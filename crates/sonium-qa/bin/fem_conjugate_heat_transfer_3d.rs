//! Conduction contract and solved transient 3D solid/fluid heat-transfer MMS.
//!
//! A conforming interface joins unequal conductivity/capacity regions. Fluid
//! advection is prescribed tangential to the interface; no flow solve is claimed.

use sonium_fem::assembly::region::RegionMap;
use sonium_fem::assembly::{TripletMatrix, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_dirichlet_values};
use sonium_fem::mesh::{Mesh, unit_cube_tetrahedra};
use sonium_fem::physics::coupled::conjugate_heat_transfer::{
    ChtConductionForm, ConjugateRegion, InterfaceHeatBalance,
    build_conjugate_heat_transfer_assembler,
};
use sonium_fem::physics::thermal::heat_conduction::HeatSourceForm;
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::{CsrMatrix, SparseLuFactorization};
use std::collections::HashMap;
const CASE: &str = "fem.conjugate-heat-transfer-3d-interface-mms.v1";
#[derive(Debug, Clone, Copy)]
enum TimeLaw {
    Linear,
    Exponential,
}
impl TimeLaw {
    fn value(self, t: f64) -> f64 {
        match self {
            Self::Linear => 1.0 + t,
            Self::Exponential => (-t).exp(),
        }
    }
    fn derivative(self, t: f64) -> f64 {
        match self {
            Self::Linear => 1.0,
            Self::Exponential => -(-t).exp(),
        }
    }
}
#[derive(Debug, Clone, Copy)]
enum Control {
    Full,
    NoCapacity,
    NoAdvection,
    NoSource,
    HomogeneousConductivity,
}
fn coefficients(solid: bool) -> (f64, f64, [f64; 3]) {
    if solid {
        (2.0, 1.3, [0.0; 3])
    } else {
        (0.7, 0.9, [0.0, 0.3, -0.2])
    }
}
fn reference(x: f64, y: f64, z: f64) -> (f64, [f64; 3], f64) {
    let (k, _, _) = coefficients(x < 0.5);
    let xi = if x < 0.5 {
        x / 2.0
    } else {
        0.25 + (x - 0.5) / 0.7
    };
    (
        1.0 + xi + xi * xi + 2.0 * y * y + 3.0 * z * z + y * z + xi * y + 2.0 * xi * z,
        [
            (1.0 + 2.0 * xi + y + 2.0 * z) / k,
            4.0 * y + z + xi,
            6.0 * z + y + 2.0 * xi,
        ],
        2.0 / (k * k) + 10.0,
    )
}
fn forcing(x: f64, y: f64, z: f64, t: f64, law: TimeLaw) -> f64 {
    let (k, c, v) = coefficients(x < 0.5);
    let (q, grad, lap) = reference(x, y, z);
    c * law.derivative(t) * q
        + law.value(t) * (-k * lap + c * v.iter().zip(grad).map(|(a, b)| a * b).sum::<f64>())
}
fn triplet(a: &CsrMatrix<f64>) -> TripletMatrix<f64> {
    let mut out = TripletMatrix::new(a.num_rows, a.num_cols);
    for row in 0..a.num_rows {
        for k in a.row_ptrs[row]..a.row_ptrs[row + 1] {
            out.push(row, a.col_indices[k], a.values[k]);
        }
    }
    out
}
fn constrained(
    a: &CsrMatrix<f64>,
    rhs: Vec<f64>,
    mesh: &Mesh,
    space: &H1Space,
    t: f64,
    law: TimeLaw,
) -> Result<FemSystem<f64>, String> {
    let mut system = FemSystem::new(triplet(a), rhs);
    try_apply_dirichlet_values(
        &mut system,
        mesh,
        space,
        1,
        &|x, y, z| law.value(t) * reference(x, y, z).0,
        None,
    )
    .map_err(|e| e.to_string())?;
    Ok(system)
}
fn point(mesh: &Mesh, node: usize) -> [f64; 3] {
    let p = &mesh.nodes[node];
    [p.x, p.y, p.z]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn gradient_at(
    mesh: &Mesh,
    space: &H1Space,
    values: &[f64],
    element: usize,
    p: [f64; 3],
) -> [f64; 3] {
    let nodes = &mesh.elements[element].nodes;
    let origin = point(mesh, nodes[0]);
    let a = sub(point(mesh, nodes[1]), origin);
    let b = sub(point(mesh, nodes[2]), origin);
    let c = sub(point(mesh, nodes[3]), origin);
    let r = sub(p, origin);
    let det = dot(a, cross(b, c));
    let xi = dot(r, cross(b, c)) / det;
    let eta = dot(a, cross(r, c)) / det;
    let zeta = dot(a, cross(b, r)) / det;
    let tab = space.tabulate(mesh, element, xi, eta, zeta);
    let dofs = space.local_to_global(mesh, element);
    std::array::from_fn(|axis| {
        dofs.iter()
            .enumerate()
            .map(|(i, &d)| values[d] * tab.grad(i, axis))
            .sum()
    })
}
fn interface_flux(
    mesh: &Mesh,
    space: &H1Space,
    values: &[f64],
    ks: f64,
    kf: f64,
) -> Result<(f64, f64), String> {
    let mut faces: HashMap<[usize; 3], Vec<usize>> = HashMap::new();
    for (element, cell) in mesh.elements.iter().enumerate() {
        for local in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
            let mut face = local.map(|i| cell.nodes[i]);
            if face.iter().all(|&i| (mesh.nodes[i].x - 0.5).abs() < 1e-12) {
                face.sort_unstable();
                faces.entry(face).or_default().push(element);
            }
        }
    }
    if faces.is_empty() {
        return Err("no solid/fluid interface facets".into());
    }
    let mut solid = 0.0;
    let mut fluid = 0.0;
    for (face, adjacent) in faces {
        if adjacent.len() != 2 {
            return Err("interface lacks two adjacent elements".into());
        }
        let se = *adjacent
            .iter()
            .find(|&&e| mesh.elements[e].marker == 1)
            .ok_or("missing solid interface element")?;
        let fe = *adjacent
            .iter()
            .find(|&&e| mesh.elements[e].marker == 2)
            .ok_or("missing fluid interface element")?;
        let p = face.map(|i| point(mesh, i));
        let normal = cross(sub(p[1], p[0]), sub(p[2], p[0]));
        let area = 0.5 * dot(normal, normal).sqrt();
        for bary in [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ] {
            let q = std::array::from_fn(|axis| (0..3).map(|i| bary[i] * p[i][axis]).sum());
            solid -= area / 3.0 * ks * gradient_at(mesh, space, values, se, q)[0];
            fluid += area / 3.0 * kf * gradient_at(mesh, space, values, fe, q)[0];
        }
    }
    Ok((solid, fluid))
}
#[derive(Debug)]
struct Metrics {
    value: f64,
    flux: f64,
    residual: f64,
    boundary: f64,
    interface_error: f64,
    interface_jump: f64,
    energy_balance: f64,
    history_error: f64,
    history_change: f64,
    steps: usize,
    free: usize,
}
fn solve(
    n: usize,
    degree: PolynomialDegree,
    steps: usize,
    law: TimeLaw,
    control: Control,
) -> Result<Metrics, String> {
    let mut mesh = unit_cube_tetrahedra(n);
    for face in &mut mesh.boundaries {
        face.marker = 1;
    }
    for element in &mut mesh.elements {
        let center = element.nodes.iter().map(|&i| mesh.nodes[i].x).sum::<f64>()
            / element.nodes.len() as f64;
        element.marker = if center < 0.5 { 1 } else { 2 };
    }
    let space = H1Space::new(degree);
    let count = space.num_global_dofs(&mesh);
    let coordinates = space
        .dof_coordinates(&mesh)
        .ok_or("missing H1 coordinates")?;
    let boundary = space.boundary_dofs(&mesh, 1);
    let mut fixed = vec![false; count];
    for &d in &boundary {
        fixed[d] = true;
    }
    let free = count - boundary.len();
    if free == 0 {
        return Err("CHT case has no free unknowns".into());
    }
    let ks = 2.0;
    let kf = if matches!(control, Control::HomogeneousConductivity) {
        2.0
    } else {
        0.7
    };
    let mut regions = RegionMap::new();
    regions.insert(1, ConjugateRegion::solid(ks, 1.0, 1.3));
    regions.insert(2, ConjugateRegion::fluid(kf, 1.0, 0.9, [0.0, 0.3, -0.2]));
    let assembler = build_conjugate_heat_transfer_assembler(&mesh, degree, &regions)
        .map_err(|e| e.to_string())?;
    let dt = 1.0 / steps as f64;
    let capacity = if matches!(control, Control::NoCapacity) {
        0.0
    } else {
        1.0
    };
    let advection = if matches!(control, Control::NoAdvection) {
        0.0
    } else {
        1.0
    };
    let lhs = assembler.assemble_real(&[0.5, capacity / dt, 0.5 * advection]);
    let mass = assembler.assemble_real(&[0.0, capacity, 0.0]);
    let adv = assembler.assemble_real(&[0.0, 0.0, advection]);
    let factor = SparseLuFactorization::factorize(
        &constrained(&lhs, vec![0.0; count], &mesh, &space, 0.0, law)?
            .matrix
            .to_csr(),
    )
    .map_err(|e| e.to_string())?;
    let source = |time: f64| {
        if matches!(control, Control::NoSource) {
            vec![0.0; count]
        } else {
            assemble_linear(
                &HeatSourceForm {
                    source_fn: |x, y, z| forcing(x, y, z, time, law),
                },
                &space,
                &mesh,
                4,
            )
        }
    };
    let mut values: Vec<f64> = coordinates
        .iter()
        .map(|p| reference(p[0], p[1], p[2]).0)
        .collect();
    let initial = values.clone();
    let mut previous_source = source(0.0);
    let mut residual = 0.0_f64;
    let mut boundary_error = 0.0_f64;
    let mut energy_balance = 0.0_f64;
    let mut history_error = 0.0_f64;
    for step in 1..=steps {
        let time = step as f64 * dt;
        let current_source = source(time);
        let old_mass = mass.matvec(&values);
        let old_lhs = lhs.matvec(&values);
        let old = values.clone();
        let rhs: Vec<_> = (0..count)
            .map(|d| {
                2.0 * old_mass[d] / dt - old_lhs[d] + 0.5 * (current_source[d] + previous_source[d])
            })
            .collect();
        values = factor
            .solve(&constrained(&lhs, rhs.clone(), &mesh, &space, time, law)?.rhs)
            .map_err(|e| e.to_string())?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite CHT history".into());
        }
        let action = lhs.matvec(&values);
        let error: f64 = (0..count)
            .filter(|&d| !fixed[d])
            .map(|d| (action[d] - rhs[d]).powi(2))
            .sum();
        let norm: f64 = (0..count)
            .filter(|&d| !fixed[d])
            .map(|d| rhs[d].powi(2))
            .sum();
        residual = residual.max((error / norm).sqrt());
        let delta: Vec<_> = values.iter().zip(&old).map(|(a, b)| (a - b) / dt).collect();
        let mean: Vec<_> = values
            .iter()
            .zip(&old)
            .map(|(a, b)| 0.5 * (a + b))
            .collect();
        let storage: f64 = mass.matvec(&delta).iter().sum();
        let transported: f64 = adv.matvec(&mean).iter().sum();
        let injected: f64 = current_source
            .iter()
            .zip(&previous_source)
            .map(|(a, b)| 0.5 * (a + b))
            .sum();
        let outward: f64 = -boundary.iter().map(|&d| action[d] - rhs[d]).sum::<f64>();
        let scale = storage
            .abs()
            .max(transported.abs())
            .max(injected.abs())
            .max(outward.abs());
        energy_balance =
            energy_balance.max((storage + transported + outward - injected).abs() / scale);
        let mut error = 0.0;
        let mut norm = 0.0;
        for (d, p) in coordinates.iter().enumerate() {
            let expected = law.value(time) * reference(p[0], p[1], p[2]).0;
            if fixed[d] {
                boundary_error = boundary_error.max((values[d] - expected).abs());
            } else {
                error += (values[d] - expected).powi(2);
                norm += expected.powi(2);
            }
        }
        history_error = history_error.max((error / norm).sqrt());
        previous_source = current_source;
    }
    let mut error = 0.0;
    let mut norm = 0.0;
    let mut flux_error = 0.0;
    let mut flux_norm = 0.0;
    let g = law.value(1.0);
    for element in 0..mesh.num_elements() {
        let region = *regions
            .get_explicit(mesh.elements[element].marker)
            .ok_or("missing region")?;
        let solid = mesh.elements[element].marker == 1;
        let (kr, cr, vr) = coefficients(solid);
        let dofs = space.local_to_global(&mesh, element);
        for qp in QuadratureRule::new(mesh.elements[element].element_type, 5).iter() {
            let tab = space.tabulate(&mesh, element, qp.xi(), qp.eta(), qp.zeta());
            let [x, y, z] = tab.physical_coords;
            let (q, grad, _) = reference(x, y, z);
            let value: f64 = dofs
                .iter()
                .enumerate()
                .map(|(i, &d)| values[d] * tab.value(i))
                .sum();
            let actual_grad: [f64; 3] = std::array::from_fn(|a| {
                dofs.iter()
                    .enumerate()
                    .map(|(i, &d)| values[d] * tab.grad(i, a))
                    .sum()
            });
            let w = qp.weight * tab.det_j;
            error += w * (value - g * q).powi(2);
            norm += w * (g * q).powi(2);
            let actual_adv = region.advective_heat_flux(value);
            for a in 0..3 {
                let actual = -region.thermal_conductivity * actual_grad[a] + actual_adv[a];
                let expected = g * (-kr * grad[a] + cr * vr[a] * q);
                flux_error += w * (actual - expected).powi(2);
                flux_norm += w * expected.powi(2);
            }
        }
    }
    let (solid, fluid) = interface_flux(&mesh, &space, &values, ks, kf)?;
    let balance = InterfaceHeatBalance::new(solid, fluid);
    let expected = 3.0 * g;
    let interface_error = (solid + expected).abs().max((fluid - expected).abs()) / expected;
    let history_change = values
        .iter()
        .zip(&initial)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
        / initial.iter().map(|v| v * v).sum::<f64>().sqrt();
    Ok(Metrics {
        value: (error / norm).sqrt(),
        flux: (flux_error / flux_norm).sqrt(),
        residual,
        boundary: boundary_error,
        interface_error,
        interface_jump: balance.relative_imbalance,
        energy_balance,
        history_error,
        history_change,
        steps,
        free,
    })
}
fn run() -> Result<QaResult, String> {
    let mut spatial = Vec::new();
    for n in [6, 12, 24] {
        let m = solve(n, PolynomialDegree::P1, 8, TimeLaw::Linear, Control::Full)?;
        println!("CHT spatial n={n}: {m:?}");
        spatial.push(m);
    }
    let mut temporal = Vec::new();
    for steps in [8, 16, 32] {
        let m = solve(
            2,
            PolynomialDegree::P2,
            steps,
            TimeLaw::Exponential,
            Control::Full,
        )?;
        println!("CHT temporal {steps}: {m:?}");
        temporal.push(m);
    }
    let patch = solve(2, PolynomialDegree::P2, 8, TimeLaw::Linear, Control::Full)?;
    println!("CHT exact P2: {patch:?}");
    let mut negative = f64::INFINITY;
    for control in [
        Control::NoCapacity,
        Control::NoAdvection,
        Control::NoSource,
        Control::HomogeneousConductivity,
    ] {
        let m = solve(2, PolynomialDegree::P2, 8, TimeLaw::Linear, control)?;
        println!("CHT {control:?}: {m:?}");
        negative = negative.min(m.history_error);
    }
    let rate = |select: fn(&Metrics) -> f64, levels: &[Metrics]| {
        levels
            .windows(2)
            .map(|m| (select(&m[0]) / select(&m[1])).log2())
            .fold(f64::INFINITY, f64::min)
    };
    let all: Vec<_> = spatial.iter().chain(&temporal).chain([&patch]).collect();
    let mut r = QaResult::passed(CASE)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (name, value, bound) in [
        ("fine_temperature_error", spatial[2].value, 0.005),
        ("fine_total_heat_flux_error", spatial[2].flux, 0.1),
        (
            "fine_interface_flux_error",
            spatial[2].interface_error,
            0.08,
        ),
        ("fine_interface_jump", spatial[2].interface_jump, 0.08),
        (
            "fine_temporal_error",
            temporal[2]
                .value
                .max(temporal[2].flux)
                .max(temporal[2].interface_error),
            0.001,
        ),
        (
            "p2_exact_error",
            patch
                .value
                .max(patch.flux)
                .max(patch.interface_error)
                .max(patch.interface_jump),
            1e-9,
        ),
        (
            "physical_residual",
            all.iter().map(|m| m.residual).fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "weak_energy_balance",
            all.iter().map(|m| m.energy_balance).fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "boundary_error",
            all.iter().map(|m| m.boundary).fold(0.0_f64, f64::max),
            1e-10,
        ),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name}={value:.6e} exceeds {bound}"));
        }
        r = r.metric(name, value).maximum(name, bound);
    }
    for (name, value, bound) in [
        ("spatial_temperature_rate", rate(|m| m.value, &spatial), 1.8),
        ("spatial_flux_rate", rate(|m| m.flux, &spatial), 0.9),
        (
            "spatial_interface_flux_rate",
            rate(|m| m.interface_error, &spatial),
            0.8,
        ),
        (
            "temporal_temperature_rate",
            rate(|m| m.value, &temporal),
            1.8,
        ),
        ("temporal_flux_rate", rate(|m| m.flux, &temporal), 1.8),
        (
            "minimum_history_change",
            all.iter()
                .map(|m| m.history_change)
                .fold(f64::INFINITY, f64::min),
            0.1,
        ),
        ("minimum_omitted_term_error", negative, 1e-3),
    ] {
        if !value.is_finite() || value < bound {
            return Err(format!("{name}={value:.6e} below {bound}"));
        }
        r = r.metric(name, value).minimum(name, bound);
    }
    Ok(r.metric("executed_time_steps",all.iter().map(|m|m.steps).sum::<usize>() as f64).minimum("executed_time_steps",88.0).metric("minimum_free_unknowns",all.iter().map(|m|m.free).min().unwrap_or(0) as f64).minimum("minimum_free_unknowns",1.0).metric("negative_controls",4.0).minimum("negative_controls",4.0)
        .reference("independent piecewise-quadratic temperature in resistive coordinate xi with continuous T and k*T_x across x=.5; exact interface rates +/-3g(t), regional capacity and tangential fluid advection")
        .reference_checksum("analytic:3d-transient-conforming-solid-fluid-CHT:v1").input_checksum("P1n6-12-24:linear8:P2n2:exp8-16-32:linear8:ks2-kf.7:Cs1.3-Cf.9:fluid-v0,.3,-.2:four-controls")
        .dimension("3D tetrahedral cube with conforming x=.5 solid/fluid interface, unequal material coefficients and mixed xyz temperature curvature")
        .formulation("native region-routed conduction/capacity/fluid-advection assembly, CN actual history, sparse LU, independently integrated two-sided interface gradient traces and total heat flux")
        .tolerance("temperature spatial/time rates>=1.8; flux spatial/time>=0.9/1.8; P2/residual/weak energy<=1e-9; four all-history omitted-term controls>=1e-3"))
}
fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.conjugate-heat-transfer-3d.v1",
        &ChtConductionForm {
            thermal_conductivity: 1.0,
        },
        true,
    );
    let result = match run() {
        Ok(r) => r,
        Err(e) => QaResult::failed(CASE, e),
    };
    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(e) => {
            eprintln!("FAILED: {e}");
            std::process::exit(1);
        }
    }
    if result.status == QaStatus::Passed {
        println!("SUCCESS");
    } else {
        std::process::exit(1);
    }
}
