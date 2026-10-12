//! Solved capacitance of two circular cylinders with spatial/truncation refinement.

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_dirichlet_values};
use sonium_fem::forms::LaplacianForm;
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::physics::electromagnetics::electrostatic::{EPSILON_0, ElectrostaticSolvedField};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::analytical::electrostatics_two_cylinder_capacitance;
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::direct::SparseLuFactorization;
use std::f64::consts::PI;

const CASE_ID: &str = "fem.electrostatics-two-cylinder.v1";
const RADIUS: f64 = 0.1;
const HALF_SEPARATION: f64 = 0.25;

fn bipolar_parameters() -> (f64, f64) {
    (
        (HALF_SEPARATION.powi(2) - RADIUS.powi(2)).sqrt(),
        (HALF_SEPARATION / RADIUS).acosh(),
    )
}

fn reference(x: f64, y: f64) -> (f64, [f64; 2]) {
    let (focus, limit) = bipolar_parameters();
    let left_squared = (x + focus).powi(2) + y * y;
    let right_squared = (x - focus).powi(2) + y * y;
    let potential = (left_squared / right_squared).ln() / (4.0 * limit);
    let field = [
        -((x + focus) / left_squared - (x - focus) / right_squared) / (2.0 * limit),
        -y * (1.0 / left_squared - 1.0 / right_squared) / (2.0 * limit),
    ];
    (potential, field)
}

fn bipolar_mesh(n: usize) -> Result<(Mesh, f64), String> {
    let (focus, limit) = bipolar_parameters();
    // The omitted v sectors shrink with h. Their exact missing capacitance is
    // C_full * cut / pi; the raw computed charge must converge to C_full too.
    let cut = PI / n as f64;
    let nv = 4 * n;
    let mut mesh = Mesh::new(2);
    for j in 0..=nv {
        let v = cut + (2.0 * PI - 2.0 * cut) * j as f64 / nv as f64;
        for i in 0..=n {
            let u = -limit + 2.0 * limit * i as f64 / n as f64;
            let denominator = u.cosh() - v.cos();
            mesh.add_node(Point::new_2d(
                focus * u.sinh() / denominator,
                focus * v.sin() / denominator,
            ));
        }
    }
    let node = |i: usize, j: usize| j * (n + 1) + i;
    for j in 0..nv {
        for i in 0..n {
            for nodes in [
                [node(i, j), node(i + 1, j + 1), node(i + 1, j)],
                [node(i, j), node(i, j + 1), node(i + 1, j + 1)],
            ] {
                let [a, b, c] = nodes.map(|index| mesh.nodes[index]);
                let determinant = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
                if !determinant.is_finite() || determinant <= 0.0 {
                    return Err("bipolar mesh has an inverted or degenerate element".into());
                }
                mesh.add_element(ElementType::Triangle, nodes.to_vec());
            }
        }
    }
    mesh.detect_boundaries();
    for face in &mut mesh.boundaries {
        face.marker = if face.nodes.iter().all(|i| i % (n + 1) == 0) {
            2
        } else if face.nodes.iter().all(|i| i % (n + 1) == n) {
            3
        } else {
            1
        };
    }
    Ok((mesh, cut))
}

#[derive(Debug)]
struct CylinderMetrics {
    capacitance_error: f64,
    boundary_error: f64,
    finite_domain_error: f64,
    flux_error: f64,
    potential_error: f64,
    field_error: f64,
    charge_balance: f64,
    energy_charge_error: f64,
    residual: f64,
    truncation_fraction: f64,
    free_dofs: usize,
}

fn solve(n: usize) -> Result<CylinderMetrics, String> {
    let (mesh, cut) = bipolar_mesh(n)?;
    let (_, limit) = bipolar_parameters();
    let full_capacitance = PI * EPSILON_0 / limit;
    let finite_capacitance = full_capacitance * (1.0 - cut / PI);
    let space = H1Space::new(PolynomialDegree::P1);
    let matrix = assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 2);
    let raw = matrix.to_csr();
    let mut system = FemSystem::new(matrix, vec![0.0; mesh.num_nodes()]);
    // v=constant outer curves have zero exact normal flux. Leaving their weak
    // boundary terms natural introduces no artificial box Dirichlet trace.
    for (marker, voltage) in [(2, -0.5), (3, 0.5)] {
        try_apply_dirichlet_values(&mut system, &mesh, &space, marker, &|_, _, _| voltage, None)
            .map_err(|e| format!("electrode boundary data: {e}"))?;
    }
    let electrodes = [space.boundary_dofs(&mesh, 2), space.boundary_dofs(&mesh, 3)];
    let mut constrained = vec![false; mesh.num_nodes()];
    for &i in electrodes.iter().flatten() {
        constrained[i] = true;
    }
    let free_dofs = constrained.iter().filter(|v| !**v).count();
    if free_dofs == 0 {
        return Err("two-cylinder mesh has no free unknowns".into());
    }
    let csr = system.matrix.to_csr();
    let factor = SparseLuFactorization::factorize(&csr).map_err(|e| e.to_string())?;
    let values = factor.solve(&system.rhs).map_err(|e| e.to_string())?;
    if values.iter().any(|x| !x.is_finite()) {
        return Err("two-cylinder solve returned nonfinite potential".into());
    }
    let boundary_error = electrodes
        .iter()
        .zip([-0.5, 0.5])
        .flat_map(|(nodes, voltage)| nodes.iter().map(move |&i| (i, voltage)))
        .map(|(i, voltage)| (values[i] - voltage).abs())
        .fold(0.0_f64, f64::max);
    let action = raw.matvec(&values);
    let charges: [f64; 2] = electrodes
        .each_ref()
        .map(|nodes| EPSILON_0 * nodes.iter().map(|&i| action[i]).sum::<f64>());
    let capacitance = charges[1]; // imposed voltage difference is 1V
    let residual = action
        .iter()
        .enumerate()
        .filter(|(i, _)| !constrained[*i])
        .map(|(_, v)| v * v)
        .sum::<f64>()
        .sqrt()
        / system.rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    let field = ElectrostaticSolvedField::new(&mesh, &space, &values, EPSILON_0)
        .map_err(|e| e.to_string())?;
    let energy_capacitance = 2.0 * field.energy(&raw).map_err(|e| e.to_string())?;
    // Integrate the recovered displacement normal to the positive electrode,
    // independently of the weak reaction/energy capacitance observables.
    let mut flux_charge = 0.0;
    for face in mesh.boundaries.iter().filter(|face| face.marker == 3) {
        let a = mesh.nodes[face.nodes[0]];
        let b = mesh.nodes[face.nodes[1]];
        let tab = space.tabulate(&mesh, face.element_idx, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let mut normal_length = [b.y - a.y, a.x - b.x];
        if normal_length[0] * ((a.x + b.x) / 2.0 - tab.physical_coords[0])
            + normal_length[1] * ((a.y + b.y) / 2.0 - tab.physical_coords[1])
            < 0.0
        {
            normal_length = normal_length.map(|v| -v);
        }
        let electric = field
            .electric_field_at_reference(face.element_idx, 1.0 / 3.0, 1.0 / 3.0, 0.0)
            .map_err(|e| e.to_string())?;
        flux_charge -=
            EPSILON_0 * (electric[0] * normal_length[0] + electric[1] * normal_length[1]);
    }
    let (mut potential_error, mut potential_norm, mut field_error, mut field_norm) =
        (0.0, 0.0, 0.0, 0.0);
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(&mesh, element);
        for qp in QuadratureRule::new(ElementType::Triangle, 4).iter() {
            let tab = space.tabulate(&mesh, element, qp.xi(), qp.eta(), 0.0);
            let (expected, expected_field) =
                reference(tab.physical_coords[0], tab.physical_coords[1]);
            let actual = dofs
                .iter()
                .enumerate()
                .map(|(local, &i)| values[i] * tab.value(local))
                .sum::<f64>();
            let electric = field
                .electric_field_at_reference(element, qp.xi(), qp.eta(), 0.0)
                .map_err(|e| e.to_string())?;
            let weight = qp.weight * tab.det_j;
            potential_error += weight * (actual - expected).powi(2);
            potential_norm += weight * expected.powi(2);
            field_error += weight
                * (0..2)
                    .map(|d| (electric[d] - expected_field[d]).powi(2))
                    .sum::<f64>();
            field_norm += weight * expected_field.iter().map(|v| v * v).sum::<f64>();
        }
    }
    Ok(CylinderMetrics {
        capacitance_error: (capacitance / full_capacitance - 1.0).abs(),
        boundary_error,
        finite_domain_error: (capacitance / finite_capacitance - 1.0).abs(),
        flux_error: (flux_charge / full_capacitance - 1.0).abs(),
        potential_error: (potential_error / potential_norm).sqrt(),
        field_error: (field_error / field_norm).sqrt(),
        charge_balance: (charges[0] + charges[1]).abs() / full_capacitance,
        energy_charge_error: (energy_capacitance / capacitance - 1.0).abs(),
        residual,
        truncation_fraction: cut / PI,
        free_dofs,
    })
}

fn run() -> Result<QaResult, String> {
    let coarse = solve(16)?;
    println!("cylinders n=16: {coarse:?}");
    let medium = solve(32)?;
    println!("cylinders n=32: {medium:?}");
    let fine = solve(64)?;
    println!("cylinders n=64: {fine:?}");
    let rate = (medium.capacitance_error / fine.capacitance_error).log2();
    let field_rate = (medium.field_error / fine.field_error).log2();
    let flux_rate = (medium.flux_error / fine.flux_error).log2();
    if !(coarse.capacitance_error > medium.capacitance_error
        && medium.capacitance_error > fine.capacitance_error
        && coarse.flux_error > medium.flux_error
        && medium.flux_error > fine.flux_error
        && coarse.finite_domain_error > medium.finite_domain_error
        && medium.finite_domain_error > fine.finite_domain_error
        && coarse.potential_error > medium.potential_error
        && medium.potential_error > fine.potential_error
        && coarse.field_error > medium.field_error
        && medium.field_error > fine.field_error)
    {
        return Err("two-cylinder capacitance/flux/field errors do not contract".into());
    }
    let (focus, limit) = bipolar_parameters();
    let helper = electrostatics_two_cylinder_capacitance(RADIUS, 2.0 * HALF_SEPARATION, EPSILON_0);
    let helper_error = (helper / (PI * EPSILON_0 / limit) - 1.0).abs();
    let mut record = QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (key, value, bound) in [
        ("capacitance_error", fine.capacitance_error, 0.02),
        ("finite_domain_error", fine.finite_domain_error, 0.003),
        ("integrated_flux_error", fine.flux_error, 0.04),
        ("potential_error", fine.potential_error, 0.03),
        ("electric_field_error", fine.field_error, 0.06),
        ("charge_balance", fine.charge_balance, 1e-9),
        ("energy_charge_error", fine.energy_charge_error, 1e-9),
        ("physical_residual", fine.residual, 1e-9),
        ("analytical_helper_error", helper_error, 1e-12),
        ("boundary_error", fine.boundary_error, 1e-12),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{key} {value:.3e} exceeds {bound:.3e}"));
        }
        record = record.metric(key, value).maximum(key, bound);
    }
    if [rate, field_rate, flux_rate]
        .iter()
        .any(|r| !r.is_finite() || *r < 0.7)
    {
        return Err("two-cylinder capacitance/field/flux convergence rate below 0.7".into());
    }
    Ok(record.metric("coarse_capacitance_error",coarse.capacitance_error).metric("medium_capacitance_error",medium.capacitance_error)
        .metric("capacitance_refinement_rate",rate).minimum("capacitance_refinement_rate",0.7)
        .metric("truncation_fraction",fine.truncation_fraction).metric("free_dofs",fine.free_dofs as f64).minimum("free_dofs",1.0)
        .reference("bipolar potential log(r_minus/r_plus)/(2 acosh(c/r)); full C'=pi epsilon/acosh(c/r)")
        .reference_checksum("analytic:bipolar-two-cylinder:v1")
        .input_checksum("bipolar:n16-n32-n64:nv4n:shrinking-v-cut:v1")
        .dimension(format!("2D cylinders radius {RADIUS}, centres +/-{HALF_SEPARATION}, focal distance {focus}"))
        .formulation("production P1 Laplacian and solved electrostatic field; +/-0.5V electrodes; natural outer flux")
        .tolerance("raw full-domain capacitance and integrated displacement converge as spatial mesh and omitted far-field sectors refine"))
}

fn main() {
    let result = match run() {
        Ok(record) => record,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
    if result.status != QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
