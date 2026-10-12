//! Diffusion contract and solved 3D linearized PNP/Butler-Volmer reference.
//!
//! Two charged species feed Poisson charge and receive native c0-linearized
//! migration. An anodic electrode injects cations with nonlinear BV kinetics
//! and solved concentration activity. No fully nonlinear c*grad(phi), transient
//! electrochemistry, or resolved double-layer device accuracy is claimed.

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{BilinearForm, LaplacianForm, LinearForm, LocalMatrix, LocalVector};
use sonium_fem::mesh::{ElementType, Mesh, unit_cube_tetrahedra};
use sonium_fem::physics::coupled::electrochemistry::{
    ButlerVolmerBoundaryFluxForm, ButlerVolmerKinetics, ButlerVolmerTangentForm,
    ElectroMigrationForm, IonicChargeSourceForm, IonicSpecies, NernstPlanckDiffusionForm,
};
use sonium_fem::physics::thermal::heat_conduction::HeatSourceForm;
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, H1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::SparseLuFactorization;

const CASE: &str = "fem.electrochemistry-3d-pnp-electrode.v1";
const F: f64 = 96_485.332_12;
const VT: f64 = 8.314_462_618 * 300.0 / F;
const EPSILON: f64 = 10.0 * F;
const C0: f64 = 0.1;
const KINETICS: ButlerVolmerKinetics = ButlerVolmerKinetics {
    exchange_current_density: F * 0.2,
    alpha_anodic: 0.5,
    alpha_cathodic: 0.5,
    temperature: 300.0,
};
#[derive(Debug, Clone, Copy)]
enum Control {
    Full,
    NoMigration,
    NoCharge,
    NoReaction,
    FrozenActivity,
}
fn reference(x: f64, y: f64, z: f64) -> ([f64; 3], [[f64; 3]; 3], [f64; 3]) {
    let phi = 0.03 * (1.0 + x * x + y * y + z * z + x * y + x * z + y * z);
    let dphi = [
        0.03 * (2.0 * x + y + z),
        0.03 * (2.0 * y + x + z),
        0.03 * (2.0 * z + x + y),
    ];
    let r = 1.0 - x;
    let a = C0 / VT;
    (
        [
            phi,
            1.0 + 0.1 * (r * r + 2.0 * y * y + 3.0 * z * z + r * y + 2.0 * r * z + y * z),
            0.8 + 0.1 * (2.0 * y * y + 3.0 * z * z + y * z) + a * phi,
        ],
        [
            dphi,
            [
                -0.1 * (2.0 * r + y + 2.0 * z),
                0.1 * (4.0 * y + r + z),
                0.1 * (6.0 * z + 2.0 * r + y),
            ],
            [
                a * dphi[0],
                0.1 * (4.0 * y + z) + a * dphi[1],
                0.1 * (6.0 * z + y) + a * dphi[2],
            ],
        ],
        [0.18, 1.2, 1.0 + a * 0.18],
    )
}
fn source(field: usize, x: f64, y: f64, z: f64) -> f64 {
    let (v, _, lap) = reference(x, y, z);
    if field == 0 {
        -EPSILON * lap[0] - F * (v[1] - v[2])
    } else {
        let d = [0.5, 0.7][field - 1];
        let beta = [1.0, -1.0][field - 1] * d * C0 / VT;
        -d * lap[field] - beta * lap[0]
    }
}
fn electrode(y: f64, z: f64) -> (f64, f64) {
    let (v, grad, _) = reference(1.0, y, z);
    let flux = 0.5 * grad[1][0] + 0.5 * C0 / VT * grad[0][0];
    let current = F * flux;
    // Independent inverse symmetric BV law and Nernst concentration activity.
    (
        v[0] + VT * v[1].ln() + 2.0 * VT * (current / (2.0 * F * 0.2)).asinh(),
        current,
    )
}
struct ChargeCoupling {
    valence: f64,
}
impl BilinearForm<f64> for ChargeCoupling {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        w: f64,
        local: &mut LocalMatrix<f64>,
    ) {
        for j in 0..trial.n_dofs {
            let mut column = LocalVector::zeros(test.n_dofs);
            IonicChargeSourceForm {
                valence: self.valence,
                concentration: trial.value(j),
            }
            .integrate_point(test, w, &mut column);
            for i in 0..test.n_dofs {
                local.add(i, j, column.values[i]);
            }
        }
    }
}
fn append(
    out: &mut TripletMatrix<f64>,
    a: &TripletMatrix<f64>,
    row: usize,
    col: usize,
    scale: f64,
) {
    for k in 0..a.nnz() {
        out.push(row + a.rows[k], col + a.cols[k], scale * a.values[k]);
    }
}
fn p(mesh: &Mesh, node: usize) -> [f64; 3] {
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
struct ElectrodePoint {
    tab: BasisTabulation,
    dofs: Vec<usize>,
    weight: f64,
    metal: f64,
    reference_current: f64,
}
fn electrode_points(mesh: &Mesh, space: &H1Space) -> Vec<ElectrodePoint> {
    let mut out = Vec::new();
    for (e, cell) in mesh.elements.iter().enumerate() {
        for face in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
            let nodes = face.map(|i| cell.nodes[i]);
            if !nodes.iter().all(|&i| (mesh.nodes[i].x - 1.0).abs() < 1e-12) {
                continue;
            }
            let corners = nodes.map(|i| p(mesh, i));
            let area_scale = dot(
                cross(sub(corners[1], corners[0]), sub(corners[2], corners[0])),
                cross(sub(corners[1], corners[0]), sub(corners[2], corners[0])),
            )
            .sqrt();
            let origin = p(mesh, cell.nodes[0]);
            let a = sub(p(mesh, cell.nodes[1]), origin);
            let b = sub(p(mesh, cell.nodes[2]), origin);
            let c = sub(p(mesh, cell.nodes[3]), origin);
            let det = dot(a, cross(b, c));
            for qp in QuadratureRule::new(ElementType::Triangle, 4).iter() {
                let point = std::array::from_fn(|axis| {
                    corners[0][axis] * (1.0 - qp.xi() - qp.eta())
                        + corners[1][axis] * qp.xi()
                        + corners[2][axis] * qp.eta()
                });
                let r = sub(point, origin);
                let tab = space.tabulate(
                    mesh,
                    e,
                    dot(r, cross(b, c)) / det,
                    dot(a, cross(r, c)) / det,
                    dot(a, cross(b, r)) / det,
                );
                let (metal, current) = electrode(point[1], point[2]);
                out.push(ElectrodePoint {
                    tab,
                    dofs: space.local_to_global(mesh, e),
                    weight: qp.weight * area_scale,
                    metal,
                    reference_current: current,
                });
            }
        }
    }
    out
}
fn reaction(
    points: &[ElectrodePoint],
    values: &[f64],
    count: usize,
    control: Control,
) -> Result<(Vec<f64>, TripletMatrix<f64>, f64, f64), String> {
    let mut rhs = vec![0.0; 3 * count];
    let mut jac = TripletMatrix::new(3 * count, 3 * count);
    let mut current = 0.0;
    let mut expected = 0.0;
    for point in points {
        let n = point.tab.n_dofs;
        let phi: f64 = point
            .dofs
            .iter()
            .enumerate()
            .map(|(i, &d)| values[d] * point.tab.value(i))
            .sum();
        let concentration: f64 = point
            .dofs
            .iter()
            .enumerate()
            .map(|(i, &d)| values[count + d] * point.tab.value(i))
            .sum();
        if !concentration.is_finite() || concentration <= 0.0 {
            return Err("nonpositive electrode activity".into());
        }
        expected += point.weight * point.reference_current;
        if matches!(control, Control::NoReaction) {
            continue;
        }
        let activity = if matches!(control, Control::FrozenActivity) {
            1.0
        } else {
            concentration
        };
        let eta = point.metal - phi - VT * activity.ln();
        let mut flux = LocalVector::zeros(n);
        let mut tangent = LocalMatrix::zeros(n);
        let weight = point.weight / point.tab.det_j;
        ButlerVolmerBoundaryFluxForm {
            kinetics: KINETICS,
            overpotential: eta,
            stoichiometric_coefficient: 1.0,
        }
        .integrate_point(&point.tab, weight, &mut flux);
        ButlerVolmerTangentForm {
            kinetics: KINETICS,
            overpotential: eta,
        }
        .integrate_point(&point.tab, &point.tab, weight, &mut tangent);
        current += F * flux.values.iter().sum::<f64>();
        for (i, &di) in point.dofs.iter().enumerate() {
            rhs[count + di] += flux.values[i];
            for (j, &dj) in point.dofs.iter().enumerate() {
                let value = tangent.get(i, j) / F;
                jac.push(count + di, dj, value);
                if !matches!(control, Control::FrozenActivity) {
                    jac.push(count + di, count + dj, value * VT / concentration);
                }
            }
        }
    }
    Ok((rhs, jac, current, expected))
}
#[derive(Debug)]
struct Metrics {
    errors: [f64; 3],
    fluxes: [f64; 3],
    current: f64,
    residual: f64,
    balance: f64,
    boundary: f64,
    iterations: usize,
    free: usize,
}
fn solve(n: usize, degree: PolynomialDegree, control: Control) -> Result<Metrics, String> {
    let mut mesh = unit_cube_tetrahedra(n);
    for face in &mut mesh.boundaries {
        face.marker = 1;
    }
    let space = H1Space::new(degree);
    let count = space.num_global_dofs(&mesh);
    let coords = space
        .dof_coordinates(&mesh)
        .ok_or("H1 coordinates missing")?;
    let mut fixed = vec![false; 3 * count];
    let boundary = space.boundary_dofs(&mesh, 1);
    for &d in &boundary {
        fixed[d] = true;
        let q = coords[d];
        if q[0] < 1.0 - 1e-12
            || q[1] < 1e-12
            || q[1] > 1.0 - 1e-12
            || q[2] < 1e-12
            || q[2] > 1.0 - 1e-12
        {
            fixed[count + d] = true;
            fixed[2 * count + d] = true;
        }
    }
    let free = (0..3)
        .map(|f| (f * count..(f + 1) * count).filter(|&d| !fixed[d]).count())
        .min()
        .unwrap_or(0);
    if free == 0 {
        return Err("PNP reference lacks free field unknowns".into());
    }
    let species = [
        IonicSpecies::from_diffusivity(0.5, 1.0, 300.0),
        IonicSpecies::from_diffusivity(0.7, -1.0, 300.0),
    ];
    let mut base = TripletMatrix::new(3 * count, 3 * count);
    let lap = assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 4);
    append(&mut base, &lap, 0, 0, EPSILON);
    for f in 0..2 {
        let d = assemble_bilinear(
            &NernstPlanckDiffusionForm {
                diffusivity: species[f].diffusivity,
            },
            &space,
            &space,
            &mesh,
            4,
        );
        append(&mut base, &d, (f + 1) * count, (f + 1) * count, 1.0);
        if !matches!(control, Control::NoMigration) {
            let m = assemble_bilinear(
                &ElectroMigrationForm {
                    species: species[f],
                    reference_concentration: C0,
                },
                &space,
                &space,
                &mesh,
                4,
            );
            append(&mut base, &m, (f + 1) * count, 0, 1.0);
        }
        if !matches!(control, Control::NoCharge) {
            let c = assemble_bilinear(
                &ChargeCoupling {
                    valence: species[f].valence,
                },
                &space,
                &space,
                &mesh,
                4,
            );
            append(&mut base, &c, 0, (f + 1) * count, -1.0);
        }
    }
    let raw = base.to_csr();
    let external: Vec<f64> = (0..3)
        .flat_map(|field| {
            assemble_linear(
                &HeatSourceForm {
                    source_fn: move |x, y, z| source(field, x, y, z),
                },
                &space,
                &mesh,
                4,
            )
        })
        .collect();
    let points = electrode_points(&mesh, &space);
    if points.is_empty() {
        return Err("missing nonlinear electrode facets".into());
    }
    let mut values = vec![1.0; 3 * count];
    values[..count].fill(0.0);
    for f in 0..3 {
        for d in 0..count {
            if fixed[f * count + d] {
                let q = coords[d];
                values[f * count + d] = reference(q[0], q[1], q[2]).0[f];
            }
        }
    }
    let mut iterations = 0;
    let mut converged = false;
    for iteration in 1..=30 {
        let (boundary_rhs, jac, _, _) = reaction(&points, &values, count, control)?;
        let action = raw.matvec(&values);
        let residual: Vec<f64> = (0..3 * count)
            .map(|d| external[d] + boundary_rhs[d] - action[d])
            .collect();
        let mut matrix = base.clone();
        append(&mut matrix, &jac, 0, 0, 1.0);
        let mut rhs = residual;
        for k in 0..matrix.nnz() {
            if fixed[matrix.rows[k]] || fixed[matrix.cols[k]] {
                matrix.values[k] = 0.0;
            }
        }
        for (d, &is_fixed) in fixed.iter().enumerate() {
            if is_fixed {
                matrix.push(d, d, 1.0);
                rhs[d] = 0.0;
            }
        }
        let delta = SparseLuFactorization::factorize(&matrix.to_csr())
            .and_then(|f| f.solve(&rhs))
            .map_err(|e| e.to_string())?;
        let mut damping = 1.0;
        while (count..3 * count).any(|d| values[d] + damping * delta[d] <= 0.0) {
            damping *= 0.5;
            if damping < 1e-6 {
                return Err("Newton update cannot preserve positive ionic concentrations".into());
            }
        }
        for (v, d) in values.iter_mut().zip(delta) {
            *v += damping * d;
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite coupled PNP solve".into());
        }
        iterations = iteration;
        let (boundary_rhs, _, _, _) = reaction(&points, &values, count, control)?;
        let action = raw.matvec(&values);
        let mut error = 0.0_f64;
        for f in 0..3 {
            let r: f64 = (f * count..(f + 1) * count)
                .filter(|&d| !fixed[d])
                .map(|d| (action[d] - external[d] - boundary_rhs[d]).powi(2))
                .sum();
            let b: f64 = (f * count..(f + 1) * count)
                .filter(|&d| !fixed[d])
                .map(|d| (external[d] + boundary_rhs[d]).powi(2))
                .sum();
            error = error.max((r / b).sqrt());
        }
        if error < 1e-11 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err("nonlinear PNP/electrode solve did not converge".into());
    }
    let (boundary_rhs, _, current, expected_current) = reaction(&points, &values, count, control)?;
    let action = raw.matvec(&values);
    let mut residual = 0.0_f64;
    let mut balance = 0.0_f64;
    let mut boundary_error = 0.0_f64;
    for f in 0..3 {
        let range = f * count..(f + 1) * count;
        let error: f64 = range
            .clone()
            .filter(|&d| !fixed[d])
            .map(|d| (action[d] - external[d] - boundary_rhs[d]).powi(2))
            .sum();
        let norm: f64 = range
            .clone()
            .filter(|&d| !fixed[d])
            .map(|d| (external[d] + boundary_rhs[d]).powi(2))
            .sum();
        residual = residual.max((error / norm).sqrt());
        let reaction: f64 = range
            .clone()
            .filter(|&d| fixed[d])
            .map(|d| action[d] - external[d] - boundary_rhs[d])
            .sum();
        let volume: f64 = range.clone().map(|d| external[d] + boundary_rhs[d]).sum();
        let charge = if f == 0 {
            let mut s = 0.0;
            for element in 0..mesh.num_elements() {
                let dofs = space.local_to_global(&mesh, element);
                for qp in QuadratureRule::new(mesh.elements[element].element_type, 4).iter() {
                    let tab = space.tabulate(&mesh, element, qp.xi(), qp.eta(), qp.zeta());
                    let density: f64 = dofs
                        .iter()
                        .enumerate()
                        .map(|(i, &d)| {
                            F * (values[count + d] - values[2 * count + d]) * tab.value(i)
                        })
                        .sum();
                    s += qp.weight * tab.det_j * density;
                }
            }
            if matches!(control, Control::NoCharge) {
                0.0
            } else {
                s
            }
        } else {
            0.0
        };
        balance = balance.max(
            (reaction + volume + charge).abs()
                / (volume.abs() + charge.abs() + reaction.abs()).max(1e-20),
        );
        for d in range {
            if fixed[d] {
                let q = coords[d % count];
                boundary_error =
                    boundary_error.max((values[d] - reference(q[0], q[1], q[2]).0[f]).abs());
            }
        }
    }
    let mut errors = [0.0; 3];
    let mut norms = [0.0; 3];
    let mut flux_errors = [0.0; 3];
    let mut flux_norms = [0.0; 3];
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(&mesh, element);
        for qp in QuadratureRule::new(mesh.elements[element].element_type, 5).iter() {
            let tab = space.tabulate(&mesh, element, qp.xi(), qp.eta(), qp.zeta());
            let [x, y, z] = tab.physical_coords;
            let (expected, grad, _) = reference(x, y, z);
            let w = qp.weight * tab.det_j;
            let actual: [f64; 3] = std::array::from_fn(|f| {
                dofs.iter()
                    .enumerate()
                    .map(|(i, &d)| values[f * count + d] * tab.value(i))
                    .sum()
            });
            let actual_grad: [[f64; 3]; 3] = std::array::from_fn(|f| {
                std::array::from_fn(|a| {
                    dofs.iter()
                        .enumerate()
                        .map(|(i, &d)| values[f * count + d] * tab.grad(i, a))
                        .sum()
                })
            });
            for f in 0..3 {
                errors[f] += w * (actual[f] - expected[f]).powi(2);
                norms[f] += w * expected[f].powi(2);
                for a in 0..3 {
                    let (flux, expected_flux) = if f == 0 {
                        (-actual_grad[0][a], -grad[0][a])
                    } else {
                        let d = [0.5, 0.7][f - 1];
                        let beta = [1.0, -1.0][f - 1] * d * C0 / VT;
                        let b = species[f - 1].valence * species[f - 1].mobility * C0;
                        (
                            -species[f - 1].diffusivity * actual_grad[f][a] - b * actual_grad[0][a],
                            -d * grad[f][a] - beta * grad[0][a],
                        )
                    };
                    flux_errors[f] += w * (flux - expected_flux).powi(2);
                    flux_norms[f] += w * expected_flux.powi(2);
                }
            }
        }
    }
    Ok(Metrics {
        errors: std::array::from_fn(|f| (errors[f] / norms[f]).sqrt()),
        fluxes: std::array::from_fn(|f| (flux_errors[f] / flux_norms[f]).sqrt()),
        current: (current - expected_current).abs() / expected_current.abs(),
        residual,
        balance,
        boundary: boundary_error,
        iterations,
        free,
    })
}
fn run() -> Result<QaResult, String> {
    let mut levels = Vec::new();
    for n in [4, 8, 16] {
        let m = solve(n, PolynomialDegree::P1, Control::Full)?;
        println!("PNP/electrode P1 n={n}: {m:?}");
        levels.push(m);
    }
    let patch = solve(2, PolynomialDegree::P2, Control::Full)?;
    println!("PNP/electrode exact P2: {patch:?}");
    let mut negative = f64::INFINITY;
    for control in [
        Control::NoMigration,
        Control::NoCharge,
        Control::NoReaction,
        Control::FrozenActivity,
    ] {
        let m = solve(2, PolynomialDegree::P2, control)?;
        println!("PNP/electrode {control:?}: {m:?}");
        negative = negative.min(m.errors.into_iter().fold(0.0_f64, f64::max).max(m.current));
    }
    let mut r = QaResult::passed(CASE)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for f in 0..3 {
        let rate = levels
            .windows(2)
            .map(|m| (m[0].errors[f] / m[1].errors[f]).log2())
            .fold(f64::INFINITY, f64::min);
        let flux_rate = levels
            .windows(2)
            .map(|m| (m[0].fluxes[f] / m[1].fluxes[f]).log2())
            .fold(f64::INFINITY, f64::min);
        for (name, value, bound, minimum) in [
            (
                format!("field_{f}_fine_error"),
                levels[2].errors[f],
                0.005,
                false,
            ),
            (
                format!("field_{f}_fine_flux_error"),
                levels[2].fluxes[f],
                0.1,
                false,
            ),
            (format!("field_{f}_value_rate"), rate, 1.8, true),
            (format!("field_{f}_flux_rate"), flux_rate, 0.9, true),
        ] {
            if !value.is_finite()
                || (if minimum {
                    value < bound
                } else {
                    value > bound
                })
            {
                return Err(format!("{name}={value:.6e} fails {bound}"));
            }
            r = r.metric(&name, value);
            r = if minimum {
                r.minimum(&name, bound)
            } else {
                r.maximum(&name, bound)
            };
        }
    }
    let all: Vec<_> = levels.iter().chain([&patch]).collect();
    for (name, value, bound) in [
        ("fine_electrode_current_error", levels[2].current, 0.01),
        (
            "exact_p2_error",
            patch
                .errors
                .into_iter()
                .chain(patch.fluxes)
                .chain([patch.current])
                .fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "physical_coupled_residual",
            all.iter().map(|m| m.residual).fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "weak_charge_species_balance",
            all.iter().map(|m| m.balance).fold(0.0_f64, f64::max),
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
    let current_rate = levels
        .windows(2)
        .map(|m| (m[0].current / m[1].current).log2())
        .fold(f64::INFINITY, f64::min);
    if !current_rate.is_finite() || current_rate < 1.8 {
        return Err(format!(
            "electrode current does not converge: {current_rate}"
        ));
    }
    r = r
        .metric("electrode_current_rate", current_rate)
        .minimum("electrode_current_rate", 1.8);
    if !negative.is_finite() || negative < 1e-3 {
        return Err(format!("undetected omitted-term control: {negative}"));
    }
    Ok(r.metric("minimum_omitted_term_error",negative).minimum("minimum_omitted_term_error",1e-3).metric("minimum_free_unknowns",all.iter().map(|m|m.free).min().unwrap_or(0) as f64).minimum("minimum_free_unknowns",1.0).metric("executed_newton_solves",all.iter().map(|m|m.iterations).sum::<usize>() as f64).minimum("executed_newton_solves",4.0).metric("negative_controls",4.0).minimum("negative_controls",4.0)
        .reference("independent mixed-xyz quadratic potential/two-ion strong forms, blocked anion electrode flux and inverse symmetric BV metal voltage with concentration activity; electric and ionic fluxes")
        .reference_checksum("analytic:3d-linearized-PNP-nonlinear-BV-activity-electrode:v1").input_checksum("P1n4-8-16:P2n2:Dplus.5-Dminus.7:z+1,-1:c0.1:T300:epsilon10F:j0.2F:four-controls")
        .dimension("3D tetrahedral cube, three free field blocks, nonlinear x=1 electrode concentration trace, mixed xyz curvature")
        .formulation("native diffusion/c0-linearized migration/ionic charge and BV flux/tangent forms; monolithic positive-concentration Newton; independently recovered electric/ion flux and electrode current")
        .tolerance("three value/flux spatial rates>=1.8/0.9; P2/residual/charge-species balance<=1e-9; four omitted-term controls>=1e-3"))
}
fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.electrochemistry-3d.v1",
        &NernstPlanckDiffusionForm { diffusivity: 1.0 },
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
