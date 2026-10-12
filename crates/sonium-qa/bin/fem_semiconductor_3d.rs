//! Carrier form contract and self-consistent steady 3D transport/Poisson MMS.
//!
//! Covers the native nonconservative drift operator E.grad(c), linear carrier
//! recombination, Einstein diffusivity, and charge feedback. It does not claim
//! conservative div(c*E) transport, nonlinear SRH kinetics or transient accuracy.

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_dirichlet_values};
use sonium_fem::forms::{BilinearForm, LaplacianForm, LinearForm, LocalMatrix, LocalVector};
use sonium_fem::mesh::{Mesh, unit_cube_tetrahedra};
use sonium_fem::physics::electromagnetics::semiconductor::{
    CarrierDiffusionForm, CarrierDriftForm, CarrierKind, CarrierMaterial, CarrierRecombinationForm,
    DopingChargeForm, DopingProfile,
};
use sonium_fem::physics::thermal::heat_conduction::HeatSourceForm;
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, H1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::SparseLuFactorization;
use std::collections::HashMap;

const CASE: &str = "fem.semiconductor-3d-coupled-mms.v1";
const ORDER: usize = 4;
const CHARGE: f64 = 1.602_176_634e-19;
const DENSITY_SCALE: f64 = 0.1 / CHARGE;
const CARRIERS: [CarrierMaterial; 2] = [
    CarrierMaterial {
        mobility: 2.0,
        lifetime: 2.0,
        temperature: 300.0,
        kind: CarrierKind::Electron,
    },
    CarrierMaterial {
        mobility: 1.5,
        lifetime: 3.0,
        temperature: 300.0,
        kind: CarrierKind::Hole,
    },
];

// Independent polynomial fields, gradients and Laplacians. Every field has
// curvature in all three axes and mixed derivatives including xz and yz.
fn reference(x: f64, y: f64, z: f64) -> ([f64; 3], [[f64; 3]; 3], [f64; 3]) {
    (
        [
            0.01 * (1.0
                + x
                + 2.0 * y
                + 3.0 * z
                + x * x
                + 2.0 * y * y
                + 3.0 * z * z
                + x * y
                + 2.0 * y * z
                + 3.0 * x * z),
            1.0 + 0.2 * (x + x * x + 2.0 * y * y + 3.0 * z * z + x * y + y * z + 2.0 * x * z),
            0.8 + 0.1 * (2.0 * x * x + y * y + z * z + 2.0 * x * y + 3.0 * y * z + x * z),
        ],
        [
            [
                0.01 * (1.0 + 2.0 * x + y + 3.0 * z),
                0.01 * (2.0 + 4.0 * y + x + 2.0 * z),
                0.01 * (3.0 + 6.0 * z + 2.0 * y + 3.0 * x),
            ],
            [
                0.2 * (1.0 + 2.0 * x + y + 2.0 * z),
                0.2 * (4.0 * y + x + z),
                0.2 * (6.0 * z + y + 2.0 * x),
            ],
            [
                0.1 * (4.0 * x + 2.0 * y + z),
                0.1 * (2.0 * y + 2.0 * x + 3.0 * z),
                0.1 * (2.0 * z + 3.0 * y + x),
            ],
        ],
        [0.12, 2.4, 0.8],
    )
}
fn forcing(field: usize, x: f64, y: f64, z: f64) -> f64 {
    let (value, grad, lap) = reference(x, y, z);
    if field == 0 {
        return -lap[0] - 0.1 * (3.0 + value[2] - value[1]);
    }
    let mobility = [2.0, 1.5][field - 1];
    let diffusion = mobility * 1.380_649e-23 * 300.0 / 1.602_176_634e-19;
    let drift = [-2.0, 1.5][field - 1];
    let lifetime = [2.0, 3.0][field - 1];
    -diffusion * lap[field]
        - drift
            * grad[0]
                .iter()
                .zip(grad[field])
                .map(|(a, b)| a * b)
                .sum::<f64>()
        + value[field] / lifetime
}
#[derive(Debug, Clone, Copy)]
enum Control {
    Full,
    NoDrift,
    NoRecombination,
    NoSpaceCharge,
    NoDoping,
    NoZField,
}

// Actual FE potential gradient and carrier densities at assembly quadrature.
#[derive(Debug)]
struct Snapshot(HashMap<[u64; 3], [f64; 5]>);
impl Snapshot {
    fn build(mesh: &Mesh, space: &H1Space, values: &[Vec<f64>]) -> Self {
        let mut points = HashMap::new();
        for element in 0..mesh.num_elements() {
            let dofs = space.local_to_global(mesh, element);
            for qp in QuadratureRule::new(mesh.elements[element].element_type, ORDER).iter() {
                let tab = space.tabulate(mesh, element, qp.xi(), qp.eta(), qp.zeta());
                let mut state = [0.0; 5];
                for (i, &d) in dofs.iter().enumerate() {
                    for (axis, e) in state.iter_mut().take(3).enumerate() {
                        *e -= values[0][d] * tab.grad(i, axis);
                    }
                    state[3] += values[1][d] * tab.value(i);
                    state[4] += values[2][d] * tab.value(i);
                }
                points.insert(tab.physical_coords.map(f64::to_bits), state);
            }
        }
        Self(points)
    }
    fn at(&self, p: [f64; 3]) -> [f64; 5] {
        self.0[&p.map(f64::to_bits)]
    }
}
struct CarrierOperator<'a> {
    carrier: CarrierMaterial,
    snapshot: &'a Snapshot,
    control: Control,
}
impl BilinearForm<f64> for CarrierOperator<'_> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local: &mut LocalMatrix<f64>,
    ) {
        CarrierDiffusionForm::new(&self.carrier).integrate_point(test, trial, weight, local);
        if !matches!(self.control, Control::NoDrift) {
            let state = self.snapshot.at(test.physical_coords);
            let field = [
                state[0],
                state[1],
                if matches!(self.control, Control::NoZField) {
                    0.0
                } else {
                    state[2]
                },
            ];
            CarrierDriftForm {
                electric_field: field,
                drift_coefficient: self.carrier.drift_coefficient(),
            }
            .integrate_point(test, trial, weight, local);
        }
        if !matches!(self.control, Control::NoRecombination) {
            CarrierRecombinationForm {
                lifetime: self.carrier.lifetime,
            }
            .integrate_point(test, trial, weight, local);
        }
    }
}
struct ChargeSource<'a> {
    snapshot: &'a Snapshot,
    control: Control,
}
impl LinearForm<f64> for ChargeSource<'_> {
    fn integrate_point(&self, test: &BasisTabulation, weight: f64, local: &mut LocalVector<f64>) {
        let state = self.snapshot.at(test.physical_coords);
        let carriers = if matches!(self.control, Control::NoSpaceCharge) {
            [0.0, 0.0]
        } else {
            [state[3], state[4]]
        };
        let doping = if matches!(self.control, Control::NoDoping) {
            [0.0, 0.0]
        } else {
            [4.0, 1.0]
        };
        DopingChargeForm {
            profile: DopingProfile {
                donors: DENSITY_SCALE * (doping[0] + carriers[1]),
                acceptors: DENSITY_SCALE * (doping[1] + carriers[0]),
            },
        }
        .integrate_point(test, weight, local);
    }
}
fn constrained(
    matrix: TripletMatrix<f64>,
    rhs: Vec<f64>,
    mesh: &Mesh,
    space: &H1Space,
    field: usize,
) -> Result<FemSystem<f64>, String> {
    let mut system = FemSystem::new(matrix, rhs);
    try_apply_dirichlet_values(
        &mut system,
        mesh,
        space,
        1,
        &|x, y, z| reference(x, y, z).0[field],
        None,
    )
    .map_err(|e| e.to_string())?;
    Ok(system)
}
#[derive(Debug)]
struct Metrics {
    errors: [f64; 3],
    gradients: [f64; 3],
    residual: f64,
    boundary: f64,
    charge_balance: f64,
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
    let boundary = space.boundary_dofs(&mesh, 1);
    let mut fixed = vec![false; count];
    for &d in &boundary {
        fixed[d] = true;
    }
    let free = 3 * (count - boundary.len());
    if free == 0 {
        return Err("semiconductor case has no free unknowns".into());
    }
    let poisson = assemble_bilinear(&LaplacianForm, &space, &space, &mesh, ORDER);
    let factor = SparseLuFactorization::factorize(
        &constrained(poisson.clone(), vec![0.0; count], &mesh, &space, 0)?
            .matrix
            .to_csr(),
    )
    .map_err(|e| e.to_string())?;
    let external: Vec<_> = (0..3)
        .map(|field| {
            assemble_linear(
                &HeatSourceForm {
                    source_fn: move |x, y, z| forcing(field, x, y, z),
                },
                &space,
                &mesh,
                ORDER,
            )
        })
        .collect();
    let mut values = vec![vec![0.0; count], vec![1.0; count], vec![0.8; count]];
    let mut iterations = 0;
    let mut converged = false;
    for iteration in 1..=60 {
        let snapshot = Snapshot::build(&mesh, &space, &values);
        let charge = assemble_linear(
            &ChargeSource {
                snapshot: &snapshot,
                control,
            },
            &space,
            &mesh,
            ORDER,
        );
        let rhs = external[0].iter().zip(charge).map(|(a, b)| a + b).collect();
        let system = constrained(poisson.clone(), rhs, &mesh, &space, 0)?;
        let potential = factor.solve(&system.rhs).map_err(|e| e.to_string())?;
        let mut next = values.clone();
        next[0] = potential;
        let snapshot = Snapshot::build(&mesh, &space, &next);
        for field in 1..3 {
            let matrix = assemble_bilinear(
                &CarrierOperator {
                    carrier: CARRIERS[field - 1],
                    snapshot: &snapshot,
                    control,
                },
                &space,
                &space,
                &mesh,
                ORDER,
            );
            let system = constrained(matrix, external[field].clone(), &mesh, &space, field)?;
            next[field] = SparseLuFactorization::factorize(&system.matrix.to_csr())
                .and_then(|f| f.solve(&system.rhs))
                .map_err(|e| e.to_string())?;
        }
        let mut change = 0.0_f64;
        for field in 0..3 {
            if next[field].iter().any(|v| !v.is_finite())
                || (field > 0 && next[field].iter().any(|v| *v <= 0.0))
            {
                return Err("nonfinite potential or nonpositive carrier density".into());
            }
            let error: f64 = next[field]
                .iter()
                .zip(&values[field])
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            let norm: f64 = next[field].iter().map(|v| v * v).sum();
            change = change.max((error / norm).sqrt());
        }
        values = next;
        iterations = iteration;
        if change < 1e-12 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err("self-consistent carrier/Poisson iteration did not converge".into());
    }
    let snapshot = Snapshot::build(&mesh, &space, &values);
    let charge = assemble_linear(
        &ChargeSource {
            snapshot: &snapshot,
            control,
        },
        &space,
        &mesh,
        ORDER,
    );
    let mut residual = 0.0_f64;
    let mut charge_balance = 0.0_f64;
    for field in 0..3 {
        let matrix = if field == 0 {
            poisson.clone()
        } else {
            assemble_bilinear(
                &CarrierOperator {
                    carrier: CARRIERS[field - 1],
                    snapshot: &snapshot,
                    control,
                },
                &space,
                &space,
                &mesh,
                ORDER,
            )
        };
        let rhs: Vec<f64> = if field == 0 {
            external[0]
                .iter()
                .zip(&charge)
                .map(|(a, b)| a + b)
                .collect()
        } else {
            external[field].clone()
        };
        let action = matrix.to_csr().matvec(&values[field]);
        let numerator: f64 = (0..count)
            .filter(|&d| !fixed[d])
            .map(|d| (action[d] - rhs[d]).powi(2))
            .sum();
        let denominator: f64 = (0..count)
            .filter(|&d| !fixed[d])
            .map(|d| rhs[d].powi(2))
            .sum();
        residual = residual.max((numerator / denominator).sqrt());
        if field == 0 {
            // Integrated weak boundary flux plus assembled volume charge/source.
            let outward: f64 = boundary.iter().map(|&d| rhs[d] - action[d]).sum();
            let volume: f64 = rhs.iter().sum();
            charge_balance = (outward - volume).abs() / volume.abs().max(1e-20);
        }
    }
    let coordinates = space
        .dof_coordinates(&mesh)
        .ok_or("missing H1 coordinates")?;
    let mut boundary_error = 0.0_f64;
    for &d in &boundary {
        let p = coordinates[d];
        for field in 0..3 {
            boundary_error =
                boundary_error.max((values[field][d] - reference(p[0], p[1], p[2]).0[field]).abs());
        }
    }
    let mut errors = [0.0; 3];
    let mut norms = [0.0; 3];
    let mut gradients = [0.0; 3];
    let mut gradnorms = [0.0; 3];
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(&mesh, element);
        for qp in QuadratureRule::new(mesh.elements[element].element_type, 5).iter() {
            let tab = space.tabulate(&mesh, element, qp.xi(), qp.eta(), qp.zeta());
            let p = tab.physical_coords;
            let (expected, gradient, _) = reference(p[0], p[1], p[2]);
            let w = qp.weight * tab.det_j;
            for field in 0..3 {
                let value: f64 = dofs
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| values[field][d] * tab.value(i))
                    .sum();
                errors[field] += w * (value - expected[field]).powi(2);
                norms[field] += w * expected[field].powi(2);
                for axis in 0..3 {
                    let value: f64 = dofs
                        .iter()
                        .enumerate()
                        .map(|(i, &d)| values[field][d] * tab.grad(i, axis))
                        .sum();
                    gradients[field] += w * (value - gradient[field][axis]).powi(2);
                    gradnorms[field] += w * gradient[field][axis].powi(2);
                }
            }
        }
    }
    Ok(Metrics {
        errors: std::array::from_fn(|i| (errors[i] / norms[i]).sqrt()),
        gradients: std::array::from_fn(|i| (gradients[i] / gradnorms[i]).sqrt()),
        residual,
        boundary: boundary_error,
        charge_balance,
        iterations,
        free,
    })
}
fn run() -> Result<QaResult, String> {
    let mut levels = Vec::new();
    for n in [4, 8, 16] {
        let m = solve(n, PolynomialDegree::P1, Control::Full)?;
        println!("semiconductor P1 n={n}: {m:?}");
        levels.push(m);
    }
    let patch = solve(2, PolynomialDegree::P2, Control::Full)?;
    println!("semiconductor P2 exact: {patch:?}");
    let mut control_error = f64::INFINITY;
    for control in [
        Control::NoDrift,
        Control::NoRecombination,
        Control::NoSpaceCharge,
        Control::NoDoping,
        Control::NoZField,
    ] {
        let m = solve(2, PolynomialDegree::P2, control)?;
        println!("semiconductor {control:?}: {m:?}");
        control_error = control_error.min(m.errors.into_iter().fold(0.0_f64, f64::max));
    }
    let mut result = QaResult::passed(CASE)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for field in 0..3 {
        let value_rate = levels
            .windows(2)
            .map(|m| (m[0].errors[field] / m[1].errors[field]).log2())
            .fold(f64::INFINITY, f64::min);
        let gradient_rate = levels
            .windows(2)
            .map(|m| (m[0].gradients[field] / m[1].gradients[field]).log2())
            .fold(f64::INFINITY, f64::min);
        for (name, value, bound, minimum) in [
            (
                format!("field_{field}_fine_error"),
                levels[2].errors[field],
                0.005,
                false,
            ),
            (
                format!("field_{field}_fine_gradient_error"),
                levels[2].gradients[field],
                0.1,
                false,
            ),
            (format!("field_{field}_value_rate"), value_rate, 1.8, true),
            (
                format!("field_{field}_gradient_rate"),
                gradient_rate,
                0.9,
                true,
            ),
        ] {
            if !value.is_finite()
                || (if minimum {
                    value < bound
                } else {
                    value > bound
                })
            {
                return Err(format!("{name}={value:.3e} fails {bound}"));
            }
            result = result.metric(&name, value);
            result = if minimum {
                result.minimum(&name, bound)
            } else {
                result.maximum(&name, bound)
            };
        }
    }
    let all: Vec<_> = levels.iter().chain([&patch]).collect();
    for (name, value, bound) in [
        (
            "p2_exact_patch_error",
            patch
                .errors
                .into_iter()
                .chain(patch.gradients)
                .fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "fresh_coupled_residual",
            all.iter().map(|m| m.residual).fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "weak_charge_balance",
            all.iter().map(|m| m.charge_balance).fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "boundary_error",
            all.iter().map(|m| m.boundary).fold(0.0_f64, f64::max),
            1e-10,
        ),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name}={value:.3e} exceeds {bound}"));
        }
        result = result.metric(name, value).maximum(name, bound);
    }
    if !control_error.is_finite() || control_error < 1e-3 {
        return Err(format!("missing-term controls undetected: {control_error}"));
    }
    Ok(result.metric("minimum_omitted_term_error",control_error).minimum("minimum_omitted_term_error",1e-3)
        .metric("minimum_free_unknowns",all.iter().map(|m|m.free).min().unwrap_or(0) as f64).minimum("minimum_free_unknowns",1.0)
        .metric("executed_coupled_iterations",all.iter().map(|m|m.iterations).sum::<usize>() as f64).minimum("executed_coupled_iterations",8.0)
        .metric("negative_controls",5.0).minimum("negative_controls",5.0)
        .reference("independent 3D quadratic potential/electron/hole strong-form forcing, native Einstein diffusivity and charge-sign mapping, self-consistent field/charge feedback")
        .reference_checksum("analytic:3d-nonconservative-drift-linear-recombination-Poisson:v1")
        .input_checksum("P1n4-8-16:P2n2:electrons-mu2-tau2:holes-mu1.5-tau3:T300:qN0.1:doping4-1:five-controls")
        .dimension("3D tetrahedral unit cube; potential and both carriers vary and curve in x/y/z with mixed derivatives")
        .formulation("native carrier diffusion/drift/recombination and doping forms plus scalar Poisson, sequential self-consistent solved-state feedback; steady native E.grad(c) operator")
        .tolerance("value/gradient rates>=1.8/0.9; P2 exact<=1e-9; physical residual and weak charge balance<=1e-9; omitted terms>=1e-3"))
}
fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.semiconductor-3d.v1",
        &CarrierDiffusionForm { diffusivity: 1.0 },
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
