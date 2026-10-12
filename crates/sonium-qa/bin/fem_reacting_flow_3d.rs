//! Form contract and solved prescribed-flow 3D reacting-species/heat MMS.

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_dirichlet_values};
use sonium_fem::forms::{BilinearForm, LocalMatrix};
use sonium_fem::mesh::{Mesh, unit_cube_tetrahedra};
use sonium_fem::physics::fluid::reacting_flow::{
    ArrheniusReaction, ChemicalSpecies, ReactionHeatSourceForm, ReactionMechanism,
    RecoveredReactionSourceForm, SpeciesAdvectionForm, SpeciesDiffusionForm,
};
use sonium_fem::physics::thermal::heat_conduction::{
    AdvectionForm, ConductionForm, HeatSourceForm,
};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, H1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::SparseLuFactorization;
use std::collections::HashMap;

const CASE_ID: &str = "fem.reacting-flow-3d-mms.v1";
const ORDER: usize = 4;
const VELOCITY: [f64; 3] = [0.3, -0.2, 0.5];
const DIFFUSIVITY: [f64; 4] = [1.0, 1.3, 0.8, 1.5];
const LAPLACIAN: [f64; 4] = [1.2, -1.2, 0.64, -120.0];
const STOICHIOMETRY: [f64; 3] = [-1.0, -1.0, 1.0];

fn reference(x: f64, y: f64, z: f64) -> ([f64; 4], [[f64; 3]; 4]) {
    (
        [
            1.0 + 0.1 * (x * x + 2.0 * y * y + 3.0 * z * z + x * y + x * z + y * z),
            0.8 + 0.15
                * (x * (1.0 - x)
                    + 2.0 * y * (1.0 - y)
                    + z * (1.0 - z)
                    + x * y
                    + 2.0 * x * z
                    + y * z),
            0.6 + 0.08 * (2.0 * x * x + y * y + z * z + x * y + 2.0 * x * z + y * z),
            10.0 * (x * (1.0 - x) + 2.0 * y * (1.0 - y) + 3.0 * z * (1.0 - z)),
        ],
        [
            [
                0.1 * (2.0 * x + y + z),
                0.1 * (4.0 * y + x + z),
                0.1 * (6.0 * z + x + y),
            ],
            [
                0.15 * (1.0 - 2.0 * x + y + 2.0 * z),
                0.15 * (2.0 - 4.0 * y + x + z),
                0.15 * (1.0 - 2.0 * z + 2.0 * x + y),
            ],
            [
                0.08 * (4.0 * x + y + 2.0 * z),
                0.08 * (2.0 * y + x + z),
                0.08 * (2.0 * z + 2.0 * x + y),
            ],
            [
                10.0 * (1.0 - 2.0 * x),
                20.0 * (1.0 - 2.0 * y),
                30.0 * (1.0 - 2.0 * z),
            ],
        ],
    )
}
fn mechanism() -> ReactionMechanism {
    ReactionMechanism {
        species: ["A", "B", "AB"]
            .into_iter()
            .enumerate()
            .map(|(i, name)| ChemicalSpecies {
                name: name.into(),
                molar_mass: if i == 2 { 2.0 } else { 1.0 },
                diffusivity: DIFFUSIVITY[i],
                specific_heat: 1.0,
            })
            .collect(),
        reactions: vec![ArrheniusReaction {
            stoichiometry: STOICHIOMETRY.to_vec(),
            orders: vec![1.0, 1.0, 0.0],
            pre_exponential_factor: 20.0,
            activation_energy: 5000.0,
            heat_release: 5.0,
        }],
    }
}
// Strong-equation forcing is differentiated independently of all FE operators
// and native reaction helpers. Species are concentrations, density is one,
// and the thermal variable is the temperature rise above 300 K.
fn forcing(field: usize, x: f64, y: f64, z: f64) -> f64 {
    let (state, gradient) = reference(x, y, z);
    let rate = 20.0 * (-5000.0 / (8.314_462_618 * (300.0 + state[3]))).exp() * state[0] * state[1];
    let reaction = if field < 3 {
        STOICHIOMETRY[field] * rate
    } else {
        5.0 * rate
    };
    -DIFFUSIVITY[field] * LAPLACIAN[field]
        + VELOCITY
            .iter()
            .zip(gradient[field])
            .map(|(a, b)| a * b)
            .sum::<f64>()
        - reaction
}
#[derive(Debug, Clone, Copy)]
struct Operator {
    diffusivity: f64,
    advection: bool,
    thermal: bool,
}
impl BilinearForm<f64> for Operator {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local: &mut LocalMatrix<f64>,
    ) {
        if self.thermal {
            ConductionForm::from_conductivity(self.diffusivity)
                .integrate_point(test, trial, weight, local);
            if self.advection {
                AdvectionForm {
                    rho_c: 1.0,
                    velocity_fn: |_, _, _| VELOCITY.to_vec(),
                }
                .integrate_point(test, trial, weight, local);
            }
        } else {
            SpeciesDiffusionForm {
                density_diffusivity: self.diffusivity,
            }
            .integrate_point(test, trial, weight, local);
            if self.advection {
                SpeciesAdvectionForm {
                    density: 1.0,
                    velocity: |_, _, _| VELOCITY,
                }
                .integrate_point(test, trial, weight, local);
            }
        }
    }
}
#[derive(Debug, Clone, Copy)]
enum Control {
    Full,
    NoReaction,
    NoHeat,
    FrozenTemperature,
    NoAdvection,
}

fn loads(
    mesh: &Mesh,
    space: &H1Space,
    values: &[Vec<f64>],
    chemistry: &ReactionMechanism,
    control: Control,
) -> [Vec<f64>; 4] {
    let mut points = HashMap::new();
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(mesh, element);
        for qp in QuadratureRule::new(mesh.elements[element].element_type, ORDER).iter() {
            let tab = space.tabulate(mesh, element, qp.xi(), qp.eta(), qp.zeta());
            let state: [f64; 4] = std::array::from_fn(|field| {
                dofs.iter()
                    .enumerate()
                    .map(|(i, &d)| values[field][d] * tab.value(i))
                    .sum()
            });
            points.insert(tab.physical_coords.map(f64::to_bits), state);
        }
    }
    std::array::from_fn(|field| {
        if matches!(control, Control::NoReaction)
            || (field == 3 && matches!(control, Control::NoHeat))
        {
            return vec![0.0; space.num_global_dofs(mesh)];
        }
        let source = |x: f64, y: f64, z: f64| {
            let state = points[&[x.to_bits(), y.to_bits(), z.to_bits()]];
            let temperature = if matches!(control, Control::FrozenTemperature) {
                300.0
            } else {
                300.0 + state[3]
            };
            if field < 3 {
                chemistry.sources(&state[..3], temperature)[field]
            } else {
                chemistry.heat_release_rate(&state[..3], temperature)
            }
        };
        if field < 3 {
            assemble_linear(&RecoveredReactionSourceForm { source }, space, mesh, ORDER)
        } else {
            assemble_linear(&ReactionHeatSourceForm { source }, space, mesh, ORDER)
        }
    })
}
fn boundary_system(
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
    species_error: f64,
    temperature_error: f64,
    species_gradient: f64,
    temperature_gradient: f64,
    residual: f64,
    boundary: f64,
    mass_balance: f64,
    heat_transfer: f64,
    iterations: usize,
    free_dofs: usize,
}
fn solve(n: usize, degree: PolynomialDegree, control: Control) -> Result<Metrics, String> {
    let mut mesh = unit_cube_tetrahedra(n);
    for face in &mut mesh.boundaries {
        face.marker = 1;
    }
    let space = H1Space::new(degree);
    let count = space.num_global_dofs(&mesh);
    let boundary = space.boundary_dofs(&mesh, 1);
    let mut constrained = vec![false; count];
    for &i in &boundary {
        constrained[i] = true;
    }
    let free_dofs = 4 * (count - boundary.len());
    if free_dofs == 0 {
        return Err("reacting-flow MMS has no free unknowns".into());
    }
    let chemistry = mechanism();
    chemistry
        .validate()
        .map_err(|e| format!("mechanism: {e:?}"))?;
    let matrices: Vec<_> = DIFFUSIVITY
        .into_iter()
        .enumerate()
        .map(|(field, diffusivity)| {
            assemble_bilinear(
                &Operator {
                    diffusivity,
                    thermal: field == 3,
                    advection: !matches!(control, Control::NoAdvection),
                },
                &space,
                &space,
                &mesh,
                ORDER,
            )
        })
        .collect();
    let raw: Vec<_> = matrices.iter().map(TripletMatrix::to_csr).collect();
    let factors: Vec<_> = matrices
        .iter()
        .enumerate()
        .map(|(field, matrix)| {
            SparseLuFactorization::factorize(
                &boundary_system(matrix.clone(), vec![0.0; count], &mesh, &space, field)?
                    .matrix
                    .to_csr(),
            )
            .map_err(|e| e.to_string())
        })
        .collect::<Result<_, String>>()?;
    let external: Vec<_> = (0..4)
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
    let mut values: Vec<Vec<f64>> = [1.0, 0.8, 0.6, 0.0]
        .into_iter()
        .map(|c| vec![c; count])
        .collect();
    let mut iterations = 0;
    let mut converged = false;
    for iteration in 1..=80 {
        let reaction = loads(&mesh, &space, &values, &chemistry, control);
        let mut next = Vec::new();
        let mut change = 0.0_f64;
        for field in 0..4 {
            let rhs: Vec<_> = external[field]
                .iter()
                .zip(&reaction[field])
                .map(|(a, b)| a + b)
                .collect();
            let system = boundary_system(matrices[field].clone(), rhs, &mesh, &space, field)?;
            let state = factors[field]
                .solve(&system.rhs)
                .map_err(|e| e.to_string())?;
            if state.iter().any(|v| !v.is_finite()) || (field < 3 && state.iter().any(|v| *v < 0.0))
            {
                return Err(
                    "reacting-flow iteration returned nonfinite or negative concentrations".into(),
                );
            }
            let numerator = state
                .iter()
                .zip(&values[field])
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>();
            let denominator = state.iter().map(|v| v * v).sum::<f64>();
            change = change.max((numerator / denominator).sqrt());
            next.push(state);
        }
        values = next;
        iterations = iteration;
        if change < 1e-12 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err("coupled Arrhenius species/heat iteration did not converge".into());
    }
    let reaction = loads(&mesh, &space, &values, &chemistry, control);
    let mut residual = 0.0_f64;
    let mut mass_balance = 0.0_f64;
    let mut heat_transfer = 0.0_f64;
    let scale = reaction
        .iter()
        .take(3)
        .flat_map(|r| r.iter())
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    for i in 0..count {
        mass_balance = mass_balance
            .max((reaction[0][i] + reaction[1][i] + 2.0 * reaction[2][i]).abs() / scale);
        heat_transfer =
            heat_transfer.max((reaction[3][i] - 5.0 * reaction[2][i]).abs() / (5.0 * scale));
    }
    for field in 0..4 {
        let action = raw[field].matvec(&values[field]);
        let rhs: Vec<_> = external[field]
            .iter()
            .zip(&reaction[field])
            .map(|(a, b)| a + b)
            .collect();
        let numerator = (0..count)
            .filter(|&i| !constrained[i])
            .map(|i| (action[i] - rhs[i]).powi(2))
            .sum::<f64>();
        let denominator = (0..count)
            .filter(|&i| !constrained[i])
            .map(|i| rhs[i].powi(2))
            .sum::<f64>();
        residual = residual.max((numerator / denominator).sqrt());
    }
    let coordinates = space
        .dof_coordinates(&mesh)
        .ok_or("reacting-flow H1 coordinates missing")?;
    let mut boundary_error = 0.0_f64;
    for &i in &boundary {
        let p = coordinates[i];
        let expected = reference(p[0], p[1], p[2]).0;
        for field in 0..4 {
            boundary_error = boundary_error.max((values[field][i] - expected[field]).abs());
        }
    }
    let mut errors = [0.0; 4];
    let mut norms = [0.0; 4];
    let mut grad_errors = [0.0; 4];
    let mut grad_norms = [0.0; 4];
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(&mesh, element);
        for qp in QuadratureRule::new(mesh.elements[element].element_type, 5).iter() {
            let tab = space.tabulate(&mesh, element, qp.xi(), qp.eta(), qp.zeta());
            let p = tab.physical_coords;
            let (expected, gradient) = reference(p[0], p[1], p[2]);
            let w = qp.weight * tab.det_j;
            for field in 0..4 {
                let actual: f64 = dofs
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| values[field][d] * tab.value(i))
                    .sum();
                errors[field] += w * (actual - expected[field]).powi(2);
                norms[field] += w * expected[field].powi(2);
                for axis in 0..3 {
                    let actual: f64 = dofs
                        .iter()
                        .enumerate()
                        .map(|(i, &d)| values[field][d] * tab.grad(i, axis))
                        .sum();
                    grad_errors[field] += w * (actual - gradient[field][axis]).powi(2);
                    grad_norms[field] += w * gradient[field][axis].powi(2);
                }
            }
        }
    }
    Ok(Metrics {
        species_error: (0..3)
            .map(|i| (errors[i] / norms[i]).sqrt())
            .fold(0.0_f64, f64::max),
        temperature_error: (errors[3] / norms[3]).sqrt(),
        species_gradient: (0..3)
            .map(|i| (grad_errors[i] / grad_norms[i]).sqrt())
            .fold(0.0_f64, f64::max),
        temperature_gradient: (grad_errors[3] / grad_norms[3]).sqrt(),
        residual,
        boundary: boundary_error,
        mass_balance,
        heat_transfer,
        iterations,
        free_dofs,
    })
}
fn run() -> Result<QaResult, String> {
    let mut levels = Vec::new();
    for n in [4, 8, 16] {
        let m = solve(n, PolynomialDegree::P1, Control::Full)?;
        println!("reacting-flow P1 n={n}: {m:?}");
        levels.push(m);
    }
    let patch = solve(2, PolynomialDegree::P2, Control::Full)?;
    println!("reacting-flow P2: {patch:?}");
    let mut control_errors = Vec::new();
    for control in [
        Control::NoReaction,
        Control::NoHeat,
        Control::FrozenTemperature,
        Control::NoAdvection,
    ] {
        let m = solve(2, PolynomialDegree::P2, control)?;
        println!("reacting-flow {control:?}: {m:?}");
        control_errors.push(m.species_error.max(m.temperature_error));
    }
    if !levels.windows(2).all(|p| {
        p[0].species_error > p[1].species_error
            && p[0].temperature_error > p[1].temperature_error
            && p[0].species_gradient > p[1].species_gradient
            && p[0].temperature_gradient > p[1].temperature_gradient
    }) {
        return Err("reacting-flow 3D fields and gradients do not contract".into());
    }
    let mut r = QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (name, value, bound) in [
        ("fine_species_error", levels[2].species_error, 0.005),
        (
            "fine_temperature_rise_error",
            levels[2].temperature_error,
            0.02,
        ),
        (
            "fine_species_gradient_error",
            levels[2].species_gradient,
            0.1,
        ),
        (
            "fine_temperature_gradient_error",
            levels[2].temperature_gradient,
            0.15,
        ),
        (
            "p2_exact_field_error",
            patch
                .species_error
                .max(patch.temperature_error)
                .max(patch.species_gradient)
                .max(patch.temperature_gradient),
            1e-9,
        ),
        (
            "physical_residual",
            levels
                .iter()
                .chain([&patch])
                .map(|m| m.residual)
                .fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "boundary_error",
            levels
                .iter()
                .chain([&patch])
                .map(|m| m.boundary)
                .fold(0.0_f64, f64::max),
            1e-10,
        ),
        (
            "reaction_mass_balance",
            levels
                .iter()
                .chain([&patch])
                .map(|m| m.mass_balance)
                .fold(0.0_f64, f64::max),
            1e-12,
        ),
        (
            "reaction_heat_transfer_error",
            levels
                .iter()
                .chain([&patch])
                .map(|m| m.heat_transfer)
                .fold(0.0_f64, f64::max),
            1e-12,
        ),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name}: {value:.3e}>{bound:.3e}"));
        }
        r = r.metric(name, value).maximum(name, bound);
    }
    for (name, value, bound) in [
        (
            "species_rate",
            (levels[1].species_error / levels[2].species_error).log2(),
            1.8,
        ),
        (
            "temperature_rate",
            (levels[1].temperature_error / levels[2].temperature_error).log2(),
            1.8,
        ),
        (
            "species_gradient_rate",
            (levels[1].species_gradient / levels[2].species_gradient).log2(),
            0.8,
        ),
        (
            "temperature_gradient_rate",
            (levels[1].temperature_gradient / levels[2].temperature_gradient).log2(),
            0.8,
        ),
        (
            "minimum_omitted_term_error",
            control_errors.into_iter().fold(f64::INFINITY, f64::min),
            1e-4,
        ),
    ] {
        if !value.is_finite() || value < bound {
            return Err(format!("{name}: {value:.3e}<{bound:.3e}"));
        }
        r = r.metric(name, value).minimum(name, bound);
    }
    Ok(r.metric("minimum_free_dofs",levels.iter().chain([&patch]).map(|m|m.free_dofs).min().unwrap_or(0) as f64).minimum("minimum_free_dofs",1.0)
        .metric("minimum_coupling_iterations",levels.iter().chain([&patch]).map(|m|m.iterations).min().unwrap_or(0) as f64).minimum("minimum_coupling_iterations",2.0)
        .metric("negative_controls",4.0).minimum("negative_controls",4.0)
        .seed("A+B-to-AB:Arrhenius20-E5000-Q5:3d-xyz:velocity0.3,-0.2,0.5:P1n4-8-16:P2n2")
        .input_checksum("native-3species-thermal-Arrhenius-feedback:4-controls:v1")
        .reference("independent polynomial strong ADR equations; concentrations and temperature vary in xyz; rate=20 exp(-5000/(R*T))*cA*cB, sources=(-r,-r,r), heat=5r")
        .reference_checksum("analytic:3d-reacting-species-heat-mms:v1")
        .dimension("3D unit cube, prescribed nonzero 3-component flow, three species and temperature with free interior DOFs")
        .formulation("production species/thermal diffusion/advection, recovered Arrhenius species and heat sources; coupled Picard sparse solves, P1 refinement and exact P2 fields")
        .tolerance("field rates>=1.8, gradient rates>=0.8; P2 fields/gradients<=1e-9; per-field residual<=1e-9; reaction mass/heat transfer<=1e-12; four missing-term controls"))
}
fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.reacting-flow-3d.v1",
        &SpeciesDiffusionForm {
            density_diffusivity: 1.0,
        },
        true,
    );
    let result = match run() {
        Ok(r) => r,
        Err(e) => QaResult::failed(CASE_ID, e),
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
        println!("FAILED");
        std::process::exit(1);
    }
}
