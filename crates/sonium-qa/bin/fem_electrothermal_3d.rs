//! Constant-temperature form contract and solved nonlinear 3D electrothermal MMS.

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_dirichlet_values};
use sonium_fem::mesh::{Mesh, unit_cube_tetrahedra};
use sonium_fem::physics::coupled::electrothermal::{
    DcCurrentForm, DcJouleHeatForm, ElectrothermalMaterial,
};
use sonium_fem::physics::thermal::heat_conduction::{ConductionForm, HeatSourceForm};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::SparseLuFactorization;
use std::collections::HashMap;

const CASE_ID: &str = "fem.electrothermal-3d-mms.v1";
const ORDER: usize = 4;
const MATERIAL: ElectrothermalMaterial = ElectrothermalMaterial {
    reference_electrical_conductivity: 2.0,
    resistivity_temperature_coefficient: 0.02,
    reference_temperature: 293.15,
    relative_permeability: 1.0,
    thermal_conductivity: 1.5,
    density: 1.0,
    specific_heat: 1.0,
};

fn reference(x: f64, y: f64, z: f64) -> (f64, [f64; 3], f64, [f64; 3]) {
    (
        1.0 + x + 2.0 * y + 3.0 * z + x * y + y * z + 2.0 * x * z,
        [1.0 + y + 2.0 * z, 2.0 + x + z, 3.0 + y + 2.0 * x],
        10.0 * (x * (1.0 - x) + 2.0 * y * (1.0 - y) + 3.0 * z * (1.0 - z)),
        [
            10.0 * (1.0 - 2.0 * x),
            20.0 * (1.0 - 2.0 * y),
            30.0 * (1.0 - 2.0 * z),
        ],
    )
}

// Independent analytic source: both fields depend on z. No reference nodal
// vector is multiplied by a production operator to manufacture the load.
fn electrical_source(x: f64, y: f64, z: f64) -> f64 {
    let (_, dv, theta, dt) = reference(x, y, z);
    let denominator = 1.0 + 0.02 * theta;
    2.0 * 0.02 * dt.iter().zip(dv).map(|(a, b)| a * b).sum::<f64>() / denominator.powi(2)
}
fn thermal_source(x: f64, y: f64, z: f64) -> f64 {
    let (_, dv, theta, _) = reference(x, y, z);
    // -k*Laplacian(T)=120*k, less the independently evaluated exact Joule heat.
    120.0 * 1.5 - 2.0 / (1.0 + 0.02 * theta) * dv.iter().map(|v| v * v).sum::<f64>()
}

// Production forms accept point callbacks. Tabulate the actual FE state at
// exactly their quadrature points, so no analytic field is used as feedback.
// Bit keys come from the same deterministic space.tabulate operation used by
// assemble_bilinear/linear; all quadrature points are interior to their cells.
#[derive(Debug)]
struct Snapshot(HashMap<[u64; 3], [f64; 4]>);
impl Snapshot {
    fn build(mesh: &Mesh, space: &H1Space, theta: &[f64], voltage: &[f64]) -> Self {
        let mut points = HashMap::new();
        for element in 0..mesh.num_elements() {
            let dofs = space.local_to_global(mesh, element);
            for qp in QuadratureRule::new(mesh.elements[element].element_type, ORDER).iter() {
                let tab = space.tabulate(mesh, element, qp.xi(), qp.eta(), qp.zeta());
                let mut state = [MATERIAL.reference_temperature, 0.0, 0.0, 0.0];
                for (i, &dof) in dofs.iter().enumerate() {
                    state[0] += theta[dof] * tab.value(i);
                    for axis in 0..3 {
                        state[axis + 1] += voltage[dof] * tab.grad(i, axis);
                    }
                }
                points.insert(tab.physical_coords.map(f64::to_bits), state);
            }
        }
        Self(points)
    }
    fn at(&self, x: f64, y: f64, z: f64) -> [f64; 4] {
        *self
            .0
            .get(&[x.to_bits(), y.to_bits(), z.to_bits()])
            .expect("production assembly must use the declared quadrature snapshot")
    }
}

fn constrained(
    matrix: TripletMatrix<f64>,
    rhs: Vec<f64>,
    mesh: &Mesh,
    space: &H1Space,
    thermal: bool,
) -> Result<FemSystem<f64>, String> {
    let mut system = FemSystem::new(matrix, rhs);
    try_apply_dirichlet_values(
        &mut system,
        mesh,
        space,
        1,
        &|x, y, z| {
            let (v, _, theta, _) = reference(x, y, z);
            if thermal { theta } else { v }
        },
        None,
    )
    .map_err(|e| e.to_string())?;
    Ok(system)
}

#[derive(Debug)]
struct Metrics {
    voltage: f64,
    temperature: f64,
    voltage_gradient: f64,
    temperature_gradient: f64,
    current_residual: f64,
    heat_residual: f64,
    boundary: f64,
    joule_power_error: f64,
    electrical_balance: f64,
    thermal_balance: f64,
    iterations: usize,
    free_dofs: usize,
}

fn solve(n: usize, feedback: bool, joule: bool) -> Result<Metrics, String> {
    let mut mesh = unit_cube_tetrahedra(n);
    for face in &mut mesh.boundaries {
        face.marker = 1;
    }
    let space = H1Space::new(PolynomialDegree::P1);
    let count = mesh.num_nodes();
    let boundary = space.boundary_dofs(&mesh, 1);
    let mut is_boundary = vec![false; count];
    for &i in &boundary {
        is_boundary[i] = true;
    }
    let free_dofs = count - boundary.len();
    if free_dofs == 0 {
        return Err("electrothermal case has no free unknowns".into());
    }
    let electrical_rhs = assemble_linear(
        &HeatSourceForm {
            source_fn: electrical_source,
        },
        &space,
        &mesh,
        ORDER,
    );
    let external_heat = assemble_linear(
        &HeatSourceForm {
            source_fn: thermal_source,
        },
        &space,
        &mesh,
        ORDER,
    );
    let thermal_matrix = assemble_bilinear(
        &ConductionForm::from_conductivity(MATERIAL.thermal_conductivity),
        &space,
        &space,
        &mesh,
        ORDER,
    );
    let thermal_raw = thermal_matrix.to_csr();
    let thermal_factor = SparseLuFactorization::factorize(
        &constrained(
            thermal_matrix.clone(),
            vec![0.0; count],
            &mesh,
            &space,
            true,
        )?
        .matrix
        .to_csr(),
    )
    .map_err(|e| e.to_string())?;
    let mut voltage = vec![0.0; count];
    let mut theta = vec![0.0; count];
    let mut iterations = 0;
    let mut converged = false;
    for iteration in 1..=40 {
        let snapshot = Snapshot::build(&mesh, &space, &theta, &voltage);
        let temperature = |x, y, z| {
            if feedback {
                snapshot.at(x, y, z)[0]
            } else {
                MATERIAL.reference_temperature
            }
        };
        let matrix = assemble_bilinear(
            &DcCurrentForm {
                material: MATERIAL,
                temperature,
            },
            &space,
            &space,
            &mesh,
            ORDER,
        );
        let system = constrained(matrix, electrical_rhs.clone(), &mesh, &space, false)?;
        let new_voltage = SparseLuFactorization::factorize(&system.matrix.to_csr())
            .and_then(|f| f.solve(&system.rhs))
            .map_err(|e| e.to_string())?;
        let snapshot = Snapshot::build(&mesh, &space, &theta, &new_voltage);
        let heat = if joule {
            assemble_linear(
                &DcJouleHeatForm {
                    material: MATERIAL,
                    temperature: |x, y, z| {
                        if feedback {
                            snapshot.at(x, y, z)[0]
                        } else {
                            MATERIAL.reference_temperature
                        }
                    },
                    potential_gradient: |x, y, z| {
                        let s = snapshot.at(x, y, z);
                        [s[1], s[2], s[3]]
                    },
                },
                &space,
                &mesh,
                ORDER,
            )
        } else {
            vec![0.0; count]
        };
        let rhs: Vec<f64> = external_heat
            .iter()
            .zip(&heat)
            .map(|(a, b)| a + b)
            .collect();
        let system = constrained(thermal_matrix.clone(), rhs, &mesh, &space, true)?;
        let new_theta = thermal_factor
            .solve(&system.rhs)
            .map_err(|e| e.to_string())?;
        if new_theta.iter().chain(&new_voltage).any(|v| !v.is_finite()) {
            return Err("nonfinite coupled electrothermal state".into());
        }
        let change_t = new_theta
            .iter()
            .zip(&theta)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
            / new_theta.iter().map(|v| v * v).sum::<f64>().sqrt();
        let change_v = new_voltage
            .iter()
            .zip(&voltage)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
            / new_voltage.iter().map(|v| v * v).sum::<f64>().sqrt();
        theta = new_theta;
        voltage = new_voltage;
        iterations = iteration;
        if change_t.max(change_v) < 1e-11 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err("electrothermal Picard feedback did not converge".into());
    }
    // Reassemble at the final temperature, rather than reusing the lagged
    // coefficient from the last Picard subsolve, and check both original PDEs.
    let snapshot = Snapshot::build(&mesh, &space, &theta, &voltage);
    let current_raw = assemble_bilinear(
        &DcCurrentForm {
            material: MATERIAL,
            temperature: |x, y, z| {
                if feedback {
                    snapshot.at(x, y, z)[0]
                } else {
                    MATERIAL.reference_temperature
                }
            },
        },
        &space,
        &space,
        &mesh,
        ORDER,
    )
    .to_csr();
    let heat = if joule {
        assemble_linear(
            &DcJouleHeatForm {
                material: MATERIAL,
                temperature: |x, y, z| {
                    if feedback {
                        snapshot.at(x, y, z)[0]
                    } else {
                        MATERIAL.reference_temperature
                    }
                },
                potential_gradient: |x, y, z| {
                    let s = snapshot.at(x, y, z);
                    [s[1], s[2], s[3]]
                },
            },
            &space,
            &mesh,
            ORDER,
        )
    } else {
        vec![0.0; count]
    };
    let heat_rhs: Vec<f64> = external_heat
        .iter()
        .zip(&heat)
        .map(|(a, b)| a + b)
        .collect();
    let current_action = current_raw.matvec(&voltage);
    let thermal_action = thermal_raw.matvec(&theta);
    let residual = |action: &[f64], rhs: &[f64]| {
        (action
            .iter()
            .zip(rhs)
            .enumerate()
            .filter(|(i, _)| !is_boundary[*i])
            .map(|(_, (&a, &b))| (a - b).powi(2))
            .sum::<f64>()
            / rhs
                .iter()
                .enumerate()
                .filter(|(i, _)| !is_boundary[*i])
                .map(|(_, v)| v * v)
                .sum::<f64>())
        .sqrt()
    };
    let current_residual = residual(&current_action, &electrical_rhs);
    let heat_residual = residual(&thermal_action, &heat_rhs);
    let joule_power = heat.iter().sum::<f64>();
    let electrical_input = boundary
        .iter()
        .map(|&i| voltage[i] * (current_action[i] - electrical_rhs[i]))
        .sum::<f64>()
        + voltage
            .iter()
            .zip(&electrical_rhs)
            .map(|(a, b)| a * b)
            .sum::<f64>();
    let thermal_outflow = -boundary
        .iter()
        .map(|&i| thermal_action[i] - heat_rhs[i])
        .sum::<f64>();
    let thermal_input = heat_rhs.iter().sum::<f64>();
    let electrical_balance = if joule {
        (electrical_input - joule_power).abs() / joule_power
    } else {
        0.0
    };
    let thermal_balance = (thermal_input - thermal_outflow).abs() / thermal_input.abs();
    let mut errors = [0.0; 4];
    let mut norms = [0.0; 4];
    let mut expected_power = 0.0;
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(&mesh, element);
        for qp in QuadratureRule::new(mesh.elements[element].element_type, 5).iter() {
            let tab = space.tabulate(&mesh, element, qp.xi(), qp.eta(), qp.zeta());
            let [x, y, z] = tab.physical_coords;
            let (v, dv, t, dt) = reference(x, y, z);
            let weight = qp.weight * tab.det_j;
            let actual_v: f64 = dofs
                .iter()
                .enumerate()
                .map(|(i, &d)| voltage[d] * tab.value(i))
                .sum();
            let actual_t: f64 = dofs
                .iter()
                .enumerate()
                .map(|(i, &d)| theta[d] * tab.value(i))
                .sum();
            errors[0] += weight * (actual_v - v).powi(2);
            norms[0] += weight * v * v;
            errors[1] += weight * (actual_t - t).powi(2);
            norms[1] += weight * t * t;
            for axis in 0..3 {
                let gv: f64 = dofs
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| voltage[d] * tab.grad(i, axis))
                    .sum();
                let gt: f64 = dofs
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| theta[d] * tab.grad(i, axis))
                    .sum();
                errors[2] += weight * (gv - dv[axis]).powi(2);
                norms[2] += weight * dv[axis].powi(2);
                errors[3] += weight * (gt - dt[axis]).powi(2);
                norms[3] += weight * dt[axis].powi(2);
            }
            expected_power +=
                weight * 2.0 / (1.0 + 0.02 * t) * dv.iter().map(|g| g * g).sum::<f64>();
        }
    }
    let boundary_error = boundary
        .iter()
        .map(|&i| {
            let p = mesh.nodes[i];
            let (v, _, t, _) = reference(p.x, p.y, p.z);
            (voltage[i] - v).abs().max((theta[i] - t).abs())
        })
        .fold(0.0_f64, f64::max);
    Ok(Metrics {
        voltage: (errors[0] / norms[0]).sqrt(),
        temperature: (errors[1] / norms[1]).sqrt(),
        voltage_gradient: (errors[2] / norms[2]).sqrt(),
        temperature_gradient: (errors[3] / norms[3]).sqrt(),
        current_residual,
        heat_residual,
        boundary: boundary_error,
        joule_power_error: (joule_power - expected_power).abs() / expected_power,
        electrical_balance,
        thermal_balance,
        iterations,
        free_dofs,
    })
}

fn run() -> Result<QaResult, String> {
    let mut levels = Vec::new();
    for n in [4, 8, 16] {
        let m = solve(n, true, true)?;
        println!("electrothermal n={n}: {m:?}");
        levels.push(m);
    }
    if !levels.windows(2).all(|p| {
        p[0].voltage > p[1].voltage
            && p[0].temperature > p[1].temperature
            && p[0].voltage_gradient > p[1].voltage_gradient
            && p[0].temperature_gradient > p[1].temperature_gradient
    }) {
        return Err("coupled 3D field/gradient errors do not contract".into());
    }
    let frozen = solve(8, false, true)?;
    let unheated = solve(8, true, false)?;
    println!("electrothermal frozen-conductivity control: {frozen:?}");
    println!("electrothermal omitted-Joule control: {unheated:?}");
    let mut r = QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (name, value, bound) in [
        ("fine_voltage_error", levels[2].voltage, 0.005),
        ("fine_temperature_rise_error", levels[2].temperature, 0.02),
        (
            "fine_voltage_gradient_error",
            levels[2].voltage_gradient,
            0.06,
        ),
        (
            "fine_temperature_gradient_error",
            levels[2].temperature_gradient,
            0.15,
        ),
        ("fine_joule_power_error", levels[2].joule_power_error, 0.005),
        (
            "current_residual",
            levels
                .iter()
                .map(|m| m.current_residual)
                .fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "heat_residual",
            levels
                .iter()
                .map(|m| m.heat_residual)
                .fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "boundary_error",
            levels.iter().map(|m| m.boundary).fold(0.0_f64, f64::max),
            1e-10,
        ),
        (
            "electrical_power_balance",
            levels
                .iter()
                .map(|m| m.electrical_balance)
                .fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "thermal_power_balance",
            levels
                .iter()
                .map(|m| m.thermal_balance)
                .fold(0.0_f64, f64::max),
            1e-9,
        ),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name}: {value:.3e}>{bound:.3e}"));
        }
        r = r.metric(name, value).maximum(name, bound);
    }
    for (name, value, bound) in [
        (
            "voltage_rate",
            (levels[1].voltage / levels[2].voltage).log2(),
            1.8,
        ),
        (
            "temperature_rate",
            (levels[1].temperature / levels[2].temperature).log2(),
            1.8,
        ),
        (
            "voltage_gradient_rate",
            (levels[1].voltage_gradient / levels[2].voltage_gradient).log2(),
            0.8,
        ),
        (
            "temperature_gradient_rate",
            (levels[1].temperature_gradient / levels[2].temperature_gradient).log2(),
            0.8,
        ),
        ("frozen_feedback_voltage_error", frozen.voltage, 0.001),
        (
            "frozen_feedback_error_ratio",
            frozen.voltage / levels[1].voltage,
            1.5,
        ),
        (
            "omitted_joule_temperature_error",
            unheated.temperature,
            0.01,
        ),
        (
            "omitted_joule_error_ratio",
            unheated.temperature / levels[1].temperature,
            4.0,
        ),
    ] {
        if !value.is_finite() || value < bound {
            return Err(format!("{name}: {value:.3e}<{bound:.3e}"));
        }
        r = r.metric(name, value).minimum(name, bound);
    }
    Ok(r.metric("minimum_picard_iterations",levels.iter().map(|m|m.iterations).min().unwrap_or(0) as f64)
        .minimum("minimum_picard_iterations",2.0)
        .metric("minimum_free_dofs",levels.iter().map(|m|m.free_dofs).min().unwrap_or(0) as f64).minimum("minimum_free_dofs",1.0)
        .seed("coupled-3d-xyz:alpha0.02:sigma2:k1.5:n4-8-16:feedback-controls")
        .input_checksum("native-current-joule-conduction:picard:quadrature4:v1")
        .reference("independent strong current/heat equations; mixed xyz voltage and quadratic xyz temperature; sigma=2/(1+0.02*(T-293.15))")
        .reference_checksum("analytic:3d-temperature-feedback-joule-mms:v1")
        .dimension("3D unit cube; free interior current and temperature; nonzero z gradients and temperature rise")
        .formulation("production DcCurrentForm/DcJouleHeatForm/ConductionForm; actual FE quadrature-state feedback; converged Picard and original-equation residuals")
        .tolerance("P1 value rates>=1.8, gradients>=0.8; fine temperature rise<=2%; voltage/power<=0.5%; current/heat residual and power balances<=1e-9"))
}

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.electrothermal-3d.v1",
        &DcCurrentForm {
            material: MATERIAL,
            temperature: |_, _, _| MATERIAL.reference_temperature,
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
