//! Finite-element counterpart of the conducting solid in Wolfram's published
//! Joule-heating model collection.
//!
//! Source:
//! <https://reference.wolfram.com/language/PDEModels/tutorial/Multiphysics/ModelCollection/JouleHeating.html?view=all>
//!
//! The unit-square conductor has fixed voltages on its left and right faces,
//! electrically and thermally insulated horizontal faces, and fixed end
//! temperatures.  Sonium first solves the electrical conduction problem,
//! obtains current and power from the unconstrained electrical residual, then
//! feeds the resulting `I^2 R / volume` source into the thermal FEM solve.

use serde::Deserialize;
use sonium_fem::{
    assembly::{assemble_bilinear, assemble_linear},
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_dirichlet_values},
    mesh::generators::unit_square_triangles,
    physics::thermal::heat_conduction::{ConductionForm, HeatSourceForm, JouleHeatingSource},
    spaces::H1Space,
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{GmresConfig, gmres};
use std::{env, path::PathBuf, process::Command};

const CELLS_PER_SIDE: usize = 24;
const VOLTAGE_V: f64 = 2.0;
const ELECTRICAL_CONDUCTIVITY_S_M: f64 = 5.0;
const THERMAL_CONDUCTIVITY_W_M_K: f64 = 10.0;
const END_TEMPERATURE_K: f64 = 300.0;
const SAMPLE_X_M: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

struct SoniumSolution {
    potential_v: Vec<f64>,
    temperature_k: Vec<f64>,
    current_a_per_m: f64,
    electrical_power_w_per_m: f64,
    thermal_outflow_w_per_m: f64,
}

fn solve_system(system: &FemSystem<f64>, name: &str) -> Vec<f64> {
    let result = gmres(
        &system.matrix.to_csr(),
        &system.rhs,
        &GmresConfig {
            max_iterations: 2_000,
            restart: 50,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "{name} solve did not converge");
    result.x
}

fn sample_centerline(field: &[f64]) -> Vec<f64> {
    let iy = CELLS_PER_SIDE / 2;
    SAMPLE_X_M
        .iter()
        .map(|&x| {
            let ix = (x * CELLS_PER_SIDE as f64).round() as usize;
            field[iy * (CELLS_PER_SIDE + 1) + ix]
        })
        .collect()
}

fn sonium_solution() -> SoniumSolution {
    let mesh = unit_square_triangles(CELLS_PER_SIDE);
    let space = H1Space::new(PolynomialDegree::P1);
    let zero_rhs = vec![0.0; (CELLS_PER_SIDE + 1).pow(2)];

    // Electrical conduction: -div(sigma grad(phi)) = 0.  The thermal
    // conduction form is the public weighted-Laplacian adapter available for
    // scalar conductivity problems.
    let electrical_matrix = assemble_bilinear(
        &ConductionForm::from_conductivity(ELECTRICAL_CONDUCTIVITY_S_M),
        &space,
        &space,
        &mesh,
        2,
    );
    let electrical_csr = electrical_matrix.to_csr();
    let mut electrical_system = FemSystem::new(electrical_matrix, zero_rhs);
    apply_dirichlet_values(
        &mut electrical_system,
        &mesh,
        &space,
        1,
        &|_, _, _| 0.0,
        None,
    );
    apply_dirichlet_values(
        &mut electrical_system,
        &mesh,
        &space,
        2,
        &|_, _, _| VOLTAGE_V,
        None,
    );
    let potential = solve_system(&electrical_system, "electrical");

    // K phi is the unconstrained boundary-current residual.  Summing the
    // right electrode gives current per unit out-of-plane depth.
    let electrical_reaction = electrical_csr.matvec(&potential);
    let right_current = (0..=CELLS_PER_SIDE)
        .map(|iy| electrical_reaction[iy * (CELLS_PER_SIDE + 1) + CELLS_PER_SIDE])
        .sum::<f64>();
    let electrical_power = VOLTAGE_V * right_current;
    let resistance = VOLTAGE_V / right_current;
    let joule_source = JouleHeatingSource::new(resistance, right_current, 1.0);
    let heat_density = joule_source.heat_generation_rate();
    assert!((heat_density - electrical_power).abs() < 1.0e-10);

    // Thermal conduction: -div(k grad(T)) = sigma |grad(phi)|^2.  For this
    // homogeneous bar the solved current produces a uniform volumetric source.
    let thermal_matrix = assemble_bilinear(
        &ConductionForm::from_conductivity(THERMAL_CONDUCTIVITY_W_M_K),
        &space,
        &space,
        &mesh,
        2,
    );
    let thermal_csr = thermal_matrix.to_csr();
    let source = HeatSourceForm {
        source_fn: move |_, _, _| heat_density,
    };
    let thermal_rhs = assemble_linear(&source, &space, &mesh, 2);
    let unconstrained_thermal_rhs = thermal_rhs.clone();
    let mut thermal_system = FemSystem::new(thermal_matrix, thermal_rhs);
    for marker in [1, 2] {
        apply_dirichlet_values(
            &mut thermal_system,
            &mesh,
            &space,
            marker,
            &|_, _, _| END_TEMPERATURE_K,
            None,
        );
    }
    let temperature = solve_system(&thermal_system, "thermal");

    let thermal_internal = thermal_csr.matvec(&temperature);
    let thermal_outflow = -(0..thermal_internal.len())
        .map(|i| thermal_internal[i] - unconstrained_thermal_rhs[i])
        .sum::<f64>();

    SoniumSolution {
        potential_v: sample_centerline(&potential),
        temperature_k: sample_centerline(&temperature),
        current_a_per_m: right_current,
        electrical_power_w_per_m: electrical_power,
        thermal_outflow_w_per_m: thermal_outflow,
    }
}

fn exact_potential(x: f64) -> f64 {
    VOLTAGE_V * x
}

fn exact_temperature(x: f64) -> f64 {
    let heat_density = ELECTRICAL_CONDUCTIVITY_S_M * VOLTAGE_V.powi(2);
    END_TEMPERATURE_K + heat_density * x * (1.0 - x) / (2.0 * THERMAL_CONDUCTIVITY_W_M_K)
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    sample_x_m: Vec<f64>,
    potential_v: Vec<f64>,
    temperature_k: Vec<f64>,
    current_a_per_m: f64,
    electrical_power_w_per_m: f64,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/joule_heating_bar.wls");
    let output = match Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Wolfram Engine not available; using analytical Joule-heating reference");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram Joule-heating reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram Joule-heating output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

fn assert_close(name: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() < tolerance,
            "{name} sample {index}: Sonium={got:.12e}, reference={want:.12e}"
        );
    }
}

#[test]
fn sonium_reproduces_wolfram_joule_heated_conductor_and_power_balance() {
    let sonium = sonium_solution();
    let exact_potential = SAMPLE_X_M.map(exact_potential);
    let exact_temperature = SAMPLE_X_M.map(exact_temperature);
    let exact_current = ELECTRICAL_CONDUCTIVITY_S_M * VOLTAGE_V;
    let exact_power = exact_current * VOLTAGE_V;

    assert_close("potential", &sonium.potential_v, &exact_potential, 1.0e-10);
    assert_close(
        "temperature",
        &sonium.temperature_k,
        &exact_temperature,
        5.0e-4,
    );
    assert!((sonium.current_a_per_m - exact_current).abs() < 1.0e-9);
    assert!((sonium.electrical_power_w_per_m - exact_power).abs() < 1.0e-9);
    assert!(
        (sonium.thermal_outflow_w_per_m - sonium.electrical_power_w_per_m).abs() < 1.0e-8,
        "electrical input {} W/m != thermal outflow {} W/m",
        sonium.electrical_power_w_per_m,
        sonium.thermal_outflow_w_per_m
    );

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.sample_x_m, SAMPLE_X_M);
        assert_close(
            "Wolfram potential",
            &sonium.potential_v,
            &reference.potential_v,
            1.0e-10,
        );
        assert_close(
            "Wolfram temperature",
            &sonium.temperature_k,
            &reference.temperature_k,
            5.0e-4,
        );
        assert!((sonium.current_a_per_m - reference.current_a_per_m).abs() < 1.0e-9);
        assert!(
            (sonium.electrical_power_w_per_m - reference.electrical_power_w_per_m).abs() < 1.0e-9
        );
    }
}
