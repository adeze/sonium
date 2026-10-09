//! Release-facing matrix-free transmission contracts.

use num_complex::Complex64;
use sonium_bem::core::assembly::build_transmission_tbem_system;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::{
    PhysicsParams, check_transmission_passivity_contract, check_transmission_power_balance,
    check_transmission_reciprocity_contract, check_transmission_reciprocity_operator,
};
use sonium_qa::QaResult;
use sonium_solvers::traits::LinearOperator;
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-operator-contract.v1";
const FREQUENCY_HZ: f64 = 420.0;
const MAX_COMPRESSED_RECIPROCITY_ERROR: f64 = 5.0e-4;

struct PassiveOperator;

impl LinearOperator<Complex64> for PassiveOperator {
    fn num_rows(&self) -> usize {
        2
    }

    fn num_cols(&self) -> usize {
        2
    }

    fn apply_into(&self, input: &[Complex64], output: &mut [Complex64]) {
        output[0] = input[1];
        output[1] = Complex64::new(0.0, -1.0) * input[0];
    }

    fn apply_transpose_into(&self, input: &[Complex64], output: &mut [Complex64]) {
        output[0] = Complex64::new(0.0, -1.0) * input[1];
        output[1] = input[0];
    }
}

fn probes(dimension: usize) -> Vec<Vec<Complex64>> {
    (0..3)
        .map(|seed| {
            (0..dimension)
                .map(|index| {
                    let phase = (index + 1 + seed * 5) as f64;
                    Complex64::new(0.15 + 0.013 * phase, -0.2 + 0.009 * phase)
                })
                .collect()
        })
        .collect()
}

fn run() -> Result<(f64, f64, f64, f64), String> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let exterior = PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false)
        .map_err(|error| format!("exterior medium: {error}"))?;
    let interior = PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
        .map_err(|error| format!("interior medium: {error}"))?;
    let system =
        build_transmission_tbem_system(&mesh.elements, &mesh.nodes, exterior.clone(), interior)
            .map_err(|error| format!("transmission assembly: {error}"))?;
    let test_probes = probes(system.total_dofs());
    let dense = check_transmission_reciprocity_contract(
        system.matrix(),
        system.interface_dofs(),
        exterior.omega,
        exterior.harmonic_factor,
        &test_probes,
        1.0e-6,
    )
    .map_err(|error| format!("dense reciprocity: {error:?}"))?;
    let compressed = system
        .compress(8, 1.0e-6, 8)
        .map_err(|error| format!("transmission compression: {error}"))?;
    let compressed_report = check_transmission_reciprocity_operator(
        &compressed,
        system.interface_dofs(),
        exterior.omega,
        exterior.harmonic_factor,
        &test_probes,
        MAX_COMPRESSED_RECIPROCITY_ERROR,
    )
    .map_err(|error| format!("compressed reciprocity: {error:?}"))?;

    let passivity =
        check_transmission_passivity_contract(&PassiveOperator, 1, 1.0, 1.0, &probes(2), 1.0e-12)
            .map_err(|error| format!("weighted passivity: {error:?}"))?;
    let pressure = [Complex64::new(2.0, 1.0), Complex64::new(1.0, -0.5)];
    let velocity = [Complex64::new(0.1, -0.2), Complex64::new(0.2, 0.1)];
    let power = check_transmission_power_balance(
        &[0.5, 0.75],
        &pressure,
        &velocity,
        &pressure,
        &velocity,
        1.0e-12,
    )
    .map_err(|error| format!("power balance: {error:?}"))?;
    Ok((
        dense.relative_error,
        compressed_report.relative_error,
        passivity.minimum_power,
        power.relative_error,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((dense_error, compressed_error, minimum_power, power_error)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("dense_reciprocity_error", dense_error)
            .metric("compressed_reciprocity_error", compressed_error)
            .metric("transmission_passivity_minimum_power", minimum_power)
            .metric("transmission_power_balance_error", power_error)
            .minimum("transmission_passivity_minimum_power", 0.0)
            .maximum("dense_reciprocity_error", 1.0e-6)
            .maximum("compressed_reciprocity_error", MAX_COMPRESSED_RECIPROCITY_ERROR)
            .maximum("transmission_power_balance_error", 1.0e-12)
            .input_checksum("bem:icosphere:subdivision-0:constant-transmission")
            .reference("independent dense and matrix-free weighted transmission contracts")
            .reference_checksum("analytic/deterministic-fixture:bem-transmission-operator-contract-v1")
            .dimension("3-D CPU; 0.2 m 20-panel two-trace compressed operator")
            .formulation("dense Müller plus block-ACA matrix-free reciprocity, weighted passivity, and panel power balance")
            .tolerance("dense <= 1e-6; compressed <= 5e-4; power <= 1e-12")
            .seed("bem-transmission-operator-contract:icosphere-level-0:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .input_checksum("bem:icosphere:subdivision-0:constant-transmission")
            .reference_checksum("analytic/deterministic-fixture:bem-transmission-operator-contract-v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
