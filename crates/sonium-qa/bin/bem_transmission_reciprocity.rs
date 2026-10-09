//! Release-facing weighted reciprocity contract for a dense Müller transmission matrix.

use num_complex::Complex64;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::{
    PhysicsParams, build_transmission_tbem_system, check_transmission_reciprocity_contract,
};
use sonium_qa::QaResult;
use std::panic::catch_unwind;
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-reciprocity.v1";
const FREQUENCY_HZ: f64 = 420.0;
const MAX_RELATIVE_ERROR: f64 = 1.0e-6;

fn run() -> Result<(usize, f64), String> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let exterior = PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false)
        .map_err(|error| format!("exterior medium: {error}"))?;
    let interior = PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
        .map_err(|error| format!("interior medium: {error}"))?;
    let system =
        build_transmission_tbem_system(&mesh.elements, &mesh.nodes, exterior.clone(), interior)
            .map_err(|error| format!("transmission assembly: {error}"))?;

    let probes = (0..3)
        .map(|seed| {
            (0..system.total_dofs())
                .map(|index| {
                    let phase = (index + 1 + seed * 3) as f64;
                    Complex64::new(0.15 + 0.03 * phase, -0.2 + 0.017 * (phase + seed as f64))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let report = check_transmission_reciprocity_contract(
        system.matrix(),
        system.interface_dofs(),
        exterior.omega,
        exterior.harmonic_factor,
        &probes,
        MAX_RELATIVE_ERROR,
    )
    .map_err(|error| format!("weighted reciprocity contract: {error}"))?;
    if !report.relative_error.is_finite() || report.relative_error > MAX_RELATIVE_ERROR {
        return Err(format!(
            "weighted reciprocity error {:.3e} exceeds {:.3e}",
            report.relative_error, MAX_RELATIVE_ERROR
        ));
    }
    Ok((system.interface_dofs(), report.relative_error))
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(run) {
        Ok(Ok((interface_dofs, relative_error))) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("interface_dofs", interface_dofs as f64)
            .metric("probe_count", 3.0)
            .metric("pair_count", 3.0)
            .metric("weighted_reciprocity_relative_error", relative_error)
            .maximum("weighted_reciprocity_relative_error", MAX_RELATIVE_ERROR)
            .input_checksum("bem:icosphere:subdivision-0:constant-transmission")
            .reference_checksum("analytic/deterministic-fixture:bem-transmission-reciprocity-v1")
            .reference("weighted transpose identity J A = (J A)^T")
            .dimension("3-D CPU; radius 0.2 m; 20-panel constant Müller transmission")
            .formulation("dense two-trace Müller transmission matrix")
            .tolerance("weighted reciprocity relative error <= 1e-6")
            .seed("bem-transmission-reciprocity:icosphere-level-0:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .input_checksum("bem:icosphere:subdivision-0:constant-transmission")
            .reference_checksum("analytic/deterministic-fixture:bem-transmission-reciprocity-v1")
            .dimension("3-D CPU; radius 0.2 m; 20-panel constant Müller transmission")
            .formulation("dense two-trace Müller transmission matrix")
            .tolerance("weighted reciprocity relative error <= 1e-6")
            .seed("bem-transmission-reciprocity:icosphere-level-0:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(_) => QaResult::failed(CASE_ID, "panic while checking transmission reciprocity")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
