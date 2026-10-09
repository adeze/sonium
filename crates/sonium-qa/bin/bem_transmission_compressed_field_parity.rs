//! Checked forward/adjoint field parity for compressed transmission solves.

use num_complex::Complex64;
use sonium_bem::core::assembly::build_transmission_tbem_system_with_continuous_quadratic;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::solver::transmission::{
    solve_incident_transmission, solve_transmission_adjoint_system,
};
use sonium_bem::core::types::PhysicsParams;
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::GmresConfig;
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-compressed-field-parity.v1";

fn relative_l2(actual: &[Complex64], reference: &[Complex64]) -> f64 {
    let numerator = actual
        .iter()
        .zip(reference)
        .map(|(value, expected)| (*value - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = reference
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(1.0e-15);
    numerator / denominator
}

fn run() -> Result<(usize, f64, f64, f64, f64), String> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let exterior = PhysicsParams::new(42.0, 343.0, 1.21, false);
    let interior = PhysicsParams::new(42.0, 343.0, 1.21, true);
    let system = build_transmission_tbem_system_with_continuous_quadratic(
        &mesh.elements,
        &mesh.nodes,
        exterior,
        interior,
    )
    .map_err(|error| error.to_string())?;
    let compressed = system
        .compress(16, 1.0e-3, 32)
        .map_err(|error| error.to_string())?;
    let dimension = system.total_dofs();
    let mut nearfield = system.matrix().clone();
    for index in 0..dimension {
        nearfield[[index, index]] += Complex64::new(1.0, 0.0);
    }
    let mut adjoint_nearfield = DenseMatrix::zeros(dimension, dimension);
    for row in 0..dimension {
        for column in 0..dimension {
            adjoint_nearfield[[row, column]] = system.matrix()[[column, row]].conj();
        }
        adjoint_nearfield[[row, row]] += Complex64::new(1.0, 0.0);
    }
    let config = GmresConfig::<f64> {
        max_iterations: 120,
        restart: 32,
        tolerance: 1.0e-10,
        exact_residual_check_interval: 1,
        ..GmresConfig::default()
    };
    let incident = IncidentField::plane_wave_z();
    let dense_forward = solve_incident_transmission(&system, &incident, 1.0e-8)
        .map_err(|error| error.to_string())?;
    let rhs = system
        .incident_rhs(&incident)
        .map_err(|error| error.to_string())?;
    let compressed_forward = compressed
        .solve_with_ilu(&rhs, &nearfield, &config, 1.0e-8)
        .map_err(|error| error.to_string())?;
    let points = [[0.0, 0.0, 0.8], [0.2, 0.0, 0.7]];
    let dense_field = dense_forward
        .evaluate_exterior_field_checked(&system, &points, &incident)
        .map_err(|error| error.to_string())?;
    let compressed_field = compressed_forward
        .evaluate_exterior_field_checked(&system, &points, &incident)
        .map_err(|error| error.to_string())?;
    let dense_pressure: Vec<_> = dense_field.iter().map(|field| field.p_total).collect();
    let compressed_pressure: Vec<_> = compressed_field.iter().map(|field| field.p_total).collect();
    let forward_error = relative_l2(&compressed_pressure, &dense_pressure);
    if compressed_forward
        .evaluate_exterior_field_checked(&system, &[[f64::NAN, 0.0, 0.8]], &incident)
        .is_ok()
    {
        return Err("compressed forward field accepted a non-finite point".into());
    }

    let adjoint_rhs = (0..dimension)
        .map(|index| Complex64::new(0.17 + index as f64 / 31.0, -0.09 + index as f64 / 43.0))
        .collect::<Vec<_>>();
    let dense_adjoint = solve_transmission_adjoint_system(&system, &adjoint_rhs, 1.0e-8)
        .map_err(|error| error.to_string())?;
    let compressed_adjoint = compressed
        .solve_hermitian_with_ilu(&adjoint_rhs, &adjoint_nearfield, &config, 2.0e-6)
        .map_err(|error| error.to_string())?;
    let dense_adjoint_field = dense_adjoint
        .evaluate_exterior_adjoint_pressure_checked(&system, &points)
        .map_err(|error| error.to_string())?;
    let compressed_adjoint_field = compressed_adjoint
        .evaluate_exterior_adjoint_pressure_checked(&system, &points)
        .map_err(|error| error.to_string())?;
    let adjoint_error = relative_l2(&compressed_adjoint_field, &dense_adjoint_field);
    if compressed_adjoint
        .evaluate_exterior_adjoint_pressure_checked(&system, &[[f64::NAN, 0.0, 0.8]])
        .is_ok()
    {
        return Err("compressed adjoint field accepted a non-finite point".into());
    }
    Ok((
        dimension,
        compressed_forward.relative_residual(),
        compressed_adjoint.relative_residual(),
        forward_error,
        adjoint_error,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((dimension, forward_residual, adjoint_residual, forward_error, adjoint_error)) => {
            QaResult::passed(CASE_ID)
                .metric("dimension", dimension as f64)
                .metric("forward_residual", forward_residual)
                .metric("adjoint_residual", adjoint_residual)
                .metric("forward_field_error", forward_error)
                .metric("adjoint_field_error", adjoint_error)
                .maximum("forward_residual", 1.0e-8)
                .maximum("adjoint_residual", 2.0e-6)
                .maximum("forward_field_error", 0.2)
                .maximum("adjoint_field_error", 0.2)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .input_checksum("bem:icosphere:subdivision-0:continuous-quadratic-transmission")
                .reference_checksum(
                    "analytic/deterministic-fixture:bem-transmission-compressed-field-parity-v1",
                )
                .dimension("3-D CPU continuous-quadratic compressed transmission field")
                .formulation(
                    "dense-source block-ACA with checked forward and Hermitian-adjoint fields",
                )
                .tolerance(
                    "forward residual <= 1e-8; adjoint residual <= 2e-6; field errors <= 0.2",
                )
                .seed("bem-transmission-compressed-field-parity:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
        }
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
