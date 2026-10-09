//! Independent transparent-interface limit for the constant-panel Müller solve.
//!
//! When both media are identical, a closed interface has no acoustic contrast:
//! the exterior solution must reproduce the incident plane wave and its
//! scattered field must remain at the discretization-error level.  The
//! reference is evaluated independently from the production transmission
//! solve using the analytic plane-wave field.

use sonium_bem::build_transmission_tbem_system_from_xem;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::solver::solve_incident_transmission;
use sonium_bem::core::types::Element;
use sonium_qa::QaResult;
use sonium_xem::AcousticMedium;
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-transparent-limit.v1";
const FREQUENCY_HZ: f64 = 420.0;
const FIELD_TOLERANCE: f64 = 1.0e-3;
const RESIDUAL_TOLERANCE: f64 = 1.0e-10;

fn run() -> Result<(f64, f64, f64), String> {
    let mesh = generate_icosphere_mesh(0.2, 2);
    let mut elements: Vec<Element> = mesh.elements;
    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = vec![index];
    }

    let exterior = AcousticMedium::try_new(FREQUENCY_HZ, 343.0, 1.21)
        .map_err(|error| format!("exterior medium: {error}"))?;
    let interior = AcousticMedium::try_new(FREQUENCY_HZ, 343.0, 1.21)
        .map_err(|error| format!("interior medium: {error}"))?;
    let system =
        build_transmission_tbem_system_from_xem(&elements, &mesh.nodes, exterior, interior)
            .map_err(|error| format!("transparent transmission assembly: {error}"))?;

    let incident = IncidentField::plane_wave_z();
    let solution = solve_incident_transmission(&system, &incident, RESIDUAL_TOLERANCE)
        .map_err(|error| format!("transparent transmission solve: {error}"))?;
    let points = [[0.0, 0.0, 1.0], [0.3, 0.0, 0.0]];
    let field = solution
        .evaluate_exterior_field_checked(&system, &points, &incident)
        .map_err(|error| format!("transparent exterior field: {error}"))?;
    let expected = incident.evaluate_pressure(&points, &system.exterior);
    let max_total_error = field
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual.p_total - expected).norm())
        .fold(0.0, f64::max);
    let max_scattered_pressure = field
        .iter()
        .map(|point| point.p_scattered.norm())
        .fold(0.0, f64::max);
    if !max_total_error.is_finite()
        || !max_scattered_pressure.is_finite()
        || max_total_error > FIELD_TOLERANCE
        || max_scattered_pressure > FIELD_TOLERANCE
    {
        return Err(format!(
            "transparent limit exceeded: total error {max_total_error:.3e}, scattered pressure {max_scattered_pressure:.3e}, tolerance {FIELD_TOLERANCE:.3e}"
        ));
    }
    Ok((
        max_total_error,
        max_scattered_pressure,
        solution.relative_residual,
    ))
}

fn main() {
    let started = Instant::now();
    match std::panic::catch_unwind(run) {
        Ok(Ok((max_total_error, max_scattered_pressure, residual))) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("max_total_field_error", max_total_error)
                .metric("max_scattered_pressure", max_scattered_pressure)
                .metric("relative_residual", residual)
                .maximum("max_total_field_error", FIELD_TOLERANCE)
                .maximum("max_scattered_pressure", FIELD_TOLERANCE)
                .maximum("relative_residual", RESIDUAL_TOLERANCE)
                .input_checksum("analytic/deterministic-fixture:bem-transparent-limit-v1")
                .reference("independent analytic plane-wave field in the equal-medium transparent-interface limit")
                .reference_checksum("analytic/deterministic-fixture:bem-transparent-limit-v1")
                .dimension("3D closed icosphere level 2, constant-panel equal-medium interface")
                .formulation("dense constant-panel Müller transmission solve and checked exterior total-field evaluation")
                .tolerance("total-field and scattered-pressure errors <= 1e-3; certified relative residual <= 1e-10")
                .seed("bem-transmission-transparent-limit:icosphere-level-2:plane-wave-z:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("transparent-limit QA result must serialize")
            );
            println!("SUCCESS");
        }
        Ok(Err(error)) => {
            let record = QaResult::failed(CASE_ID, error.clone())
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("transparent-limit QA failure must serialize")
            );
            eprintln!("BEM TRANSPARENT LIMIT FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
        Err(payload) => {
            let message = if let Some(message) = payload.downcast_ref::<&str>() {
                (*message).to_owned()
            } else if let Some(message) = payload.downcast_ref::<String>() {
                message.clone()
            } else {
                "unknown panic".to_owned()
            };
            let record = QaResult::failed(CASE_ID, message.clone())
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("transparent-limit panic result must serialize")
            );
            eprintln!("BEM TRANSPARENT LIMIT PANICKED: {message}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
