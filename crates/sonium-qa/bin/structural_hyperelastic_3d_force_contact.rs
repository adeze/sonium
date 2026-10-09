//! Compile-only QA contract for the force-controlled 3-D hyperelastic contact reference.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{
    Hyperelastic3DForceContactStudy, run_hyperelastic_3d_force_contact,
};

const CASE_ID: &str = "structural.hyperelastic-3d-force-contact.v1";

fn run() -> Result<(usize, usize, usize, usize, usize, f64, f64), String> {
    let report =
        run_hyperelastic_3d_force_contact(&Hyperelastic3DForceContactStudy::introductory())
            .map_err(|error| error.to_string())?;
    if !report.converged
        || report.steps.len() != report.study.load_steps
        || report.steps.iter().any(|step| !step.converged)
        || report.contact_engaged_step.is_none()
        || report.bottom_contact_faces == 0
        || report.top_contact_faces == 0
        || !report.max_penetration_m.is_finite()
        || !report.final_residual.is_finite()
        || report.final_average_top_displacement_m >= 0.0
    {
        return Err(format!(
            "3-D force-controlled hyperelastic contact gates failed: {report:#?}"
        ));
    }

    Ok((
        report.nodes,
        report.elements,
        report.bottom_contact_faces,
        report.top_contact_faces,
        report.contact_engaged_step.unwrap_or_default(),
        report.max_penetration_m,
        report.final_average_top_displacement_m,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            nodes,
            elements,
            bottom_contact_faces,
            top_contact_faces,
            contact_engaged_step,
            max_penetration_m,
            final_average_top_displacement_m,
        )) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("nodes", nodes as f64)
                .metric("elements", elements as f64)
                .metric("bottom_contact_faces", bottom_contact_faces as f64)
                .metric("top_contact_faces", top_contact_faces as f64)
                .metric("contact_engaged_step", contact_engaged_step as f64)
                .metric("max_penetration_m", max_penetration_m)
                .metric("final_average_top_displacement_m", final_average_top_displacement_m)
                .reference_checksum("analytic/deterministic-fixture:hyperelastic-3d-force-contact-v1")
                .dimension("3-D force-controlled opposing-face hyperelastic contact FEM")
                .formulation(
                    "Neo-Hookean total-Lagrangian FEM with distributed external load and augmented-Lagrangian contact",
                )
                .seed("structural-hyperelastic-3d-force-contact:0.4m-two-block:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone())
                .seed("structural-hyperelastic-3d-force-contact:0.4m-two-block:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("STRUCTURAL HYPERELASTIC 3-D FORCE CONTACT QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
