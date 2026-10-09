//! Compile-only QA contract for the 3-D hyperelastic contact reference.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{Hyperelastic3DContactStudy, run_hyperelastic_3d_contact};

const CASE_ID: &str = "structural.hyperelastic-3d-contact.v1";

fn run() -> Result<(usize, usize, f64, usize), String> {
    let report = run_hyperelastic_3d_contact(&Hyperelastic3DContactStudy::introductory())
        .map_err(|error| error.to_string())?;
    let engaged = report.contact_engaged_step.ok_or_else(|| {
        "3-D hyperelastic contact reference did not report contact engagement".to_string()
    })?;
    if !report.converged
        || report.bottom_contact_faces == 0
        || report.top_contact_faces == 0
        || report.steps.len() != report.study.load_steps
        || report.steps.iter().any(|step| !step.converged)
        || !report.max_penetration_m.is_finite()
        || !report.final_residual.is_finite()
    {
        return Err(format!(
            "3-D hyperelastic contact gates failed: {report:#?}"
        ));
    }
    Ok((
        report.bottom_contact_faces,
        report.top_contact_faces,
        report.max_penetration_m,
        engaged,
    ))
}
fn main() {
    let started = Instant::now();
    match run() {
        Ok((bottom_faces, top_faces, max_penetration_m, engaged_step)) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("bottom_contact_faces", bottom_faces as f64)
                .metric("top_contact_faces", top_faces as f64)
                .metric("max_penetration_m", max_penetration_m)
                .metric("contact_engaged_step", engaged_step as f64)
                .reference_checksum("analytic/deterministic-fixture:hyperelastic-3d-contact-v1")
                .dimension("3-D P1 tetrahedral hyperelastic block and opposing-face contact")
                .formulation("Neo-Hookean total-Lagrangian FEM with augmented-Lagrangian contact")
                .seed("structural-hyperelastic-3d-contact:0.4m-span-two-block:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone())
                .seed("structural-hyperelastic-3d-contact:0.4m-span-two-block:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("STRUCTURAL HYPERELASTIC 3-D CONTACT QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
