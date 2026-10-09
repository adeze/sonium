//! Compile-only QA contract for the 3-D force-controlled hyperelastic reference.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{Hyperelastic3DForceStudy, run_hyperelastic_3d_force};

const CASE_ID: &str = "structural.hyperelastic-3d-force.v1";

fn run() -> Result<(usize, usize, f64), String> {
    let report = run_hyperelastic_3d_force(&Hyperelastic3DForceStudy::introductory())
        .map_err(|error| error.to_string())?;
    if !report.converged
        || report.steps.len() != report.study.load_steps
        || report.steps.iter().any(|step| !step.converged)
        || report.bottom_nodes == 0
        || report.top_nodes == 0
        || !report.final_residual.is_finite()
        || report.final_average_top_displacement_m >= 0.0
    {
        return Err(format!(
            "3-D force reference gates failed: converged={}, steps={}, bottom={}, top={}, residual={}, displacement={}",
            report.converged,
            report.steps.len(),
            report.bottom_nodes,
            report.top_nodes,
            report.final_residual,
            report.final_average_top_displacement_m
        ));
    }
    Ok((
        report.nodes,
        report.elements,
        report.final_average_top_displacement_m,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((nodes, elements, displacement)) => {
            let record = QaResult::passed(CASE_ID)
                .reference_checksum("analytic/deterministic-fixture:hyperelastic-3d-force-v1")
                .dimension("3-D guided force-controlled hyperelastic FEM")
                .formulation("Neo-Hookean total-Lagrangian FEM with explicit external load vector")
                .seed("structural-hyperelastic-3d-force:0.4m-block:cpu")
                .metric("nodes", nodes as f64)
                .metric("elements", elements as f64)
                .metric("final_average_top_displacement_m", displacement)
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone())
                .seed("structural-hyperelastic-3d-force:0.4m-block:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("STRUCTURAL HYPERELASTIC 3-D FORCE QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
