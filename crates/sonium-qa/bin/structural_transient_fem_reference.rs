//! Compile-only QA contract for the bounded assembled 2-D FEM transient reference.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{
    StructuralModelAssumption, TransientFemStudy, run_transient_fem,
};

const CASE_ID: &str = "structural.transient-fem-reference.v1";
const RESIDUAL_TOLERANCE_N: f64 = 1.0e-5;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    nodes: usize,
    elements: usize,
    dofs: usize,
    free_dofs: usize,
    steps: usize,
    first_bending_frequency_hz: f64,
    max_dynamic_equilibrium_residual_n: f64,
    max_total_energy_j: f64,
    damping_coefficient_s: f64,
}

fn run() -> Result<Metrics, String> {
    let mut study = TransientFemStudy::introductory();
    study.bracket.geometry.nx = 2;
    study.bracket.geometry.ny = 2;
    study.load_scales = vec![0.0, 0.25, 0.5, 1.0, 0.0];
    study.equilibrium_tolerance_n = RESIDUAL_TOLERANCE_N;

    let report = run_transient_fem(&study).map_err(|error| error.to_string())?;
    let final_displacement_is_nonzero = report
        .final_displacement
        .iter()
        .any(|displacement| displacement[0].abs() + displacement[1].abs() > 0.0);
    let finite_steps = report.steps.iter().all(|step| {
        step.time_s.is_finite()
            && step.load_scale.is_finite()
            && step.max_displacement_m.is_finite()
            && step.max_velocity_m_per_s.is_finite()
            && step.max_acceleration_m_per_s2.is_finite()
            && step.dynamic_equilibrium_residual_n.is_finite()
            && step.total_energy_j.is_finite()
            && step.damping_power_w.is_finite()
    });
    if report.schema_version != 1
        || report.model_assumption != StructuralModelAssumption::TransientFem2D
        || report.polynomial_order != study.polynomial_order
        || report.nodes == 0
        || report.elements == 0
        || report.dofs == 0
        || report.free_dofs == 0
        || report.steps.len() != study.load_scales.len()
        || report.final_displacement.len() != report.nodes
        || report.final_velocity.len() != report.nodes
        || report.final_acceleration.len() != report.nodes
        || !report.first_bending_frequency_hz.is_finite()
        || report.first_bending_frequency_hz <= 0.0
        || !report.damping_coefficient_s.is_finite()
        || !report.max_dynamic_equilibrium_residual_n.is_finite()
        || report.max_dynamic_equilibrium_residual_n > RESIDUAL_TOLERANCE_N
        || !report.max_total_energy_j.is_finite()
        || !finite_steps
        || !report.converged
        || !final_displacement_is_nonzero
        || report.execution.capability_id
            != "sonium-structural.cpu-reference.f64.transient-fem-2d.v1"
    {
        return Err(format!(
            "transient FEM reference gates failed: schema={}, assumption={:?}, order={:?}, nodes={}, elements={}, dofs={}, free_dofs={}, steps={}, frequency={}, residual={}, energy={}, damping={}, converged={}, nonzero_final={}",
            report.schema_version,
            report.model_assumption,
            report.polynomial_order,
            report.nodes,
            report.elements,
            report.dofs,
            report.free_dofs,
            report.steps.len(),
            report.first_bending_frequency_hz,
            report.max_dynamic_equilibrium_residual_n,
            report.max_total_energy_j,
            report.damping_coefficient_s,
            report.converged,
            final_displacement_is_nonzero,
        ));
    }

    Ok(Metrics {
        nodes: report.nodes,
        elements: report.elements,
        dofs: report.dofs,
        free_dofs: report.free_dofs,
        steps: report.steps.len(),
        first_bending_frequency_hz: report.first_bending_frequency_hz,
        max_dynamic_equilibrium_residual_n: report.max_dynamic_equilibrium_residual_n,
        max_total_energy_j: report.max_total_energy_j,
        damping_coefficient_s: report.damping_coefficient_s,
    })
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("nodes", metrics.nodes as f64)
                .metric("elements", metrics.elements as f64)
                .metric("dofs", metrics.dofs as f64)
                .metric("free_dofs", metrics.free_dofs as f64)
                .metric("steps", metrics.steps as f64)
                .metric(
                    "first_bending_frequency_hz",
                    metrics.first_bending_frequency_hz,
                )
                .metric(
                    "max_dynamic_equilibrium_residual_n",
                    metrics.max_dynamic_equilibrium_residual_n,
                )
                .metric("max_total_energy_j", metrics.max_total_energy_j)
                .metric("damping_coefficient_s", metrics.damping_coefficient_s)
                .minimum("nodes", 1.0)
                .minimum("elements", 1.0)
                .minimum("dofs", 1.0)
                .minimum("free_dofs", 1.0)
                .minimum("steps", 2.0)
                .minimum("first_bending_frequency_hz", f64::MIN_POSITIVE)
                .maximum(
                    "max_dynamic_equilibrium_residual_n",
                    RESIDUAL_TOLERANCE_N,
                )
                .reference("bounded assembled linear-elastic 2-D FEM transient reference")
                .reference_checksum(
                    "analytic/deterministic-fixture:structural-transient-fem-reference-v1",
                )
                .dimension("2-D P1 bracket FEM with consistent mass and prescribed load history")
                .formulation(
                    "linear elasticity, consistent mass, mass-proportional damping, Newmark average acceleration",
                )
                .tolerance("dynamic equilibrium residual <= 1e-5 N; finite trajectory diagnostics")
                .seed("structural-transient-fem-reference:2x2-p1:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("structural-transient-fem-reference:2x2-p1:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("STRUCTURAL TRANSIENT FEM QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
