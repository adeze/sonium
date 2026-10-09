//! Compile-only release QA contract for the bounded robust 3-D topology path.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{
    Solid3DRobustTopologyStudy, StructuralModelAssumption, run_solid3d_robust_topology,
};

const CASE_ID: &str = "structural.solid-topology-3d-robust.v1";
const RESIDUAL_TOLERANCE: f64 = 1.0e-8;
const EQUILIBRIUM_TOLERANCE: f64 = 1.0e-6;
const ENERGY_TOLERANCE: f64 = 1.0e-8;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    cases: usize,
    nominal_volume_fraction: f64,
    eroded_volume_fraction: f64,
    dilated_volume_fraction: f64,
    nominal_overhang_slope: f64,
    aggregated_stress_pa: f64,
    worst_case_compliance_j: f64,
    max_solver_residual: f64,
    max_equilibrium_residual_n: f64,
    max_force_balance_residual_n: f64,
    max_energy_balance_relative_error: f64,
    gradient_taylor_relative_error: f64,
}

fn run() -> Result<Metrics, String> {
    let mut study = Solid3DRobustTopologyStudy::introductory();
    study.base.base.geometry.nx = 2;
    study.base.base.geometry.ny = 1;
    study.base.base.geometry.nz = 1;
    study.base.volume_fraction = 0.90;
    study.base.iterations = 2;
    study.base.allowable_stress_pa = 1.0e14;
    study.allowable_compliance_j = 1.0;

    let result = run_solid3d_robust_topology(&study).map_err(|error| error.to_string())?;
    let expected_cases = 3 * study.uncertainty_cases.len() * study.base.load_cases.len();
    if result.model_assumption != StructuralModelAssumption::SimpTopology3D
        || result.cases.len() != expected_cases
        || !result.factorization_reused_within_scenario
        || !result.erosion_connected
        || !result.dilation_connected
        || !result.overhang_feasible
        || !result.stress_feasible
        || !result.morphology_feasible
        || !result.uncertainty_feasible
        || !result.robust_feasible
        || result.eroded_volume_fraction > result.nominal_volume_fraction + 1.0e-12
        || result.dilated_volume_fraction + 1.0e-12 < result.nominal_volume_fraction
    {
        return Err("robust morphology/uncertainty feasibility contract failed".to_string());
    }

    let mut metrics = Metrics {
        cases: result.cases.len(),
        nominal_volume_fraction: result.nominal_volume_fraction,
        eroded_volume_fraction: result.eroded_volume_fraction,
        dilated_volume_fraction: result.dilated_volume_fraction,
        nominal_overhang_slope: result.nominal.overhang_slope,
        aggregated_stress_pa: result.aggregated_stress_pa,
        worst_case_compliance_j: result.worst_case_compliance_j,
        max_solver_residual: 0.0,
        max_equilibrium_residual_n: 0.0,
        max_force_balance_residual_n: 0.0,
        max_energy_balance_relative_error: 0.0,
        gradient_taylor_relative_error: result.gradient_taylor_relative_error,
    };
    for case in &result.cases {
        if !case.converged
            || !case.feasible
            || !case.compliance_j.is_finite()
            || !case.aggregated_stress_pa.is_finite()
            || !case.solver_residual.is_finite()
            || !case.free_equilibrium_residual_n.is_finite()
            || !case.force_balance_residual_n.is_finite()
            || !case.energy_balance_relative_error.is_finite()
        {
            return Err(format!("robust case failed: {}", case.load_case));
        }
        metrics.max_solver_residual = metrics.max_solver_residual.max(case.solver_residual);
        metrics.max_equilibrium_residual_n = metrics
            .max_equilibrium_residual_n
            .max(case.free_equilibrium_residual_n);
        metrics.max_force_balance_residual_n = metrics
            .max_force_balance_residual_n
            .max(case.force_balance_residual_n);
        metrics.max_energy_balance_relative_error = metrics
            .max_energy_balance_relative_error
            .max(case.energy_balance_relative_error);
    }
    let scalars = [
        metrics.nominal_volume_fraction,
        metrics.eroded_volume_fraction,
        metrics.dilated_volume_fraction,
        metrics.nominal_overhang_slope,
        metrics.aggregated_stress_pa,
        metrics.worst_case_compliance_j,
        metrics.gradient_taylor_relative_error,
    ];
    if scalars.iter().any(|value| !value.is_finite())
        || metrics.max_solver_residual > RESIDUAL_TOLERANCE
        || metrics.max_equilibrium_residual_n > EQUILIBRIUM_TOLERANCE
        || metrics.max_force_balance_residual_n > EQUILIBRIUM_TOLERANCE
        || metrics.max_energy_balance_relative_error > ENERGY_TOLERANCE
    {
        return Err("robust 3-D metric gate failed".to_string());
    }
    Ok(metrics)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("cases", metrics.cases as f64)
                .metric("nominal_volume_fraction", metrics.nominal_volume_fraction)
                .metric("eroded_volume_fraction", metrics.eroded_volume_fraction)
                .metric("dilated_volume_fraction", metrics.dilated_volume_fraction)
                .metric("nominal_overhang_slope", metrics.nominal_overhang_slope)
                .metric("aggregated_stress_pa", metrics.aggregated_stress_pa)
                .metric("worst_case_compliance_j", metrics.worst_case_compliance_j)
                .metric("max_solver_residual", metrics.max_solver_residual)
                .metric(
                    "max_equilibrium_residual_n",
                    metrics.max_equilibrium_residual_n,
                )
                .metric(
                    "max_force_balance_residual_n",
                    metrics.max_force_balance_residual_n,
                )
                .metric(
                    "max_energy_balance_relative_error",
                    metrics.max_energy_balance_relative_error,
                )
                .metric(
                    "gradient_taylor_relative_error",
                    metrics.gradient_taylor_relative_error,
                )
                .minimum("cases", 6.0)
                .maximum("max_solver_residual", RESIDUAL_TOLERANCE)
                .maximum("max_equilibrium_residual_n", EQUILIBRIUM_TOLERANCE)
                .maximum("max_force_balance_residual_n", EQUILIBRIUM_TOLERANCE)
                .maximum("max_energy_balance_relative_error", ENERGY_TOLERANCE)
                .maximum(
                    "nominal_overhang_slope",
                    Solid3DRobustTopologyStudy::introductory()
                        .base
                        .maximum_overhang_slope,
                )
                .maximum(
                    "aggregated_stress_pa",
                    Solid3DRobustTopologyStudy::introductory()
                        .base
                        .allowable_stress_pa,
                )
                .reference("assembled P1 tetrahedral robust morphology and uncertainty reanalysis")
                .reference_checksum("analytic/deterministic-fixture:structural-solid-topology-3d-robust-v1")
                .dimension("3-D linear tetrahedra; 3 morphologies x 2 explicit uncertainty cases x 2 load cases")
                .formulation("nominal SIMP design with graph erosion/dilation, 3-D overhang screening, and shared-factorization static reanalysis")
                .tolerance("solver <= 1e-8; equilibrium/force <= 1e-6; energy <= 1e-8")
                .seed("structural-solid-topology-3d-robust:2x1x1:cpu")
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
                .seed("structural-solid-topology-3d-robust:2x1x1:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("STRUCTURAL ROBUST 3-D TOPOLOGY QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
