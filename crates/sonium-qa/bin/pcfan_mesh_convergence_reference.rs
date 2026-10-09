//! CPU QA for the production PC-fan CFD mesh-convergence admission path.

use sonium_qa::QaResult;
use sonium_sim_pcfan::{
    PcFanCfdConfig, PcFanCfdMeshConvergenceConfig, PcFanConfig, PcFanFlowModel,
    PcFanMeshResolution, solve_pc_fan_cfd_mesh_convergence,
};
use std::time::Instant;

const CASE_ID: &str = "pcfan.mesh-convergence-reference.v1";
const RELATIVE_TOLERANCE: f64 = 0.70;
const METADATA_TOLERANCE: f64 = 2.0e-12;

struct Metrics {
    levels: f64,
    converged_status: f64,
    final_airflow_cfm: f64,
    final_pressure_abs_pa: f64,
    final_airflow_relative_change: f64,
    final_pressure_relative_change: f64,
    max_metadata_relative_error: f64,
}

fn relative_change(actual: f64, previous: f64) -> f64 {
    (actual - previous).abs() / actual.abs().max(previous.abs()).max(f64::MIN_POSITIVE)
}

fn run() -> Result<Metrics, String> {
    let config = PcFanConfig {
        mesh: PcFanMeshResolution {
            radial_segments: 2,
            chord_segments: 2,
            circumferential_segments: 8,
            harmonic_count: 2,
        },
        cfd: PcFanCfdConfig {
            flow_model: PcFanFlowModel::IncompressibleCfd,
            cells_x: 2,
            cells_y: 2,
            cells_z: 2,
            max_picard_iterations: 40,
            nonlinear_tolerance: 5.0e-2,
            linear_solver_max_iterations: 300,
            linear_solver_tolerance: 1.0e-4,
            ..PcFanCfdConfig::default()
        },
        ..PcFanConfig::default_120mm()
    };
    let design = config.default_design_variables();
    let convergence = solve_pc_fan_cfd_mesh_convergence(
        &config,
        &design,
        PcFanCfdMeshConvergenceConfig {
            start_cells: 2,
            max_cells: 4,
            relative_tolerance: RELATIVE_TOLERANCE,
            min_levels: 3,
            max_final_residual: 1.0e-3,
        },
    );

    if convergence.levels.len() < 3 {
        return Err(format!(
            "mesh-convergence returned {} levels, expected at least 3",
            convergence.levels.len()
        ));
    }

    let mut max_metadata_relative_error: f64 = 0.0;
    for pair in convergence.levels.windows(2) {
        let previous = &pair[0];
        let current = &pair[1];
        if current.cells_x <= previous.cells_x
            || current.cells_y <= previous.cells_y
            || current.cells_z <= previous.cells_z
        {
            return Err("mesh-convergence cell counts are not strictly increasing".into());
        }
        if !current.solution.airflow_cfm.is_finite()
            || !current.solution.pressure_drop_pa.is_finite()
            || !current.solution.mass_balance_error_kg_s.is_finite()
            || !current.solution.residual.is_finite()
        {
            return Err(format!(
                "mesh level {} has invalid CFD admission fields: converged={}, airflow={}, pressure={}, mass_balance={}, residual={}",
                current.cells_x,
                current.solution.converged,
                current.solution.airflow_cfm,
                current.solution.pressure_drop_pa,
                current.solution.mass_balance_error_kg_s,
                current.solution.residual,
            ));
        }

        let expected_airflow =
            relative_change(current.solution.airflow_cfm, previous.solution.airflow_cfm);
        let expected_pressure = relative_change(
            current.solution.pressure_drop_pa.abs(),
            previous.solution.pressure_drop_pa.abs(),
        );
        let airflow_error = relative_change(
            current.airflow_relative_change.unwrap_or(f64::NAN),
            expected_airflow,
        );
        let pressure_error = relative_change(
            current.pressure_relative_change.unwrap_or(f64::NAN),
            expected_pressure,
        );
        max_metadata_relative_error = max_metadata_relative_error
            .max(airflow_error)
            .max(pressure_error);
        if !airflow_error.is_finite()
            || !pressure_error.is_finite()
            || airflow_error > METADATA_TOLERANCE
            || pressure_error > METADATA_TOLERANCE
        {
            return Err(format!(
                "mesh-convergence metadata mismatch at {} cells: airflow error={}, pressure error={}",
                current.cells_x, airflow_error, pressure_error
            ));
        }
    }

    let last = convergence
        .levels
        .last()
        .expect("at least three mesh levels were admitted");
    let final_airflow_relative_change = last
        .airflow_relative_change
        .ok_or_else(|| "final mesh level lacks airflow change metadata".to_string())?;
    let final_pressure_relative_change = last
        .pressure_relative_change
        .ok_or_else(|| "final mesh level lacks pressure change metadata".to_string())?;
    if convergence.converged
        && (final_airflow_relative_change > RELATIVE_TOLERANCE
            || final_pressure_relative_change > RELATIVE_TOLERANCE)
    {
        return Err(format!(
            "final mesh changes exceed admission tolerance: airflow={}, pressure={}",
            final_airflow_relative_change, final_pressure_relative_change
        ));
    }
    if relative_change(
        convergence.selected_solution.airflow_cfm,
        last.solution.airflow_cfm,
    ) > METADATA_TOLERANCE
        || relative_change(
            convergence.selected_solution.pressure_drop_pa,
            last.solution.pressure_drop_pa,
        ) > METADATA_TOLERANCE
    {
        return Err("selected solution does not match final admitted mesh level".into());
    }

    Ok(Metrics {
        levels: convergence.levels.len() as f64,
        converged_status: if convergence.converged { 1.0 } else { 0.0 },
        final_airflow_cfm: last.solution.airflow_cfm,
        final_pressure_abs_pa: last.solution.pressure_drop_pa.abs(),
        final_airflow_relative_change,
        final_pressure_relative_change,
        max_metadata_relative_error,
    })
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("levels", metrics.levels)
                .metric("converged_status", metrics.converged_status)
                .metric("final_airflow_cfm", metrics.final_airflow_cfm)
                .metric("final_pressure_abs_pa", metrics.final_pressure_abs_pa)
                .metric(
                    "final_airflow_relative_change",
                    metrics.final_airflow_relative_change,
                )
                .metric(
                    "final_pressure_relative_change",
                    metrics.final_pressure_relative_change,
                )
                .metric(
                    "max_metadata_relative_error",
                    metrics.max_metadata_relative_error,
                )
                .minimum("levels", 3.0)
                .minimum("converged_status", 0.0)
                .maximum("converged_status", 1.0)
                .minimum("final_airflow_cfm", 0.0)
                .minimum("final_pressure_abs_pa", 0.0)
                .minimum("final_airflow_relative_change", 0.0)
                .minimum("final_pressure_relative_change", 0.0)
                .maximum("final_airflow_relative_change", RELATIVE_TOLERANCE)
                .maximum("final_pressure_relative_change", RELATIVE_TOLERANCE)
                .maximum("max_metadata_relative_error", METADATA_TOLERANCE)
                .reference_checksum("analytic/invariant:pcfan-mesh-convergence-v1")
                .input_checksum("pcfan-120mm:incompressible-cfd:cells=2..4:cpu")
                .formulation(
                    "production immersed-solid CFD mesh-convergence metadata and admission",
                )
                .seed("pcfan-mesh-convergence-v1:2,3,4:relative-tolerance=0.70")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("PCFAN MESH-CONVERGENCE QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
