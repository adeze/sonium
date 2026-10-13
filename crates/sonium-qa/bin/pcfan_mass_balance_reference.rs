//! CPU QA for converged PC-fan CFD mass conservation and true residual.

// Rust guideline compliant 2026-02-21

use sonium_qa::{QaResult, QaStatus};
use sonium_sim_pcfan::{
    PcFanCfdBackend, PcFanCfdConfig, PcFanConfig, PcFanDesignVariables, PcFanFlowModel,
    PcFanMeshResolution, evaluate_pc_fan_cfd_newton_state_linearization,
    solve_pc_fan_cfd_mean_flow,
};
use std::time::Instant;

const CASE_ID: &str = "pcfan.mass-balance-reference.v1";
const M3S_TO_CFM: f64 = 2_118.880_003_289_3;
// One percent boundary flux conservation on the native volumetric fixture.
const MASS_BALANCE_TOLERANCE: f64 = 0.01;
const NONLINEAR_TOLERANCE: f64 = 1.0e-7;
const TRUE_RESIDUAL_TOLERANCE: f64 = 1.0e-6;

fn run() -> Result<QaResult, String> {
    let config = PcFanConfig {
        mesh: PcFanMeshResolution {
            radial_segments: 2,
            chord_segments: 2,
            circumferential_segments: 8,
            harmonic_count: 2,
        },
        cfd: PcFanCfdConfig {
            flow_model: PcFanFlowModel::IncompressibleCfd,
            backend: PcFanCfdBackend::Cpu,
            cells_x: 8,
            cells_y: 8,
            cells_z: 8,
            max_picard_iterations: 2000,
            nonlinear_tolerance: NONLINEAR_TOLERANCE,
            linear_solver_max_iterations: 2000,
            linear_solver_tolerance: 1.0e-10,
            use_direct_solver: true,
            max_relative_mass_imbalance: MASS_BALANCE_TOLERANCE,
            ..PcFanCfdConfig::default()
        },
        ..PcFanConfig::default_120mm()
    };
    run_with_config(config)
}

fn run_with_config(config: PcFanConfig) -> Result<QaResult, String> {
    let design = PcFanDesignVariables::default();
    let solution = solve_pc_fan_cfd_mean_flow(&config, &design);
    println!(
        "native CFD: converged={} iterations={} residual={:.6e} inlet={:.6e} outlet={:.6e}",
        solution.converged,
        solution.iterations,
        solution.residual,
        solution.inlet_airflow_cfm,
        solution.airflow_cfm
    );
    let values = [
        solution.airflow_cfm,
        solution.inlet_airflow_cfm,
        solution.mass_balance_error_kg_s,
        solution.pressure_drop_pa,
        solution.residual,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err("production CFD returned a non-finite mass-balance observable".into());
    }
    if solution.airflow_cfm <= 0.0 || solution.inlet_airflow_cfm <= 0.0 {
        return Err("production CFD returned non-positive airflow".into());
    }
    let recomputed_mass_balance = config.operating_point.air_density_kg_m3
        * ((solution.airflow_cfm - solution.inlet_airflow_cfm) / M3S_TO_CFM);
    let mass_scale =
        config.operating_point.air_density_kg_m3 * (solution.inlet_airflow_cfm / M3S_TO_CFM);
    let disagreement = (recomputed_mass_balance - solution.mass_balance_error_kg_s).abs();
    if disagreement > 1.0e-10 * mass_scale.max(1.0) {
        return Err(
            "independent inlet/outlet recomputation disagrees with production mass balance".into(),
        );
    }

    // Reassemble the full constrained nonlinear residual at the returned field.
    // This API performs one map probe at exactly the prescribed state; verify
    // state identity so a subsequent Picard iterate cannot stand in for it.
    let mut state = solution
        .velocity_m_s
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    state.extend_from_slice(&solution.pressure_pa);
    let observation = evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &state)
        .map_err(|error| format!("independent residual assembly failed: {error}"))?;
    observation.validate().map_err(|error| error.to_string())?;
    if observation.state.len() != state.len() {
        return Err("independent residual assembly changed the state dimension".into());
    }
    let state_error = observation
        .state
        .iter()
        .zip(&state)
        .map(|(value, expected)| (value - expected).abs())
        .fold(0.0_f64, f64::max);
    let true_residual = observation
        .state_residual
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    // CFM observables expose flow magnitudes. Also check signed native boundary
    // fluxes so equal magnitudes with reversed flow cannot imply conservation.
    let inlet_mass = solution.conservation.inlet_mass_flow_kg_s;
    let outlet_mass = solution.conservation.outlet_mass_flow_kg_s;
    let signed_mass_error = (outlet_mass - inlet_mass).abs()
        / inlet_mass
            .abs()
            .max(outlet_mass.abs())
            .max(f64::MIN_POSITIVE);
    let record = QaResult::passed(CASE_ID).requested_backend("cpu").actual_backend("cpu")
        .metric("converged_status", if solution.converged { 1.0 } else { 0.0 }).minimum("converged_status", 1.0)
        .metric("nonlinear_update_residual", solution.residual).maximum("nonlinear_update_residual", NONLINEAR_TOLERANCE)
        .metric("independent_true_residual", true_residual).maximum("independent_true_residual", TRUE_RESIDUAL_TOLERANCE)
        .metric("residual_state_identity_error", state_error).maximum("residual_state_identity_error", 1.0e-12)
        .metric("relative_mass_balance_error", recomputed_mass_balance.abs() / mass_scale.max(f64::MIN_POSITIVE))
        .maximum("relative_mass_balance_error", MASS_BALANCE_TOLERANCE)
        .metric("signed_inlet_mass_flow_kg_s", inlet_mass)
        .minimum("signed_inlet_mass_flow_kg_s", f64::MIN_POSITIVE)
        .metric("signed_outlet_mass_flow_kg_s", outlet_mass)
        .minimum("signed_outlet_mass_flow_kg_s", f64::MIN_POSITIVE)
        .metric("signed_relative_mass_balance_error", signed_mass_error)
        .maximum("signed_relative_mass_balance_error", MASS_BALANCE_TOLERANCE)
        .metric("absolute_mass_balance_error_kg_s", recomputed_mass_balance.abs())
        .metric("mass_balance_observable_disagreement_kg_s", disagreement)
        .metric("airflow_cfm", solution.airflow_cfm).minimum("airflow_cfm", f64::MIN_POSITIVE)
        .metric("inlet_airflow_cfm", solution.inlet_airflow_cfm).minimum("inlet_airflow_cfm", f64::MIN_POSITIVE)
        .metric("pressure_drop_pa", solution.pressure_drop_pa)
        .metric("iterations", solution.iterations as f64)
        .metric("node_count", solution.fem_node_count as f64)
        .metric("element_count", solution.fem_element_count as f64)
        .input_checksum(format!("pcfan-120mm:incompressible-cfd:{}x{}x{}:cpu:nonlinear-1e-7", config.cfd.cells_x, config.cfd.cells_y, config.cfd.cells_z))
        .reference("independent inlet/outlet mass recomputation and constrained nonlinear residual reassembly at the returned native field")
        .reference_checksum("analytic/deterministic-fixture:pcfan-mass-balance-v1")
        .dimension(format!("{}x{}x{} symmetric tetrahedral incompressible PC-fan CFD", config.cfd.cells_x, config.cfd.cells_y, config.cfd.cells_z))
        .formulation("native steady CFD; exact inner solves; full residual reassembly and boundary flux conservation")
        .tolerance("all solves converge; nonlinear update <=1e-7; true residual <=1e-6; relative mass imbalance <=0.01")
        .seed("pcfan-mass-balance-reference:cpu:v1");
    Ok(record)
}

fn main() {
    let started = Instant::now();
    let mut result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    }
    .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
    if result.status == QaStatus::Passed
        && let Err(error) = result.release_eligible()
    {
        result.status = QaStatus::Failed;
        result.message = Some(error);
    }
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("serialize PC-fan mass-balance result")
    );
    if result.status == QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_native_budget_cannot_be_reported_as_mass_balance_success() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 2,
            },
            cfd: PcFanCfdConfig {
                flow_model: PcFanFlowModel::IncompressibleCfd,
                backend: PcFanCfdBackend::Cpu,
                cells_x: 2,
                cells_y: 2,
                cells_z: 2,
                max_picard_iterations: 4,
                nonlinear_tolerance: NONLINEAR_TOLERANCE,
                use_direct_solver: true,
                ..PcFanCfdConfig::default()
            },
            ..PcFanConfig::default_120mm()
        };
        let mut record =
            run_with_config(config).expect("bounded native probe must produce measurable fields");
        assert_eq!(record.metrics["converged_status"], 0.0);
        assert!(record.release_eligible().is_err());
        // A forged convergence flag must not hide the reassembled equation residual.
        record.metrics.insert("converged_status".into(), 1.0);
        record
            .metrics
            .insert("nonlinear_update_residual".into(), 0.0);
        assert!(record.metrics["independent_true_residual"] > TRUE_RESIDUAL_TOLERANCE);
        assert!(record.release_eligible().is_err());
    }
}
