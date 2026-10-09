//! CPU QA for the production PC-fan CFD mass-balance observable.

use sonium_qa::QaResult;
use sonium_sim_pcfan::{
    PcFanCfdConfig, PcFanConfig, PcFanDesignVariables, PcFanFlowModel, PcFanMeshResolution,
    solve_pc_fan_cfd_mean_flow,
};

const CASE_ID: &str = "pcfan.mass-balance-reference.v1";

const M3S_TO_CFM: f64 = 2_118.880_003_289_3;

fn run() -> Result<(f64, f64, f64, f64, f64, f64), String> {
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
            max_picard_iterations: 4,
            nonlinear_tolerance: 1.0e-3,
            linear_solver_max_iterations: 300,
            linear_solver_tolerance: 1.0e-4,
            ..PcFanCfdConfig::default()
        },
        ..PcFanConfig::default_120mm()
    };

    let solution = solve_pc_fan_cfd_mean_flow(&config, &PcFanDesignVariables::default());
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
    // Recompute the mismatch from the independently exposed inlet and outlet
    // CFM values. Convert CFM to m³/s before applying air density; dividing
    // kg/s by CFM would mix units.
    let recomputed_mass_balance_error_kg_s = config.operating_point.air_density_kg_m3
        * ((solution.airflow_cfm - solution.inlet_airflow_cfm) / M3S_TO_CFM);
    let mass_balance_scale_kg_s =
        config.operating_point.air_density_kg_m3 * (solution.inlet_airflow_cfm / M3S_TO_CFM);
    let observable_disagreement =
        (recomputed_mass_balance_error_kg_s - solution.mass_balance_error_kg_s).abs();
    if observable_disagreement > 1.0e-10 * mass_balance_scale_kg_s.max(1.0) {
        return Err(format!(
            "independent mass-balance recomputation disagrees with production observable: recomputed={recomputed_mass_balance_error_kg_s:.6e}, production={:.6e}",
            solution.mass_balance_error_kg_s,
        ));
    }
    let relative_mass_balance_error =
        recomputed_mass_balance_error_kg_s.abs() / mass_balance_scale_kg_s.max(1.0e-12);
    Ok((
        relative_mass_balance_error,
        recomputed_mass_balance_error_kg_s.abs(),
        observable_disagreement,
        solution.airflow_cfm,
        solution.pressure_drop_pa,
        solution.residual,
    ))
}

fn main() {
    match run() {
        Ok((
            relative_error,
            absolute_error,
            observable_disagreement,
            airflow,
            pressure_drop,
            residual,
        )) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("relative_mass_balance_error", relative_error)
                .metric("absolute_mass_balance_error_kg_s", absolute_error)
                .metric(
                    "mass_balance_observable_disagreement_kg_s",
                    observable_disagreement,
                )
                .metric("airflow_cfm", airflow)
                .metric("pressure_drop_pa", pressure_drop)
                .metric("residual", residual)
                .maximum("relative_mass_balance_error", 0.25)
                .input_checksum("analytic/deterministic-fixture:pcfan-mass-balance-v1")
                .reference(
                    "independent inlet/outlet CFM recomputation converted with configured air density",
                )
                .reference_checksum("analytic/deterministic-fixture:pcfan-mass-balance-v1")
                .dimension("2x2x2 tetrahedral incompressible PC-fan CFD fixture")
                .formulation("production PC-fan mean-flow CFD mass-balance observable")
                .tolerance("finite positive airflow and relative mass-balance error <= 0.25")
                .seed("pcfan-mass-balance-reference:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("PC-fan mass-balance result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("PC-fan mass-balance failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
