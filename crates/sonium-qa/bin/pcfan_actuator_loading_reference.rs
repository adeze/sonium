//! Release-backed PC-fan actuator force/torque/power ledger reference.

use sonium_qa::QaResult;
use sonium_sim_pcfan::{
    PcFanCfdConfig, PcFanConfig, PcFanDesignVariables, PcFanFlowModel, PcFanMeshResolution,
    solve_pc_fan_cfd_mean_flow,
};

const CASE_ID: &str = "pcfan.actuator-loading-reference.v1";

fn run() -> Result<(f64, f64, f64, f64), String> {
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
    let loading = solution
        .actuator_loading
        .ok_or_else(|| "incompressible actuator path did not publish loading".to_string())?;
    let power_identity = (loading.shaft_power_to_fluid_w
        - loading.torque_on_fluid_nm * loading.angular_speed_rad_s)
        .abs();
    let finite = [
        loading.force_on_fluid_n[0],
        loading.force_on_fluid_n[1],
        loading.force_on_fluid_n[2],
        loading.torque_on_fluid_nm,
        loading.angular_speed_rad_s,
        loading.shaft_power_to_fluid_w,
    ]
    .into_iter()
    .all(f64::is_finite);
    if !finite {
        return Err("actuator loading contains a non-finite force, torque, speed, or power".into());
    }
    if loading.angular_speed_rad_s <= 0.0 || loading.shaft_power_to_fluid_w <= 0.0 {
        return Err("actuator loading must transfer positive shaft power at positive speed".into());
    }
    if power_identity > 1.0e-10 {
        return Err(format!("shaft-power identity error {power_identity:e}"));
    }
    Ok((
        power_identity,
        loading.force_on_fluid_n[2],
        loading.torque_on_fluid_nm,
        loading.shaft_power_to_fluid_w,
    ))
}

fn main() {
    match run() {
        Ok((power_identity, axial_force, torque, shaft_power)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("shaft_power_identity_error", power_identity)
                .metric("axial_force_on_fluid_n", axial_force)
                .metric("torque_on_fluid_nm", torque)
                .metric("shaft_power_to_fluid_w", shaft_power)
                .maximum("shaft_power_identity_error", 1.0e-10)
                .input_checksum("analytic/deterministic-fixture:pcfan-actuator-loading-v1")
                .reference("independent volume-force torque integral and shaft-power = torque * angular speed")
                .reference_checksum("analytic/deterministic-fixture:pcfan-actuator-loading-v1")
                .dimension("2x2x2 tetrahedral CFD mesh; actuator force/torque/power ledger")
                .formulation("production PC-fan incompressible CFD actuator-force integration")
                .tolerance("finite positive loading and shaft-power identity error <= 1e-10")
                .seed("pcfan-actuator-loading-reference:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("PC-fan actuator result must serialize")
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
                    .expect("PC-fan actuator failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
