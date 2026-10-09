//! Temporary perf harness for the motor topology optimizer (review item D1).
//! Run explicitly: `cargo test -p sonium-sim-speaker-modeling --test motor_topology_perf -- --ignored --nocapture`

use sonium_sim_speaker_modeling::axisymmetric_linear::AxisymmetricLinearConfig;
use sonium_sim_speaker_modeling::motor_topology::{MotorTopologyConfig, optimize_motor_topology};

#[test]
#[ignore = "perf measurement; run explicitly"]
fn motor_topology_default_config_timing() {
    let driver = AxisymmetricLinearConfig::default();
    let start = std::time::Instant::now();
    let result = optimize_motor_topology(&driver, &MotorTopologyConfig::default()).unwrap();
    let elapsed = start.elapsed();
    let iterations = result.iterations.max(1);
    eprintln!(
        "motor topology: {} iterations in {:.3}s ({:.3}s/iter), gray fraction {:.4}, final BL {:.6} Tm",
        result.iterations,
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / iterations as f64,
        result.gray_fraction,
        result.bl_history_tm.last().copied().unwrap_or(f64::NAN),
    );
}
