//! Diagnostic native CFD convergence and independently assembled residual.

// Rust guideline compliant 2026-02-21

use sonium_sim_pcfan::{
    PcFanCfdBackend, PcFanCfdConfig, PcFanConfig, PcFanFlowModel,
    evaluate_pc_fan_cfd_newton_state_linearization, solve_pc_fan_cfd_mean_flow,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cells = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "4".into())
        .parse::<usize>()?;
    let iterations = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "2000".into())
        .parse::<usize>()?;
    let config = PcFanConfig {
        cfd: PcFanCfdConfig {
            flow_model: PcFanFlowModel::IncompressibleCfd,
            backend: PcFanCfdBackend::Cpu,
            cells_x: cells,
            cells_y: cells,
            cells_z: cells,
            max_picard_iterations: iterations,
            nonlinear_tolerance: 1.0e-7,
            linear_solver_tolerance: 1.0e-10,
            linear_solver_max_iterations: 2000,
            use_direct_solver: true,
            ..PcFanCfdConfig::default()
        },
        ..PcFanConfig::default_120mm()
    };
    let design = config.default_design_variables();
    println!("Starting native {cells}^3 CFD: maximum {iterations} Picard iterations");
    let solution = solve_pc_fan_cfd_mean_flow(&config, &design);
    println!(
        "converged={} iterations={} residual={:.6e} airflow={:.6e} inlet={:.6e} pressure={:.6e} nodes={} elements={} conservation={:?}",
        solution.converged,
        solution.iterations,
        solution.residual,
        solution.airflow_cfm,
        solution.inlet_airflow_cfm,
        solution.pressure_drop_pa,
        solution.fem_node_count,
        solution.fem_element_count,
        solution.conservation
    );
    let mut state = solution
        .velocity_m_s
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    state.extend_from_slice(&solution.pressure_pa);
    let observation = evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &state)?;
    let residual = observation
        .state_residual
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let state_error = observation
        .state
        .iter()
        .zip(&state)
        .map(|(value, expected)| (value - expected).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "independently assembled nonlinear residual={residual:.6e} state_reassembly_error={state_error:.6e} provenance={}",
        observation.provenance
    );
    Ok(())
}
