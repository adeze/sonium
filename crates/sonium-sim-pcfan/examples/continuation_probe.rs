//! Diagnostic load continuation ending at the unchanged native fan operating point.

// Rust guideline compliant 2026-02-21

use sonium_fem::mesh::generators::box_mesh_tetrahedra_symmetric;
use sonium_sim_pcfan::{
    PcFanCfdBackend, PcFanCfdConfig, PcFanConfig, PcFanDesignVariables, PcFanFlowModel,
    PcFanMeshResolution, evaluate_pc_fan_cfd_newton_state_linearization,
};
use sonium_solvers::{CsrMatrix, direct::SparseLuFactorization};

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cells = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "4".into())
        .parse::<usize>()?;
    let mut config = PcFanConfig {
        mesh: PcFanMeshResolution {
            radial_segments: 2,
            chord_segments: 2,
            circumferential_segments: 8,
            harmonic_count: 2,
        },
        cfd: PcFanCfdConfig {
            flow_model: PcFanFlowModel::IncompressibleCfd,
            backend: PcFanCfdBackend::Cpu,
            cells_x: cells,
            cells_y: cells,
            cells_z: cells,
            use_direct_solver: true,
            ..PcFanCfdConfig::default()
        },
        ..PcFanConfig::default_120mm()
    };
    let design = PcFanDesignVariables::default();
    let domain = config.build_geometry(&design).build_cfd_domain(&config);
    let mesh = box_mesh_tetrahedra_symmetric(
        domain.min_corner.x,
        domain.max_corner.x,
        domain.min_corner.y,
        domain.max_corner.y,
        domain.min_corner.z,
        domain.max_corner.z,
        cells,
        cells,
        cells,
    );
    let mut state = vec![0.0; 4 * mesh.num_nodes()];
    let full_rpm = config.operating_point.rpm;
    for fraction in [
        1.0 / 600.0,
        2.0 / 600.0,
        4.0 / 600.0,
        8.0 / 600.0,
        16.0 / 600.0,
        24.0 / 600.0,
        32.0 / 600.0,
        48.0 / 600.0,
        64.0 / 600.0,
        96.0 / 600.0,
        128.0 / 600.0,
        192.0 / 600.0,
        256.0 / 600.0,
        384.0 / 600.0,
        512.0 / 600.0,
        1.0,
    ] {
        config.operating_point.rpm = full_rpm * fraction;
        println!(
            "Load continuation fraction {fraction}; rpm {}",
            config.operating_point.rpm
        );
        let mut converged = false;
        for iteration in 0..80 {
            let observation =
                evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &state)?;
            let residual = norm(&observation.state_residual);
            println!("Newton iteration {iteration}: residual {residual:.6e}");
            if residual <= 1.0e-10 {
                converged = true;
                break;
            }
            let j = observation.state_jacobian;
            let matrix = CsrMatrix::try_from_raw_parts(
                j.dimension,
                j.dimension,
                j.row_ptrs,
                j.column_indices,
                j.values,
            )?;
            let rhs = observation
                .state_residual
                .iter()
                .map(|value| -value)
                .collect::<Vec<_>>();
            let delta = SparseLuFactorization::factorize(&matrix)?.solve(&rhs)?;
            let mut step = 1.0;
            let mut accepted = false;
            for _ in 0..16 {
                let trial = state
                    .iter()
                    .zip(&delta)
                    .map(|(value, update)| value + step * update)
                    .collect::<Vec<_>>();
                let trial_observation =
                    evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &trial)?;
                let trial_residual = norm(&trial_observation.state_residual);
                if trial_residual.is_finite() && trial_residual <= (1.0 - 1.0e-4 * step) * residual
                {
                    println!("accepted step {step:.6e}; trial residual {trial_residual:.6e}");
                    state = trial;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if !accepted {
                return Err("Newton line search exhausted".into());
            }
        }
        if !converged {
            return Err("load-continuation Newton budget exhausted".into());
        }
    }
    if config.operating_point.rpm != full_rpm {
        return Err("continuation did not restore the original RPM".into());
    }
    if let Ok(path) = std::env::var("P2_PCFAN_CHECKPOINT") {
        std::fs::write(
            path,
            serde_json::to_vec(&serde_json::json!({
                "configuration": config, "design": design, "state": state,
                "accepted_iterations": 0, "true_equation_residual_budget": 1e-10,
            }))?,
        )?;
    }
    println!(
        "Full original operating point converged with independently assembled residual <=1e-10"
    );
    Ok(())
}
