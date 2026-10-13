//! Diagnostic viscosity continuation ending at the unchanged native fan operating point.

// Rust guideline compliant 2026-02-21

use sonium_fem::mesh::generators::box_mesh_tetrahedra_symmetric;
use sonium_sim_pcfan::{
    PcFanCfdBackend, PcFanCfdConfig, PcFanConfig, PcFanFlowModel,
    evaluate_pc_fan_cfd_newton_state_linearization,
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
    let design = config.default_design_variables();
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
    let full_viscosity = config.operating_point.dynamic_viscosity_pa_s;
    for multiplier in [
        1024.0, 512.0, 256.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0,
    ] {
        config.operating_point.dynamic_viscosity_pa_s = full_viscosity * multiplier;
        println!(
            "Viscosity continuation multiplier {multiplier}; viscosity {}",
            config.operating_point.dynamic_viscosity_pa_s
        );
        let mut converged = false;
        for iteration in 0..80 {
            let observation =
                evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &state)?;
            let residual = norm(&observation.state_residual);
            println!("Newton iteration {iteration}: residual {residual:.6e}");
            if residual <= 1.0e-8 {
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
            return Err("viscosity-continuation Newton budget exhausted".into());
        }
    }
    println!(
        "Full original operating point converged with independently assembled residual <=1e-8"
    );
    Ok(())
}
