//! Diagnostic regularized Newton startup for the unchanged native fan equations.

// Rust guideline compliant 2026-02-21

use sonium_fem::mesh::generators::box_mesh_tetrahedra_symmetric;
use sonium_sim_pcfan::{
    PcFanCfdBackend, PcFanCfdConfig, PcFanConfig, PcFanFlowModel, PcFanMeshResolution,
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
    let velocity_dimension = 3 * mesh.num_nodes();
    let mut regularization = 1000.0_f64;
    for iteration in 0..100 {
        let observation = evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &state)?;
        let residual = norm(&observation.state_residual);
        println!("Newton iteration {iteration}: residual {residual:.6e}");
        if residual <= 1.0e-8 {
            return Ok(());
        }
        let j = observation.state_jacobian;
        let mut matrix = CsrMatrix::try_from_raw_parts(
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
        // Only the Newton linear system is regularized. The measured residual
        // above and every trial residual below use the original steady equations.
        for row in 0..velocity_dimension {
            let begin = matrix.row_ptrs[row];
            let end = matrix.row_ptrs[row + 1];
            let scale = (begin..end)
                .filter(|&entry| matrix.col_indices[entry] < velocity_dimension)
                .map(|entry| matrix.values[entry].abs())
                .fold(0.0_f64, f64::max);
            if let Some(diagonal) = (begin..end).find(|&entry| matrix.col_indices[entry] == row) {
                matrix.values[diagonal] += regularization * scale;
            }
        }
        let delta = SparseLuFactorization::factorize(&matrix)?.solve(&rhs)?;
        println!("velocity regularization {regularization:.6e}");
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
            if trial_residual.is_finite() && trial_residual <= (1.0 - 1.0e-4 * step) * residual {
                println!("accepted step {step:.6e}; trial residual {trial_residual:.6e}");
                state = trial;
                regularization = if step >= 0.5 {
                    (regularization * 0.5).max(1.0e-8)
                } else {
                    regularization * 2.0
                };
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            regularization *= 10.0;
            println!("No accepted step; increase velocity regularization to {regularization:.6e}");
        }
    }
    Err("Newton iteration budget exhausted".into())
}
