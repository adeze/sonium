//! Diagnostic certification of the native fan Newton linear solve.

// Rust guideline compliant 2026-02-21

use sonium_fem::mesh::generators::box_mesh_tetrahedra_symmetric;
use sonium_sim_pcfan::{
    PcFanCfdBackend, PcFanCfdConfig, PcFanConfig, PcFanFlowModel,
    evaluate_pc_fan_cfd_newton_state_linearization,
};
use sonium_solvers::traits::LinearOperator;
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
    for iteration in 0..40 {
        let observation = evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &state)?;
        let residual = norm(&observation.state_residual);
        println!("Newton iteration {iteration}: residual {residual:.6e}");
        if residual <= 1.0e-8 {
            return Ok(());
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
        let factors = SparseLuFactorization::factorize(&matrix)?;
        let delta = factors.solve(&rhs)?;
        let difference = matrix
            .matvec(&delta)
            .iter()
            .zip(&rhs)
            .map(|(value, expected)| value - expected)
            .collect::<Vec<_>>();
        println!(
            "Newton linear step relative residual {:.6e}",
            norm(&difference) / norm(&rhs)
        );
        if iteration == 1 {
            return Ok(());
        }
        if iteration == 1 && std::env::args().nth(2).as_deref() == Some("tangent") {
            let mut analytic = vec![0.0; state.len()];
            matrix.apply_into(&delta, &mut analytic);
            for epsilon in [1.0e-4, 1.0e-5, 1.0e-6] {
                let plus = state
                    .iter()
                    .zip(&delta)
                    .map(|(value, update)| value + epsilon * update)
                    .collect::<Vec<_>>();
                let minus = state
                    .iter()
                    .zip(&delta)
                    .map(|(value, update)| value - epsilon * update)
                    .collect::<Vec<_>>();
                let plus = evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &plus)?;
                let minus =
                    evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &minus)?;
                let difference = plus
                    .state_residual
                    .iter()
                    .zip(&minus.state_residual)
                    .zip(&analytic)
                    .map(|((plus, minus), expected)| (plus - minus) / (2.0 * epsilon) - expected)
                    .collect::<Vec<_>>();
                println!(
                    "directional tangent epsilon {epsilon:.1e}: relative error {:.6e}",
                    norm(&difference) / norm(&analytic)
                );
            }
            return Ok(());
        }
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
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            return Err("Newton line search exhausted".into());
        }
    }
    Err("Newton iteration budget exhausted".into())
}
