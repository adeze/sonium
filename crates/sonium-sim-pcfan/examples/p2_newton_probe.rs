//! Diagnostic damped Newton solve of the unchanged native fan equations.

// Rust guideline compliant 2026-02-21

use sonium_fem::mesh::generators::box_mesh_tetrahedra_symmetric;
use sonium_sim_pcfan::{
    PcFanCfdBackend, PcFanCfdConfig, PcFanConfig, PcFanFlowModel, PcFanMeshResolution,
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
    let variant = std::env::args().nth(2).unwrap_or_default();
    let max_iterations = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "40".into())
        .parse::<usize>()?;
    if cells == 0 || max_iterations == 0 {
        return Err("positive mesh cells and iteration budget required".into());
    }
    let geometry_mesh = if variant == "qa" {
        PcFanMeshResolution {
            radial_segments: 2,
            chord_segments: 2,
            circumferential_segments: 8,
            harmonic_count: 2,
        }
    } else {
        PcFanConfig::default_120mm().mesh
    };
    let config = PcFanConfig {
        mesh: geometry_mesh,
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
    let line_search_steps = std::env::var("P2_PCFAN_LINE_SEARCH_STEPS")
        .map(|value| value.parse::<usize>())
        .unwrap_or(Ok(16))?;
    if !(1..=48).contains(&line_search_steps) {
        return Err("line-search trial budget must be between 1 and 48".into());
    }
    let equilibrated_merit = match std::env::var("P2_PCFAN_MERIT").as_deref() {
        Ok("row-equilibrated") => true,
        Ok("raw") | Err(_) => false,
        _ => return Err("merit must be raw or row-equilibrated".into()),
    };
    let mut initial_accepted_iterations = 0;
    if let Ok(path) = std::env::var("P2_PCFAN_INITIAL_STATE") {
        let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        if saved["configuration"] != serde_json::to_value(&config)?
            || saved["design"] != serde_json::to_value(&design)?
        {
            return Err("checkpoint physics or geometry differs from the requested fixture".into());
        }
        let initial: Vec<f64> = serde_json::from_value(saved["state"].clone())?;
        if initial.len() != state.len() || initial.iter().any(|value| !value.is_finite()) {
            return Err("checkpoint state is invalid".into());
        }
        initial_accepted_iterations = saved["accepted_iterations"]
            .as_u64()
            .ok_or("checkpoint lacks accepted-iteration provenance")?
            as usize;
        state = initial;
    }
    for iteration in 0..max_iterations {
        let observation = evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &state)?;
        let residual = norm(&observation.state_residual);
        println!("Newton iteration {iteration}: absolute residual norm {residual:.6e}");
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
        // Positive row weights normalize the local equation units for Armijo
        // acceptance. Keep them fixed throughout this iteration, retain every
        // equation, and keep the raw residual convergence target unchanged.
        let weights: Vec<_> = (0..matrix.num_rows)
            .map(|row| {
                if !equilibrated_merit {
                    return 1.0;
                }
                let norm = matrix.values[matrix.row_ptrs[row]..matrix.row_ptrs[row + 1]]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt();
                if norm == 0.0 { 1.0 } else { norm.recip() }
            })
            .collect();
        if weights.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err("invalid equation merit weights".into());
        }
        let merit = |values: &[f64]| {
            values
                .iter()
                .zip(&weights)
                .map(|(r, w)| (r * w).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let current_merit = merit(&observation.state_residual);
        if equilibrated_merit {
            println!("row-equilibrated full-equation merit {current_merit:.6e}");
        }
        let rhs = observation
            .state_residual
            .iter()
            .map(|value| -value)
            .collect::<Vec<_>>();
        let delta = SparseLuFactorization::factorize(&matrix)?.solve(&rhs)?;
        if std::env::var("P2_PCFAN_CHECK_TANGENT").as_deref() == Ok("1")
            || (iteration == 1 && std::env::args().nth(2).as_deref() == Some("tangent"))
        {
            let linear_residual = matrix
                .apply(&delta)
                .iter()
                .zip(&observation.state_residual)
                .map(|(j, r)| j + r)
                .collect::<Vec<_>>();
            println!(
                "checkpoint accepted iterations={initial_accepted_iterations}, state norm={:.6e}, Newton direction norm={:.6e}, relative linear equation residual={:.6e}",
                norm(&state),
                norm(&delta),
                norm(&linear_residual) / residual
            );
            let mut analytic = vec![0.0; state.len()];
            matrix.apply_into(&delta, &mut analytic);
            for epsilon in [1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8] {
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
        for _ in 0..line_search_steps {
            let trial = state
                .iter()
                .zip(&delta)
                .map(|(value, update)| value + step * update)
                .collect::<Vec<_>>();
            let trial_observation =
                evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &trial)?;
            let trial_residual = norm(&trial_observation.state_residual);
            let trial_merit = merit(&trial_observation.state_residual);
            if trial_residual.is_finite()
                && trial_merit.is_finite()
                && trial_merit <= (1.0 - 1.0e-4 * step) * current_merit
            {
                println!("accepted step {step:.6e}; trial residual {trial_residual:.6e}");
                state = trial;
                if let Ok(path) = std::env::var("P2_PCFAN_CHECKPOINT") {
                    let path = std::path::PathBuf::from(path);
                    let temporary = path.with_extension("pending.json");
                    let saved = serde_json::json!({
                        "configuration": config, "design": design, "state": state,
                        "accepted_iterations": initial_accepted_iterations + iteration + 1,
                        "true_equation_residual": trial_residual,
                    });
                    std::fs::write(&temporary, serde_json::to_vec(&saved)?)?;
                    std::fs::rename(temporary, path)?;
                }
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
