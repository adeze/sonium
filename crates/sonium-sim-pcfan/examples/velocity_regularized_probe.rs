//! Diagnose velocity-regularized Newton steps on the original fan equations.
//!
//! The temporary diagonal affects proposals only. Pressure equations retain
//! their full original rows. Admission and convergence use the unmodified
//! physical residual; conservation and the original mesh ladder remain required.

// Rust guideline compliant 2026-02-21

use sonium_sim_pcfan::{
    PcFanCfdBackend, PcFanCfdConfig, PcFanConfig, PcFanFlowModel, PcFanMeshResolution,
    evaluate_pc_fan_cfd_newton_state_linearization,
};
use sonium_solvers::{CsrMatrix, direct::SparseLuFactorization, traits::LinearOperator};

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn velocity_weights(
    matrix: &CsrMatrix<f64>,
    velocity_dofs: usize,
) -> Result<Vec<f64>, &'static str> {
    if matrix.num_rows != matrix.num_cols || velocity_dofs == 0 || velocity_dofs >= matrix.num_rows
    {
        return Err("invalid velocity/pressure layout");
    }
    let mut weights = vec![0.0; matrix.num_rows];
    for (row, weight) in weights.iter_mut().enumerate().take(velocity_dofs) {
        let identity = matrix.row_range(row).all(|e| {
            if matrix.col_indices[e] == row {
                matrix.values[e] == 1.0
            } else {
                matrix.values[e] == 0.0
            }
        }) && matrix.row_range(row).any(|e| matrix.col_indices[e] == row);
        if identity {
            continue;
        }
        *weight = matrix
            .row_range(row)
            .filter(|&e| matrix.col_indices[e] < velocity_dofs)
            .map(|e| matrix.values[e].powi(2))
            .sum::<f64>()
            .sqrt();
        if !weight.is_finite() || *weight <= 0.0 {
            return Err("invalid free velocity row scale");
        }
    }
    Ok(weights)
}

fn shifted_matrix(
    matrix: &CsrMatrix<f64>,
    weights: &[f64],
    shift: f64,
) -> Result<CsrMatrix<f64>, Box<dyn std::error::Error>> {
    if weights.len() != matrix.num_rows
        || !shift.is_finite()
        || shift <= 0.0
        || weights.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err("invalid positive velocity regularization".into());
    }
    let mut rows = Vec::with_capacity(matrix.row_ptrs.len());
    let mut columns = Vec::with_capacity(matrix.values.len() + matrix.num_rows);
    let mut values = Vec::with_capacity(columns.capacity());
    rows.push(0);
    for (row, &weight) in weights.iter().enumerate() {
        let addition = shift * weight;
        if !addition.is_finite() {
            return Err("velocity shift overflow".into());
        }
        let mut inserted = false;
        for entry in matrix.row_range(row) {
            let column = matrix.col_indices[entry];
            if column > row && !inserted && addition != 0.0 {
                columns.push(row);
                values.push(addition);
                inserted = true;
            }
            columns.push(column);
            if column == row {
                values.push(matrix.values[entry] + addition);
                inserted = true;
            } else {
                values.push(matrix.values[entry]);
            }
        }
        if !inserted && addition != 0.0 {
            columns.push(row);
            values.push(addition);
        }
        rows.push(values.len());
    }
    Ok(CsrMatrix::try_from_raw_parts(
        matrix.num_rows,
        matrix.num_cols,
        rows,
        columns,
        values,
    )?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cells: usize = std::env::args().nth(1).map_or(Ok(8), |v| v.parse())?;
    let budget: usize = std::env::args().nth(2).map_or(Ok(100), |v| v.parse())?;
    if !(2..=16).contains(&cells) || budget == 0 {
        return Err("cells in[2,16] and positive budget required".into());
    }
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
            ..Default::default()
        },
        ..PcFanConfig::default_120mm()
    };
    let design = config.default_design_variables();
    let saved: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::var("P2_PCFAN_INITIAL_STATE")?)?)?;
    if saved["configuration"] != serde_json::to_value(&config)?
        || saved["design"] != serde_json::to_value(&design)?
        || saved["viscosity_multiplier"].as_f64().unwrap_or(1.0) != 1.0
    {
        return Err("checkpoint differs from original physical fan fixture".into());
    }
    let mut state: Vec<f64> = serde_json::from_value(saved["state"].clone())?;
    // The production state-only API assembles P1 velocity (three components)
    // and P1 pressure on the same nodes, in consecutive blocks.
    if state.is_empty() || !state.len().is_multiple_of(4) || state.iter().any(|v| !v.is_finite()) {
        return Err("invalid original P1/P1 checkpoint layout".into());
    }
    let velocity_dofs = 3 * (state.len() / 4);
    let mut accepted = saved["accepted_iterations"]
        .as_u64()
        .ok_or("missing accepted-state provenance")?;
    let path = std::path::PathBuf::from(std::env::var("P2_PCFAN_CHECKPOINT")?);
    let mut shift: f64 = std::env::var("P2_PCFAN_VELOCITY_SHIFT")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(1.0);
    if !shift.is_finite() || shift <= 0.0 {
        return Err("positive finite velocity shift required".into());
    }
    for iteration in 0..budget {
        let current = evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &state)?;
        current.validate()?;
        if current.state != state {
            return Err("assembly changed supplied state".into());
        }
        let r = &current.state_residual;
        let raw = norm(r);
        println!(
            "Velocity regularization iteration{iteration}:original residual={raw:.12e},shift={shift:.6e}"
        );
        if raw <= 1e-8 {
            println!("Original residual converged;conservation/fullmeshQA unproven");
            return Ok(());
        }
        let j = current.state_jacobian;
        let matrix = CsrMatrix::try_from_raw_parts(
            j.dimension,
            j.dimension,
            j.row_ptrs,
            j.column_indices,
            j.values,
        )?;
        let weights = velocity_weights(&matrix, velocity_dofs)?;
        let rhs = r.iter().map(|v| -v).collect::<Vec<_>>();
        let mut accepted_trial = false;
        for attempt in 0..12 {
            let regularized = shifted_matrix(&matrix, &weights, shift)?;
            let delta = SparseLuFactorization::factorize(&regularized)?.solve(&rhs)?;
            let error = regularized
                .apply(&delta)
                .iter()
                .zip(r)
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>();
            let linear_relative = norm(&error) / raw;
            if !linear_relative.is_finite() || linear_relative > 1e-8 {
                return Err("regularized linear equation did not converge".into());
            }
            let slope = dot(r, &matrix.apply(&delta));
            if !slope.is_finite() || slope >= 0.0 {
                println!("Non-descent shifted direction at shift={shift:.6e};reduce shift");
                shift *= 0.25;
                continue;
            }
            let mut alpha = 1.0;
            for backtrack in 0..24 {
                let trial = state
                    .iter()
                    .zip(&delta)
                    .map(|(u, d)| u + alpha * d)
                    .collect::<Vec<_>>();
                let measured =
                    evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &trial)?;
                measured.validate()?;
                if measured.state != trial {
                    return Err("trial assembly changed supplied state".into());
                }
                let trial_raw = norm(&measured.state_residual);
                println!(
                    "Velocity regularization attempt{attempt}/{backtrack}:shift={shift:.6e},alpha={alpha:.6e},original residual={trial_raw:.12e},linear residual={linear_relative:.6e}"
                );
                // Armijo uses the actual original derivative and every raw
                // residual row. No temporary mass enters this measurement.
                if trial_raw < raw && trial_raw * trial_raw <= raw * raw + 2e-4 * alpha * slope {
                    state = trial;
                    accepted += 1;
                    let used_shift = shift;
                    shift = if alpha >= 0.5 {
                        (shift * 0.5).max(1e-12)
                    } else {
                        (shift * 2.0).min(1e12)
                    };
                    let temporary = path.with_extension("pending.json");
                    std::fs::write(
                        &temporary,
                        serde_json::to_vec(&serde_json::json!({"configuration":config,
                        "design":design,"state":state,"accepted_iterations":accepted,"true_equation_residual":trial_raw,
                        "used_velocity_shift":used_shift,"next_velocity_shift":shift,"step_fraction":alpha,
                        "independent_regularized_linear_relative_residual":linear_relative,
                        "diagnostic":"original full equations;velocity-only regularized Newton;originalQA unproven"}))?,
                    )?;
                    std::fs::rename(temporary, &path)?;
                    accepted_trial = true;
                    break;
                }
                alpha *= 0.5;
            }
            if accepted_trial {
                break;
            }
            shift *= 4.0;
            if !shift.is_finite() || !(1e-14..=1e14).contains(&shift) {
                return Err("velocity regularization range exhausted".into());
            }
        }
        if !accepted_trial {
            return Err("no accepted original-equation regularized update".into());
        }
    }
    Err("regularized iteration budget exhausted;original convergence unproven".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifted_saddle_preserves_pressure_equation_and_independent_linear_identity() {
        let matrix = CsrMatrix::try_from_raw_parts(
            2,
            2,
            vec![0, 2, 3],
            vec![0, 1, 0],
            vec![2.0, -1.0, -1.0],
        )
        .unwrap();
        let weights = velocity_weights(&matrix, 1).unwrap();
        let shifted = shifted_matrix(&matrix, &weights, 3.0).unwrap();
        let delta = SparseLuFactorization::factorize(&shifted)
            .unwrap()
            .solve(&[-1.0, -0.25])
            .unwrap();
        assert!((delta[0] - 0.25).abs() < 1e-14);
        assert!((delta[1] - 3.0).abs() < 1e-14);
        let original = matrix.apply(&delta);
        assert!((original[0] + 1.0 + 1.5).abs() < 1e-14);
        assert!((original[1] + 0.25).abs() < 1e-14);
        // A half step decreases the independent original affine residual.
        assert!(norm(&[1.0 + original[0] * 0.5, 0.25 + original[1] * 0.5]) < norm(&[1.0, 0.25]));
    }

    #[test]
    fn velocity_constraints_and_pressure_rows_are_unchanged() {
        let matrix = CsrMatrix::try_from_raw_parts(
            3,
            3,
            vec![0, 1, 3, 4],
            vec![0, 1, 2, 1],
            vec![1.0, 2.0, -1.0, -1.0],
        )
        .unwrap();
        let shifted = shifted_matrix(&matrix, &velocity_weights(&matrix, 2).unwrap(), 3.0).unwrap();
        assert_eq!(shifted.apply(&[2.0, 1.0, 4.0]), vec![2.0, 4.0, -1.0]);
        assert!(shifted_matrix(&matrix, &[0.0, 2.0, 0.0], f64::NAN).is_err());
    }
}
