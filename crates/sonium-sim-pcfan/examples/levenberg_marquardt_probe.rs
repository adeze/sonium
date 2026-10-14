//! Diagnose globalization of the unchanged original steady fan equations.
//!
//! Uses matrix-free damped normal equations with positive row/column scaling.
//! Every trial and the final convergence check evaluate the original native
//! equations. Checkpoints preserve exact original physics/design identity.
//! No original QA convergence, conservation or mesh accuracy is assumed.

// Rust guideline compliant 2026-02-21

use sonium_sim_pcfan::{
    PcFanCfdBackend, PcFanCfdConfig, PcFanConfig, PcFanFlowModel, PcFanMeshResolution,
    evaluate_pc_fan_cfd_newton_state_linearization,
};
use sonium_solvers::traits::LinearOperator;
use sonium_solvers::{CgConfig, CgSolution, CsrMatrix, try_cg};

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

struct Normal<'a> {
    scaled: &'a CsrMatrix<f64>,
    damping: f64,
}

fn transpose_action(matrix: &CsrMatrix<f64>, values: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; matrix.num_cols];
    for (row, &value) in values.iter().enumerate() {
        for entry in matrix.row_range(row) {
            result[matrix.col_indices[entry]] += matrix.values[entry] * value;
        }
    }
    result
}

impl LinearOperator<f64> for Normal<'_> {
    fn num_rows(&self) -> usize {
        self.num_cols()
    }
    fn num_cols(&self) -> usize {
        self.scaled.num_cols
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        let forward = self.scaled.apply(x);
        let adjoint = transpose_action(self.scaled, &forward);
        for ((target, &value), &input) in y.iter_mut().zip(&adjoint).zip(x) {
            *target = value + self.damping * input;
        }
    }
    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        // B^T B plus a real diagonal is symmetric.
        self.apply_into(x, y);
    }
}

fn solve_normal_equation(
    operator: &Normal<'_>,
    rhs: &[f64],
    config: &CgConfig<f64>,
) -> Result<CgSolution<f64>, Box<dyn std::error::Error>> {
    let size = norm(rhs);
    if !size.is_finite() {
        return Err("nonfinite normal-equation right-hand-side norm".into());
    }
    // CG's dot-product breakdown guard has an absolute floor. Solving with
    // a unit right-hand side avoids premature breakdown for small gradients.
    // Scaling the update back preserves the original damped equation exactly;
    // the caller independently checks that equation before accepting a step.
    if size == 0.0 {
        return Ok(try_cg(operator, rhs, config)?);
    }
    let unit_rhs = rhs.iter().map(|v| v / size).collect::<Vec<_>>();
    let mut result = try_cg(operator, &unit_rhs, config)?;
    for value in &mut result.x {
        *value *= size;
    }
    Ok(result)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cells: usize = std::env::args().nth(1).map_or(Ok(8), |v| v.parse())?;
    let max_iterations: usize = std::env::args().nth(2).map_or(Ok(200), |v| v.parse())?;
    if !(2..=16).contains(&cells) || max_iterations == 0 {
        return Err("cells in[2,16] and positive iteration budget required".into());
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
    {
        return Err("checkpoint differs from original physical fan fixture".into());
    }
    let mut state: Vec<f64> = serde_json::from_value(saved["state"].clone())?;
    if state.is_empty() || state.iter().any(|v| !v.is_finite()) {
        return Err("invalid checkpoint state".into());
    }
    let accepted_initial = saved["accepted_iterations"]
        .as_u64()
        .ok_or("checkpoint provenance missing")?;
    let checkpoint = std::path::PathBuf::from(std::env::var("P2_PCFAN_CHECKPOINT")?);
    // On balanced coordinates this starts with moderate regularization. The
    // gain ratio changes only the update system, never the physical equations.
    let mut damping = 1e-3_f64;
    let mut growth = 2.0;
    for iteration in 0..max_iterations {
        let observation = evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &state)?;
        let raw = norm(&observation.state_residual);
        println!(
            "LM iteration{iteration}: original absolute residual={raw:.12e},damping={damping:.6e},state norm={:.6e}",
            norm(&state)
        );
        if !raw.is_finite() {
            return Err("nonfinite original steady residual".into());
        }
        if raw <= 1e-8 {
            println!(
                "Original steady residual converged; diagnostic only, conservation and original QA unproven"
            );
            return Ok(());
        }
        let j = observation.state_jacobian;
        let mut balanced = CsrMatrix::try_from_raw_parts(
            j.dimension,
            j.dimension,
            j.row_ptrs,
            j.column_indices,
            j.values,
        )?;
        let row_weights = (0..balanced.num_rows)
            .map(|row| {
                let size = norm(&balanced.values[balanced.row_range(row)]);
                if size == 0.0 { 1.0 } else { size.recip() }
            })
            .collect::<Vec<_>>();
        for (row, &weight) in row_weights.iter().enumerate() {
            for entry in balanced.row_range(row) {
                balanced.values[entry] *= weight;
            }
        }
        let mut column_scales = vec![0.0; balanced.num_cols];
        for entry in 0..balanced.values.len() {
            column_scales[balanced.col_indices[entry]] += balanced.values[entry].powi(2);
        }
        for scale in &mut column_scales {
            *scale = if *scale == 0.0 {
                1.0
            } else {
                scale.sqrt().recip()
            };
        }
        if row_weights
            .iter()
            .chain(&column_scales)
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err("invalid positive equation/variable scaling".into());
        }
        for entry in 0..balanced.values.len() {
            balanced.values[entry] *= column_scales[balanced.col_indices[entry]];
        }
        let weighted = observation
            .state_residual
            .iter()
            .zip(&row_weights)
            .map(|(&r, &w)| r * w)
            .collect::<Vec<_>>();
        let current_squared = norm(&weighted).powi(2);
        let rhs = transpose_action(&balanced, &weighted)
            .into_iter()
            .map(|v| -v)
            .collect::<Vec<_>>();
        let mut accepted = false;
        for attempt in 0..24 {
            let operator = Normal {
                scaled: &balanced,
                damping,
            };
            let solution = solve_normal_equation(
                &operator,
                &rhs,
                &CgConfig {
                    max_iterations: 4000,
                    tolerance: 1e-9,
                    print_interval: 0,
                },
            )?;
            // Independently inspect the actual damped normal equation; CG's
            // success flag or recurrence residual alone cannot admit an update.
            let actual = operator
                .apply(&solution.x)
                .iter()
                .zip(&rhs)
                .map(|(&a, &b)| a - b)
                .collect::<Vec<_>>();
            let relative = norm(&actual) / norm(&rhs).max(1e-300);
            println!(
                "LM attempt{attempt}: CG iterations={},actual linear residual={relative:.6e},damping={damping:.6e}",
                solution.iterations
            );
            if !solution.converged || !relative.is_finite() || relative > 1e-8 {
                damping *= growth;
                growth *= 2.0;
                continue;
            }
            let prediction = balanced
                .apply(&solution.x)
                .iter()
                .zip(&weighted)
                .map(|(&a, &b)| a + b)
                .collect::<Vec<_>>();
            let predicted_reduction = current_squared - norm(&prediction).powi(2);
            let trial = state
                .iter()
                .zip(&solution.x)
                .zip(&column_scales)
                .map(|((&u, &z), &s)| u + s * z)
                .collect::<Vec<_>>();
            let result = evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &trial)?;
            let trial_raw = norm(&result.state_residual);
            let trial_weighted = result
                .state_residual
                .iter()
                .zip(&row_weights)
                .map(|(&r, &w)| r * w)
                .collect::<Vec<_>>();
            let gain = (current_squared - norm(&trial_weighted).powi(2)) / predicted_reduction;
            println!(
                "LM actual original trial residual={trial_raw:.12e},weighted merit={:.6e},gain={gain:.6e}",
                norm(&trial_weighted)
            );
            if predicted_reduction > 0.0 && gain.is_finite() && gain > 1e-4 && trial_raw.is_finite()
            {
                state = trial;
                damping *= (1.0 - (2.0 * gain - 1.0).powi(3)).clamp(1.0 / 3.0, 2.0);
                damping = damping.clamp(1e-12, 1e12);
                growth = 2.0;
                let temporary = checkpoint.with_extension("pending.json");
                std::fs::write(
                    &temporary,
                    serde_json::to_vec(&serde_json::json!({
                        "configuration":config,"design":design,"state":state,
                        "accepted_iterations":accepted_initial+iteration as u64+1,
                        "true_equation_residual":trial_raw,"diagnostic":"original equations; matrix-free scaled LM; original QA unproven",
                        "damping":damping,"weighted_equation_merit":norm(&trial_weighted),"gain_ratio":gain,
                        "independent_normal_equation_relative_residual":relative
                    }))?,
                )?;
                std::fs::rename(temporary, &checkpoint)?;
                accepted = true;
                break;
            }
            damping *= growth;
            growth *= 2.0;
            if !damping.is_finite() || damping > 1e12 {
                return Err("LM trust-region proposals exhausted".into());
            }
        }
        if !accepted {
            return Err("LM iteration has no accepted original-equation update".into());
        }
    }
    Err("LM iteration budget exhausted; original convergence unproven".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_free_normal_solve_matches_independent_dense_solution() {
        let matrix = CsrMatrix::try_from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let operator = Normal {
            scaled: &matrix,
            damping: 0.1,
        };
        assert_eq!(operator.apply(&[1.0, 0.0]), [10.1, 14.0]);
        assert_eq!(operator.apply(&[0.0, 1.0]), [14.0, 20.1]);
        let result = try_cg(
            &operator,
            &[1.0, 2.0],
            &CgConfig {
                max_iterations: 10,
                tolerance: 1e-12,
                print_interval: 0,
            },
        )
        .unwrap();
        assert!(result.converged);
        let expected = [-7.9 / 7.01, 6.2 / 7.01];
        for (&x, &e) in result.x.iter().zip(&expected) {
            assert!((x - e).abs() < 1e-12);
        }
        let residual = operator
            .apply(&result.x)
            .iter()
            .zip([1.0, 2.0])
            .map(|(&a, b)| a - b)
            .collect::<Vec<_>>();
        assert!(norm(&residual) < 1e-12);
    }

    #[test]
    fn small_normal_rhs_retains_independent_relative_accuracy() {
        let matrix = CsrMatrix::try_from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let operator = Normal {
            scaled: &matrix,
            damping: 0.1,
        };
        for scale in [1e-16, 1e-8, 1.0, 1e8] {
            let rhs = [scale, 2.0 * scale];
            let result = solve_normal_equation(
                &operator,
                &rhs,
                &CgConfig {
                    max_iterations: 10,
                    tolerance: 1e-12,
                    print_interval: 0,
                },
            )
            .unwrap();
            assert!(result.converged);
            for (&x, exact) in result.x.iter().zip([-7.9 / 7.01, 6.2 / 7.01]) {
                assert!((x / scale - exact).abs() < 1e-12);
            }
            let actual = operator
                .apply(&result.x)
                .iter()
                .zip(rhs)
                .map(|(&a, b)| a - b)
                .collect::<Vec<_>>();
            assert!(norm(&actual) / norm(&rhs) < 1e-12);
        }
    }
}
