//! Diagnose trust-region convergence of the unchanged original fan equations.
//!
//! Positive variable scaling affects proposals only. Every trial
//! evaluates the original physical residual; convergence requires its full
//! unscaled norm. Checkpoints are diagnostics, not conservation or mesh QA.

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

fn dogleg(cauchy: &[f64], newton: &[f64], radius: f64) -> Vec<f64> {
    if norm(newton) <= radius {
        return newton.to_vec();
    }
    let size = norm(cauchy);
    if size >= radius {
        return cauchy.iter().map(|v| v * (radius / size)).collect();
    }
    let direction = newton
        .iter()
        .zip(cauchy)
        .map(|(n, c)| n - c)
        .collect::<Vec<_>>();
    let a = dot(&direction, &direction);
    let b = dot(cauchy, &direction);
    let remaining = radius * radius - dot(cauchy, cauchy);
    let discriminant = (b * b + a * remaining).sqrt();
    // Rationalize the positive root when b>=0 to avoid cancellation.
    let fraction = if b >= 0.0 {
        remaining / (discriminant + b)
    } else {
        (discriminant - b) / a
    };
    cauchy
        .iter()
        .zip(direction)
        .map(|(c, d)| c + fraction * d)
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cells: usize = std::env::args().nth(1).map_or(Ok(8), |v| v.parse())?;
    let budget: usize = std::env::args().nth(2).map_or(Ok(100), |v| v.parse())?;
    if !(2..=16).contains(&cells) || budget == 0 {
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
        || saved["viscosity_multiplier"].as_f64().unwrap_or(1.0) != 1.0
    {
        return Err("checkpoint differs from original physical fan fixture".into());
    }
    let mut state: Vec<f64> = serde_json::from_value(saved["state"].clone())?;
    if state.is_empty() || state.iter().any(|v| !v.is_finite()) {
        return Err("invalid checkpoint state".into());
    }
    let mut accepted = saved["accepted_iterations"]
        .as_u64()
        .ok_or("missing checkpoint provenance")?;
    let path = std::path::PathBuf::from(std::env::var("P2_PCFAN_CHECKPOINT")?);
    // Radius is in positively equilibrated variable coordinates. It limits
    // the proposal only; no physical mass,viscosity or equation is modified.
    let mut radius = 1.0_f64;
    for iteration in 0..budget {
        let observation = evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &state)?;
        observation.validate()?;
        if observation.state != state {
            return Err("assembly changed supplied state".into());
        }
        let r = &observation.state_residual;
        let raw = norm(r);
        println!("Dogleg iteration{iteration}:original residual={raw:.12e},radius={radius:.6e}");
        if raw <= 1e-8 {
            println!("Original residual converged;conservation and fullmesh QA unproven");
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
        let rhs = r.iter().map(|v| -v).collect::<Vec<_>>();
        let delta = SparseLuFactorization::factorize(&matrix)?.solve(&rhs)?;
        let linear_error = matrix
            .apply(&delta)
            .iter()
            .zip(r)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        let linear_relative = norm(&linear_error) / raw;
        if !linear_relative.is_finite() || linear_relative > 1e-8 {
            return Err("full Newton equation did not converge".into());
        }
        // The merit uses the raw residual required for final admission.
        // Row-equilibrated Cauchy descent can increase this norm even in the
        // infinitesimal limit; rejecting such steps by shrinking cannot help.
        // Keep equation weights equal to one and scale variables instead.
        let weights = vec![1.0; matrix.num_rows];
        let mut balanced = matrix.clone();
        let mut scales = vec![0.0; matrix.num_cols];
        for (row, &weight) in weights.iter().enumerate() {
            for entry in balanced.row_range(row) {
                balanced.values[entry] *= weight;
                scales[balanced.col_indices[entry]] += balanced.values[entry].powi(2);
            }
        }
        for scale in &mut scales {
            *scale = if *scale == 0.0 {
                1.0
            } else {
                scale.sqrt().recip()
            };
        }
        if weights
            .iter()
            .chain(&scales)
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err("invalid positive scaling".into());
        }
        for entry in 0..balanced.values.len() {
            balanced.values[entry] *= scales[balanced.col_indices[entry]];
        }
        let weighted = r
            .iter()
            .zip(&weights)
            .map(|(r, w)| r * w)
            .collect::<Vec<_>>();
        let mut gradient = vec![0.0; state.len()];
        for (row, &value) in weighted.iter().enumerate() {
            for entry in balanced.row_range(row) {
                gradient[balanced.col_indices[entry]] += balanced.values[entry] * value;
            }
        }
        let forward = balanced.apply(&gradient);
        let denominator = dot(&forward, &forward);
        if denominator <= 0.0 || !denominator.is_finite() {
            return Err("nonzero residual has no finite descent direction".into());
        }
        let alpha = dot(&gradient, &gradient) / denominator;
        let cauchy = gradient.iter().map(|g| -alpha * g).collect::<Vec<_>>();
        let newton = delta
            .iter()
            .zip(&scales)
            .map(|(d, s)| d / s)
            .collect::<Vec<_>>();
        let current = dot(&weighted, &weighted);
        let mut accepted_trial = false;
        for attempt in 0..32 {
            let step = dogleg(&cauchy, &newton, radius);
            let model = balanced
                .apply(&step)
                .iter()
                .zip(&weighted)
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>();
            let predicted = current - dot(&model, &model);
            let trial = state
                .iter()
                .zip(&step)
                .zip(&scales)
                .map(|((u, z), s)| u + s * z)
                .collect::<Vec<_>>();
            let measured =
                evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &trial)?;
            measured.validate()?;
            let trial_raw = norm(&measured.state_residual);
            let trial_weighted = measured
                .state_residual
                .iter()
                .zip(&weights)
                .map(|(r, w)| r * w)
                .collect::<Vec<_>>();
            let gain = (current - dot(&trial_weighted, &trial_weighted)) / predicted;
            println!(
                "Dogleg attempt{attempt}:radius={radius:.6e},original trial residual={trial_raw:.12e},gain={gain:.6e},Newton linear residual={linear_relative:.6e}"
            );
            if gain < 0.25 || !gain.is_finite() {
                radius *= 0.25;
            } else if gain > 0.75 && norm(&step) >= 0.99 * radius {
                radius = (2.0 * radius).min(1e8);
            }
            // Require raw improvement as well as fixed-weight model agreement;
            // every equation participates in both admission measurements.
            if predicted > 0.0 && gain.is_finite() && gain > 1e-4 && trial_raw < raw {
                state = trial;
                accepted += 1;
                let temporary = path.with_extension("pending.json");
                std::fs::write(
                    &temporary,
                    serde_json::to_vec(&serde_json::json!({"configuration":config,
                    "design":design,"state":state,"accepted_iterations":accepted,"true_equation_residual":trial_raw,
                    "radius":radius,"gain_ratio":gain,"independent_newton_linear_relative_residual":linear_relative,
                    "diagnostic":"original full equations;scaled dogleg trust region;originalQA unproven"}))?,
                )?;
                std::fs::rename(temporary, &path)?;
                accepted_trial = true;
                break;
            }
            if gain >= 0.25 {
                radius *= 0.25;
            }
            if radius < 1e-14 {
                return Err("trust region exhausted before original convergence".into());
            }
        }
        if !accepted_trial {
            return Err("no accepted original-equation dogleg update".into());
        }
    }
    Err("dogleg iteration budget exhausted;original convergence unproven".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dogleg_preserves_newton_or_intersects_radius_on_both_segments() {
        let cauchy = [1.0, 0.0];
        let newton = [2.0, 2.0];
        assert_eq!(dogleg(&cauchy, &newton, 3.0), newton);
        assert_eq!(dogleg(&cauchy, &newton, 0.5), [0.5, 0.0]);
        let step = dogleg(&cauchy, &newton, 2.0);
        assert!((norm(&step) - 2.0).abs() < 1e-14);
        assert!((step[1] - 2.0 * (step[0] - 1.0)).abs() < 1e-14);
        // Opposing components exercise the other quadratic-root branch.
        let step = dogleg(&[1.0, 0.0], &[-2.0, 3.0], 2.0);
        assert!((norm(&step) - 2.0).abs() < 1e-14);
        assert!((step[1] - (1.0 - step[0])).abs() < 1e-14);
    }

    #[test]
    fn dogleg_reduces_independent_raw_quadratic_on_every_path_segment() {
        // J=[[1,2],[3,4]],r=[1,2]. Exact Newton is[0,-.5]; the
        // steepest-descent minimizer is -(149/4450)*[7,10].
        let cauchy = [-7.0 * 149.0 / 4450.0, -10.0 * 149.0 / 4450.0];
        for radius in [1e-4, 0.45, 1.0] {
            let step = dogleg(&cauchy, &[0.0, -0.5], radius);
            let residual = [
                1.0 + step[0] + 2.0 * step[1],
                2.0 + 3.0 * step[0] + 4.0 * step[1],
            ];
            assert!(dot(&residual, &residual) < 5.0);
            assert!(norm(&step) <= radius + 1e-14);
            if radius == 1.0 {
                assert_eq!(residual, [0.0; 2]);
            }
        }
    }
}
