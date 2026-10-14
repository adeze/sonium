//! Locate the remaining original steady fan residual without advancing its state.
//!
//! Reports momentum and continuity equation norms plus the scaled least-squares
//! gradient. All rows come from the native constrained Newton assembly. This
//! read-only diagnostic provides no convergence or conservation admission.

// Rust guideline compliant 2026-02-21

use sonium_sim_pcfan::{
    PcFanConfig, PcFanDesignVariables, evaluate_pc_fan_cfd_newton_state_linearization,
};
use sonium_solvers::{CsrMatrix, direct::SparseLuFactorization, traits::LinearOperator};

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn checkpoint_configuration(
    saved: &serde_json::Value,
) -> Result<PcFanConfig, Box<dyn std::error::Error>> {
    let original: PcFanConfig = serde_json::from_value(saved["configuration"].clone())?;
    let multiplier = saved["viscosity_multiplier"].as_f64().unwrap_or(1.0);
    if !multiplier.is_finite() || multiplier <= 0.0 {
        return Err("finite positive checkpoint viscosity multiplier required".into());
    }
    if saved.get("effective_configuration").is_none() {
        if multiplier != 1.0 {
            return Err("modified viscosity checkpoint lacks effective configuration".into());
        }
        return Ok(original);
    }
    let mut expected = original;
    expected.operating_point.dynamic_viscosity_pa_s *= multiplier;
    if saved["effective_configuration"] != serde_json::to_value(&expected)? {
        return Err("effective checkpoint changes more than the declared viscosity".into());
    }
    Ok(expected)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var("P2_PCFAN_INITIAL_STATE")?;
    let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    let config = checkpoint_configuration(&saved)?;
    let design: PcFanDesignVariables = serde_json::from_value(saved["design"].clone())?;
    let state: Vec<f64> = serde_json::from_value(saved["state"].clone())?;
    let observation = evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &state)?;
    observation.validate()?;
    if observation.state != state || !state.len().is_multiple_of(4) {
        return Err("native assembly changed the supplied four-field state".into());
    }
    let nodes = state.len() / 4;
    let j = &observation.state_jacobian;
    let r = &observation.state_residual;
    let mut weighted = vec![0.0; r.len()];
    let mut row_weights = vec![0.0; r.len()];
    let mut gradient = vec![0.0; r.len()];
    let mut column_squared = vec![0.0; r.len()];
    for row in 0..r.len() {
        let range = j.row_ptrs[row]..j.row_ptrs[row + 1];
        let size = norm(&j.values[range.clone()]);
        let weight = if size == 0.0 { 1.0 } else { size.recip() };
        row_weights[row] = weight;
        weighted[row] = weight * r[row];
        for entry in range {
            let column = j.column_indices[entry];
            let value = weight * j.values[entry];
            gradient[column] += value * weighted[row];
            column_squared[column] += value * value;
        }
    }
    for (g, size) in gradient.iter_mut().zip(column_squared) {
        if size > 0.0 {
            *g /= size.sqrt();
        }
    }
    let mut largest = (0..r.len()).collect::<Vec<_>>();
    largest.sort_by(|&a, &b| r[b].abs().total_cmp(&r[a].abs()));
    let rows = largest
        .into_iter()
        .take(12)
        .map(|row| {
            let (equation, node) = if row < 3 * nodes {
                (["momentum_x", "momentum_y", "momentum_z"][row % 3], row / 3)
            } else {
                ("continuity", row - 3 * nodes)
            };
            serde_json::json!({"row":row,"node_index":node,"equation":equation,
            "original_residual":r[row],"row_scaled_residual":weighted[row]})
        })
        .collect::<Vec<_>>();
    let tangent = match std::env::var("P2_PCFAN_CHECK_TANGENT").as_deref() {
        Ok("1") => {
            let matrix = CsrMatrix::try_from_raw_parts(
                j.dimension,
                j.dimension,
                j.row_ptrs.clone(),
                j.column_indices.clone(),
                j.values.clone(),
            )?;
            let rhs = r.iter().map(|v| -v).collect::<Vec<_>>();
            let direction_kind =
                std::env::var("P2_PCFAN_TANGENT_DIRECTION").unwrap_or_else(|_| "newton".into());
            let delta = match direction_kind.as_str() {
                "newton" => {
                    let factors = match std::env::var("SONIUM_PCFAN_LU_BACKEND").as_deref() {
                        Ok("supernodal") => SparseLuFactorization::factorize_supernodal(&matrix)?,
                        Ok("simplicial") | Err(_) => SparseLuFactorization::factorize(&matrix)?,
                        _ => {
                            return Err(
                                "SONIUM_PCFAN_LU_BACKEND must be simplicial or supernodal".into()
                            );
                        }
                    };
                    factors.solve(&rhs)?
                }
                "deterministic" => {
                    // Perturb every unconstrained velocity and pressure field,
                    // independently of the potentially singular Newton solve.
                    // Keep identity-row prescribed values fixed, but include
                    // all rows in each measured derivative residual below.
                    (0..state.len())
                        .map(|i| {
                            let prescribed = matrix.row_range(i).all(|entry| {
                                let value = matrix.values[entry];
                                value == 0.0 || (matrix.col_indices[entry] == i && value == 1.0)
                            }) && matrix.get(i, i) == 1.0;
                            if prescribed {
                                0.0
                            } else {
                                ((i * 17 % 37) as f64 - 18.0) / 18.0 * state[i].abs().max(1e-3)
                            }
                        })
                        .collect()
                }
                _ => {
                    return Err("P2_PCFAN_TANGENT_DIRECTION must be newton or deterministic".into());
                }
            };
            let analytic = matrix.apply(&delta);
            if !norm(&analytic).is_finite() || norm(&analytic) == 0.0 {
                return Err("diagnostic tangent direction has zero or invalid action".into());
            }
            let linear_error = analytic
                .iter()
                .zip(r)
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>();
            let mut samples = Vec::new();
            // Include tiny steps because the failed Armijo search exhausted
            // after updates below1e-8. Central differences do not advance state.
            for epsilon in [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12] {
                let plus = state
                    .iter()
                    .zip(&delta)
                    .map(|(v, d)| v + epsilon * d)
                    .collect::<Vec<_>>();
                let minus = state
                    .iter()
                    .zip(&delta)
                    .map(|(v, d)| v - epsilon * d)
                    .collect::<Vec<_>>();
                let plus = evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &plus)?;
                let minus =
                    evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &minus)?;
                plus.validate()?;
                minus.validate()?;
                let error = plus
                    .state_residual
                    .iter()
                    .zip(&minus.state_residual)
                    .zip(&analytic)
                    .map(|((a, b), expected)| (a - b) / (2.0 * epsilon) - expected)
                    .collect::<Vec<_>>();
                samples.push(serde_json::json!({"epsilon":epsilon,
                    "relative_directional_derivative_error":norm(&error)/norm(&analytic),
                    "momentum_directional_derivative_error_norm":norm(&error[..3*nodes]),
                    "continuity_directional_derivative_error_norm":norm(&error[3*nodes..]),
                    "momentum_analytic_action_norm":norm(&analytic[..3*nodes]),
                    "continuity_analytic_action_norm":norm(&analytic[3*nodes..])}));
            }
            let direction_scan = match std::env::var("P2_PCFAN_SCAN_DIRECTION").as_deref() {
                Ok("1") => {
                    let size = norm(&delta);
                    if !size.is_finite() || size == 0.0 {
                        return Err("invalid direction norm for signed scan".into());
                    }
                    let scale = norm(&state).max(1.0) / size;
                    let direction = delta.iter().map(|d| d * scale).collect::<Vec<_>>();
                    let mut scan = Vec::new();
                    // This is a read-only initial-state investigation. Signed
                    // finite perturbations are not accepted nonlinear steps,
                    // and retain all original residual equations and weights.
                    for magnitude in [0.001, 0.01, 0.1, 1.0, 2.0, 4.0] {
                        for step in [-magnitude, magnitude] {
                            let trial = state
                                .iter()
                                .zip(&direction)
                                .map(|(v, d)| v + step * d)
                                .collect::<Vec<_>>();
                            let actual = evaluate_pc_fan_cfd_newton_state_linearization(
                                &config, &design, &trial,
                            )?;
                            actual.validate()?;
                            if actual.state != trial {
                                return Err("signed scan assembly changed the trial state".into());
                            }
                            let residual = actual.state_residual;
                            let merit = residual
                                .iter()
                                .zip(&row_weights)
                                .map(|(r, w)| r * w)
                                .collect::<Vec<_>>();
                            scan.push(serde_json::json!({"relative_state_step":step,
                                "raw_equation_residual_norm":norm(&residual),
                                "fixed_row_weighted_merit":norm(&merit)}));
                        }
                    }
                    Some(
                        serde_json::json!({"initial_fixed_row_weighted_merit":norm(&weighted),
                        "normalized_direction_norm":norm(&direction),"samples":scan}),
                    )
                }
                Ok("0") | Err(std::env::VarError::NotPresent) => None,
                _ => return Err("P2_PCFAN_SCAN_DIRECTION must be0or1".into()),
            };
            Some(
                serde_json::json!({"state_norm":norm(&state),"direction_kind":direction_kind,
                    "direction_norm":norm(&delta),
                    "newton_direction_norm":(direction_kind=="newton").then(|| norm(&delta)),
                    "actual_linear_relative_residual":(direction_kind=="newton").then(|| norm(&linear_error)/norm(r)),
                    "central_difference_samples":samples,"signed_direction_scan":direction_scan}),
            )
        }
        Ok("0") | Err(_) => None,
        _ => return Err("P2_PCFAN_CHECK_TANGENT must be0 or1".into()),
    };
    let result = serde_json::json!({"scope":"read-only declared checkpoint native constrained equations; no QA admission",
        "viscosity_multiplier":saved["viscosity_multiplier"],"effective_configuration":config,
        "checkpoint_newton_tangent":tangent,
        "checkpoint":path,"accepted_iterations":saved["accepted_iterations"],"state_dofs":state.len(),
        "original_residual_norm":norm(r),"momentum_residual_norm":norm(&r[..3*nodes]),
        "continuity_residual_norm":norm(&r[3*nodes..]),
        "row_scaled_momentum_residual_norm":norm(&weighted[..3*nodes]),
        "row_scaled_continuity_residual_norm":norm(&weighted[3*nodes..]),
        "scaled_normal_equation_gradient_norm":norm(&gradient),"largest_original_residual_rows":rows});
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modified_checkpoint_requires_exact_declared_effective_physics() {
        let original = PcFanConfig::default_120mm();
        let mut effective = original.clone();
        effective.operating_point.dynamic_viscosity_pa_s *= 64.0;
        let mut saved = serde_json::json!({"configuration":original,
            "effective_configuration":effective,"viscosity_multiplier":64.0});
        assert_eq!(
            serde_json::to_value(checkpoint_configuration(&saved).unwrap()).unwrap(),
            saved["effective_configuration"]
        );
        saved["effective_configuration"]["blade_count"] = serde_json::json!(99);
        assert!(checkpoint_configuration(&saved).is_err());
        saved
            .as_object_mut()
            .unwrap()
            .remove("effective_configuration");
        assert!(checkpoint_configuration(&saved).is_err());
        saved["viscosity_multiplier"] = serde_json::json!(1.0);
        assert!(checkpoint_configuration(&saved).is_ok());
    }
}
