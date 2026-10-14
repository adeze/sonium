//! Continue a saved original fan state through higher-viscosity equations.
//!
//! Intermediate stages change viscosity explicitly and provide no original QA
//! admission. Geometry, forces, boundary conditions and design stay unchanged.
//! Every stage must converge before the exact original viscosity is reached.

// Rust guideline compliant 2026-02-21

use sonium_sim_pcfan::{
    PcFanCfdBackend, PcFanCfdConfig, PcFanConfig, PcFanDesignVariables, PcFanFlowModel,
    PcFanMeshResolution, evaluate_pc_fan_cfd_newton_state_linearization,
};
use sonium_solvers::{CsrMatrix, direct::SparseLuFactorization, traits::LinearOperator};

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn initial_state(
    saved: &serde_json::Value,
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
) -> Result<(Vec<f64>, u64), Box<dyn std::error::Error>> {
    if saved["configuration"] != serde_json::to_value(config)?
        || saved["design"] != serde_json::to_value(design)?
        || saved["viscosity_multiplier"].as_f64().unwrap_or(1.0) != 1.0
    {
        return Err("checkpoint differs from the original physical fan fixture".into());
    }
    let state: Vec<f64> = serde_json::from_value(saved["state"].clone())?;
    if state.is_empty() || state.iter().any(|v| !v.is_finite()) {
        return Err("invalid checkpoint state".into());
    }
    let accepted = saved["accepted_iterations"]
        .as_u64()
        .ok_or("checkpoint provenance missing")?;
    Ok((state, accepted))
}

fn checkpoint(
    path: &std::path::Path,
    value: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let temporary = path.with_extension("pending.json");
    std::fs::write(&temporary, serde_json::to_vec(value)?)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cells = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "8".into())
        .parse::<usize>()?;
    let budget: usize = std::env::args().nth(2).map_or(Ok(100), |v| v.parse())?;
    if !(2..=16).contains(&cells) || budget == 0 {
        return Err("cells in[2,16] and positive stage iteration budget required".into());
    }
    let original = PcFanConfig {
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
    let design = original.default_design_variables();
    let saved: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::var("P2_PCFAN_INITIAL_STATE")?)?)?;
    let (mut state, mut accepted_iterations) = initial_state(&saved, &original, &design)?;
    let path = std::path::PathBuf::from(std::env::var("P2_PCFAN_CHECKPOINT")?);
    let full_viscosity = original.operating_point.dynamic_viscosity_pa_s;
    let mut stages = Vec::new();
    // Geometric continuation gradually restores convection relative to viscosity.
    // Every stage requires the independently assembled raw residual <=1e-8.
    for multiplier in [64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0] {
        let mut config = original.clone();
        config.operating_point.dynamic_viscosity_pa_s = full_viscosity * multiplier;
        println!(
            "Viscosity continuation multiplier {multiplier}; viscosity {}",
            config.operating_point.dynamic_viscosity_pa_s
        );
        let mut converged = false;
        for iteration in 0..=budget {
            let observation =
                evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &state)?;
            observation.validate()?;
            if observation.state != state {
                return Err("native assembly changed the supplied stage state".into());
            }
            let residual = norm(&observation.state_residual);
            println!("Newton iteration {iteration}: residual {residual:.6e}");
            if residual <= 1.0e-8 {
                stages.push(serde_json::json!({"viscosity_multiplier":multiplier,
                    "actual_stage_residual":residual,"iterations":iteration}));
                checkpoint(
                    &path,
                    &serde_json::json!({"configuration":original,
                    "effective_configuration":config,"design":design,"state":state,
                    "accepted_iterations":accepted_iterations,"viscosity_multiplier":multiplier,
                    "actual_stage_residual":residual,"converged_stages":stages,
                    "diagnostic":"stage convergence only; original conservation/mesh QA unproven"}),
                )?;
                converged = true;
                break;
            }
            if !residual.is_finite() || iteration == budget {
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
            let actual = matrix
                .apply(&delta)
                .iter()
                .zip(&rhs)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();
            let linear_residual = norm(&actual) / norm(&rhs);
            println!("Independent Newton linear residual {linear_residual:.12e}");
            if !linear_residual.is_finite() || linear_residual > 1e-8 {
                return Err("stage Newton linear equation did not converge".into());
            }
            let weights = (0..matrix.num_rows)
                .map(|row| {
                    let size = norm(&matrix.values[matrix.row_range(row)]);
                    if size == 0.0 { 1.0 } else { size.recip() }
                })
                .collect::<Vec<_>>();
            if weights.iter().any(|w| !w.is_finite() || *w <= 0.0) {
                return Err("invalid positive equation scaling".into());
            }
            // Fixed positive weights retain all equations and affect only Armijo
            // acceptance, never the raw stage/final stopping criterion.
            let merit = |r: &[f64]| {
                norm(
                    &r.iter()
                        .zip(&weights)
                        .map(|(r, w)| r * w)
                        .collect::<Vec<_>>(),
                )
            };
            let current_merit = merit(&observation.state_residual);
            let mut step = 1.0;
            let mut accepted = false;
            for _ in 0..32 {
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
                    accepted_iterations += 1;
                    checkpoint(
                        &path,
                        &serde_json::json!({"configuration":original,
                        "effective_configuration":config,"design":design,"state":state,
                        "accepted_iterations":accepted_iterations,"viscosity_multiplier":multiplier,
                        "actual_stage_residual":trial_residual,"converged_stages":stages,
                        "independent_linear_equation_relative_residual":linear_residual,
                        "diagnostic":"modified intermediate viscosity; original QA unproven"}),
                    )?;
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
    let final_observation =
        evaluate_pc_fan_cfd_newton_state_linearization(&original, &design, &state)?;
    let residual = norm(&final_observation.state_residual);
    if !residual.is_finite() || residual > 1e-8 {
        return Err("final exact original equations did not converge".into());
    }
    checkpoint(
        &path,
        &serde_json::json!({"configuration":original,"design":design,"state":state,
        "accepted_iterations":accepted_iterations,"viscosity_multiplier":1.0,
        "true_equation_residual":residual,"converged_stages":stages,
        "diagnostic":"exact original equations only; conservation/mesh QA unproven"}),
    )?;
    println!("Exact original equations residual={residual:.12e}; conservation/mesh QA unproven");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_rejects_changed_geometry_design_and_intermediate_physics() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let saved = serde_json::json!({"configuration":config,"design":design,
            "state":[0.0,1.0,2.0,3.0],"accepted_iterations":469});
        assert_eq!(initial_state(&saved, &config, &design).unwrap().1, 469);
        let mut changed = saved.clone();
        changed["configuration"]["mesh"]["radial_segments"] = 123.into();
        assert!(initial_state(&changed, &config, &design).is_err());
        changed = saved.clone();
        changed["design"] = serde_json::json!({});
        assert!(initial_state(&changed, &config, &design).is_err());
        changed = saved;
        changed["viscosity_multiplier"] = 2.0.into();
        assert!(initial_state(&changed, &config, &design).is_err());
    }
}
