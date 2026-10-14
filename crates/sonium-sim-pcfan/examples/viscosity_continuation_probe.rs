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

/// Compute the Cauchy step for a row- and column-scaled residual norm.
fn cauchy_step(
    matrix: &CsrMatrix<f64>,
    residual: &[f64],
    weights: &[f64],
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if residual.len() != matrix.num_rows || weights.len() != matrix.num_rows {
        return Err("Cauchy residual and weights must match the matrix rows".into());
    }
    if residual.iter().any(|r| !r.is_finite())
        || weights.iter().any(|w| !w.is_finite() || *w <= 0.0)
    {
        return Err("Cauchy residual and positive weights must be finite".into());
    }
    let mut column_norm_squared = vec![0.0; matrix.num_cols];
    let mut gradient = vec![0.0; matrix.num_cols];
    for row in 0..matrix.num_rows {
        for index in matrix.row_range(row) {
            let column = matrix.col_indices[index];
            let value = weights[row] * matrix.values[index];
            column_norm_squared[column] += value * value;
            gradient[column] += value * weights[row] * residual[row];
        }
    }
    // Column scaling balances velocity and pressure coordinates without
    // removing equations or changing the physical residual used for stopping.
    let direction = gradient
        .iter()
        .zip(&column_norm_squared)
        .map(|(g, size)| if *size == 0.0 { 0.0 } else { -g / size })
        .collect::<Vec<_>>();
    let action = matrix.apply(&direction);
    let slope = action
        .iter()
        .zip(residual)
        .zip(weights)
        .map(|((a, r), w)| a * w * r * w)
        .sum::<f64>();
    let curvature = action
        .iter()
        .zip(weights)
        .map(|(a, w)| (a * w).powi(2))
        .sum::<f64>();
    if !slope.is_finite() || slope >= 0.0 || !curvature.is_finite() || curvature <= 0.0 {
        return Err("no finite Cauchy descent direction at this state".into());
    }
    let scale = -slope / curvature;
    let delta = direction.iter().map(|d| scale * d).collect::<Vec<_>>();
    if delta.iter().any(|d| !d.is_finite()) {
        return Err("nonfinite scaled Cauchy step".into());
    }
    Ok(delta)
}

/// Compute a regularized least-squares direction without forming normal equations.
fn regularized_step(
    matrix: &CsrMatrix<f64>,
    residual: &[f64],
    weights: &[f64],
    damping: f64,
    iteration_budget: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if residual.len() != matrix.num_rows
        || weights.len() != matrix.num_rows
        || !damping.is_finite()
        || damping <= 0.0
        || !(1..=16384).contains(&iteration_budget)
        || residual.iter().any(|v| !v.is_finite())
        || weights.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || matrix.values.iter().any(|v| !v.is_finite())
    {
        return Err("invalid regularized direction inputs".into());
    }
    let mut column_norm_squared = vec![0.0; matrix.num_cols];
    for row in 0..matrix.num_rows {
        for index in matrix.row_range(row) {
            column_norm_squared[matrix.col_indices[index]] +=
                (weights[row] * matrix.values[index]).powi(2);
        }
    }
    let scaling = column_norm_squared
        .iter()
        .map(|v| if *v == 0.0 { 1.0 } else { v.sqrt().recip() })
        .collect::<Vec<_>>();
    if scaling.iter().any(|v| !v.is_finite()) {
        return Err("invalid regularized column scaling".into());
    }
    // B = W J D. The SPD operator B^T B + damping I only regularizes
    // the search direction; every original equation remains in B and in
    // the independently assembled raw nonlinear stopping criterion.
    let apply_b = |p: &[f64]| {
        let physical = p
            .iter()
            .zip(&scaling)
            .map(|(v, s)| v * s)
            .collect::<Vec<_>>();
        matrix
            .apply(&physical)
            .iter()
            .zip(weights)
            .map(|(v, w)| v * w)
            .collect::<Vec<_>>()
    };
    let apply_bt = |p: &[f64]| {
        let mut result = vec![0.0; matrix.num_cols];
        for row in 0..matrix.num_rows {
            for index in matrix.row_range(row) {
                let column = matrix.col_indices[index];
                result[column] += scaling[column] * matrix.values[index] * weights[row] * p[row];
            }
        }
        result
    };
    let weighted = residual
        .iter()
        .zip(weights)
        .map(|(v, w)| v * w)
        .collect::<Vec<_>>();
    let rhs = apply_bt(&weighted)
        .into_iter()
        .map(|v| -v)
        .collect::<Vec<_>>();
    let rhs_norm = norm(&rhs);
    if !rhs_norm.is_finite() || rhs_norm == 0.0 {
        return Err("no regularized descent gradient at this state".into());
    }
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    let mut p = vec![0.0; matrix.num_cols];
    let mut r = rhs.clone();
    let mut direction = r.clone();
    let mut squared = dot(&r, &r);
    let mut iterations = 0;
    // This is an inexact globalization direction, never an equation solve
    // certificate. Its actual normal residual and descent are checked below.
    for iteration in 0..iteration_budget {
        let mut action = apply_bt(&apply_b(&direction));
        for (value, d) in action.iter_mut().zip(&direction) {
            *value += damping * d;
        }
        let curvature = dot(&direction, &action);
        if !curvature.is_finite() || curvature <= 0.0 {
            return Err("regularized normal operator lost positive curvature".into());
        }
        let alpha = squared / curvature;
        for ((value, residual), (d, a)) in
            p.iter_mut().zip(&mut r).zip(direction.iter().zip(&action))
        {
            *value += alpha * d;
            *residual -= alpha * a;
        }
        iterations = iteration + 1;
        let next = dot(&r, &r);
        if !next.is_finite() {
            return Err("nonfinite regularized normal residual".into());
        }
        if next.sqrt() <= 1e-6 * rhs_norm {
            break;
        }
        let beta = next / squared;
        for (d, residual) in direction.iter_mut().zip(&r) {
            *d = residual + beta * *d;
        }
        squared = next;
    }
    let bp = apply_b(&p);
    let slope = dot(&bp, &weighted);
    let predicted = weighted
        .iter()
        .zip(&bp)
        .map(|(r, a)| r + a)
        .collect::<Vec<_>>();
    let mut actual = apply_bt(&bp);
    for ((a, v), rhs) in actual.iter_mut().zip(&p).zip(&rhs) {
        *a += damping * v - rhs;
    }
    let actual_relative = norm(&actual) / rhs_norm;
    println!(
        "Regularized direction damping={damping:.6e},CG_iterations={iterations},actual_normal_relative_residual={actual_relative:.12e}"
    );
    if !actual_relative.is_finite()
        || actual_relative > 1e-5
        || !slope.is_finite()
        || slope >= 0.0
        || norm(&predicted) >= norm(&weighted)
    {
        return Err("regularized direction failed independent residual or descent check".into());
    }
    let delta = p
        .iter()
        .zip(&scaling)
        .map(|(v, s)| v * s)
        .collect::<Vec<_>>();
    if delta.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite regularized physical direction".into());
    }
    Ok(delta)
}

fn adapt_damping(
    damping: f64,
    current_merit: f64,
    predicted_merit: f64,
    actual_merit: f64,
    step: f64,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let predicted_reduction = current_merit.powi(2) - predicted_merit.powi(2);
    let actual_reduction = current_merit.powi(2) - actual_merit.powi(2);
    if [damping, current_merit, predicted_merit, actual_merit, step]
        .iter()
        .any(|v| !v.is_finite())
        || damping <= 0.0
        || current_merit <= 0.0
        || predicted_merit < 0.0
        || actual_merit < 0.0
        || step <= 0.0
        || step > 1.0
        || predicted_reduction <= 0.0
        || actual_reduction < 0.0
    {
        return Err("invalid regularized model reduction".into());
    }
    let ratio = actual_reduction / predicted_reduction;
    if !ratio.is_finite() {
        return Err("nonfinite regularized reduction ratio".into());
    }
    // Standard trust-region quality bands: relax regularization when a full
    // step accurately predicts merit reduction; strengthen it for a poor
    // prediction or a step that required backtracking. These affect only
    // the next search direction, never residual admission or physical data.
    let next = if ratio < 0.25 || step < 1.0 {
        damping * 10.0
    } else if ratio > 0.75 {
        damping * 0.3
    } else {
        damping
    };
    Ok((next.clamp(1e-8, 1e6), ratio))
}

fn linear_step(
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    backend: &str,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    let factors = match backend {
        "simplicial" => SparseLuFactorization::factorize(matrix)?,
        "supernodal" => SparseLuFactorization::factorize_supernodal(matrix)?,
        _ => return Err("SONIUM_PCFAN_LU_BACKEND must be simplicial or supernodal".into()),
    };
    let factor_seconds = start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let delta = factors.solve(rhs)?;
    println!(
        "Newton LU backend={backend},threads={},factor_seconds={factor_seconds:.6},solve_seconds={:.6}",
        rayon::current_num_threads(),
        start.elapsed().as_secs_f64()
    );
    Ok(delta)
}

fn benchmark_linear_step(
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut baseline: Option<Vec<f64>> = None;
    for (backend, threads) in [("simplicial", 1), ("supernodal", 1), ("supernodal", 16)] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        // Return a Send error across the scoped pool boundary without retaining
        // any global thread setting or changing the assembled physical equations.
        let delta = pool
            .install(|| linear_step(matrix, rhs, backend).map_err(|e| e.to_string()))
            .map_err(std::io::Error::other)?;
        let actual = matrix
            .apply(&delta)
            .iter()
            .zip(rhs)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let relative_residual = norm(&actual) / norm(rhs);
        println!(
            "Linear benchmark backend={backend},threads={threads},actual_relative_residual={relative_residual:.12e}"
        );
        if !relative_residual.is_finite() || relative_residual > 1e-8 {
            return Err("benchmark linear equation did not converge".into());
        }
        if let Some(reference) = &baseline {
            let difference = delta
                .iter()
                .zip(reference)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();
            let relative_difference = norm(&difference) / norm(reference);
            println!("Linear benchmark relative_update_difference={relative_difference:.12e}");
            if !relative_difference.is_finite() || relative_difference > 1e-6 {
                return Err("benchmark Newton updates disagree".into());
            }
        } else {
            baseline = Some(delta);
        }
    }
    println!("Linear backend benchmark only; original nonlinear/conservation/mesh QA unproven");
    Ok(())
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

/// Restore an explicitly requested intermediate continuation state.
fn resumed_state(
    saved: &serde_json::Value,
    original: &PcFanConfig,
    design: &PcFanDesignVariables,
) -> Result<(Vec<f64>, u64, f64), Box<dyn std::error::Error>> {
    let multiplier = saved["viscosity_multiplier"]
        .as_f64()
        .ok_or("resumed viscosity multiplier missing")?;
    if !multiplier.is_finite() || !(1.0..=65536.0).contains(&multiplier) {
        return Err("invalid resumed viscosity multiplier".into());
    }
    let mut effective = original.clone();
    effective.operating_point.dynamic_viscosity_pa_s *= multiplier;
    if saved["configuration"] != serde_json::to_value(original)?
        || saved["design"] != serde_json::to_value(design)?
        || saved["effective_configuration"] != serde_json::to_value(effective)?
    {
        return Err("resumed continuation physics, geometry or design differs".into());
    }
    let state: Vec<f64> = serde_json::from_value(saved["state"].clone())?;
    if state.is_empty() || state.iter().any(|value| !value.is_finite()) {
        return Err("invalid resumed continuation state".into());
    }
    let accepted = saved["accepted_iterations"]
        .as_u64()
        .ok_or("resumed iteration provenance missing")?;
    Ok((state, accepted, multiplier))
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
    let start_multiplier: usize = std::env::args().nth(3).map_or(Ok(64), |v| v.parse())?;
    let continuation_ratio: f64 = std::env::args().nth(4).map_or(Ok(2.0), |v| v.parse())?;
    if !(2..=16).contains(&cells) || budget == 0 {
        return Err("cells in[2,16] and positive stage iteration budget required".into());
    }
    if !start_multiplier.is_power_of_two() || start_multiplier > 65536 {
        return Err("initial viscosity multiplier must be a power of two in[1,65536]".into());
    }
    // A ratio bounded away from one keeps the declared stage ladder finite.
    // Smaller steps change initialization only, never the final equations.
    if !continuation_ratio.is_finite() || !(1.05..=2.0).contains(&continuation_ratio) {
        return Err("viscosity continuation ratio must be finite in[1.05,2]".into());
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
    let initial_path = std::env::var("P2_PCFAN_INITIAL_STATE")?;
    let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(&initial_path)?)?;
    let resume = std::env::var("SONIUM_PCFAN_RESUME_CONTINUATION").as_deref() == Ok("1");
    let (mut state, mut accepted_iterations, start_multiplier) = if resume {
        resumed_state(&saved, &original, &design)?
    } else {
        let (state, accepted) = initial_state(&saved, &original, &design)?;
        (state, accepted, start_multiplier as f64)
    };
    let path = std::path::PathBuf::from(std::env::var("P2_PCFAN_CHECKPOINT")?);
    let full_viscosity = original.operating_point.dynamic_viscosity_pa_s;
    let backend = std::env::var("SONIUM_PCFAN_LU_BACKEND").unwrap_or_else(|_| "simplicial".into());
    if !matches!(backend.as_str(), "simplicial" | "supernodal") {
        return Err("SONIUM_PCFAN_LU_BACKEND must be simplicial or supernodal".into());
    }
    let linear_benchmark = std::env::var("SONIUM_PCFAN_LINEAR_BENCH").as_deref() == Ok("1");
    if linear_benchmark && resume {
        return Err("linear benchmark requires an original-physics checkpoint".into());
    }
    let cauchy_fallback = std::env::var("SONIUM_PCFAN_CAUCHY_FALLBACK").as_deref() == Ok("1");
    let regularized = match std::env::var("SONIUM_PCFAN_REGULARIZED_DIRECTION") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "0" => false,
        Ok(value) if value == "1" => true,
        _ => return Err("SONIUM_PCFAN_REGULARIZED_DIRECTION must be0or1".into()),
    };
    let mut damping: f64 = std::env::var("SONIUM_PCFAN_LM_DAMPING")
        .unwrap_or_else(|_| "0.01".into())
        .parse()?;
    if !damping.is_finite() || !(1e-8..=1e6).contains(&damping) {
        return Err("regularized damping must be finite in[1e-8,1e6]".into());
    }
    let adaptive = match std::env::var("SONIUM_PCFAN_LM_ADAPTIVE") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "0" => false,
        Ok(value) if value == "1" && regularized => true,
        _ => return Err("SONIUM_PCFAN_LM_ADAPTIVE requires0or1and regularized directions".into()),
    };
    let regularized_iteration_budget: usize = match std::env::var("SONIUM_PCFAN_LM_CG_ITERATIONS") {
        Err(std::env::VarError::NotPresent) => 512,
        Ok(value) => value.parse()?,
        Err(error) => return Err(error.into()),
    };
    if !(1..=16384).contains(&regularized_iteration_budget) {
        return Err("regularized CG budget must be in[1,16384]".into());
    }
    // Benchmark the exact original equations; continuation alone uses modified
    // intermediate viscosities and restores the same final physical viscosity.
    let multipliers = if linear_benchmark {
        vec![1.0]
    } else {
        let mut stages = Vec::new();
        let mut multiplier = start_multiplier;
        loop {
            stages.push(multiplier);
            if multiplier == 1.0 {
                break;
            }
            multiplier = (multiplier / continuation_ratio).max(1.0);
        }
        stages
    };
    let mut stages = Vec::new();
    // Geometric continuation gradually restores convection relative to viscosity.
    // Every stage requires the independently assembled raw residual <=1e-8.
    for multiplier in multipliers {
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
                    "continuation_resumed":resume,"initial_checkpoint_path":initial_path,
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
            if linear_benchmark {
                return benchmark_linear_step(&matrix, &rhs);
            }
            let (mut delta, linear_residual) = if regularized {
                (Vec::new(), None)
            } else {
                match linear_step(&matrix, &rhs, &backend) {
                    Ok(delta) => {
                        let actual = matrix
                            .apply(&delta)
                            .iter()
                            .zip(&rhs)
                            .map(|(a, b)| a - b)
                            .collect::<Vec<_>>();
                        let relative = norm(&actual) / norm(&rhs);
                        println!("Independent Newton linear residual {relative:.12e}");
                        (delta, Some(relative))
                    }
                    Err(error) if cauchy_fallback => {
                        println!(
                            "Newton factorization/solve rejected: {error}; trying Cauchy descent"
                        );
                        (Vec::new(), None)
                    }
                    Err(error) => return Err(error),
                }
            };
            let newton_converged = linear_residual.is_some_and(|r| r.is_finite() && r <= 1e-8);
            if !newton_converged && !cauchy_fallback && !regularized {
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
            let mut accepted = false;
            for attempt in 0..=usize::from(cauchy_fallback) {
                if attempt == 0 && !newton_converged && !regularized {
                    println!(
                        "Newton direction rejected by independent residual; trying Cauchy descent"
                    );
                    continue;
                }
                let direction_name = if attempt == 0 {
                    if regularized { "Regularized" } else { "Newton" }
                } else {
                    "Cauchy"
                };
                if attempt == 0 && regularized {
                    let mut retries = 0;
                    loop {
                        match regularized_step(
                            &matrix,
                            &observation.state_residual,
                            &weights,
                            damping,
                            regularized_iteration_budget,
                        ) {
                            Ok(direction) => {
                                delta = direction;
                                break;
                            }
                            Err(error) if adaptive && retries < 6 && damping < 1e6 => {
                                damping = (damping * 10.0).min(1e6);
                                retries += 1;
                                println!(
                                    "Regularized direction rejected: {error}; retry damping={damping:.6e}"
                                );
                            }
                            Err(error) => return Err(error),
                        }
                    }
                }
                if attempt != 0 {
                    delta = cauchy_step(&matrix, &observation.state_residual, &weights)?;
                }
                let merit_slope = if attempt == 0 && !regularized {
                    -current_merit
                } else {
                    matrix
                        .apply(&delta)
                        .iter()
                        .zip(&observation.state_residual)
                        .zip(&weights)
                        .map(|((a, r), w)| a * w * r * w)
                        .sum::<f64>()
                        / current_merit
                };
                if !merit_slope.is_finite() || merit_slope >= 0.0 {
                    return Err("line-search direction does not decrease the full merit".into());
                }
                let mut step = 1.0;
                let used_damping = damping;
                let model_action = if adaptive && attempt == 0 {
                    Some(matrix.apply(&delta))
                } else {
                    None
                };
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
                    // Retain the original Newton acceptance calculation exactly.
                    let armijo_bound = if attempt == 0 && !regularized {
                        (1.0 - 1.0e-4 * step) * current_merit
                    } else {
                        current_merit + 1.0e-4 * step * merit_slope
                    };
                    if trial_residual.is_finite()
                        && trial_merit.is_finite()
                        && trial_merit <= armijo_bound
                    {
                        println!(
                            "accepted {direction_name} step {step:.6e}; trial residual {trial_residual:.6e}"
                        );
                        let mut reduction_ratio = None;
                        if let Some(action) = &model_action {
                            let prediction = observation
                                .state_residual
                                .iter()
                                .zip(action)
                                .map(|(r, a)| r + step * a)
                                .collect::<Vec<_>>();
                            let (next, ratio) = adapt_damping(
                                damping,
                                current_merit,
                                merit(&prediction),
                                trial_merit,
                                step,
                            )?;
                            println!(
                                "Regularized model reduction ratio={ratio:.6e},next_damping={next:.6e}"
                            );
                            damping = next;
                            reduction_ratio = Some(ratio);
                        }
                        state = trial;
                        accepted_iterations += 1;
                        checkpoint(
                            &path,
                            &serde_json::json!({"configuration":original,
                        "effective_configuration":config,"design":design,"state":state,
                        "accepted_iterations":accepted_iterations,"viscosity_multiplier":multiplier,
                        "actual_stage_residual":trial_residual,"converged_stages":stages,
                        "independent_linear_equation_relative_residual":linear_residual,
                        "newton_linear_solve_converged":newton_converged,
                        "continuation_resumed":resume,"initial_checkpoint_path":initial_path,
                        "accepted_direction":direction_name,"merit_directional_derivative":merit_slope,
                        "regularized_direction_enabled":regularized,"regularized_direction_damping":used_damping,
                        "regularized_adaptive":adaptive,"regularized_next_damping":damping,
                        "regularized_cg_iteration_budget":regularized_iteration_budget,
                        "regularized_model_reduction_ratio":reduction_ratio,
                        "diagnostic":"modified intermediate viscosity; original QA unproven"}),
                        )?;
                        accepted = true;
                        break;
                    }
                    step *= 0.5;
                }
                if accepted {
                    break;
                }
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
        "continuation_resumed":resume,"initial_checkpoint_path":initial_path,
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
    fn adaptive_regularization_uses_independently_measured_nonlinear_reduction() {
        // r(x)=x^2-1 at x=2: r=3, J=4, delta=-.5. The linear
        // prediction is1, but the actual residual at1.5 is1.25.
        let (next, ratio) = adapt_damping(0.01, 3.0, 1.0, 1.25, 1.0).unwrap();
        assert!((ratio - 0.9296875).abs() < 1e-14);
        assert!((next - 0.003).abs() < 1e-14);
        let (next, ratio) = adapt_damping(0.01, 3.0, 1.0, 2.9, 1.0).unwrap();
        assert!(ratio < 0.25);
        assert_eq!(next, 0.1);
        // Backtracking is evidence that the initial regularized step was too
        // ambitious, even when the smaller step has an accurate prediction.
        assert_eq!(adapt_damping(0.01, 3.0, 2.0, 2.0, 0.5).unwrap().0, 0.1);
        assert_eq!(adapt_damping(1e-8, 3.0, 1.0, 1.0, 1.0).unwrap().0, 1e-8);
        assert_eq!(adapt_damping(1e6, 3.0, 1.0, 2.9, 1.0).unwrap().0, 1e6);
    }

    #[test]
    fn adaptive_regularization_rejects_false_model_reduction() {
        assert!(adapt_damping(0.01, 3.0, 3.0, 2.0, 1.0).is_err());
        assert!(adapt_damping(0.01, 3.0, 1.0, 3.1, 1.0).is_err());
        assert!(adapt_damping(0.01, 3.0, 1.0, 1.0, 0.0).is_err());
        assert!(adapt_damping(0.01, 3.0, f64::NAN, 1.0, 1.0).is_err());
        assert!(adapt_damping(0.0, 3.0, 1.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn cauchy_descent_retains_inconsistent_equations_when_the_jacobian_is_singular() {
        let matrix = CsrMatrix::try_from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 1.0, 2.0, 2.0],
        )
        .unwrap();
        let residual = [1.0, 0.0];
        let direction = cauchy_step(&matrix, &residual, &[1.0, 1.0]).unwrap();
        let action = matrix.apply(&direction);
        let after = residual
            .iter()
            .zip(action)
            .map(|(r, a)| r + a)
            .collect::<Vec<_>>();
        assert!(norm(&after) < norm(&residual));
        // The equations are inconsistent. Descent must not claim a root.
        assert!(norm(&after) > 0.1);
    }

    #[test]
    fn resumed_checkpoint_requires_exact_intermediate_physics_and_provenance() {
        let original = PcFanConfig::default_120mm();
        let design = original.default_design_variables();
        let mut effective = original.clone();
        effective.operating_point.dynamic_viscosity_pa_s *= 92.25;
        let saved = serde_json::json!({"configuration":original,"design":design,
            "effective_configuration":effective,"viscosity_multiplier":92.25,
            "state":[1.0,2.0,3.0],"accepted_iterations":58});
        let result = resumed_state(&saved, &original, &design).unwrap();
        assert_eq!((result.1, result.2), (58, 92.25));
        assert!(initial_state(&saved, &original, &design).is_err());
        let mut changed = saved.clone();
        changed["effective_configuration"]["operating_point"]["dynamic_viscosity_pa_s"] =
            1.0.into();
        assert!(resumed_state(&changed, &original, &design).is_err());
        changed = saved.clone();
        changed["accepted_iterations"] = serde_json::Value::Null;
        assert!(resumed_state(&changed, &original, &design).is_err());
        changed = saved;
        changed["configuration"]["mesh"]["radial_segments"] = 999.into();
        assert!(resumed_state(&changed, &original, &design).is_err());
    }

    #[test]
    fn scaled_cauchy_step_decreases_an_independent_linear_merit() {
        let matrix = CsrMatrix::try_from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 3.0, -1.0],
        )
        .unwrap();
        let residual = [4.0, -2.0];
        let weights = [0.5, 2.0];
        let delta = cauchy_step(&matrix, &residual, &weights).unwrap();
        let action = matrix.apply(&delta);
        let before = (0..2)
            .map(|i| (weights[i] * residual[i]).powi(2))
            .sum::<f64>();
        let after = (0..2)
            .map(|i| (weights[i] * (residual[i] + action[i])).powi(2))
            .sum::<f64>();
        assert!(after < before);
        assert!(cauchy_step(&matrix, &[0.0, 0.0], &weights).is_err());
        assert!(cauchy_step(&matrix, &residual, &[1.0, 0.0]).is_err());
        assert!(cauchy_step(&matrix, &[f64::NAN, 1.0], &weights).is_err());
    }

    #[test]
    fn regularized_direction_matches_independent_two_by_two_normal_solution() {
        let matrix = CsrMatrix::try_from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 3.0, -1.0],
        )
        .unwrap();
        let residual = [4.0, -2.0];
        let weights = [0.5, 2.0];
        let s0 = 36.25_f64.sqrt();
        let s1 = 5.0_f64.sqrt();
        let cross = -11.5 / (s0 * s1);
        let g0 = -23.0 / s0;
        let g1 = 10.0 / s1;
        // One iteration cannot solve this non-diagonal two-dimensional system.
        // A larger budget must still satisfy the independently measured normal
        // residual; it changes computational effort, never acceptance accuracy.
        assert!(regularized_step(&matrix, &residual, &weights, 0.01, 1).is_err());
        assert!(regularized_step(&matrix, &residual, &weights, 0.01, 2).is_ok());
        for budget in [0, 16385] {
            assert!(regularized_step(&matrix, &residual, &weights, 0.01, budget).is_err());
        }
        for damping in [1e-4, 0.01, 1.0, 100.0] {
            let diagonal = 1.0 + damping;
            let determinant = diagonal * diagonal - cross * cross;
            let expected = [
                (-diagonal * g0 + cross * g1) / determinant / s0,
                (cross * g0 - diagonal * g1) / determinant / s1,
            ];
            let actual = regularized_step(&matrix, &residual, &weights, damping, 512).unwrap();
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| (a - b).abs() < 1e-11)
            );
        }
        for damping in [0.0, -1.0, f64::NAN] {
            assert!(regularized_step(&matrix, &residual, &weights, damping, 512).is_err());
        }
        assert!(regularized_step(&matrix, &[0.0, 0.0], &weights, 0.01, 512).is_err());
        assert!(regularized_step(&matrix, &residual, &[0.0, 1.0], 0.01, 512).is_err());
    }

    #[test]
    fn regularization_cannot_discard_an_inconsistent_equation() {
        let matrix =
            CsrMatrix::try_from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 0], vec![1.0, 1.0]).unwrap();
        let residual = [2.0, -1.0];
        let delta = regularized_step(&matrix, &residual, &[1.0, 1.0], 0.01, 512).unwrap();
        assert_eq!(delta[1], 0.0);
        let remaining = matrix
            .apply(&delta)
            .iter()
            .zip(residual)
            .map(|(a, r)| a + r)
            .collect::<Vec<_>>();
        assert!(norm(&remaining) < norm(&residual));
        assert!(norm(&remaining) > 2.0);
    }

    #[test]
    fn cauchy_step_preserves_the_physical_direction_under_variable_unit_changes() {
        let matrix = CsrMatrix::try_from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 3.0, -1.0],
        )
        .unwrap();
        // The second unknown is expressed in a unit a million times smaller.
        let transformed = CsrMatrix::try_from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2e6, 3.0, -1e6],
        )
        .unwrap();
        let residual = [4.0, -2.0];
        let weights = [0.5, 2.0];
        let original = cauchy_step(&matrix, &residual, &weights).unwrap();
        let changed = cauchy_step(&transformed, &residual, &weights).unwrap();
        assert!((original[0] - changed[0]).abs() < 1e-12);
        assert!((original[1] - 1e6 * changed[1]).abs() < 1e-12);
        for damping in [0.01, 1.0] {
            let original = regularized_step(&matrix, &residual, &weights, damping, 512).unwrap();
            let changed =
                regularized_step(&transformed, &residual, &weights, damping, 512).unwrap();
            assert!((original[0] - changed[0]).abs() < 1e-12);
            assert!((original[1] - 1e6 * changed[1]).abs() < 1e-12);
        }
    }

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
