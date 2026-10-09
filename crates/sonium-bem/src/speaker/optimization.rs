//! Loudspeaker parameter optimisation.
//!
//! Optimises driver positions and waveguide shape for flat on-axis
//! response and smooth radiation pattern (flat PIR with ≤2 dB/oct slope).
//!
//! Uses the augmented Lagrangian method from `sonium-solvers` with
//! L-BFGS-B inner solver and central finite-difference gradients.

use sonium_solvers::optimization::{
    augmented_lagrangian, AugmentedLagrangianConfig, Bound, Constraint, ConstraintKind,
    InnerSolver,
};
use sonium_solvers::optimization::util::finite_diff::gradient_central;

use super::cea2034_output::{self, ANGLES_PER_PLANE};
use super::model::LoudspeakerModel;
use super::simulation::{self, SimulationResult, SolverConfig};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of the optimisation.
pub struct OptimizationResult {
    /// Optimised model.
    pub model: LoudspeakerModel,
    /// Final objective value.
    pub objective: f64,
    /// Number of outer iterations.
    pub iterations: usize,
    /// Whether the optimiser converged.
    pub converged: bool,
}

/// Configuration for the optimisation.
pub struct OptimizationConfig {
    /// Maximum outer iterations for augmented Lagrangian.
    pub max_iterations: usize,
    /// Weight for the flat on-axis response term.
    pub weight_flat: f64,
    /// Weight for the smooth PIR term.
    pub weight_pir: f64,
    /// Number of frequency points used during optimisation (coarser = faster).
    pub optim_freq_points: usize,
    /// Maximum mesh frequency during optimisation (Hz, coarser mesh = faster).
    pub optim_max_mesh_freq: f64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            weight_flat: 1.0,
            weight_pir: 1.0,
            optim_freq_points: 50,
            optim_max_mesh_freq: 5000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Parameter packing
// ---------------------------------------------------------------------------

/// Pack optimisable parameters from a model into a flat vector.
///
/// Layout: [woofer_y, tweeter_y, wg_h_spline..., wg_v_spline..., wg_depth]
fn pack_parameters(model: &LoudspeakerModel) -> Vec<f64> {
    let mut params = Vec::new();
    params.push(model.woofer.y_offset);
    params.push(model.tweeter.y_offset);
    for pt in &model.waveguide.horizontal_spline {
        params.push(pt[1]); // radial values only (t is fixed)
    }
    for pt in &model.waveguide.vertical_spline {
        params.push(pt[1]);
    }
    params.push(model.waveguide.depth);
    params
}

/// Unpack a flat parameter vector back into a model (cloning the base model).
fn unpack_parameters(x: &[f64], base: &LoudspeakerModel) -> LoudspeakerModel {
    let mut model = base.clone();
    let mut idx = 0;

    model.woofer.y_offset = x[idx];
    idx += 1;
    model.tweeter.y_offset = x[idx];
    idx += 1;

    for pt in &mut model.waveguide.horizontal_spline {
        pt[1] = x[idx];
        idx += 1;
    }
    for pt in &mut model.waveguide.vertical_spline {
        pt[1] = x[idx];
        idx += 1;
    }
    model.waveguide.depth = x[idx];
    #[allow(unused_assignments)]
    { idx += 1; }

    model
}

/// Compute box constraints for the parameter vector.
fn compute_bounds(model: &LoudspeakerModel) -> Vec<Bound<f64>> {
    let h2 = model.cabinet.height / 2.0;
    let w2 = model.cabinet.width / 2.0;
    let sep = model.separator.y_position;
    let sep_half = model.separator.thickness / 2.0;

    let mut bounds = Vec::new();

    // Woofer y: bottom of cabinet to below separator
    bounds.push(Bound::both(
        -h2 + model.woofer.radius,
        sep - sep_half - model.woofer.radius,
    ));
    // Tweeter y: above separator to top of cabinet
    bounds.push(Bound::both(
        sep + sep_half + model.tweeter.radius,
        h2 - model.tweeter.radius,
    ));

    // Waveguide H spline radii
    for _ in &model.waveguide.horizontal_spline {
        bounds.push(Bound::both(model.tweeter.radius, w2 * 0.8));
    }
    // Waveguide V spline radii
    for _ in &model.waveguide.vertical_spline {
        bounds.push(Bound::both(model.tweeter.radius, h2 * 0.5));
    }
    // Waveguide depth
    bounds.push(Bound::both(0.005, model.cabinet.depth / 3.0));

    bounds
}

// ---------------------------------------------------------------------------
// Objective and constraints
// ---------------------------------------------------------------------------

/// Compute the combined objective from a simulation result.
fn compute_objective(
    result: &SimulationResult,
    config: &OptimizationConfig,
) -> f64 {
    let n_freqs = result.frequencies.len();

    // On-axis SPL (horizontal 0° = index 0)
    let on_axis: Vec<f64> = (0..n_freqs).map(|fi| result.spl[fi][0]).collect();

    // Find F3: the low-frequency -3dB point.
    // Scan from high to low and find the last frequency still within 3dB of peak.
    // Everything from that point upward is "above F3".
    let peak_spl = on_axis.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let threshold = peak_spl - 3.0;
    let f3_idx = on_axis
        .iter()
        .enumerate()
        .rev()
        .find(|(_, s)| **s < threshold)
        .map(|(i, _)| i + 1)
        .unwrap_or(0);

    // L_flat: MSE of on-axis above F3 relative to mean
    let above_f3: Vec<f64> = on_axis[f3_idx..].to_vec();
    let mean_above = if above_f3.is_empty() {
        0.0
    } else {
        above_f3.iter().sum::<f64>() / above_f3.len() as f64
    };
    let l_flat: f64 = above_f3
        .iter()
        .map(|&s| (s - mean_above).powi(2))
        .sum::<f64>()
        / above_f3.len().max(1) as f64;

    // L_pir: deviation from flat + slope penalty
    // Use simplified PIR: RMS of all horizontal angles
    let pir: Vec<f64> = (0..n_freqs)
        .map(|fi| {
            let sum_p2: f64 = (0..ANGLES_PER_PLANE)
                .map(|ai| {
                    let p = 10f64.powf((result.spl[fi][ai] - 105.0) / 20.0);
                    p * p
                })
                .sum();
            let rms_p = (sum_p2 / ANGLES_PER_PLANE as f64).sqrt();
            20.0 * rms_p.log10() + 105.0
        })
        .collect();

    let pir_mean = pir.iter().sum::<f64>() / pir.len().max(1) as f64;
    let l_pir_flat: f64 = pir.iter().map(|&s| (s - pir_mean).powi(2)).sum::<f64>()
        / pir.len().max(1) as f64;

    // Slope penalty: max dB change per octave
    let l_slope = compute_max_octave_slope(&result.frequencies, &pir);
    let slope_penalty = (l_slope - 2.0).max(0.0).powi(2);

    config.weight_flat * l_flat + config.weight_pir * (l_pir_flat + 10.0 * slope_penalty)
}

/// Maximum absolute SPL change per octave across the frequency range.
fn compute_max_octave_slope(freqs: &[f64], spl: &[f64]) -> f64 {
    if freqs.len() < 2 {
        return 0.0;
    }
    let mut max_slope: f64 = 0.0;
    for i in 1..freqs.len() {
        let octaves = (freqs[i] / freqs[i - 1]).log2();
        if octaves > 0.01 {
            let slope = (spl[i] - spl[i - 1]).abs() / octaves;
            if slope > max_slope {
                max_slope = slope;
            }
        }
    }
    max_slope
}

/// Compute the PIR slope constraint value: `2.0 - max_slope >= 0`.
fn pir_slope_constraint(result: &SimulationResult) -> f64 {
    let n_freqs = result.frequencies.len();
    let pir: Vec<f64> = (0..n_freqs)
        .map(|fi| {
            let sum_p2: f64 = (0..ANGLES_PER_PLANE)
                .map(|ai| {
                    let p = 10f64.powf((result.spl[fi][ai] - 105.0) / 20.0);
                    p * p
                })
                .sum();
            let rms_p = (sum_p2 / ANGLES_PER_PLANE as f64).sqrt();
            20.0 * rms_p.log10() + 105.0
        })
        .collect();

    let max_slope = compute_max_octave_slope(&result.frequencies, &pir);
    2.0 - max_slope
}

// ---------------------------------------------------------------------------
// Simulation cache
// ---------------------------------------------------------------------------

use std::cell::RefCell;

struct SimCache {
    last_x: Option<Vec<f64>>,
    last_result: Option<SimulationResult>,
}

impl SimCache {
    fn new() -> Self {
        Self {
            last_x: None,
            last_result: None,
        }
    }

    fn get_or_compute(
        &mut self,
        x: &[f64],
        base_model: &LoudspeakerModel,
        eval_points: &[[f64; 3]],
        solver_config: &SolverConfig,
    ) {
        let x_vec: Vec<f64> = x.to_vec();
        if self.last_x.as_ref() == Some(&x_vec) && self.last_result.is_some() {
            return;
        }

        let model = unpack_parameters(x, base_model);
        let result = simulation::run_simulation_with_config(&model, eval_points, solver_config);
        self.last_x = Some(x_vec);
        self.last_result = Some(result);
    }
}

// ---------------------------------------------------------------------------
// Main optimisation entry point
// ---------------------------------------------------------------------------

/// Run loudspeaker optimisation.
pub fn optimize(
    model: &LoudspeakerModel,
    config: &OptimizationConfig,
) -> OptimizationResult {
    // Build a model variant with coarser settings for optimisation
    let mut optim_model = model.clone();
    optim_model.simulation.freq_points = config.optim_freq_points;
    optim_model.simulation.max_mesh_frequency = Some(config.optim_max_mesh_freq);

    let eval_points = cea2034_output::generate_eval_points(optim_model.baffle_center());
    let solver_config = SolverConfig {
        tolerance: 1e-4, // relaxed for speed
        max_iterations: 100,
        restart: 20,
        ..Default::default()
    };

    let x0 = pack_parameters(&optim_model);
    let bounds = compute_bounds(&optim_model);

    // Thread-local simulation cache
    let cache = RefCell::new(SimCache::new());

    let objective = |x: &[f64]| -> f64 {
        cache
            .borrow_mut()
            .get_or_compute(x, &optim_model, &eval_points, &solver_config);
        let borrow = cache.borrow();
        compute_objective(borrow.last_result.as_ref().unwrap(), config)
    };

    let grad_fn = |x: &[f64]| -> Vec<f64> {
        let h = f64::EPSILON.powf(1.0 / 3.0);
        gradient_central(&objective, x, h)
    };

    // PIR slope constraint: c(x) >= 0
    let constraint = Constraint {
        kind: ConstraintKind::Inequality,
        dim: 1,
        fun: Box::new({
            let optim_model = optim_model.clone();
            let eval_points = eval_points.clone();
            let solver_config_clone = SolverConfig {
                tolerance: 1e-4,
                max_iterations: 100,
                restart: 20,
                ..Default::default()
            };
            move |x: &[f64]| {
                let model = unpack_parameters(x, &optim_model);
                let result = simulation::run_simulation_with_config(
                    &model,
                    &eval_points,
                    &solver_config_clone,
                );
                vec![pir_slope_constraint(&result)]
            }
        }),
        jac: None,
    };

    let al_config = AugmentedLagrangianConfig {
        max_outer_iterations: config.max_iterations,
        max_inner_iterations: 10, // few inner iterations (expensive objective)
        bounds,
        inner_solver: InnerSolver::Lbfgsb,
        ..Default::default()
    };

    log::info!(
        "Starting optimisation with {} parameters",
        x0.len()
    );

    let result = augmented_lagrangian(&objective, &grad_fn, &x0, &[constraint], &al_config);

    match result {
        Ok(opt) => {
            let final_model = unpack_parameters(&opt.x, model);
            OptimizationResult {
                model: final_model,
                objective: opt.fun,
                iterations: opt.iterations,
                converged: opt.converged,
            }
        }
        Err(e) => {
            log::warn!("Optimisation failed: {e}");
            // Return the original model as fallback
            OptimizationResult {
                model: model.clone(),
                objective: f64::INFINITY,
                iterations: 0,
                converged: false,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::model::*;

    fn sample_model() -> LoudspeakerModel {
        LoudspeakerModel {
            cabinet: CabinetDimensions {
                height: 0.40,
                width: 0.22,
                depth: 0.28,
            },
            woofer: DriverConfig {
                radius: 0.065,
                y_offset: -0.08,
                piston_velocity: 0.001,
                transfer_function: None,
            },
            tweeter: DriverConfig {
                radius: 0.013,
                y_offset: 0.10,
                piston_velocity: 0.001,
                transfer_function: None,
            },
            separator: SeparatorConfig {
                y_position: 0.02,
                thickness: 0.018,
            },
            waveguide: WaveguideConfig {
                horizontal_spline: vec![[0.0, 0.05], [0.5, 0.03], [1.0, 0.013]],
                vertical_spline: vec![[0.0, 0.04], [0.5, 0.025], [1.0, 0.013]],
                depth: 0.025,
            },
            simulation: SimulationConfig {
                freq_start: 1000.0,
                freq_end: 5000.0,
                freq_points: 10,
                speed_of_sound: 343.0,
                air_density: 1.21,
                max_mesh_frequency: Some(5000.0),
            },
        }
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let model = sample_model();
        let x = pack_parameters(&model);
        let restored = unpack_parameters(&x, &model);

        assert!((restored.woofer.y_offset - model.woofer.y_offset).abs() < 1e-15);
        assert!((restored.tweeter.y_offset - model.tweeter.y_offset).abs() < 1e-15);
        assert!((restored.waveguide.depth - model.waveguide.depth).abs() < 1e-15);
        for (a, b) in restored
            .waveguide
            .horizontal_spline
            .iter()
            .zip(model.waveguide.horizontal_spline.iter())
        {
            assert!((a[1] - b[1]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_bounds_count() {
        let model = sample_model();
        let x = pack_parameters(&model);
        let bounds = compute_bounds(&model);
        assert_eq!(bounds.len(), x.len());
    }

    #[test]
    fn test_initial_point_within_bounds() {
        let model = sample_model();
        let x = pack_parameters(&model);
        let bounds = compute_bounds(&model);
        for (i, (xi, b)) in x.iter().zip(bounds.iter()).enumerate() {
            if let Some(lb) = b.lower {
                assert!(
                    *xi >= lb - 1e-10,
                    "param {i}: {xi} < lower bound {lb}"
                );
            }
            if let Some(ub) = b.upper {
                assert!(
                    *xi <= ub + 1e-10,
                    "param {i}: {xi} > upper bound {ub}"
                );
            }
        }
    }

    #[test]
    fn test_max_octave_slope() {
        // Flat response → 0 dB/oct
        let freqs = vec![100.0, 200.0, 400.0, 800.0];
        let spl = vec![80.0, 80.0, 80.0, 80.0];
        assert!(compute_max_octave_slope(&freqs, &spl) < 0.01);

        // 3 dB/oct tilt
        let spl_tilt = vec![80.0, 83.0, 86.0, 89.0];
        let slope = compute_max_octave_slope(&freqs, &spl_tilt);
        assert!((slope - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_f3_idx_skips_rolloff() {
        // Simulate a response that rolls off at low freq then is flat.
        // F3 should be at the transition, not at index 0.
        let _freqs = vec![100.0, 200.0, 400.0, 800.0, 1600.0];
        let on_axis = vec![70.0, 76.0, 82.0, 85.0, 85.0]; // rolloff below 400 Hz

        let peak_spl = 85.0;
        let threshold = peak_spl - 3.0; // 82.0
        let f3_idx = on_axis
            .iter()
            .enumerate()
            .rev()
            .find(|(_, s)| **s < threshold)
            .map(|(i, _)| i + 1)
            .unwrap_or(0);

        // Should skip the first two points (70, 76) that are below threshold
        assert_eq!(f3_idx, 2, "F3 index should be 2 (400 Hz), got {f3_idx}");
    }
}
