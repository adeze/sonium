use super::adaptive_state::AdaptiveState;
pub(crate) use super::argmin::argmin;
use super::deconfig::DEConfig;
use super::dereport::DEReport;
pub use super::error::{DEError, Result};
use super::strategy::Strategy;
use super::types::CallbackAction;
use super::types::Crossover;
use super::types::DEIntermediate;
use super::types::Init;
use super::types::PenaltyTuple;
use rand::rngs::{SmallRng, StdRng};
use rand::{Rng, SeedableRng};
use std::sync::RwLock;
use std::time::Instant;

/// Mean of a slice, or `0.0` when empty (mirrors `ndarray`'s checked mean).
fn vec_mean_or_zero(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Population standard deviation (ddof = 0), mirroring `ndarray`'s `std`.
fn vec_std_population(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mean = vec_mean_or_zero(values);
    (values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

/// Differential Evolution optimizer.
///
/// A population-based stochastic optimizer for continuous functions.
/// Use [`DifferentialEvolution::new`] to create an instance, configure
/// with [`config_mut`](Self::config_mut), then call [`solve`](Self::solve).
pub struct DifferentialEvolution<'a, F>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    pub(crate) func: &'a F,
    pub(crate) lower: Vec<f64>,
    pub(crate) upper: Vec<f64>,
    pub(crate) config: DEConfig,
}

impl<'a, F> DifferentialEvolution<'a, F>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    /// Creates a new DE optimizer with objective `func` and bounds [lower, upper].
    ///
    /// # Errors
    ///
    /// Returns `DEError::BoundsMismatch` if `lower` and `upper` have different lengths.
    /// Returns `DEError::InvalidBounds` if any lower bound exceeds its corresponding upper bound.
    pub fn new(func: &'a F, lower: Vec<f64>, upper: Vec<f64>) -> Result<Self> {
        if lower.len() != upper.len() {
            return Err(DEError::BoundsMismatch {
                lower_len: lower.len(),
                upper_len: upper.len(),
            });
        }

        // Validate that lower <= upper for all dimensions
        for i in 0..lower.len() {
            if lower[i] > upper[i] {
                return Err(DEError::InvalidBounds {
                    index: i,
                    lower: lower[i],
                    upper: upper[i],
                });
            }
        }

        Ok(Self {
            func,
            lower,
            upper,
            config: DEConfig::default(),
        })
    }

    /// Mutable access to configuration
    pub fn config_mut(&mut self) -> &mut DEConfig {
        &mut self.config
    }

    /// Run the optimization and return a report
    pub fn solve(&mut self) -> DEReport {
        use super::apply_integrality::apply_integrality;
        use super::apply_wls::apply_wls_in_place;
        use super::crossover_binomial::binomial_crossover_clamp_into;
        use super::crossover_exponential::exponential_crossover_clamp_into;
        use super::init_latin_hypercube::init_latin_hypercube;
        use super::init_random::init_random;
        use super::mutant_adaptive::mutant_adaptive_into;
        use super::mutant_best1::mutant_best1_into;
        use super::mutant_best2::mutant_best2_into;
        use super::mutant_current_to_best1::mutant_current_to_best1_into;
        use super::mutant_current_to_pbest1::mutant_current_to_pbest1_into;
        use super::mutant_rand_to_best1::mutant_rand_to_best1_into;
        use super::mutant_rand1::mutant_rand1_into;
        use super::mutant_rand2::mutant_rand2_into;
        use super::mutation::Mutation;
        use crate::util::parallel_eval::evaluate_population_parallel_slice;
        use rayon::prelude::*;
        use std::cell::RefCell;
        use std::sync::Arc;

        // Thread-local scratch used to copy a slice into an Array1 before calling
        // the user-provided objective function (which expects &[f64]).
        thread_local! {
            static X_SCRATCH: RefCell<Vec<f64>> = RefCell::new(vec![0.0; 0]);
        }

        let n = self.lower.len();

        // Identify fixed (equal-bounds) and free variables
        let mut is_free: Vec<bool> = Vec::with_capacity(n);
        for i in 0..n {
            is_free.push((self.upper[i] - self.lower[i]).abs() > 0.0);
        }
        let n_free = is_free.iter().filter(|&&b| b).count();
        let _n_equal = n - n_free;
        if n_free == 0 {
            // All fixed; just evaluate x = lower
            let x_fixed = self.lower.clone();
            let mut x_eval = x_fixed.clone();
            if let Some(mask) = &self.config.integrality {
                apply_integrality(&mut x_eval, mask, &self.lower, &self.upper);
            }
            let f = (self.func)(&x_eval);
            return DEReport {
                x: x_eval,
                fun: f,
                success: true,
                message: "All variables fixed by bounds".into(),
                nit: 0,
                nfev: 1,
                population: vec![vec![0.0; n]; 1],
                population_energies: vec![f],
            };
        }

        let is_lshade = matches!(
            self.config.strategy,
            Strategy::LShadeBin | Strategy::LShadeExp
        );
        let mut npop = if is_lshade {
            self.config.lshade.initial_population_size(n_free)
        } else {
            self.config.popsize * n_free
        };
        let max_nfev = if is_lshade {
            self.config.maxiter * npop
        } else {
            0
        };

        if self.config.disp {
            eprintln!(
                "DE Init: {} dimensions ({} free), population={}, maxiter={}",
                n, n_free, npop, self.config.maxiter
            );
            eprintln!(
                "  Strategy: {:?}, Mutation: {:?}, Crossover: CR={:.3}",
                self.config.strategy, self.config.mutation, self.config.recombination
            );
            eprintln!(
                "  Tolerances: tol={:.2e}, atol={:.2e}",
                self.config.tol, self.config.atol
            );
        }

        // Timing toggle via env var
        let timing_enabled = std::env::var("AUTOEQ_DE_TIMING")
            .map(|v| v != "0")
            .unwrap_or(false);

        // RNG
        let mut rng: StdRng = match self.config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                let mut thread_rng = rand::rng();
                StdRng::from_rng(&mut thread_rng)
            }
        };

        // Initialize population in [lower, upper]
        let mut pop = match self.config.init {
            Init::LatinHypercube => {
                if self.config.disp {
                    eprintln!("  Using Latin Hypercube initialization");
                }
                init_latin_hypercube(n, npop, &self.lower, &self.upper, &is_free, &mut rng)
            }
            Init::Random => {
                if self.config.disp {
                    eprintln!("  Using Random initialization");
                }
                init_random(n, npop, &self.lower, &self.upper, &is_free, &mut rng)
            }
        };

        // Evaluate energies (objective + penalties)
        let mut nfev: usize = 0;
        if self.config.disp {
            eprintln!("  Evaluating initial population of {} individuals...", npop);
        }

        // Prepare population for evaluation (apply integrality constraints).
        // Borrow the population directly when no integrality is required to
        // avoid a redundant full copy.
        let t_integrality0 = Instant::now();
        let eval_pop = if let Some(mask) = &self.config.integrality {
            let mut ep = pop.clone();
            for row in ep.iter_mut() {
                apply_integrality(row, mask, &self.lower, &self.upper);
            }
            std::borrow::Cow::Owned(ep)
        } else {
            std::borrow::Cow::Borrowed(&pop)
        };
        let t_integrality = t_integrality0.elapsed();

        // Build thread-safe energy function that includes penalties
        let func_ref = self.func;
        let penalty_ineq_vec: Vec<PenaltyTuple> = self
            .config
            .penalty_ineq
            .iter()
            .map(|(f, w)| (f.clone(), *w))
            .collect();
        let penalty_eq_vec: Vec<PenaltyTuple> = self
            .config
            .penalty_eq
            .iter()
            .map(|(f, w)| (f.clone(), *w))
            .collect();
        let linear_penalty = self.config.linear_penalty.clone();
        let no_penalties =
            penalty_ineq_vec.is_empty() && penalty_eq_vec.is_empty() && linear_penalty.is_none();

        let energy_fn = Arc::new(move |x: &[f64]| -> f64 {
            // Copy the slice into a reusable per-thread Array1 so the public
            // objective function signature (&[f64]) is preserved without
            // a per-evaluation heap allocation.
            let energy = X_SCRATCH.with(|xs| {
                let mut scratch = xs.borrow_mut();
                if scratch.len() != x.len() {
                    *scratch = vec![0.0; x.len()];
                }
                scratch.copy_from_slice(x);
                let x_arr = &*scratch;

                if no_penalties {
                    // Fast path: most DE benchmarks/use cases have no penalties,
                    // so avoid constructing empty iterators and the linear-penalty
                    // branch inside every evaluation.
                    (func_ref)(x_arr)
                } else {
                    let base = (func_ref)(x_arr);
                    let mut p = 0.0;
                    for (f, w) in &penalty_ineq_vec {
                        let v = f(x_arr);
                        let viol = v.max(0.0);
                        p += w * viol * viol;
                    }
                    for (h, w) in &penalty_eq_vec {
                        let v = h(x_arr);
                        p += w * v * v;
                    }
                    if let Some(ref lp) = linear_penalty {
                        for ((row, &lo), &hi) in lp.a.iter().zip(lp.lb.iter()).zip(lp.ub.iter()) {
                            let v: f64 = row.iter().zip(x_arr.iter()).map(|(&a, &xj)| a * xj).sum();
                            if v < lo {
                                let d = lo - v;
                                p += lp.weight * d * d;
                            } else if v > hi {
                                let d = v - hi;
                                p += lp.weight * d * d;
                            }
                        }
                    }
                    base + p
                }
            });
            if energy.is_finite() {
                energy
            } else {
                f64::INFINITY
            }
        });

        let t_eval0 = Instant::now();
        let mut energies = evaluate_population_parallel_slice(
            eval_pop.as_ref(),
            energy_fn.clone(),
            &self.config.parallel,
        );
        drop(eval_pop); // release the borrow of `pop` before the mutable main loop
        let t_eval_init = t_eval0.elapsed();
        nfev += npop;
        if timing_enabled {
            eprintln!(
                "TIMING init: integrality={:.3} ms, eval={:.3} ms",
                t_integrality.as_secs_f64() * 1e3,
                t_eval_init.as_secs_f64() * 1e3
            );
        }

        // Report initial population statistics
        let pop_mean = vec_mean_or_zero(&energies);
        let pop_std = vec_std_population(&energies);
        if self.config.disp {
            eprintln!(
                "  Initial population: mean={:.6e}, std={:.6e}",
                pop_mean, pop_std
            );
        }

        // If x0 provided, override the best member
        if let Some(x0) = &self.config.x0 {
            let mut x0c = x0.clone();
            // Clip to bounds
            for ((xc, &lo), &hi) in x0c.iter_mut().zip(self.lower.iter()).zip(self.upper.iter()) {
                *xc = xc.clamp(lo, hi);
            }
            if let Some(mask) = &self.config.integrality {
                apply_integrality(&mut x0c, mask, &self.lower, &self.upper);
            }
            let f0 = self.energy(&x0c);
            nfev += 1;
            // find current best
            let (best_idx, _best_f) = argmin(&energies);
            pop[best_idx].copy_from_slice(&x0c);
            energies[best_idx] = f0;
        }

        let (mut best_idx, mut best_f) = argmin(&energies);
        let mut best_x = pop[best_idx].clone();

        if self.config.disp {
            eprintln!(
                "  Initial best: fitness={:.6e} at index {}",
                best_f, best_idx
            );
            let param_summary: Vec<String> = (0..best_x.len() / 3)
                .map(|i| {
                    let freq = 10f64.powf(best_x[i * 3]);
                    let q = best_x[i * 3 + 1];
                    let gain = best_x[i * 3 + 2];
                    format!("f{:.0}Hz/Q{:.2}/G{:.2}dB", freq, q, gain)
                })
                .collect();
            eprintln!("  Initial best params: [{}]", param_summary.join(", "));
        }

        if self.config.disp {
            eprintln!("DE iter {:4}  best_f={:.6e}", 0, best_f);
        }

        // Initialize adaptive state if adaptive strategies are enabled
        let mut adaptive_state = if matches!(
            self.config.strategy,
            Strategy::AdaptiveBin | Strategy::AdaptiveExp
        ) || self.config.adaptive.adaptive_mutation
        {
            Some(AdaptiveState::new(&self.config.adaptive))
        } else {
            None
        };

        // Initialize external archive for L-SHADE strategies
        let external_archive: Option<Arc<RwLock<super::external_archive::ExternalArchive>>> = if matches!(
            self.config.strategy,
            Strategy::LShadeBin | Strategy::LShadeExp
        ) {
            Some(Arc::new(RwLock::new(
                super::external_archive::ExternalArchive::with_population_size(
                    npop,
                    self.config.lshade.arc_rate,
                ),
            )))
        } else {
            None
        };

        // Main loop
        let mut success = false;
        let mut message = String::new();
        let mut nit = 0;
        let mut accepted_trials;
        let mut improvement_count;

        let mut t_build_tot = std::time::Duration::ZERO;
        let mut t_eval_tot = std::time::Duration::ZERO;
        let mut t_select_tot = std::time::Duration::ZERO;
        let mut t_iter_tot = std::time::Duration::ZERO;

        // Reusable trial matrix and per-trial parameter storage, allocated once
        // outside the hot loop.  They are only reallocated when L-SHADE shrinks
        // the population.
        let mut trials_buf: Vec<Vec<f64>> = vec![vec![0.0; n]; npop];
        // Plain `f64` vectors for F/CR and trial energies: writes are to disjoint
        // indices and the parallel loop joins before the sequential read phase,
        // so atomics are unnecessary here.
        let mut trial_f: Vec<f64> = vec![0.0; npop];
        let mut trial_cr: Vec<f64> = vec![0.0; npop];
        let mut trial_energies: Vec<f64> = vec![0.0; npop];

        // One scratch vector per rayon thread for the mutant vector.
        thread_local! {
            static MUTANT_SCRATCH: RefCell<Vec<f64>> = RefCell::new(vec![0.0; 0]);
        }

        // Pre-extract contiguous slices for bound clipping; avoids repeated
        // `as_slice()` calls inside the per-individual hot path.
        let lower_slice = self.lower.as_slice();
        let upper_slice = self.upper.as_slice();

        // Guarded fast path for the benchmark's default configuration:
        // Best1Bin strategy, default dithered mutation, no penalties, no
        // integrality, no WLS, and no adaptive/L-SHADE machinery. This path
        // inlines the trial build and evaluation, removing strategy/crossover
        // enum dispatch and the penalty-free energy wrapper. Non-default configs
        // fall through to the general loop below unchanged.
        let use_fast_path = npop >= 3
            && matches!(self.config.strategy, Strategy::Best1Bin)
            && matches!(self.config.mutation, Mutation::Range { min: 0.0, max: 2.0 })
            && self.config.penalty_ineq.is_empty()
            && self.config.penalty_eq.is_empty()
            && self.config.linear_penalty.is_none()
            && self.config.integrality.is_none()
            && !self.config.adaptive.wls_enabled
            && adaptive_state.is_none()
            && !is_lshade
            && self.config.parallel.enabled;

        for iter in 1..=self.config.maxiter {
            nit = iter;
            accepted_trials = 0;
            improvement_count = 0;

            let iter_start = Instant::now();

            // Generate all trials into the reusable pre-allocated matrix, then
            // evaluate in parallel.
            let t_build0 = Instant::now();

            if use_fast_path {
                // Fast path for the default Best1Bin/binomial/no-penalty
                // configuration. Mirrors the general loop's raw-pointer layout
                // but removes strategy/crossover enum dispatch, skips F/CR
                // storage, and calls the objective directly instead of through
                // the Arc penalty wrapper.
                let func_ref = self.func;
                let recombination = self.config.recombination;
                let seed = self.config.seed;
                trials_buf
                    .par_iter_mut()
                    .zip_eq(trial_energies.par_iter_mut())
                    .enumerate()
                    .with_min_len(16)
                    .for_each(|(i, (trial_row, trial_energy))| {
                        let mut local_rng: SmallRng = if let Some(base_seed) = seed {
                            SmallRng::seed_from_u64(
                                base_seed
                                    .wrapping_add((iter as u64) << 32)
                                    .wrapping_add(i as u64),
                            )
                        } else {
                            let mut thread_rng = rand::rng();
                            SmallRng::from_rng(&mut thread_rng)
                        };

                        let f = local_rng.random_range(0.0..2.0);
                        let cr = recombination;

                        MUTANT_SCRATCH.with(|ms| {
                            let mut scratch = ms.borrow_mut();
                            if scratch.len() != n {
                                *scratch = vec![0.0; n];
                            }
                            mutant_best1_into(&mut scratch, i, &pop, best_idx, f, &mut local_rng);
                            let target_slice = &pop[i];
                            let mutant_slice = scratch.as_slice();
                            let trial_slice = trial_row.as_mut_slice();
                            binomial_crossover_clamp_into(
                                trial_slice,
                                target_slice,
                                mutant_slice,
                                cr,
                                lower_slice,
                                upper_slice,
                                &mut local_rng,
                            );
                        });

                        let trial_slice = trial_row.as_slice();
                        let energy = X_SCRATCH.with(|xs| {
                            let mut x = xs.borrow_mut();
                            if x.len() != n {
                                *x = vec![0.0; n];
                            }
                            x.copy_from_slice(trial_slice);
                            (func_ref)(&x)
                        });

                        *trial_energy = if energy.is_finite() {
                            energy
                        } else {
                            f64::INFINITY
                        };
                    });
            } else {
                // Pre-sort indices for adaptive/L-SHADE strategies to avoid re-sorting in the loop
                let sorted_indices = if matches!(
                    self.config.strategy,
                    Strategy::AdaptiveBin
                        | Strategy::AdaptiveExp
                        | Strategy::LShadeBin
                        | Strategy::LShadeExp
                ) {
                    let mut indices: Vec<usize> = (0..npop).collect();
                    indices.sort_by(|&a, &b| {
                        energies[a]
                            .partial_cmp(&energies[b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    indices
                } else {
                    Vec::new() // Not needed for other strategies
                };

                // Writes go through disjoint `par_iter_mut` rows/slots; the loop
                // joins before the selection phase reads them back.
                let eval_fn = energy_fn.clone();
                // Strategy/crossover type is constant for the whole iteration; resolve
                // it once instead of inside every parallel task.
                let crossover_type = self.config.strategy.crossover();
                trials_buf
                    .par_iter_mut()
                    .zip_eq(trial_f.par_iter_mut())
                    .zip_eq(trial_cr.par_iter_mut())
                    .zip_eq(trial_energies.par_iter_mut())
                    .enumerate()
                    .with_min_len(16)
                    .for_each(|(i, (((trial_row, trial_f), trial_cr), trial_energy))| {
                        // Create a fast, deterministic per-individual RNG from a composite
                        // seed. SmallRng seeds much cheaper than StdRng while preserving
                        // reproducibility for a fixed configuration/seed.
                        let mut local_rng: SmallRng = if let Some(base_seed) = self.config.seed {
                            SmallRng::seed_from_u64(
                                base_seed
                                    .wrapping_add((iter as u64) << 32)
                                    .wrapping_add(i as u64),
                            )
                        } else {
                            // Use thread_rng for unseeded runs
                            let mut thread_rng = rand::rng();
                            SmallRng::from_rng(&mut thread_rng)
                        };

                        // Sample mutation factor and crossover rate (adaptive or fixed)
                        let (f, cr) = if let Some(ref adaptive) = adaptive_state {
                            // Use adaptive parameter sampling
                            let adaptive_f = adaptive.sample_f(&mut local_rng);
                            let adaptive_cr = adaptive.sample_cr(&mut local_rng);
                            (adaptive_f, adaptive_cr)
                        } else {
                            // Use fixed or dithered parameters
                            (
                                self.config.mutation.sample(&mut local_rng),
                                self.config.recombination,
                            )
                        };

                        MUTANT_SCRATCH.with(|ms| {
                            let mut scratch = ms.borrow_mut();
                            if scratch.len() != n {
                                *scratch = vec![0.0; n];
                            }

                            // Generate mutant in place based on strategy
                            match self.config.strategy {
                                Strategy::Best1Bin | Strategy::Best1Exp => {
                                    mutant_best1_into(
                                        &mut scratch,
                                        i,
                                        &pop,
                                        best_idx,
                                        f,
                                        &mut local_rng,
                                    );
                                }
                                Strategy::Rand1Bin | Strategy::Rand1Exp => {
                                    mutant_rand1_into(&mut scratch, i, &pop, f, &mut local_rng);
                                }
                                Strategy::Rand2Bin | Strategy::Rand2Exp => {
                                    mutant_rand2_into(&mut scratch, i, &pop, f, &mut local_rng);
                                }
                                Strategy::CurrentToBest1Bin | Strategy::CurrentToBest1Exp => {
                                    mutant_current_to_best1_into(
                                        &mut scratch,
                                        i,
                                        &pop,
                                        best_idx,
                                        f,
                                        &mut local_rng,
                                    );
                                }
                                Strategy::Best2Bin | Strategy::Best2Exp => {
                                    mutant_best2_into(
                                        &mut scratch,
                                        i,
                                        &pop,
                                        best_idx,
                                        f,
                                        &mut local_rng,
                                    );
                                }
                                Strategy::RandToBest1Bin | Strategy::RandToBest1Exp => {
                                    mutant_rand_to_best1_into(
                                        &mut scratch,
                                        i,
                                        &pop,
                                        best_idx,
                                        f,
                                        &mut local_rng,
                                    );
                                }
                                Strategy::AdaptiveBin | Strategy::AdaptiveExp => {
                                    if let Some(ref adaptive) = adaptive_state {
                                        mutant_adaptive_into(
                                            &mut scratch,
                                            i,
                                            &pop,
                                            &sorted_indices,
                                            adaptive.current_w,
                                            f,
                                            &mut local_rng,
                                        );
                                    } else {
                                        mutant_rand1_into(&mut scratch, i, &pop, f, &mut local_rng);
                                    }
                                }
                                Strategy::LShadeBin | Strategy::LShadeExp => {
                                    let pbest_size =
                                        super::mutant_current_to_pbest1::compute_pbest_size(
                                            self.config.lshade.p,
                                            pop.len(),
                                        );
                                    let archive_ref =
                                        external_archive.as_ref().and_then(|a| a.read().ok());
                                    mutant_current_to_pbest1_into(
                                        &mut scratch,
                                        i,
                                        &pop,
                                        &sorted_indices,
                                        pbest_size,
                                        archive_ref.as_deref(),
                                        f,
                                        &mut local_rng,
                                    );
                                }
                            };

                            // Crossover from target row into the trial row.
                            let target_slice = &pop[i];
                            let mutant_slice = scratch.as_slice();
                            let trial_slice = trial_row.as_mut_slice();
                            match crossover_type {
                                Crossover::Binomial => binomial_crossover_clamp_into(
                                    trial_slice,
                                    target_slice,
                                    mutant_slice,
                                    cr,
                                    lower_slice,
                                    upper_slice,
                                    &mut local_rng,
                                ),
                                Crossover::Exponential => exponential_crossover_clamp_into(
                                    trial_slice,
                                    target_slice,
                                    mutant_slice,
                                    cr,
                                    lower_slice,
                                    upper_slice,
                                    &mut local_rng,
                                ),
                            }
                        });

                        // Apply WLS if enabled, in place on the trial row.
                        if self.config.adaptive.wls_enabled
                            && local_rng.random::<f64>() < self.config.adaptive.wls_prob
                        {
                            apply_wls_in_place(
                                trial_row.as_mut_slice(),
                                &self.lower,
                                &self.upper,
                                self.config.adaptive.wls_scale,
                                &mut local_rng,
                            );
                        }

                        // Apply integrality if provided
                        if let Some(mask) = &self.config.integrality {
                            apply_integrality(
                                trial_row.as_mut_slice(),
                                mask,
                                &self.lower,
                                &self.upper,
                            );
                        }

                        let trial_slice = trial_row.as_slice();
                        let energy = eval_fn(trial_slice);

                        *trial_f = f;
                        *trial_cr = cr;
                        *trial_energy = energy;
                    });
            }

            let t_build = t_build0.elapsed();
            let t_eval = std::time::Duration::ZERO;
            nfev += npop;

            let t_select0 = Instant::now();
            // Selection phase: update population based on trial results
            for (i, trial_energy) in trial_energies.iter().enumerate() {
                let f = trial_f[i];
                let cr = trial_cr[i];

                // Selection: replace if better
                if *trial_energy <= energies[i] {
                    // For L-SHADE: add replaced solution to archive
                    if let Some(ref archive) = external_archive
                        && *trial_energy < energies[i]
                        && let Ok(mut arch) = archive.write()
                    {
                        arch.add_with_rng(pop[i].clone(), &mut rng);
                    }
                    pop[i].copy_from_slice(&trials_buf[i]);
                    energies[i] = *trial_energy;
                    accepted_trials += 1;

                    // Update adaptive parameters if improvement
                    if let Some(ref mut adaptive) = adaptive_state {
                        adaptive.record_success(f, cr);
                    }

                    // Track if this is an improvement over the current best
                    if *trial_energy < best_f {
                        improvement_count += 1;
                    }
                }
            }
            let t_select = t_select0.elapsed();

            // L-SHADE: linear population size reduction
            if is_lshade {
                let current_npop = self
                    .config
                    .lshade
                    .current_population_size(n_free, nfev, max_nfev);
                if current_npop < pop.len() {
                    let mut indices: Vec<usize> = (0..pop.len()).collect();
                    indices.sort_by(|&a, &b| {
                        energies[a]
                            .partial_cmp(&energies[b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let keep: Vec<usize> = indices.into_iter().take(current_npop).collect();

                    let mut new_pop = vec![vec![0.0; n]; current_npop];
                    let mut new_energies = vec![0.0; current_npop];
                    for (new_i, &old_i) in keep.iter().enumerate() {
                        new_pop[new_i].copy_from_slice(&pop[old_i]);
                        new_energies[new_i] = energies[old_i];
                    }
                    pop = new_pop;
                    energies = new_energies;
                    npop = current_npop;
                    trials_buf = vec![vec![0.0; n]; npop];
                    trial_f = vec![0.0; npop];
                    trial_cr = vec![0.0; npop];
                    trial_energies = vec![0.0; npop];

                    if let Some(ref archive) = external_archive
                        && let Ok(mut arch) = archive.write()
                    {
                        arch.resize(self.config.lshade.current_archive_size(current_npop));
                    }
                }
            }

            t_build_tot += t_build;
            t_eval_tot += t_eval;
            t_select_tot += t_select;
            let iter_dur = iter_start.elapsed();
            t_iter_tot += iter_dur;

            if timing_enabled && (iter <= 5 || iter % 10 == 0) {
                eprintln!(
                    "TIMING iter {:4}: build={:.3} ms, eval={:.3} ms, select={:.3} ms, total={:.3} ms",
                    iter,
                    t_build.as_secs_f64() * 1e3,
                    t_eval.as_secs_f64() * 1e3,
                    t_select.as_secs_f64() * 1e3,
                    iter_dur.as_secs_f64() * 1e3,
                );
            }

            // Update adaptive parameters after each generation
            if let Some(ref mut adaptive) = adaptive_state {
                adaptive.update(&self.config.adaptive, iter, self.config.maxiter);
            }

            // Update best solution after generation
            let (new_best_idx, new_best_f) = argmin(&energies);
            if new_best_f < best_f {
                best_idx = new_best_idx;
                best_f = new_best_f;
                best_x.copy_from_slice(&pop[best_idx]);
            }

            // Convergence check
            let pop_mean = vec_mean_or_zero(&energies);
            let pop_std = vec_std_population(&energies);
            let convergence_threshold = self.config.atol + self.config.tol * pop_mean.abs();

            if self.config.disp {
                eprintln!(
                    "DE iter {:4}  best_f={:.6e}  std={:.3e}  accepted={}/{}, improved={}",
                    iter, best_f, pop_std, accepted_trials, npop, improvement_count
                );
            }

            // Callback
            if let Some(ref mut cb) = self.config.callback {
                let intermediate = DEIntermediate {
                    x: best_x.clone(),
                    fun: best_f,
                    convergence: pop_std,
                    iter,
                };
                match cb(&intermediate) {
                    CallbackAction::Stop => {
                        success = true;
                        message = "Optimization stopped by callback".to_string();
                        break;
                    }
                    CallbackAction::Continue => {}
                }
            }

            if pop_std <= convergence_threshold {
                success = true;
                message = format!(
                    "Converged: std(pop_f)={:.3e} <= threshold={:.3e}",
                    pop_std, convergence_threshold
                );
                break;
            }
        }

        if !success {
            message = format!("Maximum iterations reached: {}", self.config.maxiter);
        }

        if self.config.disp {
            eprintln!("DE finished: {}", message);
        }

        // Polish if configured
        let (final_x, final_f, polish_nfev) = if let Some(ref polish_cfg) = self.config.polish {
            if polish_cfg.enabled {
                self.polish(&best_x)
            } else {
                (best_x.clone(), best_f, 0)
            }
        } else {
            (best_x.clone(), best_f, 0)
        };

        if timing_enabled {
            eprintln!(
                "TIMING total: build={:.3} s, eval={:.3} s, select={:.3} s, iter_total={:.3} s",
                t_build_tot.as_secs_f64(),
                t_eval_tot.as_secs_f64(),
                t_select_tot.as_secs_f64(),
                t_iter_tot.as_secs_f64()
            );
        }

        self.finish_report(
            pop,
            energies,
            final_x,
            final_f,
            success,
            message,
            nit,
            nfev + polish_nfev,
        )
    }
}
