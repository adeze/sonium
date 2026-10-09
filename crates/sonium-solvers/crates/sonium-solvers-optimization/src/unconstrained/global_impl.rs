//! Deterministic bounded global optimizers matching Wolfram's global-method
//! examples: Differential Evolution, Simulated Annealing, and Random Search.

use crate::types::{Bounds, OptimError, OptimResult, OptimStatus, validate_bounds};

#[derive(Debug, Clone)]
pub struct DifferentialEvolutionConfig {
    pub bounds: Bounds<f64>,
    /// Population size; zero selects `max(12, 10 * dimension)`.
    pub population_size: usize,
    pub max_generations: usize,
    pub mutation_factor: f64,
    pub crossover_probability: f64,
    pub target_f: f64,
    pub seed: u64,
}

impl Default for DifferentialEvolutionConfig {
    fn default() -> Self {
        Self {
            bounds: Vec::new(),
            population_size: 0,
            max_generations: 1_000,
            mutation_factor: 0.8,
            crossover_probability: 0.9,
            target_f: f64::NEG_INFINITY,
            seed: 0x5eed_d1ff_e12e,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SimulatedAnnealingConfig {
    pub bounds: Bounds<f64>,
    /// Optional initial point; the box centre is used when absent.
    pub x0: Option<Vec<f64>>,
    pub max_iterations: usize,
    pub initial_temperature: f64,
    pub cooling_rate: f64,
    /// Proposal standard deviation as a fraction of each box width at the
    /// initial temperature.
    pub step_scale: f64,
    pub target_f: f64,
    pub seed: u64,
}

impl Default for SimulatedAnnealingConfig {
    fn default() -> Self {
        Self {
            bounds: Vec::new(),
            x0: None,
            max_iterations: 20_000,
            initial_temperature: 1.0,
            cooling_rate: 0.999,
            step_scale: 0.15,
            target_f: f64::NEG_INFINITY,
            seed: 0x5eed_a11e_a11e,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RandomSearchConfig {
    pub bounds: Bounds<f64>,
    pub samples: usize,
    /// Include the box centre as the first start.
    pub include_center: bool,
    pub target_f: f64,
    pub seed: u64,
}

impl Default for RandomSearchConfig {
    fn default() -> Self {
        Self {
            bounds: Vec::new(),
            samples: 10_000,
            include_center: true,
            target_f: f64::NEG_INFINITY,
            seed: 0x5eed_5ea2_c001,
        }
    }
}

pub fn differential_evolution<F>(
    f: &F,
    config: &DifferentialEvolutionConfig,
) -> Result<OptimResult<f64>, OptimError>
where
    F: Fn(&[f64]) -> f64 + Sync + ?Sized,
{
    validate_target(config.target_f, "Differential Evolution")?;
    let (lower, upper) = finite_box(&config.bounds)?;
    if config.max_generations == 0
        || !config.mutation_factor.is_finite()
        || config.mutation_factor <= 0.0
        || config.mutation_factor > 2.0
        || !config.crossover_probability.is_finite()
        || !(0.0..=1.0).contains(&config.crossover_probability)
    {
        return invalid("invalid Differential Evolution configuration");
    }
    let n = lower.len();
    let population_size = if config.population_size == 0 {
        (10 * n).max(12)
    } else {
        config.population_size
    };
    if population_size < 4 {
        return invalid("Differential Evolution needs at least four population members");
    }
    let mut rng = XorShift64Star::new(config.seed);
    let mut population: Vec<Vec<f64>> = (0..population_size)
        .map(|_| random_point(&lower, &upper, &mut rng))
        .collect();
    let mut values = Vec::with_capacity(population_size);
    for point in &population {
        values.push(evaluate(f, point, 0)?);
    }
    let mut evaluations = population_size;
    let mut best = argmin(&values);
    let mut generations = 0;

    while generations < config.max_generations && values[best] > config.target_f {
        for target in 0..population_size {
            let [a, b, c] = distinct_indices(population_size, target, &mut rng);
            let forced = rng.index(n);
            let mut trial = population[target].clone();
            for coordinate in 0..n {
                if coordinate == forced || rng.uniform() < config.crossover_probability {
                    trial[coordinate] = (population[a][coordinate]
                        + config.mutation_factor
                            * (population[b][coordinate] - population[c][coordinate]))
                        .clamp(lower[coordinate], upper[coordinate]);
                }
            }
            let trial_value = evaluate(f, &trial, generations + 1)?;
            evaluations += 1;
            if trial_value <= values[target] {
                population[target] = trial;
                values[target] = trial_value;
                if trial_value < values[best] {
                    best = target;
                }
            }
        }
        generations += 1;
        best = argmin(&values);
    }
    let result = global_result(
        population[best].clone(),
        values[best],
        generations,
        evaluations,
        values[best] <= config.target_f,
    );
    super::validate_result(&result)?;
    Ok(result)
}

pub fn simulated_annealing<F>(
    f: &F,
    config: &SimulatedAnnealingConfig,
) -> Result<OptimResult<f64>, OptimError>
where
    F: Fn(&[f64]) -> f64 + Sync + ?Sized,
{
    validate_target(config.target_f, "Simulated Annealing")?;
    let (lower, upper) = finite_box(&config.bounds)?;
    if config.max_iterations == 0
        || !config.initial_temperature.is_finite()
        || config.initial_temperature <= 0.0
        || !config.cooling_rate.is_finite()
        || !(0.0..1.0).contains(&config.cooling_rate)
        || !config.step_scale.is_finite()
        || config.step_scale <= 0.0
    {
        return invalid("invalid Simulated Annealing configuration");
    }
    let n = lower.len();
    let mut current = config.x0.clone().unwrap_or_else(|| {
        lower
            .iter()
            .zip(&upper)
            .map(|(&lo, &hi)| 0.5 * (lo + hi))
            .collect()
    });
    if current.len() != n {
        return Err(OptimError::DimensionMismatch {
            expected: n,
            got: current.len(),
        });
    }
    for coordinate in 0..n {
        current[coordinate] = current[coordinate].clamp(lower[coordinate], upper[coordinate]);
    }
    let mut rng = XorShift64Star::new(config.seed);
    let mut current_value = evaluate(f, &current, 0)?;
    let mut best = current.clone();
    let mut best_value = current_value;
    let mut temperature = config.initial_temperature;
    let mut iterations = 0;
    while iterations < config.max_iterations && best_value > config.target_f {
        let mut proposal = current.clone();
        for coordinate in 0..n {
            let width = upper[coordinate] - lower[coordinate];
            proposal[coordinate] = (proposal[coordinate]
                + config.step_scale * width * temperature.sqrt() * rng.standard_normal())
            .clamp(lower[coordinate], upper[coordinate]);
        }
        let proposal_value = evaluate(f, &proposal, iterations + 1)?;
        let accept = proposal_value <= current_value
            || rng.uniform() < ((current_value - proposal_value) / temperature).exp();
        if accept {
            current = proposal;
            current_value = proposal_value;
            if current_value < best_value {
                best.clone_from(&current);
                best_value = current_value;
            }
        }
        temperature = (temperature * config.cooling_rate).max(f64::MIN_POSITIVE);
        iterations += 1;
    }
    let result = global_result(
        best,
        best_value,
        iterations,
        iterations + 1,
        best_value <= config.target_f,
    );
    super::validate_result(&result)?;
    Ok(result)
}

pub fn random_search<F>(f: &F, config: &RandomSearchConfig) -> Result<OptimResult<f64>, OptimError>
where
    F: Fn(&[f64]) -> f64 + Sync + ?Sized,
{
    validate_target(config.target_f, "Random Search")?;
    let (lower, upper) = finite_box(&config.bounds)?;
    if config.samples == 0 {
        return invalid("Random Search needs at least one sample");
    }
    let mut rng = XorShift64Star::new(config.seed);
    let mut best = if config.include_center {
        lower
            .iter()
            .zip(&upper)
            .map(|(&lo, &hi)| 0.5 * (lo + hi))
            .collect()
    } else {
        random_point(&lower, &upper, &mut rng)
    };
    let mut best_value = evaluate(f, &best, 0)?;
    let mut evaluations = 1;
    while evaluations < config.samples && best_value > config.target_f {
        let point = random_point(&lower, &upper, &mut rng);
        let value = evaluate(f, &point, evaluations)?;
        evaluations += 1;
        if value < best_value {
            best = point;
            best_value = value;
        }
    }
    let result = global_result(
        best,
        best_value,
        evaluations,
        evaluations,
        best_value <= config.target_f,
    );
    super::validate_result(&result)?;
    Ok(result)
}

fn finite_box(bounds: &Bounds<f64>) -> Result<(Vec<f64>, Vec<f64>), OptimError> {
    if bounds.is_empty() {
        return invalid("global optimizer needs at least one bounded variable");
    }
    validate_bounds(bounds, bounds.len())?;
    let mut lower = Vec::with_capacity(bounds.len());
    let mut upper = Vec::with_capacity(bounds.len());
    for (index, bound) in bounds.iter().enumerate() {
        let (Some(lo), Some(hi)) = (bound.lower, bound.upper) else {
            return invalid(&format!(
                "variable {index} does not have finite two-sided bounds"
            ));
        };
        lower.push(lo);
        upper.push(hi);
    }
    Ok((lower, upper))
}

/// Validate the optional target objective used as an early-stop threshold.
/// `-∞` is the documented sentinel meaning "no target"; `NaN` and `+∞`
/// would otherwise make the comparison non-informative and can report a
/// successful result without evaluating a meaningful target.
fn validate_target(target_f: f64, method: &str) -> Result<(), OptimError> {
    if target_f.is_nan() || target_f == f64::INFINITY {
        return invalid(&format!("invalid {method} target_f"));
    }
    Ok(())
}

fn evaluate<F>(f: &F, x: &[f64], iteration: usize) -> Result<f64, OptimError>
where
    F: Fn(&[f64]) -> f64 + Sync + ?Sized,
{
    let value = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(x))).map_err(|_| {
        OptimError::CallbackPanicked {
            iteration,
            stage: "global objective".into(),
        }
    })?;
    if !value.is_finite() {
        return Err(OptimError::NumericalError {
            iteration,
            message: "global objective returned a non-finite value".into(),
        });
    }
    Ok(value)
}

fn global_result(
    x: Vec<f64>,
    fun: f64,
    iterations: usize,
    fun_evals: usize,
    reached_target: bool,
) -> OptimResult<f64> {
    OptimResult {
        x,
        fun,
        grad: None,
        iterations,
        fun_evals,
        grad_evals: 0,
        hess_evals: 0,
        grad_norm: 0.0,
        converged: reached_target,
        status: if reached_target {
            OptimStatus::FunctionTolerance
        } else {
            OptimStatus::MaxIterationsReached
        },
        kkt: None,
    }
}

fn random_point(lower: &[f64], upper: &[f64], rng: &mut XorShift64Star) -> Vec<f64> {
    lower
        .iter()
        .zip(upper)
        .map(|(&lo, &hi)| lo + (hi - lo) * rng.uniform())
        .collect()
}

fn argmin(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .min_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |entry| entry.0)
}

fn distinct_indices(size: usize, excluded: usize, rng: &mut XorShift64Star) -> [usize; 3] {
    let mut result = [0; 3];
    for slot in 0..3 {
        loop {
            let candidate = rng.index(size);
            if candidate != excluded && !result[..slot].contains(&candidate) {
                result[slot] = candidate;
                break;
            }
        }
    }
    result
}

fn invalid<T>(message: &str) -> Result<T, OptimError> {
    Err(OptimError::NumericalError {
        iteration: 0,
        message: message.into(),
    })
}

struct XorShift64Star {
    state: u64,
    spare_normal: Option<f64>,
}

impl XorShift64Star {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
            spare_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }

    fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.spare_normal.take() {
            return value;
        }
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * self.uniform();
        self.spare_normal = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Bound;

    fn shifted_sphere(x: &[f64]) -> f64 {
        (x[0] - 0.25).powi(2) + (x[1] + 0.5).powi(2)
    }

    #[test]
    fn differential_evolution_finds_global_quadratic_minimum() {
        let result = differential_evolution(
            &shifted_sphere,
            &DifferentialEvolutionConfig {
                bounds: vec![Bound::both(-2.0, 2.0); 2],
                max_generations: 200,
                target_f: 1.0e-10,
                seed: 42,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.converged, "{result:?}");
        assert!(result.fun < 1.0e-10);
    }

    #[test]
    fn simulated_annealing_escapes_cosine_local_wells() {
        let objective = |x: &[f64]| {
            x[0] * x[0] + x[1] * x[1]
                - 0.1
                    * ((6.0 * std::f64::consts::PI * x[0]).cos()
                        + (6.0 * std::f64::consts::PI * x[1]).cos())
        };
        let result = simulated_annealing(
            &objective,
            &SimulatedAnnealingConfig {
                bounds: vec![Bound::both(-1.0, 1.0); 2],
                x0: Some(vec![0.8, -0.8]),
                max_iterations: 30_000,
                target_f: -0.199,
                seed: 7,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.fun < -0.195, "{result:?}");
    }

    #[test]
    fn random_search_is_seed_reproducible_and_multistart() {
        let config = RandomSearchConfig {
            bounds: vec![Bound::both(-1.0, 1.0); 2],
            samples: 20_000,
            include_center: false,
            seed: 99,
            ..Default::default()
        };
        let first = random_search(&shifted_sphere, &config).unwrap();
        let second = random_search(&shifted_sphere, &config).unwrap();
        assert_eq!(first.x, second.x);
        assert_eq!(first.fun.to_bits(), second.fun.to_bits());
        assert!(first.fun < 5.0e-4, "{first:?}");
    }

    #[test]
    fn invalid_target_thresholds_are_rejected_by_all_global_methods() {
        let bounds = vec![Bound::both(-1.0, 1.0)];
        let objective = |x: &[f64]| x[0] * x[0];

        let differential = differential_evolution(
            &objective,
            &DifferentialEvolutionConfig {
                bounds: bounds.clone(),
                target_f: f64::INFINITY,
                ..DifferentialEvolutionConfig::default()
            },
        );
        let annealing = simulated_annealing(
            &objective,
            &SimulatedAnnealingConfig {
                bounds: bounds.clone(),
                target_f: f64::NAN,
                ..SimulatedAnnealingConfig::default()
            },
        );
        let random = random_search(
            &objective,
            &RandomSearchConfig {
                bounds,
                target_f: f64::INFINITY,
                ..RandomSearchConfig::default()
            },
        );

        for result in [differential, annealing, random] {
            assert!(
                matches!(result, Err(OptimError::NumericalError { iteration: 0, .. })),
                "invalid target threshold was accepted: {result:?}"
            );
        }
    }
}
