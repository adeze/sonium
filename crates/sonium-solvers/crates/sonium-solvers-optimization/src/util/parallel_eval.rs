use rayon::prelude::*;
use std::sync::Arc;

/// Parallel evaluation configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Enable parallel evaluation
    pub enabled: bool,
    /// Requested worker count for application-owned pool construction.
    ///
    /// Solvers deliberately do not mutate Rayon's process-global pool. Callers
    /// that require an exact worker count must invoke the solver from inside a
    /// [`rayon::ThreadPool::install`] closure built with this value.
    pub num_threads: Option<usize>,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            num_threads: None, // Use rayon's default (typically num_cpus)
        }
    }
}

/// Evaluate a population in parallel
///
/// # Arguments
/// * `population` - slice of individuals, one vector per row
/// * `eval_fn` - Function to evaluate each individual
/// * `config` - Parallel configuration
///
/// # Returns
/// Array of fitness values for each individual
pub fn evaluate_population_parallel<F>(
    population: &[Vec<f64>],
    eval_fn: Arc<F>,
    config: &ParallelConfig,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    let npop = population.len();

    if !config.enabled || npop < 4 {
        // Sequential evaluation for small populations or when disabled
        let mut energies = vec![0.0; npop];
        for i in 0..npop {
            energies[i] = eval_fn(&population[i]);
        }
        return energies;
    }

    // Always use global thread pool (configured once in solver)
    (0..npop)
        .into_par_iter()
        .map(|i| eval_fn(&population[i]))
        .collect::<Vec<f64>>()
}

/// Evaluate trials in parallel for differential evolution
///
/// This function evaluates multiple trial vectors in parallel, which is useful
/// during the main DE loop where we generate and evaluate one trial per individual.
///
/// # Arguments
/// * `trials` - Vector of trial vectors to evaluate
/// * `eval_fn` - Function to evaluate each trial
/// * `config` - Parallel configuration
///
/// # Returns
/// Vector of fitness values for each trial
pub fn evaluate_trials_parallel<F>(
    trials: &[Vec<f64>],
    eval_fn: Arc<F>,
    config: &ParallelConfig,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    if !config.enabled || trials.len() < 4 {
        // Sequential evaluation for small batches or when disabled
        return trials.iter().map(|trial| eval_fn(trial)).collect();
    }

    // Always use global thread pool (configured once in solver)
    trials.par_iter().map(|trial| eval_fn(trial)).collect()
}

/// Evaluate rows of a population in parallel.
///
/// This avoids constructing a `Vec<Vec<f64>>` when the trials are already
/// stored as rows of a population matrix.
pub fn evaluate_rows_parallel<F>(
    rows: &[Vec<f64>],
    eval_fn: Arc<F>,
    config: &ParallelConfig,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    let npop = rows.len();
    if !config.enabled || npop < 4 {
        return (0..npop).map(|i| eval_fn(&rows[i])).collect();
    }

    (0..npop)
        .into_par_iter()
        .map(|i| eval_fn(&rows[i]))
        .collect()
}

/// Slice-based variant of [`evaluate_population_parallel`] used by the DE hot
/// path. The evaluator receives a contiguous row slice, avoiding a per-row
/// heap allocation (rows are already owned vectors, so this is identical to
/// [`evaluate_population_parallel`] after the `ndarray` port).
pub(crate) fn evaluate_population_parallel_slice<F>(
    population: &[Vec<f64>],
    eval_fn: Arc<F>,
    config: &ParallelConfig,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    evaluate_population_parallel(population, eval_fn, config)
}

/// Structure to batch evaluate individuals with their indices.
pub struct IndexedEvaluation {
    /// Index of the individual in the population.
    pub index: usize,
    /// The individual's solution vector.
    pub individual: Vec<f64>,
    /// The computed fitness value.
    pub fitness: f64,
}

/// Evaluate population with indices preserved for tracking
pub fn evaluate_population_indexed<F>(
    population: &[Vec<f64>],
    eval_fn: Arc<F>,
    config: &ParallelConfig,
) -> Vec<IndexedEvaluation>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    let npop = population.len();

    if !config.enabled || npop < 4 {
        // Sequential evaluation
        let mut results = Vec::with_capacity(npop);
        for (i, member) in population.iter().enumerate() {
            let individual = member.clone();
            let fitness = eval_fn(&individual);
            results.push(IndexedEvaluation {
                index: i,
                individual,
                fitness,
            });
        }
        return results;
    }

    // Parallel evaluation (global thread pool)
    (0..npop)
        .into_par_iter()
        .map(|i| {
            let individual = population[i].clone();
            let fitness = eval_fn(&individual);
            IndexedEvaluation {
                index: i,
                individual,
                fitness,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_evaluation() {
        // Simple quadratic function
        let eval_fn = Arc::new(|x: &[f64]| -> f64 { x.iter().map(|&xi| xi * xi).sum() });

        // Create a small population
        let mut population = vec![vec![0.0; 3]; 10];
        for (i, row) in population.iter_mut().enumerate() {
            for (j, entry) in row.iter_mut().enumerate() {
                *entry = (i as f64) * 0.1 + (j as f64) * 0.01;
            }
        }

        // Test with parallel enabled
        let config = ParallelConfig {
            enabled: true,
            num_threads: Some(2),
        };
        let energies = evaluate_population_parallel(&population, eval_fn.clone(), &config);

        // Verify results
        assert_eq!(energies.len(), 10);
        for (i, row) in population.iter().enumerate() {
            let expected = row.iter().map(|&x| x * x).sum::<f64>();
            assert!((energies[i] - expected).abs() < 1e-10);
        }

        // Test with parallel disabled
        let config_seq = ParallelConfig {
            enabled: false,
            num_threads: None,
        };
        let energies_seq = evaluate_population_parallel(&population, eval_fn, &config_seq);

        // Results should be identical
        for (a, b) in energies.iter().zip(energies_seq.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_indexed_evaluation() {
        let eval_fn = Arc::new(|x: &[f64]| -> f64 { x.iter().sum() });

        let mut population = vec![vec![0.0; 2]; 5];
        for (i, row) in population.iter_mut().enumerate() {
            row[0] = i as f64;
            row[1] = (i * 2) as f64;
        }

        let config = ParallelConfig::default();
        let results = evaluate_population_indexed(&population, eval_fn, &config);

        assert_eq!(results.len(), 5);
        for result in results {
            let expected: f64 = population[result.index].iter().sum();
            assert_eq!(result.fitness, expected);
        }
    }
}
