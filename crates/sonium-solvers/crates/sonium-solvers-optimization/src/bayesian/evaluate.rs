use super::bayes_opt_config::BayesOptConfig;
use super::misc::normalize;
use rayon::prelude::*;

pub(super) fn evaluate_and_store<F>(
    f: &F,
    candidates: &mut [Vec<f64>],
    config: &BayesOptConfig,
    xs_norm: &mut Vec<Vec<f64>>,
    ys: &mut Vec<f64>,
    best_x: &mut Vec<f64>,
    best_y: &mut f64,
) where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let values = if config.parallel.enabled && candidates.len() >= 4 {
        candidates.par_iter().map(|x| f(x)).collect::<Vec<_>>()
    } else {
        candidates.iter().map(|x| f(x)).collect::<Vec<_>>()
    };

    for (x, y) in candidates.iter().zip(values) {
        xs_norm.push(normalize(x, &config.bounds));
        ys.push(y);
        if y < *best_y {
            *best_y = y;
            *best_x = x.to_vec();
        }
    }
}

pub(super) fn evaluate_multi_and_store<F>(
    f: &F,
    candidates: &mut [Vec<f64>],
    config: &BayesOptConfig,
    xs_norm: &mut Vec<Vec<f64>>,
    values: &mut Vec<Vec<f64>>,
) where
    F: Fn(&[f64]) -> Vec<f64> + Sync,
{
    let ys = if config.parallel.enabled && candidates.len() >= 4 {
        candidates.par_iter().map(|x| f(x)).collect::<Vec<_>>()
    } else {
        candidates.iter().map(|x| f(x)).collect::<Vec<_>>()
    };

    for (x, y) in candidates.iter().zip(ys) {
        xs_norm.push(normalize(x, &config.bounds));
        values.push(y);
    }
}
