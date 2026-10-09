//! Optimization fixtures for integration tests

pub struct OptimizationProblem {
    pub dim: usize,
    pub objective: fn(&[f64]) -> f64,
    pub gradient: fn(&[f64]) -> Vec<f64>,
    pub initial_guess: Vec<f64>,
    pub global_minimum: Vec<f64>,
    pub minimum_value: f64,
}

pub fn sphere_problem(dim: usize) -> OptimizationProblem {
    let dim_copy = dim;
    OptimizationProblem {
        dim,
        objective: move |x: &[f64]| x.iter().map(|&xi| xi * xi).sum(),
        gradient: move |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect(),
        initial_guess: vec![1.0; dim_copy],
        global_minimum: vec![0.0; dim_copy],
        minimum_value: 0.0,
    }
}

pub fn rosenbrock_problem(dim: usize) -> OptimizationProblem {
    assert!(dim >= 2);
    OptimizationProblem {
        dim,
        objective: |x: &[f64]| {
            let mut sum = 0.0;
            for i in 0..x.len() - 1 {
                sum += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
            }
            sum
        },
        gradient: |x: &[f64]| {
            let mut g = vec![0.0; x.len()];
            g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
            for i in 1..x.len() - 1 {
                g[i] = -400.0 * x[i] * (x[i + 1] - x[i] * x[i])
                    + 400.0 * (x[i] - x[i - 1] * x[i - 1])
                    - 2.0 * (1.0 - x[i]);
            }
            g[x.len() - 1] = 200.0 * (x[x.len() - 1] - x[x.len() - 2] * x[x.len() - 2]);
            g
        },
        initial_guess: vec![-1.2; 2],
        global_minimum: vec![1.0; 2],
        minimum_value: 0.0,
    }
}

pub fn booth_problem() -> OptimizationProblem {
    OptimizationProblem {
        dim: 2,
        objective: |x: &[f64]| {
            (x[0] + 2.0 * x[1] - 7.0).powi(2) + (2.0 * x[0] + x[1] - 5.0).powi(2)
        },
        gradient: |x: &[f64]| {
            vec![
                10.0 * x[0] + 8.0 * x[1] - 34.0,
                8.0 * x[0] + 10.0 * x[1] - 38.0,
            ]
        },
        initial_guess: vec![1.0; 2],
        global_minimum: vec![1.0, 3.0],
        minimum_value: 0.0,
    }
}

pub fn constraint_equality(x: &[f64]) -> Vec<f64> {
    vec![x[0].powi(2) + x[1].powi(2) - 1.0]
}

pub fn constraint_inequality(x: &[f64]) -> Vec<f64> {
    vec![x[0] + x[1] - 1.0]
}

pub fn constraint_jacobian_equality(x: &[f64]) -> Vec<Vec<f64>> {
    vec![vec![2.0 * x[0], 2.0 * x[1]]]
}
