//! High-level topology optimization driver
//!
//! Ties together the density field, filter, objective, constraints, and optimizer
//! into a single optimization loop.

use super::density::{DensityField, InterpolationScheme};
use super::filter::DensityFilter;
use super::optimizer::{MmaOptimizer, OcOptimizer, TopOptConfig, TopologyOptimizerKind};

/// Result of a topology optimization run.
#[derive(Debug, Clone)]
pub struct TopOptResult {
    /// Optimized density field.
    pub densities: Vec<f64>,
    /// Objective function history.
    pub objective_history: Vec<f64>,
    /// Volume fraction history.
    pub volume_history: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether convergence was achieved.
    pub converged: bool,
}

/// Run structural topology optimization (compliance minimization).
///
/// The caller provides closures for the FEM solve and sensitivity computation:
/// - `solve_and_evaluate`: given densities, solve the FEM problem and return
///   (objective_value, element_compliances)
/// - `compute_sensitivities`: given densities and element_compliances, return
///   dC/dp for each element
pub fn run_topology_optimization<F, G>(
    mesh: &crate::mesh::Mesh,
    config: &TopOptConfig,
    element_volumes: &[f64],
    solve_and_evaluate: F,
    compute_sensitivities: G,
) -> TopOptResult
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
    G: Fn(&[f64], &[f64]) -> Vec<f64>,
{
    let n_elem = mesh.elements.len();
    let scheme = InterpolationScheme::simp_with_penalty(config.simp_penalty);
    let mut density = DensityField::uniform(n_elem, config.volume_fraction, scheme);
    let filter = DensityFilter::build(mesh, config.filter_radius);
    let optimizer = OcOptimizer::new();
    let mut mma_optimizer = MmaOptimizer::new();

    let mut obj_history = Vec::new();
    let mut vol_history = Vec::new();
    let mut converged = false;

    for iter in 0..config.max_iterations {
        // Filter densities
        let filtered_rho = filter.filter_densities(&density.densities);

        // Solve and evaluate
        let (obj_val, element_compliances) = solve_and_evaluate(&filtered_rho);
        obj_history.push(obj_val);

        let vf = density.volume_fraction(element_volumes);
        vol_history.push(vf);

        if config.verbosity >= 1 {
            log::info!("TopOpt iter {iter}: obj = {obj_val:.6e}, vol_frac = {vf:.4}");
        }

        // Compute sensitivities
        let raw_sens = compute_sensitivities(&filtered_rho, &element_compliances);
        let filtered_sens = filter.filter_sensitivities(&density.densities, &raw_sens);

        // Update densities
        let new_rho = match config.optimizer {
            TopologyOptimizerKind::Mma => mma_optimizer.update(
                &filtered_sens,
                &density.densities,
                config.volume_fraction,
                element_volumes,
            ),
            TopologyOptimizerKind::Oc => optimizer.update(
                &filtered_sens,
                &density.densities,
                config.volume_fraction,
                element_volumes,
            ),
        };

        // Check convergence
        let max_change: f64 = density
            .densities
            .iter()
            .zip(new_rho.iter())
            .map(|(&old, &new)| (old - new).abs())
            .fold(0.0, f64::max);

        density.densities = new_rho;
        density.clamp();

        if max_change < config.tolerance {
            converged = true;
            if config.verbosity >= 1 {
                log::info!("TopOpt converged at iteration {iter}: max_change = {max_change:.6e}");
            }
            break;
        }
    }

    let iterations = obj_history.len();
    TopOptResult {
        densities: density.densities,
        objective_history: obj_history,
        volume_history: vol_history,
        iterations,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_optimization_mock() {
        // Mock optimization: no real FEM, just test the iteration loop
        let mesh = crate::mesh::unit_square_triangles(3);
        let n_elem = mesh.elements.len();
        let volumes = vec![1.0 / n_elem as f64; n_elem];

        let config = TopOptConfig {
            max_iterations: 5,
            tolerance: 0.001,
            volume_fraction: 0.5,
            filter_radius: 0.5,
            simp_penalty: 3.0,
            verbosity: 0,
            ..Default::default()
        };

        let result = run_topology_optimization(
            &mesh,
            &config,
            &volumes,
            |rho| {
                // Mock: compliance = sum of 1/rho (penalized)
                let obj: f64 = rho.iter().map(|&r| 1.0 / r.max(1e-3)).sum();
                let elem_c: Vec<f64> = rho.iter().map(|&r| 1.0 / r.max(1e-3)).collect();
                (obj, elem_c)
            },
            |rho, elem_c| {
                // Mock: sensitivity = -c_e / rho_e
                rho.iter()
                    .zip(elem_c)
                    .map(|(&r, &c)| -c / r.max(1e-3))
                    .collect()
            },
        );

        assert!(result.iterations >= 1 && result.iterations <= 5);
        assert!(!result.objective_history.is_empty());
        // Densities should be in valid range
        for &rho in &result.densities {
            assert!((1e-3..=1.0).contains(&rho));
        }
    }
}
