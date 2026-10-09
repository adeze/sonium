use crate::{DEConfig, DEError, DEReport, DifferentialEvolution, Result};

/// Runs Differential Evolution optimization on a function.
///
/// This is a convenience function that mirrors SciPy's `differential_evolution` API.
/// It creates a DE optimizer with the given bounds and configuration, then runs
/// the optimization to find the global minimum.
///
/// # Arguments
///
/// * `func` - The objective function to minimize, mapping `&[f64]` to `f64`
/// * `bounds` - Vector of (lower, upper) bound pairs for each dimension
/// * `config` - DE configuration (use `DEConfigBuilder` to construct)
///
/// # Returns
///
/// Returns `Ok(DEReport)` containing the optimization result on success.
///
/// # Errors
///
/// Returns `DEError::InvalidBounds` if any bound pair has upper < lower.
///
/// # Example
///
/// ```rust
/// use sonium_solvers_optimization::unconstrained::de::{differential_evolution, DEConfigBuilder};
///
/// let config = DEConfigBuilder::new()
///     .maxiter(50)
///     .seed(42)
///     .build()
///     .expect("invalid config");
///
/// let result = differential_evolution(
///     &|x| x[0].powi(2) + x[1].powi(2),
///     &[(-5.0, 5.0), (-5.0, 5.0)],
///     config,
/// ).expect("optimization failed");
///
/// assert!(result.fun < 0.01);
/// ```
pub fn differential_evolution<F>(
    func: &F,
    bounds: &[(f64, f64)],
    config: DEConfig,
) -> Result<DEReport>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let n = bounds.len();
    let mut lower = vec![0.0; n];
    let mut upper = vec![0.0; n];
    for (i, (lo, hi)) in bounds.iter().enumerate() {
        lower[i] = *lo;
        upper[i] = *hi;
        if hi < lo {
            return Err(DEError::InvalidBounds {
                index: i,
                lower: *lo,
                upper: *hi,
            });
        }
    }
    let mut de = DifferentialEvolution::new(func, lower, upper)?;
    *de.config_mut() = config;
    Ok(de.solve())
}
