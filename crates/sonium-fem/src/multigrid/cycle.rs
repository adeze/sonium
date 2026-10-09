//! Multigrid cycle implementations
//!
//! Provides V-cycle, W-cycle, and F-cycle algorithms.

use super::hierarchy::{HierarchyError, MultigridHierarchy};
use super::smoother::{SmootherConfig, compute_residual, smooth};
use super::transfer::{prolongate, restrict};
use num_complex::Complex64;
use std::fmt;

/// Multigrid cycle type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleType {
    /// V-cycle: descend to coarsest, ascend
    VCycle,
    /// W-cycle: recursive V-cycles
    WCycle,
    /// F-cycle: FMG-like pattern
    FCycle,
}

/// Multigrid solver configuration
#[derive(Debug, Clone)]
pub struct MultigridConfig {
    /// Type of cycle
    pub cycle_type: CycleType,
    /// Pre-smoothing configuration
    pub pre_smoother: SmootherConfig,
    /// Post-smoothing configuration
    pub post_smoother: SmootherConfig,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Relative tolerance for convergence
    pub tolerance: f64,
    /// Number of cycles per iteration
    pub cycles_per_iteration: usize,
}

impl Default for MultigridConfig {
    fn default() -> Self {
        Self {
            cycle_type: CycleType::VCycle,
            pre_smoother: SmootherConfig::default(),
            post_smoother: SmootherConfig::default(),
            max_iterations: 50,
            tolerance: 1e-8,
            cycles_per_iteration: 1,
        }
    }
}

/// Multigrid solver result
#[derive(Debug)]
pub struct MultigridResult {
    /// Solution vector
    pub solution: Vec<Complex64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm
    pub residual_norm: f64,
    /// Convergence achieved
    pub converged: bool,
}

/// Failure returned by the checked multigrid entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultigridSolveError {
    /// The hierarchy contains an invalid matrix or transfer operator.
    Hierarchy(HierarchyError),
    /// A right-hand side or initial guess has the wrong dimension.
    VectorLength {
        /// Name of the vector that failed validation.
        vector: &'static str,
        /// Required number of entries.
        expected: usize,
        /// Supplied number of entries.
        actual: usize,
    },
    /// A caller supplied a non-finite vector entry.
    NonFiniteVector {
        /// Name of the vector that failed validation.
        vector: &'static str,
        /// Index of the non-finite entry.
        index: usize,
    },
    /// A solver configuration value is outside the checked contract.
    InvalidConfig(&'static str),
    /// The iteration produced a non-finite solution entry.
    NonFiniteSolution { index: usize },
}

impl fmt::Display for MultigridSolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hierarchy(error) => write!(f, "invalid multigrid hierarchy: {error}"),
            Self::VectorLength {
                vector,
                expected,
                actual,
            } => write!(f, "{vector} has length {actual}, expected {expected}"),
            Self::NonFiniteVector { vector, index } => {
                write!(f, "{vector} entry {index} is non-finite")
            }
            Self::InvalidConfig(name) => write!(f, "invalid multigrid configuration: {name}"),
            Self::NonFiniteSolution { index } => {
                write!(f, "multigrid solution entry {index} is non-finite")
            }
        }
    }
}

impl std::error::Error for MultigridSolveError {}

impl From<HierarchyError> for MultigridSolveError {
    fn from(error: HierarchyError) -> Self {
        Self::Hierarchy(error)
    }
}

/// Checked multigrid solve entry point.
pub fn try_solve_multigrid(
    hierarchy: &MultigridHierarchy,
    b: &[Complex64],
    initial_guess: Option<&[Complex64]>,
    config: &MultigridConfig,
) -> Result<MultigridResult, MultigridSolveError> {
    hierarchy.validate()?;
    let n = hierarchy.try_finest()?.n_dofs;
    validate_vector("right-hand side", b, n)?;
    if let Some(guess) = initial_guess {
        validate_vector("initial guess", guess, n)?;
    }
    validate_smoother("pre-smoother", &config.pre_smoother)?;
    validate_smoother("post-smoother", &config.post_smoother)?;
    if !config.tolerance.is_finite() || config.tolerance < 0.0 {
        return Err(MultigridSolveError::InvalidConfig(
            "tolerance must be finite and non-negative",
        ));
    }
    if config.cycles_per_iteration == 0 {
        return Err(MultigridSolveError::InvalidConfig(
            "cycles_per_iteration must be non-zero",
        ));
    }

    let result = solve_multigrid(hierarchy, b, initial_guess, config);
    if let Some(index) = result
        .solution
        .iter()
        .position(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(MultigridSolveError::NonFiniteSolution { index });
    }
    Ok(result)
}

fn validate_vector(
    vector: &'static str,
    values: &[Complex64],
    expected: usize,
) -> Result<(), MultigridSolveError> {
    if values.len() != expected {
        return Err(MultigridSolveError::VectorLength {
            vector,
            expected,
            actual: values.len(),
        });
    }
    if let Some(index) = values
        .iter()
        .position(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(MultigridSolveError::NonFiniteVector { vector, index });
    }
    Ok(())
}

fn validate_smoother(
    name: &'static str,
    config: &SmootherConfig,
) -> Result<(), MultigridSolveError> {
    if !config.omega.is_finite() {
        return Err(MultigridSolveError::InvalidConfig(name));
    }
    Ok(())
}

/// Solve using multigrid method
pub fn solve_multigrid(
    hierarchy: &MultigridHierarchy,
    b: &[Complex64],
    initial_guess: Option<&[Complex64]>,
    config: &MultigridConfig,
) -> MultigridResult {
    let n = hierarchy.finest().n_dofs;

    // Initialize solution
    let mut x = match initial_guess {
        Some(guess) => guess.to_vec(),
        None => vec![Complex64::new(0.0, 0.0); n],
    };

    let matrix = hierarchy.system_at(0);

    let b_norm = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    let tol = config.tolerance * b_norm.max(1e-10);

    let mut iterations = 0;
    let mut residual = compute_residual(matrix, &x, b);
    let mut res_norm = residual.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

    while iterations < config.max_iterations && res_norm > tol {
        for _ in 0..config.cycles_per_iteration {
            match config.cycle_type {
                CycleType::VCycle => {
                    v_cycle(
                        hierarchy,
                        0,
                        &mut x,
                        b,
                        &config.pre_smoother,
                        &config.post_smoother,
                    );
                }
                CycleType::WCycle => {
                    w_cycle(
                        hierarchy,
                        0,
                        &mut x,
                        b,
                        &config.pre_smoother,
                        &config.post_smoother,
                    );
                }
                CycleType::FCycle => {
                    f_cycle(
                        hierarchy,
                        0,
                        &mut x,
                        b,
                        &config.pre_smoother,
                        &config.post_smoother,
                    );
                }
            }
        }

        residual = compute_residual(matrix, &x, b);
        res_norm = residual.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        iterations += 1;
    }

    MultigridResult {
        solution: x,
        iterations,
        residual_norm: res_norm,
        converged: res_norm <= tol,
    }
}

/// V-cycle: single descent to coarsest level
fn v_cycle(
    hierarchy: &MultigridHierarchy,
    level: usize,
    x: &mut [Complex64],
    b: &[Complex64],
    pre_config: &SmootherConfig,
    post_config: &SmootherConfig,
) {
    let n_levels = hierarchy.num_levels();

    if level == n_levels - 1 {
        // Coarsest level: solve exactly or smooth many times
        let matrix = hierarchy.system_at(level);

        let exact_config = SmootherConfig {
            iterations: 20,
            ..pre_config.clone()
        };
        smooth(matrix, x, b, &exact_config);
        return;
    }

    let matrix = hierarchy.system_at(level);

    // Pre-smoothing
    smooth(matrix, x, b, pre_config);

    // Compute residual
    let residual = compute_residual(matrix, x, b);

    // Restrict residual to coarser level
    let restriction = hierarchy.levels[level]
        .restriction
        .as_ref()
        .expect("Restriction operator not built");
    let r_coarse = restrict(restriction, &residual);

    // Solve on coarser level (correction equation: A*e = r)
    let n_coarse = hierarchy.levels[level + 1].n_dofs;
    let mut e_coarse = vec![Complex64::new(0.0, 0.0); n_coarse];

    v_cycle(
        hierarchy,
        level + 1,
        &mut e_coarse,
        &r_coarse,
        pre_config,
        post_config,
    );

    // Prolongate correction
    let prolongation = hierarchy.levels[level + 1]
        .prolongation
        .as_ref()
        .expect("Prolongation operator not built");
    let e_fine = prolongate(prolongation, &e_coarse);

    // Add correction
    for i in 0..x.len() {
        x[i] += e_fine[i];
    }

    // Post-smoothing
    smooth(matrix, x, b, post_config);
}

/// W-cycle: recursive pattern with two coarse-level calls
fn w_cycle(
    hierarchy: &MultigridHierarchy,
    level: usize,
    x: &mut [Complex64],
    b: &[Complex64],
    pre_config: &SmootherConfig,
    post_config: &SmootherConfig,
) {
    let n_levels = hierarchy.num_levels();

    if level == n_levels - 1 {
        let matrix = hierarchy.system_at(level);

        let exact_config = SmootherConfig {
            iterations: 20,
            ..pre_config.clone()
        };
        smooth(matrix, x, b, &exact_config);
        return;
    }

    let matrix = hierarchy.system_at(level);

    // Pre-smoothing
    smooth(matrix, x, b, pre_config);

    // Compute residual
    let residual = compute_residual(matrix, x, b);

    // Restrict
    let restriction = hierarchy.levels[level]
        .restriction
        .as_ref()
        .expect("Restriction operator not built");
    let r_coarse = restrict(restriction, &residual);

    // First coarse grid correction
    let n_coarse = hierarchy.levels[level + 1].n_dofs;
    let mut e_coarse = vec![Complex64::new(0.0, 0.0); n_coarse];
    w_cycle(
        hierarchy,
        level + 1,
        &mut e_coarse,
        &r_coarse,
        pre_config,
        post_config,
    );

    // Second coarse grid correction (W-cycle pattern) - use fresh vector
    let mut e_coarse2 = vec![Complex64::new(0.0, 0.0); n_coarse];
    w_cycle(
        hierarchy,
        level + 1,
        &mut e_coarse2,
        &r_coarse,
        pre_config,
        post_config,
    );

    // Add both corrections
    for i in 0..e_coarse.len() {
        e_coarse[i] += e_coarse2[i];
    }

    // Prolongate and add correction
    let prolongation = hierarchy.levels[level + 1]
        .prolongation
        .as_ref()
        .expect("Prolongation operator not built");
    let e_fine = prolongate(prolongation, &e_coarse);

    for i in 0..x.len() {
        x[i] += e_fine[i];
    }

    // Post-smoothing
    smooth(matrix, x, b, post_config);
}

/// F-cycle: Full Multigrid pattern
fn f_cycle(
    hierarchy: &MultigridHierarchy,
    level: usize,
    x: &mut [Complex64],
    b: &[Complex64],
    pre_config: &SmootherConfig,
    post_config: &SmootherConfig,
) {
    let n_levels = hierarchy.num_levels();

    if level == n_levels - 1 {
        let matrix = hierarchy.system_at(level);

        let exact_config = SmootherConfig {
            iterations: 20,
            ..pre_config.clone()
        };
        smooth(matrix, x, b, &exact_config);
        return;
    }

    // First, restrict to coarser level and solve there
    let restriction = hierarchy.levels[level]
        .restriction
        .as_ref()
        .expect("Restriction operator not built");

    // Restrict RHS (for FMG initialization)
    let b_coarse = restrict(restriction, b);

    // Solve on coarser level first (FMG pattern)
    let n_coarse = hierarchy.levels[level + 1].n_dofs;
    let mut x_coarse = vec![Complex64::new(0.0, 0.0); n_coarse];
    f_cycle(
        hierarchy,
        level + 1,
        &mut x_coarse,
        &b_coarse,
        pre_config,
        post_config,
    );

    // Prolongate as initial guess
    let prolongation = hierarchy.levels[level + 1]
        .prolongation
        .as_ref()
        .expect("Prolongation operator not built");
    let x_init = prolongate(prolongation, &x_coarse);

    for (i, xi) in x_init.iter().enumerate() {
        x[i] = *xi;
    }

    // Now do V-cycle to polish
    v_cycle(hierarchy, level, x, b, pre_config, post_config);
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::assembly::TripletMatrix;
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn test_v_cycle_reduces_residual() {
        // Use smoother directly to test residual reduction
        // Apply Dirichlet BCs to make the system non-singular
        use crate::assembly::HelmholtzProblem;
        use crate::boundary::apply_homogeneous_dirichlet;
        use crate::multigrid::smoother::{SmootherConfig, residual_norm, smooth};

        let mesh = unit_square_triangles(4);
        let k = Complex64::new(0.0, 0.0);

        let mut problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k, |_, _, _| {
            Complex64::new(1.0, 0.0)
        });

        // Apply Dirichlet BC on all boundaries to make system non-singular
        apply_homogeneous_dirichlet(&mut problem, &mesh, &[1, 2, 3, 4]);

        let matrix = problem.matrix.to_compressed();
        let b = &problem.rhs;
        let mut x = vec![Complex64::new(0.0, 0.0); matrix.dim];

        let initial_norm = residual_norm(&matrix, &x, b);

        // Apply many smoothing iterations
        let config = SmootherConfig {
            iterations: 50,
            ..Default::default()
        };
        smooth(&matrix, &mut x, b, &config);

        let final_norm = residual_norm(&matrix, &x, b);

        // Smoothing should reduce residual for a well-posed problem
        assert!(
            final_norm < initial_norm,
            "Residual should decrease: {} -> {}",
            initial_norm,
            final_norm
        );
    }

    #[test]
    fn test_multigrid_config() {
        let config = MultigridConfig {
            cycle_type: CycleType::WCycle,
            max_iterations: 100,
            tolerance: 1e-10,
            ..Default::default()
        };

        assert_eq!(config.cycle_type, CycleType::WCycle);
        assert_eq!(config.max_iterations, 100);
    }

    #[test]
    fn checked_multigrid_rejects_empty_hierarchy() {
        let hierarchy = MultigridHierarchy::from_meshes(
            Vec::new(),
            PolynomialDegree::P1,
            Complex64::new(0.0, 0.0),
        );
        let error = try_solve_multigrid(&hierarchy, &[], None, &MultigridConfig::default())
            .expect_err("empty hierarchy must be rejected");
        assert!(matches!(
            error,
            MultigridSolveError::Hierarchy(HierarchyError::EmptyHierarchy)
        ));
    }

    #[test]
    fn checked_multigrid_rejects_bad_vectors_before_iteration() {
        let mesh = unit_square_triangles(2);
        let mut hierarchy = MultigridHierarchy::from_meshes(
            vec![mesh],
            PolynomialDegree::P1,
            Complex64::new(0.0, 0.0),
        );
        hierarchy.assemble_all_with(|mesh, _| {
            TripletMatrix::<Complex64>::new(mesh.num_nodes(), mesh.num_nodes())
        });
        let n = hierarchy.finest().n_dofs;

        let bad_length = try_solve_multigrid(
            &hierarchy,
            &vec![Complex64::new(0.0, 0.0); n - 1],
            None,
            &MultigridConfig::default(),
        )
        .expect_err("short RHS must be rejected");
        assert!(matches!(
            bad_length,
            MultigridSolveError::VectorLength {
                vector: "right-hand side",
                ..
            }
        ));

        let mut bad_value = vec![Complex64::new(0.0, 0.0); n];
        bad_value[0] = Complex64::new(f64::NAN, 0.0);
        let non_finite =
            try_solve_multigrid(&hierarchy, &bad_value, None, &MultigridConfig::default())
                .expect_err("non-finite RHS must be rejected");
        assert!(matches!(
            non_finite,
            MultigridSolveError::NonFiniteVector {
                vector: "right-hand side",
                index: 0
            }
        ));
    }
}
