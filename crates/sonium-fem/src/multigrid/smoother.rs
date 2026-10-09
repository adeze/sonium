//! Smoothers for multigrid methods
//!
//! Provides Gauss-Seidel and Jacobi smoothers for reducing high-frequency error.
//! Physics-agnostic: operates on any [`CooOperator`](super::CooOperator).

use super::CooOperator;
use num_complex::Complex64;
use std::collections::HashMap;

/// Pre-built CSR-like structure for efficient row access during smoothing.
/// Built once from a CooOperator and reused across all sweep iterations.
struct CsrStructure {
    row_data: HashMap<usize, Vec<(usize, Complex64)>>,
    diagonals: Vec<Complex64>,
}

impl CsrStructure {
    fn from_coo(matrix: &dyn CooOperator) -> Self {
        let n = matrix.dim();
        let mut row_data: HashMap<usize, Vec<(usize, Complex64)>> = HashMap::new();
        let mut diagonals: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];

        let rows = matrix.rows();
        let cols = matrix.cols();
        let values = matrix.values();

        for k in 0..rows.len() {
            let row = rows[k];
            let col = cols[k];
            let val = values[k];

            if row == col {
                diagonals[row] += val;
            } else {
                row_data.entry(row).or_default().push((col, val));
            }
        }

        Self {
            row_data,
            diagonals,
        }
    }
}

/// Smoother type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmootherType {
    /// Gauss-Seidel (sequential)
    GaussSeidel,
    /// Damped Jacobi
    Jacobi,
    /// Symmetric Gauss-Seidel (forward then backward)
    SymmetricGaussSeidel,
}

/// Smoother configuration
#[derive(Debug, Clone)]
pub struct SmootherConfig {
    /// Type of smoother
    pub smoother_type: SmootherType,
    /// Number of smoothing iterations
    pub iterations: usize,
    /// Damping factor (for Jacobi)
    pub omega: f64,
}

impl Default for SmootherConfig {
    fn default() -> Self {
        Self {
            smoother_type: SmootherType::GaussSeidel,
            iterations: 2,
            omega: 2.0 / 3.0, // Optimal for Laplacian
        }
    }
}

/// Apply smoothing iterations to reduce high-frequency error
///
/// Solves A*x = b approximately, updating x in place.
/// Works with any matrix type implementing [`CooOperator`].
pub fn smooth(
    matrix: &dyn CooOperator,
    x: &mut [Complex64],
    b: &[Complex64],
    config: &SmootherConfig,
) {
    let csr = CsrStructure::from_coo(matrix);

    match config.smoother_type {
        SmootherType::GaussSeidel => {
            for _ in 0..config.iterations {
                gauss_seidel_sweep(&csr, x, b, false);
            }
        }
        SmootherType::Jacobi => {
            for _ in 0..config.iterations {
                jacobi_sweep(&csr, x, b, config.omega);
            }
        }
        SmootherType::SymmetricGaussSeidel => {
            for _ in 0..config.iterations {
                gauss_seidel_sweep(&csr, x, b, false);
                gauss_seidel_sweep(&csr, x, b, true);
            }
        }
    }
}

/// Single Gauss-Seidel sweep using pre-built CSR structure
fn gauss_seidel_sweep(csr: &CsrStructure, x: &mut [Complex64], b: &[Complex64], backward: bool) {
    let n = csr.diagonals.len();

    // Sweep order
    let indices: Box<dyn Iterator<Item = usize>> = if backward {
        Box::new((0..n).rev())
    } else {
        Box::new(0..n)
    };

    for i in indices {
        if csr.diagonals[i].norm() < 1e-15 {
            continue; // Skip zero diagonal
        }

        let mut sigma = Complex64::new(0.0, 0.0);
        if let Some(row_entries) = csr.row_data.get(&i) {
            for &(j, val) in row_entries {
                sigma += val * x[j];
            }
        }

        x[i] = (b[i] - sigma) / csr.diagonals[i];
    }
}

/// Single Jacobi sweep with damping using pre-built CSR structure
fn jacobi_sweep(csr: &CsrStructure, x: &mut [Complex64], b: &[Complex64], omega: f64) {
    let n = csr.diagonals.len();
    let mut x_new = vec![Complex64::new(0.0, 0.0); n];

    for i in 0..n {
        if csr.diagonals[i].norm() < 1e-15 {
            x_new[i] = x[i];
            continue;
        }

        let mut sigma = Complex64::new(0.0, 0.0);
        if let Some(row_entries) = csr.row_data.get(&i) {
            for &(j, val) in row_entries {
                sigma += val * x[j];
            }
        }

        let x_gs = (b[i] - sigma) / csr.diagonals[i];
        x_new[i] = Complex64::new(omega, 0.0) * x_gs + Complex64::new(1.0 - omega, 0.0) * x[i];
    }

    x.copy_from_slice(&x_new);
}

/// Compute residual r = b - A*x
pub fn compute_residual(
    matrix: &dyn CooOperator,
    x: &[Complex64],
    b: &[Complex64],
) -> Vec<Complex64> {
    let mut r = b.to_vec();

    let rows = matrix.rows();
    let cols = matrix.cols();
    let values = matrix.values();

    for k in 0..rows.len() {
        r[rows[k]] -= values[k] * x[cols[k]];
    }

    r
}

/// Compute residual norm
pub fn residual_norm(matrix: &dyn CooOperator, x: &[Complex64], b: &[Complex64]) -> f64 {
    let r = compute_residual(matrix, x, b);
    r.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::HelmholtzProblem;
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn test_gauss_seidel_reduces_residual() {
        let mesh = unit_square_triangles(4);
        let k = Complex64::new(0.0, 0.0); // Laplacian

        let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k, |x, y, _z| {
            Complex64::new((x * y).sin(), 0.0)
        });

        let matrix = problem.matrix.to_compressed();
        let b = &problem.rhs;

        let mut x = vec![Complex64::new(0.0, 0.0); matrix.dim];
        let initial_norm = residual_norm(&matrix, &x, b);

        let config = SmootherConfig::default();
        smooth(&matrix, &mut x, b, &config);

        let final_norm = residual_norm(&matrix, &x, b);
        assert!(final_norm < initial_norm, "Residual should decrease");
    }

    #[test]
    fn test_jacobi_with_damping() {
        let mesh = unit_square_triangles(4);
        let k = Complex64::new(0.0, 0.0);

        let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k, |_, _, _| {
            Complex64::new(1.0, 0.0)
        });

        let matrix = problem.matrix.to_compressed();
        let b = &problem.rhs;

        let mut x = vec![Complex64::new(0.0, 0.0); matrix.dim];
        let initial_norm = residual_norm(&matrix, &x, b);

        let config = SmootherConfig {
            smoother_type: SmootherType::Jacobi,
            iterations: 5,
            omega: 0.6,
        };
        smooth(&matrix, &mut x, b, &config);

        let final_norm = residual_norm(&matrix, &x, b);
        assert!(final_norm < initial_norm, "Jacobi should reduce residual");
    }

    /// Verify smoother works with TripletMatrix<Complex64> (not just HelmholtzMatrix)
    #[test]
    fn test_smoother_with_triplet_matrix() {
        use crate::assembly::TripletMatrix;

        // Simple 3x3 SPD system
        let mut tri = TripletMatrix::<Complex64>::new(3, 3);
        tri.push(0, 0, Complex64::new(2.0, 0.0));
        tri.push(0, 1, Complex64::new(-1.0, 0.0));
        tri.push(1, 0, Complex64::new(-1.0, 0.0));
        tri.push(1, 1, Complex64::new(2.0, 0.0));
        tri.push(1, 2, Complex64::new(-1.0, 0.0));
        tri.push(2, 1, Complex64::new(-1.0, 0.0));
        tri.push(2, 2, Complex64::new(2.0, 0.0));

        let b = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let mut x = vec![Complex64::new(0.0, 0.0); 3];
        let initial_norm = residual_norm(&tri, &x, &b);

        let config = SmootherConfig::default();
        smooth(&tri, &mut x, &b, &config);

        let final_norm = residual_norm(&tri, &x, &b);
        assert!(
            final_norm < initial_norm,
            "Smoother should work with TripletMatrix"
        );
    }
}
