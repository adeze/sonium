//! Geometric multigrid solver for finite element problems
//!
//! Provides V-cycle, W-cycle, and F-cycle multigrid methods with
//! Gauss-Seidel smoothing and linear interpolation transfer operators.
//!
//! The multigrid module is physics-agnostic: it operates on any sparse
//! matrix presented via the [`CooOperator`] trait.

use num_complex::Complex64;

mod cycle;
mod hierarchy;
mod smoother;
mod transfer;

pub use cycle::*;
pub use hierarchy::*;
pub use smoother::*;
pub use transfer::*;

/// Minimal COO (coordinate) sparse matrix interface for multigrid operations.
///
/// Implemented by any type that stores a sparse matrix in coordinate format
/// (rows, cols, values, dimension). This decouples the multigrid module
/// from specific matrix types (`HelmholtzMatrix`, `TripletMatrix`, etc.).
pub trait CooOperator {
    fn dim(&self) -> usize;
    fn rows(&self) -> &[usize];
    fn cols(&self) -> &[usize];
    fn values(&self) -> &[Complex64];
}

// Implement for HelmholtzMatrix (legacy)
impl CooOperator for crate::assembly::HelmholtzMatrix {
    fn dim(&self) -> usize {
        self.dim
    }
    fn rows(&self) -> &[usize] {
        &self.rows
    }
    fn cols(&self) -> &[usize] {
        &self.cols
    }
    fn values(&self) -> &[Complex64] {
        &self.values
    }
}

// Implement for TripletMatrix<Complex64> (generic)
impl CooOperator for crate::assembly::TripletMatrix<Complex64> {
    fn dim(&self) -> usize {
        self.num_rows
    }
    fn rows(&self) -> &[usize] {
        &self.rows
    }
    fn cols(&self) -> &[usize] {
        &self.cols
    }
    fn values(&self) -> &[Complex64] {
        &self.values
    }
}
