//! Core variational form traits

use crate::spaces::BasisTabulation;

/// Dense element-local matrix (row-major, `dim x dim`).
///
/// Accumulates quadrature-point contributions before scattering to global.
#[derive(Debug, Clone)]
pub struct LocalMatrix<T> {
    pub values: Vec<T>,
    pub dim: usize,
}

impl<T: Copy + Default> LocalMatrix<T> {
    /// Create a zero-initialized local matrix.
    pub fn zeros(dim: usize) -> Self {
        Self {
            values: vec![T::default(); dim * dim],
            dim,
        }
    }

    /// Read element `(i, j)`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> T {
        self.values[i * self.dim + j]
    }
}

impl<T: Copy + Default + std::ops::AddAssign> LocalMatrix<T> {
    /// Accumulate `val` into element `(i, j)`.
    #[inline]
    pub fn add(&mut self, i: usize, j: usize, val: T) {
        self.values[i * self.dim + j] += val;
    }
}

/// Dense element-local vector.
#[derive(Debug, Clone)]
pub struct LocalVector<T> {
    pub values: Vec<T>,
}

impl<T: Copy + Default> LocalVector<T> {
    /// Create a zero-initialized local vector.
    pub fn zeros(dim: usize) -> Self {
        Self {
            values: vec![T::default(); dim],
        }
    }
}

impl<T: Copy + Default + std::ops::AddAssign> LocalVector<T> {
    /// Accumulate `val` into element `i`.
    #[inline]
    pub fn add(&mut self, i: usize, val: T) {
        self.values[i] += val;
    }
}

/// A bilinear form `a(u, v)` that can be assembled into a global matrix.
///
/// The assembler calls [`integrate_point`](BilinearForm::integrate_point) once per quadrature
/// point per element, passing pre-tabulated basis functions. The implementation
/// accumulates its contribution into the provided `LocalMatrix`.
///
/// Physics-specific parameters (wavenumber, material properties) are stored
/// in the implementing struct.
///
/// # Type parameter
///
/// `T` is the scalar type of the assembled matrix entries (`f64` or `Complex64`).
pub trait BilinearForm<T: Copy + Default + std::ops::AddAssign>: Send + Sync {
    /// Accumulate the integrand at one quadrature point into the element matrix.
    ///
    /// - `test`, `trial`: pre-tabulated basis functions (same for Galerkin)
    /// - `weight`: quadrature weight (det_j is already in the tabulation's `det_j` field)
    /// - `local_matrix`: accumulator (caller initializes to zero once per element)
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<T>,
    );
}

/// A bilinear form with different test and trial spaces (mixed formulation).
///
/// This is used for mixed finite elements where the test and trial spaces differ:
/// - e.g., H1 × H(div) for mixed acoustics (pressure × flux)
/// - e.g., H(curl) × H(div) for EM formulations
///
/// The block matrix structure is `[A B; C D]` where:
/// - A: test_space × trial_space (same block)
/// - B: test_space × trial_space2 (cross block)
/// - C: test_space2 × trial_space (cross block)
/// - D: test_space2 × trial_space2 (second block)
pub trait MixedBilinearForm<T: Copy + Default + std::ops::AddAssign>: Send + Sync {
    /// Number of local DOFs in the first (test) space.
    fn test_num_dofs(&self) -> usize;

    /// Number of local DOFs in the second (trial) space (for cross blocks).
    fn trial_num_dofs(&self) -> usize;

    /// Integrate the (1,1) block: test_space × trial_space.
    fn integrate_11(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<T>,
    );

    /// Integrate the (1,2) block: test_space × trial_space2 (if applicable).
    fn integrate_12(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<T>,
    );

    /// Integrate the (2,1) block: test_space2 × trial_space (if applicable).
    fn integrate_21(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<T>,
    );

    /// Integrate the (2,2) block: test_space2 × trial_space2.
    fn integrate_22(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<T>,
    );
}

/// Assert a caller-supplied closure vector carries at least `dim` components.
///
/// Form integrands run inside hot element loops with no `Result` channel, so a
/// short closure return must fail loudly here instead of silently truncating
/// the physics (wrong matrix, no error).
pub(crate) fn require_closure_components(field: &str, values: &[f64], dim: usize) {
    assert!(
        values.len() >= dim,
        "{field} closure must return at least {dim} components for a {dim}D mesh (got {})",
        values.len()
    );
}

/// A linear form `l(v)` that can be assembled into a global vector.
///
/// Analogous to [`BilinearForm`] but produces a vector (RHS) rather than a matrix.
pub trait LinearForm<T: Copy + Default + std::ops::AddAssign>: Send + Sync {
    /// Accumulate the integrand at one quadrature point into the element vector.
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<T>,
    );
}

/// A bilinear form with interior facet contributions for DG methods.
///
/// Interior penalty, DG, and other discontinuous methods require integrating
/// jump and average operators over interior facets (shared edges/faces between
/// elements). This trait extends [`BilinearForm`] with a facet integration
/// method.
///
/// The assembler calls [`integrate_facet`](FacetBilinearForm::integrate_facet)
/// once per quadrature point on each interior facet, passing basis tabulations
/// from both adjacent elements ("plus" and "minus" sides).
pub trait FacetBilinearForm<T: Copy + Default + std::ops::AddAssign>: Send + Sync {
    /// Accumulate the facet integrand at one quadrature point.
    ///
    /// Called for each interior facet between `elem_plus` and `elem_minus`.
    ///
    /// - `test_plus`, `trial_plus`: basis on the "plus" element side
    /// - `test_minus`, `trial_minus`: basis on the "minus" element side
    /// - `normal`: unit outward normal from plus toward minus
    /// - `h`: facet mesh size (for penalty parameter scaling)
    /// - `weight`: quadrature weight × facet area
    /// - `block`: 2×2 block matrix accumulator:
    ///   - `block[0]`: (plus, plus) contributions
    ///   - `block[1]`: (plus, minus) contributions
    ///   - `block[2]`: (minus, plus) contributions
    ///   - `block[3]`: (minus, minus) contributions
    #[allow(clippy::too_many_arguments)]
    fn integrate_facet(
        &self,
        test_plus: &BasisTabulation,
        trial_plus: &BasisTabulation,
        test_minus: &BasisTabulation,
        trial_minus: &BasisTabulation,
        normal: [f64; 3],
        h: f64,
        weight: f64,
        block: &mut [LocalMatrix<T>; 4],
    );
}

/// An element-level form that produces full element matrices directly.
///
/// Bridges element-based formulations (beam, shell, DKT) with the generic
/// assembly pipeline. Unlike [`BilinearForm`] which operates per-quadrature-point,
/// `ElementForm` computes the entire element matrix in one call, allowing
/// analytical or specialised integration strategies.
pub trait ElementForm<T: Copy + Default + std::ops::AddAssign>: Send + Sync {
    /// Total number of elements.
    fn num_elements(&self) -> usize;

    /// Number of local DOFs for element `elem_idx`.
    fn num_local_dofs(&self, elem_idx: usize) -> usize;

    /// Map local DOF indices to global DOF indices for element `elem_idx`.
    fn local_to_global(&self, elem_idx: usize) -> Vec<usize>;

    /// Compute the full element matrix (dense, row-major, `n_local x n_local`).
    fn element_matrix(&self, elem_idx: usize) -> Vec<T>;
}

/// Nonlinear element-level form for Newton-Raphson iteration.
///
/// Same as [`ElementForm`] but receives the current global solution and
/// produces both the element residual vector and tangent stiffness matrix.
/// Used for geometrically nonlinear beam/shell formulations (corotational).
pub trait NonlinearElementForm<T: Copy + Default + std::ops::AddAssign>: Send + Sync {
    /// Total number of elements.
    fn num_elements(&self) -> usize;

    /// Number of local DOFs for element `elem_idx`.
    fn num_local_dofs(&self, elem_idx: usize) -> usize;

    /// Map local DOF indices to global DOF indices for element `elem_idx`.
    fn local_to_global(&self, elem_idx: usize) -> Vec<usize>;

    /// Total number of global DOFs.
    fn num_global_dofs(&self) -> usize;

    /// Compute the element residual vector given the current global solution.
    fn element_residual(&self, elem_idx: usize, global_solution: &[T]) -> Vec<T>;

    /// Compute the element tangent stiffness matrix (dense, row-major) given
    /// the current global solution.
    fn element_tangent(&self, elem_idx: usize, global_solution: &[T]) -> Vec<T>;
}

/// A nonlinear form for Newton-Raphson iteration.
///
/// Provides both the residual `R(u)` and tangent stiffness `dR/du`
/// evaluated at the current solution.
pub trait NonlinearForm<T: Copy + Default + std::ops::AddAssign>: Send + Sync {
    /// Accumulate the element residual contribution at one quadrature point.
    ///
    /// `local_solution` contains the current solution values at element DOFs.
    fn element_residual(
        &self,
        test: &BasisTabulation,
        local_solution: &[T],
        weight: f64,
        local_vector: &mut LocalVector<T>,
    );

    /// Accumulate the element tangent stiffness contribution at one quadrature point.
    fn element_tangent(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        local_solution: &[T],
        weight: f64,
        local_matrix: &mut LocalMatrix<T>,
    );
}
