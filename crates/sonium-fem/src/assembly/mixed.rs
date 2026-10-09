//! Mixed bilinear form assembly for saddle-point systems
//!
//! Assembles 2×2 block systems arising from mixed formulations:
//!
//! ```text
//! | A   B | | σ |   | f |
//! | B^T 0 | | u | = | g |
//! ```
//!
//! where σ ∈ V (e.g., H(div)) and u ∈ Q (e.g., L² / DG).

use crate::forms::{BilinearForm, LinearForm, LocalMatrix};
use crate::mesh::Mesh;
use crate::quadrature::QuadratureRule;
use crate::spaces::{FunctionSpace, SpaceValidationError};
use sonium_compute::BlockLayout;
use sonium_solvers::ComplexField;
use sonium_solvers::sparse::BlockOperator;

use super::triplet::TripletMatrix;

/// Assemble a bilinear form with two different function spaces (test ≠ trial).
///
/// This is needed for coupling blocks in mixed formulations.
pub(crate) fn assemble_bilinear_two_spaces<T, F, S0, S1>(
    form: &F,
    test_space: &S0,
    trial_space: &S1,
    mesh: &Mesh,
    quad_order: usize,
) -> TripletMatrix<T>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F: BilinearForm<T>,
    S0: FunctionSpace,
    S1: FunctionSpace,
{
    let n_test = test_space.num_global_dofs(mesh);
    let n_trial = trial_space.num_global_dofs(mesh);
    let mut global = TripletMatrix::new(n_test, n_trial);

    for elem_idx in 0..mesh.num_elements() {
        let elem = &mesh.elements[elem_idx];
        let test_dofs = test_space.local_to_global(mesh, elem_idx);
        let trial_dofs = trial_space.local_to_global(mesh, elem_idx);
        let n_t = test_dofs.len();
        let n_r = trial_dofs.len();

        let quad = QuadratureRule::new(elem.element_type, quad_order);
        let mut local_mat = LocalMatrix::zeros(n_t.max(n_r));

        for qp in quad.iter() {
            let test_tab = test_space.tabulate(mesh, elem_idx, qp.xi(), qp.eta(), qp.zeta());
            let trial_tab = trial_space.tabulate(mesh, elem_idx, qp.xi(), qp.eta(), qp.zeta());
            form.integrate_point(&test_tab, &trial_tab, qp.weight, &mut local_mat);
        }

        for (i, &gi) in test_dofs.iter().enumerate() {
            for (j, &gj) in trial_dofs.iter().enumerate() {
                let val = local_mat.get(i, j);
                global.push(gi, gj, val);
            }
        }
    }
    global
}

/// Checked assembly for a bilinear form with distinct test and trial spaces.
///
/// This is the public cross-space counterpart to
/// [`assemble_bilinear_checked`](super::assemble_bilinear_checked). It admits
/// both spaces before element tabulation, so mixed-space DOF/topology errors
/// are returned as [`SpaceValidationError`] instead of surfacing as a scatter
/// panic. Forms used with this API must honor the independent `test.n_dofs`
/// and `trial.n_dofs` cardinalities.
pub fn assemble_bilinear_two_spaces_checked<T, F, S0, S1>(
    form: &F,
    test_space: &S0,
    trial_space: &S1,
    mesh: &Mesh,
    quad_order: usize,
) -> Result<TripletMatrix<T>, SpaceValidationError>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F: BilinearForm<T>,
    S0: FunctionSpace,
    S1: FunctionSpace,
{
    test_space.validate(mesh)?;
    trial_space.validate(mesh)?;
    Ok(assemble_bilinear_two_spaces(
        form,
        test_space,
        trial_space,
        mesh,
        quad_order,
    ))
}

/// Assemble a cross-space bilinear form whose coefficients may vary by
/// element. This is the cross-space counterpart of
/// [`crate::assembly::assemble_bilinear_per_element`].
pub fn assemble_bilinear_two_spaces_per_element<T, F, S0, S1>(
    forms: &[F],
    test_space: &S0,
    trial_space: &S1,
    mesh: &Mesh,
    quad_order: usize,
) -> TripletMatrix<T>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F: BilinearForm<T>,
    S0: FunctionSpace,
    S1: FunctionSpace,
{
    assert_eq!(
        forms.len(),
        mesh.num_elements(),
        "one cross-space form is required for every mesh element"
    );

    let n_test = test_space.num_global_dofs(mesh);
    let n_trial = trial_space.num_global_dofs(mesh);
    let mut global = TripletMatrix::new(n_test, n_trial);

    for (element_index, element) in mesh.elements.iter().enumerate() {
        let test_dofs = test_space.local_to_global(mesh, element_index);
        let trial_dofs = trial_space.local_to_global(mesh, element_index);
        let n_test_local = test_dofs.len();
        let n_trial_local = trial_dofs.len();
        let quadrature = QuadratureRule::new(element.element_type, quad_order);
        let mut local_matrix = LocalMatrix::zeros(n_test_local.max(n_trial_local));

        for point in quadrature.iter() {
            let test_tabulation =
                test_space.tabulate(mesh, element_index, point.xi(), point.eta(), point.zeta());
            let trial_tabulation =
                trial_space.tabulate(mesh, element_index, point.xi(), point.eta(), point.zeta());
            forms[element_index].integrate_point(
                &test_tabulation,
                &trial_tabulation,
                point.weight,
                &mut local_matrix,
            );
        }

        for (local_row, &global_row) in test_dofs.iter().enumerate() {
            for (local_column, &global_column) in trial_dofs.iter().enumerate() {
                global.push(
                    global_row,
                    global_column,
                    local_matrix.get(local_row, local_column),
                );
            }
        }
    }

    global
}

/// Result of mixed assembly: the four blocks of the 2×2 system.
pub struct MixedSystem<T> {
    /// Top-left block: a(σ,τ) — e.g., ∫σ·τ for mixed Poisson
    pub block_00: TripletMatrix<T>,
    /// Top-right block: b(τ,u) — e.g., ∫u·div(τ) coupling
    pub block_01: TripletMatrix<T>,
    /// Bottom-left block: b(σ,v) — transpose of coupling
    pub block_10: TripletMatrix<T>,
    /// Bottom-right block: c(u,v) — often zero for saddle-point
    pub block_11: TripletMatrix<T>,
    /// Number of DOFs in first space (σ).
    pub n_dofs_0: usize,
    /// Number of DOFs in second space (u).
    pub n_dofs_1: usize,
}

impl<T: Copy + Default + std::ops::AddAssign + Send + Sync> MixedSystem<T> {
    /// Assemble to a single monolithic TripletMatrix with DOF offset.
    ///
    /// The first `n_dofs_0` rows/cols correspond to space 0 (σ),
    /// the next `n_dofs_1` to space 1 (u).
    pub fn to_monolithic(&self) -> TripletMatrix<T> {
        let n_total = self.n_dofs_0 + self.n_dofs_1;
        let total_nnz =
            self.block_00.nnz() + self.block_01.nnz() + self.block_10.nnz() + self.block_11.nnz();
        let mut mono = TripletMatrix::with_capacity(n_total, n_total, total_nnz);
        let offset = self.n_dofs_0;

        // Block (0,0): no offset
        for k in 0..self.block_00.nnz() {
            mono.push(
                self.block_00.rows[k],
                self.block_00.cols[k],
                self.block_00.values[k],
            );
        }
        // Block (0,1): column offset
        for k in 0..self.block_01.nnz() {
            mono.push(
                self.block_01.rows[k],
                self.block_01.cols[k] + offset,
                self.block_01.values[k],
            );
        }
        // Block (1,0): row offset
        for k in 0..self.block_10.nnz() {
            mono.push(
                self.block_10.rows[k] + offset,
                self.block_10.cols[k],
                self.block_10.values[k],
            );
        }
        // Block (1,1): both offsets
        for k in 0..self.block_11.nnz() {
            mono.push(
                self.block_11.rows[k] + offset,
                self.block_11.cols[k] + offset,
                self.block_11.values[k],
            );
        }

        mono
    }
}

impl<T: ComplexField> MixedSystem<T> {
    /// Assemble the mixed system as a backend-neutral block operator.
    ///
    /// This keeps the block structure visible to solver/preconditioner code
    /// instead of flattening the system into a monolithic sparse matrix.
    pub fn to_block_operator(&self) -> BlockOperator<T> {
        let layout = BlockLayout::new(
            vec![self.n_dofs_0, self.n_dofs_1],
            vec![self.n_dofs_0, self.n_dofs_1],
        );

        BlockOperator::new(layout)
            .with_block(0, 0, Box::new(self.block_00.to_csr()))
            .with_block(0, 1, Box::new(self.block_01.to_csr()))
            .with_block(1, 0, Box::new(self.block_10.to_csr()))
            .with_block(1, 1, Box::new(self.block_11.to_csr()))
    }
}

/// Assemble a mixed bilinear form with two separate forms for the
/// diagonal blocks and a coupling form for the off-diagonals.
///
/// - `form_00`: bilinear form for (σ,τ) block (e.g., mass in H(div))
/// - `form_01`: bilinear form for (τ,u) coupling (e.g., div·scalar)
/// - `space_0`: function space for σ (e.g., HdivSpace)
/// - `space_1`: function space for u (e.g., DGSpace)
///
/// The (1,0) block is assembled as the transpose of (0,1).
pub fn assemble_mixed<T, F00, F01, S0, S1>(
    form_00: &F00,
    form_01: &F01,
    space_0: &S0,
    space_1: &S1,
    mesh: &Mesh,
    quad_order: usize,
) -> MixedSystem<T>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F00: BilinearForm<T>,
    F01: BilinearForm<T>,
    S0: FunctionSpace,
    S1: FunctionSpace,
{
    let n0 = space_0.num_global_dofs(mesh);
    let n1 = space_1.num_global_dofs(mesh);

    // Assemble block (0,0): form_00 on space_0 × space_0
    let block_00 = super::assemble_bilinear(form_00, space_0, space_0, mesh, quad_order);

    // Assemble block (0,1): form_01 on space_0 (test) × space_1 (trial)
    let block_01 = assemble_bilinear_two_spaces(form_01, space_0, space_1, mesh, quad_order);

    // Block (1,0): transpose of (0,1)
    let mut block_10 = TripletMatrix::new(n1, n0);
    for k in 0..block_01.nnz() {
        block_10.push(block_01.cols[k], block_01.rows[k], block_01.values[k]);
    }

    // Block (1,1): zero (saddle-point structure)
    let block_11 = TripletMatrix::new(n1, n1);

    MixedSystem {
        block_00,
        block_01,
        block_10,
        block_11,
        n_dofs_0: n0,
        n_dofs_1: n1,
    }
}

/// Checked variant of [`assemble_mixed`].
///
/// Both spaces are admitted before any element tabulation or block scatter,
/// so malformed H(div)/scalar topology is reported as a typed error instead
/// of surfacing as an indexing panic in a mixed block.
pub fn assemble_mixed_checked<T, F00, F01, S0, S1>(
    form_00: &F00,
    form_01: &F01,
    space_0: &S0,
    space_1: &S1,
    mesh: &Mesh,
    quad_order: usize,
) -> Result<MixedSystem<T>, SpaceValidationError>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F00: BilinearForm<T>,
    F01: BilinearForm<T>,
    S0: FunctionSpace,
    S1: FunctionSpace,
{
    space_0.validate(mesh)?;
    space_1.validate(mesh)?;
    Ok(assemble_mixed(
        form_00, form_01, space_0, space_1, mesh, quad_order,
    ))
}

/// Assemble all four blocks of a checked mixed finite-element system.
///
/// Unlike [`assemble_mixed_checked`], this API does not infer the lower-left
/// block by transposition and does not assume that the lower-right block is
/// zero. Each form is evaluated with its own test/trial orientation, which is
/// required for nonsymmetric couplings, reaction terms, and stabilized mixed
/// formulations.
pub fn assemble_mixed_full_checked<T, F00, F01, F10, F11, S0, S1>(
    form_00: &F00,
    form_01: &F01,
    form_10: &F10,
    form_11: &F11,
    space_0: &S0,
    space_1: &S1,
    mesh: &Mesh,
    quad_order: usize,
) -> Result<MixedSystem<T>, SpaceValidationError>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F00: BilinearForm<T>,
    F01: BilinearForm<T>,
    F10: BilinearForm<T>,
    F11: BilinearForm<T>,
    S0: FunctionSpace,
    S1: FunctionSpace,
{
    space_0.validate(mesh)?;
    space_1.validate(mesh)?;

    let n_dofs_0 = space_0.num_global_dofs(mesh);
    let n_dofs_1 = space_1.num_global_dofs(mesh);
    Ok(MixedSystem {
        block_00: super::assemble_bilinear(form_00, space_0, space_0, mesh, quad_order),
        block_01: assemble_bilinear_two_spaces(form_01, space_0, space_1, mesh, quad_order),
        block_10: assemble_bilinear_two_spaces(form_10, space_1, space_0, mesh, quad_order),
        block_11: super::assemble_bilinear(form_11, space_1, space_1, mesh, quad_order),
        n_dofs_0,
        n_dofs_1,
    })
}

/// Assemble the RHS for a mixed system.
///
/// Returns `(rhs_0, rhs_1)` vectors for the two spaces.
pub fn assemble_mixed_rhs<T, L0, L1, S0, S1>(
    form_0: &L0,
    form_1: &L1,
    space_0: &S0,
    space_1: &S1,
    mesh: &Mesh,
    quad_order: usize,
) -> (Vec<T>, Vec<T>)
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    L0: LinearForm<T>,
    L1: LinearForm<T>,
    S0: FunctionSpace,
    S1: FunctionSpace,
{
    let rhs_0 = super::assemble_linear(form_0, space_0, mesh, quad_order);
    let rhs_1 = super::assemble_linear(form_1, space_1, mesh, quad_order);
    (rhs_0, rhs_1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::forms::{LaplacianForm, MassForm};
    use crate::mesh::unit_square_triangles;
    use crate::spaces::H1Space;
    use sonium_solvers::LinearOperator;

    #[test]
    fn test_mixed_system_monolithic_dimensions() {
        let mesh = unit_square_triangles(2);
        let space_0 = H1Space::new(PolynomialDegree::P1);
        let space_1 = H1Space::new(PolynomialDegree::P1);
        let n0 = space_0.num_global_dofs(&mesh);
        let n1 = space_1.num_global_dofs(&mesh);

        let system = assemble_mixed(&MassForm, &LaplacianForm, &space_0, &space_1, &mesh, 2);

        assert_eq!(system.n_dofs_0, n0);
        assert_eq!(system.n_dofs_1, n1);

        let mono = system.to_monolithic();
        assert_eq!(mono.num_rows, n0 + n1);
        assert_eq!(mono.num_cols, n0 + n1);
    }

    #[test]
    fn test_block_10_is_transpose_of_01() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);

        let system = assemble_mixed(&MassForm, &LaplacianForm, &space, &space, &mesh, 2);

        // Convert blocks to CSR and compare
        let csr_01 = system.block_01.to_csr();
        let csr_10 = system.block_10.to_csr();

        assert_eq!(csr_01.num_rows, csr_10.num_cols);
        assert_eq!(csr_01.num_cols, csr_10.num_rows);
    }

    #[test]
    fn test_mixed_system_block_operator_dimensions() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let system = assemble_mixed(&MassForm, &LaplacianForm, &space, &space, &mesh, 2);

        let block_op = system.to_block_operator();

        assert_eq!(block_op.num_rows(), system.n_dofs_0 + system.n_dofs_1);
        assert_eq!(block_op.num_cols(), system.n_dofs_0 + system.n_dofs_1);
        assert_eq!(block_op.layout().row_range(0), 0..system.n_dofs_0);
        assert_eq!(
            block_op.layout().row_range(1),
            system.n_dofs_0..system.n_dofs_0 + system.n_dofs_1
        );
        assert!(block_op.block(0, 0).is_some());
        assert!(block_op.block(0, 1).is_some());
        assert!(block_op.block(1, 0).is_some());
        assert!(block_op.block(1, 1).is_some());
    }

    #[test]
    fn test_mixed_system_block_operator_matches_monolithic_matvec() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let system = assemble_mixed(&MassForm, &LaplacianForm, &space, &space, &mesh, 2);

        let block_op = system.to_block_operator();
        let monolithic = system.to_monolithic().to_csr();
        let x: Vec<f64> = (0..block_op.num_cols()).map(|i| 1.0 + i as f64).collect();

        let y_block = block_op.apply(&x);
        let y_mono = monolithic.apply(&x);

        assert_eq!(y_block.len(), y_mono.len());
        for (actual, expected) in y_block.iter().zip(y_mono.iter()) {
            assert!((*actual - *expected).abs() < 1.0e-10);
        }
    }
}
