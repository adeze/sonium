//! Acoustics-specific assembler for efficient frequency sweeps
//!
//! Wraps [`PrebakedAssembler<2>`] with acoustics-oriented convenience methods.
//! Component 0 = stiffness (K), component 1 = mass (M).
//! Assembles the Helmholtz system `A = K - k^2 M` at O(nnz) per frequency.

use crate::assembly::TripletMatrix;
use crate::assembly::prebaked::PrebakedAssembler;
use crate::basis::PolynomialDegree;
use crate::forms::{LaplacianForm, MassForm};
use crate::mesh::Mesh;
use crate::spaces::H1Space;
use num_complex::Complex64;
use sonium_solvers::CsrMatrix;
use std::collections::HashMap;

use super::viscothermal::EquivalentFluid;

/// Type alias for the acoustics pre-baked assembler (stiffness + mass).
pub type AcousticsAssembler = PrebakedAssembler<2>;

/// Build an [`AcousticsAssembler`] from a mesh and polynomial degree.
///
/// Assembles the stiffness and mass matrices using the generic assembler,
/// then pre-bakes them into a shared CSR sparsity pattern.
pub fn build_acoustics_assembler(mesh: &Mesh, degree: PolynomialDegree) -> AcousticsAssembler {
    let space = H1Space::new(degree);
    let quad_stiffness = crate::quadrature::required_order_for_stiffness(degree.degree());
    let quad_mass = crate::quadrature::required_order_for_mass(degree.degree());

    let stiffness: TripletMatrix<f64> =
        crate::assembly::assemble_bilinear(&LaplacianForm, &space, &space, mesh, quad_stiffness);
    let mass: TripletMatrix<f64> =
        crate::assembly::assemble_bilinear(&MassForm, &space, &space, mesh, quad_mass);

    PrebakedAssembler::from_triplets([stiffness, mass], &[])
}

/// Build an [`AcousticsAssembler`] with boundary mass matrices.
///
/// `boundary_tags` specifies which boundary markers to include.
/// Each boundary mass matrix integrates `integral_Gamma phi_i phi_j dGamma`
/// on the boundary faces with that marker.
pub fn build_acoustics_assembler_with_boundaries(
    mesh: &Mesh,
    degree: PolynomialDegree,
    boundary_tags: &[i32],
) -> AcousticsAssembler {
    let space = H1Space::new(degree);
    let quad_stiffness = crate::quadrature::required_order_for_stiffness(degree.degree());
    let quad_mass = crate::quadrature::required_order_for_mass(degree.degree());

    let stiffness =
        crate::assembly::assemble_bilinear(&LaplacianForm, &space, &space, mesh, quad_stiffness);
    let mass = crate::assembly::assemble_bilinear(&MassForm, &space, &space, mesh, quad_mass);

    // Assemble boundary mass matrices using the legacy function for now
    // (boundary surface integration is not yet in the generic assembler)
    let mut boundary_triplets = Vec::new();
    for &tag in boundary_tags {
        let b_mass = crate::assembly::assemble_boundary_mass(mesh, degree, tag);
        if b_mass.nnz() > 0 {
            let triplet = TripletMatrix {
                rows: b_mass.rows,
                cols: b_mass.cols,
                values: b_mass.values,
                num_rows: b_mass.dim,
                num_cols: b_mass.dim,
            };
            boundary_triplets.push((tag as usize, triplet));
        }
    }

    PrebakedAssembler::from_triplets([stiffness, mass], &boundary_triplets)
}

/// Extension methods for the acoustics assembler.
pub trait AcousticsAssemblerExt {
    /// Assemble the Helmholtz system `A = K - k^2 M` at the given wavenumber.
    fn assemble_helmholtz(
        &self,
        k: Complex64,
        boundary_coeffs: &HashMap<usize, Complex64>,
    ) -> CsrMatrix<Complex64>;

    /// Assemble the shifted-Laplacian preconditioner `P = K + (alpha + i*beta) M`.
    fn assemble_shifted_laplacian(&self, alpha: f64, beta: f64) -> CsrMatrix<Complex64>;

    /// Assemble a real-valued combination `result = k_coeff * K + m_coeff * M`.
    /// Used by WaveHoltz for time-stepping matrices.
    fn assemble_real_combination(&self, k_coeff: f64, m_coeff: f64) -> CsrMatrix<f64>;

    /// Assemble the viscothermal Helmholtz system using the equivalent fluid model.
    ///
    /// For spatially uniform fluid properties:
    /// `A = (1/ρ_eff(ω)) K − (ω²/K_eff(ω)) M + boundary terms`
    ///
    /// The boundary coefficients can include thermal BLI wall admittance
    /// (see [`thermal_wall_admittance`](super::thermal_wall_admittance)) or
    /// any other impedance condition.
    fn assemble_viscothermal(
        &self,
        equivalent_fluid: &EquivalentFluid,
        omega: f64,
        boundary_coeffs: &HashMap<usize, Complex64>,
    ) -> CsrMatrix<Complex64>;
}

impl AcousticsAssemblerExt for AcousticsAssembler {
    fn assemble_helmholtz(
        &self,
        k: Complex64,
        boundary_coeffs: &HashMap<usize, Complex64>,
    ) -> CsrMatrix<Complex64> {
        let k_sq = k * k;
        self.assemble_complex(&[Complex64::new(1.0, 0.0), -k_sq], boundary_coeffs)
    }

    fn assemble_shifted_laplacian(&self, alpha: f64, beta: f64) -> CsrMatrix<Complex64> {
        self.assemble_complex(
            &[Complex64::new(1.0, 0.0), Complex64::new(alpha, beta)],
            &HashMap::new(),
        )
    }

    fn assemble_real_combination(&self, k_coeff: f64, m_coeff: f64) -> CsrMatrix<f64> {
        self.assemble_real(&[k_coeff, m_coeff])
    }

    fn assemble_viscothermal(
        &self,
        equivalent_fluid: &EquivalentFluid,
        omega: f64,
        boundary_coeffs: &HashMap<usize, Complex64>,
    ) -> CsrMatrix<Complex64> {
        let coeffs = equivalent_fluid.prebaked_coefficients(omega);
        self.assemble_complex(&coeffs, boundary_coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn test_build_acoustics_assembler() {
        let mesh = unit_square_triangles(4);
        let assembler = build_acoustics_assembler(&mesh, PolynomialDegree::P1);

        assert!(assembler.nnz() > 0);
        assert_eq!(assembler.num_rows, mesh.num_nodes());
    }

    #[test]
    fn test_assemble_helmholtz_at_k0_is_laplacian() {
        let mesh = unit_square_triangles(4);
        let assembler = build_acoustics_assembler(&mesh, PolynomialDegree::P1);

        let k = Complex64::new(0.0, 0.0);
        let csr = assembler.assemble_helmholtz(k, &HashMap::new());

        // At k=0: A = K. Row sums of Laplacian should be ~0.
        for i in 0..csr.num_rows {
            let row_sum: Complex64 = csr.row_entries(i).map(|(_, v)| v).sum();
            assert!(
                row_sum.norm() < 1e-10,
                "Row {i} sum = {row_sum}, expected ~0"
            );
        }
    }

    #[test]
    fn test_real_combination() {
        let mesh = unit_square_triangles(4);
        let assembler = build_acoustics_assembler(&mesh, PolynomialDegree::P1);

        // K + M (both positive, so diagonal should be positive)
        let csr = assembler.assemble_real_combination(1.0, 1.0);
        for i in 0..csr.num_rows {
            let diag = csr.row_entries(i).find(|(j, _)| *j == i).map(|(_, v)| v);
            assert!(
                diag.is_some_and(|v| v > 0.0),
                "Diagonal at {i} should be positive"
            );
        }
    }
}
