//! Axisymmetric convected Helmholtz forms for duct aeroacoustics
//!
//! Extends the Linearized Potential Flow (LPF) formulation to axisymmetric
//! geometries. The mesh lives in the (r, z) meridional plane where x = r.
//!
//! The convected Helmholtz equation in axisymmetric form (m=0 mode):
//!
//! ```text
//! ∫ r [∇φ·∇ψ − (M·∇φ)(M·∇ψ) − k²φψ + 2ik(M·∇φ)ψ] dr dz = 0
//! ```
//!
//! Uses `PrebakedAssembler<4>` for O(nnz) frequency sweeps — identical
//! coefficient structure to the standard LPF but with r-weighted forms.

use crate::assembly::TripletMatrix;
use crate::assembly::prebaked::PrebakedAssembler;
use crate::basis::PolynomialDegree;
use crate::forms::{BilinearForm, LocalMatrix};
use crate::mesh::Mesh;
use crate::physics::acoustics::pressure::axisymmetric::{
    AxiLaplacianForm, AxiMassForm, axisymmetric_quadrature_orders,
};
use crate::spaces::{BasisTabulation, H1Space};
use num_complex::Complex64;
use sonium_solvers::CsrMatrix;
use std::collections::HashMap;

use super::mean_flow::MeanFlow;

// ---------------------------------------------------------------------------
// Forms
// ---------------------------------------------------------------------------

/// Axisymmetric scalar convection form: `c(φ,ψ) = ∫ r (M·∇φ) ψ dr dz`
pub struct AxiScalarConvectionForm {
    pub advection: [f64; 3],
}

impl BilinearForm<f64> for AxiScalarConvectionForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let r = test.physical_coords[0];
        let w = weight * test.det_j * r;

        for i in 0..n {
            let psi_i = test.value(i);
            for j in 0..n {
                let m_dot_grad: f64 = (0..dim).map(|d| self.advection[d] * trial.grad(j, d)).sum();
                local_matrix.add(i, j, psi_i * m_dot_grad * w);
            }
        }
    }
}

/// Axisymmetric streaming stiffness form: `s(φ,ψ) = ∫ r (M·∇φ)(M·∇ψ) dr dz`
pub struct AxiStreamingStiffnessForm {
    pub mach: [f64; 3],
}

impl BilinearForm<f64> for AxiStreamingStiffnessForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let r = test.physical_coords[0];
        let w = weight * test.det_j * r;

        for i in 0..n {
            let m_dot_grad_i: f64 = (0..dim).map(|d| self.mach[d] * test.grad(i, d)).sum();
            for j in 0..n {
                let m_dot_grad_j: f64 = (0..dim).map(|d| self.mach[d] * trial.grad(j, d)).sum();
                local_matrix.add(i, j, m_dot_grad_i * m_dot_grad_j * w);
            }
        }
    }
}

/// One-shot axisymmetric convected Helmholtz form at a fixed frequency.
///
/// `a(φ,ψ) = ∫ r [(∇φ·∇ψ − (M·∇φ)(M·∇ψ)) − k²φψ + 2ik(M·∇φ)ψ] dr dz`
pub struct AxiConvectedHelmholtzForm {
    pub mach: [f64; 3],
    pub wavenumber: Complex64,
}

impl AxiConvectedHelmholtzForm {
    pub fn new(mean_flow: &MeanFlow, omega: f64) -> Self {
        Self {
            mach: mean_flow.mach_vector(),
            wavenumber: Complex64::new(mean_flow.wavenumber(omega), 0.0),
        }
    }
}

impl BilinearForm<Complex64> for AxiConvectedHelmholtzForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<Complex64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let k = self.wavenumber;
        let k_sq = k * k;
        let two_ik = Complex64::new(0.0, 2.0) * k;
        let r = test.physical_coords[0];
        let w = weight * test.det_j * r;

        for i in 0..n {
            let psi_i = test.value(i);
            let m_dot_grad_i: f64 = (0..dim).map(|d| self.mach[d] * test.grad(i, d)).sum();

            for j in 0..n {
                let mut grad_dot = 0.0;
                let mut m_dot_grad_j = 0.0;
                for d in 0..dim {
                    grad_dot += test.grad(i, d) * trial.grad(j, d);
                    m_dot_grad_j += self.mach[d] * trial.grad(j, d);
                }
                let mass = psi_i * trial.value(j);
                let streaming = m_dot_grad_i * m_dot_grad_j;

                let real_part = (grad_dot - streaming) * w;
                let val = Complex64::new(real_part, 0.0) - k_sq * mass * w
                    + two_ik * psi_i * m_dot_grad_j * w;

                local_matrix.add(i, j, val);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Assembler
// ---------------------------------------------------------------------------

/// Pre-baked assembler for axisymmetric LPF frequency sweeps.
///
/// Components: `[stiffness, mass, convection, streaming]` — all r-weighted.
pub type AxiLpfAssembler = PrebakedAssembler<4>;

/// Build an axisymmetric LPF assembler from mesh, polynomial degree, and mean flow.
pub fn build_axi_lpf_assembler(
    mesh: &Mesh,
    degree: PolynomialDegree,
    mean_flow: &MeanFlow,
) -> AxiLpfAssembler {
    let space = H1Space::new(degree);
    let (quad_stiffness, quad_mass) = axisymmetric_quadrature_orders(degree);

    let stiffness: TripletMatrix<f64> =
        crate::assembly::assemble_bilinear(&AxiLaplacianForm, &space, &space, mesh, quad_stiffness);
    let mass: TripletMatrix<f64> =
        crate::assembly::assemble_bilinear(&AxiMassForm, &space, &space, mesh, quad_mass);

    let mach = mean_flow.mach_vector();
    let conv_form = AxiScalarConvectionForm { advection: mach };
    let convection: TripletMatrix<f64> =
        crate::assembly::assemble_bilinear(&conv_form, &space, &space, mesh, quad_stiffness);

    let stream_form = AxiStreamingStiffnessForm { mach };
    let streaming: TripletMatrix<f64> =
        crate::assembly::assemble_bilinear(&stream_form, &space, &space, mesh, quad_stiffness);

    PrebakedAssembler::from_triplets([stiffness, mass, convection, streaming], &[])
}

/// Build an axisymmetric LPF assembler with boundary mass matrices.
pub fn build_axi_lpf_assembler_with_boundaries(
    mesh: &Mesh,
    degree: PolynomialDegree,
    mean_flow: &MeanFlow,
    boundary_tags: &[i32],
) -> AxiLpfAssembler {
    let space = H1Space::new(degree);
    let (quad_stiffness, quad_mass) = axisymmetric_quadrature_orders(degree);

    let stiffness =
        crate::assembly::assemble_bilinear(&AxiLaplacianForm, &space, &space, mesh, quad_stiffness);
    let mass = crate::assembly::assemble_bilinear(&AxiMassForm, &space, &space, mesh, quad_mass);

    let mach = mean_flow.mach_vector();
    let conv_form = AxiScalarConvectionForm { advection: mach };
    let convection =
        crate::assembly::assemble_bilinear(&conv_form, &space, &space, mesh, quad_stiffness);

    let stream_form = AxiStreamingStiffnessForm { mach };
    let streaming =
        crate::assembly::assemble_bilinear(&stream_form, &space, &space, mesh, quad_stiffness);

    let mut boundary_triplets = Vec::new();
    for &tag in boundary_tags {
        let b_mass = crate::assembly::assemble_axi_boundary_mass(mesh, degree, tag);
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

    PrebakedAssembler::from_triplets([stiffness, mass, convection, streaming], &boundary_triplets)
}

/// Extension methods for the axisymmetric LPF assembler.
pub trait AxiLpfAssemblerExt {
    /// Assemble the axisymmetric convected Helmholtz system.
    ///
    /// `A(ω) = K − k²M + 2ikC − S` where all components are r-weighted.
    fn assemble_axi_convected_helmholtz(
        &self,
        mean_flow: &MeanFlow,
        omega: f64,
        boundary_coeffs: &HashMap<usize, Complex64>,
    ) -> CsrMatrix<Complex64>;
}

impl AxiLpfAssemblerExt for AxiLpfAssembler {
    fn assemble_axi_convected_helmholtz(
        &self,
        mean_flow: &MeanFlow,
        omega: f64,
        boundary_coeffs: &HashMap<usize, Complex64>,
    ) -> CsrMatrix<Complex64> {
        let k = Complex64::new(mean_flow.wavenumber(omega), 0.0);

        let coeffs = [
            Complex64::new(1.0, 0.0),     // K
            -(k * k),                     // -k² M
            Complex64::new(0.0, 2.0) * k, // 2ik C
            Complex64::new(-1.0, 0.0),    // -S
        ];

        self.assemble_complex(&coeffs, boundary_coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;
    use crate::physics::acoustics::pressure::AcousticsAssemblerExt;
    use crate::physics::acoustics::pressure::axisymmetric::build_axi_acoustics_assembler;

    #[test]
    fn test_axi_lpf_reduces_to_helmholtz_at_zero_mach() {
        let mesh = unit_square_triangles(4);
        let zero_flow = MeanFlow::air_standard([0.0, 0.0, 0.0]);

        let omega = 2000.0 * std::f64::consts::PI;
        let k = Complex64::new(zero_flow.wavenumber(omega), 0.0);

        for degree in [PolynomialDegree::P1, PolynomialDegree::P2] {
            let lpf = build_axi_lpf_assembler(&mesh, degree, &zero_flow);
            let helmholtz = build_axi_acoustics_assembler(&mesh, degree);
            let a_lpf = lpf.assemble_axi_convected_helmholtz(&zero_flow, omega, &HashMap::new());
            let a_helm = helmholtz.assemble_helmholtz(k, &HashMap::new());

            // At M=0, convection and streaming vanish; LPF should equal Helmholtz.
            let n = a_lpf.num_rows;
            let mut max_diff = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    let diff = (a_lpf.get(i, j) - a_helm.get(i, j)).norm();
                    max_diff = max_diff.max(diff);
                }
            }
            assert!(
                max_diff < 1e-10,
                "P{} LPF at M=0 should match Helmholtz, max diff = {max_diff:.2e}",
                degree.degree()
            );
        }
    }
}
