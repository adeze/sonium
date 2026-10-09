//! Linearized Potential Flow (LPF) interface
//!
//! Solves the convected Helmholtz equation for the velocity potential phi
//! in the presence of a uniform, irrotational background mean flow.
//!
//! # Governing equation (frequency domain)
//!
//! ```text
//! nabla^2 phi - (M . nabla)^2 phi + k^2 phi - 2ik(M . nabla) phi = Q
//! ```
//!
//! where k = omega/c_0 is the wavenumber and M = U_0/c_0 is the Mach vector.
//!
//! # Weak form (4 bilinear components)
//!
//! ```text
//! A(omega) = K - S - k^2 M_mass + 2ik C
//! ```
//!
//! - K: integral(grad phi . grad psi) dOmega       (stiffness)
//! - M_mass: integral(phi psi) dOmega               (mass)
//! - C: 1/2 * integral((M . grad phi) psi - phi (M . grad psi)) dOmega
//!   (skew convection)
//! - S: integral((M . grad phi)(M . grad psi)) dOmega (streaming stiffness)
//!
//! Uses [`PrebakedAssembler<4>`] for O(nnz) frequency sweeps.
//!
//! # Modal source decomposition
//!
//! For duct acoustics (e.g. turbofan engines), use the port conditions
//! from [`super::boundary`] to impose propagating and non-propagating
//! modes at inlet/outlet boundaries.

use crate::assembly::TripletMatrix;
use crate::assembly::prebaked::PrebakedAssembler;
use crate::basis::PolynomialDegree;
use crate::forms::{BilinearForm, LaplacianForm, LocalMatrix, MassForm};
use crate::mesh::Mesh;
use crate::spaces::{BasisTabulation, H1Space};
use num_complex::Complex64;
use sonium_solvers::CsrMatrix;
use std::collections::HashMap;

use super::mean_flow::MeanFlow;

// ---------------------------------------------------------------------------
// Forms
// ---------------------------------------------------------------------------

/// Scalar convection form:
/// `c(phi, psi) = 1/2 * integral((M . grad phi) psi - phi (M . grad psi)) dOmega`
///
/// Skew-symmetric form coupling the mean flow Mach vector to gradients of the
/// acoustic potential. This drops the boundary term from the raw advective
/// operator and keeps the discrete convection block energy-conserving.
pub struct ScalarConvectionForm {
    /// Mach number vector M = U_0 / c_0 (or any advection direction).
    pub advection: [f64; 3],
}

impl BilinearForm<f64> for ScalarConvectionForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j;

        for i in 0..n {
            let psi_i = test.value(i);
            let m_dot_grad_i: f64 = (0..dim).map(|d| self.advection[d] * test.grad(i, d)).sum();
            for j in 0..n {
                let m_dot_grad: f64 = (0..dim).map(|d| self.advection[d] * trial.grad(j, d)).sum();
                let phi_j = trial.value(j);
                local_matrix.add(i, j, 0.5 * (psi_i * m_dot_grad - phi_j * m_dot_grad_i) * w);
            }
        }
    }
}

/// Streaming stiffness form: `s(phi, psi) = integral((M . grad phi)(M . grad psi)) dOmega`
///
/// Symmetric form arising from double material-derivative convection.
/// Accounts for the effect of the mean flow on the effective wave speed.
pub struct StreamingStiffnessForm {
    /// Mach number vector M = U_0 / c_0.
    pub mach: [f64; 3],
}

impl BilinearForm<f64> for StreamingStiffnessForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j;

        for i in 0..n {
            let m_dot_grad_i: f64 = (0..dim).map(|d| self.mach[d] * test.grad(i, d)).sum();
            for j in 0..n {
                let m_dot_grad_j: f64 = (0..dim).map(|d| self.mach[d] * trial.grad(j, d)).sum();
                local_matrix.add(i, j, m_dot_grad_i * m_dot_grad_j * w);
            }
        }
    }
}

/// One-shot convected Helmholtz form at a fixed frequency.
///
/// Combines all four components into a single complex-valued form:
/// `a(phi, psi) = (K - S) - k^2 M + 2ik C`
///
/// Use this for single-frequency solves. For frequency sweeps,
/// prefer [`LpfAssembler`] via [`build_lpf_assembler`].
pub struct ConvectedHelmholtzForm {
    pub mach: [f64; 3],
    pub wavenumber: Complex64,
}

impl ConvectedHelmholtzForm {
    pub fn new(mean_flow: &MeanFlow, omega: f64) -> Self {
        Self {
            mach: mean_flow.mach_vector(),
            wavenumber: Complex64::new(mean_flow.wavenumber(omega), 0.0),
        }
    }
}

impl BilinearForm<Complex64> for ConvectedHelmholtzForm {
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
        let w = weight * test.det_j;

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

                let convection = 0.5 * (psi_i * m_dot_grad_j - trial.value(j) * m_dot_grad_i);

                // A = K - S - k^2 M + 2ik C (matching HelmholtzForm convention)
                let real_part = (grad_dot - streaming) * w;
                let val =
                    Complex64::new(real_part, 0.0) - k_sq * mass * w + two_ik * convection * w;

                local_matrix.add(i, j, val);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Assembler
// ---------------------------------------------------------------------------

/// Pre-baked assembler for LPF frequency sweeps.
///
/// Components: `[stiffness, mass, convection, streaming]`
///
/// Reassembly is needed when the mean flow (Mach number) changes,
/// since convection and streaming operators depend on M.
pub type LpfAssembler = PrebakedAssembler<4>;

/// Build an LPF assembler from mesh, polynomial degree, and mean flow.
///
/// Assembles the four component matrices (K, M, C, S) once, then
/// frequency sweeps cost only O(nnz) per frequency point.
pub fn build_lpf_assembler(
    mesh: &Mesh,
    degree: PolynomialDegree,
    mean_flow: &MeanFlow,
) -> LpfAssembler {
    let space = H1Space::new(degree);
    let quad_stiffness = crate::quadrature::required_order_for_stiffness(degree.degree());
    let quad_mass = crate::quadrature::required_order_for_mass(degree.degree());

    let stiffness: TripletMatrix<f64> =
        crate::assembly::assemble_bilinear(&LaplacianForm, &space, &space, mesh, quad_stiffness);
    let mass: TripletMatrix<f64> =
        crate::assembly::assemble_bilinear(&MassForm, &space, &space, mesh, quad_mass);

    let mach = mean_flow.mach_vector();
    let conv_form = ScalarConvectionForm { advection: mach };
    let convection: TripletMatrix<f64> =
        crate::assembly::assemble_bilinear(&conv_form, &space, &space, mesh, quad_stiffness);

    let stream_form = StreamingStiffnessForm { mach };
    let streaming: TripletMatrix<f64> =
        crate::assembly::assemble_bilinear(&stream_form, &space, &space, mesh, quad_stiffness);

    PrebakedAssembler::from_triplets([stiffness, mass, convection, streaming], &[])
}

/// Extension methods for the LPF assembler.
pub trait LpfAssemblerExt {
    /// Assemble the convected Helmholtz system at angular frequency omega.
    ///
    /// `A(omega) = K - k^2 M + 2ik C - S`
    ///
    /// where k = omega / c_0.
    fn assemble_convected_helmholtz(
        &self,
        mean_flow: &MeanFlow,
        omega: f64,
        boundary_coeffs: &HashMap<usize, Complex64>,
    ) -> CsrMatrix<Complex64>;
}

impl LpfAssemblerExt for LpfAssembler {
    fn assemble_convected_helmholtz(
        &self,
        mean_flow: &MeanFlow,
        omega: f64,
        boundary_coeffs: &HashMap<usize, Complex64>,
    ) -> CsrMatrix<Complex64> {
        let k = Complex64::new(mean_flow.wavenumber(omega), 0.0);

        // Components: [K, M, C, S]
        // A = K - k^2 M + 2ik C - S (matching HelmholtzForm convention)
        let coeffs = [
            Complex64::new(1.0, 0.0),
            -(k * k),
            Complex64::new(0.0, 2.0) * k,
            Complex64::new(-1.0, 0.0),
        ];

        self.assemble_complex(&coeffs, boundary_coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;
    use crate::physics::acoustics::pressure::AcousticsAssemblerExt;
    use crate::spaces::FunctionSpace;

    #[test]
    fn test_convection_form_antisymmetric() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let form = ScalarConvectionForm {
            advection: [1.0, 0.0, 0.0],
        };
        let mut mat = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..n {
            assert!(mat.get(i, i).abs() < 1e-14);
            for j in i + 1..n {
                assert!(
                    (mat.get(i, j) + mat.get(j, i)).abs() < 1e-14,
                    "ScalarConvectionForm not skew-symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_streaming_stiffness_symmetry() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let form = StreamingStiffnessForm {
            mach: [0.3, 0.1, 0.0],
        };
        let mut mat = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..n {
            for j in i + 1..n {
                assert!(
                    (mat.get(i, j) - mat.get(j, i)).abs() < 1e-14,
                    "StreamingStiffness not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_convected_helmholtz_reduces_to_helmholtz_at_m0() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        // Zero Mach number: convected Helmholtz should reduce to standard Helmholtz
        let flow = MeanFlow::air_standard([0.0, 0.0, 0.0]);
        let omega = 1000.0;
        let form = ConvectedHelmholtzForm::new(&flow, omega);
        let mut mat_conv = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut mat_conv);

        let k = Complex64::new(flow.wavenumber(omega), 0.0);
        let helm = crate::physics::acoustics::pressure::HelmholtzForm::new(k);
        let mut mat_helm = LocalMatrix::zeros(n);
        helm.integrate_point(&tab, &tab, 1.0, &mut mat_helm);

        for i in 0..n {
            for j in 0..n {
                let diff = (mat_conv.get(i, j) - mat_helm.get(i, j)).norm();
                assert!(
                    diff < 1e-10,
                    "At M=0, convected Helmholtz should equal Helmholtz at ({i},{j}): diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_lpf_assembler_frequency_sweep() {
        let mesh = unit_square_triangles(4);
        let flow = MeanFlow::air_standard([34.3, 0.0, 0.0]); // M = 0.1
        let assembler = build_lpf_assembler(&mesh, PolynomialDegree::P1, &flow);

        assert!(assembler.nnz() > 0);

        // Sweep over a few frequencies
        for &omega in &[100.0, 500.0, 1000.0, 5000.0] {
            let csr = assembler.assemble_convected_helmholtz(&flow, omega, &HashMap::new());
            assert_eq!(csr.num_rows, mesh.num_nodes());
            assert!(csr.nnz() > 0);
        }
    }

    #[test]
    fn test_lpf_assembler_at_m0_matches_acoustics() {
        let mesh = unit_square_triangles(4);
        let flow = MeanFlow::air_standard([0.0, 0.0, 0.0]); // M = 0
        let lpf = build_lpf_assembler(&mesh, PolynomialDegree::P1, &flow);
        let acoustics = crate::physics::acoustics::pressure::build_acoustics_assembler(
            &mesh,
            PolynomialDegree::P1,
        );

        let omega = 2000.0;
        let k = Complex64::new(flow.wavenumber(omega), 0.0);

        let csr_lpf = lpf.assemble_convected_helmholtz(&flow, omega, &HashMap::new());
        let csr_helm = acoustics.assemble_helmholtz(k, &HashMap::new());

        // At M=0 the two should produce identical matrices
        assert_eq!(csr_lpf.num_rows, csr_helm.num_rows);
        for i in 0..csr_lpf.num_rows {
            let row_lpf: Complex64 = csr_lpf.row_entries(i).map(|(_, v)| v).sum();
            let row_helm: Complex64 = csr_helm.row_entries(i).map(|(_, v)| v).sum();
            assert!(
                (row_lpf - row_helm).norm() < 1e-10,
                "Row {i} mismatch: lpf={row_lpf}, helm={row_helm}"
            );
        }
    }
}
