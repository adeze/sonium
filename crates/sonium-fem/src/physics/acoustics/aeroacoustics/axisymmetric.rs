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
use crate::mesh::{Mesh, MeshTopology};
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
///
/// Advection is in MESH frame (`[M_r, M_z, _]`): mesh-x is the radius,
/// mesh-y the axis. Pass uniform axial flow as `[0, M_z, 0]`.
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
///
/// The mean velocity is read in MESH frame (`[U_r, U_z, _]`); uniform
/// axial flow is `[0, U_z, 0]`.
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

/// Azimuthal mass form for spinning LPF modes: `a(φ,ψ) = ∫ (m²/r) φψ dr dz`.
///
/// The `m²/r²` term of the cylindrical Laplacian, r-weighted. For
/// `|m| ≥ 1` constrain the axis nodes (see [`axi_axis_scalar_dofs`):
/// nodal constraints make every basis function vanish linearly toward
/// the axis, so all entries stay integrable. Vanishes for `m = 0`.
pub struct AxiAzimuthalMassForm {
    /// Unsigned azimuthal order.
    pub azimuthal_order: u32,
}

impl BilinearForm<f64> for AxiAzimuthalMassForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let order_sq = f64::from(self.azimuthal_order) * f64::from(self.azimuthal_order);
        if order_sq == 0.0 {
            return;
        }
        let r = test.physical_coords[0];
        if r <= 0.0 {
            // Standard rules never sample the axis; guard stays silent.
            return;
        }
        let n = test.n_dofs;
        let w = weight * test.det_j * order_sq / r;
        for i in 0..n {
            let psi_i = test.value(i);
            for j in 0..n {
                local_matrix.add(i, j, psi_i * trial.value(j) * w);
            }
        }
    }
}

/// Scalar axis DOFs of an H1 space on a meridional mesh.
///
/// Collects P1 vertex DOFs with `r ≈ 0` plus, for P2, the edge-midpoint
/// DOFs of axis edges via the mesh topology (standard layout; meshes
/// with explicit higher-order nodes use the vertex rule, which is
/// complete there since every DOF is a node). Degrees above P2 fall
/// back to the vertex rule, documented as partial.
pub fn axi_axis_scalar_dofs(mesh: &Mesh, degree: PolynomialDegree) -> Vec<usize> {
    let max_r = mesh
        .nodes
        .iter()
        .map(|node| node.x.abs())
        .fold(0.0f64, f64::max)
        .max(1e-300);
    let tolerance = 1e-9 * max_r.max(1.0);
    let mut dofs: Vec<usize> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.x.abs() <= tolerance)
        .map(|(index, _)| index)
        .collect();
    if degree == PolynomialDegree::P1 {
        dofs.sort_unstable();
        return dofs;
    }
    let explicit_p2 = mesh
        .elements
        .iter()
        .any(|element| element.nodes.len() > element.element_type.num_vertices());
    if explicit_p2 || degree != PolynomialDegree::P2 {
        dofs.sort_unstable();
        return dofs;
    }
    let topology = MeshTopology::build(mesh);
    let node_offset = mesh.num_nodes();
    for (edge_index, &(first, second)) in topology.edges.iter().enumerate() {
        if mesh.nodes[first].x.abs() <= tolerance && mesh.nodes[second].x.abs() <= tolerance {
            dofs.push(node_offset + edge_index);
        }
    }
    dofs.sort_unstable();
    dofs.dedup();
    dofs
}

/// Pre-baked assembler for spinning axisymmetric LPF frequency sweeps.
///
/// Components: `[stiffness, mass, convection, streaming, azimuthal]` —
/// all r-weighted. The azimuthal component carries the `m²/r` term.
pub type AxiLpfModeAssembler = PrebakedAssembler<5>;

/// Build a spinning-mode axisymmetric LPF assembler.
pub fn build_axi_lpf_mode_assembler(
    mesh: &Mesh,
    degree: PolynomialDegree,
    mean_flow: &MeanFlow,
    azimuthal: sonium_wave::analytical::azimuthal::AzimuthalMode,
) -> AxiLpfModeAssembler {
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

    let azimuthal_form = AxiAzimuthalMassForm {
        azimuthal_order: azimuthal.abs_order(),
    };
    let azimuthal_mass: TripletMatrix<f64> =
        crate::assembly::assemble_bilinear(&azimuthal_form, &space, &space, mesh, quad_mass);

    PrebakedAssembler::from_triplets(
        [stiffness, mass, convection, streaming, azimuthal_mass],
        &[],
    )
}

/// Extension methods for the spinning-mode axisymmetric LPF assembler.
pub trait AxiLpfModeAssemblerExt {
    /// Assemble the spinning-mode convected Helmholtz system.
    ///
    /// `A(ω) = K − k²M + 2ikC − S − Aφ` with r-weighted components.
    fn assemble_axi_modal_convected_helmholtz(
        &self,
        mean_flow: &MeanFlow,
        omega: f64,
        boundary_coeffs: &HashMap<usize, Complex64>,
    ) -> CsrMatrix<Complex64>;
}

impl AxiLpfModeAssemblerExt for AxiLpfModeAssembler {
    fn assemble_axi_modal_convected_helmholtz(
        &self,
        mean_flow: &MeanFlow,
        omega: f64,
        boundary_coeffs: &HashMap<usize, Complex64>,
    ) -> CsrMatrix<Complex64> {
        let k = Complex64::new(mean_flow.wavenumber(omega), 0.0);
        let coeffs = [
            Complex64::new(1.0, 0.0),
            -(k * k),
            Complex64::new(0.0, 2.0) * k,
            Complex64::new(-1.0, 0.0),
            Complex64::new(-1.0, 0.0),
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
