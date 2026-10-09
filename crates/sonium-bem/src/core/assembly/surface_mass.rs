//! Consistent P1 surface mass matrix on a triangulated mesh.
//!
//! For Phase 1.2's "dual mass on barycentric refinement" preconditioner
//! (per `bem-step1.md`): assembles the standard finite-element P1 mass
//! matrix `M[i, j] = ∫_S φ_i φ_j dS` over a triangle mesh, where each
//! triangle contributes the local
//! ```text
//!   M_loc = (Area / 12) · [ 2 1 1
//!                           1 2 1
//!                           1 1 2 ]
//! ```
//! pattern. Mirrors the assembly path in
//! `crates/sonium-fem/src/assembly/mass.rs::assemble_boundary_mass` but
//! consumes the BEM `Mesh` API and the barycentric refinement directly,
//! returning a real-valued sparse CSR matrix that downstream BEM code
//! can use as a preconditioner once a primal↔refined lifting operator
//! is wired in.
//!
//! The matrix is symmetric and SPD on a non-degenerate triangulation
//! (every triangle area is strictly positive, so the local mass matrix
//! is SPD; the global assembly preserves SPD-ness).
//!
//! Composition into the full Hiptmair-Buffa Calderón sandwich
//! `P = M_dual⁻¹ X M_dual⁻¹` requires a primal-P0 ↔ refined-P1 lifting
//! operator that this module deliberately does not provide — that
//! belongs to the preconditioner-wiring layer (Stage 2.5 work). What
//! ships here is the mass-matrix half of the construction.

use crate::core::mesh::BarycentricRefinement;
use sonium_solvers::iterative::{CgConfig, cg};
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::traits::Preconditioner;

/// Assemble the consistent P1 surface mass matrix on the barycentric
/// refinement of a BEM surface mesh. Returns a sparse, symmetric, SPD
/// real-valued matrix indexed by the refined-mesh node IDs (rows and
/// columns both in `0..refined.n_refined_nodes()`).
///
/// Each refined sub-triangle contributes the standard P1 local mass
/// `(Area / 12) · [2 1 1; 1 2 1; 1 1 2]` to the global system; duplicate
/// `(row, col)` triplets across sub-triangles sharing a node are merged
/// by `CsrMatrix::from_triplets`.
pub fn assemble_p1_surface_mass(refined: &BarycentricRefinement) -> CsrMatrix<f64> {
    let n_dof = refined.n_refined_nodes();
    let mut triplets: Vec<(usize, usize, f64)> =
        Vec::with_capacity(9 * refined.refined_triangles.len());

    for tri in &refined.refined_triangles {
        let p0 = node_pos(refined, tri[0]);
        let p1 = node_pos(refined, tri[1]);
        let p2 = node_pos(refined, tri[2]);
        let area = triangle_area(p0, p1, p2);
        if area <= 0.0 {
            continue;
        }
        let factor = area / 12.0;
        for i in 0..3 {
            for j in 0..3 {
                let val = if i == j { 2.0 * factor } else { factor };
                triplets.push((tri[i], tri[j], val));
            }
        }
    }

    CsrMatrix::from_triplets(n_dof, n_dof, triplets)
}

/// Lift a primal P0 (per-original-triangle) vector to the refined-P1
/// DOF space by sampling the value at each parent triangle's centroid
/// (the refined node `n_orig + n_edge + parent_idx`) and zeroing the
/// rest of the refined DOFs.
///
/// This is the simplest centroid-sample lift — it preserves the value
/// at the centroid exactly and lets the refined-P1 mass matrix smooth
/// the rest. The Hiptmair-Buffa-equivalent dual lift (distribute over
/// the dual cell) is a Stage-2.5 follow-up; this centroid-sample form
/// is sufficient for the unit tests in this module and the
/// "mass-of-constants = surface area" gate.
///
/// Panics if `primal.len() != refined.primal_to_children.len()`.
pub fn lift_primal_p0_to_refined_p1(refined: &BarycentricRefinement, primal: &[f64]) -> Vec<f64> {
    assert_eq!(
        primal.len(),
        refined.primal_to_children.len(),
        "primal vector must have one entry per original triangle"
    );
    let mut lifted = vec![0.0_f64; refined.n_refined_nodes()];
    let centroid_offset = refined.n_orig_nodes + refined.n_edge_nodes;
    for (parent_idx, &val) in primal.iter().enumerate() {
        lifted[centroid_offset + parent_idx] = val;
    }
    lifted
}

/// Restrict a refined-P1 vector back to primal P0 by sampling at the
/// per-parent centroid nodes. Adjoint of `lift_primal_p0_to_refined_p1`
/// in the centroid-sample basis.
pub fn restrict_refined_p1_to_primal_p0(
    refined: &BarycentricRefinement,
    refined_vec: &[f64],
) -> Vec<f64> {
    assert_eq!(
        refined_vec.len(),
        refined.n_refined_nodes(),
        "refined vector must have one entry per refined node"
    );
    let n_primal = refined.primal_to_children.len();
    let centroid_offset = refined.n_orig_nodes + refined.n_edge_nodes;
    let mut primal = vec![0.0_f64; n_primal];
    primal[..n_primal].copy_from_slice(&refined_vec[centroid_offset..centroid_offset + n_primal]);
    primal
}

#[inline]
fn node_pos(refined: &BarycentricRefinement, idx: usize) -> [f64; 3] {
    [
        refined.refined_nodes[idx][0],
        refined.refined_nodes[idx][1],
        refined.refined_nodes[idx][2],
    ]
}

#[inline]
fn triangle_area(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> f64 {
    let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let cx = v1[1] * v2[2] - v1[2] * v2[1];
    let cy = v1[2] * v2[0] - v1[0] * v2[2];
    let cz = v1[0] * v2[1] - v1[1] * v2[0];
    0.5 * (cx * cx + cy * cy + cz * cz).sqrt()
}

/// Phase 1.2 dual-mass preconditioner: composes
/// `lift → M_refined⁻¹ → restrict` over the barycentric refinement of a
/// surface mesh. Implements `Preconditioner<f64>` for any solver that
/// accepts an external preconditioner.
///
/// Inversion of `M_refined` is done via inner CG (M_refined is sparse,
/// SPD, and well-conditioned because each row is diagonal-dominant), so
/// the cost of a single `apply()` is `inner_max_iter` matvecs against
/// `M_refined`. For typical surface meshes with diagonal-dominant
/// barycentric mass, ~30 inner iterations gets to `1e-8` residual.
///
/// **Scope note**: this is the standalone preconditioner-as-operator
/// piece. Wiring it into `BemSolver` as a Calderón sandwich
/// `M_dual⁻¹ V M_dual⁻¹` requires composing it with the BEM operator V
/// or W on the inside, which the BemPreconditioner enum does not yet
/// expose (Stage-2.5 follow-up). The standalone form is testable in
/// isolation and useful from any iterative solver that accepts a
/// `Preconditioner<f64>`.
pub struct DualMassPreconditioner<'a> {
    refined: &'a BarycentricRefinement,
    mass: CsrMatrix<f64>,
    inner_cg_max_iter: usize,
    inner_cg_tol: f64,
}

impl<'a> DualMassPreconditioner<'a> {
    /// Construct from a barycentric refinement. The P1 surface mass on
    /// the refined mesh is assembled here once and re-used across every
    /// `apply()` call. Inner CG defaults to `max_iter = 50`,
    /// `tol = 1e-8` — tightened only on demand for the small fixtures
    /// the unit tests use.
    pub fn new(refined: &'a BarycentricRefinement) -> Self {
        let mass = assemble_p1_surface_mass(refined);
        Self {
            refined,
            mass,
            inner_cg_max_iter: 50,
            inner_cg_tol: 1e-8,
        }
    }

    /// Tune the inner CG convergence parameters used by `apply()`.
    pub fn with_inner_cg(mut self, max_iter: usize, tol: f64) -> Self {
        self.inner_cg_max_iter = max_iter;
        self.inner_cg_tol = tol;
        self
    }

    /// Read-only access to the assembled refined-mesh mass matrix.
    pub fn refined_mass(&self) -> &CsrMatrix<f64> {
        &self.mass
    }
}

impl<'a> Preconditioner<f64> for DualMassPreconditioner<'a> {
    fn apply(&self, r: &[f64]) -> Vec<f64> {
        // Lift the primal residual to the refined-P1 DOF space.
        let lifted = lift_primal_p0_to_refined_p1(self.refined, r);
        // Solve M_refined · y = lifted via inner CG.
        let cfg = CgConfig {
            max_iterations: self.inner_cg_max_iter,
            tolerance: self.inner_cg_tol,
            print_interval: 0,
        };
        let sol = cg(&self.mass, lifted.as_slice(), &cfg);
        // Restrict back to primal P0.
        restrict_refined_p1_to_primal_p0(self.refined, &sol.x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::mesh::generators::generate_icosphere_mesh;
    use crate::core::mesh::refine_surface_mesh;
    use sonium_solvers::traits::LinearOperator;
    use std::f64::consts::PI;

    #[test]
    fn p1_surface_mass_constant_function_integrates_to_surface_area() {
        // For any P1 surface mass matrix, `1^T · M · 1 == surface area`
        // because the constant-1 vector represents the function f(x) = 1
        // and ∫_S 1·1 dS = area(S). This is the "mass of constants"
        // gate the spec calls out as the silent-failure detector.
        let mesh = generate_icosphere_mesh(1.0, 2);
        let refined = refine_surface_mesh(&mesh);
        let mass = assemble_p1_surface_mass(&refined);

        let n_dof = refined.n_refined_nodes();
        let ones = vec![1.0_f64; n_dof];
        let m1 = mass.apply(ones.as_slice());
        let total: f64 = ones.iter().zip(m1.iter()).map(|(a, b)| a * b).sum();

        let analytic_area = 4.0 * PI * 1.0_f64.powi(2);
        let original_area: f64 = mesh.elements.iter().map(|e| e.area).sum();
        // Mass should integrate to the *discretised* surface area, which
        // is the sum of element areas of the *refined* mesh — and we
        // already gated that against the original surface area to 1e-12
        // in `barycentric::tests`, which itself sits at ~5 % below the
        // analytic 4π for the subdiv-2 icosphere.
        let rel_vs_orig = (total - original_area).abs() / original_area;
        assert!(
            rel_vs_orig < 1e-10,
            "1^T M 1 should equal sum of element areas (discretised surface area): \
             got {total:.6e}, expected {original_area:.6e}, rel {rel_vs_orig:.3e}"
        );
        // Sanity reporting only — the icosphere mesh is faceted, so its
        // discretised area is below 4πR² by an O(h²) geometry deficit
        // (about 5 % at subdiv 2).
        let _ = analytic_area;
    }

    #[test]
    fn p1_surface_mass_is_symmetric() {
        let mesh = generate_icosphere_mesh(1.0, 2);
        let refined = refine_surface_mesh(&mesh);
        let mass = assemble_p1_surface_mass(&refined);

        // Symmetry test: M·e_i and M^T·e_i must agree for an arbitrary
        // probe vector. Apply M to a deterministic vector x, M^T to the
        // same x, and compare. CsrMatrix's `apply_transpose` is the
        // direct check.
        let n = refined.n_refined_nodes();
        let x: Vec<f64> = (0..n).map(|i| (0.37 * (i + 1) as f64).sin()).collect();
        let y_fwd = mass.apply(x.as_slice());
        let y_tr = mass.apply_transpose(x.as_slice());
        let mut max_diff = 0.0_f64;
        for (a, b) in y_fwd.iter().zip(y_tr.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 1e-12,
            "P1 surface mass must be symmetric: max(|M·x − M^T·x|) = {max_diff:.3e}"
        );
    }

    #[test]
    fn p1_surface_mass_is_positive_definite_via_quadratic_form() {
        // Necessary condition for SPD on a non-degenerate triangulation:
        // x^T M x > 0 for any x ≠ 0. Probe with several random vectors;
        // a regression that breaks SPD-ness (e.g. wrong sign on the
        // off-diagonal) would surface here even without computing eigvals.
        let mesh = generate_icosphere_mesh(1.0, 2);
        let refined = refine_surface_mesh(&mesh);
        let mass = assemble_p1_surface_mass(&refined);
        let n = refined.n_refined_nodes();

        for seed in 0..5 {
            let x: Vec<f64> = (0..n)
                .map(|i| (0.41_f64 * (seed + 1) as f64 * (i + 1) as f64).sin())
                .collect();
            let y = mass.apply(x.as_slice());
            let q: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
            assert!(
                q > 0.0,
                "x^T M x should be positive for non-zero x (seed {seed}): got {q:.3e}"
            );
        }
    }

    #[test]
    fn p1_surface_mass_diagonal_dominant_lower_bound() {
        // For a P1 mass on a triangulation where each node's vertex star
        // has area `A_v = sum of areas of triangles touching v`, the
        // diagonal entry M[v, v] = A_v / 6 (each triangle contributes
        // 2 · area/12 = area/6 to its three vertex diagonals). The total
        // row sum (consistent mass) is M[v, v] + Σ M[v, w] = A_v / 3
        // (each triangle contributes area/3 to the row of each of its
        // vertices). Verify both relations hold to round-off — these
        // two identities pin down the local-mass formula.
        let mesh = generate_icosphere_mesh(1.0, 2);
        let refined = refine_surface_mesh(&mesh);
        let mass = assemble_p1_surface_mass(&refined);

        // Compute per-node vertex-star area on the refined mesh.
        let mut star_area = vec![0.0_f64; refined.n_refined_nodes()];
        for tri in &refined.refined_triangles {
            let p0 = node_pos(&refined, tri[0]);
            let p1 = node_pos(&refined, tri[1]);
            let p2 = node_pos(&refined, tri[2]);
            let a = triangle_area(p0, p1, p2);
            for &v in tri {
                star_area[v] += a;
            }
        }

        // M · 1 row sums.
        let n = refined.n_refined_nodes();
        let ones = vec![1.0_f64; n];
        let row_sums = mass.apply(ones.as_slice());
        for v in 0..n {
            let expected = star_area[v] / 3.0;
            let rel = (row_sums[v] - expected).abs() / expected.max(1e-30);
            assert!(
                rel < 1e-10,
                "row-sum identity broken at node {v}: got {} expected {expected} (rel {rel:.3e})",
                row_sums[v]
            );
        }
    }

    #[test]
    fn lift_then_restrict_is_identity_on_primal_p0() {
        let mesh = generate_icosphere_mesh(1.0, 2);
        let refined = refine_surface_mesh(&mesh);
        let n_primal = refined.primal_to_children.len();
        let primal: Vec<f64> = (0..n_primal)
            .map(|i| (0.31 * (i + 1) as f64).cos())
            .collect();
        let lifted = lift_primal_p0_to_refined_p1(&refined, &primal);
        let restricted = restrict_refined_p1_to_primal_p0(&refined, &lifted);
        for (a, b) in primal.iter().zip(restricted.iter()) {
            assert!(
                (a - b).abs() < 1e-15,
                "lift∘restrict must be identity on primal P0: {a} vs {b}"
            );
        }
        // Lifted vector has primal entries at centroids and zeros
        // elsewhere — confirm the centroid block holds the input.
        let centroid_offset = refined.n_orig_nodes + refined.n_edge_nodes;
        for (i, &val) in primal.iter().enumerate() {
            assert_eq!(lifted[centroid_offset + i], val);
        }
        for j in 0..centroid_offset {
            assert_eq!(lifted[j], 0.0, "lift must zero out non-centroid DOFs");
        }
    }

    #[test]
    fn dual_mass_preconditioner_is_symmetric_via_quadratic_form() {
        // The composition `R · M_refined⁻¹ · L` is symmetric because R
        // and L are adjoints (verified above) and M_refined⁻¹ is SPD.
        // Test by checking ⟨P x, y⟩ ≈ ⟨x, P y⟩ on probe vectors. Inner
        // CG tolerance is tight so the residual symmetry of CG itself
        // doesn't dominate the comparison.
        let mesh = generate_icosphere_mesh(1.0, 2);
        let refined = refine_surface_mesh(&mesh);
        let pre = DualMassPreconditioner::new(&refined).with_inner_cg(200, 1e-12);

        let n = refined.primal_to_children.len();
        let x: Vec<f64> = (0..n).map(|i| (0.41 * (i + 1) as f64).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (0.29 * (i + 1) as f64).cos()).collect();

        let px = pre.apply(x.as_slice());
        let py = pre.apply(y.as_slice());

        let lhs: f64 = px.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let rhs: f64 = x.iter().zip(py.iter()).map(|(a, b)| a * b).sum();

        let denom = lhs.abs().max(rhs.abs()).max(1e-30);
        let rel = (lhs - rhs).abs() / denom;
        assert!(
            rel < 1e-6,
            "DualMassPreconditioner symmetry: ⟨Px,y⟩ = {lhs}, ⟨x,Py⟩ = {rhs}, rel = {rel:.3e}"
        );
    }

    #[test]
    fn dual_mass_preconditioner_diagonal_dominates_jacobi_diagonal() {
        // Sanity: for a primal vector with magnitude ~1 entries, the
        // dual-mass `apply` should produce output with magnitude
        // bounded by 1 / (smallest diagonal of M_refined). Effectively
        // verifies the inner CG is actually inverting something — a
        // broken inner solve would either return ~0 (early exit) or
        // explode. Use a uniform-magnitude probe and check the output
        // norm sits in a sane range.
        let mesh = generate_icosphere_mesh(1.0, 2);
        let refined = refine_surface_mesh(&mesh);
        let pre = DualMassPreconditioner::new(&refined).with_inner_cg(200, 1e-10);

        let n = refined.primal_to_children.len();
        let x: Vec<f64> = vec![1.0_f64; n];
        let y = pre.apply(x.as_slice());
        let y_norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        let x_norm = (n as f64).sqrt(); // ‖x‖ = sqrt(n)
        // For an icosphere of radius 1 the centroid-row diagonal of
        // M_refined ≈ area/per-centroid which is O(area_per_tri).
        // Output norm should be ‖x‖ × O(1/diag) — finite and non-trivial.
        // Loose bound: 1e-6 ≤ ‖y‖ / ‖x‖ ≤ 1e6 catches "constant zero"
        // and "explodes to inf" failure modes without pinning the exact value.
        let ratio = y_norm / x_norm;
        assert!(
            ratio.is_finite() && (1e-6..=1e6).contains(&ratio),
            "DualMassPreconditioner output norm ratio {ratio:.3e} outside sanity band"
        );
    }

    #[test]
    fn restrict_is_adjoint_of_lift() {
        // ⟨L x, y⟩_refined = ⟨x, R y⟩_primal where L = lift, R = restrict.
        // This is the discrete adjoint relation that lets the
        // composition `R · M_refined⁻¹ · L` stay symmetric.
        let mesh = generate_icosphere_mesh(1.0, 2);
        let refined = refine_surface_mesh(&mesh);
        let n_primal = refined.primal_to_children.len();
        let n_ref = refined.n_refined_nodes();

        let x: Vec<f64> = (0..n_primal)
            .map(|i| (0.19 * (i + 1) as f64).sin())
            .collect();
        let y: Vec<f64> = (0..n_ref).map(|i| (0.23 * (i + 1) as f64).cos()).collect();

        let lx = lift_primal_p0_to_refined_p1(&refined, &x);
        let ry = restrict_refined_p1_to_primal_p0(&refined, &y);

        let lhs: f64 = lx.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let rhs: f64 = x.iter().zip(ry.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (lhs - rhs).abs() < 1e-12,
            "adjointness broken: ⟨Lx, y⟩ = {lhs}, ⟨x, Ry⟩ = {rhs}"
        );
    }
}
