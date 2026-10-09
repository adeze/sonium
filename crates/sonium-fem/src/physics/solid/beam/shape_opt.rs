//! Shape optimisation of H-beam cross-section via adjoint method.
//!
//! Finds the optimal H-section parameters (b, h, t) that minimise tip deflection
//! of a cantilever beam under a point load, subject to a cross-sectional area constraint.
//!
//! # Lagrangian formulation
//!
//! The PDE-constrained optimisation is:
//!
//! ```text
//! min  f(u) = w_tip          (tip deflection)
//!  p
//! s.t. K(p) u = F            (state equation)
//!      A(p) ≤ A_max          (material budget)
//!      p_lo ≤ p ≤ p_hi       (manufacturing bounds)
//! ```
//!
//! The Lagrangian with adjoint variable λ:
//!
//! ```text
//! L(u, λ, p) = f(u) + λᵀ (K(p) u − F)
//! ```
//!
//! Stationarity conditions (saddle point):
//!
//! ```text
//! ∂L/∂λ = 0  →  K(p) u = F                     (forward / state)
//! ∂L/∂u = 0  →  Kᵀ λ = −∂f/∂u                  (adjoint)
//! ∂L/∂p_i    =  λᵀ (∂K/∂p_i) u                  (sensitivity)
//! ```
//!
//! Because K is symmetric, the adjoint reuses the same factorisation.
//! The stiffness matrix is linear in (I, A), so `∂K/∂p_i` decomposes as:
//!
//! ```text
//! ∂K/∂p_i = (∂I/∂p_i / I) K_bending + (∂A/∂p_i / A) K_axial
//! ```

use sonium_solvers::{
    AugmentedLagrangianConfig, Bound, CgConfig, Constraint, ConstraintKind, OptimResult,
    augmented_lagrangian, dense::DenseMatrix,
};

use super::element::{BeamElement2D, assemble_beam_stiffness};
use super::section::{BeamSection, HSection};

/// Problem definition for cantilever H-beam shape optimisation.
#[derive(Debug, Clone)]
pub struct HBeamShapeOpt {
    /// Beam length (m).
    pub length: f64,
    /// Number of elements along the beam.
    pub n_elements: usize,
    /// Young's modulus (Pa).
    pub youngs_modulus: f64,
    /// Density (kg/m³).
    pub density: f64,
    /// Tip load (N), positive downward.
    pub tip_load: f64,
    /// Maximum cross-sectional area (m²).
    pub area_max: f64,
    /// Bounds on design variables: [(b_lo, b_hi), (h_lo, h_hi), (t_lo, t_hi)].
    pub bounds: [(f64, f64); 3],
}

/// Result of shape optimisation.
#[derive(Debug, Clone)]
pub struct ShapeOptResult {
    /// Optimal H-section parameters.
    pub section: HSection,
    /// Tip deflection at optimum (m).
    pub tip_deflection: f64,
    /// Cross-sectional area at optimum (m²).
    pub area: f64,
    /// Optimiser result (iterations, convergence, etc.).
    pub optim: OptimResult<f64>,
}

impl HBeamShapeOpt {
    /// Build beam elements for a given H-section.
    fn build_elements(&self, section: &HSection) -> Vec<BeamElement2D> {
        let beam_section = section.to_beam_section(self.youngs_modulus, self.density);
        let dx = self.length / self.n_elements as f64;
        (0..self.n_elements)
            .map(|i| {
                BeamElement2D::new(
                    [i, i + 1],
                    i as f64 * dx,
                    0.0,
                    (i + 1) as f64 * dx,
                    0.0,
                    beam_section,
                )
            })
            .collect()
    }

    fn n_dofs(&self) -> usize {
        (self.n_elements + 1) * 3
    }

    /// Tip transverse-displacement DOF index.
    fn tip_w_dof(&self) -> usize {
        self.n_elements * 3 + 1 // w at last node
    }

    /// Assemble stiffness, apply clamped BC, solve, return full solution vector.
    fn forward_solve(&self, elements: &[BeamElement2D]) -> Vec<f64> {
        let n = self.n_dofs();
        let triplet = assemble_beam_stiffness(elements, n);

        let mut rhs = vec![0.0; n];
        rhs[self.tip_w_dof()] = -self.tip_load; // downward

        // Clamp node 0: u=0, v=0, θ=0
        let clamped = [0_usize, 1, 2];
        let (triplet, rhs) = apply_dirichlet(triplet, rhs, &clamped);

        let csr = triplet.to_csr();
        let result = sonium_solvers::cg(
            &csr,
            &rhs,
            &CgConfig {
                max_iterations: 1000,
                tolerance: 1e-10,
                print_interval: 0,
            },
        );
        assert!(
            result.converged,
            "forward solve did not converge: residual={}, iterations={}, status={:?}",
            result.residual, result.iterations, result.status
        );
        result.x
    }

    /// Assemble separate bending and axial stiffness matrices (both with Dirichlet BC).
    fn assemble_bending_and_axial(
        &self,
        elements: &[BeamElement2D],
    ) -> (
        sonium_solvers::CsrMatrix<f64>,
        sonium_solvers::CsrMatrix<f64>,
    ) {
        let n = self.n_dofs();
        let clamped = [0_usize, 1, 2];

        // Build bending-only elements (A=0 → no axial contribution)
        let section = elements[0].section;
        let bending_section = BeamSection::euler_bernoulli(
            section.youngs_modulus,
            section.moment_of_inertia,
            0.0, // zero area eliminates axial stiffness
            0.0,
        );
        let axial_section = BeamSection::euler_bernoulli(
            section.youngs_modulus,
            0.0, // zero I eliminates bending stiffness
            section.area,
            0.0,
        );

        let dx = self.length / self.n_elements as f64;
        let make_elems = |sec: BeamSection| -> Vec<BeamElement2D> {
            (0..self.n_elements)
                .map(|i| {
                    BeamElement2D::new(
                        [i, i + 1],
                        i as f64 * dx,
                        0.0,
                        (i + 1) as f64 * dx,
                        0.0,
                        sec,
                    )
                })
                .collect()
        };

        let bending_elems = make_elems(bending_section);
        let axial_elems = make_elems(axial_section);

        let k_bend = assemble_beam_stiffness(&bending_elems, n);
        let k_axial = assemble_beam_stiffness(&axial_elems, n);

        // Apply Dirichlet to both (zero rows/cols, identity on diagonal)
        let rhs_dummy = vec![0.0; n];
        let (k_bend, _) = apply_dirichlet(k_bend, rhs_dummy.clone(), &clamped);
        let (k_axial, _) = apply_dirichlet(k_axial, rhs_dummy, &clamped);

        (k_bend.to_csr(), k_axial.to_csr())
    }

    /// Evaluate objective f(p) = |w_tip| and its gradient via the adjoint method.
    ///
    /// Returns `(f, ∇f)` where `∇f = [∂f/∂b, ∂f/∂h, ∂f/∂t]`.
    pub fn evaluate(&self, params: &[f64; 3]) -> (f64, [f64; 3]) {
        let section = HSection::new(params[0], params[1], params[2]);
        let elements = self.build_elements(&section);

        // --- Forward solve: K u = F ---
        let u = self.forward_solve(&elements);
        let tip_deflection = -u[self.tip_w_dof()]; // positive downward

        // --- Adjoint solve: K λ = -∂f/∂u ---
        // f = -u[tip_w_dof] (we minimise tip deflection magnitude)
        // ∂f/∂u = -e_{tip_w_dof}
        let n = self.n_dofs();
        let mut df_du = vec![0.0_f64; n];
        df_du[self.tip_w_dof()] = -1.0;

        // K is symmetric so adjoint solve = forward solve with -df_du as RHS
        let neg_df_du: Vec<f64> = df_du.iter().map(|v| -*v).collect();
        let triplet = assemble_beam_stiffness(&elements, n);
        let clamped = [0_usize, 1, 2];
        let (triplet_bc, _) = apply_dirichlet(triplet, vec![0.0; n], &clamped);
        let csr = triplet_bc.to_csr();
        let adjoint_result = sonium_solvers::cg(
            &csr,
            &neg_df_du,
            &CgConfig {
                max_iterations: 1000,
                tolerance: 1e-10,
                print_interval: 0,
            },
        );
        assert!(adjoint_result.converged, "adjoint solve did not converge");
        let lambda = adjoint_result.x;

        // --- Sensitivity: ∂L/∂p_i = λᵀ (∂K/∂p_i) u ---
        //
        // K = (I) * K_bending_unit + (A) * K_axial_unit
        // where K_*_unit are assembled with unit I or unit A.
        //
        // ∂K/∂p_i = (∂I/∂p_i) * K_bending_unit + (∂A/∂p_i) * K_axial_unit
        //         = (∂I/∂p_i / I) * K_bending  + (∂A/∂p_i / A) * K_axial
        //
        // So: λᵀ (∂K/∂p_i) u = (∂I/∂p_i / I) * λᵀ K_bending u
        //                     + (∂A/∂p_i / A) * λᵀ K_axial u

        let (k_bend_csr, k_axial_csr) = self.assemble_bending_and_axial(&elements);
        let k_bend_u = spmv(&k_bend_csr, &u);
        let k_axial_u = spmv(&k_axial_csr, &u);

        let lambda_k_bend_u = lambda
            .iter()
            .zip(k_bend_u.iter())
            .map(|(l, ku)| l * ku)
            .sum::<f64>();
        let lambda_k_axial_u = lambda
            .iter()
            .zip(k_axial_u.iter())
            .map(|(l, ku)| l * ku)
            .sum::<f64>();

        let ii = section.moment_of_inertia();
        let aa = section.area();
        assert!(
            ii > 0.0,
            "moment of inertia is zero — check HSection parameters"
        );
        assert!(
            aa > 0.0,
            "cross-sectional area is zero — check HSection parameters"
        );
        let di = section.inertia_derivatives();
        let da = section.area_derivatives();

        let mut grad = [0.0; 3];
        for i in 0..3 {
            grad[i] = (di[i] / ii) * lambda_k_bend_u + (da[i] / aa) * lambda_k_axial_u;
        }

        (tip_deflection, grad)
    }

    /// Run the shape optimisation.
    ///
    /// Returns the optimal H-section, tip deflection, and optimiser diagnostics.
    pub fn optimise(&self, initial: &HSection) -> ShapeOptResult {
        let x0 = vec![initial.b, initial.h, initial.t];

        // Area constraint: A(p) ≤ A_max  →  A_max − A(p) ≥ 0
        let area_max = self.area_max;
        let area_constraint = Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(move |x: &[f64]| {
                let sec = HSection::new(x[0], x[1], x[2]);
                vec![area_max - sec.area()]
            }),
            jac: Some(Box::new(move |x: &[f64]| {
                let sec = HSection::new(x[0], x[1], x[2]);
                let da = sec.area_derivatives();
                DenseMatrix::from_row_major_vec(1, 3, vec![-da[0], -da[1], -da[2]])
            })),
        };

        let bounds = vec![
            Bound::both(self.bounds[0].0, self.bounds[0].1),
            Bound::both(self.bounds[1].0, self.bounds[1].1),
            Bound::both(self.bounds[2].0, self.bounds[2].1),
        ];

        // Objective: minimise tip deflection
        let objective = |x: &[f64]| -> f64 {
            let (f, _) = self.evaluate(&[x[0], x[1], x[2]]);
            f
        };

        let gradient = |x: &[f64]| -> Vec<f64> {
            let (_, g) = self.evaluate(&[x[0], x[1], x[2]]);
            g.to_vec()
        };

        let config = AugmentedLagrangianConfig {
            max_outer_iterations: 50,
            max_inner_iterations: 200,
            grad_tolerance: 1e-8,
            constraint_tolerance: 1e-8,
            inner_solver: sonium_solvers::InnerSolver::Lbfgsb,
            bounds,
            ..Default::default()
        };

        let result = augmented_lagrangian(&objective, &gradient, &x0, &[area_constraint], &config)
            .expect("optimisation failed");

        let opt_section = HSection::new(result.x[0], result.x[1], result.x[2]);
        let tip_deflection = result.fun;

        ShapeOptResult {
            section: opt_section,
            tip_deflection,
            area: opt_section.area(),
            optim: result,
        }
    }
}

/// Sparse matrix-vector product for CsrMatrix.
fn spmv(csr: &sonium_solvers::CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    csr.matvec(x)
}

/// Apply homogeneous Dirichlet BC by zeroing rows/cols and setting diagonal to 1.
fn apply_dirichlet(
    mut triplet: crate::assembly::TripletMatrix<f64>,
    mut rhs: Vec<f64>,
    clamped_dofs: &[usize],
) -> (crate::assembly::TripletMatrix<f64>, Vec<f64>) {
    let clamped: std::collections::HashSet<usize> = clamped_dofs.iter().copied().collect();

    let mut new_r = Vec::new();
    let mut new_c = Vec::new();
    let mut new_v = Vec::new();

    for i in 0..triplet.nnz() {
        let r = triplet.rows[i];
        let c = triplet.cols[i];
        if !clamped.contains(&r) && !clamped.contains(&c) {
            new_r.push(r);
            new_c.push(c);
            new_v.push(triplet.values[i]);
        }
    }
    for &d in clamped_dofs {
        new_r.push(d);
        new_c.push(d);
        new_v.push(1.0);
        rhs[d] = 0.0;
    }
    triplet.rows = new_r;
    triplet.cols = new_c;
    triplet.values = new_v;

    (triplet, rhs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Validate H-section area and I against manual computation.
    #[test]
    fn test_hsection_properties() {
        // b=0.1, h=0.2, t=0.01
        let sec = HSection::new(0.1, 0.2, 0.01);
        // A = t(2b + h) = 0.01 * (0.2 + 0.2) = 0.004
        assert!((sec.area() - 0.004).abs() < 1e-12);

        // I = [b(h+2t)³ − (b−t)h³] / 12
        //   = [0.1 * 0.22³ − 0.09 * 0.2³] / 12
        let big_h = 0.22_f64;
        let expected_i = (0.1 * big_h.powi(3) - 0.09 * 0.2_f64.powi(3)) / 12.0;
        assert!(
            (sec.moment_of_inertia() - expected_i).abs() < 1e-15,
            "I: got {}, expected {}",
            sec.moment_of_inertia(),
            expected_i
        );
    }

    /// Validate analytical derivatives against finite differences.
    #[test]
    fn test_hsection_derivatives_vs_fd() {
        let sec = HSection::new(0.05, 0.10, 0.005);
        let eps = 1e-7;

        let di_analytical = sec.inertia_derivatives();
        let da_analytical = sec.area_derivatives();

        // Finite difference for each parameter
        let params = [sec.b, sec.h, sec.t];
        for i in 0..3 {
            let mut p_plus = params;
            let mut p_minus = params;
            p_plus[i] += eps;
            p_minus[i] -= eps;
            let s_plus = HSection::new(p_plus[0], p_plus[1], p_plus[2]);
            let s_minus = HSection::new(p_minus[0], p_minus[1], p_minus[2]);

            let di_fd = (s_plus.moment_of_inertia() - s_minus.moment_of_inertia()) / (2.0 * eps);
            let da_fd = (s_plus.area() - s_minus.area()) / (2.0 * eps);

            let rel_i = (di_analytical[i] - di_fd).abs() / di_analytical[i].abs().max(1e-15);
            let rel_a = (da_analytical[i] - da_fd).abs() / da_analytical[i].abs().max(1e-15);

            assert!(
                rel_i < 1e-5,
                "dI/dp[{i}]: analytical={}, fd={}, rel_err={rel_i:.2e}",
                di_analytical[i],
                di_fd
            );
            assert!(
                rel_a < 1e-5,
                "dA/dp[{i}]: analytical={}, fd={}, rel_err={rel_a:.2e}",
                da_analytical[i],
                da_fd
            );
        }
    }

    /// Validate the full adjoint gradient against finite differences.
    #[test]
    fn test_adjoint_gradient_vs_fd() {
        let problem = HBeamShapeOpt {
            length: 1.0,
            n_elements: 10,
            youngs_modulus: 200e9,
            density: 7800.0,
            tip_load: 1000.0,
            area_max: 0.01,
            bounds: [(0.01, 0.1), (0.02, 0.2), (0.002, 0.02)],
        };

        let p0 = [0.05, 0.10, 0.005];
        let (f0, grad) = problem.evaluate(&p0);
        let eps = 1e-7;

        for i in 0..3 {
            let mut p_plus = p0;
            let mut p_minus = p0;
            p_plus[i] += eps;
            p_minus[i] -= eps;
            let (f_plus, _) = problem.evaluate(&p_plus);
            let (f_minus, _) = problem.evaluate(&p_minus);
            let fd = (f_plus - f_minus) / (2.0 * eps);

            let rel_err = (grad[i] - fd).abs() / fd.abs().max(1e-15);
            assert!(
                rel_err < 1e-4,
                "∂f/∂p[{i}]: adjoint={:.6e}, fd={:.6e}, rel_err={rel_err:.2e} (f0={f0:.6e})",
                grad[i],
                fd,
            );
        }
    }

    /// Run the full optimisation and verify convergence.
    #[test]
    fn test_shape_optimisation_converges() {
        let problem = HBeamShapeOpt {
            length: 1.0,
            n_elements: 10,
            youngs_modulus: 200e9,
            density: 7800.0,
            tip_load: 1000.0,
            area_max: 0.004, // constrained area budget
            bounds: [(0.01, 0.10), (0.02, 0.20), (0.002, 0.015)],
        };

        // Start from a feasible but non-optimal point
        let initial = HSection::new(0.04, 0.06, 0.008);
        assert!(
            initial.area() <= problem.area_max + 1e-10,
            "initial must be feasible: A={} > A_max={}",
            initial.area(),
            problem.area_max
        );

        let result = problem.optimise(&initial);

        // The optimised deflection should be less than the initial
        let (initial_defl, _) = problem.evaluate(&[initial.b, initial.h, initial.t]);

        assert!(
            result.tip_deflection < initial_defl,
            "optimised deflection ({:.6e}) should be less than initial ({:.6e})",
            result.tip_deflection,
            initial_defl,
        );

        // Area constraint should be active (or near-active) at optimum
        assert!(
            result.area <= problem.area_max + 1e-6,
            "area constraint violated: {} > {}",
            result.area,
            problem.area_max
        );

        // For a bending-dominated problem, optimal H-section should have:
        // - large h (maximize moment arm)
        // - b driven by area constraint
        // - t near minimum (thin flanges/web)
        assert!(
            result.section.h > result.section.b,
            "optimal h ({}) should exceed b ({}) for bending-dominated beam",
            result.section.h,
            result.section.b
        );
    }
}
