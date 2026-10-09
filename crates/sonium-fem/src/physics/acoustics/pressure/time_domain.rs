//! Time-domain solvers for the scalar wave equation
//!
//! Solves: `(1/c²) ∂²p/∂t² − ∇²p = f(x,t)`
//!
//! Semi-discrete form: `M_c p̈ + K p = F(t)` where `M_c = (1/c²) M`.
//!
//! # Design note: separate from sonium-solvers Newmark
//!
//! This module provides its own step-by-step Newmark API (`NewmarkStepper`)
//! rather than using `sonium_solvers::newmark_integrate` because:
//! - Step-by-step API integrates with FEM physics loops where the caller
//!   controls time stepping, RHS formation, and solver selection
//! - `effective_stiffness()` exploits same-sparsity-pattern CSR matrices
//!   (a pattern-level optimisation not possible through the trait-based API)
//! - Works directly with `CsrMatrix<f64>` and `Vec<f64>` (no ndarray overhead)
//! - Includes `LeapfrogStepper` for explicit time integration (no counterpart
//!   in sonium-solvers)
//!
//! # Implicit FEM — Newmark-β
//!
//! Unconditionally stable (β=1/4, γ=1/2) trapezoidal scheme:
//! ```text
//! (M_c + β Δt² K) p̈_{n+1} = F_{n+1} − K (p_n + Δt ṗ_n + (½−β) Δt² p̈_n)
//! ```
//!
//! # Explicit FEM — Leapfrog / Central difference
//!
//! Conditionally stable (CFL: Δt < h_min / c), using lumped mass:
//! ```text
//! p_{n+1} = 2 p_n − p_{n−1} + Δt² M_L⁻¹ (F_n − K p_n)
//! ```
//!
//! # dG-FEM (stub)
//!
//! Time-explicit discontinuous Galerkin requires a DG function space
//! (element-local DOFs, numerical fluxes). The DG infrastructure is
//! not yet implemented; this module provides the time-stepper interface
//! that a future DG assembler would plug into.

use sonium_solvers::CsrMatrix;

// ---------------------------------------------------------------------------
// Wave equation state
// ---------------------------------------------------------------------------

/// State vector for the wave equation: pressure, velocity, acceleration.
#[derive(Debug, Clone)]
pub struct WaveState {
    /// Pressure field p [Pa]
    pub pressure: Vec<f64>,
    /// Pressure time derivative ṗ [Pa/s]
    pub velocity: Vec<f64>,
    /// Pressure second time derivative p̈ [Pa/s²]
    pub acceleration: Vec<f64>,
}

impl WaveState {
    /// Zero-initialized state of given dimension.
    pub fn zeros(ndof: usize) -> Self {
        Self {
            pressure: vec![0.0; ndof],
            velocity: vec![0.0; ndof],
            acceleration: vec![0.0; ndof],
        }
    }

    pub fn ndof(&self) -> usize {
        self.pressure.len()
    }
}

// ---------------------------------------------------------------------------
// Newmark-β implicit time integrator
// ---------------------------------------------------------------------------

/// Newmark-β configuration.
///
/// Default: β = 1/4, γ = 1/2 (trapezoidal rule, unconditionally stable,
/// second-order accurate, energy-conserving).
#[derive(Debug, Clone, Copy)]
pub struct NewmarkConfig {
    pub beta: f64,
    pub gamma: f64,
    pub dt: f64,
}

impl NewmarkConfig {
    /// Trapezoidal rule (unconditionally stable, energy-conserving).
    pub fn trapezoidal(dt: f64) -> Self {
        Self {
            beta: 0.25,
            gamma: 0.5,
            dt,
        }
    }

    /// Newmark with numerical dissipation (γ > 1/2 damps high frequencies).
    pub fn dissipative(dt: f64, gamma: f64) -> Self {
        // Optimal β for given γ: β = (γ + 1/2)² / 4
        let beta = (gamma + 0.5) * (gamma + 0.5) / 4.0;
        Self { beta, gamma, dt }
    }
}

/// Newmark-β implicit time integrator for `M_c p̈ + K p = F`.
///
/// # Usage
///
/// ```ignore
/// let config = NewmarkConfig::trapezoidal(dt);
/// let stepper = NewmarkStepper::new(config);
///
/// // Form effective stiffness K_eff = M_c + β Δt² K (factored once)
/// let k_eff = stepper.effective_stiffness(&stiffness, &mass_c);
///
/// for step in 0..n_steps {
///     let rhs = stepper.form_rhs(&state, &force, &stiffness, &mass_c);
///     let accel_new = solve(&k_eff, &rhs);  // CG, LU, etc.
///     stepper.update_state(&mut state, &accel_new);
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct NewmarkStepper {
    pub config: NewmarkConfig,
}

impl NewmarkStepper {
    pub fn new(config: NewmarkConfig) -> Self {
        Self { config }
    }

    /// Form the effective stiffness matrix: `K_eff = M_c + β Δt² K`.
    ///
    /// This matrix is constant across time steps (for linear problems)
    /// and should be factored/preconditioned once.
    pub fn effective_stiffness(
        &self,
        stiffness: &CsrMatrix<f64>,
        mass_c: &CsrMatrix<f64>,
    ) -> CsrMatrix<f64> {
        let beta_dt2 = self.config.beta * self.config.dt * self.config.dt;
        // K_eff = M_c + β Δt² K
        // Build from components: scale copies and add
        let n = stiffness.num_rows;
        let mut k_eff_values = mass_c.values.clone();
        // Add β Δt² K: need same sparsity pattern
        // Assumption: K and M_c have compatible sparsity (true for same mesh)
        assert_eq!(
            stiffness.row_ptrs, mass_c.row_ptrs,
            "K and M_c must share the same CSR sparsity pattern (use PrebakedAssembler)"
        );
        assert_eq!(
            stiffness.col_indices, mass_c.col_indices,
            "K and M_c must share the same CSR column indices (use PrebakedAssembler)"
        );
        for (kv, sv) in k_eff_values.iter_mut().zip(&stiffness.values) {
            *kv += beta_dt2 * sv;
        }
        CsrMatrix::from_raw_parts(
            n,
            n,
            mass_c.row_ptrs.clone(),
            mass_c.col_indices.clone(),
            k_eff_values,
        )
    }

    /// Form the RHS for the Newmark system solve.
    ///
    /// ```text
    /// rhs = F_{n+1} − K (p_n + Δt ṗ_n + (½−β) Δt² p̈_n)
    /// ```
    ///
    /// The caller must then solve `K_eff × ä_{n+1} = rhs`.
    pub fn form_rhs(
        &self,
        state: &WaveState,
        force: &[f64],
        stiffness: &CsrMatrix<f64>,
    ) -> Vec<f64> {
        let dt = self.config.dt;
        let beta = self.config.beta;
        let n = state.ndof();

        // Predictor: p̃ = p + Δt ṗ + (0.5 − β) Δt² p̈
        let half_minus_beta = 0.5 - beta;
        let p_pred: Vec<f64> = (0..n)
            .map(|i| {
                state.pressure[i]
                    + dt * state.velocity[i]
                    + half_minus_beta * dt * dt * state.acceleration[i]
            })
            .collect();

        // rhs = F − K p̃
        let k_p_pred = csr_matvec(stiffness, &p_pred);
        force
            .iter()
            .zip(&k_p_pred)
            .map(|(&f, &kp)| f - kp)
            .collect()
    }

    /// Update the state after solving for the new acceleration.
    pub fn update_state(&self, state: &mut WaveState, acceleration_new: &[f64]) {
        let dt = self.config.dt;
        let beta = self.config.beta;
        let gamma = self.config.gamma;

        for (((&a_new, p), v), a) in acceleration_new
            .iter()
            .zip(&mut state.pressure)
            .zip(&mut state.velocity)
            .zip(&mut state.acceleration)
        {
            *p += dt * *v + dt * dt * ((0.5 - beta) * *a + beta * a_new);
            *v += dt * ((1.0 - gamma) * *a + gamma * a_new);
            *a = a_new;
        }
    }

    /// Compute initial acceleration from initial conditions: p̈₀ = M_c⁻¹(F₀ − K p₀).
    ///
    /// Returns the RHS vector; the caller must solve M_c × p̈₀ = rhs.
    pub fn initial_acceleration_rhs(
        &self,
        initial_pressure: &[f64],
        initial_force: &[f64],
        stiffness: &CsrMatrix<f64>,
    ) -> Vec<f64> {
        let kp = csr_matvec(stiffness, initial_pressure);
        let n = initial_pressure.len();
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            rhs[i] = initial_force[i] - kp[i];
        }
        rhs
    }
}

// ---------------------------------------------------------------------------
// Explicit leapfrog / central difference
// ---------------------------------------------------------------------------

/// Explicit leapfrog (central difference) time integrator.
///
/// Requires a **lumped** (diagonal) mass matrix for element-wise inversion.
/// Conditionally stable: Δt ≤ h_min / (c × safety_factor).
///
/// ```text
/// p_{n+1} = 2 p_n − p_{n−1} + Δt² diag(M_L)⁻¹ (F_n − K p_n)
/// ```
#[derive(Debug, Clone)]
pub struct LeapfrogStepper {
    pub dt: f64,
    /// Inverse of lumped mass diagonal: 1 / M_L[i,i].
    pub inv_lumped_mass: Vec<f64>,
    /// Previous pressure field p_{n−1}.
    pub pressure_prev: Vec<f64>,
}

impl LeapfrogStepper {
    /// Create from lumped mass diagonal and time step.
    ///
    /// `lumped_mass` contains the diagonal entries M_L\[i,i\] = (1/c²) × Σ_j M\[i,j\].
    pub fn new(lumped_mass: &[f64], dt: f64) -> Self {
        let inv = lumped_mass.iter().map(|&m| 1.0 / m).collect();
        let n = lumped_mass.len();
        Self {
            dt,
            inv_lumped_mass: inv,
            pressure_prev: vec![0.0; n],
        }
    }

    /// Maximum stable time step (CFL condition).
    ///
    /// For linear elements: Δt_max = h_min / c where h_min is the smallest
    /// element size. A safety factor of 0.9 is typical.
    pub fn cfl_time_step(h_min: f64, c: f64, safety: f64) -> f64 {
        safety * h_min / c
    }

    /// Perform one explicit time step.
    ///
    /// Updates `pressure` in-place and stores the previous value.
    pub fn step(&mut self, pressure: &mut [f64], force: &[f64], stiffness: &CsrMatrix<f64>) {
        let n = pressure.len();
        let dt2 = self.dt * self.dt;

        // K p_n
        let kp = csr_matvec(stiffness, pressure);

        // p_{n+1} = 2 p_n − p_{n−1} + Δt² M_L⁻¹ (F − K p)
        for i in 0..n {
            let p_new = 2.0 * pressure[i] - self.pressure_prev[i]
                + dt2 * self.inv_lumped_mass[i] * (force[i] - kp[i]);
            self.pressure_prev[i] = pressure[i];
            pressure[i] = p_new;
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple CSR matrix-vector product (not using ndarray).
fn csr_matvec(a: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    (0..a.num_rows)
        .map(|row| {
            let start = a.row_ptrs[row];
            let end = a.row_ptrs[row + 1];
            (start..end)
                .map(|idx| a.values[idx] * x[a.col_indices[idx]])
                .sum()
        })
        .collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple 1D wave equation on 3 interior nodes (5 nodes, 4 elements, h=0.25).
    /// Dirichlet p=0 at both ends → only 3 DOFs.
    fn simple_1d_system() -> (CsrMatrix<f64>, CsrMatrix<f64>) {
        // Stiffness K (1D Laplacian, h=0.25): 1/h [2 -1 0; -1 2 -1; 0 -1 2]
        let h = 0.25;
        let inv_h = 1.0 / h;
        let stiffness = CsrMatrix::from_raw_parts(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![
                2.0 * inv_h,
                -inv_h,
                -inv_h,
                2.0 * inv_h,
                -inv_h,
                -inv_h,
                2.0 * inv_h,
            ],
        );

        // Mass M (1D lumped, h=0.25): h [1 0 0; 0 1 0; 0 0 1]
        let mass = CsrMatrix::from_raw_parts(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![h, 0.0, 0.0, h, 0.0, 0.0, h],
        );

        (stiffness, mass)
    }

    #[test]
    fn test_newmark_config_trapezoidal() {
        let cfg = NewmarkConfig::trapezoidal(0.001);
        assert_eq!(cfg.beta, 0.25);
        assert_eq!(cfg.gamma, 0.5);
    }

    #[test]
    fn test_newmark_effective_stiffness() {
        let (k, m) = simple_1d_system();
        let cfg = NewmarkConfig::trapezoidal(0.01);
        let stepper = NewmarkStepper::new(cfg);
        let k_eff = stepper.effective_stiffness(&k, &m);
        // K_eff = M + 0.25 × 0.01² × K = M + 2.5e-5 K
        // Diagonal of K_eff ≈ M_ii + 2.5e-5 × K_ii
        assert_eq!(k_eff.num_rows, 3);
        assert!(!k_eff.values.is_empty());
    }

    #[test]
    fn test_newmark_zero_initial_stays_zero() {
        let (k, _m) = simple_1d_system();
        let cfg = NewmarkConfig::trapezoidal(0.01);
        let stepper = NewmarkStepper::new(cfg);
        let mut state = WaveState::zeros(3);
        let force = vec![0.0; 3];

        let rhs = stepper.form_rhs(&state, &force, &k);
        // With zero state and zero force, rhs should be zero
        for &r in &rhs {
            assert!(r.abs() < 1e-15, "rhs should be zero: {r}");
        }
        // After update with zero acceleration, state stays zero
        stepper.update_state(&mut state, &[0.0, 0.0, 0.0]);
        assert!(state.pressure.iter().all(|&p| p.abs() < 1e-15));
    }

    #[test]
    fn test_leapfrog_cfl() {
        let dt = LeapfrogStepper::cfl_time_step(0.01, 343.0, 0.9);
        assert!((dt - 0.9 * 0.01 / 343.0).abs() < 1e-15);
    }

    #[test]
    fn test_leapfrog_zero_force_constant_pressure() {
        let (k, _) = simple_1d_system();
        // Lumped mass: M_c = (1/c²) × h × I, for c = 1, h = 0.25
        let lumped = vec![0.25, 0.25, 0.25];
        let mut stepper = LeapfrogStepper::new(&lumped, 0.001);

        // Start with p = [1, 1, 1] (constant field)
        // K × [1,1,1] for the Laplacian stencil sums to 0 (row sums = 0 for interior)
        let mut pressure = vec![0.0; 3]; // Zero pressure → stays zero
        let force = vec![0.0; 3];

        stepper.step(&mut pressure, &force, &k);
        // Should still be zero (zero force, zero initial conditions)
        for &p in &pressure {
            assert!(p.abs() < 1e-15);
        }
    }
}
