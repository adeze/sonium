//! Transient heat equation time stepping.
//!
//! Solves the semi-discrete thermal system
//!
//! ```text
//! M dT/dt + K T = f
//! ```
//!
//! where `M` is assembled with [`HeatCapacityForm`](super::HeatCapacityForm)
//! and `K` with [`ConductionForm`](super::ConductionForm) plus any linear
//! Robin/advection contributions included by the caller.

use sonium_solvers::{
    CsrMatrix, LinearSolveConfig, LinearSolveError, LinearSolveResult, solve_real,
};

/// One-parameter theta schemes for the heat equation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThermalTimeScheme {
    /// Backward Euler: first-order, L-stable.
    BackwardEuler,
    /// Crank-Nicolson: second-order, A-stable.
    CrankNicolson,
    /// General theta method.
    ///
    /// `theta = 1` is backward Euler, `theta = 0.5` is Crank-Nicolson.
    Theta(f64),
}

impl ThermalTimeScheme {
    /// Return the implicitness parameter.
    pub fn theta(self) -> f64 {
        match self {
            Self::BackwardEuler => 1.0,
            Self::CrankNicolson => 0.5,
            Self::Theta(theta) => theta,
        }
    }
}

/// Step-by-step theta integrator for `M dT/dt + K T = f`.
#[derive(Debug, Clone, Copy)]
pub struct ThermalTimeStepper {
    /// Time step size.
    pub dt: f64,
    /// Time integration scheme.
    pub scheme: ThermalTimeScheme,
}

impl ThermalTimeStepper {
    /// Backward Euler stepper.
    pub fn backward_euler(dt: f64) -> Self {
        Self {
            dt,
            scheme: ThermalTimeScheme::BackwardEuler,
        }
    }

    /// Crank-Nicolson stepper.
    pub fn crank_nicolson(dt: f64) -> Self {
        Self {
            dt,
            scheme: ThermalTimeScheme::CrankNicolson,
        }
    }

    /// General theta-method stepper.
    pub fn theta(dt: f64, theta: f64) -> Self {
        assert!((0.0..=1.0).contains(&theta), "theta must lie in [0, 1]");
        Self {
            dt,
            scheme: ThermalTimeScheme::Theta(theta),
        }
    }

    /// Assemble `A = M/dt + theta K`.
    ///
    /// The matrix is constant for linear time-invariant thermal problems and
    /// can be reused across steps.
    pub fn effective_matrix(
        &self,
        mass: &CsrMatrix<f64>,
        stiffness: &CsrMatrix<f64>,
    ) -> CsrMatrix<f64> {
        assert_compatible_square(mass, stiffness);
        scaled_sum_csr(mass, 1.0 / self.dt, stiffness, self.scheme.theta())
    }

    /// Form the RHS for one theta-method step:
    ///
    /// ```text
    /// b = (M/dt - (1-theta)K) T_n + theta f_{n+1} + (1-theta) f_n
    /// ```
    ///
    /// For backward Euler, pass `source_current = None`.
    pub fn form_rhs(
        &self,
        temperature: &[f64],
        source_next: &[f64],
        source_current: Option<&[f64]>,
        mass: &CsrMatrix<f64>,
        stiffness: &CsrMatrix<f64>,
    ) -> Vec<f64> {
        assert_compatible_square(mass, stiffness);
        assert_eq!(temperature.len(), mass.num_cols);
        assert_eq!(source_next.len(), mass.num_rows);
        if let Some(source_current) = source_current {
            assert_eq!(source_current.len(), mass.num_rows);
        }

        let theta = self.scheme.theta();
        let mt = mass.matvec(temperature);
        let kt = stiffness.matvec(temperature);
        let mut rhs = vec![0.0; mass.num_rows];

        for i in 0..rhs.len() {
            rhs[i] = mt[i] / self.dt - (1.0 - theta) * kt[i] + theta * source_next[i];
            if let Some(source_current) = source_current {
                rhs[i] += (1.0 - theta) * source_current[i];
            }
        }

        rhs
    }

    /// Solve one thermal step and return `T_{n+1}` plus solver metadata.
    pub fn step(
        &self,
        temperature: &[f64],
        source_next: &[f64],
        source_current: Option<&[f64]>,
        mass: &CsrMatrix<f64>,
        stiffness: &CsrMatrix<f64>,
        solve_config: &LinearSolveConfig,
    ) -> Result<LinearSolveResult<f64>, LinearSolveError> {
        let effective = self.effective_matrix(mass, stiffness);
        let rhs = self.form_rhs(temperature, source_next, source_current, mass, stiffness);
        solve_real(&effective, &rhs, solve_config)
    }
}

/// Convenience one-shot backward Euler thermal step.
pub fn thermal_bdf_step(
    temperature: &[f64],
    source_next: &[f64],
    mass: &CsrMatrix<f64>,
    stiffness: &CsrMatrix<f64>,
    dt: f64,
    solve_config: &LinearSolveConfig,
) -> Result<Vec<f64>, LinearSolveError> {
    let result = ThermalTimeStepper::backward_euler(dt).step(
        temperature,
        source_next,
        None,
        mass,
        stiffness,
        solve_config,
    )?;
    Ok(result.x)
}

fn assert_compatible_square(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) {
    assert_eq!(a.num_rows, a.num_cols, "mass matrix must be square");
    assert_eq!(b.num_rows, b.num_cols, "stiffness matrix must be square");
    assert_eq!(a.num_rows, b.num_rows, "matrix row counts must match");
}

fn scaled_sum_csr(a: &CsrMatrix<f64>, alpha: f64, b: &CsrMatrix<f64>, beta: f64) -> CsrMatrix<f64> {
    let mut triplets = Vec::with_capacity(a.nnz() + b.nnz());
    for row in 0..a.num_rows {
        for idx in a.row_ptrs[row]..a.row_ptrs[row + 1] {
            triplets.push((row, a.col_indices[idx], alpha * a.values[idx]));
        }
    }
    for row in 0..b.num_rows {
        for idx in b.row_ptrs[row]..b.row_ptrs[row + 1] {
            triplets.push((row, b.col_indices[idx], beta * b.values[idx]));
        }
    }
    CsrMatrix::from_triplets(a.num_rows, a.num_cols, triplets)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagonal(values: &[f64]) -> CsrMatrix<f64> {
        let mut triplets = Vec::new();
        for (i, &v) in values.iter().enumerate() {
            triplets.push((i, i, v));
        }
        CsrMatrix::from_triplets(values.len(), values.len(), triplets)
    }

    #[test]
    fn test_effective_matrix_backward_euler() {
        let m = diagonal(&[2.0, 4.0]);
        let k = diagonal(&[10.0, 20.0]);
        let stepper = ThermalTimeStepper::backward_euler(0.5);
        let a = stepper.effective_matrix(&m, &k);

        assert_eq!(a.matvec(&[1.0, 1.0]), vec![14.0, 28.0]);
    }

    #[test]
    fn test_form_rhs_crank_nicolson() {
        let m = diagonal(&[2.0]);
        let k = diagonal(&[10.0]);
        let stepper = ThermalTimeStepper::crank_nicolson(0.5);
        let rhs = stepper.form_rhs(&[3.0], &[8.0], Some(&[4.0]), &m, &k);

        // M/dt*T - 0.5*K*T + 0.5*f_{n+1} + 0.5*f_n
        // = 2/0.5*3 - 15 + 4 + 2 = 3
        assert!((rhs[0] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_backward_euler_rhs_zero_source() {
        let m = diagonal(&[2.0]);
        let k = diagonal(&[10.0]);
        let stepper = ThermalTimeStepper::backward_euler(0.5);
        let rhs = stepper.form_rhs(&[3.0], &[0.0], None, &m, &k);

        assert!((rhs[0] - 12.0).abs() < 1e-14);
    }
}
