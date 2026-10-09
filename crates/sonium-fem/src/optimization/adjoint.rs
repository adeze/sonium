//! Adjoint sensitivity analysis
//!
//! Computes dF/dp efficiently via the adjoint method:
//!
//! ```text
//! Forward:  K(p) u = f
//! Adjoint:  K(p)^T lambda = -dF/du
//! Sensitivity: dF/dp_i = lambda^T (dK/dp_i) u + dF/dp_i|_explicit
//! ```
//!
//! One adjoint solve gives sensitivities for ALL design variables simultaneously.

/// Adjoint sensitivity result.
#[derive(Debug, Clone)]
pub struct AdjointSensitivity {
    /// Adjoint solution vector lambda.
    pub adjoint_solution: Vec<f64>,
    /// Sensitivity dF/dp for each design variable.
    pub sensitivities: Vec<f64>,
}

/// Complex frequency-domain adjoint result, including the solved adjoint.
#[derive(Debug, Clone)]
pub struct ComplexAdjointSensitivity {
    /// Adjoint solution of `Kᴴ λ = -∂F/∂ū`.
    pub adjoint_solution: Vec<num_complex::Complex64>,
    /// Real sensitivities for every design variable.
    pub sensitivities: Vec<f64>,
}

/// Compute adjoint sensitivities for a real-valued linear system.
///
/// Given:
/// - `system`: the system matrix K(p)
/// - `solution`: the forward solution u
/// - `df_du`: derivative of objective w.r.t. solution, dF/du
/// - `dk_dp_times_u`: for each design variable p_i, the product `(dK/dp_i) u`
/// - `linear_solve`: solves `K^T lambda = rhs` (for symmetric K, same as forward solve)
///
/// Returns the adjoint solution and sensitivities.
pub fn compute_adjoint_sensitivity<S>(
    df_du: &[f64],
    dk_dp_times_u: &[Vec<f64>],
    linear_solve: S,
) -> AdjointSensitivity
where
    S: Fn(&[f64]) -> Vec<f64>,
{
    // Solve adjoint: K^T lambda = -dF/du
    let neg_df_du: Vec<f64> = df_du.iter().map(|v| -*v).collect();
    let lambda = linear_solve(&neg_df_du);

    // Sensitivity: dF/dp_i = lambda^T (dK/dp_i) u
    let sensitivities: Vec<f64> = dk_dp_times_u
        .iter()
        .map(|dk_u| {
            lambda
                .iter()
                .zip(dk_u.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
        })
        .collect();

    AdjointSensitivity {
        adjoint_solution: lambda,
        sensitivities,
    }
}

/// Compute adjoint sensitivities for a complex-valued system (frequency domain).
///
/// Used for acoustic frequency response optimization.
/// The adjoint equation for complex systems is: `K^H lambda = -dF/du_conj`
/// where `K^H` is the Hermitian transpose.
pub fn compute_adjoint_sensitivity_complex<S>(
    df_du_conj: &[num_complex::Complex64],
    dk_dp_times_u: &[Vec<num_complex::Complex64>],
    linear_solve_hermitian: S,
) -> Vec<f64>
where
    S: Fn(&[num_complex::Complex64]) -> Vec<num_complex::Complex64>,
{
    compute_residual_adjoint_sensitivity_complex(
        df_du_conj,
        dk_dp_times_u,
        &vec![0.0; dk_dp_times_u.len()],
        linear_solve_hermitian,
    )
    .sensitivities
}

/// Compute complex adjoint sensitivities from residual derivatives.
///
/// For `R(u,p) = K(p)u - b(p) = 0` and real objective `F(u,p)`:
///
/// ```text
/// Kᴴ λ = -∂F/∂ū
/// dF/dpᵢ = ∂F/∂pᵢ|u + Re[λᴴ ∂R/∂pᵢ]
/// ```
///
/// `residual_derivatives[i]` must therefore contain
/// `(∂K/∂pᵢ)u - ∂b/∂pᵢ`. Including the RHS derivative is essential for
/// boundary-source, radiation, and load design variables.
pub fn compute_residual_adjoint_sensitivity_complex<S>(
    df_du_conj: &[num_complex::Complex64],
    residual_derivatives: &[Vec<num_complex::Complex64>],
    explicit_derivatives: &[f64],
    linear_solve_hermitian: S,
) -> ComplexAdjointSensitivity
where
    S: Fn(&[num_complex::Complex64]) -> Vec<num_complex::Complex64>,
{
    assert_eq!(
        residual_derivatives.len(),
        explicit_derivatives.len(),
        "one explicit derivative is required per residual derivative"
    );
    assert!(
        residual_derivatives
            .iter()
            .all(|derivative| derivative.len() == df_du_conj.len()),
        "residual derivative dimensions must match the state dimension"
    );
    let neg_df: Vec<num_complex::Complex64> = df_du_conj.iter().map(|v| -*v).collect();
    let lambda = linear_solve_hermitian(&neg_df);
    assert_eq!(
        lambda.len(),
        df_du_conj.len(),
        "adjoint solve returned the wrong state dimension"
    );

    let sensitivities = residual_derivatives
        .iter()
        .zip(explicit_derivatives)
        .map(|(d_residual, &explicit)| {
            explicit
                + lambda
                    .iter()
                    .zip(d_residual)
                    .map(|(l, d)| (l.conj() * d).re)
                    .sum::<f64>()
        })
        .collect();

    ComplexAdjointSensitivity {
        adjoint_solution: lambda,
        sensitivities,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjoint_sensitivity_simple() {
        // Simple 2-DOF system: K = [[2, -1], [-1, 2]], u = [1, 1]
        // Objective: F = u[0] → dF/du = [1, 0]
        // K^T = K (symmetric), so lambda = K^{-1} [-1, 0] = [-2/3, -1/3]
        // dK/dp = [[1, 0], [0, 0]] (only affects K[0,0])
        // (dK/dp) u = [1, 0]
        // dF/dp = lambda^T [1, 0] = -2/3

        let df_du = vec![1.0, 0.0];
        let dk_dp_u = vec![vec![1.0, 0.0]];

        let result = compute_adjoint_sensitivity(&df_du, &dk_dp_u, |rhs| {
            // Solve K lambda = rhs for K = [[2,-1],[-1,2]]
            // K^{-1} = (1/3) [[2,1],[1,2]]
            vec![(2.0 * rhs[0] + rhs[1]) / 3.0, (rhs[0] + 2.0 * rhs[1]) / 3.0]
        });

        assert!(
            (result.sensitivities[0] - (-2.0 / 3.0)).abs() < 1e-12,
            "dF/dp = {}, expected -2/3",
            result.sensitivities[0]
        );
    }

    #[test]
    fn complex_residual_adjoint_includes_matrix_rhs_and_explicit_terms() {
        use num_complex::Complex64;

        // Scalar system: (2+p i)u = 1+2p, F = |u|² + 3p.
        let p = 0.2;
        let matrix = Complex64::new(2.0, p);
        let rhs = Complex64::new(1.0 + 2.0 * p, 0.0);
        let u = rhs / matrix;
        // With dF = Re[(2u)ᴴ du], ∂F/∂ū in this convention is 2u.
        let df_du_conj = [2.0 * u];
        let d_residual = [vec![
            Complex64::new(0.0, 1.0) * u - Complex64::new(2.0, 0.0),
        ]];
        let result = compute_residual_adjoint_sensitivity_complex(
            &df_du_conj,
            &d_residual,
            &[3.0],
            |adjoint_rhs| vec![adjoint_rhs[0] / matrix.conj()],
        );

        let h = 1.0e-6;
        let objective = |parameter: f64| {
            let state = Complex64::new(1.0 + 2.0 * parameter, 0.0) / Complex64::new(2.0, parameter);
            state.norm_sqr() + 3.0 * parameter
        };
        let finite_difference = (objective(p + h) - objective(p - h)) / (2.0 * h);
        assert!(
            (result.sensitivities[0] - finite_difference).abs() < 1.0e-8,
            "adjoint={}, finite difference={finite_difference}",
            result.sensitivities[0]
        );
    }
}
