//! Optimization algorithms
//!
//! MMA (Method of Moving Asymptotes) and OC (Optimality Criteria) for
//! constrained topology/shape optimization.

/// Optimality Criteria (OC) update for compliance-based topology optimization.
///
/// Simple and efficient for compliance minimization with a single volume constraint.
/// Updates densities using a fixed-point iteration with bisection on the Lagrange multiplier.
pub struct OcOptimizer {
    /// Move limit (maximum density change per iteration).
    pub move_limit: f64,
    /// Damping factor (0-1, typically 0.5).
    pub damping: f64,
}

impl OcOptimizer {
    pub fn new() -> Self {
        Self {
            move_limit: 0.2,
            damping: 0.5,
        }
    }

    /// Update densities using OC method.
    ///
    /// `sensitivities`: dC/dp (compliance sensitivity, should be negative)
    /// `volume_sensitivities`: dg/dp (volume constraint gradient)
    /// `densities`: current design variables
    /// `target_volume_fraction`: volume constraint target
    /// `element_volumes`: volume of each element
    ///
    /// Returns the updated densities.
    pub fn update(
        &self,
        sensitivities: &[f64],
        densities: &[f64],
        target_volume_fraction: f64,
        element_volumes: &[f64],
    ) -> Vec<f64> {
        let n = densities.len();
        let total_volume: f64 = element_volumes.iter().sum();
        let target_vol = target_volume_fraction * total_volume;

        // Bisection on Lagrange multiplier
        let mut lam_lo = 0.0_f64;
        let mut lam_hi = 1e9_f64;

        let mut new_rho = vec![0.0; n];

        for _ in 0..100 {
            let lam_mid = 0.5 * (lam_lo + lam_hi);

            for i in 0..n {
                // OC update: rho_new = rho * (-dC/dp / (lam * dg/dp))^eta
                let be = (-sensitivities[i] / (lam_mid * element_volumes[i] / total_volume))
                    .max(0.0)
                    .powf(self.damping);
                let candidate = densities[i] * be;

                // Apply move limits and bounds
                new_rho[i] = candidate
                    .min(densities[i] + self.move_limit)
                    .min(1.0)
                    .max(densities[i] - self.move_limit)
                    .max(1e-3);
            }

            let current_vol: f64 = new_rho
                .iter()
                .zip(element_volumes)
                .map(|(&r, &v)| r * v)
                .sum();

            if current_vol > target_vol {
                lam_lo = lam_mid;
            } else {
                lam_hi = lam_mid;
            }

            if (lam_hi - lam_lo) / (lam_hi + lam_lo) < 1e-6 {
                break;
            }
        }

        new_rho
    }
}

impl Default for OcOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// One linearized constraint for [`MmaOptimizer::update_multi`].
///
/// Convention: the constraint is `value(x) <= bound`, linearized at the
/// current densities `x0`:
///
/// ```text
/// g_j(x) = value + grad . (x - x0) - bound <= 0
/// ```
///
/// `value` is the constraint value at `x0` and `grad` its gradient there.
/// Linear constraints (volume, average-density bridges) are represented
/// exactly. Constraints should be scaled to O(1) magnitudes (the built-in
/// volume constraint is normalized by the total element volume) so the dual
/// tolerances are meaningful.
#[derive(Debug, Clone)]
pub struct MmaConstraint {
    /// Constraint gradient at the current densities.
    pub grad: Vec<f64>,
    /// Constraint value at the current densities.
    pub value: f64,
    /// Upper bound: feasible means `value(x) <= bound`.
    pub bound: f64,
}

/// One-constraint Method of Moving Asymptotes optimizer.
///
/// This solves the separable convex MMA subproblem for an arbitrary objective
/// gradient and one linear volume constraint. Unlike OC, it does not require
/// every objective sensitivity to be negative, which is important for
/// magnetic BL-linearity and acoustic response objectives.
///
/// [`MmaOptimizer::update_multi`] generalizes the dual solve to several
/// linearized constraints; `update` delegates to it with the normalized
/// volume constraint.
pub struct MmaOptimizer {
    /// Maximum density change in one iteration.
    pub move_limit: f64,
    lower_asymptotes: Vec<f64>,
    upper_asymptotes: Vec<f64>,
    previous: Option<Vec<f64>>,
    previous_previous: Option<Vec<f64>>,
}

impl MmaOptimizer {
    pub fn new() -> Self {
        Self {
            move_limit: 0.2,
            lower_asymptotes: Vec::new(),
            upper_asymptotes: Vec::new(),
            previous: None,
            previous_previous: None,
        }
    }

    /// Update bounded densities by solving the MMA dual for a volume limit.
    ///
    /// Delegates to [`update_multi`](Self::update_multi) with the normalized
    /// volume constraint `sum(x_i * V_i) / sum(V_i) <= target_volume_fraction`.
    pub fn update(
        &mut self,
        sensitivities: &[f64],
        densities: &[f64],
        target_volume_fraction: f64,
        element_volumes: &[f64],
    ) -> Vec<f64> {
        assert_eq!(element_volumes.len(), densities.len());
        assert!((0.0..=1.0).contains(&target_volume_fraction));
        let total_volume = element_volumes.iter().sum::<f64>().max(f64::MIN_POSITIVE);
        let volume = densities
            .iter()
            .zip(element_volumes)
            .map(|(&density, &element_volume)| density * element_volume)
            .sum::<f64>();
        let volume_constraint = MmaConstraint {
            grad: element_volumes
                .iter()
                .map(|&element_volume| element_volume / total_volume)
                .collect(),
            value: volume / total_volume,
            bound: target_volume_fraction,
        };
        self.update_multi(
            sensitivities,
            densities,
            &[volume_constraint],
            element_volumes,
        )
    }

    /// Update bounded densities by solving the MMA dual with several
    /// linearized constraints (see [`MmaConstraint`] for the convention).
    ///
    /// Dual method: the MMA dual `phi(lambda) = min_x [f_mma(x) +
    /// sum_j lambda_j * g_j(x)]` is concave in `lambda >= 0` (minimum of
    /// affine functions), with supergradient `g_j(x*(lambda))`. It is
    /// maximized by **cyclic coordinate ascent**: each constraint multiplier
    /// in turn is maximized exactly by bisection on its (monotone decreasing)
    /// constraint residual, reusing the single-constraint machinery. One
    /// constraint therefore reproduces the classic single-lambda bisection.
    /// Sweeps stop when every KKT violation drops below `1e-8`
    /// (complementary residuals for active multipliers, positive residuals
    /// for inactive ones), capped at 60 sweeps.
    ///
    /// Safeguard: if the sweeps do not converge, the method falls back to the
    /// single most-violated constraint (all other multipliers zeroed, exact
    /// bisection on that constraint alone). The fallback satisfies the most
    /// violated constraint but may leave others mildly violated; it only
    /// triggers when coordinate ascent stalls. Note the stall is expected
    /// when the linearized subproblem is infeasible (no point of the
    /// move-limited box satisfies all constraints): callers introducing
    /// lower-bound constraints that the current design violates should ramp
    /// their bounds (as the motor-topology bridges do) so each subproblem
    /// stays feasible.
    ///
    /// `element_volumes` is accepted for interface symmetry with
    /// [`update`](Self::update) and validated for length; the constraints
    /// carry their own gradients.
    pub fn update_multi(
        &mut self,
        obj_grad: &[f64],
        densities: &[f64],
        constraints: &[MmaConstraint],
        element_volumes: &[f64],
    ) -> Vec<f64> {
        let (updated, _) = self.solve_mma_subproblem(obj_grad, densities, constraints);
        debug_assert_eq!(element_volumes.len(), densities.len());
        updated
    }

    /// Full MMA update: asymptote adaptation, separable approximation, dual
    /// solve. Returns the updated densities and the dual multipliers (the
    /// latter feeds KKT-residual checks in tests).
    fn solve_mma_subproblem(
        &mut self,
        sensitivities: &[f64],
        densities: &[f64],
        constraints: &[MmaConstraint],
    ) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(sensitivities.len(), densities.len());
        assert!(!constraints.is_empty());
        let n = densities.len();
        for constraint in constraints {
            assert_eq!(constraint.grad.len(), n);
        }
        if self.lower_asymptotes.len() != n {
            self.lower_asymptotes = densities.iter().map(|&x| (x - 0.5).max(1.0e-3)).collect();
            self.upper_asymptotes = densities.iter().map(|&x| (x + 0.5).min(1.0)).collect();
        } else if let (Some(previous), Some(previous_previous)) =
            (&self.previous, &self.previous_previous)
        {
            for i in 0..n {
                let trend = (densities[i] - previous[i]) * (previous[i] - previous_previous[i]);
                let factor = if trend > 0.0 { 1.2 } else { 0.7 };
                self.lower_asymptotes[i] =
                    (densities[i] - factor * (previous[i] - self.lower_asymptotes[i])).max(1.0e-3);
                self.upper_asymptotes[i] =
                    (densities[i] + factor * (self.upper_asymptotes[i] - previous[i])).min(1.0);
            }
        }

        let mut alpha = vec![0.0; n];
        let mut beta = vec![0.0; n];
        let mut p = vec![0.0; n];
        let mut q = vec![0.0; n];
        for i in 0..n {
            let x = densities[i];
            let lower = self.lower_asymptotes[i].min(x - 1.0e-5);
            let upper = self.upper_asymptotes[i].max(x + 1.0e-5);
            self.lower_asymptotes[i] = lower.max(1.0e-3);
            self.upper_asymptotes[i] = upper.min(1.0);
            alpha[i] = (x - self.move_limit)
                .max(1.0e-3)
                .max(lower + 0.1 * (x - lower));
            beta[i] = (x + self.move_limit)
                .min(1.0)
                .min(upper - 0.1 * (upper - x));
            let positive = sensitivities[i].max(0.0) + 1.0e-12;
            let negative = (-sensitivities[i]).max(0.0) + 1.0e-12;
            p[i] = (upper - x).powi(2) * positive;
            q[i] = (x - lower).powi(2) * negative;
        }

        // Linearized residual `g_j(x) = offset_j + grad_j . x` (<= 0 feasible).
        let offsets: Vec<f64> = constraints
            .iter()
            .map(|constraint| {
                constraint.value
                    - constraint.bound
                    - constraint
                        .grad
                        .iter()
                        .zip(densities)
                        .map(|(&gradient, &x)| gradient * x)
                        .sum::<f64>()
            })
            .collect();
        let residuals = |x: &[f64]| {
            constraints
                .iter()
                .zip(&offsets)
                .map(|(constraint, &offset)| {
                    offset
                        + constraint
                            .grad
                            .iter()
                            .zip(x)
                            .map(|(&gradient, &x)| gradient * x)
                            .sum::<f64>()
                })
                .collect::<Vec<_>>()
        };

        // Primal inner solve `x*(lambda)`: per-coordinate bisection of the
        // monotone stationarity equation of the separable convex subproblem.
        let solve_primal = |lambda: &[f64]| {
            (0..n)
                .map(|i| {
                    let lower = self.lower_asymptotes[i];
                    let upper = self.upper_asymptotes[i];
                    let dual_term: f64 = lambda
                        .iter()
                        .zip(constraints)
                        .map(|(&multiplier, constraint)| multiplier * constraint.grad[i])
                        .sum();
                    let derivative = |x: f64| {
                        p[i] / (upper - x).powi(2) - q[i] / (x - lower).powi(2) + dual_term
                    };
                    if derivative(alpha[i]) >= 0.0 {
                        return alpha[i];
                    }
                    if derivative(beta[i]) <= 0.0 {
                        return beta[i];
                    }
                    let mut lo = alpha[i];
                    let mut hi = beta[i];
                    for _ in 0..60 {
                        let mid = 0.5 * (lo + hi);
                        if derivative(mid) > 0.0 {
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    }
                    0.5 * (lo + hi)
                })
                .collect::<Vec<_>>()
        };

        // Exact coordinate maximization of the dual over `lambda_j >= 0`
        // (bisection on the residual, which decreases monotonically in
        // `lambda_j`; the maximizer sits at the residual root or at 0).
        let ascend_coordinate = |lambda: &mut [f64], j: usize| {
            lambda[j] = 0.0;
            if residuals(&solve_primal(lambda))[j] <= 0.0 {
                return;
            }
            let mut lambda_hi = 1.0;
            lambda[j] = lambda_hi;
            while residuals(&solve_primal(lambda))[j] > 0.0 && lambda_hi < 1.0e16 {
                lambda_hi *= 10.0;
                lambda[j] = lambda_hi;
            }
            let mut lambda_lo = 0.0;
            for _ in 0..80 {
                let mid = 0.5 * (lambda_lo + lambda_hi);
                lambda[j] = mid;
                if residuals(&solve_primal(lambda))[j] > 0.0 {
                    lambda_lo = mid;
                } else {
                    lambda_hi = mid;
                }
            }
            lambda[j] = 0.5 * (lambda_lo + lambda_hi);
        };

        // Parallel-tangent (secant) acceleration: cyclic coordinate ascent
        // zigzags on elongated dual bowls (near-parallel constraint
        // gradients), crawling at a constant per-sweep rate. After each
        // sweep, maximize the (concave) dual along the sweep-secant ray
        // `lambda + t * d` by bisection on the directional residual
        // `sum_j d_j * g_j(x*(lambda + t d))`, which decreases monotonically
        // in `t`. The ray is clipped where a multiplier would go negative.
        let ascend_secant = |lambda: &mut [f64], direction: &[f64]| {
            let m = lambda.len();
            // Largest t keeping lambda + t*d >= 0 (infinity if d >= 0).
            let t_boundary = (0..m)
                .filter(|&j| direction[j] < 0.0)
                .map(|j| -lambda[j] / direction[j])
                .fold(f64::INFINITY, f64::min);
            let directional_residual = |t: f64| {
                let trial: Vec<f64> = lambda
                    .iter()
                    .zip(direction)
                    .map(|(&multiplier, &step)| multiplier + t * step)
                    .collect();
                let x = solve_primal(&trial);
                residuals(&x)
                    .iter()
                    .zip(direction)
                    .map(|(&residual, &step)| residual * step)
                    .sum::<f64>()
            };
            if directional_residual(0.0) <= 0.0 {
                return;
            }
            let mut t_hi = 1.0_f64;
            while directional_residual(t_hi) > 0.0 && t_hi < t_boundary && t_hi < 1.0e6 {
                t_hi *= 2.0;
            }
            if t_hi > t_boundary {
                lambda
                    .iter_mut()
                    .zip(direction)
                    .for_each(|(multiplier, &step)| *multiplier += t_boundary * step);
                return;
            }
            let mut t_lo = 0.0;
            for _ in 0..60 {
                let mid = 0.5 * (t_lo + t_hi);
                if directional_residual(mid) > 0.0 {
                    t_lo = mid;
                } else {
                    t_hi = mid;
                }
            }
            let t = 0.5 * (t_lo + t_hi);
            lambda
                .iter_mut()
                .zip(direction)
                .for_each(|(multiplier, &step)| *multiplier += t * step);
        };

        const MAX_SWEEPS: usize = 60;
        const RESIDUAL_TOL: f64 = 1.0e-8;
        let m = constraints.len();
        let mut lambda = vec![0.0; m];
        let kkt_violation = |lambda: &[f64]| {
            let x = solve_primal(lambda);
            residuals(&x)
                .iter()
                .enumerate()
                .map(|(j, &residual)| {
                    if lambda[j] > 0.0 {
                        residual.abs()
                    } else {
                        residual.max(0.0)
                    }
                })
                .fold(0.0_f64, f64::max)
        };
        let mut converged = kkt_violation(&lambda) <= RESIDUAL_TOL;
        if !converged {
            let mut previous_sweep: Option<Vec<f64>> = None;
            for _ in 0..MAX_SWEEPS {
                for j in 0..m {
                    ascend_coordinate(&mut lambda, j);
                }
                if let Some(previous) = &previous_sweep {
                    let direction: Vec<f64> = lambda
                        .iter()
                        .zip(previous)
                        .map(|(&current, &old)| current - old)
                        .collect();
                    if direction.iter().any(|&step| step != 0.0) {
                        ascend_secant(&mut lambda, &direction);
                    }
                }
                previous_sweep = Some(lambda.clone());
                if kkt_violation(&lambda) <= RESIDUAL_TOL {
                    converged = true;
                    break;
                }
            }
        }
        if !converged {
            // Safeguard: dual ascent stalled — satisfy the single
            // most-violated constraint exactly and drop the rest.
            let x = solve_primal(&lambda);
            let violation = residuals(&x);
            let worst = violation
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(j, _)| j)
                .unwrap_or(0);
            lambda = vec![0.0; m];
            ascend_coordinate(&mut lambda, worst);
        }

        let new_densities = solve_primal(&lambda);
        self.previous_previous = self.previous.take();
        self.previous = Some(densities.to_vec());
        (new_densities, lambda)
    }
}

impl Default for MmaOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Density-update algorithm used by the topology driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyOptimizerKind {
    /// Method of Moving Asymptotes; supports sign-changing sensitivities.
    Mma,
    /// Optimality Criteria; efficient for compliance-like objectives.
    Oc,
}

/// Configuration for the topology optimization loop.
#[derive(Debug, Clone)]
pub struct TopOptConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on density change.
    pub tolerance: f64,
    /// Target volume fraction.
    pub volume_fraction: f64,
    /// Filter radius (m).
    pub filter_radius: f64,
    /// Penalization exponent for SIMP.
    pub simp_penalty: f64,
    /// Whether to use continuation (gradually increase penalty).
    pub continuation: bool,
    /// Density-update optimizer.
    pub optimizer: TopologyOptimizerKind,
    /// Verbosity level.
    pub verbosity: usize,
}

impl Default for TopOptConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 0.01,
            volume_fraction: 0.5,
            filter_radius: 0.05,
            simp_penalty: 3.0,
            continuation: false,
            optimizer: TopologyOptimizerKind::Mma,
            verbosity: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oc_update_moves_toward_target() {
        let oc = OcOptimizer::new();

        // Start above target, sensitivities favor removing material
        let densities = vec![0.8, 0.8, 0.8, 0.8];
        let sensitivities = vec![-10.0, -20.0, -5.0, -15.0];
        let volumes = vec![1.0; 4];

        let new_rho = oc.update(&sensitivities, &densities, 0.4, &volumes);

        let old_vf = 0.8;
        let new_vf: f64 = new_rho.iter().sum::<f64>() / 4.0;
        // Should move toward target (0.4), i.e. decrease from 0.8
        assert!(
            new_vf < old_vf,
            "Volume fraction should decrease: old={old_vf}, new={new_vf}"
        );
    }

    #[test]
    fn test_oc_respects_move_limit() {
        let oc = OcOptimizer {
            move_limit: 0.1,
            damping: 0.5,
        };

        let densities = vec![0.5; 4];
        let sensitivities = vec![-1000.0; 4]; // very strong sensitivity
        let volumes = vec![1.0; 4];

        let new_rho = oc.update(&sensitivities, &densities, 0.3, &volumes);

        for (i, (&old, &new)) in densities.iter().zip(new_rho.iter()).enumerate() {
            assert!(
                (new - old).abs() <= 0.1 + 1e-10,
                "Element {i}: move = {}, exceeds limit 0.1",
                (new - old).abs()
            );
        }
    }

    /// Reproduce the first-call MMA subproblem coefficients (asymptotes,
    /// bounds, p/q) for a fresh optimizer, so tests can solve the exact same
    /// subproblem by brute force.
    fn first_call_subproblem(
        x0: &[f64],
        grad: &[f64],
        move_limit: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = x0.len();
        let mut lower = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut alpha = vec![0.0; n];
        let mut beta = vec![0.0; n];
        let mut p = vec![0.0; n];
        let mut q = vec![0.0; n];
        for i in 0..n {
            let x = x0[i];
            lower[i] = ((x - 0.5).max(1.0e-3)).min(x - 1.0e-5).max(1.0e-3);
            upper[i] = ((x + 0.5).min(1.0)).max(x + 1.0e-5).min(1.0);
            alpha[i] = (x - move_limit)
                .max(1.0e-3)
                .max(lower[i] + 0.1 * (x - lower[i]));
            beta[i] = (x + move_limit)
                .min(1.0)
                .min(upper[i] - 0.1 * (upper[i] - x));
            p[i] = (upper[i] - x).powi(2) * (grad[i].max(0.0) + 1.0e-12);
            q[i] = (x - lower[i]).powi(2) * ((-grad[i]).max(0.0) + 1.0e-12);
        }
        (lower, upper, alpha, beta, p, q)
    }

    /// Exact MMA subproblem objective `sum_i p_i/(U_i-x_i) + q_i/(x_i-L_i)`.
    fn subproblem_objective(x: &[f64], lower: &[f64], upper: &[f64], p: &[f64], q: &[f64]) -> f64 {
        (0..x.len())
            .map(|i| p[i] / (upper[i] - x[i]) + q[i] / (x[i] - lower[i]))
            .sum()
    }

    /// Linearized constraint residuals `value + grad.(x - x0) - bound`.
    fn residuals_at(x: &[f64], x0: &[f64], constraints: &[MmaConstraint]) -> Vec<f64> {
        constraints
            .iter()
            .map(|constraint| {
                constraint.value - constraint.bound
                    + constraint
                        .grad
                        .iter()
                        .zip(x.iter().zip(x0))
                        .map(|(&gradient, (&x, &x0))| gradient * (x - x0))
                        .sum::<f64>()
            })
            .collect()
    }

    /// Brute-force grid minimizer of the exact MMA subproblem over the
    /// feasible (linearized constraints <= tolerance) box [alpha, beta].
    fn grid_solve(
        x0: &[f64],
        grad: &[f64],
        constraints: &[MmaConstraint],
        step: f64,
    ) -> (Vec<f64>, f64) {
        let (lower, upper, alpha, beta, p, q) = first_call_subproblem(x0, grad, 0.2);
        let axes: Vec<Vec<f64>> = (0..x0.len())
            .map(|i| {
                let mut axis = Vec::new();
                let mut x = alpha[i];
                while x <= beta[i] + 1.0e-12 {
                    axis.push(x.min(beta[i]));
                    x += step;
                }
                axis
            })
            .collect();
        let mut best = vec![0.0; x0.len()];
        let mut best_objective = f64::INFINITY;
        let mut current = vec![0.0; x0.len()];
        fn recurse(
            depth: usize,
            axes: &[Vec<f64>],
            current: &mut [f64],
            x0: &[f64],
            constraints: &[MmaConstraint],
            lower: &[f64],
            upper: &[f64],
            p: &[f64],
            q: &[f64],
            best: &mut [f64],
            best_objective: &mut f64,
        ) {
            if depth == axes.len() {
                if residuals_at(current, x0, constraints)
                    .iter()
                    .all(|&residual| residual <= 1.0e-9)
                {
                    let objective = subproblem_objective(current, lower, upper, p, q);
                    if objective < *best_objective {
                        *best_objective = objective;
                        best.copy_from_slice(current);
                    }
                }
                return;
            }
            for &x in &axes[depth] {
                current[depth] = x;
                recurse(
                    depth + 1,
                    axes,
                    current,
                    x0,
                    constraints,
                    lower,
                    upper,
                    p,
                    q,
                    best,
                    best_objective,
                );
            }
        }
        recurse(
            0,
            &axes,
            &mut current,
            x0,
            constraints,
            &lower,
            &upper,
            &p,
            &q,
            &mut best,
            &mut best_objective,
        );
        assert!(best_objective.is_finite(), "grid found no feasible point");
        (best, best_objective)
    }

    /// Verify KKT conditions of the exact MMA subproblem at `(x, lambda)`:
    /// primal feasibility, dual feasibility, complementarity, and
    /// stationarity (with bound complementarity for alpha/beta-active
    /// coordinates).
    fn assert_kkt(
        x0: &[f64],
        grad: &[f64],
        constraints: &[MmaConstraint],
        x: &[f64],
        lambda: &[f64],
    ) {
        let (lower, upper, alpha, beta, p, q) = first_call_subproblem(x0, grad, 0.2);
        let residuals = residuals_at(x, x0, constraints);
        for (j, &residual) in residuals.iter().enumerate() {
            assert!(residual <= 1.0e-6, "constraint {j} violated: {residual:e}");
            assert!(lambda[j] >= 0.0, "negative multiplier {j}");
            assert!(
                (lambda[j] * residual).abs() <= 1.0e-6,
                "complementarity {j}: lambda*g = {:e}",
                lambda[j] * residual
            );
        }
        let mut worst_stationarity = 0.0_f64;
        for i in 0..x.len() {
            let dual_term: f64 = lambda
                .iter()
                .zip(constraints)
                .map(|(&multiplier, constraint)| multiplier * constraint.grad[i])
                .sum();
            let stationarity =
                p[i] / (upper[i] - x[i]).powi(2) - q[i] / (x[i] - lower[i]).powi(2) + dual_term;
            if x[i] <= alpha[i] + 1.0e-9 {
                assert!(
                    stationarity >= -1.0e-6,
                    "lower-bound KKT {i}: {stationarity:e}"
                );
            } else if x[i] >= beta[i] - 1.0e-9 {
                assert!(
                    stationarity <= 1.0e-6,
                    "upper-bound KKT {i}: {stationarity:e}"
                );
            } else {
                worst_stationarity = worst_stationarity.max(stationarity.abs());
            }
        }
        eprintln!("max interior stationarity residual: {worst_stationarity:e}");
        assert!(
            worst_stationarity <= 1.0e-5,
            "interior stationarity residual {worst_stationarity:e}"
        );
    }

    /// Assert `update_multi`'s result matches the brute-force grid optimum of
    /// the exact MMA subproblem.
    fn assert_matches_grid(
        x0: &[f64],
        grad: &[f64],
        constraints: &[MmaConstraint],
        x: &[f64],
        step: f64,
    ) {
        let (grid_x, grid_objective) = grid_solve(x0, grad, constraints, step);
        let (lower, upper, _, _, p, q) = first_call_subproblem(x0, grad, 0.2);
        let objective = subproblem_objective(x, &lower, &upper, &p, &q);
        let scale = grid_objective.abs().max(1.0e-12);
        eprintln!(
            "subproblem objective: update_multi={objective:e}, grid={grid_objective:e}, rel={:e}",
            (objective - grid_objective).abs() / scale
        );
        assert!(
            (objective - grid_objective).abs() / scale <= 5.0e-2,
            "update_multi objective {objective:e} deviates from grid optimum {grid_objective:e}"
        );
        for (i, (&x, &grid_x)) in x.iter().zip(&grid_x).enumerate() {
            assert!(
                (x - grid_x).abs() <= 2.0 * step + 1.0e-9,
                "coordinate {i}: update_multi={x:e}, grid={grid_x:e}"
            );
        }
    }

    fn volume_constraint(x0: &[f64], fraction: f64) -> MmaConstraint {
        let n = x0.len() as f64;
        MmaConstraint {
            grad: vec![1.0 / n; x0.len()],
            value: x0.iter().sum::<f64>() / n,
            bound: fraction,
        }
    }

    /// D2 (a): two constraints, volume binds alone.
    #[test]
    fn mma_multi_volume_binds_alone() {
        let x0 = vec![0.5; 3];
        let grad = vec![-0.5, -0.1, -0.5]; // quadratic bowl centered at [1, 0.6, 1]
        let constraints = vec![
            volume_constraint(&x0, 0.4),
            // loose: mean >= 0.3
            MmaConstraint {
                grad: vec![-1.0 / 3.0; 3],
                value: -0.5,
                bound: -0.3,
            },
        ];
        let mut mma = MmaOptimizer::new();
        let (x, lambda) = mma.solve_mma_subproblem(&grad, &x0, &constraints);
        eprintln!("x={x:?}, lambda={lambda:?}");
        assert!(lambda[0] > 1.0e-6, "volume multiplier should be active");
        assert_eq!(lambda[1], 0.0, "loose constraint should stay inactive");
        assert!(x.iter().sum::<f64>() <= 1.2 + 1.0e-6);
        assert_kkt(&x0, &grad, &constraints, &x, &lambda);
        assert_matches_grid(&x0, &grad, &constraints, &x, 0.005);
    }

    /// D2 (b): two constraints, the second (average-density floor) binds alone.
    #[test]
    fn mma_multi_second_constraint_binds_alone() {
        let x0 = vec![0.5; 3];
        let grad = vec![0.5, 0.5, 0.5]; // quadratic bowl centered at origin
        let constraints = vec![
            volume_constraint(&x0, 0.95), // loose
            // mean >= 0.55
            MmaConstraint {
                grad: vec![-1.0 / 3.0; 3],
                value: -0.5,
                bound: -0.55,
            },
        ];
        let mut mma = MmaOptimizer::new();
        let (x, lambda) = mma.solve_mma_subproblem(&grad, &x0, &constraints);
        eprintln!("x={x:?}, lambda={lambda:?}");
        assert_eq!(lambda[0], 0.0, "volume should stay inactive");
        assert!(lambda[1] > 1.0e-6, "floor multiplier should be active");
        let mean = x.iter().sum::<f64>() / 3.0;
        assert!((mean - 0.55).abs() <= 1.0e-6, "mean={mean:e}");
        assert_kkt(&x0, &grad, &constraints, &x, &lambda);
        assert_matches_grid(&x0, &grad, &constraints, &x, 0.005);
    }

    /// D2 (c): two constraints, both bind simultaneously.
    #[test]
    fn mma_multi_both_constraints_bind() {
        let x0 = vec![0.5; 3];
        let grad = vec![-0.5, 0.5, -0.5]; // bowl centered at [1, 0, 1]
        let constraints = vec![
            volume_constraint(&x0, 0.4), // sum(x) <= 1.2
            // x_2 >= 0.42
            MmaConstraint {
                grad: vec![0.0, -1.0, 0.0],
                value: -0.5,
                bound: -0.42,
            },
        ];
        let mut mma = MmaOptimizer::new();
        let (x, lambda) = mma.solve_mma_subproblem(&grad, &x0, &constraints);
        eprintln!("x={x:?}, lambda={lambda:?}");
        assert!(lambda[0] > 1.0e-6, "volume multiplier should be active");
        assert!(lambda[1] > 1.0e-6, "floor multiplier should be active");
        assert!(x.iter().sum::<f64>() <= 1.2 + 1.0e-6);
        assert!(x[1] >= 0.42 - 1.0e-6);
        assert_kkt(&x0, &grad, &constraints, &x, &lambda);
        assert_matches_grid(&x0, &grad, &constraints, &x, 0.005);
    }

    /// D2: an infeasible start (floor constraint violated at x0) recovers
    /// within a few iterations and the floor stays satisfied afterwards,
    /// mimicking the motor-topology bridge iteration pattern.
    #[test]
    fn mma_multi_recovers_from_infeasible_start() {
        let mut mma = MmaOptimizer::new();
        let mut x = vec![0.65; 4];
        let volumes = vec![1.0; 4];
        for iteration in 0..8 {
            let grad = vec![0.5; 4]; // objective pushes all densities down
            let constraints = vec![
                volume_constraint(&x, 0.65), // sum(x) <= 2.6
                // mean(x0, x1) >= 0.95
                MmaConstraint {
                    grad: vec![-0.5, -0.5, 0.0, 0.0],
                    value: -0.5 * (x[0] + x[1]),
                    bound: -0.95,
                },
            ];
            let (next, lambda) = mma.solve_mma_subproblem(&grad, &x, &constraints);
            let mean = 0.5 * (next[0] + next[1]);
            let sum: f64 = next.iter().sum();
            eprintln!(
                "iter {iteration}: x={next:?}, mean={mean:.4}, sum={sum:.4}, lambda={lambda:?}"
            );
            x = next;
            if iteration >= 2 {
                assert!(mean >= 0.95 - 1.0e-6, "iter {iteration}: floor violated");
                assert!(sum <= 2.6 + 1.0e-6, "iter {iteration}: volume violated");
            }
        }
        let _ = &volumes;
    }

    /// D2 regression: `update` (volume-only delegation) keeps its exact
    /// legacy behavior on the pre-D2 test case.
    #[test]
    fn mma_update_matches_update_multi_with_volume_constraint() {
        let densities = vec![0.5; 6];
        let sensitivities = vec![-2.0, 1.0, -0.5, 3.0, -4.0, 0.25];
        let volumes = vec![1.0; 6];
        let mut legacy = MmaOptimizer::new();
        let via_update = legacy.update(&sensitivities, &densities, 0.4, &volumes);
        let mut multi = MmaOptimizer::new();
        let constraint = volume_constraint(&densities, 0.4);
        let via_multi = multi.update_multi(&sensitivities, &densities, &[constraint], &volumes);
        let max_diff = via_update
            .iter()
            .zip(&via_multi)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        eprintln!("update vs update_multi max diff: {max_diff:e}");
        assert!(max_diff <= 1.0e-10);
    }

    #[test]
    fn mma_respects_move_and_volume_limits_with_mixed_sensitivities() {
        let mut mma = MmaOptimizer::new();
        let densities = vec![0.5; 6];
        let sensitivities = vec![-2.0, 1.0, -0.5, 3.0, -4.0, 0.25];
        let volumes = vec![1.0; 6];
        let updated = mma.update(&sensitivities, &densities, 0.4, &volumes);
        assert!(
            updated
                .iter()
                .all(|&density| (1.0e-3..=1.0).contains(&density))
        );
        assert!(
            updated
                .iter()
                .zip(&densities)
                .all(|(&new, &old)| (new - old).abs() <= mma.move_limit + 1.0e-12)
        );
        assert!(updated.iter().sum::<f64>() <= 0.4 * volumes.len() as f64 + 1.0e-8);
        assert!(
            updated[0] > updated[1],
            "negative sensitivity should retain more material than positive sensitivity"
        );
    }
}
