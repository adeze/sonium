//! Trust region subproblem solvers.
//!
//! Cauchy point, dogleg, Steihaug-CG, and exact More-Sorensen for solving:
//!   min  g^T d + 0.5 d^T H d   subject to  ||d|| <= delta

use crate::types::RealScalar;

/// Selects how the trust-region subproblem is solved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrustRegionSubproblemSolver {
    /// Steihaug-CG: iterative, Hessian-vector products only, approximate.
    #[default]
    SteihaugCg,
    /// More-Sorensen: exact dense solve via repeated Cholesky of `H + lambda*I`,
    /// including the hard case. Requires a dense Hessian (O(n^3) per solve).
    MoreSorensen,
}

/// Compute the Cauchy point: the steepest descent direction clipped to the trust region.
///
/// Returns the Cauchy step `d_c = -alpha * g` where `alpha = min(||g||^2/(g^T H g), delta/||g||)`.
pub fn cauchy_point<T, Hv>(g: &[T], hess_vec_fn: &Hv, x: &[T], delta: T) -> Vec<T>
where
    T: RealScalar,
    Hv: Fn(&[T], &[T]) -> Vec<T>,
{
    let g_norm = T::vec_norm(g);
    if g_norm < T::machine_epsilon() {
        return vec![T::zero(); g.len()];
    }

    let hg = hess_vec_fn(x, g);
    let ghg = T::vec_dot(g, &hg);

    let alpha = if ghg > T::zero() {
        T::min(g_norm * g_norm / ghg, delta / g_norm)
    } else {
        delta / g_norm
    };

    g.iter().map(|&gi| -alpha * gi).collect::<Vec<_>>()
}

/// Dogleg step: blend Cauchy point and full Newton step.
///
/// If the Newton step `d_n = -H^{-1} g` is within the trust region, use it.
/// Otherwise, interpolate between the Cauchy point and the Newton step along the
/// dogleg path.
pub fn dogleg<T: RealScalar>(_g: &[T], newton_step: &[T], cauchy_step: &[T], delta: T) -> Vec<T> {
    let n_norm = T::vec_norm(newton_step);

    // Full Newton step inside trust region
    if n_norm <= delta {
        return newton_step.to_vec();
    }

    let c_norm = T::vec_norm(cauchy_step);

    // Cauchy point outside trust region: scale to boundary
    if c_norm >= delta {
        return cauchy_step
            .iter()
            .map(|&ci| ci * (delta / c_norm))
            .collect::<Vec<_>>();
    }

    // Interpolate: d = cauchy + tau * (newton - cauchy), find tau so ||d|| = delta
    let diff = sonium_solvers_core::vec_ops::vec_sub(newton_step, cauchy_step);
    let a = T::vec_dot(&diff, &diff);
    let b = T::from_f64(2.0).unwrap() * T::vec_dot(cauchy_step, &diff);
    let c = T::vec_dot(cauchy_step, cauchy_step) - delta * delta;

    let discriminant = b * b - T::from_f64(4.0).unwrap() * a * c;
    let tau = if discriminant >= T::zero() && a > T::machine_epsilon() {
        (-b + discriminant.sqrt()) / (T::from_f64(2.0).unwrap() * a)
    } else {
        T::one()
    };
    let tau = T::min(T::max(tau, T::zero()), T::one());

    let scaled_diff = diff.iter().map(|&di| di * tau).collect::<Vec<_>>();
    sonium_solvers_core::vec_ops::vec_add(cauchy_step, &scaled_diff)
}

/// Steihaug-CG: solve the trust region subproblem using CG with trust region constraint.
///
/// Approximately minimises `m(d) = g^T d + 0.5 d^T H d` subject to `||d|| <= delta`.
/// Returns the step `d` and the number of Hessian-vector products used.
pub fn steihaug_cg<T, Hv>(
    g: &[T],
    hess_vec_fn: &Hv,
    x: &[T],
    delta: T,
    max_iters: usize,
    tol: T,
) -> (Vec<T>, usize)
where
    T: RealScalar,
    Hv: Fn(&[T], &[T]) -> Vec<T>,
{
    let n = g.len();
    let mut d = vec![T::zero(); n];
    let mut r = g.to_vec();
    let mut p = sonium_solvers_core::vec_ops::vec_neg(&r);
    let mut hess_evals = 0;

    let r_norm = T::vec_norm(&r);
    if r_norm < tol {
        return (d, 0);
    }

    let mut rr = r_norm * r_norm;

    for _ in 0..max_iters {
        let hp = hess_vec_fn(x, &p);
        hess_evals += 1;
        let php = T::vec_dot(&p, &hp);

        // Negative curvature or zero curvature: go to trust region boundary
        if php <= T::zero() {
            let tau = trust_region_boundary_step(&d, &p, delta);
            let step = p.iter().map(|&pi| pi * tau).collect::<Vec<_>>();
            return (sonium_solvers_core::vec_ops::vec_add(&d, &step), hess_evals);
        }

        let alpha = rr / php;

        // Check if step would exceed trust region
        let step = p.iter().map(|&pi| pi * alpha).collect::<Vec<_>>();
        let d_new = sonium_solvers_core::vec_ops::vec_add(&d, &step);
        if T::vec_norm(&d_new) >= delta {
            let tau = trust_region_boundary_step(&d, &p, delta);
            let step = p.iter().map(|&pi| pi * tau).collect::<Vec<_>>();
            return (sonium_solvers_core::vec_ops::vec_add(&d, &step), hess_evals);
        }

        d = d_new;
        T::vec_axpy(alpha, &hp, &mut r);

        let rr_new = T::vec_dot(&r, &r);
        if rr_new.sqrt() < tol {
            break;
        }

        let beta = rr_new / rr;
        let neg_r = sonium_solvers_core::vec_ops::vec_neg(&r);
        let beta_p = p.iter().map(|&pi| pi * beta).collect::<Vec<_>>();
        p = sonium_solvers_core::vec_ops::vec_add(&neg_r, &beta_p);
        rr = rr_new;
    }

    (d, hess_evals)
}

/// Find the positive scalar `tau` such that `||d + tau * p|| = delta`.
fn trust_region_boundary_step<T: RealScalar>(d: &[T], p: &[T], delta: T) -> T {
    let dd = T::vec_dot(d, d);
    let dp = T::vec_dot(d, p);
    let pp = T::vec_dot(p, p);

    let a = pp;
    let b = T::from_f64(2.0).unwrap() * dp;
    let c = dd - delta * delta;

    let discriminant = b * b - T::from_f64(4.0).unwrap() * a * c;
    if discriminant < T::zero() || a < T::machine_epsilon() {
        return T::zero();
    }

    (-b + discriminant.sqrt()) / (T::from_f64(2.0).unwrap() * a)
}

/// Quadratic model value `m(d) = g^T d + 0.5 d^T H d`.
fn trust_model_value<T: RealScalar>(
    g: &[T],
    h: &sonium_solvers_core::dense::DenseMatrix<T>,
    d: &[T],
) -> T {
    let n = g.len();
    let m = T::vec_dot(g, d);
    let mut dhd = T::zero();
    for i in 0..n {
        let mut hd = T::zero();
        for j in 0..n {
            hd += h[[i, j]] * d[j];
        }
        dhd += d[i] * hd;
    }
    m + T::from_f64(0.5).unwrap() * dhd
}

/// Cholesky of `a`, or — on the first non-positive pivot `j` — an approximate
/// null-space direction `z` via the LINPACK partial-factorisation trick:
/// `z_j = 1`, `z_i = 0` for `i > j`, and back-substitution into the completed
/// leading factor so that `L^T z` vanishes on rows `< j`.
fn cholesky_or_null_direction<T: RealScalar>(
    a: &sonium_solvers_core::dense::DenseMatrix<T>,
) -> Result<sonium_solvers_core::dense::DenseMatrix<T>, Vec<T>> {
    let n = a.nrows();
    let mut l = sonium_solvers_core::dense::DenseMatrix::zeros(n, n);
    for j in 0..n {
        let mut sum = T::zero();
        for k in 0..j {
            sum += l[[j, k]] * l[[j, k]];
        }
        let diag = a[[j, j]] - sum;
        if diag <= T::zero() {
            let mut z = vec![T::zero(); n];
            z[j] = T::one();
            for i in (0..j).rev() {
                let mut s = T::zero();
                for k in (i + 1)..=j {
                    s += l[[k, i]] * z[k];
                }
                z[i] = -s / l[[i, i]];
            }
            return Err(z);
        }
        l[[j, j]] = diag.sqrt();
        for i in (j + 1)..n {
            let mut s = T::zero();
            for k in 0..j {
                s += l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = (a[[i, j]] - s) / l[[j, j]];
        }
    }
    Ok(l)
}

/// More-Sorensen exact trust-region subproblem solver (Nocedal & Wright,
/// Algorithm 4.3) for a dense symmetric Hessian.
///
/// Solves `min g^T d + 0.5 d^T H d` subject to `||d|| <= delta` to high
/// accuracy by Newton iteration on the secular equation `||d(lambda)|| = delta`
/// with Cholesky factorisations of `H + lambda*I`. Handles the hard case
/// (H + lambda*I singular at the optimum, g orthogonal to the corresponding
/// eigenspace) by augmenting the step with an approximate null direction.
pub fn more_sorensen<T: RealScalar>(
    g: &[T],
    h: &sonium_solvers_core::dense::DenseMatrix<T>,
    delta: T,
) -> Vec<T> {
    let n = g.len();
    let g_norm = T::vec_norm(g);
    if n == 0 || g_norm < T::machine_epsilon() {
        return vec![T::zero(); n];
    }

    // Symmetrise defensively: the caller may assemble H from Hessian-vector
    // products with slight roundoff asymmetry.
    let mut hs = sonium_solvers_core::dense::DenseMatrix::zeros(n, n);
    let half = T::from_f64(0.5).unwrap();
    for i in 0..n {
        for j in 0..n {
            hs[[i, j]] = half * (h[[i, j]] + h[[j, i]]);
        }
    }

    // Interior solution: H positive definite and the Newton step fits.
    if let Ok(l) = cholesky_or_null_direction(&hs) {
        let neg_g = sonium_solvers_core::vec_ops::vec_neg(g);
        let y = crate::util::dense_linalg::solve_lower_triangular(&l, &neg_g);
        let d = crate::util::dense_linalg::solve_lower_triangular_transpose(&l, &y);
        if T::vec_norm(&d) <= delta {
            return d;
        }
    }

    // Norm of H (max abs entry) for bracketing.
    let mut h_norm = T::one();
    for i in 0..n {
        for j in 0..n {
            let v = hs[[i, j]].abs();
            if v > h_norm {
                h_norm = v;
            }
        }
    }

    // Bracket: lambda_lo gives non-PD or ||d|| > delta; lambda_hi gives PD and
    // ||d|| <= delta. For lambda >= ||g||/delta + ||H||, ||d|| <= ||g||/(lambda -
    // ||H||) <= delta, so lambda_hi is a valid upper bound.
    let mut lambda_lo = T::zero();
    let mut lambda_hi = g_norm / delta + h_norm;
    let mut lambda = T::max(
        T::machine_epsilon() * h_norm,
        g_norm / (delta * T::from_f64(10.0).unwrap()),
    );
    let tol = T::sqrt_epsilon();
    let mut hard_direction: Option<Vec<T>> = None;
    let mut best_d: Option<Vec<T>> = None; // d at the smallest successful lambda_hi

    for _ in 0..60 {
        // H + lambda*I
        let mut h_lambda = hs.clone();
        for i in 0..n {
            h_lambda[[i, i]] += lambda;
        }

        let l = match cholesky_or_null_direction(&h_lambda) {
            Err(z) => {
                // H + lambda*I not PD: lambda is below -lambda_min(H).
                hard_direction = Some(z);
                lambda_lo = T::max(lambda_lo, lambda);
                // Increase lambda within the bracket.
                lambda = T::max(
                    (lambda_lo * lambda_hi).sqrt(),
                    lambda_lo + T::from_f64(0.5).unwrap() * (lambda_hi - lambda_lo),
                );
                if lambda_hi - lambda_lo <= T::machine_epsilon() * T::max(T::one(), lambda_hi) {
                    break;
                }
                continue;
            }
            Ok(l) => l,
        };

        // Solve (H + lambda*I) d = -g.
        let neg_g = sonium_solvers_core::vec_ops::vec_neg(g);
        let y = crate::util::dense_linalg::solve_lower_triangular(&l, &neg_g);
        let d = crate::util::dense_linalg::solve_lower_triangular_transpose(&l, &y);
        let nd = T::vec_norm(&d);

        if (nd - delta).abs() <= tol * delta {
            return d;
        }

        if nd < delta {
            lambda_hi = lambda;
            best_d = Some(d.clone());
            // Hard case: bracket nearly collapsed with ||d|| < delta throughout
            // and a known near-null direction from a failed factorisation.
            if let Some(ref z) = hard_direction
                && lambda_hi - lambda_lo <= tol * T::max(T::one(), lambda_lo)
            {
                return hard_case_step(g, &hs, &d, z, delta);
            }
        } else {
            lambda_lo = lambda;
        }

        // Newton update on the secular equation (N&W eq. 4.44).
        let q = crate::util::dense_linalg::solve_lower_triangular(&l, &d);
        let nq = T::vec_norm(&q);
        let lambda_newton = if nq > T::machine_epsilon() {
            lambda + (nd / nq) * (nd / nq) * ((nd - delta) / delta)
        } else {
            lambda_hi
        };

        // Safeguard: keep lambda strictly inside the bracket.
        lambda = if lambda_newton <= lambda_lo || lambda_newton >= lambda_hi {
            T::max(
                (lambda_lo * lambda_hi).sqrt(),
                lambda_lo + T::from_f64(0.5).unwrap() * (lambda_hi - lambda_lo),
            )
        } else {
            lambda_newton
        };

        if lambda_hi - lambda_lo <= T::machine_epsilon() * T::max(T::one(), lambda_hi) {
            break;
        }
    }

    // Fallback: best boundary/interior step found, else scaled gradient step.
    if let (Some(d), Some(z)) = (&best_d, &hard_direction) {
        return hard_case_step(g, &hs, d, z, delta);
    }
    match best_d {
        Some(d) => d,
        None => {
            let scale = delta / g_norm;
            g.iter().map(|&gi| -scale * gi).collect::<Vec<_>>()
        }
    }
}

/// Combine `d` with a near-null direction `z` so that `||d + tau*z|| = delta`,
/// choosing the sign of `tau` that gives the lower model value.
fn hard_case_step<T: RealScalar>(
    g: &[T],
    h: &sonium_solvers_core::dense::DenseMatrix<T>,
    d: &[T],
    z: &[T],
    delta: T,
) -> Vec<T> {
    let zz = T::vec_dot(z, z);
    if zz < T::machine_epsilon() {
        return d.to_vec();
    }
    let dz = T::vec_dot(d, z);
    let dd = T::vec_dot(d, d);
    let disc = dz * dz - zz * (dd - delta * delta);
    if disc < T::zero() {
        return d.to_vec();
    }
    let root = disc.sqrt();
    let mut best = d.to_vec();
    let mut best_m = trust_model_value(g, h, d);
    for tau in [(-dz + root) / zz, (-dz - root) / zz] {
        let cand = sonium_solvers_core::vec_ops::vec_add_scaled(d, z, tau);
        let m = trust_model_value(g, h, &cand);
        if m < best_m {
            best_m = m;
            best = cand;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use sonium_solvers_core::traits::ComplexField;

    #[test]
    fn test_cauchy_point_quadratic() {
        // H = I, g = [2, 0], delta = 10
        // Cauchy point should be at -g (full step since within trust region)
        let g = sonium_solvers_core::array![2.0_f64, 0.0];
        let hv = |_x: &[f64], v: &[f64]| v.to_vec();
        let x = sonium_solvers_core::array![0.0, 0.0];
        let dc = cauchy_point(&g, &hv, &x, 10.0);
        assert_relative_eq!(dc[0], -2.0, epsilon = 1e-10);
        assert_relative_eq!(dc[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cauchy_point_clipped() {
        // H = I, g = [10, 0], delta = 1
        // Cauchy point should be clipped to ||dc|| = 1
        let g = sonium_solvers_core::array![10.0_f64, 0.0];
        let hv = |_x: &[f64], v: &[f64]| v.to_vec();
        let x = sonium_solvers_core::array![0.0, 0.0];
        let dc = cauchy_point(&g, &hv, &x, 1.0);
        assert_relative_eq!(f64::vec_norm(&dc), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_steihaug_cg_quadratic() {
        // min g^T d + 0.5 d^T H d, H = I, g = [2, 3]
        // Unconstrained min: d = -g = [-2, -3]
        let g = sonium_solvers_core::array![2.0_f64, 3.0];
        let hv = |_x: &[f64], v: &[f64]| v.to_vec();
        let x = sonium_solvers_core::array![0.0, 0.0];
        let (d, _) = steihaug_cg(&g, &hv, &x, 100.0, 10, 1e-12);
        assert_relative_eq!(d[0], -2.0, epsilon = 1e-8);
        assert_relative_eq!(d[1], -3.0, epsilon = 1e-8);
    }

    #[test]
    fn test_dogleg_newton_inside() {
        let g = sonium_solvers_core::array![1.0_f64, 1.0];
        let newton = sonium_solvers_core::array![-0.5, -0.5];
        let cauchy = sonium_solvers_core::array![-0.3, -0.3];
        let d = dogleg(&g, &newton, &cauchy, 10.0);
        assert_relative_eq!(d[0], -0.5, epsilon = 1e-10);
        assert_relative_eq!(d[1], -0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_dogleg_cauchy_outside_trust_region() {
        // Cauchy step has norm > delta, so dogleg should scale cauchy to trust boundary
        let g = sonium_solvers_core::array![1.0_f64, 1.0];
        let newton = sonium_solvers_core::array![-10.0, -10.0]; // far outside
        let cauchy = sonium_solvers_core::array![-5.0, -5.0]; // ||cauchy|| = 5*sqrt(2) ~ 7.07
        let delta = 1.0;
        let d = dogleg(&g, &newton, &cauchy, delta);
        let d_norm = f64::vec_norm(&d);
        assert_relative_eq!(d_norm, delta, epsilon = 1e-10);
        // Direction should be proportional to cauchy
        let ratio = d[0] / d[1];
        assert_relative_eq!(ratio, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dogleg_interpolation() {
        // Cauchy inside, Newton outside -> interpolates along dogleg path with ||d|| = delta
        let g = sonium_solvers_core::array![1.0_f64, 0.0];
        let cauchy = sonium_solvers_core::array![-0.5, 0.0]; // ||cauchy|| = 0.5
        let newton = sonium_solvers_core::array![-3.0, 0.0]; // ||newton|| = 3.0
        let delta = 1.5;
        let d = dogleg(&g, &newton, &cauchy, delta);
        let d_norm = f64::vec_norm(&d);
        assert_relative_eq!(d_norm, delta, epsilon = 1e-10);
        // Should be along the dogleg path between cauchy and newton (same direction here)
        assert!(
            d[0] < cauchy[0],
            "interpolated step should be beyond cauchy"
        );
    }

    #[test]
    fn test_steihaug_cg_negative_curvature() {
        // H with a negative eigenvalue: H = [[-1, 0], [0, 1]]
        // Steihaug should detect negative curvature (p^T H p < 0) and go to trust boundary
        let g = sonium_solvers_core::array![1.0_f64, 0.0];
        let hv = |_x: &[f64], v: &[f64]| sonium_solvers_core::array![-v[0], v[1]];
        let x = sonium_solvers_core::array![0.0, 0.0];
        let delta = 2.0;
        let (d, hess_evals) = steihaug_cg(&g, &hv, &x, delta, 10, 1e-12);
        let d_norm = f64::vec_norm(&d);
        // Should go to trust region boundary
        assert_relative_eq!(d_norm, delta, epsilon = 1e-8);
        assert!(hess_evals >= 1);
    }

    #[test]
    fn test_steihaug_cg_trust_boundary() {
        // Large gradient, small delta -> step hits trust region boundary
        let g = sonium_solvers_core::array![100.0_f64, 100.0];
        let hv = |_x: &[f64], v: &[f64]| v.to_vec(); // H = I
        let x = sonium_solvers_core::array![0.0, 0.0];
        let delta = 0.1;
        let (d, _) = steihaug_cg(&g, &hv, &x, delta, 10, 1e-12);
        let d_norm = f64::vec_norm(&d);
        // Unconstrained solution is d = -g = [-100, -100], norm ~ 141
        // Should be clipped to trust boundary
        assert_relative_eq!(d_norm, delta, epsilon = 1e-8);
    }

    #[test]
    fn test_more_sorensen_interior() {
        // H = 2I, g = [2, 4], delta large: unconstrained minimiser d = [-1, -2]
        let g = sonium_solvers_core::array![2.0_f64, 4.0];
        let h = sonium_solvers_core::array![[2.0_f64, 0.0], [0.0, 2.0]];
        let d = more_sorensen(&g, &h, 10.0);
        assert_relative_eq!(d[0], -1.0, epsilon = 1e-10);
        assert_relative_eq!(d[1], -2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_more_sorensen_boundary() {
        // H = I, g = [3, 4], delta = 1: exact boundary solution d = -g/(1+lambda)
        // with lambda = 4 -> d = [-0.6, -0.8], model value -4.5.
        let g = sonium_solvers_core::array![3.0_f64, 4.0];
        let h = sonium_solvers_core::array![[1.0_f64, 0.0], [0.0, 1.0]];
        let d = more_sorensen(&g, &h, 1.0);
        assert_relative_eq!(f64::vec_norm(&d), 1.0, epsilon = 1e-8);
        assert_relative_eq!(d[0], -0.6, epsilon = 1e-8);
        assert_relative_eq!(d[1], -0.8, epsilon = 1e-8);
        let m = trust_model_value(&g, &h, &d);
        assert_relative_eq!(m, -4.5, epsilon = 1e-8);
    }

    #[test]
    fn test_more_sorensen_hard_case() {
        // Hard case: H = diag(-1, 2), g = [0, 1] is orthogonal to the
        // eigenvector of the negative eigenvalue. Optimal lambda* = 1 makes
        // H + lambda*I singular; the solution is on the boundary with a
        // component along e0: d = [+-sqrt(8)/3, -1/3], model value -2/3.
        let g = sonium_solvers_core::array![0.0_f64, 1.0];
        let h = sonium_solvers_core::array![[-1.0_f64, 0.0], [0.0, 2.0]];
        let delta = 1.0;
        let d = more_sorensen(&g, &h, delta);
        let d_norm = f64::vec_norm(&d);
        assert_relative_eq!(d_norm, delta, epsilon = 1e-6);
        assert_relative_eq!(d[1], -1.0 / 3.0, epsilon = 1e-4);
        assert_relative_eq!(d[0].abs(), 8.0_f64.sqrt() / 3.0, epsilon = 1e-4);
        let m = trust_model_value(&g, &h, &d);
        assert_relative_eq!(m, -2.0 / 3.0, epsilon = 1e-6);

        // Steihaug-CG stops at the first boundary hit and is strictly worse.
        let hv = |_x: &[f64], v: &[f64]| sonium_solvers_core::array![-v[0], 2.0 * v[1]];
        let x = sonium_solvers_core::array![0.0, 0.0];
        let (d_cg, _) = steihaug_cg(&g, &hv, &x, delta, 50, 1e-12);
        let m_cg = trust_model_value(&g, &h, &d_cg);
        assert!(
            m <= m_cg + 1e-12,
            "More-Sorensen model value {m} should not exceed Steihaug-CG {m_cg}"
        );
    }

    #[test]
    fn test_more_sorensen_near_singular() {
        // Nearly singular H = diag(1e-8, 1), g = [1, 1], delta = 2.
        // Boundary solution with lambda ~ 0.529: d ~ [-1.89, -0.654].
        let g = sonium_solvers_core::array![1.0_f64, 1.0];
        let h = sonium_solvers_core::array![[1e-8_f64, 0.0], [0.0, 1.0]];
        let delta = 2.0;
        let d = more_sorensen(&g, &h, delta);
        let d_norm = f64::vec_norm(&d);
        assert_relative_eq!(d_norm, delta, epsilon = 1e-6);
        assert_relative_eq!(d[0], -1.89, epsilon = 1e-2);
        assert_relative_eq!(d[1], -0.654, epsilon = 1e-2);
    }

    #[test]
    fn test_more_sorensen_f32() {
        let g: Vec<f32> = sonium_solvers_core::array![3.0_f32, 4.0];
        let h: sonium_solvers_core::dense::DenseMatrix<f32> =
            sonium_solvers_core::array![[1.0_f32, 0.0], [0.0, 1.0]];
        let d = more_sorensen(&g, &h, 1.0_f32);
        let d_norm = f32::vec_norm(&d);
        assert!((d_norm - 1.0).abs() < 1e-3, "d_norm={d_norm}");
        assert!((d[0] + 0.6).abs() < 1e-3, "d[0]={}", d[0]);
        assert!((d[1] + 0.8).abs() < 1e-3, "d[1]={}", d[1]);
    }
}
