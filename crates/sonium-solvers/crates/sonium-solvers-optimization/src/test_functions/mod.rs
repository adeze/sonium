#![doc = include_str!("README.md")]
#![doc = include_str!("REFERENCES.md")]
#![allow(unused)]

pub use functions::*;

pub mod functions;

mod get;
mod misc;
mod types;

pub use get::*;
pub use misc::*;
pub use types::*;

// ── Analytic gradients, starts and minima for the classic subset ──
// (kept from the previous `test_functions` module; the value functions
// above are the ported 106-function registry).

pub fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
    let dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
    let dy = 200.0 * (x[1] - x[0] * x[0]);
    sonium_solvers_core::array![dx, dy]
}

pub fn rosenbrock_hess_vec(x: &[f64], v: &[f64]) -> Vec<f64> {
    // H = [[2 - 400*(y - 3*x^2), -400*x],
    //      [-400*x,                200  ]]
    let h00 = 2.0 - 400.0 * (x[1] - 3.0 * x[0] * x[0]);
    let h01 = -400.0 * x[0];
    let h11 = 200.0;
    sonium_solvers_core::array![h00 * v[0] + h01 * v[1], h01 * v[0] + h11 * v[1]]
}

pub fn rosenbrock_start() -> Vec<f64> {
    sonium_solvers_core::array![-1.2, 1.0]
}

pub fn rosenbrock_min() -> Vec<f64> {
    sonium_solvers_core::array![1.0, 1.0]
}

pub fn sphere_grad(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>()
}

pub fn sphere_hess_vec(_x: &[f64], v: &[f64]) -> Vec<f64> {
    v.iter().map(|&vi| 2.0 * vi).collect::<Vec<_>>()
}

pub fn beale_grad(x: &[f64]) -> Vec<f64> {
    let (a, b) = (x[0], x[1]);
    let t1 = 1.5 - a + a * b;
    let t2 = 2.25 - a + a * b * b;
    let t3 = 2.625 - a + a * b * b * b;
    let da = 2.0 * t1 * (-1.0 + b) + 2.0 * t2 * (-1.0 + b * b) + 2.0 * t3 * (-1.0 + b * b * b);
    let db = 2.0 * t1 * a + 2.0 * t2 * 2.0 * a * b + 2.0 * t3 * 3.0 * a * b * b;
    sonium_solvers_core::array![da, db]
}

pub fn beale_start() -> Vec<f64> {
    sonium_solvers_core::array![0.0, 0.0]
}

pub fn beale_min() -> Vec<f64> {
    sonium_solvers_core::array![3.0, 0.5]
}

pub fn booth_grad(x: &[f64]) -> Vec<f64> {
    let a = x[0] + 2.0 * x[1] - 7.0;
    let b = 2.0 * x[0] + x[1] - 5.0;
    sonium_solvers_core::array![2.0 * a + 4.0 * b, 4.0 * a + 2.0 * b]
}

pub fn booth_start() -> Vec<f64> {
    sonium_solvers_core::array![0.0, 0.0]
}

pub fn booth_min() -> Vec<f64> {
    sonium_solvers_core::array![1.0, 3.0]
}

pub fn himmelblau_grad(x: &[f64]) -> Vec<f64> {
    let a = x[0] * x[0] + x[1] - 11.0;
    let b = x[0] + x[1] * x[1] - 7.0;
    sonium_solvers_core::array![4.0 * x[0] * a + 2.0 * b, 2.0 * a + 4.0 * x[1] * b]
}

#[allow(dead_code)]
pub fn himmelblau_start() -> Vec<f64> {
    sonium_solvers_core::array![0.0, 0.0]
}

/// N-dimensional Rosenbrock: `f(x) = sum_{i=0}^{n-2} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]`
pub fn rosenbrock_nd(x: &[f64]) -> f64 {
    let n = x.len();
    let mut f = 0.0;
    for i in 0..(n - 1) {
        let a = 1.0 - x[i];
        let b = x[i + 1] - x[i] * x[i];
        f += a * a + 100.0 * b * b;
    }
    f
}

pub fn rosenbrock_nd_grad(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut g = vec![0.0; n];
    for i in 0..(n - 1) {
        g[i] += -2.0 * (1.0 - x[i]) - 400.0 * x[i] * (x[i + 1] - x[i] * x[i]);
        g[i + 1] += 200.0 * (x[i + 1] - x[i] * x[i]);
    }
    g
}

pub fn rosenbrock_nd_start(n: usize) -> Vec<f64> {
    let mut x = vec![1.0; n];
    for i in (0..n).step_by(2) {
        x[i] = -1.2;
    }
    x
}

/// Styblinski-Tang: `f(x) = 0.5 * sum(x_i^4 - 16*x_i^2 + 5*x_i)`
/// Minimum at x_i ≈ -2.903534, f ≈ -39.16617 * n.
pub fn styblinski_tang(x: &[f64]) -> f64 {
    0.5 * x
        .iter()
        .map(|&xi| xi.powi(4) - 16.0 * xi * xi + 5.0 * xi)
        .sum::<f64>()
}

pub fn styblinski_tang_grad(x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&xi| 0.5 * (4.0 * xi.powi(3) - 32.0 * xi + 5.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::finite_diff::{gradient_central_auto, hessian_vec_product};
    use approx::assert_relative_eq;

    // ── 1. Function value at known minimum ──────────────────────────────

    #[test]
    fn rosenbrock_value_at_minimum() {
        let x = rosenbrock_min();
        assert_relative_eq!(rosenbrock(&x), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn sphere_value_at_minimum() {
        let x = vec![0.0; 5];
        assert_relative_eq!(sphere(&x), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn beale_value_at_minimum() {
        let x = beale_min();
        assert_relative_eq!(beale(&x), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn booth_value_at_minimum() {
        let x = booth_min();
        assert_relative_eq!(booth(&x), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn himmelblau_value_at_minimum() {
        let x = sonium_solvers_core::array![3.0, 2.0];
        assert_relative_eq!(himmelblau(&x), 0.0, epsilon = 1e-15);
    }

    // ── 2. Gradient is zero at minimum ──────────────────────────────────

    #[test]
    fn rosenbrock_grad_zero_at_minimum() {
        let x = rosenbrock_min();
        let g = rosenbrock_grad(&x);
        for &gi in g.iter() {
            assert_relative_eq!(gi, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn sphere_grad_zero_at_minimum() {
        let x = vec![0.0; 5];
        let g = sphere_grad(&x);
        for &gi in g.iter() {
            assert_relative_eq!(gi, 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn beale_grad_zero_at_minimum() {
        let x = beale_min();
        let g = beale_grad(&x);
        for &gi in g.iter() {
            assert_relative_eq!(gi, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn booth_grad_zero_at_minimum() {
        let x = booth_min();
        let g = booth_grad(&x);
        for &gi in g.iter() {
            assert_relative_eq!(gi, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn himmelblau_grad_zero_at_minimum() {
        let x = sonium_solvers_core::array![3.0, 2.0];
        let g = himmelblau_grad(&x);
        for &gi in g.iter() {
            assert_relative_eq!(gi, 0.0, epsilon = 1e-12);
        }
    }

    // ── 3. Gradient matches finite differences ──────────────────────────

    fn assert_grad_matches_finite_diff<F, G>(f: F, grad_fn: G, x: &[f64], tol: f64)
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let g_analytic = grad_fn(x);
        let g_numeric = gradient_central_auto(&f, x);
        assert_eq!(g_analytic.len(), g_numeric.len());
        for i in 0..g_analytic.len() {
            assert_relative_eq!(g_analytic[i], g_numeric[i], epsilon = tol);
        }
    }

    #[test]
    fn rosenbrock_grad_vs_finite_diff() {
        let x = rosenbrock_start();
        assert_grad_matches_finite_diff(rosenbrock, rosenbrock_grad, &x, 1e-7);
    }

    #[test]
    fn sphere_grad_vs_finite_diff() {
        let x = sonium_solvers_core::array![3.0, 4.0];
        assert_grad_matches_finite_diff(sphere, sphere_grad, &x, 1e-8);
    }

    #[test]
    fn beale_grad_vs_finite_diff() {
        let x = beale_start();
        assert_grad_matches_finite_diff(beale, beale_grad, &x, 1e-7);
    }

    #[test]
    fn booth_grad_vs_finite_diff() {
        let x = booth_start();
        assert_grad_matches_finite_diff(booth, booth_grad, &x, 1e-8);
    }

    #[test]
    fn rosenbrock_nd_grad_vs_finite_diff() {
        let x = rosenbrock_nd_start(5);
        assert_grad_matches_finite_diff(rosenbrock_nd, rosenbrock_nd_grad, &x, 1e-7);
    }

    #[test]
    fn styblinski_tang_grad_vs_finite_diff() {
        let x = sonium_solvers_core::array![1.0, -1.0];
        assert_grad_matches_finite_diff(styblinski_tang, styblinski_tang_grad, &x, 1e-7);
    }

    // ── 4. Hessian-vector product matches finite differences ────────────

    #[test]
    fn rosenbrock_hess_vec_vs_finite_diff() {
        let x = rosenbrock_start();
        let h = f64::EPSILON.sqrt();

        let v1 = sonium_solvers_core::array![1.0, 0.0];
        let hv_analytic = rosenbrock_hess_vec(&x, &v1);
        let hv_numeric = hessian_vec_product(&rosenbrock_grad, &x, &v1, h);
        for i in 0..2 {
            assert_relative_eq!(
                hv_analytic[i],
                hv_numeric[i],
                max_relative = 1e-4,
                epsilon = 1e-8
            );
        }

        let v2 = sonium_solvers_core::array![0.0, 1.0];
        let hv_analytic = rosenbrock_hess_vec(&x, &v2);
        let hv_numeric = hessian_vec_product(&rosenbrock_grad, &x, &v2, h);
        for i in 0..2 {
            assert_relative_eq!(
                hv_analytic[i],
                hv_numeric[i],
                max_relative = 1e-4,
                epsilon = 1e-8
            );
        }
    }

    #[test]
    fn sphere_hess_vec_vs_finite_diff() {
        let x = sonium_solvers_core::array![3.0, 4.0, -1.0];
        let v = sonium_solvers_core::array![0.5, -0.3, 1.7];
        let h = f64::EPSILON.sqrt();

        let hv_analytic = sphere_hess_vec(&x, &v);
        let hv_numeric = hessian_vec_product(&sphere_grad, &x, &v, h);
        for i in 0..3 {
            assert_relative_eq!(hv_analytic[i], hv_numeric[i], epsilon = 1e-5);
        }
    }
}
