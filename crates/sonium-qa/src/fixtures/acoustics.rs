//! Acoustic fixtures for BEM/FEM integration tests

use num_complex::Complex64;
use sonium_solvers::{CsrBuilder, CsrMatrix};

pub fn rigid_sphere_bem_problem(
    radius: f64,
    frequency: f64,
    n_theta: usize,
    n_phi: usize,
) -> (Vec<[f64; 3]>, Vec<[usize; 3]>, f64, f64) {
    use std::f64::consts::PI;
    let c0 = 343.0;
    let rho = 1.21;
    let k = 2.0 * PI * frequency / c0;

    let mut points = Vec::with_capacity(n_theta * n_phi);
    let mut elements = Vec::with_capacity(2 * (n_theta - 1) * (n_phi - 1));

    for i in 0..n_theta {
        let theta = PI * (i as f64) / ((n_theta - 1) as f64);
        for j in 0..n_phi {
            let phi = 2.0 * PI * (j as f64) / (n_phi as f64);
            let x = radius * theta.sin() * phi.cos();
            let y = radius * theta.sin() * phi.sin();
            let z = radius * theta.cos();
            points.push([x, y, z]);
        }
    }

    let idx = |i, j| i * n_phi + j;
    for i in 0..n_theta - 1 {
        for j in 0..n_phi - 1 {
            elements.push([idx(i, j), idx(i + 1, j), idx(i, j + 1)]);
            elements.push([idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]);
        }
    }

    (points, elements, k, rho)
}

pub fn plane_wave_pressure(
    points: &[[f64; 3]],
    k: f64,
    theta_inc: f64,
    phi_inc: f64,
) -> Vec<Complex64> {
    let inc_x = theta_inc.sin() * phi_inc.cos();
    let inc_y = theta_inc.sin() * phi_inc.sin();
    let inc_z = theta_inc.cos();

    let mut p_inc = vec![Complex64::new(0.0, 0.0); points.len()];
    for (i, pt) in points.iter().enumerate() {
        let phase = k * (inc_x * pt[0] + inc_y * pt[1] + inc_z * pt[2]);
        p_inc[i] = Complex64::new(phase.cos(), phase.sin());
    }
    p_inc
}

pub fn helmholtz_1d_matrix(n: usize, k: f64) -> CsrMatrix<Complex64> {
    let h = 1.0 / (n as f64);
    let h2 = h * h;

    let mut builder = CsrBuilder::new(n, n);
    for i in 0..n {
        let mut entries = vec![(i, Complex64::new(2.0 / h2 - k * k, 0.0))];
        if i > 0 {
            entries.push((i - 1, Complex64::new(-1.0 / h2, 0.0)));
        }
        if i < n - 1 {
            entries.push((i + 1, Complex64::new(-1.0 / h2, 0.0)));
        }
        builder.add_row_entries(entries.into_iter());
    }
    builder.finish()
}

pub fn analytical_1d_helmholtz(n: usize, k: f64) -> Vec<Complex64> {
    let h = 1.0 / (n as f64);
    let mut u = vec![Complex64::new(0.0, 0.0); n];
    for i in 0..n {
        let x = (i as f64 + 0.5) * h;
        u[i] = Complex64::new((k * x).cos(), (k * x).sin());
    }
    u
}
