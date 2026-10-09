//! Validate scalable high-order triangle quadrature for oscillatory BEM factors.
//!
//! The reference uses an independently written 32×32 tensor Gauss integral
//! through a Duffy map. Candidate rules come from `triangle_quadrature`.

use num_complex::Complex64;
use sonium_bem::core::integration::{gauss_legendre, triangle_quadrature};
use sonium_qa::relative_error;

fn plane_wave(x: f64, y: f64, phase_x: f64, phase_y: f64) -> Complex64 {
    let phase = phase_x * x + phase_y * y;
    Complex64::new(phase.cos(), phase.sin())
}

fn tensor_gauss_reference(phase_x: f64, phase_y: f64) -> Complex64 {
    let (points, weights) = gauss_legendre(32);
    let mut integral = Complex64::new(0.0, 0.0);
    for (&u_raw, &u_weight_raw) in points.iter().zip(&weights) {
        let u = 0.5 * (u_raw + 1.0);
        let u_weight = 0.5 * u_weight_raw;
        for (&v_raw, &v_weight_raw) in points.iter().zip(&weights) {
            let v = 0.5 * (v_raw + 1.0);
            let v_weight = 0.5 * v_weight_raw;
            let x = u;
            let y = (1.0 - u) * v;
            integral += u_weight * v_weight * (1.0 - u) * plane_wave(x, y, phase_x, phase_y);
        }
    }
    integral
}

fn candidate_integral(order: usize, phase_x: f64, phase_y: f64) -> Complex64 {
    triangle_quadrature(order)
        .iter()
        .map(|&(x, y, weight)| weight * plane_wave(x, y, phase_x, phase_y))
        .sum()
}

fn test_high_order_quad() {
    println!("=== High-Order Triangle Quadrature ===");
    for (phase_x, phase_y) in [(0.7, -0.4), (3.0, -2.0), (8.0, 5.0)] {
        let reference = tensor_gauss_reference(phase_x, phase_y);
        let mut errors = Vec::new();
        println!("\nphase=({phase_x:.1}, {phase_y:.1})");
        for order in [1, 2, 3, 4, 8, 12] {
            let candidate = candidate_integral(order, phase_x, phase_y);
            let error = relative_error(&[candidate], &[reference]);
            assert!(error.is_finite(), "non-finite error for order {order}");
            errors.push(error);
            println!("  order={order:2}: complex relative error={error:.3e}");
        }

        assert!(
            errors[5] < 1e-11,
            "order-12 oscillatory triangle integral must match the independent reference: {errors:?}"
        );
        assert!(
            errors[5] < errors[3] * 1e-3,
            "requested high order must materially improve on the fixed 13-point rule: {errors:?}"
        );
    }
}

fn main() {
    let result = std::panic::catch_unwind(test_high_order_quad);
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
