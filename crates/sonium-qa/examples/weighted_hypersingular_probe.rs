//! Independent polynomial-density check of the weighted singular kernel.

// Rust guideline compliant 2026-02-21

use sonium_bem::core::integration::singular::{
    QuadratureParams, singular_integration_linear_with_params,
};
use sonium_bem::core::types::{ElementType, PhysicsParams};
use sonium_solvers::dense::DenseMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let coordinates = DenseMatrix::from_row_major_vec(
        4,
        3,
        vec![
            1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, -1.0, 0.0,
        ],
    );
    let mut physics = PhysicsParams::new(1.0, 343.0, 1.21, false);
    // Isolate the Laplace kernel without calling a positive-frequency
    // constructor with an invalid zero frequency.
    physics.wave_number = 0.0;
    // phi(x,y)=x^2 vanishes quadratically at the source, so its
    // hypersingular integral is ordinary and absolutely integrable:
    // int_square x^2/(4*pi*r^3) dA = asinh(1)/pi.
    let density = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let exact = 1.0_f64.asinh() / std::f64::consts::PI;
    let mut final_error = f64::INFINITY;
    for order in [8, 12, 16] {
        let result = singular_integration_linear_with_params(
            &[0.0; 3],
            &[0.0, 0.0, 1.0],
            &coordinates,
            ElementType::Quad4,
            &physics,
            9,
            &QuadratureParams {
                edge_gauss_order: order,
                subelement_gauss_order: order,
                edge_sections: 4,
                subtriangles_per_section: 2,
            },
            None,
        );
        let actual = result
            .basis_results
            .iter()
            .zip(density)
            .map(|(basis, value)| basis.d2g_dnxdny_integral * value)
            .sum::<num_complex::Complex64>();
        final_error = (actual - exact).norm() / exact;
        println!(
            "order={order}, weighted_W={actual}, exact={exact:.12e}, relative_error={final_error:.12e}"
        );
    }
    if !final_error.is_finite() || final_error > 1e-8 {
        return Err("weighted singular kernel fails independent quadratic-density integral".into());
    }
    Ok(())
}
