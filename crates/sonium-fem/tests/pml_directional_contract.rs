//! Directional PML coefficient contract with an independent complex-stretch reference.

use num_complex::Complex64;
use sonium_fem::boundary::{PmlDirection, PmlRegion};
use sonium_fem::mesh::Point;

fn close(actual: Complex64, expected: Complex64) -> bool {
    (actual - expected).norm() < 1.0e-12
}

pub fn run_directional_pml_contract() {
    let y_pml = PmlRegion::y_positive(0.5, 0.25, 100.0, 8.0);
    assert_eq!(y_pml.direction, PmlDirection::Y);
    let inner = Point::new_2d(0.2, 0.5);
    let midpoint = Point::new_2d(0.2, 0.625);
    let outer = Point::new_2d(0.2, 0.75);
    let sigma_mid = 100.0 * 0.5_f64.powi(2);
    let sigma_outer = 100.0;
    let sy_mid = Complex64::new(1.0, sigma_mid / 8.0);
    let sy_outer = Complex64::new(1.0, sigma_outer / 8.0);
    assert!(close(y_pml.stretching_y(inner.y), Complex64::new(1.0, 0.0)));
    assert!(close(y_pml.stretching_y(midpoint.y), sy_mid));
    assert!(close(y_pml.stretching_y(outer.y), sy_outer));
    assert!(close(
        y_pml.stretching_x(midpoint.x),
        Complex64::new(1.0, 0.0)
    ));
    assert!(close(y_pml.mass_coefficient(&midpoint), sy_mid));
    assert!(close(
        y_pml.stiffness_coefficient_y(&midpoint),
        Complex64::new(1.0, 0.0) / sy_mid
    ));
    assert!(close(y_pml.stiffness_coefficient_x(&midpoint), sy_mid));

    let xy_pml = PmlRegion::new(
        PmlDirection::XY,
        [0.5, 0.5, f64::NEG_INFINITY],
        [0.75, 0.75, f64::INFINITY],
        100.0,
        8.0,
    );
    let corner = Point::new_2d(0.625, 0.625);
    let expected_factor = sy_mid * sy_mid;
    assert!(close(xy_pml.mass_coefficient(&corner), expected_factor));

    let x_negative = PmlRegion::x_negative(0.0, 0.25, 100.0, 8.0);
    assert_eq!(x_negative.direction, PmlDirection::X);
    let x_inner = Point::new_2d(0.0, 0.2);
    let x_midpoint = Point::new_2d(-0.125, 0.2);
    let x_outer = Point::new_2d(-0.25, 0.2);
    let sx_mid = Complex64::new(1.0, sigma_mid / 8.0);
    let sx_outer = Complex64::new(1.0, sigma_outer / 8.0);
    assert!(close(
        x_negative.stretching_x(x_inner.x),
        Complex64::new(1.0, 0.0)
    ));
    assert!(close(x_negative.stretching_x(x_midpoint.x), sx_mid));
    assert!(close(x_negative.stretching_x(x_outer.x), sx_outer));
    assert!(close(
        x_negative.stretching_y(x_midpoint.y),
        Complex64::new(1.0, 0.0)
    ));
    assert!(close(x_negative.mass_coefficient(&x_midpoint), sx_mid));
    assert!(close(
        x_negative.stiffness_coefficient_x(&x_midpoint),
        Complex64::new(1.0, 0.0) / sx_mid
    ));
    assert!(close(
        x_negative.stiffness_coefficient_y(&x_midpoint),
        sx_mid
    ));

    let y_negative = PmlRegion::y_negative(0.0, 0.25, 100.0, 8.0);
    assert_eq!(y_negative.direction, PmlDirection::Y);
    let y_inner = Point::new_2d(0.2, 0.0);
    let y_midpoint = Point::new_2d(0.2, -0.125);
    let y_outer = Point::new_2d(0.2, -0.25);
    assert!(close(
        y_negative.stretching_y(y_inner.y),
        Complex64::new(1.0, 0.0)
    ));
    assert!(close(y_negative.stretching_y(y_midpoint.y), sy_mid));
    assert!(close(y_negative.stretching_y(y_outer.y), sy_outer));
    assert!(close(
        y_negative.stretching_x(y_midpoint.x),
        Complex64::new(1.0, 0.0)
    ));
    assert!(close(y_negative.mass_coefficient(&y_midpoint), sy_mid));
    assert!(close(
        y_negative.stiffness_coefficient_y(&y_midpoint),
        Complex64::new(1.0, 0.0) / sy_mid
    ));
    assert!(close(
        y_negative.stiffness_coefficient_x(&y_midpoint),
        sy_mid
    ));

    assert!(close(xy_pml.stretching_x(corner.x), sy_mid));
    assert!(close(xy_pml.stretching_y(corner.y), sy_mid));
    assert!(close(
        xy_pml.stiffness_coefficient_x(&corner),
        Complex64::new(1.0, 0.0)
    ));
    assert!(close(
        xy_pml.stiffness_coefficient_y(&corner),
        Complex64::new(1.0, 0.0)
    ));
}

#[test]
fn directional_pml_matches_complex_stretch_reference() {
    run_directional_pml_contract();
}
