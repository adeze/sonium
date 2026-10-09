//! Independent convergence reference for the CPU 2-D multi-region BEM path.
//!
//! This is a dimensional-reduction reference, not a 3-D axisymmetric solver:
//! the production operator is the constant-panel Müller transmission BEM for
//! an infinite penetrable circular cylinder.  The reference is an independently
//! evaluated cylindrical Fourier series from `sonium-wave`.

use num_complex::Complex64;
use sonium_bem::{TwoDFluidCylinder, TwoDPlaneWave};
use sonium_wave::analytical::FluidCylinderTransmission2d;
use sonium_xem::{
    AcousticMedium, ConventionRegistry, DensityKgM3, FrequencyHz, LengthM, SpeedOfSoundMPerS,
};
use std::f64::consts::PI;

const RADIUS_M: f64 = 0.4;
const EXTERIOR_K: f64 = 2.0;
const INTERIOR_K: f64 = 3.0;
const EXTERIOR_C: f64 = 343.0;
const EXTERIOR_RHO: f64 = 1.21;
const INTERIOR_RHO: f64 = 1.5;
const FREQUENCY_HZ: f64 = EXTERIOR_K * EXTERIOR_C / (2.0 * PI);
const INTERIOR_C: f64 = 2.0 * PI * FREQUENCY_HZ / INTERIOR_K;
const FOURIER_TERMS: usize = 40;
const POWER_SAMPLES: usize = 512;
const REFERENCE_FAR_FIELD_RADIUS_M: f64 = 40.0;
const RESIDUAL_TOLERANCE: f64 = 1.0e-10;

const FORMULATION: &str = "CPU 2-D constant-panel Muller transmission BEM";
const QUANTITY: &str =
    "complex peak pressure [Pa] and scattered acoustic power per unit length [W/m]";
const REFERENCE: &str = "independent cylindrical Fourier transmission series from sonium-wave";
const PROVENANCE: &str = "analytic/deterministic-fixture:bem-2d-fluid-cylinder-convergence-v1";

fn medium(speed: f64, density: f64) -> AcousticMedium {
    let mut conventions = ConventionRegistry::default();
    // The solve carries peak complex pressure amplitudes.  The production
    // power helper applies the 1/2 factor for peak phasors.
    conventions.rms_amplitudes = false;
    AcousticMedium::try_new_with_conventions(
        FrequencyHz::try_new(FREQUENCY_HZ).expect("finite frequency"),
        SpeedOfSoundMPerS::try_new(speed).expect("finite sound speed"),
        DensityKgM3::try_new(density).expect("finite density"),
        conventions,
    )
    .expect("validated XEM acoustic medium")
}

fn solve(panel_count: usize) -> sonium_bem::TwoDFluidCylinderSolution {
    TwoDFluidCylinder::try_from_xem(
        LengthM::try_new(RADIUS_M).expect("finite radius"),
        medium(EXTERIOR_C, EXTERIOR_RHO),
        medium(INTERIOR_C, INTERIOR_RHO),
        panel_count,
    )
    .expect("valid multi-region XEM problem")
    .solve(TwoDPlaneWave::plane_wave_x(), RESIDUAL_TOLERANCE)
    .expect("CPU Müller transmission solve")
}

fn reference() -> FluidCylinderTransmission2d {
    FluidCylinderTransmission2d::try_new(
        RADIUS_M,
        EXTERIOR_K,
        INTERIOR_K,
        EXTERIOR_RHO,
        INTERIOR_RHO,
        FOURIER_TERMS,
    )
    .expect("independent Fourier reference")
}

fn exterior_points() -> Vec<[f64; 2]> {
    [0.65_f64, 0.82, 1.05, 1.30]
        .into_iter()
        .zip([0.37_f64, 0.97, 1.83, 2.71])
        .map(|(radius, angle)| [radius * angle.cos(), radius * angle.sin()])
        .collect()
}

fn interior_points() -> Vec<[f64; 2]> {
    [0.08_f64, 0.23, 0.34]
        .into_iter()
        .zip([0.37_f64, 1.41, 2.62])
        .map(|(radius, angle)| [radius * angle.cos(), radius * angle.sin()])
        .collect()
}

fn relative_complex_error(actual: Complex64, expected: Complex64) -> f64 {
    (actual - expected).norm() / actual.norm().max(expected.norm()).max(f64::MIN_POSITIVE)
}

fn max_field_errors(
    solution: &sonium_bem::TwoDFluidCylinderSolution,
    reference: &FluidCylinderTransmission2d,
) -> (f64, f64) {
    let exterior = exterior_points();
    let interior = interior_points();
    let exterior_actual = solution
        .evaluate_exterior_total_pressure(&exterior)
        .expect("exterior field evaluation");
    let interior_actual = solution
        .evaluate_interior_pressure(&interior)
        .expect("interior field evaluation");
    let exterior_error = exterior_actual
        .iter()
        .zip(&exterior)
        .map(|(actual, point)| {
            relative_complex_error(
                *actual,
                reference.exterior_pressure(point[0].hypot(point[1]), point[1].atan2(point[0])),
            )
        })
        .fold(0.0, f64::max);
    let interior_error = interior_actual
        .iter()
        .zip(&interior)
        .map(|(actual, point)| {
            relative_complex_error(
                *actual,
                reference.interior_pressure(point[0].hypot(point[1]), point[1].atan2(point[0])),
            )
        })
        .fold(0.0, f64::max);
    (exterior_error, interior_error)
}

fn reference_scattered_power(reference: &FluidCylinderTransmission2d) -> f64 {
    let angular_integral = (0..POWER_SAMPLES)
        .map(|index| {
            let angle = 2.0 * PI * index as f64 / POWER_SAMPLES as f64;
            let point = [
                REFERENCE_FAR_FIELD_RADIUS_M * angle.cos(),
                REFERENCE_FAR_FIELD_RADIUS_M * angle.sin(),
            ];
            let incident = Complex64::from_polar(1.0, EXTERIOR_K * point[0]);
            let scattered =
                reference.exterior_pressure(REFERENCE_FAR_FIELD_RADIUS_M, angle) - incident;
            let phase =
                Complex64::from_polar(1.0, -(EXTERIOR_K * REFERENCE_FAR_FIELD_RADIUS_M + PI / 4.0));
            let far_field =
                scattered * (8.0 * PI * EXTERIOR_K * REFERENCE_FAR_FIELD_RADIUS_M).sqrt() * phase;
            far_field.norm_sqr()
        })
        .sum::<f64>()
        * (2.0 * PI / POWER_SAMPLES as f64);
    // The Fourier reference uses unit peak pressure and the same outgoing
    // convention as the XEM `exp(-iωt)` solve.  The production API reports
    // RMS power, hence the explicit peak-to-RMS 1/2 factor.
    0.5 * angular_integral / (8.0 * PI * EXTERIOR_K * EXTERIOR_RHO * EXTERIOR_C)
}

fn relative_power_error(
    solution: &sonium_bem::TwoDFluidCylinderSolution,
    reference_power: f64,
) -> f64 {
    let actual = solution
        .scattered_power_per_length(POWER_SAMPLES)
        .expect("scattered power per unit length");
    (actual - reference_power).abs() / actual.abs().max(reference_power.abs()).max(1.0e-12)
}

pub fn multi_region_constant_panel_refinement_converges_in_field_and_power() {
    let reference = reference();
    let reference_power = reference_scattered_power(&reference);
    let coarse = solve(64);
    let medium = solve(128);
    let fine = solve(256);

    let coarse_errors = max_field_errors(&coarse, &reference);
    let medium_errors = max_field_errors(&medium, &reference);
    let fine_errors = max_field_errors(&fine, &reference);
    let coarse_power_error = relative_power_error(&coarse, reference_power);
    let medium_power_error = relative_power_error(&medium, reference_power);
    let fine_power_error = relative_power_error(&fine, reference_power);

    for value in [
        reference_power,
        coarse.relative_residual,
        medium.relative_residual,
        fine.relative_residual,
        coarse_errors.0,
        coarse_errors.1,
        medium_errors.0,
        medium_errors.1,
        fine_errors.0,
        fine_errors.1,
        coarse_power_error,
        medium_power_error,
        fine_power_error,
    ] {
        assert!(value.is_finite(), "non-finite reference metric: {value}");
    }

    assert!(coarse.relative_residual <= RESIDUAL_TOLERANCE);
    assert!(medium.relative_residual <= RESIDUAL_TOLERANCE);
    assert!(fine.relative_residual <= RESIDUAL_TOLERANCE);
    assert!(fine_errors.0 < medium_errors.0 && medium_errors.0 < coarse_errors.0);
    assert!(fine_errors.1 < medium_errors.1 && medium_errors.1 < coarse_errors.1);
    // Power is integrated from a finite-radius far-field proxy, so its
    // discretisation error need not decrease monotonically at every mesh.
    // The fine mesh must nevertheless improve on the coarse mesh and stay
    // below the bounded production-reference budget.
    assert!(fine_power_error < coarse_power_error);
    assert!(fine_power_error < 1.0e-3);
    assert!(fine_errors.0 < 0.12 && fine_errors.1 < 0.12);
    assert!(fine_power_error < 0.12);

    println!(
        "BEM_EVIDENCE: {{\"case_id\":\"bem.2d-fluid-cylinder-convergence.v1\",\"status\":\"passed\",\"formulation\":\"{FORMULATION}\",\"quantity\":\"{QUANTITY}\",\"reference\":\"{REFERENCE}\",\"provenance\":\"{PROVENANCE}\",\"reference_power_w_per_m\":{reference_power:.9e},\"coarse_exterior_error\":{:.9e},\"medium_exterior_error\":{:.9e},\"fine_exterior_error\":{:.9e},\"coarse_interior_error\":{:.9e},\"medium_interior_error\":{:.9e},\"fine_interior_error\":{:.9e},\"coarse_power_error\":{coarse_power_error:.9e},\"medium_power_error\":{medium_power_error:.9e},\"fine_power_error\":{fine_power_error:.9e},\"fine_residual\":{:.9e}}}",
        coarse_errors.0,
        medium_errors.0,
        fine_errors.0,
        coarse_errors.1,
        medium_errors.1,
        fine_errors.1,
        fine.relative_residual,
    );
}

#[test]
fn multi_region_constant_panel_refinement_test() {
    multi_region_constant_panel_refinement_converges_in_field_and_power();
}
