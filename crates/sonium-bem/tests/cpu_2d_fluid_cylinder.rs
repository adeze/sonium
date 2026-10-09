//! Production CPU 2-D penetrable-cylinder transmission reference.

use num_complex::Complex64;
use sonium_bem::{TwoDBemError, TwoDFluidCylinder, TwoDPlaneWave};
use sonium_wave::analytical::FluidCylinderTransmission2d;
use sonium_xem::{
    AcousticMedium, ConventionRegistry, DensityKgM3, FrequencyHz, LengthM, SpeedOfSoundMPerS,
};
use std::f64::consts::PI;

const RADIUS: f64 = 0.4;
const EXTERIOR_WAVE_NUMBER: f64 = 2.0;
const INTERIOR_WAVE_NUMBER: f64 = 3.0;
const EXTERIOR_SPEED: f64 = 343.0;
const EXTERIOR_DENSITY: f64 = 1.21;
const INTERIOR_DENSITY: f64 = 1.5;
const FREQUENCY: f64 = EXTERIOR_WAVE_NUMBER * EXTERIOR_SPEED / (2.0 * PI);
const INTERIOR_SPEED: f64 = 2.0 * PI * FREQUENCY / INTERIOR_WAVE_NUMBER;
const FAR_FIELD_REFERENCE_RADIUS: f64 = 40.0;
const FAR_FIELD_ANGLES: usize = 24;

fn problem() -> TwoDFluidCylinder {
    let mut conventions = ConventionRegistry::default();
    conventions.rms_amplitudes = false;
    let exterior = AcousticMedium::try_new_with_conventions(
        FrequencyHz::try_new(FREQUENCY).unwrap(),
        SpeedOfSoundMPerS::try_new(EXTERIOR_SPEED).unwrap(),
        DensityKgM3::try_new(EXTERIOR_DENSITY).unwrap(),
        conventions,
    )
    .unwrap();
    let interior = AcousticMedium::try_new_with_conventions(
        FrequencyHz::try_new(FREQUENCY).unwrap(),
        SpeedOfSoundMPerS::try_new(INTERIOR_SPEED).unwrap(),
        DensityKgM3::try_new(INTERIOR_DENSITY).unwrap(),
        conventions,
    )
    .unwrap();
    TwoDFluidCylinder::try_from_xem(LengthM::try_new(RADIUS).unwrap(), exterior, interior, 128)
        .unwrap()
}

fn reference() -> FluidCylinderTransmission2d {
    FluidCylinderTransmission2d::try_new(
        RADIUS,
        EXTERIOR_WAVE_NUMBER,
        INTERIOR_WAVE_NUMBER,
        EXTERIOR_DENSITY,
        INTERIOR_DENSITY,
        40,
    )
    .unwrap()
}

fn reference_incident(point: [f64; 2]) -> Complex64 {
    Complex64::from_polar(1.0, EXTERIOR_WAVE_NUMBER * point[0])
}

fn reference_far_field_amplitude(angle: f64) -> Complex64 {
    let radius = FAR_FIELD_REFERENCE_RADIUS;
    let point = [radius * angle.cos(), radius * angle.sin()];
    let scattered = reference().exterior_pressure(radius, angle) - reference_incident(point);
    let outgoing_phase = Complex64::from_polar(1.0, -(EXTERIOR_WAVE_NUMBER * radius + PI / 4.0));
    scattered * (8.0 * PI * EXTERIOR_WAVE_NUMBER * radius).sqrt() * outgoing_phase
}

fn reference_scattered_power(radius: f64, samples: usize) -> f64 {
    let reference = reference();
    let sum = (0..samples)
        .map(|index| {
            let angle = 2.0 * PI * index as f64 / samples as f64;
            let point = [radius * angle.cos(), radius * angle.sin()];
            let scattered = reference.exterior_pressure(radius, angle) - reference_incident(point);
            let phase = Complex64::from_polar(1.0, -(EXTERIOR_WAVE_NUMBER * radius + PI / 4.0));
            let far_field = scattered * (8.0 * PI * EXTERIOR_WAVE_NUMBER * radius).sqrt() * phase;
            far_field.norm_sqr()
        })
        .sum::<f64>();
    let angular_integral = 2.0 * PI * sum / samples as f64;
    0.5 * angular_integral / (8.0 * PI * EXTERIOR_WAVE_NUMBER * EXTERIOR_DENSITY * EXTERIOR_SPEED)
}

pub fn run_production_2d_transmission_matches_cylindrical_reference() {
    let solution = problem()
        .solve(TwoDPlaneWave::plane_wave_x(), 1.0e-10)
        .unwrap();
    assert!(solution.relative_residual < 1.0e-10);

    let exterior_points = [
        [0.65 * 0.97_f64.cos(), 0.65 * 0.97_f64.sin()],
        [0.82 * 1.83_f64.cos(), 0.82 * 1.83_f64.sin()],
        [1.05 * 2.71_f64.cos(), 1.05 * 2.71_f64.sin()],
    ];
    let interior_points = [
        [0.08 * 0.37_f64.cos(), 0.08 * 0.37_f64.sin()],
        [0.23 * 1.41_f64.cos(), 0.23 * 1.41_f64.sin()],
        [0.34 * 2.62_f64.cos(), 0.34 * 2.62_f64.sin()],
    ];

    let reference = reference();
    let exterior_actual = solution
        .evaluate_exterior_total_pressure(&exterior_points)
        .unwrap();
    let exterior_expected = exterior_points
        .iter()
        .map(|point| {
            reference.exterior_pressure(point[0].hypot(point[1]), point[1].atan2(point[0]))
        })
        .collect::<Vec<_>>();
    let interior_actual = solution
        .evaluate_interior_pressure(&interior_points)
        .unwrap();
    let interior_expected = interior_points
        .iter()
        .map(|point| {
            reference.interior_pressure(point[0].hypot(point[1]), point[1].atan2(point[0]))
        })
        .collect::<Vec<_>>();

    let exterior_error = exterior_actual
        .iter()
        .zip(&exterior_expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max);
    let interior_error = interior_actual
        .iter()
        .zip(&interior_expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max);
    assert!(exterior_error < 0.08, "exterior error {exterior_error}");
    assert!(interior_error < 0.08, "interior error {interior_error}");
    assert!(
        solution.pressure_continuity_error() < 0.08,
        "pressure continuity error {}",
        solution.pressure_continuity_error()
    );
    assert!(
        solution.velocity_continuity_error() < 0.08,
        "velocity continuity error {}",
        solution.velocity_continuity_error()
    );

    let actual_power = solution.scattered_power_per_length(512).unwrap();
    let expected_power = reference_scattered_power(40.0, 512);
    let power_error = (actual_power - expected_power).abs() / expected_power.max(1.0e-12);
    assert!(power_error < 0.12, "power error {power_error}");
}

pub fn run_production_2d_transmission_matches_cylindrical_far_field() {
    let solution = problem()
        .solve(TwoDPlaneWave::plane_wave_x(), 1.0e-10)
        .expect("CPU Müller transmission solve");

    let mut maximum_relative_error: f64 = 0.0;
    for index in 0..FAR_FIELD_ANGLES {
        let angle = 2.0 * PI * index as f64 / FAR_FIELD_ANGLES as f64;
        let actual = solution
            .far_field_amplitude(angle)
            .expect("finite production far-field amplitude");
        let expected = reference_far_field_amplitude(angle);
        let relative_error =
            (actual - expected).norm() / actual.norm().max(expected.norm()).max(f64::MIN_POSITIVE);
        maximum_relative_error = maximum_relative_error.max(relative_error);
    }
    assert!(
        maximum_relative_error < 0.12,
        "2-D transmission far-field relative error: {maximum_relative_error}"
    );
}

pub fn run_production_2d_transmission_rejects_invalid_field_queries() {
    let solution = problem()
        .solve(TwoDPlaneWave::plane_wave_x(), 1.0e-10)
        .expect("CPU Müller transmission solve");

    assert!(matches!(
        solution.far_field_amplitude(f64::NAN),
        Err(TwoDBemError::NonFiniteInput {
            quantity: "far-field angle"
        })
    ));
    assert!(matches!(
        solution.evaluate_exterior_total_pressure(&[[0.1, 0.0]]),
        Err(TwoDBemError::InvalidEvaluationPoint { index: 0 })
    ));
    assert!(matches!(
        solution.evaluate_exterior_scattered_pressure(&[[f64::NAN, 0.0]]),
        Err(TwoDBemError::InvalidEvaluationPoint { index: 0 })
    ));
    assert!(matches!(
        solution.evaluate_interior_pressure(&[[0.8, 0.0]]),
        Err(TwoDBemError::InvalidEvaluationPoint { index: 0 })
    ));
}

#[test]
fn production_2d_transmission_rejects_invalid_field_queries() {
    run_production_2d_transmission_rejects_invalid_field_queries();
}

#[test]
fn production_2d_transmission_matches_cylindrical_far_field() {
    run_production_2d_transmission_matches_cylindrical_far_field();
}

#[test]
fn production_2d_transmission_matches_cylindrical_reference() {
    run_production_2d_transmission_matches_cylindrical_reference();
}

#[test]
fn production_2d_transmission_batch_reuses_factorization() {
    let problem = TwoDFluidCylinder::try_from_xem(
        LengthM::try_new(RADIUS).unwrap(),
        {
            let mut conventions = ConventionRegistry::default();
            conventions.rms_amplitudes = false;
            AcousticMedium::try_new_with_conventions(
                FrequencyHz::try_new(FREQUENCY).unwrap(),
                SpeedOfSoundMPerS::try_new(EXTERIOR_SPEED).unwrap(),
                DensityKgM3::try_new(EXTERIOR_DENSITY).unwrap(),
                conventions,
            )
            .unwrap()
        },
        {
            let mut conventions = ConventionRegistry::default();
            conventions.rms_amplitudes = false;
            AcousticMedium::try_new_with_conventions(
                FrequencyHz::try_new(FREQUENCY).unwrap(),
                SpeedOfSoundMPerS::try_new(INTERIOR_SPEED).unwrap(),
                DensityKgM3::try_new(INTERIOR_DENSITY).unwrap(),
                conventions,
            )
            .unwrap()
        },
        64,
    )
    .unwrap();
    let incidents = [
        TwoDPlaneWave::plane_wave_x(),
        TwoDPlaneWave::try_new([0.0, 1.0], Complex64::new(0.7, -0.2)).unwrap(),
    ];
    let batched = problem.solve_many(&incidents, 1.0e-10).unwrap();
    assert_eq!(batched.len(), incidents.len());
    for (incident, batch_solution) in incidents.iter().zip(&batched) {
        let independent = problem.solve(*incident, 1.0e-10).unwrap();
        assert!(batch_solution.relative_residual <= 1.0e-10);
        let exterior_error = batch_solution
            .exterior_density()
            .iter()
            .zip(independent.exterior_density())
            .map(|(batch, single)| (*batch - *single).norm())
            .fold(0.0, f64::max);
        let interior_error = batch_solution
            .interior_density()
            .iter()
            .zip(independent.interior_density())
            .map(|(batch, single)| (*batch - *single).norm())
            .fold(0.0, f64::max);
        assert!(
            exterior_error < 1.0e-11,
            "exterior batch error {exterior_error}"
        );
        assert!(
            interior_error < 1.0e-11,
            "interior batch error {interior_error}"
        );
    }
    assert!(problem.solve_many(&[], 1.0e-10).unwrap().is_empty());
    assert!(matches!(
        problem.solve_many(&incidents, f64::NAN),
        Err(sonium_bem::TwoDBemError::InvalidTolerance { .. })
    ));
}
