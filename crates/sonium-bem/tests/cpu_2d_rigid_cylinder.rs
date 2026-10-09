//! Production CPU 2-D line-BEM field and power contract.

use num_complex::Complex64;
use sonium_bem::{TwoDBemError, TwoDPlaneWave, TwoDRigidCylinder};
use sonium_wave::analytical::cylinder_scattering_2d;
use sonium_xem::{AcousticMedium, ConventionRegistry, LengthM};
use std::f64::consts::PI;

pub fn run_production_2d_line_bem_matches_cylinder_far_field() {
    let solution = TwoDRigidCylinder::try_from_xem(
        LengthM::try_new(RADIUS).expect("finite radius"),
        medium(false),
        128,
    )
    .expect("typed 2-D cylinder construction")
    .solve(TwoDPlaneWave::plane_wave_x(), 1.0e-11)
    .expect("2-D line-BEM solve");

    let mut maximum_relative_error: f64 = 0.0;
    for index in 0..FAR_FIELD_ANGLES {
        let angle = 2.0 * PI * index as f64 / FAR_FIELD_ANGLES as f64;
        let actual = solution
            .far_field_amplitude(angle)
            .expect("finite production far-field amplitude");
        let expected = analytical_far_field_amplitude(angle);
        let relative_error =
            (actual - expected).norm() / actual.norm().max(expected.norm()).max(f64::MIN_POSITIVE);
        maximum_relative_error = maximum_relative_error.max(relative_error);
    }
    assert!(
        maximum_relative_error < 0.08,
        "2-D far-field relative error: {maximum_relative_error}"
    );
}

pub fn run_production_2d_rigid_rejects_invalid_field_queries() {
    let solution = TwoDRigidCylinder::try_from_xem(
        LengthM::try_new(RADIUS).expect("finite radius"),
        medium(false),
        96,
    )
    .expect("typed 2-D cylinder construction")
    .solve(TwoDPlaneWave::plane_wave_x(), 1.0e-11)
    .expect("2-D line-BEM solve");

    assert!(matches!(
        solution.far_field_amplitude(f64::NAN),
        Err(TwoDBemError::NonFiniteInput {
            quantity: "far-field angle"
        })
    ));
    assert!(matches!(
        solution.evaluate_total_pressure(&[[0.1, 0.0]]),
        Err(TwoDBemError::InvalidEvaluationPoint { index: 0 })
    ));
    assert!(matches!(
        solution.evaluate_scattered_pressure(&[[f64::NAN, 0.0]]),
        Err(TwoDBemError::InvalidEvaluationPoint { index: 0 })
    ));
}

#[test]
fn production_2d_line_bem_matches_cylinder_far_field() {
    run_production_2d_line_bem_matches_cylinder_far_field();
}

#[test]
fn production_2d_rigid_rejects_invalid_field_queries() {
    run_production_2d_rigid_rejects_invalid_field_queries();
}

const RADIUS: f64 = 0.4;
const WAVE_NUMBER: f64 = 2.0;
const SPEED_OF_SOUND: f64 = 343.0;
const DENSITY: f64 = 1.21;
const FREQUENCY: f64 = WAVE_NUMBER * SPEED_OF_SOUND / (2.0 * PI);
const FAR_FIELD_REFERENCE_RADIUS: f64 = 40.0;
const FAR_FIELD_ANGLES: usize = 24;

fn medium(rms_amplitudes: bool) -> AcousticMedium {
    let mut conventions = ConventionRegistry::default();
    conventions.rms_amplitudes = rms_amplitudes;
    AcousticMedium::try_new_with_conventions(
        sonium_xem::FrequencyHz::try_new(FREQUENCY).unwrap(),
        sonium_xem::SpeedOfSoundMPerS::try_new(SPEED_OF_SOUND).unwrap(),
        sonium_xem::DensityKgM3::try_new(DENSITY).unwrap(),
        conventions,
    )
    .unwrap()
}

fn analytical_pressure(radius: f64, angle: f64) -> Complex64 {
    cylinder_scattering_2d(WAVE_NUMBER, RADIUS, 28, vec![radius], vec![angle]).pressure[0]
}

fn analytical_far_field_amplitude(angle: f64) -> Complex64 {
    // Recover the documented outgoing amplitude from the independent
    // cylindrical-series pressure; production panel quadrature is not reused.
    let radius = FAR_FIELD_REFERENCE_RADIUS;
    let incident = Complex64::from_polar(1.0, WAVE_NUMBER * radius * angle.cos());
    let scattered = analytical_pressure(radius, angle) - incident;
    let outgoing_phase = Complex64::from_polar(1.0, -(WAVE_NUMBER * radius + PI / 4.0));
    scattered * (8.0 * PI * WAVE_NUMBER * radius).sqrt() * outgoing_phase
}

fn analytical_rms_power_per_length(radius: f64, samples: usize) -> f64 {
    let mut sum = 0.0;
    for index in 0..samples {
        let angle = 2.0 * PI * index as f64 / samples as f64;
        let point = [radius * angle.cos(), radius * angle.sin()];
        let incident = Complex64::from_polar(1.0, WAVE_NUMBER * point[0]);
        let scattered = analytical_pressure(radius, angle) - incident;
        sum += scattered.norm_sqr();
    }
    radius * 2.0 * PI * sum / samples as f64 / (DENSITY * SPEED_OF_SOUND)
}

pub fn run_production_2d_line_bem_matches_cylinder_field_and_power() {
    let problem =
        TwoDRigidCylinder::try_from_xem(LengthM::try_new(RADIUS).unwrap(), medium(true), 128)
            .expect("typed 2-D cylinder construction");
    let solution = problem
        .solve(TwoDPlaneWave::plane_wave_x(), 1.0e-11)
        .expect("2-D line-BEM solve");

    assert!(solution.relative_residual < 1.0e-11);
    let radius = 1.4;
    let angles: [f64; 8] = [0.0, 0.37, 0.91, 1.63, 2.41, 3.08, 4.27, 5.62];
    let points: Vec<[f64; 2]> = angles
        .iter()
        .map(|&angle| [radius * angle.cos(), radius * angle.sin()])
        .collect();
    let actual = solution
        .evaluate_total_pressure(&points)
        .expect("exterior total field");
    let expected: Vec<Complex64> = angles
        .iter()
        .map(|&angle| analytical_pressure(radius, angle))
        .collect();
    let error = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let reference_norm = expected
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!(
        error / reference_norm < 0.08,
        "2-D field relative error: {}",
        error / reference_norm
    );

    let actual_power = solution
        .scattered_power_per_length(512)
        .expect("2-D scattered power");
    let reference_power = analytical_rms_power_per_length(40.0, 512);
    assert!(reference_power > 0.0);
    assert!(
        (actual_power - reference_power).abs() / reference_power < 0.08,
        "2-D power per length: actual={actual_power:e}, reference={reference_power:e}"
    );
}

#[test]
fn production_2d_line_bem_matches_cylinder_field_and_power() {
    run_production_2d_line_bem_matches_cylinder_field_and_power();
}

pub fn run_production_2d_power_observes_xem_peak_rms_convention() {
    let rms = TwoDRigidCylinder::try_from_xem(LengthM::try_new(RADIUS).unwrap(), medium(true), 96)
        .unwrap()
        .solve(TwoDPlaneWave::plane_wave_x(), 1.0e-11)
        .unwrap()
        .scattered_power_per_length(256)
        .unwrap();
    let peak =
        TwoDRigidCylinder::try_from_xem(LengthM::try_new(RADIUS).unwrap(), medium(false), 96)
            .unwrap()
            .solve(TwoDPlaneWave::plane_wave_x(), 1.0e-11)
            .unwrap()
            .scattered_power_per_length(256)
            .unwrap();

    assert!(rms.is_finite() && rms > 0.0);
    assert!((peak * 2.0 - rms).abs() / rms < 1.0e-12);
}

#[test]
fn production_2d_power_observes_xem_peak_rms_convention() {
    run_production_2d_power_observes_xem_peak_rms_convention();
}

#[test]
fn production_2d_rigid_transmission_batch_reuses_factorization() {
    let problem =
        TwoDRigidCylinder::try_from_xem(LengthM::try_new(RADIUS).unwrap(), medium(false), 64)
            .unwrap();
    let incidents = [
        TwoDPlaneWave::plane_wave_x(),
        TwoDPlaneWave::try_new([0.0, 1.0], Complex64::new(0.6, -0.1)).unwrap(),
    ];
    let batched = problem.solve_many(&incidents, 1.0e-10).unwrap();
    assert_eq!(batched.len(), incidents.len());
    for (incident, batch_solution) in incidents.iter().zip(&batched) {
        let independent = problem.solve(*incident, 1.0e-10).unwrap();
        assert!(batch_solution.relative_residual <= 1.0e-10);
        let error = batch_solution
            .density()
            .iter()
            .zip(independent.density())
            .map(|(batch, single)| (*batch - *single).norm())
            .fold(0.0, f64::max);
        assert!(error < 1.0e-11, "batch density error {error}");
    }
    assert!(problem.solve_many(&[], 1.0e-10).unwrap().is_empty());
}
