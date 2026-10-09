//! Independent production 2-D transmission matrix and adjoint contract.

use num_complex::Complex64;
use sonium_bem::{TwoDFluidCylinder, TwoDPlaneWave};
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
const PANEL_COUNT: usize = 32;
const FREQUENCY: f64 = EXTERIOR_WAVE_NUMBER * EXTERIOR_SPEED / (2.0 * PI);
const INTERIOR_SPEED: f64 = 2.0 * PI * FREQUENCY / INTERIOR_WAVE_NUMBER;

#[derive(Debug, Clone, Copy)]
pub struct TwoDTransmissionAdjointMetrics {
    pub matrix_dimension: usize,
    pub forward_residual: f64,
    pub forward_solution_error: f64,
    pub adjoint_field_norm: f64,
    pub adjoint_invalid_point_rejections: usize,
    pub adjoint_residual: f64,
    pub adjoint_batch_residual: f64,
    pub inner_product_error: f64,
}

fn problem() -> TwoDFluidCylinder {
    let mut conventions = ConventionRegistry::default();
    conventions.rms_amplitudes = false;
    let exterior = AcousticMedium::try_new_with_conventions(
        FrequencyHz::try_new(FREQUENCY).unwrap(),
        SpeedOfSoundMPerS::try_new(EXTERIOR_SPEED).unwrap(),
        DensityKgM3::try_new(EXTERIOR_DENSITY).unwrap(),
        conventions.clone(),
    )
    .unwrap();
    let interior = AcousticMedium::try_new_with_conventions(
        FrequencyHz::try_new(FREQUENCY).unwrap(),
        SpeedOfSoundMPerS::try_new(INTERIOR_SPEED).unwrap(),
        DensityKgM3::try_new(INTERIOR_DENSITY).unwrap(),
        conventions,
    )
    .unwrap();
    TwoDFluidCylinder::try_from_xem(
        LengthM::try_new(RADIUS).unwrap(),
        exterior,
        interior,
        PANEL_COUNT,
    )
    .unwrap()
}

fn inner(lhs: &[Complex64], rhs: &[Complex64]) -> Complex64 {
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| left.conj() * right)
        .sum()
}

fn relative_error(lhs: Complex64, rhs: Complex64) -> f64 {
    (lhs - rhs).norm() / lhs.norm().max(rhs.norm()).max(1.0e-30)
}

pub fn run_two_d_transmission_adjoint_contract() -> TwoDTransmissionAdjointMetrics {
    let problem = problem();
    let incident = TwoDPlaneWave::plane_wave_x();
    let (matrix, assembled_rhs) = problem.assemble_transmission_system(incident).unwrap();
    let dimension = assembled_rhs.len();
    assert_eq!(dimension, 2 * PANEL_COUNT);

    let forward = problem.solve(incident, 1.0e-10).unwrap();
    assert!(forward.relative_residual <= 1.0e-10);
    let mut forward_density = Vec::with_capacity(dimension);
    forward_density.extend_from_slice(forward.exterior_density());
    forward_density.extend_from_slice(forward.interior_density());

    let independently_solved = sonium_solvers::direct::lu_solve(&matrix, &assembled_rhs).unwrap();
    let forward_solution_error = forward_density
        .iter()
        .zip(&independently_solved)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / independently_solved
            .iter()
            .map(Complex64::norm_sqr)
            .sum::<f64>()
            .sqrt()
            .max(1.0e-30);
    assert!(forward_solution_error <= 1.0e-10);
    let mut action = vec![Complex64::default(); dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            action[row] += matrix[[row, column]] * forward_density[column];
        }
    }
    let forward_residual = action
        .iter()
        .zip(&assembled_rhs)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / assembled_rhs
            .iter()
            .map(Complex64::norm_sqr)
            .sum::<f64>()
            .sqrt()
            .max(1.0e-30);
    assert!(forward_residual <= 1.0e-10);

    let trial = (0..dimension)
        .map(|index| {
            Complex64::new(
                (index + 1) as f64 / dimension as f64,
                -((index % 5) as f64) / (dimension as f64 + 1.0),
            )
        })
        .collect::<Vec<_>>();
    let adjoint_rhs = (0..dimension)
        .map(|index| Complex64::new(0.3 + (index % 7) as f64 / 11.0, (index % 3) as f64 / 13.0))
        .collect::<Vec<_>>();
    let adjoint = problem.solve_adjoint(&adjoint_rhs, 1.0e-10).unwrap();
    assert!(adjoint.relative_residual() <= 1.0e-10);
    let adjoint_field = adjoint
        .evaluate_exterior_adjoint_pressure(&[[0.0, 0.7], [0.25, 0.55]])
        .unwrap();
    let adjoint_field_norm = adjoint_field.iter().map(|value| value.norm()).sum::<f64>();
    assert!(adjoint_field_norm.is_finite());
    let adjoint_invalid_point_rejections = usize::from(
        adjoint
            .evaluate_exterior_adjoint_pressure(&[[f64::NAN, 0.7]])
            .is_err(),
    );
    assert_eq!(adjoint_invalid_point_rejections, 1);
    let second_adjoint_rhs = adjoint_rhs
        .iter()
        .enumerate()
        .map(|(index, value)| *value + Complex64::new(index as f64 / 41.0, 0.1))
        .collect::<Vec<_>>();
    let adjoint_batch = problem
        .solve_adjoint_many(&[adjoint_rhs.clone(), second_adjoint_rhs], 1.0e-10)
        .unwrap();
    let adjoint_batch_residual = adjoint_batch
        .iter()
        .map(|solution| solution.relative_residual())
        .fold(0.0, f64::max);
    assert!(adjoint_batch_residual <= 1.0e-10);

    let mut trial_action = vec![Complex64::default(); dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            trial_action[row] += matrix[[row, column]] * trial[column];
        }
    }
    let inner_product_error = relative_error(
        inner(&trial_action, &adjoint.coefficients()),
        inner(&trial, &adjoint_rhs),
    );
    assert!(
        inner_product_error <= 2.0e-12,
        "Hermitian-adjoint identity error {inner_product_error:e}"
    );

    assert!(
        problem
            .solve_adjoint(&adjoint_rhs[..dimension - 1], 1.0e-10)
            .is_err()
    );
    assert!(
        problem
            .solve_adjoint(&vec![Complex64::new(f64::NAN, 0.0); dimension], 1.0e-10,)
            .is_err()
    );

    TwoDTransmissionAdjointMetrics {
        matrix_dimension: dimension,
        forward_residual,
        forward_solution_error,
        adjoint_field_norm,
        adjoint_invalid_point_rejections,
        adjoint_residual: adjoint.relative_residual(),
        adjoint_batch_residual,
        inner_product_error,
    }
}

#[test]
fn two_d_transmission_matrix_and_adjoint_contract() {
    let metrics = run_two_d_transmission_adjoint_contract();
    assert_eq!(metrics.matrix_dimension, 2 * PANEL_COUNT);
    assert!(metrics.forward_residual <= 1.0e-10);
    assert!(metrics.forward_solution_error <= 1.0e-10);
    assert!(metrics.adjoint_field_norm.is_finite());
    assert_eq!(metrics.adjoint_invalid_point_rejections, 1);
    assert!(metrics.adjoint_residual <= 1.0e-10);
    assert!(metrics.inner_product_error <= 2.0e-12);
}

#[test]
fn two_d_transmission_adjoint_many_matches_independent_solves() {
    let problem = problem();
    let dimension = 2 * PANEL_COUNT;
    let rhs_columns = (0..2)
        .map(|column| {
            (0..dimension)
                .map(|index| {
                    Complex64::new(
                        0.2 + (index + column) as f64 / 17.0,
                        -(((index + 2 * column) % 5) as f64) / 23.0,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let batch = problem.solve_adjoint_many(&rhs_columns, 1.0e-10).unwrap();
    assert_eq!(batch.len(), rhs_columns.len());
    for (batched, rhs) in batch.iter().zip(&rhs_columns) {
        let independent = problem.solve_adjoint(rhs, 1.0e-10).unwrap();
        assert!(batched.relative_residual() <= 1.0e-10);
        let coefficient_error = batched
            .coefficients()
            .iter()
            .zip(independent.coefficients())
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0, f64::max);
        assert!(coefficient_error <= 1.0e-12);
    }

    assert!(problem.solve_adjoint_many(&[], 1.0e-10).unwrap().is_empty());
    assert!(
        problem
            .solve_adjoint_many(&[vec![Complex64::default(); dimension - 1]], 1.0e-10)
            .is_err()
    );
    assert!(
        problem
            .solve_adjoint_many(&[vec![Complex64::new(f64::NAN, 0.0); dimension]], 1.0e-10,)
            .is_err()
    );
}
