//! Independent Mie reference for the reduced axisymmetric penetrable path.

use num_complex::Complex64;
use sonium_bem::{AxisymmetricPenetrableSphere, PhysicsParams};
use sonium_wave::analytical::FluidSphereTransmission3d;
use sonium_xem::{AcousticMedium, ConventionRegistry, DensityKgM3, FrequencyHz, SpeedOfSoundMPerS};
use std::f64::consts::PI;

const RADIUS: f64 = 0.2;
const EXTERIOR_SPEED: f64 = 343.0;
const EXTERIOR_DENSITY: f64 = 1.21;
const INTERIOR_SPEED: f64 = 240.0;
const INTERIOR_DENSITY: f64 = 2.4;
const EXTERIOR_KA: f64 = 1.0;
const EXTERIOR_WAVE_NUMBER: f64 = EXTERIOR_KA / RADIUS;
const FREQUENCY: f64 = EXTERIOR_WAVE_NUMBER * EXTERIOR_SPEED / (2.0 * PI);
const INTERIOR_WAVE_NUMBER: f64 = 2.0 * PI * FREQUENCY / INTERIOR_SPEED;

fn far_field_angles() -> Vec<f64> {
    (0..24)
        .map(|index| PI * (index as f64 + 0.5) / 24.0)
        .collect()
}

#[derive(Debug, Clone, Copy)]
pub struct AxisymmetricPenetrableMetrics {
    pub coarse_exterior_error: f64,
    pub fine_exterior_error: f64,
    pub coarse_interior_error: f64,
    pub fine_interior_error: f64,
    pub coarse_far_field_error: f64,
    pub fine_far_field_error: f64,
    pub fine_residual: f64,
    pub forward_solution_error: f64,
    pub pressure_continuity_error: f64,
    pub velocity_continuity_error: f64,
    pub adjoint_field_norm: f64,
    pub adjoint_invalid_point_rejections: usize,
    pub adjoint_residual: f64,
    pub adjoint_batch_residual: f64,
    pub inner_product_error: f64,
    pub compressed_action_error: f64,
    pub compressed_block_count: usize,
}

fn physics() -> (PhysicsParams, PhysicsParams) {
    let mut conventions = ConventionRegistry::default();
    conventions.rms_amplitudes = false;
    let exterior_medium = AcousticMedium::try_new_with_conventions(
        FrequencyHz::try_new(FREQUENCY).unwrap(),
        SpeedOfSoundMPerS::try_new(EXTERIOR_SPEED).unwrap(),
        DensityKgM3::try_new(EXTERIOR_DENSITY).unwrap(),
        conventions.clone(),
    )
    .unwrap();
    let interior_medium = AcousticMedium::try_new_with_conventions(
        FrequencyHz::try_new(FREQUENCY).unwrap(),
        SpeedOfSoundMPerS::try_new(INTERIOR_SPEED).unwrap(),
        DensityKgM3::try_new(INTERIOR_DENSITY).unwrap(),
        conventions,
    )
    .unwrap();
    (
        PhysicsParams::try_new_medium(exterior_medium, false).unwrap(),
        PhysicsParams::try_new_medium(interior_medium, false).unwrap(),
    )
}

fn field_error(actual: &[Complex64], reference: &[Complex64]) -> f64 {
    let numerator = actual
        .iter()
        .zip(reference)
        .map(|(value, exact)| (*value - *exact).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = reference
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(1.0e-14);
    numerator / denominator
}

fn inner(lhs: &[Complex64], rhs: &[Complex64]) -> Complex64 {
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| left.conj() * right)
        .sum()
}

fn solve_level(
    meridional_points: usize,
    exterior_reference: &[Complex64],
    interior_reference: &[Complex64],
    far_field_reference: &[Complex64],
) -> (f64, f64, f64, f64) {
    let (exterior, interior) = physics();
    let problem = AxisymmetricPenetrableSphere::try_new(
        RADIUS,
        meridional_points,
        0.65,
        64,
        exterior,
        interior,
    )
    .unwrap();
    let solution = problem.solve(Complex64::new(1.0, 0.0), 1.0e-10).unwrap();
    let exterior_points: [[f64; 2]; 4] = [[0.0, 1.0], [0.24, 0.97], [0.4, 0.85], [0.7, 0.7]];
    let interior_points: [[f64; 2]; 4] = [[0.0, 0.0], [0.04, 0.08], [0.1, 0.05], [0.12, -0.04]];
    let exterior_values = solution
        .evaluate_exterior_total_pressure(&exterior_points)
        .unwrap();
    let interior_values = solution
        .evaluate_interior_pressure(&interior_points)
        .unwrap();
    let far_field_points = far_field_angles()
        .iter()
        .map(|&theta| [theta.sin(), theta.cos()])
        .collect::<Vec<_>>();
    let far_field_values = solution
        .evaluate_exterior_total_pressure(&far_field_points)
        .unwrap();
    (
        field_error(&exterior_values, exterior_reference),
        field_error(&interior_values, interior_reference),
        field_error(&far_field_values, far_field_reference),
        solution.relative_residual(),
    )
}

pub fn run_axisymmetric_penetrable_sphere_mie_convergence() -> AxisymmetricPenetrableMetrics {
    let mie = FluidSphereTransmission3d::try_new(
        RADIUS,
        EXTERIOR_WAVE_NUMBER,
        INTERIOR_WAVE_NUMBER,
        EXTERIOR_DENSITY,
        INTERIOR_DENSITY,
        32,
    )
    .unwrap();
    let exterior_points: [[f64; 2]; 4] = [[0.0, 1.0], [0.24, 0.97], [0.4, 0.85], [0.7, 0.7]];
    let interior_points: [[f64; 2]; 4] = [[0.0, 0.0], [0.04, 0.08], [0.1, 0.05], [0.12, -0.04]];
    let exterior_reference = exterior_points
        .iter()
        .map(|&[radial, axial]| {
            mie.exterior_pressure(radial.hypot(axial), radial.atan2(axial))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let interior_reference = interior_points
        .iter()
        .map(|&[radial, axial]| {
            mie.interior_pressure(radial.hypot(axial), radial.atan2(axial))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let far_field_reference = far_field_angles()
        .iter()
        .map(|&theta| mie.exterior_pressure(1.0, theta).unwrap())
        .collect::<Vec<_>>();
    let (coarse_exterior_error, coarse_interior_error, coarse_far_field_error, _) = solve_level(
        8,
        &exterior_reference,
        &interior_reference,
        &far_field_reference,
    );
    let (fine_exterior_error, fine_interior_error, fine_far_field_error, fine_residual) =
        solve_level(
            16,
            &exterior_reference,
            &interior_reference,
            &far_field_reference,
        );
    assert!(fine_exterior_error < coarse_exterior_error);
    assert!(fine_interior_error < coarse_interior_error);
    assert!(fine_residual <= 1.0e-10);

    let (exterior, interior) = physics();
    let adjoint_problem =
        AxisymmetricPenetrableSphere::try_new(RADIUS, 8, 0.65, 64, exterior, interior).unwrap();
    let (matrix, assembled_rhs) = adjoint_problem
        .assemble_system(Complex64::new(1.0, 0.0))
        .unwrap();
    let forward = adjoint_problem
        .solve(Complex64::new(1.0, 0.0), 1.0e-10)
        .unwrap();
    let pressure_continuity_error = forward.pressure_continuity_error().unwrap();
    let velocity_continuity_error = forward.velocity_continuity_error().unwrap();
    assert!(pressure_continuity_error <= 1.0e-10);
    assert!(velocity_continuity_error <= 1.0e-10);
    let mut forward_coefficients = Vec::with_capacity(matrix.nrows());
    forward_coefficients.extend_from_slice(forward.exterior_coefficients());
    forward_coefficients.extend_from_slice(forward.interior_coefficients());
    let independently_solved = sonium_solvers::direct::lu_solve(&matrix, &assembled_rhs).unwrap();
    let forward_solution_error = forward_coefficients
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
    let dimension = matrix.nrows();
    let trial = (0..dimension)
        .map(|index| Complex64::new(0.2 + index as f64 / 17.0, -(index as f64) / 29.0))
        .collect::<Vec<_>>();
    let compressed = adjoint_problem.compress(4, 1.0e-2, 16).unwrap();
    let compressed_action = sonium_solvers::traits::LinearOperator::apply(&compressed, &trial);
    let dense_action = (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| matrix[[row, column]] * trial[column])
                .sum::<Complex64>()
        })
        .collect::<Vec<_>>();
    let compressed_action_error = compressed_action
        .iter()
        .zip(&dense_action)
        .map(|(actual, reference)| (*actual - *reference).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / dense_action
            .iter()
            .map(Complex64::norm_sqr)
            .sum::<f64>()
            .sqrt()
            .max(1.0e-15);
    let compressed_block_count = compressed.compressed_block_count();
    assert!(compressed_block_count > 0);
    assert!(compressed_action_error <= 2.0e-2);
    let adjoint_rhs = (0..dimension)
        .map(|index| Complex64::new(0.4 + index as f64 / 13.0, index as f64 / 31.0))
        .collect::<Vec<_>>();
    let adjoint = adjoint_problem
        .solve_adjoint(&adjoint_rhs, 1.0e-10)
        .unwrap();
    let adjoint_field = adjoint
        .evaluate_exterior_adjoint_pressure(&[[0.0, 0.8], [0.2, 0.7]])
        .unwrap();
    let adjoint_field_norm = adjoint_field.iter().map(|value| value.norm()).sum::<f64>();
    assert!(adjoint_field_norm.is_finite());
    let adjoint_invalid_point_rejections = usize::from(
        adjoint
            .evaluate_exterior_adjoint_pressure(&[[f64::NAN, 0.8]])
            .is_err(),
    );
    assert_eq!(adjoint_invalid_point_rejections, 1);
    let second_adjoint_rhs = adjoint_rhs
        .iter()
        .enumerate()
        .map(|(index, value)| *value + Complex64::new(index as f64 / 37.0, 0.07))
        .collect::<Vec<_>>();
    let adjoint_batch = adjoint_problem
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
    let inner_product_error =
        (inner(&trial_action, adjoint.coefficients()) - inner(&trial, &adjoint_rhs)).norm()
            / inner(&trial, &adjoint_rhs).norm().max(1.0e-30);
    assert!(inner_product_error <= 2.0e-12);
    assert!(
        adjoint_problem
            .solve_adjoint(&adjoint_rhs[..dimension - 1], 1.0e-10)
            .is_err()
    );
    AxisymmetricPenetrableMetrics {
        coarse_exterior_error,
        fine_exterior_error,
        coarse_interior_error,
        fine_interior_error,
        coarse_far_field_error,
        fine_far_field_error,
        fine_residual,
        forward_solution_error,
        pressure_continuity_error,
        velocity_continuity_error,
        adjoint_field_norm,
        adjoint_invalid_point_rejections,
        adjoint_residual: adjoint.relative_residual(),
        adjoint_batch_residual,
        inner_product_error,
        compressed_action_error,
        compressed_block_count,
    }
}

#[test]
fn axisymmetric_penetrable_sphere_converges_to_mie() {
    let metrics = run_axisymmetric_penetrable_sphere_mie_convergence();
    assert!(metrics.fine_exterior_error < 0.2);
    assert!(metrics.fine_interior_error < 0.2);
    assert!(metrics.fine_far_field_error < metrics.coarse_far_field_error);
    assert!(metrics.fine_far_field_error / metrics.coarse_far_field_error <= 0.01);
    assert!(metrics.fine_far_field_error < 0.2);
    assert!(metrics.pressure_continuity_error <= 1.0e-10);
    assert!(metrics.velocity_continuity_error <= 1.0e-10);
    assert!(metrics.adjoint_field_norm.is_finite());
    assert_eq!(metrics.adjoint_invalid_point_rejections, 1);
}

#[test]
fn axisymmetric_penetrable_sphere_adjoint_many_matches_independent_solves() {
    let (exterior, interior) = physics();
    let problem =
        AxisymmetricPenetrableSphere::try_new(RADIUS, 8, 0.65, 64, exterior, interior).unwrap();
    let dimension = problem
        .assemble_system(Complex64::new(1.0, 0.0))
        .unwrap()
        .0
        .nrows();
    let rhs_columns = (0..2)
        .map(|column| {
            (0..dimension)
                .map(|index| {
                    Complex64::new(
                        0.15 + (index + column) as f64 / 19.0,
                        (index + 3 * column) as f64 / 29.0,
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

#[test]
fn axisymmetric_penetrable_sphere_compressed_action_matches_dense_operator() {
    let (exterior, interior) = physics();
    let problem =
        AxisymmetricPenetrableSphere::try_new(RADIUS, 8, 0.65, 64, exterior, interior).unwrap();
    let (dense, _) = problem.assemble_system(Complex64::new(1.0, 0.0)).unwrap();
    let compressed = problem.compress(4, 1.0e-2, 16).unwrap();
    assert_eq!(compressed.interface_dofs(), 8);
    assert_eq!(compressed.total_dofs(), 16);
    assert!(compressed.compressed_block_count() > 0);

    let trial = (0..16)
        .map(|index| Complex64::new(0.13 + index as f64 / 19.0, -0.07 + index as f64 / 31.0))
        .collect::<Vec<_>>();
    let dense_action = (0..16)
        .map(|row| {
            (0..16)
                .map(|column| dense[[row, column]] * trial[column])
                .sum::<Complex64>()
        })
        .collect::<Vec<_>>();
    let compressed_action = sonium_solvers::traits::LinearOperator::apply(&compressed, &trial);
    let action_error = compressed_action
        .iter()
        .zip(&dense_action)
        .map(|(actual, reference)| (*actual - *reference).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let action_scale = dense_action
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(1.0e-15);
    assert!(action_error / action_scale <= 2.0e-2);
    assert!(
        compressed_action
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(matches!(
        problem.compress(0, 1.0e-2, 16),
        Err(sonium_bem::core::assembly::TransmissionCompressionError::InvalidBlockSize)
    ));
}
