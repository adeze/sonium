//! Independent Mie convergence for the reduced axisymmetric ring-source path.

use num_complex::Complex64;
use sonium_bem::core::{AxisymmetricRigidSphere, PhysicsParams};
use sonium_wave::analytical::sphere_scattering_3d;
use std::f64::consts::PI;

const RADIUS_M: f64 = 0.1;
const KA: f64 = 0.35;
const SPEED_OF_SOUND_M_PER_S: f64 = 343.0;
const DENSITY_KG_PER_M3: f64 = 1.21;
const EVALUATION_RADIUS_M: f64 = 1.2;

pub struct AxisymmetricRingMieMetrics {
    pub coarse_l2_error: f64,
    pub fine_l2_error: f64,
    pub coarse_pointwise_error: f64,
    pub fine_pointwise_error: f64,
    pub coarse_far_field_error: f64,
    pub fine_far_field_error: f64,
    pub coarse_residual: f64,
    pub fine_residual: f64,
    pub coarse_dofs: usize,
    pub fine_dofs: usize,
}

fn polar_angles() -> Vec<f64> {
    vec![0.2, 0.55, 0.9, 1.25, 1.6, 1.95, 2.3, 2.65, 2.95]
}

fn far_field_angles() -> Vec<f64> {
    (0..24)
        .map(|index| PI * (index as f64 + 0.5) / 24.0)
        .collect()
}

fn points_for_angles(angles: &[f64]) -> Vec<[f64; 2]> {
    angles
        .iter()
        .map(|&theta| {
            [
                EVALUATION_RADIUS_M * theta.sin(),
                EVALUATION_RADIUS_M * theta.cos(),
            ]
        })
        .collect()
}

fn solve(dofs: usize) -> (Vec<Complex64>, Vec<Complex64>, f64) {
    let frequency_hz = KA * SPEED_OF_SOUND_M_PER_S / (2.0 * PI * RADIUS_M);
    let physics = PhysicsParams::try_new(
        frequency_hz,
        SPEED_OF_SOUND_M_PER_S,
        DENSITY_KG_PER_M3,
        false,
    )
    .expect("checked physics");
    let problem = AxisymmetricRigidSphere::try_new(RADIUS_M, dofs, 0.65, 64, physics)
        .expect("checked reduced sphere");
    assert_eq!(problem.num_dofs(), dofs);
    let solution = problem
        .solve(Complex64::new(1.0, 0.0), 1.0e-10)
        .expect("reduced axisymmetric solve");
    assert_eq!(solution.num_dofs(), dofs);
    let points: Vec<_> = polar_angles()
        .into_iter()
        .map(|theta| {
            [
                EVALUATION_RADIUS_M * theta.sin(),
                EVALUATION_RADIUS_M * theta.cos(),
            ]
        })
        .collect();
    let far_field_points = points_for_angles(&far_field_angles());
    (
        solution
            .evaluate_total_pressure(&points)
            .expect("checked exterior points"),
        solution
            .evaluate_total_pressure(&far_field_points)
            .expect("checked far-field points"),
        solution.relative_residual(),
    )
}

fn errors(actual: &[Complex64], reference: &[Complex64]) -> (f64, f64) {
    let numerator = actual
        .iter()
        .zip(reference)
        .map(|(&actual, &reference)| (actual - reference).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = reference
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let pointwise = actual
        .iter()
        .zip(reference)
        .map(|(&actual, &reference)| {
            (actual - reference).norm() / actual.norm().max(reference.norm()).max(f64::MIN_POSITIVE)
        })
        .fold(0.0, f64::max);
    (numerator / denominator, pointwise)
}

pub fn run_axisymmetric_ring_mie_convergence() -> AxisymmetricRingMieMetrics {
    let reference = sphere_scattering_3d(
        KA / RADIUS_M,
        RADIUS_M,
        32,
        vec![EVALUATION_RADIUS_M],
        polar_angles(),
    )
    .pressure;
    let (coarse, coarse_far_field, coarse_residual) = solve(12);
    let (fine, fine_far_field, fine_residual) = solve(20);
    let (coarse_l2_error, coarse_pointwise_error) = errors(&coarse, &reference);
    let (fine_l2_error, fine_pointwise_error) = errors(&fine, &reference);
    let far_field_reference = sphere_scattering_3d(
        KA / RADIUS_M,
        RADIUS_M,
        32,
        vec![EVALUATION_RADIUS_M],
        far_field_angles(),
    )
    .pressure;
    let (_, coarse_far_field_error) = errors(&coarse_far_field, &far_field_reference);
    let (_, fine_far_field_error) = errors(&fine_far_field, &far_field_reference);

    AxisymmetricRingMieMetrics {
        coarse_l2_error,
        fine_l2_error,
        coarse_pointwise_error,
        fine_pointwise_error,
        coarse_far_field_error,
        fine_far_field_error,
        coarse_residual,
        fine_residual,
        coarse_dofs: 12,
        fine_dofs: 20,
    }
}

#[test]
fn reduced_axisymmetric_ring_path_converges_to_mie() {
    let metrics = run_axisymmetric_ring_mie_convergence();
    println!(
        "BEM_EVIDENCE: {{\"case_id\":\"bem.axisymmetric-ring-mie-convergence.v1\",\"status\":\"passed\",\"coarse_l2_error\":{:.6e},\"fine_l2_error\":{:.6e},\"coarse_pointwise_error\":{:.6e},\"fine_pointwise_error\":{:.6e},\"coarse_far_field_error\":{:.6e},\"fine_far_field_error\":{:.6e},\"coarse_residual\":{:.6e},\"fine_residual\":{:.6e},\"coarse_dofs\":{},\"fine_dofs\":{}}}",
        metrics.coarse_l2_error,
        metrics.fine_l2_error,
        metrics.coarse_pointwise_error,
        metrics.fine_pointwise_error,
        metrics.coarse_far_field_error,
        metrics.fine_far_field_error,
        metrics.coarse_residual,
        metrics.fine_residual,
        metrics.coarse_dofs,
        metrics.fine_dofs,
    );
    assert!(metrics.fine_l2_error / metrics.coarse_l2_error <= 0.01);
    assert!(metrics.fine_pointwise_error / metrics.coarse_pointwise_error <= 0.01);
    assert!(metrics.fine_far_field_error / metrics.coarse_far_field_error <= 0.01);
    assert!(metrics.fine_l2_error <= 1.0e-7);
    assert!(metrics.fine_pointwise_error <= 2.0e-7);
    assert!(metrics.fine_far_field_error <= 2.0e-7);
    assert!(metrics.coarse_residual <= 1.0e-10);
    assert!(metrics.fine_residual <= 1.0e-10);
}
