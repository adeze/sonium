//! Star-shaped Fourier-deformed 2-D rigid cylinder for the Stage-2 pilot.

use num_complex::Complex64;
use sonium_bem::core::types::PhysicsParams;
use sonium_bem::{TwoDBemError, TwoDFourierMode, TwoDPlaneWave, TwoDRigidCylinder};
use std::f64::consts::PI;

const RADIUS: f64 = 0.4;
const WAVE_NUMBER: f64 = 2.0;
const SPEED_OF_SOUND: f64 = 343.0;
const DENSITY: f64 = 1.21;
const FREQUENCY: f64 = WAVE_NUMBER * SPEED_OF_SOUND / (2.0 * PI);

fn physics() -> PhysicsParams {
    PhysicsParams::try_new(FREQUENCY, SPEED_OF_SOUND, DENSITY, false).expect("valid physics")
}

fn mode(order: usize, cosine: f64, sine: f64) -> TwoDFourierMode {
    TwoDFourierMode {
        order,
        cosine,
        sine,
    }
}

fn max_density_difference(left: &[Complex64], right: &[Complex64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max)
}

fn density_norm(density: &[Complex64]) -> f64 {
    density
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE)
}

pub fn run_deformed_cylinder_with_zero_modes_matches_circle() {
    let circle = TwoDRigidCylinder::try_new(RADIUS, 256, physics()).expect("circle problem");
    let deformed = TwoDRigidCylinder::try_new_deformed(
        RADIUS,
        256,
        physics(),
        &[mode(2, 0.0, 0.0), mode(3, 0.0, 0.0)],
    )
    .expect("deformed problem");
    assert_eq!(deformed.panels().len(), circle.panels().len());
    for (deformed_panel, circle_panel) in deformed.panels().iter().zip(circle.panels()) {
        for component in 0..2 {
            assert!(
                (deformed_panel.start[component] - circle_panel.start[component]).abs() < 1.0e-12,
                "endpoint drift without deformation"
            );
            assert!(
                (deformed_panel.normal[component] - circle_panel.normal[component]).abs() < 1.0e-9,
                "normal drift without deformation"
            );
        }
        assert!(
            (deformed_panel.length - circle_panel.length).abs() < 1.0e-12,
            "length drift without deformation"
        );
    }
    let circle_density = circle
        .solve(TwoDPlaneWave::plane_wave_x(), 1.0e-11)
        .expect("circle solve")
        .density()
        .to_vec();
    let deformed_density = deformed
        .solve(TwoDPlaneWave::plane_wave_x(), 1.0e-11)
        .expect("deformed solve")
        .density()
        .to_vec();
    let difference = max_density_difference(&circle_density, &deformed_density);
    assert!(
        difference / density_norm(&circle_density) < 1.0e-9,
        "zero-mode density mismatch: {difference:.3e}"
    );
}

pub fn run_deformed_cylinder_rejects_invalid_shapes() {
    assert!(matches!(
        TwoDRigidCylinder::try_new_deformed(RADIUS, 64, physics(), &[mode(0, 0.1, 0.0)]),
        Err(TwoDBemError::InvalidDeformation { .. })
    ));
    assert!(matches!(
        TwoDRigidCylinder::try_new_deformed(
            RADIUS,
            64,
            physics(),
            &[mode(2, 0.2, 0.0), mode(3, 0.0, 0.15)]
        ),
        Err(TwoDBemError::InvalidDeformation { .. })
    ));
    assert!(matches!(
        TwoDRigidCylinder::try_new_deformed(RADIUS, 64, physics(), &[mode(2, f64::NAN, 0.0)]),
        Err(TwoDBemError::InvalidDeformation { .. })
    ));
    assert!(matches!(
        TwoDRigidCylinder::try_new_deformed(RADIUS, 4, physics(), &[mode(2, 0.1, 0.0)]),
        Err(TwoDBemError::InvalidPanelCount { value: 4 })
    ));
    assert!(matches!(
        TwoDRigidCylinder::try_new_deformed(-RADIUS, 64, physics(), &[mode(2, 0.1, 0.0)]),
        Err(TwoDBemError::InvalidRadius { .. })
    ));
    let problem = TwoDRigidCylinder::try_new_deformed(
        RADIUS,
        64,
        physics(),
        &[mode(2, 0.2, 0.0), mode(3, 0.0, 0.05)],
    )
    .expect("bounding-radius problem");
    assert!((problem.bounding_radius() - RADIUS * 1.25).abs() < 1.0e-9);
    assert_eq!(problem.deformation_modes().len(), 2);
}

pub fn run_deformed_cylinder_normals_point_outward() {
    let problem = TwoDRigidCylinder::try_new_deformed(
        RADIUS,
        128,
        physics(),
        &[mode(2, 0.12, 0.05), mode(3, -0.06, 0.02)],
    )
    .expect("deformed problem");
    for panel in problem.panels() {
        assert!(panel.length > 0.0, "degenerate panel");
        assert!(
            (panel.normal[0].hypot(panel.normal[1]) - 1.0).abs() < 1.0e-12,
            "non-unit normal"
        );
        let outward = panel.normal[0] * panel.center[0] + panel.normal[1] * panel.center[1];
        assert!(outward > 0.0, "inward normal at {:?}", panel.center);
    }
}

pub fn run_deformed_cylinder_small_deformation_is_continuous() {
    let circle = TwoDRigidCylinder::try_new(RADIUS, 128, physics()).expect("circle problem");
    let deformed =
        TwoDRigidCylinder::try_new_deformed(RADIUS, 128, physics(), &[mode(2, 1.0e-4, 0.0)])
            .expect("deformed problem");
    let circle_density = circle
        .solve(TwoDPlaneWave::plane_wave_x(), 1.0e-11)
        .expect("circle solve")
        .density()
        .to_vec();
    let deformed_density = deformed
        .solve(TwoDPlaneWave::plane_wave_x(), 1.0e-11)
        .expect("deformed solve")
        .density()
        .to_vec();
    let relative =
        max_density_difference(&circle_density, &deformed_density) / density_norm(&circle_density);
    assert!(
        relative < 1.0e-2,
        "discontinuous small deformation: {relative:.3e}"
    );
}

pub fn run_deformed_cylinder_exterior_check_respects_bulge() {
    let deformed = TwoDRigidCylinder::try_new_deformed(RADIUS, 96, physics(), &[mode(2, 0.2, 0.0)])
        .expect("deformed problem")
        .solve(TwoDPlaneWave::plane_wave_x(), 1.0e-11)
        .expect("deformed solve");
    // Inside the 1.2R bounding radius but outside the base circle: rejected.
    assert!(matches!(
        deformed.evaluate_total_pressure(&[[RADIUS * 1.1, 0.0]]),
        Err(TwoDBemError::InvalidEvaluationPoint { index: 0 })
    ));
    // Well outside the bulge: accepted and finite.
    let far = deformed
        .evaluate_total_pressure(&[[RADIUS * 3.0, 0.0]])
        .expect("far evaluation");
    assert!(far[0].is_finite());
}

pub fn run_deformed_cylinder_refines_with_panels() {
    let modes = [mode(2, 0.1, 0.04), mode(3, -0.05, 0.06)];
    let mut amplitudes = Vec::new();
    for panels in [64_usize, 128] {
        let solution = TwoDRigidCylinder::try_new_deformed(RADIUS, panels, physics(), &modes)
            .expect("deformed problem")
            .solve(TwoDPlaneWave::plane_wave_x(), 1.0e-11)
            .expect("deformed solve");
        let mut row = Vec::new();
        for index in 0..12 {
            let angle = 2.0 * PI * index as f64 / 12.0;
            row.push(
                solution
                    .far_field_amplitude(angle)
                    .expect("far-field amplitude"),
            );
        }
        amplitudes.push(row);
    }
    let mut maximum = 0.0_f64;
    for (coarse, fine) in amplitudes[0].iter().zip(&amplitudes[1]) {
        let denominator = coarse.norm().max(fine.norm()).max(f64::MIN_POSITIVE);
        maximum = maximum.max((coarse - fine).norm() / denominator);
    }
    assert!(
        maximum < 0.05,
        "panel refinement change too large: {maximum:.3e}"
    );
}

#[test]
fn deformed_cylinder_with_zero_modes_matches_circle() {
    run_deformed_cylinder_with_zero_modes_matches_circle();
}

#[test]
fn deformed_cylinder_rejects_invalid_shapes() {
    run_deformed_cylinder_rejects_invalid_shapes();
}

#[test]
fn deformed_cylinder_normals_point_outward() {
    run_deformed_cylinder_normals_point_outward();
}

#[test]
fn deformed_cylinder_small_deformation_is_continuous() {
    run_deformed_cylinder_small_deformation_is_continuous();
}

#[test]
fn deformed_cylinder_exterior_check_respects_bulge() {
    run_deformed_cylinder_exterior_check_respects_bulge();
}

#[test]
fn deformed_cylinder_refines_with_panels() {
    run_deformed_cylinder_refines_with_panels();
}
