//! Panel ladder qualifying the deformed-cylinder BEM truth for Stage 2.

use num_complex::Complex64;
use sonium_bem::core::types::PhysicsParams;
use sonium_bem::{TwoDFourierMode, TwoDPlaneWave, TwoDRigidCylinder};
use sonium_wave::analytical::cylinder_scattering_2d;
use std::f64::consts::PI;

const RADIUS: f64 = 0.4;
const SPEED_OF_SOUND: f64 = 343.0;
const DENSITY: f64 = 1.21;
const KA_VALUES: [f64; 3] = [0.2, 1.0, 2.0];
const PANEL_LEVELS: [usize; 4] = [32, 64, 128, 256];
const PROBE_RADIUS: f64 = 1.2;
const PROBE_ANGLES: [f64; 8] = [0.2, 0.7, 1.2, 1.7, 2.2, 2.6, 3.0, 3.4];
const FAR_FIELD_ANGLES: usize = 24;

fn physics_for_ka(ka: f64) -> PhysicsParams {
    let frequency = ka / RADIUS * SPEED_OF_SOUND / (2.0 * PI);
    PhysicsParams::try_new(frequency, SPEED_OF_SOUND, DENSITY, false).expect("valid physics")
}

fn mode(order: usize, cosine: f64) -> TwoDFourierMode {
    TwoDFourierMode {
        order,
        cosine,
        sine: 0.0,
    }
}

fn shapes() -> Vec<(&'static str, Vec<TwoDFourierMode>)> {
    vec![
        ("circle", vec![]),
        ("corner-pp", vec![mode(2, 0.15), mode(3, 0.10)]),
        ("corner-pm", vec![mode(2, 0.15), mode(3, -0.10)]),
        ("corner-mp", vec![mode(2, -0.15), mode(3, 0.10)]),
        ("corner-mm", vec![mode(2, -0.15), mode(3, -0.10)]),
        ("midpoint", vec![mode(2, 0.07), mode(3, 0.05)]),
    ]
}

struct LevelOutputs {
    probes: Vec<Complex64>,
    far_field: Vec<Complex64>,
    power: f64,
}

fn solve_level(panels: usize, ka: f64, modes: &[TwoDFourierMode]) -> LevelOutputs {
    let physics = physics_for_ka(ka);
    let problem = if modes.is_empty() {
        TwoDRigidCylinder::try_new(RADIUS, panels, physics).expect("circle problem")
    } else {
        TwoDRigidCylinder::try_new_deformed(RADIUS, panels, physics, modes)
            .expect("deformed problem")
    };
    let solution = problem
        .solve(TwoDPlaneWave::plane_wave_x(), 1.0e-11)
        .expect("ladder solve");
    let points: Vec<[f64; 2]> = PROBE_ANGLES
        .iter()
        .map(|&theta| [PROBE_RADIUS * theta.cos(), PROBE_RADIUS * theta.sin()])
        .collect();
    let probes = solution
        .evaluate_total_pressure(&points)
        .expect("probe pressures");
    let mut far_field = Vec::with_capacity(FAR_FIELD_ANGLES);
    for index in 0..FAR_FIELD_ANGLES {
        let angle = 2.0 * PI * index as f64 / FAR_FIELD_ANGLES as f64;
        far_field.push(
            solution
                .far_field_amplitude(angle)
                .expect("far-field amplitude"),
        );
    }
    let power = solution
        .scattered_power_per_length(64)
        .expect("scattered power");
    LevelOutputs {
        probes,
        far_field,
        power,
    }
}

fn max_relative_change(coarse: &[Complex64], fine: &[Complex64]) -> f64 {
    coarse
        .iter()
        .zip(fine)
        .map(|(a, b)| (a - b).norm() / a.norm().max(b.norm()).max(f64::MIN_POSITIVE))
        .fold(0.0_f64, f64::max)
}

/// Pattern-level far-field change: the per-angle relative metric blows up in
/// scattering nulls at low ka, so the maximum change is normalized by the
/// maximum pattern amplitude instead.
fn far_field_pattern_change(coarse: &[Complex64], fine: &[Complex64]) -> f64 {
    let change = coarse
        .iter()
        .zip(fine)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max);
    let scale = coarse
        .iter()
        .chain(fine)
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    change / scale
}

pub fn run_deformed_cylinder_panel_ladder_qualifies_truth() {
    println!("panels | shape | ka | probe-change | far-field-change | power-change");
    let mut worst_128 = 0.0_f64;
    let mut worst_256 = 0.0_f64;
    for (name, modes) in &shapes() {
        for &ka in &KA_VALUES {
            let levels: Vec<LevelOutputs> = PANEL_LEVELS
                .iter()
                .map(|&panels| solve_level(panels, ka, modes))
                .collect();
            for pair in 0..3 {
                let probe = max_relative_change(&levels[pair].probes, &levels[pair + 1].probes);
                let far =
                    far_field_pattern_change(&levels[pair].far_field, &levels[pair + 1].far_field);
                let power = (levels[pair].power - levels[pair + 1].power).abs()
                    / levels[pair + 1].power.max(f64::MIN_POSITIVE);
                println!(
                    "{}->{} | {name:8} | {ka:.1} | {probe:.3e} | {far:.3e} | {power:.3e}",
                    PANEL_LEVELS[pair],
                    PANEL_LEVELS[pair + 1],
                );
                let worst = probe.max(far).max(power);
                match PANEL_LEVELS[pair + 1] {
                    128 => worst_128 = worst_128.max(worst),
                    256 => worst_256 = worst_256.max(worst),
                    _ => {}
                }
            }
        }
    }
    println!("worst adjacent change into 128 panels: {worst_128:.3e}");
    println!("worst adjacent change into 256 panels: {worst_256:.3e}");
    assert!(
        worst_128 < 0.02,
        "128-panel ladder gate failed: {worst_128:.3e}"
    );
    assert!(
        worst_256 < 0.02,
        "256-panel ladder gate failed: {worst_256:.3e}"
    );
}

pub fn run_deformed_cylinder_circle_matches_analytical_series() {
    let wave_number = 1.0 / RADIUS;
    for &panels in &[128_usize, 256] {
        let solution = TwoDRigidCylinder::try_new(RADIUS, panels, physics_for_ka(1.0))
            .expect("circle problem")
            .solve(TwoDPlaneWave::plane_wave_x(), 1.0e-11)
            .expect("circle solve");
        let mut maximum = 0.0_f64;
        for &angle in &PROBE_ANGLES {
            let actual = solution
                .evaluate_total_pressure(&[[
                    PROBE_RADIUS * angle.cos(),
                    PROBE_RADIUS * angle.sin(),
                ]])
                .expect("probe pressure")[0];
            let expected =
                cylinder_scattering_2d(wave_number, RADIUS, 28, vec![PROBE_RADIUS], vec![angle])
                    .pressure[0];
            maximum = maximum.max(
                (actual - expected).norm()
                    / actual.norm().max(expected.norm()).max(f64::MIN_POSITIVE),
            );
        }
        println!("circle vs analytical at {panels} panels, ka=1: {maximum:.3e}");
        assert!(
            maximum < 0.02,
            "analytical agreement failed at {panels} panels: {maximum:.3e}"
        );
    }
}

#[test]
fn deformed_cylinder_panel_ladder_qualifies_truth() {
    run_deformed_cylinder_panel_ladder_qualifies_truth();
}

#[test]
fn deformed_cylinder_circle_matches_analytical_series() {
    run_deformed_cylinder_circle_matches_analytical_series();
}
