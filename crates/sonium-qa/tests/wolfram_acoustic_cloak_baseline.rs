//! Wolfram Acoustic Cloak uncloaked rigid-cylinder baseline.
//!
//! Wolfram publishes a maximum total-pressure amplitude of 1.89447 Pa for a
//! unit incident plane wave at 800 Hz in water, scattered by a rigid cylinder
//! of radius 1 m. This test checks that scalar against Sonium's independent
//! Bessel/Hankel series. The production `qa-fem-acoustics-cylinder-scattering`
//! executable separately checks Sonium FEM against the same analytical family.

use sonium_bem::analytical::cylinder_scattering_2d;
use std::f64::consts::PI;

const CYLINDER_RADIUS_M: f64 = 1.0;
const FREQUENCY_HZ: f64 = 800.0;
const WATER_SOUND_SPEED_M_S: f64 = 1481.0;
const WOLFRAM_MAX_PRESSURE_PA: f64 = 1.89447;

#[test]
fn sonium_series_reproduces_wolfram_uncloaked_cylinder_maximum() {
    let wave_number = 2.0 * PI * FREQUENCY_HZ / WATER_SOUND_SPEED_M_S;
    let mut radii = Vec::new();
    let mut angles = Vec::new();
    // The maximum for this rigid-cylinder case lies on or immediately outside
    // the cylinder. Sample five near-surface rings at half-degree resolution;
    // this avoids turning a scalar golden-value check into a costly field scan.
    for radius in [1.0, 1.025, 1.05, 1.1, 1.2] {
        for index in 0..=360 {
            radii.push(radius);
            angles.push(index as f64 * PI / 360.0);
        }
    }
    let solution = cylinder_scattering_2d(wave_number, CYLINDER_RADIUS_M, 30, radii, angles);
    let maximum = solution
        .pressure
        .iter()
        .map(|pressure| pressure.norm())
        .fold(0.0_f64, f64::max);

    assert!((wave_number * CYLINDER_RADIUS_M - 3.394).abs() < 0.01);
    assert!(
        (maximum - WOLFRAM_MAX_PRESSURE_PA).abs() < 0.04,
        "series maximum {maximum:.6} Pa, Wolfram {WOLFRAM_MAX_PRESSURE_PA:.6} Pa"
    );
}
