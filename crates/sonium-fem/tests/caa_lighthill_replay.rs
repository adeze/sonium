//! Wolfram-derived golden for the compact Lighthill kernel.
//!
//! `fixtures/caa_lighthill.wls` (Wolfram 15.0.0, 50-digit precision)
//! independently derives exact and far-field-only pressures of an axial
//! quadrupole at `kr = 1, 10, 100`, the transverse far-field null, the
//! directional factor at five polar angles plus its sphere
//! normalization, the `1/r` spreading ratio, and a reciprocity delta.
//! This test replays the checked-in `caa_lighthill_golden.json`
//! through the production `lighthill_pressure` /
//! `lighthill_far_field_pressure` path plus the `replay` check helpers
//! (`directivity_factor`, `lighthill_reciprocity_gap`) — no live engine
//! needed per commit. Regenerate with:
//! `wolframscript -script fixtures/caa_lighthill.wls > fixtures/caa_lighthill_golden.json`.

use num_complex::Complex64;
use serde::Deserialize;
use sonium_fem::physics::acoustics::aeroacoustics::lighthill::{
    LighthillQuadrupole, lighthill_far_field_pressure, lighthill_pressure,
};
use sonium_fem::physics::acoustics::aeroacoustics::replay::{
    directivity_factor, lighthill_reciprocity_gap,
};

const GOLDEN: &str = include_str!("../fixtures/caa_lighthill_golden.json");

#[derive(Debug, Deserialize)]
struct Golden {
    schema_version: u32,
    case_id: String,
    angular_frequency: f64,
    speed_of_sound: f64,
    stress_moment: [[[f64; 2]; 3]; 3],
    kr_cases: Vec<KrCase>,
    transverse: FieldCase,
    directivity: Directivity,
    spreading: Spreading,
    reciprocity: Reciprocity,
    tolerances: Tolerances,
}

#[derive(Debug, Deserialize)]
struct KrCase {
    kr: f64,
    observer: [f64; 3],
    exact: [f64; 2],
    far: [f64; 2],
    reactive: [f64; 2],
}

#[derive(Debug, Deserialize)]
struct FieldCase {
    observer: [f64; 3],
    exact: [f64; 2],
    far: [f64; 2],
}

#[derive(Debug, Deserialize)]
struct Directivity {
    angles_rad: Vec<f64>,
    factors: Vec<[f64; 2]>,
    normalization: f64,
}

#[derive(Debug, Deserialize)]
struct Spreading {
    distance_near: f64,
    distance_far: f64,
    ratio: f64,
}

#[derive(Debug, Deserialize)]
struct Reciprocity {
    point_a: [f64; 3],
    point_b: [f64; 3],
    delta: [f64; 2],
}

#[derive(Debug, Deserialize)]
struct Tolerances {
    pressure_abs: f64,
    factor_abs: f64,
    ratio_abs: f64,
    norm_rel: f64,
}

fn pair(values: [f64; 2]) -> Complex64 {
    Complex64::new(values[0], values[1])
}

fn tensor(moment: [[[f64; 2]; 3]; 3]) -> [[Complex64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| pair(moment[i][j])))
}

/// Mean-square directional factor over the sphere (midpoint rule).
///
/// Integrates `|D|^2` with a uniform theta/phi midpoint grid; the
/// smooth `cos^4` integrand keeps the documented `norm_rel` band at
/// this resolution with wide margin.
fn sphere_mean_square(moment: [[Complex64; 3]; 3]) -> f64 {
    const MERIDIONAL: usize = 720;
    const HOOP: usize = 720;
    let dtheta = std::f64::consts::PI / MERIDIONAL as f64;
    let dphi = 2.0 * std::f64::consts::PI / HOOP as f64;
    let mut sum = 0.0;
    for m in 0..MERIDIONAL {
        let theta = (m as f64 + 0.5) * dtheta;
        let (sin_theta, cos_theta) = theta.sin_cos();
        for p in 0..HOOP {
            let phi = (p as f64 + 0.5) * dphi;
            let direction = [sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta];
            sum += directivity_factor(moment, direction).norm_sqr() * sin_theta;
        }
    }
    sum * dtheta * dphi / (4.0 * std::f64::consts::PI)
}

#[test]
fn wolfram_lighthill_replays_through_production_path() {
    let golden: Golden = serde_json::from_str(GOLDEN).expect("parse golden");
    assert_eq!(golden.schema_version, 1);
    assert_eq!(golden.case_id, "fem.caa-lighthill.v1");
    let moment = tensor(golden.stress_moment);
    let source = LighthillQuadrupole::new([0.0; 3], moment).unwrap();
    let (omega, sound) = (golden.angular_frequency, golden.speed_of_sound);

    for case in &golden.kr_cases {
        let exact = lighthill_pressure(&source, case.observer, omega, sound).unwrap();
        let far = lighthill_far_field_pressure(&source, case.observer, omega, sound).unwrap();
        assert!(
            (exact - pair(case.exact)).norm() < golden.tolerances.pressure_abs,
            "kr={} exact: got {exact:?}, want {:?}",
            case.kr,
            pair(case.exact)
        );
        assert!(
            (far - pair(case.far)).norm() < golden.tolerances.pressure_abs,
            "kr={} far: got {far:?}, want {:?}",
            case.kr,
            pair(case.far)
        );
        assert!(
            ((exact - far) - pair(case.reactive)).norm() < golden.tolerances.pressure_abs,
            "kr={} reactive: got {:?}, want {:?}",
            case.kr,
            exact - far,
            pair(case.reactive)
        );
    }

    // Transverse far-field null: the axial quadrupole does not radiate
    // across its axis, while the exact field keeps a reactive remainder.
    let exact = lighthill_pressure(&source, golden.transverse.observer, omega, sound).unwrap();
    let far =
        lighthill_far_field_pressure(&source, golden.transverse.observer, omega, sound).unwrap();
    assert!(
        (exact - pair(golden.transverse.exact)).norm() < golden.tolerances.pressure_abs,
        "transverse exact: got {exact:?}"
    );
    assert_eq!(pair(golden.transverse.far), Complex64::new(0.0, 0.0));
    assert!(far.norm() == 0.0, "transverse far-field null: got {far:?}");
    assert!(
        exact.norm() > 0.0,
        "transverse exact field must keep a reactive remainder"
    );

    // Directional factor point values plus sphere normalization.
    assert_eq!(
        golden.directivity.angles_rad.len(),
        golden.directivity.factors.len()
    );
    for (angle, want) in golden
        .directivity
        .angles_rad
        .iter()
        .zip(golden.directivity.factors.iter())
    {
        let direction = [angle.cos(), angle.sin(), 0.0];
        let got = directivity_factor(moment, direction);
        assert!(
            (got - pair(*want)).norm() < golden.tolerances.factor_abs,
            "D({angle}): got {got:?}, want {want:?}"
        );
    }
    let normalization = sphere_mean_square(moment);
    assert!(
        (normalization - golden.directivity.normalization).abs()
            < golden.tolerances.norm_rel * golden.directivity.normalization,
        "normalization: got {normalization}, want {}",
        golden.directivity.normalization
    );

    // Spreading of the radiating term and reciprocity of the kernel.
    let near = lighthill_far_field_pressure(
        &source,
        [golden.spreading.distance_near, 0.0, 0.0],
        omega,
        sound,
    )
    .unwrap();
    let far = lighthill_far_field_pressure(
        &source,
        [golden.spreading.distance_far, 0.0, 0.0],
        omega,
        sound,
    )
    .unwrap();
    let ratio = far.norm() / near.norm();
    assert!(
        (ratio - golden.spreading.ratio).abs() < golden.tolerances.ratio_abs,
        "spreading: got {ratio}, want {}",
        golden.spreading.ratio
    );
    let gap = lighthill_reciprocity_gap(
        moment,
        golden.reciprocity.point_a,
        golden.reciprocity.point_b,
        omega,
        sound,
    )
    .unwrap();
    assert_eq!(pair(golden.reciprocity.delta), Complex64::new(0.0, 0.0));
    assert!(gap < 1e-12, "reciprocity gap: {gap}");
}
