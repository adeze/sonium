//! Wolfram-derived golden for the spinning Lighthill ring.
//!
//! `fixtures/lighthill_ring.wls` (Wolfram 15.0.0) symbolically
//! differentiates the free-space Green function, rotates the
//! cylindrical moment per hoop angle, and integrates with
//! Clenshaw-Curtis quadrature at 40-digit working precision for an
//! axisymmetric case, a rotation-active m = 2 case, and an on-axis
//! spinning null. This test replays the checked-in
//! `lighthill_ring_golden.json` through the production
//! `lighthill_ring_pressure` path — no live engine needed per commit.
//! Regenerate with:
//! `wolframscript -script fixtures/lighthill_ring.wls > fixtures/lighthill_ring_golden.json`.

use num_complex::Complex64;
use serde::Deserialize;
use sonium_fem::physics::acoustics::aeroacoustics::{
    AzimuthalMode, LighthillRingSource, lighthill_ring_pressure,
};

const GOLDEN: &str = include_str!("../fixtures/lighthill_ring_golden.json");

#[derive(Debug, Deserialize)]
struct Golden {
    schema_version: u32,
    case_id: String,
    angular_frequency: f64,
    speed_of_sound: f64,
    ring_radius_m: f64,
    axial_position_m: f64,
    case_a: Case,
    case_b: Case,
    case_c: Case,
    tolerances: Tolerances,
}

#[derive(Debug, Deserialize)]
struct Case {
    azimuthal_order: i32,
    observer: [f64; 3],
    stress_moment: [[[f64; 2]; 3]; 3],
    pressure: [f64; 2],
}

#[derive(Debug, Deserialize)]
struct Tolerances {
    pressure_abs: f64,
}

fn pair(values: [f64; 2]) -> Complex64 {
    Complex64::new(values[0], values[1])
}

fn check_case(name: &str, golden: &Golden, case: &Case) {
    let moment = case.stress_moment.map(|row| row.map(pair));
    let source = LighthillRingSource::new(
        golden.ring_radius_m,
        golden.axial_position_m,
        AzimuthalMode::new(case.azimuthal_order),
        moment,
    )
    .unwrap();
    let actual = lighthill_ring_pressure(
        &source,
        case.observer,
        golden.angular_frequency,
        golden.speed_of_sound,
        1e-12,
    )
    .unwrap();
    let expected = pair(case.pressure);
    assert!(
        (actual - expected).norm() < golden.tolerances.pressure_abs,
        "{name}: got {actual:?}, want {expected:?}"
    );
}

#[test]
fn spinning_lighthill_ring_matches_wolfram() {
    let golden: Golden = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(golden.schema_version, 1);
    assert_eq!(golden.case_id, "fem.lighthill-ring.v1");
    check_case("case_a", &golden, &golden.case_a);
    check_case("case_b", &golden, &golden.case_b);
    check_case("case_c", &golden, &golden.case_c);
}
