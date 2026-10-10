//! Wolfram-derived golden for the spinning revolved FW-H strip.
//!
//! `fixtures/fwh_revolved.wls` (Wolfram 15.0.0) symbolically
//! differentiates the free-space Green function and integrates one
//! meridional strip (thickness and loading, modal phase exp(+im phi))
//! with nested Gauss-Kronrod x Clenshaw-Curtis quadrature at 40-digit
//! working precision for an oblique observer and an on-axis spinning
//! null. This test replays the checked-in
//! `fwh_revolved_golden.json` through the production
//! `stationary_fwh_ring_pressure` path — no live engine needed per
//! commit. Regenerate with:
//! `wolframscript -script fixtures/fwh_revolved.wls > fixtures/fwh_revolved_golden.json`.

use num_complex::Complex64;
use serde::Deserialize;
use sonium_fem::physics::acoustics::aeroacoustics::{
    AzimuthalMode, FwhRevolvedPatch, RevolvedQuadrature, stationary_fwh_ring_pressure,
};

const GOLDEN: &str = include_str!("../fixtures/fwh_revolved_golden.json");

#[derive(Debug, Deserialize)]
struct Golden {
    schema_version: u32,
    case_id: String,
    angular_frequency: f64,
    density: f64,
    speed_of_sound: f64,
    azimuthal_order: i32,
    node_a: [f64; 2],
    node_b: [f64; 2],
    outward_normal: [f64; 2],
    pressure: [f64; 2],
    normal_velocity: [f64; 2],
    case_a: Case,
    case_b: Case,
    tolerances: Tolerances,
}

#[derive(Debug, Deserialize)]
struct Case {
    observer: [f64; 3],
    thickness: [f64; 2],
    loading: [f64; 2],
    total: [f64; 2],
}

#[derive(Debug, Deserialize)]
struct Tolerances {
    component_abs: f64,
}

fn pair(values: [f64; 2]) -> Complex64 {
    Complex64::new(values[0], values[1])
}

fn check_case(name: &str, golden: &Golden, case: &Case) {
    let patch = FwhRevolvedPatch::new(
        golden.node_a,
        golden.node_b,
        golden.outward_normal,
        AzimuthalMode::new(golden.azimuthal_order),
        pair(golden.pressure),
        pair(golden.normal_velocity),
    )
    .unwrap();
    let actual = stationary_fwh_ring_pressure(
        std::slice::from_ref(&patch),
        case.observer,
        golden.angular_frequency,
        golden.density,
        golden.speed_of_sound,
        RevolvedQuadrature::new(1e-12, 12).unwrap(),
    )
    .unwrap();
    for (label, got, want) in [
        ("thickness", actual.thickness, pair(case.thickness)),
        ("loading", actual.loading, pair(case.loading)),
        ("total", actual.total, pair(case.total)),
    ] {
        assert!(
            (got - want).norm() < golden.tolerances.component_abs,
            "{name}/{label}: got {got:?}, want {want:?}"
        );
    }
}

#[test]
fn spinning_revolved_fwh_matches_wolfram() {
    let golden: Golden = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(golden.schema_version, 1);
    assert_eq!(golden.case_id, "fem.fwh-revolved.v1");
    check_case("case_a", &golden, &golden.case_a);
    check_case("case_b", &golden, &golden.case_b);
}
