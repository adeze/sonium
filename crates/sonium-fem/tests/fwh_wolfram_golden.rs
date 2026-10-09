//! Wolfram-derived golden for the stationary FW-H kernel.
//!
//! `fixtures/fwh_monopole_patch.wls` (Wolfram 15.0.0, 50-digit precision)
//! independently derives single-patch thickness / loading / total pressures
//! for a loading-only, a thickness-only, and a combined oblique fixture.
//! This test replays the checked-in `fwh_monopole_patch_golden.json`
//! through the production `stationary_fwh_pressure` path — no live engine
//! needed per commit. Regenerate with:
//! `wolframscript -script fixtures/fwh_monopole_patch.wls`.

use num_complex::Complex64;
use serde::Deserialize;
use sonium_fem::physics::acoustics::aeroacoustics::{FwhSurfacePatch, stationary_fwh_pressure};

const GOLDEN: &str = include_str!("../fixtures/fwh_monopole_patch_golden.json");

#[derive(Debug, Deserialize)]
struct Golden {
    schema_version: u32,
    case_id: String,
    angular_frequency: f64,
    density: f64,
    speed_of_sound: f64,
    patch: Patch,
    case_a: Case,
    case_b: Case,
    case_c: Case,
    tolerances: Tolerances,
}

#[derive(Debug, Deserialize)]
struct Patch {
    position: [f64; 3],
    outward_normal: [f64; 3],
    area: f64,
}

#[derive(Debug, Deserialize)]
struct Case {
    observer: [f64; 3],
    pressure: [f64; 2],
    normal_velocity: [f64; 2],
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
    let patch = FwhSurfacePatch::new(
        golden.patch.position,
        golden.patch.outward_normal,
        golden.patch.area,
        pair(case.pressure),
        pair(case.normal_velocity),
    )
    .unwrap();
    let actual = stationary_fwh_pressure(
        &[patch],
        case.observer,
        golden.angular_frequency,
        golden.density,
        golden.speed_of_sound,
    )
    .unwrap();
    for (label, got, want) in [
        ("thickness", actual.thickness, pair(case.thickness)),
        ("loading", actual.loading, pair(case.loading)),
        ("total", actual.total, pair(case.total)),
    ] {
        assert!(
            (got - want).norm() < golden.tolerances.component_abs,
            "{name} {label}: got {got:?}, want {want:?}"
        );
    }
    // The kernel contract: total is the sum of the two mechanisms.
    assert!((actual.total - (actual.thickness + actual.loading)).norm() < 1e-15);
}

#[test]
fn wolfram_fwh_single_patch_replays_through_production_path() {
    let golden: Golden = serde_json::from_str(GOLDEN).expect("parse golden");
    assert_eq!(golden.schema_version, 1);
    assert_eq!(golden.case_id, "fem.fwh-single-patch.v1");
    check_case("loading-only", &golden, &golden.case_a);
    check_case("thickness-only", &golden, &golden.case_b);
    check_case("combined-oblique", &golden, &golden.case_c);
}
