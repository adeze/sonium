//! Wolfram-derived golden for the planar FW-H kernel.
//!
//! `fixtures/caa_fwh.wls` (Wolfram 15.0.0, 50-digit precision)
//! independently derives single-segment thickness / loading / total
//! pressures with the aero-convention Hankel kernel for a
//! loading-only, a thickness-only, and a combined oblique fixture,
//! plus a closed four-segment square contour fed with line-source
//! data and the Hankel spreading ratio at `kr = 100`. This test
//! replays the checked-in `caa_fwh_golden.json` through the
//! production `stationary_fwh_pressure_2d` path — no live engine
//! needed per commit. Regenerate with:
//! `wolframscript -script fixtures/caa_fwh.wls > fixtures/caa_fwh_golden.json`.

use num_complex::Complex64;
use serde::Deserialize;
use sonium_fem::physics::acoustics::aeroacoustics::replay::fwh_component_gap;
use sonium_fem::physics::acoustics::aeroacoustics::{
    FwhContourSegment, stationary_fwh_pressure_2d,
};

const GOLDEN: &str = include_str!("../fixtures/caa_fwh_golden.json");

#[derive(Debug, Deserialize)]
struct Golden {
    schema_version: u32,
    case_id: String,
    angular_frequency: f64,
    density: f64,
    speed_of_sound: f64,
    segment: Segment,
    case_a: Case,
    case_b: Case,
    case_c: Case,
    case_d: ClosedContour,
    spreading: Spreading,
    tolerances: Tolerances,
}

#[derive(Debug, Deserialize)]
struct Segment {
    position: [f64; 2],
    outward_normal: [f64; 2],
    length: f64,
}

#[derive(Debug, Deserialize)]
struct Case {
    observer: [f64; 2],
    pressure: [f64; 2],
    normal_velocity: [f64; 2],
    thickness: [f64; 2],
    loading: [f64; 2],
    total: [f64; 2],
}

#[derive(Debug, Deserialize)]
struct ClosedContour {
    observer: [f64; 2],
    segments: Vec<ContourSegment>,
    thickness: [f64; 2],
    loading: [f64; 2],
    total: [f64; 2],
}

#[derive(Debug, Deserialize)]
struct ContourSegment {
    position: [f64; 2],
    outward_normal: [f64; 2],
    length: f64,
    pressure: [f64; 2],
    normal_velocity: [f64; 2],
    thickness: [f64; 2],
    loading: [f64; 2],
    total: [f64; 2],
}

#[derive(Debug, Deserialize)]
struct Spreading {
    distance_near: f64,
    distance_far: f64,
    ratio: f64,
}

#[derive(Debug, Deserialize)]
struct Tolerances {
    component_abs: f64,
    ratio_abs: f64,
}

fn pair(values: [f64; 2]) -> Complex64 {
    Complex64::new(values[0], values[1])
}

fn check_case(name: &str, golden: &Golden, case: &Case) {
    let segment = FwhContourSegment::new(
        golden.segment.position,
        golden.segment.outward_normal,
        golden.segment.length,
        pair(case.pressure),
        pair(case.normal_velocity),
    )
    .unwrap();
    let actual = stationary_fwh_pressure_2d(
        &[segment],
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
    assert!(fwh_component_gap(&actual) < 1e-15, "{name} component sum");
}

#[test]
fn wolfram_fwh_2d_replays_through_production_path() {
    let golden: Golden = serde_json::from_str(GOLDEN).expect("parse golden");
    assert_eq!(golden.schema_version, 1);
    assert_eq!(golden.case_id, "fem.caa-fwh-2d.v1");
    check_case("loading-only", &golden, &golden.case_a);
    check_case("thickness-only", &golden, &golden.case_b);
    check_case("combined-oblique", &golden, &golden.case_c);

    // Closed square contour: per-segment replay plus the summed field.
    assert_eq!(golden.case_d.segments.len(), 4);
    let segments: Vec<FwhContourSegment> = golden
        .case_d
        .segments
        .iter()
        .map(|entry| {
            FwhContourSegment::new(
                entry.position,
                entry.outward_normal,
                entry.length,
                pair(entry.pressure),
                pair(entry.normal_velocity),
            )
            .unwrap()
        })
        .collect();
    for (index, entry) in golden.case_d.segments.iter().enumerate() {
        let actual = stationary_fwh_pressure_2d(
            &[segments[index]],
            golden.case_d.observer,
            golden.angular_frequency,
            golden.density,
            golden.speed_of_sound,
        )
        .unwrap();
        for (label, got, want) in [
            ("thickness", actual.thickness, pair(entry.thickness)),
            ("loading", actual.loading, pair(entry.loading)),
            ("total", actual.total, pair(entry.total)),
        ] {
            assert!(
                (got - want).norm() < golden.tolerances.component_abs,
                "contour segment {index} {label}: got {got:?}, want {want:?}"
            );
        }
    }
    let summed = stationary_fwh_pressure_2d(
        &segments,
        golden.case_d.observer,
        golden.angular_frequency,
        golden.density,
        golden.speed_of_sound,
    )
    .unwrap();
    for (label, got, want) in [
        ("thickness", summed.thickness, pair(golden.case_d.thickness)),
        ("loading", summed.loading, pair(golden.case_d.loading)),
        ("total", summed.total, pair(golden.case_d.total)),
    ] {
        assert!(
            (got - want).norm() < golden.tolerances.component_abs,
            "contour sum {label}: got {got:?}, want {want:?}"
        );
    }
    assert!(fwh_component_gap(&summed) < 1e-15, "contour component sum");

    // Hankel spreading ratio between two ranges (thickness-only probe).
    let probe = FwhContourSegment::new(
        golden.segment.position,
        golden.segment.outward_normal,
        golden.segment.length,
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    )
    .unwrap();
    let near = stationary_fwh_pressure_2d(
        &[probe],
        [
            golden.segment.position[0] + golden.spreading.distance_near,
            golden.segment.position[1],
        ],
        golden.angular_frequency,
        golden.density,
        golden.speed_of_sound,
    )
    .unwrap();
    let far = stationary_fwh_pressure_2d(
        &[probe],
        [
            golden.segment.position[0] + golden.spreading.distance_far,
            golden.segment.position[1],
        ],
        golden.angular_frequency,
        golden.density,
        golden.speed_of_sound,
    )
    .unwrap();
    let ratio = far.thickness.norm() / near.thickness.norm();
    assert!(
        (ratio - golden.spreading.ratio).abs() < golden.tolerances.ratio_abs,
        "spreading: got {ratio}, want {}",
        golden.spreading.ratio
    );
}
