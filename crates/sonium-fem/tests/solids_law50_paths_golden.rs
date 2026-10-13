//! Wolfram golden replay: LAW50 honeycomb strain paths.
//!
//! Replays `fixtures/solids_law50_paths_golden.json` (derived by
//! `fixtures/solids_law50_paths.wls` at 100-digit working precision,
//! exported at 50 digits) through the production LAW50 API. Regenerate
//! with:
//! `wolframscript -script fixtures/solids_law50_paths.wls > fixtures/solids_law50_paths_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::funct::Funct1D;
use sonium_fem::physics::solid::plasticity::law_foams::{HoneycombLaw, HoneycombState};
use sonium_fem::physics::solid::plasticity::update::{StressUpdateModel, von_mises_c1};

const GOLDEN: &str = include_str!("../fixtures/solids_law50_paths_golden.json");

/// Steps per leg, matching the derivation script.
const STEPS_PER_LEG: usize = 100;
/// Crush extent of every path: 0.3 drives each axis deep into its
/// rated crush curve (yield starts near 1% strain).
const EMAX: f64 = 0.3;
/// Step strain rate in 1/s: dt = de / STEP_RATE pins the midpoint of
/// the rated (0, 10 /s) yield tables on every step.
const STEP_RATE: f64 = 5.0;

/// Accumulated result of one driven strain path.
struct PathResult {
    sigma: [f64; 6],
    epbar: f64,
    yielding: Vec<bool>,
    epbar_stations: Vec<f64>,
}

fn golden_values(doc: &Value, name: &str) -> Vec<f64> {
    doc["values"][name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .iter()
        .map(|v| {
            v.as_str()
                .unwrap_or_else(|| panic!("golden lacks {name}"))
                .parse()
                .unwrap()
        })
        .collect()
}

fn check_all(doc: &Value, name: &str, actual: &[f64]) {
    let expected = golden_values(doc, name);
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= abs + rel * want.abs(),
            "{name}[{i}]: production {got:.6e} vs golden {want:.6e}"
        );
    }
}

fn check_flags(doc: &Value, name: &str, actual: &[bool]) {
    let numeric: Vec<f64> = actual.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    check_all(doc, name, &numeric);
}

/// Builds the directional LAW50 crush law of the derivation script.
///
/// Per-direction rated yield curves (rates 0 and 10 /s) with
/// directional crush emphasis: axis 3 crushes hardest.
fn crush_law() -> HoneycombLaw {
    let curve = |lo: [f64; 2], hi: [f64; 2]| {
        vec![
            (
                Funct1D::new(vec![0.0, 1.0], vec![lo[0], lo[1]]).unwrap(),
                0.0,
            ),
            (
                Funct1D::new(vec![0.0, 1.0], vec![hi[0], hi[1]]).unwrap(),
                10.0,
            ),
        ]
    };
    let rated = [
        curve([1.0, 3.0], [2.0, 4.0]),
        curve([1.5, 4.0], [2.5, 5.0]),
        curve([2.0, 5.0], [3.0, 6.0]),
        curve([1.0, 1.0], [2.0, 2.0]),
        curve([1.0, 1.0], [2.0, 2.0]),
        curve([1.0, 1.0], [2.0, 2.0]),
    ];
    HoneycombLaw::law50([100.0; 6], rated).unwrap()
}

/// Drives fixed strain increments, recording stations (1-based steps).
///
/// Each step runs the full production triple (`update_with_rate`,
/// `commit`, `advance_totals`), exactly what `step` bundles.
fn drive_fixed(
    law: &HoneycombLaw,
    state: &mut HoneycombState,
    steps: &[[f64; 6]],
    stations: &[usize],
    dt: f64,
) -> PathResult {
    let mut yielding = Vec::new();
    let mut epbar_stations = Vec::new();
    for (i, step) in steps.iter().enumerate() {
        let update = law.update_with_rate(state, step, dt);
        law.commit(state, &update);
        law.advance_totals(state, step, dt);
        if stations.contains(&(i + 1)) {
            yielding.push(update.yielding);
            epbar_stations.push(state.equiv_plastic);
        }
    }
    PathResult {
        sigma: state.stresses,
        epbar: state.equiv_plastic,
        yielding,
        epbar_stations,
    }
}

#[test]
fn law50_paths_match_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["units"], "SI");
    let law = crush_law();
    let stations_100 = vec![20, 40, 60, 80, 100];
    let stations_cycle = vec![80, 160, 240, 320, 400];

    let de = EMAX / STEPS_PER_LEG as f64;
    let dt = de / STEP_RATE;

    // Axis-1 compression to 30% crush.
    let mut state = HoneycombState::new();
    let ax1 = drive_fixed(
        &law,
        &mut state,
        &vec![[-de, 0.0, 0.0, 0.0, 0.0, 0.0]; STEPS_PER_LEG],
        &stations_100,
        dt,
    );
    check_all(&doc, "ax1_sigma", &ax1.sigma);
    check_all(&doc, "ax1_epbar", &[ax1.epbar]);
    check_flags(&doc, "ax1_yielding", &ax1.yielding);
    check_all(&doc, "ax1_epbar_st", &ax1.epbar_stations);
    assert!(ax1.yielding.iter().any(|&y| y), "axis 1 must crush");
    assert_eq!(&ax1.sigma[1..], &[0.0; 5], "laterals must stay zero");

    // Axis-2 compression to 30% crush.
    let mut state = HoneycombState::new();
    let ax2 = drive_fixed(
        &law,
        &mut state,
        &vec![[0.0, -de, 0.0, 0.0, 0.0, 0.0]; STEPS_PER_LEG],
        &stations_100,
        dt,
    );
    check_all(&doc, "ax2_sigma", &ax2.sigma);
    check_all(&doc, "ax2_epbar", &[ax2.epbar]);
    check_flags(&doc, "ax2_yielding", &ax2.yielding);
    check_all(&doc, "ax2_epbar_st", &ax2.epbar_stations);
    assert!(ax2.yielding.iter().any(|&y| y), "axis 2 must crush");

    // Axis-3 compression to 30% crush.
    let mut state = HoneycombState::new();
    let ax3 = drive_fixed(
        &law,
        &mut state,
        &vec![[0.0, 0.0, -de, 0.0, 0.0, 0.0]; STEPS_PER_LEG],
        &stations_100,
        dt,
    );
    check_all(&doc, "ax3_sigma", &ax3.sigma);
    check_all(&doc, "ax3_epbar", &[ax3.epbar]);
    check_flags(&doc, "ax3_yielding", &ax3.yielding);
    check_all(&doc, "ax3_epbar_st", &ax3.epbar_stations);
    assert!(ax3.yielding.iter().any(|&y| y), "axis 3 must crush");
    assert!(
        ax3.sigma[2].abs() > ax2.sigma[1].abs() && ax2.sigma[1].abs() > ax1.sigma[0].abs(),
        "directional crush emphasis must order the axes"
    );

    // Symmetric axis-1 cycle 0 -> +emax -> -emax -> 0.
    let mut des = vec![de; STEPS_PER_LEG];
    des.extend(vec![-de; 2 * STEPS_PER_LEG]);
    des.extend(vec![de; STEPS_PER_LEG]);
    let steps: Vec<[f64; 6]> = des.iter().map(|&d| [d, 0.0, 0.0, 0.0, 0.0, 0.0]).collect();
    let mut state = HoneycombState::new();
    let cyc = drive_fixed(&law, &mut state, &steps, &stations_cycle, dt);
    check_all(&doc, "cycle_sigma", &cyc.sigma);
    check_all(&doc, "cycle_epbar", &[cyc.epbar]);
    check_flags(&doc, "cycle_yielding", &cyc.yielding);
    check_all(&doc, "cycle_epbar_st", &cyc.epbar_stations);
    assert!(cyc.yielding.iter().any(|&y| y), "cycle must crush");
    check_all(&doc, "cycle_closure", &[von_mises_c1(&cyc.sigma)]);
}
