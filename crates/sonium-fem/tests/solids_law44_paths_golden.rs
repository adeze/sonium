//! Wolfram golden replay: LAW44 Cowper-Symonds strain paths.
//!
//! Replays `fixtures/solids_law44_paths_golden.json` (derived by
//! `fixtures/solids_law44_paths.wls` at 400-digit working precision,
//! exported at 50 digits) through the production LAW44 API. Regenerate
//! with:
//! `wolframscript -script fixtures/solids_law44_paths.wls > fixtures/solids_law44_paths_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::law_metals::{Law44, Law44State};
use sonium_fem::physics::solid::plasticity::update::{StressUpdateModel, von_mises_c1};

const GOLDEN: &str = include_str!("../fixtures/solids_law44_paths_golden.json");

/// Steps per leg, matching the derivation script.
const STEPS_PER_LEG: usize = 100;
/// Lateral bisection passes per axial step, matching the derivation.
const LATERAL_ITERATIONS: usize = 50;
/// Axial extent of every path: 0.02 drives the card metal to 1.7%
/// plastic strain on the monotonic legs (well past yield).
const EMAX: f64 = 0.02;
/// Nominal strain rates in 1/s of the slow and fast rate legs.
const RATE_SLOW: f64 = 1.0e-3;
const RATE_FAST: f64 = 1.0e2;

/// Accumulated result of one driven strain path.
struct PathResult {
    sigma: [f64; 6],
    epbar: f64,
    yielding: Vec<bool>,
    epbar_stations: Vec<f64>,
    back_stress: [f64; 6],
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

/// Trial update honoring an optional step time for the strain rate.
fn trial(law: &Law44, state: &Law44State, d: &[f64; 6], dt: Option<f64>) -> [f64; 6] {
    match dt {
        Some(dt) => law.update_with_rate(state, d, dt).stress,
        None => law.update(state, d).stress,
    }
}

/// Solves the lateral strain of one axial step for zero lateral stress.
///
/// Bisects the pure production update (50 passes over a widening
/// bracket), mirroring the derivation script exactly.
fn solve_lateral(law: &Law44, state: &Law44State, de: f64, dt: Option<f64>) -> f64 {
    let (mut lo, mut hi) = if de >= 0.0 { (-de, 0.0) } else { (0.0, -de) };
    let probe = |dl: f64| trial(law, state, &[de, dl, dl, 0.0, 0.0, 0.0], dt)[1];
    let mut blo = probe(lo);
    let mut bhi = probe(hi);
    for _ in 0..20 {
        if blo < 0.0 && bhi > 0.0 {
            break;
        }
        let width = (hi - lo).abs();
        lo -= width;
        hi += width;
        blo = probe(lo);
        bhi = probe(hi);
    }
    assert!(blo < 0.0 && bhi > 0.0, "lateral bracket failed for de={de}");
    for _ in 0..LATERAL_ITERATIONS {
        let mid = 0.5 * (lo + hi);
        if probe(mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Drives fixed strain increments, recording stations (1-based steps).
fn drive_fixed(
    law: &Law44,
    state: &mut Law44State,
    steps: &[[f64; 6]],
    stations: &[usize],
    dt: Option<f64>,
) -> PathResult {
    let mut yielding = Vec::new();
    let mut epbar_stations = Vec::new();
    for (i, step) in steps.iter().enumerate() {
        let update = match dt {
            Some(dt) => law.update_with_rate(state, step, dt),
            None => law.update(state, step),
        };
        law.commit(state, &update);
        if stations.contains(&(i + 1)) {
            yielding.push(update.yielding);
            epbar_stations.push(state.equiv_plastic_strain);
        }
    }
    PathResult {
        sigma: state.stress,
        epbar: state.equiv_plastic_strain,
        yielding,
        epbar_stations,
        back_stress: state.back_stress,
    }
}

/// Drives axial increments with the lateral solve, recording stations.
fn drive_axial(
    law: &Law44,
    state: &mut Law44State,
    des: &[f64],
    stations: &[usize],
    dt: Option<f64>,
) -> PathResult {
    let mut yielding = Vec::new();
    let mut epbar_stations = Vec::new();
    for (i, &de) in des.iter().enumerate() {
        let dl = solve_lateral(law, state, de, dt);
        let d = [de, dl, dl, 0.0, 0.0, 0.0];
        let update = match dt {
            Some(dt) => law.update_with_rate(state, &d, dt),
            None => law.update(state, &d),
        };
        law.commit(state, &update);
        if stations.contains(&(i + 1)) {
            yielding.push(update.yielding);
            epbar_stations.push(state.equiv_plastic_strain);
        }
    }
    PathResult {
        sigma: state.stress,
        epbar: state.equiv_plastic_strain,
        yielding,
        epbar_stations,
        back_stress: state.back_stress,
    }
}

/// Von Mises norm of the relative (over) stress.
fn relative_mises(sigma: &[f64; 6], back: &[f64; 6]) -> f64 {
    let rel: [f64; 6] = std::array::from_fn(|i| sigma[i] - back[i]);
    von_mises_c1(&rel)
}

#[test]
fn law44_paths_match_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["units"], "SI");
    let law = Law44::card_metal();
    let stations_100 = vec![20, 40, 60, 80, 100];
    let stations_cycle = vec![80, 160, 240, 320, 400];

    // Uniaxial extension with solved lateral strain.
    let de = EMAX / STEPS_PER_LEG as f64;
    let mut state = Law44State::new();
    let uni = drive_axial(
        &law,
        &mut state,
        &vec![de; STEPS_PER_LEG],
        &stations_100,
        None,
    );
    check_all(&doc, "uni_sigma", &uni.sigma);
    check_all(&doc, "uni_epbar", &[uni.epbar]);
    check_flags(&doc, "uni_yielding", &uni.yielding);
    check_all(&doc, "uni_epbar_st", &uni.epbar_stations);
    check_all(&doc, "uni_back", &uni.back_stress);
    assert!(uni.yielding.iter().any(|&y| y), "uniaxial must yield");
    assert!(
        uni.back_stress.iter().any(|&b| b != 0.0),
        "kinematic hardening must translate the back-stress"
    );

    // Pure shear ramp.
    let dg = EMAX / STEPS_PER_LEG as f64;
    let mut state = Law44State::new();
    let shear = drive_fixed(
        &law,
        &mut state,
        &vec![[0.0, 0.0, 0.0, dg, 0.0, 0.0]; STEPS_PER_LEG],
        &stations_100,
        None,
    );
    check_all(&doc, "shear_sigma", &shear.sigma);
    check_all(&doc, "shear_epbar", &[shear.epbar]);
    check_flags(&doc, "shear_yielding", &shear.yielding);
    check_all(&doc, "shear_epbar_st", &shear.epbar_stations);
    assert!(shear.yielding.iter().any(|&y| y), "shear must yield");

    // Volumetric ramp: J2 never yields hydrostatically.
    let dv = EMAX / STEPS_PER_LEG as f64;
    let mut state = Law44State::new();
    let vol = drive_fixed(
        &law,
        &mut state,
        &vec![[dv, dv, dv, 0.0, 0.0, 0.0]; STEPS_PER_LEG],
        &stations_100,
        None,
    );
    check_all(&doc, "vol_sigma", &vol.sigma);
    check_all(&doc, "vol_epbar", &[vol.epbar]);
    check_flags(&doc, "vol_yielding", &vol.yielding);
    check_all(&doc, "vol_epbar_st", &vol.epbar_stations);
    assert!(
        vol.yielding.iter().all(|&y| !y),
        "volumetric must stay elastic"
    );
    assert_eq!(vol.epbar, 0.0);

    // Symmetric strain cycle with solved lateral strain.
    let mut des = vec![de; STEPS_PER_LEG];
    des.extend(vec![-de; 2 * STEPS_PER_LEG]);
    des.extend(vec![de; STEPS_PER_LEG]);
    let mut state = Law44State::new();
    let cyc = drive_axial(&law, &mut state, &des, &stations_cycle, None);
    check_all(&doc, "cycle_sigma", &cyc.sigma);
    check_all(&doc, "cycle_epbar", &[cyc.epbar]);
    check_flags(&doc, "cycle_yielding", &cyc.yielding);
    check_all(&doc, "cycle_epbar_st", &cyc.epbar_stations);
    assert!(cyc.yielding.iter().any(|&y| y), "cycle must yield");
    let closure = relative_mises(&cyc.sigma, &cyc.back_stress) - law.flow_stress(0.0, 0.0);
    check_all(&doc, "cycle_closure", &[closure]);

    // Slow rate leg: Cowper-Symonds hardens at any positive rate.
    let dt_slow = de / RATE_SLOW;
    let mut state = Law44State::new();
    let slow = drive_axial(
        &law,
        &mut state,
        &vec![de; STEPS_PER_LEG],
        &stations_100,
        Some(dt_slow),
    );
    check_all(&doc, "slow_sigma", &slow.sigma);
    check_all(&doc, "slow_epbar", &[slow.epbar]);
    check_flags(&doc, "slow_yielding", &slow.yielding);
    check_all(&doc, "slow_epbar_st", &slow.epbar_stations);
    assert!(
        slow.sigma[0] > uni.sigma[0],
        "rate hardening must raise the slow-leg stress"
    );

    // Fast rate leg: the Cowper-Symonds factor roughly doubles the flow.
    let dt_fast = de / RATE_FAST;
    let mut state = Law44State::new();
    let fast = drive_axial(
        &law,
        &mut state,
        &vec![de; STEPS_PER_LEG],
        &stations_100,
        Some(dt_fast),
    );
    check_all(&doc, "fast_sigma", &fast.sigma);
    check_all(&doc, "fast_epbar", &[fast.epbar]);
    check_flags(&doc, "fast_yielding", &fast.yielding);
    check_all(&doc, "fast_epbar_st", &fast.epbar_stations);
    assert!(
        fast.sigma[0] > slow.sigma[0],
        "rate hardening must raise the fast-leg stress above the slow leg"
    );
    assert!(
        fast.epbar < uni.epbar,
        "rate hardening must cut the fast-leg plastic strain"
    );
}
