//! Wolfram golden replay: LAW62 hyper-viscoelastic strain paths.
//!
//! Replays `fixtures/solids_law62_paths_golden.json` (derived by
//! `fixtures/solids_law62_paths.wls` at 100-digit working precision,
//! exported at 50 digits) through the production LAW62 API. Regenerate
//! with:
//! `wolframscript -script fixtures/solids_law62_paths.wls > fixtures/solids_law62_paths_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::law_metals::{Law62, Law62State};
use sonium_fem::physics::solid::plasticity::update::{StressUpdateModel, von_mises_c1};

const GOLDEN: &str = include_str!("../fixtures/solids_law62_paths_golden.json");

/// Steps per leg, matching the derivation script.
const STEPS_PER_LEG: usize = 100;
/// Lateral bisection passes per axial step, matching the derivation.
const LATERAL_ITERATIONS: usize = 50;
/// Step time in s: beta = dt / tau = 0.1 exercises the Prony branch.
const DT: f64 = 1.0e-3;
/// Axial/shear extent of the deviatoric paths: 0.2 Hencky strain is
/// well into the Ogden stiffening regime.
const EMAX: f64 = 0.2;
/// Volumetric extent per axis: 0.01 Hencky keeps the
/// near-incompressible bulk response finite.
const EVMAX: f64 = 0.01;

/// Accumulated result of one driven strain path.
struct PathResult {
    sigma: [f64; 6],
    yielding: Vec<bool>,
    epbar_stations: Vec<f64>,
    hencky: [f64; 6],
    branch: [f64; 6],
    branch_stations: Vec<f64>,
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

/// Solves the lateral strain of one axial step for zero lateral stress.
///
/// Bisects the pure production `update_with_rate` (50 passes over a
/// widening bracket), mirroring the derivation script exactly.
fn solve_lateral(law: &Law62, state: &Law62State, de: f64) -> f64 {
    let probe = |dl: f64| {
        law.update_with_rate(state, &[de, dl, dl, 0.0, 0.0, 0.0], DT)
            .stress[1]
    };
    let (mut lo, mut hi) = if de >= 0.0 { (-de, 0.0) } else { (0.0, -de) };
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

/// Records one station of a hyper-viscoelastic path.
///
/// LAW62 carries no plastic accumulator, so the station epbar is the
/// step increment itself (always zero, pinned exactly like the flags).
fn record(
    state: &Law62State,
    update: &sonium_fem::physics::solid::plasticity::update::StressUpdate,
    yielding: &mut Vec<bool>,
    epbar_stations: &mut Vec<f64>,
    branch_stations: &mut Vec<f64>,
) {
    yielding.push(update.yielding);
    epbar_stations.push(update.delta_equiv_plastic_strain);
    branch_stations.push(von_mises_c1(&state.branches[0]));
}

/// Drives fixed strain increments, recording stations (1-based steps).
fn drive_fixed(
    law: &Law62,
    state: &mut Law62State,
    steps: &[[f64; 6]],
    stations: &[usize],
) -> PathResult {
    let mut yielding = Vec::new();
    let mut epbar_stations = Vec::new();
    let mut branch_stations = Vec::new();
    for (i, step) in steps.iter().enumerate() {
        let update = law.update_with_rate(state, step, DT);
        law.commit(state, &update);
        if stations.contains(&(i + 1)) {
            record(
                state,
                &update,
                &mut yielding,
                &mut epbar_stations,
                &mut branch_stations,
            );
        }
    }
    PathResult {
        sigma: state.stress,
        yielding,
        epbar_stations,
        hencky: state.hencky,
        branch: state.branches[0],
        branch_stations,
    }
}

/// Drives axial increments with the lateral solve, recording stations.
fn drive_axial(law: &Law62, state: &mut Law62State, des: &[f64], stations: &[usize]) -> PathResult {
    let mut yielding = Vec::new();
    let mut epbar_stations = Vec::new();
    let mut branch_stations = Vec::new();
    for (i, &de) in des.iter().enumerate() {
        let dl = solve_lateral(law, state, de);
        let d = [de, dl, dl, 0.0, 0.0, 0.0];
        let update = law.update_with_rate(state, &d, DT);
        law.commit(state, &update);
        if stations.contains(&(i + 1)) {
            record(
                state,
                &update,
                &mut yielding,
                &mut epbar_stations,
                &mut branch_stations,
            );
        }
    }
    PathResult {
        sigma: state.stress,
        yielding,
        epbar_stations,
        hencky: state.hencky,
        branch: state.branches[0],
        branch_stations,
    }
}

#[test]
fn law62_paths_match_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["units"], "SI");
    let law = Law62::rubber_prony();
    assert!(law.is_rate_dependent());
    assert_eq!(law.prony.len(), 1);
    let stations_100 = vec![20, 40, 60, 80, 100];
    let stations_cycle = vec![80, 160, 240, 320, 400];

    // Uniaxial extension with solved lateral strain.
    let de = EMAX / STEPS_PER_LEG as f64;
    let mut state = Law62State::new(law.prony.len());
    let uni = drive_axial(&law, &mut state, &vec![de; STEPS_PER_LEG], &stations_100);
    check_all(&doc, "uni_sigma", &uni.sigma);
    check_flags(&doc, "uni_yielding", &uni.yielding);
    check_all(&doc, "uni_epbar_st", &uni.epbar_stations);
    check_all(&doc, "uni_hencky", &uni.hencky);
    check_all(&doc, "uni_branch", &uni.branch);
    check_all(&doc, "uni_branch_st", &uni.branch_stations);
    assert!(
        uni.yielding.iter().all(|&y| !y),
        "hyper-viscoelasticity never yields"
    );
    assert!(
        uni.branch.iter().any(|&b| b != 0.0),
        "the Prony branch must carry overstress"
    );
    assert!(
        uni.sigma[0] > 0.0 && uni.hencky[1] < 0.0,
        "uniaxial tension must pull in the sides"
    );

    // Pure shear ramp (pins the Poynting normal stresses).
    let dg = EMAX / STEPS_PER_LEG as f64;
    let mut state = Law62State::new(law.prony.len());
    let shear = drive_fixed(
        &law,
        &mut state,
        &vec![[0.0, 0.0, 0.0, dg, 0.0, 0.0]; STEPS_PER_LEG],
        &stations_100,
    );
    check_all(&doc, "shear_sigma", &shear.sigma);
    check_flags(&doc, "shear_yielding", &shear.yielding);
    check_all(&doc, "shear_epbar_st", &shear.epbar_stations);
    assert!(
        shear.sigma[0] > 0.0,
        "Ogden shear must develop Poynting stresses"
    );

    // Small volumetric ramp: the near-incompressible bulk response.
    let dv = EVMAX / STEPS_PER_LEG as f64;
    let mut state = Law62State::new(law.prony.len());
    let vol = drive_fixed(
        &law,
        &mut state,
        &vec![[dv, dv, dv, 0.0, 0.0, 0.0]; STEPS_PER_LEG],
        &stations_100,
    );
    check_all(&doc, "vol_sigma", &vol.sigma);
    check_flags(&doc, "vol_yielding", &vol.yielding);
    check_all(&doc, "vol_epbar_st", &vol.epbar_stations);
    assert!(vol.sigma[0] > uni.sigma[0], "bulk must be stiffer");

    // Symmetric strain cycle with solved lateral strain.
    let mut des = vec![de; STEPS_PER_LEG];
    des.extend(vec![-de; 2 * STEPS_PER_LEG]);
    des.extend(vec![de; STEPS_PER_LEG]);
    let mut state = Law62State::new(law.prony.len());
    let cyc = drive_axial(&law, &mut state, &des, &stations_cycle);
    check_all(&doc, "cycle_sigma", &cyc.sigma);
    check_flags(&doc, "cycle_yielding", &cyc.yielding);
    check_all(&doc, "cycle_epbar_st", &cyc.epbar_stations);
    check_all(&doc, "cycle_closure", &[von_mises_c1(&cyc.sigma)]);
    assert!(
        von_mises_c1(&cyc.sigma) > 0.0,
        "branch memory must leave a residual stress"
    );
}
