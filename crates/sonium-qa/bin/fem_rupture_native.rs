//! Phase 7 lane 7C rupture leg: native LAW36 + FAIL/TAB + Wierzbicki.
//!
//! Replays the miniqa `RUPTURE/FAIL_TAB/ELEM_SAMP` uniaxial shells and the
//! `RUPTURE/Wierzbcki` outer-fiber strain paths through
//! [`CoupledPoint`](sonium_fem::physics::solid::fracture::coupled::CoupledPoint)
//! and checks pre-rupture stress plus the erosion stamp against the golden
//! `OFF` flag. exit 0 = all probes pass; exit 1 = FAIL with a report.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::fracture::coupled::{
    CoupledPoint, RuptureCriterion, SolveOptions, TabCriterion, WierzbickiCriterion,
};
use sonium_fem::physics::solid::fracture::fail_tab::FailTab;
use sonium_fem::physics::solid::fracture::wierzbicki::Wierzbicki;
use sonium_fem::physics::solid::plasticity::law_metals::{Law36, Law36State, RateCurve};
use sonium_fem::physics::solid::plasticity::update::StressUpdateModel;
use std::path::PathBuf;

/// Golden schema accepted by this leg (see `goldens/SCHEMA.md`).
const GOLDEN_SCHEMA: &str = "opencourant-golden-v1";

/// Vendor near-zero skip from SCHEMA.md (`Zero_tolerance`).
const ZERO_TOLERANCE: f64 = 5e-14;

/// ELEM_SAMP shell reference volumes in mm^3 (5x5x1, 10x10x1).
const ELEM1_VOLUME: f64 = 25.0;
const ELEM9_VOLUME: f64 = 100.0;

/// ELEM_SAMP formulation band (relative) plus floors (GPa, energy, damage).
///
/// Measured 2026-10-12: the ELEM_SAMP force/energy replays (145 and
/// 240 stamps) pass at a 0.001 band; the 0.002 band keeps 2x headroom
/// for golden-regen channel shift (re-measure after any regen).
const RUPTURE_REL_BAND: f64 = 0.002;
const RUPTURE_STRESS_FLOOR: f64 = 0.03;
const RUPTURE_ENERGY_FLOOR: f64 = 0.1;
const RUPTURE_ENERGY_BAND: f64 = 0.10;

/// Wierzbicki outer-fiber stress band (relative) plus floor (MPa).
const WIERZ_REL_BAND: f64 = 0.03;
const WIERZ_STRESS_FLOOR: f64 = 5.0;

/// Erosion-stamp tolerances in golden stamps (discrete rupture uncertainty).
const ELEM_SAMP_STAMP_TOL: usize = 3;
const WIERZ_STAMP_TOL: usize = 25;

/// Linear EMAX extension past the frozen golden tail (rupture search).
const WIERZ_EXTENSION: usize = 60;

fn goldens_root() -> PathBuf {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    here.join("goldens").join("miniqa")
}

fn load_golden(root: &std::path::Path, case: &str) -> Value {
    let path = root.join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("golden {case} unreadable: {err}"));
    let doc: Value =
        serde_json::from_str(&text).unwrap_or_else(|err| panic!("golden {case} invalid: {err}"));
    assert_eq!(
        doc["schema"].as_str().unwrap_or(""),
        GOLDEN_SCHEMA,
        "golden {case} schema drift"
    );
    doc
}

fn time_grid(doc: &Value) -> Vec<f64> {
    doc["time"]
        .as_array()
        .expect("golden lacks time grid")
        .iter()
        .map(|v| v.as_f64().expect("time must be numeric"))
        .collect()
}

fn channel(doc: &Value, name: &str) -> Vec<f64> {
    doc["channels"][name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks channel {name}"))
        .iter()
        .map(|v| v.as_f64().expect("channel sample must be numeric"))
        .collect()
}

fn channel_tol(doc: &Value, name: &str) -> (f64, f64) {
    let abs = doc["channels"][name]["abs_tol"].as_f64().unwrap_or(1e-12);
    let rel = doc["channels"][name]["rel_tol"].as_f64().unwrap_or(6.6e-4);
    (abs, rel)
}

/// Band comparison: golden tolerance widened by a formulation band.
fn within_band(actual: f64, expected: f64, abs: f64, rel: f64, band: f64, floor: f64) -> bool {
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    if actual.abs() < ZERO_TOLERANCE && expected.abs() < ZERO_TOLERANCE {
        return true;
    }
    let allowance = abs + rel * expected.abs() + band * expected.abs() + floor;
    (actual - expected).abs() <= allowance
}

/// Transcribed ELEM_SAMP LAW36 steel (deck `/MAT/PLAS_TAB/2`).
///
/// Deck units (kg/mm/ms, stress in GPa): `E = 210`, `nu = 0.3`, four
/// rated `FUNCT/14` rows (scales 1/1/1.2/1.2 at 1e-6/1e-5/0.1/1.0 /ms),
/// all strain limits open (`Eps_p_max = Eps_t1 = Eps_t2 = Eps_f = 0`).
fn elem_samp_law() -> Law36 {
    let (x, y) = funct_elem_samp_14();
    let curves = [(1.0e-6, 1.0), (1.0e-5, 1.0), (0.1, 1.2), (1.0, 1.2)]
        .iter()
        .map(|(rate, scale)| {
            RateCurve::new(*rate, x.clone(), y.clone(), *scale).expect("deck curve is valid")
        })
        .collect();
    Law36::new(210.0, 0.3, curves, 0.0, 1.0e20, 1.0e20, 2.0e20, 3.0e20)
}

/// Transcribed Wierzbicki LAW36 (deck `/MAT/PLAS_TAB/59`).
///
/// Deck units (Mg/mm/s, stress in MPa): `E = 150000`, `nu = 0.3`,
/// single `FUNCT/5` row (`Fscale = 0` maps to 1), `Eps_p_max = 3`,
/// `Eps_f = 1`, tensile limits open. Element 11 (`/MAT/36`) adds a
/// `nu = 0` plus `Eps_t/Eps_m` tensile window; the engine holds both
/// elements to the same Wierzbicki rupture strain, so the replay
/// shares this law and notes the window as observed-inactive.
fn wierz_law() -> Law36 {
    let (x, y) = funct_wierz_5();
    let curve = RateCurve::new(0.0, x, y, 1.0).expect("deck curve is valid");
    Law36::new(150000.0, 0.3, vec![curve], 0.0, 3.0, 1.0e20, 2.0e20, 1.0)
}

/// One coupled uniaxial replay along a prescribed axial path.
struct RuptureReplay {
    /// Axial stress per stamp (law units).
    stress_xx: Vec<f64>,
    /// Accumulated damage per stamp.
    damage: Vec<f64>,
    /// Accumulated equivalent plastic strain per stamp.
    ep: Vec<f64>,
    /// Work density per stamp (law energy units).
    energy: Vec<f64>,
    /// First stamp with the erosion latch set.
    erosion_stamp: Option<usize>,
}

/// Drives a coupled point through uniaxial stress along `e1(time)`.
///
/// Each golden step solves the symmetric lateral increment for zero
/// transverse stress (secant, previous-ratio seed) through rate-aware
/// trials, then accepts the step through the coupled point (law
/// commit, damage, erosion latch, work booking).
fn uniaxial_coupled_replay<M, C>(
    law: M,
    state: M::State,
    criterion: C,
    poisson: f64,
    time: &[f64],
    e1: &[f64],
) -> RuptureReplay
where
    M: StressUpdateModel<StrainIncrement = [f64; 6]>,
    M::State: core::fmt::Debug + Clone,
    C: RuptureCriterion,
{
    assert_eq!(time.len(), e1.len());
    let options = SolveOptions::shell();
    let mut point = CoupledPoint::new(law, state, criterion);
    let mut replay = RuptureReplay {
        stress_xx: vec![0.0],
        damage: vec![0.0],
        ep: vec![0.0],
        energy: vec![0.0],
        erosion_stamp: None,
    };
    let mut lat_prev = 0.0f64;
    let mut de11_prev = 0.0f64;
    for i in 1..time.len() {
        let dt = (time[i] - time[i - 1]).max(1e-15);
        let de11 = e1[i] - e1[i - 1];
        let mut lat = if de11_prev.abs() > 1e-15 {
            lat_prev / de11_prev * de11
        } else {
            -poisson * de11
        };
        for _ in 0..25 {
            let trial = if point.law.is_rate_dependent() {
                point
                    .law
                    .update_with_rate(&point.state, &[de11, lat, lat, 0.0, 0.0, 0.0], dt)
            } else {
                point
                    .law
                    .update(&point.state, &[de11, lat, lat, 0.0, 0.0, 0.0])
            };
            let f1 = trial.stress[1];
            let scale = trial.stress[0].abs().max(1.0);
            if f1.abs() < 1e-9 * scale {
                break;
            }
            let h = 1e-7 * de11.abs().max(1e-12);
            let trial_h = if point.law.is_rate_dependent() {
                point.law.update_with_rate(
                    &point.state,
                    &[de11, lat + h, lat + h, 0.0, 0.0, 0.0],
                    dt,
                )
            } else {
                point
                    .law
                    .update(&point.state, &[de11, lat + h, lat + h, 0.0, 0.0, 0.0])
            };
            let slope = (trial_h.stress[1] - f1) / h;
            if slope.abs() < 1e-9 {
                break;
            }
            lat -= f1 / slope;
        }
        let accepted = point.step(&[de11, lat, lat, 0.0, 0.0, 0.0], dt, &options);
        lat_prev = lat;
        de11_prev = de11;
        replay.stress_xx.push(accepted.stress[0]);
        replay.damage.push(accepted.damage);
        replay.ep.push(point.ep);
        replay.energy.push(point.energy_density);
        if accepted.eroded && replay.erosion_stamp.is_none() {
            replay.erosion_stamp = Some(i);
        }
    }
    replay
}

/// First stamp whose `OFF` flag reads deleted (golden rupture stamp).
fn golden_rupture(off: &[f64]) -> usize {
    off.iter()
        .position(|v| *v < 0.5)
        .expect("golden never ruptures")
}

/// Cumulative backward-Euler integral of a rate channel (strain path).
///
/// The TH rate reads end-of-step while the imposed velocity steps at
/// t = 0, so the rectangle rule (not the trapezoid) recovers the
/// displacement-consistent path.
fn integrate_rate(time: &[f64], rate: &[f64], upto: usize) -> Vec<f64> {
    let mut path = vec![0.0; upto + 1];
    for i in 1..=upto {
        path[i] = path[i - 1] + rate[i] * (time[i] - time[i - 1]).max(0.0);
    }
    path
}

fn main() {
    let root = goldens_root();
    let elem = load_golden(&root, "RUPTURE/FAIL_TAB/ELEM_SAMP");
    let wierz = load_golden(&root, "RUPTURE/Wierzbcki/User_law/shell/input_51");
    let mut passed = 0;
    let mut total = 0;

    // R1a: ELEM_SAMP element 1 (5 mm uniaxial-X shell) vs golden TH.
    total += 1;
    let time_e = time_grid(&elem);
    let f1 = channel(&elem, "Default 1 ~1");
    let iem = channel(&elem, "Default 1 ~7");
    let off = channel(&elem, "Default 1 ~9");
    // Shell DEF carries no total E1: EMIN/EMAX (~10/~11) read plastic
    // strain (F76b). The axial TRUE path integrates the EPSD (~12)
    // effective-rate channel, which matches v/L throughout.
    let epsd = channel(&elem, "Default 1 ~12");
    let rupt_gold = golden_rupture(&off);
    let e1path = integrate_rate(&time_e, &epsd, rupt_gold);
    let tab = FailTab::elem_samp_steel();
    let replay = uniaxial_coupled_replay(
        elem_samp_law(),
        Law36State::new(),
        TabCriterion::new(tab),
        0.3,
        &time_e[..=rupt_gold],
        &e1path,
    );
    let (f_abs, f_rel) = channel_tol(&elem, "Default 1 ~1");
    let (e_abs, e_rel) = channel_tol(&elem, "Default 1 ~7");
    let mut r1a = true;
    let edge = rupt_gold.saturating_sub(ELEM_SAMP_STAMP_TOL);
    for i in 1..edge {
        let ok_f = within_band(
            replay.stress_xx[i],
            f1[i],
            f_abs,
            f_rel,
            RUPTURE_REL_BAND,
            RUPTURE_STRESS_FLOOR,
        );
        let ok_e = within_band(
            replay.energy[i] * ELEM1_VOLUME,
            iem[i],
            e_abs,
            e_rel,
            RUPTURE_ENERGY_BAND,
            RUPTURE_ENERGY_FLOOR,
        );
        if !(ok_f && ok_e) && r1a {
            println!(
                "  R1a el1 t {:.4}: F1 {:.4e} vs {:.4e} | IE {:.4e} vs {:.4e} | D {:.4}",
                time_e[i],
                replay.stress_xx[i],
                f1[i],
                replay.energy[i] * ELEM1_VOLUME,
                iem[i],
                replay.damage[i]
            );
        }
        r1a = r1a && ok_f && ok_e;
    }
    let rupt_nat = replay.erosion_stamp.unwrap_or(usize::MAX);
    let stamp_ok = rupt_gold.abs_diff(rupt_nat) <= ELEM_SAMP_STAMP_TOL;
    if !stamp_ok {
        println!("  R1a el1 erosion stamp {rupt_nat} vs golden {rupt_gold}");
    }
    r1a = r1a && stamp_ok;
    println!(
        "PROBE R1a elem-samp-el1: F1 + IE over {} stamps, erosion {} vs {} -> {}",
        edge,
        rupt_nat,
        rupt_gold,
        if r1a { "PASS" } else { "FAIL" }
    );
    passed += usize::from(r1a);

    // R1b: ELEM_SAMP element 9 (10 mm uniaxial-Y shell) vs golden TH.
    total += 1;
    let f2 = channel(&elem, "Default 9 ~2");
    let iem9 = channel(&elem, "Default 9 ~7");
    let off9 = channel(&elem, "Default 9 ~9");
    let epsd9 = channel(&elem, "Default 9 ~12");
    let rupt_gold9 = golden_rupture(&off9);
    let e9path = integrate_rate(&time_e, &epsd9, rupt_gold9);
    let tab9 = FailTab::elem_samp_steel();
    let replay9 = uniaxial_coupled_replay(
        elem_samp_law(),
        Law36State::new(),
        TabCriterion::new(tab9),
        0.3,
        &time_e[..=rupt_gold9],
        &e9path,
    );
    let (f9_abs, f9_rel) = channel_tol(&elem, "Default 9 ~2");
    let (e9_abs, e9_rel) = channel_tol(&elem, "Default 9 ~7");
    let mut r1b = true;
    let edge9 = rupt_gold9.saturating_sub(ELEM_SAMP_STAMP_TOL);
    for i in 1..edge9 {
        let ok_f = within_band(
            replay9.stress_xx[i],
            f2[i],
            f9_abs,
            f9_rel,
            RUPTURE_REL_BAND,
            RUPTURE_STRESS_FLOOR,
        );
        let ok_e = within_band(
            replay9.energy[i] * ELEM9_VOLUME,
            iem9[i],
            e9_abs,
            e9_rel,
            RUPTURE_ENERGY_BAND,
            RUPTURE_ENERGY_FLOOR,
        );
        if !(ok_f && ok_e) && r1b {
            println!(
                "  R1b el9 t {:.4}: F2 {:.4e} vs {:.4e} | IE {:.4e} vs {:.4e} | D {:.4}",
                time_e[i],
                replay9.stress_xx[i],
                f2[i],
                replay9.energy[i] * ELEM9_VOLUME,
                iem9[i],
                replay9.damage[i]
            );
        }
        r1b = r1b && ok_f && ok_e;
    }
    let rupt_nat9 = replay9.erosion_stamp.unwrap_or(usize::MAX);
    let stamp_ok9 = rupt_gold9.abs_diff(rupt_nat9) <= ELEM_SAMP_STAMP_TOL;
    if !stamp_ok9 {
        println!("  R1b el9 erosion stamp {rupt_nat9} vs golden {rupt_gold9}");
    }
    r1b = r1b && stamp_ok9;
    println!(
        "PROBE R1b elem-samp-el9: F2 + IE over {} stamps, erosion {} vs {} -> {}",
        edge9,
        rupt_nat9,
        rupt_gold9,
        if r1b { "PASS" } else { "FAIL" }
    );
    passed += usize::from(r1b);

    // R2a/R2b: Wierzbicki outer-fiber replay (uniaxial strain = EMAX).
    // The bending test ruptures at the outer fiber (~uniaxial stress,
    // triaxiality 1/3, shell Lode 1); the native locus
    // emax(1/3) = 0.87 exp(-1.77/3) = 0.482 meets the golden 0.473 to
    // 2.5%. Damage at the golden OFF stamp must read ~1, and the native
    // erosion stamp (linear EMAX extension past the frozen tail) must
    // land within the locus-band window. Bending resultants stay out:
    // single-point replay cannot integrate the moment.
    for (tag, prefix, stamp_tol) in [
        ("R2a", "New TH 2 1 New TH 2_group_0", WIERZ_STAMP_TOL),
        ("R2b", "New TH 2 11", WIERZ_STAMP_TOL),
    ] {
        total += 1;
        let time_w = time_grid(&wierz);
        let off_w = channel(&wierz, &format!("{prefix} ~9"));
        let emax = channel(&wierz, &format!("{prefix} ~11"));
        let rupt_gold_w = golden_rupture(&off_w);
        // Drop the frozen post-deletion tail and extend the EMAX path
        // linearly past rupture at the pre-rupture rate so the native
        // erosion search can complete.
        let r = rupt_gold_w;
        let rate = (emax[r] - emax[r - 10]) / (time_w[r] - time_w[r - 10]).max(1e-15);
        let dt_tail = (time_w[r] - time_w[r - 1]).max(1e-15);
        let mut time_x = time_w[..=r].to_vec();
        let mut emax_x = emax[..=r].to_vec();
        for k in 1..=WIERZ_EXTENSION {
            time_x.push(time_w[r] + dt_tail * k as f64);
            emax_x.push(emax[r] + rate * dt_tail * k as f64);
        }
        let locus = Wierzbicki::deck_shell();
        let replay_w = uniaxial_coupled_replay(
            wierz_law(),
            Law36State::new(),
            WierzbickiCriterion::new(locus),
            0.3,
            &time_x,
            &emax_x,
        );
        let mut ok = true;
        // Damage reads ~1 at the golden rupture stamp (locus proof).
        let d_rupt = replay_w.damage[rupt_gold_w];
        let damage_ok = (d_rupt - 1.0).abs() <= 0.10;
        if !damage_ok {
            println!("  {tag} damage at golden rupture: {d_rupt:.4}");
        }
        ok = ok && damage_ok;
        // Erosion stamp lands in the locus-band window.
        let rupt_nat_w = replay_w.erosion_stamp.unwrap_or(usize::MAX);
        let stamp_ok_w = rupt_gold_w.abs_diff(rupt_nat_w) <= stamp_tol;
        if !stamp_ok_w {
            println!("  {tag} erosion stamp {rupt_nat_w} vs golden {rupt_gold_w}");
        }
        ok = ok && stamp_ok_w;
        // Outer-fiber stress tracks the hardening curve pre-rupture.
        let law_ref = wierz_law();
        let (s_abs, s_rel) = channel_tol(&wierz, &format!("{prefix} ~11"));
        let edge_w = rupt_gold_w.saturating_sub(25);
        for i in (1..edge_w).step_by(7) {
            let ep = replay_w.ep[i];
            let hardening = law_ref.nominal_yield(ep.max(0.0), 0.0);
            // Pre-yield the fiber reads the elastic line, not the curve.
            let want = hardening.min(150000.0 * emax_x[i]);
            let ok_s = within_band(
                replay_w.stress_xx[i],
                want,
                s_abs,
                s_rel,
                WIERZ_REL_BAND,
                WIERZ_STRESS_FLOOR,
            );
            if !ok_s && ok {
                println!(
                    "  {tag} t {:.6}: fiber stress {:.4e} vs curve {:.4e} (ep {ep:.4})",
                    time_w[i], replay_w.stress_xx[i], want
                );
            }
            ok = ok && ok_s;
        }
        println!(
            "PROBE {tag} wierz-{prefix}: D(rupt) {d_rupt:.4}, erosion {rupt_nat_w} vs {rupt_gold_w} -> {}",
            if ok { "PASS" } else { "FAIL" }
        );
        passed += usize::from(ok);
    }

    if passed == total {
        println!("PASS: qa-fem-rupture-native ({passed}/{total})");
        println!("SUCCESS: qa-fem-rupture-native");
    } else {
        println!("FAIL: qa-fem-rupture-native ({passed}/{total})");
        println!("FAILED: qa-fem-rupture-native");
        std::process::exit(1);
    }
}

/// Deck FUNCT/14 steel hardening (machine-transcribed, do not hand-edit).
fn funct_elem_samp_14() -> (Vec<f64>, Vec<f64>) {
    let x = vec![
        0.0, 0.00112, 0.00218, 0.003, 0.00404, 0.00517, 0.00613, 0.0071, 0.00806, 0.00901, 0.0102,
        0.0121, 0.013, 0.014, 0.015, 0.0159, 0.0171, 0.0181, 0.0204, 0.0303, 0.0405, 0.0502, 0.06,
        0.0702, 0.0807, 0.09, 0.0997, 0.101, 0.11, 0.15000001, 0.2, 0.30000001, 0.5, 1.0,
    ];
    let y = vec![
        0.306, 0.415, 0.445, 0.461, 0.474, 0.489, 0.498, 0.505, 0.512, 0.522, 0.53, 0.543, 0.55,
        0.555, 0.561, 0.567, 0.572, 0.577, 0.592, 0.632, 0.663, 0.687, 0.706, 0.722, 0.737, 0.749,
        0.758, 0.759, 0.768, 0.805, 0.84, 0.9, 1.0, 1.21,
    ];
    (x, y)
}

/// Deck FUNCT/5 hardening (machine-transcribed, do not hand-edit).
fn funct_wierz_5() -> (Vec<f64>, Vec<f64>) {
    let x = vec![
        0.0, 0.001048, 0.004507, 0.016289, 0.02891, 0.041452, 0.05345, 0.065447, 0.07663, 0.088459,
        0.099323, 0.110154, 0.12117, 0.131963, 0.143266, 0.153501, 0.172727, 0.181857, 0.183558,
        0.185272,
    ];
    let y = vec![
        260.47, 285.474, 297.373, 326.307, 354.258, 377.438, 396.082, 412.1, 425.068, 437.155,
        446.97, 455.8, 463.91, 471.204, 478.244, 484.115, 494.108, 498.605, 499.394, 500.216,
    ];
    (x, y)
}
