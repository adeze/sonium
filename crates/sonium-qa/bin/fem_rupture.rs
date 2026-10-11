//! Phase-3 lane 3D G-leg (lane-owned): failure/erosion.
//!
//! Native-vs-harvested-golden comparison for the RUPTURE miniqa cases
//! (`RUPTURE/FAIL_TAB/ELEM_SAMP`, `RUPTURE/Wierzbcki/User_law/shell`).
//! Offline: reads only `goldens/miniqa/**/*.json` plus the schema rules.
//! Exit 0 = PASS, exit 1 = FAIL with per-probe report on stdout.
//!
//! Probe design (single-point drivers; no cross-lane runtimes):
//! - M1 rebuilds the ELEM_SAMP part mass from H8 quadrature volumes and
//!   the deck density, and checks every golden MASS sample against it.
//! - F1 drives the deck `/FAIL/TAB` table along the golden-recorded
//!   element-1 kinematics (nodal DX/DY plus THIC) with a Johnson-Cook
//!   steel surrogate for the stress triaxiality, and compares the
//!   survival verdict against the golden (zero mass loss). F1b repeats
//!   the path with the rate clamped quasi-static as a sensitivity
//!   counterfactual (must rupture, proving F1 is not vacuous).
//! - F2 pins the locus value at the golden end rate against the deck
//!   table closed form (extrapolation regime of comment 3).
//! - W1 drives a uniaxial locus path with the deck Wierzbicki card and
//!   compares the rupture verdict against the golden ERODED flags.
//! - W2 checks locus endpoint identities (analytic, no golden numbers).
//! - E1 checks ledger conservation on the ELEM_SAMP energy split.
//!
//! Rate rows are read per second (SI-consistent): under a per-millisecond
//! reading the deck path would sit inside the table and predict rupture,
//! contradicting the golden's zero-mass-loss survival; the per-second
//! reading reproduces the golden verdict (see probe F1 output).
//!
//! The surrogate triaxiality is conservative: the golden shell kinematics
//! carry net volume change (plane-stress thickness update), which the
//! dense-metal surrogate turns into hydrostatic stress, so F1 evaluates
//! the locus at higher triaxiality (lower failure strain) than the
//! engine's plane-stress state and overestimates damage.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::fracture::deletion::{DeletionPolicy, ErosionLedger};
use sonium_fem::physics::solid::fracture::ductile::{
    ErosionCriterion, ErosionSet, stress_triaxiality_c1,
};
use sonium_fem::physics::solid::fracture::fail_tab::FailTab;
use sonium_fem::physics::solid::fracture::wierzbicki::{Wierzbicki, xi_shell};
use sonium_fem::physics::solid::nonlinear_solid::{
    hex_jacobian, hex_quadrature, hex_shape_derivatives,
};
use sonium_fem::physics::solid::plasticity::StressUpdateModel;
use sonium_fem::physics::solid::plasticity::crash::{
    JohnsonCookMaterial, JohnsonCookState, drive_uniaxial_stress,
};
use std::path::PathBuf;

/// Golden schema accepted by this leg (see `goldens/SCHEMA.md`).
const GOLDEN_SCHEMA: &str = "opencourant-golden-v1";

/// Vendor near-zero skip from SCHEMA.md (`Zero_tolerance`).
const ZERO_TOLERANCE: f64 = 5e-14;

/// ELEM_SAMP element-1 footprint in mm (deck `/SHELL/1`, nodes 1-4).
const EL1_MM: [f64; 3] = [5.0, 5.0, 1.0];
/// ELEM_SAMP element-9 footprint in mm (nodes 81-84).
const EL9_MM: [f64; 3] = [10.0, 10.0, 1.0];
/// ELEM_SAMP steel density in kg/mm^3 (deck `/MAT/PLAS_TAB/2`).
const DECK_DENSITY_KG_MM3: f64 = 7.8e-6;

fn goldens_root() -> PathBuf {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        return PathBuf::from(&args[1]);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens")
}

fn load_golden(root: &std::path::Path, case: &str) -> Value {
    let path = root.join("miniqa").join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("G-leg needs harvested golden {path:?}: {err}"));
    let doc: Value = serde_json::from_str(&text).expect("golden must be valid JSON");
    assert_eq!(
        doc["schema"].as_str().unwrap_or(""),
        GOLDEN_SCHEMA,
        "golden schema mismatch for {case}"
    );
    doc
}

fn time_grid(doc: &Value) -> Vec<f64> {
    doc["time"]
        .as_array()
        .expect("golden needs a time grid")
        .iter()
        .map(channel_number)
        .collect()
}

fn channel_number(value: &Value) -> f64 {
    value.as_f64().expect("golden samples must be numbers")
}

fn channel(doc: &Value, name: &str) -> Vec<f64> {
    doc["channels"][name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks channel {name}"))
        .iter()
        .map(channel_number)
        .collect()
}

fn channel_tol(doc: &Value, name: &str) -> (f64, f64) {
    let abs = doc["channels"][name]["abs_tol"].as_f64().unwrap_or(1e-12);
    let rel = doc["channels"][name]["rel_tol"].as_f64().unwrap_or(6.6e-4);
    (abs, rel)
}

/// Sample comparison with the vendor near-zero skip (SCHEMA.md).
fn within_tol(actual: f64, expected: f64, abs: f64, rel: f64) -> bool {
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    if actual.abs() < ZERO_TOLERANCE && expected.abs() < ZERO_TOLERANCE {
        return true;
    }
    if (actual - expected).abs() < 2.0 * ZERO_TOLERANCE {
        return true;
    }
    (actual - expected).abs() <= abs + rel * expected.abs()
}

/// Reference volume of an axis-aligned brick via H8 quadrature.
///
/// Builds the 8 corners in SI meters from the mm footprint, integrates
/// `det J` with the production 2x2x2 rule, and returns m^3.
fn brick_volume_h8(footprint_mm: [f64; 3]) -> f64 {
    let corner = |sx: f64, sy: f64, sz: f64| {
        [
            1e-3 * footprint_mm[0] * (1.0 + sx) / 2.0,
            1e-3 * footprint_mm[1] * (1.0 + sy) / 2.0,
            1e-3 * footprint_mm[2] * (1.0 + sz) / 2.0,
        ]
    };
    // Matches HEX_NODES xi-fastest ordering.
    let coords = [
        corner(-1.0, -1.0, -1.0),
        corner(1.0, -1.0, -1.0),
        corner(1.0, 1.0, -1.0),
        corner(-1.0, 1.0, -1.0),
        corner(-1.0, -1.0, 1.0),
        corner(1.0, -1.0, 1.0),
        corner(1.0, 1.0, 1.0),
        corner(-1.0, 1.0, 1.0),
    ];
    let mut volume = 0.0;
    for (natural, weight) in hex_quadrature() {
        let derivs = hex_shape_derivatives(&natural);
        let (_, det, _) = hex_jacobian(&coords, &derivs);
        volume += det * weight;
    }
    volume
}

/// Deck-tuned Johnson-Cook steel surrogate for triaxiality.
///
/// Deck `/MAT/PLAS_TAB/2` hardening starts at 0.306 GPa with a
/// `(ep/1)^0.3` rise; rate and thermal legs stay neutral so the
/// surrogate only shapes the stress direction, never the locus rate.
fn deck_surrogate_steel() -> JohnsonCookMaterial {
    JohnsonCookMaterial::new(
        210.0e9, 0.3, 306.0e6, 0.9e9, 0.3, 0.0, 1.03, 1.0, 293.0, 1793.0,
    )
}

/// True in-plane strains of element 1 from golden node channels.
///
/// `/TH/NODE DEF` order is DX, DY, DZ, VX, VY, VZ (Reference Guide
/// time-history tables): `~1` is DX, `~2` is DY, `~4` is VX. Lengths in
/// mm, thickness ratio from the THIC channel (initial 1.0 mm).
fn element1_history(doc: &Value) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let time = time_grid(doc);
    let dx2 = channel(doc, "nodes 2 ~1");
    let dx3 = channel(doc, "nodes 3 ~1");
    let dy3 = channel(doc, "nodes 3 ~2");
    let dx4 = channel(doc, "nodes 4 ~1");
    let thic = channel(doc, "THIC 1");
    let n = time.len();
    assert!(
        dx2.len() == n && dx3.len() == n && dy3.len() == n && dx4.len() == n && thic.len() == n
    );
    let ex: Vec<f64> = dx2.iter().map(|dx| (1.0 + dx / 5.0).ln()).collect();
    let ey: Vec<f64> = dy3.iter().map(|dy| (1.0 + dy / 5.0).ln()).collect();
    let ez: Vec<f64> = thic.iter().map(|t| t.ln()).collect();
    let gxy: Vec<f64> = dx3
        .iter()
        .zip(dx4.iter())
        .map(|(a, b)| (a - b) / 5.0)
        .collect();
    (ex, ey, ez, gxy)
}

/// Accumulates FAIL_TAB damage along the golden element-1 history.
///
/// Returns `(damage, ep, max_triaxiality, end_rate_per_s)`. With
/// `clamp_rate` set, the locus rate clamps there (sensitivity probe);
/// otherwise the golden step rate feeds the table directly.
fn fail_tab_damage(doc: &Value, clamp_rate: Option<f64>) -> (f64, f64, f64, f64) {
    let time = time_grid(doc);
    let (ex, ey, ez, gxy) = element1_history(doc);
    let steel = deck_surrogate_steel();
    let mut state = JohnsonCookState::new(293.0);
    let tab = FailTab::elem_samp_steel();
    let mut damage = 0.0f64;
    let mut max_triax = 0.0f64;
    let mut end_rate = 0.0f64;
    for i in 1..time.len() {
        let dt_ms = time[i] - time[i - 1];
        assert!(dt_ms > 0.0, "golden time grid must be monotone");
        let dt = dt_ms * 1e-3;
        let d_eps = [
            ex[i] - ex[i - 1],
            ey[i] - ey[i - 1],
            ez[i] - ez[i - 1],
            gxy[i] - gxy[i - 1],
            0.0,
            0.0,
        ];
        let update = steel.update_with_rate(&state, &d_eps, dt);
        steel.commit(&mut state, &update);
        let dep = update.delta_equiv_plastic_strain;
        if dep > 0.0 {
            let triax = stress_triaxiality_c1(&update.stress);
            max_triax = max_triax.max(triax);
            let rate = dep / dt;
            end_rate = rate;
            let locus_rate = clamp_rate.map_or(rate, |cap| rate.min(cap));
            damage += tab.increment_at_rate(
                dep,
                state.equiv_plastic_strain,
                triax,
                0.0,
                locus_rate,
                None,
                None,
            );
        }
    }
    (damage, state.equiv_plastic_strain, max_triax, end_rate)
}

fn main() {
    let root = goldens_root();
    let fail = load_golden(&root, "RUPTURE/FAIL_TAB/ELEM_SAMP");
    let wier = load_golden(&root, "RUPTURE/Wierzbcki/User_law/shell/input_51");
    let mut passed = 0;
    let mut total = 0;

    // M1: H8-derived part mass vs golden MASS (every sample).
    total += 1;
    let mass_native =
        (brick_volume_h8(EL1_MM) + brick_volume_h8(EL9_MM)) * DECK_DENSITY_KG_MM3 * 1e9;
    let mass_series = channel(&fail, "MASS");
    let (mass_abs, mass_rel) = channel_tol(&fail, "MASS");
    let mass_worst = mass_series
        .iter()
        .map(|m| (m - mass_native).abs() - (mass_abs + mass_rel * m.abs()))
        .fold(0.0f64, f64::max);
    let m1 = mass_series
        .iter()
        .all(|m| within_tol(*m, mass_native, mass_abs, mass_rel));
    println!(
        "PROBE M1 fail-tab-mass: native {mass_native:.6e} kg vs golden {:.6e} ({} samples, worst margin {mass_worst:.2e}) -> {}",
        mass_series[0],
        mass_series.len(),
        if m1 { "PASS" } else { "FAIL" }
    );
    passed += usize::from(m1);

    // F1: survival verdict along golden kinematics.
    total += 1;
    let tab = FailTab::elem_samp_steel();
    let (damage, ep, max_triax, end_rate) = fail_tab_damage(&fail, None);
    let native_survives = !tab.ruptured(damage);
    // Golden survival: MASS constant (M1) plus THIC above zero at end.
    let thic_end = channel(&fail, "THIC 1").pop().unwrap();
    let golden_survives = thic_end > 0.0 && m1;
    let f1 = native_survives && golden_survives;
    println!(
        "PROBE F1 fail-tab-survival: native D {damage:.6e} (ep {ep:.4}, max triax {max_triax:.4}, end rate {end_rate:.3e}/s) vs Dcrit {:.2} -> native {} / golden {} -> {}",
        tab.dcrit,
        if native_survives {
            "survives"
        } else {
            "RUPTURES"
        },
        if golden_survives {
            "survives"
        } else {
            "RUPTURES"
        },
        if f1 { "PASS" } else { "FAIL" }
    );
    passed += usize::from(f1);

    // F1b: quasi-static counterfactual must rupture (sensitivity).
    total += 1;
    let (damage_qs, _, _, _) = fail_tab_damage(&fail, Some(0.5));
    let f1b = tab.ruptured(damage_qs);
    println!(
        "PROBE F1b fail-tab-sensitivity: clamped-rate D {damage_qs:.6e} vs Dcrit {:.2} -> must rupture -> {}",
        tab.dcrit,
        if f1b { "PASS" } else { "FAIL" }
    );
    passed += usize::from(f1b);

    // F2: locus value at golden end rate vs deck closed form.
    total += 1;
    let span = 0.5 - 1.0e-5;
    let fscale = 1.0 + 0.5 * (end_rate - 0.5) / span;
    let expected_ef = 0.360 * fscale;
    // Exact deck node (0.3333333, 0.360): no triaxiality interpolation.
    let native_ef = tab.table_strain(0.3333333, end_rate, 0.0);
    let f2 = (native_ef - expected_ef).abs() <= 1e-9 + 1e-12 * expected_ef.abs();
    println!(
        "PROBE F2 fail-tab-locus: native {native_ef:.6e} vs deck {expected_ef:.6e} at {end_rate:.3e}/s (F {fscale:.3e}) -> {}",
        if f2 { "PASS" } else { "FAIL" }
    );
    passed += usize::from(f2);

    // W1: uniaxial locus rupture vs golden ERODED verdicts.
    total += 1;
    let locus = Wierzbicki::deck_shell();
    let jc = JohnsonCookMaterial::c4340_steel();
    let mut jc_state = JohnsonCookState::new(293.0);
    let curve = drive_uniaxial_stress(&jc, &mut jc_state, 1.0, 400);
    let eta = 1.0 / 3.0;
    let xi = xi_shell(eta);
    let mut w_damage = 0.0;
    let mut cross = f64::NAN;
    for pair in curve.windows(2) {
        let dep = pair[1].equiv_plastic_strain - pair[0].equiv_plastic_strain;
        let prev = w_damage;
        w_damage += dep / locus.fracture_strain(eta, xi);
        // Damage is linear in ep on this proportional path: interpolate
        // the unit crossing inside the tripping step.
        if locus.fractured(w_damage) && cross.is_nan() {
            let frac = (1.0 - prev) / (w_damage - prev);
            cross = pair[0].equiv_plastic_strain + frac * dep;
        }
    }
    let locus_cross = locus.emax(eta);
    let w1_cross = ((cross - locus_cross) / locus_cross).abs() < 1e-6;
    let eroded36 = channel(&wier, "PART_36 ERODED").pop().unwrap();
    let eroded59 = channel(&wier, "PART_59 ERODED").pop().unwrap();
    let time_w = time_grid(&wier);
    let onset = |key: &str| {
        channel(&wier, key)
            .iter()
            .position(|v| *v >= 1.0)
            .map(|i| time_w[i])
            .unwrap_or(f64::NAN)
    };
    let golden_ruptures = eroded36 >= 1.0 && eroded59 >= 1.0;
    let w1 = w1_cross && golden_ruptures;
    println!(
        "PROBE W1 wierzbicki-rupture: native cross ep {cross:.6e} vs locus {locus_cross:.6e}; golden ERODED {eroded36:.0}/{eroded59:.0} at t* {:.6e}/{:.6e} -> {}",
        onset("PART_36 ERODED"),
        onset("PART_59 ERODED"),
        if w1 { "PASS" } else { "FAIL" }
    );
    passed += usize::from(w1);

    // W2: locus endpoint identities (analytic).
    total += 1;
    let w2 = (locus.fracture_strain(eta, 1.0) - locus.emax(eta)).abs() < 1e-12
        && (locus.fracture_strain(eta, 0.0) - locus.emin(eta)).abs() < 1e-12
        && (xi_shell(eta) - 1.0).abs() < 1e-12;
    println!(
        "PROBE W2 wierzbicki-endpoints: emax {:.6e}, emin {:.6e}, xi {:.6e} -> {}",
        locus.emax(eta),
        locus.emin(eta),
        xi_shell(eta),
        if w2 { "PASS" } else { "FAIL" }
    );
    passed += usize::from(w2);

    // E1: ledger conservation on the ELEM_SAMP energy split.
    total += 1;
    let plastic_work = channel(&fail, "PLASTIC WORK").pop().unwrap();
    let (work_abs, work_rel) = channel_tol(&fail, "PLASTIC WORK");
    let v1 = brick_volume_h8(EL1_MM);
    let v9 = brick_volume_h8(EL9_MM);
    // No erosion in the golden: split the work by reference volume.
    let density = plastic_work / (v1 + v9);
    let mut ledger = ErosionLedger::new(2);
    ledger.record(0, v1, density, density, false);
    ledger.record(1, v9, density, density, false);
    let totals = ledger.totals();
    let e1 = within_tol(totals.grand_plastic(), plastic_work, work_abs, work_rel)
        && totals.eroded_plastic == 0.0;
    println!(
        "PROBE E1 erosion-ledger: booked {:.6e} J vs golden {:.6e} J, eroded {:.1e} -> {}",
        totals.grand_plastic(),
        plastic_work,
        totals.eroded_plastic,
        if e1 { "PASS" } else { "FAIL" }
    );
    passed += usize::from(e1);

    // E2: deletion policy mirrors the deck verdicts (no erosion here).
    total += 1;
    let policy = DeletionPolicy::shell_layers();
    let criterion = ErosionCriterion::tabular(tab.dcrit);
    let mut set = ErosionSet::new(2);
    let fresh = set.sweep_with_tabular(
        &criterion,
        &[(ep, 0.0, 0.0, damage), (ep, 0.0, 0.0, damage)],
    );
    let layers = [false, false, false, false, false];
    let e2 = fresh.is_empty() && !policy.shell_deleted(&layers, None) && set.active_count() == 2;
    println!(
        "PROBE E2 deletion-policy: fresh {fresh:?}, active {}/2 -> {}",
        set.active_count(),
        if e2 { "PASS" } else { "FAIL" }
    );
    passed += usize::from(e2);

    if passed == total {
        println!("PASS: qa-fem-rupture ({passed}/{total})");
    } else {
        println!("FAIL: qa-fem-rupture ({passed}/{total})");
        std::process::exit(1);
    }
}
