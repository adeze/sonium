//! Phase-3 lane 3C G-leg (lane-owned): foam/extended laws.
//!
//! Native-vs-harvested-golden comparison for the LOIS foam-law miniqa
//! cases (`LOIS/FOAM/FOAM429/input_100/isolid_101.json`,
//! `LOIS/LOI50/test1/pid_6/input_44.json`,
//! `LOIS/LOI51/JWL/01-PG_degenerated/01a_P-E.json` and `01b_dP-E.json`,
//! `LOIS/LOI76/shell/test1/compression.json`). Offline: reads only
//! `goldens/miniqa/**/*.json` plus the `SCHEMA.md` contract compiled
//! into the probes below. No engine, no network.
//!
//! Each probe recomputes a quantity natively from the harvested raw
//! series and compares it against the harvested summary or a physics
//! identity, with tolerances justified per probe:
//!
//! - `schema`: `schema` tag equals `opencourant-golden-v1` (SCHEMA.md).
//! - `termination`: engine run terminated normally.
//! - `grid`: time grid strictly increasing with at least 2 samples.
//! - `decimation`: kept-row record matches the series length, or the
//!   record is null (no decimation).
//! - `shapes`: every channel has exactly N finite samples.
//! - `summary`: natively recomputed final/min/max match the harvested
//!   summaries within each channel's own `abs_tol`/`rel_tol`.
//! - `energy`: `KE + IE + HE + RE + CE - EXTW` reproduces the total
//!   `NO NAME` channel within 1e-6 scale-relative (7-digit TH rounding;
//!   observed worst 3.1e-7 on all five cases). Scale-relative, not
//!   `NO NAME`-relative, because `NO NAME` crosses zero.
//! - `momentum`: `|p|^2 <= 2 m KE` at every sample within 1e-6
//!   (Cauchy-Schwarz over the mass distribution; exact physics).
//! - `mass`: MASS spread within 1e-9 absolute (observed exactly 0);
//!   cycle TOTALMASS within 0.1% of the last MASS sample.
//! - `cycle`: MASERR within 1e-12 of zero; TSTEP, CYCLE, TIME
//!   positive. ERROR (energy-error percent) is skipped: meaningless
//!   on micro-cases. TIME-vs-grid-end matching is skipped: TH cadence
//!   trails the cycle cadence by a case-dependent 0.2-7%.
//! - `law33_bound` (FOAM): principal stresses of `S + P_air I` stay
//!   under the yield `Y(gamma)` of Theory Manual equations 1024-1027
//!   within 1% + 1e-9 on all 20 TH solids x 122 samples (observed 0
//!   violations, worst ratio 1.0000006: the bound is active in crush
//!   as expected; the margin covers TH rounding). Deck constants (P0,
//!   porosity, A, B, C) are compiled research facts from
//!   `FOAM_V10_0000.rad`; the runtime reads only the golden JSON.
//! - `foam_part` (FOAM): part `FOAM LAW 33` IE and HE equal the global
//!   channels (single foam part) within golden tolerances; ERODED is
//!   all zero. KE/MASS are skipped: the model carries other parts.
//! - `foam_off` (FOAM): every solid `~1` activity flag equals 1.0.
//! - `solide_ie` (LOI50): element `~8` equals global INTERNAL ENERGY
//!   within golden tolerances (single-element case).
//! - `solide_flag` (LOI50): `~1` equals 1.0 on all samples.
//! - `vol_positive` (LOI50): `~9` (density) stays positive.
//! - `plastic_zero` (LOI50): PLASTIC WORK stays zero (elastic case).
//! - `hydrostatic` (LOI51): brick `~2 == ~3 == ~4` within golden
//!   tolerances and `~5..~7` zero (hydrostatic JWL brick).
//! - `pe_cross` (LOI51): 01a `P - P0` versus 01b `dP` on the shared
//!   energy grid within 1.0 + 1e-5 relative (observed 0.03 absolute;
//!   independent runs plus TH rounding plus interpolation).
//! - `trajectory` (LOI51): the two runs' energy grids agree within
//!   5e-3 relative (observed 1.5e-3; run-to-run cadence jitter).
//! - `brick_flag` (LOI51): `~1` and `~12` equal 1.0 on all samples.
//! - `spec_ie` (LOI76): element `~7` equals global INTERNAL ENERGY.
//! - `compression` (LOI76): `~1` stays non-positive, ending negative.
//! - `top_bottom` (LOI76): `~10` equals `~11` (symmetric faces).
//! - `plastic` (LOI76): PLASTIC WORK is non-decreasing, positive at
//!   the end, and below total internal energy (SAMP yields).
//! - `spec_flag` (LOI76): `~9` equals 1.0 on all samples.
//!
//! Exit 0 = all probes pass on all cases; exit 1 = any failure, with
//! a per-probe report on stdout.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::law_foams::jacobi_eigen_c1;
use std::path::PathBuf;
use std::process::ExitCode;

/// One probe outcome.
struct Probe {
    name: &'static str,
    pass: bool,
    detail: String,
}

impl Probe {
    fn ok(name: &'static str, detail: String) -> Self {
        Self {
            name,
            pass: true,
            detail,
        }
    }

    fn fail(name: &'static str, detail: String) -> Self {
        Self {
            name,
            pass: false,
            detail,
        }
    }
}

/// Tolerance-band float comparison: `|a - b| <= abs + rel * |b|`.
fn close(a: f64, b: f64, abs: f64, rel: f64) -> bool {
    (a - b).abs() <= abs + rel * b.abs()
}

/// Raw series of a channel.
fn series<'a>(doc: &'a Value, name: &str) -> &'a [Value] {
    doc["channels"][name]["values"]
        .as_array()
        .map_or(&[], Vec::as_slice)
}

/// Native min/max over a raw series.
fn min_max(values: &[Value]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for v in values {
        let x = v.as_f64().unwrap_or(f64::NAN);
        min = min.min(x);
        max = max.max(x);
    }
    (min, max)
}

fn num(values: &[Value], i: usize) -> f64 {
    values[i].as_f64().unwrap_or(f64::NAN)
}

/// Golden tolerances of a channel.
fn tols(doc: &Value, name: &str) -> (f64, f64) {
    (
        doc["channels"][name]["abs_tol"].as_f64().unwrap_or(0.0),
        doc["channels"][name]["rel_tol"].as_f64().unwrap_or(0.0),
    )
}

fn probe_schema(doc: &Value) -> Probe {
    let schema = doc["schema"].as_str().unwrap_or("<missing>");
    if schema == "opencourant-golden-v1" {
        return Probe::ok("schema", format!("schema tag {schema:?}"));
    }
    Probe::fail(
        "schema",
        format!("schema tag {schema:?}, want opencourant-golden-v1"),
    )
}

fn probe_termination(doc: &Value) -> Probe {
    if doc["meta"]["normal_termination"].as_bool().unwrap_or(false) {
        Probe::ok("termination", "normal_termination = true".to_string())
    } else {
        Probe::fail("termination", "run did not terminate normally".to_string())
    }
}

fn probe_grid(time: &[f64]) -> Probe {
    if time.len() < 2 {
        return Probe::fail("grid", format!("only {} samples", time.len()));
    }
    for w in time.windows(2) {
        if w[1].is_nan() || w[1] <= w[0] {
            return Probe::fail("grid", format!("time not increasing at t = {}", w[0]));
        }
    }
    Probe::ok(
        "grid",
        format!(
            "N = {}, t0 = {:.6}, tN = {:.6}",
            time.len(),
            time[0],
            time[time.len() - 1]
        ),
    )
}

fn probe_decimation(doc: &Value, n: usize) -> Probe {
    let record = &doc["meta"]["decimation"];
    if record.is_null() {
        return Probe::ok("decimation", "no decimation record (null)".to_string());
    }
    let kept = record["kept"].as_u64().unwrap_or(u64::MAX) as usize;
    if kept == n {
        Probe::ok(
            "decimation",
            format!("kept {kept} of {} rows", record["of"].as_u64().unwrap_or(0)),
        )
    } else {
        Probe::fail(
            "decimation",
            format!("kept {kept} rows but series have {n}"),
        )
    }
}

fn probe_shapes(doc: &Value, n: usize) -> Probe {
    let channels = doc["channels"].as_object().map_or(0, |m| m.len());
    let mut bad_len = 0;
    let mut bad_finite = 0;
    if let Some(map) = doc["channels"].as_object() {
        for (name, ch) in map {
            let values = ch["values"].as_array().map_or(&[][..], Vec::as_slice);
            if values.len() != n {
                bad_len += 1;
                let _ = name;
            }
            if values
                .iter()
                .any(|v| v.as_f64().is_none_or(|x| !x.is_finite()))
            {
                bad_finite += 1;
            }
        }
    }
    if bad_len == 0 && bad_finite == 0 {
        Probe::ok(
            "shapes",
            format!("{channels} channels x {n} finite samples"),
        )
    } else {
        Probe::fail(
            "shapes",
            format!("{bad_len} length mismatches, {bad_finite} non-finite channels"),
        )
    }
}

fn probe_summary(doc: &Value) -> Probe {
    let mut checked = 0;
    let mut worst = -1.0f64;
    let mut worst_name = String::from("<none>");
    if let Some(map) = doc["channels"].as_object() {
        for (name, ch) in map {
            let values = ch["values"].as_array().map_or(&[][..], Vec::as_slice);
            if values.is_empty() {
                continue;
            }
            let abs = ch["abs_tol"].as_f64().unwrap_or(0.0);
            let rel = ch["rel_tol"].as_f64().unwrap_or(0.0);
            let native_final = values[values.len() - 1].as_f64().unwrap_or(f64::NAN);
            let (native_min, native_max) = min_max(values);
            for (field, native) in [
                ("final", native_final),
                ("min", native_min),
                ("max", native_max),
            ] {
                let stored = ch[field].as_f64().unwrap_or(f64::NAN);
                let err = (native - stored).abs() / (abs + rel * stored.abs()).max(1.0e-300);
                if err > worst {
                    worst = err;
                    worst_name = format!("{name}/{field}");
                }
                if !close(native, stored, abs, rel) {
                    return Probe::fail(
                        "summary",
                        format!("{name}/{field}: native {native:.6e} vs stored {stored:.6e}"),
                    );
                }
            }
            checked += 1;
        }
    }
    Probe::ok(
        "summary",
        format!("{checked} channels recomputed, worst band fraction {worst:.3} ({worst_name})"),
    )
}

fn probe_energy(doc: &Value, n: usize) -> Probe {
    let get = |name: &str| series(doc, name);
    let (ke, ie, he, re, ce, ew, nn) = (
        get("KINETIC ENERGY"),
        get("INTERNAL ENERGY"),
        get("HOURGLASS ENERGY"),
        get("ROTATION ENERGY"),
        get("CONTACT ENERGY"),
        get("EXTERNAL WORK"),
        get("NO NAME"),
    );
    if [ke, ie, he, re, ce, ew, nn].iter().any(|s| s.len() != n) {
        return Probe::fail("energy", "energy channel missing or short".to_string());
    }
    let mut worst = 0.0f64;
    for i in 0..n {
        let parts = [
            num(ke, i),
            num(ie, i),
            num(he, i),
            num(re, i),
            num(ce, i),
            -num(ew, i),
            -num(nn, i),
        ];
        let resid: f64 = parts.iter().sum();
        let scale: f64 = parts.iter().map(|x| x.abs()).sum();
        worst = worst.max(resid.abs() / scale.max(1.0e-300));
    }
    if worst <= 1.0e-6 {
        Probe::ok(
            "energy",
            format!("KE+IE+HE+RE+CE-EXTW = total, worst scale-rel {worst:.3e}"),
        )
    } else {
        Probe::fail(
            "energy",
            format!("balance worst scale-rel {worst:.3e} > 1e-6"),
        )
    }
}

fn probe_momentum(doc: &Value, n: usize) -> Probe {
    let (xm, ym, zm, m, ke) = (
        series(doc, "X-MOMENTUM"),
        series(doc, "Y-MOMENTUM"),
        series(doc, "Z-MOMENTUM"),
        series(doc, "MASS"),
        series(doc, "KINETIC ENERGY"),
    );
    if [xm, ym, zm, m, ke].iter().any(|s| s.len() != n) {
        return Probe::fail("momentum", "momentum channel missing or short".to_string());
    }
    let mut worst = 0.0f64;
    for i in 0..n {
        let p2 = num(xm, i).powi(2) + num(ym, i).powi(2) + num(zm, i).powi(2);
        let bound = 2.0 * num(m, i) * num(ke, i);
        if bound > 0.0 {
            worst = worst.max(p2 / bound - 1.0);
        }
    }
    if worst <= 1.0e-6 {
        Probe::ok(
            "momentum",
            format!("|p|^2 <= 2m KE, worst excess {worst:.3e}"),
        )
    } else {
        Probe::fail("momentum", format!("bound excess {worst:.3e} > 1e-6"))
    }
}

fn probe_mass(doc: &Value, n: usize) -> Probe {
    let m = series(doc, "MASS");
    if m.len() != n {
        return Probe::fail("mass", "MASS channel missing or short".to_string());
    }
    let (min, max) = min_max(m);
    if max - min > 1.0e-9 {
        return Probe::fail("mass", format!("MASS spread {:.3e} > 1e-9", max - min));
    }
    let total = doc["cycle"]["TOTALMASS"].as_f64().unwrap_or(f64::NAN);
    let last = num(m, n - 1);
    if !close(total, last, 0.0, 1.0e-3) {
        return Probe::fail(
            "mass",
            format!("TOTALMASS {total:.6e} vs MASS last {last:.6e}"),
        );
    }
    Probe::ok(
        "mass",
        format!("MASS spread {:.3e}, TOTALMASS matches", max - min),
    )
}

fn probe_cycle(doc: &Value) -> Probe {
    let maserr = doc["cycle"]["MASERR"].as_f64().unwrap_or(f64::NAN);
    if !maserr.is_finite() || maserr.abs() > 1.0e-12 {
        return Probe::fail("cycle", format!("MASERR {maserr:.3e}"));
    }
    for key in ["TSTEP", "CYCLE", "TIME"] {
        let v = doc["cycle"][key].as_f64().unwrap_or(f64::NAN);
        if !v.is_finite() || v <= 0.0 {
            return Probe::fail("cycle", format!("{key} = {v:.3e} not positive"));
        }
    }
    Probe::ok(
        "cycle",
        format!(
            "MASERR {maserr:.1e}, TSTEP {:.3e}, CYCLE {:.0}, TIME {:.4}",
            doc["cycle"]["TSTEP"].as_f64().unwrap_or(f64::NAN),
            doc["cycle"]["CYCLE"].as_f64().unwrap_or(f64::NAN),
            doc["cycle"]["TIME"].as_f64().unwrap_or(f64::NAN),
        ),
    )
}

// ---------------------------------------------------------------------------
// FOAM (LAW33) probes
// ---------------------------------------------------------------------------

/// LAW33 deck research constants from `FOAM_V10_0000.rad`, compiled in.
///
/// Ambient pressure, porosity, offset, and the three yield-curve
/// coefficients of Theory Manual equations 1024-1027. The runtime
/// reads only the golden JSON; these deck facts are fixed inputs.
const LAW33_P0: f64 = 1e-4;
const LAW33_PHI: f64 = 0.1;
const LAW33_G0: f64 = 0.0;
const LAW33_A: f64 = -1.77e-4;
const LAW33_B: f64 = 6.77e-4;
const LAW33_C: f64 = -3.34;

/// Element ids behind `FOAM429D00_th_3Dsolid {eid} ~1` keys.
fn foam_element_ids(doc: &Value) -> Vec<String> {
    let mut ids = Vec::new();
    if let Some(map) = doc["channels"].as_object() {
        for name in map.keys() {
            if let Some(rest) = name.strip_prefix("FOAM429D00_th_3Dsolid ")
                && let Some(eid) = rest.strip_suffix(" ~1")
            {
                ids.push(eid.to_string());
            }
        }
    }
    ids.sort();
    ids
}

fn probe_law33_bound(doc: &Value, n: usize) -> Probe {
    let ids = foam_element_ids(doc);
    if ids.is_empty() {
        return Probe::fail("law33_bound", "no TH solid groups".to_string());
    }
    let mut worst = 0.0f64;
    let mut checked = 0usize;
    let mut violations = 0usize;
    for eid in &ids {
        let key = |k: u8| format!("FOAM429D00_th_3Dsolid {eid} ~{k}");
        let chans: Vec<&[Value]> = (1..=9).map(|k| series(doc, &key(k))).collect();
        if chans.iter().any(|s| s.len() != n) {
            return Probe::fail("law33_bound", format!("element {eid} short channel"));
        }
        let rho0 = num(chans[8], 0);
        for i in 0..n {
            if num(chans[0], i) < 0.5 {
                continue;
            }
            let rho = num(chans[8], i);
            if !rho.is_finite() || rho <= 0.0 {
                return Probe::fail(
                    "law33_bound",
                    format!("element {eid} sample {i}: bad density {rho:.3e}"),
                );
            }
            let gam = rho0 / rho - 1.0 + LAW33_G0;
            let pair = -LAW33_P0 * gam / (1.0 + gam - LAW33_PHI);
            let yield_ = LAW33_A + LAW33_B * (1.0 + LAW33_C * gam);
            let stress = [
                num(chans[1], i) + pair,
                num(chans[2], i) + pair,
                num(chans[3], i) + pair,
                num(chans[4], i),
                num(chans[5], i),
                num(chans[6], i),
            ];
            let (values, _) = jacobi_eigen_c1(&stress, 12);
            let most = values[0].abs().max(values[1].abs()).max(values[2].abs());
            checked += 1;
            worst = worst.max(most / yield_.max(1.0e-30));
            if most > yield_ * 1.01 + 1e-9 {
                violations += 1;
            }
        }
    }
    if violations == 0 {
        Probe::ok(
            "law33_bound",
            format!("{checked} samples, 0 violations, worst ratio {worst:.3}"),
        )
    } else {
        Probe::fail(
            "law33_bound",
            format!("{violations}/{checked} violations, worst {worst:.3}"),
        )
    }
}

fn probe_foam_part(doc: &Value, n: usize) -> Probe {
    for (part, global) in [
        ("FOAM LAW 33 IE", "INTERNAL ENERGY"),
        ("FOAM LAW 33 HE", "HOURGLASS ENERGY"),
    ] {
        let (a, b) = (series(doc, part), series(doc, global));
        if a.len() != n || b.len() != n {
            return Probe::fail("foam_part", format!("{part} missing or short"));
        }
        let (abs, rel) = tols(doc, global);
        let worst = (0..n)
            .map(|i| (num(a, i) - num(b, i)).abs() / (abs + rel * num(b, i).abs()).max(1.0e-300))
            .fold(0.0f64, f64::max);
        if worst > 1.0 {
            return Probe::fail(
                "foam_part",
                format!("{part} vs {global}: band fraction {worst:.3}"),
            );
        }
    }
    let eroded = series(doc, "FOAM LAW 33 ERODED");
    if eroded.len() != n {
        return Probe::fail("foam_part", "ERODED missing or short".to_string());
    }
    let (_, max) = min_max(eroded);
    if max > 0.0 {
        return Probe::fail("foam_part", format!("ERODED max {max:.3e}"));
    }
    Probe::ok(
        "foam_part",
        "part IE/HE match globals, ERODED zero".to_string(),
    )
}

fn probe_foam_off(doc: &Value, n: usize) -> Probe {
    let ids = foam_element_ids(doc);
    for eid in &ids {
        let name = format!("FOAM429D00_th_3Dsolid {eid} ~1");
        let s = series(doc, &name);
        if s.len() != n {
            return Probe::fail("foam_off", format!("element {eid} flag short"));
        }
        let (abs, rel) = tols(doc, &name);
        for i in 0..n {
            if !close(num(s, i), 1.0, abs, rel) {
                return Probe::fail(
                    "foam_off",
                    format!("element {eid} sample {i}: flag {:.3e}", num(s, i)),
                );
            }
        }
    }
    Probe::ok("foam_off", format!("{} elements active", ids.len()))
}

// ---------------------------------------------------------------------------
// LOI50 (LAW50) probes
// ---------------------------------------------------------------------------

fn probe_solide_ie(doc: &Value, n: usize) -> Probe {
    let (a, b) = (
        series(doc, "solide 1 solide ~8"),
        series(doc, "INTERNAL ENERGY"),
    );
    if a.len() != n || b.len() != n {
        return Probe::fail("solide_ie", "channel missing or short".to_string());
    }
    let (abs, rel) = tols(doc, "INTERNAL ENERGY");
    let worst = (0..n)
        .map(|i| (num(a, i) - num(b, i)).abs() / (abs + rel * num(b, i).abs()).max(1.0e-300))
        .fold(0.0f64, f64::max);
    if worst > 1.0 {
        Probe::fail(
            "solide_ie",
            format!("element IE vs global: band fraction {worst:.3}"),
        )
    } else {
        Probe::ok(
            "solide_ie",
            format!("element ~8 matches global IE, band fraction {worst:.3}"),
        )
    }
}

fn probe_solide_flag(doc: &Value, n: usize) -> Probe {
    let s = series(doc, "solide 1 solide ~1");
    if s.len() != n {
        return Probe::fail("solide_flag", "flag channel short".to_string());
    }
    let (abs, rel) = tols(doc, "solide 1 solide ~1");
    for i in 0..n {
        if !close(num(s, i), 1.0, abs, rel) {
            return Probe::fail("solide_flag", format!("sample {i}: flag {:.3e}", num(s, i)));
        }
    }
    Probe::ok("solide_flag", "activity flag 1.0 throughout".to_string())
}

fn probe_vol_positive(doc: &Value, n: usize) -> Probe {
    let s = series(doc, "solide 1 solide ~9");
    if s.len() != n {
        return Probe::fail("vol_positive", "density channel short".to_string());
    }
    let (min, _) = min_max(s);
    if min > 0.0 {
        Probe::ok("vol_positive", format!("density min {min:.3e} > 0"))
    } else {
        Probe::fail("vol_positive", format!("density min {min:.3e}"))
    }
}

fn probe_plastic_zero(doc: &Value, n: usize) -> Probe {
    let s = series(doc, "PLASTIC WORK");
    if s.len() != n {
        return Probe::fail("plastic_zero", "PLASTIC WORK short".to_string());
    }
    let (abs, _) = tols(doc, "PLASTIC WORK");
    let (_, max) = min_max(s);
    if max <= abs {
        Probe::ok(
            "plastic_zero",
            format!("PLASTIC WORK max {max:.3e} within abs tol"),
        )
    } else {
        Probe::fail(
            "plastic_zero",
            format!("PLASTIC WORK max {max:.3e} exceeds abs tol {abs:.1e}"),
        )
    }
}

// ---------------------------------------------------------------------------
// LOI51 (LAW51) probes
// ---------------------------------------------------------------------------

fn probe_hydrostatic(doc: &Value, n: usize) -> Probe {
    let p: Vec<&[Value]> = [2, 3, 4]
        .iter()
        .map(|k| series(doc, &format!("brick 2 brick_2 ~{k}")))
        .collect();
    if p.iter().any(|s| s.len() != n) {
        return Probe::fail("hydrostatic", "pressure channel short".to_string());
    }
    let (abs, rel) = tols(doc, "brick 2 brick_2 ~2");
    let mut worst = 0.0f64;
    for i in 0..n {
        for pair in [(0, 1), (0, 2)] {
            let err = (num(p[pair.0], i) - num(p[pair.1], i)).abs()
                / (abs + rel * num(p[0], i).abs()).max(1.0e-300);
            worst = worst.max(err);
        }
    }
    if worst > 1.0 {
        return Probe::fail(
            "hydrostatic",
            format!("normal spread band fraction {worst:.3}"),
        );
    }
    for k in [5, 6, 7] {
        let name = format!("brick 2 brick_2 ~{k}");
        let s = series(doc, &name);
        if s.len() != n {
            return Probe::fail("hydrostatic", format!("{name} short"));
        }
        let (sabs, _) = tols(doc, &name);
        let (min, max) = min_max(s);
        if min.abs().max(max.abs()) > sabs {
            return Probe::fail("hydrostatic", format!("{name} shear nonzero"));
        }
    }
    Probe::ok(
        "hydrostatic",
        format!("normals equal, shears zero, band fraction {worst:.3}"),
    )
}

fn probe_brick_flag(doc: &Value, n: usize) -> Probe {
    for k in [1, 12] {
        let name = format!("brick 2 brick_2 ~{k}");
        let s = series(doc, &name);
        if s.len() != n {
            return Probe::fail("brick_flag", format!("{name} short"));
        }
        let (abs, rel) = tols(doc, &name);
        for i in 0..n {
            if !close(num(s, i), 1.0, abs, rel) {
                return Probe::fail("brick_flag", format!("{name} sample {i} off"));
            }
        }
    }
    Probe::ok("brick_flag", "~1 and ~12 equal 1.0".to_string())
}

/// Linear interpolation with edge clamping over ascending `(x, y)`.
fn interp_clamped(pairs: &[(f64, f64)], x: f64) -> f64 {
    if x <= pairs[0].0 {
        return pairs[0].1;
    }
    let last = pairs.len() - 1;
    if x >= pairs[last].0 {
        return pairs[last].1;
    }
    for w in pairs.windows(2) {
        if x <= w[1].0 {
            let frac = (x - w[0].0) / (w[1].0 - w[0].0);
            return w[0].1 + frac * (w[1].1 - w[0].1);
        }
    }
    pairs[last].1
}

fn floats(doc: &Value, name: &str) -> Vec<f64> {
    series(doc, name)
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect()
}

fn strictly_decreasing(values: &[f64]) -> bool {
    values
        .windows(2)
        .all(|w| w[0].is_finite() && w[1].is_finite() && w[1] < w[0])
}

fn probe_pe_cross(pa: &Value, pb: &Value) -> Probe {
    let press = floats(pa, "brick 2 brick_2 ~2");
    let ea = floats(pa, "brick 2 brick_2 ~8");
    let dp = floats(pb, "brick 2 brick_2 ~2");
    let eb = floats(pb, "brick 2 brick_2 ~8");
    if press.len() != ea.len() || dp.len() != eb.len() || press.is_empty() {
        return Probe::fail("pe_cross", "P/E channel length mismatch".to_string());
    }
    if !strictly_decreasing(&ea) || !strictly_decreasing(&eb) {
        return Probe::fail(
            "pe_cross",
            "energy grid not strictly decreasing".to_string(),
        );
    }
    let mut pairs: Vec<(f64, f64)> = eb.iter().copied().zip(dp.iter().copied()).collect();
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let p0 = press[0];
    let mut worst = 0.0f64;
    for i in 0..press.len() {
        let expected = interp_clamped(&pairs, ea[i]);
        let got = press[i] - p0;
        let err = (got - expected).abs() / (1.0 + 1.0e-5 * expected.abs());
        worst = worst.max(err);
        if !close(got, expected, 1.0, 1.0e-5) {
            return Probe::fail(
                "pe_cross",
                format!("sample {i}: dP {got:.6e} vs 01b {expected:.6e}"),
            );
        }
    }
    Probe::ok(
        "pe_cross",
        format!("01a P-P0 vs 01b dP on E grid, worst band fraction {worst:.3}"),
    )
}

fn probe_trajectory(pa: &Value, pb: &Value) -> Probe {
    let ea = floats(pa, "brick 2 brick_2 ~8");
    let eb = floats(pb, "brick 2 brick_2 ~8");
    if ea.len() != eb.len() || ea.is_empty() {
        return Probe::fail("trajectory", "energy grid length mismatch".to_string());
    }
    let worst = ea
        .iter()
        .zip(eb.iter())
        .map(|(a, b)| (a - b).abs() / a.abs().max(1.0e-300))
        .fold(0.0f64, f64::max);
    if worst <= 5.0e-3 {
        Probe::ok(
            "trajectory",
            format!("run-to-run energy jitter {worst:.3e} <= 5e-3"),
        )
    } else {
        Probe::fail("trajectory", format!("energy jitter {worst:.3e} > 5e-3"))
    }
}

// ---------------------------------------------------------------------------
// LOI76 (LAW76) probes
// ---------------------------------------------------------------------------

fn probe_spec_ie(doc: &Value, n: usize) -> Probe {
    let (a, b) = (
        series(doc, "spec 1000001 compr ~7"),
        series(doc, "INTERNAL ENERGY"),
    );
    if a.len() != n || b.len() != n {
        return Probe::fail("spec_ie", "channel missing or short".to_string());
    }
    let (abs, rel) = tols(doc, "INTERNAL ENERGY");
    let worst = (0..n)
        .map(|i| (num(a, i) - num(b, i)).abs() / (abs + rel * num(b, i).abs()).max(1.0e-300))
        .fold(0.0f64, f64::max);
    if worst > 1.0 {
        Probe::fail(
            "spec_ie",
            format!("element IE vs global: band fraction {worst:.3}"),
        )
    } else {
        Probe::ok(
            "spec_ie",
            format!("element ~7 matches global IE, band fraction {worst:.3}"),
        )
    }
}

fn probe_compression(doc: &Value, n: usize) -> Probe {
    let name = "spec 1000001 compr ~1";
    let s = series(doc, name);
    if s.len() != n {
        return Probe::fail("compression", "stress channel short".to_string());
    }
    let (abs, _) = tols(doc, name);
    let (min, max) = min_max(s);
    if max > abs {
        return Probe::fail("compression", format!("tensile peak {max:.3e}"));
    }
    if num(s, n - 1) >= 0.0 {
        return Probe::fail("compression", "final stress not compressive".to_string());
    }
    Probe::ok(
        "compression",
        format!("~1 in [{min:.3e}, {max:.3e}], compressive"),
    )
}

fn probe_top_bottom(doc: &Value, n: usize) -> Probe {
    let (a, b) = (
        series(doc, "spec 1000001 compr ~10"),
        series(doc, "spec 1000001 compr ~11"),
    );
    if a.len() != n || b.len() != n {
        return Probe::fail("top_bottom", "face channel short".to_string());
    }
    let (abs, rel) = tols(doc, "spec 1000001 compr ~11");
    let worst = (0..n)
        .map(|i| (num(a, i) - num(b, i)).abs() / (abs + rel * num(b, i).abs()).max(1.0e-300))
        .fold(0.0f64, f64::max);
    if worst > 1.0 {
        Probe::fail(
            "top_bottom",
            format!("face mismatch band fraction {worst:.3}"),
        )
    } else {
        Probe::ok(
            "top_bottom",
            format!("~10 equals ~11, band fraction {worst:.3}"),
        )
    }
}

fn probe_plastic(doc: &Value, n: usize) -> Probe {
    let s = series(doc, "PLASTIC WORK");
    let ie = series(doc, "INTERNAL ENERGY");
    if s.len() != n || ie.len() != n {
        return Probe::fail("plastic", "work channel short".to_string());
    }
    let (abs, _) = tols(doc, "PLASTIC WORK");
    for i in 1..n {
        if num(s, i) < num(s, i - 1) - abs {
            return Probe::fail("plastic", "PLASTIC WORK decreases".to_string());
        }
    }
    let final_pw = num(s, n - 1);
    if final_pw <= 0.0 {
        return Probe::fail("plastic", "no plastic work accumulated".to_string());
    }
    if final_pw > num(ie, n - 1) {
        return Probe::fail(
            "plastic",
            "plastic work exceeds internal energy".to_string(),
        );
    }
    Probe::ok(
        "plastic",
        format!("PLASTIC WORK monotone, final {final_pw:.6e}"),
    )
}

fn probe_spec_flag(doc: &Value, n: usize) -> Probe {
    let name = "spec 1000001 compr ~9";
    let s = series(doc, name);
    if s.len() != n {
        return Probe::fail("spec_flag", "flag channel short".to_string());
    }
    let (abs, rel) = tols(doc, name);
    for i in 0..n {
        if !close(num(s, i), 1.0, abs, rel) {
            return Probe::fail("spec_flag", format!("sample {i} off"));
        }
    }
    Probe::ok("spec_flag", "~9 equals 1.0".to_string())
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Loads one golden document.
fn load_doc(path: &PathBuf) -> Result<Value, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| format!("cannot read {}: {err}", path.display()))?;
    serde_json::from_str(&text).map_err(|err| format!("bad JSON: {err}"))
}

/// Common probes shared by every case.
fn common_probes(doc: &Value, time: &[f64]) -> Vec<Probe> {
    let n = time.len();
    vec![
        probe_schema(doc),
        probe_termination(doc),
        probe_grid(time),
        probe_decimation(doc, n),
        probe_shapes(doc, n),
        probe_summary(doc),
        probe_energy(doc, n),
        probe_momentum(doc, n),
        probe_mass(doc, n),
        probe_cycle(doc),
    ]
}

/// Case-specific probes selected by file stem.
fn case_probes(doc: &Value, n: usize, case: &str) -> Vec<Probe> {
    match case {
        "foam" => vec![
            probe_law33_bound(doc, n),
            probe_foam_part(doc, n),
            probe_foam_off(doc, n),
        ],
        "loi50" => vec![
            probe_solide_ie(doc, n),
            probe_solide_flag(doc, n),
            probe_vol_positive(doc, n),
            probe_plastic_zero(doc, n),
        ],
        "loi51a" | "loi51b" => vec![probe_hydrostatic(doc, n), probe_brick_flag(doc, n)],
        "loi76" => vec![
            probe_spec_ie(doc, n),
            probe_compression(doc, n),
            probe_top_bottom(doc, n),
            probe_plastic(doc, n),
            probe_spec_flag(doc, n),
        ],
        _ => vec![Probe::fail("case", format!("unknown case {case}"))],
    }
}

fn main() -> ExitCode {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens/miniqa/LOIS");
    let cases = [
        (root.join("FOAM/FOAM429/input_100/isolid_101.json"), "foam"),
        (root.join("LOI50/test1/pid_6/input_44.json"), "loi50"),
        (
            root.join("LOI51/JWL/01-PG_degenerated/01a_P-E.json"),
            "loi51a",
        ),
        (
            root.join("LOI51/JWL/01-PG_degenerated/01b_dP-E.json"),
            "loi51b",
        ),
        (root.join("LOI76/shell/test1/compression.json"), "loi76"),
    ];
    println!("qa-fem-foam-laws: lane 3C G-leg (LOIS foam-law cases)");
    println!("goldens root: {}", root.display());
    let mut total = 0;
    let mut failed = 0;
    let mut cross: Vec<(&str, Value)> = Vec::new();
    for (path, case) in &cases {
        let label = path
            .file_name()
            .map_or_else(|| "?".to_string(), |s| s.to_string_lossy().into_owned());
        println!("--- {label} ({})", path.display());
        let doc = match load_doc(path) {
            Ok(doc) => doc,
            Err(err) => {
                println!("FAIL load        {err}");
                total += 1;
                failed += 1;
                continue;
            }
        };
        if *case == "loi51a" || *case == "loi51b" {
            cross.push((case, doc.clone()));
        }
        let time: Vec<f64> = doc["time"].as_array().map_or(Vec::new(), |t| {
            t.iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect()
        });
        let mut probes = common_probes(&doc, &time);
        probes.extend(case_probes(&doc, time.len(), case));
        for probe in &probes {
            total += 1;
            if !probe.pass {
                failed += 1;
            }
            println!(
                "{} {:<12} {}",
                if probe.pass { "PASS" } else { "FAIL" },
                probe.name,
                probe.detail
            );
        }
    }
    println!("--- cross-file LOI51 01a vs 01b");
    if cross.len() == 2 {
        let pa = cross.iter().find(|(c, _)| *c == "loi51a").map(|(_, d)| d);
        let pb = cross.iter().find(|(c, _)| *c == "loi51b").map(|(_, d)| d);
        if let (Some(pa), Some(pb)) = (pa, pb) {
            for probe in [probe_pe_cross(pa, pb), probe_trajectory(pa, pb)] {
                total += 1;
                if !probe.pass {
                    failed += 1;
                }
                println!(
                    "{} {:<12} {}",
                    if probe.pass { "PASS" } else { "FAIL" },
                    probe.name,
                    probe.detail
                );
            }
        }
    } else {
        println!("FAIL cross        LOI51 pair incomplete, skipping");
        total += 1;
        failed += 1;
    }
    println!(
        "summary: {} passed, {} failed of {total}",
        total - failed,
        failed
    );
    if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
