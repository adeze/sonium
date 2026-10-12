//! Phase-3 lane 3A G-leg (lane-owned): crash golden-integrity audit.
//!
//! The CRASH fixtures retain integrity-only checks, reported with zero
//! numerical solves. A separate native CST/DKT shell dynamics case compares
//! a clamped circular plate with an independently harvested engine response.
//! The original 140-Q4 CRASH box additionally evolves native shell, material,
//! wall and rigid-body states and compares the full harvested impact history.
//! The circular-plate gate covers elastic pressure-hold equilibrium, not impact
//! plasticity or reproduction of the CRASH vendor models. All fixtures are
//! checked in; no engine or network is required at QA runtime.
//!
//! Each probe re-derives a quantity from the harvested raw series and
//! compares it against the harvested summary or a physics identity, with
//! tolerances justified per probe:
//!
//! - `schema`: `schema` tag equals `opencourant-golden-v1` (SCHEMA.md).
//! - `termination`: engine run terminated normally.
//! - `grid`: time grid strictly increasing with at least 2 samples.
//! - `decimation`: kept-row record matches the series length, or the
//!   record is null (no decimation).
//! - `shapes`: every channel has exactly N finite samples.
//! - `summary`: recomputed final/min/max match the harvested summaries
//!   within each channel's own `abs_tol`/`rel_tol`.
//! - `energy`: `KE + IE + HE + RE + CE - EXTW` reproduces the total
//!   `NO NAME` channel within 1e-6 relative (7-digit TH rounding;
//!   observed worst 3e-7 on both cases).
//! - `momentum`: `|p|^2 <= 2 m KE` at every sample within 1e-6
//!   (Cauchy-Schwarz over the mass distribution; exact physics).
//! - `mass`: MASS constant within 1e-9 absolute; cycle TOTALMASS
//!   within 0.1% (observed <= 0.02%).
//! - `cycle`: last-cycle `.out` scalars vs last TH samples within
//!   cadence bands: energies 2% (observed <= 1.2%), TSTEP 0.5%
//!   (observed <= 0.14%), TIME within 2% of the grid end (observed
//!   <= 1.1%), rotation energy and external work scale-relative,
//!   mass/error flags exactly zero.
//! - `hgfrac`: hourglass-to-internal energy below 5% (explicit-code
//!   acceptance; observed 0.9% and 0.0%).
//! - `tstep`: step stays positive; reports min/max.
//! - `shells`: element `~9` active flags equal 1.0, or no shell TH
//!   channels (reported, still passing).
//!
//! Exit 0 = all probes pass on both cases; exit 1 = any failure, with
//! a per-probe report on stdout.

// Rust guideline compliant 2026-02-21

#[path = "support/crash_box.rs"]
mod crash_box;
#[path = "support/crash_box_golden.rs"]
mod crash_box_golden;
#[path = "support/shell_plate_golden.rs"]
mod shell_plate_golden;

use serde_json::Value;
use sha2::{Digest, Sha256};
use sonium_qa::QaResult;
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

/// Energy-balance worst relative residual cap (7-digit TH rounding).
const ENERGY_BALANCE_REL_TOL: f64 = 1.0e-6;
/// Momentum-bound worst excess cap (exact Cauchy-Schwarz physics).
const MOMENTUM_EXCESS_TOL: f64 = 1.0e-6;
/// MASS absolute drift cap across the run.
const MASS_DRIFT_ABS_TOL: f64 = 1.0e-9;
/// Cycle TOTALMASS relative cap against the last MASS sample.
const MASS_CYCLE_REL_TOL: f64 = 1.0e-3;
/// Cycle-vs-TH cadence caps for TIME and bulk energies.
const CYCLE_TIME_REL_TOL: f64 = 0.02;
const CYCLE_ENERGY_REL_TOL: f64 = 0.02;
/// Cycle-vs-TH cadence cap for TSTEP.
const CYCLE_TSTEP_REL_TOL: f64 = 0.005;
/// Rotation-energy mismatch cap, scale-relative to peak KE.
const CYCLE_ROTATION_REL_TOL: f64 = 1.0e-5;
/// External-work mismatch cap, scale-relative to its final value.
const CYCLE_EXTWORK_REL_TOL: f64 = 1.0e-3;
/// Hourglass-to-internal energy acceptance cap (explicit-code practice).
const HOURGLASS_FRACTION_MAX: f64 = 0.05;
/// Zero guard for relative-error denominators.
const ZERO_FLOOR: f64 = 1.0e-300;

/// Raw series of a channel.
fn series<'a>(doc: &'a Value, name: &str) -> &'a [Value] {
    doc["channels"][name]["values"]
        .as_array()
        .map_or(&[], Vec::as_slice)
}

/// Harvested scalar summary of a channel.
fn summary(doc: &Value, name: &str, field: &str) -> f64 {
    doc["channels"][name][field].as_f64().unwrap_or(f64::NAN)
}

/// Min/max over a raw series.
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

fn probe_grid(doc: &Value, time: &[f64]) -> Probe {
    if time.len() < 2 {
        return Probe::fail("grid", format!("only {} samples", time.len()));
    }
    for w in time.windows(2) {
        if w[1].is_nan() || w[1] <= w[0] {
            return Probe::fail("grid", format!("time not increasing at t = {}", w[0]));
        }
    }
    let _ = doc;
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
            let recomputed_final = values[values.len() - 1].as_f64().unwrap_or(f64::NAN);
            let (recomputed_min, recomputed_max) = min_max(values);
            for (field, recomputed) in [
                ("final", recomputed_final),
                ("min", recomputed_min),
                ("max", recomputed_max),
            ] {
                let stored = ch[field].as_f64().unwrap_or(f64::NAN);
                let err = (recomputed - stored).abs() / (abs + rel * stored.abs()).max(ZERO_FLOOR);
                if err > worst {
                    worst = err;
                    worst_name = format!("{name}/{field}");
                }
                if !close(recomputed, stored, abs, rel) {
                    return Probe::fail(
                        "summary",
                        format!(
                            "{name}/{field}: recomputed {recomputed:.6e} vs stored {stored:.6e}"
                        ),
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

fn num(values: &[Value], i: usize) -> f64 {
    values[i].as_f64().unwrap_or(f64::NAN)
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
        let total = num(ke, i) + num(ie, i) + num(he, i) + num(re, i) + num(ce, i) - num(ew, i);
        let err = (total - num(nn, i)).abs() / num(nn, i).abs().max(ZERO_FLOOR);
        worst = worst.max(err);
    }
    if worst <= ENERGY_BALANCE_REL_TOL {
        Probe::ok(
            "energy",
            format!("KE+IE+HE+RE+CE-EXTW = total, worst rel {worst:.3e}"),
        )
    } else {
        Probe::fail(
            "energy",
            format!("balance worst rel {worst:.3e} > {ENERGY_BALANCE_REL_TOL:.1e}"),
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
    if worst <= MOMENTUM_EXCESS_TOL {
        Probe::ok(
            "momentum",
            format!("|p|^2 <= 2m KE, worst excess {worst:.3e}"),
        )
    } else {
        Probe::fail(
            "momentum",
            format!("bound excess {worst:.3e} > {MOMENTUM_EXCESS_TOL:.1e}"),
        )
    }
}

fn probe_mass(doc: &Value, n: usize) -> Probe {
    let m = series(doc, "MASS");
    if m.len() != n {
        return Probe::fail("mass", "MASS channel missing or short".to_string());
    }
    let m0 = num(m, 0);
    let drift = (0..n)
        .map(|i| (num(m, i) - m0).abs())
        .fold(0.0f64, f64::max);
    let total = doc["cycle"]["TOTALMASS"].as_f64().unwrap_or(f64::NAN);
    let cycle_err = (total - num(m, n - 1)).abs() / num(m, n - 1).abs().max(ZERO_FLOOR);
    if drift <= MASS_DRIFT_ABS_TOL && cycle_err <= MASS_CYCLE_REL_TOL {
        Probe::ok(
            "mass",
            format!("MASS const (drift {drift:.3e}), cycle TOTALMASS rel {cycle_err:.3e}"),
        )
    } else {
        Probe::fail(
            "mass",
            format!(
                "drift {drift:.3e} (want <= {MASS_DRIFT_ABS_TOL:.1e}), \
                 cycle rel {cycle_err:.3e} (want <= {MASS_CYCLE_REL_TOL:.1e})"
            ),
        )
    }
}

fn probe_cycle(doc: &Value, time: &[f64]) -> Probe {
    let n = time.len();
    let cycle = &doc["cycle"];
    let mut notes = Vec::new();
    let mut fail = Vec::new();
    // TIME vs grid end (TH vs .out print cadence).
    let t_end = time[n - 1];
    let ct = cycle["TIME"].as_f64().unwrap_or(f64::NAN);
    let dt_err = (ct - t_end).abs() / t_end.abs().max(ZERO_FLOOR);
    notes.push(format!("TIME rel {dt_err:.3e}"));
    if dt_err > CYCLE_TIME_REL_TOL {
        fail.push("TIME");
    }
    // Bulk energies.
    for (field, channel) in [
        ("IENERGY", "INTERNAL ENERGY"),
        ("KENERGYT", "KINETIC ENERGY"),
    ] {
        let c = cycle[field].as_f64().unwrap_or(f64::NAN);
        let f = summary(doc, channel, "final");
        let err = (c - f).abs() / f.abs().max(ZERO_FLOOR);
        notes.push(format!("{field} rel {err:.3e}"));
        if err > CYCLE_ENERGY_REL_TOL {
            fail.push(field);
        }
    }
    // TSTEP.
    let ts = cycle["TSTEP"].as_f64().unwrap_or(f64::NAN);
    let tf = summary(doc, "TIME STEP", "final");
    let ts_err = (ts - tf).abs() / tf.abs().max(ZERO_FLOOR);
    notes.push(format!("TSTEP rel {ts_err:.3e}"));
    if ts_err > CYCLE_TSTEP_REL_TOL {
        fail.push("TSTEP");
    }
    // Rotation energy and external work scale-relative to KE peak.
    let (_, ke_max) = min_max(series(doc, "KINETIC ENERGY"));
    let kr = cycle["KENERGYR"].as_f64().unwrap_or(f64::NAN);
    let krf = summary(doc, "ROTATION ENERGY", "final");
    let kr_err = (kr - krf).abs() / ke_max.abs().max(ZERO_FLOOR);
    notes.push(format!("KENERGYR/KE rel {kr_err:.3e}"));
    if kr_err > CYCLE_ROTATION_REL_TOL {
        fail.push("KENERGYR");
    }
    let ew = cycle["EXTWORK"].as_f64().unwrap_or(f64::NAN);
    let ewf = summary(doc, "EXTERNAL WORK", "final");
    let ew_err = (ew - ewf).abs() / (ewf.abs().max(ZERO_FLOOR) + 1.0e-9 * ke_max.abs());
    notes.push(format!("EXTWORK rel {ew_err:.3e}"));
    if ew_err > CYCLE_EXTWORK_REL_TOL {
        fail.push("EXTWORK");
    }
    // Clean flags exactly zero.
    for field in ["ADDEDMASS", "MASERR", "ERROR"] {
        let v = cycle[field].as_f64().unwrap_or(f64::NAN);
        notes.push(format!("{field} = {v}"));
        if v != 0.0 {
            fail.push(field);
        }
    }
    if fail.is_empty() {
        Probe::ok("cycle", notes.join(", "))
    } else {
        Probe::fail(
            "cycle",
            format!("{} failed: {}", fail.join(","), notes.join(", ")),
        )
    }
}

fn probe_hgfrac(doc: &Value) -> Probe {
    let (_, ie_max) = min_max(series(doc, "INTERNAL ENERGY"));
    let (_, he_max) = min_max(series(doc, "HOURGLASS ENERGY"));
    let frac = he_max / ie_max.abs().max(ZERO_FLOOR);
    if frac < HOURGLASS_FRACTION_MAX {
        Probe::ok(
            "hgfrac",
            format!(
                "max HE / max IE = {frac:.3e} < {:.0}%",
                100.0 * HOURGLASS_FRACTION_MAX
            ),
        )
    } else {
        Probe::fail(
            "hgfrac",
            format!(
                "max HE / max IE = {frac:.3e} >= {:.0}%",
                100.0 * HOURGLASS_FRACTION_MAX
            ),
        )
    }
}

fn probe_tstep(doc: &Value) -> Probe {
    let (min, max) = min_max(series(doc, "TIME STEP"));
    if min > 0.0 {
        Probe::ok("tstep", format!("dt in [{min:.4e}, {max:.4e}]"))
    } else {
        Probe::fail("tstep", format!("non-positive dt min {min:.4e}"))
    }
}

fn probe_shells(doc: &Value, n: usize) -> Probe {
    let mut flags = 0;
    let mut bad = 0;
    if let Some(map) = doc["channels"].as_object() {
        for (name, ch) in map {
            if name.starts_with("Shells ") && name.ends_with(" ~9") {
                flags += 1;
                let values = ch["values"].as_array().map_or(&[][..], Vec::as_slice);
                if values.len() != n || values.iter().any(|v| v.as_f64() != Some(1.0)) {
                    bad += 1;
                }
            }
        }
    }
    if flags == 0 {
        Probe::ok("shells", "no shell TH channels harvested".to_string())
    } else if bad == 0 {
        Probe::ok("shells", format!("{flags} element ~9 flags all 1.0"))
    } else {
        Probe::fail("shells", format!("{bad}/{flags} ~9 flags deviate from 1.0"))
    }
}

/// Runs all probes on one golden file.
fn run_case(path: &PathBuf) -> (String, Vec<Probe>) {
    let label = path
        .file_name()
        .map_or_else(|| "?".to_string(), |s| s.to_string_lossy().into_owned());
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) => {
            return (
                label,
                vec![Probe::fail(
                    "load",
                    format!("cannot read {}: {err}", path.display()),
                )],
            );
        }
    };
    let doc: Value = match serde_json::from_str(&text) {
        Ok(doc) => doc,
        Err(err) => return (label, vec![Probe::fail("load", format!("bad JSON: {err}"))]),
    };
    let time: Vec<f64> = doc["time"].as_array().map_or(Vec::new(), |t| {
        t.iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect()
    });
    let n = time.len();
    let probes = vec![
        probe_schema(&doc),
        probe_termination(&doc),
        probe_grid(&doc, &time),
        probe_decimation(&doc, n),
        probe_shapes(&doc, n),
        probe_summary(&doc),
        probe_energy(&doc, n),
        probe_momentum(&doc, n),
        probe_mass(&doc, n),
        probe_cycle(&doc, &time),
        probe_hgfrac(&doc),
        probe_tstep(&doc),
        probe_shells(&doc, n),
    ];
    (label, probes)
}

fn main() -> ExitCode {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens/miniqa/CRASH");
    let cases = [
        root.join("crashsection/box1_42.json"),
        root.join("crashtriangles.json"),
    ];
    println!(
        "qa-fem-crash-shells: lane 3A G-leg, golden-integrity only \
         (CRASH/crashsection+crashtriangles; separate native plate gate follows)"
    );
    println!("goldens root: {}", root.display());
    let mut total = 0;
    let mut failed = 0;
    for path in &cases {
        let (label, probes) = run_case(path);
        println!("--- {label} ({})", path.display());
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
    println!(
        "summary: {} passed, {} failed of {total}",
        total - failed,
        failed
    );
    let mut integrity = QaResult::passed("fem.crash-shells-integrity.v1")
        .formulation("external CRASH golden schema and harvested-series self-consistency only")
        .dimension("two external CRASH fixtures; no native field")
        .reference("checked-in CRASH golden series and schema")
        .reference_checksum("checked per-fixture harvested series")
        .metric("numerical_solve_executed", 0.0)
        .maximum("numerical_solve_executed", 0.0)
        .metric("failed_probes", failed as f64)
        .maximum("failed_probes", 0.0);
    for path in &cases {
        match std::fs::read(path) {
            Ok(bytes) => {
                let checksum = format!("{:x}", Sha256::digest(bytes));
                integrity = integrity.artifact_with_provenance(
                    path.strip_prefix(env!("CARGO_MANIFEST_DIR"))
                        .map(|relative| relative.display().to_string())
                        .unwrap_or_else(|_| "goldens/miniqa/CRASH/unknown.json".into()),
                    "external CRASH miniqa engine golden",
                    "harvested TH series; no native CRASH evolution",
                    "opencourant-golden-v1",
                    checksum,
                );
            }
            Err(e) => {
                eprintln!("CRASH artifact read failed: {e}");
                failed += 1;
            }
        }
    }
    if failed != 0 {
        integrity.status = sonium_qa::QaStatus::Failed;
    }
    match integrity.to_json_line() {
        Ok(line) => println!("{line}"),
        Err(e) => {
            eprintln!("integrity reporting failed: {e}");
            failed += 1;
        }
    }
    match shell_plate_golden::run().and_then(|r| r.to_json_line().map_err(|e| e.to_string())) {
        Ok(line) => println!("{line}"),
        Err(e) => {
            eprintln!("native shell gate failed: {e}");
            failed += 1;
        }
    }
    match crash_box_golden::run().and_then(|r| r.to_json_line().map_err(|e| e.to_string())) {
        Ok(line) => println!("{line}"),
        Err(e) => {
            eprintln!("native original impact gate failed: {e}");
            failed += 1;
        }
    }
    if failed == 0 {
        println!("SUCCESS: qa-fem-crash-shells");
        ExitCode::SUCCESS
    } else {
        println!("FAILED: qa-fem-crash-shells");
        ExitCode::FAILURE
    }
}
