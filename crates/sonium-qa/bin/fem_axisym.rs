//! Phase-3 lane 3E G-leg (lane-owned): axisymmetric solids.
//!
//! Native-vs-harvested-golden comparison for axisymmetric SOLIDES
//! miniqa cases. Offline: reads only `goldens/miniqa/**/*.json` plus
//! the schema rules. Exit 0 = PASS, exit 1 = FAIL with per-probe
//! report on stdout.
//!
//! Audit result (W-only leg): no harvested SOLIDES golden is an
//! axisymmetric case. All seven carry 3D `/BRICK` decks (`N2D3D = 0`;
//! no `/QUAD` card anywhere under vendor `miniqa/SOLIDES/`), so per
//! the Phase-3 contracts this leg runs integrity + presence probes
//! instead of a native comparison:
//! - S0 enumerates `goldens/miniqa/SOLIDES/**/*.json` recursively.
//! - S1 validates each golden against `goldens/SCHEMA.md`: schema
//!   tag, case/path identity, strictly increasing finite time grid,
//!   per-channel sample counts, finite samples, tolerances present,
//!   and normal engine termination.
//! - S2 scans case ids, rootnames, and channel keys for axisymmetric
//!   markers and fails closed on any new unaudited SOLIDES file, so
//!   a future axisymmetric harvest cannot slip past this leg.
//!
//! G-leg independence holds trivially: this bin uses only `std` plus
//! `serde_json` (no fem runtimes of any lane).

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use std::path::{Path, PathBuf};

/// Golden schema accepted by this leg (see `goldens/SCHEMA.md`).
const GOLDEN_SCHEMA: &str = "opencourant-golden-v1";

/// Audited SOLIDES golden set: case ids relative to `miniqa/`, with
/// the vendor starter-deck evidence that each is a 3D `/BRICK` case
/// (`N2D3D = 0`, no `/QUAD` under vendor `miniqa/SOLIDES/`).
const AUDITED_CASES: [&str; 7] = [
    "SOLIDES/TYPE6_LAW28",
    "SOLIDES/inibri_stress",
    "SOLIDES/nv_elts/poutre_flex/flexion/H101",
    "SOLIDES/nv_elts/poutre_flex/flexion/H112",
    "SOLIDES/nv_elts/poutre_flex/flexion/HEPH/icpre_1",
    "SOLIDES/nv_elts/poutre_flex/flexion/HSEPH",
    "SOLIDES/nv_elts/poutre_flex/flexion/PA6",
];

/// Case-insensitive whole-token markers of axisymmetric content in
/// case ids, rootnames, and channel keys. Whole-token matching keeps
/// the standard `SPRING ENERGY` channel from tripping the `ring`
/// marker. `/QUAD` deck-card spellings match as substrings.
const AXISYM_MARKERS: [&str; 5] = ["axisym", "quad", "n2d3d", "taylor", "ring"];

fn goldens_root() -> PathBuf {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        return PathBuf::from(&args[1]);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens")
}

/// Collects `*.json` files under `dir` recursively (lexical order).
fn collect_json(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries =
        std::fs::read_dir(dir).map_err(|err| format!("cannot list {}: {err}", dir.display()))?;
    let mut entries: Vec<_> = entries
        .collect::<Result<_, _>>()
        .map_err(|err| format!("cannot list {}: {err}", dir.display()))?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_json(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "json") {
            out.push(path);
        }
    }
    Ok(())
}

fn load_golden(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| format!("cannot read {}: {err}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|err| format!("{} is not valid JSON: {err}", path.display()))
}

/// Validates one golden against the schema rules; returns the case id.
fn check_schema(doc: &Value, path: &Path, miniqa: &Path) -> Result<String, String> {
    if doc["schema"].as_str().unwrap_or("") != GOLDEN_SCHEMA {
        return Err(format!("{}: schema mismatch", path.display()));
    }
    let case = doc["case"]
        .as_str()
        .ok_or_else(|| format!("{}: golden lacks case id", path.display()))?;
    let relative = path
        .strip_prefix(miniqa)
        .map_err(|_| format!("{}: outside miniqa root", path.display()))?;
    let expected = relative.with_extension("");
    if Path::new(case) != expected {
        return Err(format!(
            "{}: case id {case} does not match path",
            path.display()
        ));
    }
    let time = doc["time"]
        .as_array()
        .ok_or_else(|| format!("{case}: golden lacks time grid"))?;
    if time.is_empty() && doc["meta"]["th_note"].is_null() {
        return Err(format!("{case}: empty grid without th_note"));
    }
    let mut prev = f64::NEG_INFINITY;
    for sample in time {
        let t = sample
            .as_f64()
            .ok_or_else(|| format!("{case}: non-numeric time sample"))?;
        if !t.is_finite() || t <= prev {
            return Err(format!("{case}: time grid not strictly increasing"));
        }
        prev = t;
    }
    let channels = doc["channels"]
        .as_object()
        .ok_or_else(|| format!("{case}: golden lacks channels"))?;
    for (name, channel) in channels {
        let values = channel["values"]
            .as_array()
            .ok_or_else(|| format!("{case}: channel {name} lacks values"))?;
        if values.len() != time.len() {
            return Err(format!(
                "{case}: channel {name} has {} samples for {} stamps",
                values.len(),
                time.len()
            ));
        }
        if values
            .iter()
            .any(|v| !v.as_f64().is_some_and(f64::is_finite))
        {
            return Err(format!("{case}: channel {name} has non-finite samples"));
        }
        for key in ["abs_tol", "rel_tol"] {
            let tol = channel[key]
                .as_f64()
                .ok_or_else(|| format!("{case}: channel {name} lacks {key}"))?;
            if !tol.is_finite() || tol < 0.0 {
                return Err(format!("{case}: channel {name} has bad {key}"));
            }
        }
    }
    if doc["meta"]["normal_termination"].as_bool() != Some(true) {
        return Err(format!("{case}: abnormal engine termination"));
    }
    Ok(case.to_string())
}

/// Reports the first axisymmetric marker found, if any.
fn axisym_marker(case: &str, doc: &Value) -> Option<String> {
    let mut haystacks = vec![case.to_lowercase()];
    if let Some(root) = doc["meta"]["rootname"].as_str() {
        haystacks.push(root.to_lowercase());
    }
    if let Some(channels) = doc["channels"].as_object() {
        for name in channels.keys() {
            haystacks.push(name.to_lowercase());
        }
    }
    for hay in &haystacks {
        if hay.contains("/quad") {
            return Some(format!("{case}: marker /quad in {hay}"));
        }
        for token in hay.split(|c: char| !c.is_ascii_alphanumeric()) {
            for marker in AXISYM_MARKERS {
                if token == marker {
                    return Some(format!("{case}: marker {marker} in {hay}"));
                }
            }
        }
    }
    None
}

fn main() {
    let root = goldens_root();
    let miniqa = root.join("miniqa");
    let solides = miniqa.join("SOLIDES");
    let mut failures = 0;

    // S0: enumerate.
    let mut files = Vec::new();
    if let Err(err) = collect_json(&solides, &mut files) {
        println!("S0 COLLECT: FAIL ({err})");
        std::process::exit(1);
    }
    println!("S0 COLLECT: PASS ({} SOLIDES goldens)", files.len());

    // S1: schema integrity per case.
    let mut cases = Vec::new();
    for path in &files {
        match load_golden(path).and_then(|doc| check_schema(&doc, path, &miniqa)) {
            Ok(case) => {
                let doc = load_golden(path).expect("golden loads twice");
                let nch = doc["channels"].as_object().map_or(0, |c| c.len());
                let nt = doc["time"].as_array().map_or(0, Vec::len);
                println!("S1 {case}: PASS (schema ok, {nch} channels, {nt} stamps)");
                cases.push(case);
            }
            Err(err) => {
                println!("S1 {}: FAIL ({err})", path.display());
                failures += 1;
            }
        }
    }

    // S2: axisymmetry presence audit + fail-closed set check.
    for case in &cases {
        let path = miniqa.join(format!("{case}.json"));
        let doc = load_golden(&path).expect("audited golden loads");
        match axisym_marker(case, &doc) {
            Some(hit) => {
                println!("S2 {case}: FAIL (axisymmetric marker: {hit})");
                failures += 1;
            }
            None => println!("S2 {case}: PASS (no axisymmetric markers)"),
        }
    }
    for case in &cases {
        if !AUDITED_CASES.contains(&case.as_str()) {
            println!("S2 {case}: FAIL (unaudited SOLIDES file: lane must extend probes)");
            failures += 1;
        }
    }
    for wanted in AUDITED_CASES {
        if !cases.iter().any(|case| case == wanted) {
            println!("S2 {wanted}: FAIL (audited golden missing)");
            failures += 1;
        }
    }

    if failures == 0 {
        println!(
            "VERDICT: PASS (W-only: no axisymmetric SOLIDES golden exists; \
             all {} audited cases are 3D /BRICK)",
            AUDITED_CASES.len()
        );
    } else {
        println!("VERDICT: FAIL ({failures} probe failures)");
        std::process::exit(1);
    }
}
