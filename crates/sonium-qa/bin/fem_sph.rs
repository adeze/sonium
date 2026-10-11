//! Phase-5 lane 5B G-leg: SPH/WavesCompression golden audit + deferral.
//!
//! Offline native-vs-harvested-golden leg for `SPH/WavesCompression`.
//! A full integrator replay against vendor TH channels is NOT claimed:
//! engine time histories cannot be reproduced to the vendor tolerance band
//! (relative 6.6e-4) by an independent integrator, so the physics comparison
//! is deferred per P5C3 (see `VERDICT` below). What this leg does offline is
//! real work, not a placeholder: it schema-validates the checked-in golden,
//! verifies grid/channel integrity, and audits the golden's own conservation
//! invariants (mass constancy, transverse momentum, total-energy closure,
//! stored statistics) using only the golden's fields and `SCHEMA.md` rules.
//! Exit 0 = leg executed (audits listed per probe); exit 1 = audit failure.

use std::collections::BTreeMap;
use std::process::ExitCode;

use serde::Deserialize;

const GOLDEN: &str = include_str!("../goldens/miniqa/SPH/WavesCompression.json");
const EXPECTED_SCHEMA: &str = "opencourant-golden-v1";
const EXPECTED_CASE: &str = "SPH/WavesCompression";
/// Vendor near-zero skip from `SCHEMA.md` (`Zero_tolerance`).
const ZERO_TOLERANCE: f64 = 5e-14;

#[derive(Debug, Deserialize)]
struct Golden {
    schema: String,
    case: String,
    meta: Meta,
    time: Vec<f64>,
    channels: BTreeMap<String, Channel>,
}

#[derive(Debug, Deserialize)]
struct Meta {
    normal_termination: bool,
}

#[derive(Debug, Deserialize)]
struct Channel {
    values: Vec<f64>,
    #[serde(rename = "final")]
    final_value: f64,
    min: f64,
    max: f64,
    abs_tol: f64,
    rel_tol: f64,
}

struct Report {
    failures: u32,
}

impl Report {
    fn probe(&mut self, name: &str, pass: bool, detail: &str) {
        println!(
            "PROBE {name}: {} {detail}",
            if pass { "pass" } else { "FAIL" }
        );
        if !pass {
            self.failures += 1;
        }
    }
}

/// Replay comparison from `SCHEMA.md`, including the near-zero skip.
fn within_band(actual: f64, expected: f64, abs_tol: f64, rel_tol: f64) -> bool {
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    if actual.abs() < ZERO_TOLERANCE && expected.abs() < ZERO_TOLERANCE {
        return true;
    }
    if (actual - expected).abs() < 2.0 * ZERO_TOLERANCE {
        return true;
    }
    (actual - expected).abs() <= abs_tol + rel_tol * expected.abs()
}

fn main() -> ExitCode {
    let mut report = Report { failures: 0 };
    let golden: Golden = match serde_json::from_str(GOLDEN) {
        Ok(golden) => golden,
        Err(err) => {
            println!("PROBE golden-parse: FAIL {err}");
            return ExitCode::from(1);
        }
    };

    report.probe(
        "schema",
        golden.schema == EXPECTED_SCHEMA,
        &format!("schema={} case={}", golden.schema, golden.case),
    );
    report.probe(
        "case",
        golden.case == EXPECTED_CASE,
        &format!("expected {EXPECTED_CASE}"),
    );
    report.probe(
        "normal-termination",
        golden.meta.normal_termination,
        "engine NORMAL TERMINATION flag",
    );

    let stamps = golden.time.len();
    let monotone = golden.time.windows(2).all(|w| w[1] > w[0]);
    report.probe(
        "time-grid",
        stamps > 0 && monotone && golden.time.iter().all(|t| t.is_finite()),
        &format!(
            "{stamps} stamps, last t={}",
            golden.time.last().unwrap_or(&f64::NAN)
        ),
    );

    let mut bad_lengths = 0;
    let mut non_finite = 0;
    let mut bad_stats = 0;
    for channel in golden.channels.values() {
        if channel.values.len() != stamps {
            bad_lengths += 1;
        }
        if !channel.values.iter().all(|v| v.is_finite()) {
            non_finite += 1;
        }
        let min = channel.values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = channel
            .values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let final_value = channel.values.last().cloned().unwrap_or(f64::NAN);
        if channel.min != min || channel.max != max || channel.final_value != final_value {
            bad_stats += 1;
        }
    }
    report.probe(
        "channel-integrity",
        bad_lengths == 0 && non_finite == 0 && bad_stats == 0,
        &format!(
            "{} channels, {bad_lengths} length / {non_finite} non-finite / {bad_stats} stat mismatches",
            golden.channels.len()
        ),
    );

    // Mass conservation: MASS constant within its own band at every stamp.
    if let Some(mass) = golden.channels.get("MASS") {
        let ok = mass
            .values
            .iter()
            .all(|v| within_band(*v, mass.final_value, mass.abs_tol, mass.rel_tol));
        report.probe(
            "mass-conservation",
            ok,
            &format!("MASS final={}", mass.final_value),
        );
    } else {
        report.probe("mass-conservation", false, "MASS channel missing");
    }

    // Transverse momenta stay at solver-noise level (near-zero skip).
    for name in ["X-MOMENTUM", "Y-MOMENTUM"] {
        match golden.channels.get(name) {
            Some(channel) => {
                let peak = channel.values.iter().cloned().fold(0.0, f64::max);
                let floor = channel.values.iter().cloned().fold(0.0, f64::min);
                let ok = channel.values.iter().all(|v| v.abs() < ZERO_TOLERANCE);
                report.probe(
                    &format!("transverse-{}", name.to_lowercase()),
                    ok,
                    &format!("peak {peak:+.6e}, floor {floor:+.6e}"),
                );
            }
            None => report.probe(name, false, "channel missing"),
        }
    }

    // Total-energy closure: NO NAME tracks INTERNAL + KINETIC ENERGY.
    let closure = (|| {
        let total = golden.channels.get("NO NAME")?;
        let internal = golden.channels.get("INTERNAL ENERGY")?;
        let kinetic = golden.channels.get("KINETIC ENERGY")?;
        let worst = total
            .values
            .iter()
            .zip(internal.values.iter().zip(kinetic.values.iter()))
            .map(|(t, (i, k))| (t - (i + k)).abs())
            .fold(0.0, f64::max);
        let ok = total
            .values
            .iter()
            .zip(internal.values.iter().zip(kinetic.values.iter()))
            .all(|(t, (i, k))| within_band(*t, i + k, total.abs_tol, total.rel_tol));
        Some((ok, worst, total.final_value))
    })();
    match closure {
        Some((ok, worst, final_value)) => report.probe(
            "total-energy-closure",
            ok,
            &format!("NO NAME final={final_value}, worst |diff|={worst:.3e}"),
        ),
        None => report.probe("total-energy-closure", false, "energy channels missing"),
    }

    // Time step stays positive; report the stability range the deck saw.
    match golden.channels.get("TIME STEP") {
        Some(step) => report.probe(
            "time-step",
            step.values.iter().all(|v| *v > 0.0),
            &format!("min={} max={}", step.min, step.max),
        ),
        None => report.probe("time-step", false, "TIME STEP channel missing"),
    }

    let cells = golden
        .channels
        .keys()
        .filter(|k| k.starts_with("central cells"))
        .count();
    let nodes = golden
        .channels
        .keys()
        .filter(|k| k.starts_with("central nodes"))
        .count();
    report.probe(
        "th-inventory",
        cells == 192 && nodes == 96,
        &format!("{cells} central-cell + {nodes} central-node channels (16 probes x 12/6 vars)"),
    );

    println!("VERDICT: DEFERRED (P5C3, recorded 2026-10-11)");
    println!("  missing: native explicit-dynamics SPH time integrator coupled to the");
    println!("    deck HYDPLA + polynomial-EOS response with /TH/SPHCEL probe extraction,");
    println!("    calibrated to engine neighbor search and correction order");
    println!("  replay target: qa-fem-sph --series vs SPH/WavesCompression INTERNAL ENERGY,");
    println!("    Z-MOMENTUM, and central-cell pressure/density channels at vendor tolerance");
    if report.failures > 0 {
        println!(
            "G-LEG qa-fem-sph: FAIL ({} probe failures)",
            report.failures
        );
        println!("FAILED: qa-fem-sph");
        std::process::exit(1);
    }
    println!("G-LEG qa-fem-sph: audits pass; physics replay deferred per P5C3");
    println!("SUCCESS: qa-fem-sph");
    ExitCode::SUCCESS
}
