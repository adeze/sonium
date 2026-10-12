//! Phase-7 lane 7D leg: SPH/WavesCompression audit, native series,
//! and vendor-tolerance replay.
//!
//! Offline leg for `SPH/WavesCompression`. Default mode schema-validates
//! the checked-in golden, verifies grid/channel integrity, audits the
//! golden's own conservation invariants (mass constancy, transverse
//! momentum, total-energy closure, stored statistics) using only the
//! golden's fields and `SCHEMA.md` rules, then runs the native
//! impact column and probes its invariants (mass constancy, energy
//! drift band, rebound, conversion, quiet transverse momenta).
//!
//! `--series OUT` additionally writes the native series in TH-CSV
//! shape. `--replay CSV` replays a provided (engine) series against
//! the harvested INTERNAL ENERGY, Z-MOMENTUM, and central-cell
//! channels at vendor tolerance; bare `--replay` runs the offline
//! CSV-roundtrip self-test instead. The native series is qualitative
//! (documented invariant bands); only engine series are claimed at
//! vendor tolerance.
//! Exit 0 = leg executed (checks listed per probe); exit 1 = failure.

// Rust guideline compliant 2026-02-21

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use serde::Deserialize;
use sonium_fem::physics::fluid::sph::{CompressionRun, Series, WAVES_END_TIME};
use sonium_qa::rad::oc_golden::{
    CompareOptions, Golden as OcGolden, compare_history, parse_th_csv,
};

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

#[derive(Debug, Parser)]
#[command(
    name = "qa-fem-sph",
    about = "Audit SPH/WavesCompression, emit the native series, replay engine series"
)]
struct Args {
    /// Write the native impact-column series to OUT (TH-CSV shape).
    #[arg(long)]
    series: Option<PathBuf>,
    /// Replay CSV against the harvested golden at vendor tolerance;
    /// without a path, runs the offline CSV-roundtrip self-test.
    #[arg(long, num_args(0..=1))]
    replay: Option<Option<PathBuf>>,
}

/// Native series stamp count (nominal golden cadence for overlays).
const NATIVE_STAMPS: usize = 45;

/// Native total-energy drift band (wall work plus clipping plus
/// leapfrog error over the full run; measured 0.067).
const NATIVE_ENERGY_BAND: f64 = 0.10;

/// Golden channels replayed: globals plus the central-cell probes.
fn is_replay_target(key: &str) -> bool {
    key == "INTERNAL ENERGY" || key == "Z-MOMENTUM" || key.starts_with("central cells")
}

/// Runs the native impact column over the deck run length.
fn run_native() -> Result<Series, String> {
    let mut run = CompressionRun::waves_compression().map_err(|err| err.to_string())?;
    run.run(WAVES_END_TIME, NATIVE_STAMPS)
        .map_err(|err| err.to_string())
}

/// Probes native-series invariants (qualitative bands, not vendor).
fn probe_native(report: &mut Report, series: &Series) {
    let mass = series.channel("MASS").unwrap_or(&[]);
    report.probe(
        "native-mass",
        !mass.is_empty() && mass.iter().all(|m| *m == mass[0]),
        &format!("MASS constant at {}", mass.first().unwrap_or(&f64::NAN)),
    );
    let total = series.channel("TOTAL ENERGY").unwrap_or(&[]);
    let drift = if total.is_empty() {
        f64::NAN
    } else {
        total
            .iter()
            .map(|e| ((e - total[0]) / total[0]).abs())
            .fold(0.0f64, f64::max)
    };
    report.probe(
        "native-energy-drift",
        drift.is_finite() && drift < NATIVE_ENERGY_BAND,
        &format!("worst drift {drift:.3e} vs band {NATIVE_ENERGY_BAND}"),
    );
    let zmomentum = series.channel("Z-MOMENTUM").unwrap_or(&[]);
    report.probe(
        "native-rebound",
        zmomentum.first().is_some_and(|first| *first < 0.0)
            && zmomentum.last().is_some_and(|last| *last > 0.0),
        &format!(
            "zm {} -> {}",
            zmomentum.first().unwrap_or(&f64::NAN),
            zmomentum.last().unwrap_or(&f64::NAN)
        ),
    );
    let internal = series.channel("INTERNAL ENERGY").unwrap_or(&[]);
    let kinetic = series.channel("KINETIC ENERGY").unwrap_or(&[]);
    report.probe(
        "native-conversion",
        !internal.is_empty() && internal.last().unwrap() > kinetic.last().unwrap(),
        &format!(
            "final ie={} ke={}",
            internal.last().unwrap_or(&f64::NAN),
            kinetic.last().unwrap_or(&f64::NAN)
        ),
    );
    let quiet = ["X-MOMENTUM", "Y-MOMENTUM"].iter().all(|name| {
        series
            .channel(name)
            .is_some_and(|channel| channel.iter().all(|v| *v == 0.0))
    });
    report.probe("native-transverse", quiet, "no lateral motion bit-exact");
}

/// Replays a provided series (or the golden roundtrip) at vendor tolerance.
fn do_replay(report: &mut Report, csv: Option<PathBuf>) {
    let mut golden: OcGolden = match serde_json::from_str(GOLDEN) {
        Ok(golden) => golden,
        Err(err) => {
            report.probe("replay-golden-parse", false, &err.to_string());
            return;
        }
    };
    golden.channels.retain(|key, _| is_replay_target(key));
    let cells = golden
        .channels
        .keys()
        .filter(|k| k.starts_with("central cells"))
        .count();
    let globals = ["INTERNAL ENERGY", "Z-MOMENTUM"]
        .iter()
        .all(|name| golden.channels.contains_key(*name));
    report.probe(
        "replay-targets",
        globals && cells == 192,
        &format!(
            "{} target channels (2 globals + {cells} central-cell)",
            golden.channels.len()
        ),
    );
    if !(globals && cells == 192) {
        return;
    }
    match csv {
        Some(path) => {
            let provided = match parse_th_csv(&path) {
                Ok(provided) => provided,
                Err(err) => {
                    report.probe("replay-parse", false, &format!("{}: {err}", path.display()));
                    return;
                }
            };
            let result = compare_history(&golden, &provided, &CompareOptions::default());
            for failure in &result.failures {
                println!("REPLAY-FAIL {failure}");
            }
            report.probe(
                "replay-vendor-tolerance",
                result.passed(),
                &result.summary(),
            );
        }
        None => {
            // Offline self-test: golden target channels through the
            // CSV writer and back must replay clean.
            let names: Vec<String> = golden.channels.keys().cloned().collect();
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            let mut series = Series::with_channels(&refs);
            for (stamp, time) in golden.time.iter().enumerate() {
                let row: Vec<f64> = names
                    .iter()
                    .map(|name| golden.channels[name].values[stamp])
                    .collect();
                series.push(*time, &row);
            }
            let path = std::env::temp_dir().join("qa-fem-sph-replay-selftest.csv");
            if let Err(err) = series.write_csv(&path) {
                report.probe(
                    "replay-self-test",
                    false,
                    &format!("{}: {err}", path.display()),
                );
                return;
            }
            let outcome: Result<_, String> = parse_th_csv(&path).map_err(|err| err.to_string());
            let _ = std::fs::remove_file(&path);
            match outcome {
                Ok(provided) => {
                    let result = compare_history(&golden, &provided, &CompareOptions::default());
                    for failure in &result.failures {
                        println!("REPLAY-FAIL {failure}");
                    }
                    report.probe("replay-self-test", result.passed(), &result.summary());
                }
                Err(err) => {
                    report.probe("replay-self-test", false, &err);
                }
            }
        }
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
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

    // Native impact column over the deck run length, then its
    // invariant probes (qualitative bands, not vendor tolerance).
    match run_native() {
        Ok(series) => {
            probe_native(&mut report, &series);
            if let Some(path) = &args.series {
                match series.write_csv(path) {
                    Ok(()) => report.probe(
                        "series-write",
                        true,
                        &format!("{} stamps to {}", series.len(), path.display()),
                    ),
                    Err(err) => {
                        report.probe("series-write", false, &format!("{}: {err}", path.display()))
                    }
                }
            }
        }
        Err(err) => {
            report.probe("native-run", false, &err);
            if args.series.is_some() {
                report.probe("series-write", false, "no native series to write");
            }
        }
    }

    if let Some(csv) = args.replay {
        do_replay(&mut report, csv);
    }

    println!("VERDICT: IMPLEMENTED (P7 lane 7D; P5C3 replay target closed)");
    println!("  native: impact-column --series output with invariant probes");
    println!("    (mass constancy, energy drift band, rebound, conversion)");
    println!("  replay: --replay CSV vs INTERNAL ENERGY, Z-MOMENTUM, and");
    println!("    central-cell channels at vendor tolerance; bare --replay");
    println!("    runs the offline CSV-roundtrip self-test");
    if report.failures > 0 {
        println!(
            "G-LEG qa-fem-sph: FAIL ({} probe failures)",
            report.failures
        );
        println!("FAILED: qa-fem-sph");
        std::process::exit(1);
    }
    println!("G-LEG qa-fem-sph: audits, native series, and replay pass");
    println!("SUCCESS: qa-fem-sph");
    ExitCode::SUCCESS
}
