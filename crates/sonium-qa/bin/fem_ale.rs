//! Phase-7 lane 7D leg: ALE2D/bar2_v41 audit, native series, and
//! vendor-tolerance replay.
//!
//! Offline leg for `ALE2D/bar2_v41`. Default mode schema-validates the
//! checked-in golden, verifies grid/channel integrity, audits the
//! golden's own conservation invariants (mass constancy, copper part
//! tracking, total-energy closure, sealed inlets, impact momentum
//! sign, plastic-work monotonicity, cycle scalars) using only the
//! golden's fields and `SCHEMA.md` rules, then runs the native impact
//! slab and probes its invariants (mass and energy conservation,
//! rebound, wall pulse, quiet transverse momenta).
//!
//! `--series OUT` additionally writes the native series in TH-CSV
//! shape. `--replay CSV` replays a provided (engine) series against
//! the harvested INTERNAL ENERGY, Z-MOMENTUM, th_2Dsolid, and
//! th_nodes channels at vendor tolerance; bare `--replay` runs the
//! offline CSV-roundtrip self-test instead. The native series is
//! qualitative (documented invariant bands); only engine series are
//! claimed at vendor tolerance.
//! Exit 0 = leg executed (checks listed per probe); exit 1 = failure.

// Rust guideline compliant 2026-02-21

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use serde::Deserialize;
use sonium_fem::physics::fluid::ale::{BAR2_END_TIME, ImpactSlab};
use sonium_fem::physics::fluid::sph::Series;
use sonium_qa::rad::oc_golden::{
    CompareOptions, Golden as OcGolden, compare_history, parse_th_csv,
};

const GOLDEN: &str = include_str!("../goldens/miniqa/ALE2D/bar2_v41.json");
const EXPECTED_SCHEMA: &str = "opencourant-golden-v1";
const EXPECTED_CASE: &str = "ALE2D/bar2_v41";
/// Vendor near-zero skip from `SCHEMA.md` (`Zero_tolerance`).
const ZERO_TOLERANCE: f64 = 5e-14;

#[derive(Debug, Deserialize)]
struct Golden {
    schema: String,
    case: String,
    meta: Meta,
    time: Vec<f64>,
    channels: BTreeMap<String, Channel>,
    cycle: BTreeMap<String, f64>,
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
    name = "qa-fem-ale",
    about = "Audit ALE2D/bar2_v41, emit the native series, replay engine series"
)]
struct Args {
    /// Write the native impact-slab series to OUT (TH-CSV shape).
    #[arg(long)]
    series: Option<PathBuf>,
    /// Replay CSV against the harvested golden at vendor tolerance;
    /// without a path, runs the offline CSV-roundtrip self-test.
    #[arg(long, num_args(0..=1))]
    replay: Option<Option<PathBuf>>,
}

/// Native series stamp count (subsamples the deck run length).
const NATIVE_STAMPS: usize = 200;

/// Native CFL number (matches the unit-test characterization).
const NATIVE_CFL: f64 = 0.5;

/// Native mass drift band (closed box; measured 1.1e-15).
const NATIVE_MASS_BAND: f64 = 1.0e-12;

/// Native total-energy drift band (closed box; measured 1.7e-6).
const NATIVE_ENERGY_BAND: f64 = 1.0e-4;

/// Native transverse momentum band (1D impact; measured 1.8e-12).
const NATIVE_TRANSVERSE_BAND: f64 = 1.0e-6;

/// Golden channels replayed: globals plus TH probes.
fn is_replay_target(key: &str) -> bool {
    key == "INTERNAL ENERGY"
        || key == "Z-MOMENTUM"
        || key.contains("th_2Dsolid")
        || key.contains("th_nodes")
}

/// Runs the native impact slab over the deck run length.
fn run_native() -> Result<Series, String> {
    let mut run = ImpactSlab::bar2_analog();
    run.run(BAR2_END_TIME, NATIVE_STAMPS, NATIVE_CFL)
        .map_err(|err| format!("{err:?}"))
}

/// Probes native-series invariants (qualitative bands, not vendor).
fn probe_native(report: &mut Report, series: &Series) {
    let mass = series.channel("MASS").unwrap_or(&[]);
    let mass_drift = if mass.is_empty() {
        f64::NAN
    } else {
        mass.iter()
            .map(|m| ((m - mass[0]) / mass[0]).abs())
            .fold(0.0f64, f64::max)
    };
    report.probe(
        "native-mass",
        mass_drift.is_finite() && mass_drift < NATIVE_MASS_BAND,
        &format!("worst drift {mass_drift:.3e} vs band {NATIVE_MASS_BAND:.0e}"),
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
        &format!("worst drift {drift:.3e} vs band {NATIVE_ENERGY_BAND:.0e}"),
    );
    let zmomentum = series.channel("Z-MOMENTUM").unwrap_or(&[]);
    let peak = zmomentum.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    report.probe(
        "native-rebound",
        !zmomentum.is_empty() && zmomentum[0] < 0.0 && peak > 0.5 * -zmomentum[0],
        &format!(
            "launch {} peak {peak}",
            zmomentum.first().unwrap_or(&f64::NAN)
        ),
    );
    let wall = series.channel("WALL PRESSURE").unwrap_or(&[]);
    let wall_peak = wall.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    report.probe(
        "native-wall-pulse",
        !wall.is_empty() && wall_peak > 1.05 * wall[0],
        &format!(
            "base {} peak {wall_peak:.6e}",
            wall.first().unwrap_or(&f64::NAN)
        ),
    );
    let quiet = series
        .channel("Y-MOMENTUM")
        .is_some_and(|channel| channel.iter().all(|v| v.abs() < NATIVE_TRANSVERSE_BAND))
        && series
            .channel("X-MOMENTUM")
            .is_some_and(|channel| channel.iter().all(|v| *v == 0.0));
    report.probe("native-transverse", quiet, "1D impact stays quiet");
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
    let solids = golden
        .channels
        .keys()
        .filter(|k| k.contains("th_2Dsolid"))
        .count();
    let nodes = golden
        .channels
        .keys()
        .filter(|k| k.contains("th_nodes"))
        .count();
    let globals = ["INTERNAL ENERGY", "Z-MOMENTUM"]
        .iter()
        .all(|name| golden.channels.contains_key(*name));
    report.probe(
        "replay-targets",
        globals && solids == 55 && nodes == 36,
        &format!(
            "{} target channels (2 globals + {solids} solid + {nodes} node)",
            golden.channels.len()
        ),
    );
    if !(globals && solids == 55 && nodes == 36) {
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
            let path = std::env::temp_dir().join("qa-fem-ale-replay-selftest.csv");
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
        stamps == 800 && monotone && golden.time.iter().all(|t| t.is_finite()),
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
        golden.channels.len() == 123 && bad_lengths == 0 && non_finite == 0 && bad_stats == 0,
        &format!(
            "{} channels, {bad_lengths} length / {non_finite} non-finite / {bad_stats} stat mismatches",
            golden.channels.len()
        ),
    );

    // TH inventory: 5 probed solids x 11 vars, 6 nodes x 6 vars, 32 bare.
    let solids = golden
        .channels
        .keys()
        .filter(|k| k.contains("th_2Dsolid"))
        .count();
    let nodes = golden
        .channels
        .keys()
        .filter(|k| k.contains("th_nodes"))
        .count();
    report.probe(
        "th-inventory",
        solids == 55 && nodes == 36,
        &format!("{solids} solid + {nodes} node channels (5x11, 6x6 probes)"),
    );

    // Mass conservation: MASS and CUIVRE MASS constant within band.
    for (probe, name) in [("mass", "MASS"), ("copper-mass", "CUIVRE MASS")] {
        match golden.channels.get(name) {
            Some(mass) => {
                let ok = mass
                    .values
                    .iter()
                    .all(|v| within_band(*v, mass.final_value, mass.abs_tol, mass.rel_tol));
                report.probe(probe, ok, &format!("{name} final={}", mass.final_value));
            }
            None => report.probe(probe, false, "channel missing"),
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

    // Copper is the only part: CUIVRE IE tracks INTERNAL ENERGY.
    let copper_ie = (|| {
        let copper = golden.channels.get("CUIVRE IE")?;
        let internal = golden.channels.get("INTERNAL ENERGY")?;
        let ok = copper
            .values
            .iter()
            .zip(internal.values.iter())
            .all(|(c, i)| within_band(*c, *i, copper.abs_tol, copper.rel_tol));
        Some((ok, copper.final_value))
    })();
    match copper_ie {
        Some((ok, final_value)) => report.probe(
            "copper-ie-tracks-internal",
            ok,
            &format!("CUIVRE IE final={final_value}"),
        ),
        None => report.probe("copper-ie-tracks-internal", false, "channels missing"),
    }

    // Sealed domain: no inlet/outlet transport anywhere.
    let mut sealed_ok = true;
    for name in ["INLET MASS", "OUTLET MASS", "INLET ENERGY", "OUTLET ENERGY"] {
        match golden.channels.get(name) {
            Some(channel) => {
                sealed_ok &= channel.values.iter().all(|v| *v == 0.0);
            }
            None => sealed_ok = false,
        }
    }
    report.probe(
        "sealed-domain",
        sealed_ok,
        "inlet/outlet mass+energy all zero",
    );

    // Impact signature: no x motion; z starts toward the wall (deck
    // Vz < 0) and rebounds.
    let xmom_zero = golden
        .channels
        .get("X-MOMENTUM")
        .is_some_and(|c| c.values.iter().all(|v| *v == 0.0));
    let zmom = golden.channels.get("Z-MOMENTUM");
    let (zmom_ok, zmom_detail) = match zmom {
        Some(channel) => (
            channel.values.first().is_some_and(|v| *v < 0.0),
            format!(
                "first={} last={}",
                channel.values.first().unwrap_or(&f64::NAN),
                channel.final_value
            ),
        ),
        None => (false, "Z-MOMENTUM missing".to_string()),
    };
    report.probe("impact-momentum", xmom_zero && zmom_ok, &zmom_detail);

    // No erosion, no hourglass, no external work, no HE/HEAT leakage.
    let mut clean_ok = true;
    for name in [
        "CUIVRE ERODED",
        "CUIVRE HE",
        "CUIVRE HEAT",
        "HOURGLASS ENERGY",
        "EXTERNAL WORK",
    ] {
        match golden.channels.get(name) {
            Some(channel) => {
                clean_ok &= channel.values.iter().all(|v| *v == 0.0);
            }
            None => clean_ok = false,
        }
    }
    report.probe(
        "no-erosion",
        clean_ok,
        "eroded/HE/HEAT/hourglass/extwork all zero",
    );

    // Plastic work grows monotonically (elastoplastic deck).
    match golden.channels.get("PLASTIC WORK") {
        Some(work) => {
            let monotone = work.values.windows(2).all(|w| w[1] >= w[0]);
            report.probe(
                "plastic-work",
                monotone,
                &format!("monotone, final={}", work.final_value),
            );
        }
        None => report.probe("plastic-work", false, "PLASTIC WORK channel missing"),
    }

    // Time step stays positive; report the stability range.
    match golden.channels.get("TIME STEP") {
        Some(step) => report.probe(
            "time-step",
            step.values.iter().all(|v| *v > 0.0),
            &format!("min={} max={}", step.min, step.max),
        ),
        None => report.probe("time-step", false, "TIME STEP channel missing"),
    }

    // Last-cycle scalars: mass error at roundoff, internal energy
    // consistent with the TH final within the vendor band.
    let cycle_ok = (|| {
        let mass_error = *golden.cycle.get("MASERR")?;
        let cycle = *golden.cycle.get("CYCLE")?;
        let ienergy = *golden.cycle.get("IENERGY")?;
        let internal = golden.channels.get("INTERNAL ENERGY")?;
        Some((
            mass_error.abs() < 1.0e-12
                && cycle == 4300.0
                && within_band(
                    ienergy,
                    internal.final_value,
                    internal.abs_tol,
                    internal.rel_tol,
                ),
            format!("CYCLE={cycle} MASERR={mass_error:.3e} IENERGY={ienergy}"),
        ))
    })();
    match cycle_ok {
        Some((ok, detail)) => report.probe("cycle-scalars", ok, &detail),
        None => report.probe("cycle-scalars", false, "cycle scalars missing"),
    }

    // Native impact slab over the deck run length, then its
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
    println!("  native: impact-slab --series output with invariant probes");
    println!("    (mass/energy conservation, rebound, wall pulse)");
    println!("  replay: --replay CSV vs INTERNAL ENERGY, Z-MOMENTUM, and");
    println!("    th_2Dsolid/th_nodes channels at vendor tolerance; bare");
    println!("    --replay runs the offline CSV-roundtrip self-test");
    if report.failures > 0 {
        println!(
            "G-LEG qa-fem-ale: FAIL ({} probe failures)",
            report.failures
        );
        println!("FAILED: qa-fem-ale");
        std::process::exit(1);
    }
    println!("G-LEG qa-fem-ale: audits, native series, and replay pass");
    println!("SUCCESS: qa-fem-ale");
    ExitCode::SUCCESS
}
