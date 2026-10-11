//! Phase-5 lane 5C G-leg: ALE2D/bar2_v41 golden audit + deferral.
//!
//! Offline native-vs-harvested-golden leg for `ALE2D/bar2_v41`. A full
//! integrator replay against vendor TH channels is NOT claimed:
//! engine time histories cannot be reproduced to the vendor tolerance
//! band (relative 6.6e-4) by an independent integrator, so the physics
//! comparison is deferred per P5C3 (see `VERDICT` below). What this leg
//! does offline is real work, not a placeholder: it schema-validates
//! the checked-in golden, verifies grid/channel integrity, and audits
//! the golden's own conservation invariants (mass constancy, copper
//! part tracking, total-energy closure, sealed inlets, impact momentum
//! sign, plastic-work monotonicity, cycle scalars) using only the
//! golden's fields and `SCHEMA.md` rules. Exit 0 = leg executed
//! (audits listed per probe); exit 1 = audit failure.

// Rust guideline compliant 2026-02-21

use std::collections::BTreeMap;
use std::process::ExitCode;

use serde::Deserialize;

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

    println!("VERDICT: DEFERRED (P5C3, recorded 2026-10-11)");
    println!("  missing: native explicit-dynamics ALE solver with /MAT/LAW3");
    println!("    elastoplastic response, /ALE/GRID/DONEA grid motion, /ALE/BCS");
    println!("    walls, and /TH/PART+N+QUAD probe extraction calibrated to");
    println!("    engine transport and hourglass treatment");
    println!("  replay target: qa-fem-ale --series vs ALE2D/bar2_v41 INTERNAL ENERGY,");
    println!("    Z-MOMENTUM, and th_2Dsolid/th_nodes channels at vendor tolerance");
    if report.failures > 0 {
        println!(
            "G-LEG qa-fem-ale: FAIL ({} probe failures)",
            report.failures
        );
        println!("FAILED: qa-fem-ale");
        std::process::exit(1);
    }
    println!("G-LEG qa-fem-ale: audits pass; physics replay deferred per P5C3");
    println!("SUCCESS: qa-fem-ale");
    ExitCode::SUCCESS
}
