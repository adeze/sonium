//! Phase-5 lane 5E G-leg: submodel/restart.
//!
//! Offline gate over the SUBMODEL and RESTART miniqa goldens plus the
//! native restart round-trip. It replays the TRANSF transform chain on
//! transcribed deck excerpts through the shipped library, checks
//! `//SUBMODEL` split/merge logic on transcribed deck skeletons, replays
//! restart continuity (run-1/run-2 join) from the checked-in goldens,
//! and proves native restart determinism (same state twice gives
//! identical bytes) through the Phase-1 `sonium-compute` restart API.
//!
//! Transcriptions (deck file, verified dev-time against the vendor tree
//! and a fresh double-precision starter run):
//!
//! * Run stops and TH intervals from the RESTART engine decks:
//!   `CRA2V44_0001.rad` stops 2.5, `CRA2V44_0002.rad` stops 5.0,
//!   `dT_HIS` 0.05; `BOULE1V4_0001.rad` stops 0.06, `BOULE1V4_0002.rad`
//!   stops 0.13, `dT_HIS` 0.001.
//! * TRANSF probe nodes (1000, 1016, 1039) plus reference nodes (1086,
//!   1089, 1095, 1114, 1115) from `BOULE1V5_0000.rad` `/NODE`, with
//!   per-stage echoes from the fresh starter `NEW NODE COORDINATES`
//!   tables.
//! * The `//SUBMODEL/1` header skeletons of the three `update_v_100`
//!   starter decks (41/77/38 cards inside, zero offsets).
//!
//! Report lines read `PASS|FAIL <id> measured=<m> bound=<b> <detail>`;
//! the process exits 0 when every probe passes, 1 otherwise.

// Rust guideline compliant 2026-02-21

use clap::Parser;
use sha2::{Digest, Sha256};
use sonium_compute::explicit_restart::{decode_restart, encode_restart};
use sonium_compute::explicit_state::ExplicitState;
use sonium_qa::rad::oc_golden::{GOLDEN_SCHEMA, Golden, golden_path, load_golden};
use sonium_qa::rad_cards::parse_header;
use sonium_qa::rad_parse::{DeckCard, RadDeck};
use sonium_qa::rad_submodel::{
    apply_transforms, collect_transforms, merge_submodels, section_inventory, split_submodels,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Phase-5 lane 5E G-leg: submodel/restart")]
struct Args {
    /// Goldens root; defaults to the crate `goldens/` dir.
    #[arg(long)]
    goldens: Option<PathBuf>,
}

/// One report line.
struct Row {
    id: String,
    pass: bool,
    measured: String,
    bound: String,
    detail: String,
}

/// Collects a report line.
fn push(
    rows: &mut Vec<Row>,
    id: &str,
    pass: bool,
    measured: String,
    bound: String,
    detail: String,
) {
    rows.push(Row {
        id: id.to_string(),
        pass,
        measured,
        bound,
        detail,
    });
}

/// Restart run-stop transcription for one case.
struct RunStops {
    case: &'static str,
    rootname: &'static str,
    starter_deck: &'static str,
    engine_decks: [&'static str; 2],
    stop_run1: f64,
    stop_run2: f64,
    dt_his: f64,
}

/// SUBMODEL golden cases with provenance expectations.
const SUBMODEL_CASES: [(&str, &str, &str, &[&str]); 4] = [
    (
        "SUBMODEL/TRANSF",
        "BOULE1V5",
        "BOULE1V5_0000.rad",
        &["BOULE1V5_0001.rad"],
    ),
    (
        "SUBMODEL/update_v_100/ACTIV",
        "HYPE_SUB",
        "HYPE_SUB_0000.rad",
        &["HYPE_SUB_0001.rad"],
    ),
    (
        "SUBMODEL/update_v_100/PLOAD",
        "BOXBEAM_SUB",
        "BOXBEAM_SUB_0000.rad",
        &["BOXBEAM_SUB_0001.rad"],
    ),
    (
        "SUBMODEL/update_v_100/FAIL/Jonhson_Cook",
        "SPECT_SUB",
        "SPECT_SUB_0000.rad",
        &["SPECT_SUB_0001.rad"],
    ),
];

/// RESTART golden cases with run-stop transcriptions.
const RESTART_CASES: [RunStops; 2] = [
    RunStops {
        case: "RESTART/cra2",
        rootname: "CRA2V44",
        starter_deck: "CRA2V44_0000.rad",
        engine_decks: ["CRA2V44_0001.rad", "CRA2V44_0002.rad"],
        stop_run1: 2.5,
        stop_run2: 5.0,
        dt_his: 0.05,
    },
    RunStops {
        case: "RESTART/int7_small_boule_rst",
        rootname: "BOULE1V4",
        starter_deck: "BOULE1V4_0000.rad",
        engine_decks: ["BOULE1V4_0001.rad", "BOULE1V4_0002.rad"],
        stop_run1: 0.06,
        stop_run2: 0.13,
        dt_his: 0.001,
    },
];

/// Reports whether a string is lowercase hex of a length.
fn is_hex(text: &str, len: usize) -> bool {
    text.len() == len && text.chars().all(|c| c.is_ascii_hexdigit())
}

/// Checks golden provenance against transcribed deck facts.
fn check_provenance(
    rows: &mut Vec<Row>,
    golden: &Golden,
    case: &str,
    rootname: &str,
    starter: &str,
    engines: &[&str],
) {
    let id = format!("provenance:{case}");
    let mut defects = Vec::new();
    if golden.schema != GOLDEN_SCHEMA {
        defects.push(format!("schema {}", golden.schema));
    }
    if golden.case != case {
        defects.push(format!("case {}", golden.case));
    }
    if !golden.meta.normal_termination {
        defects.push("no NORMAL TERMINATION".to_string());
    }
    if golden.meta.rootname != rootname {
        defects.push(format!("rootname {}", golden.meta.rootname));
    }
    if golden.meta.starter_deck != starter {
        defects.push(format!("starter {}", golden.meta.starter_deck));
    }
    let want: Vec<String> = engines.iter().map(ToString::to_string).collect();
    if golden.meta.engine_decks != want {
        defects.push(format!("engines {:?}", golden.meta.engine_decks));
    }
    if !is_hex(&golden.meta.deck_sha256.starter, 64) {
        defects.push("starter hash".to_string());
    }
    match &golden.meta.deck_sha256.engine {
        Some(hash) if is_hex(hash, 64) => {}
        _ => defects.push("engine hash".to_string()),
    }
    if !is_hex(&golden.meta.engine_commit, 40) {
        defects.push("engine commit".to_string());
    }
    if golden.meta.precision != "dp" {
        defects.push(format!("precision {}", golden.meta.precision));
    }
    let detail = if defects.is_empty() {
        format!("{} engine deck(s), hashes pinned", engines.len())
    } else {
        defects.join("; ")
    };
    push(
        rows,
        &id,
        defects.is_empty(),
        format!("{} defects", defects.len()),
        "0 defects".to_string(),
        detail,
    );
}

/// Checks the shared grid and every channel shape.
fn check_grid(rows: &mut Vec<Row>, golden: &Golden, case: &str) {
    let id = format!("grid:{case}");
    let time = &golden.time;
    let mut defects = Vec::new();
    if time.len() < 2 {
        defects.push(format!("{} stamps", time.len()));
    }
    if time.iter().any(|t| !t.is_finite()) {
        defects.push("non-finite stamp".to_string());
    }
    if time.windows(2).any(|pair| pair[1] <= pair[0]) {
        defects.push("non-increasing grid".to_string());
    }
    for (name, channel) in &golden.channels {
        if channel.values.len() != time.len() {
            defects.push(format!("{name} length {}", channel.values.len()));
        }
        if channel.values.iter().any(|v| !v.is_finite()) {
            defects.push(format!("{name} non-finite"));
        }
    }
    let detail = if defects.is_empty() {
        format!("{} stamps x {} channels", time.len(), golden.channels.len())
    } else {
        defects.join("; ")
    };
    push(
        rows,
        &id,
        defects.is_empty(),
        format!("{} defects", defects.len()),
        "0 defects".to_string(),
        detail,
    );
}

/// Channel value with missing channels read as zero.
fn series_or_zero(golden: &Golden, name: &str, index: usize) -> f64 {
    golden
        .channels
        .get(name)
        .map(|channel| channel.values[index])
        .unwrap_or(0.0)
}

/// Channel tolerance with missing channels read as zero.
fn tolerance_or_zero(golden: &Golden, name: &str, value: f64) -> f64 {
    golden
        .channels
        .get(name)
        .map(|channel| channel.abs_tol + channel.rel_tol * value.abs())
        .unwrap_or(0.0)
}

/// Checks the total-energy account identity per stamp.
///
/// `NO NAME` must equal internal plus kinetic plus hourglass plus
/// rotation plus contact minus external work within the summed
/// channel tolerances; absent terms read as zero (a nonzero absent
/// term would break the identity and fail closed).
fn check_energy_identity(rows: &mut Vec<Row>, golden: &Golden, case: &str) {
    const TERMS: [&str; 6] = [
        "INTERNAL ENERGY",
        "KINETIC ENERGY",
        "HOURGLASS ENERGY",
        "ROTATION ENERGY",
        "CONTACT ENERGY",
        "EXTERNAL WORK",
    ];
    let id = format!("energy-identity:{case}");
    let Some(total) = golden.channels.get("NO NAME") else {
        push(
            rows,
            &id,
            false,
            "missing".to_string(),
            "present".to_string(),
            "NO NAME absent".to_string(),
        );
        return;
    };
    let mut worst = 0.0_f64;
    let mut at = 0;
    for index in 0..golden.time.len() {
        let predicted = series_or_zero(golden, TERMS[0], index)
            + series_or_zero(golden, TERMS[1], index)
            + series_or_zero(golden, TERMS[2], index)
            + series_or_zero(golden, TERMS[3], index)
            + series_or_zero(golden, TERMS[4], index)
            - series_or_zero(golden, TERMS[5], index);
        let mut band = total.abs_tol + total.rel_tol * total.values[index].abs();
        for term in TERMS {
            band += tolerance_or_zero(golden, term, series_or_zero(golden, term, index));
        }
        let ratio = (total.values[index] - predicted).abs() / band.max(1e-300);
        if ratio > worst {
            worst = ratio;
            at = index;
        }
    }
    push(
        rows,
        &id,
        worst <= 1.0,
        format!("{worst:.3}"),
        "1.000".to_string(),
        format!("worst stamp {at} of {}", golden.time.len()),
    );
}

/// Checks mass constancy, or mass-minus-added for erosion cases.
fn check_mass(rows: &mut Vec<Row>, golden: &Golden, case: &str, minus_added: bool) {
    let id = format!("mass:{case}");
    let Some(mass) = golden.channels.get("MASS") else {
        push(
            rows,
            &id,
            false,
            "missing".to_string(),
            "present".to_string(),
            "MASS absent".to_string(),
        );
        return;
    };
    let added = golden.channels.get("ADDED MASS");
    let mut series = mass.values.clone();
    let mut band = 0.0_f64;
    let mut note = String::new();
    if minus_added {
        let Some(added) = added else {
            push(
                rows,
                &id,
                false,
                "missing".to_string(),
                "present".to_string(),
                "ADDED MASS absent".to_string(),
            );
            return;
        };
        for (slot, extra) in series.iter_mut().zip(added.values.iter()) {
            *slot -= extra;
        }
        note = format!(
            "t0 activation step {:.3}; ",
            mass.values[1] - mass.values[0]
        );
    }
    for (index, value) in series.iter().enumerate() {
        band = band.max(mass.abs_tol + mass.rel_tol * value.abs());
        if minus_added {
            let channel = added.expect("ADDED MASS checked");
            let extra = channel.values[index];
            band = band.max(channel.abs_tol + channel.rel_tol * extra.abs());
        }
    }
    let range = series.iter().fold(0.0_f64, |high, v| high.max(*v))
        - series.iter().fold(f64::INFINITY, |low, v| low.min(*v));
    push(
        rows,
        &id,
        range <= band,
        format!("{range:.3e}"),
        format!("{band:.3e}"),
        format!("{note}{} stamps", series.len()),
    );
}

/// Checks a part-mass sum against the global mass.
fn check_part_sum(rows: &mut Vec<Row>, golden: &Golden, case: &str, parts: &[&str]) {
    let id = format!("part-sum:{case}");
    let Some(mass) = golden.channels.get("MASS") else {
        push(
            rows,
            &id,
            false,
            "missing".to_string(),
            "present".to_string(),
            "MASS absent".to_string(),
        );
        return;
    };
    for part in parts {
        if !golden.channels.contains_key(*part) {
            push(
                rows,
                &id,
                false,
                "missing".to_string(),
                "present".to_string(),
                format!("{part} absent"),
            );
            return;
        }
    }
    let mut worst = 0.0_f64;
    for index in 0..golden.time.len() {
        let mut predicted = 0.0;
        let mut band = mass.abs_tol + mass.rel_tol * mass.values[index].abs();
        for part in parts {
            let value = series_or_zero(golden, part, index);
            predicted += value;
            band += tolerance_or_zero(golden, part, value);
        }
        worst = worst.max((mass.values[index] - predicted).abs() / band.max(1e-300));
    }
    push(
        rows,
        &id,
        worst <= 1.0,
        format!("{worst:.3}"),
        "1.000".to_string(),
        format!("{} parts", parts.len()),
    );
}

/// Checks run coverage: grid starts at zero and ends near run-2 stop.
fn check_coverage(rows: &mut Vec<Row>, golden: &Golden, stops: &RunStops) {
    let id = format!("coverage:{}", stops.case);
    let time = &golden.time;
    let Some((&first, &last)) = time.first().zip(time.last()) else {
        push(
            rows,
            &id,
            false,
            "empty".to_string(),
            "stamps".to_string(),
            "no grid".to_string(),
        );
        return;
    };
    let tail = stops.stop_run2 - last;
    let pass = first.abs() <= 1e-12 && tail > 0.0 && tail <= 2.0 * stops.dt_his;
    push(
        rows,
        &id,
        pass,
        format!("t0={first:.1e} tail={tail:.3e}"),
        format!("t0<=1e-12 tail<={:.1e}", 2.0 * stops.dt_his),
        format!("run-2 stop {}", stops.stop_run2),
    );
}

/// Checks restart-join continuity across the run-1 stop.
///
/// The join index is the first stamp past the run-1 stop: its time
/// gap must match the nominal output interval, and every channel step
/// across it must stay within its tolerance plus twice the largest
/// neighboring step (a restart that dropped state would spike here).
fn check_join(rows: &mut Vec<Row>, golden: &Golden, stops: &RunStops) {
    let id = format!("join:{}", stops.case);
    let time = &golden.time;
    let join = time.partition_point(|t| *t <= stops.stop_run1);
    if join < 1 || join + 1 >= time.len() {
        push(
            rows,
            &id,
            false,
            format!("j={join}"),
            "interior".to_string(),
            "join outside grid".to_string(),
        );
        return;
    }
    let mut gaps: Vec<f64> = time.windows(2).map(|pair| pair[1] - pair[0]).collect();
    gaps.sort_by(f64::total_cmp);
    let median = gaps[gaps.len() / 2];
    let gap = time[join] - time[join - 1];
    let gap_ok = gap >= 0.5 * median && gap <= 2.0 * median;
    let mut worst = 0.0_f64;
    let mut channel = String::new();
    for (name, series) in &golden.channels {
        let values = &series.values;
        let step = (values[join] - values[join - 1]).abs();
        let lo = join.saturating_sub(10);
        let hi = (join + 10).min(values.len() - 1);
        let mut neighbor = 0.0_f64;
        for at in lo..hi {
            if at + 1 == join {
                continue;
            }
            neighbor = neighbor.max((values[at + 1] - values[at]).abs());
        }
        let tolerance =
            series.abs_tol + series.rel_tol * values[join].abs().max(values[join - 1].abs());
        let denom = tolerance + 2.0 * neighbor;
        let score = if denom > 0.0 {
            (step - tolerance).max(0.0) / denom
        } else if step > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        if score > worst {
            worst = score;
            channel.clone_from(name);
        }
    }
    let pass = gap_ok && worst <= 1.0;
    push(
        rows,
        &id,
        pass,
        format!("gap={gap:.3e} step={worst:.3}"),
        format!("gap~{median:.1e} step<=1.000"),
        format!("j={join} t={:.6} worst {channel}", time[join]),
    );
}

/// Interpolates a series at a query time (clamped ends).
fn interpolate(time: &[f64], values: &[f64], query: f64) -> f64 {
    let mut at = time.partition_point(|t| *t < query);
    if at == 0 {
        return values[0];
    }
    if at >= time.len() {
        at = time.len() - 1;
        return values[at];
    }
    let span = time[at] - time[at - 1];
    if span <= 0.0 {
        return values[at];
    }
    let fraction = (query - time[at - 1]) / span;
    values[at - 1] + fraction * (values[at] - values[at - 1])
}

/// Checks last-cycle scalars against the series.
///
/// Cycle lines print every `/PRINT` interval with few significant
/// figures, so the band adds the local step drift plus one percent
/// of scale to the channel tolerance; this guards gross drift while
/// the join probe carries the tight restart proof.
fn check_cycle(rows: &mut Vec<Row>, golden: &Golden, case: &str) {
    const KEYS: [(&str, &str); 4] = [
        ("INTERNAL ENERGY", "IENERGY"),
        ("KINETIC ENERGY", "KENERGYT"),
        ("EXTERNAL WORK", "EXTWORK"),
        ("MASS", "TOTALMASS"),
    ];
    let id = format!("cycle:{case}");
    let Some(runtime) = golden.cycle.get("TIME").copied() else {
        push(
            rows,
            &id,
            false,
            "missing".to_string(),
            "present".to_string(),
            "cycle TIME absent".to_string(),
        );
        return;
    };
    let mut worst = 0.0_f64;
    let mut key = String::new();
    for (channel_name, cycle_key) in KEYS {
        let (Some(series), Some(scalar)) = (
            golden.channels.get(channel_name),
            golden.cycle.get(cycle_key).copied(),
        ) else {
            push(
                rows,
                &id,
                false,
                "missing".to_string(),
                "present".to_string(),
                format!("{channel_name}/{cycle_key} absent"),
            );
            return;
        };
        let reference = interpolate(&golden.time, &series.values, runtime);
        let end = series.values.len() - 1;
        let at = golden.time.partition_point(|t| *t < runtime).clamp(1, end);
        let mut step = (series.values[at] - series.values[at - 1]).abs();
        if at < end {
            step = step.max((series.values[at + 1] - series.values[at]).abs());
        }
        let band = series.abs_tol
            + series.rel_tol * reference.abs()
            + 3.0 * step
            + 0.01 * scalar.abs().max(reference.abs());
        let score = (scalar - reference).abs() / band.max(1e-300);
        if score > worst {
            worst = score;
            key.clone_from(&cycle_key.to_string());
        }
    }
    push(
        rows,
        &id,
        worst <= 1.0,
        format!("{worst:.3}"),
        "1.000".to_string(),
        format!("TIME={runtime:.4} worst {key}"),
    );
}

/// Checks total-energy conservation for a closed case.
fn check_conservation(rows: &mut Vec<Row>, golden: &Golden, case: &str, band: f64) {
    let id = format!("conservation:{case}");
    let (Some(total), Some(kinetic)) = (
        golden.channels.get("NO NAME"),
        golden.channels.get("KINETIC ENERGY"),
    ) else {
        push(
            rows,
            &id,
            false,
            "missing".to_string(),
            "present".to_string(),
            "NO NAME/KE absent".to_string(),
        );
        return;
    };
    let initial = kinetic.values[0];
    let mut worst = 0.0_f64;
    let mut at = 0;
    for (index, value) in total.values.iter().enumerate() {
        let drift = (value - initial).abs() / initial.abs().max(1e-300);
        if drift > worst {
            worst = drift;
            at = index;
        }
    }
    push(
        rows,
        &id,
        worst <= band,
        format!("{worst:.3e}"),
        format!("{band:.1e}"),
        format!("worst stamp {at} of {}", total.values.len()),
    );
}

/// TRANSF probe deck, transcribed rows.
///
/// Nodes 1000, 1016, 1039 plus the five reference nodes from
/// `BOULE1V5_0000.rad` `/NODE`; group 18 spans the three probes;
/// the four transform rows are token-identical to the deck rows.
const TRANSF_PROBE_DECK: &str = "/NODE
1000 -10.015957411335 -1.0753791459204 1.2391039848
1016 -10.103259229593 -.90077550940476 .49999998143592
1039 -10.03837446944 -1.1836183064494 -.2391039848
1086 -.899999976 .500000119 -.5
1089 -3.9000001 .500000358 -.5
1095 .50000002 -.1 .5
1114 2.10000002 -2.5 2.5
1115 .50000002 -.1 2.5
/GRNOD/NODE/18
group
1000 1016 1039
/TRANSFORM/SYM/1
sym_back
18 1089 1086
/TRANSFORM/SCA/2
scale_back
18 5 2.5 1.25 1114
/TRANSFORM/ROT/3
rot_back
18 1095 1115
-90
/TRANSFORM/TRA/4
tra_back
18 -5
/END
";

/// Per-stage probe echoes `id x y z` (SYM, SCA, ROT, TRA).
///
/// Fresh-starter `NEW NODE COORDINATES` values for the three probes.
const TRANSF_PROBE_ECHOES: [[&str; 3]; 4] = [
    [
        "1000 0.2215956960324E+01 -0.1075380120396E+01 0.1239103984800E+01",
        "1016 0.2303258806403E+01 -0.9007764977907E+00 0.4999999814359E+00",
        "1039 0.2238374001183E+01 -0.1183619284497E+01 -0.2391039848000E+00",
    ],
    [
        "1000 0.2679784721622E+01 0.1061549699009E+01 0.9238799810000E+00",
        "1016 0.3116293952013E+01 0.1498058755523E+01 -0.2320510006371E-07",
        "1039 0.2791869925917E+01 0.7909517887575E+00 -0.9238799810000E+00",
    ],
    [
        "1000 0.1661549719009E+01 -0.2279784701622E+01 0.9238799810000E+00",
        "1016 0.2098058775523E+01 -0.2716293932013E+01 -0.2320510006371E-07",
        "1039 0.1390951808758E+01 -0.2391869905917E+01 -0.9238799810000E+00",
    ],
    [
        "1000 -0.3338450280991E+01 -0.2279784701622E+01 0.9238799810000E+00",
        "1016 -0.2901941224477E+01 -0.2716293932013E+01 -0.2320510006371E-07",
        "1039 -0.3609048191242E+01 -0.2391869905917E+01 -0.9238799810000E+00",
    ],
];

/// Vendor agreement band for transcribed starter echoes.
const VENDOR_BAND: f64 = 1e-9;

/// Replays the TRANSF chain on transcribed probes.
fn check_transf_chain(rows: &mut Vec<Row>) {
    const ID: &str = "transf-chain:SUBMODEL/TRANSF";
    let deck = match RadDeck::parse_str(TRANSF_PROBE_DECK) {
        Ok(deck) => deck,
        Err(error) => {
            push(
                rows,
                ID,
                false,
                "parse".to_string(),
                "ok".to_string(),
                error.to_string(),
            );
            return;
        }
    };
    let transforms = match collect_transforms(&deck) {
        Ok(parsed) => parsed,
        Err(error) => {
            push(
                rows,
                ID,
                false,
                "collect".to_string(),
                "ok".to_string(),
                error.to_string(),
            );
            return;
        }
    };
    if transforms.len() != 4 {
        push(
            rows,
            ID,
            false,
            format!("{} transforms", transforms.len()),
            "4 transforms".to_string(),
            "wrong chain".to_string(),
        );
        return;
    }
    let applied = match apply_transforms(&deck) {
        Ok(applied) => applied,
        Err(error) => {
            push(
                rows,
                ID,
                false,
                "apply".to_string(),
                "ok".to_string(),
                error.to_string(),
            );
            return;
        }
    };
    let mut worst = 0.0_f64;
    let mut stage_name = "";
    for (stage, echoes) in applied.stages.iter().zip(TRANSF_PROBE_ECHOES.iter()) {
        for echo in echoes {
            let mut parts = echo.split_whitespace();
            let id: i64 = parts.next().expect("echo id").parse().expect("echo int");
            let expected: [f64; 3] = [
                parts.next().expect("x").parse().expect("echo real"),
                parts.next().expect("y").parse().expect("echo real"),
                parts.next().expect("z").parse().expect("echo real"),
            ];
            let got = stage.nodes.get(&id).expect("probe member");
            for axis in 0..3 {
                let dev = (got[axis] - expected[axis]).abs();
                if dev > worst {
                    worst = dev;
                    stage_name = match stage.id {
                        1 => "SYM",
                        2 => "SCA",
                        3 => "ROT",
                        _ => "TRA",
                    };
                }
            }
        }
    }
    push(
        rows,
        ID,
        worst <= VENDOR_BAND,
        format!("{worst:.3e}"),
        format!("{VENDOR_BAND:.1e}"),
        format!("4 stages x 3 probes, worst {stage_name}"),
    );
}

/// `//SUBMODEL/1` header skeletons, deck-transcribed.
///
/// Header sequences of the three `update_v_100` starter decks; bodies
/// are empty because split/merge/inventory only read headers (offsets
/// rows are empty in every deck, matching the real decks).
const ACTIV_SKELETON: &str = "/BEGIN
/TITLE
/MEMORY
/SPMD
/IOFLAG
/ANALY
/DEF_SOLID
/DEF_SHELL
/SHFRA/V4
/RANDOM
//SUBMODEL/1
/MAT/HYD_JCOOK/1
/MAT/HYD_JCOOK/2
/MAT/HYD_JCOOK/3
/NODE
/BCS/1
/GRNOD/NODE/2
/BCS/2
/GRNOD/NODE/3
/PART/1
/QUAD/1
/PART/2
/QUAD/2
/PART/3
/QUAD/3
/PROP/SOLID/1
/PROP/SOLID/2
/PROP/SOLID/3
/INIVEL/TRA/1
/GRNOD/NODE/4
/SENSOR/TIME/2
/SENSOR/SENS/3
/SENSOR/TIME/5
/INTER/TYPE5/1
/GRNOD/SURF/6
/SURF/SEG/5
/SURF/SEG/7
/INTER/TYPE3/2
/SURF/SEG/8
/SURF/SEG/9
/INTER/TYPE3/3
/SURF/SEG/10
/SURF/SEG/11
/INTER/TYPE5/4
/GRNOD/SURF/13
/SURF/SEG/12
/SURF/SEG/14
/ACTIV/4
/GRQUAD/QUAD/18
/TH/PART/1
/TH/NODE/2
/TH/QUAD/3
//ENDSUB
/END
";

const PLOAD_SKELETON: &str = "/BEGIN
/TITLE
/MEMORY
/SPMD
/IOFLAG
/ANALY
/DEF_SHELL
/SHFRA/V4
/RANDOM
//SUBMODEL/1
/MAT/PLAS_JOHNS/2
/NODE
/BCS/1
/GRNOD/NODE/2
/BCS/2
/GRNOD/NODE/3
/BCS/3
/GRNOD/NODE/33
/BCS/4
/GRNOD/NODE/7
/PART/1
/SHELL/1
/PART/5
/SHELL/5
/PART/7
/SHELL/7
/PART/8
/SHELL/8
/PART/9
/SHELL/9
/SUBSET/1
/SUBSET/6
/SUBSET/10
/PROP/SHELL/2
/FUNCT/1
/INTER/TYPE11/1
/LINE/EDGE/1
/SURF/PART/1
/INTER/TYPE7/2
/GRNOD/SURF/9
/RWALL/PLANE/1
/PLOAD/1
/SURF/PART/999
/FUNCT/999
/RBODY/1
/GRNOD/GRNOD/44
/GRNOD/PART/43
/GRNOD/NODE/42
/RBODY/2
/GRNOD/PART/45
/RBODY/3
/GRNOD/PART/46
/IMPVEL/1
/GRNOD/NODE/4
/GRNOD/NODE/12
/GRNOD/NODE/13
/GRNOD/NODE/14
/GRNOD/NODE/15
/GRNOD/GRNOD/16
/GRNOD/NODE/17
/GRNOD/NODE/18
/GRNOD/GRNOD/19
/GRNOD/NODE/20
/GRNOD/NODE/21
/GRNOD/GRNOD/22
/GRNOD/NODE/23
/GRNOD/NODE/24
/GRNOD/GRNOD/25
/GRNOD/NODE/28
/GRNOD/NODE/29
/GRNOD/NODE/30
/GRNOD/NODE/31
/GRNOD/GRNOD/32
/GRNOD/NODE/34
/GRNOD/NODE/35
/GRNOD/PART/36
/GRNOD/PART/37
/GRNOD/GRNOD/38
/GRNOD/PART/39
/GRNOD/PART/40
/GRNOD/GRNOD/41
/TH/PART/1
/TH/NODE/2
/TH/RWALL/4
/TH/INTER/5
/TH/INTER/6
/TH/RBODY/7
//ENDSUB
/END
";

const FAIL_SKELETON: &str = "/BEGIN
/TITLE
/MEMORY
/SPMD
/IOFLAG
/ANALY
/DEF_SOLID
/DEF_SHELL
/SHFRA/V4
/RANDOM
//SUBMODEL/1
/MAT/PLAS_JOHNS/4
/FAIL/JOHNSON/4
/NODE
/BCS/6
/GRNOD/GRNOD/111
/GRNOD/GRNOD/113
/GRNOD/NODE/112
/BCS/7
/GRNOD/GRNOD/114
/GRNOD/GRNOD/116
/GRNOD/NODE/115
/BCS/8
/GRNOD/GRNOD/125
/GRNOD/GRNOD/127
/GRNOD/NODE/126
/BCS/9
/GRNOD/GRNOD/128
/GRNOD/GRNOD/130
/GRNOD/NODE/129
/PART/2
/SHELL/2
/SUBSET/1
/SUBSET/5
/PROP/TYPE1/4
/FUNCT/1
/FUNCT/5
/FUNCT/6
/FUNCT/7
/FUNCT/14
/FUNCT/15
/FUNCT/17
/FUNCT/18
/FUNCT/20
/FUNCT/21
/IMPDISP/2
/GRNOD/NODE/103
/TH/PART/9
/TH/SHEL/11
//ENDSUB
/END
";

/// Checks one skeleton: split shape, zero offsets, merge identity.
///
/// Cards are header-only (`parse_header` plus empty bodies): the real
/// decks carry full bodies, but split/merge/inventory read headers
/// alone, and offset rows are empty in every deck.
fn check_skeleton(rows: &mut Vec<Row>, case: &str, skeleton: &str, inside: usize, functs: usize) {
    let id = format!("submodel-merge:{case}");
    let mut cards = Vec::new();
    for header in skeleton.lines().filter(|line| !line.is_empty()) {
        let Some((kind, card_id, sub_id)) = parse_header(header) else {
            push(
                rows,
                &id,
                false,
                "parse".to_string(),
                "ok".to_string(),
                format!("unknown header {header}"),
            );
            return;
        };
        cards.push(DeckCard {
            kind,
            id: card_id,
            sub_id,
            title: String::new(),
            fields: Vec::new(),
            typed: None,
        });
    }
    let deck = RadDeck { cards };
    let sections = match split_submodels(&deck) {
        Ok(sections) => sections,
        Err(error) => {
            push(
                rows,
                &id,
                false,
                "split".to_string(),
                "ok".to_string(),
                error.to_string(),
            );
            return;
        }
    };
    if sections.len() != 1
        || sections[0].id != 1
        || sections[0].level != 1
        || !sections[0].offsets.is_zero()
        || sections[0].end - sections[0].start != inside
    {
        push(
            rows,
            &id,
            false,
            format!("{sections:?}"),
            format!("1 section, {inside} inside"),
            "shape mismatch".to_string(),
        );
        return;
    }
    let merged = match merge_submodels(&deck) {
        Ok(merged) => merged,
        Err(error) => {
            push(
                rows,
                &id,
                false,
                "merge".to_string(),
                "ok".to_string(),
                error.to_string(),
            );
            return;
        }
    };
    let counts = section_inventory(&deck, &sections[0]);
    let pass = merged.cards.len() + 2 == deck.cards.len()
        && counts.get("/FUNCT").copied().unwrap_or(0) == functs;
    push(
        rows,
        &id,
        pass,
        format!(
            "{} inside, {} FUNCT",
            sections[0].end - sections[0].start,
            counts.get("/FUNCT").copied().unwrap_or(0)
        ),
        format!("{inside} inside, {functs} FUNCT"),
        format!("{} cards merged", merged.cards.len()),
    );
}

/// Builds a deterministic nonzero explicit state.
fn demo_state() -> ExplicitState {
    let mut state = ExplicitState::new(4);
    state.positions = vec![
        1.0, 2.0, 3.0, 0.0, -1.0, 0.5, -2.0, 0.0, 1.0, 3.0, 3.0, -3.0,
    ];
    state.velocities = vec![
        0.5, 0.0, -0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0, -1.0, 0.25, 0.0,
    ];
    state.forces = vec![0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.3, 0.4, 0.4, 0.4];
    state.masses = vec![1.0, 2.0, 0.5, 4.0];
    state.time = 1.25;
    state.step = 7;
    state.dt = 0.001;
    state
}

/// Proves native restart determinism: same state, identical bytes.
fn check_restart_determinism(rows: &mut Vec<Row>) {
    const ID_BYTES: &str = "native-determinism:bytes";
    const ID_ROUNDTRIP: &str = "native-determinism:roundtrip";
    const ID_SENSITIVITY: &str = "native-determinism:sensitivity";
    let state = demo_state();
    let (first, second) = match (encode_restart(&state), encode_restart(&state)) {
        (Ok(first), Ok(second)) => (first, second),
        _ => {
            push(
                rows,
                ID_BYTES,
                false,
                "encode".to_string(),
                "ok".to_string(),
                "encode failed".to_string(),
            );
            return;
        }
    };
    let digest = format!("{:x}", Sha256::digest(&first));
    push(
        rows,
        ID_BYTES,
        first == second,
        format!("{} bytes sha256 {digest}", first.len()),
        "identical bytes".to_string(),
        "same state encoded twice".to_string(),
    );
    match decode_restart(&first) {
        Ok(restored) => push(
            rows,
            ID_ROUNDTRIP,
            restored == state,
            if restored == state {
                "equal".to_string()
            } else {
                "drifted".to_string()
            },
            "equal".to_string(),
            "decode inverts encode".to_string(),
        ),
        Err(error) => push(
            rows,
            ID_ROUNDTRIP,
            false,
            "decode".to_string(),
            "ok".to_string(),
            error.to_string(),
        ),
    }
    let mut mutated = state;
    mutated.positions[0] += 1.0;
    let changed = encode_restart(&mutated).is_ok_and(|bytes| bytes != first);
    push(
        rows,
        ID_SENSITIVITY,
        changed,
        if changed {
            "differ".to_string()
        } else {
            "identical".to_string()
        },
        "differ".to_string(),
        "one position moved by 1.0".to_string(),
    );
}

/// Loads a golden or records the failure.
fn load_case(root: &std::path::Path, case: &str, rows: &mut Vec<Row>) -> Option<Golden> {
    match load_golden(&golden_path(root, case)) {
        Ok(golden) => Some(golden),
        Err(error) => {
            push(
                rows,
                &format!("load:{case}"),
                false,
                "missing".to_string(),
                "present".to_string(),
                error.to_string(),
            );
            None
        }
    }
}

fn main() {
    let args = Args::parse();
    let root = args
        .goldens
        .clone()
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens"));
    let mut rows = Vec::new();

    check_transf_chain(&mut rows);
    check_skeleton(
        &mut rows,
        "SUBMODEL/update_v_100/ACTIV",
        ACTIV_SKELETON,
        41,
        0,
    );
    check_skeleton(
        &mut rows,
        "SUBMODEL/update_v_100/PLOAD",
        PLOAD_SKELETON,
        77,
        2,
    );
    check_skeleton(
        &mut rows,
        "SUBMODEL/update_v_100/FAIL/Jonhson_Cook",
        FAIL_SKELETON,
        38,
        10,
    );

    for (case, rootname, starter, engines) in SUBMODEL_CASES {
        let Some(golden) = load_case(&root, case, &mut rows) else {
            continue;
        };
        check_provenance(&mut rows, &golden, case, rootname, starter, engines);
        check_grid(&mut rows, &golden, case);
        check_energy_identity(&mut rows, &golden, case);
        check_mass(&mut rows, &golden, case, case.ends_with("Jonhson_Cook"));
    }
    if let Some(golden) = load_case(&root, "SUBMODEL/TRANSF", &mut rows) {
        check_part_sum(
            &mut rows,
            &golden,
            "SUBMODEL/TRANSF",
            &["BOULE MASS", "COQUE ARRIERE MASS"],
        );
    }

    for stops in &RESTART_CASES {
        let Some(golden) = load_case(&root, stops.case, &mut rows) else {
            continue;
        };
        check_provenance(
            &mut rows,
            &golden,
            stops.case,
            stops.rootname,
            stops.starter_deck,
            &stops.engine_decks,
        );
        check_grid(&mut rows, &golden, stops.case);
        check_energy_identity(&mut rows, &golden, stops.case);
        check_mass(&mut rows, &golden, stops.case, false);
        check_coverage(&mut rows, &golden, stops);
        check_join(&mut rows, &golden, stops);
        check_cycle(&mut rows, &golden, stops.case);
    }
    if let Some(golden) = load_case(&root, "RESTART/cra2", &mut rows) {
        check_conservation(&mut rows, &golden, "RESTART/cra2", 1e-3);
    }
    if let Some(golden) = load_case(&root, "RESTART/int7_small_boule_rst", &mut rows) {
        check_part_sum(
            &mut rows,
            &golden,
            "RESTART/int7_small_boule_rst",
            &["BOULE MASS", "COQUE ARRIERE MASS"],
        );
    }

    check_restart_determinism(&mut rows);

    let mut failures = 0;
    for row in &rows {
        if !row.pass {
            failures += 1;
        }
        println!(
            "{} {} measured={} bound={} {}",
            if row.pass { "PASS" } else { "FAIL" },
            row.id,
            row.measured,
            row.bound,
            row.detail
        );
    }
    println!("rad-submodel: {} probes, {} failures", rows.len(), failures);
    if failures > 0 {
        println!("FAILED: qa-rad-submodel");
        std::process::exit(1);
    }
    println!("SUCCESS: qa-rad-submodel");
}
