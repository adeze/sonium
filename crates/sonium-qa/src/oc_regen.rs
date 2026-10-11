//! One-command miniqa golden regen plus offline drift check.
//!
//! This module drives corpus-wide golden refreshes through the existing
//! [`super::oc_golden::harvest`] pipeline and diffs fresh (or staged)
//! goldens against the checked-in baselines, one channel at a time.
//! Live regen runs the vendor starter+engine per deck and writes
//! byte-shaped goldens through the existing golden constructors only
//! (contract P4C3: no parallel golden schema); drift-only mode stays
//! fully offline, diffs staged goldens (or self-checks the baselines),
//! and verifies engine stamps plus deck hashes.
//!
//! Entry points: [`run_regen`] runs a whole corpus from [`RegenOptions`]
//! and returns a [`RegenSummary`]; [`drift_diff`] diffs two goldens
//! into a [`DriftReport`] with per-channel max-abs/rel deltas. Progress
//! flows through a [`RegenEvent`] hook, which also forwards the
//! harvester's own [`super::oc_golden::HarvestStage`] events.
//!
//! Side effects: live regen executes vendor binaries in fresh per-case
//! scratch dirs (removed on success unless `keep_scratch` is set) and
//! writes `<out>/miniqa/<case>.json` for every harvested case, even
//! when drift exceeds tolerances — the exit code, not the write,
//! signals drift. Drift-only mode only reads.

// Rust guideline compliant 2026-02-21

use super::RadError;
use super::oc_golden::{
    DEFAULT_REL_TOL, Golden, GoldenChannel, HarvestOptions, HarvestStage, ZERO_TOL, golden_path,
    grid_monotone, harvest_with_hook, interp_on_grid, load_golden, stamp_failures,
    verify_deck_hashes, write_golden,
};
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// Case selection for a regen run.
#[derive(Debug, Clone)]
pub enum CaseSelect {
    /// Every checked-in golden under `<goldens>/miniqa/`, sorted.
    All,
    /// Exactly these miniqa-relative case ids, in listed order.
    Only(Vec<String>),
}

/// Inputs for [`run_regen`].
///
/// Mirrors [`HarvestOptions`] per case; every fresh golden is built
/// through the existing harvester, so engine-stamp and deck-hash meta
/// always match the checked-in shape.
#[derive(Debug, Clone)]
pub struct RegenOptions {
    /// Goldens root holding `<root>/miniqa/<case>.json` baselines.
    pub goldens_root: PathBuf,
    /// Where fresh goldens land (`None` = overwrite `goldens_root`).
    pub out_root: Option<PathBuf>,
    /// miniqa root for deck-hash checks (and live harvests).
    pub miniqa_root: Option<PathBuf>,
    /// Engine/starter exe or `exec/` dir (`None` = offline only).
    pub exe: Option<PathBuf>,
    /// Cases to run.
    pub cases: CaseSelect,
    /// Scratch parent; one fresh per-case dir lands inside.
    pub scratch_root: PathBuf,
    /// `OMP_NUM_THREADS` per run.
    pub threads: usize,
    /// Peak-preserving decimation budget (0 = full series). Fresh
    /// regens should use the canonical corpus budget (see
    /// [`super::oc_golden::CANONICAL_MAX_SAMPLES`]) so single-deck
    /// regens match the checked-in corpus shape.
    pub max_samples: usize,
    /// Keep per-case scratch dirs after success.
    pub keep_scratch: bool,
    /// Local vendor-tree deviations recorded verbatim in fresh goldens.
    pub local_patches: Option<String>,
    /// Skip harvests: diff staged goldens (or self-check) offline.
    pub drift_only: bool,
}

impl RegenOptions {
    /// Fresh-golden target dir: `out_root`, else `goldens_root`.
    pub fn out_dir(&self) -> PathBuf {
        self.out_root
            .clone()
            .unwrap_or_else(|| self.goldens_root.clone())
    }
}

/// List checked-in golden case ids under a goldens root.
///
/// Walks `<root>/miniqa/` for `*.json` files at any depth; each case
/// id is the `/`-separated relative path minus the `.json` suffix.
/// Non-JSON files are ignored. The list is sorted for determinism.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::oc_regen::discover_golden_cases;
///
/// let root = tempfile::tempdir().unwrap();
/// let miniqa = root.path().join("miniqa");
/// std::fs::create_dir_all(miniqa.join("sub")).unwrap();
/// std::fs::write(miniqa.join("a.json"), "{}").unwrap();
/// std::fs::write(miniqa.join("sub").join("b.json"), "{}").unwrap();
/// std::fs::write(miniqa.join("notes.txt"), "ignored").unwrap();
/// assert_eq!(
///     discover_golden_cases(root.path()).unwrap(),
///     vec!["a".to_string(), "sub/b".to_string()]
/// );
/// ```
///
/// # Errors
///
/// Returns [`RadError::Parse`] when `<root>/miniqa/` is missing or
/// holds no golden JSON files.
pub fn discover_golden_cases(goldens_root: &Path) -> Result<Vec<String>, RadError> {
    let miniqa = goldens_root.join("miniqa");
    if !miniqa.is_dir() {
        return Err(RadError::Parse(format!(
            "goldens root {} holds no miniqa/ dir",
            goldens_root.display()
        )));
    }
    let mut cases = Vec::new();
    collect_golden_cases(&miniqa, &miniqa, &mut cases)?;
    cases.sort();
    if cases.is_empty() {
        return Err(RadError::Parse(format!(
            "goldens root {} holds no miniqa/*.json goldens",
            goldens_root.display()
        )));
    }
    Ok(cases)
}

fn collect_golden_cases(root: &Path, dir: &Path, cases: &mut Vec<String>) -> Result<(), RadError> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.flatten().collect();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_golden_cases(root, &path, cases)?;
        } else if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
        {
            let relative = path.strip_prefix(root).map_err(|_| {
                RadError::Parse(format!(
                    "golden {} escapes {}",
                    path.display(),
                    root.display()
                ))
            })?;
            let mut case = relative.to_string_lossy().replace('\\', "/");
            case.truncate(case.len() - ".json".len());
            cases.push(case);
        }
    }
    Ok(())
}

/// Resolve the runnable case list for regen options.
///
/// # Errors
///
/// Returns [`RadError::Parse`] for an empty `Only` list or when
/// corpus discovery fails (see [`discover_golden_cases`]).
pub fn resolve_cases(options: &RegenOptions) -> Result<Vec<String>, RadError> {
    match &options.cases {
        CaseSelect::All => discover_golden_cases(&options.goldens_root),
        CaseSelect::Only(cases) => {
            if cases.is_empty() {
                return Err(RadError::Parse(
                    "regen case list is empty; pass --cases a,b or --all".to_string(),
                ));
            }
            Ok(cases.clone())
        }
    }
}

/// Per-channel drift between a baseline golden and a candidate.
///
/// Maxima cover every finite candidate sample over the baseline
/// stamps, including vendor-zero-skipped ones; the pass verdict
/// follows the channel's own tolerances exactly like
/// [`super::oc_golden::compare_history`].
#[derive(Debug, Clone)]
pub struct ChannelDrift {
    /// Golden channel key.
    pub channel: String,
    /// Baseline stamps covered by the candidate grid.
    pub samples: usize,
    /// Largest `|actual - expected|` over covered samples.
    pub max_abs: f64,
    /// Largest `|actual - expected| / |expected|` over covered
    /// samples with `|expected| >= ZERO_TOL` (zero when none qualify).
    pub max_rel: f64,
    /// Baseline stamp of the worst bound-ratio sample.
    pub worst_time: f64,
    /// Largest `diff / (abs_tol + rel_tol * |expected|)` over
    /// non-skipped samples (zero when every sample skips).
    pub bound_ratio: f64,
    /// Covered samples above the channel's own tolerances.
    pub out_of_tolerance: usize,
    /// Baseline stamps outside the candidate grid span.
    pub uncovered: usize,
    /// True when the candidate lacks the channel entirely.
    pub missing: bool,
}

impl ChannelDrift {
    /// True when the channel matches within its own tolerances.
    pub fn passed(&self) -> bool {
        !self.missing && self.uncovered == 0 && self.out_of_tolerance == 0
    }

    /// One-line human report with the max deltas.
    pub fn summary(&self) -> String {
        let verdict = if self.passed() { "PASS" } else { "FAIL" };
        if self.missing {
            return format!(
                "channel {}: {verdict} (missing from candidate)",
                self.channel
            );
        }
        format!(
            "channel {}: {verdict} max_abs {:.3e} max_rel {:.3e} n={} oob={} uncovered={}",
            self.channel,
            self.max_abs,
            self.max_rel,
            self.samples,
            self.out_of_tolerance,
            self.uncovered
        )
    }
}

/// Per-scalar drift of the last-cycle scalars.
///
/// The verdict mirrors [`super::oc_golden::compare_cycle`]: the
/// vendor zero skip, then [`DEFAULT_REL_TOL`] with a zero absolute
/// floor.
#[derive(Debug, Clone)]
pub struct CycleDrift {
    /// Vendor scalar key (`CYCLE`, `IENERGY`, ...).
    pub key: String,
    /// Baseline value.
    pub expected: f64,
    /// Candidate value (`None` = missing).
    pub actual: Option<f64>,
    /// `|actual - expected|` (zero when missing).
    pub abs: f64,
    /// `abs / |expected|` with the vendor zero band (zero when
    /// missing or `|expected|` sits inside the band).
    pub rel: f64,
    /// True when missing or above tolerance.
    pub failed: bool,
}

impl CycleDrift {
    /// True when the scalar matches.
    pub fn passed(&self) -> bool {
        !self.failed
    }

    /// One-line human report with the deltas.
    pub fn summary(&self) -> String {
        match self.actual {
            None => format!("cycle {}: FAIL (missing from candidate)", self.key),
            Some(actual) => {
                let verdict = if self.failed { "FAIL" } else { "PASS" };
                format!(
                    "cycle {}: {verdict} abs {:.3e} rel {:.3e} (actual {actual:.6e} vs golden {:.6e})",
                    self.key, self.abs, self.rel, self.expected
                )
            }
        }
    }
}

/// Drift of a candidate golden against its checked-in baseline.
#[derive(Debug, Clone)]
pub struct DriftReport {
    /// Baseline case id.
    pub case: String,
    /// One entry per baseline channel, in golden key order.
    pub channels: Vec<ChannelDrift>,
    /// One entry per baseline cycle scalar, in key order.
    pub cycle: Vec<CycleDrift>,
    /// Set when the candidate grid is empty or non-monotone while the
    /// baseline carries TH channels (no channel is diffed then).
    pub grid_error: Option<String>,
}

impl DriftReport {
    /// True when every channel and scalar matches.
    pub fn passed(&self) -> bool {
        self.grid_error.is_none()
            && self.channels.iter().all(ChannelDrift::passed)
            && self.cycle.iter().all(CycleDrift::passed)
    }

    /// Human failure lines for every mismatch.
    pub fn failures(&self) -> Vec<String> {
        let mut failures = Vec::new();
        if let Some(error) = &self.grid_error {
            failures.push(format!("drift {}: grid FAIL ({error})", self.case));
        }
        failures.extend(
            self.channels
                .iter()
                .filter(|channel| !channel.passed())
                .map(ChannelDrift::summary),
        );
        failures.extend(
            self.cycle
                .iter()
                .filter(|scalar| !scalar.passed())
                .map(CycleDrift::summary),
        );
        failures
    }

    /// One-line human summary with the worst bound ratio.
    pub fn summary(&self) -> String {
        let verdict = if self.passed() { "PASS" } else { "FAIL" };
        if let Some(error) = &self.grid_error {
            return format!("drift {}: {verdict} ({error})", self.case);
        }
        let channels_ok = self.channels.iter().filter(|c| c.passed()).count();
        let cycle_ok = self.cycle.iter().filter(|c| c.passed()).count();
        let worst = self
            .channels
            .iter()
            .map(|channel| channel.bound_ratio)
            .fold(0.0f64, f64::max);
        format!(
            "drift {}: {verdict} {channels_ok}/{} channels, {cycle_ok}/{} cycle scalars, worst ratio {worst:.2}x",
            self.case,
            self.channels.len(),
            self.cycle.len()
        )
    }
}

/// Diff a candidate golden against a checked-in baseline.
///
/// Grids may differ (decimation, dt drift): each candidate channel is
/// linearly interpolated onto the baseline stamps, which the
/// candidate grid must cover. A sample passes when
/// `|actual - expected| <= abs + rel * |expected|` with the channel's
/// own tolerances, under the vendor near-zero skip. Extra candidate
/// channels are ignored; cycle-only baselines diff scalars alone.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::oc_golden::parse_golden_str;
/// use sonium_qa::rad::oc_regen::drift_diff;
///
/// let text = r#"{
///   "schema": "opencourant-golden-v1",
///   "case": "DEMO/case",
///   "meta": {
///     "engine": "OpenRadioss linux64_gf CommitID deadbeef",
///     "engine_commit": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
///     "starter_exe": "starter_linux64_gf",
///     "engine_exe": "engine_linux64_gf",
///     "deck_sha256": {"starter": "aa", "engine": "bb"},
///     "starter_deck": "CASE_0000.rad",
///     "engine_decks": ["CASE_0001.rad"],
///     "starter_options": "-i CASE_0000.rad -np 1",
///     "engine_options": "-i CASE_0001.rad",
///     "mpi": "smp", "np": 1, "threads": 1, "precision": "dp",
///     "harvest_date": "2026-01-01T00:00:00Z",
///     "deck_units": ["kg", "m", "s"],
///     "rootname": "CASE",
///     "normal_termination": true,
///     "th_tool": "th_to_csv_linux64_gf",
///     "exe_sha256": {"starter": "s", "engine": "e", "th_to_csv": "t"},
///     "decimation": null
///   },
///   "time": [0.0, 1.0],
///   "channels": {
///     "FORCE": {"values": [1.0, 2.0], "final": 2.0, "min": 1.0,
///               "max": 2.0, "abs_tol": 2e-12, "rel_tol": 6.6e-4}
///   },
///   "cycle": {"CYCLE": 2.0}
/// }"#;
/// let baseline = parse_golden_str(text, "demo").unwrap();
/// assert!(drift_diff(&baseline, &baseline).passed());
/// let mut drifted = baseline.clone();
/// drifted.channels.get_mut("FORCE").unwrap().values[1] = 3.0;
/// let report = drift_diff(&baseline, &drifted);
/// assert!(!report.passed());
/// assert!((report.channels[0].max_abs - 1.0).abs() < 1e-12);
/// assert!((report.channels[0].max_rel - 0.5).abs() < 1e-12);
/// ```
pub fn drift_diff(baseline: &Golden, candidate: &Golden) -> DriftReport {
    let mut report = DriftReport {
        case: baseline.case.clone(),
        channels: Vec::with_capacity(baseline.channels.len()),
        cycle: Vec::with_capacity(baseline.cycle.len()),
        grid_error: None,
    };
    if baseline.channels.is_empty() {
        // Cycle-only baseline: no TH to diff; scalars decide.
    } else if candidate.time.is_empty() {
        report.grid_error = Some("candidate has an empty time grid".to_string());
    } else if !grid_monotone(&candidate.time) {
        report.grid_error = Some("candidate time grid is not monotone".to_string());
    } else {
        for (name, channel) in &baseline.channels {
            report
                .channels
                .push(diff_channel(name, &baseline.time, channel, candidate));
        }
    }
    report.cycle = diff_cycle(&baseline.cycle, &candidate.cycle);
    report
}

fn diff_channel(
    name: &str,
    baseline_time: &[f64],
    channel: &GoldenChannel,
    candidate: &Golden,
) -> ChannelDrift {
    let mut drift = ChannelDrift {
        channel: name.to_string(),
        samples: 0,
        max_abs: 0.0,
        max_rel: 0.0,
        worst_time: 0.0,
        bound_ratio: 0.0,
        out_of_tolerance: 0,
        uncovered: 0,
        missing: false,
    };
    let Some(candidate_channel) = candidate.channels.get(name) else {
        drift.missing = true;
        return drift;
    };
    let values = &candidate_channel.values;
    if values.len() != candidate.time.len() {
        // Ragged candidate: no stamp is trustworthy, fail closed.
        drift.uncovered = baseline_time.len();
        return drift;
    }
    for (stamp, expected) in baseline_time.iter().zip(channel.values.iter()) {
        let Some(actual) = interp_on_grid(&candidate.time, values, *stamp) else {
            drift.uncovered += 1;
            continue;
        };
        drift.samples += 1;
        let diff = (actual - expected).abs();
        drift.max_abs = drift.max_abs.max(diff);
        if expected.abs() >= ZERO_TOL {
            drift.max_rel = drift.max_rel.max(diff / expected.abs());
        }
        if !actual.is_finite() {
            drift.out_of_tolerance += 1;
            drift.bound_ratio = f64::INFINITY;
            drift.worst_time = *stamp;
            continue;
        }
        // Vendor Zero_tolerance skip, mirroring compare_history.
        if (expected.abs() < ZERO_TOL && actual.abs() < ZERO_TOL) || diff < 2.0 * ZERO_TOL {
            continue;
        }
        let bound = channel.abs_tol + channel.rel_tol * expected.abs();
        if diff > bound {
            drift.out_of_tolerance += 1;
            let ratio = diff / bound.max(f64::MIN_POSITIVE);
            if ratio > drift.bound_ratio {
                drift.bound_ratio = ratio;
                drift.worst_time = *stamp;
            }
        }
    }
    drift
}

fn diff_cycle(
    baseline: &BTreeMap<String, f64>,
    candidate: &BTreeMap<String, f64>,
) -> Vec<CycleDrift> {
    let mut out = Vec::with_capacity(baseline.len());
    for (key, expected) in baseline {
        let mut drift = CycleDrift {
            key: key.clone(),
            expected: *expected,
            actual: None,
            abs: 0.0,
            rel: 0.0,
            failed: false,
        };
        let Some(actual) = candidate.get(key) else {
            drift.failed = true;
            out.push(drift);
            continue;
        };
        drift.actual = Some(*actual);
        let diff = (actual - expected).abs();
        drift.abs = diff;
        if expected.abs() >= ZERO_TOL {
            drift.rel = diff / expected.abs();
        }
        // Same rule as compare_cycle: zero skip, then DEFAULT_REL_TOL.
        if (expected.abs() < ZERO_TOL && actual.abs() < ZERO_TOL) || diff < 2.0 * ZERO_TOL {
            out.push(drift);
            continue;
        }
        if diff > DEFAULT_REL_TOL * expected.abs() {
            drift.failed = true;
        }
        out.push(drift);
    }
    out
}

/// Case-level regen lifecycle stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegenStage {
    /// Case started.
    Start,
    /// Vendor-pipeline progress inside a live harvest.
    Harvest(HarvestStage),
    /// Candidate diffed against the baseline.
    Diffed,
    /// Fresh golden written to the out root.
    Written,
    /// Case finished (see the case report for the verdict).
    Done,
}

impl fmt::Display for RegenStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Start => write!(f, "start"),
            Self::Harvest(stage) => write!(f, "harvest/{stage}"),
            Self::Diffed => write!(f, "diffed"),
            Self::Written => write!(f, "written"),
            Self::Done => write!(f, "done"),
        }
    }
}

/// One case-level regen progress event.
#[derive(Debug, Clone)]
pub struct RegenEvent {
    /// Case under regen.
    pub case: String,
    /// Zero-based case index.
    pub index: usize,
    /// Total cases in the run.
    pub total: usize,
    /// Stage just reached.
    pub stage: RegenStage,
}

/// Outcome of one regen case.
#[derive(Debug, Clone)]
pub struct CaseReport {
    /// miniqa-relative case id.
    pub case: String,
    /// True when the drift passed and no operational failure hit.
    pub passed: bool,
    /// Baseline-vs-candidate drift (`None` when the baseline or the
    /// harvest was unusable).
    pub drift: Option<DriftReport>,
    /// Operational failure lines (load, harvest, stamp, hash, write).
    /// Drift detail lives in [`DriftReport`], not here.
    pub failures: Vec<String>,
    /// Non-failing notes (self-checks, skipped checks, fresh writes).
    pub notes: Vec<String>,
    /// Fresh golden path (live regen writes only).
    pub wrote: Option<PathBuf>,
}

impl CaseReport {
    /// One-line verdict; details live in `drift`, `failures`, `notes`.
    pub fn summary(&self) -> String {
        let verdict = if self.passed { "PASS" } else { "FAIL" };
        let drift = self
            .drift
            .as_ref()
            .map_or("no drift".to_string(), |report| {
                let ok = report.channels.iter().filter(|c| c.passed()).count();
                format!("{ok}/{} channels", report.channels.len())
            });
        let mut line = format!("regen {}: {verdict} ({drift})", self.case);
        if let Some(path) = &self.wrote {
            line.push_str(&format!(" -> {}", path.display()));
        }
        line
    }
}

/// Corpus-wide regen outcome: one report per case, in run order.
#[derive(Debug, Clone, Default)]
pub struct RegenSummary {
    /// Per-case reports in run order.
    pub cases: Vec<CaseReport>,
}

impl RegenSummary {
    /// True when every case passed (and at least one ran).
    pub fn passed(&self) -> bool {
        !self.cases.is_empty() && self.cases.iter().all(|case| case.passed)
    }

    /// `(passed, failed)` case counts.
    pub fn counts(&self) -> (usize, usize) {
        let passed = self.cases.iter().filter(|case| case.passed).count();
        (passed, self.cases.len() - passed)
    }

    /// One-line corpus verdict.
    pub fn summary(&self) -> String {
        let (passed, failed) = self.counts();
        let verdict = if self.passed() { "PASS" } else { "FAIL" };
        format!("regen corpus: {verdict} ({passed} passed, {failed} failed)")
    }
}

/// Regen one case: harvest fresh (live) or diff staged (offline).
///
/// Never fails hard: every problem lands in the returned
/// [`CaseReport`] so [`run_regen`] continues across the corpus. Live
/// regen always writes the fresh golden; drift above the baseline's
/// own tolerances surfaces through the report (and the exit code),
/// never by skipping the write.
pub fn regen_case(
    options: &RegenOptions,
    case: &str,
    index: usize,
    total: usize,
    hook: impl Fn(&RegenEvent),
) -> CaseReport {
    let mut report = CaseReport {
        case: case.to_string(),
        passed: true,
        drift: None,
        failures: Vec::new(),
        notes: Vec::new(),
        wrote: None,
    };
    let emit = |stage| {
        hook(&RegenEvent {
            case: case.to_string(),
            index,
            total,
            stage,
        });
    };
    emit(RegenStage::Start);
    if options.drift_only {
        drift_only_case(options, case, &mut report);
    } else {
        live_case(options, case, &emit, &mut report);
    }
    report.passed =
        report.failures.is_empty() && report.drift.as_ref().is_none_or(DriftReport::passed);
    emit(RegenStage::Done);
    report
}

fn drift_only_case(options: &RegenOptions, case: &str, report: &mut CaseReport) {
    let baseline_path = golden_path(&options.goldens_root, case);
    let baseline = match load_golden(&baseline_path) {
        Ok(golden) => golden,
        Err(error) => {
            report.failures.push(error.to_string());
            return;
        }
    };
    let staged_path = golden_path(&options.out_dir(), case);
    let staged = if staged_path != baseline_path && staged_path.is_file() {
        match load_golden(&staged_path) {
            Ok(staged) => {
                report
                    .notes
                    .push(format!("diffing staged golden {}", staged_path.display()));
                Some(staged)
            }
            Err(error) => {
                report.failures.push(format!(
                    "staged golden {} is unloadable: {error}",
                    staged_path.display()
                ));
                None
            }
        }
    } else if staged_path != baseline_path {
        report.notes.push(format!(
            "no staged golden at {}; self-checking the baseline",
            staged_path.display()
        ));
        None
    } else {
        None
    };
    let drift = match &staged {
        Some(candidate) => drift_diff(&baseline, candidate),
        None => drift_diff(&baseline, &baseline),
    };
    report.drift = Some(drift);
    report.failures.extend(stamp_failures(&baseline));
    match &options.miniqa_root {
        Some(miniqa) => {
            if let Err(error) = verify_deck_hashes(&baseline, miniqa) {
                report.failures.push(error.to_string());
            }
        }
        None => report.notes.push(
            "deck-hash check skipped: no miniqa root (--miniqa or OPENCOURANT_MINIQA)".to_string(),
        ),
    }
}

fn live_case(
    options: &RegenOptions,
    case: &str,
    emit: &impl Fn(RegenStage),
    report: &mut CaseReport,
) {
    let baseline = match load_golden(&golden_path(&options.goldens_root, case)) {
        Ok(golden) => Some(golden),
        Err(error) => {
            report.notes.push(format!(
                "no usable baseline ({error}); the harvest writes fresh"
            ));
            None
        }
    };
    let Some(miniqa_root) = options.miniqa_root.clone() else {
        report
            .failures
            .push("live regen needs --miniqa or OPENCOURANT_MINIQA".to_string());
        return;
    };
    let Some(exe) = options.exe.clone() else {
        report
            .failures
            .push("live regen needs --exe, RADISS_EXE, or OPENCOURANT_EXE".to_string());
        return;
    };
    let harvest_options = HarvestOptions {
        miniqa_root,
        case: case.to_string(),
        exe,
        scratch_root: options.scratch_root.clone(),
        threads: options.threads,
        keep_scratch: options.keep_scratch,
        max_samples: options.max_samples,
        local_patches: options.local_patches.clone(),
    };
    let fresh = match harvest_with_hook(&harvest_options, |event| {
        emit(RegenStage::Harvest(event.stage));
    }) {
        Ok(golden) => golden,
        Err(error) => {
            report.failures.push(format!("harvest failed: {error}"));
            return;
        }
    };
    if let Some(baseline) = &baseline {
        report.drift = Some(drift_diff(baseline, &fresh));
    }
    emit(RegenStage::Diffed);
    report.failures.extend(stamp_failures(&fresh));
    match write_golden(&options.out_dir(), &fresh) {
        Ok(path) => {
            report.wrote = Some(path);
            emit(RegenStage::Written);
        }
        Err(error) => report.failures.push(format!("write failed: {error}")),
    }
}

/// Regen every selected case, continuing past per-case failures.
///
/// The hook runs on the calling thread for each [`RegenEvent`]; it
/// must not block. Per-case problems land in each [`CaseReport`], so
/// one bad deck never aborts the corpus.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::oc_regen::{CaseSelect, RegenOptions, run_regen};
///
/// let root = tempfile::tempdir().unwrap();
/// let miniqa = root.path().join("goldens").join("miniqa");
/// std::fs::create_dir_all(&miniqa).unwrap();
/// std::fs::write(
///     miniqa.join("a.json"),
///     r#"{
///   "schema": "opencourant-golden-v1",
///   "case": "a",
///   "meta": {
///     "engine": "OpenRadioss linux64_gf CommitID deadbeef",
///     "engine_commit": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
///     "starter_exe": "starter_linux64_gf",
///     "engine_exe": "engine_linux64_gf",
///     "deck_sha256": {"starter": "aa", "engine": "bb"},
///     "starter_deck": "CASE_0000.rad",
///     "engine_decks": ["CASE_0001.rad"],
///     "starter_options": "-i CASE_0000.rad -np 1",
///     "engine_options": "-i CASE_0001.rad",
///     "mpi": "smp", "np": 1, "threads": 1, "precision": "dp",
///     "harvest_date": "2026-01-01T00:00:00Z",
///     "deck_units": ["kg", "m", "s"],
///     "rootname": "CASE",
///     "normal_termination": true,
///     "th_tool": "th_to_csv_linux64_gf",
///     "exe_sha256": {"starter": "s", "engine": "e", "th_to_csv": "t"},
///     "decimation": null
///   },
///   "time": [0.0, 1.0],
///   "channels": {
///     "FORCE": {"values": [1.0, 2.0], "final": 2.0, "min": 1.0,
///               "max": 2.0, "abs_tol": 2e-12, "rel_tol": 6.6e-4}
///   },
///   "cycle": {"CYCLE": 2.0}
/// }"#,
/// )
/// .unwrap();
/// let options = RegenOptions {
///     goldens_root: root.path().join("goldens"),
///     out_root: None,
///     miniqa_root: None,
///     exe: None,
///     cases: CaseSelect::All,
///     scratch_root: root.path().join("scratch"),
///     threads: 1,
///     max_samples: 2000,
///     keep_scratch: false,
///     local_patches: None,
///     drift_only: true,
/// };
/// let summary = run_regen(&options, |_| {}).unwrap();
/// assert!(summary.passed());
/// assert_eq!(summary.counts(), (1, 0));
/// ```
///
/// # Errors
///
/// Returns [`RadError::Parse`] when case resolution fails (see
/// [`resolve_cases`]); per-case problems never surface here.
pub fn run_regen(
    options: &RegenOptions,
    hook: impl Fn(&RegenEvent),
) -> Result<RegenSummary, RadError> {
    let cases = resolve_cases(options)?;
    let total = cases.len();
    let mut summary = RegenSummary::default();
    for (index, case) in cases.iter().enumerate() {
        summary
            .cases
            .push(regen_case(options, case, index, total, &hook));
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rad::TimeHistory;
    use crate::rad::oc_golden::{
        CompareOptions, compare_cycle, compare_history, current_deck_hashes, find_deck,
        golden_json, parse_golden_str,
    };

    const GOLDEN_JSON: &str = r#"{
  "schema": "opencourant-golden-v1",
  "case": "DEMO/case",
  "meta": {
    "engine": "OpenRadioss linux64_gf CommitID deadbeef",
    "engine_commit": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
    "starter_exe": "starter_linux64_gf",
    "engine_exe": "engine_linux64_gf",
    "deck_sha256": {"starter": "aa", "engine": "bb"},
    "starter_deck": "CASE_0000.rad",
    "engine_decks": ["CASE_0001.rad"],
    "starter_options": "-i CASE_0000.rad -np 1",
    "engine_options": "-i CASE_0001.rad",
    "mpi": "smp", "np": 1, "threads": 1, "precision": "dp",
    "harvest_date": "2026-01-01T00:00:00Z",
    "deck_units": ["kg", "m", "s"],
    "rootname": "CASE",
    "normal_termination": true,
    "th_tool": "th_to_csv_linux64_gf",
    "exe_sha256": {"starter": "s", "engine": "e", "th_to_csv": "t"},
    "decimation": null
  },
  "time": [0.0, 0.5, 1.0],
  "channels": {
    "ENERGY": {"values": [0.0, 1.0, 4.0], "final": 4.0, "min": 0.0,
               "max": 4.0, "abs_tol": 4e-12, "rel_tol": 6.6e-4},
    "FORCE": {"values": [10.0, 9.0, 8.0], "final": 8.0, "min": 8.0,
              "max": 10.0, "abs_tol": 1e-11, "rel_tol": 6.6e-4}
  },
  "cycle": {"CYCLE": 3.0, "IENERGY": 4.0}
}"#;

    fn sample_golden() -> Golden {
        parse_golden_str(GOLDEN_JSON, "sample").unwrap()
    }

    fn drift_options(goldens_root: PathBuf) -> RegenOptions {
        RegenOptions {
            goldens_root,
            out_root: None,
            miniqa_root: None,
            exe: None,
            cases: CaseSelect::All,
            scratch_root: PathBuf::from("scratch"),
            threads: 1,
            max_samples: 2000,
            keep_scratch: false,
            local_patches: None,
            drift_only: true,
        }
    }

    fn golden_history_of(golden: &Golden) -> TimeHistory {
        TimeHistory {
            times: golden.time.clone(),
            channels: golden
                .channels
                .iter()
                .map(|(name, channel)| (name.clone(), channel.values.clone()))
                .collect(),
        }
    }

    #[test]
    fn discover_lists_sorted_cases() {
        let root = tempfile::tempdir().unwrap();
        let miniqa = root.path().join("miniqa");
        std::fs::create_dir_all(miniqa.join("sub")).unwrap();
        std::fs::write(miniqa.join("b.json"), "{}").unwrap();
        std::fs::write(miniqa.join("sub").join("deep.json"), "{}").unwrap();
        std::fs::write(miniqa.join("a.json"), "{}").unwrap();
        std::fs::write(miniqa.join("notes.txt"), "ignored").unwrap();
        assert_eq!(
            discover_golden_cases(root.path()).unwrap(),
            vec!["a".to_string(), "b".to_string(), "sub/deep".to_string(),]
        );
    }

    #[test]
    fn discover_rejects_missing_or_empty_corpus() {
        let root = tempfile::tempdir().unwrap();
        assert!(discover_golden_cases(root.path()).is_err());
        std::fs::create_dir_all(root.path().join("miniqa")).unwrap();
        std::fs::write(root.path().join("miniqa").join("notes.txt"), "x").unwrap();
        assert!(discover_golden_cases(root.path()).is_err());
    }

    #[test]
    fn resolve_cases_only() {
        let empty = RegenOptions {
            cases: CaseSelect::Only(Vec::new()),
            ..drift_options(PathBuf::from("goldens"))
        };
        assert!(resolve_cases(&empty).is_err());
        let only = RegenOptions {
            cases: CaseSelect::Only(vec!["b".to_string(), "a".to_string()]),
            ..drift_options(PathBuf::from("goldens"))
        };
        // Explicit lists keep user order (no sorting).
        assert_eq!(
            resolve_cases(&only).unwrap(),
            vec!["b".to_string(), "a".to_string()]
        );
    }

    #[test]
    fn drift_self_passes_with_zero_deltas() {
        let golden = sample_golden();
        let report = drift_diff(&golden, &golden);
        assert!(report.passed());
        assert!(report.failures().is_empty());
        assert_eq!(report.channels.len(), 2);
        for channel in &report.channels {
            assert_eq!(channel.max_abs, 0.0);
            assert_eq!(channel.max_rel, 0.0);
            assert_eq!(channel.bound_ratio, 0.0);
            assert_eq!(channel.samples, 3);
            assert!(channel.summary().contains("PASS"));
        }
        for scalar in &report.cycle {
            assert_eq!(scalar.abs, 0.0);
            assert!(scalar.summary().contains("PASS"));
        }
        assert!(report.summary().contains("2/2 channels"));
    }

    #[test]
    fn drift_perturbed_fails_with_deltas() {
        let baseline = sample_golden();
        let mut candidate = baseline.clone();
        candidate.channels.get_mut("FORCE").unwrap().values[2] = 12.0;
        candidate.cycle.insert("IENERGY".to_string(), 8.0);
        let report = drift_diff(&baseline, &candidate);
        assert!(!report.passed());
        let force = report
            .channels
            .iter()
            .find(|channel| channel.channel == "FORCE")
            .unwrap();
        assert!((force.max_abs - 4.0).abs() < 1e-12);
        assert!((force.max_rel - 0.5).abs() < 1e-12);
        assert_eq!(force.out_of_tolerance, 1);
        assert!(force.bound_ratio > 1.0);
        assert!(force.summary().contains("FAIL"));
        let energy = report
            .channels
            .iter()
            .find(|channel| channel.channel == "ENERGY")
            .unwrap();
        assert!(energy.passed());
        let ienergy = report
            .cycle
            .iter()
            .find(|scalar| scalar.key == "IENERGY")
            .unwrap();
        assert!(!ienergy.passed());
        assert!((ienergy.abs - 4.0).abs() < 1e-12);
        assert!((ienergy.rel - 1.0).abs() < 1e-12);
        assert_eq!(report.failures().len(), 2);
    }

    #[test]
    fn drift_matches_replay_verdicts() {
        let baseline = sample_golden();
        let mut drifted = baseline.clone();
        drifted.channels.get_mut("FORCE").unwrap().values[2] *= 1.5;
        let mut missing = baseline.clone();
        missing.channels.remove("ENERGY");
        let mut short = baseline.clone();
        short.time.truncate(1);
        for channel in short.channels.values_mut() {
            channel.values.truncate(1);
        }
        for candidate in [&baseline, &drifted, &missing, &short] {
            let drift = drift_diff(&baseline, candidate);
            let history = golden_history_of(candidate);
            let replay = compare_history(&baseline, &history, &CompareOptions::default());
            assert_eq!(
                drift.channels.iter().all(ChannelDrift::passed),
                replay.passed(),
                "channel verdict diverged for {:?}",
                candidate.channels.keys().collect::<Vec<_>>()
            );
            assert_eq!(
                drift.cycle.iter().all(CycleDrift::passed),
                compare_cycle(&baseline, &candidate.cycle).is_empty()
            );
        }
    }

    #[test]
    fn drift_cycle_only_goldens() {
        let mut golden = sample_golden();
        golden.time.clear();
        golden.channels.clear();
        golden.meta.th_note = Some("converter failed; cycle scalars only".to_string());
        let report = drift_diff(&golden, &golden);
        assert!(report.passed());
        assert!(report.channels.is_empty());
        let mut drifted = golden.clone();
        drifted.cycle.insert("CYCLE".to_string(), 99.0);
        assert!(!drift_diff(&golden, &drifted).passed());
        // A full baseline against a cycle-only candidate fails closed.
        let full = sample_golden();
        let report = drift_diff(&full, &golden);
        assert!(!report.passed());
        assert!(report.grid_error.is_some());
        assert!(report.summary().contains("empty time grid"));
    }

    #[test]
    fn drift_missing_channel_and_uncovered_grid_fail() {
        let baseline = sample_golden();
        let mut missing = baseline.clone();
        missing.channels.remove("ENERGY");
        let report = drift_diff(&baseline, &missing);
        assert!(!report.passed());
        let energy = &report.channels[0];
        assert!(energy.missing);
        assert!(energy.summary().contains("missing from candidate"));
        // Candidate grid covers only the first stamp.
        let mut short = baseline.clone();
        short.time = vec![0.0, 0.25];
        for channel in short.channels.values_mut() {
            channel.values = vec![channel.values[0], channel.values[0]];
        }
        let report = drift_diff(&baseline, &short);
        assert!(!report.passed());
        assert!(report.channels.iter().all(|c| c.uncovered == 2));
        assert!(report.channels.iter().all(|c| c.samples == 1));
    }

    #[test]
    fn drift_zero_band_and_extra_channels() {
        let mut baseline = sample_golden();
        for channel in baseline.channels.values_mut() {
            channel.values = vec![1e-15, -1e-15, 2e-15];
        }
        let mut candidate = baseline.clone();
        for channel in candidate.channels.values_mut() {
            for value in channel.values.iter_mut() {
                *value *= 2.0;
            }
        }
        // Both sides sit inside the vendor zero band: noise, not drift.
        assert!(drift_diff(&baseline, &candidate).passed());
        // Extra candidate channels are ignored.
        candidate
            .channels
            .insert("EXTRA".to_string(), baseline.channels["FORCE"].clone());
        assert!(drift_diff(&baseline, &candidate).passed());
    }

    #[test]
    fn regen_stage_names() {
        assert_eq!(RegenStage::Start.to_string(), "start");
        assert_eq!(RegenStage::Diffed.to_string(), "diffed");
        assert_eq!(RegenStage::Written.to_string(), "written");
        assert_eq!(RegenStage::Done.to_string(), "done");
        assert_eq!(
            RegenStage::Harvest(HarvestStage::Converted).to_string(),
            "harvest/converted"
        );
    }

    #[test]
    fn case_and_corpus_summaries() {
        let golden = sample_golden();
        let drift = drift_diff(&golden, &golden);
        let mut pass = CaseReport {
            case: "a".to_string(),
            passed: true,
            drift: Some(drift),
            failures: Vec::new(),
            notes: Vec::new(),
            wrote: None,
        };
        assert!(pass.summary().contains("PASS"));
        pass.wrote = Some(PathBuf::from("out/miniqa/a.json"));
        assert!(pass.summary().contains("out/miniqa/a.json"));
        let fail = CaseReport {
            case: "b".to_string(),
            passed: false,
            drift: None,
            failures: vec!["boom".to_string()],
            notes: Vec::new(),
            wrote: None,
        };
        assert!(fail.summary().contains("no drift"));
        let summary = RegenSummary {
            cases: vec![pass, fail],
        };
        assert!(!summary.passed());
        assert_eq!(summary.counts(), (1, 1));
        assert!(summary.summary().contains("1 passed, 1 failed"));
        assert!(!RegenSummary::default().passed());
    }

    #[test]
    fn drift_only_self_check_offline() {
        let root = tempfile::tempdir().unwrap();
        let goldens = root.path().join("goldens");
        let miniqa = goldens.join("miniqa").join("DEMO");
        std::fs::create_dir_all(&miniqa).unwrap();
        std::fs::write(miniqa.join("case.json"), GOLDEN_JSON).unwrap();
        let options = drift_options(goldens);
        let report = regen_case(&options, "DEMO/case", 0, 1, |_| {});
        assert!(report.passed);
        assert!(report.drift.as_ref().unwrap().passed());
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("deck-hash check skipped"))
        );
        assert!(report.wrote.is_none());
    }

    #[test]
    fn drift_only_flags_unstamped_baselines() {
        let root = tempfile::tempdir().unwrap();
        let goldens = root.path().join("goldens");
        let miniqa = goldens.join("miniqa");
        std::fs::create_dir_all(&miniqa).unwrap();
        let mut golden = sample_golden();
        golden.case = "a".to_string();
        golden.meta.engine.clear();
        std::fs::write(miniqa.join("a.json"), golden_json(&golden).unwrap()).unwrap();
        let options = drift_options(goldens);
        let report = regen_case(&options, "a", 0, 1, |_| {});
        assert!(!report.passed);
        assert!(
            report
                .failures
                .iter()
                .any(|line| line.contains("meta.engine"))
        );
    }

    #[test]
    fn drift_only_checks_deck_hashes_when_miniqa_present() {
        let root = tempfile::tempdir().unwrap();
        let goldens = root.path().join("goldens");
        std::fs::create_dir_all(goldens.join("miniqa")).unwrap();
        let data = root.path().join("miniqa").join("a").join("data");
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(data.join("A_0000.rad"), "starter-v1").unwrap();
        std::fs::write(data.join("A_0001.rad"), "engine-v1").unwrap();
        let deck = find_deck(&root.path().join("miniqa"), "a").unwrap();
        let mut golden = sample_golden();
        golden.case = "a".to_string();
        golden.meta.starter_deck = "A_0000.rad".to_string();
        golden.meta.engine_decks = vec!["A_0001.rad".to_string()];
        golden.meta.deck_sha256 = current_deck_hashes(&deck).unwrap();
        std::fs::write(
            goldens.join("miniqa").join("a.json"),
            golden_json(&golden).unwrap(),
        )
        .unwrap();
        let options = RegenOptions {
            miniqa_root: Some(root.path().join("miniqa")),
            ..drift_options(goldens.clone())
        };
        let report = regen_case(&options, "a", 0, 1, |_| {});
        assert!(report.passed, "failures: {:?}", report.failures);
        std::fs::write(data.join("A_0000.rad"), "starter-v2").unwrap();
        let report = regen_case(&options, "a", 0, 1, |_| {});
        assert!(!report.passed);
        assert!(report.failures.iter().any(|line| line.contains("stale")));
    }

    #[test]
    fn drift_only_diffs_staged_candidates() {
        let root = tempfile::tempdir().unwrap();
        let goldens = root.path().join("goldens");
        let staged_root = root.path().join("staged");
        std::fs::create_dir_all(goldens.join("miniqa")).unwrap();
        let mut golden = sample_golden();
        golden.case = "a".to_string();
        std::fs::write(
            goldens.join("miniqa").join("a.json"),
            golden_json(&golden).unwrap(),
        )
        .unwrap();
        let options = RegenOptions {
            out_root: Some(staged_root.clone()),
            ..drift_options(goldens)
        };
        // No staged file: self-check with a note.
        let report = regen_case(&options, "a", 0, 1, |_| {});
        assert!(report.passed);
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("self-checking"))
        );
        // Staged drifted golden: the drift fails.
        std::fs::create_dir_all(staged_root.join("miniqa")).unwrap();
        golden.channels.get_mut("FORCE").unwrap().values[2] = 99.0;
        std::fs::write(
            staged_root.join("miniqa").join("a.json"),
            golden_json(&golden).unwrap(),
        )
        .unwrap();
        let report = regen_case(&options, "a", 0, 1, |_| {});
        assert!(!report.passed);
        assert!(!report.drift.as_ref().unwrap().passed());
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("staged golden"))
        );
    }

    #[test]
    fn live_case_without_toolchain_fails_offline() {
        let root = tempfile::tempdir().unwrap();
        let goldens = root.path().join("goldens");
        std::fs::create_dir_all(goldens.join("miniqa")).unwrap();
        let mut golden = sample_golden();
        golden.case = "a".to_string();
        std::fs::write(
            goldens.join("miniqa").join("a.json"),
            golden_json(&golden).unwrap(),
        )
        .unwrap();
        let mut options = drift_options(goldens);
        options.drift_only = false;
        let report = regen_case(&options, "a", 0, 1, |_| {});
        assert!(!report.passed);
        assert!(report.failures.iter().any(|line| line.contains("--miniqa")));
        assert!(report.wrote.is_none());
        options.miniqa_root = Some(PathBuf::from("miniqa"));
        let report = regen_case(&options, "a", 0, 1, |_| {});
        assert!(!report.passed);
        assert!(report.failures.iter().any(|line| line.contains("--exe")));
    }

    #[test]
    fn run_regen_continues_past_failures() {
        let root = tempfile::tempdir().unwrap();
        let goldens = root.path().join("goldens");
        std::fs::create_dir_all(goldens.join("miniqa")).unwrap();
        let mut golden = sample_golden();
        golden.case = "a".to_string();
        std::fs::write(
            goldens.join("miniqa").join("a.json"),
            golden_json(&golden).unwrap(),
        )
        .unwrap();
        std::fs::write(goldens.join("miniqa").join("broken.json"), "{oops").unwrap();
        let options = RegenOptions {
            cases: CaseSelect::Only(vec!["a".to_string(), "missing".to_string()]),
            ..drift_options(goldens)
        };
        let stages = std::cell::RefCell::new(Vec::new());
        let summary = run_regen(&options, |event| {
            stages.borrow_mut().push(event.stage);
        })
        .unwrap();
        assert!(!summary.passed());
        assert_eq!(summary.counts(), (1, 1));
        assert!(summary.cases[0].passed);
        assert!(!summary.cases[1].passed);
        assert!(stages.borrow().contains(&RegenStage::Start));
        assert!(stages.borrow().contains(&RegenStage::Done));
    }

    #[test]
    fn run_regen_rejects_empty_selection() {
        let options = RegenOptions {
            cases: CaseSelect::Only(Vec::new()),
            ..drift_options(PathBuf::from("goldens"))
        };
        assert!(run_regen(&options, |_| {}).is_err());
    }
}
