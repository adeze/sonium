//! Perf-floor baseline comparison for the OpenCourant bench.
//!
//! Compares a candidate `sonium.oc-bench/v1` bundle (sonium arm only)
//! against the checked-in baseline and fails closed on any schema,
//! protocol, or matrix drift. The gate catches order-of-magnitude
//! (10x) explicit-cycle regressions on noisy shared CI runners; it
//! deliberately does not chase small percentage changes.
//!
//! The bundle field names mirror the frozen schema in
//! `PHASE6_CONTRACTS.md`; this module defines its own serde types so
//! the gate keeps working even while lane 6A evolves its producer.
//! Comparison semantics reuse `compute_regression.rs` where the cell
//! shapes fit: same-hardware policy, correctness preflight, duplicate
//! and empty-matrix rejection, exact key-set matching, slowdown
//! fractions, and a `passed()` predicate that requires conclusive
//! finite comparisons. Deliberate deviations: oc-bench rows carry no
//! confidence intervals (the producer collapses repeats into one
//! `wall_s` per cell), so there is no `Inconclusive` state and no
//! `Improved` state (anything inside the 10x floor is `Stable`); hardware
//! matching requires only `arch` equality because CI runners share an
//! ISA but not a chip model; and the default threshold is 9.0 (a 10x
//! slowdown) instead of 0.10, per the lane contract.
//!
//! # Examples
//!
//! ```rust,no_run
//! use std::path::Path;
//! use sonium_bench::oc_floor::{FloorPolicy, check_floor};
//!
//! let report = check_floor(
//!     Path::new("crates/sonium-bench/baselines/oc_floor.json"),
//!     Path::new("candidate.json"),
//!     FloorPolicy::default(),
//! )
//! .expect("fail-closed floor comparison");
//! assert!(report.passed());
//! ```

// Rust guideline compliant 2026-02-21

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Frozen evidence schema tag this gate accepts.
pub const OC_BENCH_SCHEMA: &str = "sonium.oc-bench/v1";

/// Schema tag emitted on floor reports.
pub const OC_FLOOR_REPORT_SCHEMA: &str = "sonium.oc-floor-report/v1";

/// Default maximum tolerated slowdown as a fraction.
///
/// A value of 9.0 means the candidate may run up to 10x slower per
/// degree of freedom and cycle than the baseline. The bar comes from
/// the lane contract ("new waves can't silently 10x the explicit
/// cycle"): shared CI runners exhibit multi-x clock jitter under
/// load, so a tight bound would flake, while anything near 10x is
/// far outside plausible noise and signals a real algorithmic or
/// build regression. Lowering this value makes the gate flaky on
/// shared runners; raising it lets real regressions through.
pub const DEFAULT_MAX_SLOWDOWN_FRACTION: f64 = 9.0;

/// Relative tolerance for the producer's own timing identity.
///
/// `ns_per_dof_cycle` must equal `wall_s * 1e9 / (cycles * dof)`
/// within 1%. The slack covers JSON rounding of the precomputed
/// field, not measurement noise. Tightening it risks rejecting
/// valid producer output; loosening it weakens a cheap
/// defense against corrupt or hand-edited bundles.
pub const CONSISTENCY_TOLERANCE: f64 = 0.01;

/// Hardware identity from the frozen bundle schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloorHardware {
    pub arch: String,
    pub model: String,
    pub threads: String,
}

/// Timing protocol from the frozen bundle schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FloorProtocol {
    pub repeats: u64,
    pub warmup: u64,
    pub correctness: String,
}

/// One benchmark cell from the frozen bundle schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloorRow {
    pub deck: String,
    pub side: String,
    pub flavor: String,
    pub threads: u64,
    pub wall_s: f64,
    pub cycles: u64,
    pub dof: u64,
    pub ns_per_dof_cycle: f64,
    pub peak_rss_bytes: u64,
    pub correctness_passed: bool,
}

/// Deserialized subset of an oc-bench evidence bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloorBundle {
    pub schema: String,
    pub hardware: FloorHardware,
    pub protocol: FloorProtocol,
    pub rows: Vec<FloorRow>,
}

/// Tunable but explicit floor policy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FloorPolicy {
    pub max_slowdown_fraction: f64,
}

impl Default for FloorPolicy {
    fn default() -> Self {
        Self {
            max_slowdown_fraction: DEFAULT_MAX_SLOWDOWN_FRACTION,
        }
    }
}

/// Stable identity for a benchmark cell.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FloorKey {
    pub deck: String,
    pub side: String,
    pub flavor: String,
}

impl From<&FloorRow> for FloorKey {
    fn from(row: &FloorRow) -> Self {
        Self {
            deck: row.deck.clone(),
            side: row.side.clone(),
            flavor: row.flavor.clone(),
        }
    }
}

/// Classification of a comparable timing cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FloorStatus {
    Stable,
    Regressed,
}

/// Comparison result for one timing cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloorComparison {
    pub key: FloorKey,
    pub baseline_ns_per_dof_cycle: f64,
    pub candidate_ns_per_dof_cycle: f64,
    pub slowdown_fraction: f64,
    pub rss_ratio: f64,
    pub status: FloorStatus,
}

fn floor_report_schema() -> String {
    OC_FLOOR_REPORT_SCHEMA.to_string()
}

/// Machine-readable output of the floor gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloorReport {
    #[serde(default = "floor_report_schema")]
    pub schema: String,
    pub max_slowdown_fraction: f64,
    pub baseline_hardware: FloorHardware,
    pub candidate_hardware: FloorHardware,
    pub comparisons: Vec<FloorComparison>,
    pub regression_count: usize,
}

impl FloorReport {
    /// Whether the candidate clears the perf floor.
    ///
    /// A pass requires zero regressions, a finite non-negative
    /// threshold, at least one comparison, and every cell stable
    /// within the threshold.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.regression_count == 0
            && self.max_slowdown_fraction.is_finite()
            && self.max_slowdown_fraction >= 0.0
            && !self.comparisons.is_empty()
            && self.comparisons.iter().all(|cell| {
                cell.slowdown_fraction.is_finite()
                    && cell.slowdown_fraction <= self.max_slowdown_fraction
                    && matches!(cell.status, FloorStatus::Stable)
            })
    }
}

/// Load an oc-bench bundle from disk.
///
/// # Errors
///
/// Returns a message when the file cannot be read, is not valid
/// JSON, does not match the bundle shape, or carries a schema tag
/// other than `sonium.oc-bench/v1`.
pub fn load_bundle(path: &Path) -> Result<FloorBundle, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("cannot read bundle {}: {error}", path.display()))?;
    let bundle: FloorBundle = serde_json::from_slice(&bytes)
        .map_err(|error| format!("cannot parse bundle {}: {error}", path.display()))?;
    if bundle.schema != OC_BENCH_SCHEMA {
        return Err(format!(
            "bundle {} has schema {:?}, expected {OC_BENCH_SCHEMA:?}",
            path.display(),
            bundle.schema
        ));
    }
    Ok(bundle)
}

fn validate_policy(policy: FloorPolicy) -> Result<(), String> {
    if !policy.max_slowdown_fraction.is_finite() || policy.max_slowdown_fraction < 0.0 {
        return Err(format!(
            "max_slowdown_fraction must be finite and non-negative, got {}",
            policy.max_slowdown_fraction
        ));
    }
    Ok(())
}

fn validate_row(row: &FloorRow, source: &str) -> Result<(), String> {
    if row.deck.trim().is_empty() || row.side.trim().is_empty() || row.flavor.trim().is_empty() {
        return Err(format!("{source} row has empty deck, side, or flavor"));
    }
    if row.threads == 0 {
        return Err(format!(
            "{source} row {:?} has zero threads",
            FloorKey::from(row)
        ));
    }
    if !row.correctness_passed {
        return Err(format!(
            "{source} row {:?} failed its numerical correctness preflight",
            FloorKey::from(row)
        ));
    }
    if !row.wall_s.is_finite() || row.wall_s <= 0.0 {
        return Err(format!(
            "{source} row {:?} has invalid wall_s={}",
            FloorKey::from(row),
            row.wall_s
        ));
    }
    if row.cycles == 0 || row.dof == 0 {
        return Err(format!(
            "{source} row {:?} has zero cycles or dof",
            FloorKey::from(row)
        ));
    }
    if !row.ns_per_dof_cycle.is_finite() || row.ns_per_dof_cycle <= 0.0 {
        return Err(format!(
            "{source} row {:?} has invalid ns_per_dof_cycle={}",
            FloorKey::from(row),
            row.ns_per_dof_cycle
        ));
    }
    let expected = row.wall_s * 1e9 / (row.cycles as f64 * row.dof as f64);
    let drift = (row.ns_per_dof_cycle - expected).abs() / expected;
    if !drift.is_finite() || drift > CONSISTENCY_TOLERANCE {
        return Err(format!(
            "{source} row {:?} is self-inconsistent: ns_per_dof_cycle={} but wall_s/cycles/dof imply {expected}",
            FloorKey::from(row),
            row.ns_per_dof_cycle
        ));
    }
    if row.peak_rss_bytes == 0 {
        return Err(format!(
            "{source} row {:?} reports zero peak RSS; the producer must measure it",
            FloorKey::from(row)
        ));
    }
    Ok(())
}

fn index_rows<'a>(
    bundle: &'a FloorBundle,
    source: &str,
) -> Result<BTreeMap<FloorKey, &'a FloorRow>, String> {
    let mut indexed = BTreeMap::new();
    for row in &bundle.rows {
        validate_row(row, source)?;
        let key = FloorKey::from(row);
        if indexed.insert(key.clone(), row).is_some() {
            return Err(format!("{source} contains duplicate row {key:?}"));
        }
    }
    if indexed.is_empty() {
        return Err(format!("{source} contains no benchmark rows"));
    }
    Ok(indexed)
}

/// Compare a candidate bundle against the baseline floor.
///
/// # Errors
///
/// Returns a message on any fail-closed condition: invalid policy,
/// schema drift, architecture mismatch, protocol drift, row
/// validation failures, duplicate or empty matrices, deck-set
/// drift, thread/cycle/dof drift per cell, or a non-finite timing
/// ratio. Callers must treat every error as a gate failure.
pub fn compare_floor(
    baseline: &FloorBundle,
    candidate: &FloorBundle,
    policy: FloorPolicy,
) -> Result<FloorReport, String> {
    validate_policy(policy)?;
    for (source, bundle) in [("baseline", baseline), ("candidate", candidate)] {
        if bundle.schema != OC_BENCH_SCHEMA {
            return Err(format!(
                "{source} has schema {:?}, expected {OC_BENCH_SCHEMA:?}",
                bundle.schema
            ));
        }
    }
    if baseline.hardware.arch != candidate.hardware.arch {
        return Err(format!(
            "architecture mismatch: baseline={:?}, candidate={:?}; regenerate the baseline per ISA",
            baseline.hardware.arch, candidate.hardware.arch
        ));
    }
    if baseline.protocol != candidate.protocol {
        return Err(format!(
            "timing protocol mismatch: baseline={:?}, candidate={:?}",
            baseline.protocol, candidate.protocol
        ));
    }

    let baseline_rows = index_rows(baseline, "baseline")?;
    let candidate_rows = index_rows(candidate, "candidate")?;
    let baseline_keys = baseline_rows.keys().cloned().collect::<BTreeSet<_>>();
    let candidate_keys = candidate_rows.keys().cloned().collect::<BTreeSet<_>>();
    if baseline_keys != candidate_keys {
        let missing = baseline_keys
            .difference(&candidate_keys)
            .collect::<Vec<_>>();
        let extra = candidate_keys
            .difference(&baseline_keys)
            .collect::<Vec<_>>();
        return Err(format!(
            "benchmark matrix mismatch: missing={missing:?}, extra={extra:?}"
        ));
    }

    let mut comparisons = Vec::with_capacity(baseline_rows.len());
    let mut regression_count = 0;
    for (key, baseline_row) in baseline_rows {
        let candidate_row = candidate_rows[&key];
        for (field, old, new) in [
            ("threads", baseline_row.threads, candidate_row.threads),
            ("cycles", baseline_row.cycles, candidate_row.cycles),
            ("dof", baseline_row.dof, candidate_row.dof),
        ] {
            if old != new {
                return Err(format!(
                    "cell {key:?} drifted in {field}: baseline={old}, candidate={new}; regenerate the baseline"
                ));
            }
        }
        let slowdown_fraction =
            candidate_row.ns_per_dof_cycle / baseline_row.ns_per_dof_cycle - 1.0;
        if !slowdown_fraction.is_finite() {
            return Err(format!("timing ratio overflows for {key:?}"));
        }
        let rss_ratio = candidate_row.peak_rss_bytes as f64 / baseline_row.peak_rss_bytes as f64;
        let status = if slowdown_fraction > policy.max_slowdown_fraction {
            regression_count += 1;
            FloorStatus::Regressed
        } else {
            FloorStatus::Stable
        };
        comparisons.push(FloorComparison {
            key,
            baseline_ns_per_dof_cycle: baseline_row.ns_per_dof_cycle,
            candidate_ns_per_dof_cycle: candidate_row.ns_per_dof_cycle,
            slowdown_fraction,
            rss_ratio,
            status,
        });
    }

    Ok(FloorReport {
        schema: floor_report_schema(),
        max_slowdown_fraction: policy.max_slowdown_fraction,
        baseline_hardware: baseline.hardware.clone(),
        candidate_hardware: candidate.hardware.clone(),
        comparisons,
        regression_count,
    })
}

/// Load two bundles from disk and compare them.
///
/// This is the entry point the CI gate uses: it fails closed when
/// either file is missing or stale.
///
/// # Errors
///
/// Returns a message when either bundle fails to load or the
/// comparison reports a fail-closed condition.
pub fn check_floor(
    baseline_path: &Path,
    candidate_path: &Path,
    policy: FloorPolicy,
) -> Result<FloorReport, String> {
    let baseline = load_bundle(baseline_path)?;
    let candidate = load_bundle(candidate_path)?;
    compare_floor(&baseline, &candidate, policy)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(ns_per_dof_cycle: f64) -> FloorRow {
        let cycles = 900;
        let dof = 15_000;
        let wall_s = ns_per_dof_cycle * cycles as f64 * dof as f64 / 1e9;
        FloorRow {
            deck: "rail_crush".into(),
            side: "sonium".into(),
            flavor: "native".into(),
            threads: 1,
            wall_s,
            cycles,
            dof,
            ns_per_dof_cycle,
            peak_rss_bytes: 123_456_789,
            correctness_passed: true,
        }
    }

    fn bundle(row: FloorRow) -> FloorBundle {
        FloorBundle {
            schema: OC_BENCH_SCHEMA.into(),
            hardware: FloorHardware {
                arch: "x86_64".into(),
                model: "test-runner".into(),
                threads: "1".into(),
            },
            protocol: FloorProtocol {
                repeats: 1,
                warmup: 0,
                correctness: "probe-tolerances".into(),
            },
            rows: vec![row],
        }
    }

    #[test]
    fn identical_bundles_pass_as_stable() {
        let report = compare_floor(
            &bundle(row(911.111_111_111_111_1)),
            &bundle(row(911.111_111_111_111_1)),
            FloorPolicy::default(),
        )
        .unwrap();
        assert!(report.passed());
        assert_eq!(report.comparisons[0].status, FloorStatus::Stable);
    }

    #[test]
    fn tenfold_slowdown_is_the_failing_boundary() {
        let baseline = 911.111_111_111_111_1;
        let passing = compare_floor(
            &bundle(row(baseline)),
            &bundle(row(baseline * 9.99)),
            FloorPolicy::default(),
        )
        .unwrap();
        assert!(passing.passed());
        let failing = compare_floor(
            &bundle(row(baseline)),
            &bundle(row(baseline * 10.01)),
            FloorPolicy::default(),
        )
        .unwrap();
        assert!(!failing.passed());
        assert_eq!(failing.comparisons[0].status, FloorStatus::Regressed);
    }

    #[test]
    fn self_inconsistent_timing_is_rejected() {
        let mut bad = row(911.111_111_111_111_1);
        bad.ns_per_dof_cycle *= 1.5;
        let error = compare_floor(
            &bundle(row(911.111_111_111_111_1)),
            &bundle(bad),
            FloorPolicy::default(),
        )
        .unwrap_err();
        assert!(error.contains("self-inconsistent"));
    }
}
