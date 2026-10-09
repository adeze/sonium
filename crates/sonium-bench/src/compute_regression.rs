//! Statistical regression contract for release-facing compute evidence.
//!
//! The timing producer remains platform-specific, but comparison semantics
//! live here so CI, release tooling, and unit tests use one fail-closed policy.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Default maximum tolerated median slowdown before a non-overlapping
/// confidence interval becomes a release failure.
pub const DEFAULT_MAX_MEDIAN_SLOWDOWN: f64 = 0.10;

/// Deserialized subset of a compute performance evidence bundle.
///
/// `Serialize` is derived alongside `Deserialize` so snapshots can be
/// re-emitted for trend analysis. Source commits may differ, but the timing
/// protocol and complete hardware/build environment must match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSnapshot {
    pub schema: String,
    pub hardware: HardwareFingerprint,
    #[serde(default)]
    pub protocol: serde_json::Value,
    pub rows: Vec<PerformanceRow>,
}

/// Hardware fields that must agree before timing samples are comparable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HardwareFingerprint {
    pub arch: String,
    #[serde(default)]
    pub model: String,
    pub rayon_threads: String,
    /// Preserve all producer-supplied environment fields instead of silently
    /// discarding compiler/device/driver/clock identity during deserialization.
    #[serde(flatten)]
    pub environment: BTreeMap<String, serde_json::Value>,
}

/// One benchmark cell from an evidence bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRow {
    pub implementation: String,
    pub blas_level: String,
    pub operation: String,
    pub regime: String,
    pub dtype: String,
    pub shape: String,
    /// Rayon worker count the timed kernels ran under, as a string. Rows
    /// from one evidence bundle may span several counts (the macOS binary
    /// sweeps 1 and N threads for cache/memory regimes); the count is part
    /// of the cell identity so cross-thread comparisons fail closed instead
    /// of comparing incomparable timings. Pre-threads baselines deserialize
    /// with `"unknown"` via the serde default below.
    #[serde(default = "default_row_threads")]
    pub threads: String,
    pub correctness_passed: bool,
    pub median_ns: f64,
    pub ci95_low_ns: f64,
    pub ci95_high_ns: f64,
}

fn default_row_threads() -> String {
    "unknown".into()
}

/// Tunable but explicit comparison policy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RegressionPolicy {
    pub max_median_slowdown: f64,
    pub require_non_overlapping_interval: bool,
}

impl Default for RegressionPolicy {
    fn default() -> Self {
        Self {
            max_median_slowdown: DEFAULT_MAX_MEDIAN_SLOWDOWN,
            require_non_overlapping_interval: true,
        }
    }
}

/// Stable identity for a benchmark cell.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RowKey {
    pub implementation: String,
    pub blas_level: String,
    pub operation: String,
    pub regime: String,
    pub dtype: String,
    pub shape: String,
    pub threads: String,
}

impl From<&PerformanceRow> for RowKey {
    fn from(row: &PerformanceRow) -> Self {
        Self {
            implementation: row.implementation.clone(),
            blas_level: row.blas_level.clone(),
            operation: row.operation.clone(),
            regime: row.regime.clone(),
            dtype: row.dtype.clone(),
            shape: row.shape.clone(),
            threads: row.threads.clone(),
        }
    }
}

/// Classification of a comparable timing cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegressionStatus {
    Stable,
    Improved,
    Inconclusive,
    Regressed,
}

/// Comparison result for one timing cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowComparison {
    pub key: RowKey,
    pub baseline_median_ns: f64,
    pub candidate_median_ns: f64,
    pub slowdown_fraction: f64,
    pub confidence_intervals_overlap: bool,
    pub status: RegressionStatus,
}

fn regression_report_schema() -> String {
    "sonium-compute-regression-report/v2".to_string()
}

/// Machine-readable output of the regression gate.
///
/// `Deserialize` is derived so `regression.json` can be re-read for trend
/// analysis; `schema` falls back to the current report schema when absent.
/// The schema is an owned `String` (rather than `&'static str`) precisely so
/// a report deserialized from a file buffer does not borrow from it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    #[serde(default = "regression_report_schema")]
    pub schema: String,
    pub max_median_slowdown: f64,
    pub require_non_overlapping_interval: bool,
    pub comparisons: Vec<RowComparison>,
    pub regression_count: usize,
}

impl RegressionReport {
    /// A release pass requires conclusive, finite comparisons. An inconclusive
    /// slowdown requires more evidence; it is not proof of no regression.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.regression_count == 0
            && self.max_median_slowdown.is_finite()
            && self.max_median_slowdown >= 0.0
            && !self.comparisons.is_empty()
            && self.comparisons.iter().all(|row| {
                row.slowdown_fraction.is_finite()
                    && row.slowdown_fraction <= self.max_median_slowdown
                    && matches!(
                        row.status,
                        RegressionStatus::Stable | RegressionStatus::Improved
                    )
            })
    }
}

fn validate_policy(policy: RegressionPolicy) -> Result<(), String> {
    if !policy.max_median_slowdown.is_finite() || policy.max_median_slowdown < 0.0 {
        return Err(format!(
            "max_median_slowdown must be finite and non-negative, got {}",
            policy.max_median_slowdown
        ));
    }
    Ok(())
}

fn validate_row(row: &PerformanceRow, source: &str) -> Result<(), String> {
    if row
        .threads
        .parse::<usize>()
        .ok()
        .filter(|&n| n > 0)
        .is_none()
    {
        return Err(format!("{source} row has unknown or invalid thread count"));
    }
    if !row.correctness_passed {
        return Err(format!(
            "{source} row {:?} failed its numerical correctness preflight",
            RowKey::from(row)
        ));
    }
    if row.median_ns == 0.0 {
        return Err(format!(
            "{source} median_ns must be positive; increase timing batch size"
        ));
    }
    for (name, value) in [
        ("median_ns", row.median_ns),
        ("ci95_low_ns", row.ci95_low_ns),
        ("ci95_high_ns", row.ci95_high_ns),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "{source} row {:?} has invalid {name}={value}",
                RowKey::from(row)
            ));
        }
    }
    if row.ci95_low_ns > row.median_ns || row.median_ns > row.ci95_high_ns {
        return Err(format!(
            "{source} row {:?} does not bracket its median with the confidence interval",
            RowKey::from(row)
        ));
    }
    Ok(())
}

fn index_rows<'a>(
    snapshot: &'a EvidenceSnapshot,
    source: &str,
) -> Result<BTreeMap<RowKey, &'a PerformanceRow>, String> {
    let mut indexed = BTreeMap::new();
    for row in &snapshot.rows {
        validate_row(row, source)?;
        let key = RowKey::from(row);
        if indexed.insert(key.clone(), row).is_some() {
            return Err(format!("{source} contains duplicate row {key:?}"));
        }
    }
    if indexed.is_empty() {
        return Err(format!("{source} contains no performance rows"));
    }
    Ok(indexed)
}

fn validate_environment(snapshot: &EvidenceSnapshot, source: &str) -> Result<(), String> {
    fn known(value: &str) -> bool {
        !value.trim().is_empty()
            && !matches!(
                value,
                "unknown" | "unavailable" | "not_measured" | "runtime-default"
            )
    }
    let hardware = &snapshot.hardware;
    if !known(&hardware.arch)
        || !known(&hardware.model)
        || hardware
            .rayon_threads
            .parse::<usize>()
            .ok()
            .filter(|&n| n > 0)
            .is_none()
    {
        return Err(format!("{source} has incomplete hardware identity"));
    }
    for key in [
        "os",
        "rustc",
        "profile",
        "target_features",
        "device",
        "driver",
        "clocks",
        "thermal_state",
    ] {
        if !hardware
            .environment
            .get(key)
            .and_then(serde_json::Value::as_str)
            .is_some_and(known)
        {
            return Err(format!(
                "{source} lacks comparable hardware/build field {key}"
            ));
        }
    }
    let protocol = snapshot
        .protocol
        .as_object()
        .ok_or_else(|| format!("{source} lacks a timing protocol"))?;
    for key in [
        "confidence_interval",
        "correctness",
        "allocation_measurement",
        "input_generator",
    ] {
        if !protocol
            .get(key)
            .and_then(serde_json::Value::as_str)
            .is_some_and(known)
        {
            return Err(format!("{source} lacks timing protocol field {key}"));
        }
    }
    for key in ["warmup_samples", "measured_samples"] {
        if !protocol
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|n| n > 0)
        {
            return Err(format!(
                "{source} lacks positive timing protocol field {key}"
            ));
        }
    }
    Ok(())
}

/// Compare a candidate evidence bundle with a same-hardware baseline.
///
/// The matrix is fail-closed: missing/extra cells, correctness failures,
/// malformed intervals, schema drift, and hardware/thread mismatches are
/// contract errors rather than silently incomparable timing data.
pub fn compare_evidence(
    baseline: &EvidenceSnapshot,
    candidate: &EvidenceSnapshot,
    policy: RegressionPolicy,
) -> Result<RegressionReport, String> {
    validate_policy(policy)?;
    if baseline.schema != candidate.schema {
        return Err(format!(
            "evidence schema mismatch: baseline={}, candidate={}",
            baseline.schema, candidate.schema
        ));
    }
    if baseline.hardware != candidate.hardware {
        return Err(format!(
            "hardware/thread mismatch: baseline={:?}, candidate={:?}",
            baseline.hardware, candidate.hardware
        ));
    }
    for (source, snapshot) in [("baseline", baseline), ("candidate", candidate)] {
        validate_environment(snapshot, source)?;
    }
    if baseline.protocol != candidate.protocol {
        return Err("timing protocol mismatch".into());
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
        let slowdown_fraction = candidate_row.median_ns / baseline_row.median_ns - 1.0;
        if !slowdown_fraction.is_finite() {
            return Err(format!("timing ratio overflows for {key:?}"));
        }
        let confidence_intervals_overlap = candidate_row.ci95_low_ns <= baseline_row.ci95_high_ns
            && baseline_row.ci95_low_ns <= candidate_row.ci95_high_ns;
        let exceeds_threshold = slowdown_fraction > policy.max_median_slowdown;
        let statistically_separated =
            !policy.require_non_overlapping_interval || !confidence_intervals_overlap;
        let status = if exceeds_threshold && statistically_separated {
            regression_count += 1;
            RegressionStatus::Regressed
        } else if slowdown_fraction < -policy.max_median_slowdown && statistically_separated {
            RegressionStatus::Improved
        } else if exceeds_threshold {
            RegressionStatus::Inconclusive
        } else {
            RegressionStatus::Stable
        };
        comparisons.push(RowComparison {
            key,
            baseline_median_ns: baseline_row.median_ns,
            candidate_median_ns: candidate_row.median_ns,
            slowdown_fraction,
            confidence_intervals_overlap,
            status,
        });
    }

    Ok(RegressionReport {
        schema: regression_report_schema(),
        max_median_slowdown: policy.max_median_slowdown,
        require_non_overlapping_interval: policy.require_non_overlapping_interval,
        comparisons,
        regression_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(median: f64, low: f64, high: f64) -> PerformanceRow {
        PerformanceRow {
            implementation: "sonium-cpu".into(),
            blas_level: "blas-2".into(),
            operation: "gemv".into(),
            regime: "tail".into(),
            dtype: "f32".into(),
            shape: "31x29".into(),
            threads: "1".into(),
            correctness_passed: true,
            median_ns: median,
            ci95_low_ns: low,
            ci95_high_ns: high,
        }
    }

    fn snapshot(row: PerformanceRow) -> EvidenceSnapshot {
        EvidenceSnapshot {
            schema: "sonium-compute-macos-evidence/v3".into(),
            hardware: HardwareFingerprint {
                arch: "aarch64".into(),
                model: "Mac15,6".into(),
                rayon_threads: "1".into(),
                environment: [
                    "os",
                    "rustc",
                    "profile",
                    "target_features",
                    "device",
                    "driver",
                    "clocks",
                    "thermal_state",
                ]
                .into_iter()
                .map(|key| (key.to_string(), serde_json::json!("test-controlled")))
                .collect(),
            },
            protocol: serde_json::json!({
                "warmup_samples": 5, "measured_samples": 15,
                "confidence_interval": "binomial", "correctness": "independent-reference",
                "allocation_measurement": "preallocated-kernel-only", "input_generator": "test-v1",
            }),
            rows: vec![row],
        }
    }

    #[test]
    fn separated_slowdown_fails_gate() {
        let report = compare_evidence(
            &snapshot(row(100.0, 95.0, 105.0)),
            &snapshot(row(120.0, 115.0, 125.0)),
            RegressionPolicy::default(),
        )
        .unwrap();
        assert!(!report.passed());
        assert_eq!(report.comparisons[0].status, RegressionStatus::Regressed);
    }

    #[test]
    fn overlapping_interval_makes_slowdown_inconclusive() {
        let report = compare_evidence(
            &snapshot(row(100.0, 90.0, 115.0)),
            &snapshot(row(120.0, 105.0, 130.0)),
            RegressionPolicy::default(),
        )
        .unwrap();
        assert!(!report.passed());
        assert_eq!(report.comparisons[0].status, RegressionStatus::Inconclusive);
    }

    #[test]
    fn faster_candidate_is_classified_as_improved() {
        let report = compare_evidence(
            &snapshot(row(100.0, 95.0, 105.0)),
            &snapshot(row(75.0, 70.0, 80.0)),
            RegressionPolicy::default(),
        )
        .unwrap();
        assert!(report.passed());
        assert_eq!(report.comparisons[0].status, RegressionStatus::Improved);
    }

    #[test]
    fn correctness_failure_is_rejected_before_timing_comparison() {
        let baseline = snapshot(row(100.0, 95.0, 105.0));
        let mut candidate_row = row(100.0, 95.0, 105.0);
        candidate_row.correctness_passed = false;
        let error = compare_evidence(
            &baseline,
            &snapshot(candidate_row),
            RegressionPolicy::default(),
        )
        .unwrap_err();
        assert!(error.contains("correctness preflight"));
    }

    #[test]
    fn missing_extra_duplicate_and_nonfinite_rows_fail_closed() {
        let baseline = snapshot(row(100.0, 95.0, 105.0));

        let mut empty = baseline.clone();
        empty.rows.clear();
        assert!(
            compare_evidence(&baseline, &empty, RegressionPolicy::default())
                .unwrap_err()
                .contains("no performance rows")
        );

        let mut duplicate = baseline.clone();
        duplicate.rows.push(duplicate.rows[0].clone());
        assert!(
            compare_evidence(&baseline, &duplicate, RegressionPolicy::default())
                .unwrap_err()
                .contains("duplicate row")
        );

        let mut nonfinite = baseline.clone();
        nonfinite.rows[0].median_ns = f64::NAN;
        assert!(
            compare_evidence(&baseline, &nonfinite, RegressionPolicy::default())
                .unwrap_err()
                .contains("invalid median_ns")
        );
    }

    #[test]
    fn evidence_envelope_with_unknown_fields_still_deserializes() {
        // Old records remain readable for history, but cannot pass admission
        // without the compiler/build/protocol identity now required.
        let envelope = serde_json::json!({
            "schema": "sonium-compute-macos-evidence/v3",
            "git_sha": "abc123",
            "timestamp_utc": "2026-09-04T00:00:00Z",
            "hardware": {"arch": "aarch64", "model": "Mac15,6", "rayon_threads": "1"},
            "protocol": {"warmup_samples": 5},
            "rows": [{
                "implementation": "sonium-cpu", "blas_level": "blas-2",
                "operation": "gemv", "regime": "tail", "dtype": "f32",
                "shape": "31x29", "correctness_passed": true,
                "median_ns": 100.0, "ci95_low_ns": 95.0, "ci95_high_ns": 105.0,
            }],
            "limitations": ["not_measured"],
        });
        let snapshot: EvidenceSnapshot = serde_json::from_value(envelope).unwrap();
        assert_eq!(snapshot.rows.len(), 1);
        assert!(compare_evidence(&snapshot, &snapshot, RegressionPolicy::default()).is_err());
    }

    #[test]
    fn threads_dimension_is_part_of_cell_identity() {
        let mut one_thread = row(100.0, 95.0, 105.0);
        one_thread.threads = "1".into();
        let mut many_threads = row(60.0, 57.0, 63.0);
        many_threads.threads = "8".into();
        // Same timing cell at different worker counts must fail closed,
        // not compare incomparable timings.
        let error = compare_evidence(
            &snapshot(one_thread),
            &snapshot(many_threads),
            RegressionPolicy::default(),
        )
        .unwrap_err();
        assert!(
            error.contains("benchmark matrix mismatch"),
            "unexpected error: {error}"
        );
        // Identical thread counts compare normally.
        let report = compare_evidence(
            &snapshot(row(100.0, 95.0, 105.0)),
            &snapshot(row(100.0, 95.0, 105.0)),
            RegressionPolicy::default(),
        )
        .unwrap();
        assert!(report.passed());
    }

    #[test]
    fn pre_threads_baseline_rows_default_to_unknown() {
        let envelope = serde_json::json!({
            "schema": "sonium-compute-macos-evidence/v3",
            "hardware": {"arch": "aarch64", "model": "Mac15,6", "rayon_threads": "1"},
            "rows": [{
                "implementation": "sonium-cpu", "blas_level": "blas-2",
                "operation": "gemv", "regime": "tail", "dtype": "f32",
                "shape": "31x29", "correctness_passed": true,
                "median_ns": 100.0, "ci95_low_ns": 95.0, "ci95_high_ns": 105.0,
            }],
        });
        let snapshot: EvidenceSnapshot = serde_json::from_value(envelope).unwrap();
        assert_eq!(snapshot.rows[0].threads, "unknown");
    }

    #[test]
    fn regression_report_round_trips_for_trend_analysis() {
        let report = compare_evidence(
            &snapshot(row(100.0, 95.0, 105.0)),
            &snapshot(row(120.0, 115.0, 125.0)),
            RegressionPolicy::default(),
        )
        .unwrap();
        let bytes = serde_json::to_vec(&report).unwrap();
        let reread: RegressionReport = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(reread.schema, "sonium-compute-regression-report/v2");
        assert_eq!(reread.regression_count, 1);
        assert!(!reread.passed());
        assert_eq!(reread.comparisons[0].status, RegressionStatus::Regressed);
    }

    #[test]
    fn hardware_schema_and_policy_mismatches_fail_closed() {
        let baseline = snapshot(row(100.0, 95.0, 105.0));

        let mut hardware = baseline.clone();
        hardware.hardware.rayon_threads = "4".into();
        assert!(
            compare_evidence(&baseline, &hardware, RegressionPolicy::default())
                .unwrap_err()
                .contains("hardware/thread mismatch")
        );

        let mut schema = baseline.clone();
        schema.schema = "future".into();
        assert!(
            compare_evidence(&baseline, &schema, RegressionPolicy::default())
                .unwrap_err()
                .contains("schema mismatch")
        );

        let error = compare_evidence(
            &baseline,
            &baseline,
            RegressionPolicy {
                max_median_slowdown: f64::NAN,
                require_non_overlapping_interval: true,
            },
        )
        .unwrap_err();
        assert!(error.contains("finite and non-negative"));
    }

    #[test]
    fn full_environment_and_protocol_are_compared_and_required() {
        let baseline = snapshot(row(100.0, 95.0, 105.0));
        for key in [
            "os",
            "rustc",
            "profile",
            "target_features",
            "device",
            "driver",
            "clocks",
            "thermal_state",
        ] {
            let mut other = baseline.clone();
            other
                .hardware
                .environment
                .insert(key.into(), serde_json::json!("different"));
            assert!(
                compare_evidence(&baseline, &other, RegressionPolicy::default()).is_err(),
                "{key}"
            );
            other.hardware.environment.remove(key);
            assert!(
                compare_evidence(&other, &other, RegressionPolicy::default()).is_err(),
                "{key}"
            );
        }
        let mut other = baseline.clone();
        other.protocol["measured_samples"] = serde_json::json!(30);
        assert!(
            compare_evidence(&baseline, &other, RegressionPolicy::default())
                .unwrap_err()
                .contains("protocol mismatch")
        );
        other.protocol = serde_json::Value::Null;
        assert!(compare_evidence(&other, &other, RegressionPolicy::default()).is_err());
        let mut value = serde_json::to_value(&baseline).unwrap();
        value["hardware"]["gpu_uuid"] = serde_json::json!("device-1");
        let parsed: EvidenceSnapshot = serde_json::from_value(value).unwrap();
        assert!(compare_evidence(&baseline, &parsed, RegressionPolicy::default()).is_err());
    }

    #[test]
    fn zero_or_overflowing_timing_ratios_cannot_enter_reports() {
        let zero = snapshot(row(0.0, 0.0, 0.0));
        assert!(compare_evidence(&zero, &zero, RegressionPolicy::default()).is_err());
        let tiny = snapshot(row(f64::MIN_POSITIVE, 0.0, f64::MIN_POSITIVE));
        let huge = snapshot(row(f64::MAX, f64::MAX, f64::MAX));
        assert!(compare_evidence(&tiny, &huge, RegressionPolicy::default()).is_err());
        let mut report = compare_evidence(
            &snapshot(row(1.0, 1.0, 1.0)),
            &snapshot(row(1.0, 1.0, 1.0)),
            RegressionPolicy::default(),
        )
        .unwrap();
        report.comparisons[0].status = RegressionStatus::Regressed;
        assert!(
            !report.passed(),
            "stored regression_count cannot override row status"
        );
        report.comparisons.clear();
        assert!(!report.passed(), "an empty report is not a pass");
    }
}
