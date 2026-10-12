//! Frozen OpenCourant-vs-sonium comparison bundle and metrics.
//!
//! This module defines the `sonium.oc-bench/v1` evidence bundle: one row
//! per timed (deck, side, flavor, threads) cell plus the metric math and
//! the markdown report rendering. The JSON field names are frozen by
//! `PHASE6_CONTRACTS.md`; lane 6C consumes this schema for the perf floor.
//!
//! Comparison semantics follow the fail-closed spirit of
//! [`compare_evidence`](crate::compute_regression::compare_evidence)
//! (correctness gates timing; a failed row never wins) with one
//! deliberate deviation: oc-bench rows are single-point timings with no
//! confidence intervals, so the binomial-interval machinery of
//! `compute_regression` does not apply. Cross-side winners are decided
//! on `ns_per_dof_cycle` ratios instead, which normalize the differing
//! cycle counts each side's own dt policy produces over the same stop
//! time. See [`compare_pair`] for the exact rule.

// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};
use std::backtrace::Backtrace;
use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};

/// Frozen bundle schema tag; every emitted bundle carries it verbatim.
pub const BUNDLE_SCHEMA: &str = "sonium.oc-bench/v1";

/// Correctness protocol tag: every probe passed its golden tolerance.
pub const PROTOCOL_CORRECTNESS: &str = "probe-tolerances";

/// Relative agreement required between a row's stored metric and the
/// recomputed value. JSON round-trips f64 exactly, so this only guards
/// hand-written or corrupted bundles.
const METRIC_AGREEMENT: f64 = 1e-9;

/// Timed side of a comparison row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    /// Native sonium explicit run.
    Sonium,
    /// Vendor OpenCourant engine run.
    Engine,
}

/// Machine identity for one bundle. All rows share the hardware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    /// CPU architecture, e.g. `x86_64`.
    pub arch: String,
    /// CPU model string from the host inventory.
    pub model: String,
    /// Worker count the bundle ran under, as a string.
    pub threads: String,
}

/// Timing protocol shared by every row in the bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolInfo {
    /// Timed repeats aggregated into each row (median wall time).
    pub repeats: u32,
    /// Untimed warmup runs discarded before the timed repeats.
    pub warmup: u32,
    /// Correctness rule, e.g. [`PROTOCOL_CORRECTNESS`].
    pub correctness: String,
}

/// One timed cell: a deck run on one side with one thread count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchRow {
    /// Deck id: a C3 case (`taylor_bar`) or miniqa path (`LOIS/...`).
    pub deck: String,
    /// Which implementation ran.
    pub side: Side,
    /// Build flavor: `native`/`cuda`, or `serial`/`smp`/`mpi` with an
    /// optional `_sp` suffix for single-precision engine builds.
    pub flavor: String,
    /// Worker threads (SMP threads or MPI ranks) for this row.
    pub threads: u32,
    /// Wall seconds covering exactly `cycles` explicit cycles.
    pub wall_s: f64,
    /// Completed explicit cycles timed (excludes starter/setup).
    pub cycles: u64,
    /// Structural degrees of freedom in the timed model.
    pub dof: u64,
    /// Cost per dof per cycle: `wall_s * 1e9 / (cycles * dof)`.
    pub ns_per_dof_cycle: f64,
    /// Peak resident bytes observed over the timed repeats.
    pub peak_rss_bytes: u64,
    /// True when every probe passed its golden tolerance.
    pub correctness_passed: bool,
}

/// One evidence bundle: schema tag plus shared context and rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchBundle {
    /// Must equal [`BUNDLE_SCHEMA`].
    pub schema: String,
    /// Machine identity shared by every row.
    pub hardware: HardwareInfo,
    /// Timing protocol shared by every row.
    pub protocol: ProtocolInfo,
    /// Timed cells; at least one.
    pub rows: Vec<BenchRow>,
}

/// Situation-specific error for bundle construction and comparison.
#[derive(Debug)]
pub struct CompareError {
    /// Human-readable failure.
    message: String,
    /// Capture site for forensics.
    backtrace: Backtrace,
}

impl CompareError {
    /// Captures a new error with its backtrace.
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            backtrace: Backtrace::capture(),
        }
    }

    /// Returns the human-readable failure.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the capture-site backtrace.
    pub fn backtrace(&self) -> &Backtrace {
        &self.backtrace
    }
}

impl Display for CompareError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for CompareError {}

/// Computes nanoseconds per dof per cycle.
///
/// Returns the frozen metric `wall_s * 1e9 / (cycles * dof)`, or `None`
/// when any input is non-positive or `wall_s` is non-finite, so callers
/// fail closed instead of emitting an infinite or negative cost.
///
/// # Examples
///
/// ```rust
/// use sonium_bench::oc_compare::ns_per_dof_cycle;
///
/// let cost = ns_per_dof_cycle(12.3, 900, 15_000).unwrap();
/// assert!((cost - 911.111).abs() < 0.01);
/// assert!(ns_per_dof_cycle(1.0, 0, 10).is_none());
/// ```
pub fn ns_per_dof_cycle(wall_s: f64, cycles: u64, dof: u64) -> Option<f64> {
    if !wall_s.is_finite() || wall_s <= 0.0 || cycles == 0 || dof == 0 {
        return None;
    }
    let cost = wall_s * 1e9 / (cycles as f64 * dof as f64);
    cost.is_finite().then_some(cost)
}

impl BenchRow {
    /// Builds a row, deriving the cost metric from the measurement.
    ///
    /// The stored `ns_per_dof_cycle` always equals [`ns_per_dof_cycle`]
    /// applied to the inputs, so emitted rows pass [`validate_bundle`].
    ///
    /// # Errors
    ///
    /// Returns [`CompareError`] for empty deck/flavor ids, zero threads,
    /// or a non-positive/non-finite measurement.
    pub fn measured(
        deck: &str,
        side: Side,
        flavor: &str,
        threads: u32,
        wall_s: f64,
        cycles: u64,
        dof: u64,
        peak_rss_bytes: u64,
        correctness_passed: bool,
    ) -> Result<Self, CompareError> {
        if deck.trim().is_empty() {
            return Err(CompareError::new("deck id must not be empty"));
        }
        if flavor.trim().is_empty() {
            return Err(CompareError::new("flavor must not be empty"));
        }
        if threads == 0 {
            return Err(CompareError::new("row thread count must be positive"));
        }
        let Some(cost) = ns_per_dof_cycle(wall_s, cycles, dof) else {
            return Err(CompareError::new(format!(
                "invalid measurement wall_s={wall_s} cycles={cycles} dof={dof}"
            )));
        };
        if peak_rss_bytes == 0 {
            return Err(CompareError::new("peak_rss_bytes must be positive"));
        }
        Ok(Self {
            deck: deck.to_string(),
            side,
            flavor: flavor.to_string(),
            threads,
            wall_s,
            cycles,
            dof,
            ns_per_dof_cycle: cost,
            peak_rss_bytes,
            correctness_passed,
        })
    }
}

/// Stable identity for one timed cell.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RowKey {
    deck: String,
    side: Side,
    flavor: String,
    threads: u32,
}

impl From<&BenchRow> for RowKey {
    fn from(row: &BenchRow) -> Self {
        Self {
            deck: row.deck.clone(),
            side: row.side,
            flavor: row.flavor.clone(),
            threads: row.threads,
        }
    }
}

/// Validates a bundle against the frozen contract.
///
/// Checks the schema tag, shared context, per-row measurement sanity,
/// metric agreement, and cell uniqueness. Accepts nothing silently.
///
/// # Errors
///
/// Returns [`CompareError`] naming the first contract violation.
pub fn validate_bundle(bundle: &BenchBundle) -> Result<(), CompareError> {
    if bundle.schema != BUNDLE_SCHEMA {
        return Err(CompareError::new(format!(
            "schema drift: expected {BUNDLE_SCHEMA}, got {}",
            bundle.schema
        )));
    }
    if bundle.hardware.arch.trim().is_empty() || bundle.hardware.model.trim().is_empty() {
        return Err(CompareError::new("hardware arch/model must not be empty"));
    }
    if bundle
        .hardware
        .threads
        .parse::<u32>()
        .ok()
        .filter(|threads| *threads > 0)
        .is_none()
    {
        return Err(CompareError::new(
            "hardware threads must be a positive integer",
        ));
    }
    if bundle.protocol.repeats == 0 {
        return Err(CompareError::new("protocol repeats must be positive"));
    }
    if bundle.protocol.correctness.trim().is_empty() {
        return Err(CompareError::new(
            "protocol correctness tag must not be empty",
        ));
    }
    if bundle.rows.is_empty() {
        return Err(CompareError::new("bundle contains no rows"));
    }
    let mut seen = BTreeSet::new();
    for row in &bundle.rows {
        validate_row(row)?;
        let key = RowKey::from(row);
        if !seen.insert(key.clone()) {
            return Err(CompareError::new(format!(
                "duplicate cell deck={} side={:?} flavor={} threads={}",
                key.deck, key.side, key.flavor, key.threads
            )));
        }
    }
    Ok(())
}

/// Validates one row: identity, measurement sanity, metric agreement.
///
/// # Errors
///
/// Returns [`CompareError`] naming the first violation found.
fn validate_row(row: &BenchRow) -> Result<(), CompareError> {
    if row.deck.trim().is_empty() {
        return Err(CompareError::new("row deck id must not be empty"));
    }
    if row.flavor.trim().is_empty() {
        return Err(CompareError::new("row flavor must not be empty"));
    }
    if row.threads == 0 {
        return Err(CompareError::new("row thread count must be positive"));
    }
    if !row.wall_s.is_finite() || row.wall_s <= 0.0 {
        return Err(CompareError::new(format!(
            "row {} has invalid wall_s={}",
            row.deck, row.wall_s
        )));
    }
    if row.cycles == 0 {
        return Err(CompareError::new(format!(
            "row {} has zero cycles",
            row.deck
        )));
    }
    if row.dof == 0 {
        return Err(CompareError::new(format!("row {} has zero dof", row.deck)));
    }
    if row.peak_rss_bytes == 0 {
        return Err(CompareError::new(format!(
            "row {} has zero peak_rss_bytes",
            row.deck
        )));
    }
    let expected = ns_per_dof_cycle(row.wall_s, row.cycles, row.dof)
        .ok_or_else(|| CompareError::new(format!("row {} metric overflows", row.deck)))?;
    // Absolute floor of 1.0 keeps tiny costs comparable; relative term
    // dominates at realistic magnitudes (hundreds of ns).
    let bound = METRIC_AGREEMENT * expected.abs().max(1.0);
    if (row.ns_per_dof_cycle - expected).abs() > bound {
        return Err(CompareError::new(format!(
            "row {} metric {} disagrees with recomputed {expected}",
            row.deck, row.ns_per_dof_cycle
        )));
    }
    Ok(())
}

/// Parses bundle JSON, rejecting schema drift and malformed rows.
///
/// # Errors
///
/// Returns [`CompareError`] for invalid JSON or contract violations.
pub fn parse_bundle_json(text: &str) -> Result<BenchBundle, CompareError> {
    let bundle: BenchBundle = serde_json::from_str(text)
        .map_err(|error| CompareError::new(format!("invalid bundle JSON: {error}")))?;
    validate_bundle(&bundle)?;
    Ok(bundle)
}

/// Renders a validated bundle as pretty JSON.
///
/// # Errors
///
/// Returns [`CompareError`] when the bundle violates the contract.
pub fn bundle_json(bundle: &BenchBundle) -> Result<String, CompareError> {
    validate_bundle(bundle)?;
    serde_json::to_string_pretty(bundle)
        .map_err(|error| CompareError::new(format!("bundle JSON render failed: {error}")))
}

/// Winner of one cross-side comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Winner {
    /// Sonium ran fewer ns per dof per cycle with probes passing.
    Sonium,
    /// Engine ran fewer ns per dof per cycle with probes passing.
    Engine,
    /// No winner: at least one side failed its correctness probes.
    Neither,
}

/// Outcome of comparing one sonium row with one engine row.
#[derive(Debug, Clone)]
pub struct SideComparison {
    /// Deck both rows ran.
    pub deck: String,
    /// Sonium cost; lower wins when both sides pass probes.
    pub sonium_ns_per_dof_cycle: f64,
    /// Engine cost; lower wins when both sides pass probes.
    pub engine_ns_per_dof_cycle: f64,
    /// `engine / sonium`; above 1.0 means sonium is faster per cycle.
    pub speedup_sonium_vs_engine: f64,
    /// Sonium probes all passed.
    pub sonium_correct: bool,
    /// Engine probes all passed.
    pub engine_correct: bool,
    /// Winner; a failed row never wins.
    pub winner: Winner,
}

/// Compares one sonium row against one engine row.
///
/// Both rows must cover the same deck; thread counts and flavors may
/// differ (callers pair comparable cells). A row whose probes failed
/// never wins: if either side failed, the winner is [`Winner::Neither`]
/// when both failed, else the passing side.
///
/// # Errors
///
/// Returns [`CompareError`] when the sides or decks mismatch, or when
/// either row is invalid.
///
/// # Examples
///
/// ```rust
/// use sonium_bench::oc_compare::{BenchRow, Side, Winner, compare_pair};
///
/// let sonium = BenchRow::measured("taylor_bar", Side::Sonium, "native", 1, 2.0, 100, 600, 8, true).unwrap();
/// let engine = BenchRow::measured("taylor_bar", Side::Engine, "serial", 1, 4.0, 100, 600, 8, true).unwrap();
/// let compared = compare_pair(&sonium, &engine).unwrap();
/// assert_eq!(compared.winner, Winner::Sonium);
/// ```
pub fn compare_pair(sonium: &BenchRow, engine: &BenchRow) -> Result<SideComparison, CompareError> {
    if sonium.side != Side::Sonium || engine.side != Side::Engine {
        return Err(CompareError::new(
            "compare_pair needs (sonium, engine) sides",
        ));
    }
    if sonium.deck != engine.deck {
        return Err(CompareError::new(format!(
            "deck mismatch: {} vs {}",
            sonium.deck, engine.deck
        )));
    }
    validate_row(sonium)?;
    validate_row(engine)?;
    let speedup = engine.ns_per_dof_cycle / sonium.ns_per_dof_cycle;
    if !speedup.is_finite() {
        return Err(CompareError::new("timing ratio overflows"));
    }
    let winner = match (sonium.correctness_passed, engine.correctness_passed) {
        (false, false) => Winner::Neither,
        (true, false) => Winner::Sonium,
        (false, true) => Winner::Engine,
        (true, true) => {
            if speedup > 1.0 {
                Winner::Sonium
            } else if speedup < 1.0 {
                Winner::Engine
            } else {
                Winner::Neither
            }
        }
    };
    Ok(SideComparison {
        deck: sonium.deck.clone(),
        sonium_ns_per_dof_cycle: sonium.ns_per_dof_cycle,
        engine_ns_per_dof_cycle: engine.ns_per_dof_cycle,
        speedup_sonium_vs_engine: speedup,
        sonium_correct: sonium.correctness_passed,
        engine_correct: engine.correctness_passed,
        winner,
    })
}

/// Pairs comparable rows: per (deck, threads), every sonium row meets
/// every engine row. Deterministic in bundle row order.
pub fn auto_pairs(bundle: &BenchBundle) -> Vec<(BenchRow, BenchRow)> {
    let mut pairs = Vec::new();
    let mut groups: Vec<(String, u32)> = Vec::new();
    for row in &bundle.rows {
        let key = (row.deck.clone(), row.threads);
        if !groups.contains(&key) {
            groups.push(key);
        }
    }
    for (deck, threads) in groups {
        let sonium_rows: Vec<&BenchRow> = bundle
            .rows
            .iter()
            .filter(|row| row.deck == deck && row.threads == threads && row.side == Side::Sonium)
            .collect();
        let engine_rows: Vec<&BenchRow> = bundle
            .rows
            .iter()
            .filter(|row| row.deck == deck && row.threads == threads && row.side == Side::Engine)
            .collect();
        for sonium in &sonium_rows {
            for engine in &engine_rows {
                pairs.push(((*sonium).clone(), (*engine).clone()));
            }
        }
    }
    pairs
}

/// Renders a bundle as a markdown report.
///
/// Validates first, so only contract-clean bundles render. The report
/// holds a run-context header, the rows table, and the auto-paired
/// cross-side comparisons.
///
/// # Errors
///
/// Returns [`CompareError`] when the bundle violates the contract.
pub fn render_bundle_markdown(bundle: &BenchBundle) -> Result<String, CompareError> {
    validate_bundle(bundle)?;
    let mut report = String::new();
    report.push_str("# oc-bench comparison report\n\n");
    report.push_str(&format!(
        "Schema `{}`; arch `{}`; model `{}`; threads `{}`; repeats {}; warmup {}; correctness `{}`.\n\n",
        bundle.schema,
        bundle.hardware.arch,
        bundle.hardware.model,
        bundle.hardware.threads,
        bundle.protocol.repeats,
        bundle.protocol.warmup,
        bundle.protocol.correctness,
    ));
    report.push_str("| deck | side | flavor | threads | wall_s | cycles | dof | ns/dof/cycle | peak_rss_mb | correct |\n");
    report.push_str("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for row in &bundle.rows {
        report.push_str(&format!(
            "| {} | {:?} | {} | {} | {:.4} | {} | {} | {:.2} | {:.2} | {} |\n",
            row.deck,
            row.side,
            row.flavor,
            row.threads,
            row.wall_s,
            row.cycles,
            row.dof,
            row.ns_per_dof_cycle,
            row.peak_rss_bytes as f64 / 1_048_576.0,
            row.correctness_passed,
        ));
    }
    let pairs = auto_pairs(bundle);
    report.push_str("\n## Cross-side comparisons\n\n");
    if pairs.is_empty() {
        report.push_str("No (deck, threads) group holds both sides; nothing to compare.\n");
        return Ok(report);
    }
    report.push_str("| deck | threads | sonium flavor | engine flavor | speedup (sonium vs engine) | winner |\n");
    report.push_str("| --- | --- | --- | --- | --- | --- |\n");
    for (sonium, engine) in &pairs {
        let compared = compare_pair(sonium, engine)?;
        report.push_str(&format!(
            "| {} | {} | {} | {} | {:.3}x | {:?} |\n",
            compared.deck,
            sonium.threads,
            sonium.flavor,
            engine.flavor,
            compared.speedup_sonium_vs_engine,
            compared.winner,
        ));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row(side: Side, wall_s: f64, correct: bool) -> BenchRow {
        BenchRow::measured(
            "taylor_bar",
            side,
            "native",
            1,
            wall_s,
            100,
            600,
            8_000_000,
            correct,
        )
        .unwrap()
    }

    fn sample_bundle() -> BenchBundle {
        BenchBundle {
            schema: BUNDLE_SCHEMA.to_string(),
            hardware: HardwareInfo {
                arch: "x86_64".to_string(),
                model: "test-cpu".to_string(),
                threads: "1".to_string(),
            },
            protocol: ProtocolInfo {
                repeats: 3,
                warmup: 1,
                correctness: PROTOCOL_CORRECTNESS.to_string(),
            },
            rows: vec![
                sample_row(Side::Sonium, 2.0, true),
                BenchRow::measured(
                    "taylor_bar",
                    Side::Engine,
                    "serial",
                    1,
                    4.0,
                    100,
                    600,
                    9_000_000,
                    true,
                )
                .unwrap(),
            ],
        }
    }

    #[test]
    fn metric_matches_contract_example() {
        let cost = ns_per_dof_cycle(12.3, 900, 15_000).unwrap();
        assert!((cost - 911.111_111).abs() < 1e-3);
    }

    #[test]
    fn metric_rejects_degenerate_inputs() {
        assert!(ns_per_dof_cycle(0.0, 100, 600).is_none());
        assert!(ns_per_dof_cycle(-1.0, 100, 600).is_none());
        assert!(ns_per_dof_cycle(f64::NAN, 100, 600).is_none());
        assert!(ns_per_dof_cycle(f64::INFINITY, 100, 600).is_none());
        assert!(ns_per_dof_cycle(1.0, 0, 600).is_none());
        assert!(ns_per_dof_cycle(1.0, 100, 0).is_none());
    }

    #[test]
    fn valid_bundle_passes() {
        validate_bundle(&sample_bundle()).unwrap();
    }

    #[test]
    fn schema_drift_fails() {
        let mut bundle = sample_bundle();
        bundle.schema = "sonium.oc-bench/v0".to_string();
        assert!(validate_bundle(&bundle).is_err());
    }

    #[test]
    fn duplicate_cells_fail() {
        let mut bundle = sample_bundle();
        bundle.rows.push(bundle.rows[0].clone());
        let error = validate_bundle(&bundle).unwrap_err();
        assert!(error.message().contains("duplicate cell"));
    }

    #[test]
    fn metric_disagreement_fails() {
        let mut bundle = sample_bundle();
        bundle.rows[0].ns_per_dof_cycle *= 1.5;
        assert!(validate_bundle(&bundle).is_err());
    }

    #[test]
    fn faster_side_wins_when_both_pass() {
        let compared = compare_pair(
            &sample_row(Side::Sonium, 2.0, true),
            &BenchRow::measured(
                "taylor_bar",
                Side::Engine,
                "serial",
                1,
                4.0,
                100,
                600,
                8,
                true,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(compared.winner, Winner::Sonium);
        assert!((compared.speedup_sonium_vs_engine - 2.0).abs() < 1e-12);
    }

    #[test]
    fn failed_row_never_wins() {
        let fast_wrong = sample_row(Side::Sonium, 0.5, false);
        let slow_right = BenchRow::measured(
            "taylor_bar",
            Side::Engine,
            "serial",
            1,
            40.0,
            100,
            600,
            8,
            true,
        )
        .unwrap();
        let compared = compare_pair(&fast_wrong, &slow_right).unwrap();
        assert_eq!(compared.winner, Winner::Engine);
        let both_wrong = compare_pair(
            &sample_row(Side::Sonium, 0.5, false),
            &BenchRow::measured(
                "taylor_bar",
                Side::Engine,
                "serial",
                1,
                0.5,
                100,
                600,
                8,
                false,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(both_wrong.winner, Winner::Neither);
    }

    #[test]
    fn markdown_renders_rows_and_comparisons() {
        let report = render_bundle_markdown(&sample_bundle()).unwrap();
        assert!(report.contains("# oc-bench comparison report"));
        assert!(report.contains("taylor_bar"));
        assert!(report.contains("Sonium"));
    }
}
