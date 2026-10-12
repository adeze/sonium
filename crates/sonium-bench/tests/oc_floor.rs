//! Integration tests for the oc-bench perf floor.
//!
//! Offline tests use hand-built bundles that follow the frozen
//! `sonium.oc-bench/v1` field names. The ignored live test shells
//! out to `qa-oc-bench` (lane 6A) and gates the result against the
//! checked-in baseline.

// Rust guideline compliant 2026-02-21

use std::path::PathBuf;
use std::process::Command;

use sonium_bench::oc_floor::{
    FloorBundle, FloorHardware, FloorPolicy, FloorProtocol, FloorRow, FloorStatus, OC_BENCH_SCHEMA,
    OC_FLOOR_REPORT_SCHEMA, check_floor, compare_floor, load_bundle,
};

fn hardware() -> FloorHardware {
    FloorHardware {
        arch: "x86_64".into(),
        model: "ci-runner".into(),
        threads: "1".into(),
    }
}

fn protocol() -> FloorProtocol {
    FloorProtocol {
        repeats: 1,
        warmup: 0,
        correctness: "probe-tolerances".into(),
    }
}

fn row(deck: &str, ns_per_dof_cycle: f64) -> FloorRow {
    let cycles = 900;
    let dof = 15_000;
    FloorRow {
        deck: deck.into(),
        side: "sonium".into(),
        flavor: "native".into(),
        threads: 1,
        wall_s: ns_per_dof_cycle * cycles as f64 * dof as f64 / 1e9,
        cycles,
        dof,
        ns_per_dof_cycle,
        peak_rss_bytes: 123_456_789,
        correctness_passed: true,
    }
}

fn bundle(rows: Vec<FloorRow>) -> FloorBundle {
    FloorBundle {
        schema: OC_BENCH_SCHEMA.into(),
        hardware: hardware(),
        protocol: protocol(),
        rows,
    }
}

fn baseline_bundle() -> FloorBundle {
    bundle(vec![row("rail_crush", 911.0), row("taylor_bar", 450.0)])
}

fn baseline_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("baselines/oc_floor.json")
}

#[test]
fn checked_in_baseline_loads_and_passes_against_itself() {
    let baseline = load_bundle(&baseline_path()).expect("checked-in baseline must load");
    assert_eq!(baseline.schema, OC_BENCH_SCHEMA);
    assert!(
        baseline.rows.iter().all(|row| row.side == "sonium"),
        "floor baseline must be sonium-arm-only"
    );
    let report = compare_floor(&baseline, &baseline, FloorPolicy::default()).expect("self-compare");
    assert!(report.passed());
    assert_eq!(report.schema, OC_FLOOR_REPORT_SCHEMA);
    assert!(
        report
            .comparisons
            .iter()
            .all(|cell| cell.status == FloorStatus::Stable)
    );
}

#[test]
fn twofold_noise_passes() {
    let baseline = baseline_bundle();
    let candidate = bundle(vec![
        row("rail_crush", 911.0 * 2.0),
        row("taylor_bar", 450.0 * 2.0),
    ]);
    let report = compare_floor(&baseline, &candidate, FloorPolicy::default()).unwrap();
    assert!(report.passed());
}

#[test]
fn elevenfold_regression_fails() {
    let baseline = baseline_bundle();
    let candidate = bundle(vec![
        row("rail_crush", 911.0 * 11.0),
        row("taylor_bar", 450.0),
    ]);
    let report = compare_floor(&baseline, &candidate, FloorPolicy::default()).unwrap();
    assert!(!report.passed());
    assert_eq!(report.regression_count, 1);
    assert_eq!(report.comparisons[0].status, FloorStatus::Regressed);
    assert_eq!(report.comparisons[1].status, FloorStatus::Stable);
}

#[test]
fn large_speedup_passes_as_stable() {
    let baseline = baseline_bundle();
    let candidate = bundle(vec![
        row("rail_crush", 911.0 / 20.0),
        row("taylor_bar", 450.0),
    ]);
    let report = compare_floor(&baseline, &candidate, FloorPolicy::default()).unwrap();
    assert!(report.passed());
    assert_eq!(report.comparisons[0].status, FloorStatus::Stable);
}

#[test]
fn candidate_correctness_failure_errors() {
    let baseline = baseline_bundle();
    let mut bad = row("rail_crush", 911.0);
    bad.correctness_passed = false;
    let error = compare_floor(
        &baseline,
        &bundle(vec![bad, row("taylor_bar", 450.0)]),
        FloorPolicy::default(),
    )
    .unwrap_err();
    assert!(error.contains("correctness preflight"));
}

#[test]
fn baseline_correctness_failure_errors() {
    let mut bad = row("rail_crush", 911.0);
    bad.correctness_passed = false;
    let error = compare_floor(
        &bundle(vec![bad, row("taylor_bar", 450.0)]),
        &baseline_bundle(),
        FloorPolicy::default(),
    )
    .unwrap_err();
    assert!(error.contains("correctness preflight"));
}

#[test]
fn schema_drift_errors() {
    let baseline = baseline_bundle();
    let mut candidate = baseline_bundle();
    candidate.schema = "sonium.oc-bench/v2".into();
    let error = compare_floor(&baseline, &candidate, FloorPolicy::default()).unwrap_err();
    assert!(error.contains("schema"));
}

#[test]
fn missing_deck_errors() {
    let baseline = baseline_bundle();
    let candidate = bundle(vec![row("rail_crush", 911.0)]);
    let error = compare_floor(&baseline, &candidate, FloorPolicy::default()).unwrap_err();
    assert!(error.contains("missing"));
}

#[test]
fn extra_deck_errors() {
    let baseline = baseline_bundle();
    let candidate = bundle(vec![
        row("rail_crush", 911.0),
        row("taylor_bar", 450.0),
        row("shell_panel", 300.0),
    ]);
    let error = compare_floor(&baseline, &candidate, FloorPolicy::default()).unwrap_err();
    assert!(error.contains("extra"));
}

#[test]
fn protocol_drift_errors() {
    let baseline = baseline_bundle();
    let mut candidate = baseline_bundle();
    candidate.protocol.repeats = 3;
    let error = compare_floor(&baseline, &candidate, FloorPolicy::default()).unwrap_err();
    assert!(error.contains("protocol"));
}

#[test]
fn arch_mismatch_errors() {
    let baseline = baseline_bundle();
    let mut candidate = baseline_bundle();
    candidate.hardware.arch = "aarch64".into();
    let error = compare_floor(&baseline, &candidate, FloorPolicy::default()).unwrap_err();
    assert!(error.contains("Architecture mismatch") || error.contains("architecture mismatch"));
}

#[test]
fn model_difference_is_recorded_not_gated() {
    let baseline = baseline_bundle();
    let mut candidate = baseline_bundle();
    candidate.hardware.model = "other-stepping".into();
    let report = compare_floor(&baseline, &candidate, FloorPolicy::default()).unwrap();
    assert!(report.passed());
    assert_eq!(report.candidate_hardware.model, "other-stepping");
}

#[test]
fn dof_drift_errors() {
    let baseline = baseline_bundle();
    let mut drifted = row("rail_crush", 911.0);
    drifted.dof *= 2;
    drifted.wall_s *= 2.0;
    let error = compare_floor(
        &baseline,
        &bundle(vec![drifted, row("taylor_bar", 450.0)]),
        FloorPolicy::default(),
    )
    .unwrap_err();
    assert!(error.contains("drifted in dof"));
}

#[test]
fn cycles_drift_errors() {
    let baseline = baseline_bundle();
    let mut drifted = row("rail_crush", 911.0);
    drifted.cycles *= 2;
    drifted.wall_s *= 2.0;
    let error = compare_floor(
        &baseline,
        &bundle(vec![drifted, row("taylor_bar", 450.0)]),
        FloorPolicy::default(),
    )
    .unwrap_err();
    assert!(error.contains("drifted in cycles"));
}

#[test]
fn threads_drift_errors() {
    let baseline = baseline_bundle();
    let mut drifted = row("rail_crush", 911.0);
    drifted.threads = 16;
    let error = compare_floor(
        &baseline,
        &bundle(vec![drifted, row("taylor_bar", 450.0)]),
        FloorPolicy::default(),
    )
    .unwrap_err();
    assert!(error.contains("drifted in threads"));
}

#[test]
fn duplicate_rows_error() {
    let baseline = baseline_bundle();
    let candidate = bundle(vec![row("rail_crush", 911.0), row("rail_crush", 911.0)]);
    let error = compare_floor(&baseline, &candidate, FloorPolicy::default()).unwrap_err();
    assert!(error.contains("duplicate"));
}

#[test]
fn empty_matrix_errors() {
    let baseline = baseline_bundle();
    let error = compare_floor(&baseline, &bundle(vec![]), FloorPolicy::default()).unwrap_err();
    assert!(error.contains("no benchmark rows"));
}

#[test]
fn invalid_wall_time_errors() {
    for wall_s in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let baseline = baseline_bundle();
        let mut bad = row("rail_crush", 911.0);
        bad.wall_s = wall_s;
        let error = compare_floor(
            &baseline,
            &bundle(vec![bad, row("taylor_bar", 450.0)]),
            FloorPolicy::default(),
        )
        .unwrap_err();
        assert!(error.contains("wall_s"), "wall_s={wall_s}: {error}");
    }
}

#[test]
fn zero_peak_rss_errors() {
    let baseline = baseline_bundle();
    let mut bad = row("rail_crush", 911.0);
    bad.peak_rss_bytes = 0;
    let error = compare_floor(
        &baseline,
        &bundle(vec![bad, row("taylor_bar", 450.0)]),
        FloorPolicy::default(),
    )
    .unwrap_err();
    assert!(error.contains("peak RSS"));
}

#[test]
fn self_inconsistent_timing_errors() {
    let baseline = baseline_bundle();
    let mut bad = row("rail_crush", 911.0);
    bad.ns_per_dof_cycle *= 1.5;
    let error = compare_floor(
        &baseline,
        &bundle(vec![bad, row("taylor_bar", 450.0)]),
        FloorPolicy::default(),
    )
    .unwrap_err();
    assert!(error.contains("self-inconsistent"));
}

#[test]
fn invalid_policy_errors() {
    let baseline = baseline_bundle();
    for max in [-1.0, f64::NAN, f64::INFINITY] {
        let error = compare_floor(
            &baseline,
            &baseline,
            FloorPolicy {
                max_slowdown_fraction: max,
            },
        )
        .unwrap_err();
        assert!(
            error.contains("max_slowdown_fraction"),
            "max={max}: {error}"
        );
    }
}

#[test]
fn missing_baseline_file_errors() {
    let missing = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("baselines/does-not-exist.json");
    let error = check_floor(&missing, &baseline_path(), FloorPolicy::default()).unwrap_err();
    assert!(error.contains("cannot read bundle"));
}

#[test]
fn malformed_json_errors() {
    let dir = std::env::temp_dir().join(format!("sonium-oc-floor-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("malformed.json");
    std::fs::write(&path, "{ not json").unwrap();
    let error = load_bundle(&path).unwrap_err();
    assert!(error.contains("cannot parse bundle"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn wrong_schema_file_errors() {
    let dir = std::env::temp_dir().join(format!("sonium-oc-floor-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("wrong-schema.json");
    let mut bundle = baseline_bundle();
    bundle.schema = "something-else/v1".into();
    std::fs::write(&path, serde_json::to_string(&bundle).unwrap()).unwrap();
    let error = load_bundle(&path).unwrap_err();
    assert!(error.contains("schema"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn check_floor_file_round_trip() {
    let dir = std::env::temp_dir().join(format!("sonium-oc-floor-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let baseline_file = dir.join("baseline.json");
    let candidate_file = dir.join("candidate.json");
    std::fs::write(
        &baseline_file,
        serde_json::to_string(&baseline_bundle()).unwrap(),
    )
    .unwrap();
    let candidate = bundle(vec![
        row("rail_crush", 911.0 * 3.0),
        row("taylor_bar", 450.0),
    ]);
    std::fs::write(&candidate_file, serde_json::to_string(&candidate).unwrap()).unwrap();
    let report = check_floor(&baseline_file, &candidate_file, FloorPolicy::default()).unwrap();
    assert!(report.passed());
    assert_eq!(report.schema, OC_FLOOR_REPORT_SCHEMA);
    let json = serde_json::to_string(&report).unwrap();
    assert!(json.contains(OC_FLOOR_REPORT_SCHEMA));
    std::fs::remove_dir_all(&dir).ok();
}

/// Live gate: run the sonium quick arm and compare with the baseline.
///
/// Needs lane 6A's `qa-oc-bench` plus `SONIUM_FLOOR_DECK` set to the
/// coordinator-frozen `--quick` deck id:
///
/// ```sh
/// SONIUM_FLOOR_DECK=<deck> cargo test -p sonium-bench --test oc_floor -- --ignored live
/// ```
#[test]
#[ignore = "live gate: needs qa-oc-bench (lane 6A) and SONIUM_FLOOR_DECK"]
fn live_sonium_quick_gate() {
    let deck = std::env::var("SONIUM_FLOOR_DECK").expect(
        "SONIUM_FLOOR_DECK must name the coordinator-frozen --quick deck id (lane 6A subset)",
    );
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let scratch = std::env::temp_dir().join(format!("sonium-oc-floor-live-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let candidate = scratch.join("candidate.json");
    let output = Command::new("cargo")
        .current_dir(&workspace)
        .env("RAYON_NUM_THREADS", "1")
        .args([
            "run",
            "-p",
            "sonium-qa",
            "--features",
            "qa-binaries",
            "--bin",
            "qa-oc-bench",
            "--",
            "--sonium-only",
            "--quick",
            "--threads",
            "1",
            "--deck",
            &deck,
            "--emit-bundle",
        ])
        .arg(&candidate)
        .output()
        .expect("spawn qa-oc-bench");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.contains("SUCCESS: qa-oc-bench"),
        "qa-oc-bench failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let report = check_floor(&baseline_path(), &candidate, FloorPolicy::default())
        .expect("fail-closed floor comparison");
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    assert!(
        report.passed(),
        "perf floor failed: {} regression(s)",
        report.regression_count
    );
    std::fs::remove_dir_all(&scratch).ok();
}
