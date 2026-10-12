//! Integration tests for the frozen `sonium.oc-bench/v1` bundle schema.
//!
//! These tests pin the exact JSON field names from
//! `PHASE6_CONTRACTS.md`, the metric math, and the comparison and
//! rendering behavior of `sonium_bench::oc_compare`.

// Rust guideline compliant 2026-02-21

use sonium_bench::oc_compare::{
    BUNDLE_SCHEMA, BenchBundle, BenchRow, HardwareInfo, PROTOCOL_CORRECTNESS, ProtocolInfo, Side,
    Winner, auto_pairs, bundle_json, compare_pair, ns_per_dof_cycle, parse_bundle_json,
    render_bundle_markdown, validate_bundle,
};

fn bundle() -> BenchBundle {
    BenchBundle {
        schema: BUNDLE_SCHEMA.to_string(),
        hardware: HardwareInfo {
            arch: "x86_64".to_string(),
            model: "integration-cpu".to_string(),
            threads: "16".to_string(),
        },
        protocol: ProtocolInfo {
            repeats: 3,
            warmup: 1,
            correctness: PROTOCOL_CORRECTNESS.to_string(),
        },
        rows: vec![
            BenchRow::measured(
                "rail_crush",
                Side::Sonium,
                "native",
                16,
                12.3,
                900,
                15_000,
                123_456_789,
                true,
            )
            .unwrap(),
            BenchRow::measured(
                "rail_crush",
                Side::Engine,
                "smp",
                16,
                6.1,
                900,
                15_000,
                200_000_000,
                true,
            )
            .unwrap(),
        ],
    }
}

#[test]
fn frozen_field_names_match_contract() {
    let text = bundle_json(&bundle()).unwrap();
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(value["schema"], BUNDLE_SCHEMA);
    for key in ["arch", "model", "threads"] {
        assert!(
            value["hardware"].get(key).is_some(),
            "missing hardware.{key}"
        );
    }
    // Hardware threads stay a string; row threads stay a number.
    assert!(value["hardware"]["threads"].is_string());
    for key in ["repeats", "warmup", "correctness"] {
        assert!(
            value["protocol"].get(key).is_some(),
            "missing protocol.{key}"
        );
    }
    let row = &value["rows"][0];
    for key in [
        "deck",
        "side",
        "flavor",
        "threads",
        "wall_s",
        "cycles",
        "dof",
        "ns_per_dof_cycle",
        "peak_rss_bytes",
        "correctness_passed",
    ] {
        assert!(row.get(key).is_some(), "missing row.{key}");
    }
    assert!(row["threads"].is_number());
    assert_eq!(row["side"], "sonium");
    assert_eq!(value["rows"][1]["side"], "engine");
}

#[test]
fn bundle_json_round_trips_through_validation() {
    let text = bundle_json(&bundle()).unwrap();
    let parsed = parse_bundle_json(&text).unwrap();
    assert_eq!(parsed.rows.len(), 2);
    assert_eq!(parsed.schema, BUNDLE_SCHEMA);
}

#[test]
fn contract_example_metric_value() {
    let cost = ns_per_dof_cycle(12.3, 900, 15_000).unwrap();
    assert!((cost - 911.0).abs() < 1.0);
}

#[test]
fn tampered_metric_fails_closed() {
    let mut tampered = bundle_json(&bundle()).unwrap();
    tampered = tampered.replacen("911.", "100.", 1);
    assert_ne!(bundle_json(&bundle()).unwrap(), tampered);
    assert!(parse_bundle_json(&tampered).is_err());
}

#[test]
fn auto_pairs_match_same_deck_and_threads() {
    let pairs = auto_pairs(&bundle());
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].0.side, Side::Sonium);
    assert_eq!(pairs[0].1.side, Side::Engine);
}

#[test]
fn engine_win_classified_when_faster() {
    let pairs = auto_pairs(&bundle());
    let compared = compare_pair(&pairs[0].0, &pairs[0].1).unwrap();
    assert_eq!(compared.winner, Winner::Engine);
    assert!(compared.speedup_sonium_vs_engine < 1.0);
}

#[test]
fn cross_deck_comparison_rejected() {
    let sonium = BenchRow::measured(
        "taylor_bar",
        Side::Sonium,
        "native",
        1,
        2.0,
        100,
        600,
        8,
        true,
    )
    .unwrap();
    let engine = BenchRow::measured(
        "rail_crush",
        Side::Engine,
        "serial",
        1,
        2.0,
        100,
        600,
        8,
        true,
    )
    .unwrap();
    assert!(compare_pair(&sonium, &engine).is_err());
}

#[test]
fn empty_bundle_rejected() {
    let mut empty = bundle();
    empty.rows.clear();
    assert!(validate_bundle(&empty).is_err());
}

#[test]
fn markdown_report_covers_rows_and_winner() {
    let report = render_bundle_markdown(&bundle()).unwrap();
    assert!(report.contains("rail_crush"));
    assert!(report.contains("native"));
    assert!(report.contains("smp"));
    assert!(report.contains("Engine"));
}
