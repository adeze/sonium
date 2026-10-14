//! PORT-OPTIM Step 10: bounded optimisation gate (Tier 0/1 full, Tier 2 capped).
//!
//! Default (fast) tests guard the gate machinery without solving: the
//! Tier-0-only flow is deterministic and feasible, its report JSON
//! round-trips and validates, and the SC1/SC2 verdict truth tables behave
//! (pass, per-margin failure, improvement, documented null, fail-closed).
//!
//! The ignored `step10_bounded_gate_explicit` test runs the real bounded
//! gate (Tier-0 screen + refine, Tier-1 re-rank, capped Tier-2 two-profile
//! leg with the equal-airflow rescale) and asserts the full pass verdict
//! plus the comparison guards. With `PORT_OPTIM_STEP10_PRODUCTION=1` it
//! runs the Tier-2 leg at production fidelity instead of capped. Both are
//! release-only:
//!
//! ```sh
//! cargo test -p sonium-sim-speaker-modeling --release \
//!   --test port_optim_step10 -- --ignored --nocapture
//! ```

use sonium_sim_speaker_modeling::port_flow::{
    Criterion1Inputs, EqualAirflowDelta, LevelDelta, STEP10_MIN_TIER0_FEASIBLE,
    STEP10_REPORT_PROVENANCE, Step10GateConfig, Step10GateReport, check_criterion_1,
    check_criterion_2, run_step10_gate,
};

#[test]
fn tier0_only_gate_is_deterministic_and_feasible() {
    let config = Step10GateConfig::tier0_only();
    let first = run_step10_gate(&config).expect("Tier-0-only gate must run");
    let second = run_step10_gate(&config).expect("Tier-0-only gate must re-run");
    let first_json = first.to_json_string().expect("report must serialize");
    let second_json = second.to_json_string().expect("report must re-serialize");
    assert_eq!(
        first_json, second_json,
        "Tier-0-only evidence must be byte-deterministic"
    );
    first.validate().expect("report must validate");
    assert!(first.passed, "Tier-0-only gate must pass: {}", first.detail);
    assert!(
        first.tier0.straight_feasible,
        "straight tube must screen feasible"
    );
    assert!(
        first.tier0.feasible_count >= STEP10_MIN_TIER0_FEASIBLE,
        "need a feasible set worth promoting, got {}",
        first.tier0.feasible_count
    );
    assert!(
        first.tier0.refined_feasible,
        "refinement must stay feasible"
    );
    assert!(
        first
            .tier0
            .quota
            .first()
            .is_some_and(|u| u.iter().all(|v| *v == 0.0)),
        "quota must lead with the straight tube"
    );
    assert!(first.tier1.is_none() && first.tier2.is_none());
    assert_eq!(first.provenance, STEP10_REPORT_PROVENANCE);
    println!("step-10 Tier-0: {}", first.detail);
}

#[test]
fn tier0_only_report_json_round_trips() {
    let report =
        run_step10_gate(&Step10GateConfig::tier0_only()).expect("Tier-0-only gate must run");
    let json = report.to_json_string().expect("report must serialize");
    let parsed = Step10GateReport::from_json_str(&json).expect("report JSON must parse");
    parsed.validate().expect("parsed report must validate");
    assert_eq!(parsed.detail, report.detail);
    assert_eq!(parsed.passed, report.passed);
    assert_eq!(parsed.tier0.quota, report.tier0.quota);
    // Corrupt quota order: validation must fail closed.
    let mut bad = parsed.clone();
    bad.tier0.quota.reverse();
    assert!(
        bad.validate().is_err(),
        "quota not leading with straight must fail validation"
    );
}

#[test]
fn criterion_1_truth_table() {
    let good = Criterion1Inputs {
        airflow_ratio_tier2: 1.0,
        airflow_ratio_tier0: 1.0,
        f0_winner_hz: 92.6,
        f_target_hz: 92.6,
        compression_db: Some(0.001),
    };
    let verdict = check_criterion_1(&good);
    assert!(verdict.passed && verdict.outcome == "feasible");
    for (name, inputs) in [
        (
            "tier2-airflow",
            Criterion1Inputs {
                airflow_ratio_tier2: 0.94,
                ..good.clone()
            },
        ),
        (
            "tier0-airflow",
            Criterion1Inputs {
                airflow_ratio_tier0: 0.5,
                ..good.clone()
            },
        ),
        (
            "tuning",
            Criterion1Inputs {
                f0_winner_hz: 95.0,
                ..good.clone()
            },
        ),
        (
            "compression",
            Criterion1Inputs {
                compression_db: Some(1.5),
                ..good.clone()
            },
        ),
        (
            "unmeasured-compression",
            Criterion1Inputs {
                compression_db: None,
                ..good.clone()
            },
        ),
    ] {
        let verdict = check_criterion_1(&inputs);
        assert!(
            !verdict.passed && verdict.outcome == "fail",
            "{name} margin must fail SC1: {}",
            verdict.detail
        );
    }
}

#[test]
fn criterion_2_truth_table() {
    let improve = EqualAirflowDelta {
        v_straight_v: 0.1,
        v_winner_v: 0.1,
        match_residual_pct: 0.01,
        d_thd_rel: -0.2,
        d_noise_db: -0.5,
    };
    let null = EqualAirflowDelta {
        d_thd_rel: 0.05,
        d_noise_db: 0.6,
        ..improve.clone()
    };
    let verdict = check_criterion_2(std::slice::from_ref(&improve), false);
    assert!(verdict.passed && verdict.outcome == "improvement");
    let verdict = check_criterion_2(&[improve], true);
    assert!(verdict.passed && verdict.outcome == "improvement");
    // No improvement without a Pareto archive is a plain failure, not a null.
    let verdict = check_criterion_2(std::slice::from_ref(&null), false);
    assert!(!verdict.passed && verdict.outcome == "fail");
    // With the archive it is a documented null result (still passing).
    let verdict = check_criterion_2(&[null], true);
    assert!(verdict.passed && verdict.outcome == "null-documented");
    // Empty levels fail closed either way.
    for pareto in [false, true] {
        let verdict = check_criterion_2(&[], pareto);
        assert!(!verdict.passed, "empty levels must fail (pareto={pareto})");
    }
}

// ---------------------------------------------------------------------------
// Ignored release-only bounded gate
// ---------------------------------------------------------------------------

type GateError = Box<dyn std::error::Error + Send + Sync>;

/// Full Step-10 gate: Tier-0 screen + refine, Tier-1 re-rank, Tier-2 leg
/// (capped by default, production with `PORT_OPTIM_STEP10_PRODUCTION=1`),
/// verdicts, and comparison guards on the real metrics.
#[test]
#[ignore = "Step-10 bounded gate; run explicitly in release mode"]
fn step10_bounded_gate_explicit() -> Result<(), GateError> {
    if cfg!(debug_assertions) {
        return Err("run the Step-10 gate in release mode (cargo test \
             -p sonium-sim-speaker-modeling --release --test port_optim_step10 \
             -- --ignored --nocapture)"
            .into());
    }
    let production = std::env::var("PORT_OPTIM_STEP10_PRODUCTION").as_deref() == Ok("1");
    let config = if production {
        Step10GateConfig::production()
    } else {
        Step10GateConfig::bounded()
    };
    let report = run_step10_gate(&config).map_err(|error| error.to_string())?;
    report.validate().map_err(|error| error.to_string())?;
    assert!(report.passed, "bounded gate must pass: {}", report.detail);
    let tier2 = report.tier2.as_ref().ok_or("Tier-2 leg must have run")?;
    let want_prefix = if production { "production:" } else { "capped:" };
    assert!(
        tier2.fidelity_label.starts_with(want_prefix),
        "Tier-2 ran at {}, want {want_prefix}…",
        tier2.fidelity_label
    );
    assert!(tier2.all_healthy && tier2.steps_complete);
    // Comparison guards on the real metrics.
    let straight_nominal = &tier2.straight[1];
    let winner_nominal = &tier2.winner[1];
    let identity = LevelDelta::compare(straight_nominal, straight_nominal)
        .map_err(|error| error.to_string())?;
    assert_eq!(identity.airflow_ratio, 1.0);
    assert_eq!(identity.d_thd_abs, 0.0);
    assert_eq!(identity.d_noise_db, 0.0);
    assert_eq!(identity.d_mach_ratio, 1.0);
    assert!(
        LevelDelta::compare(straight_nominal, winner_nominal).is_ok(),
        "same-drive straight/winner levels must compare"
    );
    assert!(
        LevelDelta::compare(straight_nominal, &tier2.winner[2]).is_err(),
        "mismatched drive levels must be rejected"
    );
    let airflow = EqualAirflowDelta::compare(straight_nominal, &tier2.airflow_match)
        .map_err(|error| error.to_string())?;
    assert_eq!(airflow, tier2.equal_airflow[0]);
    assert!(
        tier2
            .straight
            .iter()
            .zip(tier2.winner.iter())
            .all(|(s, w)| s.health.passed && w.health.passed),
        "every Tier-2 level must be healthy"
    );
    println!(
        "step-10 explicit (production={production}): {}",
        report.detail
    );
    println!(
        "SC1[{}] SC2[pareto:{}]",
        report
            .criterion_1
            .as_ref()
            .map_or("missing", |v| v.outcome.as_str()),
        report
            .criterion_2_pareto
            .as_ref()
            .map_or("missing", |v| v.outcome.as_str()),
    );
    Ok(())
}
