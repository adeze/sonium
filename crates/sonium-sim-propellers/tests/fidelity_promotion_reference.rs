use sonium_sim_propellers::{
    FidelityPromotionError, FidelityPromotionOptions, FidelityPromotionSample,
    evaluate_fidelity_promotion,
};

fn candidates() -> Vec<FidelityPromotionSample> {
    vec![
        FidelityPromotionSample {
            candidate_id: 10,
            low_fidelity_score: 0.8,
            high_fidelity_score: 0.9,
        },
        FidelityPromotionSample {
            candidate_id: 20,
            low_fidelity_score: 1.1,
            high_fidelity_score: 1.2,
        },
        FidelityPromotionSample {
            candidate_id: 30,
            low_fidelity_score: 1.4,
            high_fidelity_score: 1.35,
        },
    ]
}

#[test]
fn public_promotion_gate_accepts_preserved_order() {
    let result = evaluate_fidelity_promotion(&candidates(), FidelityPromotionOptions::default())
        .expect("preserved ranking should pass");
    assert!(result.is_finite());
    assert_eq!(result.pair_count, 3);
    assert_eq!(result.preserved_pair_count, 3);
    assert!(result.passed);
}

#[test]
fn public_promotion_gate_retains_candidate_rank_diagnostics() {
    let result = evaluate_fidelity_promotion(&candidates(), FidelityPromotionOptions::default())
        .expect("ranking should be evaluated");
    assert_eq!(result.points.len(), 3);
    assert_eq!(result.points[0].candidate_id, 10);
    assert_eq!(result.points[0].low_fidelity_rank, 1);
    assert_eq!(result.points[0].high_fidelity_rank, 1);
    assert!(result.points.iter().all(|point| point.rank_preserved));
}

#[test]
fn public_promotion_gate_rejects_rank_regression_under_strict_thresholds() {
    let mut input = candidates();
    input[0].high_fidelity_score = 2.0;
    let result = evaluate_fidelity_promotion(
        &input,
        FidelityPromotionOptions {
            minimum_pairwise_agreement: 1.0,
            minimum_rank_correlation: 1.0,
            ..FidelityPromotionOptions::default()
        },
    )
    .expect("rank regression should remain inspectable");
    assert!(!result.passed);
    assert!(result.preserved_pair_count < result.pair_count);
}

#[test]
fn public_promotion_gate_rejects_ties_and_nonfinite_inputs() {
    let mut tied = candidates();
    tied[1].high_fidelity_score = tied[0].high_fidelity_score;
    assert!(matches!(
        evaluate_fidelity_promotion(&tied, FidelityPromotionOptions::default()),
        Err(FidelityPromotionError::TiedScore {
            fidelity: "high",
            ..
        })
    ));
    let mut invalid = candidates();
    invalid[0].low_fidelity_score = f64::NAN;
    assert!(matches!(
        evaluate_fidelity_promotion(&invalid, FidelityPromotionOptions::default()),
        Err(FidelityPromotionError::InvalidCandidate {
            field: "low_fidelity_score",
            ..
        })
    ));
}

#[test]
fn public_promotion_gate_is_deterministic() {
    let first = evaluate_fidelity_promotion(&candidates(), FidelityPromotionOptions::default())
        .expect("first promotion");
    let second = evaluate_fidelity_promotion(&candidates(), FidelityPromotionOptions::default())
        .expect("second promotion");
    assert_eq!(first, second);
}
