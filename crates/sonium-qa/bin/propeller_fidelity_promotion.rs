//! Compile-only release-contract wrapper for propeller fidelity promotion.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    FidelityPromotionOptions, FidelityPromotionSample, evaluate_fidelity_promotion,
};

const CASE_ID: &str = "propeller.fidelity-promotion-reference.v1";

fn run() -> Result<(f64, f64), String> {
    let candidates = [
        FidelityPromotionSample {
            candidate_id: 1,
            low_fidelity_score: 1.0,
            high_fidelity_score: 1.1,
        },
        FidelityPromotionSample {
            candidate_id: 2,
            low_fidelity_score: 2.0,
            high_fidelity_score: 2.2,
        },
        FidelityPromotionSample {
            candidate_id: 3,
            low_fidelity_score: 3.0,
            high_fidelity_score: 3.1,
        },
        FidelityPromotionSample {
            candidate_id: 4,
            low_fidelity_score: 4.0,
            high_fidelity_score: 4.4,
        },
    ];
    let result = evaluate_fidelity_promotion(&candidates, FidelityPromotionOptions::default())
        .map_err(|error| error.to_string())?;
    if !result.is_finite() || !result.passed {
        return Err("deterministic fidelity-promotion fixture did not pass".into());
    }
    Ok((result.pairwise_agreement, result.rank_correlation))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((agreement, correlation)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("pairwise_agreement", agreement)
                .metric("rank_correlation", correlation)
                .metric("candidate_count", 4.0)
                .input_checksum("analytic/deterministic-fixture:propeller-fidelity-promotion-v1")
                .reference_checksum("analytic/invariant:propeller-fidelity-promotion-v1")
                .reference(
                    "caller-supplied low/high objective ordering; no fidelity solver executed",
                )
                .dimension("four candidate objective pairs")
                .formulation("strict pairwise rank-preservation promotion gate")
                .tolerance("pairwise agreement = 1; rank correlation = 1")
                .seed("propeller-fidelity-promotion-v1:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serialization")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serialization")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
