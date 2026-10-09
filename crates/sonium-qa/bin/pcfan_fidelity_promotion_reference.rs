//! Compile-only QA contract for reduced-order fidelity promotion.
//!
//! The fixture is synthetic and checks the admission metrics only; it is not
//! evidence that a resolved-flow run or measured product dataset exists.

use sonium_qa::QaResult;
use sonium_sim_pcfan::{
    PcFanFidelityDataset, PcFanFidelityLevel, PcFanFidelityObservation,
    PcFanFidelityPromotionConfig, evaluate_pc_fan_fidelity_promotion,
};

const CASE_ID: &str = "pcfan.fidelity-promotion-reference.v1";

fn dataset(provenance: &str, fidelity: PcFanFidelityLevel, offset: f64) -> PcFanFidelityDataset {
    PcFanFidelityDataset {
        provenance: provenance.into(),
        observations: [1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .enumerate()
            .map(|(index, objective)| PcFanFidelityObservation {
                candidate_id: format!("candidate-{index}"),
                fidelity,
                objective: objective + offset * objective,
                feasible: true,
                airflow_cfm: 20.0 + index as f64,
                pressure_rise_pa: 100.0 + index as f64,
                acoustic_level_db: 40.0 + index as f64,
                gradient: None,
            })
            .collect(),
    }
}

fn run() -> Result<(f64, f64), String> {
    let report = evaluate_pc_fan_fidelity_promotion(
        &PcFanFidelityPromotionConfig::default(),
        &dataset(
            "solver:reduced-fixture:v1",
            PcFanFidelityLevel::ReducedOrder,
            0.0,
        ),
        &dataset(
            "solver:rotating-wall-fixture:v1",
            PcFanFidelityLevel::RotatingWallCfd,
            0.02,
        ),
    )
    .map_err(|error| error.to_string())?;
    if !report.is_valid() || !report.best_candidate_matches {
        return Err(format!("fidelity promotion contract failed: {report:?}"));
    }
    Ok((report.rank_correlation, report.max_objective_relative_error))
}

fn main() {
    match run() {
        Ok((rank_correlation, max_objective_relative_error)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("rank_correlation", rank_correlation)
                .metric("max_objective_relative_error", max_objective_relative_error)
                .minimum("rank_correlation", 0.90)
                .maximum("max_objective_relative_error", 0.10)
                .input_checksum("synthetic/deterministic-fixture:pcfan-fidelity-promotion-v1")
                .reference(
                    "candidate topology, objective error, rank, and feasibility parity admission",
                )
                .reference_checksum("synthetic/deterministic-fixture:pcfan-fidelity-promotion-v1")
                .dimension("four-candidate reduced versus rotating-wall fidelity fixture")
                .formulation("PC-fan fidelity promotion contract")
                .tolerance("rank correlation >= 0.90 and objective relative error <= 0.10")
                .seed("pcfan-fidelity-promotion-reference:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("PC-fan fidelity result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("PC-fan fidelity failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
