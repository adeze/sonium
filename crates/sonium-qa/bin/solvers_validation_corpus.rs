//! Release-facing inventory check for the deterministic solver corpus.

use sonium_qa::QaResult;
use sonium_solvers::validation_corpus::{CorpusClass, SOLVER_VALIDATION_CORPUS};
use std::collections::HashSet;
use std::time::Instant;

const CASE_ID: &str = "solvers.validation-corpus.v2";

fn main() {
    let started = Instant::now();
    let mut ids = HashSet::new();
    let mut classes = HashSet::new();
    let mut adversarial = 0.0;
    let mut invalid = 0.0;
    for case in SOLVER_VALIDATION_CORPUS {
        if !ids.insert(case.id)
            || case.dimension == 0
            || case.nonzeros < case.dimension
            || case.assumptions.is_empty()
            || case.workload.is_empty()
            || case.conditioning.is_empty()
            || case.reference_tolerance.is_empty()
        {
            invalid += 1.0;
        }
        classes.insert(case.class);
        if case.adversarial {
            adversarial += 1.0;
        }
    }
    let required_classes = [
        CorpusClass::Spd,
        CorpusClass::Hpd,
        CorpusClass::SymmetricIndefinite,
        CorpusClass::Nonsymmetric,
        CorpusClass::SaddlePoint,
        CorpusClass::IllScaled,
        CorpusClass::NearSingular,
        CorpusClass::RankDeficient,
        CorpusClass::Helmholtz,
        CorpusClass::ConvectionDiffusion,
    ];
    let missing_classes = required_classes
        .iter()
        .filter(|class| !classes.contains(class))
        .count() as f64;
    let count = SOLVER_VALIDATION_CORPUS.len() as f64;
    let result = if count >= 12.0 && invalid == 0.0 && missing_classes == 0.0 && adversarial >= 6.0
    {
        QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("fixture_count", count)
            .metric("adversarial_fixture_count", adversarial)
            .metric("missing_required_class_count", missing_classes)
            .metric("invalid_row_count", invalid)
    .minimum("fixture_count", 12.0)
    .minimum("adversarial_fixture_count", 6.0)
            .maximum("missing_required_class_count", 0.0)
            .maximum("invalid_row_count", 0.0)
        .reference_checksum("inventory:sonium-solvers-validation-corpus-v2")
    .dimension("SPD, HPD, symmetric-indefinite, nonsymmetric, saddle-point, ill-scaled, near-singular, rank-deficient, Helmholtz, convection-diffusion")
            .elapsed_ms(started.elapsed().as_millis() as u64)
    } else {
        QaResult::failed(
            CASE_ID,
            format!(
                "invalid corpus: count={count}, invalid={invalid}, missing_classes={missing_classes}, adversarial={adversarial}"
            ),
        )
        .elapsed_ms(started.elapsed().as_millis() as u64)
    };
    match result.to_json_line() {
        Ok(line) => println!("QA_RESULT: {line}"),
        Err(error) => eprintln!("QA_RESULT serialization failed: {error}"),
    }
    if matches!(&result.status, sonium_qa::QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
