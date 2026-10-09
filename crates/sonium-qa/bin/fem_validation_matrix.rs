//! Release-facing validation of the FEM physics maturity inventory.
//!
//! This makes the machine-readable promotion invariants visible to QA and
//! archives every claimed maturity/evidence row in CI logs.

use sonium_fem::validation::{
    PhysicsMaturity, PhysicsValidationRecord, physics_validation_matrix,
    validate_physics_validation_matrix,
};
use sonium_qa::QaResult;

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn verify_case_artifacts(case_id: &str, case: &toml::Value) -> Result<(), String> {
    let paths = case["artifact_paths"]
        .as_array()
        .ok_or_else(|| format!("FEM evidence case {case_id} has no artifact_paths"))?;
    let checksums = case["artifact_checksums"]
        .as_array()
        .ok_or_else(|| format!("FEM evidence case {case_id} has no artifact_checksums"))?;
    if paths.len() != checksums.len() || paths.is_empty() {
        return Err(format!(
            "FEM evidence case {case_id} has mismatched artifact paths/checksums"
        ));
    }
    let algorithm = case["artifact_checksum_algorithm"]
        .as_str()
        .ok_or_else(|| format!("FEM evidence case {case_id} has no checksum algorithm"))?;
    if algorithm != "fnv1a64" {
        return Err(format!(
            "FEM evidence case {case_id} uses unsupported checksum algorithm {algorithm}"
        ));
    }
    for (path, expected) in paths.iter().zip(checksums) {
        let path = path
            .as_str()
            .ok_or_else(|| format!("FEM evidence case {case_id} has a non-string artifact path"))?;
        let expected = expected
            .as_str()
            .ok_or_else(|| format!("FEM evidence case {case_id} has a non-string checksum"))?;
        let bytes = std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path))
            .map_err(|error| format!("FEM evidence case {case_id} cannot read {path}: {error}"))?;
        let actual = format!("fnv1a64:{:016x}", fnv1a64(&bytes));
        if actual != expected {
            return Err(format!(
                "FEM evidence case {case_id} checksum mismatch for {path}: expected {expected}, got {actual}"
            ));
        }
    }
    Ok(())
}

fn verify_manifest_evidence(matrix: &[PhysicsValidationRecord]) -> Result<(), String> {
    let manifest: toml::Value = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/validation-manifest.toml"
    ))
    .parse()
    .map_err(|error| format!("parse validation manifest: {error}"))?;
    let cases = manifest["case"]
        .as_array()
        .ok_or_else(|| "validation manifest has no [[case]] entries".to_owned())?;

    for record in matrix {
        let Some(case_id) = record.evidence_case else {
            continue;
        };
        let case = cases
            .iter()
            .find(|case| case["id"].as_str() == Some(case_id))
            .ok_or_else(|| {
                format!(
                    "FEM module {} references missing QA evidence case {case_id}",
                    record.module
                )
            })?;
        if case["owner"].as_str() != Some("fem") {
            return Err(format!(
                "FEM module {} references non-FEM QA evidence case {case_id}",
                record.module
            ));
        }
        if matches!(
            record.maturity,
            PhysicsMaturity::ConvergenceVerified
                | PhysicsMaturity::ExternallyValidated
                | PhysicsMaturity::Production
        ) {
            verify_case_artifacts(case_id, case)?;
        }
    }
    Ok(())
}

fn maturity_name(maturity: PhysicsMaturity) -> &'static str {
    match maturity {
        PhysicsMaturity::Prototype => "prototype",
        PhysicsMaturity::VerifiedForm => "verified-form",
        PhysicsMaturity::ConvergenceVerified => "convergence-verified",
        PhysicsMaturity::ExternallyValidated => "externally-validated",
        PhysicsMaturity::Production => "production",
    }
}

fn main() {
    let matrix = physics_validation_matrix();
    if let Err(error) = validate_physics_validation_matrix(matrix) {
        if let Ok(record) = QaResult::failed("fem.validation-matrix.v1", error.to_string())
            .requested_backend("cpu")
            .actual_backend("cpu")
            .to_json_line()
        {
            println!("QA_RESULT: {record}");
        }
        eprintln!("FEM VALIDATION MATRIX FAILED: {error}");
        println!("FAILED");
        std::process::exit(1);
    }
    if let Err(error) = verify_manifest_evidence(matrix) {
        if let Ok(record) = QaResult::failed("fem.validation-matrix.v1", error.clone())
            .requested_backend("cpu")
            .actual_backend("cpu")
            .to_json_line()
        {
            println!("QA_RESULT: {record}");
        }
        eprintln!("FEM VALIDATION MATRIX MANIFEST FAILED: {error}");
        println!("FAILED");
        std::process::exit(1);
    }
    for record in matrix {
        println!(
            "module={} maturity={} convergence_gate={} evidence_case={} reference={} next_gate={}",
            record.module,
            maturity_name(record.maturity),
            record.has_convergence_gate,
            record.evidence_case.unwrap_or("none registered"),
            record.reference,
            record.next_gate
        );
    }
    let record = QaResult::passed("fem.validation-matrix.v1")
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("module_count", matrix.len() as f64)
        .metric(
            "convergence_gate_count",
            matrix
                .iter()
                .filter(|record| record.has_convergence_gate)
                .count() as f64,
        );
    if let Ok(record) = record.to_json_line() {
        println!("QA_RESULT: {record}");
    }
    println!("SUCCESS");
}
