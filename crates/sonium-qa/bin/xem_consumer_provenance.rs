//! Release evidence for XEM result-envelope adoption and provenance retention.
//!
//! This case complements the typed-medium adapter matrix by exercising the
//! public problem/result handoff used by downstream consumers.  It validates
//! canonical complex-pressure admission, release provenance, serde round-trip,
//! and zero-copy borrowed field consumption.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_xem::{
    BorrowedResultField, ComplexStorage, ConventionRegistry, DataReference, DerivativeConvention,
    Domain, DomainKind, ExecutionMetadata, FidelityTag, FieldKind, FieldSpec, HarmonicConvention,
    MeshLineage, MultiphysicsProblem, MultiphysicsResult, Provenance, ScalarType, StorageOrder,
    TensorLayout,
};
use std::collections::BTreeMap;

const CASE_ID: &str = "xem.consumer-provenance.v1";

fn release_provenance() -> Provenance {
    Provenance {
        fidelity: FidelityTag::Simulated,
        producer: Some("xem-consumer-adoption-fixture".into()),
        code_revision: Some("xem-consumer-provenance-v1".into()),
        execution: Some(ExecutionMetadata {
            platform: "qa-host".into(),
            solver: "typed-xem-consumer".into(),
            tolerances: BTreeMap::from([("relative_residual".into(), 1.0e-12)]),
        }),
        parent_geometry_entity: Some("acoustic-domain-1".into()),
        design_region_masks: Vec::new(),
        mesh_lineage: Some(MeshLineage {
            mesh_id: "mesh-result".into(),
            parent_mesh_id: Some("mesh-input".into()),
            relation: "refined".into(),
        }),
        state_transfer: None,
        derivative_convention: DerivativeConvention::default(),
    }
}

fn problem_fixture() -> MultiphysicsProblem {
    let mut problem = MultiphysicsProblem::default();
    problem.id = "xem-consumer-problem-v1".into();
    problem.conventions = ConventionRegistry {
        harmonic_time: HarmonicConvention::ExpPlusIwt,
        ..ConventionRegistry::default()
    };
    problem.domains = vec![Domain {
        id: "acoustic-domain-1".into(),
        label: "bounded acoustic domain".into(),
        kind: DomainKind::Acoustics,
    }];
    problem.fields = vec![FieldSpec {
        id: "pressure".into(),
        name: "complex acoustic pressure".into(),
        domain_id: "acoustic-domain-1".into(),
        kind: FieldKind::ComplexScalar,
        units: Some("Pa".into()),
        layout: TensorLayout {
            scalar_type: ScalarType::ComplexF64,
            shape: vec![2],
            strides: vec![1],
            byte_offset: 0,
            order: StorageOrder::RowMajor,
            complex_storage: ComplexStorage::Interleaved,
            coordinate_frame: None,
        },
    }];
    problem.provenance = release_provenance();
    problem
}

fn run() -> Result<QaResult, String> {
    let problem = problem_fixture();
    problem
        .validate()
        .map_err(|error| format!("validate XEM problem: {error}"))?;

    let pressure = problem
        .acoustic_complex_pressure_field("pressure")
        .map_err(|error| format!("admit canonical pressure field: {error}"))?;
    if pressure.kind != FieldKind::ComplexScalar
        || pressure.layout.scalar_type != ScalarType::ComplexF64
        || pressure.units.as_deref() != Some("Pa")
    {
        return Err("canonical pressure field metadata mismatch".into());
    }

    let result = MultiphysicsResult::from_problem_field(
        &problem,
        "pressure",
        DataReference {
            expression: None,
            uri: Some("xem://buffers/pressure-v1".into()),
            byte_offset: 0,
            byte_length: 2 * std::mem::size_of::<Complex64>() as u64,
        },
    )
    .map_err(|error| format!("construct XEM result envelope: {error}"))?;
    result
        .validate_for_release()
        .map_err(|error| format!("validate XEM result for release: {error}"))?;

    let serialized =
        serde_json::to_string(&result).map_err(|error| format!("serialize XEM result: {error}"))?;
    let decoded: MultiphysicsResult = serde_json::from_str(&serialized)
        .map_err(|error| format!("deserialize XEM result: {error}"))?;
    decoded
        .validate_for_release()
        .map_err(|error| format!("validate deserialized XEM result: {error}"))?;
    if decoded.problem_id != problem.id
        || decoded.fields[0].field_id != "pressure"
        || decoded.provenance.code_revision.as_deref() != Some("xem-consumer-provenance-v1")
        || decoded
            .provenance
            .execution
            .as_ref()
            .map(|execution| execution.solver.as_str())
            != Some("typed-xem-consumer")
    {
        return Err("XEM result provenance or identity was not retained".into());
    }

    let values = [Complex64::new(1.25, -0.5), Complex64::new(-0.25, 0.75)];
    let borrowed = BorrowedResultField::from_result(&decoded, "pressure", &values)
        .map_err(|error| format!("consume borrowed typed pressure field: {error}"))?;
    if borrowed.field_id() != "pressure"
        || borrowed.view().values() != values.as_slice()
        || borrowed.view().layout().scalar_type != ScalarType::ComplexF64
    {
        return Err("borrowed XEM field view mismatch".into());
    }

    let mut untraceable = decoded.clone();
    untraceable.provenance.execution = None;
    let release_rejection = u64::from(untraceable.validate_for_release().is_err());

    let mut undersized = decoded;
    undersized.fields[0].data.byte_length = 1;
    let layout_rejection = u64::from(undersized.validate().is_err());
    if release_rejection != 1 || layout_rejection != 1 {
        return Err("XEM typed rejection coverage incomplete".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("canonical_pressure_field", 1.0)
        .metric("release_result_validation", 1.0)
        .metric("serde_round_trip", 1.0)
        .metric("borrowed_field_consumption", 1.0)
        .metric("typed_rejection_count", (release_rejection + layout_rejection) as f64)
        .minimum("canonical_pressure_field", 1.0)
        .minimum("release_result_validation", 1.0)
        .minimum("serde_round_trip", 1.0)
        .minimum("borrowed_field_consumption", 1.0)
        .minimum("typed_rejection_count", 2.0)
        .input_checksum("analytic/deterministic-fixture:xem-consumer-provenance-v1")
        .reference("canonical XEM pressure envelope, release provenance, serde identity, and borrowed consumer view")
        .reference_checksum("analytic/deterministic-fixture:xem-consumer-provenance-v1")
        .dimension("one complex pressure field with two Complex64 values")
        .formulation("public MultiphysicsProblem/MultiphysicsResult and BorrowedResultField consumer boundary")
        .tolerance("all four adoption invariants and two typed rejection paths must pass")
        .seed("xem-consumer-provenance:pressure-envelope:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("XEM provenance result must serialize")
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
                    .expect("XEM provenance failure result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
