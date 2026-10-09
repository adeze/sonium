//! Release-backed public XEM consumer adoption evidence.
//!
//! This integration case exercises the public handoff that BEM/FEM-style
//! consumers use: canonical complex acoustic pressure admission, result
//! envelope construction, convention/provenance retention, serialization,
//! strict release validation, and a borrowed typed-field view.

use num_complex::Complex64;
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
            platform: "test-host".into(),
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

#[test]
fn xem_consumer_provenance_v1_release_contract() -> Result<(), Box<dyn std::error::Error>> {
    let problem = problem_fixture();
    problem.validate()?;

    let pressure = problem.acoustic_complex_pressure_field("pressure")?;
    assert_eq!(pressure.kind, FieldKind::ComplexScalar);
    assert_eq!(pressure.layout.scalar_type, ScalarType::ComplexF64);
    assert_eq!(pressure.units.as_deref(), Some("Pa"));

    let result = MultiphysicsResult::from_problem_field(
        &problem,
        "pressure",
        DataReference {
            expression: None,
            uri: Some("xem://buffers/pressure-v1".into()),
            byte_offset: 0,
            byte_length: 2 * std::mem::size_of::<Complex64>() as u64,
        },
    )?;
    result.validate_for_release()?;

    let serialized = serde_json::to_string(&result)?;
    let decoded: MultiphysicsResult = serde_json::from_str(&serialized)?;
    decoded.validate_for_release()?;
    assert_eq!(decoded.problem_id, problem.id);
    assert_eq!(decoded.fields[0].field_id, "pressure");
    assert_eq!(
        decoded.provenance.code_revision.as_deref(),
        Some("xem-consumer-provenance-v1")
    );
    assert_eq!(
        decoded
            .provenance
            .execution
            .as_ref()
            .map(|execution| execution.solver.as_str()),
        Some("typed-xem-consumer")
    );

    let values = [Complex64::new(1.25, -0.5), Complex64::new(-0.25, 0.75)];
    let borrowed = BorrowedResultField::from_result(&decoded, "pressure", &values)?;
    assert_eq!(borrowed.field_id(), "pressure");
    assert_eq!(borrowed.view().values(), values.as_slice());
    assert_eq!(borrowed.view().layout().scalar_type, ScalarType::ComplexF64);

    let mut untraceable = decoded.clone();
    untraceable.provenance.execution = None;
    assert!(untraceable.validate_for_release().is_err());

    let mut undersized = decoded;
    undersized.fields[0].data.byte_length = 1;
    assert!(undersized.validate().is_err());

    eprintln!(
        "{CASE_ID}: canonical pressure field, release provenance, round-trip, and borrowed adapter passed"
    );
    Ok(())
}
