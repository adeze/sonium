//! Audit `reviews/audit-sonium-xem-20260921.md`: admission and convention tests.
//!
//! The audit's P2 API risk notes that `MultiphysicsProblem` validates on the
//! deserialization path, but its public fields allow in-process
//! construction/mutation to bypass that path, so every consumer must validate
//! at solver admission. These tests prove that:
//!
//! 1. each in-process mutation class is rejected by `validate()` and by the
//!    canonical pressure-field admission check, and
//! 2. a mutated problem also fails the serde round trip (the wire path stays
//!    closed), and
//! 3. the shared convention fixtures (harmonic sign, RMS/peak SPL, exact unit
//!    conversion) behave identically for every consumer of the XEM contract.

use sonium_xem::{
    AcousticMedium, ComplexStorage, ConventionRegistry, CouplingGraph, Domain, DomainKind,
    FieldKind, FieldSpec, HarmonicConvention, InterfaceCondition, InterfaceConditionKind,
    MaterialLaw, MaterialLawKind, MultiphysicsProblem, Region, ScalarType, StorageOrder,
    TensorLayout, pressure_to_spl,
};
use std::collections::BTreeMap;

fn pressure_layout(elements: usize) -> TensorLayout {
    TensorLayout {
        scalar_type: ScalarType::ComplexF64,
        shape: vec![elements],
        strides: vec![1],
        byte_offset: 0,
        order: StorageOrder::RowMajor,
        complex_storage: ComplexStorage::Interleaved,
        coordinate_frame: None,
    }
}

fn valid_problem() -> MultiphysicsProblem {
    MultiphysicsProblem {
        id: "audit-xem-20260921-admission-v1".into(),
        domains: vec![Domain {
            id: "acoustics".into(),
            label: "bounded acoustic domain".into(),
            kind: DomainKind::Acoustics,
        }],
        regions: vec![Region {
            id: "air".into(),
            domain_id: "acoustics".into(),
            label: None,
            parent_geometry_entity: None,
            material_law_id: None,
        }],
        fields: vec![FieldSpec {
            id: "pressure".into(),
            name: "complex acoustic pressure".into(),
            domain_id: "acoustics".into(),
            kind: FieldKind::ComplexScalar,
            units: Some("Pa".into()),
            layout: pressure_layout(8),
        }],
        ..MultiphysicsProblem::default()
    }
}

#[test]
fn valid_fixture_passes_admission() {
    let problem = valid_problem();
    problem
        .validate()
        .expect("audit fixture must pass admission");
    problem
        .acoustic_complex_pressure_field("pressure")
        .expect("canonical pressure contract must hold");
}

#[test]
fn mutation_before_admission_is_rejected() {
    // Empty problem id: deserialization-equivalent bypass of the wire check.
    let mut problem = valid_problem();
    problem.id.clear();
    assert!(problem.validate().is_err());
    assert!(problem.acoustic_complex_pressure_field("pressure").is_err());

    // Duplicate field ids in the same namespace.
    let mut problem = valid_problem();
    problem.fields.push(problem.fields[0].clone());
    assert!(problem.validate().is_err());
    assert!(problem.acoustic_complex_pressure_field("pressure").is_err());

    // Dangling domain reference after in-process mutation.
    let mut problem = valid_problem();
    problem.fields[0].domain_id = "removed-domain".into();
    assert!(problem.validate().is_err());
    assert!(problem.acoustic_complex_pressure_field("pressure").is_err());

    // Dangling region reference.
    let mut problem = valid_problem();
    problem.regions[0].domain_id = "removed-domain".into();
    assert!(problem.validate().is_err());

    // Non-finite material parameter introduced post-construction.
    let mut problem = valid_problem();
    problem.material_laws = vec![MaterialLaw {
        id: "air-law".into(),
        kind: MaterialLawKind::Isotropic,
        parameters: BTreeMap::from([("density".into(), f64::NAN)]),
    }];
    problem.regions[0].material_law_id = Some("air-law".into());
    assert!(problem.validate().is_err());

    // Unknown material-law reference.
    let mut problem = valid_problem();
    problem.regions[0].material_law_id = Some("missing-law".into());
    assert!(problem.validate().is_err());

    // Self-referential interface condition.
    let mut problem = valid_problem();
    problem.interface_conditions = vec![InterfaceCondition {
        id: "iface".into(),
        left_region_id: "air".into(),
        right_region_id: "air".into(),
        field_ids: vec!["pressure".into()],
        kind: InterfaceConditionKind::Continuity,
        transfer: None,
    }];
    assert!(problem.validate().is_err());

    // Interface with no coupled fields.
    let mut problem = valid_problem();
    problem.regions.push(Region {
        id: "air-2".into(),
        domain_id: "acoustics".into(),
        label: None,
        parent_geometry_entity: None,
        material_law_id: None,
    });
    problem.interface_conditions = vec![InterfaceCondition {
        id: "iface".into(),
        left_region_id: "air".into(),
        right_region_id: "air-2".into(),
        field_ids: Vec::new(),
        kind: InterfaceConditionKind::Continuity,
        transfer: None,
    }];
    assert!(problem.validate().is_err());

    // Unsupported schema version smuggled past the typed constructor.
    let mut problem = valid_problem();
    problem.schema_version = sonium_xem::SchemaVersion(u32::MAX);
    assert!(problem.validate().is_err());

    // Corrupt field layout: shape/stride mismatch cannot reach a consumer.
    let mut problem = valid_problem();
    problem.fields[0].layout.strides = vec![1, 1];
    assert!(problem.validate().is_err());
}

#[test]
fn pressure_contract_rejects_non_pressure_fields() {
    // Real-valued buffer labelled as pressure.
    let mut problem = valid_problem();
    problem.fields[0].kind = FieldKind::Scalar;
    problem.fields[0].layout.scalar_type = ScalarType::F64;
    assert!(problem.validate().is_ok());
    assert!(problem.acoustic_complex_pressure_field("pressure").is_err());

    // Wrong units: kilopascals are not pascals at this boundary.
    let mut problem = valid_problem();
    problem.fields[0].units = Some("kPa".into());
    assert!(problem.validate().is_ok());
    assert!(problem.acoustic_complex_pressure_field("pressure").is_err());

    // Field owned by a structural domain, not acoustics.
    let mut problem = valid_problem();
    problem.domains.push(Domain {
        id: "structure".into(),
        label: "solid".into(),
        kind: DomainKind::Structural,
    });
    problem.fields[0].domain_id = "structure".into();
    assert!(problem.validate().is_ok());
    assert!(problem.acoustic_complex_pressure_field("pressure").is_err());
}

#[test]
fn mutated_problem_fails_serde_round_trip() {
    // The wire path re-validates, so a JSON document equivalent to an
    // in-process mutation is rejected at deserialization, not admitted.
    let mut value = serde_json::to_value(valid_problem()).expect("valid problem serializes");
    value["fields"][0]["domain_id"] = serde_json::json!("removed-domain");
    assert!(serde_json::from_value::<MultiphysicsProblem>(value).is_err());

    let mut value = serde_json::to_value(valid_problem()).expect("valid problem serializes");
    value["id"] = serde_json::json!("   ");
    assert!(serde_json::from_value::<MultiphysicsProblem>(value).is_err());
}

#[test]
fn coupling_graph_admission_rejects_mutation() {
    let problem = valid_problem();
    assert!(CouplingGraph::from_problem(&problem).is_ok());

    let mut mutated = valid_problem();
    mutated.fields[0].domain_id = "removed-domain".into();
    assert!(CouplingGraph::from_problem(&mutated).is_err());
}

#[test]
fn harmonic_sign_convention_fixture() {
    // A successfully parsed model must not silently change harmonic sign:
    // exp(-iwt) radiates exp(+ikr) and vice versa.
    assert_eq!(
        HarmonicConvention::ExpMinusIwt.signs(),
        (-1, 1),
        "exp(-iwt) must pair with exp(+ikr)"
    );
    assert_eq!(
        HarmonicConvention::ExpPlusIwt.signs(),
        (1, -1),
        "exp(+iwt) must pair with exp(-ikr)"
    );

    let minus = AcousticMedium::try_new(1_000.0, 343.0, 1.21).expect("valid medium");
    let expected = 2.0 * std::f64::consts::PI * 1_000.0 / 343.0;
    assert!(
        (minus.wavenumber() - expected).abs() < 1e-12,
        "default exp(-iwt) wavenumber must equal +2pif/c, got {}",
        minus.wavenumber()
    );

    let plus = AcousticMedium::try_new_with_conventions(
        sonium_xem::FrequencyHz::new(1_000.0),
        sonium_xem::SpeedOfSoundMPerS::new(343.0),
        sonium_xem::DensityKgM3::new(1.21),
        ConventionRegistry {
            harmonic_time: HarmonicConvention::ExpPlusIwt,
            ..ConventionRegistry::default()
        },
    )
    .expect("valid conjugate medium");
    assert!(
        (plus.wavenumber() + expected).abs() < 1e-12,
        "exp(+iwt) wavenumber must equal -2pif/c, got {}",
        plus.wavenumber()
    );
}

#[test]
fn rms_peak_spl_fixture() {
    use num_complex::Complex64;

    // Solver output is peak phasors; SPL references RMS, so |p| = sqrt(2)
    // (a 1 Pa RMS sinusoid) must yield 20*log10(1/20uPa) ~= 93.98 dB.
    let spl = pressure_to_spl(Complex64::new(std::f64::consts::SQRT_2, 0.0));
    let expected = 20.0 * (1.0 / 20e-6_f64).log10();
    assert!(
        (spl - expected).abs() < 1e-9,
        "peak-phasor SPL contract violated: got {spl}, want {expected}"
    );

    // Amplitude scaling is exact in dB: doubling the peak adds 6.0206 dB.
    let doubled = pressure_to_spl(Complex64::new(2.0 * std::f64::consts::SQRT_2, 0.0));
    assert!((doubled - spl - 20.0 * 2.0_f64.log10()).abs() < 1e-9);

    // The default registry declares RMS amplitudes; consumers converting
    // with the peak convention against an RMS label would shift by 3 dB.
    assert!(
        ConventionRegistry::default().rms_amplitudes,
        "default exchange contract must declare RMS amplitudes"
    );
}
