//! Audit `reviews/audit-sonium-xem-20260921.md`: cross-consumer admission.
//!
//! The audit requires validation at solver admission in every consumer and
//! convention fixtures replayed through the XEM→FEM and XEM→BEM public
//! adapters. This test proves both adapters (`borrowed_xem_field`) reject
//! the same in-process mutations of a shared problem (duplicate ids,
//! dangling references, empty problem id, non-pressure contracts) and admit
//! the same harmonic-convention variants identically: conventions travel as
//! metadata, and a successfully parsed model cannot silently change meaning
//! between consumers.

use num_complex::Complex64;
use sonium_bem::core::{BemProblem, BemSolver};
use sonium_fem::solver::{AccuracyStatus, LinearSolveBackendProvenance, Solution, SolverStatus};
use sonium_xem::{
    ComplexStorage, ConventionRegistry, Domain, DomainKind, FieldKind, FieldSpec,
    HarmonicConvention, MultiphysicsProblem, ScalarType, StorageOrder, TensorLayout,
};

const FIELD_ID: &str = "pressure";

fn problem_fixture(element_count: usize) -> MultiphysicsProblem {
    MultiphysicsProblem {
        id: "audit-xem-20260921-dual-admission-v1".into(),
        conventions: ConventionRegistry::default(),
        domains: vec![Domain {
            id: "acoustics".into(),
            label: "shared acoustic interface".into(),
            kind: DomainKind::Acoustics,
        }],
        fields: vec![FieldSpec {
            id: FIELD_ID.into(),
            name: "complex acoustic pressure".into(),
            domain_id: "acoustics".into(),
            kind: FieldKind::ComplexScalar,
            units: Some("Pa".into()),
            layout: TensorLayout {
                scalar_type: ScalarType::ComplexF64,
                shape: vec![element_count],
                strides: vec![1],
                byte_offset: 0,
                order: StorageOrder::RowMajor,
                complex_storage: ComplexStorage::Interleaved,
                coordinate_frame: None,
            },
        }],
        ..MultiphysicsProblem::default()
    }
}

fn with_harmonic(
    mut problem: MultiphysicsProblem,
    harmonic_time: HarmonicConvention,
    rms_amplitudes: bool,
) -> MultiphysicsProblem {
    problem.conventions.harmonic_time = harmonic_time;
    problem.conventions.rms_amplitudes = rms_amplitudes;
    problem
}

#[test]
fn dual_adapter_mutation_before_admission_is_rejected() {
    let bem_problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let bem_solution = BemSolver::new()
        .solve(&bem_problem)
        .expect("small production BEM fixture must solve");
    let pressure = bem_solution.surface_pressure.clone();
    let fem_solution = Solution {
        values: pressure.clone(),
        iterations: 0,
        residual: 0.0,
        converged: true,
        status: SolverStatus::Converged,
        attainable_accuracy: AccuracyStatus::Attained,
        backend: LinearSolveBackendProvenance::cpu(),
    };

    let valid = problem_fixture(pressure.len());
    assert!(bem_solution.borrowed_xem_field(&valid, FIELD_ID).is_ok());
    assert!(fem_solution.borrowed_xem_field(&valid, FIELD_ID).is_ok());

    // Each in-process mutation must be rejected by BOTH consumers.
    let mut cases: Vec<(&str, MultiphysicsProblem)> = Vec::new();

    let mut empty_id = valid.clone();
    empty_id.id.clear();
    cases.push(("empty problem id", empty_id));

    let mut duplicate = valid.clone();
    duplicate.fields.push(duplicate.fields[0].clone());
    cases.push(("duplicate field id", duplicate));

    let mut dangling = valid.clone();
    dangling.fields[0].domain_id = "removed-domain".into();
    cases.push(("dangling domain reference", dangling));

    let mut structural = valid.clone();
    structural.domains.push(Domain {
        id: "structure".into(),
        label: "solid".into(),
        kind: sonium_xem::DomainKind::Structural,
    });
    structural.fields[0].domain_id = "structure".into();
    cases.push(("non-acoustics pressure field", structural));

    let mut wrong_units = valid.clone();
    wrong_units.fields[0].units = Some("kPa".into());
    cases.push(("non-pascal units", wrong_units));

    for (label, mutated) in &cases {
        assert!(
            bem_solution.borrowed_xem_field(mutated, FIELD_ID).is_err(),
            "BEM adapter admitted {label}"
        );
        assert!(
            fem_solution.borrowed_xem_field(mutated, FIELD_ID).is_err(),
            "FEM adapter admitted {label}"
        );
    }
}

#[test]
fn dual_adapter_convention_variants_admit_identically() {
    let bem_problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let bem_solution = BemSolver::new()
        .solve(&bem_problem)
        .expect("small production BEM fixture must solve");
    let pressure: Vec<Complex64> = bem_solution.surface_pressure.clone();
    let fem_solution = Solution {
        values: pressure.clone(),
        iterations: 0,
        residual: 0.0,
        converged: true,
        status: SolverStatus::Converged,
        attainable_accuracy: AccuracyStatus::Attained,
        backend: LinearSolveBackendProvenance::cpu(),
    };
    let base = problem_fixture(pressure.len());

    // Both harmonic conventions and both amplitude normalizations travel as
    // metadata: neither adapter may accept a variant the other rejects.
    for harmonic_time in [
        HarmonicConvention::ExpMinusIwt,
        HarmonicConvention::ExpPlusIwt,
    ] {
        for rms_amplitudes in [true, false] {
            let problem = with_harmonic(base.clone(), harmonic_time, rms_amplitudes);
            let bem_view = bem_solution
                .borrowed_xem_field(&problem, FIELD_ID)
                .expect("BEM must admit the convention variant");
            let fem_view = fem_solution
                .borrowed_xem_field(&problem, FIELD_ID)
                .expect("FEM must admit the convention variant");
            assert_eq!(bem_view.values(), fem_view.values());
            assert_eq!(bem_view.values(), pressure.as_slice());
        }
    }
}
