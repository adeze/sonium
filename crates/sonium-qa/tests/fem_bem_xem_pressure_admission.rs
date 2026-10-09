//! Bounded W2-01 evidence for the shared FEM/BEM -> XEM pressure contract.
//!
//! The test uses one small production BEM solve as the source of a valid
//! surface-pressure vector, presents the same vector through the public FEM
//! `Solution` adapter, and admits both through the canonical XEM complex
//! acoustic-pressure field.  This deliberately tests the boundary contract,
//! not numerical agreement between FEM and BEM discretisations.

use num_complex::Complex64;
use sonium_bem::core::{BemProblem, BemSolver};
use sonium_fem::solver::{AccuracyStatus, LinearSolveBackendProvenance, Solution, SolverStatus};
use sonium_xem::{
    ComplexStorage, ConventionRegistry, Domain, DomainKind, FieldKind, FieldSpec,
    MultiphysicsProblem, ScalarType, StorageOrder, TensorLayout,
};

const CASE_ID: &str = "w2-01.fem-bem-xem-pressure-admission.v1";
const FIELD_ID: &str = "pressure";

fn problem_fixture(element_count: usize) -> MultiphysicsProblem {
    MultiphysicsProblem {
        id: "w2-01-fem-bem-xem-pressure-v1".into(),
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

fn synthetic_fem_solution(values: &[Complex64]) -> Solution {
    Solution {
        values: values.to_vec(),
        iterations: 0,
        residual: 0.0,
        converged: true,
        status: SolverStatus::Converged,
        attainable_accuracy: AccuracyStatus::Attained,
        backend: LinearSolveBackendProvenance::cpu(),
    }
}

#[test]
fn w2_01_fem_bem_xem_pressure_admission_v1() {
    let bem_problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let bem_solution = BemSolver::new()
        .solve(&bem_problem)
        .expect("small production BEM fixture must solve");
    let pressure = bem_solution.surface_pressure.clone();
    assert!(!pressure.is_empty(), "fixture must produce pressure DOFs");

    let problem = problem_fixture(pressure.len());
    problem
        .validate()
        .expect("shared XEM problem must validate before either consumer");
    let field = problem
        .acoustic_complex_pressure_field(FIELD_ID)
        .expect("pressure field must satisfy the canonical acoustic contract");
    assert_eq!(field.layout.element_count().unwrap(), pressure.len());

    // The two solver families expose the same checked, borrowed hand-off.
    // Equality here is over the adapter view and layout, not an assertion
    // that independently discretised FEM and BEM solves are interchangeable.
    let fem_solution = synthetic_fem_solution(&pressure);
    let bem_view = bem_solution
        .borrowed_xem_field(&problem, FIELD_ID)
        .expect("BEM pressure must be admitted by XEM");
    let fem_view = fem_solution
        .borrowed_xem_field(&problem, FIELD_ID)
        .expect("FEM pressure must be admitted by XEM");
    assert_eq!(bem_view.layout(), fem_view.layout());
    assert_eq!(bem_view.values(), fem_view.values());
    assert_eq!(bem_view.values(), pressure.as_slice());

    // Both metadata-only result envelopes must retain the canonical layout,
    // field identity, and exact byte extent without copying the values.
    let bem_result = bem_solution
        .to_xem_result(&problem, FIELD_ID)
        .expect("BEM result envelope");
    let fem_result = fem_solution
        .to_xem_result(&problem, FIELD_ID)
        .expect("FEM result envelope");
    bem_result.validate().expect("BEM XEM result validation");
    fem_result.validate().expect("FEM XEM result validation");
    for result in [&bem_result, &fem_result] {
        assert_eq!(result.problem_id, problem.id);
        assert_eq!(result.fields[0].field_id, FIELD_ID);
        assert_eq!(result.fields[0].layout, field.layout);
        assert_eq!(
            result.fields[0].data.byte_length,
            (pressure.len() * std::mem::size_of::<Complex64>()) as u64
        );
    }
    assert_eq!(
        bem_result.fields[0].data.uri.as_deref(),
        Some("mem://sonium/bem/w2-01-fem-bem-xem-pressure-v1/pressure")
    );
    assert_eq!(
        fem_result.fields[0].data.uri.as_deref(),
        Some("mem://sonium/fem/w2-01-fem-bem-xem-pressure-v1/pressure")
    );

    // The boundary must reject the same malformed field contract from both
    // producers: wrong length and non-pressure units.
    let short_fem = synthetic_fem_solution(&pressure[..pressure.len() - 1]);
    assert!(short_fem.borrowed_xem_field(&problem, FIELD_ID).is_err());

    let mut wrong_units = problem.clone();
    wrong_units.fields[0].units = Some("kPa".into());
    assert!(
        bem_solution
            .borrowed_xem_field(&wrong_units, FIELD_ID)
            .is_err()
    );
    assert!(
        fem_solution
            .borrowed_xem_field(&wrong_units, FIELD_ID)
            .is_err()
    );

    println!(
        "QA_RESULT: {{\"case_id\":\"{CASE_ID}\",\"status\":\"passed\",\"pressure_dofs\":{},\"byte_length\":{}}}",
        pressure.len(),
        pressure.len() * std::mem::size_of::<Complex64>()
    );
}
