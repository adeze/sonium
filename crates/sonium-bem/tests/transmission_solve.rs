//! CPU integration for the typed Müller transmission assembly and direct solve.

use num_complex::Complex64;
use sonium_bem::build_transmission_tbem_system_from_xem;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::solver::{
    TransmissionSolveError, solve_incident_transmission, solve_transmission_adjoint_many,
    solve_transmission_adjoint_system, solve_transmission_system,
};
use sonium_xem::AcousticMedium;

const FREQUENCY_HZ: f64 = 420.0;
const RESIDUAL_TOLERANCE: f64 = 1.0e-10;
const TRANSPARENT_FIELD_TOLERANCE: f64 = 1.0e-3;

fn setup_system() -> sonium_bem::TransmissionTbemSystem {
    let mesh = generate_icosphere_mesh(0.2, 1);
    let incident = AcousticMedium::try_new(FREQUENCY_HZ, 343.0, 1.21).expect("incident medium");
    let transmitted =
        AcousticMedium::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0).expect("transmitted medium");
    build_transmission_tbem_system_from_xem(&mesh.elements, &mesh.nodes, incident, transmitted)
        .expect("typed transmission assembly")
}

#[test]
fn incident_transmission_helper_builds_and_solves_rhs() {
    let system = setup_system();
    let solution = solve_incident_transmission(
        &system,
        &sonium_bem::core::incident::IncidentField::plane_wave_z(),
        RESIDUAL_TOLERANCE,
    )
    .expect("incident transmission solve");
    assert!(solution.relative_residual <= RESIDUAL_TOLERANCE);
    assert!(
        solution
            .values()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}

#[test]
fn typed_transmission_system_solves_and_certifies_two_trace_blocks() {
    let system = setup_system();
    let n = system.total_dofs();
    let expected: Vec<Complex64> = (0..n)
        .map(|index| Complex64::new(0.25 + index as f64 * 0.01, -0.1 + index as f64 * 0.005))
        .collect();
    let mut rhs = vec![Complex64::default(); n];
    for row in 0..n {
        for col in 0..n {
            rhs[row] += system.matrix()[[row, col]] * expected[col];
        }
    }

    let solution = solve_transmission_system(&system, &rhs, RESIDUAL_TOLERANCE)
        .expect("assembled transmission system should solve");
    assert!(solution.relative_residual <= RESIDUAL_TOLERANCE);
    assert_eq!(solution.first_trace().len(), system.interface_dofs());
    assert_eq!(solution.second_trace().len(), system.interface_dofs());
    for (actual, expected) in solution.values().iter().zip(expected) {
        assert!((actual - expected).norm() <= 1.0e-8);
    }
}

#[test]
fn transmission_adjoint_certifies_residual_and_inner_product() {
    let system = setup_system();
    let dimension = system.total_dofs();
    let trial = (0..dimension)
        .map(|index| Complex64::new(0.15 + index as f64 / 19.0, -0.07 + index as f64 / 23.0))
        .collect::<Vec<_>>();
    let rhs = (0..dimension)
        .map(|index| Complex64::new(0.4 + index as f64 / 17.0, 0.03 + index as f64 / 29.0))
        .collect::<Vec<_>>();

    let adjoint = solve_transmission_adjoint_system(&system, &rhs, RESIDUAL_TOLERANCE)
        .expect("Hermitian-adjoint transmission solve");
    assert!(adjoint.relative_residual <= RESIDUAL_TOLERANCE);
    assert_eq!(adjoint.first_trace().len(), system.interface_dofs());
    assert_eq!(adjoint.second_trace().len(), system.interface_dofs());

    let forward_action = (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| system.matrix()[[row, column]] * trial[column])
                .sum::<Complex64>()
        })
        .collect::<Vec<_>>();
    let lhs = forward_action
        .iter()
        .zip(adjoint.values())
        .map(|(left, right)| left.conj() * right)
        .sum::<Complex64>();
    let rhs_pairing = trial
        .iter()
        .zip(&rhs)
        .map(|(left, right)| left.conj() * right)
        .sum::<Complex64>();
    assert!((lhs - rhs_pairing).norm() < 1.0e-9);
}

#[test]
fn transmission_adjoint_many_matches_independent_solves_and_rejects_bad_rhs() {
    let system = setup_system();
    let dimension = system.total_dofs();
    let first = (0..dimension)
        .map(|index| Complex64::new(0.2 + index as f64 / 13.0, -(index as f64) / 31.0))
        .collect::<Vec<_>>();
    let second = first
        .iter()
        .enumerate()
        .map(|(index, value)| *value + Complex64::new(index as f64 / 41.0, 0.05))
        .collect::<Vec<_>>();
    let batch = solve_transmission_adjoint_many(
        &system,
        &[first.clone(), second.clone()],
        RESIDUAL_TOLERANCE,
    )
    .expect("adjoint batch");
    assert_eq!(batch.len(), 2);
    for (solution, rhs) in batch.iter().zip([first, second]) {
        let independent = solve_transmission_adjoint_system(&system, &rhs, RESIDUAL_TOLERANCE)
            .expect("independent adjoint");
        assert!(solution.relative_residual <= RESIDUAL_TOLERANCE);
        assert!(
            solution
                .values()
                .iter()
                .zip(independent.values())
                .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-12)
        );
    }
    assert!(
        solve_transmission_adjoint_many(&system, &[], RESIDUAL_TOLERANCE)
            .expect("empty batch")
            .is_empty()
    );
    let mut invalid = vec![Complex64::new(0.0, 0.0); dimension];
    invalid[2] = Complex64::new(f64::NAN, 0.0);
    assert!(matches!(
        solve_transmission_adjoint_system(&system, &invalid, RESIDUAL_TOLERANCE),
        Err(TransmissionSolveError::NonFiniteRhs { index: 2 })
    ));
    assert!(matches!(
        solve_transmission_adjoint_many(&system, &[invalid], RESIDUAL_TOLERANCE),
        Err(TransmissionSolveError::NonFiniteRhs { index: 2 })
    ));
}

#[test]
fn transmission_solve_rejects_invalid_rhs_before_factorization() {
    let system = setup_system();
    let error = solve_transmission_system(&system, &[], RESIDUAL_TOLERANCE)
        .expect_err("wrong RHS length must fail closed");
    assert!(matches!(error, TransmissionSolveError::RhsLength { .. }));

    let mut rhs = vec![Complex64::new(0.0, 0.0); system.total_dofs()];
    rhs[3] = Complex64::new(f64::NAN, 0.0);
    let error = solve_transmission_system(&system, &rhs, RESIDUAL_TOLERANCE)
        .expect_err("non-finite RHS must fail before direct solve");
    assert!(matches!(
        error,
        TransmissionSolveError::NonFiniteRhs { index: 3 }
    ));
}

#[test]
fn transmission_solve_rejects_invalid_residual_tolerance() {
    let system = setup_system();
    let rhs = vec![Complex64::new(1.0, 0.0); system.total_dofs()];
    let error =
        solve_transmission_system(&system, &rhs, 0.0).expect_err("zero tolerance must fail closed");
    assert!(matches!(
        error,
        TransmissionSolveError::InvalidTolerance { value: 0.0 }
    ));
}

#[test]
fn typed_transmission_system_builds_checked_incident_rhs() {
    let system = setup_system();
    let unit = system
        .incident_rhs(&IncidentField::plane_wave_z())
        .expect("finite plane-wave trace");
    assert_eq!(unit.len(), system.total_dofs());
    assert!(
        unit.iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let doubled = system
        .incident_rhs(&IncidentField::PlaneWave {
            direction: [0.0, 0.0, 1.0],
            amplitude: Complex64::new(2.0, 0.0),
        })
        .expect("finite scaled plane-wave trace");
    for (actual, expected) in doubled.iter().zip(unit.iter()) {
        assert!((*actual - Complex64::new(2.0, 0.0) * *expected).norm() < 1.0e-10);
    }

    let malformed = IncidentField::PlaneWave {
        direction: [f64::NAN, 0.0, 1.0],
        amplitude: Complex64::new(1.0, 0.0),
    };
    assert!(system.incident_rhs(&malformed).is_err());
}

#[test]
fn incident_transmission_solution_evaluates_exterior_field() {
    let system = setup_system();
    let incident = IncidentField::plane_wave_z();
    let solution = solve_incident_transmission(&system, &incident, RESIDUAL_TOLERANCE)
        .expect("incident transmission solve");
    let field = solution
        .evaluate_exterior_field_checked(&system, &[[0.0, 0.0, 1.0], [0.3, 0.0, 0.0]], &incident)
        .expect("exterior field evaluation");

    assert_eq!(field.len(), 2);
    assert!(field.iter().all(|point| {
        point.p_incident.re.is_finite()
            && point.p_incident.im.is_finite()
            && point.p_scattered.re.is_finite()
            && point.p_scattered.im.is_finite()
            && point.p_total.re.is_finite()
            && point.p_total.im.is_finite()
    }));
}

#[test]
fn equal_media_transmission_reproduces_incident_exterior_field() {
    let mesh = generate_icosphere_mesh(0.2, 2);
    let incident = AcousticMedium::try_new(FREQUENCY_HZ, 343.0, 1.21).expect("medium");
    let transmitted = AcousticMedium::try_new(FREQUENCY_HZ, 343.0, 1.21).expect("medium");
    let system =
        build_transmission_tbem_system_from_xem(&mesh.elements, &mesh.nodes, incident, transmitted)
            .expect("equal-media transmission assembly");
    let field =
        solve_incident_transmission(&system, &IncidentField::plane_wave_z(), RESIDUAL_TOLERANCE)
            .expect("equal-media transmission solve")
            .evaluate_exterior_field_checked(
                &system,
                &[[0.0, 0.0, 1.0], [0.3, 0.0, 0.0]],
                &IncidentField::plane_wave_z(),
            )
            .expect("equal-media exterior field");
    let points = [[0.0, 0.0, 1.0], [0.3, 0.0, 0.0]];
    let expected = IncidentField::plane_wave_z().evaluate_pressure(&points, &system.exterior);
    for (actual, expected) in field.iter().zip(expected) {
        assert!(
            (actual.p_total - expected).norm() < TRANSPARENT_FIELD_TOLERANCE,
            "total={:?}, expected={:?}, scattered={:?}",
            actual.p_total,
            expected,
            actual.p_scattered
        );
        assert!(
            actual.p_scattered.norm() < TRANSPARENT_FIELD_TOLERANCE,
            "scattered={:?}, total={:?}, expected={:?}",
            actual.p_scattered,
            actual.p_total,
            expected
        );
    }
}
