//! Release-facing checked periodic/MPC condensation contract.

use num_complex::Complex64;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::boundary::periodic::{MpcError, MpcSystem, PeriodicDofConstraint};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.mpc-system-contract.v1";

fn main() {
    let result = match run() {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("complex_condensation_checks", metrics.complex_condensation_checks as f64)
            .minimum("complex_condensation_checks", 1.0)
            .metric("real_condensation_checks", metrics.real_condensation_checks as f64)
            .minimum("real_condensation_checks", 1.0)
            .metric("recovery_checks", metrics.recovery_checks as f64)
            .minimum("recovery_checks", 1.0)
            .metric("typed_rejection_checks", metrics.typed_rejection_checks as f64)
            .minimum("typed_rejection_checks", 3.0)
            .reference(
                "checked generic FemSystem integration for complex Bloch and real anti-periodic MPC condensation",
            )
            .reference_checksum("analytic/deterministic-fixture:fem-mpc-system-contract-v1")
            .dimension("2x2 and 3x3 real/complex FEM systems")
            .formulation("u_full = P u_reduced; A_reduced = P^H A P; b_reduced = P^H b")
            .tolerance("exact reduced dimensions, analytic matrix/RHS values, and typed rejection")
            .input_checksum("matrix:small-hermitian-and-antiperiodic-fixtures:v1"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };

    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("FEM MPC QA serialization failed: {error}");
            std::process::exit(1);
        }
    }

    if matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    complex_condensation_checks: usize,
    real_condensation_checks: usize,
    recovery_checks: usize,
    typed_rejection_checks: usize,
}

fn complex_fixture() -> (FemSystem<Complex64>, MpcSystem) {
    let mut matrix = TripletMatrix::new(3, 3);
    matrix.push(0, 0, Complex64::new(4.0, 0.0));
    matrix.push(0, 1, Complex64::new(1.0, 1.0));
    matrix.push(1, 0, Complex64::new(1.0, -1.0));
    matrix.push(1, 1, Complex64::new(3.0, 0.0));
    matrix.push(1, 2, Complex64::new(2.0, 0.0));
    matrix.push(2, 1, Complex64::new(2.0, 0.0));
    matrix.push(2, 2, Complex64::new(5.0, 0.0));
    let system = FemSystem::new(
        matrix,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 3.0),
        ],
    );
    let mpc = MpcSystem::from_periodic(
        3,
        &[PeriodicDofConstraint {
            master: 0,
            slave: 2,
            phase: Complex64::new(0.0, 1.0),
        }],
    )
    .expect("complex fixture constraints are valid");
    (system, mpc)
}

fn run() -> Result<Metrics, String> {
    let (complex_system, complex_mpc) = complex_fixture();
    let condensed = complex_mpc
        .condense_fem_system_complex(&complex_system)
        .map_err(|error| format!("complex condensation failed: {error}"))?;
    if condensed.matrix.num_rows != 2
        || condensed.matrix.num_cols != 2
        || condensed.rhs != vec![Complex64::new(4.0, 0.0), Complex64::new(2.0, 0.0)]
    {
        return Err("complex condensation dimensions/RHS mismatch".into());
    }
    let complex_matrix = condensed.matrix.to_csr();
    if (complex_matrix.get(0, 0) - Complex64::new(9.0, 0.0)).norm() > 1.0e-12
        || (complex_matrix.get(0, 1) - Complex64::new(1.0, -1.0)).norm() > 1.0e-12
    {
        return Err("complex condensed matrix mismatch".into());
    }

    let recovered = complex_mpc
        .recover(&[Complex64::new(2.0, 0.0), Complex64::new(-1.0, 0.5)])
        .map_err(|error| format!("complex recovery failed: {error}"))?;
    let constraint = PeriodicDofConstraint {
        master: 0,
        slave: 2,
        phase: Complex64::new(0.0, 1.0),
    };
    if constraint.residual(&recovered).norm() > 1.0e-14 {
        return Err("recovered complex field violates periodic constraint".into());
    }

    let mut real_matrix = TripletMatrix::new(2, 2);
    real_matrix.push(0, 0, 2.0);
    real_matrix.push(0, 1, -1.0);
    real_matrix.push(1, 0, -1.0);
    real_matrix.push(1, 1, 2.0);
    let real_system = FemSystem::new(real_matrix, vec![3.0, 4.0]);
    let real_mpc = MpcSystem::from_periodic(
        2,
        &[PeriodicDofConstraint {
            master: 0,
            slave: 1,
            phase: Complex64::new(-1.0, 0.0),
        }],
    )
    .map_err(|error| format!("real fixture constraints are invalid: {error}"))?;
    let condensed = real_mpc
        .condense_fem_system_real(&real_system)
        .map_err(|error| format!("real condensation failed: {error}"))?;
    if condensed.rhs != vec![-1.0] || (condensed.matrix.to_csr().get(0, 0) - 6.0).abs() > 1.0e-12 {
        return Err("real anti-periodic condensation mismatch".into());
    }

    let mut nonfinite_matrix = TripletMatrix::new(2, 2);
    nonfinite_matrix.push(0, 0, f64::NAN);
    let nonfinite = FemSystem::new(nonfinite_matrix, vec![0.0, 0.0]);
    if !matches!(
        real_mpc.condense_fem_system_real(&nonfinite),
        Err(MpcError::NonFiniteMatrixValue { triplet: 0 })
    ) {
        return Err("non-finite matrix was not rejected with its typed error".into());
    }

    let mut malformed_matrix = TripletMatrix::new(2, 2);
    malformed_matrix.push(2, 0, 1.0);
    let malformed = FemSystem::new(malformed_matrix, vec![0.0, 0.0]);
    if !matches!(
        real_mpc.condense_fem_system_real(&malformed),
        Err(MpcError::InvalidFemSystem(_))
    ) {
        return Err("malformed FEM system was not rejected".into());
    }

    let mut dirichlet = FemSystem::new(TripletMatrix::new(2, 2), vec![0.0, 0.0]);
    sonium_fem::boundary::try_apply_homogeneous_dirichlet_dofs(&mut dirichlet, &[0])
        .map_err(|error| format!("Dirichlet fixture failed: {error}"))?;
    if !matches!(
        real_mpc.condense_fem_system_real(&dirichlet),
        Err(MpcError::DirichletConstraintsPresent)
    ) {
        return Err("MPC/Dirichlet ordering was not enforced".into());
    }

    Ok(Metrics {
        complex_condensation_checks: 1,
        real_condensation_checks: 1,
        recovery_checks: 1,
        typed_rejection_checks: 3,
    })
}
