//! CPU-only order-2 H(curl) periodic face-mode contract.

use num_complex::Complex64;
use sonium_fem::boundary::periodic::{
    HcurlOrder2PeriodicFacePair, MpcError, MpcSystem, expand_order2_hcurl_face_constraints,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hcurl-order2-periodic-face-contract.v1";

fn main() {
    let result = match run() {
        Ok((constraint_count, recovery_checks, rejection_checks)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("periodic_face_mode_constraints", constraint_count as f64)
            .minimum("periodic_face_mode_constraints", 4.0)
            .metric("recovery_checks", recovery_checks as f64)
            .minimum("recovery_checks", 4.0)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 3.0)
            .dimension("affine tetrahedral order-2 Nedelec periodic triangular faces")
            .formulation("canonical two-mode face permutation with complex Floquet phase")
            .input_checksum("periodic-face-pairs:identity-cyclic-order2-fixture:v1")
            .reference_checksum("analytic/deterministic-fixture:fem-hcurl-order2-periodic-face-v1")
            .seed("fem-hcurl-order2-periodic-face:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, error),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}

fn run() -> Result<(usize, usize, usize), String> {
    let phase = Complex64::new(0.6, 0.8);
    let pairs = [
        HcurlOrder2PeriodicFacePair {
            master_face: 1,
            slave_face: 5,
            master_vertices: [10, 20, 30],
            slave_vertices: [110, 120, 130],
        },
        HcurlOrder2PeriodicFacePair {
            master_face: 2,
            slave_face: 6,
            master_vertices: [10, 20, 30],
            slave_vertices: [120, 130, 110],
        },
    ];
    let constraints = expand_order2_hcurl_face_constraints(&pairs, 12, phase)
        .map_err(|error| error.to_string())?;
    if constraints.len() != 4 {
        return Err("expected two MPC rows per periodic face".into());
    }

    let system = MpcSystem::new(26, &constraints).map_err(|error| error.to_string())?;
    let mut reduced = vec![Complex64::new(0.0, 0.0); system.num_reduced_dofs()];
    for (index, &full_dof) in system.reduced_full_dofs().iter().enumerate() {
        reduced[index] = Complex64::new(full_dof as f64 + 0.25, -0.5);
    }
    let recovered = system
        .recover(&reduced)
        .map_err(|error| error.to_string())?;
    let mut recovery_checks = 0;
    for constraint in &constraints {
        let expected: Complex64 = constraint
            .masters
            .iter()
            .map(|term| term.coefficient * recovered[term.dof])
            .sum();
        if (recovered[constraint.slave] - expected).norm() >= 1.0e-12 {
            return Err("periodic face MPC recovery residual is nonzero".into());
        }
        recovery_checks += 1;
    }

    let mut rejection_checks = 0;
    if !matches!(
        expand_order2_hcurl_face_constraints(
            &[HcurlOrder2PeriodicFacePair {
                master_face: 0,
                slave_face: 1,
                master_vertices: [1, 1, 3],
                slave_vertices: [4, 5, 6],
            }],
            12,
            phase,
        ),
        Err(MpcError::InvalidFaceVertexMap { .. })
    ) {
        return Err("duplicate face vertex was accepted".into());
    }
    rejection_checks += 1;
    if !matches!(
        expand_order2_hcurl_face_constraints(&pairs, 12, Complex64::new(f64::NAN, 0.0)),
        Err(MpcError::NonFiniteCoefficient { .. })
    ) {
        return Err("non-finite face phase was accepted".into());
    }
    rejection_checks += 1;
    if !matches!(
        expand_order2_hcurl_face_constraints(
            &[HcurlOrder2PeriodicFacePair {
                master_face: usize::MAX,
                slave_face: 0,
                master_vertices: [1, 2, 3],
                slave_vertices: [4, 5, 6],
            }],
            usize::MAX,
            phase,
        ),
        Err(MpcError::DofIndexOverflow(_))
    ) {
        return Err("overflowing face DOF index was accepted".into());
    }
    rejection_checks += 1;

    Ok((constraints.len(), recovery_checks, rejection_checks))
}
