//! CPU-only first-order H(div) periodic triangular-face contract.

use num_complex::Complex64;
use sonium_fem::boundary::periodic::{
    HdivRt1PeriodicFacePair, MpcError, MpcSystem, PeriodicTransform, SectorPeriodicMap,
    expand_hdiv_rt1_periodic_face_constraints,
};
use sonium_fem::mesh::Face;
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hdiv-rt1-periodic-face-contract.v1";

fn run() -> Result<(usize, usize, usize), String> {
    let phase = Complex64::new(0.0, 1.0);
    let pair = HdivRt1PeriodicFacePair {
        master_face: 0,
        slave_face: 1,
        vertex_permutation: [0, 2, 1],
    };
    let master_faces = [(0, Face::Triangle([0, 1, 2]))];
    let slave_faces = [(1, Face::Triangle([4, 3, 5]))];
    let map = SectorPeriodicMap {
        node_pairs: vec![(0, 3), (1, 4), (2, 5)],
        edge_pairs: vec![],
        face_pairs: vec![],
        transform: PeriodicTransform::IDENTITY,
        phase,
    };
    let constraints = map
        .hdiv_rt1_face_constraints_from_faces(&master_faces, &slave_faces)
        .map_err(|error| error.to_string())?;
    let mpc = MpcSystem::new(6, &constraints).map_err(|error| error.to_string())?;
    let mut reduced = vec![Complex64::new(0.0, 0.0); mpc.num_reduced_dofs()];
    for (index, &full_dof) in mpc.reduced_full_dofs().iter().enumerate() {
        reduced[index] = Complex64::new(full_dof as f64 + 0.125, -0.75);
    }
    let recovered = mpc.recover(&reduced).map_err(|error| error.to_string())?;
    let recovery_checks = constraints
        .iter()
        .filter(|constraint| {
            let expected: Complex64 = constraint
                .masters
                .iter()
                .map(|term| term.coefficient * recovered[term.dof])
                .sum();
            (recovered[constraint.slave] - expected).norm() < 1.0e-12
        })
        .count();
    if recovery_checks != 3 {
        return Err("RT1 periodic face recovery residual exceeded tolerance".into());
    }

    let mut rejection_checks = 0;
    if matches!(
        expand_hdiv_rt1_periodic_face_constraints(
            &[HdivRt1PeriodicFacePair {
                master_face: 0,
                slave_face: 1,
                vertex_permutation: [0, 0, 2],
            }],
            Complex64::new(1.0, 0.0),
        ),
        Err(MpcError::InvalidFaceVertexMap { .. })
    ) {
        rejection_checks += 1;
    }
    if matches!(
        expand_hdiv_rt1_periodic_face_constraints(&[pair], Complex64::new(f64::NAN, 0.0)),
        Err(MpcError::NonFiniteCoefficient { .. })
    ) {
        rejection_checks += 1;
    }
    if matches!(
        expand_hdiv_rt1_periodic_face_constraints(
            &[HdivRt1PeriodicFacePair {
                master_face: usize::MAX,
                slave_face: 1,
                vertex_permutation: [0, 1, 2],
            }],
            Complex64::new(1.0, 0.0),
        ),
        Err(MpcError::DofIndexOverflow(usize::MAX))
    ) {
        rejection_checks += 1;
    }
    Ok((constraints.len(), recovery_checks, rejection_checks))
}

fn main() {
    let result = match run() {
        Ok((constraint_count, recovery_checks, rejection_checks)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("periodic_face_mode_constraints", constraint_count as f64)
            .minimum("periodic_face_mode_constraints", 3.0)
            .metric("recovery_checks", recovery_checks as f64)
            .minimum("recovery_checks", 3.0)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 3.0)
            .dimension("affine 3-D triangular RT1 periodic faces")
            .formulation("barycentric face-mode permutation with complex Floquet phase")
            .input_checksum("periodic-face-modes:rt1-triangle-fixture:v1")
            .reference_checksum("analytic/deterministic-fixture:fem-hdiv-rt1-periodic-face-v1")
            .seed("fem-hdiv-rt1-periodic-face:cpu:v1"),
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
