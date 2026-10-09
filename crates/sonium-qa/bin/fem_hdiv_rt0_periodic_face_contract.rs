//! CPU-only lowest-order H(div) periodic face contract.

use num_complex::Complex64;
use sonium_fem::boundary::periodic::{
    MpcError, MpcSystem, PeriodicTransform, SectorPeriodicMap, pair_periodic_faces_oriented,
};
use sonium_fem::mesh::Face;
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hdiv-rt0-periodic-face-contract.v1";

fn run() -> Result<(usize, usize, usize), String> {
    let pairs = pair_periodic_faces_oriented(
        &[
            (40, Face::Triangle([0, 1, 2])),
            (41, Face::Triangle([6, 7, 8])),
        ],
        &[
            (50, Face::Triangle([3, 4, 5])),
            (51, Face::Triangle([9, 11, 10])),
        ],
        &[(0, 3), (1, 4), (2, 5), (6, 9), (7, 10), (8, 11)],
    )
    .map_err(|error| error.to_string())?;
    if pairs != vec![(40, 50, 1), (41, 51, -1)] {
        return Err(format!("unexpected oriented face pairs: {pairs:?}"));
    }

    let phase = Complex64::new(0.0, 1.0);
    let map = SectorPeriodicMap {
        node_pairs: vec![],
        edge_pairs: vec![],
        face_pairs: vec![(0, 1, 1), (2, 3, -1)],
        transform: PeriodicTransform::IDENTITY,
        phase,
    };
    let constraints = map
        .hdiv_rt0_face_constraints_3d()
        .map_err(|error| error.to_string())?;
    let mpc = MpcSystem::from_periodic(4, &constraints).map_err(|error| error.to_string())?;
    let mut reduced = vec![Complex64::new(0.0, 0.0); mpc.num_reduced_dofs()];
    for (index, &full_dof) in mpc.reduced_full_dofs().iter().enumerate() {
        reduced[index] = Complex64::new(full_dof as f64 + 0.25, -0.5);
    }
    let recovered = mpc.recover(&reduced).map_err(|error| error.to_string())?;
    let recovery_checks = constraints
        .iter()
        .filter(|constraint| constraint.residual(&recovered).norm() < 1.0e-12)
        .count();
    if recovery_checks != constraints.len() {
        return Err("periodic RT0 face recovery residual exceeded tolerance".into());
    }

    let mut rejection_checks = 0;
    if matches!(
        pair_periodic_faces_oriented(
            &[(0, Face::Triangle([0, 1, 2]))],
            &[(1, Face::Triangle([3, 4, 5]))],
            &[(0, 3), (1, 4)],
        ),
        Err(MpcError::UnpairedFaceNode(2))
    ) {
        rejection_checks += 1;
    }
    if matches!(
        pair_periodic_faces_oriented(
            &[(0, Face::Triangle([0, 1, 2]))],
            &[(1, Face::Triangle([3, 4, 5]))],
            &[(0, 3), (1, 4), (2, 6)],
        ),
        Err(MpcError::NoPeriodicFaceMatch { master_dof: 0 })
    ) {
        rejection_checks += 1;
    }
    if !matches!(
        pair_periodic_faces_oriented(
            &[(0, Face::Triangle([0, 1, 2]))],
            &[
                (1, Face::Triangle([3, 4, 5])),
                (2, Face::Triangle([3, 5, 4])),
            ],
            &[(0, 3), (1, 4), (2, 5)],
        ),
        Err(MpcError::PairCountMismatch { .. })
    ) {
        return Err("face-pair count mismatch was not rejected".into());
    }
    rejection_checks += 1;
    Ok((pairs.len(), recovery_checks, rejection_checks))
}

fn main() {
    let result = match run() {
        Ok((pair_count, recovery_checks, rejection_checks)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("oriented_face_pairs", pair_count as f64)
            .minimum("oriented_face_pairs", 2.0)
            .metric("recovery_checks", recovery_checks as f64)
            .minimum("recovery_checks", 2.0)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 3.0)
            .dimension("affine triangular RT0 periodic faces")
            .formulation("oriented global normal-flux moments with complex Floquet phase")
            .input_checksum("periodic-face-pairs:rt0-triangle-fixture:v1")
            .reference_checksum("analytic/deterministic-fixture:fem-hdiv-rt0-periodic-face-v1")
            .seed("fem-hdiv-rt0-periodic-face:cpu:v1"),
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
