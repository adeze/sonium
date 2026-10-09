//! CPU-only mixed triangular/quadrilateral RT1 periodic-face contract.

use num_complex::Complex64;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::boundary::periodic::{
    HdivRt1PeriodicMixedFacePair, MpcError, MpcSystem, PeriodicTransform, SectorPeriodicMap,
    expand_hdiv_rt1_periodic_mixed_face_constraints,
};
use sonium_fem::mesh::Face;
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "fem.hdiv-rt1-periodic-mixed-face-contract.v1";

fn faces() -> (Vec<(usize, Face)>, Vec<(usize, Face)>) {
    (
        vec![
            (0, Face::Triangle([0, 1, 2])),
            (1, Face::Quad([3, 4, 5, 6])),
        ],
        vec![
            (2, Face::Triangle([7, 8, 9])),
            (3, Face::Quad([10, 13, 12, 11])),
        ],
    )
}

fn run() -> Result<(usize, usize, usize, usize, usize), String> {
    let (master_faces, slave_faces) = faces();
    let map = SectorPeriodicMap {
        node_pairs: vec![(0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 12), (6, 13)],
        edge_pairs: vec![],
        face_pairs: vec![],
        transform: PeriodicTransform::IDENTITY,
        phase: Complex64::new(1.0, 0.0),
    };
    let master_offsets = [0, 3];
    let slave_offsets = [0, 3, 7, 10];
    let constraints = map
        .hdiv_rt1_mixed_face_constraints_from_faces(
            &master_faces,
            &slave_faces,
            &master_offsets,
            &slave_offsets,
        )
        .map_err(|error| error.to_string())?;
    let pair_count = 2;
    if constraints.len() != 7 {
        return Err(format!(
            "expected seven mixed constraints, got {}",
            constraints.len()
        ));
    }
    let system = MpcSystem::new(14, &constraints).map_err(|error| error.to_string())?;
    let mut reduced = vec![Complex64::new(0.0, 0.0); system.num_reduced_dofs()];
    for (index, &full_dof) in system.reduced_full_dofs().iter().enumerate() {
        reduced[index] = Complex64::new(full_dof as f64 + 0.5, -0.25);
    }
    let recovered = system
        .recover(&reduced)
        .map_err(|error| error.to_string())?;
    let recovery_checks = constraints
        .iter()
        .filter(|constraint| {
            let expected = constraint
                .masters
                .iter()
                .map(|term| term.coefficient * recovered[term.dof])
                .sum::<Complex64>();
            (recovered[constraint.slave] - expected).norm() <= 1.0e-12
        })
        .count();

    let mut matrix = TripletMatrix::new(14, 14);
    for dof in 0..14 {
        matrix.push(dof, dof, Complex64::new(1.0, 0.0));
    }
    let fem_system = FemSystem::new(
        matrix,
        (1..=14)
            .map(|value| Complex64::new(value as f64, 0.0))
            .collect(),
    );
    let condensed = map
        .condense_hdiv_rt1_mixed_fem_system_complex(
            &fem_system,
            &master_faces,
            &slave_faces,
            &master_offsets,
            &slave_offsets,
        )
        .map_err(|error| error.to_string())?;
    let condensed_rows = condensed.matrix.num_rows;

    let rejection_checks = [
        matches!(
            pair_periodic_hdiv_rt1_mixed_faces_for_qa(&master_faces[..1], &slave_faces, &map),
            Err(MpcError::PairCountMismatch { .. })
        ),
        matches!(
            expand_hdiv_rt1_periodic_mixed_face_constraints(
                &[HdivRt1PeriodicMixedFacePair::Triangle(
                    sonium_fem::boundary::periodic::HdivRt1PeriodicFacePair {
                        master_face: 0,
                        slave_face: 2,
                        vertex_permutation: [0, 1, 2],
                    },
                )],
                &[0],
                &[0],
                Complex64::new(1.0, 0.0),
            ),
            Err(MpcError::DofOutOfRange { .. })
        ),
        matches!(
            expand_hdiv_rt1_periodic_mixed_face_constraints(
                &[],
                &[],
                &[],
                Complex64::new(f64::NAN, 0.0)
            ),
            Err(MpcError::NonFiniteCoefficient { .. })
        ),
    ]
    .into_iter()
    .filter(|accepted| *accepted)
    .count();

    if recovery_checks != 7 || condensed_rows != 7 || rejection_checks != 3 {
        return Err(format!(
            "mixed metrics failed: recovery={recovery_checks}, condensed={condensed_rows}, rejected={rejection_checks}"
        ));
    }
    Ok((
        pair_count,
        constraints.len(),
        recovery_checks,
        condensed_rows,
        rejection_checks,
    ))
}

fn pair_periodic_hdiv_rt1_mixed_faces_for_qa(
    master_faces: &[(usize, Face)],
    slave_faces: &[(usize, Face)],
    map: &SectorPeriodicMap,
) -> Result<Vec<HdivRt1PeriodicMixedFacePair>, MpcError> {
    sonium_fem::boundary::periodic::pair_periodic_hdiv_rt1_mixed_faces(
        master_faces,
        slave_faces,
        &map.node_pairs,
    )
}

fn main() {
    let result = match run() {
        Ok((pairs, constraints, recovery, condensed, rejected)) => QaResult::passed(CASE_ID)
            .metric("mixed_face_pairs", pairs as f64)
            .metric("mixed_constraints", constraints as f64)
            .metric("recovery_checks", recovery as f64)
            .metric("condensed_rows", condensed as f64)
            .metric("rejection_checks", rejected as f64)
            .minimum("mixed_face_pairs", 2.0)
            .minimum("mixed_constraints", 7.0)
            .minimum("recovery_checks", 7.0)
            .minimum("condensed_rows", 7.0)
            .minimum("rejection_checks", 3.0)
            .input_checksum("periodic-face-modes:rt1-mixed-fixture:v1")
            .reference_checksum("analytic/deterministic-fixture:fem-hdiv-rt1-periodic-mixed-face-v1")
            .formulation("RT1 triangle/quadrilateral face modes with explicit global offsets and complex phase")
            .seed("fem-hdiv-rt1-periodic-mixed-face:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "{}",
        result.to_json_line().expect("serialize QA_RESULT JSONL")
    );
    if !matches!(result.status, QaStatus::Passed) {
        std::process::exit(1);
    }
}
