//! CPU-only first-order H(div) periodic quadrilateral-face contract.

use num_complex::Complex64;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::boundary::periodic::{
    HdivRt1PeriodicQuadFacePair, MpcError, MpcSystem, PeriodicTransform, SectorPeriodicMap,
    expand_hdiv_rt1_periodic_quad_face_constraints,
};
use sonium_fem::mesh::Face;
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "fem.hdiv-rt1-periodic-quad-face-contract.v1";

fn run() -> Result<(usize, usize, usize, usize), String> {
    let phase = Complex64::new(0.6, 0.8);
    let master_faces = [(0, Face::Quad([0, 1, 2, 3]))];
    let slave_faces = [(1, Face::Quad([4, 7, 6, 5]))];
    let map = SectorPeriodicMap {
        node_pairs: vec![(0, 4), (1, 5), (2, 6), (3, 7)],
        edge_pairs: vec![],
        face_pairs: vec![],
        transform: PeriodicTransform::IDENTITY,
        phase,
    };
    let constraints = map
        .hdiv_rt1_quadrilateral_face_constraints_from_faces(&master_faces, &slave_faces)
        .map_err(|error| error.to_string())?;
    let offset_constraints = map
        .hdiv_rt1_quadrilateral_face_constraints_from_faces_with_offsets(
            &master_faces,
            &slave_faces,
            &[0],
            &[0, 4],
        )
        .map_err(|error| error.to_string())?;
    if offset_constraints.len() != 4 {
        return Err(format!(
            "expected four offset-aware constraints, got {}",
            offset_constraints.len()
        ));
    }
    let system = MpcSystem::new(8, &constraints).map_err(|error| error.to_string())?;
    let mut reduced = vec![Complex64::new(0.0, 0.0); system.num_reduced_dofs()];
    for (index, &full_dof) in system.reduced_full_dofs().iter().enumerate() {
        reduced[index] = Complex64::new(full_dof as f64 + 0.125, -0.75);
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

    let rejected = [
        matches!(
            expand_hdiv_rt1_periodic_quad_face_constraints(
                &[HdivRt1PeriodicQuadFacePair {
                    master_face: 0,
                    slave_face: 1,
                    vertex_permutation: [0, 0, 2, 3],
                }],
                Complex64::new(1.0, 0.0),
            ),
            Err(MpcError::InvalidFaceVertexMap { .. })
        ),
        matches!(
            expand_hdiv_rt1_periodic_quad_face_constraints(
                &[HdivRt1PeriodicQuadFacePair {
                    master_face: 0,
                    slave_face: 1,
                    vertex_permutation: [0, 2, 1, 3],
                }],
                Complex64::new(1.0, 0.0),
            ),
            Err(MpcError::InvalidFaceVertexMap { .. })
        ),
        matches!(
            expand_hdiv_rt1_periodic_quad_face_constraints(
                &[HdivRt1PeriodicQuadFacePair {
                    master_face: 0,
                    slave_face: 1,
                    vertex_permutation: [0, 1, 2, 3],
                }],
                Complex64::new(f64::NAN, 0.0),
            ),
            Err(MpcError::NonFiniteCoefficient { .. })
        ),
    ]
    .into_iter()
    .filter(|accepted| *accepted)
    .count();

    if constraints.len() != 4 {
        return Err(format!(
            "expected four quadrilateral face constraints, got {}",
            constraints.len()
        ));
    }
    if recovery_checks != 4 {
        return Err(format!(
            "expected four recovery checks, got {recovery_checks}"
        ));
    }

    let mut matrix = TripletMatrix::new(8, 8);
    for dof in 0..8 {
        matrix.push(dof, dof, Complex64::new(1.0, 0.0));
    }
    let system = FemSystem::new(
        matrix,
        (1..=8)
            .map(|value| Complex64::new(value as f64, 0.0))
            .collect(),
    );
    let condensed = map
        .condense_hdiv_rt1_quadrilateral_fem_system_complex(&system, &master_faces, &slave_faces)
        .map_err(|error| error.to_string())?;
    let condensed_rows = condensed.matrix.num_rows;
    if condensed_rows != 4 || condensed.matrix.num_cols != 4 {
        return Err(format!(
            "expected 4x4 condensed system, got {}x{}",
            condensed.matrix.num_rows, condensed.matrix.num_cols
        ));
    }
    if rejected != 3 {
        return Err(format!("expected three rejection checks, got {rejected}"));
    }
    Ok((constraints.len(), recovery_checks, rejected, condensed_rows))
}

fn main() {
    let result = match run() {
        Ok((constraints, recovery, rejections, condensed_rows)) => QaResult::passed(CASE_ID)
            .metric("quadrilateral_mode_constraints", constraints as f64)
            .metric("recovery_checks", recovery as f64)
            .metric("rejection_checks", rejections as f64)
            .metric("condensed_rows", condensed_rows as f64)
            .minimum("quadrilateral_mode_constraints", 4.0)
            .minimum("recovery_checks", 4.0)
            .minimum("rejection_checks", 3.0)
            .minimum("condensed_rows", 4.0)
            .input_checksum("periodic-face-modes:rt1-quadrilateral-fixture:v1")
            .reference_checksum("analytic/deterministic-fixture:fem-hdiv-rt1-periodic-quad-face-v1")
            .formulation(
                "RT1 quadrilateral-face square-symmetry mode map with complex Floquet phase",
            )
            .seed("fem-hdiv-rt1-periodic-quad-face:cpu:v1"),
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
