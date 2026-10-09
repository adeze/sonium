//! CPU-only topology-owned RT1 3-D multi-element numbering contract.

use num_complex::Complex64;
use sonium_fem::boundary::periodic::{PeriodicTransform, SectorPeriodicMap};
use sonium_fem::mesh::Face;
use sonium_fem::mesh::unit_cube_tetrahedra;
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "fem.hdiv-rt1-multielement-ownership.v1";

fn run() -> Result<(usize, usize, usize, usize, usize), String> {
    let mesh = unit_cube_tetrahedra(1);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let layout = space
        .rt1_dof_layout_3d(&mesh)
        .ok_or_else(|| "failed to construct checked RT1 3-D layout".to_string())?;

    if layout.num_dofs() != space.num_global_dofs(&mesh) {
        return Err("layout/global DOF count mismatch".to_string());
    }
    let shared_faces = layout
        .face_owners
        .iter()
        .filter(|owners| owners.len() == 2)
        .count();
    if shared_faces == 0 {
        return Err("fixture did not contain a shared face".to_string());
    }

    let mut face_range_checks = 0;
    for (face, owners) in layout.face_owners.iter().enumerate() {
        let range = layout
            .face_dofs(face)
            .ok_or_else(|| format!("missing range for face {face}"))?;
        if range.len() != 3 || !(1..=2).contains(&owners.len()) {
            return Err(format!("invalid face ownership/range for face {face}"));
        }
        for &element in owners {
            if !layout.element_dofs[element]
                .windows(range.len())
                .any(|window| window == (range.clone()).collect::<Vec<_>>().as_slice())
            {
                return Err(format!(
                    "face {face} not present in owner element {element}"
                ));
            }
            face_range_checks += 1;
        }
    }

    let mut cell_range_checks = 0;
    for (element, dofs) in layout.element_dofs.iter().enumerate() {
        if dofs.len() != 15 || layout.cell_dofs(element).map(|range| range.len()) != Some(3) {
            return Err(format!("invalid cell range for element {element}"));
        }
        cell_range_checks += 1;
    }

    let map = SectorPeriodicMap {
        node_pairs: vec![(0, 3), (1, 4), (2, 5)],
        edge_pairs: Vec::new(),
        face_pairs: Vec::new(),
        transform: PeriodicTransform::IDENTITY,
        phase: Complex64::new(1.0, 0.0),
    };
    let constraints = map
        .hdiv_rt1_mixed_face_constraints_from_layout(
            &[(0, Face::Triangle([0, 1, 2]))],
            &[(1, Face::Triangle([3, 4, 5]))],
            &layout,
        )
        .map_err(|error| error.to_string())?;
    if constraints.len() != 3 {
        return Err("layout-backed periodic consumer did not emit three modes".to_string());
    }

    Ok((
        layout.face_owners.len(),
        shared_faces,
        face_range_checks,
        cell_range_checks,
        constraints.len(),
    ))
}

fn main() {
    let result = match run() {
        Ok((faces, shared_faces, face_checks, cell_checks, consumer_checks)) => {
            QaResult::passed(CASE_ID)
                .metric("faces", faces as f64)
                .metric("shared_faces", shared_faces as f64)
                .metric("face_range_checks", face_checks as f64)
                .metric("cell_range_checks", cell_checks as f64)
                .metric("consumer_checks", consumer_checks as f64)
                .minimum("faces", 1.0)
                .minimum("shared_faces", 1.0)
                .minimum("face_range_checks", 1.0)
                .minimum("cell_range_checks", 1.0)
                .minimum("consumer_checks", 1.0)
                .input_checksum("mesh-generator:unit-cube-tetrahedra:n1:v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:fem-hdiv-rt1-multielement-ownership-v1",
                )
                .formulation("topology-owned RT1 face/cell numbering with shared-face ownership")
                .seed("fem-hdiv-rt1-multielement-ownership:cpu:v1")
        }
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
