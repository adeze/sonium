//! Release-facing RT0 H(div) 3-D triangular-face hanging contract.

use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hdiv_rt0_hanging_tri_face_constraints_3d,
    hdiv_rt0_hanging_tri_face_mpc_system_3d,
};
use sonium_fem::mesh::{ElementType, Mesh, Point, unit_cube_hexahedra, unit_cube_tetrahedra};
use sonium_fem::spaces::{HdivOrder, HdivSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hdiv-rt0-hanging-tri-face-contract.v1";

fn main() {
    let result = match run() {
        Ok((constraints, recovery_checks, rejection_checks)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("hanging_face_constraints", constraints as f64)
            .minimum("hanging_face_constraints", 3.0)
            .metric("recovery_checks", recovery_checks as f64)
            .minimum("recovery_checks", 3.0)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 5.0)
            .dimension("3-D affine RT0 H(div) triangular-face normal-flux moments")
            .formulation("signed subface area-fraction restriction with checked MPC recovery")
            .input_checksum("mesh:coarse-triangle-with-three-fine-subfaces:v1")
            .reference_checksum(
                "analytic/deterministic-fixture:fem-hdiv-rt0-hanging-tri-face-contract-v1",
            )
            .seed("fem-hdiv-rt0-hanging-tri-face-contract:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, error),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}

fn run() -> Result<(usize, usize, usize), String> {
    let mesh = hanging_face_mesh();
    let space = HdivSpace::new();
    let constraints = hdiv_rt0_hanging_tri_face_constraints_3d(&mesh, &space, 1.0e-10)
        .map_err(|error| error.to_string())?;
    if constraints.len() != 3
        || constraints
            .iter()
            .any(|constraint| (constraint.signed_fraction.abs() - 1.0 / 3.0).abs() > 1.0e-12)
        || constraints
            .iter()
            .any(|constraint| constraint.coarse_face != constraints[0].coarse_face)
        || !constraints
            .iter()
            .any(|constraint| constraint.signed_fraction < 0.0)
    {
        return Err("expected three signed one-third subface restrictions".into());
    }

    let mpc = hdiv_rt0_hanging_tri_face_mpc_system_3d(&mesh, &space, 1.0e-10)
        .map_err(|error| error.to_string())?;
    let coarse_face = constraints[0].coarse_face;
    let coarse_reduced = mpc
        .reduced_full_dofs()
        .iter()
        .position(|&dof| dof == coarse_face)
        .ok_or_else(|| "coarse face was eliminated instead of retained".to_string())?;
    let mut reduced = vec![Complex64::new(0.0, 0.0); mpc.num_reduced_dofs()];
    reduced[coarse_reduced] = Complex64::new(6.0, 0.0);
    let recovered = mpc.recover(&reduced).map_err(|error| error.to_string())?;
    let recovery_checks = constraints
        .iter()
        .filter(|constraint| {
            let value = recovered[constraint.slave_face];
            (value.re - 6.0 * constraint.signed_fraction).abs() < 1.0e-12
                && value.im.abs() < 1.0e-14
        })
        .count();
    if recovery_checks != constraints.len() {
        return Err("3-D RT0 face MPC recovery did not preserve signed fluxes".into());
    }

    let mut rejection_checks = 0;
    if matches!(
        hdiv_rt0_hanging_tri_face_constraints_3d(
            &mesh,
            &HdivSpace::with_order(HdivOrder::Rt1),
            1.0e-10,
        ),
        Err(MixedHangingError::UnsupportedOrder(2))
    ) {
        rejection_checks += 1;
    }
    if matches!(
        hdiv_rt0_hanging_tri_face_constraints_3d(&mesh, &space, 0.0),
        Err(MixedHangingError::InvalidTolerance(0.0))
    ) {
        rejection_checks += 1;
    }

    let mut two_d = Mesh::new(2);
    two_d.add_node(Point::new_2d(0.0, 0.0));
    two_d.add_node(Point::new_2d(1.0, 0.0));
    two_d.add_node(Point::new_2d(0.0, 1.0));
    two_d.add_element(ElementType::Triangle, vec![0, 1, 2]);
    if matches!(
        hdiv_rt0_hanging_tri_face_constraints_3d(&two_d, &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedDimension(2))
    ) {
        rejection_checks += 1;
    }

    let mut curved = mesh.clone();
    curved.add_node(Point::new_3d(0.2, 0.2, -0.5));
    curved.elements[0].nodes.push(6);
    if matches!(
        hdiv_rt0_hanging_tri_face_constraints_3d(&curved, &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedGeometry)
    ) {
        rejection_checks += 1;
    }
    if matches!(
        hdiv_rt0_hanging_tri_face_constraints_3d(&unit_cube_hexahedra(1), &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedFaceTopology)
    ) {
        rejection_checks += 1;
    }
    if rejection_checks != 5 {
        return Err("3-D RT0 face validation did not reject all unsupported cases".into());
    }

    if !hdiv_rt0_hanging_tri_face_constraints_3d(&unit_cube_tetrahedra(1), &space, 1.0e-10)
        .map_err(|error| error.to_string())?
        .is_empty()
    {
        return Err("conforming tetrahedral faces must not create constraints".into());
    }

    Ok((constraints.len(), recovery_checks, rejection_checks))
}

fn hanging_face_mesh() -> Mesh {
    let mut mesh = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(1.0 / 3.0, 1.0 / 3.0, -1.0),
        Point::new_3d(1.0 / 3.0, 1.0 / 3.0, 0.0),
        Point::new_3d(1.0 / 3.0, 1.0 / 3.0, 1.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Tetrahedron, vec![0, 2, 1, 3]);
    mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 4, 5]);
    mesh.add_element(ElementType::Tetrahedron, vec![1, 2, 4, 5]);
    mesh.add_element(ElementType::Tetrahedron, vec![2, 0, 4, 5]);
    mesh
}
