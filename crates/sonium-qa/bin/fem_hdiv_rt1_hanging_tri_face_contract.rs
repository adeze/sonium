//! Release-facing affine tetrahedral RT1 H(div) hanging-face contract.

use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hdiv_rt1_hanging_tri_face_constraints_3d,
    hdiv_rt1_hanging_tri_face_mpc_system_3d,
};
use sonium_fem::mesh::{ElementType, Mesh, Point, unit_cube_hexahedra, unit_cube_tetrahedra};
use sonium_fem::spaces::{HdivOrder, HdivSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hdiv-rt1-hanging-tri-face-contract.v1";

fn hanging_face_mesh(center_x: f64, center_y: f64) -> Mesh {
    let mut mesh = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(center_x, center_y, -1.0),
        Point::new_3d(center_x, center_y, 0.0),
        Point::new_3d(center_x, center_y, 1.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Tetrahedron, vec![0, 2, 1, 3]);
    mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 4, 5]);
    mesh.add_element(ElementType::Tetrahedron, vec![1, 2, 4, 5]);
    mesh.add_element(ElementType::Tetrahedron, vec![2, 0, 4, 5]);
    mesh
}

fn run() -> Result<(usize, usize, usize), String> {
    let mesh = hanging_face_mesh(1.0 / 3.0, 1.0 / 3.0);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let constraints = hdiv_rt1_hanging_tri_face_constraints_3d(&mesh, &space, 1.0e-10)
        .map_err(|error| error.to_string())?;
    if constraints.len() != 3 {
        return Err("expected three RT1 hanging-face restrictions".into());
    }
    let coarse_face = constraints[0].coarse_face;
    if constraints
        .iter()
        .any(|constraint| constraint.coarse_face != coarse_face)
        || constraints
            .iter()
            .any(|constraint| (constraint.signed_fraction.abs() - 1.0 / 3.0).abs() > 1.0e-12)
        || constraints
            .iter()
            .flat_map(|constraint| constraint.coefficients.into_iter().flatten())
            .any(|coefficient| !coefficient.is_finite())
    {
        return Err("RT1 triangular-face restriction geometry contract failed".into());
    }

    let mpc = hdiv_rt1_hanging_tri_face_mpc_system_3d(&mesh, &space, 1.0e-10)
        .map_err(|error| error.to_string())?;
    let coarse_values = [
        Complex64::new(2.0, -1.0),
        Complex64::new(-3.0, 0.5),
        Complex64::new(1.25, 2.0),
    ];
    let mut reduced = vec![Complex64::new(0.0, 0.0); mpc.num_reduced_dofs()];
    for (mode, value) in coarse_values.into_iter().enumerate() {
        let full_dof = coarse_face * 3 + mode;
        let reduced_dof = mpc
            .reduced_full_dofs()
            .iter()
            .position(|&dof| dof == full_dof)
            .ok_or_else(|| "coarse RT1 face mode was eliminated".to_string())?;
        reduced[reduced_dof] = value;
    }
    let recovered = mpc.recover(&reduced).map_err(|error| error.to_string())?;
    let recovery_checks = constraints
        .iter()
        .flat_map(|constraint| {
            let recovered = &recovered;
            (0..3).map(move |fine_mode| {
                let expected = (0..3)
                    .map(|coarse_mode| {
                        constraint.coefficients[fine_mode][coarse_mode] * coarse_values[coarse_mode]
                    })
                    .fold(Complex64::new(0.0, 0.0), |sum, value| sum + value);
                (recovered[constraint.slave_face * 3 + fine_mode] - expected).norm() <= 1.0e-11
            })
        })
        .filter(|passed| *passed)
        .count();
    if recovery_checks != 9 {
        return Err("RT1 triangular-face MPC recovery failed".into());
    }

    let mut rejection_checks = 0;
    if matches!(
        hdiv_rt1_hanging_tri_face_constraints_3d(&mesh, &HdivSpace::new(), 1.0e-10),
        Err(MixedHangingError::UnsupportedOrder(1))
    ) {
        rejection_checks += 1;
    }
    if matches!(
        hdiv_rt1_hanging_tri_face_constraints_3d(&mesh, &space, 0.0),
        Err(MixedHangingError::InvalidTolerance(0.0))
    ) {
        rejection_checks += 1;
    }
    if matches!(
        hdiv_rt1_hanging_tri_face_constraints_3d(&unit_cube_hexahedra(1), &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedFaceTopology)
    ) {
        rejection_checks += 1;
    }
    if !hdiv_rt1_hanging_tri_face_constraints_3d(&unit_cube_tetrahedra(1), &space, 1.0e-10)
        .map_err(|error| error.to_string())?
        .is_empty()
    {
        return Err("conforming tetrahedral mesh unexpectedly has hanging faces".into());
    }
    if rejection_checks != 3 {
        return Err("RT1 triangular-face unsupported-input rejection failed".into());
    }

    Ok((constraints.len(), recovery_checks, rejection_checks))
}

fn main() {
    let result = match run() {
        Ok((constraints, recovery_checks, rejection_checks)) => QaResult::passed(CASE_ID)
            .metric("hanging_face_constraints", constraints as f64)
            .metric("recovery_checks", recovery_checks as f64)
            .metric("rejection_checks", rejection_checks as f64)
            .dimension("3-D affine tetrahedral RT1 triangular-face normal-flux moments")
            .input_checksum("mesh:coarse-triangle-with-three-fine-subfaces:rt1-v1")
            .reference_checksum(
                "analytic/deterministic-fixture:fem-hdiv-rt1-hanging-tri-face-contract-v1",
            )
            .seed("fem-hdiv-rt1-hanging-tri-face-contract:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, &error),
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
