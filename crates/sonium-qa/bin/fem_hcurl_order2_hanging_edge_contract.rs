//! Release-facing order-2 H(curl) hanging-edge trace contract.

use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hcurl_order2_hanging_edge_constraints_2d,
    hcurl_order2_hanging_edge_mpc_system_2d,
};
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hcurl-order2-hanging-edge-contract.v1";

fn main() {
    let result = match run() {
        Ok((constraints, recovery_checks, rejection_checks)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("hanging_edge_constraints", constraints as f64)
            .minimum("hanging_edge_constraints", 2.0)
            .metric("recovery_checks", recovery_checks as f64)
            .minimum("recovery_checks", 4.0)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 4.0)
            .dimension("2-D affine order-2 Nedelec two-mode edge traces")
            .formulation("signed 2x2 subedge trace restriction with checked MPC recovery")
            .input_checksum("mesh:coarse-triangle-with-two-fine-subedges:v1")
            .reference_checksum(
                "analytic/deterministic-fixture:fem-hcurl-order2-hanging-edge-contract-v1",
            )
            .seed("fem-hcurl-order2-hanging-edge-contract:cpu:v1"),
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
    let mesh = hanging_edge_mesh();
    let space = HcurlSpace::order2();
    let constraints = hcurl_order2_hanging_edge_constraints_2d(&mesh, &space, 1.0e-10)
        .map_err(|error| error.to_string())?;
    if constraints.len() != 2 {
        return Err(format!(
            "expected two hanging edges, got {}",
            constraints.len()
        ));
    }
    if !constraints.iter().all(|constraint| {
        (constraint.signed_fraction.abs() - 0.5).abs() < 1.0e-12
            && constraint.coefficients[1][0].abs() < 1.0e-14
            && (constraint.coefficients[1][1] - 0.25).abs() < 1.0e-12
            && (constraint.coefficients[0][1] - 0.25).abs() < 1.0e-12
    }) {
        return Err("order-2 trace matrix coefficients failed".into());
    }
    if !constraints
        .iter()
        .any(|constraint| constraint.signed_fraction < 0.0)
    {
        return Err("reverse fine-edge orientation was not exercised".into());
    }

    let mpc = hcurl_order2_hanging_edge_mpc_system_2d(&mesh, &space, 1.0e-10)
        .map_err(|error| error.to_string())?;
    let coarse_edge = constraints[0].coarse_edge;
    let coarse_values = [Complex64::new(1.25, -0.5), Complex64::new(-0.75, 0.25)];
    let mut reduced = vec![Complex64::new(0.0, 0.0); mpc.num_reduced_dofs()];
    for (reduced_dof, &full_dof) in mpc.reduced_full_dofs().iter().enumerate() {
        if full_dof == coarse_edge * 2 {
            reduced[reduced_dof] = coarse_values[0];
        } else if full_dof == coarse_edge * 2 + 1 {
            reduced[reduced_dof] = coarse_values[1];
        }
    }
    let recovered = mpc.recover(&reduced).map_err(|error| error.to_string())?;
    let mut recovery_checks = 0;
    for constraint in &constraints {
        let expected = [
            constraint.coefficients[0][0] * coarse_values[0]
                + constraint.coefficients[0][1] * coarse_values[1],
            constraint.coefficients[1][0] * coarse_values[0]
                + constraint.coefficients[1][1] * coarse_values[1],
        ];
        for mode in 0..2 {
            if (recovered[constraint.slave_edge * 2 + mode] - expected[mode]).norm() >= 1.0e-12 {
                return Err("MPC recovery mismatch".into());
            }
            recovery_checks += 1;
        }
    }

    let mut rejection_checks = 0;
    if !matches!(
        hcurl_order2_hanging_edge_constraints_2d(&mesh, &HcurlSpace::new(), 1.0e-10),
        Err(MixedHangingError::UnsupportedOrder(1))
    ) {
        return Err("order-1 admission was not rejected".into());
    }
    rejection_checks += 1;
    if !matches!(
        hcurl_order2_hanging_edge_constraints_2d(&mesh, &space, 0.0),
        Err(MixedHangingError::InvalidTolerance(0.0))
    ) {
        return Err("invalid tolerance was not rejected".into());
    }
    rejection_checks += 1;

    let mut mesh_3d = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
    ] {
        mesh_3d.add_node(point);
    }
    mesh_3d.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    if !matches!(
        hcurl_order2_hanging_edge_constraints_2d(&mesh_3d, &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedDimension(3))
    ) {
        return Err("3-D admission was not rejected".into());
    }
    rejection_checks += 1;

    let mut curved = mesh.clone();
    curved.add_node(Point::new_2d(0.25, 0.0));
    curved.add_node(Point::new_2d(0.75, 0.0));
    curved.elements[0].nodes = vec![0, 1, 2, 5, 6, 3];
    if !matches!(
        hcurl_order2_hanging_edge_constraints_2d(&curved, &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedGeometry)
    ) {
        return Err("curved admission was not rejected".into());
    }
    rejection_checks += 1;

    Ok((constraints.len(), recovery_checks, rejection_checks))
}

fn hanging_edge_mesh() -> Mesh {
    let mut mesh = Mesh::new(2);
    for point in [
        Point::new_2d(0.0, 0.0),
        Point::new_2d(1.0, 0.0),
        Point::new_2d(0.0, 1.0),
        Point::new_2d(0.5, 0.0),
        Point::new_2d(0.5, -1.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
    mesh.add_element(ElementType::Triangle, vec![0, 3, 4]);
    mesh.add_element(ElementType::Triangle, vec![3, 1, 4]);
    mesh
}
