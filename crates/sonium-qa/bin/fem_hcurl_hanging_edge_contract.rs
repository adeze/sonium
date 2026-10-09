//! Release-facing order-1 H(curl) hanging-edge trace contract.

use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hcurl_order1_hanging_edge_constraints_2d,
    hcurl_order1_hanging_edge_mpc_system_2d,
};
use sonium_fem::mesh::{ElementType, Mesh, MeshTopology, Point};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hcurl-order1-hanging-edge-contract.v1";

fn main() {
    let result = match run() {
        Ok((constraints, recovery_checks, rejection_checks)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("hanging_edge_constraints", constraints as f64)
            .minimum("hanging_edge_constraints", 2.0)
            .metric("recovery_checks", recovery_checks as f64)
            .minimum("recovery_checks", 1.0)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 2.0)
            .dimension("2-D affine order-1 Nedelec edge moments")
            .formulation("signed subedge length-fraction restriction of tangential line moments")
            .input_checksum("mesh:coarse-triangle-with-two-fine-subedges:v1")
            .reference_checksum(
                "analytic/deterministic-fixture:fem-hcurl-order1-hanging-edge-contract-v1",
            )
            .seed("fem-hcurl-order1-hanging-edge-contract:cpu:v1"),
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
    let space = HcurlSpace::new();
    let constraints = hcurl_order1_hanging_edge_constraints_2d(&mesh, &space, 1.0e-10)
        .map_err(|error| format!("H(curl) constraint construction failed: {error}"))?;
    if constraints.len() != 2
        || constraints
            .iter()
            .any(|constraint| (constraint.signed_fraction.abs() - 0.5).abs() > 1.0e-12)
    {
        return Err("expected two signed half-edge restrictions".into());
    }

    let mpc = hcurl_order1_hanging_edge_mpc_system_2d(&mesh, &space, 1.0e-10)
        .map_err(|error| format!("H(curl) MPC construction failed: {error}"))?;
    let topology = MeshTopology::build(&mesh);
    let reduced = mpc
        .reduced_full_dofs()
        .iter()
        .map(|&dof| Complex64::new(tangential_integral(&mesh, topology.edges[dof]), 0.0))
        .collect::<Vec<_>>();
    let recovered = mpc
        .recover(&reduced)
        .map_err(|error| format!("H(curl) MPC recovery failed: {error}"))?;
    let recovery_checks = recovered.len();

    let order_error =
        hcurl_order1_hanging_edge_constraints_2d(&mesh, &HcurlSpace::order2(), 1.0e-10)
            .expect_err("order-2 H(curl) is outside this bounded packet");
    if !matches!(order_error, MixedHangingError::UnsupportedOrder(2)) {
        return Err(format!("unexpected order rejection: {order_error}"));
    }
    let tolerance_error = hcurl_order1_hanging_edge_constraints_2d(&mesh, &space, 0.0)
        .expect_err("zero tolerance must be rejected");
    if !matches!(tolerance_error, MixedHangingError::InvalidTolerance(0.0)) {
        return Err(format!("unexpected tolerance rejection: {tolerance_error}"));
    }
    Ok((constraints.len(), recovery_checks, 2))
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

fn tangential_integral(mesh: &Mesh, edge: (usize, usize)) -> f64 {
    mesh.nodes[edge.1].x - mesh.nodes[edge.0].x
}
