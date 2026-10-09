//! Release-facing RT0 H(div) hanging-edge trace contract.

use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hdiv_rt0_hanging_edge_constraints_2d, hdiv_rt0_hanging_edge_mpc_system_2d,
};
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::spaces::{HdivOrder, HdivSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hdiv-rt0-hanging-edge-contract.v1";

fn main() {
    let result = match run() {
        Ok((constraints, recovery_checks, rejection_checks)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("hanging_edge_constraints", constraints as f64)
            .minimum("hanging_edge_constraints", 2.0)
            .metric("recovery_checks", recovery_checks as f64)
            .minimum("recovery_checks", 2.0)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 3.0)
            .dimension("2-D affine RT0 H(div) edge normal-flux moments")
            .formulation("signed subedge length-fraction restriction of normal fluxes")
            .input_checksum("mesh:coarse-triangle-with-two-fine-subedges:v1")
            .reference_checksum(
                "analytic/deterministic-fixture:fem-hdiv-rt0-hanging-edge-contract-v1",
            )
            .seed("fem-hdiv-rt0-hanging-edge-contract:cpu:v1"),
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
    let space = HdivSpace::new();
    let constraints = hdiv_rt0_hanging_edge_constraints_2d(&mesh, &space, 1.0e-10)
        .map_err(|error| error.to_string())?;
    if constraints.len() != 2
        || !constraints
            .iter()
            .any(|constraint| (constraint.signed_fraction - 0.5).abs() < 1.0e-12)
        || !constraints
            .iter()
            .any(|constraint| (constraint.signed_fraction + 0.5).abs() < 1.0e-12)
    {
        return Err("expected one positive and one negative half-edge restriction".into());
    }

    let mpc = hdiv_rt0_hanging_edge_mpc_system_2d(&mesh, &space, 1.0e-10)
        .map_err(|error| error.to_string())?;
    let coarse_edge = constraints[0].coarse_edge;
    let coarse_reduced = mpc
        .reduced_full_dofs()
        .iter()
        .position(|&dof| dof == coarse_edge)
        .ok_or_else(|| "coarse edge was eliminated instead of retained".to_string())?;
    let mut reduced = vec![Complex64::new(0.0, 0.0); mpc.num_reduced_dofs()];
    reduced[coarse_reduced] = Complex64::new(4.0, 0.0);
    let recovered = mpc.recover(&reduced).map_err(|error| error.to_string())?;
    let recovery_checks = constraints
        .iter()
        .filter(|constraint| {
            let value = recovered[constraint.slave_edge];
            (value.re - 4.0 * constraint.signed_fraction).abs() < 1.0e-12
                && value.im.abs() < 1.0e-14
        })
        .count();
    if recovery_checks != constraints.len() {
        return Err("RT0 hanging-edge MPC recovery did not preserve signed fluxes".into());
    }

    let mut rejection_checks = 0;
    if matches!(
        hdiv_rt0_hanging_edge_constraints_2d(
            &mesh,
            &HdivSpace::with_order(HdivOrder::Rt1),
            1.0e-10,
        ),
        Err(MixedHangingError::UnsupportedOrder(2))
    ) {
        rejection_checks += 1;
    }
    if matches!(
        hdiv_rt0_hanging_edge_constraints_2d(&mesh, &space, 0.0),
        Err(MixedHangingError::InvalidTolerance(0.0))
    ) {
        rejection_checks += 1;
    }

    let mut three_d = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
    ] {
        three_d.add_node(point);
    }
    three_d.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    if matches!(
        hdiv_rt0_hanging_edge_constraints_2d(&three_d, &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedDimension(3))
    ) {
        rejection_checks += 1;
    }
    if rejection_checks != 3 {
        return Err(
            "RT0 hanging-edge input validation did not reject all unsupported cases".into(),
        );
    }

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
