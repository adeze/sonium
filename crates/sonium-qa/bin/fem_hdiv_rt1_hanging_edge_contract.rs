//! Release-facing affine triangular RT1 H(div) hanging-edge contract.

use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hdiv_rt1_hanging_edge_constraints_2d, hdiv_rt1_hanging_edge_mpc_system_2d,
};
use sonium_fem::mesh::{ElementType, Mesh, Point, unit_square_quads, unit_square_triangles};
use sonium_fem::spaces::{HdivOrder, HdivSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hdiv-rt1-hanging-edge-contract.v1";

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

fn run() -> Result<(usize, usize, usize), String> {
    let mesh = hanging_edge_mesh();
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let constraints = hdiv_rt1_hanging_edge_constraints_2d(&mesh, &space, 1.0e-10)
        .map_err(|error| error.to_string())?;
    if constraints.len() != 2 {
        return Err("expected two RT1 hanging-edge restrictions".into());
    }
    for constraint in &constraints {
        let b = constraint.signed_fraction;
        let t0 = 2.0 * constraint.fine_start_parameter - 1.0;
        let expected = [[b, 3.0 * b * t0], [0.0, b * b]];
        if constraint
            .coefficients
            .iter()
            .flatten()
            .zip(expected.iter().flatten())
            .any(|(actual, expected)| (actual - expected).abs() > 1.0e-12)
        {
            return Err("RT1 restriction matrix does not match the trace contract".into());
        }
    }

    let mpc = hdiv_rt1_hanging_edge_mpc_system_2d(&mesh, &space, 1.0e-10)
        .map_err(|error| error.to_string())?;
    let coarse_edge = constraints[0].coarse_edge;
    let coarse_mode_0 = coarse_edge * 2;
    let coarse_mode_1 = coarse_mode_0 + 1;
    let reduced_mode_0 = mpc
        .reduced_full_dofs()
        .iter()
        .position(|&dof| dof == coarse_mode_0)
        .ok_or_else(|| "coarse RT1 constant moment was eliminated".to_string())?;
    let reduced_mode_1 = mpc
        .reduced_full_dofs()
        .iter()
        .position(|&dof| dof == coarse_mode_1)
        .ok_or_else(|| "coarse RT1 linear moment was eliminated".to_string())?;
    let mut reduced = vec![Complex64::new(0.0, 0.0); mpc.num_reduced_dofs()];
    let coarse_constant = Complex64::new(2.0, -1.0);
    let coarse_linear = Complex64::new(-3.0, 0.5);
    reduced[reduced_mode_0] = coarse_constant;
    reduced[reduced_mode_1] = coarse_linear;
    let recovered = mpc.recover(&reduced).map_err(|error| error.to_string())?;
    let recovery_checks = constraints
        .iter()
        .filter(|constraint| {
            let b = constraint.signed_fraction;
            let t0 = 2.0 * constraint.fine_start_parameter - 1.0;
            let expected_0 = b * (coarse_constant + 3.0 * t0 * coarse_linear);
            let expected_1 = b * b * coarse_linear;
            (recovered[constraint.slave_edge * 2] - expected_0).norm() <= 1.0e-12
                && (recovered[constraint.slave_edge * 2 + 1] - expected_1).norm() <= 1.0e-12
        })
        .count();
    if recovery_checks != constraints.len() {
        return Err("RT1 hanging-edge MPC recovery failed".into());
    }

    let mut rejection_checks = 0;
    if matches!(
        hdiv_rt1_hanging_edge_constraints_2d(&unit_square_quads(1), &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedEdgeTopology)
    ) {
        rejection_checks += 1;
    }
    if matches!(
        hdiv_rt1_hanging_edge_constraints_2d(
            &unit_square_triangles(1),
            &HdivSpace::with_order(HdivOrder::Rt0),
            1.0e-10
        ),
        Err(MixedHangingError::UnsupportedOrder(1))
    ) {
        rejection_checks += 1;
    }
    if matches!(
        hdiv_rt1_hanging_edge_constraints_2d(&mesh, &space, 0.0),
        Err(MixedHangingError::InvalidTolerance(0.0))
    ) {
        rejection_checks += 1;
    }
    if rejection_checks != 3 {
        return Err("RT1 unsupported-input rejection contract failed".into());
    }

    Ok((constraints.len(), recovery_checks, rejection_checks))
}

fn main() {
    let result = match run() {
        Ok((constraints, recovery_checks, rejection_checks)) => QaResult::passed(CASE_ID)
            .metric("constraints", constraints as f64)
            .metric("recovery_checks", recovery_checks as f64)
            .metric("rejection_checks", rejection_checks as f64)
            .dimension("2-D affine RT1 triangular normal-flux edge moments")
            .input_checksum("mesh:coarse-triangle-with-two-fine-subedges:v1")
            .reference_checksum(
                "analytic/deterministic-fixture:fem-hdiv-rt1-hanging-edge-contract-v1",
            )
            .seed("fem-hdiv-rt1-hanging-edge-contract:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, &error),
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
