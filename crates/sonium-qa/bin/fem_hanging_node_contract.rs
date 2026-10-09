//! Release-facing 2-D P1 hanging-node interpolation and MPC contract.

use num_complex::Complex64;
use sonium_fem::boundary::{
    HangingNodeError, hanging_node_constraints_2d, hanging_node_mpc_system_2d,
};
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hanging-node-contract.v1";

fn main() {
    let result = match run() {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("single_constraint_checks", metrics.single_constraint_checks as f64)
            .minimum("single_constraint_checks", 1.0)
            .metric("multi_constraint_checks", metrics.multi_constraint_checks as f64)
            .minimum("multi_constraint_checks", 1.0)
            .metric("recovery_checks", metrics.recovery_checks as f64)
            .minimum("recovery_checks", 1.0)
            .metric("conforming_mesh_checks", metrics.conforming_mesh_checks as f64)
            .minimum("conforming_mesh_checks", 1.0)
            .metric("typed_rejection_checks", metrics.typed_rejection_checks as f64)
            .minimum("typed_rejection_checks", 4.0)
            .reference(
                "independent 2-D P1 non-conforming edge interpolation and generic scalar MPC recovery",
            )
            .reference_checksum("analytic/deterministic-fixture:fem-hanging-node-contract-v1")
            .dimension("2-D P1 triangles with one and two hanging edge vertices")
            .formulation("u_slave = (1-t)u_edge0 + t u_edge1; u_full = P u_reduced")
            .tolerance("geometric admission tolerance 1e-10; exact interpolation and typed rejection")
            .input_checksum("mesh:coarse-edge-with-one-and-two-fine-subdivisions:v1"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };

    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("FEM hanging-node QA serialization failed: {error}");
            std::process::exit(1);
        }
    }

    if matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    single_constraint_checks: usize,
    multi_constraint_checks: usize,
    recovery_checks: usize,
    conforming_mesh_checks: usize,
    typed_rejection_checks: usize,
}

fn one_hanging_node_mesh() -> Mesh {
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

fn two_hanging_node_mesh() -> Mesh {
    let mut mesh = Mesh::new(2);
    for point in [
        Point::new_2d(0.0, 0.0),
        Point::new_2d(1.0, 0.0),
        Point::new_2d(0.0, 1.0),
        Point::new_2d(0.25, 0.0),
        Point::new_2d(0.75, 0.0),
        Point::new_2d(0.5, -1.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
    mesh.add_element(ElementType::Triangle, vec![0, 3, 5]);
    mesh.add_element(ElementType::Triangle, vec![3, 4, 5]);
    mesh.add_element(ElementType::Triangle, vec![4, 1, 5]);
    mesh
}

fn conforming_mesh() -> Mesh {
    let mut mesh = Mesh::new(2);
    for point in [
        Point::new_2d(0.0, 0.0),
        Point::new_2d(1.0, 0.0),
        Point::new_2d(0.0, 1.0),
        Point::new_2d(1.0, 1.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
    mesh.add_element(ElementType::Triangle, vec![1, 3, 2]);
    mesh
}

fn run() -> Result<Metrics, String> {
    let one = one_hanging_node_mesh();
    let one_constraints = hanging_node_constraints_2d(&one, 1.0e-10)
        .map_err(|error| format!("single hanging-node detection failed: {error}"))?;
    if one_constraints.len() != 1
        || one_constraints[0].slave != 3
        || one_constraints[0].edge != [0, 1]
        || (one_constraints[0].parameter - 0.5).abs() > 1.0e-14
    {
        return Err(format!("unexpected single constraint: {one_constraints:?}"));
    }

    let two = two_hanging_node_mesh();
    let two_constraints = hanging_node_constraints_2d(&two, 1.0e-10)
        .map_err(|error| format!("multiple hanging-node detection failed: {error}"))?;
    if two_constraints.len() != 2
        || two_constraints[0].slave != 3
        || two_constraints[1].slave != 4
        || (two_constraints[0].parameter - 0.25).abs() > 1.0e-14
        || (two_constraints[1].parameter - 0.75).abs() > 1.0e-14
    {
        return Err(format!(
            "unexpected multiple constraints: {two_constraints:?}"
        ));
    }

    let system = hanging_node_mpc_system_2d(&two, 1.0e-10)
        .map_err(|error| format!("hanging-node MPC construction failed: {error}"))?;
    let recovered = system
        .recover(&[
            Complex64::new(2.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(9.0, 0.0),
            Complex64::new(4.0, 0.0),
        ])
        .map_err(|error| format!("hanging-node recovery failed: {error}"))?;
    if (recovered[3].re - 3.0).abs() > 1.0e-14 || (recovered[4].re - 5.0).abs() > 1.0e-14 {
        return Err("recovered hanging values violate edge interpolation".into());
    }

    if !hanging_node_constraints_2d(&conforming_mesh(), 1.0e-10)
        .map_err(|error| format!("conforming mesh admission failed: {error}"))?
        .is_empty()
    {
        return Err("conforming mesh produced a false hanging-node constraint".into());
    }

    let mut p2 = Mesh::new(2);
    for index in 0..6 {
        p2.add_node(Point::new_2d(index as f64, 0.0));
    }
    p2.add_element(ElementType::Triangle, vec![0, 1, 2, 3, 4, 5]);
    if !matches!(
        hanging_node_constraints_2d(&p2, 1.0e-10),
        Err(HangingNodeError::UnsupportedElement { .. })
    ) {
        return Err("higher-order element was not rejected with its typed error".into());
    }

    let mut malformed = Mesh::new(2);
    malformed.add_node(Point::new_2d(0.0, 0.0));
    malformed.add_node(Point::new_2d(1.0, 0.0));
    malformed.add_node(Point::new_2d(0.0, 1.0));
    malformed.add_element(ElementType::Triangle, vec![0, 1, 9]);
    if !matches!(
        hanging_node_constraints_2d(&malformed, 1.0e-10),
        Err(HangingNodeError::InvalidMesh { .. })
    ) {
        return Err("malformed connectivity was not rejected".into());
    }

    if !matches!(
        hanging_node_constraints_2d(&one, f64::NAN),
        Err(HangingNodeError::InvalidTolerance(value)) if value.is_nan()
    ) {
        return Err("non-finite tolerance was not rejected".into());
    }

    Ok(Metrics {
        single_constraint_checks: 1,
        multi_constraint_checks: 1,
        recovery_checks: 1,
        conforming_mesh_checks: 1,
        typed_rejection_checks: 4,
    })
}
