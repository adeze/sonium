//! Release-facing vector H1 hanging-edge/face MPC contract.

use num_complex::Complex64;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::{
    HangingNodeError, MpcError, MpcSystem, h1_hanging_face_mpc_system_3d,
    h1_hanging_node_mpc_system_2d, vector_h1_hanging_face_mpc_system_3d,
    vector_h1_hanging_node_mpc_system_2d, vector_h1_hanging_quad_face_mpc_system_3d,
};
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.vector-h1-hanging-node-contract.v1";

struct Metrics {
    edge_constraints: usize,
    tetra_face_constraints: usize,
    hexa_face_constraints: usize,
    recovery_checks: usize,
    rejection_checks: usize,
}

fn main() {
    let result = match run() {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("edge_constraints", metrics.edge_constraints as f64)
            .minimum("edge_constraints", 1.0)
            .metric(
                "tetra_face_constraints",
                metrics.tetra_face_constraints as f64,
            )
            .minimum("tetra_face_constraints", 1.0)
            .metric(
                "hexa_face_constraints",
                metrics.hexa_face_constraints as f64,
            )
            .minimum("hexa_face_constraints", 1.0)
            .metric("recovery_checks", metrics.recovery_checks as f64)
            .minimum("recovery_checks", 3.0)
            .metric("rejection_checks", metrics.rejection_checks as f64)
            .minimum("rejection_checks", 1.0)
            .dimension("2-D vector H1 edges; 3-D tetrahedral and hexahedral vector H1 faces")
            .formulation(
                "component-preserving lift of scalar trace ownership into interleaved vector DOFs",
            )
            .input_checksum("mesh:vector-h1-coarse-edge-tri-face-hexa-face-patches:v1")
            .reference_checksum(
                "analytic/deterministic-fixture:fem-vector-h1-hanging-node-contract-v1",
            )
            .seed("fem-vector-h1-hanging-node-contract:cpu:v1"),
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

fn run() -> Result<Metrics, String> {
    let edge_mesh = hanging_edge_mesh();
    let edge_scalar = H1Space::new(PolynomialDegree::P2);
    let edge_scalar_mpc = h1_hanging_node_mpc_system_2d(&edge_mesh, &edge_scalar, 1.0e-10)
        .map_err(|error| format!("scalar edge contract failed: {error}"))?;
    let edge_space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let edge_mpc = vector_h1_hanging_node_mpc_system_2d(&edge_mesh, &edge_space, 1.0e-10)
        .map_err(|error| format!("vector edge contract failed: {error}"))?;
    let edge_constraints =
        assert_lift_and_recover(&edge_space, &edge_mesh, &edge_scalar_mpc, &edge_mpc, 2)?;

    let tetra_mesh = hanging_face_mesh();
    let tetra_scalar = H1Space::new(PolynomialDegree::P2);
    let tetra_scalar_mpc = h1_hanging_face_mpc_system_3d(&tetra_mesh, &tetra_scalar, 1.0e-10)
        .map_err(|error| format!("scalar tetrahedral-face contract failed: {error}"))?;
    let tetra_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let tetra_mpc = vector_h1_hanging_face_mpc_system_3d(&tetra_mesh, &tetra_space, 1.0e-10)
        .map_err(|error| format!("vector tetrahedral-face contract failed: {error}"))?;
    let tetra_face_constraints =
        assert_lift_and_recover(&tetra_space, &tetra_mesh, &tetra_scalar_mpc, &tetra_mpc, 3)?;

    let hexa_mesh = hanging_hexa_mesh();
    let hexa_scalar = H1Space::new(PolynomialDegree::P2);
    let hexa_scalar_mpc =
        sonium_fem::boundary::h1_hanging_quad_face_mpc_system_3d(&hexa_mesh, &hexa_scalar, 1.0e-10)
            .map_err(|error| format!("scalar hexahedral-face contract failed: {error}"))?;
    let hexa_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let hexa_mpc = vector_h1_hanging_quad_face_mpc_system_3d(&hexa_mesh, &hexa_space, 1.0e-10)
        .map_err(|error| format!("vector hexahedral-face contract failed: {error}"))?;
    let hexa_face_constraints =
        assert_lift_and_recover(&hexa_space, &hexa_mesh, &hexa_scalar_mpc, &hexa_mpc, 3)?;

    let zero_components = vector_h1_hanging_node_mpc_system_2d(
        &edge_mesh,
        &VectorH1Space::new(PolynomialDegree::P1, 0),
        1.0e-10,
    )
    .expect_err("zero-component vector H1 space must fail closed");
    if !matches!(
        zero_components,
        HangingNodeError::Mpc(MpcError::InvalidComponents(0))
    ) {
        return Err(format!(
            "unexpected zero-component error: {zero_components}"
        ));
    }

    Ok(Metrics {
        edge_constraints,
        tetra_face_constraints,
        hexa_face_constraints,
        recovery_checks: edge_mpc.num_full_dofs()
            + tetra_mpc.num_full_dofs()
            + hexa_mpc.num_full_dofs(),
        rejection_checks: 1,
    })
}

fn assert_lift_and_recover(
    space: &VectorH1Space,
    mesh: &Mesh,
    scalar_mpc: &MpcSystem,
    vector_mpc: &MpcSystem,
    components: usize,
) -> Result<usize, String> {
    let scalar_constraints = scalar_mpc.num_full_dofs() - scalar_mpc.num_reduced_dofs();
    let vector_constraints = vector_mpc.num_full_dofs() - vector_mpc.num_reduced_dofs();
    if scalar_constraints == 0 || vector_constraints != scalar_constraints * components {
        return Err(format!(
            "vector lift count mismatch: scalar={scalar_constraints}, vector={vector_constraints}, components={components}"
        ));
    }
    let coordinates = space
        .dof_coordinates(mesh)
        .ok_or_else(|| "vector H1 DOF coordinates unavailable".to_string())?;
    let reduced = vector_mpc
        .reduced_full_dofs()
        .iter()
        .map(|&dof| Complex64::new(vector_value(coordinates[dof], dof % components), 0.0))
        .collect::<Vec<_>>();
    let recovered = vector_mpc
        .recover(&reduced)
        .map_err(|error| format!("vector recovery failed: {error}"))?;
    for (dof, value) in recovered.iter().enumerate() {
        let expected = vector_value(coordinates[dof], dof % components);
        if (value.re - expected).abs() > 1.0e-10 || value.im.abs() > 1.0e-14 {
            return Err(format!("vector recovery mismatch at DOF {dof}"));
        }
    }
    Ok(vector_constraints)
}

fn vector_value(point: [f64; 3], component: usize) -> f64 {
    let x = point[0];
    let y = point[1];
    match component {
        0 => 1.0 + 2.0 * x + 0.5 * x * x,
        1 => -2.0 + 0.75 * x - y + 0.25 * x * x,
        _ => 3.0 - 0.5 * x + 1.25 * y + 0.125 * x * x,
    }
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

fn hanging_face_mesh() -> Mesh {
    let mut mesh = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
        Point::new_3d(0.5, 0.0, 0.0),
        Point::new_3d(0.5, 0.5, 0.0),
        Point::new_3d(0.0, 0.5, 0.0),
        Point::new_3d(0.1, 0.1, -1.0),
        Point::new_3d(0.6, 0.1, -1.0),
        Point::new_3d(0.1, 0.6, -1.0),
        Point::new_3d(0.3, 0.3, -1.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    mesh.add_element(ElementType::Tetrahedron, vec![0, 4, 6, 7]);
    mesh.add_element(ElementType::Tetrahedron, vec![4, 1, 5, 8]);
    mesh.add_element(ElementType::Tetrahedron, vec![6, 5, 2, 9]);
    mesh.add_element(ElementType::Tetrahedron, vec![4, 5, 6, 10]);
    mesh
}

fn hanging_hexa_mesh() -> Mesh {
    let mut mesh = Mesh::new(3);
    let mut nodes = [[[0usize; 3]; 3]; 3];
    for (z, z_value) in [-1.0, 0.0, 1.0].into_iter().enumerate() {
        for (y, y_value) in [0.0, 0.5, 1.0].into_iter().enumerate() {
            for (x, x_value) in [0.0, 0.5, 1.0].into_iter().enumerate() {
                nodes[z][y][x] = mesh.add_node(Point::new_3d(x_value, y_value, z_value));
            }
        }
    }
    let cube = |z0: usize, z1: usize, x0: usize, x1: usize, y0: usize, y1: usize| {
        vec![
            nodes[z0][y0][x0],
            nodes[z0][y0][x1],
            nodes[z0][y1][x1],
            nodes[z0][y1][x0],
            nodes[z1][y0][x0],
            nodes[z1][y0][x1],
            nodes[z1][y1][x1],
            nodes[z1][y1][x0],
        ]
    };
    mesh.add_element(ElementType::Hexahedron, cube(1, 2, 0, 2, 0, 2));
    for y in 0..2 {
        for x in 0..2 {
            mesh.add_element(ElementType::Hexahedron, cube(0, 1, x, x + 1, y, y + 1));
        }
    }
    mesh
}
