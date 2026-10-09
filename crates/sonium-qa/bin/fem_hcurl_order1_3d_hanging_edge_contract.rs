//! Release-facing affine tetrahedral order-1 H(curl) hanging-edge contract.

use num_complex::Complex64;
use sonium_fem::boundary::mixed_hanging::{
    MixedHangingError, hcurl_order1_hanging_edge_constraints_3d,
    hcurl_order1_hanging_edge_mpc_system_3d,
};
use sonium_fem::mesh::{
    ElementType, Mesh, MeshTopology, Point, unit_cube_hexahedra, unit_cube_tetrahedra,
    unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hcurl-order1-3d-hanging-edge-contract.v1";

fn hanging_edge_mesh() -> Mesh {
    let mut mesh = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
        Point::new_3d(0.5, 0.0, 0.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    mesh.add_element(ElementType::Tetrahedron, vec![0, 4, 2, 3]);
    mesh.add_element(ElementType::Tetrahedron, vec![4, 1, 2, 3]);
    mesh
}

fn constant_line_integral(mesh: &Mesh, edge: (usize, usize), vector: [f64; 3]) -> f64 {
    let first = mesh.nodes[edge.0];
    let second = mesh.nodes[edge.1];
    vector[0] * (second.x - first.x)
        + vector[1] * (second.y - first.y)
        + vector[2] * (second.z - first.z)
}

fn run() -> Result<(usize, usize, usize), String> {
    let mesh = hanging_edge_mesh();
    let space = HcurlSpace::new();
    let constraints = hcurl_order1_hanging_edge_constraints_3d(&mesh, &space, 1.0e-10)
        .map_err(|error| error.to_string())?;
    if constraints.len() != 2 {
        return Err(format!(
            "expected two hanging edges, got {}",
            constraints.len()
        ));
    }

    let topology = MeshTopology::build(&mesh);
    let vector = [2.0, -1.0, 0.5];
    let mut moment_checks = 0;
    for constraint in &constraints {
        let coarse_integral =
            constant_line_integral(&mesh, topology.edges[constraint.coarse_edge], vector);
        let fine_integral =
            constant_line_integral(&mesh, topology.edges[constraint.slave_edge], vector);
        let expected = constraint.signed_fraction * coarse_integral;
        if (fine_integral - expected).abs() > 1.0e-12 {
            return Err(format!(
                "signed line-moment mismatch: got {fine_integral:.16e}, expected {expected:.16e}"
            ));
        }
        moment_checks += 1;
    }

    let mpc = hcurl_order1_hanging_edge_mpc_system_3d(&mesh, &space, 1.0e-10)
        .map_err(|error| error.to_string())?;
    let coarse_edge = constraints[0].coarse_edge;
    let reduced_dof = mpc
        .reduced_full_dofs()
        .iter()
        .position(|&dof| dof == coarse_edge)
        .ok_or_else(|| "coarse edge was unexpectedly eliminated".to_owned())?;
    let mut reduced = vec![Complex64::new(0.0, 0.0); mpc.num_reduced_dofs()];
    let coarse_value = Complex64::new(4.0, -0.75);
    reduced[reduced_dof] = coarse_value;
    let recovered = mpc.recover(&reduced).map_err(|error| error.to_string())?;
    let mut recovery_checks = 0;
    for constraint in &constraints {
        let expected = coarse_value * constraint.signed_fraction;
        if (recovered[constraint.slave_edge] - expected).norm() > 1.0e-12 {
            return Err("complex MPC recovery mismatch".to_owned());
        }
        recovery_checks += 1;
    }

    let mut rejection_checks = 0;
    if !matches!(
        hcurl_order1_hanging_edge_constraints_3d(&mesh, &HcurlSpace::order2(), 1.0e-10),
        Err(MixedHangingError::UnsupportedOrder(2))
    ) {
        return Err("order-2 input was not rejected".to_owned());
    }
    rejection_checks += 1;
    if !matches!(
        hcurl_order1_hanging_edge_constraints_3d(&mesh, &space, 0.0),
        Err(MixedHangingError::InvalidTolerance(value)) if value == 0.0
    ) {
        return Err("zero tolerance was not rejected".to_owned());
    }
    rejection_checks += 1;
    if !matches!(
        hcurl_order1_hanging_edge_constraints_3d(&unit_square_triangles(1), &space, 1.0e-10,),
        Err(MixedHangingError::UnsupportedDimension(2))
    ) {
        return Err("2-D input was not rejected".to_owned());
    }
    rejection_checks += 1;
    if !matches!(
        hcurl_order1_hanging_edge_constraints_3d(&unit_cube_hexahedra(1), &space, 1.0e-10,),
        Err(MixedHangingError::UnsupportedHcurlTopology)
    ) {
        return Err("hexahedral input was not rejected".to_owned());
    }
    rejection_checks += 1;

    let mut malformed = unit_cube_tetrahedra(1);
    malformed.elements[0].nodes.push(malformed.nodes.len());
    malformed.add_node(Point::new_3d(0.25, 0.25, 0.25));
    if !matches!(
        hcurl_order1_hanging_edge_constraints_3d(&malformed, &space, 1.0e-10),
        Err(MixedHangingError::UnsupportedGeometry)
    ) {
        return Err("non-affine tetrahedral topology was not rejected".to_owned());
    }
    rejection_checks += 1;

    Ok((moment_checks, recovery_checks, rejection_checks))
}

fn main() {
    let result = match run() {
        Ok((moment_checks, recovery_checks, rejection_checks)) => QaResult::passed(CASE_ID)
            .metric("signed_line_moment_checks", moment_checks as f64)
            .metric("recovery_checks", recovery_checks as f64)
            .metric("rejection_checks", rejection_checks as f64)
            .dimension("3-D affine tetrahedral order-1 H(curl) hanging edges")
            .input_checksum("mesh:coarse-edge-with-two-fine-subedges:order1-v1")
            .reference_checksum(
                "analytic/deterministic-fixture:fem-hcurl-order1-3d-hanging-edge-contract-v1",
            )
            .seed("fem-hcurl-order1-3d-hanging-edge-contract:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, &error),
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
