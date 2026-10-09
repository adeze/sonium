//! Release-facing affine-hexahedral Q1/Q2/Q3 H1 hanging-face contract.

use num_complex::Complex64;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::{
    h1_hanging_quad_face_constraints_3d, h1_hanging_quad_face_mpc_system_3d,
};
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.higher-order-hexa-hanging-node-contract.v1";

struct Metrics {
    p1_constraints: usize,
    p2_constraints: usize,
    p3_constraints: usize,
    recovery_checks: usize,
    rejection_checks: usize,
}

fn main() {
    let result = match run() {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("p1_constraints", metrics.p1_constraints as f64)
            .minimum("p1_constraints", 5.0)
            .metric("p2_constraints", metrics.p2_constraints as f64)
            .minimum("p2_constraints", 16.0)
            .metric("p3_constraints", metrics.p3_constraints as f64)
            .minimum("p3_constraints", 33.0)
            .metric("recovery_checks", metrics.recovery_checks as f64)
            .minimum("recovery_checks", 3.0)
            .metric("rejection_checks", metrics.rejection_checks as f64)
            .minimum("rejection_checks", 2.0)
            .dimension("3-D affine hexahedral Q1/Q2/Q3 scalar H1 traces")
            .formulation("largest-containing-face ownership, tensor-product Lagrange MPC recovery")
            .input_checksum("mesh:coarse-hexahedron-with-four-fine-face-patches:v1")
            .reference_checksum(
                "analytic/deterministic-fixture:fem-higher-order-hexa-hanging-node-contract-v1",
            )
            .seed("fem-higher-order-hexa-hanging-node-contract:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, error),
    };

    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("valid QA result should serialize")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}

fn run() -> Result<Metrics, String> {
    let mesh = hanging_face_mesh();
    let mut constraint_counts = [0usize; 3];
    let mut recovery_checks = 0usize;

    for (slot, degree) in [
        PolynomialDegree::P1,
        PolynomialDegree::P2,
        PolynomialDegree::P3,
    ]
    .into_iter()
    .enumerate()
    {
        let space = H1Space::new(degree);
        let constraints = h1_hanging_quad_face_constraints_3d(&mesh, &space, 1.0e-10)
            .map_err(|error| format!("{degree:?} constraint construction failed: {error}"))?;
        let expected = [5, 16, 33][slot];
        if constraints.len() != expected {
            return Err(format!(
                "{degree:?} expected {expected} constraints, got {}",
                constraints.len()
            ));
        }
        let expected_masters = match degree {
            PolynomialDegree::P1 => 4,
            PolynomialDegree::P2 => 9,
            PolynomialDegree::P3 => 16,
        };
        for constraint in &constraints {
            if constraint.masters.len() != expected_masters {
                return Err(format!(
                    "{degree:?} expected {expected_masters} face masters, got {}",
                    constraint.masters.len()
                ));
            }
            let coefficient_sum: f64 = constraint
                .masters
                .iter()
                .map(|term| term.coefficient.re)
                .sum();
            if (coefficient_sum - 1.0).abs() > 1.0e-10 {
                return Err(format!(
                    "{degree:?} interpolation does not reproduce constants"
                ));
            }
        }

        let mpc = h1_hanging_quad_face_mpc_system_3d(&mesh, &space, 1.0e-10)
            .map_err(|error| format!("{degree:?} MPC construction failed: {error}"))?;
        let coordinates = space
            .dof_coordinates(&mesh)
            .ok_or_else(|| format!("{degree:?} did not provide DOF coordinates"))?;
        let reduced = mpc
            .reduced_full_dofs()
            .iter()
            .map(|&dof| Complex64::new(polynomial(coordinates[dof], degree), 0.0))
            .collect::<Vec<_>>();
        let recovered = mpc
            .recover(&reduced)
            .map_err(|error| format!("{degree:?} recovery failed: {error}"))?;
        for (dof, value) in recovered.iter().enumerate() {
            if (value.re - polynomial(coordinates[dof], degree)).abs() > 1.0e-9
                || value.im.abs() > 1.0e-14
            {
                return Err(format!("{degree:?} recovery mismatch at DOF {dof}"));
            }
            recovery_checks += 1;
        }
        constraint_counts[slot] = constraints.len();
    }

    let rejection_checks = check_rejections()?;
    Ok(Metrics {
        p1_constraints: constraint_counts[0],
        p2_constraints: constraint_counts[1],
        p3_constraints: constraint_counts[2],
        recovery_checks,
        rejection_checks,
    })
}

fn hanging_face_mesh() -> Mesh {
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

fn polynomial(point: [f64; 3], degree: PolynomialDegree) -> f64 {
    let mut value = 1.25 + 2.0 * point[0] - 0.75 * point[1] + 0.5 * point[2];
    if matches!(degree, PolynomialDegree::P2 | PolynomialDegree::P3) {
        value += 0.5 * point[0] * point[0];
    }
    if degree == PolynomialDegree::P3 {
        value += 0.25 * point[0] * point[0] * point[0];
    }
    value
}

fn check_rejections() -> Result<usize, String> {
    let mut tetra = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
    ] {
        tetra.add_node(point);
    }
    tetra.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    let error =
        h1_hanging_quad_face_constraints_3d(&tetra, &H1Space::new(PolynomialDegree::P1), 1.0e-10)
            .expect_err("tetrahedral input must be rejected");
    if !matches!(
        error,
        sonium_fem::boundary::HangingNodeError::UnsupportedH1Element { .. }
    ) {
        return Err(format!("unexpected tetrahedral rejection: {error}"));
    }

    let mut curved = hanging_face_mesh();
    curved.elements[0].nodes.extend([0; 19]);
    let error =
        h1_hanging_quad_face_constraints_3d(&curved, &H1Space::new(PolynomialDegree::P2), 1.0e-10)
            .expect_err("curved connectivity must be rejected");
    if !matches!(
        error,
        sonium_fem::boundary::HangingNodeError::UnsupportedGeometry { .. }
    ) {
        return Err(format!("unexpected curved-geometry rejection: {error}"));
    }
    Ok(2)
}
