//! Release-facing affine H1 P1/P2/P3 hanging-node MPC contract.

use num_complex::Complex64;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::{
    H1HangingFaceConstraint, H1HangingNodeConstraint, HangingNodeError,
    h1_hanging_face_constraints_3d, h1_hanging_face_mpc_system_3d, h1_hanging_node_constraints_2d,
    h1_hanging_node_mpc_system_2d,
};
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.higher-order-hanging-node-contract.v1";

struct Metrics {
    p1_checks: usize,
    p2_checks: usize,
    p3_checks: usize,
    quadrilateral_checks: usize,
    tetrahedral_face_checks: usize,
    recovery_checks: usize,
    rejection_checks: usize,
}

fn main() {
    let result = match run() {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("p1_checks", metrics.p1_checks as f64)
            .minimum("p1_checks", 1.0)
            .metric("p2_checks", metrics.p2_checks as f64)
            .minimum("p2_checks", 1.0)
            .metric("p3_checks", metrics.p3_checks as f64)
            .minimum("p3_checks", 1.0)
            .metric("quadrilateral_checks", metrics.quadrilateral_checks as f64)
            .minimum("quadrilateral_checks", 2.0)
            .metric(
                "tetrahedral_face_checks",
                metrics.tetrahedral_face_checks as f64,
            )
            .minimum("tetrahedral_face_checks", 3.0)
            .metric("recovery_checks", metrics.recovery_checks as f64)
            .minimum("recovery_checks", 6.0)
            .metric("rejection_checks", metrics.rejection_checks as f64)
            .minimum("rejection_checks", 5.0)
            .reference("independent affine H1 Lagrange edge/face interpolation and MPC recovery")
            .reference_checksum(
                "analytic/deterministic-fixture:fem-higher-order-hanging-node-contract-v1",
            )
            .dimension("2-D affine triangles/quadrilaterals and 3-D affine tetrahedra")
            .formulation("u_slave = sum_i L_i u_coarse_trace_i; u_full = P u_reduced")
            .tolerance(
                "geometric admission tolerance 1e-10; exact trace partition and typed rejection",
            )
            .input_checksum("mesh:coarse-edge-and-triangular-face-with-p1-p2-p3-traces:v1"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };

    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("FEM higher-order hanging-node QA serialization failed: {error}");
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

fn one_hanging_face_mesh() -> Mesh {
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

    // The first tetrahedron owns the coarse face (0, 1, 2). Four tetrahedra
    // on the opposite side use its 1-to-4 triangular subdivision. Distinct
    // fine-side apexes keep the fixture a valid conforming tetrahedral patch.
    mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    mesh.add_element(ElementType::Tetrahedron, vec![0, 4, 6, 7]);
    mesh.add_element(ElementType::Tetrahedron, vec![4, 1, 5, 8]);
    mesh.add_element(ElementType::Tetrahedron, vec![6, 5, 2, 9]);
    mesh.add_element(ElementType::Tetrahedron, vec![4, 5, 6, 10]);
    mesh
}

fn polynomial_value(point: [f64; 3], degree: PolynomialDegree) -> f64 {
    let quadratic = if degree == PolynomialDegree::P1 {
        0.0
    } else {
        0.5 * point[0] * point[0]
    };
    let cubic = if degree == PolynomialDegree::P3 {
        0.25 * point[0] * point[0] * point[0]
    } else {
        0.0
    };
    1.25 + 2.0 * point[0] - 0.75 * point[1] + 0.5 * point[2] + quadratic + cubic
}

fn check_degree(
    mesh: &Mesh,
    degree: PolynomialDegree,
    expected_constraints: usize,
    edge: [usize; 2],
) -> Result<(usize, usize), String> {
    let space = H1Space::new(degree);
    let constraints = h1_hanging_node_constraints_2d(mesh, &space, 1.0e-10)
        .map_err(|error| format!("{degree:?} constraint detection failed: {error}"))?;
    if constraints.len() != expected_constraints {
        return Err(format!(
            "{degree:?} expected {expected_constraints} constraints, got {}",
            constraints.len()
        ));
    }
    for constraint in &constraints {
        check_constraint(constraint, degree, edge)?;
    }

    let mpc = h1_hanging_node_mpc_system_2d(mesh, &space, 1.0e-10)
        .map_err(|error| format!("{degree:?} MPC construction failed: {error}"))?;
    let coordinates = space
        .dof_coordinates(mesh)
        .ok_or_else(|| format!("{degree:?} did not provide DOF coordinates"))?;
    let reduced = mpc
        .reduced_full_dofs()
        .iter()
        .map(|&dof| Complex64::new(polynomial_value(coordinates[dof], degree), 0.0))
        .collect::<Vec<_>>();
    let recovered = mpc
        .recover(&reduced)
        .map_err(|error| format!("{degree:?} recovery failed: {error}"))?;
    for (dof, value) in recovered.iter().enumerate() {
        if (value.re - polynomial_value(coordinates[dof], degree)).abs() > 1.0e-10
            || value.im.abs() > 1.0e-14
        {
            return Err(format!("{degree:?} recovery mismatch at DOF {dof}"));
        }
    }
    Ok((constraints.len(), recovered.len()))
}

fn check_constraint(
    constraint: &H1HangingNodeConstraint,
    degree: PolynomialDegree,
    edge: [usize; 2],
) -> Result<(), String> {
    if constraint.edge != edge || !(0.0..1.0).contains(&constraint.parameter) {
        return Err(format!("{degree:?} constraint has invalid coarse edge"));
    }
    let expected_trace = match degree {
        PolynomialDegree::P1 => 2,
        PolynomialDegree::P2 => 3,
        PolynomialDegree::P3 => 4,
    };
    if constraint.masters.len() != expected_trace {
        return Err(format!(
            "{degree:?} trace has {} masters, expected {expected_trace}",
            constraint.masters.len()
        ));
    }
    let sum: f64 = constraint
        .masters
        .iter()
        .map(|term| term.coefficient.re)
        .sum();
    if (sum - 1.0).abs() > 1.0e-12 {
        return Err(format!("{degree:?} trace does not reproduce constants"));
    }
    Ok(())
}

fn check_face_degree(
    mesh: &Mesh,
    degree: PolynomialDegree,
    expected_constraints: usize,
) -> Result<(usize, usize), String> {
    let space = H1Space::new(degree);
    let constraints = h1_hanging_face_constraints_3d(mesh, &space, 1.0e-10)
        .map_err(|error| format!("{degree:?} face detection failed: {error}"))?;
    if constraints.len() != expected_constraints {
        return Err(format!(
            "{degree:?} expected {expected_constraints} face constraints, got {}",
            constraints.len()
        ));
    }
    for constraint in &constraints {
        check_face_constraint(constraint, degree)?;
    }

    let mpc = h1_hanging_face_mpc_system_3d(mesh, &space, 1.0e-10)
        .map_err(|error| format!("{degree:?} face MPC construction failed: {error}"))?;
    let coordinates = space
        .dof_coordinates(mesh)
        .ok_or_else(|| format!("{degree:?} did not provide 3-D DOF coordinates"))?;
    let reduced = mpc
        .reduced_full_dofs()
        .iter()
        .map(|&dof| Complex64::new(polynomial_value(coordinates[dof], degree), 0.0))
        .collect::<Vec<_>>();
    let recovered = mpc
        .recover(&reduced)
        .map_err(|error| format!("{degree:?} face recovery failed: {error}"))?;
    for (dof, value) in recovered.iter().enumerate() {
        if (value.re - polynomial_value(coordinates[dof], degree)).abs() > 1.0e-10
            || value.im.abs() > 1.0e-14
        {
            return Err(format!("{degree:?} face recovery mismatch at DOF {dof}"));
        }
    }
    Ok((constraints.len(), recovered.len()))
}

fn check_face_constraint(
    constraint: &H1HangingFaceConstraint,
    degree: PolynomialDegree,
) -> Result<(), String> {
    let mut face = constraint.face;
    face.sort_unstable();
    if face != [0, 1, 2]
        || !constraint.reference.iter().all(|value| value.is_finite())
        || constraint.reference.iter().any(|value| *value < -1.0e-10)
        || constraint.reference.iter().sum::<f64>() > 1.0 + 1.0e-10
    {
        return Err(format!("{degree:?} constraint has invalid coarse face"));
    }
    let expected_trace = match degree {
        PolynomialDegree::P1 => 3,
        PolynomialDegree::P2 => 6,
        PolynomialDegree::P3 => 10,
    };
    if constraint.masters.len() != expected_trace {
        return Err(format!(
            "{degree:?} face trace has {} masters, expected {expected_trace}",
            constraint.masters.len()
        ));
    }
    let sum: f64 = constraint
        .masters
        .iter()
        .map(|term| term.coefficient.re)
        .sum();
    if (sum - 1.0).abs() > 1.0e-10 {
        return Err(format!(
            "{degree:?} face trace does not reproduce constants"
        ));
    }
    Ok(())
}

fn run() -> Result<Metrics, String> {
    let mesh = one_hanging_node_mesh();
    let (p1_checks, p1_recovery) = check_degree(&mesh, PolynomialDegree::P1, 1, [0, 1])?;
    let (p2_checks, p2_recovery) = check_degree(&mesh, PolynomialDegree::P2, 3, [0, 1])?;
    let (p3_checks, p3_recovery) = check_degree(&mesh, PolynomialDegree::P3, 5, [0, 1])?;

    let quad = {
        let mut mesh = Mesh::new(2);
        for point in [
            Point::new_2d(0.0, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(1.0, 1.0),
            Point::new_2d(0.0, 1.0),
            Point::new_2d(0.5, 0.0),
            Point::new_2d(0.5, -1.0),
            Point::new_2d(0.0, -1.0),
            Point::new_2d(1.0, -1.0),
        ] {
            mesh.add_node(point);
        }
        mesh.add_element(ElementType::Quadrilateral, vec![0, 1, 2, 3]);
        mesh.add_element(ElementType::Quadrilateral, vec![0, 4, 5, 6]);
        mesh.add_element(ElementType::Quadrilateral, vec![4, 1, 7, 5]);
        mesh
    };
    let (quad_p2_checks, quad_p2_recovery) = check_degree(&quad, PolynomialDegree::P2, 3, [0, 1])?;
    let (quad_p3_checks, quad_p3_recovery) = check_degree(&quad, PolynomialDegree::P3, 5, [0, 1])?;

    let tetra = one_hanging_face_mesh();
    let (tetra_p1_checks, tetra_p1_recovery) = check_face_degree(&tetra, PolynomialDegree::P1, 3)?;
    let (tetra_p2_checks, tetra_p2_recovery) = check_face_degree(&tetra, PolynomialDegree::P2, 12)?;
    let (tetra_p3_checks, tetra_p3_recovery) = check_face_degree(&tetra, PolynomialDegree::P3, 25)?;

    let mut explicit = Mesh::new(2);
    for point in [
        Point::new_2d(0.0, 0.0),
        Point::new_2d(1.0, 0.0),
        Point::new_2d(0.0, 1.0),
        Point::new_2d(0.5, 0.0),
        Point::new_2d(0.5, 0.5),
        Point::new_2d(0.0, 0.5),
    ] {
        explicit.add_node(point);
    }
    explicit.add_element(ElementType::Triangle, vec![0, 1, 2, 3, 4, 5]);
    let p2 = H1Space::new(PolynomialDegree::P2);
    let mut rejection_checks = 0;
    if !matches!(
        h1_hanging_node_constraints_2d(&explicit, &p2, 1.0e-10),
        Err(HangingNodeError::UnsupportedGeometry { .. })
    ) {
        return Err("explicit high-order geometry was not rejected".into());
    }
    rejection_checks += 1;
    if !matches!(
        h1_hanging_node_constraints_2d(&mesh, &p2, 0.0),
        Err(HangingNodeError::InvalidTolerance(0.0))
    ) {
        return Err("invalid tolerance was not rejected".into());
    }
    rejection_checks += 1;
    let mut three_d = Mesh::new(3);
    three_d.add_node(Point::new_3d(0.0, 0.0, 0.0));
    if !matches!(
        h1_hanging_node_constraints_2d(&three_d, &p2, 1.0e-10),
        Err(HangingNodeError::UnsupportedDimension(3))
    ) {
        return Err("3-D input was not rejected by the 2-D contract".into());
    }
    rejection_checks += 1;

    let mut hexa = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(1.0, 1.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
        Point::new_3d(1.0, 0.0, 1.0),
        Point::new_3d(1.0, 1.0, 1.0),
        Point::new_3d(0.0, 1.0, 1.0),
    ] {
        hexa.add_node(point);
    }
    hexa.add_element(ElementType::Hexahedron, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    if !matches!(
        h1_hanging_face_constraints_3d(&hexa, &p2, 1.0e-10),
        Err(HangingNodeError::UnsupportedH1Element { .. })
    ) {
        return Err("hexahedral face trace was not rejected".into());
    }
    rejection_checks += 1;
    let mut degenerate_face = one_hanging_face_mesh();
    degenerate_face.nodes[2] = degenerate_face.nodes[0];
    if !matches!(
        h1_hanging_face_constraints_3d(&degenerate_face, &p2, 1.0e-10),
        Err(HangingNodeError::DegenerateFace { .. })
    ) {
        return Err("degenerate triangular face was not rejected".into());
    }
    rejection_checks += 1;

    Ok(Metrics {
        p1_checks,
        p2_checks,
        p3_checks,
        quadrilateral_checks: quad_p2_checks + quad_p3_checks,
        tetrahedral_face_checks: tetra_p1_checks + tetra_p2_checks + tetra_p3_checks,
        recovery_checks: p1_recovery
            + p2_recovery
            + p3_recovery
            + quad_p2_recovery
            + quad_p3_recovery
            + tetra_p1_recovery
            + tetra_p2_recovery
            + tetra_p3_recovery,
        rejection_checks,
    })
}
