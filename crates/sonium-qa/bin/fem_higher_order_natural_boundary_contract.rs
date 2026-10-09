//! Release-facing P2 scalar natural-boundary integration contract.

use num_complex::Complex64;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::boundary::{
    BoundaryConditionError, FemSystem, try_apply_neumann_real_p2, try_apply_neumann_real_p3,
    try_apply_robin_complex_p2,
};
use sonium_fem::mesh::{BoundaryType, Mesh, Point, unit_cube_tetrahedra, unit_square_triangles};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.higher-order-natural-boundary-contract.v1";

fn main() {
    let result = match run() {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("p2_edge_neumann_checks", metrics.p2_edge_neumann_checks as f64)
            .minimum("p2_edge_neumann_checks", 1.0)
            .metric("p2_edge_robin_checks", metrics.p2_edge_robin_checks as f64)
            .minimum("p2_edge_robin_checks", 1.0)
            .metric("p2_face_neumann_checks", metrics.p2_face_neumann_checks as f64)
            .minimum("p2_face_neumann_checks", 1.0)
            .metric("p3_edge_checks", metrics.p3_edge_checks as f64)
            .minimum("p3_edge_checks", 1.0)
            .metric("p3_face_checks", metrics.p3_face_checks as f64)
            .minimum("p3_face_checks", 1.0)
            .metric("curved_trace_checks", metrics.curved_trace_checks as f64)
            .minimum("curved_trace_checks", 1.0)
            .metric("rollback_checks", metrics.rollback_checks as f64)
            .minimum("rollback_checks", 2.0)
            .reference(
                "independent quadratic Lagrange edge/face trace integration with isoparametric Jacobians",
            )
            .reference_checksum(
                "analytic/deterministic-fixture:fem-higher-order-natural-boundary-contract-v1",
            )
            .dimension("2-D/3-D P2 and P3 scalar edge/triangular-face traces")
            .formulation("Neumann ∫hφ; Robin α∫φiφj and ∫gφi with isoparametric trace geometry")
            .tolerance("constant resultants/integrals <= 1e-10; finite geometry; transactional rollback")
            .input_checksum("mesh:promoted-unit-square-and-unit-cube-p2-boundaries:v1"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };

    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("FEM higher-order natural-boundary QA serialization failed: {error}");
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
    p2_edge_neumann_checks: usize,
    p2_edge_robin_checks: usize,
    p2_face_neumann_checks: usize,
    p3_edge_checks: usize,
    p3_face_checks: usize,
    curved_trace_checks: usize,
    rollback_checks: usize,
}

fn mark_all_boundaries(mesh: &mut sonium_fem::mesh::Mesh) {
    for boundary in &mut mesh.boundaries {
        boundary.marker = 0;
    }
}

fn p3_edge_mesh() -> Mesh {
    let mut mesh = Mesh::new(2);
    for point in [
        Point::new_2d(0.0, 0.0),
        Point::new_2d(1.0, 0.0),
        Point::new_2d(1.0 / 3.0, 0.0),
        Point::new_2d(2.0 / 3.0, 0.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_boundary(vec![0, 1, 2, 3], BoundaryType::Neumann, 0, 0, 0);
    mesh
}

fn p3_face_mesh() -> Mesh {
    let mut mesh = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(1.0 / 3.0, 0.0, 0.0),
        Point::new_3d(2.0 / 3.0, 0.0, 0.0),
        Point::new_3d(2.0 / 3.0, 1.0 / 3.0, 0.0),
        Point::new_3d(1.0 / 3.0, 2.0 / 3.0, 0.0),
        Point::new_3d(0.0, 2.0 / 3.0, 0.0),
        Point::new_3d(0.0, 1.0 / 3.0, 0.0),
        Point::new_3d(1.0 / 3.0, 1.0 / 3.0, 0.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_boundary(
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        BoundaryType::Neumann,
        0,
        0,
        0,
    );
    mesh
}

fn run() -> Result<Metrics, String> {
    let mut square = unit_square_triangles(1).to_p2();
    mark_all_boundaries(&mut square);
    let mut edge_system = FemSystem::new(
        TripletMatrix::new(square.num_nodes(), square.num_nodes()),
        vec![0.0; square.num_nodes()],
    );
    try_apply_neumann_real_p2(&mut edge_system, &square, 0, &|_, _, _| 1.0)
        .map_err(|error| format!("P2 edge Neumann failed: {error}"))?;
    if (edge_system.rhs.iter().sum::<f64>() - 4.0).abs() > 1.0e-10 {
        return Err(format!(
            "P2 edge resultant was {}",
            edge_system.rhs.iter().sum::<f64>()
        ));
    }

    let mut robin = FemSystem::new(
        TripletMatrix::new(square.num_nodes(), square.num_nodes()),
        vec![Complex64::new(0.0, 0.0); square.num_nodes()],
    );
    try_apply_robin_complex_p2(
        &mut robin,
        &square,
        0,
        &|_, _, _| Complex64::new(1.0, 0.0),
        &|_, _, _| Complex64::new(2.0, 0.0),
    )
    .map_err(|error| format!("P2 edge Robin failed: {error}"))?;
    if (robin.rhs.iter().map(|value| value.re).sum::<f64>() - 8.0).abs() > 1.0e-10
        || (robin
            .matrix
            .values
            .iter()
            .map(|value| value.re)
            .sum::<f64>()
            - 4.0)
            .abs()
            > 1.0e-10
    {
        return Err("P2 edge Robin integral mismatch".into());
    }

    let mut cube = unit_cube_tetrahedra(1).to_p2();
    mark_all_boundaries(&mut cube);
    let mut face_system = FemSystem::new(
        TripletMatrix::new(cube.num_nodes(), cube.num_nodes()),
        vec![0.0; cube.num_nodes()],
    );
    try_apply_neumann_real_p2(&mut face_system, &cube, 0, &|_, _, _| 1.0)
        .map_err(|error| format!("P2 face Neumann failed: {error}"))?;
    if (face_system.rhs.iter().sum::<f64>() - 6.0).abs() > 1.0e-10 {
        return Err(format!(
            "P2 face resultant was {}",
            face_system.rhs.iter().sum::<f64>()
        ));
    }

    let p3_edge = p3_edge_mesh();
    let mut p3_edge_system = FemSystem::new(
        TripletMatrix::new(p3_edge.num_nodes(), p3_edge.num_nodes()),
        vec![0.0; p3_edge.num_nodes()],
    );
    try_apply_neumann_real_p3(&mut p3_edge_system, &p3_edge, 0, &|_, _, _| 1.0)
        .map_err(|error| format!("P3 edge Neumann failed: {error}"))?;
    if (p3_edge_system.rhs.iter().sum::<f64>() - 1.0).abs() > 1.0e-10 {
        return Err(format!(
            "P3 edge resultant was {}",
            p3_edge_system.rhs.iter().sum::<f64>()
        ));
    }

    let p3_face = p3_face_mesh();
    let mut p3_face_system = FemSystem::new(
        TripletMatrix::new(p3_face.num_nodes(), p3_face.num_nodes()),
        vec![0.0; p3_face.num_nodes()],
    );
    try_apply_neumann_real_p3(&mut p3_face_system, &p3_face, 0, &|_, _, _| 1.0)
        .map_err(|error| format!("P3 face Neumann failed: {error}"))?;
    if (p3_face_system.rhs.iter().sum::<f64>() - 0.5).abs() > 1.0e-10 {
        return Err(format!(
            "P3 face resultant was {}",
            p3_face_system.rhs.iter().sum::<f64>()
        ));
    }

    let midpoint = square
        .boundaries
        .iter()
        .find(|boundary| boundary.nodes.len() == 3)
        .ok_or_else(|| "promoted square has no quadratic boundary edge".to_string())?
        .nodes[2];
    square.nodes[midpoint].z = 0.2;
    let mut curved = FemSystem::new(
        TripletMatrix::new(square.num_nodes(), square.num_nodes()),
        vec![0.0; square.num_nodes()],
    );
    try_apply_neumann_real_p2(&mut curved, &square, 0, &|x, y, z| x + y + z)
        .map_err(|error| format!("curved P2 edge failed: {error}"))?;
    if !curved.rhs.iter().all(|value| value.is_finite()) {
        return Err("curved P2 edge produced a non-finite RHS".into());
    }

    let original_rhs = curved.rhs.clone();
    let original_nnz = curved.matrix.nnz();
    let error = try_apply_neumann_real_p2(&mut curved, &square, 0, &|_, _, _| f64::NAN);
    if !matches!(
        error,
        Err(BoundaryConditionError::NonFiniteBoundaryValue { .. })
    ) {
        return Err("non-finite P2 callback was not rejected".into());
    }
    if curved.rhs != original_rhs || curved.matrix.nnz() != original_nnz {
        return Err("P2 callback rejection changed the system".into());
    }

    let mut p1 = unit_square_triangles(1);
    mark_all_boundaries(&mut p1);
    let error = try_apply_neumann_real_p2(&mut curved, &p1, 0, &|_, _, _| 1.0);
    if !matches!(
        error,
        Err(BoundaryConditionError::UnsupportedBoundary { .. })
    ) {
        return Err("P1 trace was not rejected by P2 API".into());
    }
    if curved.rhs != original_rhs || curved.matrix.nnz() != original_nnz {
        return Err("P1 topology rejection changed the system".into());
    }

    Ok(Metrics {
        p2_edge_neumann_checks: 1,
        p2_edge_robin_checks: 1,
        p2_face_neumann_checks: 1,
        p3_edge_checks: 1,
        p3_face_checks: 1,
        curved_trace_checks: 1,
        rollback_checks: 2,
    })
}
