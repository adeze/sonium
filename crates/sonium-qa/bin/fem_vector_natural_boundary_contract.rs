//! Release-facing checked VectorH1 natural-traction contract.

use sonium_fem::assembly::TripletMatrix;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{BoundaryConditionError, FemSystem, try_apply_traction_vector};
use sonium_fem::mesh::{unit_cube_tetrahedra, unit_square_triangles};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "fem.vector-natural-boundary-contract.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    p1_edge_checks: usize,
    p2_face_checks: usize,
    rollback_checks: usize,
    topology_checks: usize,
}

fn mark_all_boundaries(mesh: &mut sonium_fem::mesh::Mesh, marker: i32) {
    for boundary in &mut mesh.boundaries {
        boundary.marker = marker;
    }
}

fn run() -> Result<Metrics, String> {
    let mut square = unit_square_triangles(1);
    square.detect_boundaries();
    mark_all_boundaries(&mut square, 3);
    let p1_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let p1_dofs = p1_space.num_global_dofs(&square);
    let mut p1 = FemSystem::new(TripletMatrix::new(p1_dofs, p1_dofs), vec![0.0; p1_dofs]);
    try_apply_traction_vector(&mut p1, &square, &p1_space, 3, &|_, _, _| vec![2.0, 3.0])
        .map_err(|error| format!("P1 vector edge traction failed: {error}"))?;
    let x_resultant: f64 = p1.rhs.iter().step_by(2).sum();
    let y_resultant: f64 = p1.rhs.iter().skip(1).step_by(2).sum();
    if (x_resultant - 8.0).abs() > 1.0e-11 || (y_resultant - 12.0).abs() > 1.0e-11 {
        return Err(format!(
            "P1 edge resultants were ({x_resultant}, {y_resultant}), expected (8, 12)"
        ));
    }

    let mut cube = unit_cube_tetrahedra(1);
    cube.detect_boundaries();
    mark_all_boundaries(&mut cube, 7);
    let p2_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let p2_dofs = p2_space.num_global_dofs(&cube);
    let mut p2 = FemSystem::new(TripletMatrix::new(p2_dofs, p2_dofs), vec![0.0; p2_dofs]);
    try_apply_traction_vector(&mut p2, &cube, &p2_space, 7, &|_, _, _| vec![2.0, 3.0, 5.0])
        .map_err(|error| format!("P2 vector face traction failed: {error}"))?;
    for (component, expected) in [12.0, 18.0, 30.0].into_iter().enumerate() {
        let resultant: f64 = p2.rhs.iter().skip(component).step_by(3).sum();
        if (resultant - expected).abs() > 1.0e-10 {
            return Err(format!(
                "P2 component {component} resultant was {resultant}, expected {expected}"
            ));
        }
    }
    if !p2.rhs[cube.num_nodes() * 3..]
        .iter()
        .any(|value| value.abs() > 1.0e-12)
    {
        return Err("P2 traction did not load edge midpoint DOFs".into());
    }

    let original_rhs = p1.rhs.clone();
    let short = try_apply_traction_vector(&mut p1, &square, &p1_space, 3, &|_, _, _| vec![1.0])
        .expect_err("short vector callback must fail");
    if !matches!(
        short,
        BoundaryConditionError::BoundaryValueDimension {
            expected: 2,
            actual: 1,
            ..
        }
    ) || p1.rhs != original_rhs
    {
        return Err("short vector callback was not rejected transactionally".into());
    }

    let non_finite = try_apply_traction_vector(&mut p1, &square, &p1_space, 3, &|_, _, _| {
        vec![f64::NAN, 0.0]
    })
    .expect_err("non-finite vector callback must fail");
    if !matches!(
        non_finite,
        BoundaryConditionError::NonFiniteBoundaryValue { .. }
    ) || p1.rhs != original_rhs
    {
        return Err("non-finite vector callback was not rejected transactionally".into());
    }

    let mut malformed = unit_square_triangles(1);
    malformed.detect_boundaries();
    let malformed_space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let malformed_dofs = malformed_space.num_global_dofs(&malformed);
    let last_node = malformed.num_nodes() - 1;
    let boundary = malformed
        .boundaries
        .iter_mut()
        .find(|boundary| boundary.nodes.len() == 2)
        .ok_or_else(|| "unit square has no edge boundary".to_string())?;
    boundary.marker = 9;
    boundary.nodes = vec![0, last_node];
    let mut malformed_system = FemSystem::new(
        TripletMatrix::new(malformed_dofs, malformed_dofs),
        vec![0.0; malformed_dofs],
    );
    let topology_error = try_apply_traction_vector(
        &mut malformed_system,
        &malformed,
        &malformed_space,
        9,
        &|_, _, _| vec![1.0, 1.0],
    )
    .expect_err("unowned P2 edge must fail");
    if !matches!(
        topology_error,
        BoundaryConditionError::BoundaryTopologyMismatch { .. }
    ) || malformed_system.rhs.iter().any(|value| *value != 0.0)
    {
        return Err("malformed P2 edge was not rejected transactionally".into());
    }

    Ok(Metrics {
        p1_edge_checks: 1,
        p2_face_checks: 1,
        rollback_checks: 2,
        topology_checks: 1,
    })
}

fn main() {
    let result = match run() {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("p1_edge_checks", metrics.p1_edge_checks as f64)
            .minimum("p1_edge_checks", 1.0)
            .metric("p2_face_checks", metrics.p2_face_checks as f64)
            .minimum("p2_face_checks", 1.0)
            .metric("rollback_checks", metrics.rollback_checks as f64)
            .minimum("rollback_checks", 2.0)
            .metric("topology_checks", metrics.topology_checks as f64)
            .minimum("topology_checks", 1.0)
            .reference("independent vector H1 traction integration with typed validation and rollback")
            .reference_checksum("analytic/deterministic-fixture:fem-vector-natural-boundary-contract-v1")
            .dimension("2-D P1 edges and 3-D implicit-geometry P2 triangular faces")
            .formulation("vector Neumann traction integral ∫Γ t·(φ_i e_c) dΓ")
            .tolerance("constant resultants <= 1e-10; finite callbacks; typed topology rejection; no mutation")
            .input_checksum("mesh:unit-square-and-unit-cube-vector-h1-traction:v1"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };
    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("FEM vector natural-boundary QA serialization failed: {error}");
            std::process::exit(1);
        }
    }
    if !matches!(result.status, QaStatus::Passed) {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
