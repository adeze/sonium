//! CPU-only MeshLang/mesher curved P2 tetrahedral refinement reference.

use sonium_fem::mesh::refinement::refine_elements_high_order_checked;
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-h1-curved-p2-tetra-refinement-reference.v1";

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let mut mesh = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
        Point::new_3d(0.5, 0.2, 0.0),
        Point::new_3d(0.0, 0.5, 0.0),
        Point::new_3d(0.0, 0.0, 0.5),
        Point::new_3d(0.5, 0.5, 0.0),
        Point::new_3d(0.5, 0.0, 0.5),
        Point::new_3d(0.0, 0.5, 0.5),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Tetrahedron, (0..10).collect());

    let result = refine_elements_high_order_checked(&mut mesh, &[0])
        .map_err(|error| format!("curved P2 tetrahedral refinement failed: {error}"))?;
    let expected = Point::new_3d(0.25, 0.15, 0.0);
    let mapped_midpoint_error = result
        .new_nodes
        .iter()
        .map(|&node| {
            let point = mesh.nodes[node];
            (point.x - expected.x)
                .abs()
                .max((point.y - expected.y).abs())
                .max((point.z - expected.z).abs())
        })
        .fold(f64::INFINITY, f64::min);
    if !mapped_midpoint_error.is_finite() || mapped_midpoint_error > 1.0e-12 {
        return Err(format!(
            "parent-map midpoint missing or inaccurate: error={mapped_midpoint_error:e}"
        ));
    }
    if result.new_elements.len() != 8 || !mesh.validate_connectivity().is_ok() {
        return Err("curved P2 tetrahedral refinement topology contract failed".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("mapped_midpoint_error", mapped_midpoint_error)
        .metric("new_elements", result.new_elements.len() as f64)
        .metric("new_nodes", result.new_nodes.len() as f64)
        .minimum("new_elements", 8.0)
        .maximum("mapped_midpoint_error", 1.0e-12)
        .dimension("3-D curved P2 tetrahedral adaptive refinement")
        .formulation("parent-isoparametric edge midpoint evaluation with canonical edge reuse")
        .tolerance("mapped midpoint error <= 1e-12; eight child tetrahedra; valid connectivity")
        .input_checksum("curved-p2-tetrahedron:meshlang-adaptive-v1")
        .reference_checksum("analytic-p2-parent-map-midpoint:meshlang-curved-p2-tetra-v1")
        .seed("meshlang-fem-h1-curved-p2-tetra-refinement:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };
    let passed = matches!(result.status, sonium_qa::QaStatus::Passed);
    println!(
        "{}",
        result
            .elapsed_ms(started.elapsed().as_millis() as u64)
            .to_json_line()
            .expect("serialize QA result JSON")
    );
    if !passed {
        std::process::exit(1);
    }
}
