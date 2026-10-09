//! Release-backed P2/Hex27 refinement reference.

use sonium_mesher::mesh::generators::{
    box_mesh_hexahedra, unit_cube_tetrahedra, unit_square_quads, unit_square_triangles,
};
use sonium_mesher::mesh::refinement::refine_elements_high_order_checked;
use sonium_mesher::mesh::{Mesh, promote_to_p2};
use sonium_qa::QaResult;

const CASE_ID: &str = "mesher.high-order-refinement.v1";

fn refine_one(mut mesh: Mesh, expected_children: usize) -> Result<(usize, usize), String> {
    let parent_id = mesh.elements[0].id;
    let result = refine_elements_high_order_checked(&mut mesh, &[0])
        .map_err(|error| format!("high-order refinement failed: {error}"))?;
    if result.new_elements.len() != expected_children
        || result.removed_elements != vec![0]
        || mesh.validate_connectivity().is_err()
        || mesh
            .elements
            .iter()
            .filter(|element| element.parent_id == Some(parent_id))
            .any(|element| element.try_polynomial_order() != Ok(2))
    {
        return Err("high-order child layout or connectivity invariant failed".into());
    }
    Ok((result.new_nodes.len(), result.new_elements.len()))
}

fn run() -> Result<QaResult, String> {
    let mut triangle_base = unit_square_triangles(1);
    triangle_base.detect_boundaries();
    let triangle = refine_one(promote_to_p2(&triangle_base), 4)?;

    let mut quadrilateral_base = unit_square_quads(1);
    quadrilateral_base.detect_boundaries();
    let quadrilateral = refine_one(promote_to_p2(&quadrilateral_base), 4)?;

    let mut tetrahedron_base = unit_cube_tetrahedra(1);
    tetrahedron_base.detect_boundaries();
    let tetrahedron = refine_one(promote_to_p2(&tetrahedron_base), 8)?;

    let mut hexahedron_base = box_mesh_hexahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1, 1);
    hexahedron_base.detect_boundaries();
    let hexahedron = refine_one(promote_to_p2(&hexahedron_base), 8)?;
    let checks = triangle.1 + quadrilateral.1 + tetrahedron.1 + hexahedron.1;

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("families_checked", 4.0)
        .metric("child_elements", checks as f64)
        .metric(
            "new_nodes",
            (triangle.0 + quadrilateral.0 + tetrahedron.0 + hexahedron.0) as f64,
        )
        .minimum("families_checked", 4.0)
        .minimum("child_elements", 24.0)
        .input_checksum("analytic/deterministic-fixture:mesher-high-order-refinement-v1")
        .reference(
            "P2 triangle/quadrilateral/tetrahedron and Hex27 child-layout/connectivity invariants",
        )
        .reference_checksum("analytic/deterministic-fixture:mesher-high-order-refinement-v1")
        .dimension("2-D and 3-D P2/Hex27 meshes")
        .formulation("checked high-order red/tensor-product refinement")
        .tolerance("four families preserve polynomial order and valid connectivity")
        .seed("mesher-high-order-refinement-reference:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("Mesher result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, &error)
                    .to_json_line()
                    .expect("Mesher failure result must serialize")
            );
            println!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
