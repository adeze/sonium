//! Release-facing P3 discontinuous-Galerkin space contract.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{
    ElementType, Mesh, unit_cube_hexahedra, unit_cube_tetrahedra, unit_square_quads,
    unit_square_triangles,
};
use sonium_fem::spaces::{DGSpace, FunctionSpace};
use sonium_qa::QaResult;
use std::collections::HashSet;

const CASE_ID: &str = "fem.dg-p3-contract.v1";
const TOLERANCE: f64 = 2.0e-12;

fn main() {
    match run() {
        Ok((element_checks, trace_checks)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("element_checks", element_checks as f64)
                .metric("boundary_trace_checks", trace_checks as f64)
                .minimum("element_checks", 4.0)
                .minimum("boundary_trace_checks", 4.0)
                .reference("independent P3 nodal partition, contiguous DG numbering, and face-trace ownership across triangle, quadrilateral, tetrahedron, and hexahedron production meshes")
                .reference_checksum("analytic/deterministic-fixture:fem-dg-p3-contract-v1")
                .dimension("2-D and 3-D affine production meshes")
                .formulation("P3 discontinuous Lagrange space with element-local numbering and boundary trace extraction")
                .tolerance("finite tabulation; partition residual <= 2e-12; all expected trace DOFs present; cell bubbles absent");
            match result.to_json_line() {
                Ok(json) => println!("QA_RESULT: {json}"),
                Err(error) => {
                    eprintln!("FEM DG P3 QA serialization failed: {error}");
                    std::process::exit(1);
                }
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("FEM DG P3 CONTRACT FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(usize, usize), String> {
    let meshes = [
        ("triangle", unit_square_triangles(1), [0.2, 0.3, 0.0]),
        ("quadrilateral", unit_square_quads(1), [0.1, -0.2, 0.0]),
        ("tetrahedron", unit_cube_tetrahedra(1), [0.1, 0.2, 0.15]),
        ("hexahedron", unit_cube_hexahedra(1), [0.1, -0.2, 0.3]),
    ];

    let mut element_checks = 0;
    let mut trace_checks = 0;
    for (name, mesh, point) in meshes {
        let (elements, traces) = check_mesh(name, &mesh, point)?;
        element_checks += elements;
        trace_checks += traces;
    }
    Ok((element_checks, trace_checks))
}

fn check_mesh(name: &str, mesh: &Mesh, point: [f64; 3]) -> Result<(usize, usize), String> {
    let space = DGSpace::new(PolynomialDegree::P3);
    space
        .validate(mesh)
        .map_err(|error| format!("{name} P3 space validation failed: {error}"))?;

    let expected = mesh.elements[0].element_type.num_nodes_p3();
    let mut element_checks = 0;
    for element in 0..mesh.elements.len() {
        let element_type = mesh.elements[element].element_type;
        if space.num_local_dofs(element_type) != expected {
            return Err(format!(
                "{name} element {element} local DOF count is {}, expected {expected}",
                space.num_local_dofs(element_type)
            ));
        }
        let local = space.local_to_global(mesh, element);
        if local.len() != expected || !local.windows(2).all(|window| window[0] + 1 == window[1]) {
            return Err(format!(
                "{name} element {element} DG numbering is not contiguous"
            ));
        }
        let tabulation = space.tabulate(mesh, element, point[0], point[1], point[2]);
        if tabulation.n_dofs != expected
            || tabulation.values.iter().any(|value| !value.is_finite())
            || tabulation
                .derivatives
                .iter()
                .any(|value| !value.is_finite())
            || (tabulation.values.iter().sum::<f64>() - 1.0).abs() > TOLERANCE
        {
            return Err(format!(
                "{name} element {element} P3 tabulation contract failed"
            ));
        }
        element_checks += 1;
    }

    let marker = mesh
        .boundaries
        .first()
        .map(|face| face.marker)
        .ok_or_else(|| format!("{name} mesh has no boundary faces"))?;
    let boundary: HashSet<usize> = space.boundary_dofs(mesh, marker).into_iter().collect();
    let mut trace_checks = 0;
    for face in mesh.boundaries.iter().filter(|face| face.marker == marker) {
        let element = &mesh.elements[face.element_idx];
        let base = space.local_to_global(mesh, face.element_idx)[0];
        let expected_trace = trace_local_dofs(
            element.element_type,
            element,
            face.local_idx,
            face.nodes.as_slice(),
        );
        for local_dof in expected_trace {
            if !boundary.contains(&(base + local_dof)) {
                return Err(format!(
                    "{name} boundary face {} is missing local trace DOF {local_dof}",
                    face.local_idx
                ));
            }
        }
        for local_dof in cell_bubble_dofs(element.element_type) {
            if boundary.contains(&(base + local_dof)) {
                return Err(format!(
                    "{name} boundary extraction incorrectly includes cell bubble {local_dof}"
                ));
            }
        }
        trace_checks += 1;
    }
    Ok((element_checks, trace_checks))
}

fn trace_local_dofs(
    element_type: ElementType,
    element: &sonium_fem::mesh::Element,
    face_index: usize,
    face_nodes: &[usize],
) -> Vec<usize> {
    let face_nodes: HashSet<usize> = face_nodes.iter().copied().collect();
    let vertices = element.vertices();
    let mut trace = Vec::new();
    for (local, &node) in vertices.iter().enumerate() {
        if face_nodes.contains(&node) {
            trace.push(local);
        }
    }
    for (edge, &(a, b)) in element_type.local_edges().iter().enumerate() {
        if face_nodes.contains(&vertices[a]) && face_nodes.contains(&vertices[b]) {
            trace.push(vertices.len() + 2 * edge);
            trace.push(vertices.len() + 2 * edge + 1);
        }
    }
    match element_type {
        ElementType::Tetrahedron => trace.push(16 + face_index),
        ElementType::Hexahedron => trace.extend(32 + 4 * face_index..36 + 4 * face_index),
        ElementType::Triangle | ElementType::Quadrilateral => {}
    }
    trace
}

fn cell_bubble_dofs(element_type: ElementType) -> Vec<usize> {
    match element_type {
        ElementType::Triangle => vec![9],
        ElementType::Quadrilateral => (12..16).collect(),
        ElementType::Tetrahedron => Vec::new(),
        ElementType::Hexahedron => (56..64).collect(),
    }
}
