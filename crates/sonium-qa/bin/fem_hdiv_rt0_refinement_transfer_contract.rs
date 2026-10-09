//! CPU-only affine RT0 H(div) refinement state-transfer contract.

use sonium_fem::mesh::{
    InterpolationError, MeshTopology, Point, transfer_hdiv_rt0_state_checked, unit_cube_tetrahedra,
    unit_square_quads, unit_square_triangles,
};
use sonium_fem::quadrature::gauss_legendre_1d;
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hdiv-rt0-refinement-transfer.v1";

fn main() {
    let result = match run() {
        Ok((edge_count, face_count, max_2d_error, max_3d_error, rejection_checks)) => {
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("target_edges", edge_count as f64)
                .minimum("target_edges", 1.0)
                .metric("target_faces", face_count as f64)
                .minimum("target_faces", 1.0)
                .metric("max_2d_error", max_2d_error)
                .maximum("max_2d_error", 2.0e-10)
                .metric("max_3d_identity_error", max_3d_error)
                .maximum("max_3d_identity_error", 2.0e-10)
                .metric("rejection_checks", rejection_checks as f64)
                .minimum("rejection_checks", 4.0)
                .dimension("affine P1 triangular/tetrahedral RT0 H(div) transfer")
                .formulation("production contravariant-Piola reconstruction and oriented normal-flux moments")
                .input_checksum("unit-square-triangles-1-to-2-and-unit-cube-tetrahedra-1:rt0-v1")
                .reference_checksum("analytic/deterministic-fixture:fem-hdiv-rt0-refinement-transfer-v1")
                .seed("fem-hdiv-rt0-refinement-transfer:cpu:v1")
        }
        Err(error) => QaResult::failed(CASE_ID, error),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}

fn affine_field(point: Point) -> [f64; 3] {
    [
        1.0 + 0.25 * point.x,
        2.0 + 0.25 * point.y,
        3.0 + 0.25 * point.z,
    ]
}

fn same_edge(left: (usize, usize), right: (usize, usize)) -> bool {
    (left.0 == right.0 && left.1 == right.1) || (left.0 == right.1 && left.1 == right.0)
}

fn edge_state(mesh: &sonium_fem::mesh::Mesh, space: &HdivSpace) -> Vec<f64> {
    let topology = MeshTopology::build(mesh);
    let quadrature = gauss_legendre_1d(8);
    topology
        .edges
        .iter()
        .enumerate()
        .map(|(edge_index, &edge)| {
            let (element_index, local_edge) = mesh
                .elements
                .iter()
                .enumerate()
                .find_map(|(element_index, element)| {
                    let vertices = element.vertices();
                    element
                        .element_type
                        .local_edges()
                        .iter()
                        .enumerate()
                        .find(|(_, candidate)| {
                            let &(a, b) = *candidate;
                            same_edge((vertices[a], vertices[b]), edge)
                        })
                        .map(|(local_edge, _)| (element_index, local_edge))
                })
                .expect("edge owner");
            let vertices = mesh.elements[element_index].vertices();
            let &(a, b) = &mesh.elements[element_index].element_type.local_edges()[local_edge];
            let p0 = mesh.nodes[vertices[a]];
            let p1 = mesh.nodes[vertices[b]];
            let centroid = {
                let mut result = Point::new_3d(0.0, 0.0, 0.0);
                let scale = 1.0 / vertices.len() as f64;
                for &node in vertices {
                    result.x += scale * mesh.nodes[node].x;
                    result.y += scale * mesh.nodes[node].y;
                    result.z += scale * mesh.nodes[node].z;
                }
                result
            };
            let tangent = [p1.x - p0.x, p1.y - p0.y];
            let midpoint = [(p0.x + p1.x) * 0.5, (p0.y + p1.y) * 0.5];
            let to_centroid = [centroid.x - midpoint[0], centroid.y - midpoint[1]];
            let mut normal = [tangent[1] * 0.5, -tangent[0] * 0.5, 0.0];
            if normal[0] * to_centroid[0] + normal[1] * to_centroid[1] > 0.0 {
                normal[0] = -normal[0];
                normal[1] = -normal[1];
            }
            let local_dof = space
                .local_to_global(mesh, element_index)
                .iter()
                .position(|&dof| dof == edge_index)
                .expect("RT0 edge owner map");
            let mut flux = 0.0;
            let mut normalization = 0.0;
            for point in &quadrature {
                let coordinate = point.xi();
                let lambda_a = 0.5 * (1.0 - coordinate);
                let lambda_b = 0.5 * (1.0 + coordinate);
                let mut barycentric = [0.0; 3];
                barycentric[a] = lambda_a;
                barycentric[b] = lambda_b;
                let tabulation = space
                    .try_tabulate(mesh, element_index, barycentric[1], barycentric[2], 0.0)
                    .expect("RT0 target tabulation");
                let field = affine_field(Point::new_3d(
                    lambda_a * p0.x + lambda_b * p1.x,
                    lambda_a * p0.y + lambda_b * p1.y,
                    lambda_a * p0.z + lambda_b * p1.z,
                ));
                let basis = [
                    tabulation.vector_value(local_dof, 0),
                    tabulation.vector_value(local_dof, 1),
                    0.0,
                ];
                flux += point.weight
                    * (field[0] * normal[0] + field[1] * normal[1] + field[2] * normal[2]);
                normalization += point.weight
                    * (basis[0] * normal[0] + basis[1] * normal[1] + basis[2] * normal[2]);
            }
            flux / normalization
        })
        .collect()
}

fn run() -> Result<(usize, usize, f64, f64, usize), String> {
    let source = unit_square_triangles(1);
    let target = unit_square_triangles(2);
    let space = HdivSpace::new();
    let source_values = edge_state(&source, &space);
    let expected = edge_state(&target, &space);
    let transferred =
        transfer_hdiv_rt0_state_checked(&source, &target, &space, &space, &source_values)
            .map_err(|error| error.to_string())?;
    let max_2d_error = transferred
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    if max_2d_error > 2.0e-10 {
        return Err(format!("2-D RT0 transfer error {max_2d_error}"));
    }

    let source_3d = unit_cube_tetrahedra(1);
    let source_3d_values: Vec<_> = (0..space.num_global_dofs(&source_3d))
        .map(|index| 0.125 * (index as f64 + 1.0))
        .collect();
    let transferred_3d =
        transfer_hdiv_rt0_state_checked(&source_3d, &source_3d, &space, &space, &source_3d_values)
            .map_err(|error| error.to_string())?;
    let max_3d_error = transferred_3d
        .iter()
        .zip(&source_3d_values)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    if max_3d_error > 2.0e-10 {
        return Err(format!("3-D RT0 identity error {max_3d_error}"));
    }

    let source_dofs = space.num_global_dofs(&source);
    let mut rejection_checks = 0;
    if matches!(
        transfer_hdiv_rt0_state_checked(
            &source,
            &target,
            &HdivSpace::with_order(HdivOrder::Rt1),
            &space,
            &vec![0.0; source_dofs],
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ) {
        rejection_checks += 1;
    }
    if matches!(
        transfer_hdiv_rt0_state_checked(
            &source,
            &target,
            &space,
            &space,
            &vec![0.0; source_dofs - 1],
        ),
        Err(InterpolationError::SourceValueCount { .. })
    ) {
        rejection_checks += 1;
    }
    if matches!(
        transfer_hdiv_rt0_state_checked(
            &source,
            &unit_square_quads(1),
            &space,
            &space,
            &vec![0.0; source_dofs],
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ) {
        rejection_checks += 1;
    }
    let mut non_finite_values = vec![0.0; source_dofs];
    non_finite_values[0] = f64::NAN;
    if matches!(
        transfer_hdiv_rt0_state_checked(&source, &target, &space, &space, &non_finite_values,),
        Err(InterpolationError::NonFiniteSource { .. })
    ) {
        rejection_checks += 1;
    }
    if rejection_checks < 4 {
        return Err(format!(
            "only {rejection_checks} RT0 rejection checks passed"
        ));
    }
    Ok((
        MeshTopology::build(&target).edges.len(),
        MeshTopology::build(&source_3d).faces.len(),
        max_2d_error,
        max_3d_error,
        rejection_checks,
    ))
}
