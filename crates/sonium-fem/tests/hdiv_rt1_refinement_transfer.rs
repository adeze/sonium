use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::mesh::{
    ElementType, InterpolationError, Mesh, Point, transfer_hdiv_rt1_state_checked,
};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};

fn triangle_mesh(origin_x: f64, origin_y: f64, scale: f64, cyclic: bool) -> Mesh {
    let mut mesh = Mesh::new(2);
    mesh.add_node(Point {
        x: origin_x,
        y: origin_y,
        z: 0.0,
    });
    mesh.add_node(Point {
        x: origin_x + scale,
        y: origin_y,
        z: 0.0,
    });
    mesh.add_node(Point {
        x: origin_x,
        y: origin_y + scale,
        z: 0.0,
    });
    let vertices = if cyclic { vec![1, 2, 0] } else { vec![0, 1, 2] };
    mesh.add_element(ElementType::Triangle, vertices);
    mesh
}

fn tetrahedron_mesh(origin: [f64; 3], scale: f64, cyclic: bool) -> Mesh {
    let mut mesh = Mesh::new(3);
    mesh.add_node(Point::new_3d(origin[0], origin[1], origin[2]));
    mesh.add_node(Point::new_3d(origin[0] + scale, origin[1], origin[2]));
    mesh.add_node(Point::new_3d(origin[0], origin[1] + scale, origin[2]));
    mesh.add_node(Point::new_3d(origin[0], origin[1], origin[2] + scale));
    let vertices = if cyclic {
        vec![1, 2, 0, 3]
    } else {
        vec![0, 1, 2, 3]
    };
    mesh.add_element(ElementType::Tetrahedron, vertices);
    mesh
}

fn reconstructed_field(
    mesh: &Mesh,
    space: &HdivSpace,
    values: &[f64],
    element: usize,
    reference: [f64; 3],
) -> [f64; 3] {
    let tabulation = space
        .try_tabulate(mesh, element, reference[0], reference[1], reference[2])
        .expect("affine RT1 tabulation should succeed");
    let local_dofs = space.local_to_global(mesh, element);
    let mut field = [0.0; 3];
    for (local_dof, &global_dof) in local_dofs.iter().enumerate() {
        for component in 0..mesh.dimension {
            field[component] += values[global_dof] * tabulation.vector_value(local_dof, component);
        }
    }
    field
}

fn physical_point(mesh: &Mesh, element: usize, reference: [f64; 3]) -> Point {
    let vertices = mesh.elements[element].vertices();
    let origin = mesh.nodes[vertices[0]];
    let v1 = mesh.nodes[vertices[1]];
    let v2 = mesh.nodes[vertices[2]];
    let v3 = if mesh.dimension == 3 {
        mesh.nodes[vertices[3]]
    } else {
        origin
    };
    Point::new_3d(
        origin.x
            + reference[0] * (v1.x - origin.x)
            + reference[1] * (v2.x - origin.x)
            + reference[2] * (v3.x - origin.x),
        origin.y
            + reference[0] * (v1.y - origin.y)
            + reference[1] * (v2.y - origin.y)
            + reference[2] * (v3.y - origin.y),
        origin.z
            + reference[0] * (v1.z - origin.z)
            + reference[1] * (v2.z - origin.z)
            + reference[2] * (v3.z - origin.z),
    )
}

fn assert_same_field(
    source: &Mesh,
    target: &Mesh,
    source_values: &[f64],
    target_values: &[f64],
    samples: &[[f64; 3]],
) {
    let source_space = HdivSpace::with_order(HdivOrder::Rt1);
    let target_space = HdivSpace::with_order(HdivOrder::Rt1);
    for &reference in samples {
        let physical = physical_point(target, 0, reference);
        let source_scale = 2.0;
        let source_reference = [
            (physical.x - source.nodes[0].x) / source_scale,
            (physical.y - source.nodes[0].y) / source_scale,
            (physical.z - source.nodes[0].z) / source_scale,
        ];
        let expected =
            reconstructed_field(source, &source_space, source_values, 0, source_reference);
        let actual = reconstructed_field(target, &target_space, target_values, 0, reference);
        for component in 0..source.dimension {
            assert!(
                (actual[component] - expected[component]).abs() < 2.0e-8,
                "component {component}: actual={} expected={} at {:?}",
                actual[component],
                expected[component],
                reference
            );
        }
    }
}

#[test]
fn rt1_transfer_preserves_affine_triangle_field_on_refined_oriented_target() {
    let source = triangle_mesh(0.0, 0.0, 2.0, false);
    let target = triangle_mesh(0.25, 0.25, 0.5, true);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let source_values: Vec<_> = (0..space.num_global_dofs(&source))
        .map(|index| 0.25 + 0.5 * index as f64)
        .collect();

    let target_values =
        transfer_hdiv_rt1_state_checked(&source, &target, &space, &space, &source_values)
            .expect("contained affine RT1 triangle should transfer");
    assert_same_field(
        &source,
        &target,
        &source_values,
        &target_values,
        &[[0.15, 0.2, 0.0], [0.2, 0.3, 0.0], [0.5, 0.1, 0.0]],
    );
}

#[test]
fn rt1_transfer_preserves_affine_tetrahedron_field_on_refined_oriented_target() {
    let source = tetrahedron_mesh([0.0, 0.0, 0.0], 2.0, false);
    let target = tetrahedron_mesh([0.15, 0.1, 0.1], 0.5, true);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let source_values: Vec<_> = (0..space.num_global_dofs(&source))
        .map(|index| -0.2 + 0.125 * index as f64)
        .collect();

    let target_values =
        transfer_hdiv_rt1_state_checked(&source, &target, &space, &space, &source_values)
            .expect("contained affine RT1 tetrahedron should transfer");
    assert_same_field(
        &source,
        &target,
        &source_values,
        &target_values,
        &[[0.1, 0.2, 0.2], [0.2, 0.1, 0.25], [0.15, 0.2, 0.1]],
    );
}

#[test]
fn rt1_transfer_accepts_multi_element_refinement_with_shared_edges() {
    let source = unit_square_triangles(1);
    let target = unit_square_triangles(2);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let source_values: Vec<_> = (0..space.num_global_dofs(&source))
        .map(|index| 0.1 + 0.07 * index as f64)
        .collect();

    let target_values =
        transfer_hdiv_rt1_state_checked(&source, &target, &space, &space, &source_values)
            .expect("RT1 transfer should handle shared source and target edges");
    assert_eq!(target_values.len(), space.num_global_dofs(&target));
    assert!(target_values.iter().all(|value| value.is_finite()));
}

#[test]
fn rt1_transfer_rejects_target_outside_source() {
    let source = triangle_mesh(0.0, 0.0, 1.0, false);
    let target = triangle_mesh(2.0, 2.0, 0.5, false);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let values = vec![0.0; space.num_global_dofs(&source)];

    assert!(matches!(
        transfer_hdiv_rt1_state_checked(&source, &target, &space, &space, &values),
        Err(InterpolationError::TargetElementOutside { .. })
    ));
}

#[test]
fn rt1_transfer_rejects_non_rt1_space() {
    let mesh = triangle_mesh(0.0, 0.0, 1.0, false);
    let rt0 = HdivSpace::with_order(HdivOrder::Rt0);
    let rt1 = HdivSpace::with_order(HdivOrder::Rt1);
    let values = vec![0.0; rt0.num_global_dofs(&mesh)];

    assert!(matches!(
        transfer_hdiv_rt1_state_checked(&mesh, &mesh, &rt0, &rt1, &values),
        Err(InterpolationError::InvalidMesh { .. })
    ));
}
