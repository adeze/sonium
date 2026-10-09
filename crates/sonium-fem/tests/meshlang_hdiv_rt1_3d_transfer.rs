#![cfg(feature = "meshlang")]

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use num_complex::Complex64;
use sonium_fem::mesh::{
    ElementType, Mesh, Point, apply_complex_hdiv_rt1_state_transfer, apply_hdiv_rt1_state_transfer,
    transfer_hdiv_rt1_state_checked,
};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};
use std::collections::BTreeMap;

fn tetrahedron_mesh() -> Mesh {
    let mut mesh = Mesh::new(3);
    mesh.add_node(Point::new_3d(0.0, 0.0, 0.0));
    mesh.add_node(Point::new_3d(1.0, 0.0, 0.0));
    mesh.add_node(Point::new_3d(0.0, 1.0, 0.0));
    mesh.add_node(Point::new_3d(0.0, 0.0, 1.0));
    mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    mesh
}

fn entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn entities(space: &HdivSpace, mesh: &Mesh) -> BTreeMap<EntityId, usize> {
    (0..space.num_global_dofs(mesh))
        .map(|index| (entity(index), index))
        .collect()
}

fn interpolate_plan(target: &BTreeMap<EntityId, usize>) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: target
            .keys()
            .map(|&entity| IrStateTransferEntry {
                entity,
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

#[test]
fn real_rt1_tetrahedral_bridge_matches_checked_kernel() {
    let source_mesh = tetrahedron_mesh();
    let target_mesh = tetrahedron_mesh();
    let source_space = HdivSpace::with_order(HdivOrder::Rt1);
    let target_space = HdivSpace::with_order(HdivOrder::Rt1);
    let source_values: Vec<_> = (0..source_space.num_global_dofs(&source_mesh))
        .map(|index| 0.25 + 0.5 * index as f64)
        .collect();
    assert_eq!(
        source_values.len(),
        15,
        "tetrahedral RT1 has 15 global DOFs"
    );
    let target_entities = entities(&target_space, &target_mesh);
    let expected = transfer_hdiv_rt1_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
    )
    .expect("checked tetrahedral RT1 transfer should succeed");
    let transferred = apply_hdiv_rt1_state_transfer::<&'static str, _>(
        &interpolate_plan(&target_entities),
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
        &BTreeMap::new(),
        &target_entities,
        |_, _, _| Err("callback must not run for interpolation"),
    )
    .expect("MeshLang tetrahedral RT1 bridge should succeed");

    for (index, (&entity, _)) in target_entities.iter().enumerate() {
        assert!((transferred[&entity] - expected[index]).abs() < 1.0e-10);
    }
}

#[test]
fn complex_rt1_tetrahedral_bridge_preserves_components() {
    let source_mesh = tetrahedron_mesh();
    let target_mesh = tetrahedron_mesh();
    let source_space = HdivSpace::with_order(HdivOrder::Rt1);
    let target_space = HdivSpace::with_order(HdivOrder::Rt1);
    let source_values: Vec<_> = (0..source_space.num_global_dofs(&source_mesh))
        .map(|index| Complex64::new(0.25 + index as f64, -0.75 - 0.25 * index as f64))
        .collect();
    assert_eq!(
        source_values.len(),
        15,
        "tetrahedral RT1 has 15 global DOFs"
    );
    let target_entities = entities(&target_space, &target_mesh);
    let real_values: Vec<_> = source_values.iter().map(|value| value.re).collect();
    let imaginary_values: Vec<_> = source_values.iter().map(|value| value.im).collect();
    let expected_real = transfer_hdiv_rt1_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &real_values,
    )
    .expect("real tetrahedral RT1 transfer should succeed");
    let expected_imaginary = transfer_hdiv_rt1_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &imaginary_values,
    )
    .expect("imaginary tetrahedral RT1 transfer should succeed");
    let transferred = apply_complex_hdiv_rt1_state_transfer::<&'static str, _>(
        &interpolate_plan(&target_entities),
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
        &BTreeMap::new(),
        &target_entities,
        |_, _, _| Err("callback must not run for interpolation"),
    )
    .expect("complex MeshLang tetrahedral RT1 bridge should succeed");

    for (index, (&entity, _)) in target_entities.iter().enumerate() {
        let value = transferred[&entity];
        assert!((value.re - expected_real[index]).abs() < 1.0e-10);
        assert!((value.im - expected_imaginary[index]).abs() < 1.0e-10);
    }
}
