#![cfg(feature = "meshlang")]

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use num_complex::Complex64;
use sonium_fem::mesh::{
    ElementType, Mesh, MeshlangTransferError, Point, apply_complex_hdiv_rt1_state_transfer,
    apply_hdiv_rt1_state_transfer, transfer_hdiv_rt1_state_checked,
};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};
use std::collections::BTreeMap;

fn entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn triangle_mesh(origin_x: f64, origin_y: f64, scale: f64) -> Mesh {
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
    mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
    mesh
}

fn entities(space: &HdivSpace, mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<EntityId, usize> {
    (0..space.num_global_dofs(mesh))
        .map(|index| (entity(index), index))
        .collect()
}

fn interpolate_plan(target_entities: &BTreeMap<EntityId, usize>) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: target_entities
            .keys()
            .map(|&entity| IrStateTransferEntry {
                entity,
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

#[test]
fn rt1_bridge_matches_checked_kernel_for_edge_and_cell_modes() {
    let source_mesh = triangle_mesh(0.0, 0.0, 2.0);
    // Keep this bridge parity test on one affine mesh.  The checked RT1
    // kernel's adaptive/refinement ownership contract is covered separately;
    // this test isolates the MeshLang consumer mapping from that geometry
    // locator.
    let target_mesh = triangle_mesh(0.25, 0.25, 0.5);
    let source_space = HdivSpace::with_order(HdivOrder::Rt1);
    let target_space = HdivSpace::with_order(HdivOrder::Rt1);
    let source_dofs = source_space.num_global_dofs(&source_mesh);
    let source_values: Vec<_> = (0..source_dofs)
        .map(|index| 0.125 + 0.75 * index as f64)
        .collect();
    let expected = transfer_hdiv_rt1_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
    )
    .expect("direct RT1 transfer should succeed");
    let target_entities = entities(&target_space, &target_mesh);
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
    .expect("MeshLang RT1 bridge should succeed");

    for (index, (&entity, _)) in target_entities.iter().enumerate() {
        assert!(
            (transferred[&entity] - expected[index]).abs() < 1.0e-10,
            "RT1 DOF {index} mismatch: got {}, expected {}",
            transferred[&entity],
            expected[index]
        );
    }
}

#[test]
fn complex_rt1_bridge_preserves_real_and_imaginary_modes() {
    let source_mesh = triangle_mesh(0.0, 0.0, 2.0);
    let target_mesh = triangle_mesh(0.25, 0.25, 0.5);
    let source_space = HdivSpace::with_order(HdivOrder::Rt1);
    let target_space = HdivSpace::with_order(HdivOrder::Rt1);
    let source_dofs = source_space.num_global_dofs(&source_mesh);
    let source_values: Vec<_> = (0..source_dofs)
        .map(|index| Complex64::new(0.25 + index as f64, -0.5 - 0.5 * index as f64))
        .collect();
    let real_values: Vec<_> = source_values.iter().map(|value| value.re).collect();
    let imaginary_values: Vec<_> = source_values.iter().map(|value| value.im).collect();
    let expected_real = transfer_hdiv_rt1_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &real_values,
    )
    .expect("real RT1 transfer should succeed");
    let expected_imaginary = transfer_hdiv_rt1_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &imaginary_values,
    )
    .expect("imaginary RT1 transfer should succeed");
    let target_entities = entities(&target_space, &target_mesh);
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
    .expect("complex MeshLang RT1 bridge should succeed");

    for (index, (&entity, _)) in target_entities.iter().enumerate() {
        let value = transferred[&entity];
        assert!((value.re - expected_real[index]).abs() < 1.0e-10);
        assert!((value.im - expected_imaginary[index]).abs() < 1.0e-10);
    }
}

#[test]
fn rt1_bridge_rejects_wrong_space_order_before_callbacks() {
    let mesh = triangle_mesh(0.0, 0.0, 2.0);
    let source_space = HdivSpace::new();
    let target_space = HdivSpace::with_order(HdivOrder::Rt1);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::None,
        entries: vec![IrStateTransferEntry {
            entity: entity(0),
            action: IrStateTransferAction::Initialize,
        }],
    };
    let mut callback_called = false;
    let error = apply_hdiv_rt1_state_transfer::<&'static str, _>(
        &plan,
        &mesh,
        &mesh,
        &source_space,
        &target_space,
        &vec![0.0; source_space.num_global_dofs(&mesh)],
        &BTreeMap::new(),
        &BTreeMap::new(),
        |_, _, _| {
            callback_called = true;
            Ok(0.0)
        },
    )
    .expect_err("an RT0/RT1 mismatch must fail closed");
    assert!(matches!(
        error,
        MeshlangTransferError::UnsupportedHdivOrder {
            source_order: HdivOrder::Rt0,
            target_order: HdivOrder::Rt1,
        }
    ));
    assert!(!callback_called);
}
