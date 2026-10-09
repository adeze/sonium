#![cfg(feature = "meshlang")]

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::mesh::{
    MeshlangTransferError, apply_hcurl_mixed_state_transfer,
    transfer_hcurl_mixed_triangle_state_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use std::collections::BTreeMap;

fn curve_id(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn interpolation_plan(dof_count: usize) -> (IrStateTransferPlan, BTreeMap<EntityId, usize>) {
    let mut entities = BTreeMap::new();
    let mut entries = Vec::with_capacity(dof_count);
    for dof in 0..dof_count {
        let entity = curve_id(dof);
        entities.insert(entity, dof);
        entries.push(IrStateTransferEntry {
            entity,
            action: IrStateTransferAction::Interpolate,
        });
    }
    (
        IrStateTransferPlan {
            responsibility: StateTransferResponsibility::Interpolate,
            entries,
        },
        entities,
    )
}

fn curved_triangle_mesh(midpoint_offset: f64) -> sonium_fem::mesh::Mesh {
    let mut mesh = unit_square_triangles(1).to_p2();
    let midpoint = mesh.elements[0].nodes[3];
    mesh.nodes[midpoint].y += midpoint_offset;
    mesh
}

fn assert_bridge_matches_kernel(source_space: HcurlSpace, target_space: HcurlSpace) {
    let mesh = unit_square_triangles(1);
    let source_dofs = source_space.num_global_dofs(&mesh);
    let target_dofs = target_space.num_global_dofs(&mesh);
    let source_values: Vec<_> = (0..source_dofs)
        .map(|dof| 0.25 + 0.125 * dof as f64 - 0.01 * (dof as f64).powi(2))
        .collect();
    let expected = transfer_hcurl_mixed_triangle_state_checked(
        &mesh,
        &mesh,
        &source_space,
        &target_space,
        &source_values,
    )
    .expect("mixed-order kernel should accept the affine unit-square mesh");
    assert_eq!(expected.len(), target_dofs);

    let (plan, target_entities) = interpolation_plan(target_dofs);
    let source_entities: BTreeMap<_, _> =
        (0..source_dofs).map(|dof| (curve_id(dof), dof)).collect();
    let actual = apply_hcurl_mixed_state_transfer::<String, _>(
        &plan,
        &mesh,
        &mesh,
        &source_space,
        &target_space,
        &source_values,
        &source_entities,
        &target_entities,
        |_, _, _| -> Result<f64, String> { panic!("interpolation must not invoke callback") },
    )
    .expect("mixed-order bridge should apply checked interpolation");

    assert_eq!(actual.len(), target_dofs);
    for (dof, expected_value) in expected.iter().enumerate() {
        assert!(
            (actual[&curve_id(dof)] - expected_value).abs() < 1.0e-11,
            "mixed transfer mismatch at DOF {dof}"
        );
    }
}

#[test]
fn mixed_hcurl_order1_to_order2_matches_checked_kernel() {
    assert_bridge_matches_kernel(HcurlSpace::new(), HcurlSpace::order2());
}

#[test]
fn mixed_hcurl_order2_to_order1_matches_checked_kernel() {
    assert_bridge_matches_kernel(HcurlSpace::order2(), HcurlSpace::new());
}

#[test]
fn mixed_hcurl_conservative_action_is_rejected_before_callback() {
    let mesh = unit_square_triangles(1);
    let space = HcurlSpace::new();
    let entity = curve_id(0);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Conservative,
        entries: vec![IrStateTransferEntry {
            entity,
            action: IrStateTransferAction::Conservative,
        }],
    };
    let mut entities = BTreeMap::new();
    entities.insert(entity, 0);

    let error = apply_hcurl_mixed_state_transfer::<String, _>(
        &plan,
        &mesh,
        &mesh,
        &space,
        &HcurlSpace::order2(),
        &vec![0.0; space.num_global_dofs(&mesh)],
        &entities,
        &entities,
        |_, _, _| -> Result<f64, String> {
            panic!("conservative rejection must precede callbacks")
        },
    )
    .expect_err("scalar conservative action is not an H(curl) transfer");
    assert!(matches!(
        error,
        MeshlangTransferError::UnsupportedAction { .. }
    ));
}

#[test]
fn mixed_hcurl_nonfinite_source_is_rejected() {
    let mesh = unit_square_triangles(1);
    let source_space = HcurlSpace::new();
    let target_space = HcurlSpace::order2();
    let source_dofs = source_space.num_global_dofs(&mesh);
    let (plan, target_entities) = interpolation_plan(target_space.num_global_dofs(&mesh));
    let source_entities: BTreeMap<_, _> =
        (0..source_dofs).map(|dof| (curve_id(dof), dof)).collect();
    let mut source_values = vec![0.0; source_dofs];
    source_values[0] = f64::NAN;

    let error = apply_hcurl_mixed_state_transfer::<String, _>(
        &plan,
        &mesh,
        &mesh,
        &source_space,
        &target_space,
        &source_values,
        &source_entities,
        &target_entities,
        |_, _, _| -> Result<f64, String> { panic!("interpolation must not invoke callback") },
    )
    .expect_err("non-finite source state must be rejected");
    assert!(matches!(error, MeshlangTransferError::Interpolation(_)));
}

#[test]
fn mixed_hcurl_transfer_matches_curved_order_mixed_kernel_both_directions() {
    let affine = unit_square_triangles(1);
    let curved_inward = curved_triangle_mesh(0.08);
    let curved_outward = curved_triangle_mesh(-0.08);

    let cases = [
        (
            affine,
            curved_inward,
            HcurlSpace::new(),
            HcurlSpace::order2(),
        ),
        (
            curved_outward,
            unit_square_triangles(1),
            HcurlSpace::order2(),
            HcurlSpace::new(),
        ),
    ];

    for (source_mesh, target_mesh, source_space, target_space) in cases {
        let source_dofs = source_space.num_global_dofs(&source_mesh);
        let target_dofs = target_space.num_global_dofs(&target_mesh);
        let source_values: Vec<_> = (0..source_dofs)
            .map(|dof| 0.17 + 0.09 * dof as f64 - 0.003 * (dof as f64).powi(2))
            .collect();
        let expected = transfer_hcurl_mixed_triangle_state_checked(
            &source_mesh,
            &target_mesh,
            &source_space,
            &target_space,
            &source_values,
        )
        .expect("curved mixed-order kernel transfer");
        let (plan, target_entities) = interpolation_plan(target_dofs);
        let source_entities: BTreeMap<_, _> =
            (0..source_dofs).map(|dof| (curve_id(dof), dof)).collect();
        let actual = apply_hcurl_mixed_state_transfer::<String, _>(
            &plan,
            &source_mesh,
            &target_mesh,
            &source_space,
            &target_space,
            &source_values,
            &source_entities,
            &target_entities,
            |_, _, _| -> Result<f64, String> {
                panic!("interpolation callback unexpectedly invoked")
            },
        )
        .expect("curved mixed-order bridge transfer");

        assert_eq!(actual.len(), target_dofs);
        for (dof, expected_value) in expected.iter().enumerate() {
            let actual_value = actual[&curve_id(dof)];
            assert!(
                (actual_value - expected_value).abs() < 2.0e-9,
                "curved mixed-order DOF {dof}: {actual_value} vs {expected_value}"
            );
        }
    }
}
