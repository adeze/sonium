#![cfg(feature = "meshlang")]

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::mesh::{
    InterpolationError, MeshlangTransferError, Point, apply_hcurl_order2_triangle_state_transfer,
    transfer_hcurl_order2_triangle_state_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use std::collections::BTreeMap;

fn dof_entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn curved_triangle_mesh(midpoint_offset: f64) -> sonium_fem::mesh::Mesh {
    let mut mesh = unit_square_triangles(1).to_p2();
    let midpoint = mesh.elements[0].nodes[3];
    mesh.nodes[midpoint].y += midpoint_offset;
    mesh
}

fn state(space: &HcurlSpace, mesh: &sonium_fem::mesh::Mesh) -> Vec<f64> {
    (0..space.num_global_dofs(mesh))
        .map(|index| 0.21 + 0.07 * index as f64 - 0.002 * (index * index) as f64)
        .collect()
}

fn entities(dofs: usize) -> BTreeMap<EntityId, usize> {
    (0..dofs)
        .map(|dof| (dof_entity(dof), dof))
        .collect::<BTreeMap<_, _>>()
}

fn interpolation_plan(dofs: usize) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..dofs)
            .map(|dof| IrStateTransferEntry {
                entity: dof_entity(dof),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

fn bridge_transfer(
    source_mesh: &sonium_fem::mesh::Mesh,
    target_mesh: &sonium_fem::mesh::Mesh,
    source_values: &[f64],
) -> Result<BTreeMap<EntityId, f64>, MeshlangTransferError> {
    let source_space = HcurlSpace::order2();
    let target_space = HcurlSpace::order2();
    let source_dofs = source_space.num_global_dofs(source_mesh);
    let target_dofs = target_space.num_global_dofs(target_mesh);

    apply_hcurl_order2_triangle_state_transfer::<&'static str, _>(
        &interpolation_plan(target_dofs),
        source_mesh,
        target_mesh,
        &source_space,
        &target_space,
        source_values,
        &entities(source_dofs),
        &entities(target_dofs),
        |_, _, _| Err("interpolation not callback"),
    )
}

#[test]
pub fn meshlang_order2_hcurl_bridge_matches_curved_kernel_in_both_directions() {
    let affine = unit_square_triangles(1);
    // The checked kernel locates target quadrature points in the source
    // element. Use an inward bend for affine→curved and an outward bend
    // for curved→affine so each target domain is contained by its source.
    let curved_inward = curved_triangle_mesh(0.08);
    let curved_outward = curved_triangle_mesh(-0.08);
    let space = HcurlSpace::order2();

    let affine_values = state(&space, &affine);
    let curved_values = state(&space, &curved_outward);
    let affine_to_curved = bridge_transfer(&affine, &curved_inward, &affine_values).unwrap();
    let curved_to_affine = bridge_transfer(&curved_outward, &affine, &curved_values).unwrap();

    let expected_affine_to_curved = transfer_hcurl_order2_triangle_state_checked(
        &affine,
        &curved_inward,
        &space,
        &space,
        &affine_values,
    )
    .unwrap();
    let expected_curved_to_affine = transfer_hcurl_order2_triangle_state_checked(
        &curved_outward,
        &affine,
        &space,
        &space,
        &curved_values,
    )
    .unwrap();

    for (dof, expected) in expected_affine_to_curved.iter().enumerate() {
        let actual = affine_to_curved[&dof_entity(dof)];
        assert!(
            (actual - expected).abs() < 2.0e-9,
            "affine→curved DOF {dof}"
        );
    }
    for (dof, expected) in expected_curved_to_affine.iter().enumerate() {
        let actual = curved_to_affine[&dof_entity(dof)];
        assert!(
            (actual - expected).abs() < 2.0e-9,
            "curved→affine DOF {dof}"
        );
    }
}

#[test]
pub fn meshlang_order2_hcurl_bridge_rejects_curved_bad_inputs_before_callbacks() {
    let curved = curved_triangle_mesh(0.08);
    let space = HcurlSpace::order2();
    let dofs = space.num_global_dofs(&curved);
    let entity_map = entities(dofs);
    let values = state(&space, &curved);

    let mut nonfinite_geometry = curved.clone();
    nonfinite_geometry.nodes[nonfinite_geometry.elements[0].nodes[3]] =
        Point::new_3d(f64::NAN, 0.0, 0.0);
    let error = apply_hcurl_order2_triangle_state_transfer::<&'static str, _>(
        &interpolation_plan(dofs),
        &nonfinite_geometry,
        &curved,
        &space,
        &space,
        &values,
        &entity_map,
        &entity_map,
        |_, _, _| Err("interpolation not callback"),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        MeshlangTransferError::Interpolation(InterpolationError::InvalidMesh { .. })
    ));

    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Conservative,
        entries: vec![IrStateTransferEntry {
            entity: dof_entity(0),
            action: IrStateTransferAction::Conservative,
        }],
    };
    let mut callback_count = 0;
    let error = apply_hcurl_order2_triangle_state_transfer::<&'static str, _>(
        &plan,
        &curved,
        &curved,
        &space,
        &space,
        &values,
        &entity_map,
        &entity_map,
        |_, _, _| {
            callback_count += 1;
            Ok(0.0)
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        MeshlangTransferError::UnsupportedAction {
            action: IrStateTransferAction::Conservative,
            ..
        }
    ));
    assert_eq!(callback_count, 0);

    let mut nonfinite_values = values;
    nonfinite_values[0] = f64::NAN;
    assert!(matches!(
        bridge_transfer(&curved, &curved, &nonfinite_values),
        Err(MeshlangTransferError::Interpolation(
            InterpolationError::NonFiniteSource { .. }
        ))
    ));
}
