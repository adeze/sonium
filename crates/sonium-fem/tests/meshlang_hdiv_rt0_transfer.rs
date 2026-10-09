#![cfg(feature = "meshlang")]

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::mesh::{
    apply_hdiv_rt0_state_transfer, transfer_hdiv_rt0_state_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HdivSpace};
use std::collections::BTreeMap;

fn entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn state(space: &HdivSpace, mesh: &sonium_fem::mesh::Mesh) -> Vec<f64> {
    (0..space.num_global_dofs(mesh))
        .map(|index| 0.25 + index as f64)
        .collect()
}

fn target_entities(space: &HdivSpace, mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<EntityId, usize> {
    (0..space.num_global_dofs(mesh))
        .map(|index| (entity(index), index))
        .collect()
}

#[test]
fn meshlang_hdiv_interpolation_matches_checked_rt0_kernel() {
    let source_mesh = unit_square_triangles(1);
    let target_mesh = unit_square_triangles(2);
    let source_space = HdivSpace::new();
    let target_space = HdivSpace::new();
    let source_values = state(&source_space, &source_mesh);
    let expected = transfer_hdiv_rt0_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
    )
    .expect("direct RT0 transfer should succeed");
    let targets = target_entities(&target_space, &target_mesh);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: vec![IrStateTransferEntry {
            entity: entity(0),
            action: IrStateTransferAction::Interpolate,
        }],
    };

    let transferred = apply_hdiv_rt0_state_transfer::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
        &BTreeMap::new(),
        &targets,
        |_, _, _| Ok(0.0),
    )
    .expect("MeshLang H(div) interpolation should succeed");
    assert_eq!(transferred.get(&entity(0)), Some(&expected[0]));
}

#[test]
fn meshlang_hdiv_conservative_uses_flux_moment_transfer() {
    let source_mesh = unit_square_triangles(1);
    let target_mesh = unit_square_triangles(2);
    let source_space = HdivSpace::new();
    let target_space = HdivSpace::new();
    let source_values = state(&source_space, &source_mesh);
    let expected = transfer_hdiv_rt0_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
    )
    .expect("direct RT0 transfer should succeed");
    let targets = target_entities(&target_space, &target_mesh);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Conservative,
        entries: vec![IrStateTransferEntry {
            entity: entity(1),
            action: IrStateTransferAction::Conservative,
        }],
    };

    let transferred = apply_hdiv_rt0_state_transfer::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
        &BTreeMap::new(),
        &targets,
        |_, _, _| Ok(0.0),
    )
    .expect("MeshLang H(div) conservative transfer should succeed");
    assert_eq!(transferred.get(&entity(1)), Some(&expected[1]));
}

#[test]
fn meshlang_hdiv_rejects_invalid_dof_maps_and_non_finite_values() {
    let source_mesh = unit_square_triangles(1);
    let target_mesh = unit_square_triangles(1);
    let source_space = HdivSpace::new();
    let target_space = HdivSpace::new();
    let source_values = state(&source_space, &source_mesh);
    let targets = target_entities(&target_space, &target_mesh);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::None,
        entries: vec![IrStateTransferEntry {
            entity: entity(0),
            action: IrStateTransferAction::Preserve,
        }],
    };

    let invalid_source = BTreeMap::from([(entity(0), source_values.len())]);
    let error = apply_hdiv_rt0_state_transfer::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
        &invalid_source,
        &targets,
        |_, _, _| Ok(0.0),
    )
    .expect_err("out-of-range source DOF must be rejected");
    assert!(matches!(
        error,
        sonium_fem::mesh::MeshlangTransferError::InvalidSourceNode { .. }
    ));

    let mut non_finite = source_values;
    non_finite[0] = f64::NAN;
    let source_entities = BTreeMap::from([(entity(0), 0usize)]);
    let error = apply_hdiv_rt0_state_transfer::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &non_finite,
        &source_entities,
        &targets,
        |_, _, _| Ok(0.0),
    )
    .expect_err("non-finite preserved state must be rejected");
    assert!(matches!(
        error,
        sonium_fem::mesh::MeshlangTransferError::NonFiniteSource { .. }
    ));
}
