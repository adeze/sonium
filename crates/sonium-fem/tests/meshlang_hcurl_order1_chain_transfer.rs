#![cfg(feature = "meshlang")]

use std::collections::BTreeMap;

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use num_complex::Complex64;
use sonium_fem::mesh::refinement::refine_elements_with_history_checked;
use sonium_fem::mesh::{
    MeshlangTransferError, RefinementTransferDirection,
    apply_complex_hcurl_order1_state_transfer_with_refinement_chain,
    apply_hcurl_order1_state_transfer_with_refinement_chain,
    transfer_hcurl_order1_state_with_refinement_chain_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};

fn entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn entities(space: &HcurlSpace, mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<EntityId, usize> {
    (0..space.num_global_dofs(mesh))
        .map(|dof| (entity(dof), dof))
        .collect()
}

fn plan(
    target_entities: &BTreeMap<EntityId, usize>,
    action: IrStateTransferAction,
) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: target_entities
            .keys()
            .map(|&entity| IrStateTransferEntry { entity, action })
            .collect(),
    }
}

fn lineage() -> (
    sonium_fem::mesh::Mesh,
    sonium_fem::mesh::Mesh,
    sonium_fem::mesh::Mesh,
    Vec<sonium_fem::mesh::refinement::RefinementHistory>,
) {
    let level0 = unit_square_triangles(1);
    let mut level1 = level0.clone();
    let (_, history0) = refine_elements_with_history_checked(&mut level1, &[0]).unwrap();
    let mut level2 = level1.clone();
    let (_, history1) = refine_elements_with_history_checked(&mut level2, &[0]).unwrap();
    (level0, level1, level2, vec![history0, history1])
}

#[test]
fn order1_hcurl_chain_bridge_matches_checked_kernel() {
    let (level0, level1, level2, histories) = lineage();
    let space0 = HcurlSpace::new();
    let space1 = HcurlSpace::new();
    let space2 = HcurlSpace::new();
    let spaces = [&space0, &space1, &space2];
    let meshes = [&level0, &level1, &level2];
    let source_entities = entities(&spaces[0], &level0);
    let target_entities = entities(&spaces[2], &level2);
    let source_values: Vec<_> = (0..spaces[0].num_global_dofs(&level0))
        .map(|dof| 0.25 + 0.125 * dof as f64)
        .collect();
    let expected = transfer_hcurl_order1_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &source_values,
        &histories,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    let actual = apply_hcurl_order1_state_transfer_with_refinement_chain::<&'static str, _>(
        &plan(&target_entities, IrStateTransferAction::Interpolate),
        &meshes,
        &spaces,
        &source_values,
        &histories,
        RefinementTransferDirection::Refine,
        &source_entities,
        &target_entities,
        |_, _, _| Err("interpolation callback must not run"),
    )
    .unwrap();
    assert_eq!(actual.len(), expected.len());
    for (&id, &dof) in &target_entities {
        assert!((actual[&id] - expected[dof]).abs() < 1.0e-11, "{id:?}");
    }
}

#[test]
fn complex_order1_hcurl_chain_bridge_matches_componentwise_kernel() {
    let (level0, level1, level2, histories) = lineage();
    let space0 = HcurlSpace::new();
    let space1 = HcurlSpace::new();
    let space2 = HcurlSpace::new();
    let spaces = [&space0, &space1, &space2];
    let meshes = [&level0, &level1, &level2];
    let source_entities = entities(&spaces[0], &level0);
    let target_entities = entities(&spaces[2], &level2);
    let source_values: Vec<_> = (0..spaces[0].num_global_dofs(&level0))
        .map(|dof| Complex64::new(0.25 + 0.125 * dof as f64, -2.0 + 0.5 * dof as f64))
        .collect();
    let real_values: Vec<_> = source_values.iter().map(|value| value.re).collect();
    let imaginary_values: Vec<_> = source_values.iter().map(|value| value.im).collect();
    let expected_real = transfer_hcurl_order1_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &real_values,
        &histories,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    let expected_imaginary = transfer_hcurl_order1_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &imaginary_values,
        &histories,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    let actual =
        apply_complex_hcurl_order1_state_transfer_with_refinement_chain::<&'static str, _>(
            &plan(&target_entities, IrStateTransferAction::Interpolate),
            &meshes,
            &spaces,
            &source_values,
            &histories,
            RefinementTransferDirection::Refine,
            &source_entities,
            &target_entities,
            |_, _, _| Err("interpolation callback must not run"),
        )
        .unwrap();
    for (&id, &dof) in &target_entities {
        let value = actual[&id];
        assert!(
            (value.re - expected_real[dof]).abs() < 1.0e-11,
            "{id:?} real"
        );
        assert!(
            (value.im - expected_imaginary[dof]).abs() < 1.0e-11,
            "{id:?} imaginary"
        );
    }
}

#[test]
fn order1_hcurl_chain_bridge_rejects_conservative_action_before_callbacks() {
    let (level0, level1, level2, histories) = lineage();
    let space0 = HcurlSpace::new();
    let space1 = HcurlSpace::new();
    let space2 = HcurlSpace::new();
    let spaces = [&space0, &space1, &space2];
    let meshes = [&level0, &level1, &level2];
    let source_entities = entities(&spaces[0], &level0);
    let target_entities = entities(&spaces[2], &level2);
    let mut callback_called = false;
    let error = apply_hcurl_order1_state_transfer_with_refinement_chain::<&'static str, _>(
        &plan(&target_entities, IrStateTransferAction::Conservative),
        &meshes,
        &spaces,
        &vec![0.0; spaces[0].num_global_dofs(&level0)],
        &histories,
        RefinementTransferDirection::Refine,
        &source_entities,
        &target_entities,
        |_, _, _| {
            callback_called = true;
            Ok(0.0)
        },
    )
    .unwrap_err();
    assert!(!callback_called);
    assert!(matches!(
        error,
        MeshlangTransferError::InvalidPlan(_) | MeshlangTransferError::UnsupportedAction { .. }
    ));
}

#[test]
fn order1_hcurl_chain_bridge_supports_coarsening() {
    let (level0, level1, level2, histories) = lineage();
    let space0 = HcurlSpace::new();
    let space1 = HcurlSpace::new();
    let space2 = HcurlSpace::new();
    let spaces = [&space0, &space1, &space2];
    let meshes = [&level0, &level1, &level2];
    let source_entities = entities(&spaces[2], &level2);
    let target_entities = entities(&spaces[0], &level0);
    let source_values: Vec<_> = (0..spaces[2].num_global_dofs(&level2))
        .map(|dof| -0.5 + 0.03125 * dof as f64)
        .collect();
    let expected = transfer_hcurl_order1_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &source_values,
        &histories,
        RefinementTransferDirection::Coarsen,
    )
    .unwrap();
    let actual = apply_hcurl_order1_state_transfer_with_refinement_chain::<&'static str, _>(
        &plan(&target_entities, IrStateTransferAction::Interpolate),
        &meshes,
        &spaces,
        &source_values,
        &histories,
        RefinementTransferDirection::Coarsen,
        &source_entities,
        &target_entities,
        |_, _, _| Err("coarsening callback must not run"),
    )
    .unwrap();
    for (&id, &dof) in &target_entities {
        assert!((actual[&id] - expected[dof]).abs() < 1.0e-11, "{id:?}");
    }
}
