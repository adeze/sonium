#![cfg(feature = "meshlang")]

use std::collections::BTreeMap;

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use num_complex::Complex64;
use sonium_fem::mesh::refinement::refine_elements_with_history_checked;
use sonium_fem::mesh::{
    RefinementTransferDirection, apply_complex_hdiv_rt0_state_transfer_with_refinement_chain,
    apply_hdiv_rt0_state_transfer_with_refinement_chain,
    transfer_hdiv_rt0_state_with_refinement_chain_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};

fn entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn entities(space: &HdivSpace, mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<EntityId, usize> {
    (0..space.num_global_dofs(mesh))
        .map(|dof| (entity(dof), dof))
        .collect()
}

fn plan(
    target_entities: &BTreeMap<EntityId, usize>,
    action: IrStateTransferAction,
) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: match action {
            IrStateTransferAction::Conservative => StateTransferResponsibility::Conservative,
            _ => StateTransferResponsibility::Interpolate,
        },
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
fn rt0_chain_bridge_matches_checked_kernel() {
    let (level0, level1, level2, histories) = lineage();
    let space0 = HdivSpace::with_order(HdivOrder::Rt0);
    let space1 = HdivSpace::with_order(HdivOrder::Rt0);
    let space2 = HdivSpace::with_order(HdivOrder::Rt0);
    let spaces = [&space0, &space1, &space2];
    let meshes = [&level0, &level1, &level2];
    let source_entities = entities(&spaces[0], &level0);
    let target_entities = entities(&spaces[2], &level2);
    let source_values: Vec<_> = (0..spaces[0].num_global_dofs(&level0))
        .map(|dof| 0.25 + 0.125 * dof as f64)
        .collect();
    let expected = transfer_hdiv_rt0_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &source_values,
        &histories,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    let actual = apply_hdiv_rt0_state_transfer_with_refinement_chain::<&'static str, _>(
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
fn complex_rt0_chain_bridge_matches_componentwise_kernel() {
    let (level0, level1, level2, histories) = lineage();
    let space0 = HdivSpace::with_order(HdivOrder::Rt0);
    let space1 = HdivSpace::with_order(HdivOrder::Rt0);
    let space2 = HdivSpace::with_order(HdivOrder::Rt0);
    let spaces = [&space0, &space1, &space2];
    let meshes = [&level0, &level1, &level2];
    let source_entities = entities(&spaces[0], &level0);
    let target_entities = entities(&spaces[2], &level2);
    let source_values: Vec<_> = (0..spaces[0].num_global_dofs(&level0))
        .map(|dof| Complex64::new(0.25 + 0.125 * dof as f64, -2.0 + 0.5 * dof as f64))
        .collect();
    let expected_real = transfer_hdiv_rt0_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &source_values
            .iter()
            .map(|value| value.re)
            .collect::<Vec<_>>(),
        &histories,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    let expected_imaginary = transfer_hdiv_rt0_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &source_values
            .iter()
            .map(|value| value.im)
            .collect::<Vec<_>>(),
        &histories,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    let actual = apply_complex_hdiv_rt0_state_transfer_with_refinement_chain::<&'static str, _>(
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
fn rt0_chain_bridge_accepts_conservative_action() {
    let (level0, level1, level2, histories) = lineage();
    let space0 = HdivSpace::with_order(HdivOrder::Rt0);
    let space1 = HdivSpace::with_order(HdivOrder::Rt0);
    let space2 = HdivSpace::with_order(HdivOrder::Rt0);
    let spaces = [&space0, &space1, &space2];
    let meshes = [&level0, &level1, &level2];
    let source_entities = entities(&spaces[0], &level0);
    let target_entities = entities(&spaces[2], &level2);
    let source_values = vec![0.0; spaces[0].num_global_dofs(&level0)];
    let actual = apply_hdiv_rt0_state_transfer_with_refinement_chain::<&'static str, _>(
        &plan(&target_entities, IrStateTransferAction::Conservative),
        &meshes,
        &spaces,
        &source_values,
        &histories,
        RefinementTransferDirection::Refine,
        &source_entities,
        &target_entities,
        |_, _, _| Err("conservative callback must not run"),
    )
    .unwrap();
    assert_eq!(actual.len(), target_entities.len());
}
