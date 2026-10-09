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
    apply_complex_hdiv_rt1_state_transfer_with_refinement_chain,
    apply_hdiv_rt1_state_transfer_with_refinement_chain,
    transfer_hdiv_rt1_state_with_refinement_chain_checked, unit_square_triangles,
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
fn rt1_chain_bridge_matches_checked_real_kernel_across_two_stages() {
    let (level0, level1, level2, histories) = lineage();
    let space0 = HdivSpace::with_order(HdivOrder::Rt1);
    let space1 = HdivSpace::with_order(HdivOrder::Rt1);
    let space2 = HdivSpace::with_order(HdivOrder::Rt1);
    let spaces = [&space0, &space1, &space2];
    let meshes = [&level0, &level1, &level2];
    let source_entities = entities(&spaces[0], &level0);
    let target_entities = entities(&spaces[2], &level2);
    let plan = interpolate_plan(&target_entities);
    let source_values: Vec<_> = (0..spaces[0].num_global_dofs(&level0))
        .map(|dof| 0.25 + 0.125 * dof as f64)
        .collect();
    let expected = transfer_hdiv_rt1_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &source_values,
        &histories,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    let mut callback_calls = 0;
    let actual = apply_hdiv_rt1_state_transfer_with_refinement_chain::<&'static str, _>(
        &plan,
        &meshes,
        &spaces,
        &source_values,
        &histories,
        RefinementTransferDirection::Refine,
        &source_entities,
        &target_entities,
        |_, _, _| {
            callback_calls += 1;
            Err("interpolation callback must not run")
        },
    )
    .unwrap();
    assert_eq!(callback_calls, 0);
    assert_eq!(actual.len(), expected.len());
    for (&id, &dof) in &target_entities {
        assert!((actual[&id] - expected[dof]).abs() < 1.0e-11, "{id:?}");
    }
}

#[test]
fn complex_rt1_chain_bridge_matches_independent_real_imaginary_kernels() {
    let (level0, level1, level2, histories) = lineage();
    let space0 = HdivSpace::with_order(HdivOrder::Rt1);
    let space1 = HdivSpace::with_order(HdivOrder::Rt1);
    let space2 = HdivSpace::with_order(HdivOrder::Rt1);
    let spaces = [&space0, &space1, &space2];
    let meshes = [&level0, &level1, &level2];
    let source_entities = entities(&spaces[0], &level0);
    let target_entities = entities(&spaces[2], &level2);
    let plan = interpolate_plan(&target_entities);
    let source_values: Vec<_> = (0..spaces[0].num_global_dofs(&level0))
        .map(|dof| {
            let real = 0.25 + 0.125 * dof as f64;
            Complex64::new(real, -2.0 * real)
        })
        .collect();
    let real: Vec<_> = source_values.iter().map(|value| value.re).collect();
    let imaginary: Vec<_> = source_values.iter().map(|value| value.im).collect();
    let expected_real = transfer_hdiv_rt1_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &real,
        &histories,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    let expected_imaginary = transfer_hdiv_rt1_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &imaginary,
        &histories,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    let actual = apply_complex_hdiv_rt1_state_transfer_with_refinement_chain::<&'static str, _>(
        &plan,
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
fn rt1_chain_bridge_rejects_conservative_action_before_callbacks() {
    let (level0, level1, level2, histories) = lineage();
    let space0 = HdivSpace::with_order(HdivOrder::Rt1);
    let space1 = HdivSpace::with_order(HdivOrder::Rt1);
    let space2 = HdivSpace::with_order(HdivOrder::Rt1);
    let spaces = [&space0, &space1, &space2];
    let meshes = [&level0, &level1, &level2];
    let source_entities = entities(&spaces[0], &level0);
    let target_entities = entities(&spaces[2], &level2);
    let mut plan = interpolate_plan(&target_entities);
    plan.entries[0].action = IrStateTransferAction::Conservative;
    let mut callback_called = false;
    let error = apply_hdiv_rt1_state_transfer_with_refinement_chain::<&'static str, _>(
        &plan,
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
    .expect_err("scalar conservative RT1 action must be rejected");
    assert!(matches!(
        error,
        MeshlangTransferError::InvalidPlan(_) | MeshlangTransferError::UnsupportedAction { .. }
    ));
    assert!(!callback_called);
}
