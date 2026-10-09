#![cfg(feature = "meshlang")]

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    IrStateTransferPlanChain, StateTransferResponsibility,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::refinement::{
    coarsen_elements_checked, refine_elements_high_order_checked,
    refine_elements_high_order_with_history_checked,
};
use sonium_fem::mesh::{MeshlangTransferError, apply_h1_p2_state_transfer, unit_square_triangles};
use sonium_fem::mesh::{
    RefinementTransferDirection, apply_h1_p2_state_transfer_with_refinement_chain,
    apply_h1_p2_state_transfer_with_refinement_history,
};
use sonium_fem::spaces::H1Space;
use std::collections::BTreeMap;

fn node_entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, index as u64)
}

fn quadratic_field(point: sonium_fem::mesh::Point) -> f64 {
    0.75 + 1.2 * point.x - 0.8 * point.y + 1.5 * point.x * point.x - 0.4 * point.x * point.y
        + 0.6 * point.y * point.y
}

fn all_node_entities(mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<EntityId, usize> {
    (0..mesh.num_nodes())
        .map(|index| (node_entity(index), index))
        .collect()
}

#[test]
fn h1_p2_bridge_transfers_quadratic_field_after_high_order_refinement() {
    let source_mesh = unit_square_triangles(1).to_p2();
    let mut target_mesh = source_mesh.clone();
    refine_elements_high_order_checked(&mut target_mesh, &[0]).unwrap();

    let source_values = source_mesh
        .nodes
        .iter()
        .copied()
        .map(quadratic_field)
        .collect::<Vec<_>>();
    let source_entities = all_node_entities(&source_mesh);
    let target_entities = all_node_entities(&target_mesh);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..target_mesh.num_nodes())
            .map(|index| IrStateTransferEntry {
                entity: node_entity(index),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    };

    let transferred = apply_h1_p2_state_transfer::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P2),
        &source_values,
        &source_entities,
        &target_entities,
        |_, _, _| Ok(0.0),
    )
    .unwrap();

    let max_error = target_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| (transferred[&node_entity(index)] - quadratic_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_error < 1.0e-9, "P2 transfer error {max_error:e}");
}

#[test]
fn h1_p2_bridge_rejects_non_quadratic_spaces_before_callbacks() {
    let source_mesh = unit_square_triangles(1);
    let target_mesh = source_mesh.clone();
    let entities = all_node_entities(&source_mesh);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: vec![IrStateTransferEntry {
            entity: node_entity(0),
            action: IrStateTransferAction::Interpolate,
        }],
    };
    let mut callback_count = 0;

    let error = apply_h1_p2_state_transfer::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &H1Space::new(PolynomialDegree::P1),
        &H1Space::new(PolynomialDegree::P2),
        &vec![0.0; source_mesh.num_nodes()],
        &entities,
        &entities,
        |_, _, _| {
            callback_count += 1;
            Ok(0.0)
        },
    )
    .unwrap_err();

    assert_eq!(callback_count, 0);
    assert!(matches!(
        error,
        MeshlangTransferError::UnsupportedH1Degree {
            expected_degree: PolynomialDegree::P2,
            source_degree: PolynomialDegree::P1,
            target_degree: PolynomialDegree::P2,
        }
    ));
}

#[test]
fn h1_p2_bridge_transfers_through_checked_refinement_history_both_directions() {
    let coarse_mesh = unit_square_triangles(1).to_p2();
    let mut fine_mesh = coarse_mesh.clone();
    let (_, history) = refine_elements_high_order_with_history_checked(&mut fine_mesh, &[0])
        .expect("checked P2 refinement history");
    let parent_id = history.parents[0].element.id;
    let source_entities = all_node_entities(&coarse_mesh);
    let fine_entities = all_node_entities(&fine_mesh);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..fine_mesh.num_nodes())
            .map(|index| IrStateTransferEntry {
                entity: node_entity(index),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    };
    let coarse_values = coarse_mesh
        .nodes
        .iter()
        .copied()
        .map(quadratic_field)
        .collect::<Vec<_>>();
    let fine_values = apply_h1_p2_state_transfer_with_refinement_history::<&'static str, _>(
        &plan,
        &coarse_mesh,
        &fine_mesh,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P2),
        &coarse_values,
        &history,
        sonium_fem::mesh::RefinementTransferDirection::Refine,
        &source_entities,
        &fine_entities,
        |_, _, _| Ok(0.0),
    )
    .expect("P2 refinement transfer");
    let fine_vector = (0..fine_mesh.num_nodes())
        .map(|index| fine_values[&node_entity(index)])
        .collect::<Vec<_>>();
    let fine_error = fine_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| (fine_vector[index] - quadratic_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    assert!(fine_error < 1.0e-9, "P2 refinement error {fine_error:e}");

    let mut restored_mesh = fine_mesh.clone();
    coarsen_elements_checked(&mut restored_mesh, &history, &[parent_id])
        .expect("checked P2 coarsening");
    let restored_entities = restored_mesh
        .elements
        .iter()
        .flat_map(|element| element.nodes.iter().copied())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|index| (node_entity(index), index))
        .collect::<BTreeMap<_, _>>();
    let coarsen_plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: restored_entities
            .values()
            .copied()
            .map(|index| IrStateTransferEntry {
                entity: node_entity(index),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    };
    let restored_values = apply_h1_p2_state_transfer_with_refinement_history::<&'static str, _>(
        &coarsen_plan,
        &fine_mesh,
        &restored_mesh,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P2),
        &fine_vector,
        &history,
        sonium_fem::mesh::RefinementTransferDirection::Coarsen,
        &fine_entities,
        &restored_entities,
        |_, _, _| Ok(0.0),
    )
    .expect("P2 coarsening transfer");
    let restored_error = restored_entities
        .values()
        .map(|&index| {
            (restored_values[&node_entity(index)] - quadratic_field(restored_mesh.nodes[index]))
                .abs()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        restored_error < 1.0e-9,
        "P2 coarsening error {restored_error:e}"
    );
}

#[test]
fn h1_p2_bridge_transfers_two_stage_history_chain_in_both_directions() {
    let level0 = unit_square_triangles(1).to_p2();
    let mut level1 = level0.clone();
    let (_, history0) = refine_elements_high_order_with_history_checked(&mut level1, &[0])
        .expect("first P2 refinement history");
    let mut level2 = level1.clone();
    let (_, history1) = refine_elements_high_order_with_history_checked(&mut level2, &[1])
        .expect("second P2 refinement history");
    let meshes = vec![&level0, &level1, &level2];
    let space0 = H1Space::new(PolynomialDegree::P2);
    let space1 = H1Space::new(PolynomialDegree::P2);
    let space2 = H1Space::new(PolynomialDegree::P2);
    let spaces = vec![&space0, &space1, &space2];
    let entities0 = all_node_entities(&level0);
    let entities1 = all_node_entities(&level1);
    let entities2 = all_node_entities(&level2);
    let entity_maps = vec![entities0.clone(), entities1.clone(), entities2.clone()];
    let plans = IrStateTransferPlanChain::new(vec![
        IrStateTransferPlan {
            responsibility: StateTransferResponsibility::Interpolate,
            entries: (0..level1.num_nodes())
                .map(|index| IrStateTransferEntry {
                    entity: node_entity(index),
                    action: IrStateTransferAction::Interpolate,
                })
                .collect(),
        },
        IrStateTransferPlan {
            responsibility: StateTransferResponsibility::Interpolate,
            entries: (0..level2.num_nodes())
                .map(|index| IrStateTransferEntry {
                    entity: node_entity(index),
                    action: IrStateTransferAction::Interpolate,
                })
                .collect(),
        },
    ])
    .expect("valid P2 refinement plan chain");
    let histories = vec![history0.clone(), history1.clone()];
    let coarse_values = level0
        .nodes
        .iter()
        .copied()
        .map(quadratic_field)
        .collect::<Vec<_>>();
    let refined = apply_h1_p2_state_transfer_with_refinement_chain::<&'static str, _>(
        &plans,
        &meshes,
        &spaces,
        &coarse_values,
        &histories,
        RefinementTransferDirection::Refine,
        &entity_maps,
        |_, _, _, _| Ok(0.0),
    )
    .expect("two-stage P2 refinement chain");
    let refined_error = level2
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| (refined[&node_entity(index)] - quadratic_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        refined_error < 1.0e-9,
        "P2 chain refine error {refined_error:e}"
    );

    let coarsening_plans = IrStateTransferPlanChain::new(vec![
        IrStateTransferPlan {
            responsibility: StateTransferResponsibility::Interpolate,
            entries: (0..level1.num_nodes())
                .map(|index| IrStateTransferEntry {
                    entity: node_entity(index),
                    action: IrStateTransferAction::Interpolate,
                })
                .collect(),
        },
        IrStateTransferPlan {
            responsibility: StateTransferResponsibility::Interpolate,
            entries: (0..level0.num_nodes())
                .map(|index| IrStateTransferEntry {
                    entity: node_entity(index),
                    action: IrStateTransferAction::Interpolate,
                })
                .collect(),
        },
    ])
    .expect("valid P2 coarsening plan chain");
    let fine_values = (0..level2.num_nodes())
        .map(|index| refined[&node_entity(index)])
        .collect::<Vec<_>>();
    let restored = apply_h1_p2_state_transfer_with_refinement_chain::<&'static str, _>(
        &coarsening_plans,
        &meshes,
        &spaces,
        &fine_values,
        &histories,
        RefinementTransferDirection::Coarsen,
        &entity_maps,
        |_, _, _, _| Ok(0.0),
    )
    .expect("two-stage P2 coarsening chain");
    let restored_error = level0
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| (restored[&node_entity(index)] - quadratic_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        restored_error < 1.0e-9,
        "P2 chain coarsen error {restored_error:e}"
    );
}

#[test]
fn h1_p2_bridge_transfers_quadratic_field_in_reverse_layout() {
    let fine_mesh = {
        let mut mesh = unit_square_triangles(1).to_p2();
        refine_elements_high_order_checked(&mut mesh, &[0]).unwrap();
        mesh
    };
    let coarse_mesh = unit_square_triangles(1).to_p2();
    let source_entities = all_node_entities(&fine_mesh);
    let target_entities = all_node_entities(&coarse_mesh);
    let source_values = fine_mesh
        .nodes
        .iter()
        .copied()
        .map(quadratic_field)
        .collect::<Vec<_>>();
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..coarse_mesh.num_nodes())
            .map(|index| IrStateTransferEntry {
                entity: node_entity(index),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    };

    let transferred = apply_h1_p2_state_transfer::<&'static str, _>(
        &plan,
        &fine_mesh,
        &coarse_mesh,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P2),
        &source_values,
        &source_entities,
        &target_entities,
        |_, _, _| Ok(0.0),
    )
    .unwrap();

    for (index, point) in coarse_mesh.nodes.iter().enumerate() {
        assert!((transferred[&node_entity(index)] - quadratic_field(*point)).abs() < 1.0e-9);
    }
}
