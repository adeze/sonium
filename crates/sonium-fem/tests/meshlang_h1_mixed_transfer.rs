#![cfg(feature = "meshlang")]

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    IrStateTransferPlanChain, StateTransferResponsibility,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::refinement::{
    refine_elements_high_order_with_history_checked, refine_elements_with_history_checked,
};
use sonium_fem::mesh::{
    ElementType, Mesh, MeshlangTransferError, Point, RefinementTransferDirection,
    apply_h1_mixed_state_transfer, apply_h1_mixed_state_transfer_with_refinement_chain,
    apply_h1_mixed_state_transfer_with_refinement_history, unit_square_triangles,
};
use sonium_fem::spaces::H1Space;
use std::cell::Cell;
use std::collections::BTreeMap;

fn node_entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, index as u64)
}

fn all_node_entities(mesh: &Mesh) -> BTreeMap<EntityId, usize> {
    (0..mesh.num_nodes())
        .map(|index| (node_entity(index), index))
        .collect()
}

fn linear_field(point: sonium_fem::mesh::Point) -> f64 {
    0.75 + 1.2 * point.x - 0.8 * point.y
}

fn quadratic_field(point: sonium_fem::mesh::Point) -> f64 {
    0.5 + 1.2 * point.x - 0.8 * point.y + 1.5 * point.x * point.x - 0.4 * point.x * point.y
        + 0.6 * point.y * point.y
}

fn p1_triangle() -> Mesh {
    let mut mesh = Mesh::new(2);
    for (x, y) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)] {
        mesh.add_node(Point::new_2d(x, y));
    }
    mesh.add_element(ElementType::Triangle, (0..3).collect());
    mesh
}

fn p2_triangle(curvature: f64) -> Mesh {
    let mut mesh = Mesh::new(2);
    for (x, y) in [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, curvature),
        (0.5, 0.5),
        (0.0, 0.5),
    ] {
        mesh.add_node(Point::new_2d(x, y));
    }
    mesh.add_element(ElementType::Triangle, (0..6).collect());
    mesh
}

fn p3_from_p2_triangle_children(source: &Mesh) -> Mesh {
    let mut target = Mesh::new(source.dimension);
    for element in &source.elements {
        assert_eq!(element.element_type, ElementType::Triangle);
        assert!(element.nodes.len() >= ElementType::Triangle.num_nodes_p2());
        let p2_map = |lambda: [f64; 3]| {
            let [l0, l1, l2] = lambda;
            let weights = [
                l0 * (2.0 * l0 - 1.0),
                l1 * (2.0 * l1 - 1.0),
                l2 * (2.0 * l2 - 1.0),
                4.0 * l0 * l1,
                4.0 * l1 * l2,
                4.0 * l2 * l0,
            ];
            element.nodes.iter().take(6).zip(weights).fold(
                Point::new_2d(0.0, 0.0),
                |point, (&node, weight)| {
                    let source = source.nodes[node];
                    Point::new_2d(point.x + weight * source.x, point.y + weight * source.y)
                },
            )
        };
        let edge = |start: usize, end: usize, fraction: f64| {
            let mut lambda = [0.0; 3];
            lambda[start] = 1.0 - fraction;
            lambda[end] = fraction;
            p2_map(lambda)
        };
        let points = [
            p2_map([1.0, 0.0, 0.0]),
            p2_map([0.0, 1.0, 0.0]),
            p2_map([0.0, 0.0, 1.0]),
            edge(0, 1, 1.0 / 3.0),
            edge(0, 1, 2.0 / 3.0),
            edge(1, 2, 1.0 / 3.0),
            edge(1, 2, 2.0 / 3.0),
            edge(2, 0, 1.0 / 3.0),
            edge(2, 0, 2.0 / 3.0),
            p2_map([1.0 / 3.0; 3]),
        ];
        let nodes = points
            .into_iter()
            .map(|point| target.add_node(point))
            .collect::<Vec<_>>();
        let index = target.add_element(ElementType::Triangle, nodes);
        target.elements[index].id = element.id;
        target.elements[index].parent_id = element.parent_id;
        target.elements[index].level = element.level;
        target.elements[index].marker = element.marker;
    }
    target
}

fn interpolation_plan(mesh: &Mesh) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..mesh.num_nodes())
            .map(|index| IrStateTransferEntry {
                entity: node_entity(index),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

#[test]
fn h1_mixed_bridge_transfers_linear_field_from_p1_to_p2() {
    let source_mesh = unit_square_triangles(1);
    let target_mesh = unit_square_triangles(2).to_p2();
    let source_values = source_mesh
        .nodes
        .iter()
        .copied()
        .map(linear_field)
        .collect::<Vec<_>>();
    let source_entities = all_node_entities(&source_mesh);
    let target_entities = all_node_entities(&target_mesh);

    let transferred = apply_h1_mixed_state_transfer::<&'static str, _>(
        &interpolation_plan(&target_mesh),
        &source_mesh,
        &target_mesh,
        &H1Space::new(PolynomialDegree::P1),
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
        .map(|(index, point)| (transferred[&node_entity(index)] - linear_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_error < 1.0e-12, "P1 -> P2 max error: {max_error}");
}

#[test]
fn h1_mixed_bridge_transfers_quadratic_field_from_p2_to_p1() {
    let source_mesh = unit_square_triangles(2).to_p2();
    let target_mesh = unit_square_triangles(1);
    let source_values = source_mesh
        .nodes
        .iter()
        .copied()
        .map(quadratic_field)
        .collect::<Vec<_>>();
    let source_entities = all_node_entities(&source_mesh);
    let target_entities = all_node_entities(&target_mesh);

    let transferred = apply_h1_mixed_state_transfer::<&'static str, _>(
        &interpolation_plan(&target_mesh),
        &source_mesh,
        &target_mesh,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P1),
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
    assert!(max_error < 1.0e-12, "P2 -> P1 max error: {max_error}");
}

#[test]
fn h1_mixed_bridge_rejects_degree_geometry_mismatch_before_callback() {
    let source_mesh = unit_square_triangles(1);
    let target_mesh = unit_square_triangles(1);
    let source_entities = all_node_entities(&source_mesh);
    let target_entities = all_node_entities(&target_mesh);
    let plan = interpolation_plan(&target_mesh);
    let mut callback_calls = 0;

    let error = apply_h1_mixed_state_transfer::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P1),
        &vec![0.0; source_mesh.num_nodes()],
        &source_entities,
        &target_entities,
        |_, _, _| {
            callback_calls += 1;
            Ok(0.0)
        },
    )
    .unwrap_err();

    assert!(matches!(
        error,
        MeshlangTransferError::InvalidH1Space { side: "source", .. }
    ));
    assert_eq!(callback_calls, 0);
}

pub fn h1_mixed_bridge_transfers_p1_p2_adaptive_lineage_both_directions() -> (f64, f64) {
    let coarse_p1 = unit_square_triangles(1);
    let mut refined_p1 = coarse_p1.clone();
    // Convert every refined parent to P2 so no unchanged P1 neighbour is
    // silently replaced by a higher-order element in the lineage contract.
    let (_, history) = refine_elements_with_history_checked(&mut refined_p1, &[0, 1]).unwrap();
    let fine_p2 = refined_p1.to_p2();
    let coarse_values = coarse_p1
        .nodes
        .iter()
        .copied()
        .map(linear_field)
        .collect::<Vec<_>>();

    let coarse_entities = all_node_entities(&coarse_p1);
    let fine_entities = all_node_entities(&fine_p2);
    let fine_plan = interpolation_plan(&fine_p2);
    let fine_values = apply_h1_mixed_state_transfer_with_refinement_history::<String, _>(
        &fine_plan,
        &coarse_p1,
        &fine_p2,
        &H1Space::new(PolynomialDegree::P1),
        &H1Space::new(PolynomialDegree::P2),
        &coarse_values,
        &history,
        RefinementTransferDirection::Refine,
        &coarse_entities,
        &fine_entities,
        |_entity, _action, _source| {
            Err::<f64, String>("mixed adaptive interpolation invoked callback".into())
        },
    )
    .unwrap();
    let fine_error = fine_values
        .iter()
        .map(|(entity, value)| {
            let node = fine_entities[entity];
            (value - linear_field(fine_p2.nodes[node])).abs()
        })
        .fold(0.0, f64::max);
    assert!(fine_error < 1.0e-12, "P1→P2 refine error={fine_error:e}");

    let coarse_plan = interpolation_plan(&coarse_p1);
    let fine_source_values = fine_p2
        .nodes
        .iter()
        .map(|point| linear_field(*point))
        .collect::<Vec<_>>();
    let coarse_result = apply_h1_mixed_state_transfer_with_refinement_history::<String, _>(
        &coarse_plan,
        &fine_p2,
        &coarse_p1,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P1),
        &fine_source_values,
        &history,
        RefinementTransferDirection::Coarsen,
        &fine_entities,
        &coarse_entities,
        |_entity, _action, _source| {
            Err::<f64, String>("mixed adaptive coarsening invoked callback".into())
        },
    )
    .unwrap();
    let coarse_error = coarse_result
        .iter()
        .map(|(entity, value)| {
            let node = coarse_entities[entity];
            (value - linear_field(coarse_p1.nodes[node])).abs()
        })
        .fold(0.0, f64::max);
    assert!(
        coarse_error < 1.0e-12,
        "P2→P1 coarsen error={coarse_error:e}"
    );

    let mut conservative_plan = fine_plan.clone();
    for entry in &mut conservative_plan.entries {
        entry.action = IrStateTransferAction::Conservative;
    }
    let callback_calls = Cell::new(0usize);
    let rejected = apply_h1_mixed_state_transfer_with_refinement_history::<String, _>(
        &conservative_plan,
        &coarse_p1,
        &fine_p2,
        &H1Space::new(PolynomialDegree::P1),
        &H1Space::new(PolynomialDegree::P2),
        &coarse_values,
        &history,
        RefinementTransferDirection::Refine,
        &coarse_entities,
        &fine_entities,
        |_entity, _action, _source| {
            callback_calls.set(callback_calls.get() + 1);
            Ok::<f64, String>(0.0)
        },
    );
    assert!(
        rejected.is_err(),
        "scalar conservative transfer must reject"
    );
    assert_eq!(callback_calls.get(), 0);
    (fine_error, coarse_error)
}

#[test]
fn h1_mixed_bridge_transfers_p1_p2_adaptive_lineage_both_directions_test() {
    let _ = h1_mixed_bridge_transfers_p1_p2_adaptive_lineage_both_directions();
}

pub fn h1_mixed_bridge_transfers_curved_higher_order_history() -> (f64, f64) {
    let coarse_p2 = p2_triangle(0.02);
    let mut fine_p2 = coarse_p2.clone();
    let (_, history) = refine_elements_high_order_with_history_checked(&mut fine_p2, &[0])
        .expect("checked curved P2 refinement history");
    let coarse_p1 = p1_triangle();
    let coarse_p1_entities = all_node_entities(&coarse_p1);
    let fine_entities = all_node_entities(&fine_p2);
    let p1_values = coarse_p1
        .nodes
        .iter()
        .copied()
        .map(linear_field)
        .collect::<Vec<_>>();
    let fine_plan = interpolation_plan(&fine_p2);
    let refined = apply_h1_mixed_state_transfer_with_refinement_history::<String, _>(
        &fine_plan,
        &coarse_p1,
        &fine_p2,
        &H1Space::new(PolynomialDegree::P1),
        &H1Space::new(PolynomialDegree::P2),
        &p1_values,
        &history,
        RefinementTransferDirection::Refine,
        &coarse_p1_entities,
        &fine_entities,
        |_entity, _action, _source| Err::<f64, String>("unexpected callback".into()),
    )
    .expect("P1-to-curved-P2 transfer with P2 history");
    let refinement_error = refined
        .iter()
        .map(|(entity, value)| {
            let node = fine_entities[entity];
            (value - linear_field(fine_p2.nodes[node])).abs()
        })
        .fold(0.0, f64::max);

    let fine_values = fine_p2
        .nodes
        .iter()
        .copied()
        .map(linear_field)
        .collect::<Vec<_>>();
    let coarse_plan = interpolation_plan(&coarse_p1);
    let coarsened = apply_h1_mixed_state_transfer_with_refinement_history::<String, _>(
        &coarse_plan,
        &fine_p2,
        &coarse_p1,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P1),
        &fine_values,
        &history,
        RefinementTransferDirection::Coarsen,
        &fine_entities,
        &coarse_p1_entities,
        |_entity, _action, _source| Err::<f64, String>("unexpected callback".into()),
    )
    .expect("curved-P2-to-P1 transfer with P2 history");
    let coarsening_error = coarsened
        .iter()
        .map(|(entity, value)| {
            let node = coarse_p1_entities[entity];
            (value - linear_field(coarse_p1.nodes[node])).abs()
        })
        .fold(0.0, f64::max);
    assert!(
        refinement_error < 1.0e-12,
        "mixed curved refine error={refinement_error:e}"
    );
    assert!(
        coarsening_error < 1.0e-12,
        "mixed curved coarsen error={coarsening_error:e}"
    );
    (refinement_error, coarsening_error)
}

#[test]
fn h1_mixed_bridge_accepts_curved_higher_order_history() {
    h1_mixed_bridge_transfers_curved_higher_order_history();
}

#[test]
fn h1_mixed_bridge_rejects_curved_history_topology_mismatch_before_callback() {
    let source = p1_triangle();
    let mut target = p2_triangle(0.02);
    let (_, mut history) = refine_elements_high_order_with_history_checked(&mut target, &[0])
        .expect("checked curved P2 refinement history");
    history.parents[0].element.nodes[0] = 1;
    let source_entities = all_node_entities(&source);
    let target_entities = all_node_entities(&target);
    let callback_calls = Cell::new(0);
    let error = apply_h1_mixed_state_transfer_with_refinement_history::<String, _>(
        &interpolation_plan(&target),
        &source,
        &target,
        &H1Space::new(PolynomialDegree::P1),
        &H1Space::new(PolynomialDegree::P2),
        &vec![0.0; source.num_nodes()],
        &history,
        RefinementTransferDirection::Refine,
        &source_entities,
        &target_entities,
        |_entity, _action, _source| {
            callback_calls.set(callback_calls.get() + 1);
            Ok(0.0)
        },
    )
    .expect_err("mismatched mixed-order history must fail closed");
    assert!(matches!(
        error,
        MeshlangTransferError::Interpolation(
            sonium_fem::mesh::InterpolationError::RefinementHistoryMismatch { .. }
        )
    ));
    assert_eq!(callback_calls.get(), 0);
}

pub fn h1_mixed_bridge_transfers_p1_p2_p3_adaptive_chain() -> (f64, f64) {
    let level0 = p1_triangle();
    let mut level1_p1 = level0.clone();
    let (_, history0) = refine_elements_with_history_checked(&mut level1_p1, &[0])
        .expect("checked P1 refinement history");
    let level1 = level1_p1.to_p2();

    let mut level2_p2 = level1.clone();
    let selected = (0..level2_p2.num_elements()).collect::<Vec<_>>();
    let (_, history1) = refine_elements_high_order_with_history_checked(&mut level2_p2, &selected)
        .expect("checked P2 refinement history");
    let level2 = p3_from_p2_triangle_children(&level2_p2);

    let meshes = vec![&level0, &level1, &level2];
    let space0 = H1Space::new(PolynomialDegree::P1);
    let space1 = H1Space::new(PolynomialDegree::P2);
    let space2 = H1Space::new(PolynomialDegree::P3);
    let spaces = vec![&space0, &space1, &space2];
    let histories = vec![history0, history1];
    let entity_maps = meshes
        .iter()
        .map(|mesh| all_node_entities(mesh))
        .collect::<Vec<_>>();
    let refine_chain = IrStateTransferPlanChain::new(vec![
        interpolation_plan(&level1),
        interpolation_plan(&level2),
    ])
    .expect("valid mixed refinement plan chain");
    let source_values = level0
        .nodes
        .iter()
        .copied()
        .map(linear_field)
        .collect::<Vec<_>>();
    let refined = apply_h1_mixed_state_transfer_with_refinement_chain::<String, _>(
        &refine_chain,
        &meshes,
        &spaces,
        &source_values,
        &histories,
        RefinementTransferDirection::Refine,
        &entity_maps,
        |_stage, _entity, _action, _previous| Err::<f64, String>("unexpected callback".into()),
    )
    .expect("P1-to-P2-to-P3 mixed refinement chain");
    let refinement_error = refined
        .iter()
        .map(|(entity, value)| {
            let node = entity_maps[2][entity];
            (value - linear_field(level2.nodes[node])).abs()
        })
        .fold(0.0, f64::max);

    let coarsen_chain = IrStateTransferPlanChain::new(vec![
        interpolation_plan(&level1),
        interpolation_plan(&level0),
    ])
    .expect("valid mixed coarsening plan chain");
    let fine_values = level2
        .nodes
        .iter()
        .copied()
        .map(linear_field)
        .collect::<Vec<_>>();
    let coarsened = apply_h1_mixed_state_transfer_with_refinement_chain::<String, _>(
        &coarsen_chain,
        &meshes,
        &spaces,
        &fine_values,
        &histories,
        RefinementTransferDirection::Coarsen,
        &entity_maps,
        |_stage, _entity, _action, _previous| Err::<f64, String>("unexpected callback".into()),
    )
    .expect("P3-to-P2-to-P1 mixed coarsening chain");
    let coarsening_error = coarsened
        .iter()
        .map(|(entity, value)| {
            let node = entity_maps[0][entity];
            (value - linear_field(level0.nodes[node])).abs()
        })
        .fold(0.0, f64::max);
    assert!(
        refinement_error < 1.0e-12,
        "mixed chain refine error={refinement_error:e}"
    );
    assert!(
        coarsening_error < 1.0e-12,
        "mixed chain coarsen error={coarsening_error:e}"
    );
    (refinement_error, coarsening_error)
}

#[test]
fn h1_mixed_bridge_transfers_p1_p2_p3_adaptive_chain_test() {
    h1_mixed_bridge_transfers_p1_p2_p3_adaptive_chain();
}

pub fn h1_mixed_bridge_transfers_curved_p2_to_p3_mixed_state() -> f64 {
    let source = p2_triangle(0.02);
    let mut refined_p2 = source.clone();
    let (_, history) = refine_elements_high_order_with_history_checked(&mut refined_p2, &[0])
        .expect("checked curved P2 refinement history");
    let target = p3_from_p2_triangle_children(&refined_p2);
    let meshes = vec![&source, &target];
    let source_space = H1Space::new(PolynomialDegree::P2);
    let target_space = H1Space::new(PolynomialDegree::P3);
    let spaces = vec![&source_space, &target_space];
    let entity_maps = meshes
        .iter()
        .map(|mesh| all_node_entities(mesh))
        .collect::<Vec<_>>();
    let source_values = source
        .nodes
        .iter()
        .copied()
        .map(linear_field)
        .collect::<Vec<_>>();
    let plan_chain = IrStateTransferPlanChain::new(vec![interpolation_plan(&target)])
        .expect("valid curved mixed adaptive plan chain");
    let transferred = apply_h1_mixed_state_transfer_with_refinement_chain::<String, _>(
        &plan_chain,
        &meshes,
        &spaces,
        &source_values,
        &[history],
        RefinementTransferDirection::Refine,
        &entity_maps,
        |_stage, _entity, _action, _previous| Err::<f64, String>("unexpected callback".into()),
    )
    .expect("curved P2-to-P3 adaptive transfer");
    let max_error = transferred
        .iter()
        .map(|(entity, value)| {
            let node = entity_maps[1][entity];
            (value - linear_field(target.nodes[node])).abs()
        })
        .fold(0.0, f64::max);
    assert!(max_error < 1.0e-10, "curved P2→P3 error {max_error}");
    max_error
}

#[test]
fn h1_mixed_bridge_transfers_curved_p2_to_p3_mixed_state_test() {
    let _ = h1_mixed_bridge_transfers_curved_p2_to_p3_mixed_state();
}
