#![cfg(feature = "meshlang")]

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    IrStateTransferPlanChain, StateTransferResponsibility,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::refinement::{
    coarsen_elements_checked, refine_elements_high_order_with_history_checked,
};
use sonium_fem::mesh::{
    ElementType, Mesh, MeshlangTransferError, Point, RefinementTransferDirection,
    apply_h1_p3_state_transfer_with_refinement_chain,
    apply_h1_p3_state_transfer_with_refinement_history,
};
use sonium_fem::spaces::H1Space;
use std::collections::{BTreeMap, BTreeSet};

fn node_entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, index as u64)
}

fn p3_triangle() -> Mesh {
    let mut mesh = Mesh::new(2);
    let points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0 / 3.0, 0.0),
        (2.0 / 3.0, 0.0),
        (2.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 2.0 / 3.0),
        (0.0, 2.0 / 3.0),
        (0.0, 1.0 / 3.0),
        (1.0 / 3.0, 1.0 / 3.0),
    ];
    for (x, y) in points {
        mesh.add_node(Point::new_2d(x, y));
    }
    mesh.add_element(sonium_fem::mesh::ElementType::Triangle, (0..10).collect());
    mesh
}

fn p3_quadrilateral() -> Mesh {
    let mut mesh = Mesh::new(2);
    let points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (1.0 / 3.0, 0.0),
        (2.0 / 3.0, 0.0),
        (1.0, 1.0 / 3.0),
        (1.0, 2.0 / 3.0),
        (2.0 / 3.0, 1.0),
        (1.0 / 3.0, 1.0),
        (0.0, 2.0 / 3.0),
        (0.0, 1.0 / 3.0),
        (1.0 / 3.0, 1.0 / 3.0),
        (2.0 / 3.0, 1.0 / 3.0),
        (2.0 / 3.0, 2.0 / 3.0),
        (1.0 / 3.0, 2.0 / 3.0),
    ];
    for (x, y) in points {
        mesh.add_node(Point::new_2d(x, y));
    }
    mesh.add_element(ElementType::Quadrilateral, (0..16).collect());
    mesh
}

fn p3_tetrahedron() -> Mesh {
    let mut mesh = Mesh::new(3);
    let barycentric_nodes = [
        [3, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 3],
        [2, 1, 0, 0],
        [1, 2, 0, 0],
        [2, 0, 1, 0],
        [1, 0, 2, 0],
        [2, 0, 0, 1],
        [1, 0, 0, 2],
        [0, 2, 1, 0],
        [0, 1, 2, 0],
        [0, 2, 0, 1],
        [0, 1, 0, 2],
        [0, 0, 2, 1],
        [0, 0, 1, 2],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
    ];
    for barycentric in barycentric_nodes {
        mesh.add_node(Point::new_3d(
            barycentric[1] as f64 / 3.0,
            barycentric[2] as f64 / 3.0,
            barycentric[3] as f64 / 3.0,
        ));
    }
    mesh.add_element(ElementType::Tetrahedron, (0..20).collect());
    mesh
}

fn p3_hexahedron() -> Mesh {
    const LATTICE: [[usize; 3]; 64] = [
        [0, 0, 0],
        [3, 0, 0],
        [3, 3, 0],
        [0, 3, 0],
        [0, 0, 3],
        [3, 0, 3],
        [3, 3, 3],
        [0, 3, 3],
        [1, 0, 0],
        [2, 0, 0],
        [3, 1, 0],
        [3, 2, 0],
        [2, 3, 0],
        [1, 3, 0],
        [0, 2, 0],
        [0, 1, 0],
        [1, 0, 3],
        [2, 0, 3],
        [3, 1, 3],
        [3, 2, 3],
        [2, 3, 3],
        [1, 3, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 0, 1],
        [0, 0, 2],
        [3, 0, 1],
        [3, 0, 2],
        [3, 3, 1],
        [3, 3, 2],
        [0, 3, 1],
        [0, 3, 2],
        [1, 1, 0],
        [2, 1, 0],
        [2, 2, 0],
        [1, 2, 0],
        [1, 1, 3],
        [2, 1, 3],
        [2, 2, 3],
        [1, 2, 3],
        [1, 0, 1],
        [2, 0, 1],
        [2, 0, 2],
        [1, 0, 2],
        [1, 3, 1],
        [2, 3, 1],
        [2, 3, 2],
        [1, 3, 2],
        [0, 1, 1],
        [0, 2, 1],
        [0, 2, 2],
        [0, 1, 2],
        [3, 1, 1],
        [3, 2, 1],
        [3, 2, 2],
        [3, 1, 2],
        [1, 1, 1],
        [2, 1, 1],
        [2, 2, 1],
        [1, 2, 1],
        [1, 1, 2],
        [2, 1, 2],
        [2, 2, 2],
        [1, 2, 2],
    ];
    let mut mesh = Mesh::new(3);
    for [i, j, k] in LATTICE {
        mesh.add_node(Point::new_3d(
            i as f64 / 3.0,
            j as f64 / 3.0,
            k as f64 / 3.0,
        ));
    }
    mesh.add_element(ElementType::Hexahedron, (0..64).collect());
    mesh
}

fn cubic_field(point: Point) -> f64 {
    0.4 + 1.1 * point.x - 0.7 * point.y + 0.8 * point.x * point.x - 0.3 * point.x * point.y
        + 0.5 * point.y * point.y
        + 0.2 * point.x * point.x * point.x
        - 0.15 * point.x * point.y * point.y
}

fn all_node_entities(mesh: &Mesh) -> BTreeMap<EntityId, usize> {
    (0..mesh.num_nodes())
        .map(|index| (node_entity(index), index))
        .collect()
}

fn active_node_entities(mesh: &Mesh) -> BTreeMap<EntityId, usize> {
    mesh.elements
        .iter()
        .flat_map(|element| element.nodes.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|index| (node_entity(index), index))
        .collect()
}

fn interpolation_plan(entities: &BTreeMap<EntityId, usize>) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: entities
            .keys()
            .copied()
            .map(|entity| IrStateTransferEntry {
                entity,
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

#[test]
fn h1_p3_bridge_transfers_cubic_field_across_refine_and_coarsen_history() {
    let coarse_mesh = p3_triangle();
    let mut fine_mesh = coarse_mesh.clone();
    let (_, history) = refine_elements_high_order_with_history_checked(&mut fine_mesh, &[0])
        .expect("checked P3 refinement");
    let source_entities = all_node_entities(&coarse_mesh);
    let fine_entities = all_node_entities(&fine_mesh);
    let source_values = coarse_mesh
        .nodes
        .iter()
        .copied()
        .map(cubic_field)
        .collect::<Vec<_>>();
    let plan = interpolation_plan(&fine_entities);

    let fine = apply_h1_p3_state_transfer_with_refinement_history::<&'static str, _>(
        &plan,
        &coarse_mesh,
        &fine_mesh,
        &H1Space::new(PolynomialDegree::P3),
        &H1Space::new(PolynomialDegree::P3),
        &source_values,
        &history,
        RefinementTransferDirection::Refine,
        &source_entities,
        &fine_entities,
        |_, _, _| Ok(0.0),
    )
    .expect("P3 refinement transfer");
    let fine_error = fine_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| (fine[&node_entity(index)] - cubic_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    assert!(fine_error < 1.0e-9, "P3 refinement error {fine_error:e}");

    let mut restored_mesh = fine_mesh.clone();
    let parent_id = history.parents[0].element.id;
    coarsen_elements_checked(&mut restored_mesh, &history, &[parent_id])
        .expect("checked P3 coarsening");
    let restored_entities = active_node_entities(&restored_mesh);
    let restored_plan = interpolation_plan(&restored_entities);
    let fine_values = (0..fine_mesh.num_nodes())
        .map(|index| fine[&node_entity(index)])
        .collect::<Vec<_>>();
    let restored = apply_h1_p3_state_transfer_with_refinement_history::<&'static str, _>(
        &restored_plan,
        &fine_mesh,
        &restored_mesh,
        &H1Space::new(PolynomialDegree::P3),
        &H1Space::new(PolynomialDegree::P3),
        &fine_values,
        &history,
        RefinementTransferDirection::Coarsen,
        &fine_entities,
        &restored_entities,
        |_, _, _| Ok(0.0),
    )
    .expect("P3 coarsening transfer");
    let restored_error = restored_entities
        .values()
        .map(|&index| {
            (restored[&node_entity(index)] - cubic_field(restored_mesh.nodes[index])).abs()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        restored_error < 1.0e-9,
        "P3 coarsening error {restored_error:e}"
    );
}

#[test]
fn h1_p3_bridge_transfers_cubic_field_across_q3_refine_and_coarsen_history() {
    let coarse_mesh = p3_quadrilateral();
    let mut fine_mesh = coarse_mesh.clone();
    let (_, history) = refine_elements_high_order_with_history_checked(&mut fine_mesh, &[0])
        .expect("checked Q3 refinement");
    let source_entities = all_node_entities(&coarse_mesh);
    let fine_entities = all_node_entities(&fine_mesh);
    let source_values = coarse_mesh
        .nodes
        .iter()
        .copied()
        .map(cubic_field)
        .collect::<Vec<_>>();
    let fine = apply_h1_p3_state_transfer_with_refinement_history::<&'static str, _>(
        &interpolation_plan(&fine_entities),
        &coarse_mesh,
        &fine_mesh,
        &H1Space::new(PolynomialDegree::P3),
        &H1Space::new(PolynomialDegree::P3),
        &source_values,
        &history,
        RefinementTransferDirection::Refine,
        &source_entities,
        &fine_entities,
        |_, _, _| Ok(0.0),
    )
    .expect("Q3 refinement transfer");
    let refinement_error = fine_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(node, point)| (fine[&node_entity(node)] - cubic_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        refinement_error < 1.0e-9,
        "Q3 refinement error {refinement_error:e}"
    );

    let refined_mesh = fine_mesh.clone();
    let parent_id = history.parents[0].element.id;
    coarsen_elements_checked(&mut fine_mesh, &history, &[parent_id])
        .expect("checked Q3 coarsening");
    let restored_entities = active_node_entities(&fine_mesh);
    let fine_values = (0..refined_mesh.num_nodes())
        .map(|node| fine[&node_entity(node)])
        .collect::<Vec<_>>();
    let restored = apply_h1_p3_state_transfer_with_refinement_history::<&'static str, _>(
        &interpolation_plan(&restored_entities),
        &refined_mesh,
        &fine_mesh,
        &H1Space::new(PolynomialDegree::P3),
        &H1Space::new(PolynomialDegree::P3),
        &fine_values,
        &history,
        RefinementTransferDirection::Coarsen,
        &all_node_entities(&refined_mesh),
        &restored_entities,
        |_, _, _| Ok(0.0),
    )
    .expect("Q3 coarsening transfer");
    let coarsening_error = restored_entities
        .values()
        .map(|&node| (restored[&node_entity(node)] - cubic_field(fine_mesh.nodes[node])).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        coarsening_error < 1.0e-9,
        "Q3 coarsening error {coarsening_error:e}"
    );
}

#[test]
fn h1_p3_bridge_transfers_cubic_field_across_p3_tetra_refine_and_coarsen_history() {
    let coarse_mesh = p3_tetrahedron();
    let mut fine_mesh = coarse_mesh.clone();
    let (_, history) = refine_elements_high_order_with_history_checked(&mut fine_mesh, &[0])
        .expect("checked P3 tetrahedral refinement");
    let source_entities = all_node_entities(&coarse_mesh);
    let fine_entities = all_node_entities(&fine_mesh);
    let source_values = coarse_mesh
        .nodes
        .iter()
        .copied()
        .map(cubic_field)
        .collect::<Vec<_>>();
    let fine = apply_h1_p3_state_transfer_with_refinement_history::<&'static str, _>(
        &interpolation_plan(&fine_entities),
        &coarse_mesh,
        &fine_mesh,
        &H1Space::new(PolynomialDegree::P3),
        &H1Space::new(PolynomialDegree::P3),
        &source_values,
        &history,
        RefinementTransferDirection::Refine,
        &source_entities,
        &fine_entities,
        |_, _, _| Ok(0.0),
    )
    .expect("P3 tetrahedral refinement transfer");
    let refinement_error = fine_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(node, point)| (fine[&node_entity(node)] - cubic_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        refinement_error < 1.0e-9,
        "P3 tetrahedral refinement error {refinement_error:e}"
    );

    let refined_mesh = fine_mesh.clone();
    let parent_id = history.parents[0].element.id;
    coarsen_elements_checked(&mut fine_mesh, &history, &[parent_id])
        .expect("checked P3 tetrahedral coarsening");
    let restored_entities = active_node_entities(&fine_mesh);
    let fine_values = (0..refined_mesh.num_nodes())
        .map(|node| fine[&node_entity(node)])
        .collect::<Vec<_>>();
    let restored = apply_h1_p3_state_transfer_with_refinement_history::<&'static str, _>(
        &interpolation_plan(&restored_entities),
        &refined_mesh,
        &fine_mesh,
        &H1Space::new(PolynomialDegree::P3),
        &H1Space::new(PolynomialDegree::P3),
        &fine_values,
        &history,
        RefinementTransferDirection::Coarsen,
        &all_node_entities(&refined_mesh),
        &restored_entities,
        |_, _, _| Ok(0.0),
    )
    .expect("P3 tetrahedral coarsening transfer");
    let coarsening_error = restored_entities
        .values()
        .map(|&node| (restored[&node_entity(node)] - cubic_field(fine_mesh.nodes[node])).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        coarsening_error < 1.0e-9,
        "P3 tetrahedral coarsening error {coarsening_error:e}"
    );
}

#[test]
fn h1_p3_bridge_transfers_two_stage_adaptive_chain_in_both_directions() {
    let level0 = p3_triangle();
    let mut level1 = level0.clone();
    let (_, history0) = refine_elements_high_order_with_history_checked(&mut level1, &[0])
        .expect("first checked P3 refinement");
    let mut level2 = level1.clone();
    let (_, history1) = refine_elements_high_order_with_history_checked(&mut level2, &[0])
        .expect("second checked P3 refinement");

    let meshes = [&level0, &level1, &level2];
    let space0 = H1Space::new(PolynomialDegree::P3);
    let space1 = H1Space::new(PolynomialDegree::P3);
    let space2 = H1Space::new(PolynomialDegree::P3);
    let spaces = [&space0, &space1, &space2];
    let entity_maps = meshes
        .iter()
        .map(|mesh| all_node_entities(mesh))
        .collect::<Vec<_>>();
    let plans = IrStateTransferPlanChain::new(vec![
        interpolation_plan(&entity_maps[1]),
        interpolation_plan(&entity_maps[2]),
    ])
    .expect("valid P3 refinement plan chain");
    let coarse_values = level0
        .nodes
        .iter()
        .copied()
        .map(cubic_field)
        .collect::<Vec<_>>();
    let refined = apply_h1_p3_state_transfer_with_refinement_chain::<&'static str, _>(
        &plans,
        &meshes,
        &spaces,
        &coarse_values,
        &[history0.clone(), history1.clone()],
        RefinementTransferDirection::Refine,
        &entity_maps,
        |_, _, _, _| Ok(0.0),
    )
    .expect("two-stage P3 refinement chain");
    let refined_error = level2
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| (refined[&node_entity(index)] - cubic_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        refined_error < 1.0e-9,
        "P3 chain refine error {refined_error:e}"
    );

    let reverse_plans = IrStateTransferPlanChain::new(vec![
        interpolation_plan(&entity_maps[1]),
        interpolation_plan(&entity_maps[0]),
    ])
    .expect("valid P3 coarsening plan chain");
    let fine_values = (0..level2.num_nodes())
        .map(|index| refined[&node_entity(index)])
        .collect::<Vec<_>>();
    let restored = apply_h1_p3_state_transfer_with_refinement_chain::<&'static str, _>(
        &reverse_plans,
        &meshes,
        &spaces,
        &fine_values,
        &[history0, history1],
        RefinementTransferDirection::Coarsen,
        &entity_maps,
        |_, _, _, _| Ok(0.0),
    )
    .expect("two-stage P3 coarsening chain");
    let restored_error = level0
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| (restored[&node_entity(index)] - cubic_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        restored_error < 1.0e-9,
        "P3 chain coarsen error {restored_error:e}"
    );
}

#[test]
fn h1_p3_bridge_transfers_cubic_field_across_p3_hexa_refine_and_coarsen_history() {
    let coarse_mesh = p3_hexahedron();
    let mut fine_mesh = coarse_mesh.clone();
    let (_, history) = refine_elements_high_order_with_history_checked(&mut fine_mesh, &[0])
        .expect("checked P3 hexahedral refinement");
    let source_entities = all_node_entities(&coarse_mesh);
    let fine_entities = all_node_entities(&fine_mesh);
    let source_values = coarse_mesh
        .nodes
        .iter()
        .copied()
        .map(cubic_field)
        .collect::<Vec<_>>();
    let fine = apply_h1_p3_state_transfer_with_refinement_history::<&'static str, _>(
        &interpolation_plan(&fine_entities),
        &coarse_mesh,
        &fine_mesh,
        &H1Space::new(PolynomialDegree::P3),
        &H1Space::new(PolynomialDegree::P3),
        &source_values,
        &history,
        RefinementTransferDirection::Refine,
        &source_entities,
        &fine_entities,
        |_, _, _| Ok(0.0),
    )
    .expect("P3 hexahedral refinement transfer");
    let refinement_error = fine_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(node, point)| (fine[&node_entity(node)] - cubic_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        refinement_error < 1.0e-9,
        "P3 hexahedral refinement error {refinement_error:e}"
    );

    let refined_mesh = fine_mesh.clone();
    let parent_id = history.parents[0].element.id;
    coarsen_elements_checked(&mut fine_mesh, &history, &[parent_id])
        .expect("checked P3 hexahedral coarsening");
    let restored_entities = active_node_entities(&fine_mesh);
    let fine_values = (0..refined_mesh.num_nodes())
        .map(|node| fine[&node_entity(node)])
        .collect::<Vec<_>>();
    let restored = apply_h1_p3_state_transfer_with_refinement_history::<&'static str, _>(
        &interpolation_plan(&restored_entities),
        &refined_mesh,
        &fine_mesh,
        &H1Space::new(PolynomialDegree::P3),
        &H1Space::new(PolynomialDegree::P3),
        &fine_values,
        &history,
        RefinementTransferDirection::Coarsen,
        &all_node_entities(&refined_mesh),
        &restored_entities,
        |_, _, _| Ok(0.0),
    )
    .expect("P3 hexahedral coarsening transfer");
    let coarsening_error = restored_entities
        .values()
        .map(|&node| (restored[&node_entity(node)] - cubic_field(fine_mesh.nodes[node])).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        coarsening_error < 1.0e-9,
        "P3 hexahedral coarsening error {coarsening_error:e}"
    );
}

#[test]
fn h1_p3_adaptive_bridge_rejects_non_p3_space_before_callbacks() {
    let source_mesh = p3_triangle();
    let target_mesh = source_mesh.clone();
    let entities = all_node_entities(&source_mesh);
    let (_, history) = {
        let mut mesh = source_mesh.clone();
        refine_elements_high_order_with_history_checked(&mut mesh, &[0]).unwrap()
    };
    let plan = interpolation_plan(&entities);
    let error = apply_h1_p3_state_transfer_with_refinement_history::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P3),
        &vec![0.0; source_mesh.num_nodes()],
        &history,
        RefinementTransferDirection::Refine,
        &entities,
        &entities,
        |_, _, _| Err("callback must not run"),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        MeshlangTransferError::UnsupportedH1Degree { .. }
    ));
}
