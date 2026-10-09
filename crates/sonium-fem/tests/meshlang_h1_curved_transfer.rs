#![cfg(feature = "meshlang")]

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{
    ElementType, Mesh, Point, apply_h1_mixed_state_transfer, apply_h1_p2_state_transfer,
    apply_h1_p3_state_transfer,
};
use sonium_fem::spaces::H1Space;
use std::collections::BTreeMap;

fn node_entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, index as u64)
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

fn p1_triangle() -> Mesh {
    let mut mesh = Mesh::new(2);
    for (x, y) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)] {
        mesh.add_node(Point::new_2d(x, y));
    }
    mesh.add_element(ElementType::Triangle, (0..3).collect());
    mesh
}

fn p3_triangle(curvature: f64) -> Mesh {
    let mut mesh = Mesh::new(2);
    for (x, y) in [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0 / 3.0, curvature),
        (2.0 / 3.0, curvature),
        (2.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 2.0 / 3.0),
        (0.0, 2.0 / 3.0),
        (0.0, 1.0 / 3.0),
        (1.0 / 3.0, 1.0 / 3.0),
    ] {
        mesh.add_node(Point::new_2d(x, y));
    }
    mesh.add_element(ElementType::Triangle, (0..10).collect());
    mesh
}

fn entities(mesh: &Mesh) -> BTreeMap<EntityId, usize> {
    (0..mesh.num_nodes())
        .map(|index| (node_entity(index), index))
        .collect()
}

fn interpolate_plan(mesh: &Mesh) -> IrStateTransferPlan {
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

fn quadratic_field(point: Point) -> f64 {
    0.8 + 1.1 * point.x - 0.6 * point.y + 0.7 * point.x * point.x - 0.25 * point.x * point.y
        + 0.45 * point.y * point.y
}

fn cubic_field(point: Point) -> f64 {
    0.4 + 1.1 * point.x - 0.7 * point.y + 0.8 * point.x * point.x - 0.3 * point.x * point.y
        + 0.5 * point.y * point.y
        + 0.2 * point.x * point.x * point.x
        - 0.15 * point.x * point.y * point.y
}

fn linear_field(point: Point) -> f64 {
    0.75 + 1.4 * point.x - 0.9 * point.y
}

fn max_error(mesh: &Mesh, values: &[f64], field: fn(Point) -> f64) -> f64 {
    mesh.nodes
        .iter()
        .zip(values)
        .map(|(point, value)| (value - field(*point)).abs())
        .fold(0.0, f64::max)
}

pub fn meshlang_h1_bridge_transfers_curved_p2_layouts() -> (f64, f64) {
    let affine = p2_triangle(0.0);
    let curved_inward = p2_triangle(-0.02);
    let curved_outward = p2_triangle(0.02);
    let space = H1Space::new(PolynomialDegree::P2);
    let affine_entities = entities(&affine);
    let inward_entities = entities(&curved_inward);
    let outward_entities = entities(&curved_outward);
    let affine_values = affine
        .nodes
        .iter()
        .copied()
        .map(quadratic_field)
        .collect::<Vec<_>>();
    let inward_values = curved_inward
        .nodes
        .iter()
        .copied()
        .map(quadratic_field)
        .collect::<Vec<_>>();

    let affine_to_curved = apply_h1_p2_state_transfer::<String, _>(
        &interpolate_plan(&curved_outward),
        &affine,
        &curved_outward,
        &space,
        &space,
        &affine_values,
        &affine_entities,
        &outward_entities,
        |_entity, _action, _source| Ok(0.0),
    )
    .expect("affine-to-curved P2 transfer should succeed");
    let affine_to_curved_values = (0..curved_outward.num_nodes())
        .map(|index| affine_to_curved[&node_entity(index)])
        .collect::<Vec<_>>();
    let affine_to_curved_error =
        max_error(&curved_outward, &affine_to_curved_values, quadratic_field);
    assert!(
        affine_to_curved_error < 2.0e-9,
        "affine-to-curved P2 error {affine_to_curved_error}"
    );

    let curved_to_affine = apply_h1_p2_state_transfer::<String, _>(
        &interpolate_plan(&affine),
        &curved_inward,
        &affine,
        &space,
        &space,
        &inward_values,
        &inward_entities,
        &affine_entities,
        |_entity, _action, _source| Ok(0.0),
    )
    .expect("curved-to-affine P2 transfer should succeed");
    let curved_to_affine_values = (0..affine.num_nodes())
        .map(|index| curved_to_affine[&node_entity(index)])
        .collect::<Vec<_>>();
    let curved_to_affine_error = max_error(&affine, &curved_to_affine_values, quadratic_field);
    assert!(
        curved_to_affine_error < 5.0e-4,
        "curved-to-affine P2 error {curved_to_affine_error}"
    );
    (affine_to_curved_error, curved_to_affine_error)
}

pub fn meshlang_h1_bridge_transfers_curved_p3_layouts() -> (f64, f64) {
    let affine = p3_triangle(0.0);
    let curved_inward = p3_triangle(-0.02);
    let curved_outward = p3_triangle(0.02);
    let space = H1Space::new(PolynomialDegree::P3);
    let affine_entities = entities(&affine);
    let inward_entities = entities(&curved_inward);
    let outward_entities = entities(&curved_outward);
    let affine_values = affine
        .nodes
        .iter()
        .copied()
        .map(cubic_field)
        .collect::<Vec<_>>();
    let inward_values = curved_inward
        .nodes
        .iter()
        .copied()
        .map(cubic_field)
        .collect::<Vec<_>>();

    let affine_to_curved = apply_h1_p3_state_transfer::<String, _>(
        &interpolate_plan(&curved_outward),
        &affine,
        &curved_outward,
        &space,
        &space,
        &affine_values,
        &affine_entities,
        &outward_entities,
        |_entity, _action, _source| Ok(0.0),
    )
    .expect("affine-to-curved P3 transfer should succeed");
    let affine_to_curved_values = (0..curved_outward.num_nodes())
        .map(|index| affine_to_curved[&node_entity(index)])
        .collect::<Vec<_>>();
    let affine_to_curved_error = max_error(&curved_outward, &affine_to_curved_values, cubic_field);
    assert!(
        affine_to_curved_error < 3.0e-8,
        "affine-to-curved P3 error {affine_to_curved_error}"
    );

    let curved_to_affine = apply_h1_p3_state_transfer::<String, _>(
        &interpolate_plan(&affine),
        &curved_inward,
        &affine,
        &space,
        &space,
        &inward_values,
        &inward_entities,
        &affine_entities,
        |_entity, _action, _source| Ok(0.0),
    )
    .expect("curved-to-affine P3 transfer should succeed");
    let curved_to_affine_values = (0..affine.num_nodes())
        .map(|index| curved_to_affine[&node_entity(index)])
        .collect::<Vec<_>>();
    let curved_to_affine_error = max_error(&affine, &curved_to_affine_values, cubic_field);
    assert!(
        curved_to_affine_error < 5.0e-4,
        "curved-to-affine P3 error {curved_to_affine_error}"
    );
    (affine_to_curved_error, curved_to_affine_error)
}

#[test]
fn meshlang_h1_curved_p2_bridge_regression() {
    meshlang_h1_bridge_transfers_curved_p2_layouts();
}

#[test]
fn meshlang_h1_curved_p3_bridge_regression() {
    meshlang_h1_bridge_transfers_curved_p3_layouts();
}
pub fn meshlang_h1_bridge_transfers_mixed_curved_layouts() -> (f64, f64) {
    let affine_p1 = p1_triangle();
    let curved_p2 = p2_triangle(0.02);
    let curved_p2_inward = p2_triangle(-0.02);
    let p1_space = H1Space::new(PolynomialDegree::P1);
    let p2_space = H1Space::new(PolynomialDegree::P2);
    let p1_entities = entities(&affine_p1);
    let p2_entities = entities(&curved_p2);
    let p2_inward_entities = entities(&curved_p2_inward);
    let p1_values = affine_p1
        .nodes
        .iter()
        .copied()
        .map(linear_field)
        .collect::<Vec<_>>();
    let p2_inward_values = curved_p2_inward
        .nodes
        .iter()
        .copied()
        .map(linear_field)
        .collect::<Vec<_>>();

    let p1_to_curved_p2 = apply_h1_mixed_state_transfer::<String, _>(
        &interpolate_plan(&curved_p2),
        &affine_p1,
        &curved_p2,
        &p1_space,
        &p2_space,
        &p1_values,
        &p1_entities,
        &p2_entities,
        |_entity, _action, _source| Ok(0.0),
    )
    .expect("P1-to-curved-P2 transfer should succeed");
    let p1_to_curved_values = (0..curved_p2.num_nodes())
        .map(|index| p1_to_curved_p2[&node_entity(index)])
        .collect::<Vec<_>>();
    let p1_to_curved_error = max_error(&curved_p2, &p1_to_curved_values, linear_field);
    assert!(
        p1_to_curved_error < 2.0e-9,
        "P1-to-curved-P2 error {p1_to_curved_error}"
    );

    let curved_p2_to_p1 = apply_h1_mixed_state_transfer::<String, _>(
        &interpolate_plan(&affine_p1),
        &curved_p2_inward,
        &affine_p1,
        &p2_space,
        &p1_space,
        &p2_inward_values,
        &p2_inward_entities,
        &p1_entities,
        |_entity, _action, _source| Ok(0.0),
    )
    .expect("curved-P2-to-P1 transfer should succeed");
    let curved_p2_to_p1_values = (0..affine_p1.num_nodes())
        .map(|index| curved_p2_to_p1[&node_entity(index)])
        .collect::<Vec<_>>();
    let curved_p2_to_p1_error = max_error(&affine_p1, &curved_p2_to_p1_values, linear_field);
    assert!(
        curved_p2_to_p1_error < 2.0e-9,
        "curved-P2-to-P1 error {curved_p2_to_p1_error}"
    );
    (p1_to_curved_error, curved_p2_to_p1_error)
}

#[test]
fn meshlang_h1_mixed_curved_bridge_regression() {
    meshlang_h1_bridge_transfers_mixed_curved_layouts();
}
