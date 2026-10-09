#![cfg(feature = "meshlang")]

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::mesh::{MeshlangTransferError, apply_h1_p3_state_transfer};
use sonium_fem::spaces::H1Space;
use std::collections::BTreeMap;

fn node_entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, index as u64)
}

fn p3_triangle(edge_fraction: f64) -> Mesh {
    let mut mesh = Mesh::new(2);
    let points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (edge_fraction, 0.0),
        (1.0 - edge_fraction, 0.0),
        (1.0 - edge_fraction, edge_fraction),
        (edge_fraction, 1.0 - edge_fraction),
        (0.0, 1.0 - edge_fraction),
        (0.0, edge_fraction),
        (1.0 / 3.0, 1.0 / 3.0),
    ];
    for (x, y) in points {
        mesh.add_node(Point::new_2d(x, y));
    }
    mesh.add_element(ElementType::Triangle, (0..10).collect());
    mesh
}

fn p3_refined_triangle() -> Mesh {
    let mut mesh = Mesh::new(2);
    let cells = [
        [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)],
        [(0.5, 0.0), (1.0, 0.0), (0.5, 0.5)],
        [(0.0, 0.5), (0.5, 0.5), (0.0, 1.0)],
        [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5)],
    ];
    for [a, b, c] in cells {
        let points = [
            a,
            b,
            c,
            (a.0 + (b.0 - a.0) / 3.0, a.1 + (b.1 - a.1) / 3.0),
            (a.0 + 2.0 * (b.0 - a.0) / 3.0, a.1 + 2.0 * (b.1 - a.1) / 3.0),
            (b.0 + (c.0 - b.0) / 3.0, b.1 + (c.1 - b.1) / 3.0),
            (b.0 + 2.0 * (c.0 - b.0) / 3.0, b.1 + 2.0 * (c.1 - b.1) / 3.0),
            (c.0 + (a.0 - c.0) / 3.0, c.1 + (a.1 - c.1) / 3.0),
            (c.0 + 2.0 * (a.0 - c.0) / 3.0, c.1 + 2.0 * (a.1 - c.1) / 3.0),
            ((a.0 + b.0 + c.0) / 3.0, (a.1 + b.1 + c.1) / 3.0),
        ];
        let indices = points
            .into_iter()
            .map(|(x, y)| mesh.add_node(Point::new_2d(x, y)))
            .collect::<Vec<_>>();
        mesh.add_element(ElementType::Triangle, indices);
    }
    mesh
}

fn p3_quadrilateral(edge_fraction: f64) -> Mesh {
    let mut mesh = Mesh::new(2);
    let points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (edge_fraction, 0.0),
        (1.0 - edge_fraction, 0.0),
        (1.0, edge_fraction),
        (1.0, 1.0 - edge_fraction),
        (1.0 - edge_fraction, 1.0),
        (edge_fraction, 1.0),
        (0.0, 1.0 - edge_fraction),
        (0.0, edge_fraction),
        (edge_fraction, edge_fraction),
        (1.0 - edge_fraction, edge_fraction),
        (1.0 - edge_fraction, 1.0 - edge_fraction),
        (edge_fraction, 1.0 - edge_fraction),
    ];
    for (x, y) in points {
        mesh.add_node(Point::new_2d(x, y));
    }
    mesh.add_element(ElementType::Quadrilateral, (0..16).collect());
    mesh
}

fn cubic_field(point: Point) -> f64 {
    0.4 + 1.1 * point.x - 0.7 * point.y + 0.8 * point.x * point.x - 0.3 * point.x * point.y
        + 0.5 * point.y * point.y
        + 0.2 * point.x * point.x * point.x
        - 0.15 * point.x * point.y * point.y
}

fn entity_map(mesh: &Mesh) -> BTreeMap<EntityId, usize> {
    (0..mesh.num_nodes())
        .map(|index| (node_entity(index), index))
        .collect()
}

#[test]
fn h1_p3_bridge_transfers_cubic_field_on_distinct_straight_layouts() {
    let source_mesh = p3_triangle(1.0 / 3.0);
    let target_mesh = p3_triangle(0.25);
    let source_entities = entity_map(&source_mesh);
    let target_entities = entity_map(&target_mesh);
    let source_values = source_mesh
        .nodes
        .iter()
        .copied()
        .map(cubic_field)
        .collect::<Vec<_>>();
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..target_mesh.num_nodes())
            .map(|index| IrStateTransferEntry {
                entity: node_entity(index),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    };

    let transferred = apply_h1_p3_state_transfer::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &H1Space::new(PolynomialDegree::P3),
        &H1Space::new(PolynomialDegree::P3),
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
        .map(|(index, point)| (transferred[&node_entity(index)] - cubic_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_error < 1.0e-10, "P3 transfer error {max_error:e}");
}

#[test]
fn h1_p3_bridge_rejects_p2_space_before_callbacks() {
    let source_mesh = p3_triangle(1.0 / 3.0);
    let target_mesh = source_mesh.clone();
    let entities = entity_map(&source_mesh);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: vec![IrStateTransferEntry {
            entity: node_entity(0),
            action: IrStateTransferAction::Interpolate,
        }],
    };
    let error = apply_h1_p3_state_transfer::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P3),
        &vec![0.0; source_mesh.num_nodes()],
        &entities,
        &entities,
        |_, _, _| Ok(0.0),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        MeshlangTransferError::UnsupportedH1Degree {
            expected_degree: PolynomialDegree::P3,
            source_degree: PolynomialDegree::P2,
            target_degree: PolynomialDegree::P3,
        }
    ));
}

#[test]
fn h1_p3_bridge_transfers_cubic_field_on_distinct_quadrilateral_layouts() {
    let source_mesh = p3_quadrilateral(1.0 / 3.0);
    let target_mesh = p3_quadrilateral(0.25);
    let source_entities = entity_map(&source_mesh);
    let target_entities = entity_map(&target_mesh);
    let source_values = source_mesh
        .nodes
        .iter()
        .copied()
        .map(cubic_field)
        .collect::<Vec<_>>();
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..target_mesh.num_nodes())
            .map(|index| IrStateTransferEntry {
                entity: node_entity(index),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    };

    let transferred = apply_h1_p3_state_transfer::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &H1Space::new(PolynomialDegree::P3),
        &H1Space::new(PolynomialDegree::P3),
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
        .map(|(index, point)| (transferred[&node_entity(index)] - cubic_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_error < 1.0e-10, "Q3 transfer error {max_error:e}");
}

#[test]
fn h1_p3_bridge_transfers_cubic_field_in_reverse_layout() {
    let fine_mesh = p3_refined_triangle();
    let coarse_mesh = p3_triangle(1.0 / 3.0);
    let source_entities = entity_map(&fine_mesh);
    let target_entities = entity_map(&coarse_mesh);
    let source_values = fine_mesh
        .nodes
        .iter()
        .copied()
        .map(cubic_field)
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

    let transferred = apply_h1_p3_state_transfer::<&'static str, _>(
        &plan,
        &fine_mesh,
        &coarse_mesh,
        &H1Space::new(PolynomialDegree::P3),
        &H1Space::new(PolynomialDegree::P3),
        &source_values,
        &source_entities,
        &target_entities,
        |_, _, _| Ok(0.0),
    )
    .unwrap();

    let max_error = coarse_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| (transferred[&node_entity(index)] - cubic_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_error < 1.0e-10,
        "reverse P3 transfer error {max_error:e}"
    );
}
