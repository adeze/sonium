//! CPU-only MeshLang-to-FEM cubic H1 state-transfer reference.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::MeshlangTransferError;
use sonium_fem::mesh::{ElementType, Mesh, Point, apply_h1_p3_state_transfer};
use sonium_fem::spaces::H1Space;
use sonium_qa::QaResult;
use std::collections::BTreeMap;
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-h1-p3-transfer-reference.v1";

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

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
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
    .map_err(|error| format!("P3 transfer: {error}"))?;
    let max_error = target_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| (transferred[&node_entity(index)] - cubic_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    if max_error > 1.0e-10 {
        return Err(format!("P3 transfer error {max_error:e}"));
    }

    let rejected = apply_h1_p3_state_transfer::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P3),
        &source_values,
        &source_entities,
        &target_entities,
        |_, _, _| Ok(0.0),
    );
    if !matches!(
        rejected,
        Err(MeshlangTransferError::UnsupportedH1Degree {
            expected_degree: PolynomialDegree::P3,
            source_degree: PolynomialDegree::P2,
            target_degree: PolynomialDegree::P3,
        })
    ) {
        return Err("non-cubic H1 space was not rejected".into());
    }

    let reverse_source_mesh = p3_refined_triangle();
    let reverse_target_mesh = p3_triangle(1.0 / 3.0);
    let reverse_source_entities = entity_map(&reverse_source_mesh);
    let reverse_target_entities = entity_map(&reverse_target_mesh);
    let reverse_values = reverse_source_mesh
        .nodes
        .iter()
        .copied()
        .map(cubic_field)
        .collect::<Vec<_>>();
    let reverse_plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..reverse_target_mesh.num_nodes())
            .map(|index| IrStateTransferEntry {
                entity: node_entity(index),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    };
    let reversed = apply_h1_p3_state_transfer::<&'static str, _>(
        &reverse_plan,
        &reverse_source_mesh,
        &reverse_target_mesh,
        &H1Space::new(PolynomialDegree::P3),
        &H1Space::new(PolynomialDegree::P3),
        &reverse_values,
        &reverse_source_entities,
        &reverse_target_entities,
        |_, _, _| Ok(0.0),
    )
    .map_err(|error| format!("P3 reverse transfer: {error}"))?;
    let max_reverse_error = reverse_target_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| (reversed[&node_entity(index)] - cubic_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    if max_reverse_error > 1.0e-10 {
        return Err(format!("P3 reverse transfer error {max_reverse_error:e}"));
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("source_nodes", source_mesh.num_nodes() as f64)
        .metric("target_nodes", target_mesh.num_nodes() as f64)
        .metric("max_h1_p3_error", max_error)
        .metric("max_reverse_h1_p3_error", max_reverse_error)
        .metric("rejection_checks", 1.0)
        .metric("reverse_transfer_checks", 1.0)
        .dimension("2-D cubic H1 MeshLang transfer")
        .input_checksum("unit-triangle-p3-layouts:meshlang-h1-p3-v1")
        .reference_checksum("analytic-cubic-field:meshlang-h1-p3-v1")
        .seed("meshlang-fem-h1-p3-transfer:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA JSON"));
}
