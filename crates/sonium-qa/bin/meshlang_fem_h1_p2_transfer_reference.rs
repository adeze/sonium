//! CPU-only MeshLang-to-FEM quadratic H1 state-transfer reference.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::refinement::refine_elements_high_order_checked;
use sonium_fem::mesh::{MeshlangTransferError, apply_h1_p2_state_transfer, unit_square_triangles};
use sonium_fem::spaces::H1Space;
use sonium_qa::QaResult;
use std::collections::BTreeMap;
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-h1-p2-transfer-reference.v1";

fn node_entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, index as u64)
}

fn quadratic_field(point: sonium_fem::mesh::Point) -> f64 {
    0.75 + 1.2 * point.x - 0.8 * point.y + 1.5 * point.x * point.x - 0.4 * point.x * point.y
        + 0.6 * point.y * point.y
}

fn entity_map(mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<EntityId, usize> {
    (0..mesh.num_nodes())
        .map(|index| (node_entity(index), index))
        .collect()
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let source_mesh = unit_square_triangles(1).to_p2();
    let mut target_mesh = source_mesh.clone();
    refine_elements_high_order_checked(&mut target_mesh, &[0])
        .map_err(|error| format!("P2 refinement: {error}"))?;
    let source_values = source_mesh
        .nodes
        .iter()
        .copied()
        .map(quadratic_field)
        .collect::<Vec<_>>();
    let source_entities = entity_map(&source_mesh);
    let target_entities = entity_map(&target_mesh);
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
    .map_err(|error| format!("P2 transfer: {error}"))?;
    let max_error = target_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| (transferred[&node_entity(index)] - quadratic_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    if max_error > 1.0e-9 {
        return Err(format!("P2 transfer error {max_error:e}"));
    }

    let rejected = apply_h1_p2_state_transfer::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &H1Space::new(PolynomialDegree::P1),
        &H1Space::new(PolynomialDegree::P2),
        &source_values,
        &source_entities,
        &target_entities,
        |_, _, _| Ok(0.0),
    );
    if !matches!(
        rejected,
        Err(MeshlangTransferError::UnsupportedH1Degree {
            expected_degree: PolynomialDegree::P2,
            source_degree: PolynomialDegree::P1,
            target_degree: PolynomialDegree::P2,
        })
    ) {
        return Err("non-quadratic H1 space was not rejected".into());
    }

    let reverse_plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..source_mesh.num_nodes())
            .map(|index| IrStateTransferEntry {
                entity: node_entity(index),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    };
    let reverse_entities = entity_map(&target_mesh);
    let reverse_values = target_mesh
        .nodes
        .iter()
        .copied()
        .map(quadratic_field)
        .collect::<Vec<_>>();
    let reversed = apply_h1_p2_state_transfer::<&'static str, _>(
        &reverse_plan,
        &target_mesh,
        &source_mesh,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P2),
        &reverse_values,
        &reverse_entities,
        &source_entities,
        |_, _, _| Ok(0.0),
    )
    .map_err(|error| format!("P2 reverse transfer: {error}"))?;
    let max_reverse_error = source_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| (reversed[&node_entity(index)] - quadratic_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    if max_reverse_error > 1.0e-9 {
        return Err(format!("P2 reverse transfer error {max_reverse_error:e}"));
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("source_nodes", source_mesh.num_nodes() as f64)
        .metric("target_nodes", target_mesh.num_nodes() as f64)
        .metric("max_h1_p2_error", max_error)
        .metric("max_reverse_h1_p2_error", max_reverse_error)
        .metric("rejection_checks", 1.0)
        .metric("reverse_transfer_checks", 1.0)
        .dimension("2-D quadratic H1 MeshLang transfer")
        .input_checksum("unit-square-triangles-p2-selective-refine:meshlang-h1-p2-v1")
        .reference_checksum("analytic-quadratic-field:meshlang-h1-p2-v1")
        .seed("meshlang-fem-h1-p2-transfer:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA JSON"));
}
