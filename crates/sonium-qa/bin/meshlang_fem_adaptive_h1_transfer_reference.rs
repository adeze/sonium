//! Release-facing MeshLang-to-FEM adaptive H1 state-transfer reference.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::mesh::refinement::refine_elements_with_history_checked;
use sonium_fem::mesh::{
    RefinementTransferDirection, apply_nodal_state_transfer_with_refinement_history,
    unit_square_triangles,
};
use sonium_qa::QaResult;
use std::collections::BTreeMap;
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-adaptive-h1-transfer-reference.v1";

fn point_entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, index as u64)
}

fn field_value(point: sonium_fem::mesh::Point) -> f64 {
    1.0 + 2.0 * point.x - 0.5 * point.y
}

fn run() -> Result<QaResult, String> {
    let source_mesh = unit_square_triangles(1);
    let mut target_mesh = source_mesh.clone();
    let (_, history) = refine_elements_with_history_checked(&mut target_mesh, &[0])
        .map_err(|error| format!("adaptive refinement: {error}"))?;
    let source_values = source_mesh
        .nodes
        .iter()
        .copied()
        .map(field_value)
        .collect::<Vec<_>>();
    let source_entities = (0..source_mesh.num_nodes())
        .map(|index| (point_entity(index), index))
        .collect::<BTreeMap<_, _>>();
    let target_entities = (0..target_mesh.num_nodes())
        .map(|index| (point_entity(index), index))
        .collect::<BTreeMap<_, _>>();

    let mut entries = vec![IrStateTransferEntry {
        entity: point_entity(0),
        action: IrStateTransferAction::Preserve,
    }];
    entries.extend(
        (source_mesh.num_nodes()..target_mesh.num_nodes()).map(|index| IrStateTransferEntry {
            entity: point_entity(index),
            action: IrStateTransferAction::Interpolate,
        }),
    );
    let initialized = target_mesh.num_nodes() + 1;
    entries.push(IrStateTransferEntry {
        entity: point_entity(initialized),
        action: IrStateTransferAction::Initialize,
    });
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries,
    };
    let transferred = apply_nodal_state_transfer_with_refinement_history::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &source_values,
        &history,
        RefinementTransferDirection::Refine,
        &source_entities,
        &target_entities,
        |_, action, _| match action {
            IrStateTransferAction::Initialize => Ok(42.0),
            _ => Err("unexpected callback action"),
        },
    )
    .map_err(|error| format!("adaptive MeshLang H1 transfer: {error}"))?;

    let max_error = (source_mesh.num_nodes()..target_mesh.num_nodes())
        .map(|index| {
            (transferred[&point_entity(index)] - field_value(target_mesh.nodes[index])).abs()
        })
        .fold(0.0_f64, f64::max);
    if max_error > 1.0e-12 {
        return Err(format!("adaptive H1 error {max_error} exceeds tolerance"));
    }
    if transferred[&point_entity(0)] != source_values[0]
        || transferred[&point_entity(initialized)] != 42.0
    {
        return Err("preserve/initialize action mismatch".into());
    }

    let conservative_plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Conservative,
        entries: vec![IrStateTransferEntry {
            entity: point_entity(0),
            action: IrStateTransferAction::Conservative,
        }],
    };
    let rejection = apply_nodal_state_transfer_with_refinement_history::<&'static str, _>(
        &conservative_plan,
        &source_mesh,
        &target_mesh,
        &source_values,
        &history,
        RefinementTransferDirection::Refine,
        &source_entities,
        &target_entities,
        |_, _, _| Ok(0.0),
    )
    .is_err();
    if !rejection {
        return Err("scalar conservative adaptive transfer was accepted".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("interpolated_nodes", (target_mesh.num_nodes() - source_mesh.num_nodes()) as f64)
        .minimum("interpolated_nodes", 1.0)
        .metric("max_h1_error", max_error)
        .maximum("max_h1_error", 1.0e-12)
        .metric("action_checks", 2.0)
        .minimum("action_checks", 2.0)
        .metric("rejection_checks", 1.0)
        .minimum("rejection_checks", 1.0)
        .dimension("2-D adaptive P1 triangular H1 MeshLang state transfer")
        .formulation(
            "canonical IrStateTransferPlan over checked parent/child lineage with preserve, interpolate, initialize, and typed conservative rejection",
        )
        .input_checksum("unit-square-triangles-1-selective-refine:meshlang-adaptive-h1-v1")
        .reference_checksum("analytic-linear-field:meshlang-adaptive-h1-v1")
        .seed("meshlang-fem-adaptive-h1-transfer:cpu:v1"))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(result) => result.elapsed_ms(started.elapsed().as_millis() as u64),
        Err(error) => {
            QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64)
        }
    };
    println!(
        "{}",
        result.to_json_line().expect("serialize QA result JSON")
    );
}
