//! Release-facing MeshLang entity state-transfer plan reference.

use meshlang_ast::{
    CanonicalGeometryIr, EntityDimension, EntityId, GeometryNode, GeometryOperation,
    IrStateTransferAction, PrimitiveKind, RemeshResponsibility, SensitivityEntry, SourceMetadata,
    StateTransferResponsibility, SymbolId,
};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "meshlang.state-transfer-reference.v1";

fn node(id: EntityId, kind: PrimitiveKind) -> GeometryNode {
    GeometryNode {
        id,
        name: format!("fixture::entity_{}", id.index),
        operation: GeometryOperation::Primitive { kind },
        inputs: Vec::new(),
        arguments: Vec::new(),
        provenance: Vec::new(),
    }
}

fn fixture(nodes: &[(u64, PrimitiveKind)]) -> CanonicalGeometryIr {
    let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
    ir.entities.extend(
        nodes
            .iter()
            .map(|(index, kind)| node(EntityId::new(EntityDimension::Volume, *index), *kind)),
    );
    ir
}

fn run() -> Result<QaResult, String> {
    let old = fixture(&[(1, PrimitiveKind::Box), (2, PrimitiveKind::Sphere)]);
    let next = fixture(&[(1, PrimitiveKind::Box), (3, PrimitiveKind::Cylinder)]);
    let plan = old.state_transfer_plan(&next);
    plan.validate().map_err(str::to_owned)?;
    if plan.responsibility != StateTransferResponsibility::ApplicationDefined
        || plan.action_for(EntityId::new(EntityDimension::Volume, 1))
            != Some(IrStateTransferAction::Preserve)
        || plan.action_for(EntityId::new(EntityDimension::Volume, 2))
            != Some(IrStateTransferAction::Drop)
        || plan.action_for(EntityId::new(EntityDimension::Volume, 3))
            != Some(IrStateTransferAction::ApplicationDefined)
    {
        return Err("preserve/drop/application-defined plan actions diverged".into());
    }

    let mut sensitive_old = fixture(&[(1, PrimitiveKind::Box)]);
    let sensitive_next = fixture(&[(1, PrimitiveKind::Sphere)]);
    sensitive_old.sensitivity.entries.push(SensitivityEntry {
        variable: SymbolId(7),
        parameters: Vec::new(),
        entities: vec![EntityId::new(EntityDimension::Volume, 1)],
        physical_groups: Vec::new(),
        remesh: RemeshResponsibility::Global,
        state_transfer: StateTransferResponsibility::Interpolate,
    });
    let interpolated = sensitive_old.state_transfer_plan(&sensitive_next);
    if interpolated.responsibility != StateTransferResponsibility::Interpolate
        || interpolated.action_for(EntityId::new(EntityDimension::Volume, 1))
            != Some(IrStateTransferAction::Interpolate)
    {
        return Err("sensitivity contract did not select interpolation".into());
    }

    let mut value_old = fixture(&[(1, PrimitiveKind::Box)]);
    let mut value_next = fixture(&[(1, PrimitiveKind::Box)]);
    value_old.entities[0]
        .arguments
        .push(meshlang_ast::IrArgument {
            name: meshlang_ast::IrArgumentName::Size,
            value: meshlang_ast::IrValue::Integer(1),
        });
    value_next.entities[0]
        .arguments
        .push(meshlang_ast::IrArgument {
            name: meshlang_ast::IrArgumentName::Size,
            value: meshlang_ast::IrValue::Integer(2),
        });
    let initialized = value_old.state_transfer_plan(&value_next);
    if initialized.responsibility != StateTransferResponsibility::None
        || initialized.action_for(EntityId::new(EntityDimension::Volume, 1))
            != Some(IrStateTransferAction::Initialize)
    {
        return Err("value-only change did not select initialization".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("preserve_drop_checks", 3.0)
        .minimum("preserve_drop_checks", 3.0)
        .metric("interpolation_checks", 2.0)
        .minimum("interpolation_checks", 2.0)
        .metric("initialization_checks", 2.0)
        .minimum("initialization_checks", 2.0)
        .metric("ordering_validation_checks", 1.0)
        .minimum("ordering_validation_checks", 1.0)
        .input_checksum("source:meshlang-state-transfer-fixtures:v1")
        .reference(
            "entity-ID state-transfer actions for stable, changed, added, and removed geometry",
        )
        .reference_checksum("analytic/deterministic-fixture:meshlang-state-transfer-v1")
        .dimension("preserve, drop, initialize, interpolate, and application-defined actions")
        .formulation("MeshLang backend-neutral remesh/state-transfer consumer plan")
        .tolerance("all action and deterministic ordering checks pass")
        .seed("meshlang-state-transfer-reference:cpu:v1"))
}

fn main() {
    let started = Instant::now();
    let (result, passed) = match run() {
        Ok(result) => (
            result.elapsed_ms(started.elapsed().as_millis() as u64),
            true,
        ),
        Err(error) => (
            QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64),
            false,
        ),
    };
    println!(
        "{}",
        result.to_json_line().expect("serialize QA result JSON")
    );
    if !passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
