//! Release-backed canonical-IR topology and change-contract reference.

use meshlang_ast::{RemeshResponsibility, StateTransferResponsibility, VersionedCanonicalIr};
use meshlang_compiler::compile_to_geo_full;
use sonium_qa::{QaResult, QaStatus};
use std::time::Instant;

const CASE_ID: &str = "meshlang.topology-diff-reference.v1";

fn compile(source: &str, file: &str) -> Result<meshlang_ast::CanonicalGeometryIr, String> {
    let result = compile_to_geo_full(source, file).map_err(|error| error.to_string())?;
    serde_json::from_str::<VersionedCanonicalIr>(&result.canonical_ir_json)
        .map(|document| document.ir)
        .map_err(|error| format!("canonical IR JSON: {error}"))
}

fn run() -> Result<QaResult, String> {
    let stable_a = compile(
        "param width = 1.0 geometry model { let body = Box(width, 1.0, 1.0) }",
        "topology-diff-reference-stable-a.mlg",
    )?;
    let stable_b = compile(
        "param width = 1.25 geometry model { let body = Box(width, 1.0, 1.0) }",
        "topology-diff-reference-stable-b.mlg",
    )?;
    let stable = stable_a.topology_diff(&stable_b);
    if stable.topology_changed() || !stable.values_changed() {
        return Err("numeric perturbation changed entity topology".into());
    }

    let branch_a = compile(
        "geometry model { if true { let shape = Box(1.0, 1.0, 1.0) } else { let shape = Sphere(r: 0.5) } }",
        "topology-diff-reference-branch-a.mlg",
    )?;
    let branch_b = compile(
        "geometry model { if false { let shape = Box(1.0, 1.0, 1.0) } else { let shape = Sphere(r: 0.5) } }",
        "topology-diff-reference-branch-b.mlg",
    )?;
    let branch = branch_a.change_contract(&branch_b);
    if !branch.diff.topology_changed()
        || branch.remesh != RemeshResponsibility::TopologyDependent
        || branch.state_transfer != StateTransferResponsibility::ApplicationDefined
        || branch.validate().is_err()
    {
        return Err("branch topology contract was not promoted to application policy".into());
    }

    let marker_a = compile(
        "geometry model { let body = Box(1.0, 1.0, 1.0) } physical_region volume left { entities: model.body }",
        "topology-diff-reference-marker-a.mlg",
    )?;
    let marker_b = compile(
        "geometry model { let body = Box(1.0, 1.0, 1.0) } physical_region volume right { entities: model.body }",
        "topology-diff-reference-marker-b.mlg",
    )?;
    let marker = marker_a.change_contract(&marker_b);
    if marker.diff.topology_changed()
        || !marker.diff.markers_changed()
        || marker.validate().is_err()
    {
        return Err("marker-only contract incorrectly requested topology work".into());
    }

    let design_a = compile(
        "design width { type: continuous initial: 1.0 lower: 0.5 upper: 2.0 } geometry model { let body = Box(width, 1.0, 1.0) }",
        "topology-diff-reference-design-a.mlg",
    )?;
    let design_b = compile(
        "design width { type: continuous initial: 1.25 lower: 0.5 upper: 2.0 } geometry model { let body = Box(width, 1.0, 1.0) }",
        "topology-diff-reference-design-b.mlg",
    )?;
    let design = design_a.change_contract(&design_b);
    if !design.diff.values_changed()
        || design.remesh != RemeshResponsibility::Global
        || design.state_transfer != StateTransferResponsibility::Interpolate
        || design.validate().is_err()
    {
        return Err("design change contract did not retain global/interpolating policy".into());
    }

    let mut invalid = branch_a.change_contract(&branch_b);
    invalid.remesh = RemeshResponsibility::Global;
    if invalid.validate().is_ok() {
        return Err("downgraded topology contract was accepted".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .reference("canonical IR topology diff and remesh/state-transfer change contract")
        .dimension("backend-neutral MeshLang geometry IR")
        .formulation("compiler-produced canonical IR; no meshing or simulation workload")
        .metric("topology_diff_checks", 6.0)
        .metric("change_contract_validation_checks", 2.0))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("serialize topology-diff QA result")
    );
    if result.status != QaStatus::Passed {
        std::process::exit(1);
    }
}
