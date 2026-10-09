//! Release-facing MeshLang-to-FEM RT0 H(div) state-transfer reference.
//!
//! This is a deterministic normal-flux fixture. It exercises the canonical
//! MeshLang plan bridge and the checked RT0 transfer kernel without invoking a
//! solver, mesher, Gmsh, or simulator workload.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::mesh::{
    apply_hdiv_rt0_state_transfer, transfer_hdiv_rt0_state_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HdivSpace};
use sonium_qa::QaResult;
use std::collections::BTreeMap;

const CASE_ID: &str = "meshlang.fem-hdiv-rt0-transfer-reference.v1";

fn edge_id(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn run() -> Result<QaResult, String> {
    let source_mesh = unit_square_triangles(1);
    let target_mesh = unit_square_triangles(2);
    let source_space = HdivSpace::new();
    let target_space = HdivSpace::new();
    let source_dofs = source_space.num_global_dofs(&source_mesh);
    let target_dofs = target_space.num_global_dofs(&target_mesh);
    let source_values: Vec<_> = (0..source_dofs).map(|index| 0.25 + index as f64).collect();
    let expected = transfer_hdiv_rt0_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
    )
    .map_err(|error| format!("direct RT0 transfer: {error}"))?;
    let target_entities = (0..target_dofs)
        .map(|index| (edge_id(index), index))
        .collect::<BTreeMap<_, _>>();
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: target_entities
            .keys()
            .map(|&entity| IrStateTransferEntry {
                entity,
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    };
    let transferred = apply_hdiv_rt0_state_transfer::<String, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
        &BTreeMap::new(),
        &target_entities,
        |_, _, _| Ok(0.0),
    )
    .map_err(|error| format!("MeshLang H(div) bridge: {error}"))?;
    let max_error = target_entities
        .iter()
        .map(|(entity, &index)| (transferred[entity] - expected[index]).abs())
        .fold(0.0_f64, f64::max);
    if max_error > 1.0e-10 {
        return Err(format!("RT0 transfer mismatch: max error {max_error:e}"));
    }

    let conservative_plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Conservative,
        entries: vec![IrStateTransferEntry {
            entity: edge_id(0),
            action: IrStateTransferAction::Conservative,
        }],
    };
    let conservative = apply_hdiv_rt0_state_transfer::<String, _>(
        &conservative_plan,
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
        &BTreeMap::new(),
        &target_entities,
        |_, _, _| Ok(0.0),
    )
    .map_err(|error| format!("MeshLang H(div) conservative bridge: {error}"))?;
    if (conservative[&edge_id(0)] - expected[0]).abs() > 1.0e-10 {
        return Err("conservative RT0 transfer did not preserve the flux moment".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .backend_passed("cpu")
        .metric("target_flux_moment_checks", target_dofs as f64)
        .minimum("target_flux_moment_checks", target_dofs as f64)
        .metric("conservative_checks", 1.0)
        .minimum("conservative_checks", 1.0)
        .metric("max_error", max_error)
        .maximum("max_error", 1.0e-10)
        .reference("analytic checked RT0 normal-flux transfer through MeshLang plan bridge")
        .reference_checksum("analytic/unit-square-rt0-state-transfer-v1")
        .dimension("2-D affine triangular RT0 source/refined target")
        .formulation("IrStateTransferPlan RT0 H(div) checked consumer")
        .tolerance("all target flux moments agree with the checked RT0 kernel")
        .elapsed_ms(0))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("RT0 MeshLang state-transfer QA result must serialize")
    );
}
