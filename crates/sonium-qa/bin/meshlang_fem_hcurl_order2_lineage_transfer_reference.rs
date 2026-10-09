//! CPU-only MeshLang-to-FEM order-2 H(curl) adaptive-lineage reference.
//!
//! This fixture exercises two checked affine-triangle refine/coarsen stages;
//! it does not start a solver, mesher, Gmsh process, or simulator.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use num_complex::Complex64;
use sonium_fem::mesh::refinement::refine_elements_with_history_checked;
use sonium_fem::mesh::{
    InterpolationError, RefinementTransferDirection,
    apply_complex_hcurl_order2_triangle_state_transfer_with_refinement_chain,
    apply_hcurl_order2_triangle_state_transfer_with_refinement_chain,
    transfer_hcurl_order2_triangle_state_with_refinement_chain_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;
use std::collections::BTreeMap;
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-hcurl-order2-lineage-transfer-reference.v1";

fn state(mesh: &sonium_fem::mesh::Mesh) -> Vec<f64> {
    let space = HcurlSpace::order2();
    (0..space.num_global_dofs(mesh))
        .map(|index| 0.125 + 0.031 * index as f64 - 0.0007 * (index * index) as f64)
        .collect()
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let level0 = unit_square_triangles(1);
    let mut level1 = level0.clone();
    let (_, history0) = refine_elements_with_history_checked(&mut level1, &[0])
        .map_err(|error| error.to_string())?;
    let level1_child = level1
        .elements
        .iter()
        .position(|element| element.parent_id == Some(history0.parents[0].element.id))
        .ok_or_else(|| "refined child not found".to_owned())?;
    let mut level2 = level1.clone();
    let (_, history1) = refine_elements_with_history_checked(&mut level2, &[level1_child])
        .map_err(|error| error.to_string())?;

    let meshes = [&level0, &level1, &level2];
    let space = HcurlSpace::order2();
    let spaces = [&space, &space, &space];
    let source = state(&level0);
    let histories = [history0, history1];
    let refined = transfer_hcurl_order2_triangle_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &source,
        &histories,
        RefinementTransferDirection::Refine,
    )
    .map_err(|error| error.to_string())?;
    let source_entities = (0..source.len())
        .map(|dof| (EntityId::new(EntityDimension::Point, dof as u64), dof))
        .collect::<BTreeMap<_, _>>();
    let target_entities = (0..refined.len())
        .map(|dof| (EntityId::new(EntityDimension::Point, dof as u64), dof))
        .collect::<BTreeMap<_, _>>();
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..refined.len())
            .map(|dof| IrStateTransferEntry {
                entity: EntityId::new(EntityDimension::Point, dof as u64),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    };
    let bridged =
        apply_hcurl_order2_triangle_state_transfer_with_refinement_chain::<&'static str, _>(
            &plan,
            &meshes,
            &spaces,
            &source,
            &histories,
            RefinementTransferDirection::Refine,
            &source_entities,
            &target_entities,
            |_, _, _| Err("unexpected chain callback"),
        )
        .map_err(|error| format!("MeshLang plan chain bridge: {error}"))?;
    let max_bridge_error = bridged
        .values()
        .zip(&refined)
        .map(|(actual, expected)| (*actual - *expected).abs())
        .fold(0.0_f64, f64::max);
    if max_bridge_error > 1.0e-12 {
        return Err(format!("plan chain bridge error {max_bridge_error:e}"));
    }
    let complex_source = source
        .iter()
        .map(|value| Complex64::new(*value, -*value * 0.5))
        .collect::<Vec<_>>();
    let complex_bridged =
        apply_complex_hcurl_order2_triangle_state_transfer_with_refinement_chain::<
            &'static str,
            _,
        >(
            &plan,
            &meshes,
            &spaces,
            &complex_source,
            &histories,
            RefinementTransferDirection::Refine,
            &source_entities,
            &target_entities,
            |_, _, _| Err("unexpected complex chain callback"),
        )
        .map_err(|error| format!("complex MeshLang plan chain bridge: {error}"))?;
    let max_complex_bridge_error = complex_bridged
        .values()
        .zip(bridged.values())
        .map(|(actual, expected)| {
            (actual.re - *expected)
                .abs()
                .max((actual.im + *expected * 0.5).abs())
        })
        .fold(0.0_f64, f64::max);
    if max_complex_bridge_error > 1.0e-12 {
        return Err(format!(
            "complex plan chain bridge error {max_complex_bridge_error:e}"
        ));
    }

    let recovered = transfer_hcurl_order2_triangle_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &refined,
        &histories,
        RefinementTransferDirection::Coarsen,
    )
    .map_err(|error| error.to_string())?;
    let max_roundtrip_error = recovered
        .iter()
        .zip(&source)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    if max_roundtrip_error > 2.0e-9 {
        return Err(format!(
            "order-2 H(curl) lineage round trip error {max_roundtrip_error:e} exceeds tolerance"
        ));
    }

    let mut rejection_checks = 0;
    let order1 = HcurlSpace::new();
    if !matches!(
        transfer_hcurl_order2_triangle_state_with_refinement_chain_checked(
            &meshes,
            &[&order1, &order1, &order1],
            &vec![0.0; order1.num_global_dofs(&level0)],
            &histories,
            RefinementTransferDirection::Refine,
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ) {
        return Err("order-1 spaces were accepted by order-2 transfer".into());
    }
    rejection_checks += 1;
    if !matches!(
        transfer_hcurl_order2_triangle_state_with_refinement_chain_checked(
            &[&level0],
            &[&space],
            &source,
            &[],
            RefinementTransferDirection::Refine,
        ),
        Err(InterpolationError::RefinementChainMismatch { .. })
    ) {
        return Err("empty lineage chain was accepted".into());
    }
    rejection_checks += 1;

    Ok(QaResult::passed(CASE_ID)
        .metric("max_roundtrip_error", max_roundtrip_error)
        .metric("plan_chain_bridge_max_error", max_bridge_error)
        .maximum("plan_chain_bridge_max_error", 1.0e-12)
        .metric("plan_chain_bridge_checks", 1.0)
        .minimum("plan_chain_bridge_checks", 1.0)
        .metric(
            "complex_plan_chain_bridge_max_error",
            max_complex_bridge_error,
        )
        .maximum("complex_plan_chain_bridge_max_error", 1.0e-12)
        .metric("complex_plan_chain_bridge_checks", 1.0)
        .minimum("complex_plan_chain_bridge_checks", 1.0)
        .metric("level2_dofs", refined.len() as f64)
        .metric("rejection_checks", rejection_checks as f64)
        .dimension("2-D affine triangular order-2 H(curl)")
        .formulation(
            "two-stage checked refine/coarsen lineage with edge moments and bubble projection",
        )
        .input_checksum("unit-square-triangles-two-stage:meshlang-hcurl-order2-lineage-v1")
        .reference_checksum("analytic/deterministic-fixture:meshlang-hcurl-order2-lineage-v1")
        .seed("meshlang-fem-hcurl-order2-lineage-transfer:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
