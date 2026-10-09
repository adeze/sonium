//! CPU-only MeshLang-to-FEM order-1 H(curl) refinement-chain reference.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use num_complex::Complex64;
use sonium_fem::mesh::refinement::refine_elements_with_history_checked;
use sonium_fem::mesh::{
    MeshlangTransferError, RefinementTransferDirection,
    apply_complex_hcurl_order1_state_transfer_with_refinement_chain,
    apply_hcurl_order1_state_transfer_with_refinement_chain,
    transfer_hcurl_order1_state_with_refinement_chain_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;
use std::collections::BTreeMap;
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-hcurl-order1-lineage-transfer-reference.v1";

fn entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn entities(space: &HcurlSpace, mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<EntityId, usize> {
    (0..space.num_global_dofs(mesh))
        .map(|dof| (entity(dof), dof))
        .collect()
}

fn plan(
    target_entities: &BTreeMap<EntityId, usize>,
    action: IrStateTransferAction,
) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: target_entities
            .keys()
            .map(|&entity| IrStateTransferEntry { entity, action })
            .collect(),
    }
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let level0 = unit_square_triangles(1);
    let mut level1 = level0.clone();
    let (_, history0) = refine_elements_with_history_checked(&mut level1, &[0])
        .map_err(|error| error.to_string())?;
    let mut level2 = level1.clone();
    let (_, history1) = refine_elements_with_history_checked(&mut level2, &[0])
        .map_err(|error| error.to_string())?;
    let histories = vec![history0, history1];
    let space0 = HcurlSpace::new();
    let space1 = HcurlSpace::new();
    let space2 = HcurlSpace::new();
    let spaces = [&space0, &space1, &space2];
    let meshes = [&level0, &level1, &level2];
    let source_entities = entities(&spaces[0], &level0);
    let target_entities = entities(&spaces[2], &level2);
    let source_values: Vec<_> = (0..spaces[0].num_global_dofs(&level0))
        .map(|dof| 0.25 + 0.125 * dof as f64)
        .collect();
    let expected = transfer_hcurl_order1_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &source_values,
        &histories,
        RefinementTransferDirection::Refine,
    )
    .map_err(|error| error.to_string())?;
    let bridged = apply_hcurl_order1_state_transfer_with_refinement_chain::<&'static str, _>(
        &plan(&target_entities, IrStateTransferAction::Interpolate),
        &meshes,
        &spaces,
        &source_values,
        &histories,
        RefinementTransferDirection::Refine,
        &source_entities,
        &target_entities,
        |_, _, _| Err("interpolation callback must not run"),
    )
    .map_err(|error| error.to_string())?;
    let max_real_error = target_entities
        .iter()
        .map(|(entity, &dof)| (bridged[entity] - expected[dof]).abs())
        .fold(0.0_f64, f64::max);

    let complex_source: Vec<_> = source_values
        .iter()
        .enumerate()
        .map(|(dof, &value)| Complex64::new(value, -2.0 + 0.5 * dof as f64))
        .collect();
    let complex_bridged =
        apply_complex_hcurl_order1_state_transfer_with_refinement_chain::<&'static str, _>(
            &plan(&target_entities, IrStateTransferAction::Interpolate),
            &meshes,
            &spaces,
            &complex_source,
            &histories,
            RefinementTransferDirection::Refine,
            &source_entities,
            &target_entities,
            |_, _, _| Err("interpolation callback must not run"),
        )
        .map_err(|error| error.to_string())?;
    let expected_imaginary = transfer_hcurl_order1_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &complex_source
            .iter()
            .map(|value| value.im)
            .collect::<Vec<_>>(),
        &histories,
        RefinementTransferDirection::Refine,
    )
    .map_err(|error| error.to_string())?;
    let max_complex_error = target_entities
        .iter()
        .map(|(entity, &dof)| {
            (complex_bridged[entity].re - expected[dof])
                .abs()
                .max((complex_bridged[entity].im - expected_imaginary[dof]).abs())
        })
        .fold(0.0_f64, f64::max);

    let mut callback_called = false;
    let rejection = apply_hcurl_order1_state_transfer_with_refinement_chain::<&'static str, _>(
        &plan(&target_entities, IrStateTransferAction::Conservative),
        &meshes,
        &spaces,
        &source_values,
        &histories,
        RefinementTransferDirection::Refine,
        &source_entities,
        &target_entities,
        |_, _, _| {
            callback_called = true;
            Ok(0.0)
        },
    );
    if callback_called
        || !matches!(
            rejection,
            Err(MeshlangTransferError::InvalidPlan(_))
                | Err(MeshlangTransferError::UnsupportedAction { .. })
        )
    {
        return Err("conservative action was not rejected before callbacks".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("target_dof_checks", target_entities.len() as f64)
        .minimum("target_dof_checks", target_entities.len() as f64)
        .metric("max_real_error", max_real_error)
        .maximum("max_real_error", 1.0e-11)
        .metric("max_complex_error", max_complex_error)
        .maximum("max_complex_error", 1.0e-11)
        .metric("rejection_checks", 1.0)
        .minimum("rejection_checks", 1.0)
        .reference("checked real/imaginary order-1 H(curl) chain transfer")
        .reference_checksum("analytic/deterministic-fixture:meshlang-hcurl-order1-lineage-v1")
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
