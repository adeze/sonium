//! Release-facing MeshLang-to-FEM RT1 H(div) refinement-chain reference.
//!
//! This CPU-only fixture exercises real and Complex64 plan-chain bridges over
//! two checked refinement stages. It does not invoke a solver, Gmsh, or a
//! simulator.

use std::collections::BTreeMap;

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use num_complex::Complex64;
use sonium_fem::mesh::refinement::refine_elements_with_history_checked;
use sonium_fem::mesh::{
    RefinementTransferDirection, apply_complex_hdiv_rt1_state_transfer_with_refinement_chain,
    apply_hdiv_rt1_state_transfer_with_refinement_chain,
    transfer_hdiv_rt1_state_with_refinement_chain_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "meshlang.fem-hdiv-rt1-chain-transfer-reference.v1";

fn entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn entities(space: &HdivSpace, mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<EntityId, usize> {
    (0..space.num_global_dofs(mesh))
        .map(|dof| (entity(dof), dof))
        .collect()
}

fn plan(target_entities: &BTreeMap<EntityId, usize>) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: target_entities
            .keys()
            .map(|&entity| IrStateTransferEntry {
                entity,
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

fn max_error(
    actual: &BTreeMap<EntityId, f64>,
    expected: &[f64],
    entities: &BTreeMap<EntityId, usize>,
) -> f64 {
    entities
        .iter()
        .map(|(entity, &dof)| (actual[entity] - expected[dof]).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<QaResult, String> {
    let level0 = unit_square_triangles(1);
    let mut level1 = level0.clone();
    let (_, history0) = refine_elements_with_history_checked(&mut level1, &[0])
        .map_err(|error| error.to_string())?;
    let mut level2 = level1.clone();
    let (_, history1) = refine_elements_with_history_checked(&mut level2, &[0])
        .map_err(|error| error.to_string())?;
    let meshes = [&level0, &level1, &level2];
    let space0 = HdivSpace::with_order(HdivOrder::Rt1);
    let space1 = HdivSpace::with_order(HdivOrder::Rt1);
    let space2 = HdivSpace::with_order(HdivOrder::Rt1);
    let spaces = [&space0, &space1, &space2];
    let histories = vec![history0, history1];
    let source_entities = entities(&space0, &level0);
    let target_entities = entities(&space2, &level2);
    let transfer_plan = plan(&target_entities);
    let source_values: Vec<_> = (0..space0.num_global_dofs(&level0))
        .map(|dof| 0.25 + 0.125 * dof as f64)
        .collect();
    let expected = transfer_hdiv_rt1_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &source_values,
        &histories,
        RefinementTransferDirection::Refine,
    )
    .map_err(|error| error.to_string())?;
    let real = apply_hdiv_rt1_state_transfer_with_refinement_chain::<String, _>(
        &transfer_plan,
        &meshes,
        &spaces,
        &source_values,
        &histories,
        RefinementTransferDirection::Refine,
        &source_entities,
        &target_entities,
        |_, _, _| Err("interpolation callback unexpectedly invoked".to_string()),
    )
    .map_err(|error| error.to_string())?;
    let real_error = max_error(&real, &expected, &target_entities);

    let complex_source: Vec<_> = source_values
        .iter()
        .copied()
        .map(|value| Complex64::new(value, -2.0 * value))
        .collect();
    let complex = apply_complex_hdiv_rt1_state_transfer_with_refinement_chain::<String, _>(
        &transfer_plan,
        &meshes,
        &spaces,
        &complex_source,
        &histories,
        RefinementTransferDirection::Refine,
        &source_entities,
        &target_entities,
        |_, _, _| Err("complex interpolation callback unexpectedly invoked".to_string()),
    )
    .map_err(|error| error.to_string())?;
    let complex_error = target_entities
        .iter()
        .map(|(entity, &dof)| {
            let expected_value = Complex64::new(expected[dof], -2.0 * expected[dof]);
            (complex[entity] - expected_value).norm()
        })
        .fold(0.0, f64::max);
    if real_error > 1.0e-11 || complex_error > 1.0e-11 {
        return Err(format!(
            "RT1 chain parity exceeded tolerance: real={real_error:e}, complex={complex_error:e}"
        ));
    }
    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("plan_chain_bridge_max_error", real_error)
        .metric("complex_plan_chain_bridge_max_error", complex_error)
        .metric("plan_chain_bridge_checks", target_entities.len() as f64)
        .metric("refinement_stages", histories.len() as f64))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("RT1 chain result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, &error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("RT1 chain failure must serialize")
            );
            println!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
