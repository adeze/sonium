//! CPU-only release reference for the checked MeshLang multi-stage plan chain.

use std::collections::BTreeMap;
use std::time::Instant;

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    IrStateTransferPlanChain, StateTransferChainError, StateTransferResponsibility,
    StateTransferReverseError,
};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "meshlang.state-transfer-chain-reference.v1";

fn id(index: u64) -> EntityId {
    EntityId::new(EntityDimension::Volume, index)
}

fn plan(
    entries: impl IntoIterator<Item = (EntityId, IrStateTransferAction)>,
) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::ApplicationDefined,
        entries: entries
            .into_iter()
            .map(|(entity, action)| IrStateTransferEntry { entity, action })
            .collect(),
    }
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let chain = IrStateTransferPlanChain::new(vec![
        plan([
            (id(1), IrStateTransferAction::Preserve),
            (id(2), IrStateTransferAction::Initialize),
            (id(3), IrStateTransferAction::Drop),
        ]),
        plan([
            (id(1), IrStateTransferAction::Preserve),
            (id(2), IrStateTransferAction::ApplicationDefined),
            (id(4), IrStateTransferAction::Initialize),
        ]),
    ])
    .map_err(|error| error.to_string())?;

    let source = BTreeMap::from([(id(1), 10_i32), (id(2), 2_i32), (id(3), 30_i32)]);
    let mut callback_checks = 0_u64;
    let result = chain
        .apply(&source, |stage, entity, action, previous| {
            callback_checks += 1;
            match (stage, entity, action, previous.copied()) {
                (0, entity, IrStateTransferAction::Initialize, Some(2)) if entity == id(2) => {
                    Ok(20)
                }
                (1, entity, IrStateTransferAction::ApplicationDefined, Some(20))
                    if entity == id(2) =>
                {
                    Ok(30)
                }
                (1, entity, IrStateTransferAction::Initialize, None) if entity == id(4) => Ok(40),
                _ => Err("unexpected callback contract"),
            }
        })
        .map_err(|error| error.to_string())?;
    if result != BTreeMap::from([(id(1), 10), (id(2), 30), (id(4), 40)]) {
        return Err(format!("unexpected final state: {result:?}"));
    }
    if callback_checks != 3 {
        return Err(format!("expected three callbacks, got {callback_checks}"));
    }

    let rejected = IrStateTransferPlanChain::new(vec![
        plan([(id(1), IrStateTransferAction::Drop)]),
        plan([(id(1), IrStateTransferAction::Preserve)]),
    ]);
    if !matches!(
        rejected,
        Err(StateTransferChainError::PreserveWithoutPreviousOutput { stage: 1, entity })
            if entity == id(1)
    ) {
        return Err("preserve-after-drop rejection was not typed".into());
    }
    let reverse = plan([
        (id(5), IrStateTransferAction::Preserve),
        (id(6), IrStateTransferAction::Initialize),
        (id(7), IrStateTransferAction::Drop),
    ])
    .reversed_for_coarsening()
    .map_err(|error| error.to_string())?;
    if reverse.entries[1].action != IrStateTransferAction::Drop
        || reverse.entries[2].action != IrStateTransferAction::Initialize
    {
        return Err("coarsening inverse action swap failed".into());
    }
    let reverse_chain = chain
        .reversed_for_coarsening()
        .map_err(|error| error.to_string())?;
    if reverse_chain.stage_count() != chain.stage_count()
        || reverse_chain.stages[0].entries[2].action != IrStateTransferAction::Drop
        || reverse_chain.stages[1].entries[1].action != IrStateTransferAction::Drop
    {
        return Err("coarsening inverse chain order/action failed".into());
    }
    if !matches!(
        plan([(id(8), IrStateTransferAction::Interpolate)]).reversed_for_coarsening(),
        Err(StateTransferReverseError::NonInvertibleAction { .. })
    ) {
        return Err("non-invertible coarsening action was not rejected".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("stage_count", chain.stage_count() as f64)
        .metric("callback_checks", callback_checks as f64)
        .metric("final_entries", result.len() as f64)
        .metric("rejection_checks", 1.0)
        .metric("reverse_checks", 3.0)
        .reference("deterministic two-stage keyed state-transfer chain")
        .reference_checksum("analytic/deterministic-fixture:meshlang-state-transfer-chain-v1")
        .dimension("preserve, initialize, application-defined, and drop across two stages")
        .formulation("validated serial plan-chain application with stage-tagged callbacks")
        .tolerance("exact integer state and typed rejection contract")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    let passed = result.status == QaStatus::Passed;
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
