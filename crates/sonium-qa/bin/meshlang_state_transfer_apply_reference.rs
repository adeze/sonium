//! Release-facing checked MeshLang state-transfer consumer reference.

use std::collections::BTreeMap;
use std::time::Instant;

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferApplyError, StateTransferResponsibility,
};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "meshlang.state-transfer-apply-reference.v1";

fn id(index: u64) -> EntityId {
    EntityId::new(EntityDimension::Volume, index)
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::ApplicationDefined,
        entries: vec![
            IrStateTransferEntry {
                entity: id(1),
                action: IrStateTransferAction::Preserve,
            },
            IrStateTransferEntry {
                entity: id(2),
                action: IrStateTransferAction::Drop,
            },
            IrStateTransferEntry {
                entity: id(3),
                action: IrStateTransferAction::Interpolate,
            },
            IrStateTransferEntry {
                entity: id(4),
                action: IrStateTransferAction::Conservative,
            },
            IrStateTransferEntry {
                entity: id(5),
                action: IrStateTransferAction::ApplicationDefined,
            },
            IrStateTransferEntry {
                entity: id(6),
                action: IrStateTransferAction::Initialize,
            },
        ],
    };
    let source = BTreeMap::from([
        (id(1), 2.0_f64),
        (id(2), 8.0),
        (id(3), 4.0),
        (id(4), 3.0),
        (id(5), 7.0),
    ]);
    let mut callback_count = 0_u64;
    let target = plan
        .apply(&source, |entity, action, previous| {
            callback_count += 1;
            match action {
                IrStateTransferAction::Initialize => Ok(entity.index as f64),
                IrStateTransferAction::Interpolate => previous
                    .map(|value| value * 0.5)
                    .ok_or("interpolation requires source state"),
                IrStateTransferAction::Conservative => previous
                    .map(|value| value + 1.0)
                    .ok_or("conservative transfer requires source state"),
                IrStateTransferAction::ApplicationDefined => {
                    Ok(previous.copied().unwrap_or_default() + 10.0)
                }
                IrStateTransferAction::Preserve | IrStateTransferAction::Drop => {
                    Err("non-producing action reached callback")
                }
            }
        })
        .map_err(|error| error.to_string())?;

    let mut checks = 1_u64;
    if target
        != BTreeMap::from([
            (id(1), 2.0),
            (id(3), 2.0),
            (id(4), 4.0),
            (id(5), 17.0),
            (id(6), 6.0),
        ])
    {
        return Err(format!("unexpected transferred values: {target:?}"));
    }
    checks += 5;
    if callback_count != 4 {
        return Err(format!(
            "expected four callback actions, got {callback_count}"
        ));
    }
    checks += 1;
    if target.keys().copied().collect::<Vec<_>>() != [id(1), id(3), id(4), id(5), id(6)] {
        return Err("target map lost deterministic ordering".into());
    }
    checks += 1;

    let stale =
        plan.apply::<f64, &'static str, _>(&BTreeMap::from([(id(9), 1.0)]), |_, _, _| Ok(0.0));
    if !matches!(stale, Err(StateTransferApplyError::UnexpectedSource(entity)) if entity == id(9)) {
        return Err(format!("stale source was not rejected: {stale:?}"));
    }
    checks += 1;

    let missing = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::None,
        entries: vec![IrStateTransferEntry {
            entity: id(1),
            action: IrStateTransferAction::Preserve,
        }],
    }
    .apply::<f64, &'static str, _>(&BTreeMap::new(), |_, _, _| Ok(0.0));
    if !matches!(missing, Err(StateTransferApplyError::MissingSource(entity)) if entity == id(1)) {
        return Err(format!(
            "missing preserve source was not rejected: {missing:?}"
        ));
    }
    checks += 1;

    let invalid = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::None,
        entries: vec![
            IrStateTransferEntry {
                entity: id(2),
                action: IrStateTransferAction::Drop,
            },
            IrStateTransferEntry {
                entity: id(1),
                action: IrStateTransferAction::Drop,
            },
        ],
    }
    .apply::<f64, &'static str, _>(&BTreeMap::new(), |_, _, _| Ok(0.0));
    if !matches!(invalid, Err(StateTransferApplyError::InvalidPlan(_))) {
        return Err(format!("unsorted plan was not rejected: {invalid:?}"));
    }
    checks += 1;

    let responsibility_mismatch = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: vec![IrStateTransferEntry {
            entity: id(7),
            action: IrStateTransferAction::Conservative,
        }],
    }
    .apply::<f64, &'static str, _>(&BTreeMap::new(), |_, _, _| Ok(0.0));
    if !matches!(
        responsibility_mismatch,
        Err(StateTransferApplyError::InvalidPlan(_))
    ) {
        return Err(format!(
            "responsibility/action mismatch was not rejected: {responsibility_mismatch:?}"
        ));
    }
    checks += 1;

    let weighted_plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::ApplicationDefined,
        entries: vec![
            IrStateTransferEntry {
                entity: id(1),
                action: IrStateTransferAction::Preserve,
            },
            IrStateTransferEntry {
                entity: id(2),
                action: IrStateTransferAction::Drop,
            },
            IrStateTransferEntry {
                entity: id(3),
                action: IrStateTransferAction::Interpolate,
            },
            IrStateTransferEntry {
                entity: id(4),
                action: IrStateTransferAction::Conservative,
            },
            IrStateTransferEntry {
                entity: id(5),
                action: IrStateTransferAction::Initialize,
            },
        ],
    };
    let weighted = weighted_plan
        .apply_weighted_f64(
            &BTreeMap::from([(id(1), 2.0), (id(2), 8.0), (id(3), 4.0), (id(4), 3.0)]),
            &BTreeMap::from([
                (id(3), vec![(id(1), 0.25), (id(3), 0.75)]),
                (id(4), vec![(id(1), 1.0), (id(4), 2.0)]),
            ]),
            |entity, action, _| match action {
                IrStateTransferAction::Initialize => Ok::<_, &'static str>(entity.index as f64),
                _ => Err("unexpected callback action"),
            },
        )
        .map_err(|error| format!("weighted transfer failed: {error}"))?;
    if weighted != BTreeMap::from([(id(1), 2.0), (id(3), 3.5), (id(4), 8.0), (id(5), 5.0)]) {
        return Err(format!("weighted transfer mismatch: {weighted:?}"));
    }
    checks += 1;

    Ok(QaResult::passed(CASE_ID)
        .metric("checks", checks as f64)
        .metric("callback_actions", callback_count as f64)
        .input_checksum("source:meshlang-state-transfer-apply-fixture:v1")
        .reference_checksum("analytic/deterministic-fixture:meshlang-state-transfer-apply-v1")
        .formulation("backend-neutral checked state-transfer consumer")
        .seed("meshlang-state-transfer-apply-reference:cpu:v1")
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
