//! Bounded geometry-aware MeshLang state-transfer weight reference.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    SpatialInterpolationError, StateTransferResponsibility,
};
use sonium_qa::QaResult;
use std::collections::BTreeMap;
use std::time::Instant;

const CASE_ID: &str = "meshlang.spatial-interpolation-reference.v1";

fn id(index: u64) -> EntityId {
    EntityId::new(EntityDimension::Volume, index)
}

fn run() -> Result<QaResult, String> {
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: vec![
            IrStateTransferEntry {
                entity: id(1),
                action: IrStateTransferAction::Interpolate,
            },
            IrStateTransferEntry {
                entity: id(2),
                action: IrStateTransferAction::Drop,
            },
            IrStateTransferEntry {
                entity: id(3),
                action: IrStateTransferAction::Drop,
            },
            IrStateTransferEntry {
                entity: id(4),
                action: IrStateTransferAction::Interpolate,
            },
        ],
    };
    let source_points = BTreeMap::from([
        (id(1), [0.0, 0.0, 0.0]),
        (id(2), [2.0, 0.0, 0.0]),
        (id(3), [0.0, 2.0, 0.0]),
    ]);
    let target_points = BTreeMap::from([(id(1), [0.0, 0.0, 0.0]), (id(4), [1.0, 0.0, 0.0])]);
    let weights = plan
        .derive_interpolation_weights(&source_points, &target_points, 2)
        .map_err(|error| error.to_string())?;
    if weights[&id(1)] != vec![(id(1), 1.0)] {
        return Err(format!(
            "exact-point weight mismatch: {:?}",
            weights[&id(1)]
        ));
    }
    let target_weights = &weights[&id(4)];
    if target_weights.len() != 2 {
        return Err(format!(
            "nearest-neighbor count mismatch: {target_weights:?}"
        ));
    }
    let total = target_weights.iter().map(|(_, weight)| weight).sum::<f64>();
    if (total - 1.0).abs() > 1.0e-12 {
        return Err(format!("interpolation weights are not normalized: {total}"));
    }
    let values = BTreeMap::from([(id(1), 10.0), (id(2), 20.0), (id(3), 30.0)]);
    let transferred = plan
        .apply_weighted_f64(&values, &weights, |entity, action, _| {
            Err::<f64, String>(format!("unexpected callback for {entity:?} {action:?}"))
        })
        .map_err(|error| error.to_string())?;
    if transferred != BTreeMap::from([(id(1), 10.0), (id(4), 15.0)]) {
        return Err(format!("weighted transfer mismatch: {transferred:?}"));
    }

    let conservative_plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Conservative,
        entries: vec![
            IrStateTransferEntry {
                entity: id(1),
                action: IrStateTransferAction::Conservative,
            },
            IrStateTransferEntry {
                entity: id(2),
                action: IrStateTransferAction::Drop,
            },
            IrStateTransferEntry {
                entity: id(3),
                action: IrStateTransferAction::Drop,
            },
            IrStateTransferEntry {
                entity: id(4),
                action: IrStateTransferAction::Conservative,
            },
        ],
    };
    let conservative_weights = conservative_plan
        .derive_conservative_weights(
            &source_points,
            &target_points,
            &BTreeMap::from([(id(1), 1.0), (id(2), 1.0), (id(3), 1.0)]),
            &BTreeMap::from([(id(1), 1.0), (id(4), 2.0)]),
            2,
        )
        .map_err(|error| error.to_string())?;
    if conservative_weights[&id(4)] != vec![(id(1), 1.0), (id(2), 1.0)] {
        return Err(format!(
            "conservative weight mismatch: {:?}",
            conservative_weights[&id(4)]
        ));
    }
    let conservative_values = conservative_plan
        .apply_weighted_f64(&values, &conservative_weights, |entity, action, _| {
            Err::<f64, String>(format!(
                "unexpected conservative callback for {entity:?} {action:?}"
            ))
        })
        .map_err(|error| error.to_string())?;
    if conservative_values != BTreeMap::from([(id(1), 10.0), (id(4), 30.0)]) {
        return Err(format!(
            "conservative transfer mismatch: {conservative_values:?}"
        ));
    }

    let invalid_plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: vec![
            IrStateTransferEntry {
                entity: id(1),
                action: IrStateTransferAction::Drop,
            },
            IrStateTransferEntry {
                entity: id(2),
                action: IrStateTransferAction::Interpolate,
            },
        ],
    };
    let source = BTreeMap::from([(id(1), [0.0, 0.0, 0.0])]);
    if !matches!(
        invalid_plan.derive_interpolation_weights(&source, &BTreeMap::new(), 1),
        Err(SpatialInterpolationError::MissingTargetPoint(entity)) if entity == id(2)
    ) {
        return Err("missing target point was accepted".into());
    }
    if !matches!(
        invalid_plan.derive_interpolation_weights(
            &source,
            &BTreeMap::from([(id(2), [f64::NAN, 0.0, 0.0])]),
            1,
        ),
        Err(SpatialInterpolationError::NonFinitePoint(entity)) if entity == id(2)
    ) {
        return Err("non-finite target point was accepted".into());
    }
    if !matches!(
        invalid_plan.derive_interpolation_weights(&source, &BTreeMap::new(), 0),
        Err(SpatialInterpolationError::InvalidNeighborCount)
    ) {
        return Err("zero-neighbor request was accepted".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .reference(
            "deterministic inverse-distance interpolation from source/target entity coordinates",
        )
        .dimension("3-D entity centroids; keyed scalar field")
        .formulation("backend-neutral nearest-neighbor interpolation adapter")
        .metric("geometry_weight_checks", 8.0)
        .minimum("geometry_weight_checks", 8.0)
        .metric("normalized_targets", 1.0)
        .minimum("normalized_targets", 1.0)
        .metric("conservative_measure_checks", 2.0)
        .minimum("conservative_measure_checks", 2.0)
        .tolerance(
            "exact-coordinate preservation; finite normalized weights; fail-closed geometry inputs",
        ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(result) => result.elapsed_ms(started.elapsed().as_millis() as u64),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    match result.to_json_line() {
        Ok(line) => println!("{line}"),
        Err(error) => {
            eprintln!("{CASE_ID}: result serialization failed: {error}");
            std::process::exit(1);
        }
    }
}
