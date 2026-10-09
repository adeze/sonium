#![cfg(feature = "meshlang")]

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::mesh::{
    MeshlangTransferError, apply_hcurl_order2_triangle_state_transfer, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use std::collections::BTreeMap;

fn curve_id(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

#[test]
fn order2_hcurl_consumer_routes_interpolation_without_callback() {
    let mesh = unit_square_triangles(1);
    let space = HcurlSpace::order2();
    let dof_count = space.num_global_dofs(&mesh);
    let source_values: Vec<f64> = (0..dof_count)
        .map(|index| 0.5 - 0.07 * index as f64)
        .collect();
    let mut entities = BTreeMap::new();
    let mut entries = Vec::new();
    for dof in 0..dof_count {
        let entity = curve_id(dof);
        entities.insert(entity, dof);
        entries.push(IrStateTransferEntry {
            entity,
            action: IrStateTransferAction::Interpolate,
        });
    }
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries,
    };

    let output = apply_hcurl_order2_triangle_state_transfer::<String, _>(
        &plan,
        &mesh,
        &mesh,
        &space,
        &space,
        &source_values,
        &entities,
        &entities,
        |_, _, _| -> Result<f64, String> { panic!("interpolation must not invoke callback") },
    )
    .expect("order-2 consumer should apply the checked interpolation path");

    for (dof, value) in source_values.iter().enumerate() {
        assert!((output[&curve_id(dof)] - value).abs() < 1.0e-11);
    }
}

#[test]
fn order2_hcurl_consumer_rejects_conservative_action() {
    let mesh = unit_square_triangles(1);
    let space = HcurlSpace::order2();
    let entity = curve_id(0);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Conservative,
        entries: vec![IrStateTransferEntry {
            entity,
            action: IrStateTransferAction::Conservative,
        }],
    };
    let mut entities = BTreeMap::new();
    entities.insert(entity, 0);
    let error = apply_hcurl_order2_triangle_state_transfer::<String, _>(
        &plan,
        &mesh,
        &mesh,
        &space,
        &space,
        &vec![0.0; space.num_global_dofs(&mesh)],
        &entities,
        &entities,
        |_, _, _| -> Result<f64, String> { Ok(0.0) },
    )
    .expect_err("conservative scalar action is not an H(curl) transfer");
    assert!(matches!(
        error,
        MeshlangTransferError::UnsupportedAction { .. }
    ));
}
