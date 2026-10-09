use std::collections::BTreeMap;

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use num_complex::Complex64;
use sonium_fem::mesh::{
    MeshTopology, MeshlangTransferError, apply_complex_hcurl_order1_state_transfer,
    unit_square_triangles,
};
use sonium_fem::spaces::HcurlSpace;

fn edge_id(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn edge_value(mesh: &sonium_fem::mesh::Mesh, edge: (usize, usize)) -> f64 {
    let p0 = mesh.nodes[edge.0];
    let p1 = mesh.nodes[edge.1];
    (p1.x - p0.x) * (1.0 + 0.25 * (p0.x + p1.x)) + (p1.y - p0.y) * (0.5 - 0.75 * (p0.y + p1.y))
}

fn plan(mesh: &sonium_fem::mesh::Mesh, action: IrStateTransferAction) -> IrStateTransferPlan {
    let edges = MeshTopology::build(mesh).edges;
    IrStateTransferPlan {
        responsibility: match action {
            IrStateTransferAction::Conservative => StateTransferResponsibility::Conservative,
            _ => StateTransferResponsibility::Interpolate,
        },
        entries: (0..edges.len())
            .map(|index| IrStateTransferEntry {
                entity: edge_id(index),
                action,
            })
            .collect(),
    }
}

#[test]
fn complex_hcurl_bridge_matches_real_and_imaginary_same_mesh_transfer() {
    let mesh = unit_square_triangles(1);
    let space = HcurlSpace::new();
    let edges = MeshTopology::build(&mesh).edges;
    let source_values = edges
        .iter()
        .map(|&edge| {
            let value = edge_value(&mesh, edge);
            Complex64::new(value, -0.5 * value)
        })
        .collect::<Vec<_>>();
    let entities = (0..edges.len())
        .map(|index| (edge_id(index), index))
        .collect::<BTreeMap<_, _>>();
    let mut callbacks = 0;

    let result = apply_complex_hcurl_order1_state_transfer::<&'static str, _>(
        &plan(&mesh, IrStateTransferAction::Interpolate),
        &mesh,
        &mesh,
        &space,
        &space,
        &source_values,
        &entities,
        &entities,
        |_entity, _action, _previous| {
            callbacks += 1;
            Err("unexpected callback")
        },
    )
    .unwrap();

    assert_eq!(callbacks, 0);
    assert_eq!(result.len(), source_values.len());
    for (index, value) in source_values.iter().enumerate() {
        assert!((result[&edge_id(index)] - value).norm() < 1.0e-12);
    }
}

#[test]
fn complex_hcurl_bridge_rejects_scalar_conservative_action_before_transfer() {
    let mesh = unit_square_triangles(1);
    let space = HcurlSpace::new();
    let edges = MeshTopology::build(&mesh).edges;
    let entities = (0..edges.len())
        .map(|index| (edge_id(index), index))
        .collect::<BTreeMap<_, _>>();
    let values = vec![Complex64::new(1.0, -0.5); edges.len()];
    let mut callbacks = 0;
    let error = apply_complex_hcurl_order1_state_transfer::<&'static str, _>(
        &plan(&mesh, IrStateTransferAction::Conservative),
        &mesh,
        &mesh,
        &space,
        &space,
        &values,
        &entities,
        &entities,
        |_entity, _action, _previous| {
            callbacks += 1;
            Err("must not run")
        },
    )
    .unwrap_err();

    assert_eq!(callbacks, 0);
    assert!(matches!(
        error,
        MeshlangTransferError::UnsupportedAction {
            action: IrStateTransferAction::Conservative,
            ..
        }
    ));
}
