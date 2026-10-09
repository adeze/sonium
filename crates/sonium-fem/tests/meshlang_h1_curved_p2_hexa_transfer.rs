#![cfg(feature = "meshlang")]

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::refinement::refine_elements_high_order_with_history_checked;
use sonium_fem::mesh::{
    RefinementTransferDirection, apply_h1_p2_state_transfer_with_refinement_history,
    unit_cube_hexahedra,
};
use sonium_fem::spaces::H1Space;
use std::collections::BTreeMap;

fn node_entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, index as u64)
}

fn entities(mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<EntityId, usize> {
    (0..mesh.num_nodes())
        .map(|index| (node_entity(index), index))
        .collect()
}

fn interpolation_plan(mesh: &sonium_fem::mesh::Mesh) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..mesh.num_nodes())
            .map(|index| IrStateTransferEntry {
                entity: node_entity(index),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

fn linear_field(point: sonium_fem::mesh::Point) -> f64 {
    0.7 + 1.25 * point.x - 0.8 * point.y + 0.55 * point.z
}

#[test]
fn curved_p2_hexahedral_refinement_preserves_linear_h1_state() {
    let mut source_mesh = unit_cube_hexahedra(1).to_p2();
    // Q2 node 20 is the centre of the bottom face in the canonical Hex27
    // layout. Move it out of plane so the parent map is genuinely curved.
    let curved_node = source_mesh.elements[0].nodes[20];
    source_mesh.nodes[curved_node].z = -0.2;

    let mut target_mesh = source_mesh.clone();
    let (_, history) = refine_elements_high_order_with_history_checked(&mut target_mesh, &[0])
        .expect("curved Hex27 refinement should succeed");

    assert_eq!(target_mesh.num_elements(), 8);
    assert_eq!(target_mesh.num_nodes(), 125);
    assert!(target_mesh.validate_connectivity().is_ok());
    assert!(target_mesh.nodes.iter().any(|point| {
        (point.x - 0.25).abs() < 1.0e-12
            && (point.y - 0.25).abs() < 1.0e-12
            && (point.z + 0.1125).abs() < 1.0e-12
    }));

    let source_entities = entities(&source_mesh);
    let target_entities = entities(&target_mesh);
    let source_values = source_mesh
        .nodes
        .iter()
        .copied()
        .map(linear_field)
        .collect::<Vec<_>>();
    let transferred = apply_h1_p2_state_transfer_with_refinement_history::<String, _>(
        &interpolation_plan(&target_mesh),
        &source_mesh,
        &target_mesh,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P2),
        &source_values,
        &history,
        RefinementTransferDirection::Refine,
        &source_entities,
        &target_entities,
        |_entity, _action, _source| Ok(0.0),
    )
    .expect("curved Hex27 H1 transfer should succeed");

    let max_error = target_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(node, point)| (transferred[&node_entity(node)] - linear_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_error < 1.0e-10,
        "curved Hex27 transfer error={max_error:e}"
    );
}
