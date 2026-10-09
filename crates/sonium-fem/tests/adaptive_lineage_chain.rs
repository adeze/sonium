use sonium_fem::mesh::refinement::refine_elements_with_history_checked;
use sonium_fem::mesh::{
    InterpolationError, MeshTopology, RefinementTransferDirection,
    transfer_h1_state_with_refinement_chain_checked,
    transfer_hcurl_order1_state_with_refinement_chain_checked,
    transfer_hdiv_rt0_state_with_refinement_chain_checked,
    transfer_hdiv_rt1_state_with_refinement_chain_checked, unit_square_triangles,
    validate_refinement_transfer_chain_checked,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace, HdivOrder, HdivSpace};

fn scalar_value(point: sonium_fem::mesh::Point) -> f64 {
    1.25 + 2.0 * point.x - 0.75 * point.y
}

fn h1_values(mesh: &sonium_fem::mesh::Mesh) -> Vec<f64> {
    mesh.nodes.iter().copied().map(scalar_value).collect()
}

fn edge_moment(point0: sonium_fem::mesh::Point, point1: sonium_fem::mesh::Point) -> f64 {
    let dx = point1.x - point0.x;
    let dy = point1.y - point0.y;
    dx * (1.0 - 0.5 * (point0.y + point1.y)) + dy * (2.0 + 0.5 * (point0.x + point1.x))
}

fn hcurl_values(mesh: &sonium_fem::mesh::Mesh) -> Vec<f64> {
    MeshTopology::build(mesh)
        .edges
        .iter()
        .map(|&(start, end)| edge_moment(mesh.nodes[start], mesh.nodes[end]))
        .collect()
}

fn assert_scalar_field(values: &[f64], mesh: &sonium_fem::mesh::Mesh) {
    assert_eq!(values.len(), mesh.num_nodes());
    for (&value, point) in values.iter().zip(&mesh.nodes) {
        assert!((value - scalar_value(*point)).abs() <= 2.0e-12);
    }
}

#[test]
fn two_level_chain_transfers_and_reverses_h1_and_hcurl() {
    let level0 = unit_square_triangles(1);
    let mut level1 = level0.clone();
    let (_, history0) = refine_elements_with_history_checked(&mut level1, &[0]).unwrap();

    let level1_child = level1
        .elements
        .iter()
        .position(|element| element.parent_id == Some(history0.parents[0].element.id))
        .unwrap();
    let mut level2 = level1.clone();
    let (_, history1) = refine_elements_with_history_checked(&mut level2, &[level1_child]).unwrap();

    let meshes = [&level0, &level1, &level2];
    let histories = [history0.clone(), history1.clone()];
    let report = validate_refinement_transfer_chain_checked(
        &meshes,
        &histories,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    assert_eq!(report.stages.len(), 2);
    assert!(
        report
            .stages
            .iter()
            .all(|stage| stage.direction == RefinementTransferDirection::Refine)
    );

    let level2_h1 = transfer_h1_state_with_refinement_chain_checked(
        &meshes,
        &h1_values(&level0),
        &histories,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    assert_scalar_field(&level2_h1, &level2);

    let hcurl_space = HcurlSpace::new();
    let spaces = [&hcurl_space, &hcurl_space, &hcurl_space];
    let level2_hcurl = transfer_hcurl_order1_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &hcurl_values(&level0),
        &histories,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    assert!(
        level2_hcurl
            .iter()
            .zip(hcurl_values(&level2))
            .all(|(actual, expected)| (actual - expected).abs() <= 2.0e-10)
    );

    let reverse = validate_refinement_transfer_chain_checked(
        &meshes,
        &histories,
        RefinementTransferDirection::Coarsen,
    )
    .unwrap();
    assert_eq!(reverse.stages.len(), 2);
    assert!(
        reverse
            .stages
            .iter()
            .all(|stage| stage.direction == RefinementTransferDirection::Coarsen)
    );

    let level0_h1 = transfer_h1_state_with_refinement_chain_checked(
        &meshes,
        &level2_h1,
        &histories,
        RefinementTransferDirection::Coarsen,
    )
    .unwrap();
    assert_scalar_field(&level0_h1, &level0);
    let level0_hcurl = transfer_hcurl_order1_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &level2_hcurl,
        &histories,
        RefinementTransferDirection::Coarsen,
    )
    .unwrap();
    assert!(
        level0_hcurl
            .iter()
            .zip(hcurl_values(&level0))
            .all(|(actual, expected)| (actual - expected).abs() <= 2.0e-10)
    );
}

#[test]
fn two_level_chain_covers_rt0_rt1_and_rejects_malformed_sequences() {
    let level0 = unit_square_triangles(1);
    let mut level1 = level0.clone();
    let (_, history0) = refine_elements_with_history_checked(&mut level1, &[0]).unwrap();
    let level1_child = level1
        .elements
        .iter()
        .position(|element| element.parent_id == Some(history0.parents[0].element.id))
        .unwrap();
    let mut level2 = level1.clone();
    let (_, history1) = refine_elements_with_history_checked(&mut level2, &[level1_child]).unwrap();
    let meshes = [&level0, &level1, &level2];
    let histories = [history0, history1];

    let rt0 = HdivSpace::with_order(HdivOrder::Rt0);
    let rt1 = HdivSpace::with_order(HdivOrder::Rt1);
    let rt0_spaces = [&rt0, &rt0, &rt0];
    let rt1_spaces = [&rt1, &rt1, &rt1];
    let rt0_level2 = transfer_hdiv_rt0_state_with_refinement_chain_checked(
        &meshes,
        &rt0_spaces,
        &vec![0.0; rt0.num_global_dofs(&level0)],
        &histories,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    let rt1_level2 = transfer_hdiv_rt1_state_with_refinement_chain_checked(
        &meshes,
        &rt1_spaces,
        &vec![0.0; rt1.num_global_dofs(&level0)],
        &histories,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    assert_eq!(rt0_level2.len(), rt0.num_global_dofs(&level2));
    assert_eq!(rt1_level2.len(), rt1.num_global_dofs(&level2));
    assert!(rt0_level2.iter().all(|value| *value == 0.0));
    assert!(rt1_level2.iter().all(|value| *value == 0.0));

    let bad_length = validate_refinement_transfer_chain_checked(
        &[&level0, &level2],
        &histories,
        RefinementTransferDirection::Refine,
    );
    assert!(matches!(
        bad_length,
        Err(InterpolationError::RefinementChainMismatch { .. })
    ));

    let bad_order = validate_refinement_transfer_chain_checked(
        &meshes,
        &[histories[1].clone(), histories[0].clone()],
        RefinementTransferDirection::Refine,
    );
    assert!(matches!(
        bad_order,
        Err(InterpolationError::RefinementChainMismatch { stage: 0, .. })
    ));

    let bad_spaces = transfer_hdiv_rt0_state_with_refinement_chain_checked(
        &meshes,
        &[&rt0, &rt0],
        &vec![0.0; rt0.num_global_dofs(&level0)],
        &histories,
        RefinementTransferDirection::Refine,
    );
    assert!(matches!(
        bad_spaces,
        Err(InterpolationError::RefinementChainMismatch { .. })
    ));
}
