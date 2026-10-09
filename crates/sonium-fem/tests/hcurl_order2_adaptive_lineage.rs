use sonium_fem::mesh::refinement::refine_elements_with_history_checked;
use sonium_fem::mesh::{
    InterpolationError, RefinementTransferDirection,
    transfer_hcurl_order2_triangle_state_with_refinement_chain_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};

fn order2_values(mesh: &sonium_fem::mesh::Mesh) -> Vec<f64> {
    let space = HcurlSpace::order2();
    (0..space.num_global_dofs(mesh))
        .map(|index| 0.125 + 0.031 * index as f64 - 0.0007 * (index * index) as f64)
        .collect()
}

#[test]
fn order2_hcurl_refinement_chain_reverses_on_affine_triangles() {
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
    let space = HcurlSpace::order2();
    let spaces = [&space, &space, &space];
    let source = order2_values(&level0);
    let histories = [history0, history1];

    let refined = transfer_hcurl_order2_triangle_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &source,
        &histories,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    assert_eq!(refined.len(), space.num_global_dofs(&level2));
    assert!(refined.iter().all(|value| value.is_finite()));

    let recovered = transfer_hcurl_order2_triangle_state_with_refinement_chain_checked(
        &meshes,
        &spaces,
        &refined,
        &histories,
        RefinementTransferDirection::Coarsen,
    )
    .unwrap();
    assert_eq!(recovered.len(), source.len());
    for (actual, expected) in recovered.iter().zip(source) {
        assert!(
            (actual - expected).abs() <= 2.0e-9,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn order2_hcurl_lineage_rejects_wrong_space_and_chain_shape() {
    let source = unit_square_triangles(1);
    let mut target = source.clone();
    let (_, history) = refine_elements_with_history_checked(&mut target, &[0]).unwrap();
    let meshes = [&source, &target];
    let order2 = HcurlSpace::order2();
    let order1 = HcurlSpace::new();
    let source_values = order2_values(&source);

    assert!(matches!(
        transfer_hcurl_order2_triangle_state_with_refinement_chain_checked(
            &meshes,
            &[&order1, &order1],
            &vec![0.0; order1.num_global_dofs(&source)],
            &[history.clone()],
            RefinementTransferDirection::Refine,
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ));
    assert!(matches!(
        transfer_hcurl_order2_triangle_state_with_refinement_chain_checked(
            &meshes,
            &[&order2],
            &source_values,
            &[history.clone()],
            RefinementTransferDirection::Refine,
        ),
        Err(InterpolationError::RefinementChainMismatch { .. })
    ));
    assert!(matches!(
        transfer_hcurl_order2_triangle_state_with_refinement_chain_checked(
            &[&source],
            &[&order2],
            &source_values,
            &[],
            RefinementTransferDirection::Refine,
        ),
        Err(InterpolationError::RefinementChainMismatch { .. })
    ));
}
