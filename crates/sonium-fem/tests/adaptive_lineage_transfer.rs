use sonium_fem::mesh::refinement::{
    coarsen_elements_checked, refine_elements_with_history_checked,
};
use sonium_fem::mesh::{
    InterpolationError, MeshTopology, RefinementTransferDirection, RefinementTransferReport,
    transfer_h1_state_with_refinement_history_checked,
    transfer_hcurl_order1_state_with_refinement_history_checked,
    transfer_hdiv_rt0_state_with_refinement_history_checked,
    transfer_hdiv_rt1_state_with_refinement_history_checked, unit_square_triangles,
    validate_refinement_transfer_lineage_checked,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace, HdivOrder, HdivSpace};

fn scalar_value(point: sonium_fem::mesh::Point) -> f64 {
    1.25 + 2.0 * point.x - 0.75 * point.y
}

fn edge_moment(point0: sonium_fem::mesh::Point, point1: sonium_fem::mesh::Point) -> f64 {
    let dx = point1.x - point0.x;
    let dy = point1.y - point0.y;
    dx * (1.0 - 0.5 * (point0.y + point1.y)) + dy * (2.0 + 0.5 * (point0.x + point1.x))
}

fn h1_values(mesh: &sonium_fem::mesh::Mesh) -> Vec<f64> {
    mesh.nodes.iter().copied().map(scalar_value).collect()
}

fn hcurl_values(mesh: &sonium_fem::mesh::Mesh) -> Vec<f64> {
    MeshTopology::build(mesh)
        .edges
        .iter()
        .map(|&(start, end)| edge_moment(mesh.nodes[start], mesh.nodes[end]))
        .collect()
}

fn assert_close(actual: &[f64], mesh: &sonium_fem::mesh::Mesh, tolerance: f64) {
    assert_eq!(actual.len(), mesh.num_nodes());
    for (value, point) in actual.iter().zip(&mesh.nodes) {
        assert!((value - scalar_value(*point)).abs() <= tolerance);
    }
}

fn assert_report(
    report: RefinementTransferReport,
    direction: RefinementTransferDirection,
    parent_count: usize,
) {
    assert_eq!(report.direction, direction);
    assert_eq!(report.parent_count, parent_count);
    assert_eq!(report.child_count, 4 * parent_count);
    assert!(report.retained_element_count > 0);
}

#[test]
fn h1_transfer_follows_lineage_in_both_refine_and_coarsen_directions() {
    let source = unit_square_triangles(1);
    let source_values = h1_values(&source);
    let mut refined = source.clone();
    let (_, history) = refine_elements_with_history_checked(&mut refined, &[0]).unwrap();

    let report = validate_refinement_transfer_lineage_checked(
        &source,
        &refined,
        &history,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    assert_report(report, RefinementTransferDirection::Refine, 1);
    let refined_values = transfer_h1_state_with_refinement_history_checked(
        &source,
        &refined,
        &source_values,
        &history,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    assert_close(&refined_values, &refined, 2.0e-12);

    let parent_id = history.parents[0].element.id;
    let mut coarsened = refined.clone();
    coarsen_elements_checked(&mut coarsened, &history, &[parent_id]).unwrap();
    let report = validate_refinement_transfer_lineage_checked(
        &refined,
        &coarsened,
        &history,
        RefinementTransferDirection::Coarsen,
    )
    .unwrap();
    assert_report(report, RefinementTransferDirection::Coarsen, 1);
    let coarsened_values = transfer_h1_state_with_refinement_history_checked(
        &refined,
        &coarsened,
        &refined_values,
        &history,
        RefinementTransferDirection::Coarsen,
    )
    .unwrap();
    assert_close(&coarsened_values, &coarsened, 2.0e-12);
}

#[test]
fn hcurl_lineage_transfer_preserves_edge_moments_across_refine_and_coarsen() {
    let source = unit_square_triangles(1);
    let source_space = HcurlSpace::new();
    let source_values = hcurl_values(&source);
    let mut refined = source.clone();
    let (_, history) = refine_elements_with_history_checked(&mut refined, &[0]).unwrap();
    let target_space = HcurlSpace::new();
    let refined_values = transfer_hcurl_order1_state_with_refinement_history_checked(
        &source,
        &refined,
        &source_space,
        &target_space,
        &source_values,
        &history,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    let expected_refined = hcurl_values(&refined);
    assert_eq!(refined_values.len(), expected_refined.len());
    assert!(
        refined_values
            .iter()
            .zip(expected_refined)
            .all(|(actual, expected)| (actual - expected).abs() <= 2.0e-10)
    );

    let parent_id = history.parents[0].element.id;
    let mut coarsened = refined.clone();
    coarsen_elements_checked(&mut coarsened, &history, &[parent_id]).unwrap();
    let coarsened_values = transfer_hcurl_order1_state_with_refinement_history_checked(
        &refined,
        &coarsened,
        &source_space,
        &target_space,
        &refined_values,
        &history,
        RefinementTransferDirection::Coarsen,
    )
    .unwrap();
    let expected_coarsened = hcurl_values(&coarsened);
    assert!(
        coarsened_values
            .iter()
            .zip(expected_coarsened)
            .all(|(actual, expected)| (actual - expected).abs() <= 2.0e-10)
    );
}

#[test]
fn hdiv_lineage_wrappers_cover_rt0_and_rt1_refinement() {
    let source = unit_square_triangles(1);
    let mut refined = source.clone();
    let (_, history) = refine_elements_with_history_checked(&mut refined, &[0]).unwrap();
    let rt0 = HdivSpace::with_order(HdivOrder::Rt0);
    let rt0_values = vec![0.0; rt0.num_global_dofs(&source)];
    let transferred_rt0 = transfer_hdiv_rt0_state_with_refinement_history_checked(
        &source,
        &refined,
        &rt0,
        &rt0,
        &rt0_values,
        &history,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    assert_eq!(transferred_rt0.len(), rt0.num_global_dofs(&refined));
    assert!(transferred_rt0.iter().all(|value| value.abs() <= 1.0e-12));

    let rt1 = HdivSpace::with_order(HdivOrder::Rt1);
    let rt1_values = vec![0.0; rt1.num_global_dofs(&source)];
    let transferred_rt1 = transfer_hdiv_rt1_state_with_refinement_history_checked(
        &source,
        &refined,
        &rt1,
        &rt1,
        &rt1_values,
        &history,
        RefinementTransferDirection::Refine,
    )
    .unwrap();
    assert_eq!(transferred_rt1.len(), rt1.num_global_dofs(&refined));
    assert!(transferred_rt1.iter().all(|value| value.abs() <= 1.0e-10));

    let parent_id = history.parents[0].element.id;
    let mut coarsened = refined.clone();
    coarsen_elements_checked(&mut coarsened, &history, &[parent_id]).unwrap();
    let refined_rt0_values = vec![0.0; rt0.num_global_dofs(&refined)];
    let coarsened_rt0 = transfer_hdiv_rt0_state_with_refinement_history_checked(
        &refined,
        &coarsened,
        &rt0,
        &rt0,
        &refined_rt0_values,
        &history,
        RefinementTransferDirection::Coarsen,
    )
    .unwrap();
    assert!(coarsened_rt0.iter().all(|value| value.abs() <= 1.0e-12));
    let refined_rt1_values = vec![0.0; rt1.num_global_dofs(&refined)];
    let coarsened_rt1 = transfer_hdiv_rt1_state_with_refinement_history_checked(
        &refined,
        &coarsened,
        &rt1,
        &rt1,
        &refined_rt1_values,
        &history,
        RefinementTransferDirection::Coarsen,
    )
    .unwrap();
    assert!(coarsened_rt1.iter().all(|value| value.abs() <= 1.0e-10));
}

#[test]
fn lineage_transfer_rejects_empty_stale_and_foreign_histories() {
    let source = unit_square_triangles(1);
    let mut refined = source.clone();
    let (_, history) = refine_elements_with_history_checked(&mut refined, &[0]).unwrap();
    let empty = sonium_fem::mesh::refinement::RefinementHistory {
        parents: Vec::new(),
        boundaries: Vec::new(),
    };
    assert!(matches!(
        validate_refinement_transfer_lineage_checked(
            &source,
            &refined,
            &empty,
            RefinementTransferDirection::Refine,
        ),
        Err(InterpolationError::RefinementHistoryMismatch { .. })
    ));

    let foreign = unit_square_triangles(2);
    assert!(matches!(
        validate_refinement_transfer_lineage_checked(
            &source,
            &foreign,
            &history,
            RefinementTransferDirection::Refine,
        ),
        Err(InterpolationError::RefinementHistoryMismatch { .. })
            | Err(InterpolationError::RefinementLineageMismatch { .. })
    ));

    let mut wrong_target = unit_square_triangles(1);
    wrong_target.elements[0].marker = 42;
    assert!(matches!(
        validate_refinement_transfer_lineage_checked(
            &source,
            &wrong_target,
            &history,
            RefinementTransferDirection::Refine,
        ),
        Err(InterpolationError::RefinementLineageMismatch { .. })
            | Err(InterpolationError::RefinementHistoryMismatch { .. })
    ));
}
