//! CPU-only adaptive FEM parent/child lineage and state-transfer contract.

use sonium_fem::mesh::refinement::{
    coarsen_elements_checked, refine_elements_with_history_checked,
};
use sonium_fem::mesh::{
    InterpolationError, MeshTopology, Point, RefinementTransferDirection,
    transfer_h1_state_with_refinement_history_checked,
    transfer_hcurl_order1_state_with_refinement_history_checked,
    transfer_hdiv_rt0_state_with_refinement_history_checked,
    transfer_hdiv_rt1_state_with_refinement_history_checked, unit_square_triangles,
    validate_refinement_transfer_lineage_checked,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace, HdivOrder, HdivSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.adaptive-lineage-transfer.v1";

fn scalar_value(point: Point) -> f64 {
    1.25 + 2.0 * point.x - 0.75 * point.y
}

fn edge_moment(p0: Point, p1: Point) -> f64 {
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    dx * (1.0 - 0.5 * (p0.y + p1.y)) + dy * (2.0 + 0.5 * (p0.x + p1.x))
}

fn run() -> Result<(usize, usize, f64, f64, usize), String> {
    let source = unit_square_triangles(1);
    let source_values: Vec<_> = source.nodes.iter().copied().map(scalar_value).collect();
    let mut refined = source.clone();
    let (_, history) = refine_elements_with_history_checked(&mut refined, &[0])
        .map_err(|error| error.to_string())?;
    let refine_report = validate_refinement_transfer_lineage_checked(
        &source,
        &refined,
        &history,
        RefinementTransferDirection::Refine,
    )
    .map_err(|error| error.to_string())?;
    let refined_values = transfer_h1_state_with_refinement_history_checked(
        &source,
        &refined,
        &source_values,
        &history,
        RefinementTransferDirection::Refine,
    )
    .map_err(|error| error.to_string())?;
    let h1_error = refined_values
        .iter()
        .zip(&refined.nodes)
        .map(|(actual, point)| (actual - scalar_value(*point)).abs())
        .fold(0.0, f64::max);
    if h1_error > 2.0e-12 {
        return Err(format!("adaptive H1 error {h1_error} exceeds tolerance"));
    }

    let hcurl = HcurlSpace::new();
    let source_hcurl: Vec<_> = MeshTopology::build(&source)
        .edges
        .iter()
        .map(|&(a, b)| edge_moment(source.nodes[a], source.nodes[b]))
        .collect();
    let refined_hcurl = transfer_hcurl_order1_state_with_refinement_history_checked(
        &source,
        &refined,
        &hcurl,
        &hcurl,
        &source_hcurl,
        &history,
        RefinementTransferDirection::Refine,
    )
    .map_err(|error| error.to_string())?;
    let hcurl_error = refined_hcurl
        .iter()
        .zip(MeshTopology::build(&refined).edges)
        .map(|(actual, (a, b))| (actual - edge_moment(refined.nodes[a], refined.nodes[b])).abs())
        .fold(0.0, f64::max);
    if hcurl_error > 2.0e-10 {
        return Err(format!(
            "adaptive H(curl) error {hcurl_error} exceeds tolerance"
        ));
    }

    let rt0 = HdivSpace::with_order(HdivOrder::Rt0);
    let rt0_values = vec![0.0; rt0.num_global_dofs(&source)];
    let rt0_target = transfer_hdiv_rt0_state_with_refinement_history_checked(
        &source,
        &refined,
        &rt0,
        &rt0,
        &rt0_values,
        &history,
        RefinementTransferDirection::Refine,
    )
    .map_err(|error| error.to_string())?;
    if rt0_target.iter().any(|value| !value.is_finite()) {
        return Err("adaptive RT0 transfer produced a non-finite value".into());
    }
    let rt1 = HdivSpace::with_order(HdivOrder::Rt1);
    let rt1_values = vec![0.0; rt1.num_global_dofs(&source)];
    let rt1_target = transfer_hdiv_rt1_state_with_refinement_history_checked(
        &source,
        &refined,
        &rt1,
        &rt1,
        &rt1_values,
        &history,
        RefinementTransferDirection::Refine,
    )
    .map_err(|error| error.to_string())?;
    if rt1_target.iter().any(|value| !value.is_finite()) {
        return Err("adaptive RT1 transfer produced a non-finite value".into());
    }

    let parent_id = history.parents[0].element.id;
    let mut coarsened = refined.clone();
    coarsen_elements_checked(&mut coarsened, &history, &[parent_id])
        .map_err(|error| error.to_string())?;
    let coarsen_report = validate_refinement_transfer_lineage_checked(
        &refined,
        &coarsened,
        &history,
        RefinementTransferDirection::Coarsen,
    )
    .map_err(|error| error.to_string())?;
    let coarsened_values = transfer_h1_state_with_refinement_history_checked(
        &refined,
        &coarsened,
        &refined_values,
        &history,
        RefinementTransferDirection::Coarsen,
    )
    .map_err(|error| error.to_string())?;
    let coarsen_error = coarsened_values
        .iter()
        .zip(&coarsened.nodes)
        .map(|(actual, point)| (actual - scalar_value(*point)).abs())
        .fold(0.0, f64::max);
    if coarsen_error > 2.0e-12 {
        return Err(format!(
            "adaptive coarsen H1 error {coarsen_error} exceeds tolerance"
        ));
    }

    let empty = sonium_fem::mesh::refinement::RefinementHistory {
        parents: Vec::new(),
        boundaries: Vec::new(),
    };
    let rejection_checks = usize::from(matches!(
        validate_refinement_transfer_lineage_checked(
            &source,
            &refined,
            &empty,
            RefinementTransferDirection::Refine,
        ),
        Err(InterpolationError::RefinementHistoryMismatch { .. })
    )) + usize::from(matches!(
        validate_refinement_transfer_lineage_checked(
            &source,
            &unit_square_triangles(2),
            &history,
            RefinementTransferDirection::Refine,
        ),
        Err(InterpolationError::RefinementHistoryMismatch { .. })
            | Err(InterpolationError::RefinementLineageMismatch { .. })
    ));
    if rejection_checks != 2 {
        return Err(format!(
            "only {rejection_checks} adaptive rejection checks passed"
        ));
    }

    if refine_report.child_count != 4 || coarsen_report.child_count != 4 {
        return Err("adaptive lineage child count is not four".into());
    }
    Ok((
        refine_report.parent_count + coarsen_report.parent_count,
        refine_report.child_count + coarsen_report.child_count,
        h1_error.max(coarsen_error),
        hcurl_error,
        rejection_checks,
    ))
}

fn main() {
    let result = match run() {
        Ok((parent_count, child_count, h1_error, hcurl_error, rejection_checks)) => {
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("parent_count", parent_count as f64)
                .minimum("parent_count", 2.0)
                .metric("child_count", child_count as f64)
                .minimum("child_count", 8.0)
                .metric("max_h1_error", h1_error)
                .maximum("max_h1_error", 2.0e-12)
                .metric("max_hcurl_error", hcurl_error)
                .maximum("max_hcurl_error", 2.0e-10)
                .metric("rejection_checks", rejection_checks as f64)
                .minimum("rejection_checks", 2.0)
                .dimension("2-D adaptive P1 triangular parent/child refine and coarsen")
                .formulation(
                    "stable element-ID/parent-ID lineage validation with H1, H(curl), RT0, and RT1 transfer",
                )
                .input_checksum("unit-square-triangles-1-selective-refine-coarsen:adaptive-lineage-v1")
                .reference_checksum("analytic-linear-field-and-zero-vector-state:adaptive-lineage-v1")
                .seed("fem-adaptive-lineage-transfer:cpu:v1")
        }
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
