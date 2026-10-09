//! CPU-only multi-level adaptive FEM lineage transfer contract.

use sonium_fem::mesh::refinement::refine_elements_with_history_checked;
use sonium_fem::mesh::{
    InterpolationError, MeshTopology, Point, RefinementTransferDirection,
    transfer_h1_state_with_refinement_chain_checked,
    transfer_hcurl_order1_state_with_refinement_chain_checked,
    transfer_hdiv_rt0_state_with_refinement_chain_checked,
    transfer_hdiv_rt1_state_with_refinement_chain_checked, unit_square_triangles,
    validate_refinement_transfer_chain_checked,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace, HdivOrder, HdivSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.adaptive-lineage-chain.v1";

fn scalar_value(point: Point) -> f64 {
    1.25 + 2.0 * point.x - 0.75 * point.y
}

fn edge_moment(p0: Point, p1: Point) -> f64 {
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    dx * (1.0 - 0.5 * (p0.y + p1.y)) + dy * (2.0 + 0.5 * (p0.x + p1.x))
}

fn max_h1_error(values: &[f64], mesh: &sonium_fem::mesh::Mesh) -> f64 {
    values
        .iter()
        .zip(&mesh.nodes)
        .map(|(value, point)| (value - scalar_value(*point)).abs())
        .fold(0.0, f64::max)
}

fn max_hcurl_error(values: &[f64], mesh: &sonium_fem::mesh::Mesh) -> f64 {
    values
        .iter()
        .zip(MeshTopology::build(mesh).edges)
        .map(|(value, (start, end))| {
            (value - edge_moment(mesh.nodes[start], mesh.nodes[end])).abs()
        })
        .fold(0.0, f64::max)
}

fn run() -> Result<(usize, usize, f64, f64, usize, usize), String> {
    let level0 = unit_square_triangles(1);
    let mut level1 = level0.clone();
    let (_, history0) = refine_elements_with_history_checked(&mut level1, &[0])
        .map_err(|error| error.to_string())?;
    let level1_child = level1
        .elements
        .iter()
        .position(|element| element.parent_id == Some(history0.parents[0].element.id))
        .ok_or_else(|| "first refinement did not produce a child".to_string())?;
    let mut level2 = level1.clone();
    let (_, history1) = refine_elements_with_history_checked(&mut level2, &[level1_child])
        .map_err(|error| error.to_string())?;

    let meshes = [&level0, &level1, &level2];
    let histories = [history0, history1];
    let report = validate_refinement_transfer_chain_checked(
        &meshes,
        &histories,
        RefinementTransferDirection::Refine,
    )
    .map_err(|error| error.to_string())?;
    if report.stages.len() != 2 {
        return Err("multi-level report does not contain two stages".into());
    }

    let h1 = transfer_h1_state_with_refinement_chain_checked(
        &meshes,
        &level0
            .nodes
            .iter()
            .copied()
            .map(scalar_value)
            .collect::<Vec<_>>(),
        &histories,
        RefinementTransferDirection::Refine,
    )
    .map_err(|error| error.to_string())?;
    let h1_error = max_h1_error(&h1, &level2);

    let hcurl_space = HcurlSpace::new();
    let hcurl_spaces = [&hcurl_space, &hcurl_space, &hcurl_space];
    let hcurl = transfer_hcurl_order1_state_with_refinement_chain_checked(
        &meshes,
        &hcurl_spaces,
        &MeshTopology::build(&level0)
            .edges
            .iter()
            .map(|&(start, end)| edge_moment(level0.nodes[start], level0.nodes[end]))
            .collect::<Vec<_>>(),
        &histories,
        RefinementTransferDirection::Refine,
    )
    .map_err(|error| error.to_string())?;
    let hcurl_error = max_hcurl_error(&hcurl, &level2);

    let rt0 = HdivSpace::with_order(HdivOrder::Rt0);
    let rt1 = HdivSpace::with_order(HdivOrder::Rt1);
    let rt0_spaces = [&rt0, &rt0, &rt0];
    let rt1_spaces = [&rt1, &rt1, &rt1];
    let rt0_state = transfer_hdiv_rt0_state_with_refinement_chain_checked(
        &meshes,
        &rt0_spaces,
        &vec![0.0; rt0.num_global_dofs(&level0)],
        &histories,
        RefinementTransferDirection::Refine,
    )
    .map_err(|error| error.to_string())?;
    let rt1_state = transfer_hdiv_rt1_state_with_refinement_chain_checked(
        &meshes,
        &rt1_spaces,
        &vec![0.0; rt1.num_global_dofs(&level0)],
        &histories,
        RefinementTransferDirection::Refine,
    )
    .map_err(|error| error.to_string())?;
    if rt0_state.len() != rt0.num_global_dofs(&level2)
        || rt1_state.len() != rt1.num_global_dofs(&level2)
        || rt0_state.iter().any(|value| !value.is_finite())
        || rt1_state.iter().any(|value| !value.is_finite())
    {
        return Err("multi-level RT0/RT1 state is invalid".into());
    }

    let rejection_checks = usize::from(matches!(
        validate_refinement_transfer_chain_checked(
            &[&level0, &level2],
            &histories,
            RefinementTransferDirection::Refine,
        ),
        Err(InterpolationError::RefinementChainMismatch { .. })
    )) + usize::from(matches!(
        validate_refinement_transfer_chain_checked(
            &meshes,
            &[histories[1].clone(), histories[0].clone()],
            RefinementTransferDirection::Refine,
        ),
        Err(InterpolationError::RefinementChainMismatch { .. })
    ));
    if rejection_checks != 2 {
        return Err(format!(
            "only {rejection_checks} chain rejection checks passed"
        ));
    }

    Ok((
        report.stages.len(),
        level2.num_elements(),
        h1_error,
        hcurl_error,
        rt0_state.len(),
        rejection_checks,
    ))
}

fn main() {
    let result = match run() {
        Ok((stage_count, element_count, h1_error, hcurl_error, rt0_dofs, rejection_checks)) => {
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("stage_count", stage_count as f64)
                .minimum("stage_count", 2.0)
                .metric("element_count", element_count as f64)
                .minimum("element_count", 1.0)
                .metric("max_h1_error", h1_error)
                .maximum("max_h1_error", 2.0e-12)
                .metric("max_hcurl_error", hcurl_error)
                .maximum("max_hcurl_error", 2.0e-10)
                .metric("rt0_dof_count", rt0_dofs as f64)
                .minimum("rt0_dof_count", 1.0)
                .metric("rejection_checks", rejection_checks as f64)
                .minimum("rejection_checks", 2.0)
                .dimension("2-D multi-level adaptive P1 triangular lineage")
                .formulation(
                    "checked sequential H1, H(curl), RT0, and RT1 refine transfer with reverse coarsening order",
                )
                .input_checksum("unit-square-triangles-1-two-level-selective-refine:adaptive-lineage-chain-v1")
                .reference_checksum("analytic-linear-H1-Hcurl-zero-Hdiv:adaptive-lineage-chain-v1")
                .seed("fem-adaptive-lineage-chain:cpu:v1")
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
