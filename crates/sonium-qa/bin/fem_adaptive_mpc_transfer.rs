//! CPU-only reduced-state adaptive FEM transfer with real MPC constraints.

use num_complex::Complex64;
use sonium_fem::boundary::periodic::{MpcError, MpcSystem, MpcTerm, MultiPointConstraint};
use sonium_fem::mesh::refinement::refine_elements_with_history_checked;
use sonium_fem::mesh::{
    InterpolationError, RefinementTransferDirection,
    transfer_h1_state_with_refinement_chain_and_mpc_checked,
    transfer_h1_state_with_refinement_history_and_mpc_checked,
    transfer_hcurl_order1_state_with_refinement_chain_and_mpc_checked,
    transfer_hdiv_rt0_state_with_refinement_chain_and_mpc_checked,
    transfer_hdiv_rt1_state_with_refinement_chain_and_mpc_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace, HdivOrder, HdivSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.adaptive-mpc-transfer.v1";
const TOLERANCE: f64 = 2.0e-10;

fn one_constraint(
    num_full_dofs: usize,
    slave: usize,
    master: usize,
    coefficient: f64,
) -> MpcSystem {
    MpcSystem::new(
        num_full_dofs,
        &[MultiPointConstraint {
            slave,
            masters: vec![MpcTerm {
                dof: master,
                coefficient: Complex64::new(coefficient, 0.0),
            }],
        }],
    )
    .map_err(|error| error.to_string())
    .expect("valid QA MPC fixture")
}

fn reduced_state(system: &MpcSystem, value: f64) -> Vec<f64> {
    vec![value; system.num_reduced_dofs()]
}

fn max_constant_error(values: &[f64], expected: f64) -> f64 {
    values
        .iter()
        .map(|value| (value - expected).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, usize, usize, usize), String> {
    let level0 = unit_square_triangles(1);
    let mut level1 = level0.clone();
    let (_, history0) = refine_elements_with_history_checked(&mut level1, &[0])
        .map_err(|error| error.to_string())?;
    let child_index = level1
        .elements
        .iter()
        .position(|element| element.parent_id == Some(history0.parents[0].element.id))
        .ok_or("missing first refined child")?;
    let mut level2 = level1.clone();
    let (_, history1) = refine_elements_with_history_checked(&mut level2, &[child_index])
        .map_err(|error| error.to_string())?;
    let meshes = [&level0, &level1, &level2];
    let histories = [history0, history1];

    let h1_mpcs = [
        one_constraint(level0.nodes.len(), 2, 0, 1.0),
        one_constraint(level1.nodes.len(), 2, 0, 1.0),
        one_constraint(level2.nodes.len(), 2, 0, 1.0),
    ];
    let h1_mpc_refs = [&h1_mpcs[0], &h1_mpcs[1], &h1_mpcs[2]];
    let h1_source = reduced_state(&h1_mpcs[0], 2.0);
    let h1_level2 = transfer_h1_state_with_refinement_chain_and_mpc_checked(
        &meshes,
        &h1_source,
        &h1_mpc_refs,
        &histories,
        RefinementTransferDirection::Refine,
        TOLERANCE,
    )
    .map_err(|error| error.to_string())?;
    let h1_error = max_constant_error(&h1_level2, 2.0);
    let h1_level0 = transfer_h1_state_with_refinement_chain_and_mpc_checked(
        &meshes,
        &h1_level2,
        &h1_mpc_refs,
        &histories,
        RefinementTransferDirection::Coarsen,
        TOLERANCE,
    )
    .map_err(|error| error.to_string())?;
    if h1_level0 != h1_source {
        return Err("H1 reverse reduced-state transfer was not exact".into());
    }

    let hcurl_spaces = [HcurlSpace::new(), HcurlSpace::new(), HcurlSpace::new()];
    let hcurl_mpcs = [
        one_constraint(hcurl_spaces[0].num_global_dofs(&level0), 1, 0, 1.0),
        one_constraint(hcurl_spaces[1].num_global_dofs(&level1), 1, 0, 1.0),
        one_constraint(hcurl_spaces[2].num_global_dofs(&level2), 1, 0, 1.0),
    ];
    let hcurl_spaces_refs = [&hcurl_spaces[0], &hcurl_spaces[1], &hcurl_spaces[2]];
    let hcurl_mpc_refs = [&hcurl_mpcs[0], &hcurl_mpcs[1], &hcurl_mpcs[2]];
    let hcurl_level2 = transfer_hcurl_order1_state_with_refinement_chain_and_mpc_checked(
        &meshes,
        &hcurl_spaces_refs,
        &reduced_state(&hcurl_mpcs[0], 0.0),
        &hcurl_mpc_refs,
        &histories,
        RefinementTransferDirection::Refine,
        TOLERANCE,
    )
    .map_err(|error| error.to_string())?;

    let rt0_spaces = [
        HdivSpace::with_order(HdivOrder::Rt0),
        HdivSpace::with_order(HdivOrder::Rt0),
        HdivSpace::with_order(HdivOrder::Rt0),
    ];
    let rt0_mpcs = [
        one_constraint(rt0_spaces[0].num_global_dofs(&level0), 1, 0, 1.0),
        one_constraint(rt0_spaces[1].num_global_dofs(&level1), 1, 0, 1.0),
        one_constraint(rt0_spaces[2].num_global_dofs(&level2), 1, 0, 1.0),
    ];
    let rt0_spaces_refs = [&rt0_spaces[0], &rt0_spaces[1], &rt0_spaces[2]];
    let rt0_mpc_refs = [&rt0_mpcs[0], &rt0_mpcs[1], &rt0_mpcs[2]];
    let rt0_level2 = transfer_hdiv_rt0_state_with_refinement_chain_and_mpc_checked(
        &meshes,
        &rt0_spaces_refs,
        &reduced_state(&rt0_mpcs[0], 0.0),
        &rt0_mpc_refs,
        &histories,
        RefinementTransferDirection::Refine,
        TOLERANCE,
    )
    .map_err(|error| error.to_string())?;

    let rt1_spaces = [
        HdivSpace::with_order(HdivOrder::Rt1),
        HdivSpace::with_order(HdivOrder::Rt1),
        HdivSpace::with_order(HdivOrder::Rt1),
    ];
    let rt1_mpcs = [
        one_constraint(rt1_spaces[0].num_global_dofs(&level0), 1, 0, 1.0),
        one_constraint(rt1_spaces[1].num_global_dofs(&level1), 1, 0, 1.0),
        one_constraint(rt1_spaces[2].num_global_dofs(&level2), 1, 0, 1.0),
    ];
    let rt1_spaces_refs = [&rt1_spaces[0], &rt1_spaces[1], &rt1_spaces[2]];
    let rt1_mpc_refs = [&rt1_mpcs[0], &rt1_mpcs[1], &rt1_mpcs[2]];
    let rt1_level2 = transfer_hdiv_rt1_state_with_refinement_chain_and_mpc_checked(
        &meshes,
        &rt1_spaces_refs,
        &reduced_state(&rt1_mpcs[0], 0.0),
        &rt1_mpc_refs,
        &histories,
        RefinementTransferDirection::Refine,
        TOLERANCE,
    )
    .map_err(|error| error.to_string())?;

    if hcurl_level2.iter().any(|value| !value.is_finite())
        || rt0_level2.iter().any(|value| !value.is_finite())
        || rt1_level2.iter().any(|value| !value.is_finite())
    {
        return Err("MPC-aware vector transfer produced a non-finite value".into());
    }

    let invalid_target = one_constraint(level1.nodes.len(), 2, 0, 2.0);
    let mut rejection_checks = 0;
    if matches!(
        transfer_h1_state_with_refinement_history_and_mpc_checked(
            &level0,
            &level1,
            &h1_source,
            &h1_mpcs[0],
            &invalid_target,
            &histories[0],
            RefinementTransferDirection::Refine,
            TOLERANCE,
        ),
        Err(InterpolationError::MpcStateError {
            role: "target",
            error: MpcError::StateConstraintResidual { .. }
        })
    ) {
        rejection_checks += 1;
    }

    let complex_source = MpcSystem::new(
        level0.nodes.len(),
        &[MultiPointConstraint {
            slave: 2,
            masters: vec![MpcTerm {
                dof: 0,
                coefficient: Complex64::new(1.0, 1.0),
            }],
        }],
    )
    .map_err(|error| error.to_string())?;
    if matches!(
        transfer_h1_state_with_refinement_history_and_mpc_checked(
            &level0,
            &level1,
            &reduced_state(&complex_source, 0.0),
            &complex_source,
            &h1_mpcs[1],
            &histories[0],
            RefinementTransferDirection::Refine,
            TOLERANCE,
        ),
        Err(InterpolationError::MpcStateError {
            role: "source",
            error: MpcError::ComplexCoefficientInRealSystem { .. }
        })
    ) {
        rejection_checks += 1;
    }

    if rejection_checks != 2 {
        return Err(format!(
            "only {rejection_checks} typed MPC rejections passed"
        ));
    }
    Ok((
        h1_error,
        hcurl_level2.len() + rt0_level2.len() + rt1_level2.len(),
        rejection_checks,
        h1_level2.len(),
    ))
}

fn main() {
    let result = match run() {
        Ok((h1_error, vector_dofs, rejection_checks, h1_dofs)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("max_h1_error", h1_error)
            .maximum("max_h1_error", TOLERANCE)
            .metric("vector_target_dofs", vector_dofs as f64)
            .minimum("vector_target_dofs", 1.0)
            .metric("h1_target_reduced_dofs", h1_dofs as f64)
            .minimum("h1_target_reduced_dofs", 1.0)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 2.0)
            .dimension("2-D affine triangular two-stage adaptive reduced-state transfer")
            .formulation(
                "source recovery, checked H1/H(curl)/RT0/RT1 lineage transfer, target MPC restriction",
            )
            .input_checksum("unit-square-triangles-1-two-stage-selective-refine:adaptive-mpc-v1")
            .reference_checksum("constant-H1-zero-vector-real-MPC:adaptive-mpc-v1")
            .seed("fem-adaptive-mpc-transfer:cpu:v1"),
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
