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
    .unwrap()
}

fn constant_reduced_state(system: &MpcSystem, value: f64) -> Vec<f64> {
    vec![value; system.num_reduced_dofs()]
}

fn zero_reduced_state(system: &MpcSystem) -> Vec<f64> {
    constant_reduced_state(system, 0.0)
}

fn three_stage_meshes() -> (
    sonium_fem::mesh::Mesh,
    sonium_fem::mesh::Mesh,
    sonium_fem::mesh::Mesh,
    sonium_fem::mesh::refinement::RefinementHistory,
    sonium_fem::mesh::refinement::RefinementHistory,
) {
    let level0 = unit_square_triangles(1);
    let mut level1 = level0.clone();
    let (_, history0) = refine_elements_with_history_checked(&mut level1, &[0]).unwrap();
    let child_index = level1
        .elements
        .iter()
        .position(|element| element.parent_id == Some(history0.parents[0].element.id))
        .unwrap();
    let mut level2 = level1.clone();
    let (_, history1) = refine_elements_with_history_checked(&mut level2, &[child_index]).unwrap();
    (level0, level1, level2, history0, history1)
}

#[test]
fn adaptive_chain_transfers_reduced_h1_hcurl_rt0_and_rt1_states() {
    let (level0, level1, level2, history0, history1) = three_stage_meshes();
    let meshes = [&level0, &level1, &level2];
    let histories = [history0.clone(), history1.clone()];

    let h1_mpcs = [
        one_constraint(level0.nodes.len(), 2, 0, 1.0),
        one_constraint(level1.nodes.len(), 2, 0, 1.0),
        one_constraint(level2.nodes.len(), 2, 0, 1.0),
    ];
    let h1_mpc_refs = [&h1_mpcs[0], &h1_mpcs[1], &h1_mpcs[2]];
    let h1_source = constant_reduced_state(&h1_mpcs[0], 2.0);
    let h1_level2 = transfer_h1_state_with_refinement_chain_and_mpc_checked(
        &meshes,
        &h1_source,
        &h1_mpc_refs,
        &histories,
        RefinementTransferDirection::Refine,
        TOLERANCE,
    )
    .unwrap();
    assert_eq!(h1_level2.len(), h1_mpcs[2].num_reduced_dofs());
    assert!(
        h1_level2
            .iter()
            .all(|value| (*value - 2.0).abs() <= TOLERANCE)
    );

    let h1_level0 = transfer_h1_state_with_refinement_chain_and_mpc_checked(
        &meshes,
        &h1_level2,
        &h1_mpc_refs,
        &histories,
        RefinementTransferDirection::Coarsen,
        TOLERANCE,
    )
    .unwrap();
    assert_eq!(h1_level0, h1_source);

    let hcurl_spaces = [HcurlSpace::new(), HcurlSpace::new(), HcurlSpace::new()];
    let hcurl_mpcs = [
        one_constraint(hcurl_spaces[0].num_global_dofs(&level0), 1, 0, 1.0),
        one_constraint(hcurl_spaces[1].num_global_dofs(&level1), 1, 0, 1.0),
        one_constraint(hcurl_spaces[2].num_global_dofs(&level2), 1, 0, 1.0),
    ];
    let hcurl_mpc_refs = [&hcurl_mpcs[0], &hcurl_mpcs[1], &hcurl_mpcs[2]];
    let hcurl_spaces_refs = [&hcurl_spaces[0], &hcurl_spaces[1], &hcurl_spaces[2]];
    let hcurl_level2 = transfer_hcurl_order1_state_with_refinement_chain_and_mpc_checked(
        &meshes,
        &hcurl_spaces_refs,
        &zero_reduced_state(&hcurl_mpcs[0]),
        &hcurl_mpc_refs,
        &histories,
        RefinementTransferDirection::Refine,
        TOLERANCE,
    )
    .unwrap();
    assert_eq!(hcurl_level2.len(), hcurl_mpcs[2].num_reduced_dofs());
    assert!(hcurl_level2.iter().all(|value| value.abs() <= TOLERANCE));

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
    let rt0_space_refs = [&rt0_spaces[0], &rt0_spaces[1], &rt0_spaces[2]];
    let rt0_mpc_refs = [&rt0_mpcs[0], &rt0_mpcs[1], &rt0_mpcs[2]];
    let rt0_level2 = transfer_hdiv_rt0_state_with_refinement_chain_and_mpc_checked(
        &meshes,
        &rt0_space_refs,
        &zero_reduced_state(&rt0_mpcs[0]),
        &rt0_mpc_refs,
        &histories,
        RefinementTransferDirection::Refine,
        TOLERANCE,
    )
    .unwrap();
    assert_eq!(rt0_level2.len(), rt0_mpcs[2].num_reduced_dofs());
    assert!(rt0_level2.iter().all(|value| value.abs() <= TOLERANCE));

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
    let rt1_space_refs = [&rt1_spaces[0], &rt1_spaces[1], &rt1_spaces[2]];
    let rt1_mpc_refs = [&rt1_mpcs[0], &rt1_mpcs[1], &rt1_mpcs[2]];
    let rt1_level2 = transfer_hdiv_rt1_state_with_refinement_chain_and_mpc_checked(
        &meshes,
        &rt1_space_refs,
        &zero_reduced_state(&rt1_mpcs[0]),
        &rt1_mpc_refs,
        &histories,
        RefinementTransferDirection::Refine,
        TOLERANCE,
    )
    .unwrap();
    assert_eq!(rt1_level2.len(), rt1_mpcs[2].num_reduced_dofs());
    assert!(rt1_level2.iter().all(|value| value.abs() <= TOLERANCE));
}

#[test]
fn adaptive_mpc_transfer_rejects_complex_bad_and_malformed_contracts() {
    let (source, target, _, history, _) = three_stage_meshes();
    let source_mpc = one_constraint(source.nodes.len(), 2, 0, 1.0);
    let source_values = constant_reduced_state(&source_mpc, 2.0);
    let invalid_target = one_constraint(target.nodes.len(), 2, 0, 2.0);
    let error = transfer_h1_state_with_refinement_history_and_mpc_checked(
        &source,
        &target,
        &source_values,
        &source_mpc,
        &invalid_target,
        &history,
        RefinementTransferDirection::Refine,
        TOLERANCE,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        InterpolationError::MpcStateError {
            role: "target",
            error: MpcError::StateConstraintResidual { .. }
        }
    ));

    let complex_source = MpcSystem::new(
        source.nodes.len(),
        &[MultiPointConstraint {
            slave: 2,
            masters: vec![MpcTerm {
                dof: 0,
                coefficient: Complex64::new(1.0, 1.0),
            }],
        }],
    )
    .unwrap();
    let complex_error = transfer_h1_state_with_refinement_history_and_mpc_checked(
        &source,
        &target,
        &constant_reduced_state(&complex_source, 0.0),
        &complex_source,
        &one_constraint(target.nodes.len(), 2, 0, 1.0),
        &history,
        RefinementTransferDirection::Refine,
        TOLERANCE,
    )
    .unwrap_err();
    assert!(matches!(
        complex_error,
        InterpolationError::MpcStateError {
            role: "source",
            error: MpcError::ComplexCoefficientInRealSystem { .. }
        }
    ));

    let target_mpc = one_constraint(target.nodes.len(), 2, 0, 1.0);
    let tolerance_error = transfer_h1_state_with_refinement_history_and_mpc_checked(
        &source,
        &target,
        &source_values,
        &source_mpc,
        &target_mpc,
        &history,
        RefinementTransferDirection::Refine,
        -1.0,
    )
    .unwrap_err();
    assert!(matches!(
        tolerance_error,
        InterpolationError::MpcStateError {
            role: "target",
            error: MpcError::InvalidTolerance(_)
        }
    ));

    let mesh_refs = [&source, &target];
    let chain_error = transfer_h1_state_with_refinement_chain_and_mpc_checked(
        &mesh_refs,
        &source_values,
        &[],
        &[history],
        RefinementTransferDirection::Refine,
        TOLERANCE,
    )
    .unwrap_err();
    assert!(matches!(
        chain_error,
        InterpolationError::MpcChainLengthMismatch {
            expected: 2,
            actual: 0
        }
    ));
}
