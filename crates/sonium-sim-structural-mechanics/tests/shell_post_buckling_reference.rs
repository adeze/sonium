use sonium_sim_structural_mechanics::{
    COROTATIONAL_SHELL_POST_BUCKLING_SCHEMA_VERSION, CorotationalShellPostBucklingStudy,
    StructuralMechanicsError, StructuralModelAssumption, run_corotational_shell_post_buckling,
};

fn assert_close(left: f64, right: f64) {
    let tolerance = 1.0e-12 * left.abs().max(right.abs()).max(1.0);
    assert!((left - right).abs() <= tolerance, "{left} != {right}");
}

#[test]
fn production_corotational_shell_continuation_converges_with_finite_diagnostics() {
    let study = CorotationalShellPostBucklingStudy::default();
    let result = run_corotational_shell_post_buckling(&study)
        .expect("bounded production-shell continuation should converge");

    assert_eq!(
        result.schema_version,
        COROTATIONAL_SHELL_POST_BUCKLING_SCHEMA_VERSION
    );
    assert_eq!(result.nodes, (study.nx + 1) * (study.ny + 1));
    assert_eq!(result.elements, 2 * study.nx * study.ny);
    assert_eq!(result.dofs, result.nodes * 6);
    assert_eq!(result.steps.len(), study.continuation_steps);
    assert!(result.converged);
    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::CorotationalShell
    );
    assert!(result.peak_load_factor.is_finite());
    assert!(result.final_load_factor.is_finite());
    assert!(result.final_control_reaction_n.is_finite());
    assert!(result.max_equilibrium_residual_n.is_finite());

    for step in &result.steps {
        assert!(step.converged);
        assert!(step.load_factor.is_finite());
        assert!(step.applied_load_n.is_finite());
        assert!(step.control_reaction_n.is_finite());
        assert!(step.equilibrium_residual_n.is_finite());
        assert!(step.control_residual_m.is_finite());
        assert!(step.max_translation_m.is_finite());
        assert!(
            step.equilibrium_residual_n
                <= study.newton_residual_tolerance * study.reference_load_n.max(1.0)
        );
        assert!(
            step.control_residual_m
                <= study.newton_residual_tolerance * step.target_displacement_m.abs().max(1.0)
        );
    }

    if let Some(turning_point) = result.turning_point_step {
        assert!(turning_point >= 2);
        assert!(turning_point <= result.steps.len());
        let before = result.steps[turning_point - 2].load_factor
            - result.steps[turning_point - 3].load_factor;
        let after = result.steps[turning_point - 1].load_factor
            - result.steps[turning_point - 2].load_factor;
        assert!(before * after < 0.0);
    }
}

#[test]
fn production_corotational_shell_result_round_trips_through_json() {
    let mut study = CorotationalShellPostBucklingStudy::default();
    study.continuation_steps = 4;
    study.max_newton_iterations = 160;
    let result = run_corotational_shell_post_buckling(&study)
        .expect("bounded production-shell continuation should converge");

    let encoded = serde_json::to_string(&result).expect("shell result should serialize");
    let decoded: sonium_sim_structural_mechanics::CorotationalShellPostBucklingResult =
        serde_json::from_str(&encoded).expect("shell result should deserialize");
    assert_eq!(decoded.schema_version, result.schema_version);
    assert_eq!(decoded.study, result.study);
    assert_eq!(decoded.nodes, result.nodes);
    assert_eq!(decoded.elements, result.elements);
    assert_eq!(decoded.dofs, result.dofs);
    assert_eq!(decoded.control_node, result.control_node);
    assert_eq!(decoded.control_dof, result.control_dof);
    assert_eq!(decoded.fixed_nodes, result.fixed_nodes);
    assert_eq!(decoded.turning_point_step, result.turning_point_step);
    assert_eq!(decoded.model_assumption, result.model_assumption);
    assert_eq!(decoded.execution, result.execution);
    assert_close(decoded.peak_load_factor, result.peak_load_factor);
    assert_close(decoded.final_load_factor, result.final_load_factor);
    assert_close(
        decoded.final_control_reaction_n,
        result.final_control_reaction_n,
    );
    assert_close(
        decoded.max_equilibrium_residual_n,
        result.max_equilibrium_residual_n,
    );
    assert_eq!(decoded.converged, result.converged);
    assert_eq!(decoded.steps.len(), result.steps.len());
    for (decoded_step, result_step) in decoded.steps.iter().zip(&result.steps) {
        assert_eq!(decoded_step.step, result_step.step);
        assert_eq!(
            decoded_step.newton_iterations,
            result_step.newton_iterations
        );
        assert_eq!(decoded_step.converged, result_step.converged);
        assert_close(
            decoded_step.target_displacement_m,
            result_step.target_displacement_m,
        );
        assert_close(decoded_step.load_factor, result_step.load_factor);
        assert_close(decoded_step.applied_load_n, result_step.applied_load_n);
        assert_close(
            decoded_step.control_reaction_n,
            result_step.control_reaction_n,
        );
        assert_close(
            decoded_step.equilibrium_residual_n,
            result_step.equilibrium_residual_n,
        );
        assert_close(
            decoded_step.control_residual_m,
            result_step.control_residual_m,
        );
        assert_close(
            decoded_step.max_translation_m,
            result_step.max_translation_m,
        );
    }
}

#[test]
fn production_corotational_shell_rejects_invalid_displacement_controls() {
    let mut study = CorotationalShellPostBucklingStudy::default();
    study.control_displacement_m = 0.0;

    let error = run_corotational_shell_post_buckling(&study)
        .expect_err("zero displacement control must be rejected before assembly");
    assert!(matches!(
        error,
        StructuralMechanicsError::InvalidConfiguration {
            field: "shell_post_buckling.displacement_controls",
            ..
        }
    ));
}
