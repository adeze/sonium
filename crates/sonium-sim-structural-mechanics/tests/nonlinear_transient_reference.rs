use sonium_sim_structural_mechanics::{
    NONLINEAR_TRANSIENT_SCHEMA_VERSION, NonlinearTransientModelAssumption, NonlinearTransientStudy,
    run_nonlinear_transient,
};

fn assert_close(left: f64, right: f64) {
    let tolerance = 1.0e-12 * left.abs().max(right.abs()).max(1.0);
    assert!(
        (left - right).abs() <= tolerance,
        "{left} differs from {right}"
    );
}

#[test]
fn nonlinear_transient_reference_is_finite_and_converged() {
    let result = run_nonlinear_transient(&NonlinearTransientStudy::introductory())
        .expect("nonlinear transient reference should converge");

    assert_eq!(result.schema_version, NONLINEAR_TRANSIENT_SCHEMA_VERSION);
    assert_eq!(
        result.model_assumption,
        NonlinearTransientModelAssumption::CubicHardeningSingleDof
    );
    assert_eq!(result.steps.len(), result.study.force_history_n.len());
    assert!(result.max_equilibrium_residual_n <= result.study.residual_tolerance_n);
    assert!(result.max_total_energy_j.is_finite());
    assert!(result.total_damping_dissipation_j >= 0.0);
    assert!(result.steps.iter().all(|step| {
        step.total_energy_j.is_finite()
            && step.nonlinear_potential_energy_j >= 0.0
            && (step.newton_iterations > 0 || step.step == 0)
    }));
    assert!(
        result
            .steps
            .iter()
            .skip(1)
            .any(|step| step.nonlinear_potential_energy_j > 0.0)
    );
}

#[test]
fn nonlinear_transient_result_round_trips() {
    let result = run_nonlinear_transient(&NonlinearTransientStudy::introductory())
        .expect("nonlinear transient reference should converge");
    let encoded = serde_json::to_string(&result).expect("serialize result");
    let decoded: sonium_sim_structural_mechanics::NonlinearTransientResult =
        serde_json::from_str(&encoded).expect("deserialize result");

    assert_eq!(decoded.schema_version, result.schema_version);
    assert_eq!(decoded.study, result.study);
    assert_eq!(decoded.steps.len(), result.steps.len());
    assert_eq!(decoded.model_assumption, result.model_assumption);
    assert_eq!(decoded.execution, result.execution);
    assert_close(decoded.final_displacement_m, result.final_displacement_m);
    assert_close(
        decoded.final_velocity_m_per_s,
        result.final_velocity_m_per_s,
    );
    assert_close(
        decoded.max_equilibrium_residual_n,
        result.max_equilibrium_residual_n,
    );
}

#[test]
fn nonlinear_transient_rejects_invalid_inputs() {
    let mut negative_cubic = NonlinearTransientStudy::introductory();
    negative_cubic.cubic_stiffness_n_per_m3 = -1.0;
    assert!(run_nonlinear_transient(&negative_cubic).is_err());

    let mut short_history = NonlinearTransientStudy::introductory();
    short_history.force_history_n.truncate(1);
    assert!(run_nonlinear_transient(&short_history).is_err());
}
