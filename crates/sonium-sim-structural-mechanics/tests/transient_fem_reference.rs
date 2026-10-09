use sonium_sim_structural_mechanics::{
    StructuralMechanicsError, TransientFemStudy, run_transient_fem,
};

fn small_study() -> TransientFemStudy {
    let mut study = TransientFemStudy::introductory();
    study.bracket.geometry.nx = 2;
    study.bracket.geometry.ny = 2;
    study.time_step_s = 1.0e-5;
    study.load_scales = vec![0.0, 0.25, 0.5, 1.0, 0.0];
    study.equilibrium_tolerance_n = 1.0e-5;
    study
}

#[test]
fn assembled_fem_transient_reports_newmark_equilibrium_and_energy() {
    let result = run_transient_fem(&small_study()).expect("assembled FEM transient");

    assert_eq!(result.schema_version, 1);
    assert_eq!(result.steps.len(), 5);
    assert_eq!(result.nodes, 9);
    assert_eq!(result.final_displacement.len(), result.nodes);
    assert!(result.first_bending_frequency_hz.is_finite());
    assert!(result.first_bending_frequency_hz > 0.0);
    assert!(result.damping_coefficient_s.is_finite());
    assert!(result.converged);
    assert!(result.max_dynamic_equilibrium_residual_n < 1.0e-7);
    assert!(result.max_total_energy_j.is_finite());
    assert!(result.steps.iter().all(|step| {
        step.dynamic_equilibrium_residual_n.is_finite()
            && step.total_energy_j.is_finite()
            && step.damping_power_w.is_finite()
    }));
    assert!(
        result
            .final_displacement
            .iter()
            .any(|displacement| displacement[0].abs() + displacement[1].abs() > 0.0)
    );
    assert_eq!(
        result.execution.capability_id,
        "sonium-structural.cpu-reference.f64.transient-fem-2d.v1"
    );
}

#[test]
fn assembled_fem_transient_rejects_nonzero_initial_load() {
    let mut study = small_study();
    study.load_scales[0] = 1.0;

    assert!(matches!(
        run_transient_fem(&study),
        Err(StructuralMechanicsError::InvalidConfiguration {
            field: "transient_fem.load_scales[0]",
            ..
        })
    ));
}
