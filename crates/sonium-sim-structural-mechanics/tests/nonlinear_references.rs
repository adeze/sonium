use sonium_sim_structural_mechanics::{
    FrictionlessContactStudy, Hyperelastic3DStudy, RandomVibrationStudy, ReliabilityStudy,
    StructuralMechanicsError, StructuralModelAssumption, run_frictionless_contact,
    run_hyperelastic_3d, run_random_vibration, run_reliability_study,
};

#[test]
fn public_hyperelastic_3d_reference_preserves_newton_and_energy_gates() {
    let study = Hyperelastic3DStudy::introductory();
    let result = run_hyperelastic_3d(&study).expect("bounded hyperelastic reference should solve");

    assert_eq!(result.load_steps, study.load_steps);
    assert!(result.converged);
    assert!(result.newton_iterations > 0);
    assert!(result.final_residual < 1.0e-8);
    assert!(result.max_affine_displacement_error_m < 1.0e-10);
    assert!(result.deformation_gradient_error < 1.0e-10);
    assert!(result.relative_energy_error < 1.0e-10);
    assert!(result.constitutive_gradient_error < 1.0e-8);
    assert!((result.taylor_remainder_ratio - 0.25).abs() < 0.02);
}

#[test]
fn public_contact_reference_reports_penalty_activation_and_balance() {
    let study = FrictionlessContactStudy::introductory();
    let result = run_frictionless_contact(&study).expect("bounded contact reference should solve");
    let expected_displacement_m = (study.max_force_n
        + study.contact_penalty_stiffness_n_per_m * study.gap_m)
        / (study.support_stiffness_n_per_m + study.contact_penalty_stiffness_n_per_m);

    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::FrictionlessContact
    );
    assert_eq!(result.contact_engaged_step, Some(6));
    assert!((result.final_displacement_m - expected_displacement_m).abs() < 1.0e-12);
    assert!(result.final_penetration_m > 0.0);
    assert!(!result.steps[4].contact_active);
    assert!(result.steps[5].contact_active);
    assert!(result.max_equilibrium_residual_n < 1.0e-9);
    assert!(result.steps.iter().all(|step| {
        step.converged
            && step.equilibrium_residual_n.is_finite()
            && step.contact_force_n.is_finite()
    }));
}

#[test]
fn public_random_vibration_and_reliability_references_preserve_screening_gates() {
    let vibration = run_random_vibration(&RandomVibrationStudy::introductory())
        .expect("bounded random-vibration reference should solve");
    assert_eq!(
        vibration.model_assumption,
        StructuralModelAssumption::RandomVibrationSingleDof
    );
    assert_eq!(vibration.displacement_psd_m2_per_hz.len(), 101);
    assert!(vibration.displacement_variance_m2 > 0.0);
    assert!(vibration.rms_displacement_m > 0.0);
    assert!(vibration.peak_displacement_m > vibration.rms_displacement_m);
    assert!(vibration.maximum_dynamic_amplification > 0.0);
    assert!(vibration.feasible);

    let reliability = run_reliability_study(&ReliabilityStudy::introductory())
        .expect("bounded reliability reference should solve");
    assert_eq!(
        reliability.model_assumption,
        StructuralModelAssumption::EmpiricalReliability
    );
    assert_eq!(reliability.failure_count, 1);
    assert!((reliability.failure_probability - 0.2).abs() < 1.0e-12);
    assert!((reliability.mean_margin_n - 22.0).abs() < 1.0e-12);
    assert!(reliability.reliability_index > 0.0);
    assert!(reliability.feasible);
}

#[test]
fn public_nonlinear_and_screening_inputs_fail_closed() {
    let mut hyperelastic = Hyperelastic3DStudy::introductory();
    hyperelastic.nx = 0;
    assert!(matches!(
        run_hyperelastic_3d(&hyperelastic),
        Err(StructuralMechanicsError::InvalidConfiguration {
            field: "mesh_resolution",
            ..
        })
    ));

    let mut contact = FrictionlessContactStudy::introductory();
    contact.gap_m = -1.0;
    assert!(matches!(
        run_frictionless_contact(&contact),
        Err(StructuralMechanicsError::InvalidConfiguration {
            field: "frictionless_contact.gap_m",
            ..
        })
    ));

    let mut vibration = RandomVibrationStudy::introductory();
    vibration.frequencies_hz[1] = vibration.frequencies_hz[0];
    assert!(matches!(
        run_random_vibration(&vibration),
        Err(StructuralMechanicsError::InvalidConfiguration {
            field: "random_vibration.frequency_psd_grid",
            ..
        })
    ));

    let mut reliability = ReliabilityStudy::introductory();
    reliability.resistance_samples_n.pop();
    assert!(matches!(
        run_reliability_study(&reliability),
        Err(StructuralMechanicsError::InvalidConfiguration {
            field: "reliability.samples",
            ..
        })
    ));
}
