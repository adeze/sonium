use sonium_sim_structural_mechanics::{
    AxisymmetricCylinderStudy, BracketShapeOptimizationStudy, BracketStudy, CompositeLaminateStudy,
    PhaseFieldDamageStudy, PhaseFieldForceControlledStudy, Plate2DStudy, ShellStaticStudy,
    Solid3DStudy, StructuralModelAssumption, ThermoelasticStudy, Viscoelastic2DStudy,
    run_axisymmetric_cylinder, run_bracket_shape_optimization, run_composite_laminate_static,
    run_phase_field_damage, run_phase_field_damage_force_controlled, run_plate2d_static,
    run_shell_static, run_solid3d_static, run_thermoelastic_static, run_viscoelastic_2d,
};

#[test]
fn axisymmetric_public_reference_preserves_lame_and_balance_gates() {
    let mut study = AxisymmetricCylinderStudy::introductory();
    study.geometry.radial_elements = 8;
    let result = run_axisymmetric_cylinder(&study).expect("bounded axisymmetric solve");

    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::AxisymmetricElasticity
    );
    assert!(result.converged);
    assert!(result.solver_residual.is_finite());
    assert!(result.free_equilibrium_residual_n < 1.0e-6);
    assert!(result.energy_balance_relative_error < 1.0e-10);
    assert!(result.max_radial_displacement_relative_error < 0.03);
    assert!(result.max_radial_stress_relative_error < 0.15);
    assert!(result.max_hoop_stress_relative_error < 0.15);
}

#[test]
fn composite_public_reference_preserves_laminate_failure_and_energy_gates() {
    let mut study = CompositeLaminateStudy::introductory();
    study.geometry.nx = 4;
    study.geometry.ny = 2;
    let result = run_composite_laminate_static(&study).expect("bounded laminate solve");

    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::CompositeLaminate
    );
    assert!(result.converged);
    assert!(result.analytical_relative_error < 1.0e-8);
    assert!(result.load_integration_relative_error < 1.0e-10);
    assert!(result.free_equilibrium_residual_n < 1.0e-6);
    assert!(result.force_balance_residual_n < 1.0e-6);
    assert!(result.energy_balance_relative_error < 1.0e-10);
    assert_eq!(result.ply_failure.len(), 1);
    assert!(result.max_tsai_hill_index.is_finite());
    assert!(result.max_stress_index.is_finite());
    assert!(result.max_tsai_hill_index < 1.0);
    assert!(result.max_stress_index < 1.0);
}

#[test]
fn thermoelastic_public_reference_preserves_affine_free_expansion() {
    let mut study = ThermoelasticStudy::introductory();
    study.geometry.nx = 4;
    study.geometry.ny = 2;
    let result = run_thermoelastic_static(&study).expect("bounded thermoelastic solve");

    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::PlaneStress
    );
    assert!(result.converged);
    assert!(result.max_affine_displacement_error_m < 1.0e-10);
    assert!(result.max_mechanical_stress_pa.is_finite());
    assert!(
        result.max_mechanical_stress_pa
            < 1.0e-9 * study.material.youngs_modulus_pa * result.thermal_strain.abs()
    );
    assert!(result.free_equilibrium_residual_n < 1.0e-2);
    assert!(result.force_balance_residual_n < 1.0e-2);
    assert!(result.energy_balance_relative_error < 1.0e-10);
    assert!(
        result
            .actual_corner_displacement_m
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
fn viscoelastic_public_reference_preserves_complex_scaling_and_phase() {
    let mut study = Viscoelastic2DStudy::introductory();
    study.geometry.nx = 4;
    study.geometry.ny = 2;
    study.material.loss_factor = 0.35;
    let result = run_viscoelastic_2d(&study).expect("bounded viscoelastic solve");

    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::ViscoelasticPlaneStress
    );
    assert!(result.converged);
    assert!(result.max_displacement_m.is_finite());
    assert!(result.free_equilibrium_residual_n < 1.0e-7);
    assert!(result.force_balance_residual_n < 1.0e-7);
    assert!(result.energy_balance_relative_error < 1.0e-10);
    assert!(result.compliance_scaling_relative_error < 1.0e-8);
    assert!(result.displacement_scaling_relative_error < 1.0e-8);
    assert!((result.phase_lag_rad + 0.35_f64.atan()).abs() < 1.0e-8);
    assert!(
        result
            .complex_compliance_j
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
fn plate_public_reference_preserves_navier_and_load_balance_gates() {
    let mut study = Plate2DStudy::introductory();
    study.geometry.nx = 20;
    study.geometry.ny = 20;
    let result = run_plate2d_static(&study).expect("bounded plate solve");

    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::MindlinPlate
    );
    assert!(result.converged);
    assert!(result.center_deflection_m > 0.0);
    assert!(result.analytical_relative_error < 0.10);
    assert!(result.load_integration_relative_error < 0.02);
    assert!(
        result.free_equilibrium_residual_n < 1.0e-6 * result.total_applied_load_n.abs().max(1.0)
    );
    assert!(result.force_balance_residual_n < 1.0e-6 * result.total_applied_load_n.abs().max(1.0));
    assert!(result.strain_energy_j > 0.0);
    assert!(result.energy_balance_relative_error < 1.0e-10);
}

#[test]
fn phase_field_public_references_preserve_monotone_damage_and_load_control() {
    let mut study = PhaseFieldDamageStudy::introductory();
    study.nx = 4;
    study.ny = 2;
    study.load_steps = 2;
    let prescribed = run_phase_field_damage(&study).expect("bounded prescribed phase-field solve");

    assert_eq!(
        prescribed.loading_control,
        sonium_sim_structural_mechanics::PhaseFieldLoadingControl::PrescribedExtension
    );
    assert!(prescribed.converged);
    assert!(prescribed.history_monotone);
    assert!(prescribed.damage_monotone);
    assert!(prescribed.max_damage.is_finite());
    assert!((0.0..=1.0).contains(&prescribed.max_damage));
    assert!(prescribed.final_mechanics_residual.is_finite());
    assert!(prescribed.final_free_equilibrium_residual.is_finite());

    let force_study = PhaseFieldForceControlledStudy {
        total_force_n: study.youngs_modulus_pa * study.height_m * study.prescribed_extension_m
            / study.width_m,
        base: study,
    };
    let force = run_phase_field_damage_force_controlled(&force_study)
        .expect("bounded force-controlled phase-field solve");
    assert_eq!(
        force.loading_control,
        sonium_sim_structural_mechanics::PhaseFieldLoadingControl::ForceControlled
    );
    assert_eq!(force.applied_force_history_n.len(), study.load_steps);
    assert_eq!(force.tip_displacement_history_m.len(), study.load_steps);
    assert!(force.history_monotone);
    assert!(force.damage_monotone);
    assert!(force.final_applied_force_n.is_finite());
}

#[test]
fn bracket_shape_public_reference_preserves_reanalysis_and_gradient_gates() {
    let mut base = BracketStudy::introductory_bracket();
    base.geometry.nx = 4;
    base.geometry.ny = 1;
    let mut study = BracketShapeOptimizationStudy::introductory(base);
    study.iterations = 1;
    study.step_size = 0.10;
    study.derivative_step_fraction = 2.0e-4;
    let result = run_bracket_shape_optimization(&study).expect("bounded shape optimization");

    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::BracketShapeOptimization
    );
    assert_eq!(result.iterations, study.iterations);
    assert!(result.body_fitted_reanalysis);
    assert!(result.final_static_result.converged);
    assert!(result.adjoint_residual < 1.0e-7);
    assert!(result.gradient_taylor_relative_error < 1.0e-6);
    assert!(result.final_objective_j <= result.initial_objective_j);
    assert!(result.final_design.length_m.is_finite());
    assert!(result.final_design.thickness_m.is_finite());
}

#[test]
fn shell_public_reference_preserves_load_and_energy_gates() {
    let mut study = ShellStaticStudy::introductory();
    study.geometry.nx = 4;
    study.geometry.ny = 2;
    let result = run_shell_static(&study).expect("bounded shell solve");

    assert_eq!(result.model_assumption, StructuralModelAssumption::Shell);
    assert!(result.converged);
    assert!(result.max_transverse_displacement_m > 0.0);
    assert!(result.force_balance_residual_n < 1.0e-7);
    assert!(result.moment_balance_residual_nm.is_finite());
    assert!(result.energy_balance_relative_error < 1.0e-9);
    assert!(result.load_integration_relative_error < 1.0e-12);
}

#[test]
fn solid_public_reference_preserves_force_energy_and_stress_gates() {
    let mut study = Solid3DStudy::introductory();
    study.geometry.nx = 4;
    study.geometry.ny = 1;
    study.geometry.nz = 1;
    study.modal_modes = 1;
    let result = run_solid3d_static(&study).expect("bounded solid solve");

    assert_eq!(result.model_assumption, StructuralModelAssumption::Solid3D);
    assert!(result.converged);
    assert!(result.max_displacement_m > 0.0);
    assert!(result.max_von_mises_pa > 0.0);
    assert!(result.strain_energy_j > 0.0);
    assert!(result.force_balance_residual_n < 1.0e-6);
    assert!(result.free_equilibrium_residual_n < 1.0e-6);
    assert!(result.energy_balance_relative_error < 1.0e-10);
    assert!(
        result
            .element_von_mises_pa
            .iter()
            .all(|value| value.is_finite())
    );
}
