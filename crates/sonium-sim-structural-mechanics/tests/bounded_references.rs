use sonium_sim_structural_mechanics::{
    ElastoplasticBarStudy, FemFatigue2DLoadBlock, FemFatigue2DStudy, StructuralMechanicsError,
    StructuralModelAssumption, StructuralRomStudy, TransientSdoFStudy, run_elastoplastic_bar,
    run_fem_fatigue_2d, run_structural_rom, run_transient_sdof,
};

#[test]
fn public_elastoplastic_bar_reference_covers_yield_and_energy() {
    let study = ElastoplasticBarStudy::introductory();
    let result = run_elastoplastic_bar(&study).expect("bounded elastoplastic bar should solve");
    let final_stress_pa = study.max_force_n / study.area_m2;
    let expected_extension_m = study.length_m
        * (final_stress_pa / study.youngs_modulus_pa
            + (final_stress_pa - study.yield_stress_pa) / study.isotropic_hardening_pa);

    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::ElastoplasticBar
    );
    assert_eq!(result.steps.len(), study.load_steps);
    assert_eq!(
        result.steps.last().unwrap().yielded_elements,
        study.elements
    );
    assert!((result.final_displacement_m - expected_extension_m).abs() < 1.0e-11);
    assert!(result.total_plastic_dissipation_j.is_finite());
    assert!(result.total_plastic_dissipation_j > 0.0);
    assert!(result.max_equilibrium_residual_n < 1.0e-9);
    assert!(result.steps.iter().all(|step| {
        step.converged
            && step.residual_n.is_finite()
            && step.internal_energy_j.is_finite()
            && step.plastic_dissipation_j.is_finite()
    }));
}

#[test]
fn public_fem_fatigue_reference_covers_miner_and_equilibrium_gates() {
    let mut study = FemFatigue2DStudy::introductory();
    study.geometry.nx = 4;
    study.geometry.ny = 1;
    study.load_blocks = vec![FemFatigue2DLoadBlock {
        minimum_force_n: [0.0, -20.0],
        maximum_force_n: [0.0, -80.0],
        applied_cycles: 10_000.0,
    }];

    let result = run_fem_fatigue_2d(&study).expect("bounded FEM fatigue reference should solve");

    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::FatiguePlaneStress2D
    );
    assert_eq!(result.blocks.len(), 1);
    assert!(result.converged);
    assert!(!result.failed);
    assert!(result.max_stress_amplitude_pa > 0.0);
    assert!(result.total_damage > 0.0);
    assert!(result.total_damage < study.damage_limit);
    assert!(result.blocks.iter().all(|block| {
        block.allowable_cycles.is_finite()
            && block.damage_increment.is_finite()
            && block.cumulative_damage.is_finite()
            && block.solver_residual <= 1.0e-7
            && block.equilibrium_residual_n <= 1.0e-7
    }));
}

#[test]
fn public_transient_reference_preserves_undamped_energy() {
    let study = TransientSdoFStudy {
        mass_kg: 2.0,
        stiffness_n_per_m: 50.0,
        damping_ns_per_m: 0.0,
        time_step_s: 0.01,
        force_history_n: vec![0.0; 1_001],
        initial_displacement_m: 0.1,
        initial_velocity_m_per_s: 0.0,
    };
    let result = run_transient_sdof(&study).expect("undamped SDOF reference should solve");
    let initial_energy_j = 0.5 * study.stiffness_n_per_m * study.initial_displacement_m.powi(2);
    let final_energy_j = result.steps.last().unwrap().mechanical_energy_j;

    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::TransientSingleDof
    );
    assert!((final_energy_j - initial_energy_j).abs() < 1.0e-10);
    assert!(result.max_equilibrium_residual_n < 1.0e-9);
    assert!(result.max_energy_balance_residual_j < 1.0e-10);
    assert!(result.total_damping_dissipation_j.abs() < 1.0e-12);
}

#[test]
fn public_rom_reference_recycles_and_enriches_with_residual_gate() {
    let mut study = StructuralRomStudy::introductory();
    study.base.geometry.elements = 4;
    study.reduced_modes = 1;
    study.base.modal_modes = 1;

    let result = run_structural_rom(&study).expect("bounded structural ROM should converge");

    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::ReducedOrderBeam
    );
    assert!(result.recycled_basis);
    assert!(result.enriched_with_full_order_snapshot);
    assert!(result.final_basis_dimension >= result.reference_basis_dimension);
    assert!(result.reduced_relative_residual <= study.residual_tolerance);
    assert!(result.full_order_relative_residual < 1.0e-8);
    assert!(result.relative_tip_error < 1.0e-7);
    assert!(result.converged);
}

#[test]
fn public_bounded_reference_inputs_fail_closed() {
    let mut bar = ElastoplasticBarStudy::introductory();
    bar.isotropic_hardening_pa = 0.0;
    assert!(matches!(
        run_elastoplastic_bar(&bar),
        Err(StructuralMechanicsError::InvalidConfiguration { .. })
    ));

    let mut fatigue = FemFatigue2DStudy::introductory();
    fatigue.load_blocks[0].applied_cycles = -1.0;
    assert!(matches!(
        run_fem_fatigue_2d(&fatigue),
        Err(StructuralMechanicsError::InvalidConfiguration { .. })
    ));

    let mut transient = TransientSdoFStudy::introductory();
    transient.force_history_n.clear();
    assert!(matches!(
        run_transient_sdof(&transient),
        Err(StructuralMechanicsError::InvalidConfiguration { .. })
    ));

    let mut rom = StructuralRomStudy::introductory();
    rom.query_youngs_modulus_scale = 0.0;
    assert!(matches!(
        run_structural_rom(&rom),
        Err(StructuralMechanicsError::InvalidConfiguration { .. })
    ));
}
