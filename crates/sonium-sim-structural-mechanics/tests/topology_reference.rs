use sonium_sim_structural_mechanics::{
    FemTopology2DMultiLoadStudy, FemTopology2DObjective, FemTopology2DRobustStudy,
    FemTopology2DStudy, Solid3DTopologyStudy, StructuralModelAssumption, run_fem_topology_2d,
    run_fem_topology_2d_multi_load, run_fem_topology_2d_robust, run_solid3d_topology,
};

#[test]
fn fem_topology_public_reference_covers_static_reanalysis_and_gradient_gates() {
    let mut study = FemTopology2DStudy::introductory();
    study.geometry.nx = 6;
    study.geometry.ny = 2;
    study.volume_fraction = 0.95;
    study.move_limit = 0.01;
    study.continuation_steps = 1;
    study.iterations_per_stage = 8;
    study.filter_radius_m = 0.02;
    study.minimum_feature_cells = 1;
    study.maximum_overhang_slope = 100.0;

    let result = run_fem_topology_2d(&study).expect("small FEM topology reference should solve");
    assert_eq!(result.objective, FemTopology2DObjective::Compliance);
    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::SimpTopology2D
    );
    assert!(result.connected);
    assert!(result.reanalysis_feasible);
    assert!(result.final_compliance_j.is_finite());
    assert!(result.projected_reanalysis_compliance_j.is_finite());
    assert!(result.body_fitted_reanalysis_compliance_j.is_finite());
    assert!(result.solver_residual <= 1.0e-7);
    assert!(result.free_equilibrium_residual_n <= 1.0e-7);
    assert!(result.force_balance_residual_n <= 1.0e-7);
    assert!(result.energy_balance_relative_error <= 1.0e-8);
    assert!(result.gradient_taylor_relative_error.is_finite());
    assert!(
        result
            .final_physical_density
            .iter()
            .all(|density| (*density >= study.minimum_density) && (*density <= 1.0))
    );
}

#[test]
fn solid_topology_public_reference_covers_multi_load_reuse_and_body_fit() {
    let mut study = Solid3DTopologyStudy::introductory();
    study.base.geometry.nx = 2;
    study.base.geometry.ny = 1;
    study.base.geometry.nz = 1;
    study.base.modal_modes = 1;
    study.volume_fraction = 0.90;
    study.iterations = 2;
    study.allowable_stress_pa = 1.0e14;

    let result = run_solid3d_topology(&study).expect("small solid topology reference should solve");
    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::SimpTopology3D
    );
    assert_eq!(result.cases.len(), 2);
    assert!(result.factorization_reused_across_cases);
    assert!(result.factorization_memory_bytes > 0);
    assert!(result.connected);
    assert!(result.stress_feasible);
    assert!(result.aggregated_stress_pa.is_finite());
    assert!(result.morphology_feasible);
    assert!(result.body_fitted_reanalysis);
    assert!(result.body_fitted_feasible);
    assert!(result.first_frequency_hz.is_finite());
    assert!(result.first_frequency_hz > 0.0);
    assert_eq!(result.initial_density.len(), result.final_density.len());
    assert!(!result.initial_density.is_empty());
    assert!(result.initial_density.iter().all(|density| {
        density.is_finite() && *density >= study.minimum_density && *density <= 1.0
    }));
    assert!(result.final_density.iter().all(|density| {
        density.is_finite() && *density >= study.minimum_density && *density <= 1.0
    }));
    let mut serialized = serde_json::to_value(&result).expect("serialize 3-D topology result");
    let serialized_object = serialized
        .as_object_mut()
        .expect("3-D topology result object");
    serialized_object.remove("initial_density");
    let legacy: sonium_sim_structural_mechanics::Solid3DTopologyResult =
        serde_json::from_value(serialized).expect("legacy 3-D topology result remains readable");
    assert!(legacy.initial_density.is_empty());
    assert!(result.cases.iter().all(|case| {
        case.converged && case.aggregated_stress_pa.is_finite() && case.solver_residual <= 1.0e-8
    }));
}

#[test]
fn solid_topology_rejects_invalid_stress_aggregation_exponent() {
    let mut study = Solid3DTopologyStudy::introductory();
    study.stress_aggregation_p = 0.5;
    assert!(matches!(
        run_solid3d_topology(&study),
        Err(
            sonium_sim_structural_mechanics::StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_topology.controls",
                ..
            }
        )
    ));
}

#[test]
fn solid_topology_mesh_convergence_covers_refinement_and_projected_rank_gates() {
    let mut study = Solid3DTopologyStudy::introductory();
    study.base.geometry.nx = 2;
    study.base.geometry.ny = 1;
    study.base.geometry.nz = 1;
    study.volume_fraction = 0.90;
    study.iterations = 8;
    study.convergence_tolerance = 0.75;
    study.allowable_stress_pa = 1.0e14;

    let report =
        sonium_sim_structural_mechanics::run_solid3d_topology_mesh_convergence(&study, 0.75)
            .expect("small 3-D topology convergence reference should solve");

    assert_eq!(
        report.schema_version,
        sonium_sim_structural_mechanics::SOLID3D_TOPOLOGY_CONVERGENCE_SCHEMA_VERSION
    );
    assert_eq!(report.coarse_resolution, [2, 1, 1]);
    assert_eq!(report.refined_resolution, [4, 2, 2]);
    let encoded = serde_json::to_string(&report).expect("convergence report should serialize");
    let decoded: sonium_sim_structural_mechanics::Solid3DTopologyConvergenceReport =
        serde_json::from_str(&encoded).expect("convergence report should deserialize");
    assert_eq!(decoded.coarse_resolution, report.coarse_resolution);
    assert_eq!(decoded.refined_resolution, report.refined_resolution);
    assert!(report.refined.nodes > report.coarse.nodes);
    assert!(report.refined.elements > report.coarse.elements);
    assert!(report.refined.dofs > report.coarse.dofs);
    assert!(report.objective_relative_change.is_finite());
    assert!(report.stress_relative_change.is_finite());
    assert!(report.volume_fraction_relative_change.is_finite());
    assert!(report.rank_pair_count > 0);
    assert!(report.rank_discordant_pair_count <= report.rank_pair_count);
    assert!(report.rank_discordance_fraction.is_finite());
    assert!(report.rank_preserved, "{report:#?}");
    assert!(report.converged, "{report:#?}");
}

#[test]
fn solid_topology_mesh_convergence_rejects_refinement_overflow_before_solving() {
    let mut study = Solid3DTopologyStudy::introductory();
    study.base.geometry.nx = usize::MAX;

    assert!(matches!(
        sonium_sim_structural_mechanics::run_solid3d_topology_mesh_convergence(&study, 0.75),
        Err(
            sonium_sim_structural_mechanics::StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_topology_convergence.nx",
                ..
            }
        )
    ));
}

#[test]
fn solid_topology_mesh_convergence_rejects_invalid_tolerance_before_solving() {
    let study = Solid3DTopologyStudy::introductory();
    assert!(matches!(
        sonium_sim_structural_mechanics::run_solid3d_topology_mesh_convergence(&study, f64::NAN),
        Err(
            sonium_sim_structural_mechanics::StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_topology_convergence.tolerance",
                ..
            }
        )
    ));
    assert!(matches!(
        sonium_sim_structural_mechanics::run_solid3d_topology_mesh_convergence(&study, 1.0),
        Err(
            sonium_sim_structural_mechanics::StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_topology_convergence.tolerance",
                ..
            }
        )
    ));
}

#[test]
fn fem_topology_multi_load_public_reference_reuses_factorization_and_selects_critical_case() {
    let mut study = FemTopology2DMultiLoadStudy::introductory();
    study.topology.geometry.nx = 6;
    study.topology.geometry.ny = 2;
    study.topology.volume_fraction = 0.95;
    study.topology.move_limit = 0.01;
    study.topology.continuation_steps = 1;
    study.topology.iterations_per_stage = 8;
    study.topology.filter_radius_m = 0.02;
    study.topology.minimum_feature_cells = 1;
    study.topology.maximum_overhang_slope = 100.0;

    let result = run_fem_topology_2d_multi_load(&study)
        .expect("small multi-load FEM topology reference should solve");
    assert_eq!(result.cases.len(), study.load_cases.len());
    assert!(result.factorization_reused);
    assert!(result.factorization_memory_bytes > 0);
    assert!(result.converged);
    assert!(result.topology.gradient_taylor_relative_error.is_finite());
    assert!(result.topology.connected);
    assert!(result.topology.reanalysis_feasible);
    assert!(
        study
            .load_cases
            .iter()
            .any(|case| case.name == result.critical_case)
    );
    assert!(result.max_compliance_j.is_finite());
    assert!(result.max_displacement_m.is_finite());
    assert!(result.max_von_mises_pa.is_finite());
    assert!(result.aggregated_stress_pa.is_finite());
    assert!(result.stress_feasible);
    assert!(result.cases.iter().all(|case| {
        case.converged
            && case.aggregated_stress_pa.is_finite()
            && case.stress_feasible
            && case.solver_residual <= 1.0e-7
            && case.free_equilibrium_residual_n <= 1.0e-7
            && case.force_balance_residual_n <= 1.0e-7
            && case.energy_balance_relative_error <= 1.0e-8
    }));
}

#[test]
fn fem_topology_robust_public_reference_covers_morphology_and_uncertainty_gates() {
    let mut study = FemTopology2DRobustStudy::introductory();
    study.envelope.topology.geometry.nx = 6;
    study.envelope.topology.geometry.ny = 2;
    study.envelope.topology.volume_fraction = 0.95;
    study.envelope.topology.move_limit = 0.01;
    study.envelope.topology.continuation_steps = 1;
    study.envelope.topology.iterations_per_stage = 3;
    study.envelope.topology.filter_radius_m = 0.02;
    study.envelope.topology.minimum_feature_cells = 1;
    study.envelope.topology.maximum_overhang_slope = 100.0;
    study.envelope.topology.edge_force_n = study.envelope.load_cases[0].edge_force_n;
    study.allowable_compliance_j = 1.0e6;

    let result = run_fem_topology_2d_robust(&study)
        .expect("small robust FEM topology reference should solve");
    assert_eq!(result.cases.len(), 12);
    assert!(result.factorization_reused_within_scenario);
    assert!(result.morphology_feasible);
    assert!(result.uncertainty_feasible);
    assert!(result.robust_feasible);
    assert!(result.erosion_connected);
    assert!(result.dilation_connected);
    assert!(result.eroded_volume_fraction <= result.nominal_volume_fraction + 1.0e-12);
    assert!(result.dilated_volume_fraction + 1.0e-12 >= result.nominal_volume_fraction);
    assert!(result.worst_case_compliance_j.is_finite());
    assert!(result.worst_case_aggregated_stress_pa.is_finite());
    assert!(result.nominal.aggregated_stress_pa.is_finite());
    assert!(result.nominal.stress_feasible);
    assert!(result.stress_feasible);
    assert!(result.gradient_taylor_relative_error.is_finite());
    assert!(result.cases.iter().all(|case| {
        case.converged
            && case.feasible
            && case.aggregated_stress_pa.is_finite()
            && case.stress_feasible
            && case.solver_residual <= 1.0e-7
            && case.free_equilibrium_residual_n <= 1.0e-7
            && case.force_balance_residual_n <= 1.0e-7
            && case.energy_balance_relative_error <= 1.0e-8
    }));
}
#[test]
fn fem_topology_robust_reports_aggregated_stress_gate_failure() {
    let mut study = FemTopology2DRobustStudy::introductory();
    study.envelope.topology.geometry.nx = 2;
    study.envelope.topology.geometry.ny = 1;
    study.envelope.topology.volume_fraction = 1.0;
    study.envelope.topology.continuation_steps = 1;
    study.envelope.topology.iterations_per_stage = 1;
    study.envelope.topology.filter_radius_m = 0.02;
    study.envelope.topology.minimum_feature_cells = 1;
    study.envelope.topology.maximum_overhang_slope = 100.0;
    study.envelope.topology.allowable_stress_pa = 1.0;
    study.allowable_compliance_j = 1.0e12;

    let result = run_fem_topology_2d_robust(&study)
        .expect("small stress-gate FEM topology reference should solve");

    assert!(result.worst_case_aggregated_stress_pa > 1.0);
    assert!(!result.stress_feasible);
    assert!(!result.robust_feasible);
    assert!(result.cases.iter().any(|case| !case.stress_feasible));
}

#[test]
fn solid_robust_topology_public_reference_covers_morphology_and_uncertainty() {
    let mut study = sonium_sim_structural_mechanics::Solid3DRobustTopologyStudy::introductory();
    study.base.base.geometry.nx = 2;
    study.base.base.geometry.ny = 1;
    study.base.base.geometry.nz = 1;
    study.base.volume_fraction = 0.90;
    study.base.iterations = 2;
    study.base.allowable_stress_pa = 1.0e14;
    study.allowable_compliance_j = 1.0;

    let result = sonium_sim_structural_mechanics::run_solid3d_robust_topology(&study)
        .expect("small 3-D robust topology reference should solve");

    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::SimpTopology3D
    );
    assert_eq!(
        result.execution.capability_id,
        "sonium-structural.cpu-reference.f64.solid3d-robust-topology.v1"
    );
    assert_eq!(
        result.cases.len(),
        3 * study.uncertainty_cases.len() * study.base.load_cases.len()
    );
    assert!(result.factorization_reused_within_scenario);
    assert!(result.erosion_connected);
    assert!(result.dilation_connected);
    assert!(result.nominal.overhang_slope.is_finite());
    assert!(result.nominal.overhang_feasible);
    assert!(result.overhang_feasible);
    assert!(result.aggregated_stress_pa.is_finite());
    assert!(result.stress_feasible);
    assert!(result.morphology_feasible);
    assert!(result.uncertainty_feasible);
    assert!(result.robust_feasible);
    assert!(result.worst_case_compliance_j.is_finite());
    assert!(result.gradient_taylor_relative_error.is_finite());
    assert!(result.eroded_volume_fraction <= result.nominal_volume_fraction + 1.0e-12);
    assert!(result.dilated_volume_fraction + 1.0e-12 >= result.nominal_volume_fraction);
    assert!(result.cases.iter().all(|case| {
        case.converged
            && case.feasible
            && case.compliance_j.is_finite()
            && case.aggregated_stress_pa.is_finite()
            && case.solver_residual <= 1.0e-8
            && case.free_equilibrium_residual_n <= 1.0e-6
            && case.force_balance_residual_n <= 1.0e-6
            && case.energy_balance_relative_error <= 1.0e-8
    }));
}

#[test]
fn solid_robust_topology_rejects_duplicate_and_nonpositive_uncertainty() {
    let mut study = sonium_sim_structural_mechanics::Solid3DRobustTopologyStudy::introductory();
    study.uncertainty_cases[1].name = study.uncertainty_cases[0].name.clone();
    assert!(matches!(
        sonium_sim_structural_mechanics::run_solid3d_robust_topology(&study),
        Err(
            sonium_sim_structural_mechanics::StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_robust_topology.uncertainty_cases",
                ..
            }
        )
    ));
    let mut study = sonium_sim_structural_mechanics::Solid3DRobustTopologyStudy::introductory();
    study.uncertainty_cases[0].load_scale = 0.0;
    assert!(matches!(
        sonium_sim_structural_mechanics::run_solid3d_robust_topology(&study),
        Err(
            sonium_sim_structural_mechanics::StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_robust_topology.uncertainty_cases",
                ..
            }
        )
    ));
    let mut study = sonium_sim_structural_mechanics::Solid3DRobustTopologyStudy::introductory();
    study.base.maximum_overhang_slope = f64::NAN;
    assert!(matches!(
        sonium_sim_structural_mechanics::run_solid3d_robust_topology(&study),
        Err(
            sonium_sim_structural_mechanics::StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_topology.controls",
                ..
            }
        )
    ));
}
#[test]
fn constrained_topology_public_reference_covers_manufacturing_and_physics_gates() {
    let mut study =
        sonium_sim_structural_mechanics::ConstrainedTopologyOptimizationStudy::introductory();
    study.base.base.iterations = 4;
    let result = sonium_sim_structural_mechanics::run_constrained_topology_optimization(&study)
        .expect("constrained topology reference should solve");

    assert!(result.aggregated_stress_pa.is_finite());
    assert!(result.stress_feasible);
    assert!(result.first_frequency_hz >= study.minimum_first_frequency_hz);
    assert!(result.frequency_feasible);
    assert!(result.buckling_load_factor >= study.minimum_buckling_factor);
    assert!(result.buckling_feasible);
    assert!(result.connected);
    assert!(result.minimum_feature_feasible);
    assert!(result.overhang_feasible);
    assert!(result.reanalysis_feasible);
    assert!(result.projected_reanalysis_relative_error <= study.reanalysis_tolerance);
    assert!(result.feasible);

    let mut infeasible = study;
    infeasible.allowable_stress_pa = 1.0;
    let rejected =
        sonium_sim_structural_mechanics::run_constrained_topology_optimization(&infeasible)
            .expect("infeasible constraints should produce evidence, not an invalid artifact");
    assert!(!rejected.stress_feasible);
    assert!(!rejected.feasible);
}
