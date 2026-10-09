use std::sync::OnceLock;

use sonium_sim_structural_mechanics::{
    SOLID3D_TOPOLOGY_PROMOTION_SCHEMA_VERSION, Solid3DTopologyResult, Solid3DTopologyStudy,
    StructuralMechanicsError, StructuralModelAssumption, promote_solid3d_topology,
    run_solid3d_topology,
};

fn reference_result() -> Solid3DTopologyResult {
    static RESULT: OnceLock<Solid3DTopologyResult> = OnceLock::new();
    RESULT
        .get_or_init(|| {
            let mut study = Solid3DTopologyStudy::introductory();
            study.base.geometry.nx = 2;
            study.base.geometry.ny = 1;
            study.base.geometry.nz = 1;
            study.base.modal_modes = 1;
            study.volume_fraction = 0.90;
            study.iterations = 2;
            study.continuation_steps = 1;
            study.move_limit = 1.0e-12;
            run_solid3d_topology(&study).expect("small 3-D topology reference should solve")
        })
        .clone()
}

#[test]
fn topology_promotion_requires_checked_body_fitted_design() {
    let result = reference_result();
    let promotion = promote_solid3d_topology(&result, 1.0).expect("reference should promote");

    assert_eq!(
        promotion.schema_version,
        SOLID3D_TOPOLOGY_PROMOTION_SCHEMA_VERSION
    );
    assert_eq!(
        promotion.model_assumption,
        StructuralModelAssumption::SimpTopology3D
    );
    assert!(promotion.accepted);
    assert!(promotion.objective_non_regression_passed);
    assert!(!promotion.final_density.is_empty());
    assert!(
        promotion
            .body_fitted_cases
            .iter()
            .all(|case| case.converged)
    );

    let encoded = serde_json::to_string(&promotion).expect("promotion should serialize");
    let decoded: sonium_sim_structural_mechanics::Solid3DTopologyPromotion =
        serde_json::from_str(&encoded).expect("promotion should deserialize");
    assert_eq!(decoded.schema_version, promotion.schema_version);
    assert_eq!(decoded.model_assumption, promotion.model_assumption);
    assert_eq!(decoded.objective, promotion.objective);
    assert_eq!(decoded.final_density, promotion.final_density);
    assert_eq!(
        decoded.body_fitted_cases.len(),
        promotion.body_fitted_cases.len()
    );
    assert!(decoded.body_fitted_cases.iter().all(|case| case.converged));
    assert_eq!(decoded.accepted, promotion.accepted);
    assert_eq!(decoded.execution, promotion.execution);
}

#[test]
fn topology_promotion_rejects_missing_reanalysis_and_objective_regression() {
    let mut missing_reanalysis = reference_result();
    missing_reanalysis.body_fitted_feasible = false;
    assert!(matches!(
        promote_solid3d_topology(&missing_reanalysis, 1.0),
        Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology.promotion.body_fitted_reanalysis",
            ..
        })
    ));

    let mut regressed = reference_result();
    regressed.final_objective_value =
        regressed.initial_objective_value + regressed.initial_objective_value.abs().max(1.0);
    assert!(matches!(
        promote_solid3d_topology(&regressed, 1.0e-6),
        Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology.promotion.objective_non_regression",
            ..
        })
    ));
}

#[test]
fn topology_promotion_rejects_invalid_controls_and_density() {
    let result = reference_result();
    assert!(matches!(
        promote_solid3d_topology(&result, -1.0),
        Err(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d_topology.promotion.max_objective_relative_increase",
            ..
        })
    ));

    let mut invalid_density = result;
    invalid_density.final_density[0] = f64::NAN;
    assert!(matches!(
        promote_solid3d_topology(&invalid_density, 1.0),
        Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology.promotion.final_density",
            ..
        })
    ));
}
