use std::sync::OnceLock;

use sonium_sim_structural_mechanics::{
    SOLID3D_ROBUST_TOPOLOGY_PROMOTION_SCHEMA_VERSION, Solid3DRobustTopologyResult,
    Solid3DRobustTopologyStudy, StructuralMechanicsError, StructuralModelAssumption,
    promote_solid3d_robust_topology, run_solid3d_robust_topology,
};

fn reference_result() -> Solid3DRobustTopologyResult {
    static RESULT: OnceLock<Solid3DRobustTopologyResult> = OnceLock::new();
    RESULT
        .get_or_init(|| {
            let mut study = Solid3DRobustTopologyStudy::introductory();
            study.base.base.geometry.nx = 2;
            study.base.base.geometry.ny = 1;
            study.base.base.geometry.nz = 1;
            study.base.volume_fraction = 0.90;
            study.base.iterations = 2;
            study.base.continuation_steps = 1;
            study.base.move_limit = 1.0e-12;
            study.base.convergence_tolerance = 1.0e-8;
            study.base.allowable_stress_pa = 1.0e14;
            study.allowable_compliance_j = 1.0;
            run_solid3d_robust_topology(&study)
                .expect("small robust topology reference should solve")
        })
        .clone()
}

#[test]
fn robust_topology_promotion_accepts_complete_uncertainty_evidence() {
    let result = reference_result();
    let promotion = promote_solid3d_robust_topology(&result, 1.0, 1.0)
        .expect("robust topology reference should promote");

    assert_eq!(
        promotion.schema_version,
        SOLID3D_ROBUST_TOPOLOGY_PROMOTION_SCHEMA_VERSION
    );
    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::SimpTopology3D
    );
    assert_eq!(promotion.scenario_count, result.cases.len());
    assert_eq!(promotion.scenario_count, 12);
    assert!(promotion.accepted);
    assert!(promotion.morphology_feasible);
    assert!(promotion.uncertainty_feasible);
    assert!(promotion.stress_feasible);
    assert!(promotion.overhang_feasible);

    let encoded = serde_json::to_string(&promotion).expect("promotion should serialize");
    let decoded: sonium_sim_structural_mechanics::Solid3DRobustTopologyPromotion =
        serde_json::from_str(&encoded).expect("promotion should deserialize");
    assert_eq!(decoded.schema_version, promotion.schema_version);
    assert_eq!(decoded.scenario_count, promotion.scenario_count);
    assert_eq!(decoded.worst_case, promotion.worst_case);
    assert_eq!(decoded.accepted, promotion.accepted);
    assert_eq!(decoded.execution, promotion.execution);
}

#[test]
fn robust_topology_promotion_rejects_tampered_robust_and_case_gates() {
    let mut robust_tampered = reference_result();
    robust_tampered.robust_feasible = false;
    assert!(matches!(
        promote_solid3d_robust_topology(&robust_tampered, 1.0, 1.0),
        Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_robust_topology.promotion.gates",
            ..
        })
    ));

    let mut case_tampered = reference_result();
    case_tampered.cases[0].feasible = false;
    assert!(matches!(
        promote_solid3d_robust_topology(&case_tampered, 1.0, 1.0),
        Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_robust_topology.promotion.gates",
            ..
        })
    ));
}

#[test]
fn robust_topology_promotion_rejects_invalid_controls_and_density() {
    let result = reference_result();
    assert!(matches!(
        promote_solid3d_robust_topology(&result, 1.0, -1.0),
        Err(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d_robust_topology.promotion.max_gradient_taylor_relative_error",
            ..
        })
    ));

    let mut density_tampered = reference_result();
    density_tampered.eroded_density[0] = f64::NAN;
    assert!(matches!(
        promote_solid3d_robust_topology(&density_tampered, 1.0, 1.0),
        Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_robust_topology.promotion.gates",
            ..
        })
    ));
}
