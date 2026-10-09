use std::sync::OnceLock;

use sonium_sim_structural_mechanics::{
    SOLID3D_TOPOLOGY_CONVERGENCE_SCHEMA_VERSION, Solid3DTopologyConvergenceReport,
    Solid3DTopologyStudy, StructuralMechanicsError, StructuralModelAssumption,
    promote_solid3d_topology_mesh_convergence, run_solid3d_topology_mesh_convergence,
};

fn reference_report() -> Solid3DTopologyConvergenceReport {
    static REPORT: OnceLock<Solid3DTopologyConvergenceReport> = OnceLock::new();
    REPORT
        .get_or_init(|| {
            let mut study = Solid3DTopologyStudy::introductory();
            study.base.geometry.nx = 2;
            study.base.geometry.ny = 1;
            study.base.geometry.nz = 1;
            study.volume_fraction = 0.90;
            study.iterations = 8;
            study.convergence_tolerance = 0.75;
            study.allowable_stress_pa = 1.0e14;
            run_solid3d_topology_mesh_convergence(&study, 0.75)
                .expect("small 3-D topology convergence reference should solve")
        })
        .clone()
}

#[test]
fn topology_convergence_promotion_accepts_refined_evidence() {
    let report = reference_report();
    let promotion = promote_solid3d_topology_mesh_convergence(&report)
        .expect("converged topology report should promote");

    assert_eq!(
        promotion.schema_version,
        SOLID3D_TOPOLOGY_CONVERGENCE_SCHEMA_VERSION
    );
    assert_eq!(
        report.model_assumption,
        StructuralModelAssumption::SimpTopology3D
    );
    assert_eq!(promotion.coarse_resolution, [2, 1, 1]);
    assert_eq!(promotion.refined_resolution, [4, 2, 2]);
    assert!(promotion.rank_preserved);
    assert!(promotion.accepted);

    let encoded = serde_json::to_string(&promotion).expect("promotion should serialize");
    let decoded: sonium_sim_structural_mechanics::Solid3DTopologyConvergencePromotion =
        serde_json::from_str(&encoded).expect("promotion should deserialize");
    assert_eq!(decoded.schema_version, promotion.schema_version);
    assert_eq!(decoded.coarse_resolution, promotion.coarse_resolution);
    assert_eq!(decoded.refined_resolution, promotion.refined_resolution);
    assert_eq!(decoded.rank_preserved, promotion.rank_preserved);
    assert_eq!(decoded.accepted, promotion.accepted);
    assert_eq!(decoded.execution, promotion.execution);
}

#[test]
fn topology_convergence_promotion_rejects_tampered_gates() {
    let mut rank_tampered = reference_report();
    rank_tampered.rank_preserved = false;
    assert!(matches!(
        promote_solid3d_topology_mesh_convergence(&rank_tampered),
        Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology_convergence.promotion.gates",
            ..
        })
    ));

    let mut metric_tampered = reference_report();
    metric_tampered.objective_relative_change = 1.0;
    assert!(matches!(
        promote_solid3d_topology_mesh_convergence(&metric_tampered),
        Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology_convergence.promotion.gates",
            ..
        })
    ));
}

#[test]
fn topology_convergence_promotion_rejects_invalid_schema_and_refinement() {
    let mut invalid_schema = reference_report();
    invalid_schema.schema_version += 1;
    assert!(matches!(
        promote_solid3d_topology_mesh_convergence(&invalid_schema),
        Err(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d_topology_convergence.schema_version",
            ..
        })
    ));

    let mut invalid_refinement = reference_report();
    invalid_refinement.refined_resolution = invalid_refinement.coarse_resolution;
    assert!(matches!(
        promote_solid3d_topology_mesh_convergence(&invalid_refinement),
        Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology_convergence.promotion.refinement",
            ..
        })
    ));
}
