use sonium_sim_speaker_modeling::port_flow::{
    PORT_FLOW_PROMOTION_SCHEMA_VERSION, PortFlowConvergenceLevel, PortFlowPhasor,
    PortFlowPromotionEvidence, PortFlowPromotionGate, PortFlowRuntimePlan,
    PortFlowRuntimePromotionArtifact, replay_port_flow_promotion, validate_port_flow_promotion,
};

fn level(mesh_size_m: f64, dt_s: f64, scale: f64) -> PortFlowConvergenceLevel {
    PortFlowConvergenceLevel {
        mesh_size_m,
        dt_s,
        frequency_hz: 10_000.0,
        port_flow: PortFlowPhasor {
            real: scale,
            imaginary: -0.2 * scale,
        },
        box_pressure: PortFlowPhasor {
            real: 4.0 * scale,
            imaginary: 0.5 * scale,
        },
        diaphragm_flow: PortFlowPhasor {
            real: 1.5 * scale,
            imaginary: 0.1 * scale,
        },
        max_linear_relative_residual: 1.0e-10,
        max_box_conservation_relative_residual: 1.0e-10,
        cap_continuity_relative_error: 1.0e-8,
        energy_algebraic_relative_closure: 1.0e-10,
        energy_physical_relative_closure: 1.0e-3,
        energy_interface_relative_residual: 1.0e-10,
    }
}

fn evidence() -> PortFlowPromotionEvidence {
    PortFlowPromotionEvidence {
        schema_version: PORT_FLOW_PROMOTION_SCHEMA_VERSION,
        provenance: "integration-port-flow-reference.v1".into(),
        frequency_hz: 10_000.0,
        spatial_levels: vec![
            level(0.004, 1.0e-5, 1.008),
            level(0.003, 1.0e-5, 1.004),
            level(0.002, 1.0e-5, 1.0),
        ],
        temporal_levels: vec![level(0.002, 2.0e-5, 1.004), level(0.002, 1.0e-5, 1.0)],
    }
}

#[test]
fn public_port_flow_promotion_api_accepts_a_complete_ladder() {
    let report = validate_port_flow_promotion(&evidence(), PortFlowPromotionGate::default())
        .expect("public promotion API should accept complete evidence");
    assert_eq!(report.promoted_mesh_size_m, 0.002);
    assert_eq!(report.promoted_dt_s, 1.0e-5);
}

#[test]
fn public_port_flow_promotion_artifact_rejects_unknown_schema() {
    let mut value = serde_json::to_value(evidence()).unwrap();
    value["schema_version"] = serde_json::json!(PORT_FLOW_PROMOTION_SCHEMA_VERSION + 1);
    assert!(serde_json::from_value::<PortFlowPromotionEvidence>(value).is_err());
}

#[test]
fn public_runtime_promotion_artifact_replays_against_its_exact_schedule() {
    let artifact = PortFlowRuntimePromotionArtifact {
        plan: PortFlowRuntimePlan {
            spatial_mesh_sizes_m: vec![0.004, 0.003, 0.002],
            temporal_dt_s: vec![2.0e-5, 1.0e-5],
            frequency_hz: 10_000.0,
            tail_cycles: 2,
            discarded_cycles: 1,
            box_volume_m3: 2.0e-5,
            density_kg_per_m3: 1.2,
            sound_speed_m_per_s: 343.0,
            provenance: "integration-port-flow-reference.v1".into(),
        },
        evidence: evidence(),
    };
    let encoded = serde_json::to_string(&artifact).expect("runtime artifact should serialize");
    let decoded: PortFlowRuntimePromotionArtifact =
        serde_json::from_str(&encoded).expect("runtime artifact should deserialize");
    let replayed = replay_port_flow_promotion(&decoded, PortFlowPromotionGate::default())
        .expect("complete runtime evidence should replay without a solver");
    assert_eq!(replayed.report.promoted_mesh_size_m, 0.002);
    assert_eq!(replayed.report.promoted_dt_s, 1.0e-5);

    let mut mismatched = decoded;
    mismatched.plan.temporal_dt_s[0] = 1.0 / (10_000.0 * 4.0);
    assert!(replay_port_flow_promotion(&mismatched, PortFlowPromotionGate::default()).is_err());
}
