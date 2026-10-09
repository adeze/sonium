//! Stable JSON-facing helpers shared by the browser API and native parity tests.

use super::{
    CrossoverConfig, DirectivityConfig, FrequencyConfig, MetadataConfig, Point3DConfig,
    RoomGeometryConfig, SimulationConfig, SimulationProvenance, SolverConfig, SourceConfig,
    VisualizationConfig, WallMaterialsConfig, default_source_orientation,
};

pub fn create_default_config() -> String {
    let config = SimulationConfig {
        room: RoomGeometryConfig::Rectangular {
            width: 5.0,
            depth: 4.0,
            height: 2.5,
        },
        sources: vec![
            SourceConfig {
                name: "Left Speaker".to_string(),
                position: Point3DConfig {
                    x: 1.5,
                    y: 0.5,
                    z: 1.2,
                },
                orientation: default_source_orientation(),
                amplitude: 1.0,
                directivity: DirectivityConfig::Omnidirectional,
                crossover: CrossoverConfig::FullRange,
                delay_ms: 0.0,
                invert_phase: false,
            },
            SourceConfig {
                name: "Right Speaker".to_string(),
                position: Point3DConfig {
                    x: 3.5,
                    y: 0.5,
                    z: 1.2,
                },
                orientation: default_source_orientation(),
                amplitude: 1.0,
                directivity: DirectivityConfig::Omnidirectional,
                crossover: CrossoverConfig::FullRange,
                delay_ms: 0.0,
                invert_phase: false,
            },
        ],
        listening_positions: vec![Point3DConfig {
            x: 2.5,
            y: 2.5,
            z: 1.2,
        }],
        frequencies: FrequencyConfig {
            min_freq: 20.0,
            max_freq: 500.0,
            num_points: 100,
            spacing: "logarithmic".to_string(),
        },
        conventions: super::default_wasm_conventions(),
        solver: SolverConfig::default(),
        visualization: VisualizationConfig::default(),
        wall_materials: WallMaterialsConfig::default(),
        metadata: MetadataConfig {
            description: "Default stereo setup".to_string(),
            author: "Room Simulator WASM".to_string(),
            date: String::new(),
            notes: String::new(),
        },
        scattering_objects: Vec::new(),
    };
    serde_json::to_string_pretty(&config).unwrap_or_else(|_| "{}".to_string())
}

pub(super) fn simulation_model_hash(config: &SimulationConfig) -> String {
    let bytes = serde_json::to_vec(config).unwrap_or_default();
    // FNV-1a is stable across browser and native builds for the same config.
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3_u64);
    }
    format!("{hash:016x}")
}

pub(super) fn simulation_provenance(
    method: &str,
    model_hash: &str,
    bem_fallback_count: usize,
    bem_fallback_reason: Option<String>,
    scattering_object_count: usize,
) -> SimulationProvenance {
    let (fidelity, mut components, mut validity) = match method {
        "bem" | "hybrid-bem" => (
            "blended_or_fallback",
            vec![
                "BEM boundary solve where available".to_string(),
                "analytic modal contribution for rectangular rooms".to_string(),
                "image-source fallback on BEM failure".to_string(),
            ],
            vec![
                "BEM validity depends on mesh quality and boundary discretization".to_string(),
                "current room-acoustics BEM uses its native default boundary model; WallMaterialConfig impedance is not applied".to_string(),
                "rectangular rooms may use analytic modal contribution".to_string(),
                "fallback occurrence is reported per result".to_string(),
            ],
        ),
        "modal" => (
            "approximate",
            vec!["analytic rectangular-room modes".to_string()],
            vec![
                "rectangular rooms only".to_string(),
                "standing-wave modal truncation and damping are configured approximations".to_string(),
            ],
        ),
        "hybrid" => (
            "blended",
            vec![
                "analytic rectangular-room modes".to_string(),
                "image-source reflections".to_string(),
            ],
            vec![
                "rectangular rooms use modal and specular image-source assumptions".to_string(),
                "crossover band is blended coherently in complex pressure".to_string(),
            ],
        ),
        _ => (
            "approximate",
            vec!["image-source reflections".to_string()],
            vec![
                "specular wall reflections and configured reflection order".to_string(),
                "diffraction and diffuse scattering are not represented".to_string(),
            ],
        ),
    };
    if scattering_object_count > 0 {
        if matches!(method, "bem" | "hybrid-bem") {
            components.push(format!(
                "{scattering_object_count} interior scattering object(s) included in the BEM mesh"
            ));
        } else {
            validity.push(format!("{scattering_object_count} configured scattering object(s) are not included by {method}; use BEM for obstacle interaction"));
        }
    }
    let fidelity = if bem_fallback_count > 0 {
        "fallback_approximate"
    } else {
        fidelity
    };
    let uncertainty = if bem_fallback_count > 0 {
        "fallback_observed"
    } else {
        "not_calibrated"
    };
    SimulationProvenance {
        model_hash: model_hash.to_string(),
        method: method.to_string(),
        fidelity: fidelity.to_string(),
        components,
        uncertainty: uncertainty.to_string(),
        error_estimate: None,
        native_validation: "not_performed".to_string(),
        validity,
        bem_fallback_count,
        bem_fallback_reason,
    }
}
