//! JSON Schema and strict-entry helpers for the versioned XEM room document.
//!
//! The project deliberately does not depend on a schema-derive crate. Keeping
//! this small schema in code makes the generated document available in the
//! same WASM build as the serde model and gives non-Rust consumers a stable
//! artifact to validate against.

use serde_json::{Value, json};

/// JSON Schema identifier for the canonical room configuration document.
pub const ROOM_CONFIG_SCHEMA_ID: &str = "https://sonium.dev/xem/room-config-v1.schema.json";

/// JSON Schema identifier for the backend-neutral multiphysics problem IR.
pub const MULTIPHYSICS_PROBLEM_SCHEMA_ID: &str =
    "https://sonium.dev/xem/multiphysics-problem-v1.schema.json";

/// Compatibility entry for a serialized room configuration version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchemaCompatibility {
    /// Serialized schema version.
    pub version: u32,
    /// Whether the current reader consumes this version directly.
    pub readable_without_migration: bool,
    /// Stable description of the migration used when direct reading is not
    /// available.
    pub migration: Option<&'static str>,
}

/// Machine-readable compatibility matrix for the published room schema.
///
/// Keep this table alongside migration code so a schema change must update
/// both the executable reader and its compatibility evidence.
pub const SCHEMA_COMPATIBILITY_MATRIX: &[SchemaCompatibility] = &[
    SchemaCompatibility {
        version: 0,
        readable_without_migration: false,
        migration: Some("v0-to-v1-explicit-schema-marker"),
    },
    SchemaCompatibility {
        version: 1,
        readable_without_migration: true,
        migration: None,
    },
];

/// Return the schema compatibility matrix used by release and CI checks.
pub const fn schema_compatibility_matrix() -> &'static [SchemaCompatibility] {
    SCHEMA_COMPATIBILITY_MATRIX
}

/// Machine-readable compatibility matrix for the published multiphysics
/// problem envelope.
///
/// The multiphysics envelope was introduced at v1 and has no implicit legacy
/// reader. Keeping the v1-only row explicit makes unsupported versions fail
/// closed and gives independent consumers a stable matrix to inspect before
/// attempting nested field decoding.
pub const MULTIPHYSICS_SCHEMA_COMPATIBILITY_MATRIX: &[SchemaCompatibility] =
    &[SchemaCompatibility {
        version: 1,
        readable_without_migration: true,
        migration: None,
    }];

/// Return the multiphysics-envelope compatibility matrix used by release and
/// CI checks.
pub const fn multiphysics_schema_compatibility_matrix() -> &'static [SchemaCompatibility] {
    MULTIPHYSICS_SCHEMA_COMPATIBILITY_MATRIX
}

/// Return the canonical JSON Schema for [`crate::RoomConfig`].
pub fn room_config_json_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": ROOM_CONFIG_SCHEMA_ID,
        "title": "Sonium XEM room configuration",
        "description": "Versioned, unit-documented input shared by Sonium room solvers.",
        "type": "object",
        "additionalProperties": false,
        "required": ["room", "sources", "listening_positions", "frequencies"],
        "properties": {
            "schema_version": {"type": "integer", "const": 1, "minimum": 1},
            "room": room_schema(),
            "sources": {
                "type": "array",
                "minItems": 1,
                "items": source_schema(),
            },
        "listening_positions": {
            "type": "array",
            "minItems": 1,
            "items": point_schema(),
        },
            "frequencies": {
                "type": "object",
                "additionalProperties": false,
                "required": ["min_freq", "max_freq", "num_points"],
                "properties": {
                    "min_freq": {"type": "number", "exclusiveMinimum": 0},
                    "max_freq": {"type": "number", "exclusiveMinimum": 0},
                    "num_points": {"type": "integer", "minimum": 1},
                    "spacing": {"type": "string", "enum": ["linear", "logarithmic"]},
                },
            },
            "boundaries": boundary_schema(),
            "medium": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "density": {"type": "number", "exclusiveMinimum": 0},
                    "speed_of_sound": {"type": "number", "exclusiveMinimum": 0},
                },
            },
            "solver": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "method": {"type": "string", "minLength": 1},
                    "mesh_resolution": {"type": "integer", "minimum": 1},
                    "gmres": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "max_iter": {"type": "integer", "minimum": 1},
                            "restart": {"type": "integer", "minimum": 1},
                            "tolerance": {"type": "number", "exclusiveMinimum": 0},
                        },
                    },
                    "ilu": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "method": {"type": "string"},
                            "scanning_degree": {"type": "string"},
                            "use_hierarchical": {"type": "boolean"}
                        }
                    },
                    "fmm": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "fmm_type": {"type": "string", "enum": ["slfmm", "mlfmm"]},
                            "expansion_order": {"type": "integer", "minimum": 1},
                            "max_particles_per_leaf": {"type": "integer", "minimum": 1}
                        }
                    },
                    "adaptive_integration": {"type": "boolean"},
                    "adaptive_meshing": {"type": ["boolean", "null"]},
                },
            },
            "visualization": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "generate_slices": {"type": "boolean"},
                    "slice_resolution": {"type": "integer", "minimum": 1},
                    "slice_frequency_indices": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                    },
                },
            },
            "metadata": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "description": {"type": "string"},
                    "author": {"type": "string"},
                    "date": {"type": "string"},
                },
            },
            "conventions": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "harmonic_time": {"enum": ["ExpMinusIwt", "ExpPlusIwt"]},
                    "coordinate_frame": {
                        "enum": ["WorldRightHanded", "SourceLocalRightHanded"],
                    },
                    "outward_normals": {"type": "boolean"},
                    "rms_amplitudes": {"type": "boolean"},
                },
            },
        },
    })
}

/// Return the schema as deterministic pretty-printed JSON.
pub fn room_config_json_schema_string() -> String {
    serde_json::to_string_pretty(&room_config_json_schema())
        .expect("the in-memory XEM JSON Schema is always serializable")
}

/// Return the published envelope schema for [`crate::MultiphysicsProblem`].
///
/// The nested IR records intentionally remain open to the Rust/serde model's
/// evolution, while the top-level envelope is strict and versioned. This is
/// enough for independent readers to reject the wrong document family or
/// schema version before attempting to interpret a field payload.
pub fn multiphysics_problem_json_schema() -> Value {
    let object_array = json!({
        "type": "array",
        "items": {"type": "object"},
    });
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": MULTIPHYSICS_PROBLEM_SCHEMA_ID,
        "title": "Sonium XEM multiphysics problem",
        "type": "object",
        "additionalProperties": false,
        "required": [
            "schema_version", "id", "conventions", "domains", "regions",
            "fields", "boundary_conditions", "interface_conditions",
            "material_laws", "couplings", "design_variables", "objectives",
            "constraints", "provenance"
        ],
        "properties": {
            "schema_version": {"type": "integer", "const": 1, "minimum": 1},
            "id": {"type": "string", "minLength": 1},
            "conventions": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "harmonic_time": {"enum": ["ExpMinusIwt", "ExpPlusIwt"]},
                    "coordinate_frame": {"enum": ["WorldRightHanded", "SourceLocalRightHanded"]},
                    "outward_normals": {"type": "boolean"},
                    "rms_amplitudes": {"type": "boolean"}
                }
            },
            "domains": object_array,
            "regions": object_array,
            "fields": object_array,
            "boundary_conditions": object_array,
            "interface_conditions": object_array,
            "material_laws": object_array,
            "couplings": object_array,
            "design_variables": object_array,
            "objectives": object_array,
            "constraints": object_array,
            "provenance": {"type": "object"}
        }
    })
}

/// Return the multiphysics problem schema as deterministic pretty-printed JSON.
pub fn multiphysics_problem_json_schema_string() -> String {
    serde_json::to_string_pretty(&multiphysics_problem_json_schema())
        .expect("the in-memory multiphysics JSON Schema is always serializable")
}

/// Reject unknown fields at the root of a room configuration document.
pub(crate) fn reject_unknown_room_config_fields(
    value: &Value,
) -> Result<(), crate::ConfigValidationError> {
    let Some(object) = value.as_object() else {
        return Err(crate::ConfigValidationError::InvalidField {
            field: "$".to_string(),
            reason: "configuration must be a JSON object".to_string(),
        });
    };

    const KNOWN: &[&str] = &[
        "schema_version",
        "room",
        "sources",
        "listening_positions",
        "frequencies",
        "boundaries",
        "medium",
        "solver",
        "visualization",
        "metadata",
        "conventions",
    ];
    if let Some(field) = object.keys().find(|field| !KNOWN.contains(&field.as_str())) {
        return Err(crate::ConfigValidationError::UnknownField {
            path: format!("$.{field}"),
        });
    }
    Ok(())
}

fn point_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["x", "y", "z"],
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"},
        },
    })
}

fn room_schema() -> Value {
    json!({
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": false,
                "required": ["type", "width", "depth", "height"],
                "properties": {
                    "type": {"const": "rectangular"},
                    "width": {"type": "number", "exclusiveMinimum": 0},
                    "depth": {"type": "number", "exclusiveMinimum": 0},
                    "height": {"type": "number", "exclusiveMinimum": 0},
                },
            },
            {
                "type": "object",
                "additionalProperties": false,
                "required": ["type", "width1", "depth1", "width2", "depth2", "height"],
                "properties": {
                    "type": {"const": "lshaped"},
                    "width1": {"type": "number", "exclusiveMinimum": 0},
                    "depth1": {"type": "number", "exclusiveMinimum": 0},
                    "width2": {"type": "number", "exclusiveMinimum": 0},
                    "depth2": {"type": "number", "exclusiveMinimum": 0},
                    "height": {"type": "number", "exclusiveMinimum": 0},
                },
            },
        ],
    })
}

fn source_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["name", "position"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "position": point_schema(),
            "amplitude": {"type": "number"},
            "directivity": {
                "oneOf": [
                    {"type": "object", "additionalProperties": false, "required": ["type"], "properties": {"type": {"const": "omnidirectional"}}},
                    {"type": "object", "additionalProperties": false, "required": ["type", "horizontal_angle", "vertical_angle"], "properties": {"type": {"const": "classical"}, "horizontal_angle": {"type": "number"}, "vertical_angle": {"type": "number"}}},
                    {"type": "object", "additionalProperties": false, "required": ["type", "horizontal_angles", "vertical_angles", "magnitude"], "properties": {"type": {"const": "custom"}, "horizontal_angles": {"type": "array", "items": {"type": "number"}}, "vertical_angles": {"type": "array", "items": {"type": "number"}}, "magnitude": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}}},
                    {"type": "object", "additionalProperties": false, "required": ["type", "horizontal_angles", "vertical_angles", "frequencies", "patterns"], "properties": {
                        "type": {"const": "complex_grid"},
                        "horizontal_angles": {"type": "array", "minItems": 2, "items": {"type": "number"}},
                        "vertical_angles": {"type": "array", "minItems": 2, "items": {"type": "number"}},
                        "frequencies": {"type": "array", "minItems": 1, "items": {"type": "number", "exclusiveMinimum": 0}},
                        "patterns": {"type": "array", "minItems": 1, "items": {"type": "array", "items": {"type": "array", "items": {
                            "type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}
                        }}}}
                    }},
                ],
            },
            "crossover": {
                "oneOf": [
                    {"type": "object", "additionalProperties": false, "required": ["type"], "properties": {"type": {"const": "fullrange"}}},
                    {"type": "object", "additionalProperties": false, "required": ["type", "cutoff_freq", "order"], "properties": {"type": {"const": "lowpass"}, "cutoff_freq": {"type": "number", "exclusiveMinimum": 0}, "order": {"type": "integer", "minimum": 1}}},
                    {"type": "object", "additionalProperties": false, "required": ["type", "cutoff_freq", "order"], "properties": {"type": {"const": "highpass"}, "cutoff_freq": {"type": "number", "exclusiveMinimum": 0}, "order": {"type": "integer", "minimum": 1}}},
                    {"type": "object", "additionalProperties": false, "required": ["type", "low_cutoff", "high_cutoff", "order"], "properties": {"type": {"const": "bandpass"}, "low_cutoff": {"type": "number", "exclusiveMinimum": 0}, "high_cutoff": {"type": "number", "exclusiveMinimum": 0}, "order": {"type": "integer", "minimum": 1}}},
                ],
            },
        },
    })
}

fn boundary_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "floor": surface_schema(),
            "ceiling": surface_schema(),
            "walls": surface_schema(),
            "front_wall": {"anyOf": [surface_schema(), {"type": "null"}]},
            "back_wall": {"anyOf": [surface_schema(), {"type": "null"}]},
            "left_wall": {"anyOf": [surface_schema(), {"type": "null"}]},
            "right_wall": {"anyOf": [surface_schema(), {"type": "null"}]},
        },
    })
}

fn surface_schema() -> Value {
    json!({
        "oneOf": [
            {"type": "object", "additionalProperties": false, "required": ["type"], "properties": {"type": {"const": "rigid"}}},
            {"type": "object", "additionalProperties": false, "required": ["type", "coefficient"], "properties": {"type": {"const": "absorption"}, "coefficient": {"type": "number", "minimum": 0, "maximum": 1}}},
            {"type": "object", "additionalProperties": false, "required": ["type", "real", "imag"], "properties": {"type": {"const": "impedance"}, "real": {"type": "number"}, "imag": {"type": "number"}}},
            {"type": "object", "additionalProperties": false, "required": ["type", "real", "imag"], "properties": {"type": {"const": "admittance"}, "real": {"type": "number"}, "imag": {"type": "number"}}},
            {"type": "object", "additionalProperties": false, "required": ["type", "model", "thickness", "flow_resistivity"], "properties": {
                "type": {"const": "porous_layer"},
                "model": {"type": "string", "enum": ["delany_bazley", "miki"]},
                "thickness": {"type": "number", "exclusiveMinimum": 0},
                "flow_resistivity": {"type": "number", "exclusiveMinimum": 0},
                "air_gap": {"anyOf": [{"type": "number", "exclusiveMinimum": 0}, {"type": "null"}]},
                "ambient": {
                    "type": "object", "additionalProperties": false,
                    "properties": {
                        "density": {"type": "number", "exclusiveMinimum": 0},
                        "speed_of_sound": {"type": "number", "exclusiveMinimum": 0}
                    }
                }
            }},
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_describes_version_and_strict_root() {
        let schema = room_config_json_schema();
        assert_eq!(schema["$id"], ROOM_CONFIG_SCHEMA_ID);
        assert_eq!(schema["properties"]["schema_version"]["const"], 1);
        assert_eq!(
            schema_compatibility_matrix()
                .iter()
                .map(|entry| entry.version)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            schema_compatibility_matrix()[0].migration,
            Some("v0-to-v1-explicit-schema-marker")
        );
        assert_eq!(schema["additionalProperties"], false);
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&json!("room"))
        );
    }

    #[test]
    fn unknown_root_field_is_rejected() {
        let error = reject_unknown_room_config_fields(&json!({"room": {}, "typo": true}))
            .expect_err("unknown root fields must not be silently ignored");
        assert!(matches!(
            error,
            crate::ConfigValidationError::UnknownField { .. }
        ));
    }

    #[test]
    fn cross_language_fixture_round_trips_through_strict_reader() {
        let config =
            crate::RoomConfig::from_json_checked(include_str!("../fixtures/room-config-v1.json"))
                .expect("the checked fixture must remain consumable");
        let encoded = serde_json::to_string(&config).expect("config serializes");
        let decoded = crate::RoomConfig::from_json_checked(&encoded).expect("round-trip parses");
        assert_eq!(decoded.schema_version, crate::SchemaVersion::current());
        assert_eq!(decoded.sources.len(), 1);
        assert_eq!(decoded.listening_positions.len(), 1);
    }

    #[test]
    fn legacy_fixture_requires_and_passes_explicit_migration() {
        let error =
            crate::RoomConfig::from_json_checked(include_str!("../fixtures/room-config-v0.json"))
                .expect_err("the checked reader must not accept an unsupported version");
        assert!(error.to_string().contains("unsupported XEM schema version"));

        let migrated =
            crate::RoomConfig::from_json_migrated(include_str!("../fixtures/room-config-v0.json"))
                .expect("the registered v0 migration must remain consumable");
        assert_eq!(migrated.schema_version, crate::SchemaVersion::current());
    }

    #[test]
    fn multiphysics_schema_is_strict_and_accepts_a_valid_problem() {
        let schema = multiphysics_problem_json_schema();
        assert_eq!(schema["$id"], MULTIPHYSICS_PROBLEM_SCHEMA_ID);
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["properties"]["schema_version"]["const"], 1);

        let problem = crate::MultiphysicsProblem {
            id: "schema-fixture".into(),
            ..Default::default()
        };
        let value = serde_json::to_value(problem).expect("serialize multiphysics fixture");
        crate::io::validate_schema_document(&schema, &value).expect("valid multiphysics envelope");

        let mut unknown = value;
        unknown["unexpected"] = serde_json::Value::Bool(true);
        assert!(crate::io::validate_schema_document(&schema, &unknown).is_err());
    }

    #[test]
    fn multiphysics_compatibility_matrix_is_explicitly_v1_only() {
        assert_eq!(
            multiphysics_schema_compatibility_matrix(),
            &[SchemaCompatibility {
                version: 1,
                readable_without_migration: true,
                migration: None,
            }]
        );
        assert!(
            multiphysics_schema_compatibility_matrix()
                .iter()
                .all(|entry| entry.version == 1 && entry.readable_without_migration)
        );
    }

    #[test]
    fn checked_in_schema_artifacts_match_runtime_generators() {
        assert_eq!(
            include_str!("../schemas/room-config-v1.schema.json").trim_end(),
            room_config_json_schema_string()
        );
        assert_eq!(
            include_str!("../schemas/multiphysics-problem-v1.schema.json").trim_end(),
            multiphysics_problem_json_schema_string()
        );
    }
}
