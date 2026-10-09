//! Portable reduced preview data for browser/WASM visualization clients.
//!
//! This module deliberately contains no FEM or filesystem dependency. A
//! native solver can serialize [`StructuralReducedPreview`] and a WASM/UI
//! client can render it without being given permission to run a production
//! structural solve.

use serde::{Deserialize, Serialize};

use crate::{
    BracketStudy, StaticBracketResult, StructuralMechanicsError, StructuralPolynomialOrder,
};

/// Schema version for [`StructuralReducedPreview`].
pub const STRUCTURAL_PREVIEW_SCHEMA_VERSION: u32 = 1;

/// A bounded, portable mesh/result payload intended for reduced visualization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralReducedPreview {
    /// Stable payload version for browser clients.
    pub schema_version: u32,
    /// Descriptive source model, not a solver capability claim.
    pub source_model: String,
    /// Node coordinates in metres.
    pub nodes: Vec<[f64; 3]>,
    /// Triangle connectivity into [`Self::nodes`].
    pub triangles: Vec<[u32; 3]>,
    /// Translational displacement at each node in metres.
    pub displacement: Vec<[f64; 3]>,
    /// Name of the scalar field carried by [`Self::scalar_values`].
    pub scalar_name: String,
    /// One scalar value per node, suitable for colour mapping.
    pub scalar_values: Vec<f64>,
}

impl StructuralReducedPreview {
    /// Construct and validate a portable preview payload.
    pub fn new(
        source_model: impl Into<String>,
        nodes: Vec<[f64; 3]>,
        triangles: Vec<[u32; 3]>,
        displacement: Vec<[f64; 3]>,
        scalar_name: impl Into<String>,
        scalar_values: Vec<f64>,
    ) -> Result<Self, StructuralMechanicsError> {
        let source_model = source_model.into();
        let scalar_name = scalar_name.into();
        if source_model.trim().is_empty() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "preview.source_model",
                value: "empty".to_string(),
            });
        }
        if scalar_name.trim().is_empty() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "preview.scalar_name",
                value: "empty".to_string(),
            });
        }
        if nodes.is_empty() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "preview.nodes",
                value: "empty".to_string(),
            });
        }
        if displacement.len() != nodes.len() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "preview.displacement",
                value: format!("{} values for {} nodes", displacement.len(), nodes.len()),
            });
        }
        if scalar_values.len() != nodes.len() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "preview.scalar_values",
                value: format!("{} values for {} nodes", scalar_values.len(), nodes.len()),
            });
        }
        if triangles
            .iter()
            .flatten()
            .any(|&index| index as usize >= nodes.len())
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "preview.triangles",
                value: "connectivity index out of range".to_string(),
            });
        }
        if nodes
            .iter()
            .chain(&displacement)
            .flatten()
            .chain(&scalar_values)
            .any(|value| !value.is_finite())
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "preview.values",
                value: "non-finite value".to_string(),
            });
        }

        Ok(Self {
            schema_version: STRUCTURAL_PREVIEW_SCHEMA_VERSION,
            source_model,
            nodes,
            triangles,
            displacement,
            scalar_name,
            scalar_values,
        })
    }

    /// Serialize the payload for a browser/WASM boundary.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize and revalidate a browser/WASM payload.
    pub fn from_json(json: &str) -> Result<Self, StructuralMechanicsError> {
        let preview: Self = serde_json::from_str(json).map_err(|error| {
            StructuralMechanicsError::InvalidConfiguration {
                field: "preview.json",
                value: error.to_string(),
            }
        })?;
        if preview.schema_version != STRUCTURAL_PREVIEW_SCHEMA_VERSION {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "preview.schema_version",
                value: preview.schema_version.to_string(),
            });
        }
        Self::new(
            preview.source_model,
            preview.nodes,
            preview.triangles,
            preview.displacement,
            preview.scalar_name,
            preview.scalar_values,
        )
    }

    /// Number of nodes carried by this reduced payload.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Build a preview for the introductory bracket without exposing FEM internals.
pub fn bracket_static_preview(
    study: &BracketStudy,
    result: &StaticBracketResult,
) -> Result<StructuralReducedPreview, StructuralMechanicsError> {
    let nx = study.geometry.nx;
    let ny = study.geometry.ny;
    if nx == 0 || ny == 0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "bracket.geometry.preview_resolution",
            value: format!("nx={nx}, ny={ny}"),
        });
    }
    if !result.converged {
        return Err(StructuralMechanicsError::NonConvergence {
            iterations: result.solver_iterations,
            residual: result.solver_residual,
        });
    }
    if result.polynomial_order != StructuralPolynomialOrder::Linear {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "bracket.result.polynomial_order",
            value: format!("expected Linear, got {:?}", result.polynomial_order),
        });
    }
    let node_count = (nx + 1) * (ny + 1);
    if result.displacement.len() != node_count {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "bracket.result.displacement",
            value: format!(
                "{} values for {node_count} preview nodes",
                result.displacement.len()
            ),
        });
    }

    let mut nodes = Vec::with_capacity(node_count);
    for row in 0..=ny {
        for column in 0..=nx {
            nodes.push([
                study.geometry.length_m * column as f64 / nx as f64,
                study.geometry.height_m * row as f64 / ny as f64,
                0.0,
            ]);
        }
    }

    let mut triangles = Vec::with_capacity(2 * nx * ny);
    for row in 0..ny {
        for column in 0..nx {
            let lower_left = (row * (nx + 1) + column) as u32;
            let lower_right = lower_left + 1;
            let upper_left = lower_left + (nx + 1) as u32;
            let upper_right = upper_left + 1;
            triangles.push([lower_left, lower_right, upper_right]);
            triangles.push([lower_left, upper_right, upper_left]);
        }
    }

    let displacement = result
        .displacement
        .iter()
        .map(|&[x, y]| [x, y, 0.0])
        .collect::<Vec<_>>();
    let scalar_values = displacement
        .iter()
        .map(|value| value[0].hypot(value[1]))
        .collect::<Vec<_>>();
    StructuralReducedPreview::new(
        "bracket-static-plane-stress-p1",
        nodes,
        triangles,
        displacement,
        "displacement_m",
        scalar_values,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_round_trips_json() {
        let preview = StructuralReducedPreview::new(
            "unit-test",
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0, 1, 2]],
            vec![[0.0, 0.0, 0.0]; 3],
            "displacement_m",
            vec![0.0; 3],
        )
        .expect("valid preview");
        let restored = StructuralReducedPreview::from_json(&preview.to_json().unwrap())
            .expect("valid preview json");
        assert_eq!(preview, restored);
    }

    #[test]
    fn preview_rejects_invalid_connectivity_and_lengths() {
        let error = StructuralReducedPreview::new(
            "unit-test",
            vec![[0.0, 0.0, 0.0]],
            vec![[0, 1, 0]],
            vec![],
            "value",
            vec![0.0],
        )
        .expect_err("invalid preview");
        assert!(error.to_string().contains("preview.displacement"));
    }

    #[test]
    fn bracket_preview_adapts_a_validated_static_result() {
        let mut study = BracketStudy::introductory_bracket();
        study.geometry.nx = 2;
        study.geometry.ny = 1;
        let result = crate::run_static_bracket(&study).expect("static bracket");
        let preview = bracket_static_preview(&study, &result).expect("bracket preview");
        assert_eq!(preview.node_count(), 6);
        assert_eq!(preview.triangles.len(), 4);
        assert_eq!(preview.scalar_name, "displacement_m");
    }
}
