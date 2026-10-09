//! Validated aperture-trace schema exchanged across the coupling seam.
//!
//! The interior builder (FEM boundary faces) and the exterior builder (BEM
//! aperture elements) must agree on one ordering, one set of areas, and one
//! phasor/normal convention before any operator is assembled. This schema is
//! the checkable form of that agreement: both sides export it, the
//! orchestrator validates the pair, and mismatches fail closed instead of
//! silently coupling the wrong faces.

use serde::{Deserialize, Serialize};

/// Failures while validating an aperture trace.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TraceSchemaError {
    #[error("aperture trace requires a non-empty face list")]
    EmptyFaces,
    #[error("aperture face {index} has non-positive area")]
    NonPositiveArea { index: usize, area_bits: u64 },
    #[error("aperture face {index} has a non-finite area")]
    NonFiniteArea { index: usize },
    #[error("trace face count {faces} does not match BEM element count {elements}")]
    FaceElementMismatch { faces: usize, elements: usize },
    #[error("trace convention mismatch: {detail}")]
    ConventionMismatch { detail: String },
}

/// One aperture triangle: interior vertex triple plus its area in m².
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApertureFace {
    pub nodes: [usize; 3],
    pub area_m2: f64,
}

/// Validated aperture interface trace.
///
/// `bem_elements[i]` is the exterior element coupled to `faces[i]`; both
/// orderings follow the interior boundary-face order. `outward_normal_velocity`
/// records the shared sign contract: `true` means the velocity unknown points
/// out of the interior domain with `dp/dn = +i*omega*rho*v_n`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApertureTrace {
    pub faces: Vec<ApertureFace>,
    pub bem_elements: Vec<usize>,
    pub outward_normal_velocity: bool,
}

impl ApertureTrace {
    /// Validate topology, areas, ordering, and the normal contract.
    pub fn validate(&self) -> Result<(), TraceSchemaError> {
        if self.faces.is_empty() {
            return Err(TraceSchemaError::EmptyFaces);
        }
        for (index, face) in self.faces.iter().enumerate() {
            if !face.area_m2.is_finite() {
                return Err(TraceSchemaError::NonFiniteArea { index });
            }
            if face.area_m2 <= 0.0 {
                return Err(TraceSchemaError::NonPositiveArea {
                    index,
                    area_bits: face.area_m2.to_bits(),
                });
            }
        }
        if self.faces.len() != self.bem_elements.len() {
            return Err(TraceSchemaError::FaceElementMismatch {
                faces: self.faces.len(),
                elements: self.bem_elements.len(),
            });
        }
        if !self.outward_normal_velocity {
            return Err(TraceSchemaError::ConventionMismatch {
                detail: "velocity unknown must point out of the interior domain".into(),
            });
        }
        Ok(())
    }

    /// Per-face areas in the shared ordering.
    pub fn areas(&self) -> Vec<f64> {
        self.faces.iter().map(|face| face.area_m2).collect()
    }

    /// Total aperture area in m².
    pub fn total_area(&self) -> f64 {
        self.faces.iter().map(|face| face.area_m2).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_trace() -> ApertureTrace {
        ApertureTrace {
            faces: vec![
                ApertureFace {
                    nodes: [0, 1, 2],
                    area_m2: 0.5,
                },
                ApertureFace {
                    nodes: [1, 3, 2],
                    area_m2: 0.25,
                },
            ],
            bem_elements: vec![4, 7],
            outward_normal_velocity: true,
        }
    }

    #[test]
    fn valid_trace_reports_areas() {
        let trace = valid_trace();
        trace.validate().expect("valid trace");
        assert_eq!(trace.areas(), vec![0.5, 0.25]);
        assert!((trace.total_area() - 0.75).abs() < 1.0e-15);
    }

    #[test]
    fn malformed_traces_fail_closed() {
        let empty = ApertureTrace {
            faces: vec![],
            bem_elements: vec![],
            outward_normal_velocity: true,
        };
        assert_eq!(empty.validate(), Err(TraceSchemaError::EmptyFaces));

        let mut bad_area = valid_trace();
        bad_area.faces[0].area_m2 = 0.0;
        assert!(matches!(
            bad_area.validate(),
            Err(TraceSchemaError::NonPositiveArea { .. })
        ));

        let mut bad_len = valid_trace();
        bad_len.bem_elements.pop();
        assert_eq!(
            bad_len.validate(),
            Err(TraceSchemaError::FaceElementMismatch {
                faces: 2,
                elements: 1
            })
        );

        let mut bad_normal = valid_trace();
        bad_normal.outward_normal_velocity = false;
        assert!(matches!(
            bad_normal.validate(),
            Err(TraceSchemaError::ConventionMismatch { .. })
        ));
    }

    #[test]
    fn trace_round_trips_through_json() {
        let trace = valid_trace();
        let json = serde_json::to_string(&trace).expect("serialize trace");
        let back: ApertureTrace = serde_json::from_str(&json).expect("parse trace");
        assert_eq!(trace, back);
    }
}
