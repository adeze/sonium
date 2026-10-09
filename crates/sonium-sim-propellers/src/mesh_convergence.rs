//! Deterministic panel-refinement evidence for the propeller geometry path.
//!
//! This module measures geometric mesh observables only. It does not claim
//! convergence of a resolved wake, aerodynamic load, acoustic field, or
//! observer directivity model.

use crate::{Blade3dConfig, BladeMeshError, PropellerDesign};
use std::fmt;

/// One level in a deterministic geometric panel-refinement ladder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropellerMeshConvergenceLevel {
    /// Integer multiplier applied to every base panel resolution.
    pub refinement: usize,
    /// Refined radial panel count.
    pub radial_segments: usize,
    /// Refined chordwise panel count.
    pub chord_segments: usize,
    /// Refined toroidal-loop panel count.
    pub loop_segments: usize,
    /// Number of generated panels.
    pub panel_count: usize,
    /// Sum of panel areas [m²].
    pub total_area_m2: f64,
    /// Fraction of area on panels with radius fraction above 0.88.
    pub tip_area_fraction: f64,
}

/// Geometric observables and pass/fail status for a refinement ladder.
#[derive(Debug, Clone, PartialEq)]
pub struct PropellerMeshConvergence {
    /// Refinement levels in the caller-provided order.
    pub levels: Vec<PropellerMeshConvergenceLevel>,
    /// Relative total-area changes between adjacent levels.
    pub relative_area_changes: Vec<f64>,
    /// Largest adjacent relative total-area change.
    pub max_relative_area_change: f64,
    /// Relative total-area change from the last two levels.
    pub final_relative_area_change: f64,
    /// Caller-declared geometric admission threshold.
    pub tolerance: f64,
    /// Whether the final adjacent levels satisfy the declared threshold.
    pub converged: bool,
}

impl PropellerMeshConvergence {
    /// Whether all recorded observables are finite and structurally valid.
    pub fn is_finite(&self) -> bool {
        self.tolerance.is_finite()
            && self.tolerance > 0.0
            && self.max_relative_area_change.is_finite()
            && self.final_relative_area_change.is_finite()
            && self.relative_area_changes.len() + 1 == self.levels.len()
            && self
                .relative_area_changes
                .iter()
                .all(|change| change.is_finite() && *change >= 0.0)
            && self.levels.iter().all(|level| {
                level.refinement > 0
                    && level.radial_segments > 0
                    && level.chord_segments > 0
                    && level.loop_segments > 0
                    && level.panel_count > 0
                    && level.total_area_m2.is_finite()
                    && level.total_area_m2 > 0.0
                    && level.tip_area_fraction.is_finite()
                    && (0.0..=1.0).contains(&level.tip_area_fraction)
            })
    }

    /// Whether the geometric ladder passed its final-level threshold.
    pub fn is_converged(&self) -> bool {
        self.converged && self.is_finite()
    }
}

/// Checked failure while constructing geometric refinement evidence.
#[derive(Debug, Clone, PartialEq)]
pub enum PropellerMeshConvergenceError {
    /// A refined panel mesh failed its normal checked geometry admission.
    Mesh(BladeMeshError),
    /// The refinement ladder or threshold is invalid.
    InvalidConfiguration {
        /// Invalid public field.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// A generated mesh contained a non-finite geometric observable.
    NonFiniteObservable {
        /// Refinement multiplier that produced the invalid observable.
        refinement: usize,
        /// Observable name.
        field: &'static str,
    },
}

impl fmt::Display for PropellerMeshConvergenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mesh(error) => write!(f, "invalid propeller refinement mesh: {error}"),
            Self::InvalidConfiguration { field, requirement } => {
                write!(f, "propeller refinement {field} must be {requirement}")
            }
            Self::NonFiniteObservable { refinement, field } => write!(
                f,
                "propeller refinement {refinement} produced a non-finite {field}"
            ),
        }
    }
}

impl std::error::Error for PropellerMeshConvergenceError {}

impl From<BladeMeshError> for PropellerMeshConvergenceError {
    fn from(error: BladeMeshError) -> Self {
        Self::Mesh(error)
    }
}

/// Build a checked geometric panel-refinement ladder.
pub fn evaluate_mesh_convergence(
    base: Blade3dConfig,
    design: &PropellerDesign,
    refinements: &[usize],
    tolerance: f64,
) -> Result<PropellerMeshConvergence, PropellerMeshConvergenceError> {
    if refinements.len() < 2
        || refinements.contains(&0)
        || refinements.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err(PropellerMeshConvergenceError::InvalidConfiguration {
            field: "refinements",
            requirement: "at least two positive strictly increasing multipliers",
        });
    }
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(PropellerMeshConvergenceError::InvalidConfiguration {
            field: "tolerance",
            requirement: "finite and positive",
        });
    }

    let mut levels = Vec::with_capacity(refinements.len());
    for &refinement in refinements {
        let config = Blade3dConfig {
            radial_segments: base.radial_segments.checked_mul(refinement).ok_or(
                PropellerMeshConvergenceError::InvalidConfiguration {
                    field: "refinements",
                    requirement: "must not overflow panel counts",
                },
            )?,
            chord_segments: base.chord_segments.checked_mul(refinement).ok_or(
                PropellerMeshConvergenceError::InvalidConfiguration {
                    field: "refinements",
                    requirement: "must not overflow panel counts",
                },
            )?,
            loop_segments: base.loop_segments.checked_mul(refinement).ok_or(
                PropellerMeshConvergenceError::InvalidConfiguration {
                    field: "refinements",
                    requirement: "must not overflow panel counts",
                },
            )?,
            ..base
        };
        let mesh = config.try_build_mesh(design)?;
        let total_area_m2 = mesh.total_area();
        let tip_area_m2 = mesh.tip_panel_area();
        let tip_area_fraction = tip_area_m2 / total_area_m2.max(f64::MIN_POSITIVE);
        if !total_area_m2.is_finite() {
            return Err(PropellerMeshConvergenceError::NonFiniteObservable {
                refinement,
                field: "total_area_m2",
            });
        }
        if !tip_area_fraction.is_finite() {
            return Err(PropellerMeshConvergenceError::NonFiniteObservable {
                refinement,
                field: "tip_area_fraction",
            });
        }
        levels.push(PropellerMeshConvergenceLevel {
            refinement,
            radial_segments: config.radial_segments,
            chord_segments: config.chord_segments,
            loop_segments: config.loop_segments,
            panel_count: mesh.panels.len(),
            total_area_m2,
            tip_area_fraction,
        });
    }

    let relative_area_changes = levels
        .windows(2)
        .map(|pair| {
            (pair[1].total_area_m2 - pair[0].total_area_m2).abs()
                / pair[1].total_area_m2.max(f64::MIN_POSITIVE)
        })
        .collect::<Vec<_>>();
    let max_relative_area_change = relative_area_changes.iter().copied().fold(0.0, f64::max);
    let final_relative_area_change = match relative_area_changes.last().copied() {
        Some(change) => change,
        None => {
            return Err(PropellerMeshConvergenceError::InvalidConfiguration {
                field: "refinements",
                requirement: "must contain at least two levels",
            });
        }
    };
    let result = PropellerMeshConvergence {
        levels,
        relative_area_changes,
        max_relative_area_change,
        final_relative_area_change,
        tolerance,
        converged: final_relative_area_change <= tolerance,
    };
    if !result.is_finite() {
        let refinement = refinements.last().copied().unwrap_or_default();
        return Err(PropellerMeshConvergenceError::NonFiniteObservable {
            refinement,
            field: "relative_area_change",
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EdgeTreatment, PropellerPlanform};
    use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;

    fn base_config() -> Blade3dConfig {
        Blade3dConfig {
            radial_segments: 4,
            chord_segments: 2,
            loop_segments: 3,
            ..Blade3dConfig::default()
        }
    }

    #[test]
    fn geometric_refinement_is_deterministic_and_finite_for_both_planforms() {
        let source = PropellerSource::drone_5inch(18_000.0);
        for planform in [
            PropellerPlanform::Conventional,
            PropellerPlanform::ToroidalLoop,
        ] {
            let design = PropellerDesign {
                source: source.clone(),
                planform,
                edge_treatment: EdgeTreatment::Smooth,
                ..PropellerDesign::conventional(source.clone())
            };
            let first = evaluate_mesh_convergence(base_config(), &design, &[1, 2, 4], 0.2).unwrap();
            let second =
                evaluate_mesh_convergence(base_config(), &design, &[1, 2, 4], 0.2).unwrap();
            assert_eq!(first, second);
            assert!(first.is_finite());
            assert_eq!(first.levels.len(), 3);
            assert_eq!(first.relative_area_changes.len(), 2);
            assert!(
                first
                    .levels
                    .windows(2)
                    .all(|pair| pair[1].panel_count > pair[0].panel_count)
            );
            assert!(
                first
                    .relative_area_changes
                    .iter()
                    .all(|change| change.is_finite() && *change >= 0.0)
            );
            assert!(first.is_converged());
        }
    }

    #[test]
    fn geometric_refinement_rejects_malformed_ladders_and_overflow() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let design = PropellerDesign::conventional(source);
        assert!(matches!(
            evaluate_mesh_convergence(base_config(), &design, &[1, 1], 0.1),
            Err(PropellerMeshConvergenceError::InvalidConfiguration {
                field: "refinements",
                ..
            })
        ));
        assert!(matches!(
            evaluate_mesh_convergence(base_config(), &design, &[1, 2], f64::NAN),
            Err(PropellerMeshConvergenceError::InvalidConfiguration {
                field: "tolerance",
                ..
            })
        ));
        let overflow_config = Blade3dConfig {
            radial_segments: usize::MAX,
            ..base_config()
        };
        assert!(matches!(
            evaluate_mesh_convergence(overflow_config, &design, &[1, 2], 0.1),
            Err(PropellerMeshConvergenceError::InvalidConfiguration {
                field: "refinements",
                ..
            }) | Err(PropellerMeshConvergenceError::Mesh(_))
        ));
    }
}
