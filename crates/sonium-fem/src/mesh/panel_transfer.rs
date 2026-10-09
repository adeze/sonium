//! Checked scalar state transfer between panelized surface meshes.
//!
//! This module owns the reusable inverse-distance panel-transfer layer
//! extracted from `sonium-sim-propellers`' `mesh_state_transfer`: deterministic
//! interpolation between panels of the same group, with explicit geometry,
//! work, neighbor, and distance budgets plus area-weighted-mean diagnostics.
//! It is a geometric field transfer reference; it is not a CFD, wake,
//! conservative finite-volume, or aeroelastic state-transfer implementation.
//! Product panel types (`BladeMesh`) stay in the simulation crates, which map
//! to [`PanelSet`] and keep their own topology checks.

use std::fmt;

/// One panel in a transferable set: centroid, area, and group identity.
///
/// The group restricts interpolation to same-group source panels (propeller:
/// same blade). Centers must be finite, areas finite and positive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Panel {
    pub center: [f64; 3],
    pub area: f64,
    pub group: usize,
}

/// A validated panel set exchanged across the transfer seam.
#[derive(Debug, Clone, PartialEq)]
pub struct PanelSet {
    pub panels: Vec<Panel>,
}

/// Bounds for checked panel-state transfer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanelStateTransferOptions {
    /// Maximum accepted source panel count.
    pub maximum_source_panels: usize,
    /// Maximum accepted target panel count.
    pub maximum_target_panels: usize,
    /// Number of nearest same-group source panels used for interpolation.
    pub maximum_neighbors: usize,
    /// Smallest distance used in inverse-distance weights [m].
    pub distance_floor_m: f64,
    /// Maximum permitted nearest-source distance [m].
    pub maximum_source_distance_m: f64,
    /// Maximum source-target distance evaluations.
    pub maximum_distance_evaluations: usize,
}

impl Default for PanelStateTransferOptions {
    fn default() -> Self {
        Self {
            maximum_source_panels: 8192,
            maximum_target_panels: 8192,
            maximum_neighbors: 4,
            distance_floor_m: 1.0e-10,
            maximum_source_distance_m: 0.05,
            maximum_distance_evaluations: 2_000_000,
        }
    }
}

/// Result of transferring one scalar value per target panel.
#[derive(Debug, Clone, PartialEq)]
pub struct PanelStateTransferResult {
    /// Interpolated state in target-panel order.
    pub values: Vec<f64>,
    /// Largest nearest-source distance observed [m].
    pub maximum_source_distance_m: f64,
    /// Mean nearest-source distance over target panels [m].
    pub mean_source_distance_m: f64,
    /// Source area-weighted mean state.
    pub source_area_weighted_mean: f64,
    /// Target area-weighted mean after transfer.
    pub target_area_weighted_mean: f64,
    /// Target minus source area-weighted mean.
    pub area_weighted_mean_delta: f64,
}

impl PanelStateTransferResult {
    /// Returns whether all diagnostics and transferred values are finite.
    pub fn is_finite(&self) -> bool {
        self.values.iter().all(|value| value.is_finite())
            && self.maximum_source_distance_m.is_finite()
            && self.maximum_source_distance_m >= 0.0
            && self.mean_source_distance_m.is_finite()
            && self.mean_source_distance_m >= 0.0
            && self.source_area_weighted_mean.is_finite()
            && self.target_area_weighted_mean.is_finite()
            && self.area_weighted_mean_delta.is_finite()
    }
}

/// Failure-closed panel-state transfer errors.
#[derive(Debug, Clone, PartialEq)]
pub enum PanelStateTransferError {
    /// An option is outside its declared domain.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// A mesh contains no panels.
    EmptyMesh {
        /// Mesh role.
        mesh: &'static str,
    },
    /// A mesh exceeds its configured panel budget.
    TooManyPanels {
        /// Mesh role.
        mesh: &'static str,
        /// Actual panel count.
        count: usize,
        /// Configured maximum.
        maximum: usize,
    },
    /// The supplied state vector does not match its mesh.
    StateLengthMismatch {
        /// Mesh role.
        mesh: &'static str,
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// Mesh metadata is incompatible with state transfer.
    TopologyMismatch {
        /// Metadata field.
        field: &'static str,
    },
    /// A mesh panel contains non-finite or degenerate geometry.
    InvalidPanel {
        /// Mesh role.
        mesh: &'static str,
        /// Panel index.
        index: usize,
        /// Required geometry property.
        requirement: &'static str,
    },
    /// A target group has no corresponding source group.
    MissingSourceGroup {
        /// Target group identity.
        group: usize,
    },
    /// Distance work exceeded the configured budget.
    DistanceBudgetExceeded {
        /// Requested evaluations.
        evaluations: usize,
        /// Configured maximum.
        maximum: usize,
    },
    /// No source panel was available for a target panel.
    NoSourceNeighbor {
        /// Target panel index.
        target_index: usize,
        /// Target group identity.
        group: usize,
    },
    /// Nearest source panel is too far from a target panel.
    SourceDistanceExceeded {
        /// Target panel index.
        target_index: usize,
        /// Nearest-source distance [m].
        distance_m: f64,
        /// Configured maximum [m].
        maximum_m: f64,
    },
    /// A derived output became non-finite.
    NonFiniteResult {
        /// Result field name.
        field: &'static str,
    },
}

impl fmt::Display for PanelStateTransferError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(
                    formatter,
                    "panel state transfer {field} must be {requirement}"
                )
            }
            Self::EmptyMesh { mesh } => {
                write!(formatter, "panel state transfer {mesh} mesh is empty")
            }
            Self::TooManyPanels {
                mesh,
                count,
                maximum,
            } => write!(
                formatter,
                "panel state transfer {mesh} mesh has {count} panels, maximum is {maximum}"
            ),
            Self::StateLengthMismatch {
                mesh,
                expected,
                actual,
            } => write!(
                formatter,
                "panel state transfer {mesh} state length {actual} does not match {expected} panels"
            ),
            Self::TopologyMismatch { field } => {
                write!(
                    formatter,
                    "panel state transfer topology mismatch in {field}"
                )
            }
            Self::InvalidPanel {
                mesh,
                index,
                requirement,
            } => write!(
                formatter,
                "panel state transfer {mesh} panel {index} must have {requirement}"
            ),
            Self::MissingSourceGroup { group } => write!(
                formatter,
                "panel state transfer has no source group {group}"
            ),
            Self::DistanceBudgetExceeded {
                evaluations,
                maximum,
            } => write!(
                formatter,
                "panel state transfer distance work {evaluations} exceeds {maximum}"
            ),
            Self::NoSourceNeighbor {
                target_index,
                group,
            } => write!(
                formatter,
                "panel state transfer target panel {target_index} has no source neighbor in group {group}"
            ),
            Self::SourceDistanceExceeded {
                target_index,
                distance_m,
                maximum_m,
            } => write!(
                formatter,
                "panel state transfer target panel {target_index} nearest distance {distance_m} exceeds {maximum_m}"
            ),
            Self::NonFiniteResult { field } => {
                write!(
                    formatter,
                    "panel state transfer result {field} is non-finite"
                )
            }
        }
    }
}

impl std::error::Error for PanelStateTransferError {}

/// Transfer one scalar state per panel between compatible 3-D meshes.
pub fn transfer_panel_scalar_state(
    source_mesh: &PanelSet,
    source_values: &[f64],
    target_mesh: &PanelSet,
    options: PanelStateTransferOptions,
) -> Result<PanelStateTransferResult, PanelStateTransferError> {
    validate_options(options)?;
    validate_mesh(
        source_mesh,
        Some(source_values),
        "source",
        options.maximum_source_panels,
    )?;
    validate_mesh(target_mesh, None, "target", options.maximum_target_panels)?;

    let mut distance_evaluations = 0usize;
    let mut transferred_values = Vec::with_capacity(target_mesh.panels.len());
    let mut maximum_source_distance_m = 0.0_f64;
    let mut distance_sum = 0.0_f64;

    for (target_index, target_panel) in target_mesh.panels.iter().enumerate() {
        let target_center = target_panel.center;
        let mut candidates = Vec::new();
        for (source_index, source_panel) in source_mesh.panels.iter().enumerate() {
            if source_panel.group != target_panel.group {
                continue;
            }
            distance_evaluations = distance_evaluations.checked_add(1).ok_or(
                PanelStateTransferError::DistanceBudgetExceeded {
                    evaluations: usize::MAX,
                    maximum: options.maximum_distance_evaluations,
                },
            )?;
            if distance_evaluations > options.maximum_distance_evaluations {
                return Err(PanelStateTransferError::DistanceBudgetExceeded {
                    evaluations: distance_evaluations,
                    maximum: options.maximum_distance_evaluations,
                });
            }
            let distance_squared = distance_squared(target_center, source_panel.center);
            if !distance_squared.is_finite() {
                return Err(PanelStateTransferError::NonFiniteResult {
                    field: "distance_squared",
                });
            }
            candidates.push((source_index, distance_squared));
        }
        if candidates.is_empty() {
            return Err(PanelStateTransferError::MissingSourceGroup {
                group: target_panel.group,
            });
        }
        candidates.sort_unstable_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let nearest_distance_m = candidates[0].1.sqrt();
        if nearest_distance_m > options.maximum_source_distance_m {
            return Err(PanelStateTransferError::SourceDistanceExceeded {
                target_index,
                distance_m: nearest_distance_m,
                maximum_m: options.maximum_source_distance_m,
            });
        }
        maximum_source_distance_m = maximum_source_distance_m.max(nearest_distance_m);
        distance_sum += nearest_distance_m;

        let nearest = candidates[0].0;
        let transferred = if nearest_distance_m <= options.distance_floor_m {
            source_values[nearest]
        } else {
            let mut weight_sum = 0.0;
            let mut value_sum = 0.0;
            for (source_index, distance_squared) in
                candidates.into_iter().take(options.maximum_neighbors)
            {
                let distance = distance_squared.sqrt().max(options.distance_floor_m);
                let weight = 1.0 / (distance * distance);
                weight_sum += weight;
                value_sum += weight * source_values[source_index];
            }
            if !weight_sum.is_finite() || weight_sum <= 0.0 {
                return Err(PanelStateTransferError::NonFiniteResult {
                    field: "interpolation_weight_sum",
                });
            }
            value_sum / weight_sum
        };
        if !transferred.is_finite() {
            return Err(PanelStateTransferError::NonFiniteResult {
                field: "transferred_value",
            });
        }
        transferred_values.push(transferred);
    }

    let source_area_weighted_mean = area_weighted_mean(&source_mesh.panels, source_values)?;
    let target_area_weighted_mean = area_weighted_mean(&target_mesh.panels, &transferred_values)?;
    let area_weighted_mean_delta = target_area_weighted_mean - source_area_weighted_mean;
    let result = PanelStateTransferResult {
        values: transferred_values,
        maximum_source_distance_m,
        mean_source_distance_m: distance_sum / target_mesh.panels.len() as f64,
        source_area_weighted_mean,
        target_area_weighted_mean,
        area_weighted_mean_delta,
    };
    if !result.is_finite() {
        return Err(PanelStateTransferError::NonFiniteResult {
            field: "transfer_result",
        });
    }
    Ok(result)
}

fn validate_options(options: PanelStateTransferOptions) -> Result<(), PanelStateTransferError> {
    if options.maximum_source_panels == 0 {
        return Err(invalid_option("maximum_source_panels", "positive"));
    }
    if options.maximum_target_panels == 0 {
        return Err(invalid_option("maximum_target_panels", "positive"));
    }
    if !(1..=32).contains(&options.maximum_neighbors) {
        return Err(invalid_option("maximum_neighbors", "within [1, 32]"));
    }
    if !options.distance_floor_m.is_finite() || options.distance_floor_m <= 0.0 {
        return Err(invalid_option("distance_floor_m", "finite and positive"));
    }
    if !options.maximum_source_distance_m.is_finite() || options.maximum_source_distance_m <= 0.0 {
        return Err(invalid_option(
            "maximum_source_distance_m",
            "finite and positive",
        ));
    }
    if options.maximum_distance_evaluations == 0 {
        return Err(invalid_option("maximum_distance_evaluations", "positive"));
    }
    Ok(())
}

fn validate_mesh(
    mesh: &PanelSet,
    values: Option<&[f64]>,
    role: &'static str,
    maximum_panels: usize,
) -> Result<(), PanelStateTransferError> {
    if mesh.panels.is_empty() {
        return Err(PanelStateTransferError::EmptyMesh { mesh: role });
    }
    if mesh.panels.len() > maximum_panels {
        return Err(PanelStateTransferError::TooManyPanels {
            mesh: role,
            count: mesh.panels.len(),
            maximum: maximum_panels,
        });
    }
    if let Some(values) = values
        && values.len() != mesh.panels.len()
    {
        return Err(PanelStateTransferError::StateLengthMismatch {
            mesh: role,
            expected: mesh.panels.len(),
            actual: values.len(),
        });
    }
    for (index, panel) in mesh.panels.iter().enumerate() {
        if let Some(values) = values
            && !values[index].is_finite()
        {
            return Err(PanelStateTransferError::InvalidPanel {
                mesh: role,
                index,
                requirement: "finite state",
            });
        }
        if !panel.area.is_finite() || panel.area <= 0.0 {
            return Err(PanelStateTransferError::InvalidPanel {
                mesh: role,
                index,
                requirement: "finite positive area",
            });
        }
        if !panel.center.iter().all(|value| value.is_finite()) {
            return Err(PanelStateTransferError::InvalidPanel {
                mesh: role,
                index,
                requirement: "finite center",
            });
        }
    }
    Ok(())
}

fn distance_squared(left: [f64; 3], right: [f64; 3]) -> f64 {
    (left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2)
}

fn area_weighted_mean(panels: &[Panel], values: &[f64]) -> Result<f64, PanelStateTransferError> {
    let mut area_sum = 0.0;
    let mut weighted_sum = 0.0;
    for (panel, value) in panels.iter().zip(values) {
        area_sum += panel.area;
        weighted_sum += panel.area * value;
    }
    if !area_sum.is_finite() || area_sum <= 0.0 {
        return Err(PanelStateTransferError::NonFiniteResult { field: "area_sum" });
    }
    let mean = weighted_sum / area_sum;
    if !mean.is_finite() {
        return Err(PanelStateTransferError::NonFiniteResult {
            field: "area_weighted_mean",
        });
    }
    Ok(mean)
}

fn invalid_option(field: &'static str, requirement: &'static str) -> PanelStateTransferError {
    PanelStateTransferError::InvalidOptions { field, requirement }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_panel_set(offset: f64) -> PanelSet {
        PanelSet {
            panels: vec![
                Panel {
                    center: [offset, 0.0, 0.0],
                    area: 0.5,
                    group: 0,
                },
                Panel {
                    center: [offset, 1.0, 0.0],
                    area: 0.5,
                    group: 0,
                },
            ],
        }
    }

    #[test]
    fn constant_field_is_preserved_with_matching_means() {
        let source = two_panel_set(0.0);
        let target = two_panel_set(0.001);
        let result = transfer_panel_scalar_state(
            &source,
            &[3.0, 3.0],
            &target,
            PanelStateTransferOptions::default(),
        )
        .expect("transfer");
        assert!(result.is_finite());
        for value in &result.values {
            assert!((value - 3.0).abs() < 1.0e-9);
        }
        assert!((result.source_area_weighted_mean - 3.0).abs() < 1.0e-12);
        assert!((result.target_area_weighted_mean - 3.0).abs() < 1.0e-9);
        assert!(result.area_weighted_mean_delta.abs() < 1.0e-9);
    }

    #[test]
    fn missing_group_and_bad_options_fail_closed() {
        let source = two_panel_set(0.0);
        let lonely = PanelSet {
            panels: vec![Panel {
                center: [10.0, 0.0, 0.0],
                area: 0.5,
                group: 7,
            }],
        };
        assert!(matches!(
            transfer_panel_scalar_state(
                &source,
                &[1.0, 1.0],
                &lonely,
                PanelStateTransferOptions::default()
            ),
            Err(PanelStateTransferError::MissingSourceGroup { group: 7 })
        ));
        let bad_options = PanelStateTransferOptions {
            maximum_neighbors: 0,
            ..PanelStateTransferOptions::default()
        };
        assert!(matches!(
            transfer_panel_scalar_state(&source, &[1.0, 1.0], &source, bad_options),
            Err(PanelStateTransferError::InvalidOptions { .. })
        ));
    }
}
