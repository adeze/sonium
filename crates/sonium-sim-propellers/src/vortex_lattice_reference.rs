//! Checked 3-D horseshoe-vortex lattice reference over [`BladePanel`] geometry.
//!
//! This is the first panel-flow lane in the propeller crate.  Each supplied
//! quadrilateral uses the corner order `[leading-root, leading-tip,
//! trailing-tip, trailing-root]`.  A quarter-chord bound vortex and a finite
//! discretised trailing wake are assembled into a dense influence matrix.
//! The resulting circulation is converted to a Kutta--Joukowski force ledger
//! using the local free-stream plus prescribed rigid-body rotation.
//!
//! The implementation is deliberately bounded and fail-closed.  It is a
//! steady incompressible VLM reference, not a viscous solver, a nonlinear
//! free-wake roll-up, RANS/URANS/LES, or a moving-surface FW-H solver.

use crate::blade3d::{BladePanel, Point3};
use std::f64::consts::PI;
use std::fmt;

const MIN_LENGTH_M: f64 = 1.0e-12;

/// Controls for the checked vortex-lattice reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VortexLatticeOptions {
    /// Constant fluid density used by the Kutta--Joukowski load ledger.
    pub density_kg_m3: f64,
    /// Uniform inertial-frame velocity at the origin.
    pub freestream_m_s: Point3,
    /// Unit-load axis and rigid-body rotation axis.
    pub rotation_axis: Point3,
    /// Prescribed angular velocity about `rotation_axis`.
    pub angular_velocity_rad_s: f64,
    /// Origin for rigid-body rotation and moment reporting.
    pub origin_m: Point3,
    /// Direction of the finite trailing wake.
    pub wake_direction: Point3,
    /// Length of each trailing wake leg.
    pub wake_length_m: f64,
    /// Number of straight segments in each trailing wake leg.
    pub wake_segments: usize,
    /// Lamb--Oseen-style core radius used to regularise Biot--Savart action.
    pub core_radius_m: f64,
    /// Hard matrix-size budget.
    pub max_panels: usize,
    /// Hard wake-segment budget per panel and leg.
    pub max_wake_segments: usize,
    /// Absolute Gaussian-elimination pivot admission threshold.
    pub pivot_tolerance: f64,
    /// Maximum permitted induced velocity at a collocation point.
    pub max_induced_velocity_m_s: f64,
    /// Maximum permitted post-solve normal residual.
    pub residual_tolerance_m_s: f64,
}

impl Default for VortexLatticeOptions {
    fn default() -> Self {
        Self {
            density_kg_m3: 1.225,
            freestream_m_s: Point3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            rotation_axis: Point3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            angular_velocity_rad_s: 0.0,
            origin_m: Point3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            wake_direction: Point3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            wake_length_m: 1.0,
            wake_segments: 8,
            core_radius_m: 1.0e-5,
            max_panels: 256,
            max_wake_segments: 64,
            pivot_tolerance: 1.0e-12,
            max_induced_velocity_m_s: 1.0e6,
            residual_tolerance_m_s: 1.0e-7,
        }
    }
}

/// Per-panel circulation and aerodynamic load result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VortexLatticePanelResult {
    pub collocation_point_m: Point3,
    pub normal: Point3,
    pub bound_vortex_start_m: Point3,
    pub bound_vortex_end_m: Point3,
    pub local_velocity_m_s: Point3,
    pub induced_velocity_m_s: Point3,
    pub effective_velocity_m_s: Point3,
    pub circulation_m2_s: f64,
    pub force_n: Point3,
    pub moment_nm: Point3,
    pub normal_residual_m_s: f64,
}

impl VortexLatticePanelResult {
    pub fn is_finite(&self) -> bool {
        finite_point(self.collocation_point_m)
            && finite_point(self.normal)
            && finite_point(self.bound_vortex_start_m)
            && finite_point(self.bound_vortex_end_m)
            && finite_point(self.local_velocity_m_s)
            && finite_point(self.induced_velocity_m_s)
            && finite_point(self.effective_velocity_m_s)
            && self.circulation_m2_s.is_finite()
            && finite_point(self.force_n)
            && finite_point(self.moment_nm)
            && self.normal_residual_m_s.is_finite()
    }
}

/// Integrated result of a checked vortex-lattice solve.
#[derive(Debug, Clone, PartialEq)]
pub struct VortexLatticeResult {
    pub panels: Vec<VortexLatticePanelResult>,
    pub total_force_n: Point3,
    pub total_moment_nm: Point3,
    pub thrust_n: f64,
    pub torque_nm: f64,
    pub maximum_normal_residual_m_s: f64,
    pub maximum_induced_velocity_m_s: f64,
    pub matrix_size: usize,
    pub wake_segment_count_per_leg: usize,
}

impl VortexLatticeResult {
    pub fn is_finite(&self) -> bool {
        finite_point(self.total_force_n)
            && finite_point(self.total_moment_nm)
            && self.thrust_n.is_finite()
            && self.torque_nm.is_finite()
            && self.maximum_normal_residual_m_s.is_finite()
            && self.maximum_induced_velocity_m_s.is_finite()
            && self.panels.iter().all(VortexLatticePanelResult::is_finite)
    }
}

/// Errors emitted before a VLM result can make a numerical claim.
#[derive(Debug, Clone, PartialEq)]
pub enum VortexLatticeError {
    EmptyPanels,
    TooManyPanels {
        count: usize,
        maximum: usize,
    },
    InvalidOption {
        field: &'static str,
        requirement: &'static str,
    },
    InvalidPanel {
        index: usize,
        field: &'static str,
        requirement: &'static str,
    },
    SingularSystem {
        pivot: usize,
        magnitude: f64,
    },
    NonFiniteSystem {
        row: usize,
        column: usize,
    },
    InducedVelocityExceeded {
        index: usize,
        velocity_m_s: f64,
        maximum_m_s: f64,
    },
    ResidualExceeded {
        maximum_m_s: f64,
        tolerance_m_s: f64,
    },
    NonFiniteResult {
        field: &'static str,
    },
}

impl fmt::Display for VortexLatticeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPanels => formatter.write_str("vortex lattice panel list must not be empty"),
            Self::TooManyPanels { count, maximum } => write!(
                formatter,
                "vortex lattice panel count {count} exceeds budget {maximum}"
            ),
            Self::InvalidOption { field, requirement } => {
                write!(
                    formatter,
                    "invalid vortex lattice option {field}: {requirement}"
                )
            }
            Self::InvalidPanel {
                index,
                field,
                requirement,
            } => write!(
                formatter,
                "invalid vortex lattice panel {index} field {field}: {requirement}"
            ),
            Self::SingularSystem { pivot, magnitude } => write!(
                formatter,
                "vortex lattice influence matrix is singular at pivot {pivot} ({magnitude:e})"
            ),
            Self::NonFiniteSystem { row, column } => write!(
                formatter,
                "vortex lattice influence matrix entry ({row}, {column}) is non-finite"
            ),
            Self::InducedVelocityExceeded {
                index,
                velocity_m_s,
                maximum_m_s,
            } => write!(
                formatter,
                "vortex lattice induced velocity at panel {index} is {velocity_m_s} m/s, above {maximum_m_s} m/s"
            ),
            Self::ResidualExceeded {
                maximum_m_s,
                tolerance_m_s,
            } => write!(
                formatter,
                "vortex lattice normal residual {maximum_m_s} m/s exceeds {tolerance_m_s} m/s"
            ),
            Self::NonFiniteResult { field } => {
                write!(formatter, "non-finite vortex lattice result field {field}")
            }
        }
    }
}

impl std::error::Error for VortexLatticeError {}

#[derive(Debug, Clone, Copy)]
struct PanelGeometry {
    collocation: Point3,
    normal: Point3,
    bound_start: Point3,
    bound_end: Point3,
}

/// Dense influence system assembled by the production horseshoe-vortex
/// lattice implementation.
///
/// The matrix is row-major and maps a circulation vector to the normal
/// induced-velocity contribution at each collocation point.  Keeping this
/// assembly contract separate lets an accelerator apply the *same* production
/// operator without claiming that the complete VLM solve or wake construction
/// has moved to the device.
#[derive(Debug, Clone, PartialEq)]
pub struct VortexLatticeLinearSystem {
    /// Row-major normal-velocity influence matrix.
    pub influence: Vec<f64>,
    /// Production local-flow right-hand side.
    pub rhs: Vec<f64>,
    /// Number of rows and columns in `influence`.
    pub dimension: usize,
}

/// Assemble the dense production VLM influence matrix and right-hand side.
///
/// This is the exact geometry/wake assembly used by
/// [`evaluate_vortex_lattice_reference`].  It is intentionally an assembly
/// API: solving the system and converting circulation to blade loads remain
/// in the checked host reference.
pub fn assemble_vortex_lattice_system(
    panels: &[BladePanel],
    options: VortexLatticeOptions,
) -> Result<VortexLatticeLinearSystem, VortexLatticeError> {
    validate_options(&options)?;
    if panels.is_empty() {
        return Err(VortexLatticeError::EmptyPanels);
    }
    if panels.len() > options.max_panels {
        return Err(VortexLatticeError::TooManyPanels {
            count: panels.len(),
            maximum: options.max_panels,
        });
    }

    let geometry = panels
        .iter()
        .enumerate()
        .map(|(index, panel)| validate_panel(index, panel))
        .collect::<Result<Vec<_>, _>>()?;
    let wake_direction =
        normalize(options.wake_direction).ok_or(VortexLatticeError::InvalidOption {
            field: "wake_direction",
            requirement: "finite and non-zero",
        })?;
    let rotation_axis =
        normalize(options.rotation_axis).ok_or(VortexLatticeError::InvalidOption {
            field: "rotation_axis",
            requirement: "finite and non-zero",
        })?;
    let n = geometry.len();
    let mut influence = vec![0.0; n.saturating_mul(n)];
    let mut rhs = vec![0.0; n];
    for (row, panel) in geometry.iter().enumerate() {
        let local_velocity = local_velocity(panel.collocation, &options, rotation_axis);
        rhs[row] = -dot(local_velocity, panel.normal);
        for (column, source) in geometry.iter().enumerate() {
            let velocity = horseshoe_velocity(
                panel.collocation,
                source,
                wake_direction,
                options.wake_length_m,
                options.wake_segments,
                options.core_radius_m,
            );
            let value = dot(velocity, panel.normal);
            if !value.is_finite() {
                return Err(VortexLatticeError::NonFiniteSystem { row, column });
            }
            influence[row * n + column] = value;
        }
    }
    Ok(VortexLatticeLinearSystem {
        influence,
        rhs,
        dimension: n,
    })
}

/// Evaluate a bounded steady horseshoe-vortex lattice on generated blade panels.
pub fn evaluate_vortex_lattice_reference(
    panels: &[BladePanel],
    options: VortexLatticeOptions,
) -> Result<VortexLatticeResult, VortexLatticeError> {
    let system = assemble_vortex_lattice_system(panels, options)?;
    let geometry = panels
        .iter()
        .enumerate()
        .map(|(index, panel)| validate_panel(index, panel))
        .collect::<Result<Vec<_>, _>>()?;
    let wake_direction =
        normalize(options.wake_direction).ok_or(VortexLatticeError::InvalidOption {
            field: "wake_direction",
            requirement: "finite and non-zero",
        })?;
    let rotation_axis =
        normalize(options.rotation_axis).ok_or(VortexLatticeError::InvalidOption {
            field: "rotation_axis",
            requirement: "finite and non-zero",
        })?;
    let n = system.dimension;
    let influence = system
        .influence
        .chunks_exact(n)
        .map(<[f64]>::to_vec)
        .collect();
    let circulation = solve_linear_system(influence, system.rhs, options.pivot_tolerance)?;
    let mut results = Vec::with_capacity(n);
    let mut total_force = zero_point();
    let mut total_moment = zero_point();
    let mut maximum_residual: f64 = 0.0;
    let mut maximum_induced: f64 = 0.0;

    for (index, panel) in geometry.iter().enumerate() {
        let local = local_velocity(panel.collocation, &options, rotation_axis);
        let mut induced = zero_point();
        for (source_index, source) in geometry.iter().enumerate() {
            induced = add(
                induced,
                scale(
                    horseshoe_velocity(
                        panel.collocation,
                        source,
                        wake_direction,
                        options.wake_length_m,
                        options.wake_segments,
                        options.core_radius_m,
                    ),
                    circulation[source_index],
                ),
            );
        }
        let induced_norm = norm(induced);
        if !induced_norm.is_finite() {
            return Err(VortexLatticeError::NonFiniteResult {
                field: "induced_velocity_m_s",
            });
        }
        if induced_norm > options.max_induced_velocity_m_s {
            return Err(VortexLatticeError::InducedVelocityExceeded {
                index,
                velocity_m_s: induced_norm,
                maximum_m_s: options.max_induced_velocity_m_s,
            });
        }
        let effective = add(local, induced);
        let residual = dot(effective, panel.normal).abs();
        maximum_residual = maximum_residual.max(residual);
        maximum_induced = maximum_induced.max(induced_norm);
        let force = scale(
            cross(effective, sub(panel.bound_end, panel.bound_start)),
            options.density_kg_m3 * circulation[index],
        );
        let moment = cross(sub(panel.collocation, options.origin_m), force);
        if !finite_point(force) || !finite_point(moment) {
            return Err(VortexLatticeError::NonFiniteResult { field: "load" });
        }
        total_force = add(total_force, force);
        total_moment = add(total_moment, moment);
        results.push(VortexLatticePanelResult {
            collocation_point_m: panel.collocation,
            normal: panel.normal,
            bound_vortex_start_m: panel.bound_start,
            bound_vortex_end_m: panel.bound_end,
            local_velocity_m_s: local,
            induced_velocity_m_s: induced,
            effective_velocity_m_s: effective,
            circulation_m2_s: circulation[index],
            force_n: force,
            moment_nm: moment,
            normal_residual_m_s: residual,
        });
    }

    if maximum_residual > options.residual_tolerance_m_s {
        return Err(VortexLatticeError::ResidualExceeded {
            maximum_m_s: maximum_residual,
            tolerance_m_s: options.residual_tolerance_m_s,
        });
    }
    let thrust = dot(total_force, rotation_axis);
    let torque = dot(total_moment, rotation_axis);
    let result = VortexLatticeResult {
        panels: results,
        total_force_n: total_force,
        total_moment_nm: total_moment,
        thrust_n: thrust,
        torque_nm: torque,
        maximum_normal_residual_m_s: maximum_residual,
        maximum_induced_velocity_m_s: maximum_induced,
        matrix_size: n,
        wake_segment_count_per_leg: options.wake_segments,
    };
    if !result.is_finite() {
        return Err(VortexLatticeError::NonFiniteResult { field: "result" });
    }
    Ok(result)
}

fn validate_options(options: &VortexLatticeOptions) -> Result<(), VortexLatticeError> {
    if !options.density_kg_m3.is_finite() || options.density_kg_m3 <= 0.0 {
        return Err(VortexLatticeError::InvalidOption {
            field: "density_kg_m3",
            requirement: "finite and positive",
        });
    }
    for (field, value) in [
        ("angular_velocity_rad_s", options.angular_velocity_rad_s),
        ("wake_length_m", options.wake_length_m),
        ("core_radius_m", options.core_radius_m),
        ("pivot_tolerance", options.pivot_tolerance),
        ("max_induced_velocity_m_s", options.max_induced_velocity_m_s),
        ("residual_tolerance_m_s", options.residual_tolerance_m_s),
    ] {
        if !value.is_finite() || value < 0.0 || (field == "wake_length_m" && value <= 0.0) {
            return Err(VortexLatticeError::InvalidOption {
                field,
                requirement: "finite and within its positive budget",
            });
        }
    }
    if options.core_radius_m <= 0.0 || options.pivot_tolerance <= 0.0 {
        return Err(VortexLatticeError::InvalidOption {
            field: "core_radius_m/pivot_tolerance",
            requirement: "strictly positive",
        });
    }
    if options.wake_segments == 0 || options.max_wake_segments == 0 {
        return Err(VortexLatticeError::InvalidOption {
            field: "wake_segments",
            requirement: "non-zero",
        });
    }
    if options.wake_segments > options.max_wake_segments {
        return Err(VortexLatticeError::InvalidOption {
            field: "wake_segments",
            requirement: "must not exceed max_wake_segments",
        });
    }
    if options.max_panels == 0 {
        return Err(VortexLatticeError::InvalidOption {
            field: "max_panels",
            requirement: "non-zero",
        });
    }
    for (field, point) in [
        ("freestream_m_s", options.freestream_m_s),
        ("rotation_axis", options.rotation_axis),
        ("origin_m", options.origin_m),
        ("wake_direction", options.wake_direction),
    ] {
        if !finite_point(point) {
            return Err(VortexLatticeError::InvalidOption {
                field,
                requirement: "all components finite",
            });
        }
    }
    if norm(options.rotation_axis) <= MIN_LENGTH_M {
        return Err(VortexLatticeError::InvalidOption {
            field: "rotation_axis",
            requirement: "non-zero",
        });
    }
    if norm(options.wake_direction) <= MIN_LENGTH_M {
        return Err(VortexLatticeError::InvalidOption {
            field: "wake_direction",
            requirement: "non-zero",
        });
    }
    Ok(())
}

fn validate_panel(index: usize, panel: &BladePanel) -> Result<PanelGeometry, VortexLatticeError> {
    if !panel.vertices.iter().copied().all(finite_point) {
        return Err(VortexLatticeError::InvalidPanel {
            index,
            field: "vertices",
            requirement: "all components finite",
        });
    }
    let root_chord = sub(panel.vertices[3], panel.vertices[0]);
    let tip_chord = sub(panel.vertices[2], panel.vertices[1]);
    let leading_span = sub(panel.vertices[1], panel.vertices[0]);
    let diagonal = sub(panel.vertices[2], panel.vertices[0]);
    let area = 0.5 * norm(cross(leading_span, diagonal))
        + 0.5 * norm(cross(diagonal, sub(panel.vertices[3], panel.vertices[0])));
    if !area.is_finite() || area <= MIN_LENGTH_M {
        return Err(VortexLatticeError::InvalidPanel {
            index,
            field: "area",
            requirement: "finite and non-degenerate",
        });
    }
    if norm(root_chord) <= MIN_LENGTH_M || norm(tip_chord) <= MIN_LENGTH_M {
        return Err(VortexLatticeError::InvalidPanel {
            index,
            field: "chord",
            requirement: "root and tip chord must be non-zero",
        });
    }
    let normal =
        normalize(cross(root_chord, leading_span)).ok_or(VortexLatticeError::InvalidPanel {
            index,
            field: "normal",
            requirement: "finite and non-zero",
        })?;
    let bound_start = add(panel.vertices[0], scale(root_chord, 0.25));
    let bound_end = add(panel.vertices[1], scale(tip_chord, 0.25));
    if norm(sub(bound_end, bound_start)) <= MIN_LENGTH_M {
        return Err(VortexLatticeError::InvalidPanel {
            index,
            field: "bound_vortex",
            requirement: "finite and non-zero span",
        });
    }
    let collocation = scale(
        add(
            add(panel.vertices[0], scale(root_chord, 0.75)),
            add(panel.vertices[1], scale(tip_chord, 0.75)),
        ),
        0.5,
    );
    Ok(PanelGeometry {
        collocation,
        normal,
        bound_start,
        bound_end,
    })
}

fn local_velocity(point: Point3, options: &VortexLatticeOptions, axis: Point3) -> Point3 {
    let angular_velocity = scale(axis, options.angular_velocity_rad_s);
    add(
        options.freestream_m_s,
        cross(angular_velocity, sub(point, options.origin_m)),
    )
}

fn horseshoe_velocity(
    point: Point3,
    panel: &PanelGeometry,
    wake_direction: Point3,
    wake_length_m: f64,
    wake_segments: usize,
    core_radius_m: f64,
) -> Point3 {
    let mut velocity =
        vortex_segment_velocity(point, panel.bound_start, panel.bound_end, core_radius_m);
    for segment in 0..wake_segments {
        let start_fraction = segment as f64 / wake_segments as f64;
        let end_fraction = (segment + 1) as f64 / wake_segments as f64;
        let root_start = add(
            panel.bound_start,
            scale(wake_direction, wake_length_m * start_fraction),
        );
        let root_end = add(
            panel.bound_start,
            scale(wake_direction, wake_length_m * end_fraction),
        );
        let tip_start = add(
            panel.bound_end,
            scale(wake_direction, wake_length_m * start_fraction),
        );
        let tip_end = add(
            panel.bound_end,
            scale(wake_direction, wake_length_m * end_fraction),
        );
        velocity = add(
            velocity,
            vortex_segment_velocity(point, root_end, root_start, core_radius_m),
        );
        velocity = add(
            velocity,
            vortex_segment_velocity(point, tip_start, tip_end, core_radius_m),
        );
    }
    velocity
}

fn vortex_segment_velocity(
    point: Point3,
    start: Point3,
    end: Point3,
    core_radius_m: f64,
) -> Point3 {
    let first = sub(point, start);
    let second = sub(point, end);
    let segment = sub(end, start);
    let cross_product = cross(first, second);
    let first_norm = (dot(first, first) + core_radius_m * core_radius_m).sqrt();
    let second_norm = (dot(second, second) + core_radius_m * core_radius_m).sqrt();
    let denominator = dot(cross_product, cross_product) + core_radius_m * core_radius_m;
    if norm(segment) <= MIN_LENGTH_M || denominator <= f64::MIN_POSITIVE {
        return zero_point();
    }
    let projection = dot(
        segment,
        sub(
            scale(first, 1.0 / first_norm),
            scale(second, 1.0 / second_norm),
        ),
    );
    scale(cross_product, projection / (4.0 * PI * denominator))
}

fn solve_linear_system(
    mut matrix: Vec<Vec<f64>>,
    mut rhs: Vec<f64>,
    pivot_tolerance: f64,
) -> Result<Vec<f64>, VortexLatticeError> {
    let n = rhs.len();
    for column in 0..n {
        let (pivot, magnitude) = (column..n)
            .map(|row| (row, matrix[row][column].abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap_or((column, 0.0));
        if !magnitude.is_finite() {
            return Err(VortexLatticeError::NonFiniteSystem { row: pivot, column });
        }
        if magnitude <= pivot_tolerance {
            return Err(VortexLatticeError::SingularSystem {
                pivot: column,
                magnitude,
            });
        }
        if pivot != column {
            matrix.swap(pivot, column);
            rhs.swap(pivot, column);
        }
        for row in (column + 1)..n {
            let factor = matrix[row][column] / matrix[column][column];
            if !factor.is_finite() {
                return Err(VortexLatticeError::NonFiniteSystem { row, column });
            }
            let (above, below) = matrix.split_at_mut(row);
            let pivot_row = &above[column];
            let current_row = &mut below[0];
            current_row[column] = 0.0;
            for (current, pivot_value) in current_row[(column + 1)..]
                .iter_mut()
                .zip(&pivot_row[(column + 1)..])
            {
                *current -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let mut value = rhs[row];
        for column in (row + 1)..n {
            value -= matrix[row][column] * solution[column];
        }
        solution[row] = value / matrix[row][row];
        if !solution[row].is_finite() {
            return Err(VortexLatticeError::NonFiniteSystem { row, column: row });
        }
    }
    Ok(solution)
}

fn finite_point(point: Point3) -> bool {
    point.x.is_finite() && point.y.is_finite() && point.z.is_finite()
}

fn zero_point() -> Point3 {
    Point3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    }
}

fn add(left: Point3, right: Point3) -> Point3 {
    Point3 {
        x: left.x + right.x,
        y: left.y + right.y,
        z: left.z + right.z,
    }
}

fn sub(left: Point3, right: Point3) -> Point3 {
    Point3 {
        x: left.x - right.x,
        y: left.y - right.y,
        z: left.z - right.z,
    }
}

fn scale(point: Point3, factor: f64) -> Point3 {
    Point3 {
        x: point.x * factor,
        y: point.y * factor,
        z: point.z * factor,
    }
}

fn dot(left: Point3, right: Point3) -> f64 {
    left.x * right.x + left.y * right.y + left.z * right.z
}

fn cross(left: Point3, right: Point3) -> Point3 {
    Point3 {
        x: left.y * right.z - left.z * right.y,
        y: left.z * right.x - left.x * right.z,
        z: left.x * right.y - left.y * right.x,
    }
}

fn norm(point: Point3) -> f64 {
    dot(point, point).sqrt()
}

fn normalize(point: Point3) -> Option<Point3> {
    let length = norm(point);
    if length.is_finite() && length > MIN_LENGTH_M {
        Some(scale(point, 1.0 / length))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panel(root_x: f64, tip_x: f64, chord_m: f64) -> BladePanel {
        BladePanel {
            vertices: [
                Point3 {
                    x: root_x,
                    y: chord_m * 0.5,
                    z: 0.0,
                },
                Point3 {
                    x: tip_x,
                    y: chord_m * 0.5,
                    z: 0.0,
                },
                Point3 {
                    x: tip_x,
                    y: -chord_m * 0.5,
                    z: 0.0,
                },
                Point3 {
                    x: root_x,
                    y: -chord_m * 0.5,
                    z: 0.0,
                },
            ],
            blade_index: 0,
            radius_fraction: 0.5,
            chord_fraction: chord_m,
            area: (tip_x - root_x) * chord_m,
        }
    }

    fn options() -> VortexLatticeOptions {
        VortexLatticeOptions {
            freestream_m_s: Point3 {
                x: 0.0,
                y: 0.0,
                z: 5.0,
            },
            angular_velocity_rad_s: 120.0,
            wake_length_m: 0.6,
            ..VortexLatticeOptions::default()
        }
    }

    #[test]
    fn rotating_panel_has_finite_load_and_normal_closure() {
        let result = evaluate_vortex_lattice_reference(&[panel(0.05, 0.10, 0.02)], options())
            .expect("single rotating panel should solve");
        assert!(result.is_finite());
        assert_eq!(result.matrix_size, 1);
        assert!(result.panels[0].circulation_m2_s.abs() > 0.0);
        assert!(result.total_force_n.z.abs() > 0.0);
        assert!(result.maximum_normal_residual_m_s < 1.0e-10);
    }

    #[test]
    fn circulation_scales_linearly_and_load_scales_quadratically() {
        let first = evaluate_vortex_lattice_reference(&[panel(0.05, 0.10, 0.02)], options())
            .expect("first solve");
        let second = evaluate_vortex_lattice_reference(
            &[panel(0.05, 0.10, 0.02)],
            VortexLatticeOptions {
                freestream_m_s: scale(options().freestream_m_s, 2.0),
                angular_velocity_rad_s: options().angular_velocity_rad_s * 2.0,
                ..options()
            },
        )
        .expect("second solve");
        assert!(
            (second.panels[0].circulation_m2_s / first.panels[0].circulation_m2_s - 2.0).abs()
                < 1.0e-12
        );
        assert!((second.total_force_n.z / first.total_force_n.z - 4.0).abs() < 1.0e-10);
    }

    #[test]
    fn wake_segmentation_preserves_finite_deterministic_replay() {
        let first = evaluate_vortex_lattice_reference(&[panel(0.05, 0.10, 0.02)], options())
            .expect("first solve");
        let second = evaluate_vortex_lattice_reference(&[panel(0.05, 0.10, 0.02)], options())
            .expect("second solve");
        assert_eq!(first, second);
        assert_eq!(first.wake_segment_count_per_leg, 8);
    }

    #[test]
    fn generated_panel_mesh_has_finite_vlm_reference() {
        let source =
            sonium_fem::physics::acoustics::aeroacoustics::PropellerSource::drone_5inch(12_000.0);
        let design = crate::source_model::PropellerDesign::conventional(source);
        let config = crate::blade3d::Blade3dConfig {
            radial_segments: 4,
            chord_segments: 2,
            loop_segments: 4,
            ..crate::blade3d::Blade3dConfig::default()
        };
        let mesh = config.try_build_mesh(&design).expect("mesh should build");
        let result = evaluate_vortex_lattice_reference(
            &mesh.panels,
            VortexLatticeOptions {
                freestream_m_s: Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 8.0,
                },
                ..VortexLatticeOptions::default()
            },
        )
        .expect("generated panel mesh should solve");
        assert!(result.is_finite());
        assert_eq!(result.panels.len(), mesh.panels.len());
    }

    #[test]
    fn invalid_inputs_fail_closed_before_matrix_work() {
        assert!(matches!(
            evaluate_vortex_lattice_reference(&[], VortexLatticeOptions::default()),
            Err(VortexLatticeError::EmptyPanels)
        ));
        assert!(matches!(
            evaluate_vortex_lattice_reference(
                &[panel(0.05, 0.10, 0.02)],
                VortexLatticeOptions {
                    wake_segments: 65,
                    max_wake_segments: 64,
                    ..VortexLatticeOptions::default()
                }
            ),
            Err(VortexLatticeError::InvalidOption {
                field: "wake_segments",
                ..
            })
        ));
        assert!(matches!(
            evaluate_vortex_lattice_reference(
                &[BladePanel {
                    vertices: [
                        Point3 {
                            x: 0.0,
                            y: 0.0,
                            z: 0.0
                        },
                        Point3 {
                            x: 1.0,
                            y: 0.0,
                            z: 0.0
                        },
                        Point3 {
                            x: 2.0,
                            y: 0.0,
                            z: 0.0
                        },
                        Point3 {
                            x: 3.0,
                            y: 0.0,
                            z: 0.0
                        },
                    ],
                    blade_index: 0,
                    radius_fraction: 0.0,
                    chord_fraction: 0.0,
                    area: 0.0,
                }],
                VortexLatticeOptions::default()
            ),
            Err(VortexLatticeError::InvalidPanel { field: "area", .. })
        ));
    }

    #[test]
    fn non_finite_options_are_rejected() {
        assert!(matches!(
            evaluate_vortex_lattice_reference(
                &[panel(0.05, 0.10, 0.02)],
                VortexLatticeOptions {
                    density_kg_m3: f64::NAN,
                    ..VortexLatticeOptions::default()
                }
            ),
            Err(VortexLatticeError::InvalidOption {
                field: "density_kg_m3",
                ..
            })
        ));
        assert!(matches!(
            evaluate_vortex_lattice_reference(
                &[panel(0.05, 0.10, 0.02)],
                VortexLatticeOptions {
                    wake_direction: Point3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    ..VortexLatticeOptions::default()
                }
            ),
            Err(VortexLatticeError::InvalidOption {
                field: "wake_direction",
                ..
            })
        ));
    }
}
