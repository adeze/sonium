//! Conservative BEM-to-3-D-panel load projection reference.
//!
//! This module maps checked radial blade-element loads onto the actual panel
//! geometry produced by [`crate::Blade3dConfig`]. It preserves the integrated
//! thrust and torque of the supplied BEM result after multiplying one blade by
//! the declared blade count, and retains panel force vectors and closure
//! residuals for VTU/gallery consumers.
//!
//! The projection is not a vortex-lattice, panel-flow, viscous, or resolved
//! toroidal aerodynamics solver. It makes that boundary explicit while
//! connecting deterministic 3-D conventional/toroidal geometry to an auditable
//! load ledger.

use std::fmt;

use crate::{BladeElementBemResult, BladeMesh, Point3};

const MIN_AXIS_NORM: f64 = 1.0e-12;
const MIN_PANEL_RADIUS_M: f64 = 1.0e-12;
const MIN_RAW_LOAD: f64 = 1.0e-14;

/// Geometry and multiplicity settings for BEM-to-panel load projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanelLoadProjectionOptions {
    /// Hub/shaft origin in the panel mesh frame [m].
    pub origin_m: [f64; 3],
    /// Rotor axis. The evaluator normalizes this vector.
    pub axis: [f64; 3],
    /// Hub radius used to map panel radius to BEM stations [m].
    pub hub_radius_m: f64,
    /// Tip radius used to map panel radius to BEM stations [m].
    pub tip_radius_m: f64,
    /// Number of identical blades represented by the mesh.
    pub blade_count: usize,
}

impl Default for PanelLoadProjectionOptions {
    fn default() -> Self {
        Self {
            origin_m: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 1.0],
            hub_radius_m: 0.009,
            tip_radius_m: 0.063,
            blade_count: 2,
        }
    }
}

/// One panel's projected aerodynamic force and torque evidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanelLoadProjection {
    /// Panel centre in the mesh frame [m].
    pub center_m: Point3,
    /// Panel area [m²].
    pub area_m2: f64,
    /// Panel radial distance from the declared rotor axis [m].
    pub radius_m: f64,
    /// Axial force carried by this panel on one blade [N].
    pub axial_force_n: f64,
    /// Tangential force carried by this panel on one blade [N].
    pub tangential_force_n: f64,
    /// Torque carried by this panel on one blade [N m].
    pub torque_nm: f64,
    /// Force vector on one blade in the mesh frame [N].
    pub force_n: [f64; 3],
}

/// Complete panel-load ledger with all-blade closure evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct PanelLoadProjectionResult {
    /// Projected panels in mesh order.
    pub panels: Vec<PanelLoadProjection>,
    /// Declared blade multiplicity.
    pub blade_count: usize,
    /// Reconstructed all-blade thrust [N].
    pub total_thrust_n: f64,
    /// Reconstructed all-blade torque [N m].
    pub total_torque_nm: f64,
    /// Reconstructed thrust minus supplied BEM thrust [N].
    pub thrust_closure_error_n: f64,
    /// Reconstructed torque minus supplied BEM torque [N m].
    pub torque_closure_error_nm: f64,
    /// Whether both closure residuals satisfy floating-point ledger tolerance.
    pub closure_passed: bool,
}

impl PanelLoadProjectionResult {
    /// Returns whether panel and closure evidence is finite and consistent.
    pub fn is_finite(&self) -> bool {
        self.blade_count > 0
            && !self.panels.is_empty()
            && self.total_thrust_n.is_finite()
            && self.total_thrust_n > 0.0
            && self.total_torque_nm.is_finite()
            && self.total_torque_nm > 0.0
            && self.thrust_closure_error_n.is_finite()
            && self.torque_closure_error_nm.is_finite()
            && self.panels.iter().all(|panel| {
                panel.center_m.x.is_finite()
                    && panel.center_m.y.is_finite()
                    && panel.center_m.z.is_finite()
                    && panel.area_m2.is_finite()
                    && panel.area_m2 > 0.0
                    && panel.radius_m.is_finite()
                    && panel.radius_m > 0.0
                    && panel.axial_force_n.is_finite()
                    && panel.tangential_force_n.is_finite()
                    && panel.torque_nm.is_finite()
                    && panel.force_n.iter().all(|value| value.is_finite())
            })
    }
}

/// Failure-closed errors for panel load projection.
#[derive(Debug, Clone, PartialEq)]
pub enum PanelLoadProjectionError {
    /// A projection option is outside its physical domain.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The mesh contains no usable panel geometry.
    EmptyMesh,
    /// A panel has invalid geometry or cannot be mapped to the rotor axis.
    InvalidPanel {
        /// Panel index in mesh order.
        panel_index: usize,
        /// Invalid panel field.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The supplied BEM result has invalid integrated or station data.
    InvalidBem {
        /// Invalid BEM field.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The interpolated raw axial panel weights cannot represent thrust.
    DegenerateAxialWeights,
    /// The interpolated raw tangential torque weights cannot represent torque.
    DegenerateTangentialWeights,
}

impl fmt::Display for PanelLoadProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => write!(
                formatter,
                "invalid panel-load option {field}: {requirement}"
            ),
            Self::EmptyMesh => write!(formatter, "panel mesh must contain at least one panel"),
            Self::InvalidPanel {
                panel_index,
                field,
                requirement,
            } => write!(
                formatter,
                "invalid panel {panel_index} field {field}: {requirement}"
            ),
            Self::InvalidBem { field, requirement } => {
                write!(formatter, "invalid BEM field {field}: {requirement}")
            }
            Self::DegenerateAxialWeights => {
                write!(formatter, "panel axial load weights are degenerate")
            }
            Self::DegenerateTangentialWeights => {
                write!(formatter, "panel tangential torque weights are degenerate")
            }
        }
    }
}

impl std::error::Error for PanelLoadProjectionError {}

/// Project checked BEM strip loads onto a generated 3-D panel mesh.
pub fn project_bem_loads_to_panel_mesh(
    mesh: &BladeMesh,
    bem: &BladeElementBemResult,
    options: PanelLoadProjectionOptions,
) -> Result<PanelLoadProjectionResult, PanelLoadProjectionError> {
    validate_options(&options)?;
    if mesh.panels.is_empty() {
        return Err(PanelLoadProjectionError::EmptyMesh);
    }
    validate_bem(bem)?;
    let axis = normalize(options.axis);
    let strips = &bem.loads.strips;
    let mut projected = Vec::with_capacity(mesh.panels.len());
    let mut raw_axial_sum = 0.0;
    let mut raw_torque_sum = 0.0;

    for (panel_index, panel) in mesh.panels.iter().enumerate() {
        let center = centroid(panel.vertices);
        let radius_vector = sub(
            [center.x, center.y, center.z],
            add(
                options.origin_m,
                scale(
                    axis,
                    dot(sub([center.x, center.y, center.z], options.origin_m), axis),
                ),
            ),
        );
        let radius_m = norm(radius_vector);
        if !center.x.is_finite()
            || !center.y.is_finite()
            || !center.z.is_finite()
            || !panel.area.is_finite()
            || panel.area <= 0.0
        {
            return Err(PanelLoadProjectionError::InvalidPanel {
                panel_index,
                field: "vertices/area",
                requirement: "finite vertices and area > 0",
            });
        }
        if !radius_m.is_finite() || radius_m <= MIN_PANEL_RADIUS_M {
            return Err(PanelLoadProjectionError::InvalidPanel {
                panel_index,
                field: "radius",
                requirement: "finite and > 1e-12 m",
            });
        }
        let radius_fraction = ((radius_m - options.hub_radius_m)
            / (options.tip_radius_m - options.hub_radius_m))
            .clamp(0.0, 1.0);
        let (chord_m, axial_force_per_m_n, tangential_force_per_m_n) = interpolate_strip(
            strips,
            options.hub_radius_m + radius_fraction * (options.tip_radius_m - options.hub_radius_m),
        );
        let raw_axial_force_n = axial_force_per_m_n * panel.area / chord_m;
        let raw_tangential_force_n = tangential_force_per_m_n * panel.area / chord_m;
        raw_axial_sum += raw_axial_force_n;
        raw_torque_sum += raw_tangential_force_n * radius_m;
        projected.push((
            center,
            panel.area,
            radius_m,
            raw_axial_force_n,
            raw_tangential_force_n,
        ));
    }

    let target_thrust_per_blade_n = bem.loads.thrust_n / options.blade_count as f64;
    let target_torque_per_blade_nm = bem.loads.torque_nm / options.blade_count as f64;
    if raw_axial_sum.abs() <= MIN_RAW_LOAD {
        return Err(PanelLoadProjectionError::DegenerateAxialWeights);
    }
    if raw_torque_sum.abs() <= MIN_RAW_LOAD {
        return Err(PanelLoadProjectionError::DegenerateTangentialWeights);
    }
    let axial_scale = target_thrust_per_blade_n / raw_axial_sum;
    let tangential_scale = target_torque_per_blade_nm / raw_torque_sum;
    let panels = projected
        .into_iter()
        .map(
            |(center, area_m2, radius_m, raw_axial_force_n, raw_tangential_force_n)| {
                let axial_force_n = raw_axial_force_n * axial_scale;
                let tangential_force_n = raw_tangential_force_n * tangential_scale;
                let radial_direction = normalize(sub(
                    [center.x, center.y, center.z],
                    add(
                        options.origin_m,
                        scale(
                            axis,
                            dot(sub([center.x, center.y, center.z], options.origin_m), axis),
                        ),
                    ),
                ));
                let tangent_direction = normalize(cross(axis, radial_direction));
                let force = add(
                    scale(axis, axial_force_n),
                    scale(tangent_direction, tangential_force_n),
                );
                PanelLoadProjection {
                    center_m: center,
                    area_m2,
                    radius_m,
                    axial_force_n,
                    tangential_force_n,
                    torque_nm: tangential_force_n * radius_m,
                    force_n: force,
                }
            },
        )
        .collect::<Vec<_>>();
    let total_thrust_n =
        options.blade_count as f64 * panels.iter().map(|panel| panel.axial_force_n).sum::<f64>();
    let total_torque_nm =
        options.blade_count as f64 * panels.iter().map(|panel| panel.torque_nm).sum::<f64>();
    let thrust_closure_error_n = total_thrust_n - bem.loads.thrust_n;
    let torque_closure_error_nm = total_torque_nm - bem.loads.torque_nm;
    let closure_passed = thrust_closure_error_n.abs()
        <= 1.0e-10 * bem.loads.thrust_n.abs().max(1.0)
        && torque_closure_error_nm.abs() <= 1.0e-10 * bem.loads.torque_nm.abs().max(1.0);
    let result = PanelLoadProjectionResult {
        panels,
        blade_count: options.blade_count,
        total_thrust_n,
        total_torque_nm,
        thrust_closure_error_n,
        torque_closure_error_nm,
        closure_passed,
    };
    debug_assert!(result.is_finite());
    Ok(result)
}

fn validate_options(options: &PanelLoadProjectionOptions) -> Result<(), PanelLoadProjectionError> {
    if options.origin_m.iter().any(|value| !value.is_finite()) {
        return Err(PanelLoadProjectionError::InvalidOptions {
            field: "origin_m",
            requirement: "all components finite",
        });
    }
    if options.axis.iter().any(|value| !value.is_finite()) || norm(options.axis) <= MIN_AXIS_NORM {
        return Err(PanelLoadProjectionError::InvalidOptions {
            field: "axis",
            requirement: "finite norm > 1e-12",
        });
    }
    if !options.hub_radius_m.is_finite()
        || !options.tip_radius_m.is_finite()
        || options.hub_radius_m < 0.0
        || options.tip_radius_m <= options.hub_radius_m
    {
        return Err(PanelLoadProjectionError::InvalidOptions {
            field: "hub_radius_m/tip_radius_m",
            requirement: "finite and 0 <= hub < tip",
        });
    }
    if options.blade_count == 0 {
        return Err(PanelLoadProjectionError::InvalidOptions {
            field: "blade_count",
            requirement: "> 0",
        });
    }
    Ok(())
}

fn validate_bem(bem: &BladeElementBemResult) -> Result<(), PanelLoadProjectionError> {
    if !bem.converged {
        return Err(PanelLoadProjectionError::InvalidBem {
            field: "converged",
            requirement: "true",
        });
    }
    if !bem.loads.thrust_n.is_finite() || bem.loads.thrust_n <= 0.0 {
        return Err(PanelLoadProjectionError::InvalidBem {
            field: "loads.thrust_n",
            requirement: "finite and > 0",
        });
    }
    if !bem.loads.torque_nm.is_finite() || bem.loads.torque_nm <= 0.0 {
        return Err(PanelLoadProjectionError::InvalidBem {
            field: "loads.torque_nm",
            requirement: "finite and > 0",
        });
    }
    if bem.loads.strips.len() < 2 {
        return Err(PanelLoadProjectionError::InvalidBem {
            field: "loads.strips",
            requirement: "at least two stations",
        });
    }
    if bem.loads.strips.iter().any(|strip| {
        !strip.radius_m.is_finite()
            || !strip.chord_m.is_finite()
            || strip.chord_m <= 0.0
            || !strip.axial_force_per_m_n.is_finite()
            || !strip.tangential_force_per_m_n.is_finite()
    }) {
        return Err(PanelLoadProjectionError::InvalidBem {
            field: "loads.strips",
            requirement: "finite positive chord and finite force densities",
        });
    }
    if bem
        .loads
        .strips
        .windows(2)
        .any(|pair| pair[1].radius_m <= pair[0].radius_m)
    {
        return Err(PanelLoadProjectionError::InvalidBem {
            field: "loads.strips.radius_m",
            requirement: "strictly increasing",
        });
    }
    Ok(())
}

fn interpolate_strip(strips: &[crate::BladeElementStripLoad], radius_m: f64) -> (f64, f64, f64) {
    if radius_m <= strips[0].radius_m {
        return (
            strips[0].chord_m,
            strips[0].axial_force_per_m_n,
            strips[0].tangential_force_per_m_n,
        );
    }
    for pair in strips.windows(2) {
        if radius_m <= pair[1].radius_m {
            let fraction = (radius_m - pair[0].radius_m) / (pair[1].radius_m - pair[0].radius_m);
            return (
                lerp(pair[0].chord_m, pair[1].chord_m, fraction),
                lerp(
                    pair[0].axial_force_per_m_n,
                    pair[1].axial_force_per_m_n,
                    fraction,
                ),
                lerp(
                    pair[0].tangential_force_per_m_n,
                    pair[1].tangential_force_per_m_n,
                    fraction,
                ),
            );
        }
    }
    let last = strips.last().expect("validated non-empty strips");
    (
        last.chord_m,
        last.axial_force_per_m_n,
        last.tangential_force_per_m_n,
    )
}

fn lerp(lower: f64, upper: f64, fraction: f64) -> f64 {
    lower + fraction * (upper - lower)
}

fn centroid(vertices: [Point3; 4]) -> Point3 {
    Point3 {
        x: vertices.iter().map(|point| point.x).sum::<f64>() / 4.0,
        y: vertices.iter().map(|point| point.y).sum::<f64>() / 4.0,
        z: vertices.iter().map(|point| point.z).sum::<f64>() / 4.0,
    }
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0].mul_add(right[0], left[1].mul_add(right[1], left[2] * right[2]))
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

fn normalize(vector: [f64; 3]) -> [f64; 3] {
    scale(vector, 1.0 / norm(vector))
}
