//! Driver-motor FEM → lumped parameter extraction pipeline.
//!
//! Automates the extraction of displacement-dependent force factor BL(x) and
//! frequency-dependent inductance Le(f) from a parametric motor geometry.
//! BL uses an axisymmetric magnetostatic FEM solve; Le(f) and Re(f) use the
//! bias-linearized axisymmetric eddy-current FEM ([`AxisymmetricEddyCurrentProblem`]).
//!
//! # Pipeline
//!
//! 1. Define motor geometry ([`MotorGeometry`])
//! 2. Generate 2D axisymmetric mesh ([`generate_motor_mesh`])
//! 3. Sweep coil displacement → magnetostatic solve → BL(x) curve
//! 4. Solve the axisymmetric eddy-current problem → Le(f), Re(f) curves
//! 5. Output [`MotorParameters`] for the modeling crate's typed driver handoff

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;

use crate::nonlinear_motor::{
    EddyCurrentLossModel, HysteresisLossModel, MotorMagneticLaw, NonlinearLePoint,
};

#[cfg(test)]
use num_complex::Complex64;

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use sonium_fem::mesh::{BoundaryType, ElementType, Mesh, Point};
use sonium_fem::physics::electromagnetics::eddy_current::{
    AxisymmetricEddyCurrentProblem, DifferentialReluctivityField,
};
use sonium_fem::physics::electromagnetics::magnetostatics::{
    AxisymmetricNonlinearMagneticProblem, MagneticConstitutiveLaw, NonlinearBiasSolveConfig,
    solve_nonlinear_axisymmetric_bias,
};
use sonium_fem::physics::electromagnetics::magnetostatics::{
    MagnetSourceForm, MagneticMaterial, PermeabilityForm, compute_b_field,
};
use sonium_fem::quadrature::gauss_triangle;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::traits::LinearOperator;

// ---------------------------------------------------------------------------
// Motor geometry definition
// ---------------------------------------------------------------------------

/// Magnet type for the motor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MagnetType {
    /// NdFeB (neodymium). Typical Br: 1.0–1.4 T, mu_r ≈ 1.05
    Neodymium { br: f64 },
    /// Ferrite. Typical Br: 0.3–0.5 T, mu_r ≈ 1.1
    Ferrite { br: f64 },
}

impl MagnetType {
    /// Fractional BL change per kelvin used by the M3 thermal handoff.
    pub const fn bl_temperature_coefficient_per_k(&self) -> f64 {
        match self {
            Self::Neodymium { .. } => -1.0e-3,
            Self::Ferrite { .. } => -2.0e-3,
        }
    }
}

/// Conductive non-magnetic insert on the center pole.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConductiveInsertSpec {
    /// Inner radius (m), normally zero for a cap and positive for a ring.
    pub inner_radius_m: f64,
    /// Outer radius (m), bounded by the center-pole radius.
    pub outer_radius_m: f64,
    /// Axial lower face (m), measured in the motor mesh's z coordinate.
    pub z_bottom_m: f64,
    /// Axial upper face (m).
    pub z_top_m: f64,
    /// Electrical conductivity (S/m); copper is approximately 5.8e7.
    pub conductivity_s_per_m: f64,
}

impl ConductiveInsertSpec {
    fn validate(self, pole: &PolePieceSpec) -> Result<(), MotorExtractionError> {
        if !self.inner_radius_m.is_finite() || self.inner_radius_m < 0.0 {
            return Err(MotorExtractionError::new(
                "conductive insert inner radius must be finite and non-negative",
            ));
        }
        validate_positive_finite("conductive insert outer radius", self.outer_radius_m)?;
        validate_positive_finite("conductive insert conductivity", self.conductivity_s_per_m)?;
        if self.inner_radius_m >= self.outer_radius_m || self.outer_radius_m > pole.radius {
            return Err(MotorExtractionError::new(
                "conductive insert radii must be ordered inside the center pole",
            ));
        }
        if !self.z_bottom_m.is_finite()
            || !self.z_top_m.is_finite()
            || self.z_bottom_m < 0.0
            || self.z_bottom_m >= self.z_top_m
            || self.z_top_m > pole.height
        {
            return Err(MotorExtractionError::new(
                "conductive insert axial bounds must be ordered inside the center pole",
            ));
        }
        Ok(())
    }
}

/// Pole piece material specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolePieceSpec {
    /// Relative permeability of the steel.
    pub mu_r: f64,
    /// Electrical conductivity (S/m) used by the eddy-loss approximation.
    pub conductivity: f64,
    /// Outer radius of the pole piece (m).
    pub radius: f64,
    /// Height of the pole piece (m).
    pub height: f64,
}

impl Default for PolePieceSpec {
    fn default() -> Self {
        Self {
            mu_r: 2000.0,
            conductivity: 6.99e6, // low-carbon steel
            radius: 0.012,
            height: 0.006,
        }
    }
}

/// Voice coil specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoilSpec {
    /// Former radius (center of winding, m).
    pub radius: f64,
    /// Winding height (m).
    pub height: f64,
    /// Number of turns.
    pub n_turns: usize,
    /// Wire diameter (m).
    pub wire_diameter: f64,
    /// Number of layers.
    pub layers: usize,
}

impl CoilSpec {
    /// Winding thickness = layers × wire_diameter.
    pub fn winding_thickness(&self) -> f64 {
        self.layers as f64 * self.wire_diameter
    }

    /// Inner radius of the winding.
    pub fn inner_radius(&self) -> f64 {
        self.radius - 0.5 * self.winding_thickness()
    }

    /// Outer radius of the winding.
    pub fn outer_radius(&self) -> f64 {
        self.radius + 0.5 * self.winding_thickness()
    }
}

/// Complete parametric motor geometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorGeometry {
    /// Magnet specification.
    pub magnet: MagnetSpec,
    /// Pole piece (top plate + center pole).
    pub pole_piece: PolePieceSpec,
    /// Voice coil.
    pub coil: CoilSpec,
    /// Axial clearance (m) between the pole-piece top and magnet bottom.
    /// The voice coil is centered on this clearance at rest.
    pub air_gap: f64,
    #[serde(default)]
    pub coil_displacement_m: f64,
    /// Optional shorting ring or copper cap embedded in the center pole.
    #[serde(default)]
    pub conductive_insert: Option<ConductiveInsertSpec>,
}

/// Magnet geometry and material.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagnetSpec {
    /// Magnet type (determines Br and mu_r).
    pub magnet_type: MagnetType,
    /// Inner radius (m) — forms the air gap.
    pub inner_radius: f64,
    /// Outer radius (m).
    pub outer_radius: f64,
    /// Height (m) — axial extent.
    pub height: f64,
}

impl MagnetSpec {
    fn material(&self) -> MagneticMaterial {
        match &self.magnet_type {
            // Magnetisation direction is axial (+z) for a standard ring magnet
            MagnetType::Neodymium { br } => MagneticMaterial::neodymium(*br, [0.0, 1.0, 0.0]),
            MagnetType::Ferrite { br } => MagneticMaterial::ferrite(*br, [0.0, 1.0, 0.0]),
        }
    }
}

impl MotorGeometry {
    /// Validate material properties and the radial/axial region ordering.
    pub fn validate(&self) -> Result<(), MotorExtractionError> {
        let br = match self.magnet.magnet_type {
            MagnetType::Neodymium { br } | MagnetType::Ferrite { br } => br,
        };
        validate_positive_finite("magnet remanence", br)?;
        validate_positive_finite("magnet inner radius", self.magnet.inner_radius)?;
        validate_positive_finite("magnet outer radius", self.magnet.outer_radius)?;
        validate_positive_finite("magnet height", self.magnet.height)?;
        if self.magnet.inner_radius >= self.magnet.outer_radius {
            return Err(MotorExtractionError::new(
                "magnet inner radius must be smaller than outer radius",
            ));
        }

        validate_positive_finite("pole-piece relative permeability", self.pole_piece.mu_r)?;
        validate_positive_finite("pole-piece conductivity", self.pole_piece.conductivity)?;
        validate_positive_finite("pole-piece radius", self.pole_piece.radius)?;
        validate_positive_finite("pole-piece height", self.pole_piece.height)?;

        validate_positive_finite("coil radius", self.coil.radius)?;
        validate_positive_finite("coil height", self.coil.height)?;
        validate_positive_finite("coil wire diameter", self.coil.wire_diameter)?;
        if self.coil.n_turns == 0 {
            return Err(MotorExtractionError::new("coil turns must be at least 1"));
        }
        if self.coil.layers == 0 {
            return Err(MotorExtractionError::new("coil layers must be at least 1"));
        }
        if self.coil.inner_radius() <= self.pole_piece.radius {
            return Err(MotorExtractionError::new(
                "coil inner radius must lie outside the pole piece",
            ));
        }
        if self.coil.outer_radius() >= self.magnet.inner_radius {
            return Err(MotorExtractionError::new(
                "coil outer radius must lie inside the magnet inner radius",
            ));
        }
        validate_positive_finite("axial air gap", self.air_gap)?;
        if !self.coil_displacement_m.is_finite() {
            return Err(MotorExtractionError::new(
                "coil displacement must be finite",
            ));
        }
        if let Some(insert) = self.conductive_insert {
            insert.validate(&self.pole_piece)?;
        }
        Ok(())
    }
}

/// Recoverable motor extraction error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MotorExtractionError {
    message: String,
}

impl MotorExtractionError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for MotorExtractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for MotorExtractionError {}

fn validate_positive_finite(name: &str, value: f64) -> Result<(), MotorExtractionError> {
    if !value.is_finite() || value <= 0.0 {
        Err(MotorExtractionError::new(format!(
            "{name} must be finite and positive, got {value}"
        )))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Extracted motor parameters
// ---------------------------------------------------------------------------

/// Boxed function type for voice coil parameter closures.
pub type VoiceCoilFn = Box<dyn Fn(f64) -> f64 + Send + Sync>;

/// A curve of (x, y) data points.
pub type Curve = Vec<(f64, f64)>;

/// Extracted motor parameters from magnetostatic FEM plus the axisymmetric
/// eddy-current FEM Le(f)/Re(f) path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorParameters {
    /// BL(x) curve: (displacement_m, force_factor_Tm).
    /// Displacement is measured from the rest position.
    pub bl_curve: Vec<(f64, f64)>,

    /// Le(f) at rest position: (frequency_Hz, inductance_H).
    pub le_vs_frequency: Vec<(f64, f64)>,

    /// Effective resistance due to eddy currents: (frequency_Hz, resistance_Ohm).
    /// This is added to the DC resistance Re.
    pub re_eddy_vs_frequency: Vec<(f64, f64)>,

    /// BL at rest position (BL(0)).
    pub bl_nominal: f64,

    /// Le at low frequency (quasi-static).
    pub le_nominal: f64,

    /// Displacement/frequency-indexed differential inductance and eddy-loss
    /// samples from displaced-coil eddy-current solves.
    #[serde(default)]
    pub le_surface: Vec<NonlinearLePoint>,
    /// Fractional force-factor change per kelvin from the magnet material.
    #[serde(default)]
    pub bl_temperature_coefficient_per_k: f64,
    /// Optional validated nonlinear magnetic law retained from FEM bias.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magnetic_law: Option<MotorMagneticLaw>,
    /// Optional calibrated hysteresis/core-loss law retained for driver
    /// promotion. Legacy extraction artifacts leave this absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hysteresis_loss: Option<HysteresisLossModel>,
    /// Optional calibrated eddy-current/skin/proximity surface retained for
    /// nonlinear driver promotion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eddy_current_loss: Option<EddyCurrentLossModel>,
}

pub const MOTOR_COMPLETION_SCHEMA_VERSION: u32 = 1;

/// Versioned production request for the M3 extraction pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorCompletionInput {
    pub schema_version: u32,
    pub geometry: MotorGeometry,
    pub extraction: ExtractionConfig,
}

/// Versioned M3 evidence artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorCompletionResult {
    pub schema_version: u32,
    pub provenance: String,
    pub maximum_current_a: f64,
    pub conductive_insert_present: bool,
    pub magnet_bl_temperature_coefficient_per_k: f64,
    pub parameters: MotorParameters,
}

pub fn run_motor_completion(
    input: &MotorCompletionInput,
) -> Result<MotorCompletionResult, MotorExtractionError> {
    if input.schema_version != MOTOR_COMPLETION_SCHEMA_VERSION {
        return Err(MotorExtractionError::new(
            "unsupported motor-completion schema version",
        ));
    }
    let maximum_current_a = input.extraction.maximum_current_a()?;
    let parameters = extract_motor_parameters(&input.geometry, &input.extraction)?;
    Ok(MotorCompletionResult {
        schema_version: MOTOR_COMPLETION_SCHEMA_VERSION,
        provenance: "AxisymmetricNonlinearBias->DifferentialReluctivity->EddyCurrentLe(x,|i|,f)"
            .into(),
        maximum_current_a,
        conductive_insert_present: input.geometry.conductive_insert.is_some(),
        magnet_bl_temperature_coefficient_per_k: parameters.bl_temperature_coefficient_per_k,
        parameters,
    })
}

impl MotorParameters {
    /// Create interpolating closures suitable for `VoiceCoilParams::nonlinear`.
    ///
    /// Returns `(bl_fn, le_fn, dle_fn)` where the inductance callbacks use
    /// the lowest-frequency, zero-current `Le(x)` slice when it is available.
    /// The legacy callback API has no current/frequency arguments; callers
    /// requiring the complete `Le(x, i, f)` law must use `nonlinear_motor`.
    pub fn to_voice_coil_fns(&self) -> (VoiceCoilFn, VoiceCoilFn, VoiceCoilFn) {
        let bl_curve = self.bl_curve.clone();
        let le_curve = self.quasi_static_le_curve();

        let bl_fn = Box::new(move |x: f64| interpolate_curve(&bl_curve, x))
            as Box<dyn Fn(f64) -> f64 + Send + Sync>;

        let le_curve_for_value = le_curve.clone();
        let le_fn = Box::new(move |x: f64| interpolate_curve(&le_curve_for_value, x))
            as Box<dyn Fn(f64) -> f64 + Send + Sync>;

        let dle_fn = Box::new(move |x: f64| interpolate_curve_derivative(&le_curve, x))
            as Box<dyn Fn(f64) -> f64 + Send + Sync>;

        (bl_fn, le_fn, dle_fn)
    }

    /// Extract the quasi-static `Le(x)` slice consumable by the legacy
    /// displacement-only voice-coil API. Prefer the lowest available
    /// zero-current frequency at each displacement and retain the historical
    /// constant nominal inductance when no usable surface is available.
    fn quasi_static_le_curve(&self) -> Vec<(f64, f64)> {
        let mut samples = self
            .le_surface
            .iter()
            .filter(|point| {
                point.current_a == 0.0
                    && point.displacement_m.is_finite()
                    && point.frequency_hz.is_finite()
                    && point.frequency_hz > 0.0
                    && point.inductance_h.is_finite()
                    && point.inductance_h > 0.0
            })
            .collect::<Vec<_>>();
        samples.sort_by(|left, right| {
            left.frequency_hz
                .total_cmp(&right.frequency_hz)
                .then_with(|| left.displacement_m.total_cmp(&right.displacement_m))
        });

        let mut curve = Vec::new();
        for point in samples {
            if !curve
                .iter()
                .any(|(displacement_m, _)| *displacement_m == point.displacement_m)
            {
                curve.push((point.displacement_m, point.inductance_h));
            }
        }
        curve.sort_by(|left, right| left.0.total_cmp(&right.0));
        if curve.is_empty() {
            vec![(0.0, self.le_nominal)]
        } else {
            curve
        }
    }
}

/// Linear interpolation on a (x, y) curve.
fn interpolate_curve(curve: &[(f64, f64)], x: f64) -> f64 {
    if curve.is_empty() {
        return 0.0;
    }
    if x <= curve[0].0 {
        return curve[0].1;
    }
    if x >= curve[curve.len() - 1].0 {
        return curve[curve.len() - 1].1;
    }
    // Binary search for the interval
    let idx = curve.partition_point(|&(xi, _)| xi < x);
    if idx == 0 {
        return curve[0].1;
    }
    let (x0, y0) = curve[idx - 1];
    let (x1, y1) = curve[idx];
    let t = (x - x0) / (x1 - x0);
    y0 + t * (y1 - y0)
}

/// Derivative of the clamped piecewise-linear curve returned by
/// [`interpolate_curve`]. Single-point curves deliberately have zero slope.
fn interpolate_curve_derivative(curve: &[(f64, f64)], x: f64) -> f64 {
    if curve.len() < 2 {
        return 0.0;
    }
    let upper = if x <= curve[0].0 {
        1
    } else if x >= curve[curve.len() - 1].0 {
        curve.len() - 1
    } else {
        curve.partition_point(|&(xi, _)| xi < x)
    };
    let lower = upper - 1;
    let dx = curve[upper].0 - curve[lower].0;
    if dx > 0.0 {
        (curve[upper].1 - curve[lower].1) / dx
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Axisymmetric motor mesh generation
// ---------------------------------------------------------------------------

/// Region markers for the motor mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotorRegion {
    Air = 0,
    Magnet = 1,
    PolePiece = 2,
    Coil = 3,
    ConductiveInsert = 4,
}

/// Generated motor mesh with region information.
pub struct MotorMesh {
    pub mesh: Mesh,
    /// Region tag for each element.
    pub element_regions: Vec<MotorRegion>,
    /// Coil element indices (for force/BL extraction).
    pub coil_elements: Vec<usize>,
}

/// Generate a 2D axisymmetric mesh of the loudspeaker motor.
///
/// The mesh is in the (r, z) plane where r ≥ 0 and z is the axial direction.
/// The geometry is a cross-section of the axisymmetric motor:
///
/// ```text
///     |<-- pole piece -->|<- gap ->|<-- magnet -->|
///     |                  |  coil   |              |
///  z  |  (steel)         | (wire)  |  (NdFeB)     |
///  ^  |                  |         |              |
///  |  +------------------+---------+--------------+--- r
///  +---> r
/// ```
///
/// The extraction configuration controls mesh resolution and the air padding
/// needed to keep the full displaced coil inside the computational domain.
pub fn generate_motor_mesh(
    geom: &MotorGeometry,
    config: &ExtractionConfig,
) -> Result<MotorMesh, MotorExtractionError> {
    geom.validate()?;
    config.validate()?;
    let n = config.mesh_resolution;
    let mut mesh = Mesh::new(2);

    // Key radial coordinates
    let r0 = 0.0; // axis of symmetry
    let r1 = geom.pole_piece.radius; // pole piece outer radius
    let r2 = geom.coil.inner_radius(); // coil inner radius
    let r3 = geom.coil.outer_radius(); // coil outer radius
    let r4 = geom.magnet.inner_radius; // magnet inner radius
    let r5 = geom.magnet.outer_radius; // magnet outer radius
    let radial_padding = r5.max(geom.magnet.height).max(geom.pole_piece.height) * 0.5;
    let r6 = r5 + radial_padding;

    // Key axial coordinates (z)
    let z0: f64 = 0.0;
    let z1 = geom.pole_piece.height; // top of pole piece
    let z2 = z1 + geom.air_gap; // bottom of magnet (above pole piece)
    let z_coil_center = z1 + 0.5 * geom.air_gap + geom.coil_displacement_m;
    let z_coil_bottom = z_coil_center - 0.5 * geom.coil.height;
    let z_coil_top = z_coil_center + 0.5 * geom.coil.height;
    let physical_bottom = z0.min(z_coil_bottom - config.x_max);
    let physical_top = (z2 + geom.magnet.height).max(z_coil_top + config.x_max);
    let axial_padding = (physical_top - physical_bottom).max(geom.magnet.outer_radius) * 0.5;
    let z_bottom = physical_bottom - axial_padding;
    let z_top = physical_top + axial_padding;

    // For simplicity, we tile the domain with a structured grid of quadrilaterals
    // split into triangles. We define radial and axial subdivisions.

    // Radial points: axis, pole_piece boundary, gap region, coil, magnet boundaries
    let mut radial_points = vec![r0, r1, r2, r3, r4, r5, r6];
    if let Some(insert) = geom.conductive_insert {
        radial_points.extend([insert.inner_radius_m, insert.outer_radius_m]);
    }
    let r_points = sorted_unique(&radial_points);
    // Axial points: bottom, pole piece top, coil top, magnet extent, domain top
    let mut axial_points = vec![
        z_bottom,
        z0,
        z_coil_bottom,
        z1,
        z_coil_top,
        z2,
        z2 + geom.magnet.height,
        z_top,
    ];
    if let Some(insert) = geom.conductive_insert {
        axial_points.extend([insert.z_bottom_m, insert.z_top_m]);
    }
    let z_points = sorted_unique(&axial_points);

    // Subdivide each strip into n segments
    let r_fine = subdivide_intervals(&r_points, n)?;
    let z_fine = subdivide_intervals(&z_points, n)?;

    let nr = r_fine.len();

    // Create nodes
    for &zv in &z_fine {
        for &rv in &r_fine {
            mesh.add_node(Point::new_2d(rv, zv));
        }
    }

    // Create triangles and assign regions
    let mut element_regions = Vec::new();
    let mut coil_elements = Vec::new();

    let bounds = RegionBounds {
        r_pole: r1,
        r_coil_in: r2,
        r_coil_out: r3,
        r_mag_in: r4,
        r_mag_out: r5,
        z_pole_bottom: z0,
        z_pole_top: z1,
        z_mag_bot: z2,
        z_mag_top: z2 + geom.magnet.height,
        z_coil_bottom,
        z_coil_top,
        conductive_insert: geom.conductive_insert,
    };

    for (j, z_pair) in z_fine.windows(2).enumerate() {
        for (i, r_pair) in r_fine.windows(2).enumerate() {
            let v0 = j * nr + i;
            let v1 = v0 + 1;
            let v2 = (j + 1) * nr + i;
            let v3 = v2 + 1;

            // Centroid for region classification
            let rc = 0.5 * (r_pair[0] + r_pair[1]);
            let zc = 0.5 * (z_pair[0] + z_pair[1]);
            let region = bounds.classify(rc, zc);

            let elem_idx = mesh.num_elements();
            mesh.add_element(ElementType::Triangle, vec![v0, v1, v2]);
            mesh.elements[elem_idx].marker = region as i32;
            element_regions.push(region);
            if region == MotorRegion::Coil {
                coil_elements.push(elem_idx);
            }

            let elem_idx = mesh.num_elements();
            mesh.add_element(ElementType::Triangle, vec![v1, v3, v2]);
            mesh.elements[elem_idx].marker = region as i32;
            element_regions.push(region);
            if region == MotorRegion::Coil {
                coil_elements.push(elem_idx);
            }
        }
    }

    mesh.detect_boundaries();
    // Mark only far truncation boundaries as Dirichlet. The r=0 boundary is
    // the symmetry axis and retains its natural zero-normal-flux condition.
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 1, |points| {
        points.iter().any(|point| point.x.abs() > 1e-12)
    });

    Ok(MotorMesh {
        mesh,
        element_regions,
        coil_elements,
    })
}

/// Cached region boundaries for fast element classification.
struct RegionBounds {
    r_pole: f64,
    r_coil_in: f64,
    r_coil_out: f64,
    r_mag_in: f64,
    r_mag_out: f64,
    z_pole_bottom: f64,
    z_pole_top: f64,
    z_mag_bot: f64,
    z_mag_top: f64,
    z_coil_bottom: f64,
    z_coil_top: f64,
    conductive_insert: Option<ConductiveInsertSpec>,
}

impl RegionBounds {
    fn classify(&self, r: f64, z: f64) -> MotorRegion {
        if self.conductive_insert.is_some_and(|insert| {
            r >= insert.inner_radius_m
                && r <= insert.outer_radius_m
                && z >= insert.z_bottom_m
                && z <= insert.z_top_m
        }) {
            return MotorRegion::ConductiveInsert;
        }
        // Pole piece: r < r_pole, z < z_pole_top
        if r <= self.r_pole && z >= self.z_pole_bottom && z <= self.z_pole_top {
            return MotorRegion::PolePiece;
        }
        // Magnet
        if r >= self.r_mag_in && r <= self.r_mag_out && z >= self.z_mag_bot && z <= self.z_mag_top {
            return MotorRegion::Magnet;
        }
        // Coil
        if r >= self.r_coil_in
            && r <= self.r_coil_out
            && z >= self.z_coil_bottom
            && z <= self.z_coil_top
        {
            return MotorRegion::Coil;
        }
        MotorRegion::Air
    }
}

/// Remove duplicates and sort a list of coordinates.
fn sorted_unique(values: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(f64::total_cmp);
    v.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    v
}

/// Subdivide each interval in `points` into `n` equal segments.
fn subdivide_intervals(points: &[f64], n: usize) -> Result<Vec<f64>, MotorExtractionError> {
    if points.len() < 2 {
        return Err(MotorExtractionError::new(
            "mesh subdivision requires at least two coordinates",
        ));
    }
    if n == 0 {
        return Err(MotorExtractionError::new(
            "mesh subdivision count must be at least 1",
        ));
    }
    let mut result = Vec::new();
    for i in 0..points.len() - 1 {
        let a = points[i];
        let b = points[i + 1];
        for k in 0..n {
            result.push(a + (b - a) * k as f64 / n as f64);
        }
    }
    let last = points
        .last()
        .copied()
        .ok_or_else(|| MotorExtractionError::new("mesh subdivision requires a final coordinate"))?;
    result.push(last);
    Ok(result)
}

// ---------------------------------------------------------------------------
// BL(x) extraction
// ---------------------------------------------------------------------------

/// Configuration for the extraction pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Displacement range: [-x_max, +x_max] in meters.
    pub x_max: f64,
    /// Number of displacement steps (total points = 2 * n_steps + 1).
    pub n_displacement_steps: usize,
    /// Mesh resolution (elements per region edge).
    pub mesh_resolution: usize,
    /// Frequencies for Le(f) extraction (Hz).
    pub le_frequencies: Vec<f64>,
    /// Rated RMS drive used to derive the current extent `Imax = V/Re`.
    pub rated_voltage_rms_v: f64,
    /// Cold DC coil resistance used only for the rated-current extent.
    pub coil_resistance_ohm: f64,
    /// Initial equal intervals on the non-negative `|i|` grid.
    pub initial_current_intervals: usize,
    /// Bounded midpoint-refinement passes.
    pub max_current_refinement_levels: usize,
    /// Refine when normalized `|dLe/di|` exceeds this fraction.
    pub current_slope_refinement_fraction: f64,
    /// Optional nonlinear B-H handoff retained in extraction evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magnetic_law: Option<MotorMagneticLaw>,
    /// Optional calibrated hysteresis/core-loss law retained in extraction
    /// evidence and copied into the driver handoff.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hysteresis_loss: Option<HysteresisLossModel>,
    /// Optional calibrated eddy-current/skin/proximity surface retained in
    /// extraction evidence and copied into the driver handoff.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eddy_current_loss: Option<EddyCurrentLossModel>,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            x_max: 0.010, // ±10 mm
            n_displacement_steps: 20,
            mesh_resolution: 4,
            le_frequencies: vec![
                20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0,
            ],
            rated_voltage_rms_v: 2.83,
            coil_resistance_ohm: 5.8,
            initial_current_intervals: 1,
            max_current_refinement_levels: 1,
            current_slope_refinement_fraction: 0.02,
            magnetic_law: None,
            hysteresis_loss: None,
            eddy_current_loss: None,
        }
    }
}

impl ExtractionConfig {
    /// Validate sweep extents, discretization, and frequency ordering.
    pub fn validate(&self) -> Result<(), MotorExtractionError> {
        validate_positive_finite("maximum coil displacement", self.x_max)?;
        if self.n_displacement_steps == 0 {
            return Err(MotorExtractionError::new(
                "n_displacement_steps must be at least 1",
            ));
        }
        if self.mesh_resolution == 0 {
            return Err(MotorExtractionError::new(
                "mesh_resolution must be at least 1",
            ));
        }
        if self.le_frequencies.is_empty() {
            return Err(MotorExtractionError::new(
                "at least one Le extraction frequency is required",
            ));
        }
        validate_positive_finite("rated RMS voltage", self.rated_voltage_rms_v)?;
        validate_positive_finite("coil resistance", self.coil_resistance_ohm)?;
        if self.initial_current_intervals == 0 {
            return Err(MotorExtractionError::new(
                "initial_current_intervals must be at least 1",
            ));
        }
        if self.max_current_refinement_levels > 8 {
            return Err(MotorExtractionError::new(
                "max_current_refinement_levels must not exceed 8",
            ));
        }
        validate_positive_finite(
            "current slope refinement fraction",
            self.current_slope_refinement_fraction,
        )?;
        let mut previous = None;
        for (index, &frequency) in self.le_frequencies.iter().enumerate() {
            validate_positive_finite(&format!("Le frequency at index {index}"), frequency)?;
            if previous.is_some_and(|value| frequency <= value) {
                return Err(MotorExtractionError::new(
                    "Le extraction frequencies must be strictly increasing",
                ));
            }
            previous = Some(frequency);
        }
        Ok(())
    }

    pub fn maximum_current_a(&self) -> Result<f64, MotorExtractionError> {
        self.validate()?;
        let current = self.rated_voltage_rms_v / self.coil_resistance_ohm;
        validate_positive_finite("maximum rated current", current)?;
        Ok(current)
    }
}

/// Run the full motor parameter extraction pipeline.
///
/// 1. Generates a motor mesh
/// 2. Solves magnetostatics at each coil displacement → BL(x)
/// 3. Solves the axisymmetric eddy-current problem → Le(f), Re_eddy(f)
pub fn extract_motor_parameters(
    geom: &MotorGeometry,
    config: &ExtractionConfig,
) -> Result<MotorParameters, MotorExtractionError> {
    geom.validate()?;
    config.validate()?;
    let motor_mesh = generate_motor_mesh(geom, config)?;

    // --- Phase 1: BL(x) extraction via magnetostatic solves ---
    let bl_curve = extract_bl_curve(geom, &motor_mesh, config)?;

    // --- Phase 2: Le(x,f)/Re_eddy(x,f) extraction ------------------------
    // Rebuild only the coil-positioned mesh for a compact displacement grid;
    // the magnetostatic BL sweep above remains the higher-resolution curve.
    let displacements = [
        -config.x_max,
        -0.5 * config.x_max,
        0.0,
        0.5 * config.x_max,
        config.x_max,
    ];
    // Each batch rebuilds the mesh and extracts the current surface
    // (milliseconds to seconds); indexed collect keeps displacement order
    // identical on both paths.
    let extract_batch = |displacement_m: f64| {
        let mut displaced_geometry = geom.clone();
        displaced_geometry.coil_displacement_m = displacement_m;
        let displaced_mesh = generate_motor_mesh(&displaced_geometry, config)?;
        let samples =
            extract_adaptive_current_surface(&displaced_geometry, &displaced_mesh, config)?;
        Ok((displacement_m, samples))
    };
    let displacement_batches: Vec<_> =
        if crate::parallelism::use_modeling_parallel_batch(displacements.len()) {
            displacements
                .into_par_iter()
                .map(&extract_batch)
                .collect::<Result<Vec<_>, MotorExtractionError>>()?
        } else {
            displacements
                .into_iter()
                .map(&extract_batch)
                .collect::<Result<Vec<_>, MotorExtractionError>>()?
        };

    let mut le_surface = Vec::new();
    let mut rest_curves = None;
    for (displacement_m, current_samples) in displacement_batches {
        for sample in current_samples {
            if displacement_m.abs() <= f64::EPSILON && sample.current_a == 0.0 {
                rest_curves = Some((sample.le_curve.clone(), sample.re_curve.clone()));
            }
            let signed_currents: &[f64] = if sample.current_a == 0.0 {
                &[0.0]
            } else {
                &[-sample.current_a, sample.current_a]
            };
            for &current_a in signed_currents {
                for ((frequency_hz, inductance_h), (_, eddy_resistance_ohm)) in
                    sample.le_curve.iter().zip(&sample.re_curve)
                {
                    le_surface.push(NonlinearLePoint {
                        displacement_m,
                        current_a,
                        frequency_hz: *frequency_hz,
                        inductance_h: *inductance_h,
                        eddy_resistance_ohm: *eddy_resistance_ohm,
                    });
                }
            }
        }
    }
    let (le_vs_frequency, re_eddy_vs_frequency) = rest_curves.ok_or_else(|| {
        MotorExtractionError::new("displaced eddy extraction did not produce a rest sample")
    })?;

    let bl_nominal = interpolate_curve(&bl_curve, 0.0);
    let le_nominal = le_vs_frequency[0].1;

    Ok(MotorParameters {
        bl_curve,
        le_vs_frequency,
        re_eddy_vs_frequency,
        bl_nominal,
        le_nominal,
        le_surface,
        bl_temperature_coefficient_per_k: geom
            .magnet
            .magnet_type
            .bl_temperature_coefficient_per_k(),
        magnetic_law: config.magnetic_law.clone(),
        hysteresis_loss: config.hysteresis_loss,
        eddy_current_loss: config.eddy_current_loss.clone(),
    })
}

#[derive(Debug, Clone)]
struct CurrentLeSample {
    current_a: f64,
    le_curve: Curve,
    re_curve: Curve,
}

fn extract_adaptive_current_surface(
    geom: &MotorGeometry,
    motor_mesh: &MotorMesh,
    config: &ExtractionConfig,
) -> Result<Vec<CurrentLeSample>, MotorExtractionError> {
    let maximum_current_a = config.maximum_current_a()?;
    let mut currents = (0..=config.initial_current_intervals)
        .map(|index| maximum_current_a * index as f64 / config.initial_current_intervals as f64)
        .collect::<Vec<_>>();
    let mut samples = Vec::new();
    for _ in 0..=config.max_current_refinement_levels {
        let missing_currents = currents
            .iter()
            .copied()
            .filter(|current_a| {
                samples
                    .iter()
                    .all(|sample: &CurrentLeSample| sample.current_a != *current_a)
            })
            .collect::<Vec<_>>();
        // Each current runs a full Le extraction; indexed collect keeps
        // current order identical on both paths (the sort below then
        // re-establishes it after appends across refinement levels).
        let extract_sample = |current_a: f64| {
            let (le_curve, re_curve) = extract_le_curve(geom, motor_mesh, config, current_a)?;
            Ok(CurrentLeSample {
                current_a,
                le_curve,
                re_curve,
            })
        };
        let mut new_samples: Vec<CurrentLeSample> =
            if crate::parallelism::use_modeling_parallel_batch(missing_currents.len()) {
                missing_currents
                    .into_par_iter()
                    .map(&extract_sample)
                    .collect::<Result<Vec<_>, MotorExtractionError>>()?
            } else {
                missing_currents
                    .into_iter()
                    .map(&extract_sample)
                    .collect::<Result<Vec<_>, MotorExtractionError>>()?
            };
        samples.append(&mut new_samples);
        samples.sort_by(|left, right| left.current_a.total_cmp(&right.current_a));
        let mut midpoints = Vec::new();
        for pair in samples.windows(2) {
            let current_span = pair[1].current_a - pair[0].current_a;
            let normalized_span = current_span / maximum_current_a;
            let indicator = pair[0]
                .le_curve
                .iter()
                .zip(&pair[1].le_curve)
                .map(|(left, right)| {
                    (right.1 - left.1).abs()
                        / left.1.abs().max(right.1.abs()).max(f64::MIN_POSITIVE)
                        / normalized_span
                })
                .fold(0.0, f64::max);
            if indicator > config.current_slope_refinement_fraction {
                midpoints.push(0.5 * (pair[0].current_a + pair[1].current_a));
            }
        }
        if midpoints.is_empty() {
            break;
        }
        currents.extend(midpoints);
        currents.sort_by(f64::total_cmp);
        currents.dedup_by(|left, right| *left == *right);
    }
    Ok(samples)
}

/// Extract BL(x) by solving magnetostatics and evaluating the force integral.
fn extract_bl_curve(
    geom: &MotorGeometry,
    motor_mesh: &MotorMesh,
    config: &ExtractionConfig,
) -> Result<Vec<(f64, f64)>, MotorExtractionError> {
    let mesh = &motor_mesh.mesh;
    let space = H1Space::new(PolynomialDegree::P1);

    // Assemble stiffness matrix per region
    // For a multi-region problem, we assemble the global stiffness as sum of
    // per-region contributions. Since all regions share the same mesh, we
    // assemble with the appropriate mu for each element.
    // Solve the magnetostatic problem once (B-field is static, independent of coil position)
    let (stiffness, rhs) = assemble_magnetostatic_system(geom, motor_mesh, &space);

    let mut system = FemSystem::new(stiffness, rhs);

    // Apply homogeneous Dirichlet on outer boundary (phi = 0 at "infinity")
    apply_homogeneous_dirichlet_manual(&mut system, mesh, &space);

    // Solve
    let csr = system.matrix.to_csr();
    let rhs_vec = system.rhs;
    let factorization = sonium_solvers::factorize_sparse(
        &csr,
        &sonium_solvers::SparseDirectConfig {
            matrix_class: sonium_solvers::MatrixClass::SymmetricPositiveDefinite,
            method: sonium_solvers::SparseDirectMethod::SimplicialCholesky,
            ordering: sonium_solvers::Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .map_err(|error| {
        MotorExtractionError::new(format!("magnetostatic factorization failed: {error}"))
    })?;
    let phi = factorization.solve(&rhs_vec).map_err(|error| {
        MotorExtractionError::new(format!("magnetostatic solve failed: {error}"))
    })?;
    let ax = csr.apply(&phi);
    let residual_norm = ax
        .iter()
        .zip(&rhs_vec)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs_vec
        .iter()
        .map(|value| value.powi(2))
        .sum::<f64>()
        .sqrt();
    let true_residual = residual_norm / rhs_norm.max(1e-30);
    if !true_residual.is_finite()
        || phi.iter().any(|value| !value.is_finite())
        || true_residual > 1e-9
    {
        return Err(MotorExtractionError::new(format!(
            "magnetostatic solve failed: true relative residual={true_residual:.3e}"
        )));
    }

    // Axisymmetric BL extraction.
    //
    // In the (r, z) plane, the B-field has components (B_r, B_z).
    // For a ring coil at radius r_coil, the force factor is:
    //   BL = 2π·r_coil·(N_turns/h_coil) · ∫ B_r(r_coil, z) dz
    //
    // where B_r is the radial component (x-component in our 2D mesh).

    let n_coil_pts = 50;
    let coil_r = geom.coil.radius;
    let coil_z_center = geom.pole_piece.height + 0.5 * geom.air_gap;
    let coil_z_bottom = coil_z_center - 0.5 * geom.coil.height;
    let dz = geom.coil.height / n_coil_pts as f64;

    // Generate displacement sweep
    let n_steps = config.n_displacement_steps;
    let displacements: Vec<f64> = (0..=2 * n_steps)
        .map(|i| -config.x_max + 2.0 * config.x_max * i as f64 / (2 * n_steps) as f64)
        .collect();

    // Indexed collect keeps displacement order identical on both paths.
    let integrate_displacement = |dx: f64| -> Result<(f64, f64), MotorExtractionError> {
        let mut radial_field_samples = Vec::with_capacity(n_coil_pts + 1);
        for k in 0..=n_coil_pts {
            let z = coil_z_bottom + k as f64 * dz + dx;
            let point = [coil_r, z, 0.0];
            let b = evaluate_b_at_point(
                &phi,
                mesh,
                &space,
                &motor_mesh.element_regions,
                geom,
                &point,
            )?;
            radial_field_samples.push(b[0]);
        }
        let bl = force_factor_from_radial_field_samples(&geom.coil, &radial_field_samples, dz)?;
        Ok((dx, bl))
    };
    if crate::parallelism::use_modeling_parallel(
        displacements.len().saturating_mul(n_coil_pts + 1),
        crate::parallelism::FIELD_EVALS_PER_THREAD,
    ) {
        displacements
            .into_par_iter()
            .map(&integrate_displacement)
            .collect()
    } else {
        displacements
            .into_iter()
            .map(&integrate_displacement)
            .collect()
    }
}

fn force_factor_from_radial_field_samples(
    coil: &CoilSpec,
    radial_field_samples: &[f64],
    dz: f64,
) -> Result<f64, MotorExtractionError> {
    if radial_field_samples.len() < 2
        || radial_field_samples.iter().any(|value| !value.is_finite())
        || !dz.is_finite()
        || dz <= 0.0
    {
        return Err(MotorExtractionError::new(
            "BL integration requires at least two finite field samples and positive spacing",
        ));
    }
    let integral = radial_field_samples
        .iter()
        .enumerate()
        .map(|(index, &field)| {
            let endpoint = index == 0 || index + 1 == radial_field_samples.len();
            if endpoint { 0.5 * field } else { field }
        })
        .sum::<f64>()
        * dz;
    let turn_density = coil.n_turns as f64 / coil.height;
    Ok(2.0 * PI * coil.radius * turn_density * integral)
}

/// Assemble the multi-region magnetostatic system.
///
/// Per-element assembly using public quadrature + tabulation APIs,
/// with per-region permeability and magnet source terms.
fn assemble_magnetostatic_system(
    geom: &MotorGeometry,
    motor_mesh: &MotorMesh,
    space: &H1Space,
) -> (sonium_solvers::sparse::TripletMatrix<f64>, Vec<f64>) {
    let mesh = &motor_mesh.mesh;
    let n_dofs = space.num_global_dofs(mesh);

    let magnet_mat = geom.magnet.material();
    let mu_0 = sonium_fem::physics::electromagnetics::magnetostatics::MU_0;
    let quad = gauss_triangle(1);
    // Indexed collect preserves element order; the merge below then runs
    // in that order on both paths.
    let assemble_element = |(elem_idx, region): (usize, MotorRegion)| {
        let mu = match region {
            MotorRegion::Air | MotorRegion::Coil | MotorRegion::ConductiveInsert => mu_0,
            MotorRegion::PolePiece => geom.pole_piece.mu_r * mu_0,
            MotorRegion::Magnet => magnet_mat.permeability(),
        };

        let form = PermeabilityForm::from_mu(mu);
        let dofs = space.local_to_global(mesh, elem_idx);
        let n_local = dofs.len();

        // Stiffness: integrate bilinear form at quadrature points
        let mut local_k: LocalMatrix<f64> = LocalMatrix::zeros(n_local);
        for qp in &quad {
            let tab = space.tabulate(mesh, elem_idx, qp.xi(), qp.eta(), qp.zeta());
            form.integrate_point(
                &tab,
                &tab,
                axisymmetric_form_weight(&tab, qp.weight),
                &mut local_k,
            );
        }

        let mut triplets = Vec::with_capacity(n_local * n_local);
        for (li, &gi) in dofs.iter().enumerate() {
            for (lj, &gj) in dofs.iter().enumerate() {
                let v = local_k.get(li, lj);
                if v.abs() > 1e-30 {
                    triplets.push((gi, gj, v));
                }
            }
        }

        let mut rhs_entries = Vec::new();
        if region == MotorRegion::Magnet {
            let source = MagnetSourceForm::new(&magnet_mat);
            let mut local_f: LocalVector<f64> = LocalVector::zeros(n_local);
            for qp in &quad {
                let tab = space.tabulate(mesh, elem_idx, qp.xi(), qp.eta(), qp.zeta());
                source.integrate_point(
                    &tab,
                    axisymmetric_form_weight(&tab, qp.weight),
                    &mut local_f,
                );
            }
            for (li, &gi) in dofs.iter().enumerate() {
                rhs_entries.push((gi, local_f.values[li]));
            }
        }

        (triplets, rhs_entries)
    };
    let element_contributions: Vec<_> = if crate::parallelism::use_modeling_parallel(
        motor_mesh.element_regions.len(),
        crate::parallelism::ASSEMBLY_ELEMENTS_PER_THREAD,
    ) {
        motor_mesh
            .element_regions
            .par_iter()
            .copied()
            .enumerate()
            .map(&assemble_element)
            .collect()
    } else {
        motor_mesh
            .element_regions
            .iter()
            .copied()
            .enumerate()
            .map(&assemble_element)
            .collect()
    };

    // Indexed parallel collection preserves element order. Merge in that
    // order so floating-point accumulation and CSR construction remain
    // deterministic across pool widths.
    let mut triplet = sonium_solvers::sparse::TripletMatrix::new(n_dofs, n_dofs);
    let mut rhs = vec![0.0; n_dofs];
    for (triplets, rhs_entries) in element_contributions {
        for (row, column, value) in triplets {
            triplet.push(row, column, value);
        }
        for (dof, value) in rhs_entries {
            rhs[dof] += value;
        }
    }

    (triplet, rhs)
}

fn axisymmetric_form_weight(
    tabulation: &sonium_fem::spaces::BasisTabulation,
    quadrature_weight: f64,
) -> f64 {
    2.0 * PI * tabulation.physical_coords[0].abs() * quadrature_weight
}

/// Evaluate B-field at an arbitrary point by finding the containing element.
fn evaluate_b_at_point(
    phi: &[f64],
    mesh: &Mesh,
    space: &H1Space,
    element_regions: &[MotorRegion],
    geom: &MotorGeometry,
    point: &[f64; 3],
) -> Result<[f64; 3], MotorExtractionError> {
    let r = point[0];
    let z = point[1];

    // Find containing element (brute force for small meshes)
    #[allow(clippy::needless_range_loop)]
    for elem_idx in 0..mesh.num_elements() {
        let verts = &mesh.elements[elem_idx].nodes;
        let p0 = &mesh.nodes[verts[0]];
        let p1 = &mesh.nodes[verts[1]];
        let p2 = &mesh.nodes[verts[2]];

        if point_in_triangle_pts(r, z, p0, p1, p2) {
            let dofs = space.local_to_global(mesh, elem_idx);
            let phi_local: Vec<f64> = dofs.iter().map(|&d| phi[d]).collect();
            let tab = space.tabulate(mesh, elem_idx, 1.0 / 3.0, 1.0 / 3.0, 0.0);

            let region = element_regions[elem_idx];
            let mu_0 = sonium_fem::physics::electromagnetics::magnetostatics::MU_0;
            let mu = match region {
                MotorRegion::Air | MotorRegion::Coil | MotorRegion::ConductiveInsert => mu_0,
                MotorRegion::PolePiece => geom.pole_piece.mu_r * mu_0,
                MotorRegion::Magnet => geom.magnet.material().permeability(),
            };

            let br = match region {
                MotorRegion::Magnet => geom.magnet.material().remanent_b(),
                MotorRegion::ConductiveInsert => [0.0; 3],
                _ => [0.0, 0.0, 0.0],
            };

            let b_vec = compute_b_field(&phi_local, &tab, mu, &br);
            let field = [
                if !b_vec.is_empty() { b_vec[0] } else { 0.0 },
                if b_vec.len() > 1 { b_vec[1] } else { 0.0 },
                0.0,
            ];
            if field.iter().any(|value| !value.is_finite()) {
                return Err(MotorExtractionError::new(format!(
                    "non-finite magnetic field at r={r}, z={z}"
                )));
            }
            return Ok(field);
        }
    }

    Err(MotorExtractionError::new(format!(
        "magnetic-field sample at r={r}, z={z} lies outside the FEM mesh"
    )))
}

/// Check if point (px, py) is inside a triangle defined by three mesh points.
fn point_in_triangle_pts(px: f64, py: f64, p0: &Point, p1: &Point, p2: &Point) -> bool {
    let d = (p1.y - p2.y) * (p0.x - p2.x) + (p2.x - p1.x) * (p0.y - p2.y);
    if d.abs() < 1e-20 {
        return false; // degenerate triangle
    }
    let l1 = ((p1.y - p2.y) * (px - p2.x) + (p2.x - p1.x) * (py - p2.y)) / d;
    let l2 = ((p2.y - p0.y) * (px - p2.x) + (p0.x - p2.x) * (py - p2.y)) / d;
    let l3 = 1.0 - l1 - l2;
    let eps = -1e-10;
    l1 >= eps && l2 >= eps && l3 >= eps
}

/// Apply homogeneous Dirichlet BCs on boundary marker 1 (far truncation only).
/// The unmarked `r = 0` symmetry axis retains its natural zero-flux condition.
fn apply_homogeneous_dirichlet_manual(system: &mut FemSystem<f64>, mesh: &Mesh, space: &H1Space) {
    let boundary_dofs: std::collections::HashSet<usize> =
        space.boundary_dofs(mesh, 1).into_iter().collect();

    // Filter matrix: remove constrained rows/cols, add diagonal identity
    let mut new_rows = Vec::new();
    let mut new_cols = Vec::new();
    let mut new_vals = Vec::new();

    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !boundary_dofs.contains(&row) && !boundary_dofs.contains(&col) {
            new_rows.push(row);
            new_cols.push(col);
            new_vals.push(system.matrix.values[i]);
        }
    }

    // RHS correction: subtract constrained columns × prescribed values (0 here, so no-op)
    // Set diagonal and RHS for constrained DOFs
    for &dof in &boundary_dofs {
        new_rows.push(dof);
        new_cols.push(dof);
        new_vals.push(1.0);
        system.rhs[dof] = 0.0;
    }

    system.matrix.rows = new_rows;
    system.matrix.cols = new_cols;
    system.matrix.values = new_vals;
}

// ---------------------------------------------------------------------------
// Le(f) extraction via the validated axisymmetric eddy-current FEM path
// ---------------------------------------------------------------------------

/// Extract Le(f) and Re_eddy(f) from a unit-current eddy-current sweep.
fn extract_le_curve(
    geom: &MotorGeometry,
    motor_mesh: &MotorMesh,
    config: &ExtractionConfig,
    bias_current_a: f64,
) -> Result<(Curve, Curve), MotorExtractionError> {
    let mesh = &motor_mesh.mesh;
    let mu_0 = sonium_fem::physics::electromagnetics::magnetostatics::MU_0;
    let differential_reluctivity_by_element =
        solve_bias_differential_reluctivity(geom, motor_mesh, bias_current_a)?;
    let differential_reluctivity = DifferentialReluctivityField {
        by_element_m_per_h: motor_mesh
            .element_regions
            .iter()
            .enumerate()
            .map(|(element_index, region)| match region {
                MotorRegion::PolePiece => differential_reluctivity_by_element[element_index],
                MotorRegion::Air | MotorRegion::Coil | MotorRegion::ConductiveInsert => 1.0 / mu_0,
                MotorRegion::Magnet => 1.0 / geom.magnet.material().permeability(),
            })
            .collect(),
    };
    let coil_area_m2: f64 = motor_mesh
        .coil_elements
        .iter()
        .map(|&element| triangle_area(mesh, mesh.elements[element].vertices()))
        .sum();
    if !coil_area_m2.is_finite() || coil_area_m2 <= 0.0 {
        return Err(MotorExtractionError::new(
            "eddy-current extraction requires a non-empty coil region",
        ));
    }
    let source_density = geom.coil.n_turns as f64 / coil_area_m2;
    let mut conductivity_s_per_m = BTreeMap::from([
        (MotorRegion::PolePiece as i32, geom.pole_piece.conductivity),
        (MotorRegion::Magnet as i32, 0.65e6),
    ]);
    if let Some(insert) = geom.conductive_insert {
        conductivity_s_per_m.insert(
            MotorRegion::ConductiveInsert as i32,
            insert.conductivity_s_per_m,
        );
    }
    let problem = AxisymmetricEddyCurrentProblem {
        mesh: mesh.clone(),
        differential_reluctivity,
        conductivity_s_per_m,
        source_current_density_per_amp_a_per_m2: BTreeMap::from([(
            MotorRegion::Coil as i32,
            source_density,
        )]),
        quadrature_order: 2,
    };
    let boundary: Vec<_> = mesh
        .boundaries
        .iter()
        .flat_map(|face| face.nodes.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    // Each frequency runs a full eddy-current FEM solve; indexed collect
    // keeps frequency order identical on both paths.
    let solve_frequency = |&frequency: &f64| {
        let solved = problem
            .solve_unit_current(frequency, &boundary)
            .map_err(|error| {
                MotorExtractionError::new(format!(
                    "axisymmetric eddy-current solve failed at {frequency} Hz: {error}"
                ))
            })?;
        if !solved.effective_inductance_h.is_finite()
            || solved.effective_inductance_h <= 0.0
            || !solved.eddy_resistance_ohm.is_finite()
            || solved.eddy_resistance_ohm < 0.0
        {
            return Err(MotorExtractionError::new(format!(
                "passive eddy-current FEM result failed at {frequency} Hz"
            )));
        }
        Ok((
            (frequency, solved.effective_inductance_h),
            (frequency, solved.eddy_resistance_ohm),
        ))
    };
    let frequency_results: Vec<_> =
        if crate::parallelism::use_modeling_parallel_batch(config.le_frequencies.len()) {
            config
                .le_frequencies
                .par_iter()
                .map(&solve_frequency)
                .collect::<Result<Vec<_>, MotorExtractionError>>()?
        } else {
            config
                .le_frequencies
                .iter()
                .map(&solve_frequency)
                .collect::<Result<Vec<_>, MotorExtractionError>>()?
        };
    let (le_vs_frequency, re_eddy_vs_frequency) = frequency_results.into_iter().unzip();

    Ok((le_vs_frequency, re_eddy_vs_frequency))
}

fn solve_bias_differential_reluctivity(
    geom: &MotorGeometry,
    motor_mesh: &MotorMesh,
    bias_current_a: f64,
) -> Result<Vec<f64>, MotorExtractionError> {
    if !bias_current_a.is_finite() || bias_current_a < 0.0 {
        return Err(MotorExtractionError::new(
            "eddy-current bias current magnitude must be finite and non-negative",
        ));
    }
    let mu_0 = sonium_fem::physics::electromagnetics::magnetostatics::MU_0;
    let coil_area_m2: f64 = motor_mesh
        .coil_elements
        .iter()
        .map(|&element| {
            triangle_area(
                &motor_mesh.mesh,
                motor_mesh.mesh.elements[element].vertices(),
            )
        })
        .sum();
    if !coil_area_m2.is_finite() || coil_area_m2 <= 0.0 {
        return Err(MotorExtractionError::new(
            "nonlinear magnetic bias requires a non-empty coil region",
        ));
    }
    let linear = |permeability| {
        MagneticConstitutiveLaw::linear(permeability)
            .map_err(|error| MotorExtractionError::new(error.to_string()))
    };
    let mut materials = BTreeMap::new();
    materials.insert(MotorRegion::Air as i32, linear(mu_0)?);
    materials.insert(
        MotorRegion::Magnet as i32,
        linear(geom.magnet.material().permeability())?,
    );
    materials.insert(
        MotorRegion::PolePiece as i32,
        MagneticConstitutiveLaw::Nonlinear(
            crate::nonlinear_bias::default_low_carbon_steel_bh_curve(),
        ),
    );
    materials.insert(MotorRegion::Coil as i32, linear(mu_0)?);
    materials.insert(MotorRegion::ConductiveInsert as i32, linear(mu_0)?);
    let remanence_t = match geom.magnet.magnet_type {
        MagnetType::Neodymium { br } | MagnetType::Ferrite { br } => br,
    };
    let source_density_per_amp = geom.coil.n_turns as f64 / coil_area_m2;
    let problem = AxisymmetricNonlinearMagneticProblem {
        mesh: motor_mesh.mesh.clone(),
        materials,
        azimuthal_current_density_a_per_m2: BTreeMap::from([
            (
                MotorRegion::Magnet as i32,
                remanence_t / (mu_0 * geom.magnet.height),
            ),
            (
                MotorRegion::Coil as i32,
                source_density_per_amp * bias_current_a,
            ),
        ]),
        quadrature_order: 2,
    };
    let constraints: Vec<_> = problem
        .mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == 1)
        .flat_map(|boundary| boundary.nodes.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let solution = solve_nonlinear_axisymmetric_bias(
        &problem,
        &constraints,
        &NonlinearBiasSolveConfig::default(),
    )
    .map_err(|error| {
        MotorExtractionError::new(format!(
            "nonlinear magnetic bias failed at |i|={bias_current_a} A: {error}"
        ))
    })?;
    let mut sum = vec![0.0; motor_mesh.mesh.num_elements()];
    let mut weight = vec![0.0; motor_mesh.mesh.num_elements()];
    for point in solution.quadrature_state {
        sum[point.element_index] +=
            point.differential_reluctivity_m_per_h * point.integration_weight_m3;
        weight[point.element_index] += point.integration_weight_m3;
    }
    for element_index in 0..sum.len() {
        if weight[element_index] > 0.0 {
            sum[element_index] /= weight[element_index];
        } else {
            sum[element_index] = match motor_mesh.element_regions[element_index] {
                MotorRegion::PolePiece => 1.0 / (geom.pole_piece.mu_r * mu_0),
                MotorRegion::Magnet => 1.0 / geom.magnet.material().permeability(),
                MotorRegion::Air | MotorRegion::Coil | MotorRegion::ConductiveInsert => 1.0 / mu_0,
            };
        }
    }
    Ok(sum)
}

fn triangle_area(mesh: &Mesh, nodes: &[usize]) -> f64 {
    let p0 = &mesh.nodes[nodes[0]];
    let p1 = &mesh.nodes[nodes[1]];
    let p2 = &mesh.nodes[nodes[2]];
    0.5 * ((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)).abs()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_motor() -> MotorGeometry {
        MotorGeometry {
            magnet: MagnetSpec {
                magnet_type: MagnetType::Neodymium { br: 1.2 },
                inner_radius: 0.018,
                outer_radius: 0.030,
                height: 0.015,
            },
            pole_piece: PolePieceSpec {
                mu_r: 2000.0,
                conductivity: 6.99e6,
                radius: 0.012,
                height: 0.006,
            },
            coil: CoilSpec {
                radius: 0.0155,
                height: 0.012,
                n_turns: 56,
                wire_diameter: 0.0002,
                layers: 2,
            },
            air_gap: 0.001,
            coil_displacement_m: 0.0,
            conductive_insert: None,
        }
    }

    #[test]
    fn extraction_config_round_trips_optional_magnetic_handoff() {
        let mut config = ExtractionConfig::default();
        config.magnetic_law = Some(
            MotorMagneticLaw::from_bh_samples(
                &[(0.0, 0.0), (0.5, 400.0), (1.0, 1_200.0)],
                2.0e-5,
                0.0,
                None,
            )
            .unwrap(),
        );
        config.hysteresis_loss = Some(HysteresisLossModel {
            coefficient_w_per_m3: 2.0e-3,
            ..HysteresisLossModel::default()
        });
        config.eddy_current_loss = Some(
            EddyCurrentLossModel::from_samples(&[
                crate::nonlinear_motor::EddyCurrentLossSample {
                    frequency_hz: 100.0,
                    flux_density_t: 0.5,
                    eddy_resistance_ohm: 0.1,
                    skin_depth_m: 0.01,
                    proximity_factor: 1.1,
                },
                crate::nonlinear_motor::EddyCurrentLossSample {
                    frequency_hz: 1_000.0,
                    flux_density_t: 0.5,
                    eddy_resistance_ohm: 0.2,
                    skin_depth_m: 0.005,
                    proximity_factor: 1.2,
                },
                crate::nonlinear_motor::EddyCurrentLossSample {
                    frequency_hz: 100.0,
                    flux_density_t: 1.0,
                    eddy_resistance_ohm: 0.2,
                    skin_depth_m: 0.009,
                    proximity_factor: 1.2,
                },
                crate::nonlinear_motor::EddyCurrentLossSample {
                    frequency_hz: 1_000.0,
                    flux_density_t: 1.0,
                    eddy_resistance_ohm: 0.4,
                    skin_depth_m: 0.004,
                    proximity_factor: 1.4,
                },
            ])
            .unwrap(),
        );
        let encoded = serde_json::to_vec(&config).unwrap();
        let decoded: ExtractionConfig = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.magnetic_law, config.magnetic_law);
        assert_eq!(decoded.hysteresis_loss, config.hysteresis_loss);
        assert_eq!(decoded.eddy_current_loss, config.eddy_current_loss);

        let mut legacy = serde_json::to_value(&config).unwrap();
        legacy.as_object_mut().unwrap().remove("magnetic_law");
        legacy.as_object_mut().unwrap().remove("hysteresis_loss");
        legacy.as_object_mut().unwrap().remove("eddy_current_loss");
        let decoded_legacy: ExtractionConfig = serde_json::from_value(legacy).unwrap();
        assert!(decoded_legacy.magnetic_law.is_none());
        assert!(decoded_legacy.hysteresis_loss.is_none());
    }

    #[test]
    fn test_motor_mesh_generation() {
        let geom = test_motor();
        let config = ExtractionConfig {
            mesh_resolution: 2,
            ..ExtractionConfig::default()
        };
        let motor_mesh = generate_motor_mesh(&geom, &config).unwrap();

        assert!(motor_mesh.mesh.num_nodes() > 0);
        assert!(motor_mesh.mesh.num_elements() > 0);
        assert_eq!(
            motor_mesh.element_regions.len(),
            motor_mesh.mesh.num_elements()
        );

        // Should have elements in all regions
        let has_air = motor_mesh.element_regions.contains(&MotorRegion::Air);
        let has_magnet = motor_mesh.element_regions.contains(&MotorRegion::Magnet);
        let has_pole = motor_mesh.element_regions.contains(&MotorRegion::PolePiece);
        let has_coil = motor_mesh.element_regions.contains(&MotorRegion::Coil);

        assert!(has_air, "Missing air region");
        assert!(has_magnet, "Missing magnet region");
        assert!(has_pole, "Missing pole piece region");
        assert!(has_coil, "Missing coil region");
    }

    #[test]
    fn test_motor_validation_rejects_invalid_geometry_and_config() {
        type GeometryMutation = Box<dyn Fn(&mut MotorGeometry)>;
        let invalid_geometry: Vec<(&str, GeometryMutation)> = vec![
            (
                "remanence NaN",
                Box::new(|g| g.magnet.magnet_type = MagnetType::Neodymium { br: f64::NAN }),
            ),
            ("pole radius zero", Box::new(|g| g.pole_piece.radius = 0.0)),
            (
                "pole permeability negative",
                Box::new(|g| g.pole_piece.mu_r = -1.0),
            ),
            (
                "pole conductivity infinite",
                Box::new(|g| g.pole_piece.conductivity = f64::INFINITY),
            ),
            ("coil radius NaN", Box::new(|g| g.coil.radius = f64::NAN)),
            ("coil height zero", Box::new(|g| g.coil.height = 0.0)),
            ("coil turns zero", Box::new(|g| g.coil.n_turns = 0)),
            ("coil layers zero", Box::new(|g| g.coil.layers = 0)),
            (
                "magnet radii reversed",
                Box::new(|g| g.magnet.inner_radius = g.magnet.outer_radius),
            ),
            (
                "coil intersects pole",
                Box::new(|g| g.coil.radius = g.pole_piece.radius),
            ),
            (
                "coil intersects magnet",
                Box::new(|g| g.magnet.inner_radius = g.coil.outer_radius() - 1e-4),
            ),
            ("air gap zero", Box::new(|g| g.air_gap = 0.0)),
        ];
        for (name, mutate) in invalid_geometry {
            let mut geometry = test_motor();
            mutate(&mut geometry);
            assert!(
                geometry.validate().is_err(),
                "accepted invalid geometry: {name}"
            );
        }

        let invalid_configs = [
            ExtractionConfig {
                x_max: f64::NAN,
                ..ExtractionConfig::default()
            },
            ExtractionConfig {
                n_displacement_steps: 0,
                ..ExtractionConfig::default()
            },
            ExtractionConfig {
                mesh_resolution: 0,
                ..ExtractionConfig::default()
            },
            ExtractionConfig {
                le_frequencies: vec![],
                ..ExtractionConfig::default()
            },
            ExtractionConfig {
                le_frequencies: vec![100.0, 50.0],
                ..ExtractionConfig::default()
            },
            ExtractionConfig {
                le_frequencies: vec![100.0, f64::NAN],
                ..ExtractionConfig::default()
            },
        ];
        for config in &invalid_configs {
            assert!(
                config.validate().is_err(),
                "accepted invalid extraction config"
            );
        }
    }

    #[test]
    fn test_constant_radial_field_bl_has_correct_dimensions() {
        let coil = test_motor().coil;
        let radial_field = 0.8;
        let samples = vec![radial_field; 101];
        let dz = coil.height / (samples.len() - 1) as f64;
        let bl = force_factor_from_radial_field_samples(&coil, &samples, dz).unwrap();
        let expected = 2.0 * PI * coil.radius * coil.n_turns as f64 * radial_field;
        assert!(
            (bl - expected).abs() / expected < 1e-12,
            "BL={bl}, expected={expected}"
        );
    }

    #[test]
    fn test_motor_mesh_pads_displacement_and_keeps_axis_natural() {
        let geom = test_motor();
        let config = ExtractionConfig {
            x_max: 0.004,
            mesh_resolution: 2,
            ..ExtractionConfig::default()
        };
        let motor_mesh = generate_motor_mesh(&geom, &config).unwrap();
        let coil_center = geom.pole_piece.height + 0.5 * geom.air_gap;
        let required_min = coil_center - 0.5 * geom.coil.height - config.x_max;
        let required_max = coil_center + 0.5 * geom.coil.height + config.x_max;
        let min_z = motor_mesh
            .mesh
            .nodes
            .iter()
            .map(|point| point.y)
            .fold(f64::INFINITY, f64::min);
        let max_z = motor_mesh
            .mesh
            .nodes
            .iter()
            .map(|point| point.y)
            .fold(f64::NEG_INFINITY, f64::max);
        let max_r = motor_mesh
            .mesh
            .nodes
            .iter()
            .map(|point| point.x)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(min_z < required_min);
        assert!(max_z > required_max);
        assert!(max_r > geom.magnet.outer_radius);

        let mut saw_axis = false;
        let mut saw_far_dirichlet = false;
        for boundary in &motor_mesh.mesh.boundaries {
            let on_axis = boundary
                .nodes
                .iter()
                .all(|&node| motor_mesh.mesh.nodes[node].x.abs() < 1e-12);
            if on_axis {
                saw_axis = true;
                assert_ne!(boundary.marker, 1, "symmetry axis was constrained");
            } else if boundary.marker == 1 {
                saw_far_dirichlet = true;
            }
        }
        assert!(saw_axis);
        assert!(saw_far_dirichlet);
    }

    #[test]
    fn test_axisymmetric_form_weight_includes_circumference() {
        let geom = test_motor();
        let config = ExtractionConfig {
            mesh_resolution: 2,
            ..ExtractionConfig::default()
        };
        let motor_mesh = generate_motor_mesh(&geom, &config).unwrap();
        let space = H1Space::new(PolynomialDegree::P1);
        let quadrature = gauss_triangle(1);
        let qp = &quadrature[0];
        let element = motor_mesh
            .mesh
            .elements
            .iter()
            .enumerate()
            .find(|(_, element)| {
                element
                    .nodes
                    .iter()
                    .all(|&node| motor_mesh.mesh.nodes[node].x > 1e-6)
            })
            .map(|(index, _)| index)
            .unwrap();
        let tab = space.tabulate(&motor_mesh.mesh, element, qp.xi(), qp.eta(), qp.zeta());
        let got = axisymmetric_form_weight(&tab, qp.weight);
        let expected = 2.0 * PI * tab.physical_coords[0] * qp.weight;
        assert!((got - expected).abs() < 1e-14);
    }

    #[test]
    fn test_field_evaluation_rejects_points_outside_mesh() {
        let geom = test_motor();
        let config = ExtractionConfig {
            mesh_resolution: 2,
            ..ExtractionConfig::default()
        };
        let motor_mesh = generate_motor_mesh(&geom, &config).unwrap();
        let space = H1Space::new(PolynomialDegree::P1);
        let phi = vec![0.0; space.num_global_dofs(&motor_mesh.mesh)];
        let outside = [geom.magnet.outer_radius * 10.0, 0.0, 0.0];
        assert!(
            evaluate_b_at_point(
                &phi,
                &motor_mesh.mesh,
                &space,
                &motor_mesh.element_regions,
                &geom,
                &outside,
            )
            .is_err()
        );
    }

    #[test]
    fn test_interpolate_curve() {
        let curve = vec![(-0.01, 5.0), (0.0, 10.0), (0.01, 5.0)];

        // Exact points
        assert!((interpolate_curve(&curve, 0.0) - 10.0).abs() < 1e-10);
        assert!((interpolate_curve(&curve, -0.01) - 5.0).abs() < 1e-10);

        // Midpoint interpolation
        assert!((interpolate_curve(&curve, 0.005) - 7.5).abs() < 1e-10);
        assert!((interpolate_curve(&curve, -0.005) - 7.5).abs() < 1e-10);

        // Extrapolation (clamped)
        assert!((interpolate_curve(&curve, -0.02) - 5.0).abs() < 1e-10);
        assert!((interpolate_curve(&curve, 0.02) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_in_triangle() {
        let p0 = Point::new_2d(0.0, 0.0);
        let p1 = Point::new_2d(1.0, 0.0);
        let p2 = Point::new_2d(0.0, 1.0);
        assert!(point_in_triangle_pts(0.25, 0.25, &p0, &p1, &p2));
        assert!(!point_in_triangle_pts(1.0, 1.0, &p0, &p1, &p2));
        // On edge
        assert!(point_in_triangle_pts(0.5, 0.0, &p0, &p1, &p2));
    }

    #[test]
    fn test_le_eddy_current_fem_is_passive() {
        let geom = test_motor();
        let config = ExtractionConfig::default();
        let motor_mesh = generate_motor_mesh(&geom, &config).unwrap();

        let (le_curve, re_curve) = extract_le_curve(&geom, &motor_mesh, &config, 0.0).unwrap();

        // Le should decrease with frequency (eddy currents expel flux)
        for i in 1..le_curve.len() {
            assert!(
                le_curve[i].1 <= le_curve[i - 1].1 + 1e-15,
                "Le should decrease: Le({:.0}Hz)={:.6e} > Le({:.0}Hz)={:.6e}",
                le_curve[i].0,
                le_curve[i].1,
                le_curve[i - 1].0,
                le_curve[i - 1].1
            );
        }

        // Le should be positive
        for &(f, le) in &le_curve {
            assert!(le > 0.0, "Le({f:.0}Hz) = {le} should be positive");
        }

        // Re_eddy should be non-negative
        for &(f, re) in &re_curve {
            assert!(re >= 0.0, "Re_eddy({f:.0}Hz) = {re} should be non-negative");
        }

        for (&(frequency, inductance), &(_, resistance)) in le_curve.iter().zip(&re_curve) {
            let reconstructed = Complex64::new(resistance, 2.0 * PI * frequency * inductance);
            assert!(reconstructed.re >= 0.0);
            assert!(reconstructed.im > 0.0);
        }
    }

    #[test]
    fn m3_current_grid_populates_even_nonincreasing_le_saturation_law() {
        let geom = test_motor();
        let config = ExtractionConfig {
            mesh_resolution: 1,
            le_frequencies: vec![1_000.0],
            initial_current_intervals: 2,
            max_current_refinement_levels: 0,
            ..ExtractionConfig::default()
        };
        let mesh = generate_motor_mesh(&geom, &config).unwrap();
        let samples = extract_adaptive_current_surface(&geom, &mesh, &config).unwrap();

        assert_eq!(samples.len(), 3);
        for pair in samples.windows(2) {
            assert!(
                pair[1].le_curve[0].1 <= pair[0].le_curve[0].1 * (1.0 + 1.0e-10),
                "Le must not increase with |i|: {samples:?}"
            );
        }
        let surface = extract_motor_parameters(
            &geom,
            &ExtractionConfig {
                x_max: 1.0e-3,
                n_displacement_steps: 1,
                mesh_resolution: 1,
                le_frequencies: vec![1_000.0],
                initial_current_intervals: 1,
                max_current_refinement_levels: 0,
                ..ExtractionConfig::default()
            },
        )
        .unwrap()
        .le_surface;
        for positive in surface.iter().filter(|point| point.current_a > 0.0) {
            let negative = surface
                .iter()
                .find(|point| {
                    point.displacement_m == positive.displacement_m
                        && point.current_a == -positive.current_a
                        && point.frequency_hz == positive.frequency_hz
                })
                .unwrap();
            assert_eq!(positive.inductance_h, negative.inductance_h);
            assert_eq!(positive.eddy_resistance_ohm, negative.eddy_resistance_ohm);
        }
    }

    #[test]
    fn m3_conductive_insert_is_meshed_and_reduces_high_frequency_inductance() {
        let plain = test_motor();
        let mut ringed = plain.clone();
        ringed.conductive_insert = Some(ConductiveInsertSpec {
            inner_radius_m: 0.0,
            outer_radius_m: 12.0e-3,
            z_bottom_m: 5.5e-3,
            z_top_m: 6.0e-3,
            conductivity_s_per_m: 5.8e7,
        });
        let config = ExtractionConfig {
            mesh_resolution: 2,
            le_frequencies: vec![1_000.0, 10_000.0],
            max_current_refinement_levels: 0,
            ..ExtractionConfig::default()
        };
        let plain_mesh = generate_motor_mesh(&plain, &config).unwrap();
        let ringed_mesh = generate_motor_mesh(&ringed, &config).unwrap();
        assert!(
            ringed_mesh
                .element_regions
                .contains(&MotorRegion::ConductiveInsert)
        );
        let (plain_le, plain_re) = extract_le_curve(&plain, &plain_mesh, &config, 0.0).unwrap();
        let (ringed_le, ringed_re) = extract_le_curve(&ringed, &ringed_mesh, &config, 0.0).unwrap();

        assert!(
            ringed_le[1].1 < plain_le[1].1,
            "ringed={ringed_le:?}, plain={plain_le:?}, ringed_re={ringed_re:?}, plain_re={plain_re:?}"
        );
        let plain_variation = (plain_re[1].1 - plain_re[0].1).abs();
        let ringed_variation = (ringed_re[1].1 - ringed_re[0].1).abs();
        assert!(ringed_variation < plain_variation);
    }

    #[test]
    fn magnet_material_defaults_reduce_bl_with_temperature() {
        assert_eq!(
            MagnetType::Neodymium { br: 1.2 }.bl_temperature_coefficient_per_k(),
            -1.0e-3
        );
        assert_eq!(
            MagnetType::Ferrite { br: 0.4 }.bl_temperature_coefficient_per_k(),
            -2.0e-3
        );
    }

    #[test]
    fn m3_versioned_production_input_emits_typed_evidence() {
        let input = MotorCompletionInput {
            schema_version: MOTOR_COMPLETION_SCHEMA_VERSION,
            geometry: test_motor(),
            extraction: ExtractionConfig {
                x_max: 1.0e-3,
                n_displacement_steps: 1,
                mesh_resolution: 1,
                le_frequencies: vec![1_000.0],
                max_current_refinement_levels: 0,
                ..ExtractionConfig::default()
            },
        };
        let result = run_motor_completion(&input).unwrap();
        assert_eq!(result.schema_version, MOTOR_COMPLETION_SCHEMA_VERSION);
        assert!(result.maximum_current_a > 0.0);
        assert_eq!(result.parameters.le_surface.len(), 15);
        assert!(result.provenance.contains("DifferentialReluctivity"));
        let encoded = serde_json::to_value(result).unwrap();
        assert_eq!(encoded["schema_version"], 1);
    }

    #[test]
    fn test_coil_spec_geometry() {
        let coil = CoilSpec {
            radius: 0.0155,
            height: 0.012,
            n_turns: 56,
            wire_diameter: 0.0002,
            layers: 2,
        };

        assert!((coil.winding_thickness() - 0.0004).abs() < 1e-10);
        assert!((coil.inner_radius() - 0.0153).abs() < 1e-10);
        assert!((coil.outer_radius() - 0.0157).abs() < 1e-10);
    }

    #[test]
    fn test_motor_parameters_to_voice_coil_fns() {
        let params = MotorParameters {
            bl_curve: vec![(-0.01, 5.0), (0.0, 10.0), (0.01, 5.0)],
            le_vs_frequency: vec![(100.0, 0.5e-3), (1000.0, 0.4e-3)],
            re_eddy_vs_frequency: vec![(100.0, 0.1), (1000.0, 0.5)],
            bl_nominal: 10.0,
            le_nominal: 0.5e-3,
            le_surface: Vec::new(),
            bl_temperature_coefficient_per_k: 0.0,
            magnetic_law: None,
            hysteresis_loss: None,
            eddy_current_loss: None,
        };

        let (bl_fn, le_fn, dle_fn) = params.to_voice_coil_fns();

        assert!((bl_fn(0.0) - 10.0).abs() < 1e-10);
        assert!((bl_fn(0.005) - 7.5).abs() < 1e-10);
        assert!((le_fn(0.0) - 0.5e-3).abs() < 1e-10);
        assert!((dle_fn(0.0)).abs() < 1e-10);
    }

    #[test]
    fn voice_coil_callbacks_use_zero_current_low_frequency_le_slice() {
        let params = MotorParameters {
            bl_curve: vec![(-0.01, 5.0), (0.0, 10.0), (0.01, 5.0)],
            le_vs_frequency: vec![(100.0, 0.5e-3)],
            re_eddy_vs_frequency: vec![(100.0, 0.1)],
            bl_nominal: 10.0,
            le_nominal: 0.5e-3,
            le_surface: vec![
                NonlinearLePoint {
                    displacement_m: -0.01,
                    current_a: 0.0,
                    frequency_hz: 100.0,
                    inductance_h: 0.4e-3,
                    eddy_resistance_ohm: 0.1,
                },
                NonlinearLePoint {
                    displacement_m: 0.0,
                    current_a: 0.0,
                    frequency_hz: 100.0,
                    inductance_h: 0.5e-3,
                    eddy_resistance_ohm: 0.1,
                },
                NonlinearLePoint {
                    displacement_m: 0.01,
                    current_a: 0.0,
                    frequency_hz: 100.0,
                    inductance_h: 0.6e-3,
                    eddy_resistance_ohm: 0.1,
                },
                // The zero-current low-frequency slice is preferred to this
                // higher-frequency sample at the same displacement.
                NonlinearLePoint {
                    displacement_m: 0.0,
                    current_a: 0.0,
                    frequency_hz: 1_000.0,
                    inductance_h: 0.2e-3,
                    eddy_resistance_ohm: 0.5,
                },
                // The displacement-only legacy API must not silently use a
                // current-dependent point.
                NonlinearLePoint {
                    displacement_m: 0.0,
                    current_a: 1.0,
                    frequency_hz: 50.0,
                    inductance_h: 0.1e-3,
                    eddy_resistance_ohm: 0.8,
                },
            ],
            bl_temperature_coefficient_per_k: 0.0,
            magnetic_law: None,
            hysteresis_loss: None,
            eddy_current_loss: None,
        };

        let (_, le_fn, dle_fn) = params.to_voice_coil_fns();
        assert!((le_fn(0.0) - 0.5e-3).abs() < 1.0e-12);
        assert!((le_fn(0.005) - 0.55e-3).abs() < 1.0e-12);
        assert!((dle_fn(0.0) - 0.01).abs() < 1.0e-12);
    }

    #[test]
    fn test_full_bl_extraction() {
        // Full pipeline: generate mesh, solve magnetostatics, extract BL(x)
        let geom = test_motor();
        let config = ExtractionConfig {
            x_max: 0.002, // small range for faster test
            n_displacement_steps: 3,
            mesh_resolution: 2, // coarse for speed
            le_frequencies: vec![100.0, 1000.0],
            max_current_refinement_levels: 0,
            ..ExtractionConfig::default()
        };

        let params = extract_motor_parameters(&geom, &config).unwrap();

        // BL curve should have the expected number of points
        assert_eq!(params.bl_curve.len(), 7); // 2*3+1
        assert_eq!(params.le_surface.len(), 30);
        assert!(
            params
                .le_surface
                .iter()
                .any(|point| point.displacement_m < 0.0)
        );
        assert!(
            params
                .le_surface
                .iter()
                .any(|point| point.displacement_m > 0.0)
        );

        // BL should be non-zero (the magnet produces a field)
        assert!(
            params.bl_nominal.abs() > 0.01,
            "BL_nominal = {} is too small — magnet field may not be captured",
            params.bl_nominal
        );

        // BL should be roughly symmetric
        let bl_neg = interpolate_curve(&params.bl_curve, -0.001);
        let bl_pos = interpolate_curve(&params.bl_curve, 0.001);
        let bl_zero = interpolate_curve(&params.bl_curve, 0.0);

        // At rest should be the peak (or close to it for a well-centered coil)
        // Allow significant tolerance — coarse mesh won't be super accurate
        assert!(bl_zero > 0.0, "BL(0) = {bl_zero} should be positive");
        assert!(
            (bl_neg - bl_pos).abs() / bl_zero.abs().max(1e-12) < 0.2,
            "BL curve is unexpectedly asymmetric: BL(-1mm)={bl_neg}, BL(+1mm)={bl_pos}"
        );
        assert!(
            params.le_nominal >= params.le_vs_frequency[0].1,
            "nominal Le must be the DC limit, not the first sampled frequency"
        );
    }
}
