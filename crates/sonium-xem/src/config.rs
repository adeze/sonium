//! JSON configuration for room acoustics simulations

use crate::conventions::{
    CURRENT_SCHEMA_VERSION, ConventionRegistry, DensityKgM3, FrequencyHz, SchemaVersion,
    SpeedOfSoundMPerS,
};
use crate::geometry::{LShapedRoom, RectangularRoom, RoomGeometry};
use crate::source::{
    ComplexDirectivityGrid, CrossoverFilter, Directivity, DirectivityGrid, Source,
};
use crate::types::{Point3D, log_space};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Complete room configuration loaded from JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoomConfig {
    /// Exchange schema version. Documents without this field default to v1;
    /// future versions are rejected until an explicit migration is provided.
    #[serde(default)]
    pub schema_version: SchemaVersion,
    /// Room geometry specification
    pub room: RoomGeometryConfig,
    /// Sound sources
    pub sources: Vec<SourceConfig>,
    /// Listening positions
    pub listening_positions: Vec<Point3DConfig>,
    /// Frequency configuration
    pub frequencies: FrequencyConfig,
    /// Boundary conditions
    #[serde(default)]
    pub boundaries: BoundaryConfig,
    /// Room-level ambient medium. Its sound speed drives
    /// [`RoomSimulation::wavenumber`] via [`Self::to_simulation`]; porous
    /// layers keep their own per-surface ambient for impedance
    /// normalization only.
    #[serde(default)]
    pub medium: AmbientMediumConfig,
    /// Solver configuration
    #[serde(default)]
    pub solver: SolverConfig,
    /// Visualization configuration
    #[serde(default)]
    pub visualization: VisualizationConfig,
    /// Simulation metadata
    #[serde(default)]
    pub metadata: MetadataConfig,
    /// Shared phasor, frame, normal, and amplitude conventions.
    #[serde(default)]
    pub conventions: ConventionRegistry,
}

/// Typed validation failure at the XEM configuration boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigValidationError {
    /// No migration is registered for the document version.
    UnsupportedSchemaVersion(u32),
    /// A named field violates its physical or structural contract.
    InvalidField { field: String, reason: String },
    /// The strict JSON entry point encountered an unknown field.
    UnknownField { path: String },
}

impl std::fmt::Display for ConfigValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion(version) => {
                write!(f, "unsupported XEM schema version {version}")
            }
            Self::InvalidField { field, reason } => write!(f, "invalid {field}: {reason}"),
            Self::UnknownField { path } => write!(f, "unknown configuration field {path}"),
        }
    }
}

impl std::error::Error for ConfigValidationError {}

fn invalid(field: impl Into<String>, reason: impl Into<String>) -> ConfigValidationError {
    ConfigValidationError::InvalidField {
        field: field.into(),
        reason: reason.into(),
    }
}

fn finite_positive(field: &str, value: f64) -> Result<(), ConfigValidationError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid(field, "must be finite and positive"))
    }
}

fn finite(field: &str, value: f64) -> Result<(), ConfigValidationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid(field, "must be finite"))
    }
}

/// Room geometry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RoomGeometryConfig {
    #[serde(rename = "rectangular")]
    /// Rectangular room geometry
    Rectangular {
        /// Width (x-dimension)
        width: f64,
        /// Depth (y-dimension)
        depth: f64,
        /// Height (z-dimension)
        height: f64,
    },
    #[serde(rename = "lshaped")]
    /// L-shaped room geometry
    LShaped {
        /// Width of the main section
        width1: f64,
        /// Depth of the main section
        depth1: f64,
        /// Width of the extension
        width2: f64,
        /// Depth of the extension
        depth2: f64,
        /// Height of the room
        height: f64,
    },
}

impl RoomGeometryConfig {
    /// Convert to RoomGeometry
    pub fn to_geometry(&self) -> Result<RoomGeometry, String> {
        self.validate().map_err(|error| error.to_string())?;
        match self {
            RoomGeometryConfig::Rectangular {
                width,
                depth,
                height,
            } => Ok(RoomGeometry::Rectangular(RectangularRoom::new(
                *width, *depth, *height,
            ))),
            RoomGeometryConfig::LShaped {
                width1,
                depth1,
                width2,
                depth2,
                height,
            } => Ok(RoomGeometry::LShaped(LShapedRoom::new(
                *width1, *depth1, *width2, *depth2, *height,
            ))),
        }
    }

    /// Validate dimensions before constructing solver geometry.
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        match self {
            Self::Rectangular {
                width,
                depth,
                height,
            } => {
                finite_positive("room.width", *width)?;
                finite_positive("room.depth", *depth)?;
                finite_positive("room.height", *height)?;
            }
            Self::LShaped {
                width1,
                depth1,
                width2,
                depth2,
                height,
            } => {
                finite_positive("room.width1", *width1)?;
                finite_positive("room.depth1", *depth1)?;
                finite_positive("room.width2", *width2)?;
                finite_positive("room.depth2", *depth2)?;
                finite_positive("room.height", *height)?;
                // The L footprint notches inward at (width2, depth1); the
                // extension must be strictly narrower than the main section
                // or the floor polygon self-intersects.
                if width2 >= width1 {
                    return Err(invalid(
                        "room.width2",
                        "L-shaped extension must be narrower than the main section (width2 < width1)",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Boundary conditions configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BoundaryConfig {
    /// Floor boundary condition
    pub floor: SurfaceConfig,
    /// Ceiling boundary condition
    pub ceiling: SurfaceConfig,
    /// Default condition for all vertical walls
    pub walls: SurfaceConfig,
    /// Override for front wall (y=0)
    pub front_wall: Option<SurfaceConfig>,
    /// Override for back wall (y=depth)
    pub back_wall: Option<SurfaceConfig>,
    /// Override for left wall (x=0)
    pub left_wall: Option<SurfaceConfig>,
    /// Override for right wall (x=width)
    pub right_wall: Option<SurfaceConfig>,
}

impl Default for BoundaryConfig {
    fn default() -> Self {
        Self {
            floor: SurfaceConfig::Rigid,
            ceiling: SurfaceConfig::Rigid,
            walls: SurfaceConfig::Rigid,
            front_wall: None,
            back_wall: None,
            left_wall: None,
            right_wall: None,
        }
    }
}

/// Surface boundary condition type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SurfaceConfig {
    /// Perfectly rigid (Neumann BC, velocity = 0)
    #[serde(rename = "rigid")]
    Rigid,
    /// Absorption coefficient (0.0 to 1.0)
    #[serde(rename = "absorption")]
    Absorption { coefficient: f64 },
    /// Specific acoustic impedance (complex)
    #[serde(rename = "impedance")]
    Impedance { real: f64, imag: f64 },
    /// Normalized acoustic admittance (rho*c/Z)
    #[serde(rename = "admittance")]
    Admittance { real: f64, imag: f64 },
    /// Rigid-backed equivalent-fluid porous layer
    #[serde(rename = "porous_layer")]
    PorousLayer {
        /// Empirical porous material model
        model: PorousModelConfig,
        /// Layer thickness in meters
        thickness: f64,
        /// Static flow resistivity in Pa*s/m^2
        flow_resistivity: f64,
        /// Optional air gap behind the layer in meters
        #[serde(default)]
        air_gap: Option<f64>,
        /// Ambient fluid properties for impedance normalization
        #[serde(default)]
        ambient: AmbientMediumConfig,
    },
}

/// Empirical porous-layer model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PorousModelConfig {
    /// Delany-Bazley fibrous absorber model
    DelanyBazley,
    /// Miki correction to Delany-Bazley
    Miki,
}

impl SurfaceConfig {
    /// Validate the physical contract of one boundary surface: absorption
    /// in `[0, 1]`, finite impedance/admittance, positive porous-layer
    /// thickness and flow resistivity, and a finite positive ambient
    /// medium. `field` names the surface for diagnostics.
    pub fn validate(&self, field: &str) -> Result<(), ConfigValidationError> {
        match self {
            SurfaceConfig::Rigid => Ok(()),
            SurfaceConfig::Absorption { coefficient } => {
                if coefficient.is_finite() && (0.0..=1.0).contains(coefficient) {
                    Ok(())
                } else {
                    Err(invalid(
                        field,
                        "absorption coefficient must be finite and in [0, 1]",
                    ))
                }
            }
            SurfaceConfig::Impedance { real, imag } => {
                finite(&format!("{field}.real"), *real)?;
                finite(&format!("{field}.imag"), *imag)
            }
            SurfaceConfig::Admittance { real, imag } => {
                finite(&format!("{field}.real"), *real)?;
                finite(&format!("{field}.imag"), *imag)
            }
            SurfaceConfig::PorousLayer {
                thickness,
                flow_resistivity,
                air_gap,
                ambient,
                ..
            } => {
                finite_positive(&format!("{field}.thickness"), *thickness)?;
                finite_positive(&format!("{field}.flow_resistivity"), *flow_resistivity)?;
                if let Some(gap) = air_gap
                    && !(gap.is_finite() && *gap >= 0.0)
                {
                    return Err(invalid(
                        format!("{field}.air_gap"),
                        "must be finite and non-negative",
                    ));
                }
                finite_positive(&format!("{field}.ambient.density"), ambient.density)?;
                finite_positive(
                    &format!("{field}.ambient.speed_of_sound"),
                    ambient.speed_of_sound,
                )
            }
        }
    }
}

impl BoundaryConfig {
    /// Validate every boundary surface. `RoomConfig::validate` calls this
    /// so unphysical boundaries (e.g. absorption 5.0) cannot pass
    /// Rust-side validation.
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        self.floor.validate("boundaries.floor")?;
        self.ceiling.validate("boundaries.ceiling")?;
        self.walls.validate("boundaries.walls")?;
        for (name, wall) in [
            ("boundaries.front_wall", &self.front_wall),
            ("boundaries.back_wall", &self.back_wall),
            ("boundaries.left_wall", &self.left_wall),
            ("boundaries.right_wall", &self.right_wall),
        ] {
            if let Some(surface) = wall {
                surface.validate(name)?;
            }
        }
        Ok(())
    }
}

/// Ambient fluid properties used by material impedance models.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct AmbientMediumConfig {
    /// Air density in kg/m^3
    pub density: f64,
    /// Sound speed in m/s
    pub speed_of_sound: f64,
}

impl Default for AmbientMediumConfig {
    fn default() -> Self {
        Self {
            density: default_air_density(),
            speed_of_sound: default_speed_of_sound(),
        }
    }
}

impl AmbientMediumConfig {
    /// Return the density through the typed exchange quantity boundary.
    pub fn typed_density(&self) -> Result<DensityKgM3, ConfigValidationError> {
        DensityKgM3::try_new(self.density)
            .map_err(|reason| invalid("boundaries.ambient.density", reason))
    }

    /// Return the sound speed through the typed exchange quantity boundary.
    pub fn typed_speed_of_sound(&self) -> Result<SpeedOfSoundMPerS, ConfigValidationError> {
        SpeedOfSoundMPerS::try_new(self.speed_of_sound)
            .map_err(|reason| invalid("boundaries.ambient.speed_of_sound", reason))
    }
}

fn default_air_density() -> f64 {
    1.21
}

fn default_speed_of_sound() -> f64 {
    crate::types::constants::SPEED_OF_SOUND_20C
}

/// 3D point configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point3DConfig {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
    /// Z coordinate
    pub z: f64,
}

impl From<Point3DConfig> for Point3D {
    fn from(p: Point3DConfig) -> Self {
        Point3D::new(p.x, p.y, p.z)
    }
}

impl From<Point3D> for Point3DConfig {
    fn from(p: Point3D) -> Self {
        Point3DConfig {
            x: p.x,
            y: p.y,
            z: p.z,
        }
    }
}

/// Source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    /// Source name
    pub name: String,
    /// Source position
    pub position: Point3DConfig,
    /// Source amplitude
    #[serde(default = "default_amplitude")]
    pub amplitude: f64,
    /// Directivity pattern
    #[serde(default)]
    pub directivity: DirectivityConfig,
    /// Crossover filter
    #[serde(default)]
    pub crossover: CrossoverConfig,
}

fn default_amplitude() -> f64 {
    1.0
}

impl SourceConfig {
    /// Validate source scalars before conversion to the runtime model.
    pub fn validate(&self, index: usize) -> Result<(), ConfigValidationError> {
        finite(&format!("sources[{index}].position.x"), self.position.x)?;
        finite(&format!("sources[{index}].position.y"), self.position.y)?;
        finite(&format!("sources[{index}].position.z"), self.position.z)?;
        finite(&format!("sources[{index}].amplitude"), self.amplitude)?;
        if self.name.trim().is_empty() {
            return Err(invalid(
                format!("sources[{index}].name"),
                "must not be empty",
            ));
        }
        self.to_source()
            .map(|_| ())
            .map_err(|reason| invalid(format!("sources[{index}].directivity"), reason))
    }

    /// Convert to Source
    pub fn to_source(&self) -> Result<Source, String> {
        let directivity = self.directivity.to_directivity()?;
        let crossover = self.crossover.to_filter();

        let source = Source::new(self.position.into(), directivity, self.amplitude)
            .with_name(self.name.clone())
            .with_crossover(crossover);

        Ok(source)
    }
}

/// Directivity pattern configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[derive(Default)]
pub enum DirectivityConfig {
    /// Omnidirectional (spherical) radiation pattern
    #[serde(rename = "omnidirectional")]
    #[default]
    Omnidirectional,
    /// Classical frequency-dependent directional source
    #[serde(rename = "classical")]
    Classical {
        /// Horizontal beamwidth (degrees)
        horizontal_angle: f64,
        /// Vertical beamwidth (degrees)
        vertical_angle: f64,
    },
    /// Custom directivity from measured data
    #[serde(rename = "custom")]
    Custom {
        /// Horizontal angles (degrees)
        horizontal_angles: Vec<f64>,
        /// Vertical angles (degrees)
        vertical_angles: Vec<f64>,
        /// Magnitude data (matrix)
        magnitude: Vec<Vec<f64>>,
    },
    /// Complex, frequency-dependent pattern (speaker BEM export or
    /// measurement; carries directivity phase). The pattern already contains
    /// the full-system crossover and multi-driver interference, so a source
    /// using it should normally keep its own `crossover` at `fullrange`.
    #[serde(rename = "complex_grid")]
    ComplexGrid(ComplexDirectivityGrid),
}

impl DirectivityConfig {
    /// Convert to Directivity
    pub fn to_directivity(&self) -> Result<Directivity, String> {
        match self {
            DirectivityConfig::Omnidirectional => {
                Ok(Directivity::Grid(DirectivityGrid::omnidirectional()))
            }
            DirectivityConfig::Classical {
                horizontal_angle,
                vertical_angle,
            } => Ok(Directivity::Classical {
                h_angle: *horizontal_angle,
                v_angle: *vertical_angle,
            }),
            DirectivityConfig::Custom {
                horizontal_angles,
                vertical_angles,
                magnitude,
            } => {
                use sonium_compute::HostMatrix;

                if magnitude.is_empty() {
                    return Err("Empty magnitude array".to_string());
                }

                let n_vert = magnitude.len();
                let n_horiz = magnitude[0].len();

                if n_vert != vertical_angles.len() {
                    return Err(format!(
                        "Vertical angles mismatch: {} vs {}",
                        n_vert,
                        vertical_angles.len()
                    ));
                }
                if n_horiz != horizontal_angles.len() {
                    return Err(format!(
                        "Horizontal angles mismatch: {} vs {}",
                        n_horiz,
                        horizontal_angles.len()
                    ));
                }
                for (row_idx, row) in magnitude.iter().enumerate() {
                    if row.len() != n_horiz {
                        return Err(format!(
                            "Row {} has {} columns, expected {}",
                            row_idx,
                            row.len(),
                            n_horiz
                        ));
                    }
                }

                let flat: Vec<f64> = magnitude
                    .iter()
                    .flat_map(|row| row.iter().copied())
                    .collect();
                let mag_array = HostMatrix::from_row_major_vec(n_vert, n_horiz, flat);

                DirectivityGrid::try_new(
                    horizontal_angles.clone(),
                    vertical_angles.clone(),
                    mag_array,
                )
                .map(Directivity::Grid)
                .map_err(|error| error.to_string())
            }
            DirectivityConfig::ComplexGrid(grid) => Ok(Directivity::ComplexGrid(grid.clone())),
        }
    }
}

/// Crossover filter configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CrossoverConfig {
    /// Full range (no filter)
    #[serde(rename = "fullrange")]
    #[default]
    FullRange,
    /// Low-pass filter
    #[serde(rename = "lowpass")]
    Lowpass {
        /// Cutoff frequency (Hz)
        cutoff_freq: f64,
        /// Filter order (e.g., 2, 4)
        order: u32,
    },
    /// High-pass filter
    #[serde(rename = "highpass")]
    Highpass {
        /// Cutoff frequency (Hz)
        cutoff_freq: f64,
        /// Filter order (e.g., 2, 4)
        order: u32,
    },
    /// Band-pass filter
    #[serde(rename = "bandpass")]
    Bandpass {
        /// Low cutoff frequency (Hz)
        low_cutoff: f64,
        /// High cutoff frequency (Hz)
        high_cutoff: f64,
        /// Filter order
        order: u32,
    },
}

impl CrossoverConfig {
    /// Convert to CrossoverFilter
    pub fn to_filter(&self) -> CrossoverFilter {
        match self {
            CrossoverConfig::FullRange => CrossoverFilter::FullRange,
            CrossoverConfig::Lowpass { cutoff_freq, order } => CrossoverFilter::Lowpass {
                cutoff_freq: *cutoff_freq,
                order: *order,
            },
            CrossoverConfig::Highpass { cutoff_freq, order } => CrossoverFilter::Highpass {
                cutoff_freq: *cutoff_freq,
                order: *order,
            },
            CrossoverConfig::Bandpass {
                low_cutoff,
                high_cutoff,
                order,
            } => CrossoverFilter::Bandpass {
                low_cutoff: *low_cutoff,
                high_cutoff: *high_cutoff,
                order: *order,
            },
        }
    }
}

/// Frequency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyConfig {
    /// Minimum frequency (Hz)
    pub min_freq: f64,
    /// Maximum frequency (Hz)
    pub max_freq: f64,
    /// Number of frequency points
    pub num_points: usize,
    /// Spacing type ("logarithmic" or "linear")
    #[serde(default = "default_spacing")]
    pub spacing: String,
}

fn default_spacing() -> String {
    "logarithmic".to_string()
}

impl FrequencyConfig {
    /// Return validated typed frequency bounds while retaining the legacy
    /// scalar fields used by the JSON schema.
    pub fn typed_bounds(&self) -> Result<(FrequencyHz, FrequencyHz), ConfigValidationError> {
        let min = FrequencyHz::try_new(self.min_freq)
            .map_err(|reason| invalid("frequencies.min_freq", reason))?;
        let max = FrequencyHz::try_new(self.max_freq)
            .map_err(|reason| invalid("frequencies.max_freq", reason))?;
        Ok((min, max))
    }

    /// Validate frequency-grid bounds and spacing policy.
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        finite_positive("frequencies.min_freq", self.min_freq)?;
        finite_positive("frequencies.max_freq", self.max_freq)?;
        if self.max_freq < self.min_freq {
            return Err(invalid(
                "frequencies",
                "max_freq must be greater than or equal to min_freq",
            ));
        }
        if self.num_points == 0 {
            return Err(invalid("frequencies.num_points", "must be non-zero"));
        }
        // A one-point sweep has an unambiguous meaning only when both bounds
        // name that same point.  Conversely, equal bounds must not be
        // repeated into a multi-point grid: duplicate frequencies make
        // interpolation and sweep bookkeeping ambiguous.
        if self.num_points == 1 && self.min_freq != self.max_freq {
            return Err(invalid(
                "frequencies",
                "a one-point grid requires min_freq to equal max_freq",
            ));
        }
        if self.min_freq == self.max_freq && self.num_points != 1 {
            return Err(invalid(
                "frequencies",
                "equal bounds require exactly one frequency point",
            ));
        }
        if !matches!(
            self.spacing.to_ascii_lowercase().as_str(),
            "linear" | "logarithmic"
        ) {
            return Err(invalid(
                "frequencies.spacing",
                "must be 'linear' or 'logarithmic'",
            ));
        }
        let generated = self.generate_frequencies();
        if generated.iter().any(|frequency| !frequency.is_finite()) {
            return Err(invalid(
                "frequencies",
                "generated grid must contain only finite values",
            ));
        }
        if generated.windows(2).any(|pair| pair[1] <= pair[0]) {
            return Err(invalid(
                "frequencies",
                "generated grid must be strictly increasing",
            ));
        }
        Ok(())
    }

    /// Generate frequency array based on configuration
    pub fn generate_frequencies(&self) -> Vec<f64> {
        if self.spacing.eq_ignore_ascii_case("linear") {
            crate::types::lin_space(self.min_freq, self.max_freq, self.num_points)
        } else {
            log_space(self.min_freq, self.max_freq, self.num_points)
        }
    }

    /// Generate a validated frequency grid with the exchange quantity type.
    ///
    /// `generate_frequencies` remains as a compatibility API for existing
    /// consumers. New solver/application boundaries should use this method so
    /// the positivity and finiteness contract is retained after interpolation
    /// rather than checked only on the two endpoints.
    pub fn validated_frequencies(&self) -> Result<Vec<FrequencyHz>, ConfigValidationError> {
        self.validate()?;
        self.generate_frequencies()
            .into_iter()
            .enumerate()
            .map(|(index, frequency)| {
                FrequencyHz::try_new(frequency)
                    .map_err(|reason| invalid(format!("frequencies[{index}]"), reason))
            })
            .collect()
    }
}

/// Solver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SolverConfig {
    /// Solver method
    pub method: String,
    /// Mesh resolution (elements per meter)
    pub mesh_resolution: usize,
    /// GMRES parameters
    pub gmres: GmresConfig,
    /// ILU preconditioner parameters
    pub ilu: IluConfig,
    /// FMM parameters
    pub fmm: FmmConfig,
    /// Adaptive integration
    pub adaptive_integration: bool,
    /// Adaptive mesh refinement
    pub adaptive_meshing: Option<bool>,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            method: default_method(),
            mesh_resolution: default_mesh_resolution(),
            gmres: GmresConfig::default(),
            ilu: IluConfig::default(),
            fmm: FmmConfig::default(),
            adaptive_integration: default_adaptive_integration(),
            adaptive_meshing: None,
        }
    }
}

fn default_method() -> String {
    "direct".to_string()
}

fn default_mesh_resolution() -> usize {
    2
}

fn default_adaptive_integration() -> bool {
    false
}

/// GMRES solver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GmresConfig {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Restart parameter
    pub restart: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for GmresConfig {
    fn default() -> Self {
        Self {
            max_iter: default_max_iter(),
            restart: default_restart(),
            tolerance: default_tolerance(),
        }
    }
}

impl GmresConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.max_iter == 0 || self.restart == 0 {
            return Err(invalid(
                "solver.gmres",
                "max_iter and restart must be non-zero",
            ));
        }
        finite_positive("solver.gmres.tolerance", self.tolerance)
    }
}

fn default_max_iter() -> usize {
    100
}

fn default_restart() -> usize {
    50
}

fn default_tolerance() -> f64 {
    1e-6
}

/// ILU preconditioner configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IluConfig {
    /// ILU method (e.g., "tbem", "slfmm")
    pub method: String,
    /// Scanning degree (e.g., "fine", "coarse")
    pub scanning_degree: String,
    /// Whether to use hierarchical preconditioning
    pub use_hierarchical: bool,
}

impl Default for IluConfig {
    fn default() -> Self {
        Self {
            method: default_ilu_method(),
            scanning_degree: default_scanning_degree(),
            use_hierarchical: false,
        }
    }
}

fn default_ilu_method() -> String {
    "tbem".to_string()
}

fn default_scanning_degree() -> String {
    "fine".to_string()
}

/// FMM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FmmConfig {
    /// FMM type (e.g., "slfmm", "mlfmm")
    pub fmm_type: String,
    /// Expansion order
    pub expansion_order: usize,
    /// Maximum particles per leaf in the octree
    pub max_particles_per_leaf: usize,
}

impl Default for FmmConfig {
    fn default() -> Self {
        Self {
            fmm_type: default_fmm_type(),
            expansion_order: default_expansion_order(),
            max_particles_per_leaf: default_max_particles(),
        }
    }
}

impl FmmConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.expansion_order == 0 || self.max_particles_per_leaf == 0 {
            return Err(invalid(
                "solver.fmm",
                "expansion_order and max_particles_per_leaf must be non-zero",
            ));
        }
        if !matches!(
            self.fmm_type.to_ascii_lowercase().as_str(),
            "slfmm" | "mlfmm"
        ) {
            return Err(invalid("solver.fmm.fmm_type", "must be 'slfmm' or 'mlfmm'"));
        }
        Ok(())
    }
}

impl SolverConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.method.trim().is_empty() {
            return Err(invalid("solver.method", "must not be empty"));
        }
        if self.mesh_resolution == 0 {
            return Err(invalid("solver.mesh_resolution", "must be non-zero"));
        }
        self.gmres.validate()?;
        self.fmm.validate()
    }
}

fn default_fmm_type() -> String {
    "slfmm".to_string()
}

fn default_expansion_order() -> usize {
    6
}

fn default_max_particles() -> usize {
    50
}

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VisualizationConfig {
    /// Whether to generate field slices
    pub generate_slices: bool,
    /// Resolution of the slices (points per side)
    pub slice_resolution: usize,
    /// Indices of frequencies to visualize (empty for all)
    pub slice_frequency_indices: Vec<usize>,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            generate_slices: default_generate_slices(),
            slice_resolution: default_slice_resolution(),
            slice_frequency_indices: Vec::new(),
        }
    }
}

impl VisualizationConfig {
    /// Validate settings used by spatial output generation.
    pub fn validate(&self, frequency_count: usize) -> Result<(), ConfigValidationError> {
        if self.generate_slices && self.slice_resolution < 2 {
            return Err(invalid(
                "visualization.slice_resolution",
                "must be at least two when slices are enabled",
            ));
        }
        if self
            .slice_frequency_indices
            .iter()
            .any(|&index| index >= frequency_count)
        {
            return Err(invalid(
                "visualization.slice_frequency_indices",
                "contains an index outside the configured frequency grid",
            ));
        }
        Ok(())
    }
}

fn default_generate_slices() -> bool {
    false
}

fn default_slice_resolution() -> usize {
    50
}

/// Simulation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MetadataConfig {
    /// Simulation description
    pub description: String,
    /// Author name
    pub author: String,
    /// Simulation date
    pub date: String,
}

impl Default for MetadataConfig {
    fn default() -> Self {
        Self {
            description: String::new(),
            author: String::new(),
            date: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        }
    }
}

impl RoomConfig {
    /// Parse a JSON document, rejecting unknown fields at the exchange root.
    ///
    /// Serde intentionally ignores unknown fields by default. That behaviour
    /// is unsafe for a versioned exchange format because a producer can typo a
    /// field and still obtain a numerically different simulation. The nested
    /// structures remain forward-compatible, while the root contract is
    /// strict and versioned.
    pub fn from_json_checked(json: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let value = crate::io::external::parse_json_reject_duplicate_keys(json)?;
        // Validate the complete published contract before serde constructs
        // nested structs. `deny_unknown_fields` protects the root model, but
        // serde's default for nested structs would otherwise silently drop a
        // misspelled boundary, source, or solver field.
        if let Some(version) = value.get("schema_version").and_then(|value| value.as_u64())
            && version != u64::from(CURRENT_SCHEMA_VERSION)
        {
            return Err(Box::new(ConfigValidationError::UnsupportedSchemaVersion(
                u32::try_from(version).unwrap_or(u32::MAX),
            )));
        }
        crate::io::validate_schema_envelope(&crate::schema::room_config_json_schema(), &value)?;
        let config: Self = serde_json::from_value(value)?;
        config.validate()?;
        Ok(config)
    }

    /// Parse a JSON document, apply the registered migration, then validate it.
    pub fn from_json_migrated(
        json: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let value = crate::io::external::parse_json_reject_duplicate_keys(json)?;
        crate::schema::reject_unknown_room_config_fields(&value)?;
        // Version 0 has the same room payload shape and only needs the
        // explicit version migration below. Validate it as the current
        // envelope while retaining the original value for `migrate()`.
        let mut schema_value = value.clone();
        let version = schema_value
            .get("schema_version")
            .and_then(serde_json::Value::as_u64);
        if version.is_none()
            || version == Some(0)
            || version == Some(u64::from(CURRENT_SCHEMA_VERSION))
        {
            if version == Some(0) {
                schema_value["schema_version"] = serde_json::Value::from(CURRENT_SCHEMA_VERSION);
            }
            crate::io::validate_schema_envelope(
                &crate::schema::room_config_json_schema(),
                &schema_value,
            )?;
        }
        let config: Self = serde_json::from_value(value)?;
        let config = config.migrate()?;
        config.validate()?;
        Ok(config)
    }

    /// Migrate a document version for which a compatibility transform exists.
    ///
    /// Version 0 had the same field layout but no explicit version marker;
    /// accepting it here makes the migration policy executable and keeps the
    /// normal validator strict for unknown future versions.
    pub fn migrate(mut self) -> Result<Self, ConfigValidationError> {
        match self.schema_version.0 {
            0 => {
                self.schema_version = SchemaVersion::current();
                Ok(self)
            }
            CURRENT_SCHEMA_VERSION => Ok(self),
            version => Err(ConfigValidationError::UnsupportedSchemaVersion(version)),
        }
    }

    /// Validate the complete exchange document before solver entry.
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.schema_version.0 != CURRENT_SCHEMA_VERSION {
            return Err(ConfigValidationError::UnsupportedSchemaVersion(
                self.schema_version.0,
            ));
        }
        self.room.validate()?;
        if self.sources.is_empty() {
            return Err(invalid("sources", "must contain at least one source"));
        }
        if self.listening_positions.is_empty() {
            return Err(invalid(
                "listening_positions",
                "must contain at least one position",
            ));
        }
        for (index, source) in self.sources.iter().enumerate() {
            source.validate(index)?;
        }
        for (index, point) in self.listening_positions.iter().enumerate() {
            finite(&format!("listening_positions[{index}].x"), point.x)?;
            finite(&format!("listening_positions[{index}].y"), point.y)?;
            finite(&format!("listening_positions[{index}].z"), point.z)?;
        }
        self.frequencies.validated_frequencies()?;
        self.boundaries.validate()?;
        finite_positive("medium.density", self.medium.density)?;
        finite_positive("medium.speed_of_sound", self.medium.speed_of_sound)?;
        self.solver.validate()?;
        self.visualization.validate(self.frequencies.num_points)?;
        Ok(())
    }

    /// Load and validate a configuration with a typed error.
    pub fn from_file_checked<P: AsRef<Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let contents = fs::read_to_string(path)?;
        Self::from_json_checked(&contents)
    }

    /// Load a configuration and apply registered schema migrations.
    pub fn from_file_migrated<P: AsRef<Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let contents = fs::read_to_string(path)?;
        Self::from_json_migrated(&contents)
    }

    /// Load configuration from JSON file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        Self::from_file_checked(path).map_err(|error| format!("Failed to load config: {error}"))
    }

    /// Serialize a validated configuration in a stable compact JSON form.
    ///
    /// Struct fields have a fixed declaration order and map-like IR payloads
    /// use ordered maps, so repeated serialization of the same value produces
    /// byte-identical output suitable for fixtures and cache keys.
    pub fn to_json_deterministic(&self) -> Result<String, String> {
        self.validate()
            .map_err(|error| format!("Failed to validate config: {error}"))?;
        serde_json::to_string(self).map_err(|error| format!("Failed to serialize config: {error}"))
    }

    /// Save configuration to JSON file
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        self.validate()
            .map_err(|error| format!("Failed to validate config: {error}"))?;
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;

        fs::write(path, json).map_err(|e| format!("Failed to write config file: {}", e))?;

        Ok(())
    }

    /// Convert to RoomSimulation
    pub fn to_simulation(&self) -> Result<RoomSimulation, String> {
        self.validate().map_err(|error| error.to_string())?;
        let room = self.room.to_geometry()?;

        let sources: Vec<Source> = self
            .sources
            .iter()
            .map(|s| s.to_source())
            .collect::<Result<Vec<_>, _>>()?;

        let listening_positions: Vec<Point3D> =
            self.listening_positions.iter().map(|&p| p.into()).collect();

        let frequencies = self
            .frequencies
            .validated_frequencies()
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(FrequencyHz::get)
            .collect();

        Ok(RoomSimulation {
            room,
            sources,
            listening_positions,
            frequencies,
            boundaries: self.boundaries.clone(), // Pass boundaries
            speed_of_sound: self.medium.speed_of_sound,
        })
    }
}

/// Room acoustics simulation configuration
#[derive(Debug, Clone)]
pub struct RoomSimulation {
    /// Room geometry
    pub room: RoomGeometry,
    /// Sound sources
    pub sources: Vec<Source>,
    /// Listening positions
    pub listening_positions: Vec<Point3D>,
    /// Frequencies to simulate
    pub frequencies: Vec<f64>,
    /// Boundary conditions
    pub boundaries: BoundaryConfig,
    /// Speed of sound (m/s)
    pub speed_of_sound: f64,
}

impl RoomSimulation {
    /// Create a new simulation with default frequency range
    pub fn new(
        room: RoomGeometry,
        sources: Vec<Source>,
        listening_positions: Vec<Point3D>,
    ) -> Self {
        let frequencies = log_space(20.0, 20000.0, 200);

        Self {
            room,
            sources,
            listening_positions,
            frequencies,
            boundaries: BoundaryConfig::default(),
            speed_of_sound: crate::types::constants::SPEED_OF_SOUND_20C,
        }
    }

    /// Create simulation with custom frequency configuration
    pub fn with_frequencies(
        room: RoomGeometry,
        sources: Vec<Source>,
        listening_positions: Vec<Point3D>,
        min_freq: f64,
        max_freq: f64,
        num_points: usize,
    ) -> Self {
        let frequencies = log_space(min_freq, max_freq, num_points);

        Self {
            room,
            sources,
            listening_positions,
            frequencies,
            boundaries: BoundaryConfig::default(),
            speed_of_sound: crate::types::constants::SPEED_OF_SOUND_20C,
        }
    }

    /// Calculate wavenumber k = 2πf/c
    pub fn wavenumber(&self, frequency: f64) -> f64 {
        crate::types::wavenumber(frequency, self.speed_of_sound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_config() {
        let config = FrequencyConfig {
            min_freq: 20.0,
            max_freq: 20000.0,
            num_points: 100,
            spacing: "logarithmic".to_string(),
        };
        let freqs = config.generate_frequencies();
        assert_eq!(freqs.len(), 100);

        let typed = config.validated_frequencies().unwrap();
        assert_eq!(typed.len(), 100);
        assert_eq!(typed.first().unwrap().get(), freqs[0]);
        assert!(typed.iter().all(|frequency| frequency.get().is_finite()));
    }

    #[test]
    fn frequency_config_has_explicit_single_point_semantics() {
        let config = FrequencyConfig {
            min_freq: 1_000.0,
            max_freq: 1_000.0,
            num_points: 1,
            spacing: "linear".to_string(),
        };
        config.validate().expect("single-point frequency is valid");
        assert_eq!(config.generate_frequencies(), vec![1_000.0]);
        assert_eq!(config.validated_frequencies().unwrap().len(), 1);

        let mut invalid = config.clone();
        invalid.max_freq = 2_000.0;
        assert!(matches!(
            invalid.validate(),
            Err(ConfigValidationError::InvalidField { field, .. }) if field == "frequencies"
        ));

        invalid = config;
        invalid.num_points = 2;
        assert!(matches!(
            invalid.validate(),
            Err(ConfigValidationError::InvalidField { field, .. }) if field == "frequencies"
        ));
    }

    #[test]
    fn frequency_config_rejects_non_monotonic_generated_grids() {
        let config = FrequencyConfig {
            min_freq: 1.0,
            max_freq: 1.0 + f64::EPSILON,
            num_points: 3,
            spacing: "linear".to_string(),
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigValidationError::InvalidField { field, .. }) if field == "frequencies"
        ));
    }

    #[test]
    fn test_room_geometry_config() {
        let config = RoomGeometryConfig::Rectangular {
            width: 5.0,
            depth: 4.0,
            height: 2.5,
        };
        let geometry = config.to_geometry().unwrap();
        let (w, d, h) = geometry.dimensions();
        assert_eq!(w, 5.0);
        assert_eq!(d, 4.0);
        assert_eq!(h, 2.5);
    }

    #[test]
    fn test_admittance_surface_roundtrip() {
        let surface = SurfaceConfig::Admittance {
            real: 0.12,
            imag: -0.03,
        };

        let json = serde_json::to_string(&surface).unwrap();
        let decoded: SurfaceConfig = serde_json::from_str(&json).unwrap();

        match decoded {
            SurfaceConfig::Admittance { real, imag } => {
                assert_eq!(real, 0.12);
                assert_eq!(imag, -0.03);
            }
            other => panic!("expected admittance surface, got {other:?}"),
        }
    }

    #[test]
    fn test_porous_layer_surface_roundtrip() {
        let surface = SurfaceConfig::PorousLayer {
            model: PorousModelConfig::Miki,
            thickness: 0.08,
            flow_resistivity: 12_000.0,
            air_gap: Some(0.05),
            ambient: AmbientMediumConfig {
                density: 1.18,
                speed_of_sound: 344.0,
            },
        };

        let json = serde_json::to_string(&surface).unwrap();
        let decoded: SurfaceConfig = serde_json::from_str(&json).unwrap();

        match decoded {
            SurfaceConfig::PorousLayer {
                model,
                thickness,
                flow_resistivity,
                air_gap,
                ambient,
            } => {
                assert_eq!(model, PorousModelConfig::Miki);
                assert_eq!(thickness, 0.08);
                assert_eq!(flow_resistivity, 12_000.0);
                assert_eq!(air_gap, Some(0.05));
                assert_eq!(ambient.density, 1.18);
                assert_eq!(ambient.speed_of_sound, 344.0);
            }
            other => panic!("expected porous layer surface, got {other:?}"),
        }
    }

    #[test]
    fn test_complex_grid_directivity_config_roundtrip() {
        // Room scenario JSON carrying a speaker-exported complex pattern.
        let json = r#"{
            "name": "Left Main",
            "position": {"x": 0.5, "y": 0.3, "z": 1.1},
            "amplitude": 1.0,
            "directivity": {
                "type": "complex_grid",
                "horizontal_angles": [0.0, 90.0, 180.0, 270.0],
                "vertical_angles": [0.0, 90.0, 180.0],
                "frequencies": [500.0, 1000.0],
                "patterns": [
                    [
                        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                        [[0.3, 0.0], [1.0, 0.0], [0.3, 0.0], [0.4, 0.1]],
                        [[0.2, 0.0], [0.2, 0.0], [0.2, 0.0], [0.2, 0.0]]
                    ],
                    [
                        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                        [[0.2, 0.0], [1.0, 0.0], [0.2, 0.0], [0.3, 0.2]],
                        [[0.1, 0.0], [0.1, 0.0], [0.1, 0.0], [0.1, 0.0]]
                    ]
                ]
            }
        }"#;

        let config: SourceConfig = serde_json::from_str(json).unwrap();
        let source = config.to_source().expect("complex grid source parses");
        let Directivity::ComplexGrid(grid) = &source.directivity else {
            panic!("expected complex grid directivity");
        };
        assert_eq!(grid.frequencies(), &[500.0, 1000.0]);
        // Default orientation (world +Y forward): the forward lobe at
        // (theta=90°, phi=90°) is exactly 1+0i, the rear is attenuated.
        let forward = source.complex_amplitude_towards(&Point3D::new(0.5, 1.3, 1.1), 1000.0);
        let rear = source.complex_amplitude_towards(&Point3D::new(0.5, -0.7, 1.1), 1000.0);
        assert!((forward - num_complex::Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!(rear.norm() < 0.5, "rear lobe should be attenuated: {rear}");

        // Serde round trip through the config enum keeps the variant.
        let encoded = serde_json::to_string(&config.directivity).unwrap();
        assert!(encoded.contains("\"type\":\"complex_grid\""), "{encoded}");
        let decoded: DirectivityConfig = serde_json::from_str(&encoded).unwrap();
        assert!(matches!(decoded, DirectivityConfig::ComplexGrid(_)));
    }

    #[test]
    fn room_config_defaults_schema_and_round_trips_conventions() {
        let config = crate::output::create_default_config();
        config.validate().expect("built-in config is valid");
        let json = serde_json::to_string(&config).expect("serialize config");
        assert!(json.contains("schema_version"));
        let decoded: RoomConfig = serde_json::from_str(&json).expect("deserialize config");
        assert_eq!(decoded.schema_version, SchemaVersion::current());
        assert_eq!(decoded.conventions, ConventionRegistry::default());

        let first = config
            .to_json_deterministic()
            .expect("deterministic serialization");
        let second = config
            .to_json_deterministic()
            .expect("deterministic serialization");
        assert_eq!(first, second);
        RoomConfig::from_json_checked(&first).expect("deterministic output is readable");
    }

    #[test]
    fn legacy_scalar_config_exposes_typed_quantity_views() {
        let config = crate::output::create_default_config();
        let (min, max) = config
            .frequencies
            .typed_bounds()
            .expect("default frequency bounds are typed-valid");
        assert_eq!(min.get(), config.frequencies.min_freq);
        assert_eq!(max.get(), config.frequencies.max_freq);

        let ambient = AmbientMediumConfig::default();
        assert_eq!(ambient.typed_density().unwrap().get(), ambient.density);
        assert_eq!(
            ambient.typed_speed_of_sound().unwrap().get(),
            ambient.speed_of_sound
        );

        let invalid = AmbientMediumConfig {
            density: 0.0,
            ..ambient
        };
        assert!(matches!(
            invalid.typed_density(),
            Err(ConfigValidationError::InvalidField { field, .. })
                if field == "boundaries.ambient.density"
        ));
    }

    #[test]
    fn room_config_rejects_unknown_version_and_invalid_grid() {
        let mut config = crate::output::create_default_config();
        config.schema_version = SchemaVersion(CURRENT_SCHEMA_VERSION + 1);
        assert!(matches!(
            config.validate(),
            Err(ConfigValidationError::UnsupportedSchemaVersion(_))
        ));
        config.schema_version = SchemaVersion(0);
        assert_eq!(
            config.clone().migrate().unwrap().schema_version,
            SchemaVersion::current()
        );

        config.schema_version = SchemaVersion::current();
        config.frequencies.min_freq = f64::NAN;
        assert!(matches!(
            config.validate(),
            Err(ConfigValidationError::InvalidField { .. })
        ));
    }

    #[test]
    fn room_config_rejects_empty_listeners_and_invalid_slice_settings() {
        let mut config = crate::output::create_default_config();
        config.listening_positions.clear();
        assert!(matches!(
            config.validate(),
            Err(ConfigValidationError::InvalidField { field, .. })
                if field == "listening_positions"
        ));

        let mut config = crate::output::create_default_config();
        config.visualization.generate_slices = true;
        config.visualization.slice_resolution = 1;
        assert!(matches!(
            config.validate(),
            Err(ConfigValidationError::InvalidField { field, .. })
                if field == "visualization.slice_resolution"
        ));

        config.visualization.slice_resolution = 2;
        config.visualization.slice_frequency_indices = vec![config.frequencies.num_points];
        assert!(matches!(
            config.validate(),
            Err(ConfigValidationError::InvalidField { field, .. })
                if field == "visualization.slice_frequency_indices"
        ));
    }

    #[test]
    fn room_config_rejects_unknown_top_level_fields() {
        let json = r#"{
            "room": {"type":"rectangular", "width":5.0, "depth":4.0, "height":2.5},
            "sources": [], "listening_positions": [],
            "frequencies": {"min_freq":20.0,"max_freq":200.0,"num_points":2},
            "unexpected": true
        }"#;
        assert!(serde_json::from_str::<RoomConfig>(json).is_err());
    }

    #[test]
    fn checked_reader_rejects_unknown_nested_fields_before_deserialization() {
        let mut value: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/room-config-v1.json"))
                .expect("fixture is JSON");
        value["boundaries"]["front_wall"] = serde_json::json!({
            "type": "rigid",
            "typo": true
        });

        let error = RoomConfig::from_json_checked(&value.to_string())
            .expect_err("nested unknown fields must not be discarded");
        assert!(error.to_string().contains("front_wall.typo"), "{error}");
    }

    #[test]
    fn published_golden_fixture_round_trips_through_typed_and_external_readers() {
        let text = include_str!("../fixtures/room-config-v1.json");
        let typed = RoomConfig::from_json_checked(text).expect("published fixture is typed-valid");
        typed
            .validate()
            .expect("published fixture is semantically valid");

        let encoded = serde_json::to_value(&typed).expect("typed config serializes");
        let external = crate::io::read_external_room_value_checked(encoded.clone())
            .expect("external reader accepts typed producer output");
        assert_eq!(external.schema_version, 1);
        assert_eq!(external.sources().len(), typed.sources.len());
        assert_eq!(
            external.listening_positions().len(),
            typed.listening_positions.len()
        );

        let reparsed = RoomConfig::from_json_checked(
            &serde_json::to_string(&encoded).expect("encoded config is JSON"),
        )
        .expect("typed reader accepts its canonical output");
        assert_eq!(reparsed.schema_version, typed.schema_version);
        assert_eq!(reparsed.sources.len(), typed.sources.len());
        assert_eq!(
            reparsed.listening_positions.len(),
            typed.listening_positions.len()
        );
        assert_eq!(reparsed.frequencies.min_freq, typed.frequencies.min_freq);
        assert_eq!(reparsed.frequencies.max_freq, typed.frequencies.max_freq);
        assert_eq!(
            reparsed.frequencies.num_points,
            typed.frequencies.num_points
        );
        assert_eq!(reparsed.frequencies.spacing, typed.frequencies.spacing);
    }

    #[test]
    fn boundary_validation_rejects_unphysical_surfaces() {
        assert!(BoundaryConfig::default().validate().is_ok());
        let bad_alpha = SurfaceConfig::Absorption { coefficient: 5.0 };
        assert!(matches!(
            bad_alpha.validate("boundaries.floor"),
            Err(ConfigValidationError::InvalidField { .. })
        ));
        let bad_layer = SurfaceConfig::PorousLayer {
            model: PorousModelConfig::Miki,
            thickness: 0.0,
            flow_resistivity: -10.0,
            air_gap: None,
            ambient: AmbientMediumConfig::default(),
        };
        assert!(bad_layer.validate("boundaries.walls").is_err());
        let mut config: RoomConfig =
            serde_json::from_str(include_str!("../fixtures/room-config-v1.json"))
                .expect("fixture is JSON");
        config.boundaries.floor = SurfaceConfig::Absorption { coefficient: 5.0 };
        assert!(config.validate().is_err());
    }

    #[test]
    fn l_shaped_room_rejects_non_notched_footprint() {
        let valid = RoomGeometryConfig::LShaped {
            width1: 5.0,
            depth1: 4.0,
            width2: 3.0,
            depth2: 3.0,
            height: 2.5,
        };
        assert!(valid.validate().is_ok());
        for width2 in [5.0, 6.0] {
            let inverted = RoomGeometryConfig::LShaped {
                width1: 5.0,
                depth1: 4.0,
                width2,
                depth2: 3.0,
                height: 2.5,
            };
            assert!(
                inverted.validate().is_err(),
                "width2 = {width2} must be rejected"
            );
        }
    }

    #[test]
    fn to_simulation_propagates_room_medium_sound_speed() {
        let mut config: RoomConfig =
            serde_json::from_str(include_str!("../fixtures/room-config-v1.json"))
                .expect("fixture is JSON");
        config.medium.speed_of_sound = 344.0;
        config.medium.density = 1.2;
        let sim = config.to_simulation().expect("config is valid");
        assert_eq!(sim.speed_of_sound, 344.0);
        let default: RoomConfig =
            serde_json::from_str(include_str!("../fixtures/room-config-v1.json"))
                .expect("fixture is JSON");
        let default_sim = default.to_simulation().expect("config is valid");
        assert_eq!(
            default_sim.speed_of_sound,
            AmbientMediumConfig::default().speed_of_sound
        );
    }
}
