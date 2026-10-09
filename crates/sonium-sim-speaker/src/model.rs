//! Parametric loudspeaker model.
//!
//! JSON-serializable description of a sealed-box loudspeaker with woofer,
//! tweeter, spline-parametrised waveguide, and internal separator plate.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_bem::materials::{AmbientMedium, PorousLayer};
use sonium_sim_speaker_modeling::{AcousticMedium, DriverModel, ProductDesign};

// ---------------------------------------------------------------------------
// Acoustic impedance constants
// ---------------------------------------------------------------------------

/// Specific acoustic impedance of MDF (Pa·s/m).
pub const Z_MDF: f64 = 3.2e6;

/// Specific acoustic impedance of hard plastic (Pa·s/m).
pub const Z_HARD_PLASTIC: f64 = 2.5e6;

/// Compute complex admittance Y = ρc / Z for a material impedance.
pub fn admittance_for_material(
    z_material: f64,
    air_density: f64,
    speed_of_sound: f64,
) -> Complex64 {
    let rho_c = air_density * speed_of_sound;
    Complex64::new(rho_c / z_material, 0.0)
}

/// Frequency-dependent acoustic material used for speaker boundary admittance.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SpeakerMaterial {
    /// Constant specific acoustic impedance in Pa*s/m.
    ConstantImpedance { specific_impedance: f64 },
    /// Rigid-backed porous layer evaluated with the shared BEM material model.
    PorousLayer { layer: PorousLayer },
}

impl SpeakerMaterial {
    /// MDF cabinet-wall default.
    pub fn mdf() -> Self {
        Self::ConstantImpedance {
            specific_impedance: Z_MDF,
        }
    }

    /// Hard-plastic waveguide default.
    pub fn hard_plastic() -> Self {
        Self::ConstantImpedance {
            specific_impedance: Z_HARD_PLASTIC,
        }
    }

    /// Normalized admittance, `rho*c / Zs`, for the material at `frequency_hz`.
    /// Panics on invalid input; use [`Self::try_admittance_at`] in fallible paths.
    pub fn admittance_at(
        self,
        frequency_hz: f64,
        air_density: f64,
        speed_of_sound: f64,
    ) -> Complex64 {
        self.try_admittance_at(frequency_hz, air_density, speed_of_sound)
            .expect("invalid speaker material admittance")
    }

    /// Checked normalized admittance, rejecting invalid parameters and overflow.
    pub fn try_admittance_at(
        self,
        frequency_hz: f64,
        air_density: f64,
        speed_of_sound: f64,
    ) -> Result<Complex64, String> {
        self.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err("material frequency must be finite and positive".into());
        }
        let medium = AmbientMedium::try_new(air_density, speed_of_sound)
            .map_err(|error| error.to_string())?;
        let value = match self {
            SpeakerMaterial::ConstantImpedance { specific_impedance } => {
                admittance_for_material(specific_impedance, air_density, speed_of_sound)
            }
            SpeakerMaterial::PorousLayer { layer } => layer
                .try_normalized_admittance(frequency_hz, medium)
                .map_err(|error| error.to_string())?,
        };
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err("speaker material admittance is nonfinite".into());
        }
        Ok(value)
    }

    fn validate(self) -> Result<(), String> {
        match self {
            SpeakerMaterial::ConstantImpedance { specific_impedance } => {
                if !specific_impedance.is_finite() || specific_impedance <= 0.0 {
                    Err(format!(
                        "specific acoustic impedance must be finite and positive, got {specific_impedance}",
                    ))
                } else {
                    Ok(())
                }
            }
            SpeakerMaterial::PorousLayer { layer } => layer.validate(),
        }
    }
}

/// Material assignment for non-driver speaker surfaces.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpeakerMaterialAssignment {
    /// Cabinet and baffle surface material.
    #[serde(default = "SpeakerMaterial::mdf")]
    pub cabinet: SpeakerMaterial,
    /// Waveguide surface material.
    #[serde(default = "SpeakerMaterial::hard_plastic")]
    pub waveguide: SpeakerMaterial,
}

impl Default for SpeakerMaterialAssignment {
    fn default() -> Self {
        Self {
            cabinet: SpeakerMaterial::mdf(),
            waveguide: SpeakerMaterial::hard_plastic(),
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

/// Complete parametric loudspeaker model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoudspeakerModel {
    /// Cabinet outer dimensions.
    pub cabinet: CabinetDimensions,
    /// Woofer driver configuration.
    pub woofer: DriverConfig,
    /// Tweeter driver configuration.
    pub tweeter: DriverConfig,
    /// Additional direct-radiator drivers. The legacy woofer/tweeter fields
    /// remain the first two entries for backwards-compatible JSON and APIs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_drivers: Vec<DriverConfig>,
    /// Optional modeling-owned product contract. When present, its driver
    /// placements must match the mesh drivers and its crossover-derived
    /// velocities drive untyped apertures.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub product_design: Option<ProductDesign>,
    /// Internal separator plate between drivers.
    pub separator: SeparatorConfig,
    /// Tweeter waveguide shape. `None` for direct-radiator (no waveguide).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub waveguide: Option<WaveguideConfig>,
    /// Materials assigned to rigid speaker surfaces.
    #[serde(default)]
    pub materials: SpeakerMaterialAssignment,
    /// Simulation parameters.
    pub simulation: SimulationConfig,
}

// ---------------------------------------------------------------------------
// Cabinet
// ---------------------------------------------------------------------------

/// Outer dimensions of the rectangular cabinet.
///
/// Coordinate convention (origin at cabinet center):
/// - x: width (left/right)
/// - y: height (bottom/top)
/// - z: depth (back/front, acoustic axis = +z)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CabinetDimensions {
    /// Total height (m).
    pub height: f64,
    /// Total width (m).
    pub width: f64,
    /// Total depth (m).
    pub depth: f64,
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Configuration for a single driver (woofer or tweeter).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriverConfig {
    /// Effective piston radius (m).
    pub radius: f64,
    /// Horizontal offset from the cabinet centre (m).
    #[serde(default)]
    pub x_offset: f64,
    /// Vertical offset from cabinet center (m, positive = up).
    /// The driver is centred horizontally (symmetric about x=0).
    pub y_offset: f64,
    /// Default piston velocity amplitude (m/s) used when no transfer
    /// function is provided.
    pub piston_velocity: f64,
    /// Optional frequency-dependent transfer function.
    /// Each entry is `(frequency_hz, magnitude, phase_degrees)`.
    /// Linearly interpolated between points.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transfer_function: Option<Vec<TransferPoint>>,
    /// Typed driver-unit model produced by `sonium-sim-speaker-modeling`.
    ///
    /// When present, its velocity transfer is the acoustic boundary condition;
    /// the legacy `transfer_function` must be absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver_model: Option<DriverModel>,
    /// Optional conical shell geometry. When `None`, the driver is a flat disk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cone: Option<ConeConfig>,
}

/// Conical shell geometry for a driver.
///
/// Models the driver as a shallow truncated cone recessing behind the baffle,
/// from `radius` (at the baffle plane) down to `dustcap_radius` (at the apex).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConeConfig {
    /// How far the cone recesses behind the baffle plane (m).
    pub cone_depth: f64,
    /// Radius of the flat dustcap at the cone apex (m).
    pub dustcap_radius: f64,
}

/// A single point in a driver transfer function.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TransferPoint {
    /// Frequency (Hz).
    pub freq: f64,
    /// Velocity magnitude (m/s).
    pub magnitude: f64,
    /// Phase (degrees).
    pub phase_deg: f64,
}

impl DriverConfig {
    /// Evaluate piston velocity at the given frequency.
    ///
    /// Returns complex velocity: magnitude * exp(j * phase).
    /// If no transfer function is set, returns a flat response
    /// at `self.piston_velocity` with zero phase.
    ///
    /// Interpolation is performed in the complex plane (real/imag
    /// components linearly interpolated). Interpolating magnitude and
    /// phase separately would produce a 359° swing through 0° whenever
    /// adjacent control points straddle ±180°, which is unphysical;
    /// complex interpolation always takes the short path on the unit
    /// circle and never wraps.
    pub fn velocity_at(&self, freq: f64) -> Complex64 {
        if let Some(model) = &self.driver_model {
            // DriverModel uses RMS phasors; the exterior BEM uses peak
            // boundary amplitudes.
            return std::f64::consts::SQRT_2 * model.velocity_at(freq);
        }
        match &self.transfer_function {
            None => Complex64::new(self.piston_velocity, 0.0),
            Some(pts) if pts.is_empty() => Complex64::new(self.piston_velocity, 0.0),
            Some(pts) => {
                // Clamp to range
                if freq <= pts[0].freq {
                    return polar(pts[0].magnitude, pts[0].phase_deg);
                }
                if freq >= pts[pts.len() - 1].freq {
                    let last = pts[pts.len() - 1];
                    return polar(last.magnitude, last.phase_deg);
                }
                // Linear interpolation in the complex plane.
                for w in pts.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    if freq >= a.freq && freq <= b.freq {
                        if !b.freq.is_finite() || !a.freq.is_finite() || b.freq <= a.freq {
                            return polar(a.magnitude, a.phase_deg);
                        }
                        let t = (freq - a.freq) / (b.freq - a.freq);
                        let ca = polar(a.magnitude, a.phase_deg);
                        let cb = polar(b.magnitude, b.phase_deg);
                        return ca + (cb - ca) * t;
                    }
                }
                Complex64::new(self.piston_velocity, 0.0)
            }
        }
    }
}

fn polar(magnitude: f64, phase_deg: f64) -> Complex64 {
    let phase_rad = phase_deg.to_radians();
    Complex64::new(magnitude * phase_rad.cos(), magnitude * phase_rad.sin())
}

// ---------------------------------------------------------------------------
// Separator
// ---------------------------------------------------------------------------

/// Internal separator plate between woofer and tweeter chambers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparatorConfig {
    /// Vertical position of the plate centre (m, relative to cabinet centre).
    pub y_position: f64,
    /// Plate thickness (m).
    pub thickness: f64,
}

// ---------------------------------------------------------------------------
// Waveguide
// ---------------------------------------------------------------------------

/// Tweeter waveguide shape definition.
///
/// The waveguide is an obloid shape around the tweeter aperture,
/// parametrised by two spline profiles (horizontal and vertical).
/// At `t=0` (baffle surface) the spline gives the outer rim radius.
/// At `t=1` (deepest point) the spline gives the tweeter aperture radius.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveguideConfig {
    /// Control points `[t, radius]` for the horizontal profile spline.
    /// `t` ranges from 0 (baffle surface) to 1 (tweeter).
    pub horizontal_spline: Vec<[f64; 2]>,
    /// Control points `[t, radius]` for the vertical profile spline.
    pub vertical_spline: Vec<[f64; 2]>,
    /// How far the waveguide protrudes forward from the baffle (m).
    pub depth: f64,
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

/// Parameters controlling the BEM simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Start frequency (Hz).
    pub freq_start: f64,
    /// End frequency (Hz).
    pub freq_end: f64,
    /// Number of frequency points.
    pub freq_points: usize,
    /// Speed of sound (m/s). Default: 343.0
    #[serde(default = "default_speed_of_sound")]
    pub speed_of_sound: f64,
    /// Air density (kg/m³). Default: 1.21
    #[serde(default = "default_air_density")]
    pub air_density: f64,
    /// Maximum frequency used for mesh sizing (λ/8 at this frequency).
    /// If not set, defaults to `freq_end`.
    #[serde(default)]
    pub max_mesh_frequency: Option<f64>,
}

fn default_speed_of_sound() -> f64 {
    343.0
}
fn default_air_density() -> f64 {
    1.21
}

impl SimulationConfig {
    /// Validate the frequency grid, medium properties, and mesh sizing inputs.
    pub fn validate(&self) -> Result<(), String> {
        validate_positive_finite("simulation start frequency", self.freq_start)?;
        validate_positive_finite("simulation end frequency", self.freq_end)?;
        if self.freq_end < self.freq_start {
            return Err(format!(
                "simulation end frequency ({}) must be at least the start frequency ({})",
                self.freq_end, self.freq_start
            ));
        }
        if self.freq_points == 0 {
            return Err("simulation frequency count must be at least 1".into());
        }
        validate_positive_finite("speed of sound", self.speed_of_sound)?;
        validate_positive_finite("air density", self.air_density)?;
        if let Some(max_mesh_frequency) = self.max_mesh_frequency {
            validate_positive_finite("maximum mesh frequency", max_mesh_frequency)?;
            if max_mesh_frequency < self.freq_end {
                return Err(format!(
                    "maximum mesh frequency ({max_mesh_frequency}) must be at least the simulation end frequency ({})",
                    self.freq_end
                ));
            }
        }
        Ok(())
    }

    /// Target maximum element edge length for the mesh.
    pub fn max_element_size(&self) -> f64 {
        let f_max = self.max_mesh_frequency.unwrap_or(self.freq_end);
        let lambda_min = self.speed_of_sound / f_max;
        lambda_min / 8.0
    }

    /// Generate logarithmically-spaced frequency vector.
    pub fn frequencies(&self) -> Vec<f64> {
        if self.freq_points <= 1 {
            return vec![self.freq_start];
        }
        let log_start = self.freq_start.ln();
        let log_end = self.freq_end.ln();
        (0..self.freq_points)
            .map(|i| {
                let t = i as f64 / (self.freq_points - 1) as f64;
                (log_start + t * (log_end - log_start)).exp()
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

impl LoudspeakerModel {
    /// All driver units in stable mesh/network order.
    pub fn drivers(&self) -> Vec<&DriverConfig> {
        std::iter::once(&self.woofer)
            .chain(std::iter::once(&self.tweeter))
            .chain(self.additional_drivers.iter())
            .collect()
    }

    /// Return the simulation-owned geometry after applying the modeling
    /// product contract.
    ///
    /// `ProductDesign` is the source of truth for product cabinet/driver
    /// geometry. Keeping this lowering at the simulator boundary prevents
    /// direct callers (JSON loaders, QA binaries, and non-optimizer clients)
    /// from silently simulating the legacy cabinet dimensions while carrying
    /// a different product design.
    pub fn with_product_design_geometry(&self) -> Result<Self, String> {
        let Some(product) = &self.product_design else {
            return Ok(self.clone());
        };
        product
            .validate()
            .map_err(|error| format!("product design is invalid: {error}"))?;
        let mut lowered = self.clone();
        let model_driver_count = lowered.drivers().len();
        if model_driver_count != product.drivers.len() {
            return Err(format!(
                "product design has {} drivers but speaker model has {}",
                product.drivers.len(),
                model_driver_count
            ));
        }
        lowered.cabinet.width = product.cabinet.baffle_width_m;
        lowered.cabinet.height = product.cabinet.baffle_height_m;
        lowered.cabinet.depth = product.cabinet.box_depth_m;
        lowered.waveguide = match &product.waveguide {
            None => None,
            Some(waveguide) => {
                let Some(tweeter) = product.drivers.get(1) else {
                    return Err("product waveguide requires a tweeter driver".into());
                };
                if waveguide.driver_id != tweeter.id {
                    return Err(format!(
                        "product waveguide target {:?} is not the speaker tweeter {:?}",
                        waveguide.driver_id, tweeter.id
                    ));
                }
                Some(WaveguideConfig {
                    horizontal_spline: waveguide.horizontal_spline.clone(),
                    vertical_spline: waveguide.vertical_spline.clone(),
                    depth: waveguide.depth_m,
                })
            }
        };
        if let Some(waveguide) = &product.waveguide {
            lowered.materials.waveguide = SpeakerMaterial::ConstantImpedance {
                specific_impedance: waveguide.surface_specific_impedance_pa_s_per_m,
            };
        }
        {
            let mut drivers = lowered.drivers_mut();
            for (driver, product_driver) in drivers.iter_mut().zip(&product.drivers) {
                driver.radius = product_driver.radius_m;
                driver.x_offset = product_driver.position_m[0];
                driver.y_offset = product_driver.position_m[1];
            }
        }
        lowered.validate_without_product_lowering()?;
        Ok(lowered)
    }

    /// Mutable access to all driver units in stable mesh/network order.
    pub fn drivers_mut(&mut self) -> Vec<&mut DriverConfig> {
        let (woofer, tweeter, additional) = (
            &mut self.woofer,
            &mut self.tweeter,
            &mut self.additional_drivers,
        );
        std::iter::once(woofer)
            .chain(std::iter::once(tweeter))
            .chain(additional.iter_mut())
            .collect()
    }

    /// Evaluate all driver velocities in modeling RMS convention, converted
    /// to the BEM peak convention used by this crate.
    pub fn velocities_at(&self, frequency_hz: f64) -> Vec<Complex64> {
        if let Some(product) = &self.product_design
            && let Ok(velocities) =
                product.piston_velocities_at(frequency_hz, Complex64::new(1.0, 0.0))
        {
            return velocities
                .into_iter()
                .map(|velocity| std::f64::consts::SQRT_2 * velocity.conj())
                .collect();
        }
        self.drivers()
            .into_iter()
            .map(|driver| driver.velocity_at(frequency_hz))
            .collect()
    }

    /// Normalized port admittance for the exterior BEM boundary.
    ///
    /// The modeling product contract is `pressure / volume_velocity` in its
    /// `e^{+iωt}` convention. BEM uses `e^{-iωt}`, so the impedance is
    /// conjugated before converting from volume impedance to specific
    /// impedance (`Z_specific = Z_volume · area`).
    pub(crate) fn port_admittance_at(&self, frequency_hz: f64) -> Option<Complex64> {
        let product = self.product_design.as_ref()?;
        let port = product.cabinet.port.as_ref()?;
        let medium = AcousticMedium {
            density_kg_per_m3: self.simulation.air_density,
            sound_speed_m_per_s: self.simulation.speed_of_sound,
            ..AcousticMedium::standard_air()
        };
        // Keep this helper fail-closed even when called independently of the
        // top-level model validation.  The surrounding Option API is used by
        // mesh lowering, so an invalid product boundary must be reported as
        // unavailable rather than becoming a process panic.
        let impedance = product
            .port_boundary_impedance_at(frequency_hz, medium)
            .ok()??;
        let specific_impedance = impedance.conj() * port.area_m2();
        let rho_c = medium.density_kg_per_m3 * medium.sound_speed_m_per_s;
        Some(Complex64::new(rho_c, 0.0) / specific_impedance)
    }

    /// Validate geometric consistency of the model.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_without_product_lowering()
    }

    fn validate_without_product_lowering(&self) -> Result<(), String> {
        validate_positive_finite("cabinet height", self.cabinet.height)?;
        validate_positive_finite("cabinet width", self.cabinet.width)?;
        validate_positive_finite("cabinet depth", self.cabinet.depth)?;
        validate_driver("woofer", &self.woofer, self.cabinet.depth)?;
        validate_driver("tweeter", &self.tweeter, self.cabinet.depth)?;
        for (index, driver) in self.additional_drivers.iter().enumerate() {
            validate_driver(
                &format!("additional driver {index}"),
                driver,
                self.cabinet.depth,
            )?;
        }
        if let Some(product) = &self.product_design {
            product
                .validate()
                .map_err(|error| format!("product design is invalid: {error}"))?;
            if let Some(port) = &product.cabinet.port {
                let radius = port.diameter_m / 2.0;
                if port.position_m[0].abs() + radius > self.cabinet.width / 2.0
                    || port.position_m[1].abs() + radius > self.cabinet.height / 2.0
                {
                    return Err("product port does not fit the speaker BEM baffle".into());
                }
            }
            let drivers = self.drivers();
            if product.drivers.len() != drivers.len() {
                return Err(format!(
                    "product design has {} drivers but speaker mesh has {}",
                    product.drivers.len(),
                    drivers.len()
                ));
            }
            for (index, (product_driver, mesh_driver)) in
                product.drivers.iter().zip(drivers).enumerate()
            {
                if (product_driver.position_m[0] - mesh_driver.x_offset).abs() > 1.0e-12
                    || (product_driver.position_m[1] - mesh_driver.y_offset).abs() > 1.0e-12
                    || product_driver.position_m[2].abs() > 1.0e-12
                {
                    return Err(format!(
                        "product driver {index} placement does not match the speaker mesh"
                    ));
                }
            }
        }
        validate_finite("separator position", self.separator.y_position)?;
        validate_positive_finite("separator thickness", self.separator.thickness)?;
        self.simulation.validate()?;

        let h2 = self.cabinet.height / 2.0;
        let w2 = self.cabinet.width / 2.0;

        // Woofer fits inside cabinet
        for (index, driver) in self.drivers().into_iter().enumerate() {
            if driver.x_offset.abs() + driver.radius > w2 {
                return Err(format!("Driver {index} extends outside cabinet width"));
            }
            if driver.y_offset - driver.radius < -h2 {
                return Err(format!("Driver {index} extends below cabinet"));
            }
            if driver.y_offset + driver.radius > h2 {
                return Err(format!("Driver {index} extends above cabinet"));
            }
        }

        // Separator between drivers
        let sep_y = self.separator.y_position;
        if sep_y - self.separator.thickness / 2.0 <= -h2
            || sep_y + self.separator.thickness / 2.0 >= h2
        {
            return Err("Separator must lie strictly inside the cabinet".into());
        }
        if self.woofer.y_offset + self.woofer.radius > sep_y - self.separator.thickness / 2.0 {
            return Err("Woofer overlaps separator".into());
        }
        if self.tweeter.y_offset - self.tweeter.radius < sep_y + self.separator.thickness / 2.0 {
            return Err("Tweeter overlaps separator".into());
        }

        // Waveguide rim must fit on baffle (if present)
        if let Some(ref wg) = self.waveguide {
            validate_positive_finite("waveguide depth", wg.depth)?;
            validate_waveguide_spline("horizontal", &wg.horizontal_spline, self.tweeter.radius)?;
            validate_waveguide_spline("vertical", &wg.vertical_spline, self.tweeter.radius)?;

            let horizontal_rim = wg.horizontal_spline[0][1];
            if horizontal_rim > w2 {
                return Err("Waveguide horizontal rim extends outside baffle width".into());
            }
            let vertical_rim = wg.vertical_spline[0][1];
            if self.tweeter.y_offset + vertical_rim > h2
                || self.tweeter.y_offset - vertical_rim < sep_y
            {
                return Err("Waveguide vertical rim extends outside baffle".into());
            }
        }

        // Driver apertures must be disjoint. The legacy pair retains its
        // stronger 10 mm vertical clearance requirement; arbitrary product
        // arrays use the geometric disk-overlap test.
        let drivers = self.drivers();
        for (left_index, left) in drivers.iter().enumerate() {
            for (right_index, right) in drivers.iter().enumerate().skip(left_index + 1) {
                let dx = left.x_offset - right.x_offset;
                let dy = left.y_offset - right.y_offset;
                let clearance = (dx * dx + dy * dy).sqrt() - left.radius - right.radius;
                if clearance < 0.0 {
                    return Err(format!("Drivers {left_index} and {right_index} overlap"));
                }
            }
        }
        let legacy_clearance =
            self.tweeter.y_offset - self.woofer.y_offset - self.woofer.radius - self.tweeter.radius;
        if legacy_clearance < 0.01 {
            return Err(format!(
                "Insufficient clearance between legacy drivers: {:.1} mm (minimum 10 mm)",
                legacy_clearance * 1000.0,
            ));
        }

        // Cone geometry validation
        if let Some(ref cone) = self.woofer.cone {
            if cone.dustcap_radius >= self.woofer.radius {
                return Err("Woofer dustcap radius must be smaller than driver radius".into());
            }
            if cone.cone_depth <= 0.0 {
                return Err("Woofer cone depth must be positive".into());
            }
        }
        if let Some(ref cone) = self.tweeter.cone {
            if cone.dustcap_radius >= self.tweeter.radius {
                return Err("Tweeter dustcap radius must be smaller than driver radius".into());
            }
            if cone.cone_depth <= 0.0 {
                return Err("Tweeter cone depth must be positive".into());
            }
        }

        self.materials.cabinet.validate()?;
        self.materials.waveguide.validate()?;

        Ok(())
    }

    /// Baffle center in world coordinates (front face center).
    pub fn baffle_center(&self) -> [f64; 3] {
        [0.0, 0.0, self.cabinet.depth / 2.0]
    }

    /// Acoustic axis direction (unit vector pointing forward).
    pub fn acoustic_axis(&self) -> [f64; 3] {
        [0.0, 0.0, 1.0]
    }
}

fn validate_finite(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} must be finite, got {value}"))
    }
}

fn validate_positive_finite(name: &str, value: f64) -> Result<(), String> {
    validate_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(format!("{name} must be positive, got {value}"))
    }
}

fn validate_driver(name: &str, driver: &DriverConfig, cabinet_depth: f64) -> Result<(), String> {
    validate_positive_finite(&format!("{name} radius"), driver.radius)?;
    validate_finite(&format!("{name} horizontal offset"), driver.x_offset)?;
    validate_finite(&format!("{name} vertical offset"), driver.y_offset)?;
    validate_finite(&format!("{name} piston velocity"), driver.piston_velocity)?;

    if let Some(model) = &driver.driver_model {
        model
            .validate()
            .map_err(|error| format!("{name} driver model is invalid: {error}"))?;
        if driver.transfer_function.is_some() {
            return Err(format!(
                "{name} must not define both driver_model and transfer_function"
            ));
        }
    }

    if let Some(points) = &driver.transfer_function {
        if points.is_empty() {
            return Err(format!("{name} transfer function must not be empty"));
        }
        let mut previous_frequency = None;
        for (index, point) in points.iter().enumerate() {
            validate_positive_finite(
                &format!("{name} transfer frequency at index {index}"),
                point.freq,
            )?;
            validate_finite(
                &format!("{name} transfer magnitude at index {index}"),
                point.magnitude,
            )?;
            if point.magnitude < 0.0 {
                return Err(format!(
                    "{name} transfer magnitude at index {index} must be non-negative"
                ));
            }
            validate_finite(
                &format!("{name} transfer phase at index {index}"),
                point.phase_deg,
            )?;
            if previous_frequency.is_some_and(|previous| point.freq <= previous) {
                return Err(format!(
                    "{name} transfer frequencies must be strictly increasing"
                ));
            }
            previous_frequency = Some(point.freq);
        }
    }

    if let Some(cone) = driver.cone {
        validate_positive_finite(&format!("{name} cone depth"), cone.cone_depth)?;
        validate_positive_finite(&format!("{name} dustcap radius"), cone.dustcap_radius)?;
        if cone.cone_depth >= cabinet_depth {
            return Err(format!(
                "{name} cone depth must be smaller than cabinet depth"
            ));
        }
    }
    Ok(())
}

fn validate_waveguide_spline(
    name: &str,
    points: &[[f64; 2]],
    aperture_radius: f64,
) -> Result<(), String> {
    if points.len() < 2 {
        return Err(format!("waveguide {name} spline needs at least 2 points"));
    }
    for (index, &[t, radius]) in points.iter().enumerate() {
        validate_finite(
            &format!("waveguide {name} spline parameter at index {index}"),
            t,
        )?;
        validate_positive_finite(
            &format!("waveguide {name} spline radius at index {index}"),
            radius,
        )?;
        if !(0.0..=1.0).contains(&t) {
            return Err(format!(
                "waveguide {name} spline parameter at index {index} must be in [0, 1]"
            ));
        }
        if index > 0 && t <= points[index - 1][0] {
            return Err(format!(
                "waveguide {name} spline parameters must be strictly increasing"
            ));
        }
    }
    if points[0][0].abs() > 1e-12 || (points[points.len() - 1][0] - 1.0).abs() > 1e-12 {
        return Err(format!(
            "waveguide {name} spline must start at t=0 and end at t=1"
        ));
    }
    let end_radius = points[points.len() - 1][1];
    let tolerance = aperture_radius.abs().max(1.0) * 1e-12;
    if (end_radius - aperture_radius).abs() > tolerance {
        return Err(format!(
            "waveguide {name} spline must end at the tweeter aperture radius"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_material_admittance_rejects_invalid_and_nonfinite_inputs() {
        let material = SpeakerMaterial::mdf();
        for frequency in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(material.try_admittance_at(frequency, 1.2, 343.0).is_err());
        }
        assert!(material.try_admittance_at(100.0, -1.0, 343.0).is_err());
        assert!(
            material
                .try_admittance_at(100.0, f64::MAX, f64::MAX)
                .is_err()
        );
        let porous = SpeakerMaterial::PorousLayer {
            layer: PorousLayer {
                model: sonium_bem::materials::PorousModel::Miki,
                thickness: -0.1,
                flow_resistivity: 10_000.0,
                air_gap: 0.0,
            },
        };
        assert!(porous.try_admittance_at(100.0, 1.2, 343.0).is_err());
        assert_eq!(
            material.try_admittance_at(100.0, 1.2, 343.0).unwrap(),
            admittance_for_material(Z_MDF, 1.2, 343.0)
        );
    }
    use sonium_sim_speaker_modeling::{
        CabinetDesign, CrossoverBranch, CrossoverNetwork, ProductDesign, ProductDriver,
        ProductWaveguideDesign,
    };

    fn sample_model() -> LoudspeakerModel {
        LoudspeakerModel {
            cabinet: CabinetDimensions {
                height: 0.40,
                width: 0.22,
                depth: 0.28,
            },
            woofer: DriverConfig {
                radius: 0.065,
                x_offset: 0.0,
                y_offset: -0.08,
                piston_velocity: 0.001,
                transfer_function: None,
                driver_model: None,
                cone: None,
            },
            tweeter: DriverConfig {
                radius: 0.013,
                x_offset: 0.0,
                y_offset: 0.10,
                piston_velocity: 0.001,
                transfer_function: None,
                driver_model: None,
                cone: None,
            },
            additional_drivers: Vec::new(),
            product_design: None,
            separator: SeparatorConfig {
                y_position: 0.02,
                thickness: 0.018,
            },
            waveguide: Some(WaveguideConfig {
                horizontal_spline: vec![[0.0, 0.05], [0.5, 0.03], [1.0, 0.013]],
                vertical_spline: vec![[0.0, 0.04], [0.5, 0.025], [1.0, 0.013]],
                depth: 0.025,
            }),
            materials: SpeakerMaterialAssignment::default(),
            simulation: SimulationConfig {
                freq_start: 200.0,
                freq_end: 20000.0,
                freq_points: 100,
                speed_of_sound: 343.0,
                air_density: 1.21,
                max_mesh_frequency: None,
            },
        }
    }

    fn sample_product_design() -> ProductDesign {
        let mut cabinet = CabinetDesign::default();
        // Keep the product port clear of the legacy woofer/tweeter locations
        // while retaining the product-level port in this integration fixture.
        cabinet.port.as_mut().unwrap().position_m = [0.075, 0.0];
        ProductDesign {
            cabinet,
            drivers: vec![
                ProductDriver {
                    id: "woofer".into(),
                    radius_m: 0.065,
                    position_m: [0.0, -0.08, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    force_factor_tm: 5.0,
                    coil_resistance_ohm: 5.0,
                    coil_reactance_ohm: 1.0,
                    mechanical_resistance_n_s_per_m: 0.5,
                    mechanical_reactance_n_s_per_m: 2.0,
                    crossover_branch: 0,
                },
                ProductDriver {
                    id: "tweeter".into(),
                    radius_m: 0.013,
                    position_m: [0.0, 0.10, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    force_factor_tm: 2.0,
                    coil_resistance_ohm: 4.0,
                    coil_reactance_ohm: 0.5,
                    mechanical_resistance_n_s_per_m: 0.3,
                    mechanical_reactance_n_s_per_m: 1.0,
                    crossover_branch: 1,
                },
            ],
            crossover: CrossoverNetwork {
                branches: vec![CrossoverBranch::default(), CrossoverBranch::default()],
            },
            waveguide: None,
            structural_acoustic: None,
        }
    }

    #[test]
    fn test_validate_ok() {
        let model = sample_model();
        assert!(model.validate().is_ok());
    }

    #[test]
    fn product_port_is_consumed_by_exterior_mesh_and_bem_boundary() {
        let mut model = sample_model();
        let mut product = sample_product_design();
        assert!(product.cabinet.port.is_some());
        model.product_design = Some(product.clone());
        model.validate().unwrap();
        let mesh = crate::mesh_gen::generate_loudspeaker_mesh(&model);
        assert!(
            mesh.region
                .contains(&crate::mesh_gen::MeshRegion::PortAperture)
        );
        let (_nodes, mut elements) = crate::simulation::speaker_mesh_to_bem(&mesh, &model).unwrap();
        crate::simulation::update_driver_bcs(&mut elements, &mesh, &model, 40.0, None).unwrap();
        assert!(elements.iter().zip(&mesh.region).any(|(element, region)| {
            *region == crate::mesh_gen::MeshRegion::PortAperture
                && matches!(
                    element.boundary_condition,
                    sonium_bem::core::types::BoundaryCondition::VelocityWithAdmittance {
                        admittance,
                        ..
                    } if admittance.norm() > 0.0
                )
        }));

        let mut one_frequency = model.clone();
        one_frequency.simulation.freq_start = 200.0;
        one_frequency.simulation.freq_end = 200.0;
        one_frequency.simulation.freq_points = 1;
        one_frequency.simulation.max_mesh_frequency = Some(200.0);
        let result = crate::simulation::run_simulation(&one_frequency, &[[0.0, 0.0, 1.0]])
            .expect("product port should be solvable in the exterior BEM path");
        assert_eq!(result.frequencies.len(), 1);
        assert!(result.complex_pressure[[0, 0]].norm().is_finite());

        product.cabinet.port = None;
        model.product_design = Some(product);
        assert!(model.validate().is_ok());
    }

    #[test]
    fn invalid_product_port_admittance_fails_closed() {
        let mut model = sample_model();
        let mut product = sample_product_design();
        product
            .cabinet
            .port
            .as_mut()
            .expect("sample product includes a port")
            .diameter_m = -1.0;
        model.product_design = Some(product);

        assert!(model.port_admittance_at(200.0).is_none());
    }

    #[test]
    fn product_geometry_lowering_is_explicit_and_does_not_mutate_input() {
        let mut model = sample_model();
        let product = sample_product_design();
        model.product_design = Some(product.clone());

        let lowered = model.with_product_design_geometry().unwrap();
        assert_eq!(model.cabinet.width, 0.22);
        assert_eq!(lowered.cabinet.width, product.cabinet.baffle_width_m);
        assert_eq!(lowered.cabinet.height, product.cabinet.baffle_height_m);
        assert_eq!(lowered.cabinet.depth, product.cabinet.box_depth_m);
        for (driver, product_driver) in lowered.drivers().iter().zip(&product.drivers) {
            assert_eq!(driver.radius, product_driver.radius_m);
            assert_eq!(driver.x_offset, product_driver.position_m[0]);
            assert_eq!(driver.y_offset, product_driver.position_m[1]);
        }
        lowered.validate().unwrap();
    }

    #[test]
    fn product_waveguide_lowering_is_owned_and_identity_bound() {
        let mut model = sample_model();
        let mut product = sample_product_design();
        product.waveguide = Some(ProductWaveguideDesign {
            driver_id: "tweeter".into(),
            horizontal_spline: vec![[0.0, 0.055], [0.5, 0.028], [1.0, 0.013]],
            vertical_spline: vec![[0.0, 0.045], [0.5, 0.024], [1.0, 0.013]],
            depth_m: 0.031,
            surface_specific_impedance_pa_s_per_m: 1.7e6,
        });
        model.waveguide = Some(WaveguideConfig {
            horizontal_spline: vec![[0.0, 0.06], [0.5, 0.04], [1.0, 0.013]],
            vertical_spline: vec![[0.0, 0.05], [0.5, 0.03], [1.0, 0.013]],
            depth: 0.020,
        });
        model.product_design = Some(product.clone());

        let lowered = model.with_product_design_geometry().unwrap();
        let lowered_waveguide = lowered.waveguide.expect("lowered product waveguide");
        assert_eq!(lowered_waveguide.depth, 0.031);
        assert_eq!(
            lowered_waveguide.horizontal_spline,
            product.waveguide.as_ref().unwrap().horizontal_spline
        );
        assert_eq!(
            lowered_waveguide.vertical_spline,
            product.waveguide.as_ref().unwrap().vertical_spline
        );
        assert!(matches!(
            lowered.materials.waveguide,
            SpeakerMaterial::ConstantImpedance {
                specific_impedance
            } if (specific_impedance - 1.7e6).abs() < 1.0e-6
        ));

        let mut wrong_target = product;
        wrong_target.waveguide.as_mut().unwrap().driver_id = "woofer".into();
        wrong_target.waveguide.as_mut().unwrap().horizontal_spline[2][1] = 0.065;
        wrong_target.waveguide.as_mut().unwrap().vertical_spline[2][1] = 0.065;
        model.product_design = Some(wrong_target);
        let error = model
            .with_product_design_geometry()
            .expect_err("waveguide must bind to the simulator tweeter");
        assert!(error.contains("not the speaker tweeter"));
    }

    #[test]
    fn product_placement_mismatch_is_rejected() {
        let mut model = sample_model();
        let mut product = sample_product_design();
        product.cabinet.port = None;
        model.product_design = Some(product);
        model.woofer.x_offset = 0.001;
        let error = model.validate().unwrap_err();
        assert!(error.contains("placement does not match"));
    }

    #[test]
    fn test_validate_woofer_below_cabinet() {
        let mut model = sample_model();
        model.woofer.y_offset = -0.19; // -0.19 - 0.065 = -0.255 < -0.20
        assert!(model.validate().is_err());
    }

    #[test]
    fn test_validate_woofer_above_cabinet() {
        let mut model = sample_model();
        model.woofer.y_offset = 0.19; // 0.19 + 0.065 = 0.255 > 0.20
        assert!(model.validate().is_err());
    }

    #[test]
    fn test_validate_tweeter_below_cabinet() {
        let mut model = sample_model();
        model.tweeter.y_offset = -0.19; // -0.19 - 0.013 = -0.203 < -0.20
        assert!(model.validate().is_err());
    }

    #[test]
    fn test_validate_overlap() {
        let mut model = sample_model();
        model.woofer.y_offset = 0.05;
        model.tweeter.y_offset = 0.05;
        assert!(model.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_non_finite_and_non_positive_geometry() {
        let invalid_cases: Vec<(&str, Box<dyn Fn(&mut LoudspeakerModel)>)> = vec![
            (
                "cabinet height NaN",
                Box::new(|m| m.cabinet.height = f64::NAN),
            ),
            ("cabinet width zero", Box::new(|m| m.cabinet.width = 0.0)),
            (
                "cabinet depth negative",
                Box::new(|m| m.cabinet.depth = -0.1),
            ),
            (
                "woofer radius NaN",
                Box::new(|m| m.woofer.radius = f64::NAN),
            ),
            ("woofer radius zero", Box::new(|m| m.woofer.radius = 0.0)),
            (
                "woofer offset infinite",
                Box::new(|m| m.woofer.y_offset = f64::INFINITY),
            ),
            (
                "woofer velocity NaN",
                Box::new(|m| m.woofer.piston_velocity = f64::NAN),
            ),
            (
                "tweeter radius negative",
                Box::new(|m| m.tweeter.radius = -0.01),
            ),
            (
                "separator position NaN",
                Box::new(|m| m.separator.y_position = f64::NAN),
            ),
            (
                "separator thickness zero",
                Box::new(|m| m.separator.thickness = 0.0),
            ),
        ];

        for (name, mutate) in invalid_cases {
            let mut model = sample_model();
            mutate(&mut model);
            assert!(model.validate().is_err(), "accepted invalid case: {name}");
        }
    }

    #[test]
    fn test_validate_rejects_invalid_transfer_functions() {
        let invalid_curves = [
            vec![],
            vec![TransferPoint {
                freq: 0.0,
                magnitude: 1.0,
                phase_deg: 0.0,
            }],
            vec![
                TransferPoint {
                    freq: 200.0,
                    magnitude: 1.0,
                    phase_deg: 0.0,
                },
                TransferPoint {
                    freq: 100.0,
                    magnitude: 1.0,
                    phase_deg: 0.0,
                },
            ],
            vec![
                TransferPoint {
                    freq: 100.0,
                    magnitude: 1.0,
                    phase_deg: 0.0,
                },
                TransferPoint {
                    freq: 100.0,
                    magnitude: 1.0,
                    phase_deg: 0.0,
                },
            ],
            vec![TransferPoint {
                freq: 100.0,
                magnitude: -1.0,
                phase_deg: 0.0,
            }],
            vec![TransferPoint {
                freq: 100.0,
                magnitude: 1.0,
                phase_deg: f64::NAN,
            }],
        ];

        for (index, curve) in invalid_curves.into_iter().enumerate() {
            let mut model = sample_model();
            model.woofer.transfer_function = Some(curve);
            assert!(
                model.validate().is_err(),
                "accepted invalid transfer function #{index}"
            );
        }
    }

    #[test]
    fn test_validate_rejects_invalid_simulation_config() {
        let invalid_cases: Vec<(&str, Box<dyn Fn(&mut SimulationConfig)>)> = vec![
            ("start frequency zero", Box::new(|s| s.freq_start = 0.0)),
            ("start frequency NaN", Box::new(|s| s.freq_start = f64::NAN)),
            ("end before start", Box::new(|s| s.freq_end = 100.0)),
            ("frequency count zero", Box::new(|s| s.freq_points = 0)),
            ("speed of sound zero", Box::new(|s| s.speed_of_sound = 0.0)),
            (
                "air density infinite",
                Box::new(|s| s.air_density = f64::INFINITY),
            ),
            (
                "mesh frequency negative",
                Box::new(|s| s.max_mesh_frequency = Some(-1.0)),
            ),
            (
                "mesh frequency below simulation end",
                Box::new(|s| s.max_mesh_frequency = Some(s.freq_end * 0.5)),
            ),
        ];

        for (name, mutate) in invalid_cases {
            let mut model = sample_model();
            mutate(&mut model.simulation);
            assert!(model.validate().is_err(), "accepted invalid case: {name}");
        }
    }

    #[test]
    fn test_validate_rejects_invalid_waveguide_splines() {
        let mut too_short = sample_model();
        too_short.waveguide.as_mut().unwrap().horizontal_spline = vec![[0.0, 0.05]];
        assert!(too_short.validate().is_err());

        let mut unsorted = sample_model();
        unsorted.waveguide.as_mut().unwrap().vertical_spline =
            vec![[0.0, 0.04], [0.8, 0.03], [0.7, 0.02], [1.0, 0.013]];
        assert!(unsorted.validate().is_err());

        let mut wrong_endpoints = sample_model();
        wrong_endpoints
            .waveguide
            .as_mut()
            .unwrap()
            .horizontal_spline[0][0] = 0.1;
        assert!(wrong_endpoints.validate().is_err());

        let mut aperture_mismatch = sample_model();
        aperture_mismatch
            .waveguide
            .as_mut()
            .unwrap()
            .vertical_spline
            .last_mut()
            .unwrap()[1] = 0.02;
        assert!(aperture_mismatch.validate().is_err());

        let mut non_finite = sample_model();
        non_finite.waveguide.as_mut().unwrap().horizontal_spline[1][1] = f64::NAN;
        assert!(non_finite.validate().is_err());
    }

    #[test]
    fn test_json_roundtrip() {
        let model = sample_model();
        let json = serde_json::to_string_pretty(&model).unwrap();
        let parsed: LoudspeakerModel = serde_json::from_str(&json).unwrap();
        assert!((parsed.cabinet.height - 0.40).abs() < 1e-10);
        assert!((parsed.woofer.radius - 0.065).abs() < 1e-10);
    }

    #[test]
    fn legacy_json_without_product_fields_keeps_defaults() {
        let model = sample_model();
        let mut value = serde_json::to_value(&model).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("additional_drivers");
        object.remove("product_design");
        object.remove("materials");
        let parsed: LoudspeakerModel = serde_json::from_value(value).unwrap();
        assert!(parsed.additional_drivers.is_empty());
        assert!(parsed.product_design.is_none());
        assert!(parsed.validate().is_ok());
    }

    #[test]
    fn test_example_configs_parse_and_validate() {
        for example in [
            "examples/waveguide_optimization.json",
            "examples/waveguide_optimization_smoke.json",
        ] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(example);
            let json = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("failed to read {example}: {e}"));
            let model: LoudspeakerModel = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("failed to parse {example}: {e}"));
            model
                .validate()
                .unwrap_or_else(|e| panic!("{example} failed validation: {e}"));
        }

        // The benchmark must carry typed drivers (velocity + impedance); the
        // smoke test keeps the flat in-phase transfer functions.
        let bench_json = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("examples/waveguide_optimization.json"),
        )
        .unwrap();
        let bench: LoudspeakerModel = serde_json::from_str(&bench_json).unwrap();
        assert!(bench.woofer.driver_model.is_some());
        assert!(bench.tweeter.driver_model.is_some());

        let smoke_json = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("examples/waveguide_optimization_smoke.json"),
        )
        .unwrap();
        let smoke: LoudspeakerModel = serde_json::from_str(&smoke_json).unwrap();
        assert!(smoke.woofer.driver_model.is_none());
        assert!(smoke.woofer.transfer_function.is_some());

        // Realistic passbands: the typed woofer must roll off toward 5 kHz and
        // the typed tweeter toward 200 Hz, with non-trivial phase — the exact
        // opposite of the smoke test's flat in-phase velocity.
        let woofer_model = bench.woofer.driver_model.as_ref().unwrap();
        let tweeter_model = bench.tweeter.driver_model.as_ref().unwrap();
        let v_w_low = woofer_model.velocity_at(500.0).norm();
        let v_w_high = woofer_model.velocity_at(5000.0).norm();
        let v_t_low = tweeter_model.velocity_at(200.0).norm();
        let v_t_high = tweeter_model.velocity_at(1000.0).norm();
        assert!(
            v_w_high < 0.2 * v_w_low,
            "woofer should roll off above its mass corner: {v_w_low:.3e} vs {v_w_high:.3e}"
        );
        assert!(
            v_t_low < 0.5 * v_t_high,
            "tweeter should roll off below its resonance: {v_t_low:.3e} vs {v_t_high:.3e}"
        );
        let phase = woofer_model.velocity_at(2200.0).arg().abs();
        assert!(
            phase > 0.5,
            "woofer velocity phase should be non-trivial in the crossover region: {phase}"
        );
    }

    #[test]
    fn test_frequencies_log_spaced() {
        let cfg = SimulationConfig {
            freq_start: 100.0,
            freq_end: 10000.0,
            freq_points: 50,
            speed_of_sound: 343.0,
            air_density: 1.21,
            max_mesh_frequency: None,
        };
        let freqs = cfg.frequencies();
        assert_eq!(freqs.len(), 50);
        assert!((freqs[0] - 100.0).abs() < 1e-10);
        assert!((freqs[49] - 10000.0).abs() < 0.1);
        // Log-spaced: ratio between consecutive should be roughly constant
        let r1 = freqs[1] / freqs[0];
        let r2 = freqs[2] / freqs[1];
        assert!((r1 - r2).abs() < 0.01);
    }

    #[test]
    fn test_velocity_at_flat() {
        let driver = DriverConfig {
            radius: 0.05,
            x_offset: 0.0,
            y_offset: 0.0,
            piston_velocity: 0.001,
            transfer_function: None,
            driver_model: None,
            cone: None,
        };
        let v = driver.velocity_at(1000.0);
        assert!((v.re - 0.001).abs() < 1e-10);
        assert!(v.im.abs() < 1e-10);
    }

    #[test]
    fn test_velocity_at_uses_typed_driver_model() {
        let study =
            sonium_sim_speaker_modeling::SpeakerDriverStudy::detailed_frequency_domain_driver();
        let frequency = study.frequency_hz[0];
        let driver_model = sonium_sim_speaker_modeling::DriverModel::from_study(&study).unwrap();
        let expected = std::f64::consts::SQRT_2 * driver_model.velocity_at(frequency);
        let driver = DriverConfig {
            radius: study.geometry.diaphragm_radius_m,
            x_offset: 0.0,
            y_offset: 0.0,
            piston_velocity: 99.0,
            transfer_function: None,
            driver_model: Some(driver_model),
            cone: None,
        };

        assert_eq!(driver.velocity_at(frequency), expected);
    }

    #[test]
    fn test_velocity_at_interpolated_takes_short_path_across_phase_wrap() {
        // Adjacent points at +170° and −170° are geometrically only 20°
        // apart on the unit circle (going through ±180°). Linear
        // magnitude/phase interpolation would walk the LONG way (340°)
        // through 0° phase, briefly producing a real-positive vector at
        // the midpoint. Complex-plane interpolation must take the short
        // path through ±180°.
        let driver = DriverConfig {
            radius: 0.05,
            x_offset: 0.0,
            y_offset: 0.0,
            piston_velocity: 0.0,
            cone: None,
            driver_model: None,
            transfer_function: Some(vec![
                TransferPoint {
                    freq: 100.0,
                    magnitude: 1.0,
                    phase_deg: 170.0,
                },
                TransferPoint {
                    freq: 200.0,
                    magnitude: 1.0,
                    phase_deg: -170.0,
                },
            ]),
        };
        let v = driver.velocity_at(150.0);
        // Short path midpoint sits near phase = ±180°, real part close
        // to −cos(10°) ≈ −0.985, imag near zero. Definitely NOT near
        // phase = 0° (which would put real part near +1.0).
        assert!(
            v.re < -0.9,
            "expected midpoint to sit near phase ±180° (real ≈ −1), got {v}"
        );
        assert!(
            v.im.abs() < 0.05,
            "expected midpoint imag near 0 across the ±180° wrap, got {v}"
        );
    }

    #[test]
    fn test_velocity_at_interpolated() {
        let driver = DriverConfig {
            radius: 0.05,
            x_offset: 0.0,
            y_offset: 0.0,
            piston_velocity: 0.001,
            cone: None,
            driver_model: None,
            transfer_function: Some(vec![
                TransferPoint {
                    freq: 100.0,
                    magnitude: 0.01,
                    phase_deg: 0.0,
                },
                TransferPoint {
                    freq: 1000.0,
                    magnitude: 0.001,
                    phase_deg: -90.0,
                },
            ]),
        };
        // At midpoint freq=550, mag should be ~0.0055, phase ~-45 deg
        let v = driver.velocity_at(550.0);
        assert!((v.norm() - 0.0055).abs() < 0.001);
    }

    #[test]
    fn test_max_element_size() {
        let cfg = SimulationConfig {
            freq_start: 200.0,
            freq_end: 20000.0,
            freq_points: 100,
            speed_of_sound: 343.0,
            air_density: 1.21,
            max_mesh_frequency: None,
        };
        // lambda_min at 20kHz = 343/20000 = 0.01715 m
        // h_max = 0.01715 / 8 = 0.00214 m
        let h = cfg.max_element_size();
        assert!(h > 0.002 && h < 0.003);
    }

    #[test]
    fn test_admittance() {
        let y = admittance_for_material(Z_MDF, 1.21, 343.0);
        // ρc = 415.03, Y = 415.03 / 3.2e6 ≈ 1.297e-4
        assert!((y.re - 1.297e-4).abs() < 1e-6);
        assert!(y.im.abs() < 1e-10);
    }

    #[test]
    fn test_constant_speaker_material_admittance_matches_legacy_value() {
        let material = SpeakerMaterial::mdf();
        let y = material.admittance_at(125.0, 1.21, 343.0);
        assert!((y.re - 1.297e-4).abs() < 1e-6);
        assert!(y.im.abs() < 1e-10);
    }

    #[test]
    fn test_porous_speaker_material_is_passive_and_frequency_dependent() {
        let material = SpeakerMaterial::PorousLayer {
            layer: PorousLayer::miki(0.05, 12_000.0),
        };
        let low = material.admittance_at(125.0, 1.21, 343.0);
        let high = material.admittance_at(2_000.0, 1.21, 343.0);

        assert!(low.re >= 0.0);
        assert!(high.re >= 0.0);
        assert!((high - low).norm() > 1e-3);
    }

    #[test]
    fn test_validate_safety_margin() {
        let mut model = sample_model();
        // Separator between drivers with thin thickness
        model.separator.y_position = 0.05;
        model.separator.thickness = 0.002;
        // woofer_y=-0.02, top=-0.02+0.065=0.045
        // sep - half = 0.05 - 0.001 = 0.049, woofer_top(0.045) < 0.049 ✓
        // tweeter_y=0.12, bottom=0.12-0.013=0.107
        // sep + half = 0.05 + 0.001 = 0.051, tweeter_bottom(0.107) > 0.051 ✓
        // gap = 0.107 - 0.045 = 0.062 -- too big.
        // Need gap < 0.01: tweeter_y - 0.013 - (-0.02 + 0.065) = tweeter_y - 0.058
        // tweeter_y - 0.058 < 0.01 → tweeter_y < 0.068
        // Also tweeter_bottom > sep + half: tweeter_y - 0.013 > 0.051 → tweeter_y > 0.064
        // So tweeter_y in (0.064, 0.068). Use 0.066.
        // gap = 0.066 - 0.013 - 0.045 = 0.008 < 0.01 ✓
        model.waveguide = None;
        model.woofer.y_offset = -0.02;
        model.tweeter.y_offset = 0.066;
        let err = model.validate().unwrap_err();
        assert!(
            err.contains("clearance"),
            "expected clearance error, got: {err}"
        );
    }

    #[test]
    fn test_validate_cone_dustcap_too_large() {
        let mut model = sample_model();
        model.woofer.cone = Some(ConeConfig {
            cone_depth: 0.008,
            dustcap_radius: 0.10, // larger than woofer radius 0.065
        });
        let err = model.validate().unwrap_err();
        assert!(
            err.contains("dustcap"),
            "expected dustcap error, got: {err}"
        );
    }

    #[test]
    fn test_validate_cone_ok() {
        let mut model = sample_model();
        model.woofer.cone = Some(ConeConfig {
            cone_depth: 0.008,
            dustcap_radius: 0.015,
        });
        model.tweeter.cone = Some(ConeConfig {
            cone_depth: 0.003,
            dustcap_radius: 0.005,
        });
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_no_waveguide_model() {
        let mut model = sample_model();
        model.waveguide = None;
        assert!(model.validate().is_ok());
    }
}
