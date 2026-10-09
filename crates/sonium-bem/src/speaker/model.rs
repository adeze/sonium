//! Parametric loudspeaker model.
//!
//! JSON-serializable description of a sealed-box loudspeaker with woofer,
//! tweeter, spline-parametrised waveguide, and internal separator plate.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Acoustic impedance constants
// ---------------------------------------------------------------------------

/// Specific acoustic impedance of MDF (Pa·s/m).
pub const Z_MDF: f64 = 3.2e6;

/// Specific acoustic impedance of hard plastic (Pa·s/m).
pub const Z_HARD_PLASTIC: f64 = 2.5e6;

/// Compute complex admittance Y = ρc / Z for a material impedance.
pub fn admittance_for_material(z_material: f64, air_density: f64, speed_of_sound: f64) -> Complex64 {
    let rho_c = air_density * speed_of_sound;
    Complex64::new(rho_c / z_material, 0.0)
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
    /// Internal separator plate between drivers.
    pub separator: SeparatorConfig,
    /// Tweeter waveguide shape.
    pub waveguide: WaveguideConfig,
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
    pub fn velocity_at(&self, freq: f64) -> Complex64 {
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
                // Linear interpolation
                for w in pts.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    if freq >= a.freq && freq <= b.freq {
                        let t = (freq - a.freq) / (b.freq - a.freq);
                        let mag = a.magnitude + t * (b.magnitude - a.magnitude);
                        let phase = a.phase_deg + t * (b.phase_deg - a.phase_deg);
                        return polar(mag, phase);
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
    /// Validate geometric consistency of the model.
    pub fn validate(&self) -> Result<(), String> {
        let h2 = self.cabinet.height / 2.0;
        let w2 = self.cabinet.width / 2.0;

        // Woofer fits inside cabinet
        if self.woofer.y_offset - self.woofer.radius < -h2 {
            return Err("Woofer extends below cabinet".into());
        }
        if self.woofer.y_offset + self.woofer.radius > h2 {
            return Err("Woofer extends above cabinet".into());
        }
        if self.woofer.radius > w2 {
            return Err("Woofer wider than cabinet".into());
        }

        // Tweeter fits inside cabinet
        if self.tweeter.y_offset - self.tweeter.radius < -h2 {
            return Err("Tweeter extends below cabinet".into());
        }
        if self.tweeter.y_offset + self.tweeter.radius > h2 {
            return Err("Tweeter extends above cabinet".into());
        }
        if self.tweeter.radius > w2 {
            return Err("Tweeter wider than cabinet".into());
        }

        // Separator between drivers
        let sep_y = self.separator.y_position;
        if self.woofer.y_offset + self.woofer.radius > sep_y - self.separator.thickness / 2.0 {
            return Err("Woofer overlaps separator".into());
        }
        if self.tweeter.y_offset - self.tweeter.radius < sep_y + self.separator.thickness / 2.0 {
            return Err("Tweeter overlaps separator".into());
        }

        // Waveguide rim must fit on baffle
        if let Some(first_h) = self.waveguide.horizontal_spline.first() {
            let rim_h = first_h[1];
            if self.tweeter.y_offset + rim_h > h2 || self.tweeter.y_offset - rim_h < sep_y {
                return Err("Waveguide horizontal rim extends outside baffle".into());
            }
        }

        // Drivers don't overlap each other
        let woofer_top = self.woofer.y_offset + self.woofer.radius;
        let tweeter_bottom = self.tweeter.y_offset - self.tweeter.radius;
        if woofer_top > tweeter_bottom {
            return Err("Woofer and tweeter overlap".into());
        }

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_model() -> LoudspeakerModel {
        LoudspeakerModel {
            cabinet: CabinetDimensions {
                height: 0.40,
                width: 0.22,
                depth: 0.28,
            },
            woofer: DriverConfig {
                radius: 0.065,
                y_offset: -0.08,
                piston_velocity: 0.001,
                transfer_function: None,
            },
            tweeter: DriverConfig {
                radius: 0.013,
                y_offset: 0.10,
                piston_velocity: 0.001,
                transfer_function: None,
            },
            separator: SeparatorConfig {
                y_position: 0.02,
                thickness: 0.018,
            },
            waveguide: WaveguideConfig {
                horizontal_spline: vec![[0.0, 0.05], [0.5, 0.03], [1.0, 0.013]],
                vertical_spline: vec![[0.0, 0.04], [0.5, 0.025], [1.0, 0.013]],
                depth: 0.025,
            },
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

    #[test]
    fn test_validate_ok() {
        let model = sample_model();
        assert!(model.validate().is_ok());
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
    fn test_json_roundtrip() {
        let model = sample_model();
        let json = serde_json::to_string_pretty(&model).unwrap();
        let parsed: LoudspeakerModel = serde_json::from_str(&json).unwrap();
        assert!((parsed.cabinet.height - 0.40).abs() < 1e-10);
        assert!((parsed.woofer.radius - 0.065).abs() < 1e-10);
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
            y_offset: 0.0,
            piston_velocity: 0.001,
            transfer_function: None,
        };
        let v = driver.velocity_at(1000.0);
        assert!((v.re - 0.001).abs() < 1e-10);
        assert!(v.im.abs() < 1e-10);
    }

    #[test]
    fn test_velocity_at_interpolated() {
        let driver = DriverConfig {
            radius: 0.05,
            y_offset: 0.0,
            piston_velocity: 0.001,
            transfer_function: Some(vec![
                TransferPoint { freq: 100.0, magnitude: 0.01, phase_deg: 0.0 },
                TransferPoint { freq: 1000.0, magnitude: 0.001, phase_deg: -90.0 },
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
}
