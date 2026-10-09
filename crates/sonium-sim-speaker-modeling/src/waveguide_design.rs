//! Product-owned body-fitted waveguide geometry.
//!
//! The speaker simulator has a mesh-native waveguide configuration, but the
//! product model must own the geometry and the driver identity it belongs to.
//! Keeping this contract in the modeling crate prevents a simulator-side
//! waveguide from becoming an untracked second source of truth.

use serde::{Deserialize, Serialize};

/// Default hard-plastic waveguide specific impedance (Pa·s/m).
pub const DEFAULT_WAVEGUIDE_SPECIFIC_IMPEDANCE_PA_S_PER_M: f64 = 2.5e6;

fn default_waveguide_specific_impedance() -> f64 {
    DEFAULT_WAVEGUIDE_SPECIFIC_IMPEDANCE_PA_S_PER_M
}

/// Serializable waveguide geometry attached to one product driver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductWaveguideDesign {
    /// Exact [`ProductDriver`](crate::ProductDriver) ID receiving the guide.
    pub driver_id: String,
    /// Control points `[t, radius_m]` for the horizontal profile.
    pub horizontal_spline: Vec<[f64; 2]>,
    /// Control points `[t, radius_m]` for the vertical profile.
    pub vertical_spline: Vec<[f64; 2]>,
    /// Guide depth measured from the baffle toward the driver (m).
    pub depth_m: f64,
    /// Constant specific acoustic impedance of the guide surface (Pa·s/m).
    #[serde(default = "default_waveguide_specific_impedance")]
    pub surface_specific_impedance_pa_s_per_m: f64,
}

impl ProductWaveguideDesign {
    /// Validate geometry and its terminal aperture radius.
    pub fn validate_for_aperture(&self, aperture_radius_m: f64) -> Result<(), String> {
        if self.driver_id.trim().is_empty() {
            return Err("waveguide driver id must not be blank".into());
        }
        if !self.depth_m.is_finite() || self.depth_m <= 0.0 {
            return Err("waveguide depth must be finite positive".into());
        }
        if !self.surface_specific_impedance_pa_s_per_m.is_finite()
            || self.surface_specific_impedance_pa_s_per_m <= 0.0
        {
            return Err("waveguide surface impedance must be finite positive".into());
        }
        if !aperture_radius_m.is_finite() || aperture_radius_m <= 0.0 {
            return Err("waveguide aperture radius must be finite positive".into());
        }
        validate_spline("horizontal", &self.horizontal_spline, aperture_radius_m)?;
        validate_spline("vertical", &self.vertical_spline, aperture_radius_m)?;
        Ok(())
    }
}

fn validate_spline(name: &str, points: &[[f64; 2]], aperture_radius_m: f64) -> Result<(), String> {
    if points.len() < 2 {
        return Err(format!("waveguide {name} spline needs at least two points"));
    }

    for (index, &[t, radius_m]) in points.iter().enumerate() {
        if !t.is_finite() || !radius_m.is_finite() || radius_m <= 0.0 {
            return Err(format!(
                "waveguide {name} spline point {index} must be finite with positive radius"
            ));
        }
        if !(0.0..=1.0).contains(&t) {
            return Err(format!(
                "waveguide {name} spline parameter {index} must be in [0, 1]"
            ));
        }
        if index > 0 && t <= points[index - 1][0] {
            return Err(format!(
                "waveguide {name} spline parameters must be strictly increasing"
            ));
        }
    }

    if points[0][0].abs() > 1.0e-12 || (points[points.len() - 1][0] - 1.0).abs() > 1.0e-12 {
        return Err(format!(
            "waveguide {name} spline must start at t=0 and end at t=1"
        ));
    }

    let terminal_radius_m = points[points.len() - 1][1];
    let tolerance = aperture_radius_m.abs().max(1.0) * 1.0e-12;
    if (terminal_radius_m - aperture_radius_m).abs() > tolerance {
        return Err(format!(
            "waveguide {name} spline must end at the driver aperture radius"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_waveguide() -> ProductWaveguideDesign {
        ProductWaveguideDesign {
            driver_id: "tweeter".into(),
            horizontal_spline: vec![[0.0, 0.05], [0.5, 0.03], [1.0, 0.013]],
            vertical_spline: vec![[0.0, 0.04], [0.5, 0.025], [1.0, 0.013]],
            depth_m: 0.025,
            surface_specific_impedance_pa_s_per_m: DEFAULT_WAVEGUIDE_SPECIFIC_IMPEDANCE_PA_S_PER_M,
        }
    }

    #[test]
    fn validates_and_round_trips() {
        let waveguide = valid_waveguide();
        waveguide
            .validate_for_aperture(0.013)
            .expect("valid product waveguide");
        let encoded = serde_json::to_string(&waveguide).expect("serialize waveguide");
        let decoded: ProductWaveguideDesign =
            serde_json::from_str(&encoded).expect("deserialize waveguide");
        assert_eq!(decoded, waveguide);
    }

    #[test]
    fn rejects_identity_and_profile_tampering() {
        let mut waveguide = valid_waveguide();
        waveguide.driver_id.clear();
        assert!(waveguide.validate_for_aperture(0.013).is_err());

        let mut waveguide = valid_waveguide();
        waveguide.horizontal_spline[1][0] = 0.0;
        assert!(waveguide.validate_for_aperture(0.013).is_err());

        let mut waveguide = valid_waveguide();
        waveguide.vertical_spline[2][1] *= 1.01;
        assert!(waveguide.validate_for_aperture(0.013).is_err());

        let mut waveguide = valid_waveguide();
        waveguide.surface_specific_impedance_pa_s_per_m = f64::NAN;
        assert!(waveguide.validate_for_aperture(0.013).is_err());
    }
}
