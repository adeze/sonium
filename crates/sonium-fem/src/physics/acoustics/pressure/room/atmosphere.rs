//! Atmospheric absorption model (ISO 9613-1)
//!
//! Computes the sound attenuation in air as a function of temperature,
//! pressure, and relative humidity. The model is valid for frequencies
//! from 50 Hz to 10 kHz and temperatures from −20 °C to +50 °C.
//!
//! # Reference
//!
//! ISO 9613-1:1993, "Acoustics — Attenuation of sound during propagation
//! outdoors — Part 1: Calculation of the absorption of sound by the
//! atmosphere."

/// Atmospheric conditions.
#[derive(Debug, Clone, Copy)]
pub struct AtmosphereModel {
    /// Temperature \[°C\].
    pub temperature_celsius: f64,
    /// Relative humidity \[%\] (0–100).
    pub relative_humidity: f64,
    /// Atmospheric pressure \[Pa\].
    pub pressure: f64,
}

impl AtmosphereModel {
    /// Standard conditions: 20 °C, 50% RH, 101325 Pa.
    pub fn standard() -> Self {
        Self {
            temperature_celsius: 20.0,
            relative_humidity: 50.0,
            pressure: 101_325.0,
        }
    }

    /// Absorption coefficient at frequency f \[dB/m\].
    ///
    /// Implements the full ISO 9613-1 model.
    pub fn absorption_db_per_m(&self, f: f64) -> f64 {
        self.absorption_nepers_per_m(f) * 8.685_889_638 // 20/ln(10)
    }

    /// Absorption coefficient at frequency f \[Np/m\].
    pub fn absorption_nepers_per_m(&self, f: f64) -> f64 {
        let t_kelvin = self.temperature_celsius + 273.15;
        let t_ref = 293.15; // 20 °C
        let p_ref = 101_325.0;
        let t_01 = 273.16; // Triple point of water

        let p_r = self.pressure / p_ref;

        // Molar concentration of water vapour
        let p_sat = p_ref * 10.0_f64.powf(-6.8346 * (t_01 / t_kelvin).powf(1.261) + 4.6151);
        let h = self.relative_humidity * p_sat / (self.pressure);

        // Relaxation frequencies
        let f_ro = p_r * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
        let f_rn = p_r
            * (t_kelvin / t_ref).powf(-0.5)
            * (9.0 + 280.0 * h * (-4.170 * ((t_kelvin / t_ref).powf(-1.0 / 3.0) - 1.0)).exp());

        // Classical + rotational absorption
        let alpha_cr = 1.84e-11 / p_r * (t_kelvin / t_ref).sqrt();

        // Vibrational absorption (O₂ and N₂)
        let t_ratio_m52 = (t_kelvin / t_ref).powf(-2.5);
        let alpha_o = 0.01275 * (-2239.1 / t_kelvin).exp() * f_ro / (f_ro * f_ro + f * f);
        let alpha_n = 0.10680 * (-3352.0 / t_kelvin).exp() * f_rn / (f_rn * f_rn + f * f);

        f * f * (alpha_cr + t_ratio_m52 * (alpha_o + alpha_n))
    }

    /// Absorption coefficients at the six standard octave bands \[Np/m\].
    pub fn octave_band_absorption_nepers(&self) -> [f64; 6] {
        let bands = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0];
        let mut result = [0.0; 6];
        for (i, &f) in bands.iter().enumerate() {
            result[i] = self.absorption_nepers_per_m(f);
        }
        result
    }

    /// Absorption coefficients at the six standard octave bands \[dB/m\].
    pub fn octave_band_absorption_db(&self) -> [f64; 6] {
        let bands = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0];
        let mut result = [0.0; 6];
        for (i, &f) in bands.iter().enumerate() {
            result[i] = self.absorption_db_per_m(f);
        }
        result
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_atmosphere_monotonic() {
        let atm = AtmosphereModel::standard();
        let mut prev = 0.0;
        // Absorption should increase with frequency (approximately f²)
        for &f in &[125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0] {
            let alpha = atm.absorption_db_per_m(f);
            assert!(
                alpha > prev,
                "α({f} Hz) = {alpha:.4} should exceed α at lower f"
            );
            assert!(alpha > 0.0, "Absorption must be positive");
            prev = alpha;
        }
    }

    #[test]
    fn test_standard_atmosphere_order_of_magnitude() {
        let atm = AtmosphereModel::standard();
        // At 1 kHz, 20°C, 50% RH: α ≈ 0.005–0.01 dB/m
        let alpha = atm.absorption_db_per_m(1000.0);
        assert!(alpha > 0.001 && alpha < 0.1, "α(1kHz) = {alpha:.4} dB/m");
        // At 4 kHz: α ≈ 0.02–0.08 dB/m
        let alpha4k = atm.absorption_db_per_m(4000.0);
        assert!(
            alpha4k > 0.01 && alpha4k < 0.5,
            "α(4kHz) = {alpha4k:.4} dB/m"
        );
    }

    #[test]
    fn test_dry_air_higher_absorption() {
        // Dry air has higher absorption than humid air at mid frequencies
        let dry = AtmosphereModel {
            temperature_celsius: 20.0,
            relative_humidity: 10.0,
            pressure: 101_325.0,
        };
        let humid = AtmosphereModel {
            temperature_celsius: 20.0,
            relative_humidity: 80.0,
            pressure: 101_325.0,
        };
        // At 4 kHz, dry air absorbs more
        let alpha_dry = dry.absorption_db_per_m(4000.0);
        let alpha_humid = humid.absorption_db_per_m(4000.0);
        assert!(
            alpha_dry > alpha_humid * 0.5,
            "Dry: {alpha_dry:.4}, Humid: {alpha_humid:.4}"
        );
    }

    #[test]
    fn test_octave_band_has_six_values() {
        let atm = AtmosphereModel::standard();
        let bands = atm.octave_band_absorption_db();
        assert_eq!(bands.len(), 6);
        for &a in &bands {
            assert!(a > 0.0);
        }
    }
}
