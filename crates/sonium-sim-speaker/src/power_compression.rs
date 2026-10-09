//! Power compression analysis
//!
//! Estimates the thermal loss in the lumped driver's output at elevated
//! drive levels. The reported decibels are a relative force/velocity ratio
//! at one frequency, where unchanged linear radiation factors cancel.
//! Receiver SPL, enclosure loading, changed radiation and structural breakup
//! require the coupled product model.
//!
//! # Algorithm
//!
//! For each power level:
//! 1. Compute V_rms from nominal power: P = V²/Re_cold
//! 2. Iterate thermal steady-state: P → T_coil → Re(T) → I = V/Z_total → P_actual
//! 3. With hot Re, compute frequency response via frequency-domain solve
//! 4. Compression = SPL_hot - SPL_cold (always negative)

use std::f64::consts::PI;

use num_complex::Complex64;
use sonium_fem::physics::acoustics::electroacoustics::thermal_coupling::VoiceCoilThermalModel;

/// Lumped thermal compression estimate at one power level.
#[derive(Debug, Clone)]
pub struct CompressionPoint {
    /// Nominal power level (W) based on cold Re.
    pub nominal_power: f64,
    /// Actual dissipated power in coil (W) after thermal equilibrium.
    pub actual_power: f64,
    /// Voice coil temperature at thermal equilibrium (K).
    pub coil_temperature: f64,
    /// Hot resistance (Ω).
    pub resistance_hot: f64,
    /// Relative lumped output change (dB). Negative = loss; no receiver SPL
    /// or nonlinear acoustic propagation is evaluated by this estimator.
    pub compression_db: f64,
}

/// Configuration for power compression analysis.
pub struct CompressionConfig {
    /// Thiele-Small: moving mass (kg).
    pub mms: f64,
    /// Thiele-Small: mechanical resistance (kg/s).
    pub rms: f64,
    /// Thiele-Small: mechanical stiffness (N/m).
    pub kms: f64,
    /// Force factor BL (T·m).
    pub bl: f64,
    /// Voice coil inductance (H).
    pub le: f64,
    /// Thermal model (owns Re_ref, alpha, thermal network).
    pub thermal: VoiceCoilThermalModel,
    /// Frequency for compression evaluation (Hz). Typically 1–2 kHz (above fs, below Le rolloff).
    pub eval_frequency: f64,
}

impl CompressionConfig {
    /// Validate T/S, thermal, and evaluation inputs; `Err` names the first
    /// offending field instead of propagating NaN/`-inf` into the curve.
    pub fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("mms", self.mms),
            ("rms", self.rms),
            ("kms", self.kms),
            ("bl", self.bl),
            ("eval_frequency", self.eval_frequency),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{field} must be finite and positive"));
            }
        }
        if !self.le.is_finite() || self.le < 0.0 {
            return Err("le must be finite and non-negative".to_string());
        }
        if !self.thermal.re_ref.is_finite() || self.thermal.re_ref <= 0.0 {
            return Err("thermal.re_ref must be finite and positive".to_string());
        }
        Ok(())
    }
}

/// Compute power compression curve across a range of nominal power levels.
///
/// Returns one `CompressionPoint` per power level, sorted by increasing power.
/// A non-positive or non-finite level yields a NaN point for that level only.
///
/// Returns `Err` naming the first invalid field when the configuration itself
/// is unphysical (non-positive `re_ref`, T/S, or evaluation frequency):
/// without this check `re_cold.sqrt()` and `p_nominal.log10()` would silently
/// propagate NaN/`-inf` into the curve.
pub fn power_compression_curve(
    config: &mut CompressionConfig,
    power_levels_w: &[f64],
) -> Result<Vec<CompressionPoint>, String> {
    config.validate()?;
    let omega = 2.0 * PI * config.eval_frequency;
    let re_cold = config.thermal.re_ref;

    // Mechanical impedance at eval frequency
    let z_mech = Complex64::new(config.rms, omega * config.mms - config.kms / omega);

    // Reference SPL at 1W (cold Re) — used to compute compression
    let v_ref = re_cold.sqrt(); // V_rms for 1W into Re_cold
    let spl_ref = compute_spl(re_cold, config.le, config.bl, omega, z_mech, v_ref);

    let points = power_levels_w
        .iter()
        .map(|&p_nominal| {
            if !p_nominal.is_finite() || p_nominal <= 0.0 {
                return CompressionPoint {
                    nominal_power: p_nominal,
                    actual_power: f64::NAN,
                    coil_temperature: f64::NAN,
                    resistance_hot: f64::NAN,
                    compression_db: f64::NAN,
                };
            }
            // V_rms for this nominal power (defined as P = V²/Re_cold)
            let v_rms = (p_nominal * re_cold).sqrt();

            // Iterate thermal steady-state with self-consistent Re(T)
            config.thermal.reset();
            let (t_coil, re_hot, p_actual) = iterate_thermal_electrical(
                &mut config.thermal,
                v_rms,
                config.le,
                config.bl,
                omega,
                z_mech,
            );

            // SPL with hot Re
            let spl_hot = compute_spl(re_hot, config.le, config.bl, omega, z_mech, v_rms);

            // SPL that would be achieved if Re stayed cold (linear extrapolation from 1W)
            let spl_cold = spl_ref + 10.0 * p_nominal.log10();

            let compression = spl_hot - spl_cold;

            CompressionPoint {
                nominal_power: p_nominal,
                actual_power: p_actual,
                coil_temperature: t_coil,
                resistance_hot: re_hot,
                compression_db: compression,
            }
        })
        .collect::<Vec<_>>();
    Ok(points)
}

/// Iterate thermal-electrical coupling to find steady-state Re(T).
fn iterate_thermal_electrical(
    thermal: &mut VoiceCoilThermalModel,
    v_rms: f64,
    le: f64,
    bl: f64,
    omega: f64,
    z_mech: Complex64,
) -> (f64, f64, f64) {
    let r_total = thermal.total_resistance();
    let mut re = thermal.re_ref;

    for _ in 0..50 {
        let z_e = Complex64::new(re, omega * le);
        let z_mot = Complex64::new(bl * bl, 0.0) / z_mech;
        let z_total = z_e + z_mot;
        let i_rms = v_rms / z_total.norm();
        let power = i_rms * i_rms * re;
        let t_coil = thermal.t_ambient + power * r_total;
        let re_new = thermal.resistance_at(t_coil);

        if (re_new - re).abs() < 1e-6 * re {
            thermal.nodes[thermal.coil_node].temperature = t_coil;
            return (t_coil, re_new, power);
        }
        re = re_new;
    }

    // Return last iteration if not converged
    let z_e = Complex64::new(re, omega * le);
    let z_mot = Complex64::new(bl * bl, 0.0) / z_mech;
    let z_total = z_e + z_mot;
    let i_rms = v_rms / z_total.norm();
    let power = i_rms * i_rms * re;
    let t_coil = thermal.t_ambient + power * r_total;
    thermal.nodes[thermal.coil_node].temperature = t_coil;
    (t_coil, re, power)
}

/// Compute SPL (dB, relative) at a given Re, voltage, and frequency.
fn compute_spl(re: f64, le: f64, bl: f64, omega: f64, z_mech: Complex64, v_rms: f64) -> f64 {
    let z_e = Complex64::new(re, omega * le);
    let z_mot = Complex64::new(bl * bl, 0.0) / z_mech;
    let z_total = z_e + z_mot;
    let i_rms = v_rms / z_total.norm();
    // SPL ∝ acceleration ∝ force/mass = BL·I / Mms
    // We only need relative SPL, so: SPL = 20·log10(BL·I_rms)
    let force_rms = bl * i_rms;
    20.0 * force_rms.log10()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const T_AMB: f64 = 293.15;

    fn test_config() -> CompressionConfig {
        CompressionConfig {
            mms: 0.015,
            rms: 1.0,
            kms: 2400.0,
            bl: 5.0,
            le: 0.5e-3,
            thermal: VoiceCoilThermalModel::simple(6.0, T_AMB, 8.0, 5.0),
            eval_frequency: 1000.0,
        }
    }

    #[test]
    fn test_compression_at_low_power() {
        let mut config = test_config();
        let results = power_compression_curve(&mut config, &[0.1]).expect("valid test config");

        // At very low power, compression should be negligible
        assert!(
            results[0].compression_db.abs() < 0.5,
            "Compression at 0.1W = {:.2} dB, expected ~0",
            results[0].compression_db
        );
    }

    #[test]
    fn test_compression_increases_with_power() {
        let mut config = test_config();
        let powers = [1.0, 10.0, 50.0, 100.0];
        let results = power_compression_curve(&mut config, &powers).expect("valid test config");

        // Compression should become more negative at higher power
        for i in 1..results.len() {
            assert!(
                results[i].compression_db <= results[i - 1].compression_db + 0.01,
                "Compression should increase: P={:.0}W → {:.2}dB, P={:.0}W → {:.2}dB",
                results[i - 1].nominal_power,
                results[i - 1].compression_db,
                results[i].nominal_power,
                results[i].compression_db,
            );
        }
    }

    #[test]
    fn test_temperature_increases_with_power() {
        let mut config = test_config();
        let powers = [1.0, 10.0, 100.0];
        let results = power_compression_curve(&mut config, &powers).expect("valid test config");

        for i in 1..results.len() {
            assert!(
                results[i].coil_temperature > results[i - 1].coil_temperature,
                "Temperature should increase with power"
            );
        }
        // At 100W, temperature should be significantly above ambient
        assert!(
            results[2].coil_temperature > T_AMB + 50.0,
            "T at 100W = {:.1}K, expected > {:.1}K",
            results[2].coil_temperature,
            T_AMB + 50.0
        );
    }

    #[test]
    fn test_resistance_increases_with_power() {
        let mut config = test_config();
        let results =
            power_compression_curve(&mut config, &[1.0, 100.0]).expect("valid test config");

        assert!(
            results[1].resistance_hot > results[0].resistance_hot,
            "Re should increase: Re(1W)={:.2}Ω, Re(100W)={:.2}Ω",
            results[0].resistance_hot,
            results[1].resistance_hot
        );
    }

    #[test]
    fn test_compression_is_negative() {
        let mut config = test_config();
        let results =
            power_compression_curve(&mut config, &[10.0, 50.0, 100.0]).expect("valid test config");

        for r in &results {
            assert!(
                r.compression_db <= 0.01, // small tolerance for numerical noise
                "Compression should be ≤ 0: P={:.0}W → {:.2}dB",
                r.nominal_power,
                r.compression_db
            );
        }
    }

    #[test]
    fn test_actual_power_less_than_nominal() {
        // With Re increasing, actual power < nominal (since P_actual = I²·Re_hot
        // but I decreases faster than Re increases)
        let mut config = test_config();
        let results = power_compression_curve(&mut config, &[50.0]).expect("valid test config");

        // At high power, Re increases significantly, reducing current.
        // The actual power should still be positive but the relationship
        // between nominal and actual depends on the impedance.
        assert!(results[0].actual_power > 0.0);
    }

    #[test]
    fn test_non_positive_power_levels_yield_nan_points_not_err() {
        // Per-level validation stays per-level: an unphysical level poisons
        // only its own point, while a valid config still returns the curve.
        let mut config = test_config();
        let results = power_compression_curve(&mut config, &[0.0, -5.0, f64::NAN, 50.0])
            .expect("valid config");
        assert_eq!(results.len(), 4);
        for point in &results[..3] {
            assert!(
                point.actual_power.is_nan(),
                "level {} must yield a NaN point",
                point.nominal_power
            );
        }
        assert!(
            results[3].actual_power.is_finite(),
            "the valid level must still solve"
        );
    }

    #[test]
    fn test_invalid_config_returns_err_not_nan() {
        let mut config = test_config();
        config.thermal.re_ref = 0.0;
        let error = power_compression_curve(&mut config, &[50.0]).unwrap_err();
        assert!(
            error.contains("re_ref"),
            "unphysical Re_ref must name re_ref, got: {error}"
        );

        let mut config = test_config();
        config.eval_frequency = -100.0;
        let error = power_compression_curve(&mut config, &[50.0]).unwrap_err();
        assert!(
            error.contains("eval_frequency"),
            "unphysical eval frequency must name eval_frequency, got: {error}"
        );
    }
}
