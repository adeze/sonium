//! Interface power ledger for aperture-coupled interior/exterior solves.
//!
//! For time-harmonic acoustics in the shared `exp(+ikr)` phasor convention,
//! the active power crossing one aperture triangle with area `a`, average
//! pressure `p`, and outward velocity `v` is `0.5 * Re(p * conj(v)) * a`.
//! Summed over the aperture, the interior side (FEM nodal pressures averaged
//! per face, aperture velocities as unknowns) and the exterior side (BEM
//! element pressures and the same velocity pattern) must agree: any mismatch
//! is lost or gained power, i.e. a coupling bug. The ledger reports both
//! sides independently plus the conservation residual, so tests can assert
//! the balance instead of only comparing nodal fields.

use num_complex::Complex64;

/// Failures while auditing interface power.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PowerLedgerError {
    #[error("power ledger needs a non-empty aperture")]
    EmptyAperture,
    #[error("power ledger length mismatch: {field} has {got} entries, expected {expected}")]
    Length {
        field: &'static str,
        got: usize,
        expected: usize,
    },
    #[error("power ledger entry {index} in {field} is not finite")]
    NonFinite { field: &'static str, index: usize },
}

/// Active power audit across one aperture interface.
///
/// All slices are per-aperture-triangle: `areas` in m², pressures in Pa
/// (complex amplitudes), velocities in m/s outward.
#[derive(Debug, Clone, PartialEq)]
pub struct InterfacePowerLedger {
    /// `0.5 * Re(p * conj(v)) * a` summed with interior pressures.
    pub interior_active_power_w: f64,
    /// `0.5 * Re(p * conj(v)) * a` summed with exterior pressures.
    pub exterior_active_power_w: f64,
    /// `|interior - exterior|`, the conservation residual in watts.
    pub conservation_residual_w: f64,
    /// Scale-aware residual: `residual / max(|interior|, |exterior|, tiny)`.
    pub relative_conservation_residual: f64,
}

impl InterfacePowerLedger {
    /// Audit one interface state.
    pub fn audit(
        areas: &[f64],
        interior_pressure: &[Complex64],
        exterior_pressure: &[Complex64],
        outward_velocity: &[Complex64],
    ) -> Result<Self, PowerLedgerError> {
        if areas.is_empty() {
            return Err(PowerLedgerError::EmptyAperture);
        }
        let n = areas.len();
        for (field, values) in [
            ("interior_pressure", interior_pressure),
            ("exterior_pressure", exterior_pressure),
            ("outward_velocity", outward_velocity),
        ] {
            if values.len() != n {
                return Err(PowerLedgerError::Length {
                    field,
                    got: values.len(),
                    expected: n,
                });
            }
        }
        for (index, area) in areas.iter().enumerate() {
            if !area.is_finite() || *area <= 0.0 {
                return Err(PowerLedgerError::NonFinite {
                    field: "areas",
                    index,
                });
            }
        }
        for (field, values) in [
            ("interior_pressure", interior_pressure),
            ("exterior_pressure", exterior_pressure),
            ("outward_velocity", outward_velocity),
        ] {
            for (index, value) in values.iter().enumerate() {
                if !value.is_finite() {
                    return Err(PowerLedgerError::NonFinite { field, index });
                }
            }
        }
        let side = |pressure: &[Complex64]| {
            pressure
                .iter()
                .zip(outward_velocity.iter())
                .zip(areas.iter())
                .map(|((p, v), a)| 0.5 * (p * v.conj()).re * a)
                .sum::<f64>()
        };
        let interior_active_power_w = side(interior_pressure);
        let exterior_active_power_w = side(exterior_pressure);
        let conservation_residual_w = (interior_active_power_w - exterior_active_power_w).abs();
        let scale = interior_active_power_w
            .abs()
            .max(exterior_active_power_w.abs())
            .max(1.0e-300);
        Ok(Self {
            interior_active_power_w,
            exterior_active_power_w,
            conservation_residual_w,
            relative_conservation_residual: conservation_residual_w / scale,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_aperture_balances_when_both_sides_agree() {
        let areas = vec![0.5, 0.5];
        let pressure = vec![Complex64::new(2.0, 1.0), Complex64::new(2.0, 1.0)];
        let velocity = vec![Complex64::new(0.1, -0.05), Complex64::new(0.1, -0.05)];
        let ledger = InterfacePowerLedger::audit(&areas, &pressure, &pressure, &velocity)
            .expect("valid ledger input");
        // 0.5 * Re((2+i)(0.1+0.05i)) * 1.0 = 0.5 * 0.15 = 0.075.
        assert!((ledger.interior_active_power_w - 0.075).abs() < 1.0e-12);
        assert_eq!(
            ledger.interior_active_power_w,
            ledger.exterior_active_power_w
        );
        assert_eq!(ledger.conservation_residual_w, 0.0);
        assert_eq!(ledger.relative_conservation_residual, 0.0);
    }

    #[test]
    fn pressure_jump_reports_as_lost_power() {
        let areas = vec![1.0];
        let interior = vec![Complex64::new(3.0, 0.0)];
        let exterior = vec![Complex64::new(1.0, 0.0)];
        let velocity = vec![Complex64::new(1.0, 0.0)];
        let ledger = InterfacePowerLedger::audit(&areas, &interior, &exterior, &velocity)
            .expect("valid ledger input");
        assert!((ledger.interior_active_power_w - 1.5).abs() < 1.0e-12);
        assert!((ledger.exterior_active_power_w - 0.5).abs() < 1.0e-12);
        assert!((ledger.conservation_residual_w - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn malformed_inputs_fail_closed() {
        assert_eq!(
            InterfacePowerLedger::audit(&[], &[], &[], &[]),
            Err(PowerLedgerError::EmptyAperture)
        );
        let areas = vec![1.0, 1.0];
        let one = vec![Complex64::new(1.0, 0.0)];
        let two = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        assert!(matches!(
            InterfacePowerLedger::audit(&areas, &one, &two, &two),
            Err(PowerLedgerError::Length { .. })
        ));
        let bad = vec![Complex64::new(f64::NAN, 0.0), Complex64::new(1.0, 0.0)];
        assert!(matches!(
            InterfacePowerLedger::audit(&areas, &bad, &two, &two),
            Err(PowerLedgerError::NonFinite { .. })
        ));
        assert!(matches!(
            InterfacePowerLedger::audit(&[0.0, 1.0], &two, &two, &two),
            Err(PowerLedgerError::NonFinite { .. })
        ));
    }
}
