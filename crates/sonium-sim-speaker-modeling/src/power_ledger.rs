//! Cross-domain harmonic power closure.
//!
//! A converged field solution is not a physically accepted solution until its
//! supplied, dissipated, and exported powers close.  This ledger is deliberately
//! independent of a particular FEM assembly so axisymmetric, sector, resolved
//! acoustic, and future full-3D paths can use the same gate.

use serde::{Deserialize, Serialize};

/// Time-average power ledger for one RMS harmonic state.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PowerLedger {
    pub supplied_w: f64,
    pub copper_loss_w: f64,
    pub magnetic_loss_w: f64,
    pub mechanical_loss_w: f64,
    pub structural_loss_w: f64,
    pub internal_acoustic_loss_w: f64,
    pub exterior_acoustic_power_w: f64,
    pub thermal_storage_w: f64,
    pub residual_w: f64,
}

/// Relative and absolute closure tolerances.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PowerLedgerTolerance {
    pub absolute_w: f64,
    pub relative: f64,
}

impl Default for PowerLedgerTolerance {
    fn default() -> Self {
        Self {
            absolute_w: 1.0e-10,
            relative: 1.0e-7,
        }
    }
}

/// Detailed result of a power-ledger gate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PowerClosure {
    pub supplied_w: f64,
    pub accounted_w: f64,
    pub residual_w: f64,
    pub relative_residual: f64,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PowerLedgerError {
    #[error("power ledger field is non-finite: {0}")]
    NonFinite(&'static str),
    #[error("power ledger field is negative: {0}")]
    Negative(&'static str),
    #[error("power ledger tolerances are invalid")]
    InvalidTolerance,
    #[error(
        "power ledger does not close: residual={residual_w:.3e} W, relative={relative_residual:.3e}"
    )]
    NotClosed {
        residual_w: f64,
        relative_residual: f64,
    },
}

impl PowerLedger {
    /// Build a ledger and compute its residual from supplied and accounted
    /// harmonic powers. `thermal_storage_w` is included for transient or
    /// slowly varying thermal states and may be signed.
    #[allow(clippy::too_many_arguments)]
    pub fn from_components(
        supplied_w: f64,
        copper_loss_w: f64,
        magnetic_loss_w: f64,
        mechanical_loss_w: f64,
        structural_loss_w: f64,
        internal_acoustic_loss_w: f64,
        exterior_acoustic_power_w: f64,
        thermal_storage_w: f64,
    ) -> Result<Self, PowerLedgerError> {
        let ledger = Self {
            supplied_w,
            copper_loss_w,
            magnetic_loss_w,
            mechanical_loss_w,
            structural_loss_w,
            internal_acoustic_loss_w,
            exterior_acoustic_power_w,
            thermal_storage_w,
            residual_w: supplied_w
                - copper_loss_w
                - magnetic_loss_w
                - mechanical_loss_w
                - structural_loss_w
                - internal_acoustic_loss_w
                - exterior_acoustic_power_w
                - thermal_storage_w,
        };
        ledger.validate_finite_and_passive()?;
        Ok(ledger)
    }

    fn validate_finite_and_passive(&self) -> Result<(), PowerLedgerError> {
        for (name, value) in [
            ("supplied_w", self.supplied_w),
            ("copper_loss_w", self.copper_loss_w),
            ("magnetic_loss_w", self.magnetic_loss_w),
            ("mechanical_loss_w", self.mechanical_loss_w),
            ("structural_loss_w", self.structural_loss_w),
            ("internal_acoustic_loss_w", self.internal_acoustic_loss_w),
            ("exterior_acoustic_power_w", self.exterior_acoustic_power_w),
            ("thermal_storage_w", self.thermal_storage_w),
            ("residual_w", self.residual_w),
        ] {
            if !value.is_finite() {
                return Err(PowerLedgerError::NonFinite(name));
            }
        }
        if self.supplied_w < 0.0 {
            return Err(PowerLedgerError::Negative("supplied_w"));
        }
        for (name, value) in [
            ("copper_loss_w", self.copper_loss_w),
            ("magnetic_loss_w", self.magnetic_loss_w),
            ("mechanical_loss_w", self.mechanical_loss_w),
            ("structural_loss_w", self.structural_loss_w),
            ("internal_acoustic_loss_w", self.internal_acoustic_loss_w),
            ("exterior_acoustic_power_w", self.exterior_acoustic_power_w),
        ] {
            if value < 0.0 {
                return Err(PowerLedgerError::Negative(name));
            }
        }
        Ok(())
    }

    /// Validate the ledger and return the accounted-power breakdown.
    pub fn close(self, tolerance: PowerLedgerTolerance) -> Result<PowerClosure, PowerLedgerError> {
        if !tolerance.absolute_w.is_finite()
            || tolerance.absolute_w < 0.0
            || !tolerance.relative.is_finite()
            || tolerance.relative < 0.0
        {
            return Err(PowerLedgerError::InvalidTolerance);
        }
        self.validate_finite_and_passive()?;
        let accounted_w = self.supplied_w - self.residual_w;
        let relative_residual = self.residual_w.abs() / self.supplied_w.abs().max(1.0e-15);
        if self.residual_w.abs() > tolerance.absolute_w && relative_residual > tolerance.relative {
            return Err(PowerLedgerError::NotClosed {
                residual_w: self.residual_w,
                relative_residual,
            });
        }
        Ok(PowerClosure {
            supplied_w: self.supplied_w,
            accounted_w,
            residual_w: self.residual_w,
            relative_residual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_closes_and_rejects_active_loss() {
        let ledger = PowerLedger::from_components(10.0, 2.0, 1.0, 1.0, 1.0, 2.0, 3.0, 0.0).unwrap();
        let closure = ledger.close(PowerLedgerTolerance::default()).unwrap();
        assert_eq!(closure.residual_w, 0.0);
        assert_eq!(closure.accounted_w, 10.0);

        assert!(PowerLedger::from_components(1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn ledger_reports_unclosed_state() {
        let mut ledger =
            PowerLedger::from_components(10.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.0).unwrap();
        ledger.residual_w = 1.0;
        assert!(matches!(
            ledger.close(PowerLedgerTolerance::default()),
            Err(PowerLedgerError::NotClosed { .. })
        ));
    }
}
