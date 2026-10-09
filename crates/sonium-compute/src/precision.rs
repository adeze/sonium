//! Explicit mixed-precision acceptance policies.
//!
//! Lower precision is useful for assembly and inner Krylov work, but an
//! optimization pipeline must not silently turn a fast approximate solve into
//! a physically accepted result.  [`MixedPrecisionPolicy`] makes the two
//! mandatory gates explicit: an f64 true residual and an f64 physical
//! observable check.

use crate::gmres_ir::GmresIrConfig;
use num_complex::Complex64;

/// Acceptance policy for an outer-f64/inner-f32 solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixedPrecisionPolicy {
    /// Relative tolerance for the outer f64 true residual.
    pub outer_tolerance: f64,
    /// Relative tolerance passed to the inner f32 solve.
    pub inner_tolerance: f32,
    /// Period for diagnostic f64 true-residual recomputation. A value of zero
    /// is normalized to one by [`Self::ir_config`].
    pub true_residual_check_period: usize,
    /// Maximum relative error in the selected f64 physical observable.
    pub physical_tolerance: f64,
}

impl Default for MixedPrecisionPolicy {
    fn default() -> Self {
        Self {
            outer_tolerance: 1.0e-8,
            inner_tolerance: 1.0e-3,
            true_residual_check_period: 1,
            physical_tolerance: 1.0e-6,
        }
    }
}

impl MixedPrecisionPolicy {
    /// Build a GMRES-IR configuration with an explicit f64 residual cadence.
    pub fn ir_config(
        self,
        max_outer_iterations: usize,
        max_inner_iterations: usize,
    ) -> GmresIrConfig<Complex64> {
        GmresIrConfig {
            max_outer_iterations,
            outer_tolerance: self.outer_tolerance,
            max_inner_iterations,
            inner_tolerance: self.inner_tolerance,
            restart_size: None,
            adaptive_inner_tolerance: true,
            true_residual_check_period: Some(self.true_residual_check_period.max(1)),
        }
    }

    /// Validate the two f64 gates required before an approximate result can
    /// enter a physical objective or optimizer score.
    pub fn validate_gates(
        self,
        true_residual_f64: f64,
        physical_relative_error_f64: f64,
    ) -> Result<(), MixedPrecisionGateError> {
        if !true_residual_f64.is_finite() {
            return Err(MixedPrecisionGateError::NonFiniteTrueResidual {
                value: true_residual_f64,
            });
        }
        if true_residual_f64 > self.outer_tolerance {
            return Err(MixedPrecisionGateError::TrueResidual {
                value: true_residual_f64,
                tolerance: self.outer_tolerance,
            });
        }
        if !physical_relative_error_f64.is_finite() {
            return Err(MixedPrecisionGateError::NonFinitePhysicalObservable {
                value: physical_relative_error_f64,
            });
        }
        if physical_relative_error_f64 > self.physical_tolerance {
            return Err(MixedPrecisionGateError::PhysicalObservable {
                value: physical_relative_error_f64,
                tolerance: self.physical_tolerance,
            });
        }
        Ok(())
    }
}

/// Failure of one of the mandatory f64 acceptance gates.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum MixedPrecisionGateError {
    #[error("f64 true residual is not finite: {value}")]
    NonFiniteTrueResidual { value: f64 },
    #[error("f64 true residual {value:.3e} exceeds tolerance {tolerance:.3e}")]
    TrueResidual { value: f64, tolerance: f64 },
    #[error("f64 physical observable error is not finite: {value}")]
    NonFinitePhysicalObservable { value: f64 },
    #[error("f64 physical observable error {value:.3e} exceeds tolerance {tolerance:.3e}")]
    PhysicalObservable { value: f64, tolerance: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_requires_both_f64_gates() {
        let policy = MixedPrecisionPolicy::default();
        assert!(policy.validate_gates(1.0e-9, 1.0e-7).is_ok());
        assert!(matches!(
            policy.validate_gates(1.0e-4, 1.0e-7),
            Err(MixedPrecisionGateError::TrueResidual { .. })
        ));
        assert!(matches!(
            policy.validate_gates(1.0e-9, 1.0e-4),
            Err(MixedPrecisionGateError::PhysicalObservable { .. })
        ));
    }

    #[test]
    fn ir_config_always_records_true_residual_checks() {
        let config = MixedPrecisionPolicy {
            true_residual_check_period: 0,
            ..MixedPrecisionPolicy::default()
        }
        .ir_config(10, 5);
        assert_eq!(config.true_residual_check_period, Some(1));
        assert!(config.adaptive_inner_tolerance);
    }
}
