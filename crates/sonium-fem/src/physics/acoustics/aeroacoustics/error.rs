//! Errors returned by aeroacoustic source and propagation models.

use thiserror::Error;

/// Invalid input to an aeroacoustic source or propagation model.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AeroacousticsError {
    /// A dimensional parameter that must be finite and strictly positive was not.
    #[error("{name} must be finite and positive, got {value}")]
    NonPositiveParameter { name: &'static str, value: f64 },
    /// A dimensional parameter that must be finite was not.
    #[error("{name} must be finite, got {value}")]
    NonFiniteParameter { name: &'static str, value: f64 },
    /// A field contains a non-finite scalar or complex component.
    #[error("{name} contains a non-finite value")]
    NonFiniteField { name: &'static str },
    /// A source/surface point and observer coincide, making the Green function singular.
    #[error("observer coincides with a source or surface point")]
    SingularObserver,
    /// A periodic DRP grid is too short for the seven-point stencil.
    #[error("DRP grid requires at least {minimum} points, got {actual}")]
    GridTooSmall { minimum: usize, actual: usize },
    /// Pressure and velocity arrays do not have the configured grid length.
    #[error("state length mismatch: expected {expected}, pressure={pressure}, velocity={velocity}")]
    StateLengthMismatch {
        expected: usize,
        pressure: usize,
        velocity: usize,
    },
    /// The explicit step exceeds the conservative CFL contract.
    #[error("CFL number must be in (0, 1], got {0}")]
    InvalidCfl(f64),
}

pub(crate) fn require_positive(name: &'static str, value: f64) -> Result<(), AeroacousticsError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(AeroacousticsError::NonPositiveParameter { name, value })
    }
}

pub(crate) fn require_finite(name: &'static str, value: f64) -> Result<(), AeroacousticsError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(AeroacousticsError::NonFiniteParameter { name, value })
    }
}
