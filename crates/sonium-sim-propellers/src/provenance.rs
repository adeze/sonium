//! Typed model-fidelity labels for propeller outputs.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::execution_evidence::PropellerExecutionProvenance;

/// Model family represented by a propeller result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropellerModelFidelity {
    /// Axisymmetric empirical/source-spectrum calculation.
    SourceSpectrumSurrogate,
    /// Deterministic 3-D panel geometry combined with source surrogates.
    PanelGeometrySurrogate,
    /// Lifting-line aerodynamic model (reserved until resolved).
    LiftingLine,
    /// Free-wake aerodynamic model (reserved until resolved).
    FreeWake,
    /// Steady RANS model (reserved until resolved).
    SteadyRans,
    /// Unsteady RANS plus acoustic analogy (reserved until resolved).
    UnsteadyRansFwh,
    /// External measured reference, not a generated simulation result.
    MeasuredReference,
}

impl fmt::Display for PropellerModelFidelity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::SourceSpectrumSurrogate => "source-spectrum-surrogate",
            Self::PanelGeometrySurrogate => "panel-geometry-surrogate",
            Self::LiftingLine => "lifting-line",
            Self::FreeWake => "free-wake",
            Self::SteadyRans => "steady-rans",
            Self::UnsteadyRansFwh => "unsteady-rans-fwh",
            Self::MeasuredReference => "measured-reference",
        };
        f.write_str(label)
    }
}

/// Provenance flags that qualify what a result is allowed to claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropellerResultProvenance {
    /// Declared model family.
    pub fidelity: PropellerModelFidelity,
    /// Whether a resolved wake solution contributed to the result.
    pub resolved_wake: bool,
    /// Whether observer directivity/phase was resolved by the model.
    pub resolved_observer_directivity: bool,
    /// Whether a declared mesh/time/revolution convergence gate passed.
    pub converged: bool,
    /// Whether public measured-data correlation passed.
    pub measured_correlation: bool,
    /// Backend, precision, host, and checksum evidence for the result.
    pub execution: PropellerExecutionProvenance,
}

impl PropellerResultProvenance {
    /// Provenance for the current production panel/source surrogate.
    pub fn panel_geometry_surrogate() -> Self {
        Self {
            fidelity: PropellerModelFidelity::PanelGeometrySurrogate,
            resolved_wake: false,
            resolved_observer_directivity: false,
            converged: false,
            measured_correlation: false,
            execution: PropellerExecutionProvenance::cpu_surrogate(
                "unbound-production-result",
                b"unbound-production-input",
                b"unbound-production-output",
            ),
        }
    }

    /// Provenance for a concrete deterministic CPU evaluation, including the
    /// canonical input/output digests used by parity and release tooling.
    pub fn panel_geometry_surrogate_for(
        seed: impl Into<String>,
        input: &[u8],
        output: &[u8],
    ) -> Self {
        Self {
            fidelity: PropellerModelFidelity::PanelGeometrySurrogate,
            resolved_wake: false,
            resolved_observer_directivity: false,
            converged: false,
            measured_correlation: false,
            execution: PropellerExecutionProvenance::cpu_surrogate(seed, input, output),
        }
    }

    /// Whether the flags are internally consistent with the bounded output
    /// path and therefore safe for the current gallery/export contract.
    pub fn is_bounded_surrogate(&self) -> bool {
        self.fidelity == PropellerModelFidelity::PanelGeometrySurrogate
            && !self.resolved_wake
            && !self.resolved_observer_directivity
            && !self.converged
            && !self.measured_correlation
            && self.execution.is_publishable_cpu_surrogate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_surrogate_provenance_is_explicit_and_fail_closed() {
        let provenance = PropellerResultProvenance::panel_geometry_surrogate();
        assert_eq!(provenance.fidelity.to_string(), "panel-geometry-surrogate");
        assert!(provenance.is_bounded_surrogate());

        let mut claimed_resolved = provenance.clone();
        claimed_resolved.resolved_wake = true;
        assert!(!claimed_resolved.is_bounded_surrogate());

        let mut claimed_measured = provenance.clone();
        claimed_measured.measured_correlation = true;
        assert!(!claimed_measured.is_bounded_surrogate());

        let mut claimed_converged = provenance.clone();
        claimed_converged.converged = true;
        assert!(!claimed_converged.is_bounded_surrogate());

        let mut missing_backend = provenance;
        missing_backend.execution.actual_backend = None;
        assert!(!missing_backend.is_bounded_surrogate());
    }
}
