//! Canonical acoustic FEM sign and orientation conventions.
//!
//! FEM boundary operators consume the same convention metadata as BEM.  Keep
//! the adaptation at this boundary so an exchange payload cannot silently
//! change the outgoing-wave sign or the meaning of a boundary normal.

use num_complex::Complex64;

/// Harmonic and orientation conventions used by the acoustic FEM operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FemConvention {
    /// Sign in the time factor `e^{time_sign i ω t}`.
    pub time_sign: i8,
    /// Sign in the outgoing spatial factor `e^{spatial_sign i k r}`.
    pub spatial_sign: i8,
    /// Whether boundary normals point out of the physical domain.
    pub outward_normals: bool,
    /// Whether amplitudes in the exchange payload are RMS values.
    pub rms_amplitudes: bool,
}

/// Failure while adapting exchange convention metadata to FEM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FemConventionError {
    /// FEM boundary traces in this API require outward physical-domain normals.
    InwardNormals,
    /// FEM geometry APIs require the global right-handed world frame.
    NonWorldCoordinateFrame,
}

/// Failure while adapting a complete XEM multiphysics problem to acoustic FEM.
#[derive(Debug, thiserror::Error)]
pub enum FemProblemError {
    /// The problem failed schema/reference validation.
    #[error("invalid XEM multiphysics problem: {0}")]
    InvalidProblem(#[from] sonium_xem::IrValidationError),
    /// The XEM coupling graph cannot be admitted by a partitioned FEM solve.
    #[error("invalid XEM coupling graph: {0}")]
    CouplingGraph(#[from] sonium_xem::CouplingGraphError),
    /// The problem does not contain an acoustics domain.
    #[error("XEM problem has no acoustics domain")]
    MissingAcousticsDomain,
    /// Its shared convention cannot be consumed by FEM.
    #[error("incompatible FEM convention: {0}")]
    Convention(#[from] FemConventionError),
    /// The selected XEM pressure field is not consumable by Helmholtz FEM.
    #[error("incompatible XEM pressure field `{field_id}`: {reason}")]
    Field {
        /// Field identifier supplied by the exchange document.
        field_id: String,
        /// Why the field cannot be bound to a FEM pressure result.
        reason: String,
    },
    /// The legacy Helmholtz backend rejected the supplied mesh or matrix.
    #[error("invalid FEM backend input: {0}")]
    Backend(String),
}

impl std::fmt::Display for FemConventionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InwardNormals => write!(f, "FEM requires outward physical-domain normals"),
            Self::NonWorldCoordinateFrame => {
                write!(f, "FEM requires the global right-handed world frame")
            }
        }
    }
}

impl std::error::Error for FemConventionError {}

impl FemConvention {
    /// Adapt one typed XEM acoustic medium to the FEM sign contract.
    pub fn from_medium(medium: &sonium_xem::AcousticMedium) -> Result<Self, FemConventionError> {
        Self::from_exchange(&medium.conventions)
    }

    /// Adapt the versioned XEM convention registry to the FEM sign contract.
    pub fn from_exchange(
        registry: &sonium_xem::ConventionRegistry,
    ) -> Result<Self, FemConventionError> {
        if !registry.outward_normals {
            return Err(FemConventionError::InwardNormals);
        }
        if !matches!(
            registry.coordinate_frame,
            sonium_xem::CoordinateFrame::WorldRightHanded
        ) {
            return Err(FemConventionError::NonWorldCoordinateFrame);
        }

        let (time_sign, spatial_sign) = registry.harmonic_time.signs();

        Ok(Self {
            time_sign,
            spatial_sign,
            outward_normals: registry.outward_normals,
            rms_amplitudes: registry.rms_amplitudes,
        })
    }

    /// Validate and adapt a complete XEM problem at the FEM boundary.
    pub fn from_problem(
        problem: &sonium_xem::MultiphysicsProblem,
    ) -> Result<Self, FemProblemError> {
        problem.validate()?;
        if !problem
            .domains
            .iter()
            .any(|domain| matches!(domain.kind, sonium_xem::DomainKind::Acoustics))
        {
            return Err(FemProblemError::MissingAcousticsDomain);
        }
        Ok(Self::from_exchange(&problem.conventions)?)
    }

    /// Validate the field contract consumed by the current Helmholtz FEM
    /// adapter. This delegates to the shared XEM pressure-field contract so
    /// BEM and FEM reject the same units, domain, and storage mismatches.
    pub fn pressure_field<'a>(
        problem: &'a sonium_xem::MultiphysicsProblem,
        field_id: &str,
    ) -> Result<&'a sonium_xem::FieldSpec, FemProblemError> {
        problem.validate()?;
        problem
            .acoustic_complex_pressure_field(field_id)
            .map_err(|error| match error {
                sonium_xem::IrValidationError::MissingReference { .. } => FemProblemError::Field {
                    field_id: field_id.to_owned(),
                    reason: "field is not declared".into(),
                },
                other => FemProblemError::Field {
                    field_id: field_id.to_owned(),
                    reason: other.to_string(),
                },
            })
    }

    /// Return the Robin coefficient for a first-order outgoing-wave boundary.
    ///
    /// The FEM Robin form is `∂p/∂n + αp = 0`.  For the outgoing spatial
    /// factor `exp(spatial_sign i k r)`, `α = -spatial_sign i k`.
    pub fn outgoing_robin_coefficient(self, wavenumber: Complex64) -> Complex64 {
        Complex64::new(0.0, -f64::from(self.spatial_sign)) * wavenumber
    }
}

/// The convention used by the existing acoustic FEM APIs.
pub const ACOUSTIC_CONVENTION: FemConvention = FemConvention {
    time_sign: -1,
    spatial_sign: 1,
    outward_normals: true,
    rms_amplitudes: true,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_exchange_convention_matches_fem() {
        let convention = FemConvention::from_exchange(&sonium_xem::ConventionRegistry::default())
            .expect("the canonical exchange convention is FEM-compatible");
        assert_eq!(convention, ACOUSTIC_CONVENTION);
    }

    #[test]
    fn typed_acoustic_medium_adapter_preserves_shared_convention_contract() {
        let medium = sonium_xem::AcousticMedium::try_new(1_000.0, 343.0, 1.2)
            .expect("canonical typed medium");
        assert_eq!(
            FemConvention::from_medium(&medium).expect("typed medium is FEM-compatible"),
            ACOUSTIC_CONVENTION
        );

        let mut invalid = medium;
        invalid.conventions.outward_normals = false;
        assert_eq!(
            FemConvention::from_medium(&invalid),
            Err(FemConventionError::InwardNormals)
        );
    }

    #[test]
    fn exchange_adaptation_tracks_time_sign_and_rejects_orientation() {
        let mut registry = sonium_xem::ConventionRegistry::default();
        registry.harmonic_time = sonium_xem::HarmonicConvention::ExpPlusIwt;
        let convention = FemConvention::from_exchange(&registry).expect("valid plus-i convention");
        assert_eq!((convention.time_sign, convention.spatial_sign), (1, -1));
        assert_eq!(
            convention.outgoing_robin_coefficient(Complex64::new(2.0, 0.0)),
            Complex64::new(0.0, 2.0)
        );

        registry.outward_normals = false;
        assert_eq!(
            FemConvention::from_exchange(&registry),
            Err(FemConventionError::InwardNormals)
        );

        registry.outward_normals = true;
        registry.coordinate_frame = sonium_xem::CoordinateFrame::SourceLocalRightHanded;
        assert_eq!(
            FemConvention::from_exchange(&registry),
            Err(FemConventionError::NonWorldCoordinateFrame)
        );
    }

    #[test]
    fn complete_xem_problem_is_validated_at_fem_boundary() {
        let mut problem = sonium_xem::MultiphysicsProblem {
            id: "acoustic-case".into(),
            domains: vec![sonium_xem::Domain {
                id: "acoustics".into(),
                label: "Acoustics".into(),
                kind: sonium_xem::DomainKind::Acoustics,
            }],
            ..Default::default()
        };
        assert_eq!(
            FemConvention::from_problem(&problem).unwrap(),
            ACOUSTIC_CONVENTION
        );

        problem.domains[0].kind = sonium_xem::DomainKind::Structural;
        assert!(matches!(
            FemConvention::from_problem(&problem),
            Err(FemProblemError::MissingAcousticsDomain)
        ));
    }
}
