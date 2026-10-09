//! Canonical acoustic BEM sign and orientation conventions.
//!
//! All BEM/FEM coupling code should reference this registry rather than
//! repeating signs locally. The convention is the usual `e^{-iωt}` time
//! dependence, an outgoing `e^{+ikr}` Green function, and normals oriented
//! out of the physical exterior domain.

/// Harmonic convention used by the acoustic boundary integral operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BemConvention {
    /// Sign in the time factor `e^{time_sign i ω t}`.
    pub time_sign: i8,
    /// Sign in the outgoing spatial factor `e^{spatial_sign i k r}`.
    pub spatial_sign: i8,
    /// Whether surface normals point out of the physical domain.
    pub outward_normals: bool,
    /// Burton–Miller coupling uses `β = +i η` for the exterior problem.
    pub exterior_burton_miller_positive_imaginary: bool,
}

/// Version of the public BEM harmonic/boundary-trace contract.
pub const BEM_CONVENTION_CONTRACT_VERSION: u32 = 1;

/// Interior/exterior domain orientation used by the BEM Green-function
/// formulation. The value is the canonical `tau` stored in `PhysicsParams`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BemDomain {
    /// Exterior scattering/radiation problem.
    Exterior,
    /// Interior cavity problem.
    Interior,
}

impl BemDomain {
    /// Return the canonical interior/exterior orientation marker.
    pub const fn tau(self) -> f64 {
        match self {
            Self::Exterior => 1.0,
            Self::Interior => -1.0,
        }
    }

    /// Convert the legacy boolean constructor argument to the typed domain.
    pub const fn from_internal(is_internal: bool) -> Self {
        if is_internal {
            Self::Interior
        } else {
            Self::Exterior
        }
    }
}

/// Meaning of the pressure unknown supplied to a boundary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureTrace {
    /// Total pressure, including the incident field.
    Total,
    /// Scattered pressure only.
    Scattered,
}

/// Direction used by a normal-velocity trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalVelocityDirection {
    /// Velocity dotted with the physical outward normal.
    Outward,
    /// Velocity dotted with the physical inward normal.
    Inward,
}

/// Typed meaning of the two acoustic boundary traces exchanged with BEM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundaryTraceConvention {
    /// Pressure quantity represented by the pressure trace.
    pub pressure: PressureTrace,
    /// Normal direction used by the velocity trace.
    pub normal_velocity: NormalVelocityDirection,
}

impl BoundaryTraceConvention {
    /// The default acoustic BEM trace pair used by current exterior solvers.
    pub const fn total_pressure_outward_velocity() -> Self {
        Self {
            pressure: PressureTrace::Total,
            normal_velocity: NormalVelocityDirection::Outward,
        }
    }
}

/// Burton–Miller imaginary coupling sign selected by the convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurtonMillerSign {
    /// `+i η` coupling sign.
    PositiveImaginary,
    /// `-i η` coupling sign.
    NegativeImaginary,
}

impl BurtonMillerSign {
    /// Return the signed imaginary unit coefficient.
    pub const fn imaginary_sign(self) -> i8 {
        match self {
            Self::PositiveImaginary => 1,
            Self::NegativeImaginary => -1,
        }
    }
}

/// Fully typed BEM contract used at solver and multiphysics boundaries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BemConventionContract {
    /// Frozen vocabulary version.
    pub version: u32,
    /// Sign in the harmonic time factor.
    pub time_sign: i8,
    /// Sign in the spatial Green-function factor.
    pub spatial_sign: i8,
    /// Interior or exterior domain.
    pub domain: BemDomain,
    /// Canonical interior/exterior marker.
    pub tau: f64,
    /// Typed boundary trace meanings.
    pub traces: BoundaryTraceConvention,
    /// Burton–Miller imaginary coupling sign.
    pub burton_miller_sign: BurtonMillerSign,
}

/// Failure while adapting an exchange-file convention to the BEM operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BemConventionError {
    /// BEM mesh normals must point out of the physical domain.
    InwardNormals,
    /// BEM operators require the global right-handed world frame.
    NonWorldCoordinateFrame,
    /// Harmonic Green-function signs must be exactly ±1.
    InvalidSign {
        /// Name of the invalid sign field.
        field: &'static str,
        /// Supplied sign value.
        value: i8,
    },
}

/// Failure while adapting a complete XEM multiphysics problem to BEM.
#[derive(Debug, thiserror::Error)]
pub enum BemProblemError {
    /// The problem failed schema/reference validation.
    #[error("invalid XEM multiphysics problem: {0}")]
    InvalidProblem(#[from] sonium_xem::IrValidationError),
    /// The XEM coupling graph cannot be admitted by a partitioned BEM solve.
    #[error("invalid XEM coupling graph: {0}")]
    CouplingGraph(#[from] sonium_xem::CouplingGraphError),
    /// The problem does not contain an acoustics domain.
    #[error("XEM problem has no acoustics domain")]
    MissingAcousticsDomain,
    /// Its shared convention cannot be consumed by BEM.
    #[error("incompatible BEM convention: {0}")]
    Convention(#[from] BemConventionError),
    /// The selected XEM pressure field is not consumable by BEM.
    #[error("incompatible XEM pressure field `{field_id}`: {reason}")]
    Field {
        /// Field identifier supplied by the exchange document.
        field_id: String,
        /// Why the field cannot be bound to a BEM pressure result.
        reason: String,
    },
    /// Backend mesh or physical parameters failed validation.
    #[error("invalid BEM backend input: {0}")]
    Backend(#[from] crate::core::bem_solver::BemError),
}

impl std::fmt::Display for BemConventionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InwardNormals => write!(f, "BEM requires outward physical-domain normals"),
            Self::NonWorldCoordinateFrame => {
                write!(f, "BEM requires the global right-handed world frame")
            }
            Self::InvalidSign { field, value } => {
                write!(
                    f,
                    "BEM convention {field} sign must be +1 or -1 (got {value})"
                )
            }
        }
    }
}

impl std::error::Error for BemConventionError {}

impl BemConvention {
    /// Version of the convention vocabulary emitted by this crate.
    pub const fn contract_version() -> u32 {
        BEM_CONVENTION_CONTRACT_VERSION
    }

    /// Convert the legacy field-based convention into a checked, typed
    /// boundary contract. BEM requires outward physical-domain normals;
    /// callers that use inward traces must adapt them before this boundary.
    pub fn typed_contract(
        &self,
        domain: BemDomain,
        pressure: PressureTrace,
    ) -> Result<BemConventionContract, BemConventionError> {
        if !matches!(self.time_sign, -1 | 1) {
            return Err(BemConventionError::InvalidSign {
                field: "time",
                value: self.time_sign,
            });
        }
        if !matches!(self.spatial_sign, -1 | 1) {
            return Err(BemConventionError::InvalidSign {
                field: "spatial",
                value: self.spatial_sign,
            });
        }
        if !self.outward_normals {
            return Err(BemConventionError::InwardNormals);
        }
        let burton_miller_sign = if self.exterior_burton_miller_positive_imaginary {
            BurtonMillerSign::PositiveImaginary
        } else {
            BurtonMillerSign::NegativeImaginary
        };
        Ok(BemConventionContract {
            version: BEM_CONVENTION_CONTRACT_VERSION,
            time_sign: self.time_sign,
            spatial_sign: self.spatial_sign,
            domain,
            tau: domain.tau(),
            traces: BoundaryTraceConvention {
                pressure,
                normal_velocity: NormalVelocityDirection::Outward,
            },
            burton_miller_sign,
        })
    }

    /// Adapt the versioned XEM convention registry to the BEM sign contract.
    ///
    /// Changing the time factor also changes the outgoing spatial Green
    /// function sign. Rejecting inward/local-frame data here prevents a
    /// coupling caller from silently solving a reflected or non-radiating
    /// problem with otherwise valid numerical inputs.
    pub fn from_exchange(
        registry: &sonium_xem::ConventionRegistry,
    ) -> Result<Self, BemConventionError> {
        if !registry.outward_normals {
            return Err(BemConventionError::InwardNormals);
        }
        if !matches!(
            registry.coordinate_frame,
            sonium_xem::CoordinateFrame::WorldRightHanded
        ) {
            return Err(BemConventionError::NonWorldCoordinateFrame);
        }
        let (time_sign, spatial_sign) = registry.harmonic_time.signs();
        Ok(Self {
            time_sign,
            spatial_sign,
            outward_normals: registry.outward_normals,
            exterior_burton_miller_positive_imaginary: true,
        })
    }

    /// Validate and adapt a complete XEM problem at the BEM boundary.
    ///
    /// This keeps schema/reference failures and convention mismatches from
    /// being discovered only after a solver has started assembling operators.
    pub fn from_problem(
        problem: &sonium_xem::MultiphysicsProblem,
    ) -> Result<Self, BemProblemError> {
        problem.validate()?;
        if !problem
            .domains
            .iter()
            .any(|domain| matches!(domain.kind, sonium_xem::DomainKind::Acoustics))
        {
            return Err(BemProblemError::MissingAcousticsDomain);
        }
        Ok(Self::from_exchange(&problem.conventions)?)
    }

    /// Validate the field contract consumed by the current acoustic BEM
    /// result adapters.  The shared XEM helper keeps this check identical for
    /// BEM and FEM while this error preserves the backend boundary context.
    pub fn pressure_field<'a>(
        problem: &'a sonium_xem::MultiphysicsProblem,
        field_id: &str,
    ) -> Result<&'a sonium_xem::FieldSpec, BemProblemError> {
        problem.validate()?;
        problem
            .acoustic_complex_pressure_field(field_id)
            .map_err(|error| match error {
                sonium_xem::IrValidationError::MissingReference { .. } => BemProblemError::Field {
                    field_id: field_id.to_owned(),
                    reason: "field is not declared".into(),
                },
                other => BemProblemError::Field {
                    field_id: field_id.to_owned(),
                    reason: other.to_string(),
                },
            })
    }
}

/// The single convention shared by the current acoustic BEM operators.
pub const ACOUSTIC_CONVENTION: BemConvention = BemConvention {
    time_sign: -1,
    spatial_sign: 1,
    outward_normals: true,
    exterior_burton_miller_positive_imaginary: true,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_exchange_convention_matches_bem() {
        let convention = BemConvention::from_exchange(&sonium_xem::ConventionRegistry::default())
            .expect("the canonical exchange convention is BEM-compatible");
        assert_eq!(convention, ACOUSTIC_CONVENTION);
    }

    #[test]
    fn exchange_adaptation_tracks_time_sign_and_rejects_orientation() {
        let mut registry = sonium_xem::ConventionRegistry::default();
        registry.harmonic_time = sonium_xem::HarmonicConvention::ExpPlusIwt;
        let convention = BemConvention::from_exchange(&registry).expect("valid plus-i convention");
        assert_eq!((convention.time_sign, convention.spatial_sign), (1, -1));

        registry.outward_normals = false;
        assert_eq!(
            BemConvention::from_exchange(&registry),
            Err(BemConventionError::InwardNormals)
        );
    }

    #[test]
    fn complete_xem_problem_is_validated_at_bem_boundary() {
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
            BemConvention::from_problem(&problem).unwrap(),
            ACOUSTIC_CONVENTION
        );

        problem.domains[0].kind = sonium_xem::DomainKind::Structural;
        assert!(matches!(
            BemConvention::from_problem(&problem),
            Err(BemProblemError::MissingAcousticsDomain)
        ));
    }
}
