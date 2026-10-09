//! Machine-readable maturity inventory for FEM physics modules.
//!
//! A module's existence is not evidence that its weak form is production
//! ready.  This inventory is deliberately explicit and conservative; each
//! record names the evidence required before a maturity level is promoted.

use std::collections::HashSet;

/// Evidence maturity assigned to a physics family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsMaturity {
    /// API and form prototype; numerical claims are not established.
    Prototype,
    /// Weak form has focused unit/form tests, but no complete convergence gate.
    VerifiedForm,
    /// Analytical or MMS convergence order is checked for the listed regime.
    ConvergenceVerified,
    /// Independent external or published reference comparison is passing.
    ExternallyValidated,
    /// Product-level evidence, failure semantics, and regression gates are complete.
    Production,
}

/// One row of the FEM validation matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicsValidationRecord {
    /// Stable physics family identifier.
    pub module: &'static str,
    /// Current evidence-backed maturity.
    pub maturity: PhysicsMaturity,
    /// Regimes covered by the current evidence.
    pub regimes: &'static str,
    /// Spatial dimensions represented by the evidence.
    pub dimensions: &'static str,
    /// Element families and orders represented by the evidence.
    pub element_orders: &'static str,
    /// Independent analytical, manufactured, or published reference.
    pub reference: &'static str,
    /// Expected convergence observable and order.
    pub expected_convergence: &'static str,
    /// Required next evidence before promotion.
    pub next_gate: &'static str,
    /// Whether a convergence test is currently registered.
    pub has_convergence_gate: bool,
    /// QA manifest case that provides the registered convergence/reference
    /// evidence. Mature records must name a case so promotion cannot rely on
    /// an untracked claim in this inventory.
    pub evidence_case: Option<&'static str>,
}

/// Failure while validating the machine-readable physics inventory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationMatrixError {
    /// Two records use the same stable module identifier.
    DuplicateModule(&'static str),
    /// A mature record is missing its required convergence evidence.
    MissingConvergenceGate(&'static str),
    /// An externally validated or production record has no independent reference.
    MissingIndependentReference(&'static str),
    /// A mature record does not point at a manifest-backed QA case.
    MissingEvidenceCase(&'static str),
    /// A required descriptive field is empty.
    MissingMetadata {
        module: &'static str,
        field: &'static str,
    },
}

impl std::fmt::Display for ValidationMatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateModule(module) => write!(f, "duplicate FEM validation module {module}"),
            Self::MissingConvergenceGate(module) => {
                write!(f, "mature FEM module {module} has no convergence gate")
            }
            Self::MissingIndependentReference(module) => {
                write!(
                    f,
                    "validated FEM module {module} has no independent reference"
                )
            }
            Self::MissingEvidenceCase(module) => {
                write!(f, "mature FEM module {module} has no QA evidence case")
            }
            Self::MissingMetadata { module, field } => {
                write!(
                    f,
                    "FEM validation module {module} has empty {field} metadata"
                )
            }
        }
    }
}

impl std::error::Error for ValidationMatrixError {}

/// Validate the promotion invariants of a physics validation matrix.
pub fn validate_physics_validation_matrix(
    matrix: &[PhysicsValidationRecord],
) -> Result<(), ValidationMatrixError> {
    let mut modules = HashSet::with_capacity(matrix.len());
    for record in matrix {
        if !modules.insert(record.module) {
            return Err(ValidationMatrixError::DuplicateModule(record.module));
        }
        for (field, value) in [
            ("module", record.module),
            ("regimes", record.regimes),
            ("dimensions", record.dimensions),
            ("element_orders", record.element_orders),
            ("reference", record.reference),
            ("expected_convergence", record.expected_convergence),
            ("next_gate", record.next_gate),
        ] {
            if value.trim().is_empty() {
                return Err(ValidationMatrixError::MissingMetadata {
                    module: record.module,
                    field,
                });
            }
        }
        if matches!(
            record.maturity,
            PhysicsMaturity::ConvergenceVerified
                | PhysicsMaturity::ExternallyValidated
                | PhysicsMaturity::Production
        ) && !record.has_convergence_gate
        {
            return Err(ValidationMatrixError::MissingConvergenceGate(record.module));
        }
        if matches!(
            record.maturity,
            PhysicsMaturity::ExternallyValidated | PhysicsMaturity::Production
        ) && (record.reference.is_empty() || record.reference == "none registered")
        {
            return Err(ValidationMatrixError::MissingIndependentReference(
                record.module,
            ));
        }
        if matches!(
            record.maturity,
            PhysicsMaturity::ConvergenceVerified
                | PhysicsMaturity::ExternallyValidated
                | PhysicsMaturity::Production
        ) && record
            .evidence_case
            .is_none_or(|case| case.trim().is_empty())
        {
            return Err(ValidationMatrixError::MissingEvidenceCase(record.module));
        }
    }
    Ok(())
}

/// A finite-difference report for a nonlinear residual/tangent pair.
#[derive(Debug, Clone, PartialEq)]
pub struct TangentCheckReport {
    /// Step sizes used by the check.
    pub steps: Vec<f64>,
    /// Norm of `R(x + h d) - R(x) - h J(x)d` for each step.
    pub remainder_norms: Vec<f64>,
    /// Observed slope between the first and last remainder.
    pub observed_order: f64,
}

/// Numerical residuals for the partition-of-unity/exact-sequence identity of
/// a scalar nodal basis at one quadrature point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasisIdentityReport {
    /// Absolute error in `Σᵢ Nᵢ = 1`.
    pub value_sum_error: f64,
    /// Euclidean norm of `Σᵢ ∇Nᵢ`.
    pub gradient_sum_error: f64,
}

/// Failure while checking a basis identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisIdentityError {
    /// No basis functions were supplied.
    Empty,
    /// The number of gradients differs from the number of values.
    DimensionMismatch,
    /// A value or gradient contains NaN or infinity.
    NonFinite,
    /// The supplied tolerance is invalid.
    InvalidTolerance,
    /// The identity residual exceeds the requested tolerance.
    Violated {
        value_sum_error: bool,
        gradient_sum_error: bool,
    },
}

impl std::fmt::Display for BasisIdentityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => f.write_str("basis identity requires at least one function"),
            Self::DimensionMismatch => f.write_str("basis values and gradients differ in length"),
            Self::NonFinite => f.write_str("basis values or gradients are non-finite"),
            Self::InvalidTolerance => {
                f.write_str("basis identity tolerance must be finite and non-negative")
            }
            Self::Violated {
                value_sum_error,
                gradient_sum_error,
            } => write!(
                f,
                "basis identity violated (value_error={value_sum_error}, gradient_error={gradient_sum_error})"
            ),
        }
    }
}

impl std::error::Error for BasisIdentityError {}

/// Check partition of unity and the zero-sum-gradient identity for a nodal
/// basis.  This is deliberately independent of element type and can be used
/// by H1, mapped scalar, and patch tests before assembly.
pub fn check_basis_partition_of_unity(
    values: &[f64],
    gradients: &[[f64; 3]],
    tolerance: f64,
) -> Result<BasisIdentityReport, BasisIdentityError> {
    if values.is_empty() {
        return Err(BasisIdentityError::Empty);
    }
    if values.len() != gradients.len() {
        return Err(BasisIdentityError::DimensionMismatch);
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(BasisIdentityError::InvalidTolerance);
    }
    if values.iter().any(|value| !value.is_finite())
        || gradients.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(BasisIdentityError::NonFinite);
    }
    let value_sum_error = (values.iter().sum::<f64>() - 1.0).abs();
    let gradient_sum = gradients.iter().fold([0.0; 3], |mut sum, gradient| {
        for component in 0..3 {
            sum[component] += gradient[component];
        }
        sum
    });
    let gradient_sum_error = gradient_sum
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let report = BasisIdentityReport {
        value_sum_error,
        gradient_sum_error,
    };
    if value_sum_error > tolerance || gradient_sum_error > tolerance {
        return Err(BasisIdentityError::Violated {
            value_sum_error: value_sum_error > tolerance,
            gradient_sum_error: gradient_sum_error > tolerance,
        });
    }
    Ok(report)
}

/// Typed failure from a nonlinear tangent consistency check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TangentCheckError {
    /// A step, state, direction, or callback result was not finite.
    NonFinite,
    /// A callback returned a vector with an unexpected size.
    DimensionMismatch,
    /// Fewer than two positive step sizes were supplied.
    InsufficientSteps,
}

impl std::fmt::Display for TangentCheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite => write!(f, "tangent check encountered a non-finite value"),
            Self::DimensionMismatch => write!(f, "tangent check callback dimensions disagree"),
            Self::InsufficientSteps => write!(f, "tangent check needs at least two positive steps"),
        }
    }
}

impl std::error::Error for TangentCheckError {}

/// Check the first-order consistency of a nonlinear residual and tangent.
///
/// For state `x` and direction `d`, a consistent Jacobian satisfies
/// `R(x + h d) - R(x) - h J(x)d = O(h²)`. The returned slope is therefore an
/// evidence-backed gate for Newton tangents rather than a solver iteration
/// count or a finite-difference derivative of a single component.
pub fn check_tangent_consistency<Residual, Tangent>(
    residual: Residual,
    tangent: Tangent,
    state: &[f64],
    direction: &[f64],
    steps: &[f64],
) -> Result<TangentCheckReport, TangentCheckError>
where
    Residual: Fn(&[f64]) -> Vec<f64>,
    Tangent: Fn(&[f64], &[f64]) -> Vec<f64>,
{
    if steps.len() < 2 {
        return Err(TangentCheckError::InsufficientSteps);
    }
    if state.len() != direction.len()
        || state
            .iter()
            .chain(direction)
            .any(|value| !value.is_finite())
        || steps.iter().any(|&step| !step.is_finite() || step <= 0.0)
    {
        return Err(TangentCheckError::NonFinite);
    }
    let base = residual(state);
    let linearized = tangent(state, direction);
    if base.len() != linearized.len() {
        return Err(TangentCheckError::DimensionMismatch);
    }
    if base
        .iter()
        .chain(linearized.iter())
        .any(|value| !value.is_finite())
    {
        return Err(TangentCheckError::NonFinite);
    }

    let mut remainder_norms = Vec::with_capacity(steps.len());
    for &step in steps {
        let perturbed: Vec<f64> = state
            .iter()
            .zip(direction)
            .map(|(&value, &delta)| value + step * delta)
            .collect();
        let shifted = residual(&perturbed);
        if shifted.len() != base.len() || shifted.iter().any(|value| !value.is_finite()) {
            return Err(TangentCheckError::DimensionMismatch);
        }
        let norm = shifted
            .iter()
            .zip(base.iter().zip(linearized.iter()))
            .map(|(&value, (&base, &linear))| (value - base - step * linear).powi(2))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() {
            return Err(TangentCheckError::NonFinite);
        }
        remainder_norms.push(norm);
    }

    let first = remainder_norms[0].max(f64::MIN_POSITIVE);
    let last = remainder_norms[remainder_norms.len() - 1].max(f64::MIN_POSITIVE);
    let first_step = steps[0];
    let last_step = steps[steps.len() - 1];
    let observed_order = (first / last).ln() / (first_step / last_step).ln();
    if !observed_order.is_finite() {
        return Err(TangentCheckError::NonFinite);
    }
    Ok(TangentCheckReport {
        steps: steps.to_vec(),
        remainder_norms,
        observed_order,
    })
}

macro_rules! validation_record {
    (
        $module:literal,
        $maturity:expr,
        $regimes:literal,
        $dimensions:literal,
        $element_orders:literal,
        $reference:literal,
        $expected_convergence:literal,
        $next_gate:literal,
        $has_convergence_gate:expr,
        $evidence_case:expr $(,)?
    ) => {
        PhysicsValidationRecord {
            module: $module,
            maturity: $maturity,
            regimes: $regimes,
            dimensions: $dimensions,
            element_orders: $element_orders,
            reference: $reference,
            expected_convergence: $expected_convergence,
            next_gate: $next_gate,
            has_convergence_gate: $has_convergence_gate,
            evidence_case: $evidence_case,
        }
    };
    (
        $module:literal,
        $maturity:expr,
        $regimes:literal,
        $dimensions:literal,
        $element_orders:literal,
        $reference:literal,
        $expected_convergence:literal,
        $next_gate:literal,
        $has_convergence_gate:expr $(,)?
    ) => {
        validation_record!(
            $module,
            $maturity,
            $regimes,
            $dimensions,
            $element_orders,
            $reference,
            $expected_convergence,
            $next_gate,
            $has_convergence_gate,
            None,
        )
    };
}

/// Conservative inventory used by QA and documentation tooling.
pub const PHYSICS_VALIDATION_MATRIX: &[PhysicsValidationRecord] = &[
    validation_record!(
        "acoustics.helmholtz",
        PhysicsMaturity::ConvergenceVerified,
        "frequency-domain scalar H1",
        "1D/2D/3D",
        "H1 P1/P2",
        "analytic square Poisson and nonzero-k Helmholtz MMS fixtures",
        "P1 nodal slope recorded per refinement at k=0 and k=2.3",
        "3D MMS convergence and independent external reference",
        true,
        Some("fem.helmholtz-mms-convergence.v1")
    ),
    validation_record!(
        "aeroacoustics",
        PhysicsMaturity::Prototype,
        "linearized flow/acoustic forms",
        "2D/3D",
        "H1 mixed P1",
        "none registered",
        "MMS residual and energy slope",
        "independent duct and published aeroacoustic reference",
        false
    ),
    validation_record!(
        "axisymmetric",
        PhysicsMaturity::Prototype,
        "weighted axisymmetric forms",
        "axisymmetric 2D",
        "H1 P1/P2",
        "none registered",
        "axisymmetric MMS slope",
        "axisymmetric patch and coordinate-singularity tests",
        false
    ),
    validation_record!(
        "beam",
        PhysicsMaturity::Prototype,
        "Euler-Bernoulli beam and design forms",
        "1D",
        "beam P1/P2",
        "cantilever analytical fixture",
        "tip displacement and energy slope",
        "patch, locking, and mesh-convergence gates",
        false
    ),
    validation_record!(
        "biharmonic",
        PhysicsMaturity::Prototype,
        "fourth-order plate forms",
        "2D",
        "C1-compatible/auxiliary H1",
        "none registered",
        "manufactured displacement slope",
        "clamped-plate patch and independent reference",
        false
    ),
    validation_record!(
        "composites",
        PhysicsMaturity::Prototype,
        "layered composite constitutive forms",
        "2D/3D",
        "H1 P1",
        "none registered",
        "laminate patch and energy slope",
        "material-interface and published laminate checks",
        false
    ),
    validation_record!(
        "compressible_flow",
        PhysicsMaturity::Prototype,
        "compressible flow forms",
        "2D/3D",
        "mixed P1",
        "none registered",
        "MMS conservation slope",
        "shock/low-Mach regime gates and tangent checks",
        false
    ),
    validation_record!(
        "contact",
        PhysicsMaturity::Prototype,
        "gap and augmented-Lagrangian contact",
        "2D/3D",
        "H1 P1",
        "none registered",
        "contact force and complementarity residual",
        "patch, active-set, and energy checks",
        false
    ),
    validation_record!(
        "eddy_current",
        PhysicsMaturity::Prototype,
        "frequency-domain eddy-current forms",
        "2D/3D",
        "H(curl) edge elements",
        "none registered",
        "skin-depth and energy slope",
        "orientation, gauge, and published benchmark",
        false
    ),
    validation_record!(
        "eigenmodes",
        PhysicsMaturity::Prototype,
        "generalized eigenvalue forms",
        "1D/2D/3D",
        "H1 P1/P2",
        "none registered",
        "eigenvalue refinement slope",
        "orthogonality and cavity-mode reference",
        false
    ),
    validation_record!(
        "elasticity.linear",
        PhysicsMaturity::VerifiedForm,
        "small-strain isotropic solids",
        "2D/3D",
        "H1 vector P1/P2",
        "cantilever and internal patch fixtures",
        "displacement/energy slope",
        "locking and NAFEMS convergence gates",
        false
    ),
    validation_record!(
        "electroacoustics",
        PhysicsMaturity::Prototype,
        "frequency/time-domain transducer coupling",
        "2D/3D",
        "mixed H1",
        "none registered",
        "power and reciprocity residual",
        "block Jacobian and coupled power-balance checks",
        false
    ),
    validation_record!(
        "electrochemistry",
        PhysicsMaturity::Prototype,
        "electrochemical transport forms",
        "2D/3D",
        "H1 P1",
        "none registered",
        "charge conservation slope",
        "manufactured concentration and current reference",
        false
    ),
    validation_record!(
        "electrostatic",
        PhysicsMaturity::Prototype,
        "electrostatic potential forms",
        "2D/3D",
        "H1 P1/P2",
        "none registered",
        "field/energy MMS slope",
        "coaxial/parallel-plate and interface tests",
        false
    ),
    validation_record!(
        "fracture",
        PhysicsMaturity::Prototype,
        "fracture and damage forms",
        "2D/3D",
        "H1 mixed",
        "none registered",
        "energy release convergence",
        "crack-tip and independent fracture reference",
        false
    ),
    validation_record!(
        "lumped_circuit",
        PhysicsMaturity::Prototype,
        "lumped acoustic/electrical networks",
        "0D/1D",
        "network elements",
        "none registered",
        "transfer-function residual",
        "impedance, passivity, and frequency-response gates",
        false
    ),
    validation_record!(
        "magnetostatics",
        PhysicsMaturity::Prototype,
        "magnetostatic and nonlinear magnetic forms",
        "2D/3D",
        "H(curl) edge elements",
        "none registered",
        "flux and energy convergence",
        "gauge/orientation and published magnetic benchmark",
        false
    ),
    validation_record!(
        "maxwell.hcurl",
        PhysicsMaturity::Prototype,
        "edge-element curl-curl forms",
        "2D/3D",
        "H(curl) Nedelec P1/P2",
        "none registered",
        "eigenfrequency/refinement slope",
        "exact-sequence, orientation, and cavity eigenmode tests",
        false
    ),
    validation_record!(
        "mindlin_plate",
        PhysicsMaturity::Prototype,
        "Mindlin-Reissner plate forms",
        "2D",
        "mixed H1 P1",
        "none registered",
        "deflection and shear-locking slope",
        "patch, locking, and published plate reference",
        false
    ),
    validation_record!(
        "nonlinear_solid",
        PhysicsMaturity::Prototype,
        "nonlinear solid constitutive forms",
        "2D/3D",
        "H1 vector P1",
        "none registered",
        "tangent finite-difference residual",
        "consistent tangent and load-path reference",
        false
    ),
    validation_record!(
        "piezoelectric",
        PhysicsMaturity::Prototype,
        "piezoelectric coupled forms",
        "2D/3D",
        "mixed H1",
        "none registered",
        "electromechanical energy slope",
        "reciprocity, units, and published transducer reference",
        false
    ),
    validation_record!(
        "plasticity",
        PhysicsMaturity::Prototype,
        "return-mapping plasticity",
        "2D/3D",
        "H1 vector P1",
        "internal tangent fixture",
        "stress/tangent finite difference",
        "yield-surface, path, and consistent-Jacobian gates",
        false
    ),
    validation_record!(
        "porous_flow",
        PhysicsMaturity::Prototype,
        "porous-flow and consolidation forms",
        "2D/3D",
        "mixed H1",
        "Terzaghi fixture",
        "mass conservation and time-step slope",
        "Terzaghi/biot independent reference",
        false
    ),
    validation_record!(
        "semiconductor",
        PhysicsMaturity::Prototype,
        "semiconductor drift-diffusion forms",
        "1D/2D/3D",
        "mixed H1",
        "none registered",
        "charge/current conservation slope",
        "manufactured carrier and nonlinear tangent checks",
        false
    ),
    validation_record!(
        "shallow_water",
        PhysicsMaturity::Prototype,
        "shallow-water forms",
        "2D",
        "H1 P1",
        "none registered",
        "mass/momentum conservation slope",
        "lake-at-rest and dam-break references",
        false
    ),
    validation_record!(
        "shell",
        PhysicsMaturity::Prototype,
        "shell and elastoplastic shell forms",
        "2D surface/3D",
        "mixed shell elements",
        "none registered",
        "patch and energy slope",
        "membrane/bending patch and NAFEMS shell reference",
        false
    ),
    validation_record!(
        "stokes",
        PhysicsMaturity::Prototype,
        "steady/transient incompressible flow",
        "2D/3D",
        "mixed velocity-pressure",
        "none registered",
        "divergence and Kovasznay slope",
        "inf-sup, stabilization, and CFD reference",
        false
    ),
    validation_record!(
        "thermal",
        PhysicsMaturity::Prototype,
        "steady and transient conduction",
        "1D/2D/3D",
        "H1 P1/P2",
        "none registered",
        "MMS temperature/flux slope",
        "manufactured-solution conservation and convergence",
        false
    ),
    validation_record!(
        "thermoelasticity",
        PhysicsMaturity::Prototype,
        "thermoelastic coupled forms",
        "2D/3D",
        "mixed H1",
        "none registered",
        "energy and displacement slope",
        "thermoelastic patch and independent reference",
        false
    ),
    validation_record!(
        "transport",
        PhysicsMaturity::Prototype,
        "advection-diffusion transport",
        "1D/2D/3D",
        "H1/DG P1",
        "none registered",
        "MMS scalar and conservation slope",
        "Peclet/stabilization and transport reference",
        false
    ),
    validation_record!(
        "turbulence",
        PhysicsMaturity::Prototype,
        "turbulence closure forms",
        "2D/3D",
        "mixed flow",
        "none registered",
        "residual and conservation slope",
        "manufactured closure and benchmark flow",
        false
    ),
];

/// Return the complete static validation matrix.
pub const fn physics_validation_matrix() -> &'static [PhysicsValidationRecord] {
    PHYSICS_VALIDATION_MATRIX
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_is_explicit_and_conservative() {
        assert!(!PHYSICS_VALIDATION_MATRIX.is_empty());
        validate_physics_validation_matrix(PHYSICS_VALIDATION_MATRIX)
            .expect("published validation matrix must satisfy promotion invariants");
        assert!(
            PHYSICS_VALIDATION_MATRIX
                .iter()
                .all(|record| !record.module.is_empty() && !record.next_gate.is_empty())
        );
        assert!(
            PHYSICS_VALIDATION_MATRIX
                .iter()
                .all(|record| !record.dimensions.is_empty()
                    && !record.element_orders.is_empty()
                    && !record.reference.is_empty()
                    && !record.expected_convergence.is_empty())
        );
        assert!(
            PHYSICS_VALIDATION_MATRIX
                .iter()
                .any(|record| record.maturity == PhysicsMaturity::Prototype)
        );
        assert!(PHYSICS_VALIDATION_MATRIX.len() >= 30);
    }

    #[test]
    fn matrix_gate_rejects_duplicate_and_unproven_mature_records() {
        let record = PhysicsValidationRecord {
            module: "duplicate",
            maturity: PhysicsMaturity::Prototype,
            regimes: "test",
            dimensions: "1D",
            element_orders: "P1",
            reference: "none registered",
            expected_convergence: "none",
            next_gate: "test",
            has_convergence_gate: false,
            evidence_case: None,
        };
        assert_eq!(
            validate_physics_validation_matrix(&[record, record]),
            Err(ValidationMatrixError::DuplicateModule("duplicate"))
        );

        let mature_without_gate = PhysicsValidationRecord {
            maturity: PhysicsMaturity::ConvergenceVerified,
            evidence_case: Some("fem.test-case.v1"),
            ..record
        };
        assert_eq!(
            validate_physics_validation_matrix(&[mature_without_gate]),
            Err(ValidationMatrixError::MissingConvergenceGate("duplicate"))
        );

        let mature_without_case = PhysicsValidationRecord {
            has_convergence_gate: true,
            evidence_case: None,
            ..mature_without_gate
        };
        assert_eq!(
            validate_physics_validation_matrix(&[mature_without_case]),
            Err(ValidationMatrixError::MissingEvidenceCase("duplicate"))
        );
    }

    #[test]
    fn matrix_gate_rejects_missing_descriptive_metadata() {
        let record = PhysicsValidationRecord {
            module: "metadata",
            maturity: PhysicsMaturity::Prototype,
            regimes: "",
            dimensions: "1D",
            element_orders: "P1",
            reference: "none registered",
            expected_convergence: "none",
            next_gate: "register evidence",
            has_convergence_gate: false,
            evidence_case: None,
        };
        assert_eq!(
            validate_physics_validation_matrix(&[record]),
            Err(ValidationMatrixError::MissingMetadata {
                module: "metadata",
                field: "regimes",
            })
        );
    }

    #[test]
    fn tangent_check_recovers_quadratic_remainder_order() {
        let report = check_tangent_consistency(
            |state| state.iter().map(|value| value * value).collect(),
            |state, direction| {
                state
                    .iter()
                    .zip(direction)
                    .map(|(&state, &direction)| 2.0 * state * direction)
                    .collect()
            },
            &[1.0, 2.0],
            &[0.5, -1.0],
            &[1e-1, 5e-2, 2.5e-2],
        )
        .expect("quadratic residual has a consistent analytic tangent");
        assert!(report.observed_order > 1.9, "{report:?}");
        assert!(report.remainder_norms[2] < report.remainder_norms[0] / 10.0);
    }

    #[test]
    fn tangent_check_rejects_bad_callback_dimensions() {
        let error = check_tangent_consistency(
            |_| vec![0.0, 1.0],
            |_, _| vec![0.0],
            &[0.0],
            &[1.0],
            &[1e-2, 1e-3],
        )
        .expect_err("a tangent with the wrong output dimension is invalid");
        assert_eq!(error, TangentCheckError::DimensionMismatch);
    }

    #[test]
    fn basis_identity_gate_checks_values_gradients_and_nonfinite_inputs() {
        let report = check_basis_partition_of_unity(
            &[0.25, 0.75],
            &[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            1e-14,
        )
        .expect("partition of unity and gradient sum should hold");
        assert!(report.value_sum_error < 1e-14);
        assert!(report.gradient_sum_error < 1e-14);
        assert_eq!(
            check_basis_partition_of_unity(&[], &[], 1e-8),
            Err(BasisIdentityError::Empty)
        );
        assert_eq!(
            check_basis_partition_of_unity(&[1.0], &[], 1e-8),
            Err(BasisIdentityError::DimensionMismatch)
        );
        assert_eq!(
            check_basis_partition_of_unity(&[f64::NAN], &[[0.0; 3]], 1e-8),
            Err(BasisIdentityError::NonFinite)
        );
        assert!(matches!(
            check_basis_partition_of_unity(&[0.5, 0.5], &[[0.0; 3], [0.0; 3]], 1e-8),
            Ok(_)
        ));
        assert!(matches!(
            check_basis_partition_of_unity(&[0.4, 0.4], &[[0.0; 3], [0.0; 3]], 1e-8),
            Err(BasisIdentityError::Violated {
                value_sum_error: true,
                ..
            })
        ));
    }
}
