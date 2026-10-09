//! Backend-independent numerical contracts for BEM operators.
//!
//! These helpers intentionally operate on the public `LinearOperator` trait.
//! Dense, H-matrix, SLFMM, and MLFMM implementations can therefore share the
//! same reciprocity/adjoint and multi-RHS checks without duplicating test logic.

use num_complex::Complex64;
use sonium_solvers::{ComplexField, LinearOperator, dense::DenseMatrix};

/// Failure while evaluating an operator contract.
#[derive(Debug, Clone, PartialEq)]
pub enum OperatorContractError {
    /// The operator or a supplied vector has an incompatible dimension.
    DimensionMismatch {
        /// Human-readable contract label.
        label: &'static str,
        /// Expected number of entries.
        expected: usize,
        /// Actual number of entries.
        actual: usize,
    },
    /// No probe vector was supplied, so an accuracy claim would be vacuous.
    InsufficientProbes,
    /// A computed contract quantity was not finite.
    NonFinite {
        /// Human-readable contract label.
        label: &'static str,
    },
    /// An approximation exceeded the explicitly requested error budget.
    AccuracyExceeded {
        /// Human-readable contract label.
        label: &'static str,
        /// Measured relative action error.
        observed: f64,
        /// Maximum permitted relative action error.
        tolerance: f64,
    },
    /// A reciprocity/passivity contract was requested for a rectangular
    /// operator, which has no square quadratic form.
    NonSquare {
        /// Human-readable contract label.
        label: &'static str,
        /// Number of operator rows.
        rows: usize,
        /// Number of operator columns.
        cols: usize,
    },
    /// The Hermitian part of an operator produced negative power.
    PassivityViolation {
        /// Probe index that exposed the violation.
        probe: usize,
        /// Measured real quadratic form.
        power: f64,
        /// Relative tolerance used by the gate.
        tolerance: f64,
    },
}

impl std::fmt::Display for OperatorContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch {
                label,
                expected,
                actual,
            } => write!(
                f,
                "{label} dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::InsufficientProbes => {
                f.write_str("operator accuracy contract requires at least one probe vector")
            }
            Self::NonFinite { label } => write!(f, "{label} produced a non-finite value"),
            Self::AccuracyExceeded {
                label,
                observed,
                tolerance,
            } => write!(
                f,
                "{label} accuracy budget exceeded: {observed:.3e} > {tolerance:.3e}"
            ),
            Self::NonSquare { label, rows, cols } => {
                write!(f, "{label} requires a square operator, got {rows}x{cols}")
            }
            Self::PassivityViolation {
                probe,
                power,
                tolerance,
            } => write!(
                f,
                "passivity contract failed at probe {probe}: power {power:.3e} < -{tolerance:.3e}"
            ),
        }
    }
}

impl std::error::Error for OperatorContractError {}

/// Inner-product error for the Hermitian-adjoint contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdjointContractReport {
    /// `|<Ax,y> - <x,Aᴴy>| / max(|<Ax,y>|, |<x,Aᴴy>|, 1)`.
    pub relative_error: f64,
    /// First inner-product value, useful for diagnosing zero-energy cases.
    pub forward_inner_product: Complex64,
    /// Second inner-product value.
    pub adjoint_inner_product: Complex64,
}

/// Relative error report for the bilinear transpose/reciprocity identity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReciprocityContractReport {
    /// `|yᵀAx - xᵀAᵀy| / max(|yᵀAx|, |xᵀAᵀy|, ε)`.
    pub relative_error: f64,
    /// Forward bilinear product.
    pub forward_product: Complex64,
    /// Transposed bilinear product.
    pub transpose_product: Complex64,
}

/// Reciprocity report for a two-trace Müller transmission operator.
///
/// The pressure/velocity block scaling obeys `J A = (J A)ᵀ` with
/// `J = [[0, i/(sω) I], [I, 0]]`, where `s` is the harmonic sign. This is the
/// bilinear (non-conjugated) acoustic reciprocity identity, distinct from the
/// Hermitian energy contract used for a single operator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransmissionReciprocityReport {
    /// Maximum relative pairwise reciprocity error.
    pub relative_error: f64,
    /// Number of supplied probe vectors.
    pub probes: usize,
    /// Number of distinct probe pairs checked.
    pub pairs: usize,
    /// Product for the pair attaining `relative_error`.
    pub forward_product: Complex64,
    /// Reversed product for the pair attaining `relative_error`.
    pub reverse_product: Complex64,
}

/// Passivity report over a set of velocity/load probe vectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassivityContractReport {
    /// Smallest real quadratic form `Re(xᴴ A x)` observed.
    pub minimum_power: f64,
    /// Number of probes evaluated.
    pub probes: usize,
}

/// Power-flow report for two transmission traces.
///
/// Both trace velocities are expressed in the same interface-normal
/// direction (the convention used by [`TransmissionTbemSystem`]): from the
/// interior/transmitted medium toward the exterior/incident medium.  Equality
/// of the two integrated powers is therefore the local lossless
/// transmission-energy contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransmissionPowerBalanceReport {
    /// Time-averaged exterior-side power in the declared normal direction.
    pub exterior_power: f64,
    /// Time-averaged interior-side power in the declared normal direction.
    pub interior_power: f64,
    /// Signed exterior-minus-interior power residual.
    pub absolute_residual: f64,
    /// Residual divided by the larger observed power scale.
    pub relative_error: f64,
}

/// Error-budget report comparing an approximate operator with a reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OperatorAccuracyReport {
    /// Maximum relative action error over all supplied probe vectors.
    pub max_relative_error: f64,
    /// Number of probe vectors evaluated.
    pub probes: usize,
}

/// Compare compressed/FMM actions with an independent reference operator.
pub fn require_operator_accuracy<A, R>(
    approximate: &A,
    reference: &R,
    probes: &[Vec<Complex64>],
    tolerance: f64,
) -> Result<OperatorAccuracyReport, OperatorContractError>
where
    A: LinearOperator<Complex64>,
    R: LinearOperator<Complex64>,
{
    if probes.is_empty() {
        return Err(OperatorContractError::InsufficientProbes);
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(OperatorContractError::NonFinite {
            label: "operator accuracy tolerance",
        });
    }
    if approximate.num_rows() != reference.num_rows() {
        return Err(OperatorContractError::DimensionMismatch {
            label: "operator rows",
            expected: reference.num_rows(),
            actual: approximate.num_rows(),
        });
    }
    if approximate.num_cols() != reference.num_cols() {
        return Err(OperatorContractError::DimensionMismatch {
            label: "operator columns",
            expected: reference.num_cols(),
            actual: approximate.num_cols(),
        });
    }

    let mut max_relative_error: f64 = 0.0;
    for probe in probes {
        if probe.len() != reference.num_cols() {
            return Err(OperatorContractError::DimensionMismatch {
                label: "operator probe",
                expected: reference.num_cols(),
                actual: probe.len(),
            });
        }
        ensure_finite("operator accuracy probe", probe)?;
        let approximate_action = approximate.apply(probe);
        let reference_action = reference.apply(probe);
        if approximate_action.len() != approximate.num_rows() {
            return Err(OperatorContractError::DimensionMismatch {
                label: "approximate operator action",
                expected: approximate.num_rows(),
                actual: approximate_action.len(),
            });
        }
        if reference_action.len() != reference.num_rows() {
            return Err(OperatorContractError::DimensionMismatch {
                label: "reference operator action",
                expected: reference.num_rows(),
                actual: reference_action.len(),
            });
        }
        ensure_finite("approximate operator action", &approximate_action)?;
        ensure_finite("reference operator action", &reference_action)?;
        let numerator = approximate_action
            .iter()
            .zip(reference_action.iter())
            .map(|(actual, expected)| (*actual - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let denominator = reference_action
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        max_relative_error = max_relative_error.max(numerator / denominator);
    }
    if !max_relative_error.is_finite() {
        return Err(OperatorContractError::NonFinite {
            label: "operator accuracy contract",
        });
    }
    let report = OperatorAccuracyReport {
        max_relative_error,
        probes: probes.len(),
    };
    if report.max_relative_error > tolerance {
        return Err(OperatorContractError::AccuracyExceeded {
            label: "operator accuracy contract",
            observed: report.max_relative_error,
            tolerance,
        });
    }
    Ok(report)
}

/// Evaluate the Hermitian-adjoint/reciprocity contract for one vector pair.
pub fn check_adjoint_contract<O: LinearOperator<Complex64>>(
    operator: &O,
    x: &[Complex64],
    y: &[Complex64],
) -> Result<AdjointContractReport, OperatorContractError> {
    check_vector_dimensions(operator, x, y)?;
    ensure_finite("Hermitian-adjoint input", x)?;
    ensure_finite("Hermitian-adjoint probe", y)?;
    let ax = operator.apply(x);
    let ahy = operator.apply_hermitian(y);
    check_action_dimensions("forward operator action", operator.num_rows(), ax.len())?;
    check_action_dimensions("Hermitian operator action", operator.num_cols(), ahy.len())?;
    ensure_finite("forward operator action", &ax)?;
    ensure_finite("Hermitian operator action", &ahy)?;
    let forward = inner_product(&ax, y);
    let adjoint = inner_product(x, &ahy);
    // Preserve a genuinely relative contract for small physical pressures or
    // velocities. A unit floor silently turns tiny actions into an absolute
    // error budget and can certify a badly scaled operator.
    let denominator = forward.norm().max(adjoint.norm()).max(f64::MIN_POSITIVE);
    let relative_error = (forward - adjoint).norm() / denominator;
    if !relative_error.is_finite() {
        return Err(OperatorContractError::NonFinite {
            label: "Hermitian-adjoint contract",
        });
    }
    Ok(AdjointContractReport {
        relative_error,
        forward_inner_product: forward,
        adjoint_inner_product: adjoint,
    })
}

/// Evaluate the bilinear transpose identity used by reciprocal acoustic
/// operators.  This intentionally differs from [`check_adjoint_contract`]:
/// reciprocity uses `Aᵀ` without conjugating the probe vectors, while energy
/// and adjoint contracts use `Aᴴ`.
pub fn check_reciprocity_contract<O: LinearOperator<Complex64>>(
    operator: &O,
    x: &[Complex64],
    y: &[Complex64],
) -> Result<ReciprocityContractReport, OperatorContractError> {
    if operator.num_rows() != operator.num_cols() {
        return Err(OperatorContractError::NonSquare {
            label: "reciprocity contract",
            rows: operator.num_rows(),
            cols: operator.num_cols(),
        });
    }
    check_vector_dimensions(operator, x, y)?;
    ensure_finite("reciprocity input", x)?;
    ensure_finite("reciprocity probe", y)?;
    let ax = operator.apply(x);
    let aty = operator.apply_transpose(y);
    check_action_dimensions("forward operator action", operator.num_rows(), ax.len())?;
    check_action_dimensions("transpose operator action", operator.num_cols(), aty.len())?;
    ensure_finite("forward operator action", &ax)?;
    ensure_finite("transpose operator action", &aty)?;
    let forward = bilinear_product(y, &ax);
    let transposed = bilinear_product(x, &aty);
    let denominator = forward.norm().max(transposed.norm()).max(f64::MIN_POSITIVE);
    let relative_error = (forward - transposed).norm() / denominator;
    if !relative_error.is_finite() {
        return Err(OperatorContractError::NonFinite {
            label: "reciprocity contract",
        });
    }
    Ok(ReciprocityContractReport {
        relative_error,
        forward_product: forward,
        transpose_product: transposed,
    })
}

/// Check the weighted bilinear reciprocity identity of a two-trace Müller
/// transmission matrix over every distinct pair of supplied probes.
///
/// `matrix` must be ordered as pressure DOFs followed by normal-velocity
/// DOFs. The off-diagonal block scaling in the production operator gives the
/// weight `i/(sω)` in the pressure row of `J`; callers supply the physical
/// angular frequency and harmonic sign explicitly so the check cannot infer a
/// convention from matrix data.
pub fn check_transmission_reciprocity_contract(
    matrix: &DenseMatrix<Complex64>,
    interface_dofs: usize,
    omega: f64,
    harmonic_factor: f64,
    probes: &[Vec<Complex64>],
    tolerance: f64,
) -> Result<TransmissionReciprocityReport, OperatorContractError> {
    if interface_dofs == 0 || matrix.nrows() != matrix.ncols() {
        return Err(OperatorContractError::NonSquare {
            label: "transmission reciprocity contract",
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        });
    }
    let expected =
        interface_dofs
            .checked_mul(2)
            .ok_or(OperatorContractError::DimensionMismatch {
                label: "transmission reciprocity matrix",
                expected: usize::MAX,
                actual: matrix.nrows(),
            })?;
    if matrix.nrows() != expected {
        return Err(OperatorContractError::DimensionMismatch {
            label: "transmission reciprocity matrix",
            expected,
            actual: matrix.nrows(),
        });
    }
    if probes.len() < 2 {
        return Err(OperatorContractError::InsufficientProbes);
    }
    if !omega.is_finite()
        || omega.abs() <= f64::MIN_POSITIVE
        || !harmonic_factor.is_finite()
        || (harmonic_factor.abs() - 1.0).abs() > f64::EPSILON
        || !tolerance.is_finite()
        || tolerance < 0.0
    {
        return Err(OperatorContractError::NonFinite {
            label: "transmission reciprocity configuration",
        });
    }
    let signed_omega = harmonic_factor * omega;
    if !signed_omega.is_finite() || signed_omega.abs() <= f64::MIN_POSITIVE {
        return Err(OperatorContractError::NonFinite {
            label: "transmission reciprocity angular frequency",
        });
    }

    let mut actions = Vec::with_capacity(probes.len());
    for probe in probes {
        if probe.len() != expected {
            return Err(OperatorContractError::DimensionMismatch {
                label: "transmission reciprocity probe",
                expected,
                actual: probe.len(),
            });
        }
        ensure_finite("transmission reciprocity probe", probe)?;
        let action = dense_matrix_action(matrix, probe);
        ensure_finite("transmission reciprocity matrix action", &action)?;
        actions.push(weighted_transmission_action(
            &action,
            interface_dofs,
            signed_omega,
        ));
    }

    let mut report = TransmissionReciprocityReport {
        relative_error: 0.0,
        probes: probes.len(),
        pairs: 0,
        forward_product: Complex64::new(0.0, 0.0),
        reverse_product: Complex64::new(0.0, 0.0),
    };
    for left in 0..probes.len() {
        for right in (left + 1)..probes.len() {
            let forward = bilinear_product(&probes[left], &actions[right]);
            let reverse = bilinear_product(&probes[right], &actions[left]);
            if !forward.re.is_finite()
                || !forward.im.is_finite()
                || !reverse.re.is_finite()
                || !reverse.im.is_finite()
            {
                return Err(OperatorContractError::NonFinite {
                    label: "transmission reciprocity product",
                });
            }
            let denominator = forward.norm().max(reverse.norm()).max(f64::MIN_POSITIVE);
            let relative_error = (forward - reverse).norm() / denominator;
            if !relative_error.is_finite() {
                return Err(OperatorContractError::NonFinite {
                    label: "transmission reciprocity relative error",
                });
            }
            report.pairs += 1;
            if relative_error >= report.relative_error {
                report.relative_error = relative_error;
                report.forward_product = forward;
                report.reverse_product = reverse;
            }
        }
    }
    if report.relative_error > tolerance {
        return Err(OperatorContractError::AccuracyExceeded {
            label: "transmission reciprocity contract",
            observed: report.relative_error,
            tolerance,
        });
    }
    Ok(report)
}

/// Check weighted bilinear reciprocity for any matrix-free two-trace
/// transmission operator.
///
/// This is the operator counterpart to
/// [`check_transmission_reciprocity_contract`].  Keeping the contract on the
/// [`LinearOperator`] trait lets compressed transmission, SLFMM, and MLFMM
/// implementations be checked without materialising their dense matrix.
/// The operator must use the same pressure-then-normal-velocity layout and
/// harmonic convention as the dense Müller system.
pub fn check_transmission_reciprocity_operator<O: LinearOperator<Complex64>>(
    operator: &O,
    interface_dofs: usize,
    omega: f64,
    harmonic_factor: f64,
    probes: &[Vec<Complex64>],
    tolerance: f64,
) -> Result<TransmissionReciprocityReport, OperatorContractError> {
    let rows = operator.num_rows();
    let cols = operator.num_cols();
    if interface_dofs == 0 || rows != cols {
        return Err(OperatorContractError::NonSquare {
            label: "transmission reciprocity operator",
            rows,
            cols,
        });
    }
    let expected =
        interface_dofs
            .checked_mul(2)
            .ok_or(OperatorContractError::DimensionMismatch {
                label: "transmission reciprocity operator",
                expected: usize::MAX,
                actual: rows,
            })?;
    if rows != expected {
        return Err(OperatorContractError::DimensionMismatch {
            label: "transmission reciprocity operator",
            expected,
            actual: rows,
        });
    }
    if probes.len() < 2 {
        return Err(OperatorContractError::InsufficientProbes);
    }
    if !omega.is_finite()
        || omega.abs() <= f64::MIN_POSITIVE
        || !harmonic_factor.is_finite()
        || (harmonic_factor.abs() - 1.0).abs() > f64::EPSILON
        || !tolerance.is_finite()
        || tolerance < 0.0
    {
        return Err(OperatorContractError::NonFinite {
            label: "transmission reciprocity operator configuration",
        });
    }
    let signed_omega = harmonic_factor * omega;
    if !signed_omega.is_finite() || signed_omega.abs() <= f64::MIN_POSITIVE {
        return Err(OperatorContractError::NonFinite {
            label: "transmission reciprocity operator angular frequency",
        });
    }

    let mut weighted_actions = Vec::with_capacity(probes.len());
    for probe in probes {
        if probe.len() != cols {
            return Err(OperatorContractError::DimensionMismatch {
                label: "transmission reciprocity operator probe",
                expected: cols,
                actual: probe.len(),
            });
        }
        ensure_finite("transmission reciprocity operator probe", probe)?;
        let action = operator.apply(probe);
        check_action_dimensions(
            "transmission reciprocity operator action",
            rows,
            action.len(),
        )?;
        ensure_finite("transmission reciprocity operator action", &action)?;
        weighted_actions.push(weighted_transmission_action(
            &action,
            interface_dofs,
            signed_omega,
        ));
    }

    let mut report = TransmissionReciprocityReport {
        relative_error: 0.0,
        probes: probes.len(),
        pairs: 0,
        forward_product: Complex64::new(0.0, 0.0),
        reverse_product: Complex64::new(0.0, 0.0),
    };
    for left in 0..probes.len() {
        for right in (left + 1)..probes.len() {
            let forward = bilinear_product(&probes[left], &weighted_actions[right]);
            let reverse = bilinear_product(&probes[right], &weighted_actions[left]);
            if !forward.re.is_finite()
                || !forward.im.is_finite()
                || !reverse.re.is_finite()
                || !reverse.im.is_finite()
            {
                return Err(OperatorContractError::NonFinite {
                    label: "transmission reciprocity operator product",
                });
            }
            let denominator = forward.norm().max(reverse.norm()).max(f64::MIN_POSITIVE);
            let relative_error = (forward - reverse).norm() / denominator;
            if !relative_error.is_finite() {
                return Err(OperatorContractError::NonFinite {
                    label: "transmission reciprocity operator relative error",
                });
            }
            report.pairs += 1;
            if relative_error >= report.relative_error {
                report.relative_error = relative_error;
                report.forward_product = forward;
                report.reverse_product = reverse;
            }
        }
    }
    if report.relative_error > tolerance {
        return Err(OperatorContractError::AccuracyExceeded {
            label: "transmission reciprocity operator",
            observed: report.relative_error,
            tolerance,
        });
    }
    Ok(report)
}

/// Check the transmission power quadratic form on a matrix-free operator.
///
/// The canonical two-trace exchange matrix is applied before evaluating
/// `Re(xᴴ J A x)`.  This keeps the passivity check consistent with the
/// weighted reciprocity convention and lets compressed/FMM operators be
/// checked without a dense materialisation.  A negative value beyond the
/// supplied tolerance is reported as a typed passivity violation.
pub fn check_transmission_passivity_contract<O: LinearOperator<Complex64>>(
    operator: &O,
    interface_dofs: usize,
    omega: f64,
    harmonic_factor: f64,
    probes: &[Vec<Complex64>],
    tolerance: f64,
) -> Result<PassivityContractReport, OperatorContractError> {
    let rows = operator.num_rows();
    let cols = operator.num_cols();
    if interface_dofs == 0 || rows != cols {
        return Err(OperatorContractError::NonSquare {
            label: "transmission passivity operator",
            rows,
            cols,
        });
    }
    let expected =
        interface_dofs
            .checked_mul(2)
            .ok_or(OperatorContractError::DimensionMismatch {
                label: "transmission passivity operator",
                expected: usize::MAX,
                actual: rows,
            })?;
    if rows != expected {
        return Err(OperatorContractError::DimensionMismatch {
            label: "transmission passivity operator",
            expected,
            actual: rows,
        });
    }
    if probes.is_empty() {
        return Err(OperatorContractError::InsufficientProbes);
    }
    if !omega.is_finite()
        || omega.abs() <= f64::MIN_POSITIVE
        || !harmonic_factor.is_finite()
        || (harmonic_factor.abs() - 1.0).abs() > f64::EPSILON
        || !tolerance.is_finite()
        || tolerance < 0.0
    {
        return Err(OperatorContractError::NonFinite {
            label: "transmission passivity operator configuration",
        });
    }
    let signed_omega = harmonic_factor * omega;
    if !signed_omega.is_finite() || signed_omega.abs() <= f64::MIN_POSITIVE {
        return Err(OperatorContractError::NonFinite {
            label: "transmission passivity operator angular frequency",
        });
    }

    let mut minimum_power = f64::INFINITY;
    for (probe_index, probe) in probes.iter().enumerate() {
        if probe.len() != cols {
            return Err(OperatorContractError::DimensionMismatch {
                label: "transmission passivity operator probe",
                expected: cols,
                actual: probe.len(),
            });
        }
        ensure_finite("transmission passivity operator probe", probe)?;
        let action = operator.apply(probe);
        check_action_dimensions("transmission passivity operator action", rows, action.len())?;
        ensure_finite("transmission passivity operator action", &action)?;
        let weighted = weighted_transmission_action(&action, interface_dofs, signed_omega);
        let power = inner_product(probe, &weighted).re;
        let scale = probe
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .max(f64::MIN_POSITIVE);
        if !power.is_finite() || !scale.is_finite() {
            return Err(OperatorContractError::NonFinite {
                label: "transmission passivity operator power",
            });
        }
        minimum_power = minimum_power.min(power);
        if power < -tolerance * scale {
            return Err(OperatorContractError::PassivityViolation {
                probe: probe_index,
                power,
                tolerance: tolerance * scale,
            });
        }
    }
    if !minimum_power.is_finite() {
        return Err(OperatorContractError::NonFinite {
            label: "transmission passivity operator minimum power",
        });
    }
    Ok(PassivityContractReport {
        minimum_power,
        probes: probes.len(),
    })
}

/// Check lossless transmission energy balance from constant-panel traces.
///
/// Pressures and normal velocities on both sides must have the same panel
/// ordering and use the same normal direction.  The function computes
/// `0.5 Re(p v*)` per panel, integrates with the supplied areas, and fails
/// closed when the exterior/interior powers differ beyond `tolerance`.
pub fn check_transmission_power_balance(
    panel_areas: &[f64],
    exterior_pressure: &[Complex64],
    exterior_normal_velocity: &[Complex64],
    interior_pressure: &[Complex64],
    interior_normal_velocity: &[Complex64],
    tolerance: f64,
) -> Result<TransmissionPowerBalanceReport, OperatorContractError> {
    if panel_areas.is_empty() {
        return Err(OperatorContractError::InsufficientProbes);
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(OperatorContractError::NonFinite {
            label: "transmission power-balance tolerance",
        });
    }
    let expected = panel_areas.len();
    for (label, actual) in [
        ("exterior pressure", exterior_pressure.len()),
        ("exterior normal velocity", exterior_normal_velocity.len()),
        ("interior pressure", interior_pressure.len()),
        ("interior normal velocity", interior_normal_velocity.len()),
    ] {
        if actual != expected {
            return Err(OperatorContractError::DimensionMismatch {
                label,
                expected,
                actual,
            });
        }
    }
    let exterior_power =
        constant_panel_power(panel_areas, exterior_pressure, exterior_normal_velocity)?;
    let interior_power =
        constant_panel_power(panel_areas, interior_pressure, interior_normal_velocity)?;
    let absolute_residual = exterior_power - interior_power;
    let scale = exterior_power
        .abs()
        .max(interior_power.abs())
        .max(f64::MIN_POSITIVE);
    let relative_error = absolute_residual.abs() / scale;
    if !absolute_residual.is_finite() || !relative_error.is_finite() {
        return Err(OperatorContractError::NonFinite {
            label: "transmission power-balance residual",
        });
    }
    if relative_error > tolerance {
        return Err(OperatorContractError::AccuracyExceeded {
            label: "transmission power balance",
            observed: relative_error,
            tolerance,
        });
    }
    Ok(TransmissionPowerBalanceReport {
        exterior_power,
        interior_power,
        absolute_residual,
        relative_error,
    })
}

fn dense_matrix_action(matrix: &DenseMatrix<Complex64>, vector: &[Complex64]) -> Vec<Complex64> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| matrix[[row, column]] * vector[column])
                .sum()
        })
        .collect()
}

fn weighted_transmission_action(
    action: &[Complex64],
    interface_dofs: usize,
    signed_omega: f64,
) -> Vec<Complex64> {
    let pressure_weight = Complex64::new(0.0, 1.0 / signed_omega);
    let mut weighted = vec![Complex64::new(0.0, 0.0); action.len()];
    for index in 0..interface_dofs {
        weighted[index] = pressure_weight * action[interface_dofs + index];
        weighted[interface_dofs + index] = action[index];
    }
    weighted
}

/// Check that the Hermitian part of a square operator is passive for all
/// supplied probes. The tolerance is relative to `||x|| ||Ax||`, preserving
/// useful behavior for tiny physical pressures and velocities.
pub fn check_passivity_contract<O: LinearOperator<Complex64>>(
    operator: &O,
    probes: &[Vec<Complex64>],
    tolerance: f64,
) -> Result<PassivityContractReport, OperatorContractError> {
    if operator.num_rows() != operator.num_cols() {
        return Err(OperatorContractError::NonSquare {
            label: "passivity contract",
            rows: operator.num_rows(),
            cols: operator.num_cols(),
        });
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(OperatorContractError::NonFinite {
            label: "passivity tolerance",
        });
    }
    if probes.is_empty() {
        return Err(OperatorContractError::InsufficientProbes);
    }
    let mut minimum_power = f64::INFINITY;
    for (probe_index, probe) in probes.iter().enumerate() {
        if probe.len() != operator.num_cols() {
            return Err(OperatorContractError::DimensionMismatch {
                label: "passivity probe",
                expected: operator.num_cols(),
                actual: probe.len(),
            });
        }
        ensure_finite("passivity probe", probe)?;
        let action = operator.apply(probe);
        check_action_dimensions(
            "passivity operator action",
            operator.num_rows(),
            action.len(),
        )?;
        ensure_finite("passivity operator action", &action)?;
        let power = inner_product(probe, &action).re;
        let scale = probe
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            * action
                .iter()
                .map(|value| value.norm_sqr())
                .sum::<f64>()
                .sqrt()
                .max(f64::MIN_POSITIVE);
        if !power.is_finite() || !scale.is_finite() {
            return Err(OperatorContractError::NonFinite {
                label: "passivity contract",
            });
        }
        minimum_power = minimum_power.min(power);
        if power < -tolerance * scale {
            return Err(OperatorContractError::PassivityViolation {
                probe: probe_index,
                power,
                tolerance,
            });
        }
    }
    Ok(PassivityContractReport {
        minimum_power,
        probes: probes.len(),
    })
}

/// Check that applying an operator to a block of RHS vectors is equivalent to
/// applying it independently to every column.
///
/// This is backend-neutral even when the implementation's production block
/// kernel is not exposed publicly: it validates the required column-wise
/// semantics that block Krylov solvers rely on.
pub fn check_block_rhs_contract<O: LinearOperator<Complex64>>(
    operator: &O,
    rhs_columns: &[Vec<Complex64>],
) -> Result<f64, OperatorContractError> {
    if rhs_columns.is_empty() {
        return Ok(0.0);
    }
    let mut input = DenseMatrix::zeros(operator.num_cols(), rhs_columns.len());
    for (column, rhs) in rhs_columns.iter().enumerate() {
        if rhs.len() != operator.num_cols() {
            return Err(OperatorContractError::DimensionMismatch {
                label: "block RHS",
                expected: operator.num_cols(),
                actual: rhs.len(),
            });
        }
        for (row, &value) in rhs.iter().enumerate() {
            input[[row, column]] = value;
        }
    }
    let mut block_output = DenseMatrix::zeros(operator.num_rows(), rhs_columns.len());
    operator.apply_block(&input, &mut block_output);

    let mut max_error: f64 = 0.0;
    for (column, rhs) in rhs_columns.iter().enumerate() {
        let expected = operator.apply(rhs);
        check_action_dimensions(
            "block RHS operator action",
            operator.num_rows(),
            expected.len(),
        )?;
        ensure_finite("block RHS operator action", &expected)?;
        let numerator = expected
            .iter()
            .enumerate()
            .map(|(row, lhs)| (*lhs - block_output[[row, column]]).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let denominator = expected
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        max_error = max_error.max(numerator / denominator);
    }
    if !max_error.is_finite() {
        return Err(OperatorContractError::NonFinite {
            label: "block RHS contract",
        });
    }
    Ok(max_error)
}

fn check_action_dimensions(
    label: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), OperatorContractError> {
    if expected != actual {
        return Err(OperatorContractError::DimensionMismatch {
            label,
            expected,
            actual,
        });
    }
    Ok(())
}

fn ensure_finite(label: &'static str, values: &[Complex64]) -> Result<(), OperatorContractError> {
    if values
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(OperatorContractError::NonFinite { label });
    }
    Ok(())
}

/// Compute the time-averaged normal acoustic power for constant-panel data.
///
/// The sign follows the supplied outward-normal velocity. Callers can use the
/// result for passivity and energy-balance gates without embedding a separate
/// pressure/amplitude convention in each solver frontend.
pub fn constant_panel_power(
    panel_areas: &[f64],
    pressure: &[Complex64],
    normal_velocity: &[Complex64],
) -> Result<f64, OperatorContractError> {
    if panel_areas.len() != pressure.len() {
        return Err(OperatorContractError::DimensionMismatch {
            label: "panel pressure",
            expected: panel_areas.len(),
            actual: pressure.len(),
        });
    }
    if panel_areas.len() != normal_velocity.len() {
        return Err(OperatorContractError::DimensionMismatch {
            label: "panel normal velocity",
            expected: panel_areas.len(),
            actual: normal_velocity.len(),
        });
    }
    let mut power = 0.0;
    for ((&area, &p), &velocity) in panel_areas.iter().zip(pressure).zip(normal_velocity) {
        if !area.is_finite() || area < 0.0 || !p.re.is_finite() || !p.im.is_finite() {
            return Err(OperatorContractError::NonFinite {
                label: "constant-panel power",
            });
        }
        if !velocity.re.is_finite() || !velocity.im.is_finite() {
            return Err(OperatorContractError::NonFinite {
                label: "constant-panel power",
            });
        }
        power += 0.5 * area * (p * velocity.conj()).re;
    }
    if !power.is_finite() {
        return Err(OperatorContractError::NonFinite {
            label: "constant-panel power",
        });
    }
    Ok(power)
}

fn check_vector_dimensions<O: LinearOperator<Complex64>>(
    operator: &O,
    x: &[Complex64],
    y: &[Complex64],
) -> Result<(), OperatorContractError> {
    if x.len() != operator.num_cols() {
        return Err(OperatorContractError::DimensionMismatch {
            label: "adjoint input",
            expected: operator.num_cols(),
            actual: x.len(),
        });
    }
    if y.len() != operator.num_rows() {
        return Err(OperatorContractError::DimensionMismatch {
            label: "adjoint output",
            expected: operator.num_rows(),
            actual: y.len(),
        });
    }
    Ok(())
}

fn inner_product<T: ComplexField>(left: &[T], right: &[T]) -> T {
    left.iter()
        .zip(right)
        .fold(T::zero(), |sum, (&left, &right)| sum + left.conj() * right)
}

fn bilinear_product(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .fold(Complex64::new(0.0, 0.0), |sum, (&left, &right)| {
            sum + left * right
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MatrixOperator(DenseMatrix<Complex64>);

    impl LinearOperator<Complex64> for MatrixOperator {
        fn num_rows(&self) -> usize {
            self.0.nrows()
        }

        fn num_cols(&self) -> usize {
            self.0.ncols()
        }

        fn apply_into(&self, input: &[Complex64], output: &mut [Complex64]) {
            output.copy_from_slice(&dense_matrix_action(&self.0, input));
        }

        fn apply_transpose_into(&self, input: &[Complex64], output: &mut [Complex64]) {
            self.0.matvec_transpose_into(input, output);
        }
    }

    struct TwoByTwo;

    impl LinearOperator<Complex64> for TwoByTwo {
        fn num_rows(&self) -> usize {
            2
        }

        fn num_cols(&self) -> usize {
            2
        }

        fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
            y[0] = Complex64::new(2.0, 1.0) * x[0] + x[1];
            y[1] = Complex64::new(-1.0, 0.5) * x[0] + Complex64::new(3.0, -2.0) * x[1];
        }

        fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
            y[0] = Complex64::new(2.0, 1.0) * x[0] + Complex64::new(-1.0, 0.5) * x[1];
            y[1] = x[0] + Complex64::new(3.0, -2.0) * x[1];
        }
    }

    struct ScaledTiny(Complex64);

    impl LinearOperator<Complex64> for ScaledTiny {
        fn num_rows(&self) -> usize {
            1
        }

        fn num_cols(&self) -> usize {
            1
        }

        fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
            y[0] = self.0 * x[0];
        }

        fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
            y[0] = self.0 * x[0];
        }
    }

    struct BrokenTranspose;

    impl LinearOperator<Complex64> for BrokenTranspose {
        fn num_rows(&self) -> usize {
            2
        }

        fn num_cols(&self) -> usize {
            2
        }

        fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
            y[0] = x[0] + Complex64::new(2.0, 0.0) * x[1];
            y[1] = Complex64::new(3.0, 0.0) * x[0] + x[1];
        }

        fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
            // Deliberately wrong: this is the identity rather than Aᵀ.
            y.copy_from_slice(x);
        }
    }

    struct Active;

    impl LinearOperator<Complex64> for Active {
        fn num_rows(&self) -> usize {
            1
        }

        fn num_cols(&self) -> usize {
            1
        }

        fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
            y[0] = -x[0];
        }

        fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
            y[0] = -x[0];
        }
    }

    struct MalformedAction;

    impl LinearOperator<Complex64> for MalformedAction {
        fn num_rows(&self) -> usize {
            2
        }

        fn num_cols(&self) -> usize {
            2
        }

        fn apply(&self, _x: &[Complex64]) -> Vec<Complex64> {
            // Deliberately violates the declared row count. The shared
            // contract must reject this before a zip can silently truncate.
            vec![Complex64::new(0.0, 0.0)]
        }

        fn apply_into(&self, _x: &[Complex64], y: &mut [Complex64]) {
            y.fill(Complex64::new(0.0, 0.0));
        }

        fn apply_transpose_into(&self, _x: &[Complex64], y: &mut [Complex64]) {
            y.fill(Complex64::new(0.0, 0.0));
        }
    }

    struct MaskingOperator;

    impl LinearOperator<Complex64> for MaskingOperator {
        fn num_rows(&self) -> usize {
            2
        }

        fn num_cols(&self) -> usize {
            2
        }

        fn apply(&self, _x: &[Complex64]) -> Vec<Complex64> {
            // Keep the action finite even for a non-finite probe so the
            // probe-admission check is independently exercised.
            vec![Complex64::new(0.0, 0.0); 2]
        }

        fn apply_into(&self, _x: &[Complex64], y: &mut [Complex64]) {
            y.fill(Complex64::new(0.0, 0.0));
        }

        fn apply_transpose_into(&self, _x: &[Complex64], y: &mut [Complex64]) {
            y.fill(Complex64::new(0.0, 0.0));
        }
    }

    #[test]
    fn adjoint_contract_accepts_complex_operator() {
        let x = vec![Complex64::new(1.0, -2.0), Complex64::new(0.5, 3.0)];
        let y = vec![Complex64::new(-2.0, 1.0), Complex64::new(4.0, 0.25)];
        let report = check_adjoint_contract(&TwoByTwo, &x, &y).expect("dimensions are valid");
        assert!(report.relative_error < 1e-12, "{report:?}");
        let block_error =
            check_block_rhs_contract(&TwoByTwo, &[x, y]).expect("block dimensions are valid");
        assert!(block_error < 1e-12);

        let accuracy = require_operator_accuracy(
            &TwoByTwo,
            &TwoByTwo,
            &[vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]],
            1e-12,
        )
        .expect("an operator is exact against itself");
        assert_eq!(accuracy.probes, 1);
    }

    #[test]
    fn panel_power_is_area_weighted_and_dimension_checked() {
        let power = constant_panel_power(
            &[2.0, 1.0],
            &[Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0)],
            &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
        )
        .expect("finite panel data");
        assert!((power - 2.5).abs() < 1e-12);
        assert!(constant_panel_power(&[1.0], &[], &[]).is_err());
    }

    #[test]
    fn accuracy_contract_rejects_vacuous_probe_set() {
        let error = require_operator_accuracy(&TwoByTwo, &TwoByTwo, &[], 1e-12)
            .expect_err("an empty probe set cannot establish operator accuracy");
        assert_eq!(error, OperatorContractError::InsufficientProbes);
    }

    #[test]
    fn accuracy_contract_rejects_malformed_action_length() {
        let error = require_operator_accuracy(
            &MalformedAction,
            &TwoByTwo,
            &[vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]],
            1e-12,
        )
        .expect_err("declared operator rows must match action length");
        assert_eq!(
            error,
            OperatorContractError::DimensionMismatch {
                label: "approximate operator action",
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn accuracy_contract_rejects_nonfinite_probe_before_operator_action() {
        let error = require_operator_accuracy(
            &MaskingOperator,
            &MaskingOperator,
            &[vec![
                Complex64::new(f64::NAN, 0.0),
                Complex64::new(0.0, 0.0),
            ]],
            1e-12,
        )
        .expect_err("non-finite probes cannot support an accuracy claim");
        assert_eq!(
            error,
            OperatorContractError::NonFinite {
                label: "operator accuracy probe"
            }
        );
    }

    #[test]
    fn physical_contracts_reject_malformed_actions() {
        let x = vec![Complex64::new(1.0, 0.0); 2];
        let y = vec![Complex64::new(0.5, -0.25); 2];

        assert_eq!(
            check_adjoint_contract(&MalformedAction, &x, &y),
            Err(OperatorContractError::DimensionMismatch {
                label: "forward operator action",
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            check_reciprocity_contract(&MalformedAction, &x, &y),
            Err(OperatorContractError::DimensionMismatch {
                label: "forward operator action",
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            check_passivity_contract(&MalformedAction, &[x.clone()], 1e-12),
            Err(OperatorContractError::DimensionMismatch {
                label: "passivity operator action",
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            check_block_rhs_contract(&MalformedAction, &[x],),
            Err(OperatorContractError::DimensionMismatch {
                label: "block RHS operator action",
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn accuracy_contract_preserves_relative_scale_for_tiny_actions() {
        let approximate = ScaledTiny(Complex64::new(2.0e-30, 0.0));
        let reference = ScaledTiny(Complex64::new(1.0e-30, 0.0));
        let error = require_operator_accuracy(
            &approximate,
            &reference,
            &[vec![Complex64::new(1.0, 0.0)]],
            0.5,
        )
        .expect_err("a 100% relative tiny-action error must fail");
        assert!(matches!(
            error,
            OperatorContractError::AccuracyExceeded { .. }
        ));
    }

    #[test]
    fn reciprocity_uses_transpose_without_conjugating_complex_probes() {
        let x = vec![Complex64::new(1.0, 2.0), Complex64::new(-0.5, 1.0)];
        let y = vec![Complex64::new(2.0, -1.0), Complex64::new(0.25, 3.0)];
        let report = check_reciprocity_contract(&TwoByTwo, &x, &y).expect("square operator");
        assert!(report.relative_error < 1e-12, "{report:?}");
        let broken = check_reciprocity_contract(&BrokenTranspose, &x, &y)
            .expect("dimensions are valid even when the identity is wrong");
        assert!(broken.relative_error > 1e-3);
    }

    #[test]
    fn passivity_gate_rejects_active_quadratic_forms() {
        let probes = vec![
            vec![Complex64::new(1.0, 0.0)],
            vec![Complex64::new(0.5, -2.0)],
        ];
        let error = check_passivity_contract(&Active, &probes, 1e-12)
            .expect_err("negative Hermitian power is active");
        assert!(matches!(
            error,
            OperatorContractError::PassivityViolation { .. }
        ));
        let passive_probes = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.5, -0.25)],
            vec![Complex64::new(0.5, -2.0), Complex64::new(-1.0, 0.75)],
        ];
        let passive = check_passivity_contract(&TwoByTwo, &passive_probes, 1e-12)
            .expect("positive Hermitian part is passive");
        assert!(passive.minimum_power > 0.0);
    }

    fn weighted_symmetric_transmission_matrix(
        interface_dofs: usize,
        omega: f64,
        harmonic_factor: f64,
    ) -> DenseMatrix<Complex64> {
        let mut matrix = DenseMatrix::zeros(2 * interface_dofs, 2 * interface_dofs);
        let signed_omega = harmonic_factor * omega;
        let i_omega = Complex64::new(0.0, signed_omega);
        for row in 0..interface_dofs {
            for column in 0..interface_dofs {
                let stiffness = Complex64::new(
                    1.0 + 0.2 * (row + column) as f64,
                    0.1 * (row as f64 - column as f64),
                );
                let transpose_stiffness = Complex64::new(
                    1.0 + 0.2 * (row + column) as f64,
                    0.1 * (column as f64 - row as f64),
                );
                let symmetric_mass = Complex64::new(
                    0.7 + 0.15 * (row + column) as f64,
                    -0.05 * (row + column) as f64,
                );
                let symmetric_hypersingular = Complex64::new(
                    2.0 + 0.3 * (row + column) as f64,
                    0.08 * (row + column) as f64,
                );
                matrix[[row, column]] = stiffness;
                matrix[[row, interface_dofs + column]] = -i_omega * symmetric_mass;
                matrix[[interface_dofs + row, column]] = symmetric_hypersingular;
                matrix[[interface_dofs + row, interface_dofs + column]] =
                    -i_omega * transpose_stiffness;
            }
        }
        matrix
    }

    #[test]
    fn transmission_reciprocity_contract_checks_weighted_transpose() {
        let matrix = weighted_symmetric_transmission_matrix(2, 120.0, -1.0);
        let probes = vec![
            vec![
                Complex64::new(1.0, 0.2),
                Complex64::new(-0.4, 0.7),
                Complex64::new(0.3, -0.8),
                Complex64::new(1.2, 0.1),
            ],
            vec![
                Complex64::new(-0.2, 1.1),
                Complex64::new(0.9, -0.3),
                Complex64::new(1.4, 0.5),
                Complex64::new(-0.7, 0.6),
            ],
            vec![
                Complex64::new(0.8, -0.4),
                Complex64::new(-1.3, 0.2),
                Complex64::new(0.1, 0.9),
                Complex64::new(0.6, -1.0),
            ],
        ];
        let report =
            check_transmission_reciprocity_contract(&matrix, 2, 120.0, -1.0, &probes, 1.0e-12)
                .expect("weighted-symmetric matrix should satisfy reciprocity");
        assert_eq!(report.probes, 3);
        assert_eq!(report.pairs, 3);
        assert!(report.relative_error < 1.0e-12, "{report:?}");
    }

    #[test]
    fn transmission_reciprocity_contract_rejects_broken_weighted_transpose() {
        let mut matrix = weighted_symmetric_transmission_matrix(2, 120.0, 1.0);
        matrix[[0, 1]] += Complex64::new(0.25, -0.4);
        let probes = vec![
            vec![
                Complex64::new(1.0, 0.2),
                Complex64::new(-0.4, 0.7),
                Complex64::new(0.3, -0.8),
                Complex64::new(1.2, 0.1),
            ],
            vec![
                Complex64::new(-0.2, 1.1),
                Complex64::new(0.9, -0.3),
                Complex64::new(1.4, 0.5),
                Complex64::new(-0.7, 0.6),
            ],
        ];
        let error =
            check_transmission_reciprocity_contract(&matrix, 2, 120.0, 1.0, &probes, 1.0e-12)
                .expect_err("broken weighted transpose must fail the gate");
        assert!(matches!(
            error,
            OperatorContractError::AccuracyExceeded {
                label: "transmission reciprocity contract",
                ..
            }
        ));
    }

    #[test]
    fn transmission_reciprocity_contract_rejects_invalid_admission() {
        let matrix = weighted_symmetric_transmission_matrix(1, 120.0, 1.0);
        let probe = vec![vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]];
        assert_eq!(
            check_transmission_reciprocity_contract(&matrix, 1, 120.0, 1.0, &probe, 1.0e-12)
                .expect_err("one probe is vacuous"),
            OperatorContractError::InsufficientProbes
        );
        let probes = vec![probe[0].clone(), probe[0].clone()];
        assert_eq!(
            check_transmission_reciprocity_contract(&matrix, 1, 120.0, 0.5, &probes, 1.0e-12)
                .expect_err("harmonic factor must be a sign"),
            OperatorContractError::NonFinite {
                label: "transmission reciprocity configuration"
            }
        );
    }

    #[test]
    fn transmission_operator_contract_checks_matrix_free_actions() {
        let matrix = weighted_symmetric_transmission_matrix(2, 120.0, -1.0);
        let operator = MatrixOperator(matrix);
        let probes = vec![
            vec![
                Complex64::new(1.0, 0.2),
                Complex64::new(-0.4, 0.7),
                Complex64::new(0.3, -0.8),
                Complex64::new(1.2, 0.1),
            ],
            vec![
                Complex64::new(-0.2, 1.1),
                Complex64::new(0.9, -0.3),
                Complex64::new(1.4, 0.5),
                Complex64::new(-0.7, 0.6),
            ],
            vec![
                Complex64::new(0.8, -0.4),
                Complex64::new(-1.3, 0.2),
                Complex64::new(0.1, 0.9),
                Complex64::new(0.6, -1.0),
            ],
        ];
        let report =
            check_transmission_reciprocity_operator(&operator, 2, 120.0, -1.0, &probes, 1.0e-12)
                .expect("matrix-free weighted-symmetric operator should pass");
        assert_eq!(report.pairs, 3);
        assert!(report.relative_error < 1.0e-12, "{report:?}");
    }

    #[test]
    fn transmission_passivity_contract_uses_weighted_power_form() {
        let mut matrix = DenseMatrix::zeros(2, 2);
        matrix[[0, 1]] = Complex64::new(1.0, 0.0);
        matrix[[1, 0]] = Complex64::new(0.0, -1.0);
        let operator = MatrixOperator(matrix.clone());
        let probes = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(0.3, -0.7), Complex64::new(-0.2, 0.4)],
        ];
        let report =
            check_transmission_passivity_contract(&operator, 1, 1.0, 1.0, &probes, 1.0e-12)
                .expect("positive weighted power form should pass");
        assert!(report.minimum_power > 0.0);

        let negative = MatrixOperator(DenseMatrix::from_fn(2, 2, |row, column| {
            -matrix[[row, column]]
        }));
        let error = check_transmission_passivity_contract(&negative, 1, 1.0, 1.0, &probes, 1.0e-12)
            .expect_err("negative weighted power form must fail");
        assert!(matches!(
            error,
            OperatorContractError::PassivityViolation { .. }
        ));
    }

    #[test]
    fn transmission_power_balance_checks_traces_and_tolerance() {
        let areas = [0.5, 0.75];
        let exterior_pressure = [Complex64::new(2.0, 1.0), Complex64::new(1.0, -0.5)];
        let exterior_velocity = [Complex64::new(0.1, -0.2), Complex64::new(0.2, 0.1)];
        let report = check_transmission_power_balance(
            &areas,
            &exterior_pressure,
            &exterior_velocity,
            &exterior_pressure,
            &exterior_velocity,
            1.0e-12,
        )
        .expect("identical traces should conserve power");
        assert_eq!(report.absolute_residual, 0.0);
        assert_eq!(report.relative_error, 0.0);

        let mut mismatched_velocity = exterior_velocity;
        mismatched_velocity[1] += Complex64::new(0.25, 0.0);
        let error = check_transmission_power_balance(
            &areas,
            &exterior_pressure,
            &exterior_velocity,
            &exterior_pressure,
            &mismatched_velocity,
            1.0e-12,
        )
        .expect_err("mismatched trace power should fail");
        assert!(matches!(
            error,
            OperatorContractError::AccuracyExceeded { .. }
        ));
    }
}
