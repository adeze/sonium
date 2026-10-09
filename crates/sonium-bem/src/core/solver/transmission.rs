//! Checked CPU solve for the dense constant-panel Müller transmission system.
//!
//! Assembly and this solve are deliberately separate: callers can inspect or
//! certify the assembled operator before supplying the two-block trace RHS.
//! The path is intentionally dense/direct and CPU-only; accelerated and
//! incident-field RHS construction remain separate follow-up work.

use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;
use thiserror::Error;

use crate::core::assembly::{TransmissionRhsError, TransmissionTbemSystem};
use crate::core::incident::IncidentField;
use crate::core::postprocess::pressure::{
    FieldPoint, compute_scattered_field_checked, compute_total_field_checked,
    map_dual_basis_coefficients_checked,
};

const DEFAULT_MIN_SCALE: f64 = f64::MIN_POSITIVE;

/// Errors returned while solving a checked transmission system.
#[derive(Debug, Error, PartialEq)]
pub enum TransmissionSolveError {
    /// The assembled system has no interface unknowns.
    #[error("transmission system has no interface degrees of freedom")]
    EmptySystem,
    /// The caller supplied a RHS with the wrong length.
    #[error("transmission RHS has length {actual}, expected {expected}")]
    RhsLength {
        /// Number of values required by the two-block system.
        expected: usize,
        /// Number of values supplied by the caller.
        actual: usize,
    },
    /// The residual tolerance is invalid.
    #[error("transmission residual tolerance must be finite and positive, got {value}")]
    InvalidTolerance {
        /// Supplied residual tolerance.
        value: f64,
    },
    /// A caller-provided RHS contains a non-finite value.
    #[error("transmission RHS contains a non-finite value at index {index}")]
    NonFiniteRhs {
        /// Position of the invalid RHS value.
        index: usize,
    },
    /// The direct factorization rejected the assembled matrix.
    #[error("transmission direct solve failed: {0}")]
    DirectSolve(String),
    /// Direct factorization rejected the Hermitian-adjoint matrix.
    #[error("transmission adjoint solve failed: {0}")]
    AdjointSolve(String),
    /// Incident-field traces could not be projected onto the assembled system.
    #[error("transmission incident RHS construction failed: {0}")]
    IncidentRhs(String),
    /// Exterior field postprocessing rejected the retained trace or geometry.
    #[error("transmission exterior field evaluation failed: {0}")]
    FieldEvaluation(String),
    /// The returned iterate did not satisfy the original assembled operator.
    #[error("transmission residual {residual:.3e} exceeds tolerance {tolerance:.3e}")]
    Residual {
        /// Certified relative residual.
        residual: f64,
        /// Maximum accepted relative residual.
        tolerance: f64,
    },
    /// The solve produced a non-finite value or residual.
    #[error("transmission solve produced a non-finite {quantity}")]
    NonFiniteSolution {
        /// Quantity that contained a non-finite value.
        quantity: &'static str,
    },
}

/// Solution of the two-block constant-panel Müller system.
///
/// `values[..interface_dofs]` contains the first trace unknown and the
/// remaining values contain the second trace unknown, in the same ordering as
/// the assembled system.  The block names are intentionally neutral because
/// the physical trace scaling belongs to the caller's formulation/RHS adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct TransmissionSolution {
    values: Vec<Complex64>,
    interface_dofs: usize,
    /// Relative residual certified against the original dense matrix.
    pub relative_residual: f64,
}

/// Certified Hermitian-adjoint solution of a dense constant-panel Müller
/// transmission system.
///
/// The values satisfy `Aᴴ y = rhs`, with the same pressure/velocity block
/// ordering as [`TransmissionSolution`].
#[derive(Debug, Clone, PartialEq)]
pub struct TransmissionAdjointSolution {
    values: Vec<Complex64>,
    interface_dofs: usize,
    /// Relative residual certified against the explicitly formed `Aᴴ` matrix.
    pub relative_residual: f64,
}

impl TransmissionAdjointSolution {
    /// Return the first interface trace block.
    pub fn first_trace(&self) -> &[Complex64] {
        &self.values[..self.interface_dofs]
    }

    /// Return the second interface trace block.
    pub fn second_trace(&self) -> &[Complex64] {
        &self.values[self.interface_dofs..]
    }

    /// Return the flattened adjoint solution in matrix ordering.
    pub fn values(&self) -> &[Complex64] {
        &self.values
    }

    /// Evaluate the exterior adjoint pressure represented by this solution.
    ///
    /// Adjoint traces are consumed as scattered Cauchy data; no incident-field
    /// contribution is added. The same checked geometry, basis-layout, finite
    /// sample, and result validation as ordinary field evaluation applies.
    pub fn evaluate_exterior_adjoint_pressure_checked(
        &self,
        system: &TransmissionTbemSystem,
        eval_points: &[[f64; 3]],
    ) -> Result<Vec<Complex64>, TransmissionSolveError> {
        if self.values.len() != system.total_dofs()
            || self.interface_dofs != system.interface_dofs()
        {
            return Err(TransmissionSolveError::NonFiniteSolution {
                quantity: "adjoint trace layout",
            });
        }
        evaluate_exterior_adjoint_pressure_values_checked(&self.values, system, eval_points)
    }
}

impl TransmissionSolution {
    /// Return the first interface trace block.
    pub fn first_trace(&self) -> &[Complex64] {
        &self.values[..self.interface_dofs]
    }

    /// Return the second interface trace block.
    pub fn second_trace(&self) -> &[Complex64] {
        &self.values[self.interface_dofs..]
    }

    /// Return the flattened two-block solution in matrix ordering.
    pub fn values(&self) -> &[Complex64] {
        &self.values
    }

    /// Evaluate the exterior total pressure from a solved transmission trace.
    ///
    /// The solve returns total interface pressure and normal velocity. The
    /// checked field kernel expects scattered Cauchy data, so the incident
    /// traces are subtracted before applying the exterior representation
    /// formula and then added back by the total-field wrapper.
    pub fn evaluate_exterior_field_checked(
        &self,
        system: &TransmissionTbemSystem,
        eval_points: &[[f64; 3]],
        incident: &IncidentField,
    ) -> Result<Vec<FieldPoint>, TransmissionSolveError> {
        let expected = system.total_dofs();
        if self.values.len() != expected || self.interface_dofs != system.interface_dofs() {
            return Err(TransmissionSolveError::RhsLength {
                expected,
                actual: self.values.len(),
            });
        }
        if self.values.iter().any(|value| !complex_is_finite(*value)) {
            return Err(TransmissionSolveError::NonFiniteSolution { quantity: "trace" });
        }

        let pressure = incident.evaluate_pressure(system.panel_centers(), &system.exterior);
        let normal_derivative = incident.evaluate_normal_derivative(
            system.panel_centers(),
            system.panel_normals(),
            &system.exterior,
        );
        let i_omega_rho = Complex64::new(0.0, system.exterior.pressure_factor);
        let incident_velocity = normal_derivative
            .into_iter()
            .map(|value| {
                if i_omega_rho.norm() > 1.0e-30 {
                    value / i_omega_rho
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .collect::<Vec<_>>();
        let scattered_pressure = self
            .first_trace()
            .iter()
            .zip(&pressure)
            .map(|(&total, &incident)| total - incident)
            .collect::<Vec<_>>();
        let scattered_velocity = self
            .second_trace()
            .iter()
            .zip(&incident_velocity)
            .map(|(&total, &incident)| total - incident)
            .collect::<Vec<_>>();
        if scattered_pressure
            .iter()
            .chain(scattered_velocity.iter())
            .any(|value| !complex_is_finite(*value))
        {
            return Err(TransmissionSolveError::NonFiniteSolution {
                quantity: "scattered trace",
            });
        }

        let (elements, nodes) = system.field_geometry();
        compute_total_field_checked(
            eval_points,
            elements,
            nodes,
            &scattered_pressure,
            Some(&scattered_velocity),
            incident,
            &system.exterior,
            system.basis_order(),
        )
        .map_err(|error| TransmissionSolveError::FieldEvaluation(error.to_string()))
    }
}

/// Solve an assembled transmission system on the CPU and certify its true
/// residual against the original dense matrix.
///
/// This is the first production integration point after typed XEM-backed
/// Müller assembly.  The RHS is already expressed in the system's two-block
/// trace convention; incident-field projection and field evaluation are not
/// silently inferred here.
pub fn solve_transmission_system(
    system: &TransmissionTbemSystem,
    rhs: &[Complex64],
    residual_tolerance: f64,
) -> Result<TransmissionSolution, TransmissionSolveError> {
    let interface_dofs = system.interface_dofs();
    let expected = system.total_dofs();
    if interface_dofs == 0 {
        return Err(TransmissionSolveError::EmptySystem);
    }
    if rhs.len() != expected {
        return Err(TransmissionSolveError::RhsLength {
            expected,
            actual: rhs.len(),
        });
    }
    if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
        return Err(TransmissionSolveError::InvalidTolerance {
            value: residual_tolerance,
        });
    }
    for (index, value) in rhs.iter().enumerate() {
        if !complex_is_finite(*value) {
            return Err(TransmissionSolveError::NonFiniteRhs { index });
        }
    }

    let values = sonium_solvers::direct::lu_solve(system.matrix(), rhs)
        .map_err(|error| TransmissionSolveError::DirectSolve(error.to_string()))?;
    if values.len() != expected || values.iter().any(|value| !complex_is_finite(*value)) {
        return Err(TransmissionSolveError::NonFiniteSolution {
            quantity: "solution",
        });
    }

    let residual = relative_residual(system.matrix(), &values, rhs);
    if !residual.is_finite() {
        return Err(TransmissionSolveError::NonFiniteSolution {
            quantity: "residual",
        });
    }
    if residual > residual_tolerance {
        return Err(TransmissionSolveError::Residual {
            residual,
            tolerance: residual_tolerance,
        });
    }

    Ok(TransmissionSolution {
        values,
        interface_dofs,
        relative_residual: residual,
    })
}

/// Build the exterior incident-field RHS and solve the checked transmission
/// system in one CPU path.
/// Solve the Hermitian-adjoint transmission system `Aᴴ y = rhs` on CPU.
///
/// The RHS uses the same flattened pressure/velocity ordering as
/// [`solve_transmission_system`]. The residual is certified against an
/// explicitly formed Hermitian transpose.
pub fn solve_transmission_adjoint_system(
    system: &TransmissionTbemSystem,
    rhs: &[Complex64],
    residual_tolerance: f64,
) -> Result<TransmissionAdjointSolution, TransmissionSolveError> {
    let interface_dofs = system.interface_dofs();
    let expected = system.total_dofs();
    if interface_dofs == 0 {
        return Err(TransmissionSolveError::EmptySystem);
    }
    if rhs.len() != expected {
        return Err(TransmissionSolveError::RhsLength {
            expected,
            actual: rhs.len(),
        });
    }
    if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
        return Err(TransmissionSolveError::InvalidTolerance {
            value: residual_tolerance,
        });
    }
    for (index, value) in rhs.iter().enumerate() {
        if !complex_is_finite(*value) {
            return Err(TransmissionSolveError::NonFiniteRhs { index });
        }
    }

    let hermitian = hermitian_transpose(system.matrix());
    let values = sonium_solvers::direct::lu_solve(&hermitian, rhs)
        .map_err(|error| TransmissionSolveError::AdjointSolve(error.to_string()))?;
    if values.len() != expected || values.iter().any(|value| !complex_is_finite(*value)) {
        return Err(TransmissionSolveError::NonFiniteSolution {
            quantity: "adjoint solution",
        });
    }
    let residual = relative_residual(&hermitian, &values, rhs);
    if !residual.is_finite() {
        return Err(TransmissionSolveError::NonFiniteSolution {
            quantity: "adjoint residual",
        });
    }
    if residual > residual_tolerance {
        return Err(TransmissionSolveError::Residual {
            residual,
            tolerance: residual_tolerance,
        });
    }
    Ok(TransmissionAdjointSolution {
        values,
        interface_dofs,
        relative_residual: residual,
    })
}

/// Solve several Hermitian-adjoint transmission systems.
///
/// The dense Hermitian matrix is formed once and each returned column carries
/// its own independently certified residual. An empty batch returns an empty
/// vector without assembling or factoring the operator.
pub fn solve_transmission_adjoint_many(
    system: &TransmissionTbemSystem,
    rhs_columns: &[Vec<Complex64>],
    residual_tolerance: f64,
) -> Result<Vec<TransmissionAdjointSolution>, TransmissionSolveError> {
    if rhs_columns.is_empty() {
        return Ok(Vec::new());
    }
    let interface_dofs = system.interface_dofs();
    let expected = system.total_dofs();
    if interface_dofs == 0 {
        return Err(TransmissionSolveError::EmptySystem);
    }
    if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
        return Err(TransmissionSolveError::InvalidTolerance {
            value: residual_tolerance,
        });
    }
    for rhs in rhs_columns {
        if rhs.len() != expected {
            return Err(TransmissionSolveError::RhsLength {
                expected,
                actual: rhs.len(),
            });
        }
        if let Some((index, _)) = rhs
            .iter()
            .enumerate()
            .find(|(_, value)| !complex_is_finite(**value))
        {
            return Err(TransmissionSolveError::NonFiniteRhs { index });
        }
    }

    let hermitian = hermitian_transpose(system.matrix());
    let mut solutions = Vec::with_capacity(rhs_columns.len());
    for rhs in rhs_columns {
        let values = sonium_solvers::direct::lu_solve(&hermitian, rhs)
            .map_err(|error| TransmissionSolveError::AdjointSolve(error.to_string()))?;
        if values.len() != expected || values.iter().any(|value| !complex_is_finite(*value)) {
            return Err(TransmissionSolveError::NonFiniteSolution {
                quantity: "adjoint solution",
            });
        }
        let residual = relative_residual(&hermitian, &values, rhs);
        if !residual.is_finite() {
            return Err(TransmissionSolveError::NonFiniteSolution {
                quantity: "adjoint residual",
            });
        }
        if residual > residual_tolerance {
            return Err(TransmissionSolveError::Residual {
                residual,
                tolerance: residual_tolerance,
            });
        }
        solutions.push(TransmissionAdjointSolution {
            values,
            interface_dofs,
            relative_residual: residual,
        });
    }
    Ok(solutions)
}

pub(crate) fn evaluate_exterior_adjoint_pressure_values_checked(
    values: &[Complex64],
    system: &TransmissionTbemSystem,
    eval_points: &[[f64; 3]],
) -> Result<Vec<Complex64>, TransmissionSolveError> {
    let expected = system.total_dofs();
    if values.len() != expected {
        return Err(TransmissionSolveError::RhsLength {
            expected,
            actual: values.len(),
        });
    }
    if values.iter().any(|value| !complex_is_finite(*value)) {
        return Err(TransmissionSolveError::NonFiniteSolution {
            quantity: "adjoint trace",
        });
    }
    let interface_dofs = system.interface_dofs();
    let (elements, nodes) = system.field_geometry();
    let primal_pressure = map_dual_basis_coefficients_checked(
        &values[..interface_dofs],
        elements,
        nodes,
        &system.exterior,
        system.basis_order(),
    )
    .map_err(|error| TransmissionSolveError::FieldEvaluation(error.to_string()))?;
    let primal_velocity = map_dual_basis_coefficients_checked(
        &values[interface_dofs..],
        elements,
        nodes,
        &system.exterior,
        system.basis_order(),
    )
    .map_err(|error| TransmissionSolveError::FieldEvaluation(error.to_string()))?;
    compute_scattered_field_checked(
        eval_points,
        elements,
        nodes,
        &primal_pressure,
        Some(&primal_velocity),
        &system.exterior,
        system.basis_order(),
    )
    .map_err(|error| TransmissionSolveError::FieldEvaluation(error.to_string()))
}

pub(crate) fn evaluate_exterior_field_values_checked(
    values: &[Complex64],
    system: &TransmissionTbemSystem,
    eval_points: &[[f64; 3]],
    incident: &IncidentField,
) -> Result<Vec<FieldPoint>, TransmissionSolveError> {
    let expected = system.total_dofs();
    if values.len() != expected {
        return Err(TransmissionSolveError::RhsLength {
            expected,
            actual: values.len(),
        });
    }
    if values.iter().any(|value| !complex_is_finite(*value)) {
        return Err(TransmissionSolveError::NonFiniteSolution { quantity: "trace" });
    }
    let pressure = incident.evaluate_pressure(system.panel_centers(), &system.exterior);
    let normal_derivative = incident.evaluate_normal_derivative(
        system.panel_centers(),
        system.panel_normals(),
        &system.exterior,
    );
    let i_omega_rho = Complex64::new(0.0, system.exterior.pressure_factor);
    let incident_velocity = normal_derivative
        .into_iter()
        .map(|value| {
            if i_omega_rho.norm() > 1.0e-30 {
                value / i_omega_rho
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect::<Vec<_>>();
    let interface_dofs = system.interface_dofs();
    let scattered_pressure = values[..interface_dofs]
        .iter()
        .zip(&pressure)
        .map(|(&total, &incident)| total - incident)
        .collect::<Vec<_>>();
    let scattered_velocity = values[interface_dofs..]
        .iter()
        .zip(&incident_velocity)
        .map(|(&total, &incident)| total - incident)
        .collect::<Vec<_>>();
    if scattered_pressure
        .iter()
        .chain(scattered_velocity.iter())
        .any(|value| !complex_is_finite(*value))
    {
        return Err(TransmissionSolveError::NonFiniteSolution {
            quantity: "scattered trace",
        });
    }
    let (elements, nodes) = system.field_geometry();
    compute_total_field_checked(
        eval_points,
        elements,
        nodes,
        &scattered_pressure,
        Some(&scattered_velocity),
        incident,
        &system.exterior,
        system.basis_order(),
    )
    .map_err(|error| TransmissionSolveError::FieldEvaluation(error.to_string()))
}

fn hermitian_transpose(matrix: &DenseMatrix<Complex64>) -> DenseMatrix<Complex64> {
    let mut result = DenseMatrix::zeros(matrix.nrows(), matrix.ncols());
    for row in 0..matrix.nrows() {
        for column in 0..matrix.ncols() {
            result[[row, column]] = matrix[[column, row]].conj();
        }
    }
    result
}

/// Build an exterior incident-field RHS and solve the checked transmission
/// system in one CPU path.
pub fn solve_incident_transmission(
    system: &TransmissionTbemSystem,
    incident: &IncidentField,
    residual_tolerance: f64,
) -> Result<TransmissionSolution, TransmissionSolveError> {
    let rhs = system
        .incident_rhs(incident)
        .map_err(|error: TransmissionRhsError| {
            TransmissionSolveError::IncidentRhs(error.to_string())
        })?;
    solve_transmission_system(system, &rhs, residual_tolerance)
}

fn relative_residual(
    matrix: &DenseMatrix<Complex64>,
    solution: &[Complex64],
    rhs: &[Complex64],
) -> f64 {
    let mut residual = vec![Complex64::default(); rhs.len()];
    for row in 0..rhs.len() {
        let mut action = Complex64::default();
        for col in 0..solution.len() {
            action += matrix[[row, col]] * solution[col];
        }
        residual[row] = action - rhs[row];
    }
    vector_norm(&residual) / vector_norm(rhs).max(DEFAULT_MIN_SCALE)
}

fn vector_norm(values: &[Complex64]) -> f64 {
    values
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}
