//! Full-state discrete adjoint engine for resolved PC-fan flow linearizations.
//!
//! A resolved Navier--Stokes caller must provide the exact converged-state
//! tangent and design residual derivatives from its assembly. The engine solves
//! `Aᵀ λ = φ_x` with sparse restarted-free GMRES and returns
//! `φ_d - R_dᵀ λ`. It deliberately does not infer a full-state adjoint from
//! finite differences of a black-box CFD solve.

use crate::model::{PcFanCfdBackend, PcFanCfdExecutionMode, PcFanFlowModel};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Sparse square state Jacobian in CSR form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedFlowSparseMatrix {
    pub dimension: usize,
    pub row_ptrs: Vec<usize>,
    pub column_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl PcFanResolvedFlowSparseMatrix {
    /// Construct and validate a CSR state Jacobian.
    pub fn new(
        dimension: usize,
        row_ptrs: Vec<usize>,
        column_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self, PcFanResolvedAdjointError> {
        let matrix = Self {
            dimension,
            row_ptrs,
            column_indices,
            values,
        };
        matrix.validate()?;
        Ok(matrix)
    }

    /// Build a CSR matrix from a dense reference matrix for small regression
    /// fixtures and assembly-level tests.
    pub fn from_dense(dense: &[Vec<f64>]) -> Result<Self, PcFanResolvedAdjointError> {
        let dimension = dense.len();
        if dense.iter().any(|row| row.len() != dimension) {
            return Err(PcFanResolvedAdjointError::InvalidMatrix(
                "dense matrix must be square",
            ));
        }
        let mut row_ptrs = Vec::with_capacity(dimension + 1);
        let mut column_indices = Vec::new();
        let mut values = Vec::new();
        row_ptrs.push(0);
        for row in dense {
            for (column, &value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(PcFanResolvedAdjointError::NonFiniteMatrixValue);
                }
                if value != 0.0 {
                    column_indices.push(column);
                    values.push(value);
                }
            }
            row_ptrs.push(values.len());
        }
        Self::new(dimension, row_ptrs, column_indices, values)
    }

    fn validate(&self) -> Result<(), PcFanResolvedAdjointError> {
        if self.row_ptrs.len() != self.dimension + 1
            || self.column_indices.len() != self.values.len()
            || self.row_ptrs.first().copied() != Some(0)
            || self.row_ptrs.last().copied() != Some(self.values.len())
        {
            return Err(PcFanResolvedAdjointError::InvalidMatrix(
                "CSR dimensions or row pointers are inconsistent",
            ));
        }
        if self.row_ptrs.windows(2).any(|window| window[0] > window[1])
            || self
                .column_indices
                .iter()
                .any(|&column| column >= self.dimension)
            || self.values.iter().any(|value| !value.is_finite())
        {
            return Err(PcFanResolvedAdjointError::InvalidMatrix(
                "CSR entries are not monotone, in range, or finite",
            ));
        }
        Ok(())
    }

    fn apply_transpose(&self, input: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; self.dimension];
        for (row, row_ptrs) in self.row_ptrs.windows(2).enumerate() {
            for entry in row_ptrs[0]..row_ptrs[1] {
                output[self.column_indices[entry]] += self.values[entry] * input[row];
            }
        }
        output
    }
}

/// The production CFD solver's published Picard/Oseen linearization.
///
/// This is deliberately a separate type from [`PcFanResolvedFlowLinearization`].
/// An Oseen matrix is useful evidence about the last fixed-point solve, but it
/// does not contain the full Newton convection tangent or design residual
/// derivatives required by a resolved-flow adjoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedFlowPicardLinearization {
    pub flow_model: PcFanFlowModel,
    pub backend: PcFanCfdBackend,
    pub provenance: String,
    pub state_jacobian: PcFanResolvedFlowSparseMatrix,
    /// State at which the one-step production Picard map was evaluated.
    pub state: Vec<f64>,
    pub fixed_point_residual: Vec<f64>,
}

impl PcFanResolvedFlowPicardLinearization {
    /// Validate the published Picard/Oseen observation without promoting it
    /// to the exact adjoint contract.
    pub fn validate(&self) -> Result<(), PcFanResolvedAdjointError> {
        if !self.flow_model.is_cfd() {
            return Err(PcFanResolvedAdjointError::UnsupportedFlowModel(
                self.flow_model,
            ));
        }
        if self.provenance.trim().is_empty() {
            return Err(PcFanResolvedAdjointError::MissingProvenance);
        }
        self.state_jacobian.validate()?;
        if self.state.len() != self.state_jacobian.dimension
            || self.fixed_point_residual.len() != self.state_jacobian.dimension
        {
            return Err(PcFanResolvedAdjointError::DimensionMismatch(
                "Picard linearization state",
            ));
        }
        if !self
            .state
            .iter()
            .chain(&self.fixed_point_residual)
            .all(|value| value.is_finite())
        {
            return Err(PcFanResolvedAdjointError::NonFiniteInput(
                "Picard linearization state",
            ));
        }
        Ok(())
    }

    /// Convert the FEM solver's boundary-conditioned sparse Oseen system.
    pub fn from_fem(
        flow_model: PcFanFlowModel,
        backend: PcFanCfdBackend,
        provenance: impl Into<String>,
        source: &sonium_fem::physics::fluid::stokes::navier_stokes::NavierStokesPicardLinearization,
    ) -> Result<Self, PcFanResolvedAdjointError> {
        let state_jacobian = PcFanResolvedFlowSparseMatrix::new(
            source.state_jacobian.num_rows,
            source.state_jacobian.row_ptrs.clone(),
            source.state_jacobian.col_indices.clone(),
            source.state_jacobian.values.clone(),
        )?;
        let converted = Self {
            flow_model,
            backend,
            provenance: provenance.into(),
            state_jacobian,
            state: source.state.clone(),
            fixed_point_residual: source.fixed_point_residual.clone(),
        };
        converted.validate()?;
        Ok(converted)
    }
}

/// Analytic state tangent published by the production FEM CFD map.
///
/// This is deliberately separate from [`PcFanResolvedFlowLinearization`]: a
/// state tangent alone is not enough for an exact design adjoint.  The latter
/// additionally requires objective state/design derivatives and the complete
/// design residual Jacobian.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedFlowNewtonLinearization {
    pub flow_model: PcFanFlowModel,
    pub backend: PcFanCfdBackend,
    pub provenance: String,
    pub state_jacobian: PcFanResolvedFlowSparseMatrix,
    pub state_residual: Vec<f64>,
    pub state: Vec<f64>,
    /// Production frozen-mesh design residual derivative, when the CFD
    /// assembly can provide all active geometry terms analytically.
    #[serde(default)]
    pub residual_design_jacobian: Option<Vec<Vec<f64>>>,
}

impl PcFanResolvedFlowNewtonLinearization {
    pub fn validate(&self) -> Result<(), PcFanResolvedAdjointError> {
        if !self.flow_model.is_cfd() {
            return Err(PcFanResolvedAdjointError::UnsupportedFlowModel(
                self.flow_model,
            ));
        }
        if self.provenance.trim().is_empty() {
            return Err(PcFanResolvedAdjointError::MissingProvenance);
        }
        let provenance = self.provenance.to_ascii_lowercase();
        if [
            "picard",
            "oseen",
            "derivative-probe",
            "finite-difference",
            "black-box",
        ]
        .iter()
        .any(|marker| provenance.contains(marker))
        {
            return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
        }
        self.state_jacobian.validate()?;
        if self.state.len() != self.state_jacobian.dimension
            || self.state_residual.len() != self.state_jacobian.dimension
        {
            return Err(PcFanResolvedAdjointError::DimensionMismatch(
                "Newton linearization state",
            ));
        }
        if !self
            .state
            .iter()
            .chain(&self.state_residual)
            .all(|value| value.is_finite())
        {
            return Err(PcFanResolvedAdjointError::NonFiniteInput(
                "Newton linearization state",
            ));
        }
        if let Some(residual_design_jacobian) = &self.residual_design_jacobian {
            if residual_design_jacobian.len() != self.state_jacobian.dimension {
                return Err(PcFanResolvedAdjointError::DimensionMismatch(
                    "Newton residual design Jacobian",
                ));
            }
            let design_dimension = residual_design_jacobian.first().map_or(0, Vec::len);
            if residual_design_jacobian.iter().any(|row| {
                row.len() != design_dimension || row.iter().any(|value| !value.is_finite())
            }) {
                return Err(PcFanResolvedAdjointError::DimensionMismatch(
                    "Newton residual design Jacobian",
                ));
            }
        }
        Ok(())
    }

    pub fn from_fem(
        flow_model: PcFanFlowModel,
        backend: PcFanCfdBackend,
        provenance: impl Into<String>,
        source: &sonium_fem::physics::fluid::stokes::navier_stokes::NavierStokesNewtonLinearization,
    ) -> Result<Self, PcFanResolvedAdjointError> {
        let converted = Self {
            flow_model,
            backend,
            provenance: provenance.into(),
            state_jacobian: PcFanResolvedFlowSparseMatrix::new(
                source.state_jacobian.num_rows,
                source.state_jacobian.row_ptrs.clone(),
                source.state_jacobian.col_indices.clone(),
                source.state_jacobian.values.clone(),
            )?,
            state_residual: source.state_residual.clone(),
            state: source.state.clone(),
            residual_design_jacobian: None,
        };
        converted.validate()?;
        Ok(converted)
    }
}

/// Exact design/objective derivative data for one resolved-flow state.
///
/// The values must come from an analytic production assembly.  This bundle is
/// deliberately separate from the diagnostic derivative-probe output so a
/// centered finite-difference estimate cannot be promoted accidentally.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedFlowDerivativeData {
    pub provenance: String,
    pub residual_design_jacobian: Vec<Vec<f64>>,
    pub objective_state_gradient: Vec<f64>,
    pub objective_design_gradient: Vec<f64>,
}

/// Exact one-way aeroacoustic output-map derivatives for a resolved flow.
///
/// The acoustic solve or source model is represented as an explicit output
/// map `a(x, d)`, where `x` is the converged resolved-flow state and `d` is
/// the design vector.  Rows in the two Jacobians correspond to the retained
/// acoustic state (for example harmonic pressure amplitudes).  This bundle
/// only carries the chain-rule data into the flow adjoint; it does not infer
/// acoustic derivatives from a black-box solve, differentiate a remesher, or
/// turn diagnostic dB floors into physical state sensitivities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedAeroacousticDerivativeData {
    pub provenance: String,
    /// `∂a/∂x`, with one row per acoustic state and one column per flow DOF.
    pub acoustic_state_jacobian: Vec<Vec<f64>>,
    /// `∂a/∂d`, with one row per acoustic state and one column per design DOF.
    pub acoustic_design_jacobian: Vec<Vec<f64>>,
    /// Direct objective derivative `∂J/∂a`.
    pub objective_acoustic_state_gradient: Vec<f64>,
    /// Direct objective derivative `∂J/∂d` at fixed acoustic state.
    pub objective_acoustic_design_gradient: Vec<f64>,
}

/// Checked implicit aeroacoustic PDE/source derivative data.
///
/// The acoustic state `a` is retained as an implicit solve with residual
/// `S(x, a, d) = 0`. The supplied rows are `S_a`, `S_x`, and `S_d`; they are
/// deliberately separate from the one-way output-map contract above so an
/// acoustic PDE cannot be admitted by pretending it is an explicit map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedAeroacousticPdeDerivativeData {
    pub provenance: String,
    /// Acoustic residual Jacobian `∂S/∂a`.
    pub acoustic_state_jacobian: PcFanResolvedFlowSparseMatrix,
    /// Rectangular acoustic residual Jacobian `∂S/∂x`.
    pub acoustic_flow_jacobian: Vec<Vec<f64>>,
    /// Acoustic residual design Jacobian `∂S/∂d`.
    pub acoustic_design_jacobian: Vec<Vec<f64>>,
    /// Acoustic residual evaluated at the retained state.
    pub acoustic_residual: Vec<f64>,
    /// Direct objective derivative `∂J/∂a`.
    pub objective_acoustic_state_gradient: Vec<f64>,
    /// Direct objective derivative `∂J/∂d` at fixed flow/acoustic states.
    pub objective_acoustic_design_gradient: Vec<f64>,
}

impl PcFanResolvedFlowDerivativeData {
    /// Build the derivative bundle from a production Newton observation and
    /// caller-supplied objective derivatives. No numerical probe or black-box
    /// fallback is inserted here.
    pub fn from_production_newton(
        source: &PcFanResolvedFlowNewtonLinearization,
        objective_provenance: impl Into<String>,
        objective_state_gradient: Vec<f64>,
        objective_design_gradient: Vec<f64>,
    ) -> Result<Self, PcFanResolvedAdjointError> {
        source.validate()?;
        let residual_design_jacobian = source
            .residual_design_jacobian
            .clone()
            .ok_or(PcFanResolvedAdjointError::MissingProductionDesignResidualJacobian)?;
        let state_dimension = source.state_jacobian.dimension;
        let design_dimension = residual_design_jacobian.first().map_or(0, Vec::len);
        if objective_state_gradient.len() != state_dimension
            || objective_design_gradient.len() != design_dimension
        {
            return Err(PcFanResolvedAdjointError::DimensionMismatch(
                "production objective derivatives",
            ));
        }
        if objective_state_gradient
            .iter()
            .chain(&objective_design_gradient)
            .any(|value| !value.is_finite())
        {
            return Err(PcFanResolvedAdjointError::NonFiniteInput(
                "production objective derivatives",
            ));
        }
        let objective_provenance = objective_provenance.into();
        if objective_provenance.trim().is_empty() {
            return Err(PcFanResolvedAdjointError::MissingProvenance);
        }
        let objective_provenance_lower = objective_provenance.to_ascii_lowercase();
        if [
            "picard",
            "oseen",
            "derivative-probe",
            "finite-difference",
            "black-box",
        ]
        .iter()
        .any(|marker| objective_provenance_lower.contains(marker))
        {
            return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
        }
        Ok(Self {
            provenance: format!(
                "production:pcfan:analytic-design-residual:v1; objective:{objective_provenance}"
            ),
            residual_design_jacobian,
            objective_state_gradient,
            objective_design_gradient,
        })
    }

    /// Compose analytic flow and one-way aeroacoustic objective derivatives.
    ///
    /// For `a = a(x, d)`, this forms
    ///
    /// `J_x = J_x^flow + (∂a/∂x)ᵀ J_a`
    ///
    /// and
    ///
    /// `J_d = J_d^flow + J_d^acoustic + (∂a/∂d)ᵀ J_a`.
    ///
    /// The returned bundle remains subject to the normal exact Newton
    /// adjoint admission checks.  In particular, every supplied derivative
    /// must be finite and the acoustic provenance must identify an analytic
    /// production path.
    pub fn from_production_newton_with_aeroacoustic(
        source: &PcFanResolvedFlowNewtonLinearization,
        objective_provenance: impl Into<String>,
        flow_objective_state_gradient: Vec<f64>,
        flow_objective_design_gradient: Vec<f64>,
        aeroacoustic: PcFanResolvedAeroacousticDerivativeData,
    ) -> Result<Self, PcFanResolvedAdjointError> {
        source.validate()?;
        let residual_design_jacobian = source
            .residual_design_jacobian
            .as_ref()
            .ok_or(PcFanResolvedAdjointError::MissingProductionDesignResidualJacobian)?;
        let state_dimension = source.state_jacobian.dimension;
        let design_dimension = residual_design_jacobian.first().map_or(0, Vec::len);

        if flow_objective_state_gradient.len() != state_dimension
            || flow_objective_design_gradient.len() != design_dimension
        {
            return Err(PcFanResolvedAdjointError::DimensionMismatch(
                "aeroacoustic flow objective derivatives",
            ));
        }

        let provenance = aeroacoustic.provenance.trim();
        if provenance.is_empty() {
            return Err(PcFanResolvedAdjointError::MissingProvenance);
        }
        let provenance_lower = provenance.to_ascii_lowercase();
        if [
            "picard",
            "oseen",
            "derivative-probe",
            "finite-difference",
            "black-box",
        ]
        .iter()
        .any(|marker| provenance_lower.contains(marker))
        {
            return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
        }

        let acoustic_dimension = aeroacoustic.objective_acoustic_state_gradient.len();
        if aeroacoustic.acoustic_state_jacobian.len() != acoustic_dimension
            || aeroacoustic.acoustic_design_jacobian.len() != acoustic_dimension
            || aeroacoustic.objective_acoustic_design_gradient.len() != design_dimension
            || aeroacoustic
                .acoustic_state_jacobian
                .iter()
                .any(|row| row.len() != state_dimension)
            || aeroacoustic
                .acoustic_design_jacobian
                .iter()
                .any(|row| row.len() != design_dimension)
        {
            return Err(PcFanResolvedAdjointError::DimensionMismatch(
                "aeroacoustic derivative dimensions",
            ));
        }
        if aeroacoustic
            .acoustic_state_jacobian
            .iter()
            .chain(&aeroacoustic.acoustic_design_jacobian)
            .flatten()
            .chain(&aeroacoustic.objective_acoustic_state_gradient)
            .chain(&aeroacoustic.objective_acoustic_design_gradient)
            .any(|value| !value.is_finite())
        {
            return Err(PcFanResolvedAdjointError::NonFiniteInput(
                "aeroacoustic derivatives",
            ));
        }

        let mut objective_state_gradient = flow_objective_state_gradient;
        let mut objective_design_gradient = flow_objective_design_gradient;
        for acoustic_index in 0..acoustic_dimension {
            let multiplier = aeroacoustic.objective_acoustic_state_gradient[acoustic_index];
            for (state_value, &derivative) in objective_state_gradient
                .iter_mut()
                .zip(&aeroacoustic.acoustic_state_jacobian[acoustic_index])
            {
                *state_value += multiplier * derivative;
            }
            for (design_value, &derivative) in objective_design_gradient
                .iter_mut()
                .zip(&aeroacoustic.acoustic_design_jacobian[acoustic_index])
            {
                *design_value += multiplier * derivative;
            }
        }
        for (design_value, &direct) in objective_design_gradient
            .iter_mut()
            .zip(&aeroacoustic.objective_acoustic_design_gradient)
        {
            *design_value += direct;
        }
        if !objective_state_gradient
            .iter()
            .chain(&objective_design_gradient)
            .all(|value| value.is_finite())
        {
            return Err(PcFanResolvedAdjointError::NonFiniteInput(
                "composed aeroacoustic objective derivatives",
            ));
        }

        let objective_provenance = objective_provenance.into();
        Self::from_production_newton(
            source,
            format!("{objective_provenance}; aeroacoustic:{provenance}"),
            objective_state_gradient,
            objective_design_gradient,
        )
    }
}

/// Exact linearization and objective derivatives for one resolved-flow state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedFlowLinearization {
    pub flow_model: PcFanFlowModel,
    pub backend: PcFanCfdBackend,
    pub provenance: String,
    pub state_jacobian: PcFanResolvedFlowSparseMatrix,
    pub state_residual: Vec<f64>,
    pub residual_design_jacobian: Vec<Vec<f64>>,
    pub objective_state_gradient: Vec<f64>,
    pub objective_design_gradient: Vec<f64>,
}

impl PcFanResolvedFlowLinearization {
    /// Combine the production FEM Newton tangent with an analytic derivative
    /// bundle for the design residual and objective.
    ///
    /// This is the only checked handoff from the state-only Newton artifact to
    /// the exact resolved-flow adjoint.  The derivative bundle is required to
    /// carry its own provenance and finite-difference, Picard/Oseen, and
    /// black-box labels are rejected by the same admission boundary used by
    /// [`solve_pc_fan_resolved_flow_adjoint`].
    pub fn from_newton(
        source: &PcFanResolvedFlowNewtonLinearization,
        derivatives: PcFanResolvedFlowDerivativeData,
    ) -> Result<Self, PcFanResolvedAdjointError> {
        source.validate()?;
        if derivatives.provenance.trim().is_empty() {
            return Err(PcFanResolvedAdjointError::MissingProvenance);
        }
        let provenance = format!(
            "{}; derivatives:{}",
            source.provenance, derivatives.provenance
        );
        let converted = Self {
            flow_model: source.flow_model,
            backend: source.backend,
            provenance,
            state_jacobian: source.state_jacobian.clone(),
            state_residual: source.state_residual.clone(),
            residual_design_jacobian: derivatives.residual_design_jacobian,
            objective_state_gradient: derivatives.objective_state_gradient,
            objective_design_gradient: derivatives.objective_design_gradient,
        };
        validate_linearization(&converted)?;
        Ok(converted)
    }
}

/// Numerical controls for the sparse transpose solve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedAdjointConfig {
    pub tolerance: f64,
    pub max_iterations: usize,
    pub max_state_residual: f64,
}

impl Default for PcFanResolvedAdjointConfig {
    fn default() -> Self {
        Self {
            tolerance: 1.0e-10,
            max_iterations: 512,
            max_state_residual: 1.0e-6,
        }
    }
}

/// Fail-closed errors for resolved-flow adjoint admission or solution.
#[derive(Debug, Clone, PartialEq)]
pub enum PcFanResolvedAdjointError {
    InvalidMatrix(&'static str),
    NonFiniteMatrixValue,
    UnsupportedFlowModel(PcFanFlowModel),
    MissingProvenance,
    InvalidConfig(&'static str, f64),
    InvalidIterationLimit,
    DimensionMismatch(&'static str),
    NonFiniteInput(&'static str),
    StateResidualTooLarge {
        norm: f64,
        maximum: f64,
    },
    /// A Picard/Oseen observation cannot satisfy an exact full-state adjoint
    /// contract.
    OseenLinearizationNotAdmissible,
    /// A black-box derivative probe is diagnostic evidence, not an exact
    /// Newton tangent admissible to the resolved-flow adjoint.
    DerivativeProbeNotAdmissible,
    /// The production FEM map did not publish the observation required by a
    /// checked resolved-flow derivative probe.
    MissingProductionPicardLinearization,
    /// The production FEM map did not publish a complete laminar Newton
    /// tangent for the prescribed state.
    MissingProductionNewtonLinearization,
    /// The production FEM map did not publish the complete frozen-mesh design
    /// residual derivative required by the exact adjoint handoff.
    MissingProductionDesignResidualJacobian,
    /// The caller supplied a production CFD result that did not converge.
    ProductionStateNotConverged,
    /// The caller supplied inconsistent backend or execution provenance.
    BackendProvenanceMismatch,
    /// The production solve used a fallback or residency mode not admitted by
    /// the checked CFD configuration.
    BackendExecutionNotAdmitted {
        requested: PcFanCfdBackend,
        actual: PcFanCfdBackend,
        mode: PcFanCfdExecutionMode,
    },
    /// A derivative probe callback failed to provide a valid observation.
    ProbeEvaluationFailed(&'static str),
    /// A derivative probe exceeded its explicit Taylor consistency budget.
    TaylorTestFailed {
        quantity: &'static str,
        error: f64,
        tolerance: f64,
    },
    DidNotConverge {
        iterations: usize,
        residual: f64,
        tolerance: f64,
    },
    SingularAdjointSystem {
        index: usize,
    },
}

impl fmt::Display for PcFanResolvedAdjointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMatrix(message) => {
                write!(formatter, "invalid resolved-flow matrix: {message}")
            }
            Self::NonFiniteMatrixValue => write!(
                formatter,
                "resolved-flow matrix contains a non-finite value"
            ),
            Self::UnsupportedFlowModel(model) => write!(
                formatter,
                "full-state adjoint requires a resolved CFD flow model, got {model:?}"
            ),
            Self::MissingProvenance => {
                write!(formatter, "resolved-flow adjoint provenance is blank")
            }
            Self::InvalidConfig(name, value) => {
                write!(formatter, "invalid resolved-adjoint {name}={value}")
            }
            Self::InvalidIterationLimit => {
                write!(formatter, "resolved-adjoint iteration limit is zero")
            }
            Self::DimensionMismatch(name) => {
                write!(formatter, "resolved-adjoint dimension mismatch in {name}")
            }
            Self::NonFiniteInput(name) => {
                write!(formatter, "resolved-adjoint input {name} is non-finite")
            }
            Self::StateResidualTooLarge { norm, maximum } => write!(
                formatter,
                "resolved-flow state residual norm {norm} exceeds admission limit {maximum}"
            ),
            Self::OseenLinearizationNotAdmissible => write!(
                formatter,
                "Picard/Oseen linearization is not admissible for the resolved-flow adjoint"
            ),
            Self::DerivativeProbeNotAdmissible => write!(
                formatter,
                "finite-difference derivative probes are not admissible for the resolved-flow adjoint"
            ),
            Self::MissingProductionPicardLinearization => write!(
                formatter,
                "production FEM Picard solve did not publish a linearization"
            ),
            Self::MissingProductionNewtonLinearization => write!(
                formatter,
                "production FEM solve did not publish a complete Newton linearization"
            ),
            Self::MissingProductionDesignResidualJacobian => write!(
                formatter,
                "production FEM solve did not publish a design residual Jacobian"
            ),
            Self::ProductionStateNotConverged => {
                write!(formatter, "production CFD state is not converged")
            }
            Self::BackendProvenanceMismatch => {
                write!(
                    formatter,
                    "production CFD backend provenance is inconsistent"
                )
            }
            Self::BackendExecutionNotAdmitted {
                requested,
                actual,
                mode,
            } => write!(
                formatter,
                "production CFD backend request {requested:?} resolved to {actual:?} in {mode:?}, which is not admitted"
            ),
            Self::ProbeEvaluationFailed(quantity) => {
                write!(
                    formatter,
                    "resolved-flow derivative probe failed for {quantity}"
                )
            }
            Self::TaylorTestFailed {
                quantity,
                error,
                tolerance,
            } => write!(
                formatter,
                "resolved-flow derivative Taylor error for {quantity} is {error}, above {tolerance}"
            ),
            Self::DidNotConverge {
                iterations,
                residual,
                tolerance,
            } => write!(
                formatter,
                "resolved adjoint did not converge after {iterations} iterations: residual {residual}, tolerance {tolerance}"
            ),
            Self::SingularAdjointSystem { index } => {
                write!(
                    formatter,
                    "singular resolved adjoint Hessenberg system at index {index}"
                )
            }
        }
    }
}

impl std::error::Error for PcFanResolvedAdjointError {}

/// Solved full-state adjoint and design gradient.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedAdjointReport {
    pub flow_model: PcFanFlowModel,
    pub backend: PcFanCfdBackend,
    pub provenance: String,
    pub adjoint: Vec<f64>,
    pub design_gradient: Vec<f64>,
    pub state_residual_norm: f64,
    pub adjoint_residual_norm: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Solve the exact resolved-flow transpose system and contract the gradient.
pub fn solve_pc_fan_resolved_flow_adjoint(
    config: &PcFanResolvedAdjointConfig,
    linearization: &PcFanResolvedFlowLinearization,
) -> Result<PcFanResolvedAdjointReport, PcFanResolvedAdjointError> {
    validate_config(config)?;
    validate_linearization(linearization)?;

    let state_residual_norm = l2_norm(&linearization.state_residual);
    if state_residual_norm > config.max_state_residual {
        return Err(PcFanResolvedAdjointError::StateResidualTooLarge {
            norm: state_residual_norm,
            maximum: config.max_state_residual,
        });
    }

    let (adjoint, iterations, adjoint_residual_norm) = solve_transpose_gmres(
        &linearization.state_jacobian,
        &linearization.objective_state_gradient,
        config.tolerance,
        config.max_iterations,
    )?;
    let design_gradient = linearization
        .objective_design_gradient
        .iter()
        .enumerate()
        .map(|(design_index, &objective_derivative)| {
            objective_derivative
                - linearization
                    .residual_design_jacobian
                    .iter()
                    .zip(&adjoint)
                    .map(|(row, &lambda)| row[design_index] * lambda)
                    .sum::<f64>()
        })
        .collect();

    Ok(PcFanResolvedAdjointReport {
        flow_model: linearization.flow_model,
        backend: linearization.backend,
        provenance: linearization.provenance.clone(),
        adjoint,
        design_gradient,
        state_residual_norm,
        adjoint_residual_norm,
        iterations,
        converged: true,
    })
}

/// Solve the exact resolved-flow adjoint directly from the production Newton
/// tangent and an analytic design/objective derivative bundle.
pub fn solve_pc_fan_resolved_flow_adjoint_from_newton(
    config: &PcFanResolvedAdjointConfig,
    newton: &PcFanResolvedFlowNewtonLinearization,
    derivatives: PcFanResolvedFlowDerivativeData,
) -> Result<PcFanResolvedAdjointReport, PcFanResolvedAdjointError> {
    let linearization = PcFanResolvedFlowLinearization::from_newton(newton, derivatives)?;
    solve_pc_fan_resolved_flow_adjoint(config, &linearization)
}

/// Exact coupled resolved-flow/aeroacoustic implicit adjoint report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedAeroacousticAdjointReport {
    pub flow_model: PcFanFlowModel,
    pub backend: PcFanCfdBackend,
    pub provenance: String,
    pub flow_adjoint: Vec<f64>,
    pub acoustic_adjoint: Vec<f64>,
    pub design_gradient: Vec<f64>,
    pub flow_state_residual_norm: f64,
    pub acoustic_state_residual_norm: f64,
    pub flow_adjoint_residual_norm: f64,
    pub acoustic_adjoint_residual_norm: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Solve the exact coupled implicit acoustic/flow transpose systems.
///
/// For `R(x, d) = 0`, `S(x, a, d) = 0`, and objective `J`, this computes
/// `S_aᵀ λ_a = J_a`, `R_xᵀ λ_x = J_x - S_xᵀ λ_a`, then returns
/// `J_d - R_dᵀ λ_x - S_dᵀ λ_a`. All derivative rows must come from the same
/// converged analytic assembly; finite-difference and black-box provenance is
/// rejected before either transpose solve.
pub fn solve_pc_fan_resolved_aeroacoustic_adjoint(
    config: &PcFanResolvedAdjointConfig,
    flow: &PcFanResolvedFlowLinearization,
    acoustic: &PcFanResolvedAeroacousticPdeDerivativeData,
) -> Result<PcFanResolvedAeroacousticAdjointReport, PcFanResolvedAdjointError> {
    validate_config(config)?;
    validate_linearization(flow)?;
    validate_aeroacoustic_pde_derivatives(flow, acoustic)?;

    let flow_state_residual_norm = l2_norm(&flow.state_residual);
    if flow_state_residual_norm > config.max_state_residual {
        return Err(PcFanResolvedAdjointError::StateResidualTooLarge {
            norm: flow_state_residual_norm,
            maximum: config.max_state_residual,
        });
    }
    let acoustic_state_residual_norm = l2_norm(&acoustic.acoustic_residual);
    if acoustic_state_residual_norm > config.max_state_residual {
        return Err(PcFanResolvedAdjointError::StateResidualTooLarge {
            norm: acoustic_state_residual_norm,
            maximum: config.max_state_residual,
        });
    }

    let (acoustic_adjoint, acoustic_iterations, acoustic_adjoint_residual_norm) =
        solve_transpose_gmres(
            &acoustic.acoustic_state_jacobian,
            &acoustic.objective_acoustic_state_gradient,
            config.tolerance,
            config.max_iterations,
        )?;

    let mut flow_rhs = flow.objective_state_gradient.clone();
    for (row, &acoustic_multiplier) in acoustic
        .acoustic_flow_jacobian
        .iter()
        .zip(&acoustic_adjoint)
    {
        for (value, &derivative) in flow_rhs.iter_mut().zip(row) {
            *value -= derivative * acoustic_multiplier;
        }
    }
    if flow_rhs.iter().any(|value| !value.is_finite()) {
        return Err(PcFanResolvedAdjointError::NonFiniteInput(
            "coupled aeroacoustic flow adjoint right-hand side",
        ));
    }

    let (flow_adjoint, flow_iterations, flow_adjoint_residual_norm) = solve_transpose_gmres(
        &flow.state_jacobian,
        &flow_rhs,
        config.tolerance,
        config.max_iterations,
    )?;

    let design_dimension = flow.objective_design_gradient.len();
    let mut design_gradient = flow.objective_design_gradient.clone();
    for (value, &direct) in design_gradient
        .iter_mut()
        .zip(&acoustic.objective_acoustic_design_gradient)
    {
        *value += direct;
    }
    for (row, &flow_multiplier) in flow.residual_design_jacobian.iter().zip(&flow_adjoint) {
        for (value, &derivative) in design_gradient.iter_mut().zip(row) {
            *value -= derivative * flow_multiplier;
        }
    }
    for (row, &acoustic_multiplier) in acoustic
        .acoustic_design_jacobian
        .iter()
        .zip(&acoustic_adjoint)
    {
        for (value, &derivative) in design_gradient.iter_mut().zip(row) {
            *value -= derivative * acoustic_multiplier;
        }
    }
    debug_assert_eq!(design_gradient.len(), design_dimension);
    if design_gradient.iter().any(|value| !value.is_finite()) {
        return Err(PcFanResolvedAdjointError::NonFiniteInput(
            "coupled aeroacoustic design gradient",
        ));
    }

    Ok(PcFanResolvedAeroacousticAdjointReport {
        flow_model: flow.flow_model,
        backend: flow.backend,
        provenance: format!(
            "{}; aeroacoustic-pde:{}",
            flow.provenance, acoustic.provenance
        ),
        flow_adjoint,
        acoustic_adjoint,
        design_gradient,
        flow_state_residual_norm,
        acoustic_state_residual_norm,
        flow_adjoint_residual_norm,
        acoustic_adjoint_residual_norm,
        iterations: flow_iterations.saturating_add(acoustic_iterations),
        converged: true,
    })
}

fn validate_aeroacoustic_pde_derivatives(
    flow: &PcFanResolvedFlowLinearization,
    acoustic: &PcFanResolvedAeroacousticPdeDerivativeData,
) -> Result<(), PcFanResolvedAdjointError> {
    let provenance = acoustic.provenance.trim();
    if provenance.is_empty() {
        return Err(PcFanResolvedAdjointError::MissingProvenance);
    }
    let provenance_lower = provenance.to_ascii_lowercase();
    if ["derivative-probe", "finite-difference", "black-box"]
        .iter()
        .any(|marker| provenance_lower.contains(marker))
    {
        return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
    }

    acoustic.acoustic_state_jacobian.validate()?;
    let acoustic_dimension = acoustic.acoustic_state_jacobian.dimension;
    let flow_dimension = flow.state_jacobian.dimension;
    let design_dimension = flow.objective_design_gradient.len();
    if acoustic.acoustic_residual.len() != acoustic_dimension
        || acoustic.objective_acoustic_state_gradient.len() != acoustic_dimension
        || acoustic.acoustic_flow_jacobian.len() != acoustic_dimension
        || acoustic.acoustic_design_jacobian.len() != acoustic_dimension
        || acoustic.objective_acoustic_design_gradient.len() != design_dimension
        || acoustic
            .acoustic_flow_jacobian
            .iter()
            .any(|row| row.len() != flow_dimension)
        || acoustic
            .acoustic_design_jacobian
            .iter()
            .any(|row| row.len() != design_dimension)
    {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "aeroacoustic PDE derivative data",
        ));
    }
    if acoustic
        .acoustic_residual
        .iter()
        .chain(&acoustic.objective_acoustic_state_gradient)
        .chain(&acoustic.objective_acoustic_design_gradient)
        .chain(acoustic.acoustic_flow_jacobian.iter().flatten())
        .chain(acoustic.acoustic_design_jacobian.iter().flatten())
        .any(|value| !value.is_finite())
    {
        return Err(PcFanResolvedAdjointError::NonFiniteInput(
            "aeroacoustic PDE derivative data",
        ));
    }
    Ok(())
}

fn validate_config(config: &PcFanResolvedAdjointConfig) -> Result<(), PcFanResolvedAdjointError> {
    for (name, value) in [
        ("tolerance", config.tolerance),
        ("max_state_residual", config.max_state_residual),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(PcFanResolvedAdjointError::InvalidConfig(name, value));
        }
    }
    if config.max_iterations == 0 {
        return Err(PcFanResolvedAdjointError::InvalidIterationLimit);
    }
    Ok(())
}

fn validate_linearization(
    linearization: &PcFanResolvedFlowLinearization,
) -> Result<(), PcFanResolvedAdjointError> {
    if !linearization.flow_model.is_cfd() {
        return Err(PcFanResolvedAdjointError::UnsupportedFlowModel(
            linearization.flow_model,
        ));
    }
    if linearization.provenance.trim().is_empty() {
        return Err(PcFanResolvedAdjointError::MissingProvenance);
    }
    let provenance = linearization.provenance.to_ascii_lowercase();
    if provenance.contains("picard") || provenance.contains("oseen") {
        return Err(PcFanResolvedAdjointError::OseenLinearizationNotAdmissible);
    }
    if provenance.contains("derivative-probe")
        || provenance.contains("finite-difference")
        || provenance.contains("black-box")
    {
        return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
    }
    linearization.state_jacobian.validate()?;
    let dimension = linearization.state_jacobian.dimension;
    if linearization.state_residual.len() != dimension
        || linearization.objective_state_gradient.len() != dimension
        || linearization.residual_design_jacobian.len() != dimension
    {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "state vectors",
        ));
    }
    let design_dimension = linearization.objective_design_gradient.len();
    if linearization
        .residual_design_jacobian
        .iter()
        .any(|row| row.len() != design_dimension)
    {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "design Jacobian",
        ));
    }
    if linearization
        .state_residual
        .iter()
        .chain(&linearization.objective_state_gradient)
        .chain(&linearization.objective_design_gradient)
        .chain(
            linearization
                .residual_design_jacobian
                .iter()
                .flat_map(|row| row.iter()),
        )
        .any(|value| !value.is_finite())
    {
        return Err(PcFanResolvedAdjointError::NonFiniteInput(
            "linearization vectors",
        ));
    }
    Ok(())
}

fn solve_transpose_gmres(
    matrix: &PcFanResolvedFlowSparseMatrix,
    rhs: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> Result<(Vec<f64>, usize, f64), PcFanResolvedAdjointError> {
    let mut solution = vec![0.0; matrix.dimension];
    let mut residual = rhs.to_vec();
    let mut residual_norm = l2_norm(&residual);
    let mut iterations = 0;
    while iterations < max_iterations && residual_norm > tolerance {
        let (correction, used, _) =
            solve_transpose_gmres_cycle(matrix, &residual, tolerance, max_iterations - iterations)?;
        for (value, delta) in solution.iter_mut().zip(correction) {
            *value += delta;
        }
        iterations += used;
        residual = rhs
            .iter()
            .zip(matrix.apply_transpose(&solution))
            .map(|(&right, value)| right - value)
            .collect();
        let next_norm = l2_norm(&residual);
        if used == 0 || next_norm >= residual_norm {
            residual_norm = next_norm;
            break;
        }
        residual_norm = next_norm;
        // Restart from the independently reassembled residual after a
        // numerical Krylov breakdown. Correction iterations share the
        // original total budget and the unscaled acceptance tolerance.
    }
    if residual_norm <= tolerance {
        Ok((solution, iterations, residual_norm))
    } else {
        Err(PcFanResolvedAdjointError::DidNotConverge {
            iterations,
            residual: residual_norm,
            tolerance,
        })
    }
}

fn solve_transpose_gmres_cycle(
    matrix: &PcFanResolvedFlowSparseMatrix,
    rhs: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> Result<(Vec<f64>, usize, f64), PcFanResolvedAdjointError> {
    let dimension = matrix.dimension;
    let beta = l2_norm(rhs);
    if beta <= tolerance {
        return Ok((vec![0.0; dimension], 0, 0.0));
    }

    let mut solution = vec![0.0; dimension];
    let mut basis = Vec::<Vec<f64>>::with_capacity(max_iterations);
    let mut hessenberg = vec![vec![0.0; max_iterations]; max_iterations + 1];
    let mut cosines = vec![0.0; max_iterations];
    let mut sines = vec![0.0; max_iterations];
    let mut transformed_rhs = vec![0.0; max_iterations + 1];
    transformed_rhs[0] = beta;
    let mut residual_estimate = beta;
    let mut used_iterations = 0;

    for iteration in 0..max_iterations {
        if iteration == 0 {
            basis.push(rhs.iter().map(|value| value / beta).collect());
        }
        let basis_vector =
            basis
                .get(iteration)
                .cloned()
                .ok_or(PcFanResolvedAdjointError::DidNotConverge {
                    iterations: used_iterations,
                    residual: residual_estimate,
                    tolerance,
                })?;
        let mut work = matrix.apply_transpose(&basis_vector);
        let work_norm = l2_norm(&work);
        // A second modified Gram-Schmidt pass preserves orthogonality for
        // the ill-conditioned velocity/pressure saddle-point operator.
        for _ in 0..2 {
            for basis_index in 0..=iteration {
                let projection = dot(&work, &basis[basis_index]);
                hessenberg[basis_index][iteration] += projection;
                for (work_value, &basis_value) in work.iter_mut().zip(&basis[basis_index]) {
                    *work_value -= projection * basis_value;
                }
            }
        }
        hessenberg[iteration + 1][iteration] = l2_norm(&work);
        let has_next_basis = hessenberg[iteration + 1][iteration] > f64::EPSILON * work_norm;
        if has_next_basis {
            basis.push(
                work.iter()
                    .map(|value| value / hessenberg[iteration + 1][iteration])
                    .collect(),
            );
        }

        for prior in 0..iteration {
            let upper = cosines[prior] * hessenberg[prior][iteration]
                + sines[prior] * hessenberg[prior + 1][iteration];
            hessenberg[prior + 1][iteration] = -sines[prior] * hessenberg[prior][iteration]
                + cosines[prior] * hessenberg[prior + 1][iteration];
            hessenberg[prior][iteration] = upper;
        }
        let denominator =
            hessenberg[iteration][iteration].hypot(hessenberg[iteration + 1][iteration]);
        if denominator == 0.0 {
            used_iterations = iteration + 1;
            break;
        }
        cosines[iteration] = hessenberg[iteration][iteration] / denominator;
        sines[iteration] = hessenberg[iteration + 1][iteration] / denominator;
        hessenberg[iteration][iteration] = denominator;
        hessenberg[iteration + 1][iteration] = 0.0;
        transformed_rhs[iteration + 1] = -sines[iteration] * transformed_rhs[iteration];
        transformed_rhs[iteration] *= cosines[iteration];
        residual_estimate = transformed_rhs[iteration + 1].abs();
        used_iterations = iteration + 1;

        let coefficients = back_substitute(&hessenberg, &transformed_rhs, used_iterations)?;
        solution.fill(0.0);
        for (coefficient, vector) in coefficients.iter().zip(&basis) {
            for (solution_value, &basis_value) in solution.iter_mut().zip(vector) {
                *solution_value += coefficient * basis_value;
            }
        }
        if residual_estimate <= tolerance {
            let actual_residual = l2_norm(
                &rhs.iter()
                    .zip(matrix.apply_transpose(&solution))
                    .map(|(&right, value)| right - value)
                    .collect::<Vec<_>>(),
            );
            if actual_residual <= tolerance {
                return Ok((solution, used_iterations, actual_residual));
            }
        }
        if !has_next_basis {
            break;
        }
    }

    let actual_residual = l2_norm(
        &rhs.iter()
            .zip(matrix.apply_transpose(&solution))
            .map(|(&right, value)| right - value)
            .collect::<Vec<_>>(),
    );
    Ok((solution, used_iterations, actual_residual))
}

fn back_substitute(
    matrix: &[Vec<f64>],
    rhs: &[f64],
    dimension: usize,
) -> Result<Vec<f64>, PcFanResolvedAdjointError> {
    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let diagonal = matrix[row][row];
        if diagonal.abs() <= f64::EPSILON {
            return Err(PcFanResolvedAdjointError::SingularAdjointSystem { index: row });
        }
        let upper_sum = ((row + 1)..dimension)
            .map(|column| matrix[row][column] * solution[column])
            .sum::<f64>();
        solution[row] = (rhs[row] - upper_sum) / diagonal;
    }
    Ok(solution)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// Explicit budgets for a production fixed-point derivative probe.
///
/// The probe differentiates the *published production Picard map* by
/// centered perturbations. It is useful for exposing missing active terms and
/// for Taylor diagnostics, but its result is deliberately not admissible to
/// [`solve_pc_fan_resolved_flow_adjoint`]. An exact adjoint still requires a
/// solver-published Newton tangent and design residual derivatives.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedDerivativeProbeConfig {
    pub state_step: f64,
    pub design_step: f64,
    pub taylor_tolerance: f64,
    pub max_state_dimension: usize,
    pub max_design_dimension: usize,
}

impl Default for PcFanResolvedDerivativeProbeConfig {
    fn default() -> Self {
        Self {
            state_step: 1.0e-6,
            design_step: 1.0e-6,
            taylor_tolerance: 1.0e-4,
            max_state_dimension: 4096,
            max_design_dimension: 256,
        }
    }
}

/// Checked derivatives and Taylor diagnostics obtained from a production
/// fixed-point map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanResolvedFlowDerivativeProbeReport {
    pub linearization: PcFanResolvedFlowLinearization,
    pub state_taylor_error: f64,
    pub design_taylor_error: f64,
    pub objective_state_taylor_error: f64,
    pub objective_design_taylor_error: f64,
}

/// Differentiate a production resolved-flow Picard map and objective.
///
/// `evaluate_state` must run exactly one production map update at the supplied
/// state. `evaluate_design` must evaluate that same map at the base state for
/// a supplied design vector. The callbacks are intentionally explicit so the
/// caller cannot accidentally differentiate a reduced-order surrogate. Each
/// returned observation is checked as a FEM-published Picard artifact, and all
/// active state/design/objective columns receive a centered Taylor check.
///
/// The result is a diagnostic finite-difference probe, not an exact Newton
/// tangent. Its provenance therefore causes the exact adjoint solver to reject
/// it fail-closed.
pub fn probe_pc_fan_resolved_flow_derivatives<StateFn, DesignFn, ObjectiveFn>(
    config: &PcFanResolvedDerivativeProbeConfig,
    base: &PcFanResolvedFlowPicardLinearization,
    design: &[f64],
    evaluate_state: StateFn,
    evaluate_design: DesignFn,
    evaluate_objective: ObjectiveFn,
) -> Result<PcFanResolvedFlowDerivativeProbeReport, PcFanResolvedAdjointError>
where
    StateFn: Fn(&[f64]) -> Result<PcFanResolvedFlowPicardLinearization, PcFanResolvedAdjointError>,
    DesignFn: Fn(&[f64]) -> Result<PcFanResolvedFlowPicardLinearization, PcFanResolvedAdjointError>,
    ObjectiveFn: Fn(&[f64], &[f64]) -> Result<f64, PcFanResolvedAdjointError>,
{
    validate_probe_config(config)?;
    base.validate()?;
    if design.len() > config.max_design_dimension {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "derivative probe design budget",
        ));
    }
    if design.iter().any(|value| !value.is_finite()) {
        return Err(PcFanResolvedAdjointError::NonFiniteInput(
            "derivative probe design",
        ));
    }

    let dimension = base.state.len();
    if dimension > config.max_state_dimension {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "derivative probe state budget",
        ));
    }
    let base_residual = picard_residual(base)?;
    let base_objective = evaluate_objective(&base.state, design)
        .map_err(|_| PcFanResolvedAdjointError::ProbeEvaluationFailed("base objective"))?;
    if !base_objective.is_finite() {
        return Err(PcFanResolvedAdjointError::NonFiniteInput(
            "base derivative probe objective",
        ));
    }

    let mut state_jacobian_dense = vec![vec![0.0; dimension]; dimension];
    let mut objective_state_gradient = vec![0.0; dimension];
    let mut state_taylor_error: f64 = 0.0;
    let mut objective_state_taylor_error: f64 = 0.0;

    for column in 0..dimension {
        let plus_state = perturb(&base.state, column, config.state_step);
        let minus_state = perturb(&base.state, column, -config.state_step);
        let plus = evaluate_state(&plus_state)
            .map_err(|_| PcFanResolvedAdjointError::ProbeEvaluationFailed("state plus"))?;
        let minus = evaluate_state(&minus_state)
            .map_err(|_| PcFanResolvedAdjointError::ProbeEvaluationFailed("state minus"))?;
        validate_probe_observation(base, &plus)?;
        validate_probe_observation(base, &minus)?;
        let plus_residual = picard_residual(&plus)?;
        let minus_residual = picard_residual(&minus)?;
        for row in 0..dimension {
            let derivative = (plus_residual[row] - minus_residual[row]) / (2.0 * config.state_step);
            if !derivative.is_finite() {
                return Err(PcFanResolvedAdjointError::NonFiniteInput(
                    "state residual derivative",
                ));
            }
            state_jacobian_dense[row][column] = derivative;
            state_taylor_error = state_taylor_error.max(relative_taylor_error(
                plus_residual[row] - base_residual[row],
                derivative * config.state_step,
            ));
        }

        let plus_objective = evaluate_objective(&plus_state, design).map_err(|_| {
            PcFanResolvedAdjointError::ProbeEvaluationFailed("objective state plus")
        })?;
        let minus_objective = evaluate_objective(&minus_state, design).map_err(|_| {
            PcFanResolvedAdjointError::ProbeEvaluationFailed("objective state minus")
        })?;
        if !plus_objective.is_finite() || !minus_objective.is_finite() {
            return Err(PcFanResolvedAdjointError::NonFiniteInput(
                "state objective derivative",
            ));
        }
        objective_state_gradient[column] =
            (plus_objective - minus_objective) / (2.0 * config.state_step);
        objective_state_taylor_error = objective_state_taylor_error.max(relative_taylor_error(
            plus_objective - base_objective,
            objective_state_gradient[column] * config.state_step,
        ));
    }

    let mut residual_design_jacobian = vec![vec![0.0; design.len()]; dimension];
    let mut objective_design_gradient = vec![0.0; design.len()];
    let mut design_taylor_error: f64 = 0.0;
    let mut objective_design_taylor_error: f64 = 0.0;
    for column in 0..design.len() {
        let plus_design = perturb(design, column, config.design_step);
        let minus_design = perturb(design, column, -config.design_step);
        let plus = evaluate_design(&plus_design)
            .map_err(|_| PcFanResolvedAdjointError::ProbeEvaluationFailed("design plus"))?;
        let minus = evaluate_design(&minus_design)
            .map_err(|_| PcFanResolvedAdjointError::ProbeEvaluationFailed("design minus"))?;
        validate_probe_observation(base, &plus)?;
        validate_probe_observation(base, &minus)?;
        let plus_residual = picard_residual(&plus)?;
        let minus_residual = picard_residual(&minus)?;
        for row in 0..dimension {
            let derivative =
                (plus_residual[row] - minus_residual[row]) / (2.0 * config.design_step);
            if !derivative.is_finite() {
                return Err(PcFanResolvedAdjointError::NonFiniteInput(
                    "design residual derivative",
                ));
            }
            residual_design_jacobian[row][column] = derivative;
            design_taylor_error = design_taylor_error.max(relative_taylor_error(
                plus_residual[row] - base_residual[row],
                derivative * config.design_step,
            ));
        }

        let plus_objective = evaluate_objective(&base.state, &plus_design).map_err(|_| {
            PcFanResolvedAdjointError::ProbeEvaluationFailed("objective design plus")
        })?;
        let minus_objective = evaluate_objective(&base.state, &minus_design).map_err(|_| {
            PcFanResolvedAdjointError::ProbeEvaluationFailed("objective design minus")
        })?;
        if !plus_objective.is_finite() || !minus_objective.is_finite() {
            return Err(PcFanResolvedAdjointError::NonFiniteInput(
                "design objective derivative",
            ));
        }
        objective_design_gradient[column] =
            (plus_objective - minus_objective) / (2.0 * config.design_step);
        objective_design_taylor_error = objective_design_taylor_error.max(relative_taylor_error(
            plus_objective - base_objective,
            objective_design_gradient[column] * config.design_step,
        ));
    }

    for (quantity, error) in [
        ("state residual", state_taylor_error),
        ("design residual", design_taylor_error),
        ("state objective", objective_state_taylor_error),
        ("design objective", objective_design_taylor_error),
    ] {
        if error > config.taylor_tolerance {
            return Err(PcFanResolvedAdjointError::TaylorTestFailed {
                quantity,
                error,
                tolerance: config.taylor_tolerance,
            });
        }
    }

    let linearization = PcFanResolvedFlowLinearization {
        flow_model: base.flow_model,
        backend: base.backend,
        provenance: "production:resolved-flow:derivative-probe:finite-difference:v1".into(),
        state_jacobian: PcFanResolvedFlowSparseMatrix::from_dense(&state_jacobian_dense)?,
        state_residual: base_residual,
        residual_design_jacobian,
        objective_state_gradient,
        objective_design_gradient,
    };
    Ok(PcFanResolvedFlowDerivativeProbeReport {
        linearization,
        state_taylor_error,
        design_taylor_error,
        objective_state_taylor_error,
        objective_design_taylor_error,
    })
}

fn validate_probe_config(
    config: &PcFanResolvedDerivativeProbeConfig,
) -> Result<(), PcFanResolvedAdjointError> {
    for (name, value) in [
        ("state_step", config.state_step),
        ("design_step", config.design_step),
        ("taylor_tolerance", config.taylor_tolerance),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(PcFanResolvedAdjointError::InvalidConfig(name, value));
        }
    }
    if config.max_state_dimension == 0 || config.max_design_dimension == 0 {
        return Err(PcFanResolvedAdjointError::InvalidIterationLimit);
    }
    Ok(())
}

fn picard_residual(
    observation: &PcFanResolvedFlowPicardLinearization,
) -> Result<Vec<f64>, PcFanResolvedAdjointError> {
    observation.validate()?;
    Ok(observation
        .fixed_point_residual
        .iter()
        .map(|value| -value)
        .collect())
}

fn validate_probe_observation(
    base: &PcFanResolvedFlowPicardLinearization,
    observation: &PcFanResolvedFlowPicardLinearization,
) -> Result<(), PcFanResolvedAdjointError> {
    observation.validate()?;
    if observation.flow_model != base.flow_model || observation.backend != base.backend {
        return Err(PcFanResolvedAdjointError::ProbeEvaluationFailed(
            "incompatible map observation",
        ));
    }
    if observation.state.len() != base.state.len() {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "derivative probe map observation",
        ));
    }
    Ok(())
}

fn perturb(values: &[f64], index: usize, delta: f64) -> Vec<f64> {
    let mut perturbed = values.to_vec();
    perturbed[index] += delta;
    perturbed
}

fn relative_taylor_error(actual: f64, predicted: f64) -> f64 {
    (actual - predicted).abs() / actual.abs().max(predicted.abs()).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linearization(flow_model: PcFanFlowModel) -> PcFanResolvedFlowLinearization {
        PcFanResolvedFlowLinearization {
            flow_model,
            backend: PcFanCfdBackend::Cpu,
            provenance: "fixture:resolved-flow-linearization:v1".into(),
            state_jacobian: PcFanResolvedFlowSparseMatrix::from_dense(&[
                vec![3.0, 1.0],
                vec![0.0, 2.0],
            ])
            .unwrap(),
            state_residual: vec![0.0, 0.0],
            residual_design_jacobian: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            objective_state_gradient: vec![5.0, 6.0],
            objective_design_gradient: vec![7.0, 8.0],
        }
    }

    fn newton_linearization() -> PcFanResolvedFlowNewtonLinearization {
        PcFanResolvedFlowNewtonLinearization {
            flow_model: PcFanFlowModel::IncompressibleCfdFull,
            backend: PcFanCfdBackend::Cpu,
            provenance: "fem:navier-stokes:newton-convection:v1".into(),
            state_jacobian: PcFanResolvedFlowSparseMatrix::from_dense(&[
                vec![3.0, 1.0],
                vec![0.0, 2.0],
            ])
            .unwrap(),
            state_residual: vec![0.0, 0.0],
            state: vec![0.25, -0.5],
            residual_design_jacobian: None,
        }
    }

    fn analytic_derivatives() -> PcFanResolvedFlowDerivativeData {
        PcFanResolvedFlowDerivativeData {
            provenance: "production:pcfan:analytic-design-objective:v1".into(),
            residual_design_jacobian: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            objective_state_gradient: vec![5.0, 6.0],
            objective_design_gradient: vec![7.0, 8.0],
        }
    }

    #[test]
    fn implicit_aeroacoustic_pde_adjoint_composes_flow_and_source_derivatives() {
        let flow = linearization(PcFanFlowModel::IncompressibleCfdFull);
        let acoustic = PcFanResolvedAeroacousticPdeDerivativeData {
            provenance: "analytic:resolved-aeroacoustic-pde:v1".into(),
            acoustic_state_jacobian: PcFanResolvedFlowSparseMatrix::from_dense(&[
                vec![2.0, 0.0],
                vec![0.0, 4.0],
            ])
            .unwrap(),
            acoustic_flow_jacobian: vec![vec![1.0, 2.0], vec![3.0, 1.0]],
            acoustic_design_jacobian: vec![vec![0.5, 1.0], vec![-2.0, 0.25]],
            acoustic_residual: vec![0.0, 0.0],
            objective_acoustic_state_gradient: vec![4.0, -2.0],
            objective_acoustic_design_gradient: vec![0.2, 0.4],
        };

        let report = solve_pc_fan_resolved_aeroacoustic_adjoint(
            &PcFanResolvedAdjointConfig::default(),
            &flow,
            &acoustic,
        )
        .unwrap();

        for (actual, expected) in report
            .acoustic_adjoint
            .iter()
            .zip([2.0, -0.5])
            .chain(report.flow_adjoint.iter().zip([1.5, 0.5]))
            .chain(report.design_gradient.iter().zip([2.2, 1.525]))
        {
            assert!((*actual - expected).abs() < 1.0e-12);
        }
        assert!(report.provenance.contains("aeroacoustic-pde"));
    }

    #[test]
    fn implicit_aeroacoustic_pde_adjoint_rejects_diagnostic_derivatives() {
        let flow = linearization(PcFanFlowModel::IncompressibleCfdFull);
        let mut acoustic = PcFanResolvedAeroacousticPdeDerivativeData {
            provenance: "finite-difference:acoustic-pde:v1".into(),
            acoustic_state_jacobian: PcFanResolvedFlowSparseMatrix::from_dense(&[vec![1.0]])
                .unwrap(),
            acoustic_flow_jacobian: vec![vec![0.0, 0.0]],
            acoustic_design_jacobian: vec![vec![0.0, 0.0]],
            acoustic_residual: vec![0.0],
            objective_acoustic_state_gradient: vec![0.0],
            objective_acoustic_design_gradient: vec![0.0, 0.0],
        };

        assert!(matches!(
            solve_pc_fan_resolved_aeroacoustic_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &flow,
                &acoustic,
            ),
            Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)
        ));
        acoustic.provenance = "analytic:acoustic-pde:v1".into();
        acoustic.acoustic_flow_jacobian[0].pop();
        assert!(matches!(
            solve_pc_fan_resolved_aeroacoustic_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &flow,
                &acoustic,
            ),
            Err(PcFanResolvedAdjointError::DimensionMismatch(
                "aeroacoustic PDE derivative data"
            ))
        ));
    }

    #[test]
    fn production_newton_tangent_wires_into_exact_resolved_adjoint() {
        let report = solve_pc_fan_resolved_flow_adjoint_from_newton(
            &PcFanResolvedAdjointConfig::default(),
            &newton_linearization(),
            analytic_derivatives(),
        )
        .unwrap();

        assert_eq!(report.flow_model, PcFanFlowModel::IncompressibleCfdFull);
        assert_eq!(report.backend, PcFanCfdBackend::Cpu);
        assert_eq!(
            report.provenance,
            "fem:navier-stokes:newton-convection:v1; derivatives:production:pcfan:analytic-design-objective:v1"
        );
        assert!((report.adjoint[0] - 5.0 / 3.0).abs() < 1.0e-10);
        assert!((report.adjoint[1] - 13.0 / 6.0).abs() < 1.0e-10);
        assert!(
            (report.design_gradient[0] - (7.0 - (5.0 / 3.0 + 3.0 * 13.0 / 6.0))).abs() < 1.0e-10
        );
        assert!(
            (report.design_gradient[1] - (8.0 - (2.0 * 5.0 / 3.0 + 4.0 * 13.0 / 6.0))).abs()
                < 1.0e-10
        );
    }

    #[test]
    fn exact_newton_handoff_rejects_probe_provenance_and_bad_dimensions() {
        let source = newton_linearization();
        let mut probe_data = analytic_derivatives();
        probe_data.provenance =
            "production:resolved-flow:derivative-probe:finite-difference:v1".into();
        assert!(matches!(
            PcFanResolvedFlowLinearization::from_newton(&source, probe_data),
            Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)
        ));

        let mut malformed = analytic_derivatives();
        malformed.objective_design_gradient = vec![7.0];
        assert!(matches!(
            PcFanResolvedFlowLinearization::from_newton(&source, malformed),
            Err(PcFanResolvedAdjointError::DimensionMismatch(
                "design Jacobian"
            ))
        ));
    }

    #[test]
    fn production_design_residual_handoff_requires_analytic_objective_provenance() {
        let mut source = newton_linearization();
        source.residual_design_jacobian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let derivatives = PcFanResolvedFlowDerivativeData::from_production_newton(
            &source,
            "analytic:pressure-objective:v1",
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        )
        .unwrap();
        assert!(derivatives.provenance.contains("analytic-design-residual"));

        assert!(matches!(
            PcFanResolvedFlowDerivativeData::from_production_newton(
                &source,
                "finite-difference:objective:v1",
                vec![5.0, 6.0],
                vec![7.0, 8.0],
            ),
            Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)
        ));
        assert!(matches!(
            PcFanResolvedFlowDerivativeData::from_production_newton(
                &newton_linearization(),
                "analytic:pressure-objective:v1",
                vec![5.0, 6.0],
                vec![7.0, 8.0],
            ),
            Err(PcFanResolvedAdjointError::MissingProductionDesignResidualJacobian)
        ));
    }

    #[test]
    fn production_derivative_bundle_rejects_malformed_objective_vectors() {
        let mut source = newton_linearization();
        source.residual_design_jacobian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        assert!(matches!(
            PcFanResolvedFlowDerivativeData::from_production_newton(
                &source,
                "analytic:pressure-objective:v1",
                vec![5.0],
                vec![7.0, 8.0],
            ),
            Err(PcFanResolvedAdjointError::DimensionMismatch(
                "production objective derivatives"
            ))
        ));
        assert!(matches!(
            PcFanResolvedFlowDerivativeData::from_production_newton(
                &source,
                "analytic:pressure-objective:v1",
                vec![5.0, f64::NAN],
                vec![7.0, 8.0],
            ),
            Err(PcFanResolvedAdjointError::NonFiniteInput(
                "production objective derivatives"
            ))
        ));
    }

    #[test]
    fn newton_linearization_rejects_diagnostic_provenance() {
        for provenance in [
            "black-box:resolved-flow:v1",
            "derivative-probe:resolved-flow:v1",
        ] {
            let mut source = newton_linearization();
            source.provenance = provenance.into();
            assert!(matches!(
                source.validate(),
                Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)
            ));
        }
    }

    #[test]
    fn resolved_adjoint_solves_transpose_and_contracts_design_gradient() {
        let report = solve_pc_fan_resolved_flow_adjoint(
            &PcFanResolvedAdjointConfig::default(),
            &linearization(PcFanFlowModel::IncompressibleCfdFull),
        )
        .unwrap();
        // A^T λ = [5, 6] gives λ=[5/3, 13/6].
        assert!((report.adjoint[0] - 5.0 / 3.0).abs() < 1.0e-10);
        assert!((report.adjoint[1] - 13.0 / 6.0).abs() < 1.0e-10);
        assert!(
            (report.design_gradient[0] - (7.0 - (5.0 / 3.0 + 3.0 * 13.0 / 6.0))).abs() < 1.0e-10
        );
        assert!(
            (report.design_gradient[1] - (8.0 - (2.0 * 5.0 / 3.0 + 4.0 * 13.0 / 6.0))).abs()
                < 1.0e-10
        );
        assert!(report.converged);
        assert!(report.adjoint_residual_norm <= 1.0e-10);
    }

    #[test]
    fn resolved_adjoint_gradient_matches_linearized_state_finite_difference() {
        let linearization = linearization(PcFanFlowModel::IncompressibleCfdFull);
        let report = solve_pc_fan_resolved_flow_adjoint(
            &PcFanResolvedAdjointConfig::default(),
            &linearization,
        )
        .unwrap();

        // R(x, d) = A x + R_d d - b, with x(0) = [1, 2]. Eliminate x
        // analytically for a centered finite-difference check of the adjoint
        // contraction.
        fn state_for_design(design: &[f64; 2]) -> [f64; 2] {
            let rhs0 = 5.0 - design[0] - 2.0 * design[1];
            let rhs1 = 4.0 - 3.0 * design[0] - 4.0 * design[1];
            [rhs0 / 3.0 - rhs1 / 6.0, rhs1 / 2.0]
        }
        fn objective(design: &[f64; 2]) -> f64 {
            let state = state_for_design(design);
            5.0 * state[0] + 6.0 * state[1] + 7.0 * design[0] + 8.0 * design[1]
        }

        for (index, expected) in report.design_gradient.iter().enumerate() {
            let mut plus = [0.0, 0.0];
            let mut minus = [0.0, 0.0];
            plus[index] = 1.0e-6;
            minus[index] = -1.0e-6;
            let finite_difference = (objective(&plus) - objective(&minus)) / 2.0e-6;
            assert!((finite_difference - expected).abs() < 1.0e-8);
        }
    }

    #[test]
    fn aeroacoustic_output_map_composes_exact_chain_rule() {
        let mut source = newton_linearization();
        source.residual_design_jacobian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let derivatives =
            PcFanResolvedFlowDerivativeData::from_production_newton_with_aeroacoustic(
                &source,
                "analytic:flow-objective:v1",
                vec![5.0, 6.0],
                vec![7.0, 8.0],
                PcFanResolvedAeroacousticDerivativeData {
                    provenance: "analytic:resolved-aeroacoustic-output:v1".into(),
                    acoustic_state_jacobian: vec![vec![0.5, -1.0], vec![2.0, 0.25]],
                    acoustic_design_jacobian: vec![vec![1.5, -0.5], vec![-2.0, 3.0]],
                    objective_acoustic_state_gradient: vec![4.0, -1.0],
                    objective_acoustic_design_gradient: vec![0.2, 0.4],
                },
            )
            .expect("analytic acoustic output map should compose");

        assert_eq!(derivatives.objective_state_gradient, vec![5.0, 1.75]);
        assert_eq!(derivatives.objective_design_gradient, vec![15.2, 3.4]);
        let linearization = PcFanResolvedFlowLinearization::from_newton(&source, derivatives)
            .expect("composed derivative data should form an exact linearization");
        let report = solve_pc_fan_resolved_flow_adjoint(
            &PcFanResolvedAdjointConfig::default(),
            &linearization,
        )
        .expect("composed acoustic adjoint should solve");

        assert!((report.adjoint[0] - 5.0 / 3.0).abs() < 1.0e-10);
        assert!((report.adjoint[1] - 1.0 / 24.0).abs() < 1.0e-10);
        assert!((report.design_gradient[0] - 13.408333333333333).abs() < 1.0e-10);
        assert!((report.design_gradient[1] + 0.1).abs() < 1.0e-10);
    }

    #[test]
    fn aeroacoustic_output_map_rejects_diagnostic_derivative_provenance() {
        let mut source = newton_linearization();
        source.residual_design_jacobian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let error = PcFanResolvedFlowDerivativeData::from_production_newton_with_aeroacoustic(
            &source,
            "analytic:flow-objective:v1",
            vec![5.0, 6.0],
            vec![7.0, 8.0],
            PcFanResolvedAeroacousticDerivativeData {
                provenance: "finite-difference:acoustic-output:v1".into(),
                acoustic_state_jacobian: vec![vec![0.5, -1.0]],
                acoustic_design_jacobian: vec![vec![1.5, -0.5]],
                objective_acoustic_state_gradient: vec![4.0],
                objective_acoustic_design_gradient: vec![0.2, 0.4],
            },
        )
        .expect_err("diagnostic acoustic derivatives must remain fail-closed");
        assert_eq!(
            error,
            PcFanResolvedAdjointError::DerivativeProbeNotAdmissible
        );
    }

    #[test]
    fn reduced_flow_and_bad_linearizations_fail_closed() {
        assert!(matches!(
            solve_pc_fan_resolved_flow_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &linearization(PcFanFlowModel::ReducedOrder)
            ),
            Err(PcFanResolvedAdjointError::UnsupportedFlowModel(_))
        ));
        let mut invalid = linearization(PcFanFlowModel::IncompressibleCfdFull);
        invalid.state_residual = vec![f64::NAN, 0.0];
        assert!(matches!(
            solve_pc_fan_resolved_flow_adjoint(&PcFanResolvedAdjointConfig::default(), &invalid),
            Err(PcFanResolvedAdjointError::NonFiniteInput(_))
        ));
    }

    #[test]
    fn picard_oseen_provenance_cannot_enter_exact_resolved_adjoint() {
        let mut linearization = linearization(PcFanFlowModel::IncompressibleCfdFull);
        linearization.provenance = "fem:navier-stokes:picard-oseen:v1".into();

        assert!(matches!(
            solve_pc_fan_resolved_flow_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &linearization,
            ),
            Err(PcFanResolvedAdjointError::OseenLinearizationNotAdmissible)
        ));
    }

    #[test]
    fn published_picard_linearization_has_its_own_checked_contract() {
        let mut observation = PcFanResolvedFlowPicardLinearization {
            flow_model: PcFanFlowModel::IncompressibleCfd,
            backend: PcFanCfdBackend::Cpu,
            provenance: "fixture:pcfan:picard-oseen:v1".into(),
            state_jacobian: PcFanResolvedFlowSparseMatrix::from_dense(&[
                vec![2.0, 0.0],
                vec![0.0, 3.0],
            ])
            .unwrap(),
            state: vec![1.0, 2.0],
            fixed_point_residual: vec![1.0e-8, -2.0e-8],
        };
        observation.validate().unwrap();

        observation.fixed_point_residual.pop();
        assert!(matches!(
            observation.validate(),
            Err(PcFanResolvedAdjointError::DimensionMismatch(_))
        ));
    }

    #[test]
    fn analytic_newton_tangent_has_a_separate_checked_contract() {
        let tangent = PcFanResolvedFlowNewtonLinearization {
            flow_model: PcFanFlowModel::IncompressibleCfdFull,
            backend: PcFanCfdBackend::Cpu,
            provenance: "fem:navier-stokes:newton-convection:v1".into(),
            state_jacobian: PcFanResolvedFlowSparseMatrix::from_dense(&[
                vec![2.0, 0.5],
                vec![0.0, 3.0],
            ])
            .unwrap(),
            state_residual: vec![0.2, -0.1],
            state: vec![1.0, 2.0],
            residual_design_jacobian: None,
        };
        tangent.validate().unwrap();

        let mut diagnostic = tangent.clone();
        diagnostic.provenance =
            "production:resolved-flow:derivative-probe:finite-difference:v1".into();
        assert!(matches!(
            diagnostic.validate(),
            Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)
        ));
    }

    #[test]
    fn sparse_matrix_rejects_invalid_csr_and_state_residual_budget() {
        assert!(matches!(
            PcFanResolvedFlowSparseMatrix::new(2, vec![0, 1], vec![0], vec![1.0]),
            Err(PcFanResolvedAdjointError::InvalidMatrix(_))
        ));
        let mut invalid = linearization(PcFanFlowModel::IncompressibleCfdFull);
        invalid.state_residual = vec![1.0, 0.0];
        let mut config = PcFanResolvedAdjointConfig::default();
        config.max_state_residual = 1.0e-3;
        assert!(matches!(
            solve_pc_fan_resolved_flow_adjoint(&config, &invalid),
            Err(PcFanResolvedAdjointError::StateResidualTooLarge { .. })
        ));
    }

    #[test]
    fn linearization_serde_round_trip_preserves_provenance() {
        let linearization = linearization(PcFanFlowModel::IncompressibleCfdFull);
        let encoded = serde_json::to_vec(&linearization).unwrap();
        let decoded: PcFanResolvedFlowLinearization = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.provenance, linearization.provenance);
        assert_eq!(decoded.state_jacobian, linearization.state_jacobian);
    }

    fn synthetic_picard_observation(
        state: &[f64],
        design: &[f64],
    ) -> PcFanResolvedFlowPicardLinearization {
        let residual = vec![
            2.0 * state[0] + 0.3 * state[1] + 1.5 * design[0] + 0.2 * design[1],
            -0.4 * state[0] + 1.7 * state[1] - 0.1 * design[0] + 0.9 * design[1],
        ];
        let output_state = state
            .iter()
            .zip(&residual)
            .map(|(value, residual)| value - residual)
            .collect();
        PcFanResolvedFlowPicardLinearization {
            flow_model: PcFanFlowModel::IncompressibleCfd,
            backend: PcFanCfdBackend::Cpu,
            provenance: "fixture:production-picard-map:v1".into(),
            state_jacobian: PcFanResolvedFlowSparseMatrix::from_dense(&[
                vec![1.0, 0.0],
                vec![0.0, 1.0],
            ])
            .unwrap(),
            state: output_state,
            fixed_point_residual: residual.iter().map(|value| -value).collect(),
        }
    }

    #[test]
    fn production_picard_derivative_probe_builds_checked_state_and_design_columns() {
        let base_state = vec![0.4, -0.3];
        let base_design = vec![0.2, -0.1];
        let mut base = synthetic_picard_observation(&base_state, &base_design);
        // The production observation is evaluated at a converged state; the
        // synthetic map stores its one-step output, so retain the prescribed
        // base input explicitly for this probe fixture.
        base.state = base_state.clone();
        let report = probe_pc_fan_resolved_flow_derivatives(
            &PcFanResolvedDerivativeProbeConfig {
                state_step: 1.0e-5,
                design_step: 1.0e-5,
                taylor_tolerance: 1.0e-8,
                ..Default::default()
            },
            &base,
            &base_design,
            |state| Ok(synthetic_picard_observation(state, &base_design)),
            |design| Ok(synthetic_picard_observation(&base_state, design)),
            |state, design| {
                Ok(
                    1.2 * state[0] * state[0] - 0.7 * state[1] + 2.0 * design[0] * design[0]
                        - 0.5 * design[1],
                )
            },
        )
        .unwrap();

        let matrix = &report.linearization.state_jacobian;
        assert_eq!(matrix.dimension, 2);
        let first_column = matrix.apply_transpose(&[1.0, 0.0]);
        assert!((first_column[0] - 2.0).abs() < 1.0e-8);
        assert!((first_column[1] - 0.3).abs() < 1.0e-8);
        for (actual, expected) in report.linearization.residual_design_jacobian[0]
            .iter()
            .zip([1.5, 0.2])
        {
            assert!((actual - expected).abs() < 1.0e-8);
        }
        for (actual, expected) in report.linearization.residual_design_jacobian[1]
            .iter()
            .zip([-0.1, 0.9])
        {
            assert!((actual - expected).abs() < 1.0e-8);
        }
        assert!(report.state_taylor_error <= 1.0e-8);
        assert!(report.design_taylor_error <= 1.0e-8);
        assert!(report.objective_state_taylor_error <= 1.0e-8);
        assert!(report.objective_design_taylor_error <= 1.0e-8);
        assert!(matches!(
            solve_pc_fan_resolved_flow_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &report.linearization,
            ),
            Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)
        ));
    }

    #[test]
    fn derivative_probe_rejects_budget_and_taylor_failures() {
        let base_state = vec![0.4, -0.3];
        let base_design = vec![0.2, -0.1];
        let mut base = synthetic_picard_observation(&base_state, &base_design);
        base.state = base_state.clone();
        let mut too_small = PcFanResolvedDerivativeProbeConfig::default();
        too_small.max_state_dimension = 1;
        assert!(matches!(
            probe_pc_fan_resolved_flow_derivatives(
                &too_small,
                &base,
                &base_design,
                |state| Ok(synthetic_picard_observation(state, &base_design)),
                |design| Ok(synthetic_picard_observation(&base_state, design)),
                |_, _| Ok(0.0),
            ),
            Err(PcFanResolvedAdjointError::DimensionMismatch(_))
        ));

        let mut strict = PcFanResolvedDerivativeProbeConfig::default();
        strict.state_step = 1.0e-2;
        strict.design_step = 1.0e-2;
        strict.taylor_tolerance = 1.0e-16;
        assert!(matches!(
            probe_pc_fan_resolved_flow_derivatives(
                &strict,
                &base,
                &base_design,
                |state| Ok(synthetic_picard_observation(state, &base_design)),
                |design| Ok(synthetic_picard_observation(&base_state, design)),
                |state, design| Ok(state[0] + design[0] * design[0]),
            ),
            Err(PcFanResolvedAdjointError::TaylorTestFailed { .. })
        ));
    }
}
