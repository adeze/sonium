//! Checked `[flow, k, omega]` coupling assembly for transported turbulence.
//!
//! This module owns the reusable block-composition layer extracted from
//! `sonium-sim-pcfan`'s `transported_turbulence`: the momentum/scalar coupled
//! linearization, the square monolithic Newton/BDF block, the segregated
//! current/history blocks, and the P1 tetrahedral shape-gradient geometry
//! they share. Scalar transport residuals, BDF stepping, and admissibility
//! live in [`super::transport`]; operating points, runner policy, and
//! provenance strings stay in the simulation crates, which stamp their own
//! provenance when delegating here.

use crate::assembly::{
    assemble_bilinear, assemble_bilinear_two_spaces_per_element, assemble_linear_per_element,
};
use crate::basis::PolynomialDegree;
use crate::forms::{BilinearForm, LocalMatrix, MassForm};
use crate::mesh::Mesh;
use crate::mesh::state_transfer::KOmegaBdfState;
use crate::spaces::{BasisTabulation, FunctionSpace, H1Space, VectorH1Space};
use sonium_solvers::CsrMatrix;
use sonium_solvers::traits::LinearOperator;

use super::forms::TurbulenceProductionForm;
use super::materials::KOmegaSstConstants;
use super::transport::{
    K_OMEGA_TRANSPORT_LINEARIZATION_PROVENANCE, KOmegaTransportElementFields, KOmegaTransportError,
    KOmegaTransportLinearization,
};

/// Fail-closed errors for k--omega coupling assembly.
#[derive(Debug, Clone, PartialEq)]
pub enum KOmegaCouplingError {
    Fem(KOmegaTransportError),
    InvalidCoupling(&'static str),
}

impl std::fmt::Display for KOmegaCouplingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fem(error) => {
                write!(formatter, "transported turbulence coupling failed: {error}")
            }
            Self::InvalidCoupling(reason) => {
                write!(
                    formatter,
                    "invalid transported turbulence coupling: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for KOmegaCouplingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Fem(error) => Some(error),
            Self::InvalidCoupling(_) => None,
        }
    }
}

impl From<KOmegaTransportError> for KOmegaCouplingError {
    fn from(error: KOmegaTransportError) -> Self {
        Self::Fem(error)
    }
}
/// Sparse coupling blocks needed to form a transported PC-fan flow/k/ω
/// tangent. Rows and columns use the production FEM global DOF order.
#[derive(Debug, Clone)]
pub struct KOmegaCoupledLinearization {
    pub provenance: &'static str,
    pub momentum_wrt_k: CsrMatrix<f64>,
    pub momentum_wrt_omega: CsrMatrix<f64>,
    pub k_wrt_velocity: CsrMatrix<f64>,
    pub omega_wrt_velocity: CsrMatrix<f64>,
    /// Nonlinear scalar-state correction to the frozen k diagonal block.
    pub k_wrt_k: CsrMatrix<f64>,
    /// Current-state derivative of the k residual with respect to omega.
    pub k_wrt_omega: CsrMatrix<f64>,
    /// Current-state derivative of the omega residual with respect to k.
    pub omega_wrt_k: CsrMatrix<f64>,
    /// Nonlinear scalar-state correction to the frozen omega diagonal block.
    pub omega_wrt_omega: CsrMatrix<f64>,
}

/// Square `[flow, k, omega]` block matrix for the current Newton/BDF map.
///
/// The flow block already contains velocity and pressure rows. The retained
/// off-diagonal coupling is velocity-only, so pressure rows/columns are
/// padded with zeros while the scalar diagonal blocks remain P1 global
/// operators. Global-mean reaction and limited-source dependencies are
/// represented by dense scalar columns in the state block.
#[derive(Debug, Clone)]
pub struct KOmegaMonolithicLinearization {
    pub provenance: &'static str,
    pub flow_dimension: usize,
    pub k_dimension: usize,
    pub omega_dimension: usize,
    pub state_jacobian: CsrMatrix<f64>,
}

impl PartialEq for KOmegaMonolithicLinearization {
    fn eq(&self, other: &Self) -> bool {
        self.provenance == other.provenance
            && self.flow_dimension == other.flow_dimension
            && self.k_dimension == other.k_dimension
            && self.omega_dimension == other.omega_dimension
            && csr_equal(&self.state_jacobian, &other.state_jacobian)
    }
}

impl PartialEq for KOmegaCoupledLinearization {
    fn eq(&self, other: &Self) -> bool {
        self.provenance == other.provenance
            && csr_equal(&self.momentum_wrt_k, &other.momentum_wrt_k)
            && csr_equal(&self.momentum_wrt_omega, &other.momentum_wrt_omega)
            && csr_equal(&self.k_wrt_velocity, &other.k_wrt_velocity)
            && csr_equal(&self.omega_wrt_velocity, &other.omega_wrt_velocity)
            && csr_equal(&self.k_wrt_k, &other.k_wrt_k)
            && csr_equal(&self.k_wrt_omega, &other.k_wrt_omega)
            && csr_equal(&self.omega_wrt_k, &other.omega_wrt_k)
            && csr_equal(&self.omega_wrt_omega, &other.omega_wrt_omega)
    }
}

fn csr_equal(left: &CsrMatrix<f64>, right: &CsrMatrix<f64>) -> bool {
    left.num_rows == right.num_rows
        && left.num_cols == right.num_cols
        && left.values == right.values
        && left.col_indices == right.col_indices
        && left.row_ptrs == right.row_ptrs
}

fn append_csr_block(
    triplets: &mut Vec<(usize, usize, f64)>,
    matrix: &CsrMatrix<f64>,
    row_offset: usize,
    column_offset: usize,
) -> Result<(), KOmegaCouplingError> {
    if matrix.row_ptrs.len() != matrix.num_rows.saturating_add(1)
        || matrix.col_indices.len() != matrix.values.len()
        || matrix.row_ptrs.last().copied() != Some(matrix.values.len())
    {
        return Err(KOmegaCouplingError::InvalidCoupling(
            "coupling block has invalid CSR structure",
        ));
    }
    for row in 0..matrix.num_rows {
        let start = matrix.row_ptrs[row];
        let end = matrix.row_ptrs[row + 1];
        if start > end {
            return Err(KOmegaCouplingError::InvalidCoupling(
                "coupling block row pointers are not monotone",
            ));
        }
        for entry in start..end {
            let column = matrix.col_indices[entry];
            let value = matrix.values[entry];
            if column >= matrix.num_cols || !value.is_finite() {
                return Err(KOmegaCouplingError::InvalidCoupling(
                    "coupling block contains an invalid entry",
                ));
            }
            triplets.push((row_offset + row, column_offset + column, value));
        }
    }
    Ok(())
}
pub fn assemble_komega_monolithic_linearization(
    flow_jacobian: &CsrMatrix<f64>,
    scalar_linearization: &KOmegaTransportLinearization,
    coupled_linearization: &KOmegaCoupledLinearization,
    coupled_provenance: &'static str,
    monolithic_provenance: &'static str,
) -> Result<KOmegaMonolithicLinearization, KOmegaCouplingError> {
    if coupled_linearization.provenance != coupled_provenance {
        return Err(KOmegaCouplingError::InvalidCoupling(
            "unexpected transported coupling provenance",
        ));
    }
    if scalar_linearization.provenance != K_OMEGA_TRANSPORT_LINEARIZATION_PROVENANCE {
        return Err(KOmegaCouplingError::InvalidCoupling(
            "unexpected scalar tangent provenance",
        ));
    }
    let flow_dimension = flow_jacobian.num_rows;
    if flow_dimension == 0 || flow_jacobian.num_cols != flow_dimension {
        return Err(KOmegaCouplingError::InvalidCoupling(
            "flow block must be non-empty and square",
        ));
    }
    let velocity_dimension = coupled_linearization.momentum_wrt_k.num_rows;
    let k_dimension = scalar_linearization.k_state_jacobian.num_rows;
    let omega_dimension = scalar_linearization.omega_state_jacobian.num_rows;
    if velocity_dimension > flow_dimension
        || coupled_linearization.momentum_wrt_k.num_cols != k_dimension
        || coupled_linearization.momentum_wrt_omega.num_rows != velocity_dimension
        || coupled_linearization.momentum_wrt_omega.num_cols != omega_dimension
        || coupled_linearization.k_wrt_velocity.num_rows != k_dimension
        || coupled_linearization.k_wrt_velocity.num_cols != velocity_dimension
        || coupled_linearization.omega_wrt_velocity.num_rows != omega_dimension
        || coupled_linearization.omega_wrt_velocity.num_cols != velocity_dimension
        || coupled_linearization.k_wrt_k.num_rows != k_dimension
        || coupled_linearization.k_wrt_k.num_cols != k_dimension
        || coupled_linearization.k_wrt_omega.num_rows != k_dimension
        || coupled_linearization.k_wrt_omega.num_cols != omega_dimension
        || coupled_linearization.omega_wrt_k.num_rows != omega_dimension
        || coupled_linearization.omega_wrt_k.num_cols != k_dimension
        || coupled_linearization.omega_wrt_omega.num_rows != omega_dimension
        || coupled_linearization.omega_wrt_omega.num_cols != omega_dimension
        || scalar_linearization.k_state_jacobian.num_cols != k_dimension
        || scalar_linearization.omega_state_jacobian.num_cols != omega_dimension
    {
        return Err(KOmegaCouplingError::InvalidCoupling(
            "flow/scalar coupling block dimensions do not agree",
        ));
    }

    let total_dimension = flow_dimension
        .checked_add(k_dimension)
        .and_then(|dimension| dimension.checked_add(omega_dimension))
        .ok_or(KOmegaCouplingError::InvalidCoupling(
            "monolithic block dimension overflow",
        ))?;
    let k_offset = flow_dimension;
    let omega_offset =
        flow_dimension
            .checked_add(k_dimension)
            .ok_or(KOmegaCouplingError::InvalidCoupling(
                "monolithic block offset overflow",
            ))?;
    let mut triplets = Vec::new();
    append_csr_block(&mut triplets, flow_jacobian, 0, 0)?;
    append_csr_block(
        &mut triplets,
        &coupled_linearization.momentum_wrt_k,
        0,
        k_offset,
    )?;
    append_csr_block(
        &mut triplets,
        &coupled_linearization.momentum_wrt_omega,
        0,
        omega_offset,
    )?;
    append_csr_block(
        &mut triplets,
        &coupled_linearization.k_wrt_velocity,
        k_offset,
        0,
    )?;
    append_csr_block(
        &mut triplets,
        &scalar_linearization.k_state_jacobian,
        k_offset,
        k_offset,
    )?;
    append_csr_block(
        &mut triplets,
        &coupled_linearization.k_wrt_k,
        k_offset,
        k_offset,
    )?;
    append_csr_block(
        &mut triplets,
        &coupled_linearization.k_wrt_omega,
        k_offset,
        omega_offset,
    )?;
    append_csr_block(
        &mut triplets,
        &coupled_linearization.omega_wrt_velocity,
        omega_offset,
        0,
    )?;
    append_csr_block(
        &mut triplets,
        &scalar_linearization.omega_state_jacobian,
        omega_offset,
        omega_offset,
    )?;
    append_csr_block(
        &mut triplets,
        &coupled_linearization.omega_wrt_k,
        omega_offset,
        k_offset,
    )?;
    append_csr_block(
        &mut triplets,
        &coupled_linearization.omega_wrt_omega,
        omega_offset,
        omega_offset,
    )?;
    let state_jacobian = CsrMatrix::try_from_triplets(total_dimension, total_dimension, triplets)
        .map_err(|_| {
        KOmegaCouplingError::InvalidCoupling("monolithic block CSR construction failed")
    })?;
    Ok(KOmegaMonolithicLinearization {
        provenance: monolithic_provenance,
        flow_dimension,
        k_dimension,
        omega_dimension,
        state_jacobian,
    })
}
pub fn assemble_komega_transient_current_linearization(
    flow_jacobian: &CsrMatrix<f64>,
    scalar_linearization: &KOmegaTransportLinearization,
    coupled_linearization: &KOmegaCoupledLinearization,
    coupled_provenance: &'static str,
    monolithic_provenance: &'static str,
) -> Result<CsrMatrix<f64>, KOmegaCouplingError> {
    let zero_coupling = KOmegaCoupledLinearization {
        provenance: coupled_linearization.provenance,
        momentum_wrt_k: CsrMatrix::new(
            coupled_linearization.momentum_wrt_k.num_rows,
            coupled_linearization.momentum_wrt_k.num_cols,
        ),
        momentum_wrt_omega: CsrMatrix::new(
            coupled_linearization.momentum_wrt_omega.num_rows,
            coupled_linearization.momentum_wrt_omega.num_cols,
        ),
        k_wrt_velocity: coupled_linearization.k_wrt_velocity.clone(),
        omega_wrt_velocity: coupled_linearization.omega_wrt_velocity.clone(),
        k_wrt_k: CsrMatrix::new(
            coupled_linearization.k_wrt_k.num_rows,
            coupled_linearization.k_wrt_k.num_cols,
        ),
        k_wrt_omega: CsrMatrix::new(
            coupled_linearization.k_wrt_omega.num_rows,
            coupled_linearization.k_wrt_omega.num_cols,
        ),
        omega_wrt_k: CsrMatrix::new(
            coupled_linearization.omega_wrt_k.num_rows,
            coupled_linearization.omega_wrt_k.num_cols,
        ),
        omega_wrt_omega: CsrMatrix::new(
            coupled_linearization.omega_wrt_omega.num_rows,
            coupled_linearization.omega_wrt_omega.num_cols,
        ),
    };
    Ok(assemble_komega_monolithic_linearization(
        flow_jacobian,
        scalar_linearization,
        &zero_coupling,
        coupled_provenance,
        monolithic_provenance,
    )?
    .state_jacobian)
}
pub fn assemble_komega_transient_history_linearization(
    flow_dimension: usize,
    scalar_linearization: &KOmegaTransportLinearization,
    coupled_linearization: &KOmegaCoupledLinearization,
    history_order: usize,
    coupled_provenance: &'static str,
) -> Result<CsrMatrix<f64>, KOmegaCouplingError> {
    if flow_dimension == 0
        || coupled_linearization.provenance != coupled_provenance
        || scalar_linearization.provenance != K_OMEGA_TRANSPORT_LINEARIZATION_PROVENANCE
    {
        return Err(KOmegaCouplingError::InvalidCoupling(
            "invalid transported history block provenance or flow dimension",
        ));
    }
    let k_dimension = scalar_linearization.k_history_jacobian.num_rows;
    let omega_dimension = scalar_linearization.omega_history_jacobian.num_rows;
    if k_dimension == 0
        || scalar_linearization.k_history_jacobian.num_cols != k_dimension
        || scalar_linearization.omega_history_jacobian.num_cols != omega_dimension
        || coupled_linearization.momentum_wrt_k.num_rows > flow_dimension
        || coupled_linearization.momentum_wrt_omega.num_rows > flow_dimension
        || coupled_linearization.momentum_wrt_k.num_cols != k_dimension
        || coupled_linearization.momentum_wrt_omega.num_cols != omega_dimension
    {
        return Err(KOmegaCouplingError::InvalidCoupling(
            "invalid transported history block dimensions",
        ));
    }
    let (k_history, omega_history) = match history_order {
        1 => (
            &scalar_linearization.k_history_jacobian,
            &scalar_linearization.omega_history_jacobian,
        ),
        2 => (
            &scalar_linearization.k_previous_history_jacobian,
            &scalar_linearization.omega_previous_history_jacobian,
        ),
        _ => {
            return Err(KOmegaCouplingError::InvalidCoupling(
                "BDF history order must be one or two",
            ));
        }
    };
    if k_history.num_rows != k_dimension
        || k_history.num_cols != k_dimension
        || omega_history.num_rows != omega_dimension
        || omega_history.num_cols != omega_dimension
    {
        return Err(KOmegaCouplingError::InvalidCoupling(
            "invalid transported scalar history dimensions",
        ));
    }
    let total_dimension = flow_dimension
        .checked_add(k_dimension)
        .and_then(|dimension| dimension.checked_add(omega_dimension))
        .ok_or(KOmegaCouplingError::InvalidCoupling(
            "transported history dimension overflow",
        ))?;
    let k_offset = flow_dimension;
    let omega_offset = flow_dimension + k_dimension;
    let mut triplets = Vec::new();
    if history_order == 1 {
        append_csr_block(
            &mut triplets,
            &coupled_linearization.momentum_wrt_k,
            0,
            k_offset,
        )?;
        append_csr_block(
            &mut triplets,
            &coupled_linearization.momentum_wrt_omega,
            0,
            omega_offset,
        )?;
    }
    append_csr_block(&mut triplets, k_history, k_offset, k_offset)?;
    append_csr_block(&mut triplets, omega_history, omega_offset, omega_offset)?;
    CsrMatrix::try_from_triplets(total_dimension, total_dimension, triplets).map_err(|_| {
        KOmegaCouplingError::InvalidCoupling("transported history CSR construction failed")
    })
}
pub fn assemble_komega_monolithic_history_linearization(
    flow_dimension: usize,
    scalar_linearization: &KOmegaTransportLinearization,
    history_order: usize,
) -> Result<CsrMatrix<f64>, KOmegaCouplingError> {
    if scalar_linearization.provenance != K_OMEGA_TRANSPORT_LINEARIZATION_PROVENANCE
        || flow_dimension == 0
        || !matches!(history_order, 1 | 2)
    {
        return Err(KOmegaCouplingError::InvalidCoupling(
            "invalid monolithic transported history contract",
        ));
    }
    let (k_history, omega_history) = if history_order == 1 {
        (
            &scalar_linearization.k_history_jacobian,
            &scalar_linearization.omega_history_jacobian,
        )
    } else {
        (
            &scalar_linearization.k_previous_history_jacobian,
            &scalar_linearization.omega_previous_history_jacobian,
        )
    };
    let k_dimension = k_history.num_rows;
    let omega_dimension = omega_history.num_rows;
    if k_dimension == 0
        || omega_dimension == 0
        || k_history.num_cols != k_dimension
        || omega_history.num_cols != omega_dimension
    {
        return Err(KOmegaCouplingError::InvalidCoupling(
            "invalid monolithic scalar history dimensions",
        ));
    }
    let total_dimension = flow_dimension
        .checked_add(k_dimension)
        .and_then(|dimension| dimension.checked_add(omega_dimension))
        .ok_or(KOmegaCouplingError::InvalidCoupling(
            "monolithic history dimension overflow",
        ))?;
    let mut triplets = Vec::new();
    append_csr_block(&mut triplets, k_history, flow_dimension, flow_dimension)?;
    append_csr_block(
        &mut triplets,
        omega_history,
        flow_dimension + k_dimension,
        flow_dimension + k_dimension,
    )?;
    CsrMatrix::try_from_triplets(total_dimension, total_dimension, triplets).map_err(|_| {
        KOmegaCouplingError::InvalidCoupling(
            "monolithic transported history CSR construction failed",
        )
    })
}
/// P1 shape gradients on one tetrahedral element, shared by turbulence
/// coupling assembly and element-field derivation.
///
/// Returns the four gradients in node order, or a coupling error for an
/// out-of-range or degenerate element.
pub fn tetrahedron_shape_gradients(
    mesh: &Mesh,
    vertices: &[usize],
) -> Result<[[f64; 3]; 4], KOmegaCouplingError> {
    let points = vertices
        .iter()
        .map(|&node| mesh.nodes.get(node).copied())
        .collect::<Option<Vec<_>>>()
        .ok_or(KOmegaCouplingError::Fem(
            KOmegaTransportError::InvalidConfig("element vertex index is out of range"),
        ))?;
    let a = vector_between(points[1], points[0]);
    let b = vector_between(points[2], points[0]);
    let c = vector_between(points[3], points[0]);
    let determinant = dot(a, cross(b, c));
    if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
        return Err(KOmegaCouplingError::Fem(
            KOmegaTransportError::InvalidConfig("degenerate tetrahedral element"),
        ));
    }
    Ok([
        scale(
            add(add(cross(b, c), cross(c, a)), cross(a, b)),
            -1.0 / determinant,
        ),
        scale(cross(b, c), 1.0 / determinant),
        scale(cross(c, a), 1.0 / determinant),
        scale(cross(a, b), 1.0 / determinant),
    ])
}

fn vector_between(a: crate::mesh::Point, b: crate::mesh::Point) -> [f64; 3] {
    [a.x - b.x, a.y - b.y, a.z - b.z]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(value: [f64; 3], factor: f64) -> [f64; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

/// Advance a transferred P1 k--omega BDF state on a target tetrahedral mesh.
///
/// `config` contains frozen mean-flow coefficients for this Picard scalar
/// update.  The target operator is assembled from the target mesh every call;
/// no source-topology matrix or residual is reused.
struct MomentumScalarCrossForm {
    density: f64,
    local_velocity: Vec<f64>,
    d_dynamic_viscosity: Vec<f64>,
}

impl BilinearForm<f64> for MomentumScalarCrossForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let mut velocity_gradient = [[0.0; 3]; 3];
        for dof in 0..test.n_dofs.min(self.local_velocity.len()) {
            let component = dof % 3;
            for (direction, gradient) in velocity_gradient[component]
                .iter_mut()
                .enumerate()
                .take(test.spatial_dim.min(3))
            {
                *gradient += self.local_velocity[dof] * test.grad(dof, direction);
            }
        }

        let scale = weight * test.det_j * self.density;
        for row in 0..test.n_dofs {
            let component = row % 3;
            let gradient_contraction = (0..test.spatial_dim.min(3))
                .map(|direction| {
                    velocity_gradient[component][direction] * test.grad(row, direction)
                })
                .sum::<f64>();
            for column in 0..trial.n_dofs.min(self.d_dynamic_viscosity.len()) {
                local_matrix.add(
                    row,
                    column,
                    scale
                        * self.d_dynamic_viscosity[column]
                        * test.vector_value(row, component)
                        * gradient_contraction,
                );
            }
        }
    }
}

#[derive(Debug, Clone)]
struct ScalarVelocityCrossForm {
    scalar_gradient: [f64; 3],
    velocity_mean_derivative: Vec<[f64; 3]>,
    production_derivative: Vec<f64>,
    production_scale: f64,
}

impl BilinearForm<f64> for ScalarVelocityCrossForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let scale = weight * test.det_j;
        for row in 0..test.n_dofs {
            for column in 0..trial
                .n_dofs
                .min(self.velocity_mean_derivative.len())
                .min(self.production_derivative.len())
            {
                let advection = (0..3)
                    .map(|direction| {
                        self.velocity_mean_derivative[column][direction]
                            * self.scalar_gradient[direction]
                    })
                    .sum::<f64>();
                let source = self.production_scale * self.production_derivative[column];
                local_matrix.add(row, column, scale * test.value(row) * (advection - source));
            }
        }
    }
}

/// Derivative of one scalar transport residual with respect to a scalar
/// coefficient field.  The transport solve freezes these coefficients for a
/// segregated update; this form publishes the missing Newton correction when
/// the element fields are recomputed from the current k--omega iterate.
#[derive(Debug, Clone)]
struct ScalarStateDerivativeForm {
    scalar_gradient: [f64; 3],
    scalar_values: Vec<f64>,
    diffusivity_derivative: Vec<f64>,
    reaction_derivative: Vec<f64>,
    source_derivative: Vec<f64>,
}

impl BilinearForm<f64> for ScalarStateDerivativeForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let scale = weight * test.det_j;
        let scalar_value = (0..trial.n_dofs.min(self.scalar_values.len()))
            .map(|column| self.scalar_values[column] * trial.value(column))
            .sum::<f64>();
        for row in 0..test.n_dofs {
            let test_value = test.value(row);
            let gradient_contraction = (0..test.spatial_dim.min(3))
                .map(|direction| self.scalar_gradient[direction] * test.grad(row, direction))
                .sum::<f64>();
            for column in 0..trial
                .n_dofs
                .min(self.diffusivity_derivative.len())
                .min(self.reaction_derivative.len())
                .min(self.source_derivative.len())
            {
                let value = self.diffusivity_derivative[column] * gradient_contraction
                    + test_value * self.reaction_derivative[column] * scalar_value
                    - test_value * self.source_derivative[column];
                local_matrix.add(row, column, scale * value);
            }
        }
    }
}

/// Assemble the exact coupling blocks for the transported PC-fan Newton/BDF
/// update.
///
/// The mean-flow block differentiates the element dynamic viscosity
/// `rho*k/omega`; scalar/velocity blocks differentiate the element advection
/// velocity and strain-production source. Scalar-state blocks include local
/// average diffusion/source terms and dense global-mean reaction/limited-source
/// terms.
#[allow(clippy::too_many_arguments)]
pub fn assemble_komega_coupled_linearization(
    mesh: &Mesh,
    velocity_space: &VectorH1Space,
    scalar_space: &H1Space,
    velocity: &[f64],
    turbulence: &KOmegaBdfState,
    fields: &KOmegaTransportElementFields,
    density: f64,
    limiter: f64,
    provenance: &'static str,
) -> Result<KOmegaCoupledLinearization, KOmegaCouplingError> {
    if mesh.dimension != 3 || velocity_space.num_components() != 3 {
        return Err(KOmegaCouplingError::InvalidCoupling(
            "coupling requires a three-dimensional vector space",
        ));
    }
    if !density.is_finite() || density <= 0.0 || !limiter.is_finite() || limiter <= 0.0 {
        return Err(KOmegaCouplingError::InvalidCoupling(
            "density and viscosity limiter must be finite and positive",
        ));
    }
    let expected_velocity = velocity_space.num_global_dofs(mesh);
    let expected_scalar = scalar_space.num_global_dofs(mesh);
    if expected_scalar == 0 {
        return Err(KOmegaCouplingError::InvalidCoupling(
            "coupling requires at least one scalar DOF",
        ));
    }
    if velocity.len() != expected_velocity {
        return Err(KOmegaCouplingError::Fem(
            KOmegaTransportError::DimensionMismatch {
                field: "velocity",
                expected: expected_velocity,
                actual: velocity.len(),
            },
        ));
    }
    if turbulence.k.len() != expected_scalar
        || turbulence.k_prev.len() != expected_scalar
        || turbulence.omega.len() != expected_scalar
        || turbulence.omega_prev.len() != expected_scalar
    {
        return Err(KOmegaCouplingError::Fem(
            KOmegaTransportError::DimensionMismatch {
                field: "transported scalar",
                expected: expected_scalar,
                actual: turbulence.k.len(),
            },
        ));
    }
    if fields.velocity.len() != mesh.num_elements()
        || fields.eddy_viscosity.len() != mesh.num_elements()
        || fields.k_production.len() != mesh.num_elements()
        || fields.omega_production.len() != mesh.num_elements()
    {
        return Err(KOmegaCouplingError::InvalidCoupling(
            "element transport fields have the wrong length",
        ));
    }

    if turbulence
        .k
        .iter()
        .chain(&turbulence.omega)
        .any(|value| !value.is_finite() || *value <= 0.0)
        || fields
            .eddy_viscosity
            .iter()
            .chain(&fields.k_production)
            .chain(&fields.omega_production)
            .any(|value| !value.is_finite() || *value < 0.0)
        || fields
            .velocity
            .iter()
            .any(|value| value.iter().any(|component| !component.is_finite()))
    {
        return Err(KOmegaCouplingError::InvalidCoupling(
            "coupling state and fields must be finite and positive",
        ));
    }

    let mut momentum_k_forms = Vec::with_capacity(mesh.num_elements());
    let mut momentum_omega_forms = Vec::with_capacity(mesh.num_elements());
    let mut k_velocity_forms = Vec::with_capacity(mesh.num_elements());
    let mut omega_velocity_forms = Vec::with_capacity(mesh.num_elements());
    let mut k_k_forms = Vec::with_capacity(mesh.num_elements());
    let mut k_omega_forms = Vec::with_capacity(mesh.num_elements());
    let mut omega_k_forms = Vec::with_capacity(mesh.num_elements());
    let mut omega_omega_forms = Vec::with_capacity(mesh.num_elements());
    let mut k_global_source_k_forms = Vec::with_capacity(mesh.num_elements());
    let mut k_global_source_omega_forms = Vec::with_capacity(mesh.num_elements());
    let scalar_local_space = H1Space::new(PolynomialDegree::P1);
    let constants = KOmegaSstConstants::default();
    let state_count = expected_scalar as f64;
    let global_mean_k = turbulence.k.iter().sum::<f64>() / state_count;
    let global_mean_omega = turbulence.omega.iter().sum::<f64>() / state_count;
    let k_reaction_wrt_omega = if global_mean_omega > 0.0 {
        constants.beta_star / state_count
    } else {
        0.0
    };
    let omega_reaction_wrt_omega = if global_mean_omega > 0.0 {
        constants.beta / state_count
    } else {
        0.0
    };
    let production_limit =
        10.0 * constants.beta_star * global_mean_k.max(0.0) * global_mean_omega.max(0.0);
    let production_limit_wrt_k = if global_mean_k > 0.0 && global_mean_omega > 0.0 {
        10.0 * constants.beta_star * global_mean_omega / state_count
    } else {
        0.0
    };
    let production_limit_wrt_omega = if global_mean_k > 0.0 && global_mean_omega > 0.0 {
        10.0 * constants.beta_star * global_mean_k / state_count
    } else {
        0.0
    };

    for (element_index, element) in mesh.elements.iter().enumerate() {
        if element.nodes.len() != 4 {
            return Err(KOmegaCouplingError::InvalidCoupling(
                "coupling requires affine tetrahedral elements",
            ));
        }
        let scalar_dofs = scalar_space.local_to_global(mesh, element_index);
        if scalar_dofs.len() != 4 {
            return Err(KOmegaCouplingError::InvalidCoupling(
                "coupling requires P1 scalar elements",
            ));
        }
        let velocity_dofs = velocity_space.local_to_global(mesh, element_index);
        let local_velocity = velocity_dofs
            .iter()
            .map(|&dof| velocity[dof])
            .collect::<Vec<_>>();
        let gradients = tetrahedron_shape_gradients(mesh, &element.nodes)?;

        let element_mean_k = element
            .nodes
            .iter()
            .map(|&node| turbulence.k[node])
            .sum::<f64>()
            / 4.0;
        let element_mean_omega = element
            .nodes
            .iter()
            .map(|&node| turbulence.omega[node])
            .sum::<f64>()
            / 4.0;
        let denominator = element_mean_omega.max(limiter);
        let d_k = density / (4.0 * denominator);
        let d_omega = if element_mean_omega > limiter {
            -density * element_mean_k / (4.0 * denominator * denominator)
        } else {
            0.0
        };
        momentum_k_forms.push(MomentumScalarCrossForm {
            density: 1.0,
            local_velocity: local_velocity.clone(),
            d_dynamic_viscosity: vec![d_k; scalar_dofs.len()],
        });
        momentum_omega_forms.push(MomentumScalarCrossForm {
            density: 1.0,
            local_velocity,
            d_dynamic_viscosity: vec![d_omega; scalar_dofs.len()],
        });

        let scalar_local_dofs = scalar_local_space.local_to_global(mesh, element_index);
        if scalar_local_dofs != scalar_dofs {
            return Err(KOmegaCouplingError::InvalidCoupling(
                "scalar space must use the P1 nodal layout",
            ));
        }
        let mut scalar_gradient = [0.0; 3];
        let mut omega_gradient = [0.0; 3];
        for (local_node, &node) in element.nodes.iter().enumerate() {
            for direction in 0..3 {
                scalar_gradient[direction] += turbulence.k[node] * gradients[local_node][direction];
                omega_gradient[direction] +=
                    turbulence.omega[node] * gradients[local_node][direction];
            }
        }

        let mut velocity_mean_derivative = vec![[0.0; 3]; velocity_dofs.len()];
        let mut k_production_derivative = vec![0.0; velocity_dofs.len()];
        let mut omega_production_derivative = vec![0.0; velocity_dofs.len()];
        let mut strain = [[0.0; 3]; 3];
        let mut velocity_gradient = [[0.0; 3]; 3];
        for (local_node, &node) in element.nodes.iter().enumerate() {
            for component in 0..3 {
                let global_dof = node * 3 + component;
                let Some(local_dof) = velocity_dofs.iter().position(|&dof| dof == global_dof)
                else {
                    continue;
                };
                let value = velocity.get(global_dof).copied().ok_or(
                    KOmegaCouplingError::InvalidCoupling("vertex velocity layout mismatch"),
                )?;
                velocity_mean_derivative[local_dof][component] = 0.25;
                for direction in 0..3 {
                    velocity_gradient[component][direction] +=
                        value * gradients[local_node][direction];
                }
            }
        }
        for row in 0..3 {
            for column in 0..3 {
                strain[row][column] =
                    0.5 * (velocity_gradient[row][column] + velocity_gradient[column][row]);
            }
        }
        let eddy_viscosity = fields.eddy_viscosity[element_index];
        let k_floor = element_mean_k.max(1.0e-12);
        for (local_node, &node) in element.nodes.iter().enumerate() {
            for component in 0..3 {
                let global_dof = node * 3 + component;
                let Some(local_dof) = velocity_dofs.iter().position(|&dof| dof == global_dof)
                else {
                    continue;
                };
                let mut derivative_strain = [[0.0; 3]; 3];
                for direction in 0..3 {
                    derivative_strain[component][direction] +=
                        0.5 * gradients[local_node][direction];
                    derivative_strain[direction][component] +=
                        0.5 * gradients[local_node][direction];
                }
                let derivative_production = 4.0
                    * eddy_viscosity
                    * (0..3)
                        .flat_map(|row| (0..3).map(move |column| (row, column)))
                        .map(|(row, column)| strain[row][column] * derivative_strain[row][column])
                        .sum::<f64>();
                k_production_derivative[local_dof] = derivative_production;
                omega_production_derivative[local_dof] = derivative_production / k_floor;
            }
        }

        let strain_square = strain
            .iter()
            .flat_map(|row| row.iter())
            .map(|value| value * value)
            .sum::<f64>();
        let element_count = scalar_dofs.len() as f64;
        let d_eddy_viscosity_d_k = if element_mean_k > 0.0 {
            1.0 / (element_count * denominator)
        } else {
            0.0
        };
        let d_eddy_viscosity_d_omega = if element_mean_k > 0.0 && element_mean_omega > limiter {
            -element_mean_k / (element_count * denominator * denominator)
        } else {
            0.0
        };
        let production_scale = 2.0 * strain_square;
        let k_production = fields.k_production[element_index];
        let k_source_is_limited = k_production >= production_limit;
        let k_source_wrt_k = if k_source_is_limited {
            0.0
        } else {
            production_scale * d_eddy_viscosity_d_k
        };
        let k_source_wrt_omega = if k_source_is_limited {
            0.0
        } else {
            production_scale * d_eddy_viscosity_d_omega
        };
        let omega_source_wrt_k = if element_mean_k > 0.0 {
            production_scale
                * (d_eddy_viscosity_d_k / element_mean_k
                    - eddy_viscosity / (element_count * element_mean_k * element_mean_k))
        } else {
            0.0
        };
        let omega_source_wrt_omega = production_scale * d_eddy_viscosity_d_omega / k_floor;

        if k_source_is_limited {
            k_global_source_k_forms.push(TurbulenceProductionForm {
                production: production_limit_wrt_k,
            });
            k_global_source_omega_forms.push(TurbulenceProductionForm {
                production: production_limit_wrt_omega,
            });
        } else {
            k_global_source_k_forms.push(TurbulenceProductionForm { production: 0.0 });
            k_global_source_omega_forms.push(TurbulenceProductionForm { production: 0.0 });
        }

        k_k_forms.push(ScalarStateDerivativeForm {
            scalar_gradient,
            scalar_values: element
                .nodes
                .iter()
                .map(|&node| turbulence.k[node])
                .collect(),
            diffusivity_derivative: vec![
                d_eddy_viscosity_d_k / constants.sigma_k;
                scalar_dofs.len()
            ],
            reaction_derivative: vec![0.0; scalar_dofs.len()],
            source_derivative: vec![k_source_wrt_k; scalar_dofs.len()],
        });
        k_omega_forms.push(ScalarStateDerivativeForm {
            scalar_gradient,
            scalar_values: element
                .nodes
                .iter()
                .map(|&node| turbulence.k[node])
                .collect(),
            diffusivity_derivative: vec![
                d_eddy_viscosity_d_omega / constants.sigma_k;
                scalar_dofs.len()
            ],
            reaction_derivative: vec![0.0; scalar_dofs.len()],
            source_derivative: vec![k_source_wrt_omega; scalar_dofs.len()],
        });
        omega_k_forms.push(ScalarStateDerivativeForm {
            scalar_gradient: omega_gradient,
            scalar_values: element
                .nodes
                .iter()
                .map(|&node| turbulence.omega[node])
                .collect(),
            diffusivity_derivative: vec![
                d_eddy_viscosity_d_k / constants.sigma_omega;
                scalar_dofs.len()
            ],
            reaction_derivative: vec![0.0; scalar_dofs.len()],
            source_derivative: vec![omega_source_wrt_k; scalar_dofs.len()],
        });
        omega_omega_forms.push(ScalarStateDerivativeForm {
            scalar_gradient: omega_gradient,
            scalar_values: element
                .nodes
                .iter()
                .map(|&node| turbulence.omega[node])
                .collect(),
            diffusivity_derivative: vec![
                d_eddy_viscosity_d_omega / constants.sigma_omega;
                scalar_dofs.len()
            ],
            reaction_derivative: vec![0.0; scalar_dofs.len()],
            source_derivative: vec![omega_source_wrt_omega; scalar_dofs.len()],
        });

        k_velocity_forms.push(ScalarVelocityCrossForm {
            scalar_gradient,
            velocity_mean_derivative: velocity_mean_derivative.clone(),
            production_derivative: k_production_derivative,
            production_scale: 1.0,
        });
        omega_velocity_forms.push(ScalarVelocityCrossForm {
            scalar_gradient,
            velocity_mean_derivative,
            production_derivative: omega_production_derivative,
            production_scale: 1.0,
        });
    }

    let quad_order = crate::quadrature::required_order_for_stiffness(2);
    let momentum_k = assemble_bilinear_two_spaces_per_element(
        &momentum_k_forms,
        velocity_space,
        scalar_space,
        mesh,
        quad_order,
    )
    .to_csr();
    let momentum_omega = assemble_bilinear_two_spaces_per_element(
        &momentum_omega_forms,
        velocity_space,
        scalar_space,
        mesh,
        quad_order,
    )
    .to_csr();
    let scalar_quad = crate::quadrature::required_order_for_stiffness(1);
    let k_velocity = assemble_bilinear_two_spaces_per_element(
        &k_velocity_forms,
        scalar_space,
        velocity_space,
        mesh,
        scalar_quad,
    )
    .to_csr();
    let omega_velocity = assemble_bilinear_two_spaces_per_element(
        &omega_velocity_forms,
        scalar_space,
        velocity_space,
        mesh,
        scalar_quad,
    )
    .to_csr();
    let k_wrt_k = assemble_bilinear_two_spaces_per_element(
        &k_k_forms,
        scalar_space,
        scalar_space,
        mesh,
        scalar_quad,
    )
    .to_csr();
    let k_wrt_omega = assemble_bilinear_two_spaces_per_element(
        &k_omega_forms,
        scalar_space,
        scalar_space,
        mesh,
        scalar_quad,
    )
    .to_csr();
    let omega_wrt_k = assemble_bilinear_two_spaces_per_element(
        &omega_k_forms,
        scalar_space,
        scalar_space,
        mesh,
        scalar_quad,
    )
    .to_csr();
    let omega_wrt_omega = assemble_bilinear_two_spaces_per_element(
        &omega_omega_forms,
        scalar_space,
        scalar_space,
        mesh,
        scalar_quad,
    )
    .to_csr();

    let scalar_mass_quad = crate::quadrature::required_order_for_mass(1);
    let scalar_mass = assemble_bilinear(
        &MassForm,
        scalar_space,
        scalar_space,
        mesh,
        scalar_mass_quad,
    )
    .to_csr();
    let k_reaction_row = scalar_mass.apply(&turbulence.k);
    let omega_reaction_row = scalar_mass.apply(&turbulence.omega);
    let k_global_source_k = assemble_linear_per_element(
        &k_global_source_k_forms,
        scalar_space,
        mesh,
        scalar_mass_quad,
    );
    let k_global_source_omega = assemble_linear_per_element(
        &k_global_source_omega_forms,
        scalar_space,
        mesh,
        scalar_mass_quad,
    );
    let k_wrt_k_global = k_global_source_k
        .iter()
        .map(|&value| -value)
        .collect::<Vec<_>>();
    let k_wrt_omega_global = k_global_source_omega
        .iter()
        .zip(&k_reaction_row)
        .map(|(&source, &reaction)| -source + k_reaction_wrt_omega * reaction)
        .collect::<Vec<_>>();
    let omega_wrt_omega_global = omega_reaction_row
        .iter()
        .map(|&reaction| omega_reaction_wrt_omega * reaction)
        .collect::<Vec<_>>();
    let k_wrt_k = add_global_state_column_coupling(k_wrt_k, &k_wrt_k_global)?;
    let k_wrt_omega = add_global_state_column_coupling(k_wrt_omega, &k_wrt_omega_global)?;
    let omega_wrt_omega =
        add_global_state_column_coupling(omega_wrt_omega, &omega_wrt_omega_global)?;

    Ok(KOmegaCoupledLinearization {
        provenance,
        momentum_wrt_k: momentum_k,
        momentum_wrt_omega: momentum_omega,
        k_wrt_velocity: k_velocity,
        omega_wrt_velocity: omega_velocity,
        k_wrt_k,
        k_wrt_omega,
        omega_wrt_k,
        omega_wrt_omega,
    })
}

fn add_global_state_column_coupling(
    matrix: CsrMatrix<f64>,
    row_weights: &[f64],
) -> Result<CsrMatrix<f64>, KOmegaCouplingError> {
    if row_weights.len() != matrix.num_rows {
        return Err(KOmegaCouplingError::InvalidCoupling(
            "global scalar coupling row dimension mismatch",
        ));
    }
    let mut triplets =
        Vec::with_capacity(matrix.values.len() + matrix.num_rows.saturating_mul(matrix.num_cols));
    append_csr_block(&mut triplets, &matrix, 0, 0)?;
    for (row, &weight) in row_weights.iter().enumerate() {
        if !weight.is_finite() {
            return Err(KOmegaCouplingError::InvalidCoupling(
                "global scalar coupling contains non-finite row",
            ));
        }
        for column in 0..matrix.num_cols {
            if weight != 0.0 {
                triplets.push((row, column, weight));
            }
        }
    }
    CsrMatrix::try_from_triplets(matrix.num_rows, matrix.num_cols, triplets).map_err(|_| {
        KOmegaCouplingError::InvalidCoupling("global scalar coupling CSR construction failed")
    })
}

#[cfg(test)]
mod tests {
    use super::super::transport::KOmegaTransportConfig;
    use super::*;
    use crate::mesh::generators::unit_cube_tetrahedra;

    fn one_by_one(value: f64) -> CsrMatrix<f64> {
        CsrMatrix::try_from_triplets(1, 1, vec![(0, 0, value)]).expect("1x1 block")
    }

    fn scalar_linearization(value: f64) -> KOmegaTransportLinearization {
        KOmegaTransportLinearization {
            provenance: K_OMEGA_TRANSPORT_LINEARIZATION_PROVENANCE,
            k_state_jacobian: one_by_one(value),
            omega_state_jacobian: one_by_one(value),
            k_history_jacobian: one_by_one(value),
            k_previous_history_jacobian: one_by_one(value),
            omega_history_jacobian: one_by_one(value),
            omega_previous_history_jacobian: one_by_one(value),
        }
    }

    fn empty_coupling() -> KOmegaCoupledLinearization {
        KOmegaCoupledLinearization {
            provenance: "test:coupling:v1",
            momentum_wrt_k: one_by_one(0.0),
            momentum_wrt_omega: one_by_one(0.0),
            k_wrt_velocity: one_by_one(0.0),
            omega_wrt_velocity: one_by_one(0.0),
            k_wrt_k: one_by_one(0.0),
            k_wrt_omega: one_by_one(0.0),
            omega_wrt_k: one_by_one(0.0),
            omega_wrt_omega: one_by_one(0.0),
        }
    }

    #[test]
    fn monolithic_block_composes_diagonal_entries() {
        let flow = one_by_one(2.0);
        let scalar = scalar_linearization(3.0);
        let coupled = empty_coupling();
        let monolithic = assemble_komega_monolithic_linearization(
            &flow,
            &scalar,
            &coupled,
            "test:coupling:v1",
            "test:monolithic:v1",
        )
        .expect("block composition");
        assert_eq!(monolithic.provenance, "test:monolithic:v1");
        assert_eq!(monolithic.flow_dimension, 1);
        assert_eq!(
            monolithic.state_jacobian.apply(&[1.0, 0.0, 0.0]),
            vec![2.0, 0.0, 0.0]
        );
        assert_eq!(
            monolithic.state_jacobian.apply(&[0.0, 1.0, 0.0]),
            vec![0.0, 3.0, 0.0]
        );
        assert_eq!(
            monolithic.state_jacobian.apply(&[0.0, 0.0, 1.0]),
            vec![0.0, 0.0, 3.0]
        );
    }

    #[test]
    fn wrong_provenance_fails_closed() {
        let flow = one_by_one(2.0);
        let scalar = scalar_linearization(3.0);
        let mut coupled = empty_coupling();
        coupled.provenance = "other:coupling:v1";
        assert!(matches!(
            assemble_komega_monolithic_linearization(
                &flow,
                &scalar,
                &coupled,
                "test:coupling:v1",
                "test:monolithic:v1"
            ),
            Err(KOmegaCouplingError::InvalidCoupling(_))
        ));
        assert!(matches!(
            assemble_komega_transient_history_linearization(
                1,
                &scalar,
                &coupled,
                1,
                "test:coupling:v1"
            ),
            Err(KOmegaCouplingError::InvalidCoupling(_))
        ));
        assert!(matches!(
            assemble_komega_transient_history_linearization(
                1,
                &scalar,
                &empty_coupling(),
                3,
                "test:coupling:v1"
            ),
            Err(KOmegaCouplingError::InvalidCoupling(_))
        ));
    }

    #[test]
    fn unit_tet_shape_gradients_match_closed_form() {
        use crate::mesh::{ElementType, Mesh, Point};
        let mut mesh = Mesh::new(3);
        for point in [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ] {
            mesh.add_node(Point {
                x: point[0],
                y: point[1],
                z: point[2],
            });
        }
        mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
        let gradients =
            tetrahedron_shape_gradients(&mesh, &[0, 1, 2, 3]).expect("unit tet gradients");
        let expected = [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        for (got, want) in gradients.iter().zip(&expected) {
            for (g, w) in got.iter().zip(want) {
                assert!((g - w).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn coupled_assembly_is_finite_on_unit_cube() {
        let mesh = unit_cube_tetrahedra(1);
        let velocity_space = VectorH1Space::new(PolynomialDegree::P1, 3);
        let scalar_space = H1Space::new(PolynomialDegree::P1);
        let velocity = vec![1.0; velocity_space.num_global_dofs(&mesh)];
        let n_scalar = scalar_space.num_global_dofs(&mesh);
        let turbulence = KOmegaBdfState {
            k: vec![0.5; n_scalar],
            k_prev: vec![0.4; n_scalar],
            omega: vec![2.0; n_scalar],
            omega_prev: vec![2.5; n_scalar],
            time: 0.01,
            step: 1,
        };
        let config = KOmegaTransportConfig::default();
        let fields = KOmegaTransportElementFields::uniform(&mesh, &config);
        let coupled = assemble_komega_coupled_linearization(
            &mesh,
            &velocity_space,
            &scalar_space,
            &velocity,
            &turbulence,
            &fields,
            1.2,
            10.0,
            "test:coupling:v1",
        )
        .expect("coupled assembly");
        assert_eq!(coupled.provenance, "test:coupling:v1");
        for matrix in [
            &coupled.momentum_wrt_k,
            &coupled.momentum_wrt_omega,
            &coupled.k_wrt_velocity,
            &coupled.omega_wrt_velocity,
            &coupled.k_wrt_k,
            &coupled.k_wrt_omega,
            &coupled.omega_wrt_k,
            &coupled.omega_wrt_omega,
        ] {
            assert!(matrix.values.iter().all(|v| v.is_finite()));
            assert_eq!(matrix.row_ptrs.len(), matrix.num_rows + 1);
        }
    }
}
