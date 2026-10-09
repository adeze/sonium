//! Bounded NVIDIA CUDA execution for the linear P1 plane-stress bracket.
//!
//! Host FEM assembly is deliberately retained as the deterministic f64
//! reference boundary. The constrained sparse operator, forward solve,
//! transpose-adjoint solve, and residual products are checked CUDA operations.

use crate::{
    BracketStudy, PlaneModel, StaticBracketResult, StressRecoveryPolicy,
    StructuralExecutionProvenance, StructuralMechanicsError, StructuralPolynomialOrder,
    apply_tip_edge_load_order, assemble_structural_matrices, boundary_component_dofs,
    component_force, dot, free_residual_norm, max_von_mises_for_order, nodal_displacement,
    reaction_force, run_static_bracket, triplet_matvec, validate_mesh, validate_study,
};
use serde::{Deserialize, Serialize};
use sonium_compute::{
    AdjointDeviceCsr, Backend, BackendError, CudaBackend, DeviceAdjointOperator, DeviceOperator,
    GmresBackendConfig, GmresBackendError, HostCsrRef, PointJacobiBackend, PrebakedAdjointSpmv,
    gmres_backend_preconditioned,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_homogeneous_dirichlet};
use sonium_fem::physics::solid::elasticity::ElasticityForm;
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

pub const CUDA_STATIC_BRACKET_SCHEMA_VERSION: u32 = 1;

/// Cooperative cancellation checked at every safe host/device boundary.
#[derive(Debug, Clone, Default)]
pub struct CudaCancellationToken(Arc<AtomicBool>);

impl CudaCancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Runtime controls and declared parity budgets for the bounded CUDA slice.
#[derive(Debug, Clone)]
pub struct CudaStaticBracketConfig {
    pub device_ordinal: usize,
    pub max_iterations: usize,
    pub relative_residual_tolerance: f64,
    pub displacement_relative_tolerance: f64,
    pub stress_relative_tolerance: f64,
    pub energy_relative_tolerance: f64,
    pub gradient_relative_tolerance: f64,
    pub cancellation: CudaCancellationToken,
}

impl Default for CudaStaticBracketConfig {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            max_iterations: 800,
            relative_residual_tolerance: 1.0e-10,
            displacement_relative_tolerance: 5.0e-6,
            stress_relative_tolerance: 5.0e-6,
            energy_relative_tolerance: 5.0e-6,
            gradient_relative_tolerance: 5.0e-6,
            cancellation: CudaCancellationToken::default(),
        }
    }
}

/// Typed failures: no backend failure is converted into an implicit CPU solve.
#[derive(Debug, Clone, PartialEq)]
pub enum StructuralCudaError {
    InvalidConfiguration {
        field: &'static str,
        value: String,
    },
    CpuReference {
        message: String,
    },
    BackendUnavailable {
        device_ordinal: usize,
        message: String,
    },
    Allocation {
        stage: &'static str,
        message: String,
    },
    DeviceLost {
        stage: &'static str,
        message: String,
    },
    Backend {
        stage: &'static str,
        message: String,
    },
    HostFallbackRejected {
        stage: &'static str,
    },
    NonConvergence {
        stage: &'static str,
        iterations: usize,
        residual: f64,
    },
    Cancelled {
        stage: &'static str,
    },
    PhysicalGate {
        check: &'static str,
        value: f64,
        limit: f64,
    },
}

impl std::fmt::Display for StructuralCudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration { field, value } => {
                write!(f, "invalid CUDA structural configuration {field}={value}")
            }
            Self::CpuReference { message } => {
                write!(f, "CPU structural reference failed: {message}")
            }
            Self::BackendUnavailable {
                device_ordinal,
                message,
            } => write!(f, "CUDA device {device_ordinal} is unavailable: {message}"),
            Self::Allocation { stage, message } => {
                write!(
                    f,
                    "CUDA structural allocation failed during {stage}: {message}"
                )
            }
            Self::DeviceLost { stage, message } => {
                write!(f, "CUDA device lost during {stage}: {message}")
            }
            Self::Backend { stage, message } => {
                write!(
                    f,
                    "CUDA structural backend failed during {stage}: {message}"
                )
            }
            Self::HostFallbackRejected { stage } => {
                write!(f, "CUDA structural {stage} would use a host fallback")
            }
            Self::NonConvergence {
                stage,
                iterations,
                residual,
            } => write!(
                f,
                "CUDA structural {stage} did not converge after {iterations} iterations: residual={residual}"
            ),
            Self::Cancelled { stage } => {
                write!(f, "CUDA structural execution cancelled at {stage}")
            }
            Self::PhysicalGate {
                check,
                value,
                limit,
            } => write!(
                f,
                "CUDA structural gate {check} failed: value={value}, limit={limit}"
            ),
        }
    }
}

impl std::error::Error for StructuralCudaError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CudaDerivativeEvidence {
    pub design_variable: String,
    pub objective: String,
    pub transpose_adjoint_executed: bool,
    pub adjoint_iterations: usize,
    pub adjoint_true_relative_residual: f64,
    pub adjoint_state_relative_error: f64,
    pub adjoint_gradient_j_per_m: f64,
    pub independent_gradient_j_per_m: f64,
    pub gradient_relative_error: f64,
    pub accepted: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CudaStaticBracketEvidence {
    pub schema_version: u32,
    pub backend: String,
    pub device_ordinal: usize,
    pub precision: String,
    pub element: String,
    pub polynomial_order: String,
    pub regime: String,
    pub solver: String,
    pub preconditioner: String,
    pub fallback_used: bool,
    pub mesh_nodes: usize,
    pub mesh_elements: usize,
    pub dofs: usize,
    pub forward_iterations: usize,
    pub forward_projected_relative_residual: f64,
    pub forward_true_relative_residual: f64,
    pub displacement_cpu_relative_error: f64,
    pub stress_cpu_relative_error: f64,
    pub energy_cpu_relative_error: f64,
    pub h2d_operations: usize,
    pub h2d_bytes: usize,
    pub d2h_operations: usize,
    pub d2h_bytes: usize,
    pub allocation_high_water_bytes: usize,
    pub live_allocation_bytes_after_cleanup: usize,
    pub assembly_time_us: u64,
    pub upload_time_us: u64,
    pub forward_solve_time_us: u64,
    pub adjoint_solve_time_us: u64,
    pub postprocess_time_us: u64,
    pub derivative: CudaDerivativeEvidence,
    pub physical_gates_passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CudaStaticBracketResult {
    pub result: StaticBracketResult,
    pub cpu_reference: StaticBracketResult,
    pub execution: StructuralExecutionProvenance,
    pub evidence: CudaStaticBracketEvidence,
}

fn cancelled(
    config: &CudaStaticBracketConfig,
    stage: &'static str,
) -> Result<(), StructuralCudaError> {
    if config.cancellation.is_cancelled() {
        Err(StructuralCudaError::Cancelled { stage })
    } else {
        Ok(())
    }
}

fn map_backend(error: BackendError, stage: &'static str) -> StructuralCudaError {
    match error {
        BackendError::Allocation { reason } => StructuralCudaError::Allocation {
            stage,
            message: reason,
        },
        BackendError::DeviceLost { reason, .. }
        | BackendError::CommandBuffer { reason, .. }
        | BackendError::Synchronization { reason, .. } => StructuralCudaError::DeviceLost {
            stage,
            message: reason,
        },
        BackendError::HostFallback { .. } => StructuralCudaError::HostFallbackRejected { stage },
        other => StructuralCudaError::Backend {
            stage,
            message: other.to_string(),
        },
    }
}

fn map_gmres(error: GmresBackendError, stage: &'static str) -> StructuralCudaError {
    match error {
        GmresBackendError::Allocation { reason, .. } => StructuralCudaError::Allocation {
            stage,
            message: reason,
        },
        GmresBackendError::Backend { source } => map_backend(source, stage),
        other => StructuralCudaError::Backend {
            stage,
            message: other.to_string(),
        },
    }
}

fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE)
}

fn check_gate(check: &'static str, value: f64, limit: f64) -> Result<(), StructuralCudaError> {
    if value.is_finite() && value <= limit {
        Ok(())
    } else {
        Err(StructuralCudaError::PhysicalGate {
            check,
            value,
            limit,
        })
    }
}

fn csr_matvec(matrix: &sonium_solvers::CsrMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; matrix.num_rows];
    for (row, output) in result.iter_mut().enumerate() {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            *output += matrix.values[index] * vector[matrix.col_indices[index]];
        }
    }
    result
}

fn csr_transpose_matvec(matrix: &sonium_solvers::CsrMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; matrix.num_cols];
    for (row, &input) in vector.iter().enumerate() {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            result[matrix.col_indices[index]] += matrix.values[index] * input;
        }
    }
    result
}

fn relative_residual(matrix: &sonium_solvers::CsrMatrix<f64>, x: &[f64], rhs: &[f64]) -> f64 {
    let applied = csr_matvec(matrix, x);
    let residual = applied
        .iter()
        .zip(rhs)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value * value).sum::<f64>().sqrt();
    residual / rhs_norm.max(f64::MIN_POSITIVE)
}

fn transpose_relative_residual(
    matrix: &sonium_solvers::CsrMatrix<f64>,
    x: &[f64],
    rhs: &[f64],
) -> f64 {
    let applied = csr_transpose_matvec(matrix, x);
    let residual = applied
        .iter()
        .zip(rhs)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value * value).sum::<f64>().sqrt();
    residual / rhs_norm.max(f64::MIN_POSITIVE)
}

struct HermitianView<'operator, 'backend> {
    operator: &'operator PrebakedAdjointSpmv<'backend, CudaBackend, f64>,
}

impl DeviceOperator<CudaBackend, f64> for HermitianView<'_, '_> {
    fn n(&self) -> usize {
        self.operator.n()
    }

    fn apply(
        &self,
        backend: &CudaBackend,
        x: &<CudaBackend as Backend>::Buffer<f64>,
        y: &mut <CudaBackend as Backend>::Buffer<f64>,
    ) {
        let _ = self.operator.try_apply_hermitian(backend, x, y);
    }

    fn try_apply(
        &self,
        backend: &CudaBackend,
        x: &<CudaBackend as Backend>::Buffer<f64>,
        y: &mut <CudaBackend as Backend>::Buffer<f64>,
    ) -> sonium_compute::BackendResult<()> {
        self.operator.try_apply_hermitian(backend, x, y)
    }
}

/// Execute the complete supported CUDA structural cell and its mandatory
/// CPU/parity, physical, and transpose-adjoint promotion gates.
pub fn run_static_bracket_cuda(
    study: &BracketStudy,
    config: &CudaStaticBracketConfig,
) -> Result<CudaStaticBracketResult, StructuralCudaError> {
    if config.max_iterations == 0 {
        return Err(StructuralCudaError::InvalidConfiguration {
            field: "max_iterations",
            value: config.max_iterations.to_string(),
        });
    }
    if !config.relative_residual_tolerance.is_finite()
        || config.relative_residual_tolerance <= 0.0
        || config.relative_residual_tolerance >= 1.0
    {
        return Err(StructuralCudaError::InvalidConfiguration {
            field: "relative_residual_tolerance",
            value: config.relative_residual_tolerance.to_string(),
        });
    }
    for (field, value) in [
        (
            "displacement_relative_tolerance",
            config.displacement_relative_tolerance,
        ),
        (
            "stress_relative_tolerance",
            config.stress_relative_tolerance,
        ),
        (
            "energy_relative_tolerance",
            config.energy_relative_tolerance,
        ),
        (
            "gradient_relative_tolerance",
            config.gradient_relative_tolerance,
        ),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(StructuralCudaError::InvalidConfiguration {
                field,
                value: value.to_string(),
            });
        }
    }
    if study.plane_model != PlaneModel::PlaneStress {
        return Err(StructuralCudaError::InvalidConfiguration {
            field: "plane_model",
            value: format!("{:?}; CUDA supports PlaneStress only", study.plane_model),
        });
    }
    validate_study(study).map_err(|error| StructuralCudaError::InvalidConfiguration {
        field: "study",
        value: error.to_string(),
    })?;
    cancelled(config, "before CPU reference")?;
    let cpu_reference =
        run_static_bracket(study).map_err(|error| StructuralCudaError::CpuReference {
            message: error.to_string(),
        })?;

    cancelled(config, "before assembly")?;
    let assembly_start = Instant::now();
    let mesh = study.mesh();
    validate_mesh(&mesh, study).map_err(|error| StructuralCudaError::InvalidConfiguration {
        field: "mesh",
        value: error.to_string(),
    })?;
    let degree = PolynomialDegree::P1;
    let space = VectorH1Space::new(degree, 2);
    let dofs = space.num_global_dofs(&mesh);
    let form = ElasticityForm::new_2d(study.material.as_fem(), study.plane_model.as_fem());
    let matrices = assemble_structural_matrices(study, &mesh, &space, &form, degree);
    let unconstrained_stiffness = matrices.stiffness.clone();
    let mut rhs = vec![0.0; dofs];
    apply_tip_edge_load_order(&mesh, study, &space, degree, &mut rhs);
    let external_rhs = rhs.clone();
    let mut system = FemSystem::new(matrices.stiffness, rhs);
    try_apply_homogeneous_dirichlet(
        &mut system,
        &mesh,
        &space,
        study.loads.fixed_boundary_marker,
    )
    .map_err(|error| StructuralCudaError::InvalidConfiguration {
        field: "fixed_boundary_marker",
        value: error.to_string(),
    })?;
    let csr = system.matrix.to_csr();
    let assembly_time_us = assembly_start.elapsed().as_micros() as u64;
    cancelled(config, "before CUDA initialization")?;

    let backend = CudaBackend::new(config.device_ordinal).map_err(|error| {
        StructuralCudaError::BackendUnavailable {
            device_ordinal: config.device_ordinal,
            message: error.to_string(),
        }
    })?;
    backend.reset_runtime_stats();
    let stats_before = backend.runtime_stats();

    let host_csr = HostCsrRef::try_new(
        csr.num_rows,
        csr.num_cols,
        &csr.row_ptrs,
        &csr.col_indices,
        &csr.values,
    )
    .map_err(|error| StructuralCudaError::InvalidConfiguration {
        field: "assembled_csr",
        value: error.to_string(),
    })?;
    let upload_start = Instant::now();
    let operator = AdjointDeviceCsr::try_from_host_csr_ref(&backend, host_csr)
        .map_err(|error| map_backend(error, "operator upload"))?;
    let prebaked = operator
        .try_prebake(&backend)
        .map_err(|error| map_backend(error, "operator prebake"))?;
    let mut diagonal = vec![0.0; csr.num_rows];
    for (row, value) in diagonal.iter_mut().enumerate() {
        for index in csr.row_ptrs[row]..csr.row_ptrs[row + 1] {
            if csr.col_indices[index] == row {
                *value = csr.values[index];
                break;
            }
        }
    }
    let preconditioner = PointJacobiBackend::try_from_diagonal(&backend, &diagonal)
        .map_err(|error| map_backend(error, "point-Jacobi upload"))?;
    let mut rhs_device =
        backend
            .raw_alloc::<f64>(dofs)
            .map_err(|error| StructuralCudaError::Allocation {
                stage: "right-hand side",
                message: error.to_string(),
            })?;
    let mut solution_device =
        backend
            .raw_alloc::<f64>(dofs)
            .map_err(|error| StructuralCudaError::Allocation {
                stage: "forward solution",
                message: error.to_string(),
            })?;
    let mut adjoint_device =
        backend
            .raw_alloc::<f64>(dofs)
            .map_err(|error| StructuralCudaError::Allocation {
                stage: "adjoint solution",
                message: error.to_string(),
            })?;
    backend
        .try_copy_h2d(&system.rhs, &mut rhs_device)
        .map_err(|error| map_backend(error, "right-hand-side upload"))?;
    let upload_time_us = upload_start.elapsed().as_micros() as u64;
    cancelled(config, "before forward solve")?;

    let gmres_config = GmresBackendConfig::<f64> {
        max_iterations: config.max_iterations.min(dofs),
        // The independently recomputed host-f64 residual below remains the
        // promotion gate. Requiring an extra decade in the projected device
        // residual can exhaust an unrestarted n-step Krylov space even when
        // the true residual already satisfies the declared contract.
        tolerance: config.relative_residual_tolerance,
    };
    let forward_start = Instant::now();
    let forward = gmres_backend_preconditioned(
        &backend,
        &prebaked,
        &preconditioner,
        &rhs_device,
        None,
        &mut solution_device,
        &gmres_config,
    )
    .map_err(|error| map_gmres(error, "forward solve"))?;
    if !forward.converged {
        return Err(StructuralCudaError::NonConvergence {
            stage: "forward solve",
            iterations: forward.iterations,
            residual: forward.final_residual,
        });
    }
    backend
        .synchronize()
        .map_err(|error| map_backend(error, "forward synchronization"))?;
    let mut solution = vec![0.0; dofs];
    backend
        .try_copy_d2h(&solution_device, &mut solution)
        .map_err(|error| map_backend(error, "forward result transfer"))?;
    let forward_solve_time_us = forward_start.elapsed().as_micros() as u64;
    let forward_true_relative_residual = relative_residual(&csr, &solution, &system.rhs);
    check_gate(
        "forward_true_relative_residual",
        forward_true_relative_residual,
        config.relative_residual_tolerance,
    )?;
    cancelled(config, "before adjoint solve")?;

    let (adjoint_status, adjoint, adjoint_solve_time_us) = {
        let adjoint_start = Instant::now();
        let hermitian = HermitianView {
            operator: &prebaked,
        };
        let adjoint_status = gmres_backend_preconditioned(
            &backend,
            &hermitian,
            &preconditioner,
            &rhs_device,
            None,
            &mut adjoint_device,
            &gmres_config,
        )
        .map_err(|error| map_gmres(error, "transpose adjoint solve"))?;
        if !adjoint_status.converged {
            return Err(StructuralCudaError::NonConvergence {
                stage: "transpose adjoint solve",
                iterations: adjoint_status.iterations,
                residual: adjoint_status.final_residual,
            });
        }
        backend
            .synchronize()
            .map_err(|error| map_backend(error, "adjoint synchronization"))?;
        let mut adjoint = vec![0.0; dofs];
        backend
            .try_copy_d2h(&adjoint_device, &mut adjoint)
            .map_err(|error| map_backend(error, "adjoint result transfer"))?;
        (
            adjoint_status,
            adjoint,
            adjoint_start.elapsed().as_micros() as u64,
        )
    };
    let adjoint_true_relative_residual = transpose_relative_residual(&csr, &adjoint, &system.rhs);
    check_gate(
        "adjoint_true_relative_residual",
        adjoint_true_relative_residual,
        config.relative_residual_tolerance,
    )?;

    cancelled(config, "before post-processing")?;
    let postprocess_start = Instant::now();
    let displacement = nodal_displacement(&solution, mesh.num_nodes());
    let tip_dofs = boundary_component_dofs(&mesh, &space, study.loads.loaded_boundary_marker, 1);
    if tip_dofs.is_empty() {
        return Err(StructuralCudaError::InvalidConfiguration {
            field: "loaded_boundary_marker",
            value: study.loads.loaded_boundary_marker.to_string(),
        });
    }
    let average_tip_deflection_m =
        tip_dofs.iter().map(|&dof| solution[dof]).sum::<f64>() / tip_dofs.len() as f64;
    let max_displacement_m = solution
        .as_chunks::<2>()
        .0
        .iter()
        .map(|d| (d[0] * d[0] + d[1] * d[1]).sqrt())
        .fold(0.0, f64::max);
    let fixed_dofs = space.boundary_dofs(&mesh, study.loads.fixed_boundary_marker);
    let internal_force = triplet_matvec(&unconstrained_stiffness, &solution);
    let applied_force_n = component_force(&external_rhs);
    let reaction_force_n = reaction_force(&internal_force, &external_rhs, &fixed_dofs);
    let free_equilibrium_residual_n =
        free_residual_norm(&internal_force, &external_rhs, &system.dirichlet_dofs);
    let force_balance_residual_n = ((applied_force_n[0] + reaction_force_n[0]).powi(2)
        + (applied_force_n[1] + reaction_force_n[1]).powi(2))
    .sqrt();
    let strain_energy_j = 0.5 * dot(&solution, &internal_force);
    let external_work_half = 0.5 * dot(&solution, &external_rhs);
    let energy_balance_relative_error =
        (strain_energy_j - external_work_half).abs() / external_work_half.abs().max(1.0);
    let force_scale = dot(&applied_force_n, &applied_force_n).sqrt().max(1.0);
    check_gate(
        "free_equilibrium",
        free_equilibrium_residual_n,
        1.0e-6 * force_scale,
    )?;
    check_gate(
        "force_balance",
        force_balance_residual_n,
        1.0e-6 * force_scale,
    )?;
    check_gate("energy_identity", energy_balance_relative_error, 1.0e-9)?;
    let max_von_mises_pa = max_von_mises_for_order(
        &mesh,
        study,
        &space,
        &solution,
        StructuralPolynomialOrder::Linear,
        &displacement,
    );
    let beam_theory_tip_deflection_m = crate::beam_tip_deflection(study);
    let beam_theory_relative_error =
        (average_tip_deflection_m.abs() - beam_theory_tip_deflection_m.abs()).abs()
            / beam_theory_tip_deflection_m.abs();
    let result = StaticBracketResult {
        nodes: mesh.num_nodes(),
        elements: mesh.num_elements(),
        dofs,
        polynomial_order: StructuralPolynomialOrder::Linear,
        solver_iterations: forward.iterations,
        solver_residual: forward_true_relative_residual,
        converged: true,
        max_displacement_m,
        average_tip_deflection_m,
        beam_theory_tip_deflection_m,
        beam_theory_relative_error,
        max_von_mises_pa,
        strain_energy_j,
        applied_force_n,
        reaction_force_n,
        free_equilibrium_residual_n,
        force_balance_residual_n,
        energy_balance_relative_error,
        model_assumption: study.model_assumption(),
        stress_recovery: StressRecoveryPolicy::ElementCentroid,
        displacement,
    };

    let displacement_cpu_relative_error = relative_error(
        result.average_tip_deflection_m,
        cpu_reference.average_tip_deflection_m,
    );
    let stress_cpu_relative_error =
        relative_error(result.max_von_mises_pa, cpu_reference.max_von_mises_pa);
    let energy_cpu_relative_error =
        relative_error(result.strain_energy_j, cpu_reference.strain_energy_j);
    check_gate(
        "cpu_displacement_parity",
        displacement_cpu_relative_error,
        config.displacement_relative_tolerance,
    )?;
    check_gate(
        "cpu_stress_parity",
        stress_cpu_relative_error,
        config.stress_relative_tolerance,
    )?;
    check_gate(
        "cpu_energy_parity",
        energy_cpu_relative_error,
        config.energy_relative_tolerance,
    )?;

    let state_norm = solution
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let adjoint_state_relative_error = solution
        .iter()
        .zip(&adjoint)
        .map(|(u, lambda)| (u - lambda).powi(2))
        .sum::<f64>()
        .sqrt()
        / state_norm.max(f64::MIN_POSITIVE);
    let stiffness_times_state = triplet_matvec(&unconstrained_stiffness, &solution);
    let adjoint_gradient_j_per_m =
        -dot(&adjoint, &stiffness_times_state) / study.geometry.thickness_m;
    let compliance_j = dot(&solution, &external_rhs);
    let independent_gradient_j_per_m = -compliance_j / study.geometry.thickness_m;
    let gradient_relative_error =
        relative_error(adjoint_gradient_j_per_m, independent_gradient_j_per_m);
    check_gate(
        "uniform_thickness_compliance_gradient",
        gradient_relative_error,
        config.gradient_relative_tolerance,
    )?;
    let derivative = CudaDerivativeEvidence {
        design_variable: "uniform_thickness_m".to_string(),
        objective: "compliance_j".to_string(),
        transpose_adjoint_executed: true,
        adjoint_iterations: adjoint_status.iterations,
        adjoint_true_relative_residual,
        adjoint_state_relative_error,
        adjoint_gradient_j_per_m,
        independent_gradient_j_per_m,
        gradient_relative_error,
        accepted: true,
    };
    let postprocess_time_us = postprocess_start.elapsed().as_micros() as u64;

    drop(adjoint_device);
    drop(solution_device);
    drop(rhs_device);
    drop(preconditioner);
    drop(prebaked);
    drop(operator);
    backend
        .try_release_cached_scratch()
        .map_err(|error| map_backend(error, "cached scratch cleanup"))?;
    let stats = backend.runtime_stats().since(stats_before);
    check_gate(
        "device_allocation_cleanup",
        stats.live_allocation_bytes as f64,
        0.0,
    )?;
    cancelled(config, "before result publication")?;

    Ok(CudaStaticBracketResult {
        result,
        cpu_reference,
        execution: StructuralExecutionProvenance::cuda_static_bracket_reference(),
        evidence: CudaStaticBracketEvidence {
            schema_version: CUDA_STATIC_BRACKET_SCHEMA_VERSION,
            backend: "cuda".to_string(),
            device_ordinal: config.device_ordinal,
            precision: "f64".to_string(),
            element: "triangle".to_string(),
            polynomial_order: "P1".to_string(),
            regime: "linear-small-strain-plane-stress".to_string(),
            solver: "unrestarted-gmres".to_string(),
            preconditioner: "point-jacobi".to_string(),
            fallback_used: false,
            mesh_nodes: mesh.num_nodes(),
            mesh_elements: mesh.num_elements(),
            dofs,
            forward_iterations: forward.iterations,
            forward_projected_relative_residual: forward.projected_residual,
            forward_true_relative_residual,
            displacement_cpu_relative_error,
            stress_cpu_relative_error,
            energy_cpu_relative_error,
            h2d_operations: stats.transfer.h2d_operations,
            h2d_bytes: stats.transfer.h2d_bytes,
            d2h_operations: stats.transfer.d2h_operations,
            d2h_bytes: stats.transfer.d2h_bytes,
            allocation_high_water_bytes: stats.allocation_high_water_bytes,
            live_allocation_bytes_after_cleanup: stats.live_allocation_bytes,
            assembly_time_us,
            upload_time_us,
            forward_solve_time_us,
            adjoint_solve_time_us,
            postprocess_time_us,
            derivative,
            physical_gates_passed: true,
        },
    })
}

impl From<StructuralMechanicsError> for StructuralCudaError {
    fn from(error: StructuralMechanicsError) -> Self {
        Self::CpuReference {
            message: error.to_string(),
        }
    }
}
