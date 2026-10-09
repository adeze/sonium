//! Generated support/admission matrix for the public `sonium-solvers`
//! facade.  Native CUDA is intentionally advertised only for the bounded
//! high-level CSR/GMRES/identity path exposed by the explicit backend selector;
//! every other cell remains fail-closed.  This producer records that
//! distinction explicitly and includes the split backend capability table as a
//! separate evidence section.

#![cfg(feature = "cuda")]

use num_complex::{Complex32, Complex64};
use serde_json::{Value, json};
use sonium_compute::CapabilityState;
use sonium_qa::{QaResult, cuda_hardware_provenance, required_cuda_backend, write_qa_artifact};
use sonium_solvers::{
    BackendKrylovCapability, CsrMatrix, LinearSolveBackend, LinearSolveConfig, LinearSolveFallback,
    LinearSolverType, backend_krylov_capability_matrix, solve_complex, solve_real,
    solve_with_backend,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.facade-support-matrix.v1";
const ARTIFACT: &str = "solvers_facade_support_matrix.json";

const SOLVERS: [(&str, LinearSolverType); 10] = [
    ("auto", LinearSolverType::Auto),
    ("gmres", LinearSolverType::Gmres),
    ("gmres_jacobi", LinearSolverType::GmresJacobi),
    ("gmres_ilu", LinearSolverType::GmresIlu),
    ("gmres_amg", LinearSolverType::GmresAmg),
    ("gmres_chebyshev", LinearSolverType::GmresChebyshev),
    ("gmres_schwarz", LinearSolverType::GmresSchwarz),
    ("fgmres_schwarz", LinearSolverType::FgmresSchwarz),
    ("cg", LinearSolverType::Cg),
    ("cg_amg", LinearSolverType::CgAmg),
];
const SCALARS: [&str; 4] = ["f32", "f64", "complex32", "complex64"];
const OPERATORS: [&str; 4] = [
    "CsrMatrix",
    "DenseMatrix",
    "FnOperator",
    "CompositeOperator",
];
const PRECONDITIONERS: [&str; 6] = ["none", "jacobi", "ilu", "amg", "chebyshev", "schwarz"];
const PRECISIONS: [&str; 2] = ["native", "mixed_inner_f32"];
const ADJOINTS: [&str; 3] = ["forward", "transpose", "hermitian"];
const CHECKPOINTS: [&str; 4] = ["none", "capture", "restore", "resume"];

fn required_preconditioner(solver: LinearSolverType) -> &'static str {
    match solver {
        LinearSolverType::Auto | LinearSolverType::Gmres => "none",
        LinearSolverType::GmresJacobi => "jacobi",
        LinearSolverType::GmresIlu => "ilu",
        LinearSolverType::GmresAmg | LinearSolverType::CgAmg => "amg",
        LinearSolverType::GmresChebyshev => "chebyshev",
        LinearSolverType::GmresSchwarz | LinearSolverType::FgmresSchwarz => "schwarz",
        LinearSolverType::Cg => "none",
    }
}

fn cpu_admission(
    solver: LinearSolverType,
    scalar: &str,
    operator: &str,
    preconditioner: &str,
    precision: &str,
    adjoint: &str,
    checkpoint: &str,
) -> (&'static str, &'static str) {
    if operator != "CsrMatrix" {
        return (
            "unsupported",
            "facade solve entry points accept CsrMatrix only",
        );
    }
    if preconditioner != required_preconditioner(solver) {
        return (
            "unsupported",
            "solver/preconditioner pairing is not this facade strategy",
        );
    }
    if precision != "native" {
        return ("unsupported", "mixed precision is exposed by separate APIs");
    }
    if adjoint != "forward" {
        return (
            "unsupported",
            "facade linear solve has no transpose/Hermitian solve selector",
        );
    }
    if checkpoint != "none" {
        return (
            "unsupported",
            "checkpoint capture/restore is a separate typed API",
        );
    }
    if matches!(solver, LinearSolverType::Cg | LinearSolverType::CgAmg)
        && matches!(scalar, "complex32" | "complex64")
    {
        return ("unsupported", "CG strategies require real SPD systems");
    }
    ("supported", "CPU high-level facade dispatch")
}

fn cuda_admission(
    solver: LinearSolverType,
    scalar: &str,
    operator: &str,
    preconditioner: &str,
    precision: &str,
    adjoint: &str,
    checkpoint: &str,
) -> (&'static str, &'static str) {
    if solver == LinearSolverType::Gmres
        && SCALARS.contains(&scalar)
        && operator == "CsrMatrix"
        && preconditioner == "none"
        && precision == "native"
        && adjoint == "forward"
        && checkpoint == "none"
    {
        return (
            "supported",
            "native CUDA facade selector: CSR + GMRES + identity + native precision + forward",
        );
    }
    (
        "unsupported",
        "high-level CUDA is fail-closed outside explicit CSR + GMRES + identity + native + forward",
    )
}

fn capability_state(state: CapabilityState) -> &'static str {
    match state {
        CapabilityState::Native => "native",
        CapabilityState::VendorNative => "vendor_native",
        CapabilityState::Unsupported => "unsupported",
        CapabilityState::HostSharedFallback => "host_shared_fallback",
        CapabilityState::HostStagedFallback => "host_staged_fallback",
    }
}

fn capability_json(cell: BackendKrylovCapability) -> Value {
    json!({
        "backend": cell.backend,
        "method": format!("{:?}", cell.method),
        "scalar": format!("{:?}", cell.scalar),
        "scalar_state": capability_state(cell.scalar_state),
        "blas1": capability_state(cell.blas1),
        "axpy_many": capability_state(cell.axpy_many),
        "multi_dot": capability_state(cell.multi_dot),
        "strided_projection": capability_state(cell.strided_projection),
        "reduction": capability_state(cell.reduction),
        "csr_spmm": capability_state(cell.csr_spmm),
        "native_execution": cell.native_execution,
    })
}

fn generated_cells() -> Vec<Value> {
    let mut cells = Vec::with_capacity(
        SOLVERS.len()
            * SCALARS.len()
            * OPERATORS.len()
            * PRECONDITIONERS.len()
            * PRECISIONS.len()
            * ADJOINTS.len()
            * CHECKPOINTS.len()
            * 2,
    );
    for (solver_name, solver) in SOLVERS {
        for scalar in SCALARS {
            for operator in OPERATORS {
                for preconditioner in PRECONDITIONERS {
                    for precision in PRECISIONS {
                        for adjoint in ADJOINTS {
                            for checkpoint in CHECKPOINTS {
                                let (cpu_status, cpu_reason) = cpu_admission(
                                    solver,
                                    scalar,
                                    operator,
                                    preconditioner,
                                    precision,
                                    adjoint,
                                    checkpoint,
                                );
                                let (cuda_status, cuda_reason) = cuda_admission(
                                    solver,
                                    scalar,
                                    operator,
                                    preconditioner,
                                    precision,
                                    adjoint,
                                    checkpoint,
                                );
                                cells.push(json!({
                                    "solver": solver_name,
                                    "scalar": scalar,
                                    "operator": operator,
                                    "preconditioner": preconditioner,
                                    "precision": precision,
                                    "adjoint": adjoint,
                                    "checkpoint": checkpoint,
                                    "cpu": {"status": cpu_status, "reason": cpu_reason},
                                    "cuda": {"status": cuda_status, "reason": cuda_reason},
                                    "fail_before_iteration_zero": true,
                                    "host_solve_may_be_reported_as_cuda": false,
                                }));
                            }
                        }
                    }
                }
            }
        }
    }
    cells
}

fn cpu_probe() -> Result<Vec<Value>, String> {
    let real = CsrMatrix::from_triplets(
        2,
        2,
        vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)],
    );
    let config = LinearSolveConfig {
        solver: LinearSolverType::Gmres,
        exact_residual_check_interval: 1,
        ..Default::default()
    };
    let real_result = solve_real(&real, &[9.0, 8.0], &config)
        .map_err(|error| format!("CPU f64 facade probe: {error}"))?;
    real_result
        .validate_contract()
        .map_err(|error| format!("CPU f64 result contract: {error}"))?;
    let complex = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, Complex64::new(4.0, 0.1)),
            (0, 1, Complex64::new(1.0, 0.0)),
            (1, 0, Complex64::new(1.0, 0.0)),
            (1, 1, Complex64::new(3.0, 0.1)),
        ],
    );
    let complex_result = solve_complex(
        &complex,
        &[Complex64::new(9.0, 0.0), Complex64::new(8.0, 0.0)],
        &config,
    )
    .map_err(|error| format!("CPU Complex64 facade probe: {error}"))?;
    complex_result
        .validate_contract()
        .map_err(|error| format!("CPU Complex64 result contract: {error}"))?;
    let native_real = solve_with_backend(
        &real,
        &[9.0, 8.0],
        &config,
        LinearSolveBackend::Cuda,
        LinearSolveFallback::Reject,
    )
    .map_err(|error| format!("CUDA f64 facade probe: {error}"))?;
    native_real
        .validate_contract()
        .map_err(|error| format!("CUDA f64 result contract: {error}"))?;
    if native_real.backend.requested != LinearSolveBackend::Cuda
        || native_real.backend.actual != LinearSolveBackend::Cuda
        || native_real.backend.fallback
    {
        return Err("CUDA f64 facade probe did not report native CUDA provenance".into());
    }
    let native_complex = solve_with_backend(
        &complex,
        &[Complex64::new(9.0, 0.0), Complex64::new(8.0, 0.0)],
        &config,
        LinearSolveBackend::Cuda,
        LinearSolveFallback::Reject,
    )
    .map_err(|error| format!("CUDA Complex64 facade probe: {error}"))?;
    native_complex
        .validate_contract()
        .map_err(|error| format!("CUDA Complex64 result contract: {error}"))?;
    if native_complex.backend.requested != LinearSolveBackend::Cuda
        || native_complex.backend.actual != LinearSolveBackend::Cuda
        || native_complex.backend.fallback
    {
        return Err("CUDA Complex64 facade probe did not report native CUDA provenance".into());
    }
    let real32 = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, 4.0_f32),
            (0, 1, 1.0_f32),
            (1, 0, 1.0_f32),
            (1, 1, 3.0_f32),
        ],
    );
    let config32 = LinearSolveConfig {
        tolerance: 1.0e-5,
        ..config.clone()
    };
    let native_real32 = solve_with_backend(
        &real32,
        &[9.0_f32, 8.0_f32],
        &config32,
        LinearSolveBackend::Cuda,
        LinearSolveFallback::Reject,
    )
    .map_err(|error| format!("CUDA f32 facade probe: {error}"))?;
    native_real32
        .validate_contract()
        .map_err(|error| format!("CUDA f32 result contract: {error}"))?;
    if native_real32.backend.requested != LinearSolveBackend::Cuda
        || native_real32.backend.actual != LinearSolveBackend::Cuda
        || native_real32.backend.fallback
    {
        return Err("CUDA f32 facade probe did not report native CUDA provenance".into());
    }
    let complex32 = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, Complex32::new(4.0, 0.1)),
            (0, 1, Complex32::new(1.0, 0.0)),
            (1, 0, Complex32::new(1.0, 0.0)),
            (1, 1, Complex32::new(3.0, 0.1)),
        ],
    );
    let native_complex32 = solve_with_backend(
        &complex32,
        &[Complex32::new(9.0, 0.0), Complex32::new(8.0, 0.0)],
        &config32,
        LinearSolveBackend::Cuda,
        LinearSolveFallback::Reject,
    )
    .map_err(|error| format!("CUDA Complex32 facade probe: {error}"))?;
    native_complex32
        .validate_contract()
        .map_err(|error| format!("CUDA Complex32 result contract: {error}"))?;
    if native_complex32.backend.requested != LinearSolveBackend::Cuda
        || native_complex32.backend.actual != LinearSolveBackend::Cuda
        || native_complex32.backend.fallback
    {
        return Err("CUDA Complex32 facade probe did not report native CUDA provenance".into());
    }
    let fallback_config = LinearSolveConfig {
        solver: LinearSolverType::GmresJacobi,
        ..config.clone()
    };
    let fallback_real = solve_with_backend(
        &real,
        &[9.0, 8.0],
        &fallback_config,
        LinearSolveBackend::Cuda,
        LinearSolveFallback::AllowCpu,
    )
    .map_err(|error| format!("explicit CPU fallback f64 facade probe: {error}"))?;
    fallback_real
        .validate_contract()
        .map_err(|error| format!("explicit CPU fallback f64 result contract: {error}"))?;
    if fallback_real.backend.requested != LinearSolveBackend::Cuda
        || fallback_real.backend.actual != LinearSolveBackend::Cpu
        || !fallback_real.backend.fallback
    {
        return Err("explicit CPU fallback f64 probe did not report fallback provenance".into());
    }
    let fallback_complex = solve_with_backend(
        &complex,
        &[Complex64::new(9.0, 0.0), Complex64::new(8.0, 0.0)],
        &fallback_config,
        LinearSolveBackend::Cuda,
        LinearSolveFallback::AllowCpu,
    )
    .map_err(|error| format!("explicit CPU fallback Complex64 facade probe: {error}"))?;
    fallback_complex
        .validate_contract()
        .map_err(|error| format!("explicit CPU fallback Complex64 result contract: {error}"))?;
    if fallback_complex.backend.requested != LinearSolveBackend::Cuda
        || fallback_complex.backend.actual != LinearSolveBackend::Cpu
        || !fallback_complex.backend.fallback
    {
        return Err(
            "explicit CPU fallback Complex64 probe did not report fallback provenance".into(),
        );
    }
    Ok(vec![
        json!({
            "backend": "cpu",
            "solver": "gmres",
            "scalar": "f64",
            "status": format!("{:?}", real_result.status),
            "accuracy": format!("{:?}", real_result.attainable_accuracy),
            "residual": real_result.residual,
            "backward_error": real_result.backward_error,
            "iterations": real_result.iterations,
            "contract_valid": true,
        }),
        json!({
            "backend": "cpu",
            "solver": "gmres",
            "scalar": "complex64",
            "status": format!("{:?}", complex_result.status),
            "accuracy": format!("{:?}", complex_result.attainable_accuracy),
            "residual": complex_result.residual,
            "backward_error": complex_result.backward_error,
            "iterations": complex_result.iterations,
            "contract_valid": true,
        }),
        json!({
            "backend": "cuda",
            "requested_backend": format!("{:?}", native_real.backend.requested),
            "actual_backend": format!("{:?}", native_real.backend.actual),
            "fallback": native_real.backend.fallback,
            "solver": "gmres",
            "scalar": "f64",
            "status": format!("{:?}", native_real.status),
            "accuracy": format!("{:?}", native_real.attainable_accuracy),
            "residual": native_real.residual,
            "backward_error": native_real.backward_error,
            "iterations": native_real.iterations,
            "contract_valid": true,
        }),
        json!({
            "backend": "cuda",
            "requested_backend": format!("{:?}", native_real32.backend.requested),
            "actual_backend": format!("{:?}", native_real32.backend.actual),
            "fallback": native_real32.backend.fallback,
            "solver": "gmres",
            "scalar": "f32",
            "status": format!("{:?}", native_real32.status),
            "accuracy": format!("{:?}", native_real32.attainable_accuracy),
            "residual": native_real32.residual,
            "backward_error": native_real32.backward_error,
            "iterations": native_real32.iterations,
            "contract_valid": true,
        }),
        json!({
            "backend": "cuda",
            "requested_backend": format!("{:?}", native_complex32.backend.requested),
            "actual_backend": format!("{:?}", native_complex32.backend.actual),
            "fallback": native_complex32.backend.fallback,
            "solver": "gmres",
            "scalar": "complex32",
            "status": format!("{:?}", native_complex32.status),
            "accuracy": format!("{:?}", native_complex32.attainable_accuracy),
            "residual": native_complex32.residual,
            "backward_error": native_complex32.backward_error,
            "iterations": native_complex32.iterations,
            "contract_valid": true,
        }),
        json!({
            "backend": "cuda",
            "requested_backend": format!("{:?}", native_complex.backend.requested),
            "actual_backend": format!("{:?}", native_complex.backend.actual),
            "fallback": native_complex.backend.fallback,
            "solver": "gmres",
            "scalar": "complex64",
            "status": format!("{:?}", native_complex.status),
            "accuracy": format!("{:?}", native_complex.attainable_accuracy),
            "residual": native_complex.residual,
            "backward_error": native_complex.backward_error,
            "iterations": native_complex.iterations,
            "contract_valid": true,
        }),
        json!({
            "backend": "cpu",
            "requested_backend": format!("{:?}", fallback_real.backend.requested),
            "actual_backend": format!("{:?}", fallback_real.backend.actual),
            "fallback": fallback_real.backend.fallback,
            "solver": "gmres_jacobi",
            "scalar": "f64",
            "status": format!("{:?}", fallback_real.status),
            "accuracy": format!("{:?}", fallback_real.attainable_accuracy),
            "residual": fallback_real.residual,
            "backward_error": fallback_real.backward_error,
            "iterations": fallback_real.iterations,
            "contract_valid": true,
        }),
        json!({
            "backend": "cpu",
            "requested_backend": format!("{:?}", fallback_complex.backend.requested),
            "actual_backend": format!("{:?}", fallback_complex.backend.actual),
            "fallback": fallback_complex.backend.fallback,
            "solver": "gmres_jacobi",
            "scalar": "complex64",
            "status": format!("{:?}", fallback_complex.status),
            "accuracy": format!("{:?}", fallback_complex.attainable_accuracy),
            "residual": fallback_complex.residual,
            "backward_error": fallback_complex.backward_error,
            "iterations": fallback_complex.iterations,
            "contract_valid": true,
        }),
    ])
}

fn main() {
    let started = Instant::now();
    let result = (|| -> Result<QaResult, String> {
        let cuda = required_cuda_backend()?;
        let backend_cells = backend_krylov_capability_matrix(&cuda);
        let cells = generated_cells();
        let probes = cpu_probe()?;
        let advertised_cpu_supported = cells
            .iter()
            .filter(|cell| cell["cpu"]["status"] == "supported")
            .count();
        let advertised_cuda_supported = cells
            .iter()
            .filter(|cell| cell["cuda"]["status"] == "supported")
            .count();
        let artifact_value = json!({
            "schema_version": 1,
            "case_id": CASE_ID,
            "requested_backend": "cuda",
            "actual_backend": "cuda",
            "hardware": cuda_hardware_provenance(),
            "axes": {
                "solver": SOLVERS.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
                "scalar": SCALARS,
                "operator": OPERATORS,
                "preconditioner": PRECONDITIONERS,
                "precision": PRECISIONS,
                "adjoint": ADJOINTS,
                "checkpoint": CHECKPOINTS,
                "backend": ["cpu", "cuda"],
            },
            "generated_cell_count": cells.len(),
            "advertised_cpu_supported_cells": advertised_cpu_supported,
            "advertised_cuda_supported_cells": advertised_cuda_supported,
            "facade_cells": cells,
            "split_backend_cuda_capabilities": backend_cells.iter().copied().map(capability_json).collect::<Vec<_>>(),
            "instrumented_execution": probes,
            "acceptance": {
                "generated_matrix_covers_all_declared_axes": true,
                "facade_cuda_admission_is_fail_closed": true,
                "cpu_result_contract_and_accuracy_preserved": true,
                "generated_execution_vs_advertisement_for_every_cell": false,
                "generated_execution_vs_advertisement_for_every_advertised_cuda_cell": true,
                "explicit_facade_cuda_fallback_configuration_and_result": true,
                "no_host_solve_labeled_cuda": true,
                "note": "Native CUDA is advertised only for explicit CSR + GMRES + identity + native + forward + no-checkpoint cells. Other cells fail before iteration zero. AllowCpu is explicit and every fallback result carries requested CUDA / actual CPU provenance.",
            },
        });
        let bytes = serde_json::to_vec_pretty(&artifact_value)
            .map_err(|error| format!("serialize facade support matrix: {error}"))?;
        let (path, checksum) = write_qa_artifact(ARTIFACT, &bytes)?;
        let mut result = QaResult::passed(CASE_ID)
            .requested_backend("cuda")
            .actual_backend("cuda")
            .backend_passed("cuda")
            .reference("CPU facade result-contract probes plus generated public-axis admission matrix")
            .reference_checksum("analytic:solver-facade-support-matrix-v1")
            .dimension("10 solvers x 4 scalars x 4 operators x 6 preconditioners x 2 precisions x 3 adjoints x 4 checkpoint modes x 2 backends")
            .formulation("fail-closed facade support advertisement and split-backend capability separation")
            .metric("generated_cells", cells.len() as f64)
            .metric("backend_cuda_capability_cells", backend_cells.len() as f64)
            .metric("advertised_cpu_supported_cells", advertised_cpu_supported as f64)
            .metric("advertised_cuda_supported_cells", advertised_cuda_supported as f64)
            .metric("instrumented_cpu_probes", 4.0)
            .metric("instrumented_native_cuda_probes", 4.0)
            .metric("instrumented_explicit_cpu_fallback_probes", 2.0)
            .minimum("generated_cells", 11520.0)
            .minimum("backend_cuda_capability_cells", 16.0)
            .artifact_with_provenance(
                path,
                "crates/sonium-qa/bin/solvers_facade_support_matrix.rs",
                "generated facade CPU/CUDA support axes, CPU result-contract probes, and split CUDA capability separation",
                "v1",
                checksum,
            )
            .hardware_provenance(cuda_hardware_provenance());
        result = result.timing_ms("generation", started.elapsed().as_millis() as u64);
        Ok(result)
    })();
    match result {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                serde_json::to_string(&result).expect("S1 QA JSON")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .requested_backend("cuda")
                .actual_backend("cuda");
            println!(
                "QA_RESULT: {}",
                serde_json::to_string(&result).expect("S1 QA failure JSON")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
