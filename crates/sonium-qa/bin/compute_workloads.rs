//! End-to-end sparse compute workload contract.

#[cfg(feature = "metal")]
use sonium_compute::MetalBackend;
use sonium_compute::{
    Backend, BlockGmresConfig, CpuBackend, DeviceCsr, IdentityDevicePreconditioner,
    gmres_backend_block_preconditioned,
};
#[cfg(feature = "cuda")]
use sonium_qa::required_cuda_backend;
use sonium_qa::{QaBackendOutcome, QaResult, QaStatus};
use std::time::Instant;

const CASE_ID: &str = "compute.sparse-workload.v1";
const MAX_RESIDUAL: f64 = 1.0e-8;
const MAX_SOLUTION_ERROR: f64 = 1.0e-8;

#[derive(Debug, Default)]
struct WorkloadSummary {
    backend_count: usize,
    max_final_residual: f64,
    max_solution_error: f64,
    backend_outcomes: Vec<QaBackendOutcome>,
}

fn main() {
    let started = Instant::now();
    let required = required_backend().ok().flatten();
    match run() {
        Ok(summary) => {
            let actual_backend = required
                .as_deref()
                .unwrap_or(if summary.backend_count == 1 {
                    "cpu"
                } else {
                    "multiple"
                });
            let mut result = QaResult::passed(CASE_ID)
                .actual_backend(actual_backend)
                .metric("max_final_residual", summary.max_final_residual)
                .metric("max_solution_error", summary.max_solution_error)
                .metric("backend_count", summary.backend_count as f64)
                .maximum("max_final_residual", MAX_RESIDUAL)
                .maximum("max_solution_error", MAX_SOLUTION_ERROR)
                .input_checksum("analytic/deterministic-fixture:csr-block-gmres-v1")
                .reference("analytical dense-system solution")
                .reference_checksum("analytic")
                .dimension("sparse 2x2, two RHS")
                .formulation("CPU CSR SpMV and two-RHS block GMRES")
                .tolerance("1e-8")
                .seed("compute-sparse-workload:csr-block-gmres:v1")
                .timing_ms("workload", started.elapsed().as_millis() as u64)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for outcome in &summary.backend_outcomes {
                result = match outcome.status {
                    QaStatus::Passed => result.backend_passed(&outcome.backend),
                    QaStatus::Skipped => result.backend_skipped(
                        &outcome.backend,
                        outcome.reason.as_deref().unwrap_or("backend unavailable"),
                    ),
                    QaStatus::Failed => result,
                };
            }
            if let Some(required) = required.as_deref() {
                result = result.requested_backend(required);
            } else if summary.backend_count == 1 {
                result = result.requested_backend("cpu");
            }
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("compute workload result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .elapsed_ms(started.elapsed().as_millis() as u64)
                .to_json_line()
                .expect("compute workload failure result must serialize");
            println!("QA_RESULT: {result}");
            eprintln!("COMPUTE WORKLOAD FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

fn required_backend() -> Result<Option<String>, String> {
    let value = match std::env::var("SONIUM_REQUIRE_BACKEND") {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    if matches!(value.as_str(), "cpu" | "cuda" | "metal") {
        Ok(Some(value))
    } else {
        Err(format!(
            "SONIUM_REQUIRE_BACKEND must be cpu, cuda, or metal for this workload; got {value:?}"
        ))
    }
}

fn run() -> Result<WorkloadSummary, String> {
    let required = required_backend()?;
    let mut summary = WorkloadSummary::default();
    if required.as_deref().is_none_or(|backend| backend == "cpu") {
        let (residual, solution_error) = run_backend("cpu", &CpuBackend::new())?;
        summary.backend_count += 1;
        summary.max_final_residual = summary.max_final_residual.max(residual);
        summary.max_solution_error = summary.max_solution_error.max(solution_error);
        summary.backend_outcomes.push(QaBackendOutcome {
            backend: "cpu".into(),
            status: QaStatus::Passed,
            reason: None,
        });
    }
    #[cfg(feature = "cuda")]
    if required.as_deref().is_none_or(|backend| backend == "cuda") {
        match required_cuda_backend() {
            Ok(backend) => {
                let (residual, solution_error) = run_backend("cuda", &backend)?;
                summary.backend_count += 1;
                summary.max_final_residual = summary.max_final_residual.max(residual);
                summary.max_solution_error = summary.max_solution_error.max(solution_error);
                summary.backend_outcomes.push(QaBackendOutcome {
                    backend: "cuda".into(),
                    status: QaStatus::Passed,
                    reason: None,
                });
            }
            Err(error) if required.as_deref() != Some("cuda") => {
                summary.backend_outcomes.push(QaBackendOutcome {
                    backend: "cuda".into(),
                    status: QaStatus::Skipped,
                    reason: Some(error.to_string()),
                });
            }
            Err(error) => return Err(format!("required CUDA backend unavailable: {error}")),
        }
    }
    #[cfg(not(feature = "cuda"))]
    if required.as_deref() == Some("cuda") {
        return Err("required CUDA backend unavailable: build with --features cuda".into());
    }
    #[cfg(not(feature = "cuda"))]
    if required.is_none() {
        summary.backend_outcomes.push(QaBackendOutcome {
            backend: "cuda".into(),
            status: QaStatus::Skipped,
            reason: Some("backend was not compiled; enable cuda feature".into()),
        });
    }
    #[cfg(feature = "metal")]
    if required.as_deref().is_none_or(|backend| backend == "metal") {
        match MetalBackend::new() {
            Ok(backend) => {
                let (residual, solution_error) = run_backend("metal", &backend)?;
                summary.backend_count += 1;
                summary.max_final_residual = summary.max_final_residual.max(residual);
                summary.max_solution_error = summary.max_solution_error.max(solution_error);
                summary.backend_outcomes.push(QaBackendOutcome {
                    backend: "metal".into(),
                    status: QaStatus::Passed,
                    reason: None,
                });
            }
            Err(error) if required.as_deref() != Some("metal") => {
                summary.backend_outcomes.push(QaBackendOutcome {
                    backend: "metal".into(),
                    status: QaStatus::Skipped,
                    reason: Some(error.to_string()),
                });
            }
            Err(error) => return Err(format!("required Metal backend unavailable: {error}")),
        }
    }
    #[cfg(not(feature = "metal"))]
    if required.as_deref() == Some("metal") {
        return Err("required Metal backend unavailable: build with --features metal".into());
    }
    #[cfg(not(feature = "metal"))]
    if required.is_none() {
        summary.backend_outcomes.push(QaBackendOutcome {
            backend: "metal".into(),
            status: QaStatus::Skipped,
            reason: Some("backend was not compiled; enable the metal feature".into()),
        });
    }
    if summary.backend_count == 0 {
        return Err("no backend was selected for sparse workload validation".into());
    }
    Ok(summary)
}

fn run_backend<B: Backend>(name: &str, backend: &B) -> Result<(f64, f64), String> {
    let matrix = DeviceCsr::try_from_host(
        backend,
        2,
        2,
        &[0, 2, 4],
        &[0, 1, 0, 1],
        &[4.0_f64, 1.0, 1.0, 3.0],
    )
    .map_err(|e| e.to_string())?;
    let mut rhs = backend.raw_alloc::<f64>(4).map_err(|e| e.to_string())?;
    backend
        .try_copy_h2d(&[9.0, 8.0, 3.0, 7.0], &mut rhs)
        .map_err(|e| e.to_string())?;
    let mut solution = backend.raw_alloc::<f64>(4).map_err(|e| e.to_string())?;
    let status = gmres_backend_block_preconditioned(
        backend,
        &matrix,
        &IdentityDevicePreconditioner,
        &rhs,
        None,
        &mut solution,
        2,
        &BlockGmresConfig {
            max_block_iterations: 4,
            tolerance: 1.0e-10,
        },
    )
    .map_err(|e| e.to_string())?;
    let mut got = [0.0; 4];
    backend
        .try_copy_d2h(&solution, &mut got)
        .map_err(|e| e.to_string())?;
    let expected = [19.0 / 11.0, 23.0 / 11.0, 2.0 / 11.0, 25.0 / 11.0];
    let max_final_residual = status
        .rhs
        .iter()
        .map(|rhs| rhs.final_residual)
        .fold(0.0, f64::max);
    let max_solution_error = got
        .iter()
        .zip(expected)
        .map(|(got, expected)| (got - expected).abs())
        .fold(0.0, f64::max);
    if !status.converged
        || !status.rhs.iter().all(|rhs| rhs.converged)
        || !max_final_residual.is_finite()
        || !max_solution_error.is_finite()
        || max_final_residual > MAX_RESIDUAL
        || max_solution_error > MAX_SOLUTION_ERROR
    {
        return Err(format!(
            "block GMRES contract failed: {status:?}; solution={got:?}"
        ));
    }
    println!(
        "backend={name} workload=csr-spmv+block-gmres rhs=2 residuals={:?}",
        status
            .rhs
            .iter()
            .map(|rhs| rhs.final_residual)
            .collect::<Vec<_>>()
    );
    Ok((max_final_residual, max_solution_error))
}
