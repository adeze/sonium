//! Native x86 facade conformance evidence.
//!
//! This producer runs the public generic linear-solve facade for all four
//! scalar families under one explicitly selected CPU dispatch profile.  The
//! stable S3 runner invokes it once with runtime AVX2/FMA and once with both
//! compute and sparse dispatch forced to scalar, then compares the resulting
//! machine-readable records.  Other public solver families remain in the
//! inventory with a precise reason when this bounded CSR fixture does not
//! exercise them; an inventory row is never silently presented as executed.

use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, ToPrimitive};
use serde_json::{Value, json};
use sonium_compute::{Backend, CapabilityState, CpuBackend};
use sonium_qa::{QaResult, host_hardware_provenance, write_qa_artifact};
use sonium_solvers::{
    ComplexField, CsrMatrix, LinearSolveConfig, LinearSolverType, SOLVER_CONFORMANCE,
    SolverConformance, detected_spmv_features, selected_spmv_kernel, solve,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.facade-native-x86.v1";
const ARTIFACT: &str = "solvers_facade_native_x86.json";
const MATRIX_SIZE: usize = 4;

fn value<T: ComplexField>(real: f64, imaginary: f64) -> T
where
    T::Real: FromPrimitive,
{
    T::from_re_im(
        T::Real::from_f64(real).expect("fixture real value is representable"),
        T::Real::from_f64(imaginary).expect("fixture imaginary value is representable"),
    )
}

fn fixture_matrix<T: ComplexField>() -> CsrMatrix<T>
where
    T::Real: FromPrimitive,
{
    CsrMatrix::from_raw_parts(
        MATRIX_SIZE,
        MATRIX_SIZE,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![
            value(4.0, 0.0),
            value(1.0, 0.0),
            value(1.0, 0.0),
            value(3.0, 0.0),
            value(1.0, 0.0),
            value(1.0, 0.0),
            value(3.0, 0.0),
            value(1.0, 0.0),
            value(1.0, 0.0),
            value(2.0, 0.0),
        ],
    )
}

fn fixture_rhs<T: ComplexField>() -> Vec<T>
where
    T::Real: FromPrimitive,
{
    // A * [1, -2, 0.5, 3]^T.  Keeping the exact solution known makes the
    // facade record independently checkable in addition to its true residual.
    [2.0, -4.5, 2.5, 6.5]
        .into_iter()
        .map(|entry| value(entry, 0.0))
        .collect()
}

fn expected_solution<T: ComplexField>() -> Vec<T>
where
    T::Real: FromPrimitive,
{
    [1.0, -2.0, 0.5, 3.0]
        .into_iter()
        .map(|entry| value(entry, 0.0))
        .collect()
}

fn scalar_label<T: ComplexField>() -> &'static str {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        "f32"
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        "f64"
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex32>() {
        "complex32"
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex64>() {
        "complex64"
    } else {
        std::any::type_name::<T>()
    }
}

fn json_scalar<T: ComplexField>(entry: T) -> Value
where
    T::Real: ToPrimitive,
{
    json!({
        "re": entry.re().to_f64().unwrap_or(f64::NAN),
        "im": entry.im().to_f64().unwrap_or(f64::NAN),
    })
}

fn max_solution_error<T: ComplexField>(actual: &[T], expected: &[T]) -> f64
where
    T::Real: ToPrimitive,
{
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            (*actual - *expected)
                .norm()
                .to_f64()
                .unwrap_or(f64::INFINITY)
        })
        .fold(0.0, f64::max)
}

fn run_linear_case<T>(tolerance: f64) -> Result<Value, String>
where
    T: ComplexField + Send + Sync,
    T::Real: FromPrimitive + ToPrimitive + Send + Sync,
{
    let matrix = fixture_matrix::<T>();
    let rhs = fixture_rhs::<T>();
    let expected = expected_solution::<T>();
    let config = LinearSolveConfig {
        solver: LinearSolverType::Gmres,
        max_iterations: 32,
        restart: 8,
        tolerance,
        exact_residual_check_interval: 1,
        ..LinearSolveConfig::default()
    };
    let result = solve(&matrix, &rhs, &config)
        .map_err(|error| format!("{} facade solve failed: {error}", scalar_label::<T>()))?;
    result
        .validate_contract()
        .map_err(|error| format!("{} result contract failed: {error}", scalar_label::<T>()))?;
    let max_error = max_solution_error(&result.x, &expected);
    if !max_error.is_finite() {
        return Err(format!(
            "{} solution error is non-finite",
            scalar_label::<T>()
        ));
    }
    Ok(json!({
        "family": "facade",
        "algorithm": "solve",
        "scalar": scalar_label::<T>(),
        "backend": "cpu",
        "solver": "gmres",
        "status": format!("{:?}", result.status),
        "converged": result.converged,
        "attainable_accuracy": format!("{:?}", result.attainable_accuracy),
        "iterations": result.iterations,
        "residual": result.residual,
        "backward_error": result.backward_error,
        "max_solution_error": max_error,
        "solution": result.x.iter().copied().map(json_scalar).collect::<Vec<_>>(),
        "contract_valid": true,
        "matrix_size": MATRIX_SIZE,
    }))
}

fn capability_state(state: CapabilityState) -> &'static str {
    match state {
        CapabilityState::Native => "native",
        CapabilityState::VendorNative => "vendor_native",
        CapabilityState::HostSharedFallback => "host_shared_fallback",
        CapabilityState::HostStagedFallback => "host_staged_fallback",
        CapabilityState::Unsupported => "unsupported",
    }
}

fn scalar_capabilities(capabilities: sonium_compute::ScalarCapabilities) -> Value {
    json!({
        "f32": capability_state(capabilities.f32),
        "f64": capability_state(capabilities.f64),
        "complex32": capability_state(capabilities.c32),
        "complex64": capability_state(capabilities.c64),
    })
}

fn backend_dispatch_record(mode: &str) -> Value {
    let cpu = CpuBackend::new();
    let capabilities = cpu.capabilities();
    let avx2 = cfg!(target_arch = "x86_64") && std::is_x86_feature_detected!("avx2");
    let fma = cfg!(target_arch = "x86_64") && std::is_x86_feature_detected!("fma");
    let force_scalar = std::env::var_os("SONIUM_COMPUTE_FORCE_SCALAR").is_some()
        || std::env::var_os("SONIUM_SPARSE_FORCE_SCALAR").is_some();
    json!({
        "backend_name": capabilities.backend_name,
        "backend": "cpu",
        "mode": mode,
        "runtime_features": {"avx2": avx2, "fma": fma},
        "force_scalar": force_scalar,
        "sparse_dispatch_features": detected_spmv_features(),
        "sparse_selected_kernels": {
            "f32": selected_spmv_kernel::<f32>().name(),
            "f64": selected_spmv_kernel::<f64>().name(),
            "complex32": selected_spmv_kernel::<Complex32>().name(),
            "complex64": selected_spmv_kernel::<Complex64>().name(),
        },
        "dense_dispatch": {
            "kernel_family": "sonium_compute::cpu::blas1+blas2+blas3",
            "selection": if force_scalar { "scalar" } else if avx2 && fma { "avx2-fma" } else { "runtime-fallback" },
            "vendor_library": "none; in-tree CPU kernels",
        },
        "capabilities": {
            "blas1": scalar_capabilities(capabilities.blas1),
            "gemv": scalar_capabilities(capabilities.gemv),
            "gemm": scalar_capabilities(capabilities.gemm),
            "csr_spmv": scalar_capabilities(capabilities.csr_spmv),
            "csr_spmm": scalar_capabilities(capabilities.csr_spmm),
            "csr_transpose": scalar_capabilities(capabilities.csr_transpose),
        },
    })
}

fn inventory_rows() -> Vec<Value> {
    SOLVER_CONFORMANCE
        .iter()
        .map(|row: &SolverConformance| {
            let executed = row.family == "facade" && row.algorithm == "solve";
            json!({
                "family": row.family,
                "algorithm": row.algorithm,
                "maturity": format!("{:?}", row.maturity),
                "scalar_types": row.scalar_types,
                "status": if executed { "executed" } else { "out_of_fixture" },
                "reason": if executed {
                    "executed for f32/f64/Complex32/Complex64 by generic public facade"
                } else if row.family == "facade" {
                    "requires a distinct block/compatibility fixture; covered by its dedicated QA contract"
                } else {
                    "method-specific family dispatch is covered by its dedicated QA contract; S3 parity fixture is generic CSR facade only"
                },
            })
        })
        .collect()
}

fn mode() -> Result<String, String> {
    let mode = std::env::var("SONIUM_S3_DISPATCH_MODE")
        .map_err(|_| "SONIUM_S3_DISPATCH_MODE must be native-avx2-fma or scalar".to_owned())?;
    if !matches!(mode.as_str(), "native-avx2-fma" | "scalar") {
        return Err(format!("unsupported S3 dispatch mode {mode:?}"));
    }
    if !cfg!(target_arch = "x86_64") {
        return Err("S3 requires a native x86_64 process; Arm64/Windows are waived".into());
    }
    if mode == "native-avx2-fma"
        && (!std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma"))
    {
        return Err("native-avx2-fma requested but runtime AVX2/FMA is unavailable".into());
    }
    Ok(mode)
}

fn produce() -> Result<QaResult, String> {
    let started = Instant::now();
    let mode = mode()?;
    let tolerance = 1.0e-6;
    let executions = vec![
        run_linear_case::<f32>(tolerance)?,
        run_linear_case::<f64>(1.0e-12)?,
        run_linear_case::<Complex32>(tolerance)?,
        run_linear_case::<Complex64>(1.0e-12)?,
    ];
    let artifact = json!({
        "schema_version": 1,
        "case_id": CASE_ID,
        "status": "passed",
        "scope": "native Linux x86_64 facade CPU dispatch; Arm64/Windows waived",
        "dispatch_mode": mode,
        "requested_backend": "cpu",
        "actual_backend": "cpu",
        "hardware": host_hardware_provenance(),
        "matrix": {
            "rows": MATRIX_SIZE,
            "columns": MATRIX_SIZE,
            "fixture": "symmetric positive definite tridiagonal CSR",
            "known_solution": [1.0, -2.0, 0.5, 3.0],
        },
        "dispatch": backend_dispatch_record(&mode),
        "executions": executions,
        "production_inventory": inventory_rows(),
        "comparison": {
            "peer_artifact": if mode == "native-avx2-fma" { "scalar/solvers_facade_native_x86.json" } else { "native/solvers_facade_native_x86.json" },
            "status_iterations_residual_solution_checked_by_runner": true,
            "budget": {"f32": 1.0e-5, "f64": 1.0e-10, "complex32": 1.0e-5, "complex64": 1.0e-10},
        },
    });
    let bytes = serde_json::to_vec_pretty(&artifact)
        .map_err(|error| format!("serialize native x86 facade artifact: {error}"))?;
    let (path, checksum) = write_qa_artifact(ARTIFACT, &bytes)?;
    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .backend_passed("cpu")
        .reference("known 4x4 SPD CSR solution and independent scalar/native dispatch comparison")
        .reference_checksum("analytic:solver-facade-native-x86-v1")
        .dimension("generic facade solve across f32/f64/Complex32/Complex64")
        .formulation("native AVX2/FMA versus forced-scalar public facade conformance")
        .metric("executed_scalar_cells", executions.len() as f64)
        .metric("production_inventory_rows", SOLVER_CONFORMANCE.len() as f64)
        .metric("out_of_fixture_rows", (SOLVER_CONFORMANCE.len() - 1) as f64)
        .minimum("executed_scalar_cells", 4.0)
        .artifact_with_provenance(
            path,
            "crates/sonium-qa/bin/solvers_facade_native_x86.rs",
            "public generic facade scalar parity and production inventory",
            "v1",
            checksum,
        )
        .hardware_provenance(host_hardware_provenance())
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    match produce() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                serde_json::to_string(&result).expect("S3 QA JSON")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error).requested_backend("cpu");
            println!(
                "QA_RESULT: {}",
                serde_json::to_string(&result).expect("S3 QA failure JSON")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
