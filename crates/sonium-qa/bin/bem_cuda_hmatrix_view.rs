//! Strict CUDA evidence for resident H-matrix actions over device-buffer
//! views.  The hierarchy is uploaded once; input gather, action, output zero,
//! and output scatter are all native CUDA work.

use num_complex::Complex64;
use serde_json::json;
use sonium_compute::cuda::{
    CudaBackend, CudaDenseBlockC64Input, CudaHMatrixC64, CudaLowRankBlockC64Input,
};
use sonium_compute::{
    Backend, BackendError, BufferView, CudaFailureInjectionPoint, MemoryPool, MemoryPoolConfig,
};
use sonium_qa::{
    QaExecutionCounts, QaFailureStage, QaResult, cuda_hardware_provenance, cuda_runtime_evidence,
    required_cuda_backend, write_qa_artifact,
};
use std::time::Instant;

const CASE_ID: &str = "bem.cuda-hmatrix-view-device-residency.v1";
const ARTIFACT: &str = "bem_cuda_hmatrix_view_report.json";
const TOLERANCE: f64 = 1.0e-11;

#[derive(Clone, Copy)]
enum Action {
    Forward,
    Transpose,
    Hermitian,
}

impl Action {
    const ALL: [Self; 3] = [Self::Forward, Self::Transpose, Self::Hermitian];

    fn label(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Transpose => "transpose",
            Self::Hermitian => "hermitian",
        }
    }
}

fn relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    if actual.len() != expected.len()
        || actual
            .iter()
            .chain(expected)
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return f64::INFINITY;
    }
    let numerator = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = expected
        .iter()
        .map(Complex64::norm_sqr)
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    numerator / denominator
}

fn matrix_fixture() -> (
    CudaDenseBlockC64Input,
    CudaLowRankBlockC64Input,
    Vec<Vec<Complex64>>,
) {
    let dense = vec![
        Complex64::new(1.0, 0.2),
        Complex64::new(-0.3, 0.7),
        Complex64::new(0.5, -0.4),
        Complex64::new(0.2, -0.6),
        Complex64::new(1.3, 0.1),
        Complex64::new(-0.8, 0.25),
        Complex64::new(-0.4, 0.9),
        Complex64::new(0.6, 0.3),
        Complex64::new(0.75, -0.5),
    ];
    let u = vec![
        Complex64::new(0.8, 0.1),
        Complex64::new(-0.2, 0.5),
        Complex64::new(0.4, -0.7),
        Complex64::new(1.1, 0.2),
    ];
    let v = vec![
        Complex64::new(0.3, -0.4),
        Complex64::new(0.6, 0.2),
        Complex64::new(-0.5, 0.8),
        Complex64::new(0.7, -0.1),
    ];
    let low_rank = CudaLowRankBlockC64Input {
        row_indices: vec![3, 4],
        col_indices: vec![3, 4],
        u: u.clone(),
        v: v.clone(),
        rank: 2,
    };
    let dense_block = CudaDenseBlockC64Input {
        row_indices: vec![0, 1, 2],
        col_indices: vec![0, 1, 2],
        matrix: dense.clone(),
    };
    let mut matrix = vec![vec![Complex64::new(0.0, 0.0); 5]; 5];
    for row in 0..3 {
        for col in 0..3 {
            matrix[row][col] = dense[row * 3 + col];
        }
    }
    for row in 0..2 {
        for col in 0..2 {
            matrix[row + 3][col + 3] = (0..2)
                .map(|rank| u[row * 2 + rank] * v[col * 2 + rank])
                .sum();
        }
    }
    (dense_block, low_rank, matrix)
}

fn cpu_action(matrix: &[Vec<Complex64>], input: &[Complex64], action: Action) -> Vec<Complex64> {
    (0..matrix.len())
        .map(|row| {
            (0..matrix.len())
                .map(|column| {
                    let value = match action {
                        Action::Forward => matrix[row][column],
                        Action::Transpose => matrix[column][row],
                        Action::Hermitian => matrix[column][row].conj(),
                    };
                    value * input[column]
                })
                .sum()
        })
        .collect()
}

fn embed(values: &[Complex64], offset: usize, stride: usize, fill: Complex64) -> Vec<Complex64> {
    let length = offset + (values.len() - 1) * stride + 3;
    let mut storage = vec![fill; length];
    for (index, value) in values.iter().enumerate() {
        storage[offset + index * stride] = *value;
    }
    storage
}

fn extract(storage: &[Complex64], view: BufferView) -> Vec<Complex64> {
    (0..view.len)
        .map(|index| storage[view.offset + index * view.stride])
        .collect()
}

fn apply(
    hmatrix: &CudaHMatrixC64,
    input: &sonium_compute::pool::Buffer<Complex64, CudaBackend>,
    input_view: BufferView,
    output: &mut sonium_compute::pool::Buffer<Complex64, CudaBackend>,
    output_view: BufferView,
    action: Action,
) -> Result<(), String> {
    let result = match action {
        Action::Forward => hmatrix.try_apply_view(input, input_view, output, output_view),
        Action::Transpose => {
            hmatrix.try_apply_transpose_view(input, input_view, output, output_view)
        }
        Action::Hermitian => {
            hmatrix.try_apply_hermitian_view(input, input_view, output, output_view)
        }
    };
    result.map_err(|error| format!("{} H-matrix view action: {error}", action.label()))
}

fn failed_cuda(error: String, actual_backend: &str, stage: QaFailureStage) -> QaResult {
    QaResult::failed(CASE_ID, error.clone())
        .requested_backend("cuda")
        .actual_backend(actual_backend)
        .backend_failed("cuda", error)
        .failure_stage(stage)
        .hardware_provenance(cuda_hardware_provenance())
}

fn run(backend: CudaBackend) -> Result<QaResult, String> {
    if !backend.has_fmm_c64() {
        return Err("CUDA FMM module is not loaded; refusing a fallback result".into());
    }
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(256 << 20))
        .map_err(|error| format!("create CUDA memory pool: {error}"))?;
    let backend = pool.backend();
    let (dense, low_rank, matrix) = matrix_fixture();
    let hmatrix = CudaHMatrixC64::new(pool.clone(), 5, vec![dense], vec![low_rank])
        .map_err(|error| format!("upload resident H-matrix: {error}"))?;
    let input_values = vec![
        Complex64::new(0.25, -0.5),
        Complex64::new(-0.75, 0.2),
        Complex64::new(0.6, 0.9),
        Complex64::new(-0.1, -0.3),
        Complex64::new(1.2, 0.4),
    ];
    let input_view =
        BufferView::new(16, 1, 5, 3).map_err(|error| format!("input view: {error}"))?;
    let output_view =
        BufferView::new(13, 2, 5, 2).map_err(|error| format!("output view: {error}"))?;
    let sentinel = Complex64::new(73.0, -11.0);
    let input_storage = embed(
        &input_values,
        input_view.offset,
        input_view.stride,
        sentinel,
    );
    let output_storage = vec![sentinel; 13];
    let mut input = pool
        .try_alloc::<Complex64>(input_storage.len())
        .map_err(|error| format!("allocate input: {error}"))?;
    let mut input_seed = pool
        .try_alloc::<Complex64>(input_storage.len())
        .map_err(|error| format!("allocate input seed: {error}"))?;
    let mut output = pool
        .try_alloc::<Complex64>(output_storage.len())
        .map_err(|error| format!("allocate output: {error}"))?;
    let mut reset = pool
        .try_alloc::<Complex64>(output_storage.len())
        .map_err(|error| format!("allocate output reset: {error}"))?;
    backend
        .try_copy_h2d(&input_storage, input.raw_mut())
        .map_err(|error| format!("upload input: {error}"))?;
    backend
        .try_copy_h2d(&input_storage, input_seed.raw_mut())
        .map_err(|error| format!("upload input seed: {error}"))?;
    backend
        .try_copy_h2d(&output_storage, output.raw_mut())
        .map_err(|error| format!("upload output: {error}"))?;
    backend
        .try_copy_h2d(&output_storage, reset.raw_mut())
        .map_err(|error| format!("upload reset: {error}"))?;

    let mut action_errors = Vec::new();
    for action in Action::ALL {
        apply(
            &hmatrix,
            &input,
            input_view,
            &mut output,
            output_view,
            action,
        )?;
        backend
            .try_synchronize()
            .map_err(|error| format!("synchronize {} action: {error}", action.label()))?;
        let mut actual_storage = vec![Complex64::new(0.0, 0.0); output_storage.len()];
        backend
            .try_copy_d2h(output.raw(), &mut actual_storage)
            .map_err(|error| format!("read {} action: {error}", action.label()))?;
        let expected = cpu_action(&matrix, &input_values, action);
        let actual = extract(&actual_storage, output_view);
        let error = relative_error(&actual, &expected);
        if error > TOLERANCE {
            return Err(format!(
                "{} view parity {error:.6e} exceeds {TOLERANCE:.1e}",
                action.label()
            ));
        }
        if actual_storage.iter().enumerate().any(|(index, value)| {
            let is_target = (0..output_view.len)
                .any(|logical| output_view.checked_element_index(logical) == Ok(index));
            !is_target && *value != sentinel
        }) {
            return Err(format!(
                "{} action modified an output padding element",
                action.label()
            ));
        }
        action_errors.push(error);
    }

    // Checked malformed and empty views must fail before touching output.
    backend
        .try_copy_h2d(&output_storage, output.raw_mut())
        .map_err(|error| format!("reset malformed-output sentinel: {error}"))?;
    let malformed = BufferView {
        offset: output_storage.len() + 1,
        len: input_values.len(),
        stride: 1,
    };
    if hmatrix
        .try_apply_view(&input, input_view, &mut output, malformed)
        .is_ok()
    {
        return Err("malformed H-matrix output view was accepted".into());
    }
    let empty = BufferView::contiguous(0, 0);
    if hmatrix
        .try_apply_view(&input, empty, &mut output, empty)
        .is_ok()
    {
        return Err(
            "empty H-matrix input/output view was accepted for a non-empty operator".into(),
        );
    }
    let mut malformed_output = vec![Complex64::new(0.0, 0.0); output_storage.len()];
    backend
        .try_copy_d2h(output.raw(), &mut malformed_output)
        .map_err(|error| format!("read malformed-output sentinel: {error}"))?;
    if malformed_output != output_storage {
        return Err("malformed or empty view changed output".into());
    }

    // The high-level preflight injects a typed launch failure before gather or
    // publication.  The resident hierarchy and caller output are retryable.
    backend
        .try_copy_h2d(&output_storage, output.raw_mut())
        .map_err(|error| format!("seed launch-failure sentinel: {error}"))?;
    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = hmatrix
        .try_apply_view(&input, input_view, &mut output, output_view)
        .expect_err("injected H-matrix view launch failure must be typed");
    if !matches!(failure, BackendError::Launch { .. }) {
        return Err(format!("launch failure was not typed: {failure}"));
    }
    backend.clear_failure();
    backend
        .recover_after_error()
        .map_err(|error| format!("recover after launch failure: {error}"))?;
    let mut preserved = vec![Complex64::new(0.0, 0.0); output_storage.len()];
    backend
        .try_copy_d2h(output.raw(), &mut preserved)
        .map_err(|error| format!("read preserved launch-failure output: {error}"))?;
    if preserved != output_storage {
        return Err("typed launch failure modified caller output".into());
    }
    apply(
        &hmatrix,
        &input,
        input_view,
        &mut output,
        output_view,
        Action::Forward,
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("synchronize retry: {error}"))?;

    // Warm replays update a resident RHS and output slot by D2D only.  Readback
    // is deliberately inside the measured interval as the publication edge.
    backend.reset_runtime_stats();
    let warm_start = backend.begin_runtime_stats();
    let mut warm_errors = Vec::new();
    for _ in 0..2 {
        backend
            .try_copy_d2d(input_seed.raw(), input.raw_mut())
            .map_err(|error| format!("resident RHS D2D refresh: {error}"))?;
        backend
            .try_copy_d2d(reset.raw(), output.raw_mut())
            .map_err(|error| format!("resident output D2D reset: {error}"))?;
        apply(
            &hmatrix,
            &input,
            input_view,
            &mut output,
            output_view,
            Action::Forward,
        )?;
        backend
            .try_synchronize()
            .map_err(|error| format!("synchronize warm H-matrix replay: {error}"))?;
        let mut actual_storage = vec![Complex64::new(0.0, 0.0); output_storage.len()];
        backend
            .try_copy_d2h(output.raw(), &mut actual_storage)
            .map_err(|error| format!("publish warm H-matrix replay: {error}"))?;
        warm_errors.push(relative_error(
            &extract(&actual_storage, output_view),
            &cpu_action(&matrix, &input_values, Action::Forward),
        ));
    }
    let warm_stats = backend.runtime_stats_since(warm_start);
    let warm_error = warm_errors.iter().copied().fold(0.0, f64::max);
    if !warm_error.is_finite()
        || warm_error > TOLERANCE
        || warm_stats.allocation_operations != 0
        || warm_stats.transfer.h2d_operations != 0
        || warm_stats.transfer.d2d_operations < 4
        || warm_stats.transfer.d2h_operations != 2
        || warm_stats.kernel_launches == 0
    {
        return Err(format!(
            "warm resident view contract failed: error={warm_error:.6e}, stats={warm_stats:?}"
        ));
    }

    let stats = backend.runtime_stats();
    let teardown_before = stats.live_allocation_bytes;
    let teardown_high_water = stats.allocation_high_water_bytes;
    let warm_metrics = json!({
        "replays": 2,
        "rhs_d2d_updates": 2,
        "structural_allocations": warm_stats.allocation_operations,
        "structural_h2d_operations": warm_stats.transfer.h2d_operations,
        "d2d_operations": warm_stats.transfer.d2d_operations,
        "d2h_publications": warm_stats.transfer.d2h_operations,
        "max_relative_error": warm_error,
        "native_kernel_launches": warm_stats.kernel_launches,
    });
    drop(hmatrix);
    drop(input);
    drop(input_seed);
    drop(output);
    drop(reset);
    let teardown_after = backend.runtime_stats().live_allocation_bytes;
    if teardown_before == 0 || teardown_after >= teardown_before {
        return Err(format!(
            "resident teardown did not release bytes: before={teardown_before}, after={teardown_after}"
        ));
    }

    let mut result = QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .hardware_provenance(cuda_hardware_provenance())
        .metric(
            "max_action_relative_error",
            action_errors.iter().copied().fold(0.0, f64::max),
        )
        .metric("warm_max_action_relative_error", warm_error)
        .metric("warm_replays", 2.0)
        .metric("warm_rhs_d2d_updates", 2.0)
        .metric(
            "warm_structural_allocations",
            warm_stats.allocation_operations as f64,
        )
        .metric(
            "warm_structural_h2d_operations",
            warm_stats.transfer.h2d_operations as f64,
        )
        .metric(
            "warm_d2d_operations",
            warm_stats.transfer.d2d_operations as f64,
        )
        .metric(
            "warm_d2h_publications",
            warm_stats.transfer.d2h_operations as f64,
        )
        .metric(
            "warm_native_kernel_launches",
            warm_stats.kernel_launches as f64,
        )
        .metric("cuda_kernel_launches", stats.kernel_launches as f64)
        .metric(
            "cuda_allocation_high_water_bytes",
            teardown_high_water as f64,
        )
        .metric("cuda_teardown_live_bytes_before", teardown_before as f64)
        .metric("cuda_teardown_live_bytes_after", teardown_after as f64)
        .metric(
            "cuda_teardown_released_bytes",
            teardown_before.saturating_sub(teardown_after) as f64,
        )
        .maximum("max_action_relative_error", TOLERANCE)
        .maximum("warm_max_action_relative_error", TOLERANCE)
        .memory_bytes(teardown_high_water as u64)
        .reference("independent CPU dense-plus-low-rank matrix oracle")
        .tolerance(format!("<= {TOLERANCE:.1e}"));
    result
        .environment
        .insert("capability.bem.hmatrix.c64".into(), "native".into());
    result
        .environment
        .insert("capability.bem.hmatrix_view.c64".into(), "native".into());
    result.environment.insert(
        "capability.bem.hmatrix_setup.c64".into(),
        "host_shared_fallback".into(),
    );
    result.environment.insert(
        "claim.scope".into(),
        "resident Complex64 H-matrix view actions only".into(),
    );
    let artifact_bytes = serde_json::to_vec_pretty(&json!({
        "case_id": CASE_ID,
        "schema": "sonium-bem-cuda-hmatrix-view/v1",
        "claim_scope": "resident Complex64 H-matrix forward/transpose/Hermitian actions over checked offset/stride views; host setup remains shared",
        "matrix": {"num_dofs": 5, "dense_block": "3x3", "low_rank_block": "2x2 rank-2"},
        "views": {
            "input": {"offset": input_view.offset, "len": input_view.len, "stride": input_view.stride},
            "output": {"offset": output_view.offset, "len": output_view.len, "stride": output_view.stride},
            "tail_length": 5,
            "padding_preserved": true,
            "malformed_rejected": true,
            "empty_rejected": true,
        },
        "actions": ["forward", "transpose", "hermitian"],
        "failure_contract": {"typed_launch": true, "output_preserved": true, "retry_passed": true},
        "warm": warm_metrics,
        "metrics": result.metrics,
        "environment": result.environment,
    }))
    .map_err(|error| format!("serialize H-matrix view artifact: {error}"))?;
    let (artifact, checksum) = write_qa_artifact(ARTIFACT, &artifact_bytes)?;
    result = result
        .artifact_with_provenance(
            artifact,
            "crates/sonium-compute/src/cuda/hmatrix.rs",
            "native CUDA resident H-matrix view action evidence",
            "v1",
            checksum,
        )
        .runtime_evidence(cuda_runtime_evidence(
            stats,
            QaExecutionCounts {
                executed: 1,
                passed: 1,
                ..QaExecutionCounts::default()
            },
        ));
    Ok(result)
}

fn main() {
    let started = Instant::now();
    let result = match required_cuda_backend() {
        Ok(backend) => match run(backend) {
            Ok(result) => result,
            Err(error) => failed_cuda(error, "cuda", QaFailureStage::Execution),
        },
        Err(error) => failed_cuda(
            format!("required NVIDIA CUDA backend unavailable: {error}"),
            "unavailable",
            QaFailureStage::Initialization,
        ),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("H-matrix view result serializes")
    );
    if result.status != sonium_qa::QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
