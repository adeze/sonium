//! CPU QA for checked grouped dense GEMV/GEMM contracts.
//!
//! The ordinary operation matrix exercises one dense operation at a time.
//! This case independently covers the grouped APIs, where per-item shape
//! validation and destination preservation are easy to regress.

use num_complex::{Complex32, Complex64};
use num_traits::ToPrimitive;
use sonium_compute::{
    Backend, BackendError, CpuBackend, DeviceScalar, GemmBatch, GemvBatch, MemoryPool,
    MemoryPoolConfig, Transpose,
};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "compute.batched-dense-contract.v1";
const TOLERANCE: f64 = 5.0e-5;

fn stored_shape(op: Transpose, rows: usize, cols: usize) -> (usize, usize) {
    match op {
        Transpose::NoTrans => (rows, cols),
        Transpose::Trans | Transpose::ConjTrans => (cols, rows),
    }
}

fn matrix_value<T: DeviceScalar>(
    data: &[T],
    leading_dimension: usize,
    op: Transpose,
    row: usize,
    column: usize,
) -> T {
    let (stored_row, stored_column) = match op {
        Transpose::NoTrans => (row, column),
        Transpose::Trans | Transpose::ConjTrans => (column, row),
    };
    let value = data[stored_row * leading_dimension + stored_column];
    if matches!(op, Transpose::ConjTrans) {
        value.conj_mul(T::one())
    } else {
        value
    }
}

fn relative_error<T: DeviceScalar>(actual: T, expected: T) -> f64
where
    T::Real: ToPrimitive,
{
    let numerator = (actual - expected)
        .norm_sqr()
        .to_f64()
        .unwrap_or(f64::INFINITY)
        .sqrt();
    let denominator = expected
        .norm_sqr()
        .to_f64()
        .unwrap_or(f64::INFINITY)
        .sqrt()
        .max(1.0);
    numerator / denominator
}

#[derive(Default)]
struct Metrics {
    scalar_types: usize,
    gemv_cases: usize,
    gemm_cases: usize,
    malformed_rejections: usize,
    max_relative_error: f64,
}

fn run_scalar<T, F>(backend: &CpuBackend, make_scalar: F, label: &str) -> Result<Metrics, String>
where
    T: DeviceScalar + PartialEq,
    T::Real: ToPrimitive,
    F: Fn(usize) -> T + Copy,
{
    let pool = MemoryPool::new(backend.clone(), MemoryPoolConfig::with_cap(1 << 20))
        .map_err(|error| format!("{label}: pool: {error}"))?;
    let backend = pool.backend();
    let mut metrics = Metrics {
        scalar_types: 1,
        ..Metrics::default()
    };
    let operations = [
        (Transpose::NoTrans, Transpose::NoTrans),
        (Transpose::Trans, Transpose::ConjTrans),
        (Transpose::ConjTrans, Transpose::Trans),
    ];

    let m = 2;
    let n = 2;
    let k = 2;

    let mut gemv_batch = Vec::with_capacity(operations.len());
    let mut gemv_expected = Vec::with_capacity(operations.len());
    let mut gemv_matrices = Vec::with_capacity(operations.len());
    let mut gemv_inputs = Vec::with_capacity(operations.len());
    let mut gemv_outputs = Vec::with_capacity(operations.len());
    for (case, &(op, _)) in operations.iter().enumerate() {
        let (a_rows, a_columns) = stored_shape(op, m, n);
        let lda = a_columns + 1;
        let a_host: Vec<_> = (0..a_rows * lda)
            .map(|i| make_scalar(100 + case * 20 + i))
            .collect();
        let x_host: Vec<_> = (0..n).map(|i| make_scalar(200 + case * 20 + i)).collect();
        let y_host: Vec<_> = (0..m).map(|i| make_scalar(300 + case * 20 + i)).collect();
        let alpha = make_scalar(400 + case);
        let beta = make_scalar(500 + case);
        let mut a = backend
            .raw_alloc::<T>(a_host.len())
            .map_err(|error| format!("{label}: GEMV {case} A allocation: {error}"))?;
        let mut x = backend
            .raw_alloc::<T>(x_host.len())
            .map_err(|error| format!("{label}: GEMV {case} x allocation: {error}"))?;
        let mut y = backend
            .raw_alloc::<T>(y_host.len())
            .map_err(|error| format!("{label}: GEMV {case} y allocation: {error}"))?;
        backend
            .try_copy_h2d(&a_host, &mut a)
            .map_err(|error| format!("{label}: GEMV {case} A upload: {error}"))?;
        backend
            .try_copy_h2d(&x_host, &mut x)
            .map_err(|error| format!("{label}: GEMV {case} x upload: {error}"))?;
        backend
            .try_copy_h2d(&y_host, &mut y)
            .map_err(|error| format!("{label}: GEMV {case} y upload: {error}"))?;
        let expected: Vec<_> = (0..m)
            .map(|row| {
                let product = (0..n).fold(T::zero(), |sum, column| {
                    sum + matrix_value(&a_host, lda, op, row, column) * x_host[column]
                });
                alpha * product + beta * y_host[row]
            })
            .collect();
        gemv_expected.push(expected);
        gemv_matrices.push(a);
        gemv_inputs.push(x);
        gemv_outputs.push(y);
    }
    let [y0, y1, y2] = &mut gemv_outputs[..] else {
        return Err(format!(
            "{label}: internal GEMV fixture cardinality mismatch"
        ));
    };
    gemv_batch.extend([
        GemvBatch {
            op: operations[0].0,
            m,
            n,
            alpha: make_scalar(400),
            a: &gemv_matrices[0],
            lda: stored_shape(operations[0].0, m, n).1 + 1,
            x: &gemv_inputs[0],
            beta: make_scalar(500),
            y: y0,
        },
        GemvBatch {
            op: operations[1].0,
            m,
            n,
            alpha: make_scalar(401),
            a: &gemv_matrices[1],
            lda: stored_shape(operations[1].0, m, n).1 + 1,
            x: &gemv_inputs[1],
            beta: make_scalar(501),
            y: y1,
        },
        GemvBatch {
            op: operations[2].0,
            m,
            n,
            alpha: make_scalar(402),
            a: &gemv_matrices[2],
            lda: stored_shape(operations[2].0, m, n).1 + 1,
            x: &gemv_inputs[2],
            beta: make_scalar(502),
            y: y2,
        },
    ]);

    backend
        .try_gemv_batched(&mut gemv_batch)
        .map_err(|error| format!("{label}: grouped GEMV dispatch: {error}"))?;
    drop(gemv_batch);
    for (index, output) in gemv_outputs.iter().enumerate() {
        let mut actual = vec![T::zero(); m];
        backend
            .try_copy_d2h(output, &mut actual)
            .map_err(|error| format!("{label}: GEMV {index} download: {error}"))?;
        for (actual, expected) in actual.into_iter().zip(&gemv_expected[index]) {
            metrics.max_relative_error = metrics
                .max_relative_error
                .max(relative_error(actual, *expected));
        }
        metrics.gemv_cases += 1;
    }

    // Grouped GEMM uses independent padded row-major fixtures and all three
    // transpose/conjugate-transpose combinations above.
    let mut gemm_batch = Vec::with_capacity(operations.len());
    let mut gemm_expected = Vec::with_capacity(operations.len());
    let mut gemm_matrices_a = Vec::with_capacity(operations.len());
    let mut gemm_matrices_b = Vec::with_capacity(operations.len());
    let mut gemm_outputs = Vec::with_capacity(operations.len());
    for (case, &(transa, transb)) in operations.iter().enumerate() {
        let (a_rows, a_columns) = stored_shape(transa, m, k);
        let (b_rows, b_columns) = stored_shape(transb, k, n);
        let lda = a_columns + 1;
        let ldb = b_columns + 1;
        let ldc = n + 1;
        let a_host: Vec<_> = (0..a_rows * lda)
            .map(|i| make_scalar(600 + case * 30 + i))
            .collect();
        let b_host: Vec<_> = (0..b_rows * ldb)
            .map(|i| make_scalar(700 + case * 30 + i))
            .collect();
        let c_host: Vec<_> = (0..m * ldc)
            .map(|i| make_scalar(800 + case * 30 + i))
            .collect();
        let alpha = make_scalar(900 + case);
        let beta = make_scalar(1000 + case);
        let mut a = backend
            .raw_alloc::<T>(a_host.len())
            .map_err(|error| error.to_string())?;
        let mut b = backend
            .raw_alloc::<T>(b_host.len())
            .map_err(|error| error.to_string())?;
        let mut c = backend
            .raw_alloc::<T>(c_host.len())
            .map_err(|error| error.to_string())?;
        backend
            .try_copy_h2d(&a_host, &mut a)
            .map_err(|error| error.to_string())?;
        backend
            .try_copy_h2d(&b_host, &mut b)
            .map_err(|error| error.to_string())?;
        backend
            .try_copy_h2d(&c_host, &mut c)
            .map_err(|error| error.to_string())?;
        let expected: Vec<_> = (0..m * ldc)
            .map(|i| {
                if i % ldc >= n {
                    c_host[i]
                } else {
                    let row = i / ldc;
                    let column = i % ldc;
                    let product = (0..k).fold(T::zero(), |sum, inner| {
                        sum + matrix_value(&a_host, lda, transa, row, inner)
                            * matrix_value(&b_host, ldb, transb, inner, column)
                    });
                    alpha * product + beta * c_host[i]
                }
            })
            .collect();
        gemm_expected.push(expected);
        gemm_matrices_a.push(a);
        gemm_matrices_b.push(b);
        gemm_outputs.push(c);
    }
    let [c0, c1, c2] = &mut gemm_outputs[..] else {
        return Err(format!(
            "{label}: internal GEMM fixture cardinality mismatch"
        ));
    };
    gemm_batch.extend([
        GemmBatch {
            transa: operations[0].0,
            transb: operations[0].1,
            m,
            n,
            k,
            alpha: make_scalar(900),
            a: &gemm_matrices_a[0],
            lda: stored_shape(operations[0].0, m, k).1 + 1,
            b: &gemm_matrices_b[0],
            ldb: stored_shape(operations[0].1, k, n).1 + 1,
            beta: make_scalar(1000),
            c: c0,
            ldc: n + 1,
        },
        GemmBatch {
            transa: operations[1].0,
            transb: operations[1].1,
            m,
            n,
            k,
            alpha: make_scalar(901),
            a: &gemm_matrices_a[1],
            lda: stored_shape(operations[1].0, m, k).1 + 1,
            b: &gemm_matrices_b[1],
            ldb: stored_shape(operations[1].1, k, n).1 + 1,
            beta: make_scalar(1001),
            c: c1,
            ldc: n + 1,
        },
        GemmBatch {
            transa: operations[2].0,
            transb: operations[2].1,
            m,
            n,
            k,
            alpha: make_scalar(902),
            a: &gemm_matrices_a[2],
            lda: stored_shape(operations[2].0, m, k).1 + 1,
            b: &gemm_matrices_b[2],
            ldb: stored_shape(operations[2].1, k, n).1 + 1,
            beta: make_scalar(1002),
            c: c2,
            ldc: n + 1,
        },
    ]);
    backend
        .try_gemm_batched(&mut gemm_batch)
        .map_err(|error| format!("{label}: grouped GEMM dispatch: {error}"))?;
    drop(gemm_batch);
    for (index, output) in gemm_outputs.iter().enumerate() {
        let mut actual = vec![T::zero(); m * (n + 1)];
        backend
            .try_copy_d2h(output, &mut actual)
            .map_err(|error| format!("{label}: GEMM {index} download: {error}"))?;
        for (actual, expected) in actual.into_iter().zip(&gemm_expected[index]) {
            metrics.max_relative_error = metrics
                .max_relative_error
                .max(relative_error(actual, *expected));
        }
        metrics.gemm_cases += 1;
    }

    // Malformed grouped entries must fail before touching the destination.
    let mut a = backend
        .raw_alloc::<T>(4)
        .map_err(|error| error.to_string())?;
    let x = backend
        .raw_alloc::<T>(1)
        .map_err(|error| error.to_string())?;
    let mut y = backend
        .raw_alloc::<T>(2)
        .map_err(|error| error.to_string())?;
    let sentinel = [make_scalar(1200), make_scalar(1201)];
    backend
        .try_copy_h2d(
            &[
                make_scalar(1),
                make_scalar(2),
                make_scalar(3),
                make_scalar(4),
            ],
            &mut a,
        )
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&sentinel, &mut y)
        .map_err(|error| error.to_string())?;
    let mut malformed = [GemvBatch {
        op: Transpose::NoTrans,
        m: 2,
        n: 2,
        alpha: make_scalar(1),
        a: &a,
        lda: 2,
        x: &x,
        beta: make_scalar(0),
        y: &mut y,
    }];
    let rejected = matches!(
        backend.try_gemv_batched(&mut malformed),
        Err(BackendError::InvalidArgument { .. })
    );
    let mut after = [T::zero(); 2];
    backend
        .try_copy_d2h(&y, &mut after)
        .map_err(|error| error.to_string())?;
    let unchanged = after == sentinel;
    if !rejected || !unchanged {
        return Err(format!(
            "{label}: malformed GEMV was not rejected transactionally"
        ));
    }
    metrics.malformed_rejections += 1;

    let mut b = backend
        .raw_alloc::<T>(2)
        .map_err(|error| error.to_string())?;
    let mut c = backend
        .raw_alloc::<T>(4)
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&[make_scalar(5), make_scalar(6)], &mut b)
        .map_err(|error| error.to_string())?;
    let gemm_sentinel = [sentinel[0], sentinel[1], sentinel[0], sentinel[1]];
    backend
        .try_copy_h2d(&gemm_sentinel, &mut c)
        .map_err(|error| error.to_string())?;
    let mut malformed_gemm = [GemmBatch {
        transa: Transpose::NoTrans,
        transb: Transpose::NoTrans,
        m: 2,
        n: 2,
        k: 2,
        alpha: make_scalar(1),
        a: &a,
        lda: 2,
        b: &b,
        ldb: 2,
        beta: make_scalar(0),
        c: &mut c,
        ldc: 2,
    }];
    let rejected_gemm = matches!(
        backend.try_gemm_batched(&mut malformed_gemm),
        Err(BackendError::InvalidArgument { .. })
    );
    let mut after_gemm = [T::zero(); 4];
    backend
        .try_copy_d2h(&c, &mut after_gemm)
        .map_err(|error| error.to_string())?;
    if !rejected_gemm || after_gemm != gemm_sentinel {
        return Err(format!(
            "{label}: malformed GEMM was not rejected transactionally"
        ));
    }
    metrics.malformed_rejections += 1;
    Ok(metrics)
}

fn main() {
    let started = Instant::now();
    let backend = CpuBackend::new();
    let runs = [
        run_scalar::<f32, _>(&backend, |i| (i as f32 * 0.125) - 1.0, "f32"),
        run_scalar::<f64, _>(&backend, |i| (i as f64 * 0.125) - 1.0, "f64"),
        run_scalar::<Complex32, _>(
            &backend,
            |i| Complex32::new(i as f32 * 0.125 - 1.0, i as f32 * 0.0625),
            "Complex32",
        ),
        run_scalar::<Complex64, _>(
            &backend,
            |i| Complex64::new(i as f64 * 0.125 - 1.0, i as f64 * 0.0625),
            "Complex64",
        ),
    ];
    let result = match runs.into_iter().collect::<Result<Vec<_>, _>>() {
        Ok(metrics) => {
            let scalar_types = metrics.iter().map(|m| m.scalar_types).sum::<usize>();
            let gemv_cases = metrics.iter().map(|m| m.gemv_cases).sum::<usize>();
            let gemm_cases = metrics.iter().map(|m| m.gemm_cases).sum::<usize>();
            let malformed_rejections = metrics
                .iter()
                .map(|m| m.malformed_rejections)
                .sum::<usize>();
            let max_relative_error = metrics
                .iter()
                .map(|m| m.max_relative_error)
                .fold(0.0, f64::max);
            if max_relative_error > TOLERANCE {
                QaResult::failed(
                    CASE_ID,
                    format!(
                        "maximum relative error {max_relative_error:.3e} exceeds {TOLERANCE:.3e}"
                    ),
                )
            } else {
                QaResult::passed(CASE_ID)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .input_checksum("analytic/deterministic-fixture:compute-batched-dense-v1")
                    .reference("independent scalar grouped GEMV/GEMM accumulation")
                    .reference_checksum("analytic/deterministic-fixture:compute-batched-dense-v1")
                    .dimension("three grouped transpose/conjugate-transpose entries for four scalar families")
                    .formulation("checked caller-owned batched dense backend contract")
                    .tolerance("relative error <= 5e-5; malformed destination-preserving rejection")
                    .seed("compute-batched-dense:v1")
                    .metric("scalar_type_count", scalar_types as f64)
                    .metric("gemv_case_count", gemv_cases as f64)
                    .metric("gemm_case_count", gemm_cases as f64)
                    .metric("malformed_rejection_count", malformed_rejections as f64)
                    .metric("max_relative_error", max_relative_error)
                    .maximum("max_relative_error", TOLERANCE)
            }
        }
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    let result = result
        .elapsed_ms(started.elapsed().as_millis() as u64)
        .timing_ms("total", started.elapsed().as_millis() as u64);
    match result.to_json_line() {
        Ok(line) => println!("QA_RESULT: {line}"),
        Err(error) => {
            eprintln!("QA_RESULT serialization failed: {error}");
            std::process::exit(1);
        }
    }
    if matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
