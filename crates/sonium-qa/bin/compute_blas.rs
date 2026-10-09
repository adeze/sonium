//! Release-facing dense BLAS contract against an independent scalar reference.

use num_complex::Complex64;
use rand::{Rng, SeedableRng, rngs::StdRng};
use sonium_compute::{Backend, CpuBackend, MemoryPool, MemoryPoolConfig, Transpose};
use sonium_qa::QaResult;
use std::time::Instant;

const SEED: u64 = 0xC0A4_2028_0809_0001;
const TOLERANCE: f64 = 3.0e-12;

fn scalar(rng: &mut StdRng) -> Complex64 {
    Complex64::new(rng.random_range(-2.0..=2.0), rng.random_range(-2.0..=2.0))
}

fn stored_shape(op: Transpose, rows: usize, cols: usize) -> (usize, usize) {
    match op {
        Transpose::NoTrans => (rows, cols),
        Transpose::Trans | Transpose::ConjTrans => (cols, rows),
    }
}

fn matrix_value(
    data: &[Complex64],
    leading_dimension: usize,
    op: Transpose,
    row: usize,
    column: usize,
) -> Complex64 {
    let (stored_row, stored_column) = match op {
        Transpose::NoTrans => (row, column),
        Transpose::Trans | Transpose::ConjTrans => (column, row),
    };
    let value = data[stored_row * leading_dimension + stored_column];
    if matches!(op, Transpose::ConjTrans) {
        value.conj()
    } else {
        value
    }
}

fn run() -> Result<(f64, f64, usize), String> {
    let backend = CpuBackend::new();
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20))
        .map_err(|error| error.to_string())?;
    let backend = pool.backend();
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut max_gemv_error = 0.0_f64;
    let mut max_gemm_error = 0.0_f64;
    let mut cases = 0usize;

    for case in 0..24 {
        let rows = 1 + (case * 5 % 11);
        let cols = 1 + (case * 7 % 10);
        let operation = match case % 3 {
            0 => Transpose::NoTrans,
            1 => Transpose::Trans,
            _ => Transpose::ConjTrans,
        };
        let (stored_rows, stored_cols) = stored_shape(operation, rows, cols);
        let leading_dimension = stored_cols + case % 3;
        let matrix: Vec<_> = (0..stored_rows * leading_dimension)
            .map(|_| scalar(&mut rng))
            .collect();
        let x_host: Vec<_> = (0..cols).map(|_| scalar(&mut rng)).collect();
        let y_initial: Vec<_> = (0..rows).map(|_| scalar(&mut rng)).collect();
        let alpha = scalar(&mut rng);
        let beta = scalar(&mut rng);
        let mut matrix_buffer = pool
            .try_alloc::<Complex64>(matrix.len())
            .map_err(|error| error.to_string())?;
        let mut x_buffer = pool
            .try_alloc::<Complex64>(x_host.len())
            .map_err(|error| error.to_string())?;
        let mut y_buffer = pool
            .try_alloc::<Complex64>(y_initial.len())
            .map_err(|error| error.to_string())?;
        backend.copy_h2d(&matrix, matrix_buffer.raw_mut());
        backend.copy_h2d(&x_host, x_buffer.raw_mut());
        backend.copy_h2d(&y_initial, y_buffer.raw_mut());
        backend
            .try_gemv(
                operation,
                rows,
                cols,
                alpha,
                matrix_buffer.raw(),
                leading_dimension,
                x_buffer.raw(),
                beta,
                y_buffer.raw_mut(),
            )
            .map_err(|error| format!("GEMV case {case} rejected: {error}"))?;
        let mut actual = vec![Complex64::default(); rows];
        backend.copy_d2h(y_buffer.raw(), &mut actual);
        for row in 0..rows {
            let expected = alpha
                * (0..cols)
                    .map(|column| {
                        matrix_value(&matrix, leading_dimension, operation, row, column)
                            * x_host[column]
                    })
                    .sum::<Complex64>()
                + beta * y_initial[row];
            let scale = expected.norm().max(1.0);
            max_gemv_error = max_gemv_error.max((actual[row] - expected).norm() / scale);
        }

        let right_cols = 1 + (case * 3 % 8);
        let left_operation = operation;
        let right_operation = match (case / 3) % 3 {
            0 => Transpose::NoTrans,
            1 => Transpose::Trans,
            _ => Transpose::ConjTrans,
        };
        let (a_rows, a_cols) = stored_shape(left_operation, rows, cols);
        let (b_rows, b_cols) = stored_shape(right_operation, cols, right_cols);
        let lda = a_cols + case % 2;
        let ldb = b_cols + (case + 1) % 2;
        let ldc = right_cols + case % 3;
        let a_host: Vec<_> = (0..a_rows * lda).map(|_| scalar(&mut rng)).collect();
        let b_host: Vec<_> = (0..b_rows * ldb).map(|_| scalar(&mut rng)).collect();
        let c_initial: Vec<_> = (0..rows * ldc).map(|_| scalar(&mut rng)).collect();
        let alpha = scalar(&mut rng);
        let beta = scalar(&mut rng);
        let mut a = pool
            .try_alloc::<Complex64>(a_host.len())
            .map_err(|error| error.to_string())?;
        let mut b = pool
            .try_alloc::<Complex64>(b_host.len())
            .map_err(|error| error.to_string())?;
        let mut c = pool
            .try_alloc::<Complex64>(c_initial.len())
            .map_err(|error| error.to_string())?;
        backend.copy_h2d(&a_host, a.raw_mut());
        backend.copy_h2d(&b_host, b.raw_mut());
        backend.copy_h2d(&c_initial, c.raw_mut());
        backend
            .try_gemm(
                left_operation,
                right_operation,
                rows,
                right_cols,
                cols,
                alpha,
                a.raw(),
                lda,
                b.raw(),
                ldb,
                beta,
                c.raw_mut(),
                ldc,
            )
            .map_err(|error| format!("GEMM case {case} rejected: {error}"))?;
        let mut actual = vec![Complex64::default(); c_initial.len()];
        backend.copy_d2h(c.raw(), &mut actual);
        for row in 0..rows {
            for column in 0..right_cols {
                let expected = alpha
                    * (0..cols)
                        .map(|inner| {
                            matrix_value(&a_host, lda, left_operation, row, inner)
                                * matrix_value(&b_host, ldb, right_operation, inner, column)
                        })
                        .sum::<Complex64>()
                    + beta * c_initial[row * ldc + column];
                let scale = expected.norm().max(1.0);
                max_gemm_error =
                    max_gemm_error.max((actual[row * ldc + column] - expected).norm() / scale);
            }
        }
        cases += 1;
    }

    // BLAS defines GEMV with zero columns as the beta epilogue, not a
    // no-op: y <- beta*y. Keep this explicit edge case in the release QA
    // contract because vendor kernels often reject a zero launch.
    let zero_column_initial = [Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)];
    let mut zero_column_a = pool
        .try_alloc::<Complex64>(0)
        .map_err(|error| error.to_string())?;
    let mut zero_column_x = pool
        .try_alloc::<Complex64>(0)
        .map_err(|error| error.to_string())?;
    let mut zero_column_y = pool
        .try_alloc::<Complex64>(zero_column_initial.len())
        .map_err(|error| error.to_string())?;
    backend.copy_h2d(&[][..], zero_column_a.raw_mut());
    backend.copy_h2d(&[][..], zero_column_x.raw_mut());
    backend.copy_h2d(&zero_column_initial, zero_column_y.raw_mut());
    let zero_column_beta = Complex64::new(0.5, 2.0);
    backend
        .try_gemv(
            Transpose::NoTrans,
            zero_column_initial.len(),
            0,
            Complex64::new(4.0, -1.0),
            zero_column_a.raw(),
            0,
            zero_column_x.raw(),
            zero_column_beta,
            zero_column_y.raw_mut(),
        )
        .map_err(|error| format!("zero-column GEMV rejected: {error}"))?;
    let mut zero_column_actual = [Complex64::default(); 2];
    backend.copy_d2h(zero_column_y.raw(), &mut zero_column_actual);
    for (actual, initial) in zero_column_actual.iter().zip(zero_column_initial) {
        let expected = zero_column_beta * initial;
        if (*actual - expected).norm() > TOLERANCE * expected.norm().max(1.0) {
            return Err(format!(
                "zero-column GEMV beta epilogue mismatch: actual={actual}, expected={expected}"
            ));
        }
    }
    cases += 1;

    if max_gemv_error > TOLERANCE || max_gemm_error > TOLERANCE {
        return Err(format!(
            "BLAS reference error exceeded tolerance: GEMV={max_gemv_error:.3e}, GEMM={max_gemm_error:.3e}, tolerance={TOLERANCE:.3e}"
        ));
    }
    Ok((max_gemv_error, max_gemm_error, cases))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((max_gemv_error, max_gemm_error, cases)) => {
            let result = QaResult::passed("compute.blas-contracts.v1")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .input_checksum(format!("seeded-blas-matrix:{SEED:x}:25-cases"))
                .reference("independent row-major GEMV/GEMM accumulation")
                .reference_checksum("analytic/deterministic fixture")
                .dimension(
                    "rectangular 1-11 by 1-10, padded leading dimensions, and zero-column GEMV",
                )
                .formulation("CPU dense BLAS against independent scalar complex reference")
                .tolerance("3e-12")
                .seed(format!("seed={SEED}"))
                .metric("max_gemv_relative_error", max_gemv_error)
                .metric("max_gemm_relative_error", max_gemm_error)
                .metric("case_count", cases as f64)
                .maximum("max_gemv_relative_error", TOLERANCE)
                .maximum("max_gemm_relative_error", TOLERANCE)
                .elapsed_ms(started.elapsed().as_millis() as u64)
                .timing_ms("total", started.elapsed().as_millis() as u64);
            match result.to_json_line() {
                Ok(json) => println!("QA_RESULT: {json}"),
                Err(error) => {
                    eprintln!("QA result serialization failed: {error}");
                    std::process::exit(1);
                }
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed("compute.blas-contracts.v1", error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .input_checksum(format!("seeded-blas-matrix:{SEED:x}:25-cases"))
                .reference("independent row-major GEMV/GEMM accumulation")
                .reference_checksum("analytic/deterministic fixture")
                .dimension(
                    "rectangular 1-11 by 1-10, padded leading dimensions, and zero-column GEMV",
                )
                .formulation("CPU dense BLAS against independent scalar complex reference")
                .tolerance("3e-12")
                .seed(format!("seed={SEED}"))
                .elapsed_ms(started.elapsed().as_millis() as u64)
                .timing_ms("total", started.elapsed().as_millis() as u64)
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("COMPUTE BLAS QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
