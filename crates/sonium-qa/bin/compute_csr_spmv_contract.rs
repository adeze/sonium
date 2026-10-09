//! CPU-only checked DeviceCsr SpMV contract.

use rand::{Rng, SeedableRng, rngs::StdRng};
use sonium_compute::{
    Backend, BackendError, CpuBackend, CsrRowBucketStats, DeviceCsr, DeviceOperator,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.csr-spmv-contract.v1";
const MAX_ERROR: f64 = 1.0e-12;

struct Metrics {
    error: f64,
    colmajor_error: f64,
    random_cases: usize,
    random_max_error: f64,
    rejection: bool,
    colmajor_rejection: bool,
    malformed_row_bucket_rejection: bool,
    prebaked_descriptor_rejection: bool,
    prebaked_no_mutation: bool,
    no_mutation: bool,
}

fn max_abs(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn randomized_rectangular_contract(backend: &CpuBackend) -> Result<(usize, f64), String> {
    let mut rng = StdRng::seed_from_u64(0xC5A5_0001);
    let mut maximum_error = 0.0_f64;

    for case in 0..64 {
        let rows = 1 + (case * 5 % 8);
        let cols = 1 + (case * 7 % 9);
        let rhs = 1 + (case * 3 % 4);
        let mut row_ptr = Vec::with_capacity(rows + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0_i32);
        for row in 0..rows {
            let mut columns: Vec<usize> =
                (0..cols).filter(|_| rng.random_range(0..4) != 0).collect();
            if case % 3 != 0 && columns.is_empty() {
                columns.push((case + row) % cols);
            }
            columns.sort_unstable();
            columns.dedup();
            for column in columns {
                col_idx.push(column as i32);
                values.push(rng.random_range(-2.0..=2.0));
            }
            row_ptr.push(col_idx.len() as i32);
        }

        let csr = DeviceCsr::try_from_host(backend, rows, cols, &row_ptr, &col_idx, &values)
            .map_err(|error| format!("random CSR case {case} upload failed: {error}"))?;
        let x_host: Vec<f64> = (0..cols * rhs)
            .map(|_| rng.random_range(-1.5..=1.5))
            .collect();
        let mut expected = vec![0.0_f64; rows * rhs];
        for column_rhs in 0..rhs {
            for row in 0..rows {
                let start = row_ptr[row] as usize;
                let end = row_ptr[row + 1] as usize;
                expected[column_rhs * rows + row] = (start..end)
                    .map(|entry| {
                        values[entry] * x_host[column_rhs * cols + col_idx[entry] as usize]
                    })
                    .sum();
            }
        }

        let mut x = backend
            .raw_alloc::<f64>(x_host.len())
            .map_err(|error| format!("random CSR case {case} input allocation failed: {error}"))?;
        let mut y = backend
            .raw_alloc::<f64>(expected.len())
            .map_err(|error| format!("random CSR case {case} output allocation failed: {error}"))?;
        backend.copy_h2d(&x_host, &mut x);
        backend.copy_h2d(&vec![91.0; expected.len()], &mut y);
        DeviceOperator::try_apply_colmajor(&csr, backend, rhs, &x, &mut y)
            .map_err(|error| format!("random CSR case {case} dispatch failed: {error}"))?;
        let mut actual = vec![0.0_f64; expected.len()];
        backend.copy_d2h(&y, &mut actual);
        let error = max_abs(&actual, &expected);
        maximum_error = maximum_error.max(error);
        if error > 2.0e-12 {
            return Err(format!(
                "random CSR case {case} exceeded tolerance: error={error}"
            ));
        }
    }

    Ok((64, maximum_error))
}

fn run() -> Result<Metrics, String> {
    let backend = CpuBackend::new();
    let row_ptr = [0_i32, 2, 3, 5];
    let col_idx = [0_i32, 2, 1, 0, 2];
    let values = [2.0_f64, -1.0, 4.0, 0.5, 3.0];
    let csr = DeviceCsr::try_from_host(&backend, 3, 3, &row_ptr, &col_idx, &values)
        .map_err(|error| format!("CSR upload failed: {error}"))?;

    let mut x = backend
        .raw_alloc::<f64>(3)
        .map_err(|error| format!("input allocation failed: {error}"))?;
    let mut y = backend
        .raw_alloc::<f64>(3)
        .map_err(|error| format!("output allocation failed: {error}"))?;
    backend.copy_h2d(&[1.0, 2.0, 3.0], &mut x);
    backend.copy_h2d(&[91.0, 92.0, 93.0], &mut y);
    DeviceOperator::try_apply(&csr, &backend, &x, &mut y)
        .map_err(|error| format!("checked CSR SpMV failed: {error}"))?;

    let mut actual = [0.0; 3];
    backend.copy_d2h(&y, &mut actual);
    let error = max_abs(&actual, &[-1.0, 8.0, 9.5]);

    let mut before = [0.0; 3];
    backend.copy_d2h(&y, &mut before);
    let short_x = backend
        .raw_alloc::<f64>(2)
        .map_err(|error| format!("short input allocation failed: {error}"))?;
    let rejection = matches!(
        DeviceOperator::try_apply(&csr, &backend, &short_x, &mut y),
        Err(BackendError::InvalidArgument { .. })
    );
    let mut after = [0.0; 3];
    backend.copy_d2h(&y, &mut after);

    let mut x_many = backend
        .raw_alloc::<f64>(6)
        .map_err(|error| format!("multi-RHS input allocation failed: {error}"))?;
    let mut y_many = backend
        .raw_alloc::<f64>(6)
        .map_err(|error| format!("multi-RHS output allocation failed: {error}"))?;
    backend.copy_h2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &mut x_many);
    DeviceOperator::try_apply_colmajor(&csr, &backend, 2, &x_many, &mut y_many)
        .map_err(|error| format!("checked CSR multi-RHS SpMV failed: {error}"))?;
    let mut actual_many = [0.0; 6];
    backend.copy_d2h(&y_many, &mut actual_many);
    let colmajor_error = max_abs(&actual_many, &[-1.0, 8.0, 9.5, 2.0, 20.0, 20.0]);

    let short_many = backend
        .raw_alloc::<f64>(5)
        .map_err(|error| format!("short multi-RHS allocation failed: {error}"))?;
    let colmajor_rejection = matches!(
        DeviceOperator::try_apply_colmajor(&csr, &backend, 2, &short_many, &mut y_many),
        Err(BackendError::InvalidArgument { .. })
    );

    let malformed_row_bucket_rejection = matches!(
        CsrRowBucketStats::try_from_row_ptr(&[0, 1, 0]),
        Err(BackendError::InvalidArgument { .. })
    );

    let mut malformed_row_ptr = backend
        .raw_alloc::<i32>(3)
        .map_err(|error| format!("malformed row-pointer allocation failed: {error}"))?;
    let mut malformed_col_idx = backend
        .raw_alloc::<i32>(1)
        .map_err(|error| format!("malformed column allocation failed: {error}"))?;
    let mut malformed_values = backend
        .raw_alloc::<f64>(1)
        .map_err(|error| format!("malformed value allocation failed: {error}"))?;
    let mut direct_x = backend
        .raw_alloc::<f64>(2)
        .map_err(|error| format!("direct input allocation failed: {error}"))?;
    let mut direct_y = backend
        .raw_alloc::<f64>(2)
        .map_err(|error| format!("direct output allocation failed: {error}"))?;
    backend.copy_h2d(&[0, 1, 0], &mut malformed_row_ptr);
    backend.copy_h2d(&[0], &mut malformed_col_idx);
    backend.copy_h2d(&[2.0], &mut malformed_values);
    backend.copy_h2d(&[3.0, 5.0], &mut direct_x);
    backend.copy_h2d(&[7.0, 11.0], &mut direct_y);
    let malformed_prebaked = backend.prebake_csr(
        2,
        2,
        &malformed_row_ptr,
        &malformed_col_idx,
        &malformed_values,
    );
    let prebaked_descriptor_rejection = matches!(
        backend.try_spmv_prebaked(&malformed_prebaked, &direct_x, &mut direct_y),
        Err(BackendError::InvalidArgument { .. })
    );
    let mut direct_y_host = [0.0; 2];
    backend.copy_d2h(&direct_y, &mut direct_y_host);

    let (random_cases, random_max_error) = randomized_rectangular_contract(&backend)?;

    Ok(Metrics {
        error,
        colmajor_error,
        random_cases,
        random_max_error,
        rejection,
        colmajor_rejection,
        malformed_row_bucket_rejection,
        prebaked_descriptor_rejection,
        prebaked_no_mutation: direct_y_host == [7.0, 11.0],
        no_mutation: after == before,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("spmv_error", metrics.error)
                .metric("colmajor_error", metrics.colmajor_error)
                .metric("random_case_count", metrics.random_cases as f64)
                .metric("random_max_error", metrics.random_max_error)
                .metric("rejection", metrics.rejection as u8 as f64)
                .metric(
                    "colmajor_rejection",
                    metrics.colmajor_rejection as u8 as f64,
                )
                .metric(
                    "malformed_row_bucket_rejection",
                    metrics.malformed_row_bucket_rejection as u8 as f64,
                )
                .metric(
                    "prebaked_descriptor_rejection",
                    metrics.prebaked_descriptor_rejection as u8 as f64,
                )
                .metric(
                    "prebaked_no_mutation",
                    metrics.prebaked_no_mutation as u8 as f64,
                )
                .metric("no_mutation", metrics.no_mutation as u8 as f64)
                .maximum("spmv_error", MAX_ERROR)
                .maximum("colmajor_error", MAX_ERROR)
                .maximum("random_max_error", MAX_ERROR)
                .minimum("random_case_count", 64.0)
                .minimum("rejection", 1.0)
                .minimum("colmajor_rejection", 1.0)
                .minimum("malformed_row_bucket_rejection", 1.0)
                .minimum("prebaked_descriptor_rejection", 1.0)
                .minimum("prebaked_no_mutation", 1.0)
                .minimum("no_mutation", 1.0)
                .reference("independent scalar CSR row accumulation")
                .reference_checksum("analytic/deterministic-fixture:compute-csr-spmv-contract-v4")
                .dimension("f64 CSR fixed 3x3 plus direct malformed pre-baked descriptor and 64 seeded rectangular multi-RHS cases")
                .seed("compute-csr-spmv-contract:3x3+prebaked+rectangular-seed-v4");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("CSR SpMV QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("CSR SpMV QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
