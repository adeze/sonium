//! Checked CSR prebake and column-major application contract.
//!
//! This is a small CPU-only QA case for the setup seam shared by device-owned,
//! pooled, and adjoint CSR operators.  It deliberately exercises both the
//! successful SpMM path and malformed device metadata, where checked APIs must
//! return a typed error rather than let a legacy prebake panic escape.

use std::panic::{AssertUnwindSafe, catch_unwind};

use num_complex::Complex64;
use sonium_compute::{
    AdjointDeviceCsr, Backend, CpuBackend, DeviceAdjointOperator, DeviceBlock, DeviceCsr,
    DeviceOperator, HostCsrRef, MemoryPool, MemoryPoolConfig, PrebakedSpmv,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.csr-prebake-contract.v1";
const MAX_ERROR: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    device_error: f64,
    pooled_error: f64,
    adjoint_block_error: f64,
    malformed_failures: usize,
}

fn max_abs(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<Metrics, String> {
    let row_ptr = [0, 2, 4];
    let col_idx = [0, 1, 0, 1];
    let values = [2.0_f64, 1.0, -1.0, 3.0];
    let columns = [4.0, 5.0, -1.0, 2.0];
    let expected = [13.0, 11.0, 0.0, 7.0];

    let backend = CpuBackend::new();
    let matrix = DeviceCsr::try_from_host(&backend, 2, 2, &row_ptr, &col_idx, &values)
        .map_err(|error| format!("device CSR upload failed: {error}"))?;
    let checked = PrebakedSpmv::try_from_csr(&backend, &matrix)
        .map_err(|error| format!("checked device prebake failed: {error}"))?;
    let mut x = backend
        .raw_alloc::<f64>(columns.len())
        .map_err(|error| format!("device input allocation failed: {error}"))?;
    let mut y = backend
        .raw_alloc::<f64>(expected.len())
        .map_err(|error| format!("device output allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&columns, &mut x)
        .map_err(|error| format!("device input upload failed: {error}"))?;
    checked
        .try_apply_colmajor(&backend, 2, &x, &mut y)
        .map_err(|error| format!("checked device SpMM failed: {error}"))?;
    let mut device_output = [0.0; 4];
    backend
        .try_copy_d2h(&y, &mut device_output)
        .map_err(|error| format!("device output download failed: {error}"))?;
    let mut block_y = backend
        .raw_alloc::<f64>(expected.len())
        .map_err(|error| format!("device block output allocation failed: {error}"))?;
    <DeviceCsr<CpuBackend, f64> as DeviceBlock<CpuBackend, f64>>::try_apply_colmajor(
        &matrix,
        &backend,
        2,
        &x,
        &mut block_y,
    )
    .map_err(|error| format!("checked device block SpMM failed: {error}"))?;
    let mut block_output = [0.0; 4];
    backend
        .try_copy_d2h(&block_y, &mut block_output)
        .map_err(|error| format!("device block output download failed: {error}"))?;
    let device_error = max_abs(&device_output, &expected).max(max_abs(&block_output, &expected));

    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(4096))
        .map_err(|error| format!("memory pool creation failed: {error}"))?;
    let pooled_backend = pool.backend();
    let pooled = DeviceCsr::try_from_pool(&pool, 2, 2, &row_ptr, &col_idx, &values)
        .map_err(|error| format!("pooled CSR upload failed: {error}"))?;
    let pooled_checked = PrebakedSpmv::try_from_pooled(pooled_backend, &pooled)
        .map_err(|error| format!("checked pooled prebake failed: {error}"))?;
    let mut pooled_x = pooled_backend
        .raw_alloc::<f64>(columns.len())
        .map_err(|error| format!("pooled input allocation failed: {error}"))?;
    let mut pooled_y = pooled_backend
        .raw_alloc::<f64>(expected.len())
        .map_err(|error| format!("pooled output allocation failed: {error}"))?;
    pooled_backend
        .try_copy_h2d(&columns, &mut pooled_x)
        .map_err(|error| format!("pooled input upload failed: {error}"))?;
    pooled_checked
        .try_apply_colmajor(pooled_backend, 2, &pooled_x, &mut pooled_y)
        .map_err(|error| format!("checked pooled SpMM failed: {error}"))?;
    let mut pooled_output = [0.0; 4];
    pooled_backend
        .try_copy_d2h(&pooled_y, &mut pooled_output)
        .map_err(|error| format!("pooled output download failed: {error}"))?;
    let pooled_error = max_abs(&pooled_output, &expected);

    let adjoint_values = [
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(-0.5, 0.25),
        Complex64::new(3.0, 2.0),
    ];
    let adjoint_row_ptr = [0_usize, 2, 4];
    let adjoint_col_idx = [0_usize, 1, 0, 1];
    let adjoint_host =
        HostCsrRef::try_new(2, 2, &adjoint_row_ptr, &adjoint_col_idx, &adjoint_values)
            .map_err(|error| format!("adjoint host CSR admission failed: {error}"))?;
    let adjoint = AdjointDeviceCsr::try_from_host_csr_ref(&backend, adjoint_host)
        .map_err(|error| format!("adjoint CSR upload failed: {error}"))?;
    let adjoint_checked = adjoint
        .try_prebake(&backend)
        .map_err(|error| format!("checked adjoint prebake failed: {error}"))?;
    let adjoint_columns = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(-2.0, 0.5),
        Complex64::new(1.5, -0.25),
    ];
    let mut adjoint_x = backend
        .raw_alloc::<Complex64>(adjoint_columns.len())
        .map_err(|error| format!("adjoint input allocation failed: {error}"))?;
    let mut adjoint_block = backend
        .raw_alloc::<Complex64>(adjoint_columns.len())
        .map_err(|error| format!("adjoint block allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&adjoint_columns, &mut adjoint_x)
        .map_err(|error| format!("adjoint input upload failed: {error}"))?;
    adjoint_checked
        .try_apply_hermitian_colmajor(&backend, 2, &adjoint_x, &mut adjoint_block)
        .map_err(|error| format!("checked adjoint block application failed: {error}"))?;
    let mut adjoint_block_host = [Complex64::new(0.0, 0.0); 4];
    backend
        .try_copy_d2h(&adjoint_block, &mut adjoint_block_host)
        .map_err(|error| format!("adjoint block download failed: {error}"))?;
    let mut adjoint_block_error: f64 = 0.0;
    for column in 0..2 {
        let range = column * 2..(column + 1) * 2;
        let mut one_x = backend
            .raw_alloc::<Complex64>(2)
            .map_err(|error| format!("adjoint column allocation failed: {error}"))?;
        let mut one_y = backend
            .raw_alloc::<Complex64>(2)
            .map_err(|error| format!("adjoint column output allocation failed: {error}"))?;
        backend
            .try_copy_h2d(&adjoint_columns[range.clone()], &mut one_x)
            .map_err(|error| format!("adjoint column upload failed: {error}"))?;
        adjoint_checked
            .try_apply_hermitian(&backend, &one_x, &mut one_y)
            .map_err(|error| format!("checked adjoint column application failed: {error}"))?;
        let mut one_y_host = [Complex64::new(0.0, 0.0); 2];
        backend
            .try_copy_d2h(&one_y, &mut one_y_host)
            .map_err(|error| format!("adjoint column download failed: {error}"))?;
        for (actual, expected) in adjoint_block_host[range].iter().zip(one_y_host) {
            adjoint_block_error = adjoint_block_error.max((*actual - expected).norm());
        }
    }

    let mut malformed_row_ptr = backend
        .raw_alloc::<i32>(2)
        .map_err(|error| format!("malformed row-pointer allocation failed: {error}"))?;
    let mut malformed_col_idx = backend
        .raw_alloc::<i32>(2)
        .map_err(|error| format!("malformed column allocation failed: {error}"))?;
    let mut malformed_values = backend
        .raw_alloc::<f64>(2)
        .map_err(|error| format!("malformed values allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&[0, 2], &mut malformed_row_ptr)
        .map_err(|error| format!("malformed row-pointer upload failed: {error}"))?;
    backend
        .try_copy_h2d(&[0, 1], &mut malformed_col_idx)
        .map_err(|error| format!("malformed column upload failed: {error}"))?;
    backend
        .try_copy_h2d(&[1.0, 2.0], &mut malformed_values)
        .map_err(|error| format!("malformed values upload failed: {error}"))?;
    let malformed = DeviceCsr {
        m: 2,
        n: 2,
        nnz: 2,
        row_ptr: malformed_row_ptr,
        col_idx: malformed_col_idx,
        values: malformed_values,
    };
    let malformed_result = catch_unwind(AssertUnwindSafe(|| {
        <DeviceCsr<CpuBackend, f64> as DeviceOperator<CpuBackend, f64>>::try_apply_colmajor(
            &malformed, &backend, 2, &x, &mut y,
        )
    }));
    let malformed_failures = match malformed_result {
        Ok(Err(_)) => 1,
        Ok(Ok(())) => return Err("malformed CSR unexpectedly succeeded".into()),
        Err(_) => return Err("checked malformed CSR path panicked".into()),
    };

    if device_error > MAX_ERROR || pooled_error > MAX_ERROR || adjoint_block_error > MAX_ERROR {
        return Err(format!(
            "CSR prebake errors exceed tolerance: device={device_error:.3e}, pooled={pooled_error:.3e}, adjoint={adjoint_block_error:.3e}"
        ));
    }

    Ok(Metrics {
        device_error,
        pooled_error,
        adjoint_block_error,
        malformed_failures,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("device_error", metrics.device_error)
                .metric("pooled_error", metrics.pooled_error)
                .metric("adjoint_block_error", metrics.adjoint_block_error)
                .metric("malformed_failures", metrics.malformed_failures as f64)
                .maximum("device_error", MAX_ERROR)
                .maximum("pooled_error", MAX_ERROR)
                .maximum("adjoint_block_error", MAX_ERROR)
                .minimum("malformed_failures", 1.0)
                .reference("independent scalar CSR column application")
                .reference_checksum("analytic/deterministic-fixture:csr-prebake-contract-v1")
                .dimension("2x2 CSR; two column-major RHS; device, pooled, adjoint")
                .seed("compute-csr-prebake-contract:2x2-matrix:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("CSR prebake QA result must serialize")
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
                    .expect("CSR prebake QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
