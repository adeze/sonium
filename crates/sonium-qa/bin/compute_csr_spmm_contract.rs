//! CPU-only checked CSR SpMM contract.
//!
//! The sparse multi-RHS path is intentionally exercised with padded
//! column-major leading dimensions.  The same fixture also checks that shape
//! failures are typed and transactional before a backend kernel can write the
//! destination.

use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_compute::{Backend, BackendError, CpuBackend, DeviceScalar};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.csr-spmm-contract.v1";
const MAX_ERROR: f64 = 2.0e-6;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    scalar_cases: usize,
    valid_cases: usize,
    rejection_cases: usize,
    maximum_error: f64,
    padding_preserved: bool,
    destination_preserved: bool,
}

fn relative_error<T: DeviceScalar>(actual: T, expected: T) -> f64 {
    let diff = ToPrimitive::to_f64(&(actual - expected).magnitude()).unwrap_or(f64::INFINITY);
    let expected = ToPrimitive::to_f64(&expected.magnitude()).unwrap_or(f64::INFINITY);
    diff / expected.max(1.0)
}

fn same_scalar<T: DeviceScalar>(actual: T, expected: T) -> bool {
    relative_error(actual, expected) == 0.0
}

fn check_scalar<T: DeviceScalar>(
    backend: &CpuBackend,
    label: &str,
) -> Result<(f64, bool, bool), String> {
    let real = |value: f64| {
        T::Real::from_f64(value)
            .map(T::from_real)
            .ok_or_else(|| format!("{label}: fixture value {value} is not representable"))
    };

    let row_ptr = [0_i32, 2, 3, 5];
    let col_idx = [0_i32, 2, 1, 0, 2];
    let values = [2.0_f64, -1.0, 4.0, 0.5, 3.0];
    let mut row = backend
        .raw_alloc::<i32>(row_ptr.len())
        .map_err(|error| format!("{label}: row allocation failed: {error}"))?;
    let mut col = backend
        .raw_alloc::<i32>(col_idx.len())
        .map_err(|error| format!("{label}: column allocation failed: {error}"))?;
    let mut matrix = backend
        .raw_alloc::<T>(values.len())
        .map_err(|error| format!("{label}: value allocation failed: {error}"))?;
    backend.copy_h2d(&row_ptr, &mut row);
    backend.copy_h2d(&col_idx, &mut col);
    let matrix_values = values
        .iter()
        .copied()
        .map(real)
        .collect::<Result<Vec<_>, _>>()?;
    backend.copy_h2d(&matrix_values, &mut matrix);
    let prebaked = backend.prebake_csr(3, 3, &row, &col, &matrix);

    // Two RHS columns use ldx=4 and ldy=5, leaving one padded element in each
    // input/output column.  The logical span is therefore eight elements,
    // rather than ldx*rhs or ldy*rhs.
    let ldx = 4;
    let ldy = 5;
    let rhs = 2;
    let x_real = [1.0, 2.0, 3.0, 99.0, -2.0, 0.5, 4.0, 77.0];
    let mut x = backend
        .raw_alloc::<T>(x_real.len())
        .map_err(|error| format!("{label}: input allocation failed: {error}"))?;
    let x_values = x_real
        .iter()
        .copied()
        .map(real)
        .collect::<Result<Vec<_>, _>>()?;
    backend.copy_h2d(&x_values, &mut x);

    let sentinel = real(123.0)?;
    let mut y = backend
        .raw_alloc::<T>(ldy * rhs)
        .map_err(|error| format!("{label}: output allocation failed: {error}"))?;
    backend.copy_h2d(&vec![sentinel; ldy * rhs], &mut y);
    backend
        .try_spmm_prebaked(&prebaked, rhs, &x, ldx, &mut y, ldy)
        .map_err(|error| format!("{label}: valid SpMM rejected: {error}"))?;

    let mut actual = vec![T::zero(); ldy * rhs];
    backend.copy_d2h(&y, &mut actual);
    let expected_columns = [[-1.0, 8.0, 9.5], [-8.0, 2.0, 11.0]];
    let mut maximum_error: f64 = 0.0;
    for (column, expected) in expected_columns.iter().enumerate() {
        for (row_index, &expected_value) in expected.iter().enumerate() {
            maximum_error = maximum_error.max(relative_error(
                actual[column * ldy + row_index],
                real(expected_value)?,
            ));
        }
    }
    let padding_preserved = same_scalar(actual[3], sentinel) && same_scalar(actual[8], sentinel);

    // A short destination must be rejected before any write.  Keep a separate
    // valid-sized destination so the no-mutation property is observable.
    let mut rejected_y = backend
        .raw_alloc::<T>(ldy * rhs)
        .map_err(|error| format!("{label}: rejection output allocation failed: {error}"))?;
    backend.copy_h2d(&vec![sentinel; ldy * rhs], &mut rejected_y);
    let short_x = backend
        .raw_alloc::<T>(6)
        .map_err(|error| format!("{label}: short input allocation failed: {error}"))?;
    let short_rejection = matches!(
        backend.try_spmm_prebaked(&prebaked, rhs, &short_x, ldx, &mut rejected_y, ldy),
        Err(BackendError::InvalidArgument { .. })
    );

    // A stride smaller than the logical matrix extent is also rejected before
    // dispatch, while the destination remains unchanged.
    let bad_stride_rejection = matches!(
        backend.try_spmm_prebaked(&prebaked, rhs, &x, 2, &mut rejected_y, ldy),
        Err(BackendError::InvalidArgument { .. })
    );
    let mut rejected_host = vec![T::zero(); ldy * rhs];
    backend.copy_d2h(&rejected_y, &mut rejected_host);
    let destination_preserved = rejected_host
        .iter()
        .copied()
        .all(|value| same_scalar(value, sentinel));

    Ok((
        maximum_error,
        padding_preserved,
        short_rejection && bad_stride_rejection && destination_preserved,
    ))
}

fn run() -> Result<Metrics, String> {
    let backend = CpuBackend::new();
    let cases = [
        check_scalar::<f32>(&backend, "f32"),
        check_scalar::<f64>(&backend, "f64"),
        check_scalar::<Complex32>(&backend, "Complex32"),
        check_scalar::<Complex64>(&backend, "Complex64"),
    ];
    let mut maximum_error: f64 = 0.0;
    let mut padding_preserved = true;
    let mut destination_preserved = true;
    for result in cases {
        let (error, padding, rejected) = result?;
        maximum_error = maximum_error.max(error);
        padding_preserved &= padding;
        destination_preserved &= rejected;
    }
    Ok(Metrics {
        scalar_cases: 4,
        valid_cases: 4,
        rejection_cases: 8,
        maximum_error,
        padding_preserved,
        destination_preserved,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("scalar_cases", metrics.scalar_cases as f64)
                .metric("valid_cases", metrics.valid_cases as f64)
                .metric("rejection_cases", metrics.rejection_cases as f64)
                .metric("maximum_relative_error", metrics.maximum_error)
                .metric("padding_preserved", metrics.padding_preserved as u8 as f64)
                .metric(
                    "destination_preserved",
                    metrics.destination_preserved as u8 as f64,
                )
                .maximum("maximum_relative_error", MAX_ERROR)
                .minimum("scalar_cases", 4.0)
                .minimum("valid_cases", 4.0)
                .minimum("rejection_cases", 8.0)
                .minimum("padding_preserved", 1.0)
                .minimum("destination_preserved", 1.0)
                .reference("independent scalar CSR multi-RHS column-major accumulation")
                .reference_checksum("analytic/deterministic-fixture:compute-csr-spmm-contract-v1")
                .dimension("f32/f64/Complex32/Complex64; 3x3 CSR; two padded RHS columns")
                .seed("compute-csr-spmm-contract:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("CSR SpMM QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("CSR SpMM QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
