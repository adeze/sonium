#![no_main]

use libfuzzer_sys::fuzz_target;
use sonium_solvers::{
    CsrMatrix, DiagonalPreconditioner, IluPreconditioner, IlutPreconditioner, Preconditioner,
};

fuzz_target!(|bytes: &[u8]| {
    let dimension = usize::from(bytes.first().copied().unwrap_or_default() % 8);
    if dimension == 0 {
        return;
    }

    // Keep the generated matrix small and structurally sparse.  Arbitrary
    // row pointers/columns exercise checked construction; valid matrices then
    // exercise setup and application of the three common preconditioners.
    let nnz = dimension.saturating_mul(dimension).min(32);
    let row_ptrs: Vec<_> = (0..=dimension)
        .map(|row| (row.saturating_mul(nnz) / dimension).min(nnz))
        .collect();
    let col_indices: Vec<_> = (0..nnz)
        .map(|index| {
            usize::from(bytes.get(index + 1).copied().unwrap_or_default())
                % dimension.saturating_add(2)
        })
        .collect();
    let values: Vec<_> = (0..nnz)
        .map(|index| {
            let byte = bytes.get(index + 1 + nnz).copied().unwrap_or_default();
            if byte == u8::MAX {
                f64::NAN
            } else {
                f64::from(byte) / 32.0 + 0.25
            }
        })
        .collect();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Ok(matrix) =
            CsrMatrix::try_from_raw_parts(dimension, dimension, row_ptrs, col_indices, values)
        else {
            return;
        };
        let rhs = vec![1.0; dimension];
        if let Ok(preconditioner) = DiagonalPreconditioner::try_from_csr(&matrix) {
            let _ = preconditioner.try_apply(&rhs);
        }
        if let Ok(preconditioner) = IluPreconditioner::try_from_csr(&matrix) {
            let _ = preconditioner.try_apply(&rhs);
        }
        if let Ok(preconditioner) = IlutPreconditioner::try_from_csr(&matrix) {
            let _ = preconditioner.try_apply(&rhs);
        }
    }));
    assert!(
        result.is_ok(),
        "checked preconditioner boundary panicked on fuzz input"
    );
});
