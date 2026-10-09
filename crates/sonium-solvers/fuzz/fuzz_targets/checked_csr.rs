#![no_main]

use libfuzzer_sys::fuzz_target;
use sonium_solvers::sparse::CsrMatrix;

// Exercise the checked sparse-storage boundary with arbitrary, bounded
// metadata. Invalid structures are expected to become typed errors; none
// may panic.
fuzz_target!(|bytes: &[u8]| {
    let rows = usize::from(bytes.first().copied().unwrap_or_default() % 16);
    let cols = usize::from(bytes.get(1).copied().unwrap_or_default() % 16);
    let nnz = usize::from(bytes.get(2).copied().unwrap_or_default() % 32);

    let mut row_ptrs = Vec::with_capacity(rows.saturating_add(1));
    for index in 0..=rows {
        let value = usize::from(bytes.get(3 + index).copied().unwrap_or_default()) % (nnz + 1);
        row_ptrs.push(value);
    }
    let col_indices = (0..nnz)
        .map(|index| {
            usize::from(bytes.get(3 + rows + index).copied().unwrap_or_default())
                % (cols.saturating_add(3).max(1))
        })
        .collect();
    let values = (0..nnz)
        .map(|index| {
            let byte = bytes
                .get(3 + rows + nnz + index)
                .copied()
                .unwrap_or_default();
            if byte == u8::MAX {
                f64::NAN
            } else {
                f64::from(byte) / 17.0 - 7.0
            }
        })
        .collect();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        CsrMatrix::<f64>::try_from_raw_parts(rows, cols, row_ptrs, col_indices, values).map(
            |matrix| {
                let _ = matrix.validate_structure();
                let _ = matrix.validate_canonical_structure();
            },
        )
    }));
    assert!(
        result.is_ok(),
        "checked CSR boundary panicked on fuzz input"
    );
});
