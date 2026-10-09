#![no_main]

use libfuzzer_sys::fuzz_target;
use sonium_solvers::CsrMatrix;

// Checked CSR construction is the untrusted-input boundary used before
// factorization and Krylov routines. This target varies dimensions, markers,
// columns, finite/non-finite values, and malformed payload lengths.
fuzz_target!(|data: &[u8]| {
    let rows = usize::from(data.first().copied().unwrap_or_default() % 16);
    let cols = usize::from(data.get(1).copied().unwrap_or_default() % 16);
    let mut row_ptrs = Vec::with_capacity(rows + 1);
    for index in 0..=rows {
        row_ptrs.push(u16::from_le_bytes(
            data.get(2 + index * 2..4 + index * 2)
                .and_then(|bytes| bytes.try_into().ok())
                .unwrap_or([0; 2]),
        ) as usize);
    }
    let mut columns = Vec::new();
    let mut values = Vec::new();
    for chunk in data
        .get(2 + (rows + 1) * 2..)
        .unwrap_or_default()
        .chunks_exact(8)
        .take(64)
    {
        columns.push(u32::from_le_bytes(chunk[..4].try_into().unwrap()) as usize);
        values.push(f32::from_le_bytes(chunk[4..].try_into().unwrap()));
    }
    let _ = CsrMatrix::<f32>::try_from_raw_parts(rows, cols, row_ptrs, columns, values);
});
