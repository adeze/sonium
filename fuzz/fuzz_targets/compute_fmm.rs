#![no_main]

use libfuzzer_sys::fuzz_target;
use num_complex::Complex64;
use sonium_compute::fmm_cpu::{
    try_indexed_dense_matvec_accumulate_c64, try_indexed_dense_matvec_c64,
};

fn byte(data: &[u8], index: usize) -> u8 {
    data.get(index).copied().unwrap_or_default()
}

fn scalar(data: &[u8], index: usize) -> Complex64 {
    let real = f64::from(byte(data, index)) / 32.0 - 4.0;
    let imag = f64::from(byte(data, index + 1)) / 32.0 - 4.0;
    Complex64::new(real, imag)
}

fuzz_target!(|data: &[u8]| {
    // Keep every input-controlled allocation bounded; malformed shapes are
    // expected to be rejected by the checked helper rather than panic.
    let rows = usize::from(byte(data, 0) % 8) + 1;
    let cols = usize::from(byte(data, 1) % 8) + 1;
    let leading_dimension = cols + usize::from(byte(data, 2) % 4);
    let source_len = usize::from(byte(data, 3) % 16) + 1;
    let matrix_len = rows.saturating_mul(leading_dimension);
    let matrix: Vec<_> = (0..matrix_len)
        .map(|index| scalar(data, 4 + index * 2))
        .collect();
    let source: Vec<_> = (0..source_len)
        .map(|index| scalar(data, 4 + matrix_len * 2 + index * 2))
        .collect();
    let src_indices: Vec<_> = (0..cols)
        .map(|index| usize::from(byte(data, 5 + index)) % source_len)
        .collect();
    let dst_indices: Vec<_> = (0..rows)
        .map(|index| usize::from(byte(data, 5 + cols + index)) % source_len)
        .collect();

    let mut gathered = vec![Complex64::default(); rows];
    let _ = try_indexed_dense_matvec_c64(
        &matrix,
        rows,
        cols,
        leading_dimension,
        &source,
        &src_indices,
        &mut gathered,
    );
    let mut scattered = vec![Complex64::default(); source_len];
    let _ = try_indexed_dense_matvec_accumulate_c64(
        &matrix,
        rows,
        cols,
        leading_dimension,
        &source,
        &src_indices,
        &mut scattered,
        &dst_indices,
    );
});
