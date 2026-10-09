#![no_main]

use libfuzzer_sys::fuzz_target;
use sonium_compute::{Backend, CpuBackend, DeviceBlock, DeviceCsr, DeviceOperator, HostCsrRef};

// Exercise checked CSR construction with arbitrary row markers, columns,
// and payload lengths. `try_new` is the descriptor boundary used before
// sparse launch/prebake APIs receive host data.
fuzz_target!(|data: &[u8]| {
    let rows = usize::from(data.first().copied().unwrap_or_default() % 16);
    let cols = usize::from(data.get(1).copied().unwrap_or_default() % 16);
    let mut row_ptr = Vec::with_capacity(rows + 1);
    let mut columns = Vec::new();
    let mut values = Vec::new();
    for index in 0..=rows {
        row_ptr.push(usize::from_le_bytes(
            data.get(2 + index * 8..10 + index * 8)
                .and_then(|bytes| bytes.try_into().ok())
                .unwrap_or([0; 8]),
        ));
    }
    for chunk in data
        .get(2 + (rows + 1) * 8..)
        .unwrap_or_default()
        .chunks_exact(8)
        .take(64)
    {
        columns.push(u32::from_le_bytes(chunk[..4].try_into().unwrap()) as usize);
        values.push(f32::from_le_bytes(chunk[4..].try_into().unwrap()));
    }
    let Ok(_host) = HostCsrRef::try_new(rows, cols, &row_ptr, &columns, &values) else {
        return;
    };
    let row_ptr_i32 = row_ptr
        .iter()
        .map(|&value| i32::try_from(value).unwrap_or(i32::MAX))
        .collect::<Vec<_>>();
    let col_idx_i32 = columns
        .iter()
        .map(|&value| i32::try_from(value).unwrap_or(i32::MAX))
        .collect::<Vec<_>>();
    let values_f64 = values
        .iter()
        .map(|&value| f64::from(value))
        .collect::<Vec<_>>();
    let backend = CpuBackend::new();
    let Ok(matrix) = DeviceCsr::try_from_host(
        &backend,
        rows,
        cols,
        &row_ptr_i32,
        &col_idx_i32,
        &values_f64,
    ) else {
        return;
    };
    let rhs = usize::from(data.get(0).copied().unwrap_or_default() % 4);
    let x = backend.raw_alloc::<f64>(cols.saturating_mul(rhs)).unwrap();
    let mut y = backend.raw_alloc::<f64>(rows.saturating_mul(rhs)).unwrap();
    let _ = <DeviceCsr<CpuBackend, f64> as DeviceOperator<CpuBackend, f64>>::try_apply_colmajor(
        &matrix, &backend, rhs, &x, &mut y,
    );
    let _ = <DeviceCsr<CpuBackend, f64> as DeviceBlock<CpuBackend, f64>>::try_apply_colmajor(
        &matrix, &backend, rhs, &x, &mut y,
    );
});
