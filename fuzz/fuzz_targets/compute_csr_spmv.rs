#![no_main]

use libfuzzer_sys::fuzz_target;
use sonium_compute::{Backend, BackendError, CpuBackend};

fn descriptor_is_valid(
    m: usize,
    n: usize,
    row_ptr: &[i32],
    col_idx: &[i32],
    values: &[f64],
) -> bool {
    row_ptr.len() == m.saturating_add(1)
        && col_idx.len() == values.len()
        && row_ptr.first() == Some(&0)
        && row_ptr
            .windows(2)
            .all(|window| window[0] >= 0 && window[1] >= window[0])
        && row_ptr.last().and_then(|&end| usize::try_from(end).ok()) == Some(values.len())
        && col_idx
            .iter()
            .all(|&column| column >= 0 && usize::try_from(column).is_ok_and(|column| column < n))
}

fuzz_target!(|data: &[u8]| {
    let m = usize::from(data.first().copied().unwrap_or_default() % 8);
    let n = usize::from(data.get(1).copied().unwrap_or_default() % 8);
    let nnz = usize::from(data.get(2).copied().unwrap_or_default() % 17);

    let mut row_ptr = vec![0_i32; m.saturating_add(1)];
    let mut col_idx = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);
    let mut offset = 3;
    for entry in &mut row_ptr {
        let byte = data.get(offset).copied().unwrap_or_default();
        *entry = i32::from(byte % 20).saturating_sub(2);
        offset = offset.saturating_add(1);
    }
    for _ in 0..nnz {
        let column = data.get(offset).copied().unwrap_or_default();
        let value = data
            .get(offset.saturating_add(1))
            .copied()
            .unwrap_or_default();
        col_idx.push(i32::from(column % 12).saturating_sub(2));
        values.push((f64::from(value) - 128.0) / 32.0);
        offset = offset.saturating_add(2);
    }

    let backend = CpuBackend::new();
    let Ok(mut row_buffer) = backend.raw_alloc::<i32>(row_ptr.len()) else {
        return;
    };
    let Ok(mut col_buffer) = backend.raw_alloc::<i32>(col_idx.len()) else {
        return;
    };
    let Ok(mut value_buffer) = backend.raw_alloc::<f64>(values.len()) else {
        return;
    };
    let Ok(mut x) = backend.raw_alloc::<f64>(n) else {
        return;
    };
    let Ok(mut y) = backend.raw_alloc::<f64>(m) else {
        return;
    };
    backend.copy_h2d(&row_ptr, &mut row_buffer);
    backend.copy_h2d(&col_idx, &mut col_buffer);
    backend.copy_h2d(&values, &mut value_buffer);
    backend.copy_h2d(&vec![1.0; n], &mut x);
    backend.copy_h2d(&vec![7.0; m], &mut y);

    // Use the legacy descriptor constructor to exercise the checked
    // steady-state boundary on both valid and malformed descriptors.
    let prebaked = backend.prebake_csr(m, n, &row_buffer, &col_buffer, &value_buffer);
    let result = backend.try_spmv_prebaked(&prebaked, &x, &mut y);
    if !descriptor_is_valid(m, n, &row_ptr, &col_idx, &values) {
        assert!(matches!(result, Err(BackendError::InvalidArgument { .. })));
    }
});
