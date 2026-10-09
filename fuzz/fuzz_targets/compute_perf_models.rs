#![no_main]

use libfuzzer_sys::fuzz_target;
use sonium_compute::CsrRowBucketStats;
use sonium_compute::perf::{
    axpy_model, bem_regular_pair_c32_model, cast_model, copy_d2d_model, copy_h2d_model,
    csr_spmv_model, dot_model, gather4_weighted_model, gemm_model, gemv_model, nrm2_model,
    scale_model,
};

// Roofline models are pure arithmetic, but their dimensions often originate
// in benchmark/QA input. Feed both ordinary and extreme shapes through every
// public model and require that the process never panics or wraps.
fuzz_target!(|data: &[u8]| {
    let byte = |index: usize| data.get(index).copied().unwrap_or_default();
    let shape = |index: usize| {
        if byte(index) == u8::MAX {
            usize::MAX
        } else {
            usize::from(byte(index) % 32)
        }
    };
    let n = shape(0);
    let m = shape(1);
    let k = shape(2);
    let rows = shape(3);
    let nnz = shape(4);

    let _ = copy_h2d_model::<f64>(n);
    let _ = copy_d2d_model::<f32>(n);
    let _ = scale_model::<num_complex::Complex32>(n);
    let _ = dot_model::<num_complex::Complex64>(n);
    let _ = axpy_model::<num_complex::Complex32>(n);
    let _ = nrm2_model::<num_complex::Complex64>(n);
    let _ = gemv_model::<f64>(m, n);
    let _ = gemm_model::<num_complex::Complex64>(m, n, k);
    let _ = cast_model::<f32, num_complex::Complex64>(n);
    let _ = gather4_weighted_model::<num_complex::Complex32>(rows);
    let _ = csr_spmv_model::<f64>(CsrRowBucketStats {
        rows,
        nnz,
        empty_rows: 0,
        short_rows: rows,
        simd_rows: 0,
        cta_rows: 0,
        max_nnz_per_row: nnz,
    });
    let _ = bem_regular_pair_c32_model(n);
});
