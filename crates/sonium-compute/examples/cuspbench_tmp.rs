//! Temporary GPU baseline: our `try_spmv_prebaked` vs raw `cusparseSpMV`
//! on a 2D Laplacian. Scratch probe, not a committed test.

#![cfg(feature = "cuda")]

use cudarc::cusparse::sys as cusparse_sys;
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use sonium_compute::{Backend, CudaBackend};

/// 5-point 2D Laplacian in CSR (i32 indices), grid `g` x `g`.
fn laplacian(g: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let n = g * g;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for i in 0..g {
        for j in 0..g {
            let r = i * g + j;
            let mut nbrs = vec![r as i32];
            if i > 0 {
                nbrs.push((r - g) as i32);
            }
            if i + 1 < g {
                nbrs.push((r + g) as i32);
            }
            if j > 0 {
                nbrs.push((r - 1) as i32);
            }
            if j + 1 < g {
                nbrs.push((r + 1) as i32);
            }
            nbrs.sort_unstable();
            for &c in &nbrs {
                col_idx.push(c);
                values.push(if c as usize == r { 4.0 } else { -1.0 });
            }
            row_ptr.push(col_idx.len() as i32);
        }
    }
    let _ = n;
    (row_ptr, col_idx, values)
}

fn bench_spmv(g: usize, reps: usize) {
    let n = g * g;
    let (row_ptr, col_idx, values) = laplacian(g);
    let nnz = values.len();
    let flops = 2.0 * nnz as f64;
    // Traffic model: values(8B) + colidx(4B) streamed, rowptr + x gathered,
    // y written once.
    let bytes = (12 * nnz + 4 * n + 8 * n + 8 * n) as f64;
    let x0: Vec<f64> = (0..n).map(|i| (i as f64 * 0.013).sin()).collect();

    // ---- Our backend ----
    let cuda = CudaBackend::new_default().expect("cuda backend");
    let mut rp = cuda.raw_alloc::<i32>(row_ptr.len()).expect("alloc");
    let mut ci = cuda.raw_alloc::<i32>(col_idx.len()).expect("alloc");
    let mut vs = cuda.raw_alloc::<f64>(values.len()).expect("alloc");
    let mut xd = cuda.raw_alloc::<f64>(n).expect("alloc");
    let mut yd = cuda.raw_alloc::<f64>(n).expect("alloc");
    cuda.try_copy_h2d(&row_ptr, &mut rp).expect("h2d");
    cuda.try_copy_h2d(&col_idx, &mut ci).expect("h2d");
    cuda.try_copy_h2d(&values, &mut vs).expect("h2d");
    cuda.try_copy_h2d(&x0, &mut xd).expect("h2d");
    let pre = cuda.prebake_csr(n, n, &rp, &ci, &vs);
    cuda.try_spmv_prebaked(&pre, &xd, &mut yd).expect("warmup");
    cuda.synchronize().expect("sync");
    let mut got = vec![0.0; n];
    cuda.try_copy_d2h(&yd, &mut got).expect("d2h");

    // ---- Raw cuSPARSE ----
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let mut rrp = stream.alloc_zeros::<i32>(row_ptr.len()).expect("alloc");
    let mut rci = stream.alloc_zeros::<i32>(col_idx.len()).expect("alloc");
    let mut rvs = stream.alloc_zeros::<f64>(values.len()).expect("alloc");
    let mut rx = stream.alloc_zeros::<f64>(n).expect("alloc");
    let mut ry = stream.alloc_zeros::<f64>(n).expect("alloc");
    stream.memcpy_htod(&row_ptr, &mut rrp).expect("h2d");
    stream.memcpy_htod(&col_idx, &mut rci).expect("h2d");
    stream.memcpy_htod(&values, &mut rvs).expect("h2d");
    stream.memcpy_htod(&x0, &mut rx).expect("h2d");
    let (rp_ptr, _a) = rrp.device_ptr(&stream);
    let (ci_ptr, _b) = rci.device_ptr(&stream);
    let (vs_ptr, _c) = rvs.device_ptr(&stream);
    let (x_ptr, _d) = rx.device_ptr(&stream);
    let y_ptr = ry.device_ptr_mut(&stream).0;
    // NOTE: the `Record` guard is intentionally dropped here; all raw uses
    // below are followed by explicit synchronisation before host access.
    let mut handle = std::ptr::null_mut();
    assert_eq!(
        unsafe { cusparse_sys::cusparseCreate(&mut handle) },
        cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS
    );
    let mut mat = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            cusparse_sys::cusparseCreateCsr(
                &mut mat,
                n as i64,
                n as i64,
                nnz as i64,
                rp_ptr as *mut _,
                ci_ptr as *mut _,
                vs_ptr as *mut _,
                cusparse_sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                cusparse_sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                cusparse_sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                cusparse_sys::cudaDataType::CUDA_R_64F,
            )
        },
        cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS
    );
    let mut vec_x = std::ptr::null_mut();
    let mut vec_y = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            cusparse_sys::cusparseCreateDnVec(
                &mut vec_x,
                n as i64,
                x_ptr as *mut _,
                cusparse_sys::cudaDataType::CUDA_R_64F,
            )
        },
        cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS
    );
    assert_eq!(
        unsafe {
            cusparse_sys::cusparseCreateDnVec(
                &mut vec_y,
                n as i64,
                y_ptr as *mut _,
                cusparse_sys::cudaDataType::CUDA_R_64F,
            )
        },
        cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS
    );
    let alpha = 1.0f64;
    let beta = 0.0f64;
    let mut buf_bytes = 0usize;
    assert_eq!(
        unsafe {
            cusparse_sys::cusparseSpMV_bufferSize(
                handle,
                cusparse_sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha as *const f64 as *const _,
                mat,
                vec_x,
                &beta as *const f64 as *const _,
                vec_y,
                cusparse_sys::cudaDataType::CUDA_R_64F,
                cusparse_sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
                &mut buf_bytes,
            )
        },
        cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS
    );
    let mut buf = stream.alloc_zeros::<u8>(buf_bytes.max(1)).expect("alloc");
    let (buf_ptr, _f) = buf.device_ptr_mut(&stream);
    let run_raw = || unsafe {
        cusparse_sys::cusparseSpMV(
            handle,
            cusparse_sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const f64 as *const _,
            mat,
            vec_x,
            &beta as *const f64 as *const _,
            vec_y,
            cusparse_sys::cudaDataType::CUDA_R_64F,
            cusparse_sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
            buf_ptr as *mut _,
        )
    };
    assert_eq!(
        run_raw(),
        cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS
    );
    stream.synchronize().expect("sync");
    // Interleaved best-of: alternate ours/raw per rep so GPU clock drift
    // affects both sides equally.
    let mut best_ours = f64::INFINITY;
    let mut best_raw = f64::INFINITY;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        cuda.try_spmv_prebaked(&pre, &xd, &mut yd).expect("spmv");
        cuda.synchronize().expect("sync");
        best_ours = best_ours.min(t.elapsed().as_secs_f64());
        let t = std::time::Instant::now();
        assert_eq!(
            run_raw(),
            cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS
        );
        stream.synchronize().expect("sync");
        best_raw = best_raw.min(t.elapsed().as_secs_f64());
    }
    // Correctness: ours vs raw.
    let mut want = vec![0.0; n];
    stream.memcpy_dtoh(&ry, &mut want).expect("d2h");
    let err = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    println!(
        "gpu_spmv laplacian {g}x{g} (n={n}, nnz={nnz}): ours {:7.3}ms {:6.1} GB/s | raw {:7.3}ms {:6.1} GB/s | maxdiff {err:.1e}",
        best_ours * 1000.0,
        bytes / best_ours / 1e9,
        best_raw * 1000.0,
        bytes / best_raw / 1e9
    );
    let _ = flops;
}

fn main() {
    for g in [256, 512, 1024] {
        bench_spmv(g, 20);
    }
}
