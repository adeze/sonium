//! Temporary GPU baseline 2: our `CudaBackend` vs raw cuBLAS/cuSPARSE for
//! complex GEMM, SYRK, and CSR SpMM. Scratch probe, not a committed test.

#![cfg(feature = "cuda")]

use cudarc::cublas::sys as bl;
use cudarc::cusparse::sys as sp;
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use num_complex::{Complex32, Complex64};
use sonium_compute::cpu::blas3::trsm::TrsmUplo;
use sonium_compute::{Backend, CudaBackend, Transpose};
use std::sync::Arc;

fn backend() -> CudaBackend {
    CudaBackend::new_default().expect("cuda backend")
}

fn raw_handle(ctx: &Arc<CudaContext>) -> bl::cublasHandle_t {
    let stream = ctx.default_stream();
    let mut handle = std::ptr::null_mut();
    assert_eq!(
        unsafe { bl::cublasCreate_v2(&mut handle) },
        bl::cublasStatus_t::CUBLAS_STATUS_SUCCESS
    );
    assert_eq!(
        unsafe { bl::cublasSetStream_v2(handle, stream.cu_stream() as _) },
        bl::cublasStatus_t::CUBLAS_STATUS_SUCCESS
    );
    handle
}

fn interleave32(v: &[Complex32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for z in v {
        out.push(z.re);
        out.push(z.im);
    }
    out
}

fn interleave64(v: &[Complex64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for z in v {
        out.push(z.re);
        out.push(z.im);
    }
    out
}

fn bench_cgemm(n: usize, reps: usize) {
    let flops = 8.0 * (n as f64).powi(3); // complex MAC = 8 flops
    let mk = |f: &dyn Fn(usize) -> Complex32| (0..n * n).map(f).collect::<Vec<_>>();
    let a = mk(&|i| Complex32::new((i as f32 * 0.0123).sin(), (i as f32 * 0.004).cos()));
    let b = mk(&|i| Complex32::new((i as f32 * 0.0071).cos(), (i as f32 * 0.003).sin()));
    let c0 = mk(&|i| Complex32::new((i % 5) as f32, 0.0));

    let cuda = backend();
    let mut ad = cuda.raw_alloc::<Complex32>(n * n).expect("alloc");
    let mut bd = cuda.raw_alloc::<Complex32>(n * n).expect("alloc");
    let mut cd = cuda.raw_alloc::<Complex32>(n * n).expect("alloc");
    cuda.try_copy_h2d(&a, &mut ad).expect("h2d");
    cuda.try_copy_h2d(&b, &mut bd).expect("h2d");
    cuda.try_copy_h2d(&c0, &mut cd).expect("h2d");
    cuda.try_gemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        n,
        n,
        n,
        Complex32::new(1.0, 0.0),
        &ad,
        n,
        &bd,
        n,
        Complex32::new(0.0, 0.0),
        &mut cd,
        n,
    )
    .expect("warmup");
    cuda.synchronize().expect("sync");

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let handle = raw_handle(&ctx);
    let mut ra = stream.alloc_zeros::<f32>(2 * n * n).expect("alloc");
    let mut rb = stream.alloc_zeros::<f32>(2 * n * n).expect("alloc");
    let mut rc = stream.alloc_zeros::<f32>(2 * n * n).expect("alloc");
    stream.memcpy_htod(&interleave32(&a), &mut ra).expect("h2d");
    stream.memcpy_htod(&interleave32(&b), &mut rb).expect("h2d");
    let (ap, _g1) = ra.device_ptr(&stream);
    let (bp, _g2) = rb.device_ptr(&stream);
    let alpha = Complex32::new(1.0, 0.0);
    let beta = Complex32::new(0.0, 0.0);
    let run_raw = |cp: u64| unsafe {
        bl::cublasCgemm_v2(
            handle,
            bl::cublasOperation_t::CUBLAS_OP_N,
            bl::cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            n as i32,
            n as i32,
            &alpha as *const Complex32 as *const bl::cuComplex,
            ap as *const bl::cuComplex,
            n as i32,
            bp as *const bl::cuComplex,
            n as i32,
            &beta as *const Complex32 as *const bl::cuComplex,
            cp as *mut bl::cuComplex,
            n as i32,
        )
    };
    stream
        .memcpy_htod(&interleave32(&c0), &mut rc)
        .expect("h2d");
    let cp0 = rc.device_ptr_mut(&stream).0;
    assert_eq!(run_raw(cp0), bl::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    stream.synchronize().expect("sync");
    drop(rc);

    // Saturate clocks before timing: 3s of untimed alternating reps.
    let mut wrc = stream.alloc_zeros::<f32>(2 * n * n).expect("alloc");
    let w0 = std::time::Instant::now();
    let mut it = 0u32;
    while w0.elapsed().as_secs_f64() < 3.0 {
        it += 1;
        cuda.try_gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            n,
            n,
            n,
            Complex32::new(1.0, 0.0),
            &ad,
            n,
            &bd,
            n,
            Complex32::new(0.0, 0.0),
            &mut cd,
            n,
        )
        .expect("gemm");
        let wcp = wrc.device_ptr_mut(&stream).0;
        assert_eq!(run_raw(wcp), bl::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        if it % 25 == 0 {
            cuda.synchronize().expect("sync");
            stream.synchronize().expect("sync");
        }
    }
    cuda.synchronize().expect("sync");
    stream.synchronize().expect("sync");
    drop(wrc);
    let mut best_ours = f64::INFINITY;
    let mut best_raw = f64::INFINITY;
    for _ in 0..reps {
        cuda.try_copy_h2d(&c0, &mut cd).expect("h2d");
        let t = std::time::Instant::now();
        cuda.try_gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            n,
            n,
            n,
            Complex32::new(1.0, 0.0),
            &ad,
            n,
            &bd,
            n,
            Complex32::new(0.0, 0.0),
            &mut cd,
            n,
        )
        .expect("gemm");
        cuda.synchronize().expect("sync");
        best_ours = best_ours.min(t.elapsed().as_secs_f64());
        let mut rc = stream.alloc_zeros::<f32>(2 * n * n).expect("alloc");
        stream
            .memcpy_htod(&interleave32(&c0), &mut rc)
            .expect("h2d");
        let cp = rc.device_ptr_mut(&stream).0;
        let t = std::time::Instant::now();
        assert_eq!(run_raw(cp), bl::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        stream.synchronize().expect("sync");
        best_raw = best_raw.min(t.elapsed().as_secs_f64());
    }
    unsafe { bl::cublasDestroy_v2(handle) };
    println!(
        "gpu_cgemm n={n}: ours {:8.1} GF/s | raw {:8.1}",
        flops / best_ours / 1e9,
        flops / best_raw / 1e9
    );
}

fn bench_zgemm(n: usize, reps: usize) {
    let flops = 8.0 * (n as f64).powi(3);
    let mk = |f: &dyn Fn(usize) -> Complex64| (0..n * n).map(f).collect::<Vec<_>>();
    let a = mk(&|i| Complex64::new((i as f64 * 0.0123).sin(), (i as f64 * 0.004).cos()));
    let b = mk(&|i| Complex64::new((i as f64 * 0.0071).cos(), (i as f64 * 0.003).sin()));
    let c0 = mk(&|i| Complex64::new((i % 5) as f64, 0.0));

    let cuda = backend();
    let mut ad = cuda.raw_alloc::<Complex64>(n * n).expect("alloc");
    let mut bd = cuda.raw_alloc::<Complex64>(n * n).expect("alloc");
    let mut cd = cuda.raw_alloc::<Complex64>(n * n).expect("alloc");
    cuda.try_copy_h2d(&a, &mut ad).expect("h2d");
    cuda.try_copy_h2d(&b, &mut bd).expect("h2d");
    cuda.try_copy_h2d(&c0, &mut cd).expect("h2d");
    cuda.try_gemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        n,
        n,
        n,
        Complex64::new(1.0, 0.0),
        &ad,
        n,
        &bd,
        n,
        Complex64::new(0.0, 0.0),
        &mut cd,
        n,
    )
    .expect("warmup");
    cuda.synchronize().expect("sync");

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let handle = raw_handle(&ctx);
    let mut ra = stream.alloc_zeros::<f64>(2 * n * n).expect("alloc");
    let mut rb = stream.alloc_zeros::<f64>(2 * n * n).expect("alloc");
    let mut rc = stream.alloc_zeros::<f64>(2 * n * n).expect("alloc");
    stream.memcpy_htod(&interleave64(&a), &mut ra).expect("h2d");
    stream.memcpy_htod(&interleave64(&b), &mut rb).expect("h2d");
    let (ap, _g1) = ra.device_ptr(&stream);
    let (bp, _g2) = rb.device_ptr(&stream);
    let alpha = Complex64::new(1.0, 0.0);
    let beta = Complex64::new(0.0, 0.0);
    let run_raw = |cp: u64| unsafe {
        bl::cublasZgemm_v2(
            handle,
            bl::cublasOperation_t::CUBLAS_OP_N,
            bl::cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            n as i32,
            n as i32,
            &alpha as *const Complex64 as *const bl::cuDoubleComplex,
            ap as *const bl::cuDoubleComplex,
            n as i32,
            bp as *const bl::cuDoubleComplex,
            n as i32,
            &beta as *const Complex64 as *const bl::cuDoubleComplex,
            cp as *mut bl::cuDoubleComplex,
            n as i32,
        )
    };
    stream
        .memcpy_htod(&interleave64(&c0), &mut rc)
        .expect("h2d");
    let cp0 = rc.device_ptr_mut(&stream).0;
    assert_eq!(run_raw(cp0), bl::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    stream.synchronize().expect("sync");
    drop(rc);

    // Saturate clocks before timing: 3s of untimed alternating reps.
    let mut wrc = stream.alloc_zeros::<f64>(2 * n * n).expect("alloc");
    let w0 = std::time::Instant::now();
    let mut it = 0u32;
    while w0.elapsed().as_secs_f64() < 3.0 {
        it += 1;
        cuda.try_gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            n,
            n,
            n,
            Complex64::new(1.0, 0.0),
            &ad,
            n,
            &bd,
            n,
            Complex64::new(0.0, 0.0),
            &mut cd,
            n,
        )
        .expect("gemm");
        let wcp = wrc.device_ptr_mut(&stream).0;
        assert_eq!(run_raw(wcp), bl::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        if it % 25 == 0 {
            cuda.synchronize().expect("sync");
            stream.synchronize().expect("sync");
        }
    }
    cuda.synchronize().expect("sync");
    stream.synchronize().expect("sync");
    drop(wrc);
    let mut best_ours = f64::INFINITY;
    let mut best_raw = f64::INFINITY;
    for _ in 0..reps {
        cuda.try_copy_h2d(&c0, &mut cd).expect("h2d");
        let t = std::time::Instant::now();
        cuda.try_gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            n,
            n,
            n,
            Complex64::new(1.0, 0.0),
            &ad,
            n,
            &bd,
            n,
            Complex64::new(0.0, 0.0),
            &mut cd,
            n,
        )
        .expect("gemm");
        cuda.synchronize().expect("sync");
        best_ours = best_ours.min(t.elapsed().as_secs_f64());
        let mut rc = stream.alloc_zeros::<f64>(2 * n * n).expect("alloc");
        stream
            .memcpy_htod(&interleave64(&c0), &mut rc)
            .expect("h2d");
        let cp = rc.device_ptr_mut(&stream).0;
        let t = std::time::Instant::now();
        assert_eq!(run_raw(cp), bl::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        stream.synchronize().expect("sync");
        best_raw = best_raw.min(t.elapsed().as_secs_f64());
    }
    unsafe { bl::cublasDestroy_v2(handle) };
    println!(
        "gpu_zgemm n={n}: ours {:8.1} GF/s | raw {:8.1}",
        flops / best_ours / 1e9,
        flops / best_raw / 1e9
    );
}

fn bench_ssyrk(n: usize, k: usize, reps: usize) {
    let flops = 1.0 * (n as f64) * (n as f64) * (k as f64); // ~n^2 k (one triangle)
    let a: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.0123).sin()).collect();
    let c0: Vec<f32> = (0..n * n).map(|i| (i % 5) as f32).collect();

    let cuda = backend();
    let mut ad = cuda.raw_alloc::<f32>(n * k).expect("alloc");
    let mut cd = cuda.raw_alloc::<f32>(n * n).expect("alloc");
    cuda.try_copy_h2d(&a, &mut ad).expect("h2d");
    cuda.try_copy_h2d(&c0, &mut cd).expect("h2d");
    cuda.try_syrk(
        TrsmUplo::Upper,
        Transpose::NoTrans,
        n,
        k,
        1.0,
        &ad,
        k,
        0.0,
        &mut cd,
        n,
    )
    .expect("warmup");
    cuda.synchronize().expect("sync");

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let handle = raw_handle(&ctx);
    let mut ra = stream.alloc_zeros::<f32>(n * k).expect("alloc");
    let mut rc = stream.alloc_zeros::<f32>(n * n).expect("alloc");
    stream.memcpy_htod(&a, &mut ra).expect("h2d");
    let (ap, _g1) = ra.device_ptr(&stream);
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let run_raw = |cp: u64| unsafe {
        bl::cublasSsyrk_v2(
            handle,
            bl::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
            bl::cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            k as i32,
            &alpha,
            ap as *const f32,
            k as i32,
            &beta,
            cp as *mut f32,
            n as i32,
        )
    };
    stream.memcpy_htod(&c0, &mut rc).expect("h2d");
    let cp0 = rc.device_ptr_mut(&stream).0;
    assert_eq!(run_raw(cp0), bl::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    stream.synchronize().expect("sync");
    drop(rc);

    // Saturate clocks before timing: 3s of untimed alternating reps.
    let mut wrc = stream.alloc_zeros::<f32>(n * n).expect("alloc");
    let w0 = std::time::Instant::now();
    let mut it = 0u32;
    while w0.elapsed().as_secs_f64() < 3.0 {
        it += 1;
        cuda.try_syrk(
            TrsmUplo::Upper,
            Transpose::NoTrans,
            n,
            k,
            1.0,
            &ad,
            k,
            0.0,
            &mut cd,
            n,
        )
        .expect("syrk");
        let wcp = wrc.device_ptr_mut(&stream).0;
        assert_eq!(run_raw(wcp), bl::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        if it % 25 == 0 {
            cuda.synchronize().expect("sync");
            stream.synchronize().expect("sync");
        }
    }
    cuda.synchronize().expect("sync");
    stream.synchronize().expect("sync");
    drop(wrc);
    let mut best_ours = f64::INFINITY;
    let mut best_raw = f64::INFINITY;
    for _ in 0..reps {
        cuda.try_copy_h2d(&c0, &mut cd).expect("h2d");
        let t = std::time::Instant::now();
        cuda.try_syrk(
            TrsmUplo::Upper,
            Transpose::NoTrans,
            n,
            k,
            1.0,
            &ad,
            k,
            0.0,
            &mut cd,
            n,
        )
        .expect("syrk");
        cuda.synchronize().expect("sync");
        best_ours = best_ours.min(t.elapsed().as_secs_f64());
        let mut rc = stream.alloc_zeros::<f32>(n * n).expect("alloc");
        stream.memcpy_htod(&c0, &mut rc).expect("h2d");
        let cp = rc.device_ptr_mut(&stream).0;
        let t = std::time::Instant::now();
        assert_eq!(run_raw(cp), bl::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        stream.synchronize().expect("sync");
        best_raw = best_raw.min(t.elapsed().as_secs_f64());
    }
    unsafe { bl::cublasDestroy_v2(handle) };
    println!(
        "gpu_ssyrk {n}x{k}: ours {:8.1} GF/s | raw {:8.1}",
        flops / best_ours / 1e9,
        flops / best_raw / 1e9
    );
}

fn laplacian(g: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let n = g * g;
    let mut rp = Vec::with_capacity(n + 1);
    let mut ci = Vec::new();
    let mut vs = Vec::new();
    rp.push(0);
    for i in 0..n {
        let (r, c) = (i / g, i % g);
        let mut row = vec![(i as i32, 4.0)];
        if r > 0 {
            row.push(((i - g) as i32, -1.0));
        }
        if r + 1 < g {
            row.push(((i + g) as i32, -1.0));
        }
        if c > 0 {
            row.push(((i - 1) as i32, -1.0));
        }
        if c + 1 < g {
            row.push(((i + 1) as i32, -1.0));
        }
        row.sort_by_key(|&(j, _)| j);
        for (j, v) in row {
            ci.push(j);
            vs.push(v);
        }
        rp.push(ci.len() as i32);
    }
    (rp, ci, vs)
}

fn bench_spmm(g: usize, nrhs: usize, reps: usize) {
    let n = g * g;
    let (row_ptr, col_idx, values) = laplacian(g);
    let nnz = values.len();
    let flops = 2.0 * nnz as f64 * nrhs as f64;
    let x0: Vec<f64> = (0..n * nrhs).map(|i| (i as f64 * 0.013).sin()).collect();

    let cuda = backend();
    let mut rp = cuda.raw_alloc::<i32>(row_ptr.len()).expect("alloc");
    let mut ci = cuda.raw_alloc::<i32>(col_idx.len()).expect("alloc");
    let mut vs = cuda.raw_alloc::<f64>(values.len()).expect("alloc");
    let mut xd = cuda.raw_alloc::<f64>(n * nrhs).expect("alloc");
    let mut yd = cuda.raw_alloc::<f64>(n * nrhs).expect("alloc");
    cuda.try_copy_h2d(&row_ptr, &mut rp).expect("h2d");
    cuda.try_copy_h2d(&col_idx, &mut ci).expect("h2d");
    cuda.try_copy_h2d(&values, &mut vs).expect("h2d");
    cuda.try_copy_h2d(&x0, &mut xd).expect("h2d");
    let pre = cuda.prebake_csr(n, n, &rp, &ci, &vs);
    cuda.try_spmm_prebaked(&pre, nrhs, &xd, n, &mut yd, n)
        .expect("warmup");
    cuda.synchronize().expect("sync");

    // ---- Raw cuSPARSE SpMM, same descriptors/algorithm ----
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let mut handle = std::ptr::null_mut();
    assert_eq!(
        unsafe { sp::cusparseCreate(&mut handle) },
        sp::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS
    );
    assert_eq!(
        unsafe { sp::cusparseSetStream(handle, stream.cu_stream() as _) },
        sp::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS
    );
    let mut rrp = stream.alloc_zeros::<i32>(row_ptr.len()).expect("alloc");
    let mut rci = stream.alloc_zeros::<i32>(col_idx.len()).expect("alloc");
    let mut rvs = stream.alloc_zeros::<f64>(values.len()).expect("alloc");
    let mut rx = stream.alloc_zeros::<f64>(n * nrhs).expect("alloc");
    let mut ry = stream.alloc_zeros::<f64>(n * nrhs).expect("alloc");
    stream.memcpy_htod(&row_ptr, &mut rrp).expect("h2d");
    stream.memcpy_htod(&col_idx, &mut rci).expect("h2d");
    stream.memcpy_htod(&values, &mut rvs).expect("h2d");
    stream.memcpy_htod(&x0, &mut rx).expect("h2d");
    let (rp_ptr, _a) = rrp.device_ptr(&stream);
    let (ci_ptr, _b) = rci.device_ptr(&stream);
    let (vs_ptr, _c) = rvs.device_ptr(&stream);
    let (x_ptr, _d) = rx.device_ptr(&stream);
    let y_ptr = ry.device_ptr_mut(&stream).0;
    let mut mat = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            sp::cusparseCreateCsr(
                &mut mat,
                n as i64,
                n as i64,
                nnz as i64,
                rp_ptr as *mut _,
                ci_ptr as *mut _,
                vs_ptr as *mut _,
                sp::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                sp::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                sp::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                sp::cudaDataType::CUDA_R_64F,
            )
        },
        sp::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS
    );
    let mut vec_x = std::ptr::null_mut();
    let mut vec_y = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            sp::cusparseCreateDnMat(
                &mut vec_x,
                n as i64,
                nrhs as i64,
                n as i64,
                x_ptr as *mut _,
                sp::cudaDataType::CUDA_R_64F,
                sp::cusparseOrder_t::CUSPARSE_ORDER_COL,
            )
        },
        sp::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS
    );
    assert_eq!(
        unsafe {
            sp::cusparseCreateDnMat(
                &mut vec_y,
                n as i64,
                nrhs as i64,
                n as i64,
                y_ptr as *mut _,
                sp::cudaDataType::CUDA_R_64F,
                sp::cusparseOrder_t::CUSPARSE_ORDER_COL,
            )
        },
        sp::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS
    );
    let alpha = 1.0f64;
    let beta = 0.0f64;
    let op = sp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE;
    let alg = sp::cusparseSpMMAlg_t::CUSPARSE_SPMM_ALG_DEFAULT;
    let mut buf_bytes = 0usize;
    assert_eq!(
        unsafe {
            sp::cusparseSpMM_bufferSize(
                handle,
                op,
                op,
                &alpha as *const f64 as *const _,
                mat,
                vec_x,
                &beta as *const f64 as *const _,
                vec_y,
                sp::cudaDataType::CUDA_R_64F,
                alg,
                &mut buf_bytes,
            )
        },
        sp::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS
    );
    let mut wbuf = stream.alloc_zeros::<u8>(buf_bytes.max(1)).expect("wbuf");
    let (wptr, _w) = wbuf.device_ptr(&stream);
    let run_raw = |yp: u64| unsafe {
        // NOTE: vec_y wraps the recycled output buffer; recreate per rep.
        let mut vy = std::ptr::null_mut();
        assert_eq!(
            sp::cusparseCreateDnMat(
                &mut vy,
                n as i64,
                nrhs as i64,
                n as i64,
                yp as *mut _,
                sp::cudaDataType::CUDA_R_64F,
                sp::cusparseOrder_t::CUSPARSE_ORDER_COL,
            ),
            sp::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS
        );
        let st = sp::cusparseSpMM(
            handle,
            op,
            op,
            &alpha as *const f64 as *const _,
            mat,
            vec_x,
            &beta as *const f64 as *const _,
            vy,
            sp::cudaDataType::CUDA_R_64F,
            alg,
            wptr as *mut _,
        );
        sp::cusparseDestroyDnMat(vy);
        st
    };
    assert_eq!(
        run_raw(y_ptr),
        sp::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS
    );
    stream.synchronize().expect("sync");

    // Saturate clocks before timing: 3s of untimed alternating reps.
    let w0 = std::time::Instant::now();
    let mut it = 0u32;
    while w0.elapsed().as_secs_f64() < 3.0 {
        it += 1;
        cuda.try_spmm_prebaked(&pre, nrhs, &xd, n, &mut yd, n)
            .expect("spmm");
        assert_eq!(
            run_raw(y_ptr),
            sp::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS
        );
        if it % 25 == 0 {
            cuda.synchronize().expect("sync");
            stream.synchronize().expect("sync");
        }
    }
    cuda.synchronize().expect("sync");
    stream.synchronize().expect("sync");
    let mut best_ours = f64::INFINITY;
    let mut best_raw = f64::INFINITY;
    for _ in 0..reps {
        cuda.try_copy_h2d(&x0, &mut xd).expect("h2d");
        let t = std::time::Instant::now();
        cuda.try_spmm_prebaked(&pre, nrhs, &xd, n, &mut yd, n)
            .expect("spmm");
        cuda.synchronize().expect("sync");
        best_ours = best_ours.min(t.elapsed().as_secs_f64());
        let mut ry = stream.alloc_zeros::<f64>(n * nrhs).expect("alloc");
        let yp = ry.device_ptr_mut(&stream).0;
        let t = std::time::Instant::now();
        assert_eq!(run_raw(yp), sp::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS);
        stream.synchronize().expect("sync");
        best_raw = best_raw.min(t.elapsed().as_secs_f64());
    }
    unsafe {
        sp::cusparseDestroyDnMat(vec_x);
        sp::cusparseDestroyDnMat(vec_y);
        sp::cusparseDestroySpMat(mat);
        sp::cusparseDestroy(handle);
    }
    println!(
        "gpu_spmm laplacian {g}x{g} nrhs={nrhs}: ours {:8.1} GF/s | raw {:8.1}",
        flops / best_ours / 1e9,
        flops / best_raw / 1e9
    );
}

fn main() {
    let reps = 10;
    for n in [2048, 4096] {
        bench_cgemm(n, reps);
    }
    for n in [2048, 4096] {
        bench_zgemm(n, reps);
    }
    bench_ssyrk(4096, 4096, reps);
    bench_spmm(512, 16, 20);
    bench_spmm(512, 64, 20);
}
