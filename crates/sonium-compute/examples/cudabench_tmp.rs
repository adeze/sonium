//! Temporary GPU baseline: our `CudaBackend::try_gemm` vs raw cuBLAS
//! (cudarc) on column-major squares. Scratch probe, not a committed test.

#![cfg(feature = "cuda")]

use cudarc::cublas::sys as cublas_sys;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::CudaContext;
use sonium_compute::{Backend, CudaBackend, Transpose};

fn bench_f64(n: usize, reps: usize) {
    let flops = 2.0 * (n as f64).powi(3);
    // Column-major native layout for cuBLAS; our backend takes the same
    // buffers with NoTrans/NoTrans (row-major API on col-major data still
    // computes C = A*B elementwise-identically for the timing purpose —
    // correctness is covered by the cuda acceptance tests, not here).
    let a: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.0123).sin()).collect();
    let b: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.0071).cos()).collect();
    let c0: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.0033).sin()).collect();

    // ---- Our backend ----
    let cuda = CudaBackend::new_default().expect("cuda backend");
    let mut ad = cuda.raw_alloc::<f64>(n * n).expect("alloc a");
    let mut bd = cuda.raw_alloc::<f64>(n * n).expect("alloc b");
    let mut cd = cuda.raw_alloc::<f64>(n * n).expect("alloc c");
    cuda.try_copy_h2d(&a, &mut ad).expect("h2d a");
    cuda.try_copy_h2d(&b, &mut bd).expect("h2d b");
    cuda.try_copy_h2d(&c0, &mut cd).expect("h2d c");
    cuda.try_gemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        n,
        n,
        n,
        1.0,
        &ad,
        n,
        &bd,
        n,
        0.0,
        &mut cd,
        n,
    )
    .expect("warmup gemm");
    cuda.synchronize().expect("sync");
    // End-to-end: refresh inputs, run, download (timed together).
    let mut cd2 = cuda.raw_alloc::<f64>(n * n).expect("alloc c2");
    let t = std::time::Instant::now();
    cuda.try_copy_h2d(&a, &mut ad).expect("h2d a");
    cuda.try_copy_h2d(&b, &mut bd).expect("h2d b");
    cuda.try_copy_h2d(&c0, &mut cd2).expect("h2d c");
    cuda.try_gemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        n,
        n,
        n,
        1.0,
        &ad,
        n,
        &bd,
        n,
        0.0,
        &mut cd2,
        n,
    )
    .expect("gemm");
    let mut got = vec![0.0; n * n];
    cuda.try_copy_d2h(&cd2, &mut got).expect("d2h");
    cuda.synchronize().expect("sync");
    let e2e = t.elapsed().as_secs_f64();
    std::hint::black_box(&got);

    // ---- Raw cuBLAS via cudarc ----
    let ctx = CudaContext::new(0).expect("cuda ctx");
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).expect("cublas");
    let mut ra = stream.alloc_zeros::<f64>(n * n).expect("alloc");
    let mut rb = stream.alloc_zeros::<f64>(n * n).expect("alloc");
    let mut rc = stream.alloc_zeros::<f64>(n * n).expect("alloc");
    stream.memcpy_htod(&a, &mut ra).expect("h2d");
    stream.memcpy_htod(&b, &mut rb).expect("h2d");
    stream.memcpy_htod(&c0, &mut rc).expect("h2d");
    let cfg = GemmConfig {
        transa: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: n as i32,
        k: n as i32,
        alpha: 1.0,
        lda: n as i32,
        ldb: n as i32,
        beta: 0.0,
        ldc: n as i32,
    };
    unsafe { blas.gemm(cfg, &ra, &rb, &mut rc) }.expect("warmup");
    stream.synchronize().expect("sync");
    // Interleaved best-of: alternate ours/raw per rep so GPU clock drift
    // affects both sides equally (sequential loops showed up to 40% phantom
    // gaps from boost drift between loops).
    let mut best_ours = f64::INFINITY;
    let mut best_raw = f64::INFINITY;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        cuda.try_gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            n,
            n,
            n,
            1.0,
            &ad,
            n,
            &bd,
            n,
            0.0,
            &mut cd,
            n,
        )
        .expect("gemm");
        cuda.synchronize().expect("sync");
        best_ours = best_ours.min(t.elapsed().as_secs_f64());
        let t = std::time::Instant::now();
        unsafe { blas.gemm(cfg, &ra, &rb, &mut rc) }.expect("gemm");
        stream.synchronize().expect("sync");
        best_raw = best_raw.min(t.elapsed().as_secs_f64());
    }

    println!(
        "gpu_dgemm n={n}: ours_kernel {:8.1} GF/s | raw_cublas {:8.1} | e2e_ours {:8.1} GB/s-equiv",
        flops / best_ours / 1e9,
        flops / best_raw / 1e9,
        (4.0 * (n as f64).powi(2) * 8.0) / e2e / 1e9
    );
}

macro_rules! bench_f32 {
    ($n:expr, $reps:expr) => {{
        let n = $n;
        let reps = $reps;
        let flops = 2.0 * (n as f64).powi(3);
        let a: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.0123).sin()).collect();
        let b: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.0071).cos()).collect();
        let c0: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.0033).sin()).collect();
        let cuda = CudaBackend::new_default().expect("cuda backend");
        let mut ad = cuda.raw_alloc::<f32>(n * n).expect("alloc a");
        let mut bd = cuda.raw_alloc::<f32>(n * n).expect("alloc b");
        let mut cd = cuda.raw_alloc::<f32>(n * n).expect("alloc c");
        cuda.try_copy_h2d(&a, &mut ad).expect("h2d a");
        cuda.try_copy_h2d(&b, &mut bd).expect("h2d b");
        cuda.try_copy_h2d(&c0, &mut cd).expect("h2d c");
        cuda.try_gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            n,
            n,
            n,
            1.0f32,
            &ad,
            n,
            &bd,
            n,
            0.0f32,
            &mut cd,
            n,
        )
        .expect("warmup gemm");
        cuda.synchronize().expect("sync");
        let ctx = CudaContext::new(0).expect("cuda ctx");
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).expect("cublas");
        let mut ra = stream.alloc_zeros::<f32>(n * n).expect("alloc");
        let mut rb = stream.alloc_zeros::<f32>(n * n).expect("alloc");
        let mut rc = stream.alloc_zeros::<f32>(n * n).expect("alloc");
        stream.memcpy_htod(&a, &mut ra).expect("h2d");
        stream.memcpy_htod(&b, &mut rb).expect("h2d");
        stream.memcpy_htod(&c0, &mut rc).expect("h2d");
        let cfg = GemmConfig {
            transa: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: n as i32,
            k: n as i32,
            alpha: 1.0f32,
            lda: n as i32,
            ldb: n as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };
        unsafe { blas.gemm(cfg, &ra, &rb, &mut rc) }.expect("warmup");
        stream.synchronize().expect("sync");
        // Interleaved best-of (see bench_f64): clock drift hits both sides.
        let mut best_ours = f64::INFINITY;
        let mut best_raw = f64::INFINITY;
        for _ in 0..reps {
            let t = std::time::Instant::now();
            cuda.try_gemm(
                Transpose::NoTrans,
                Transpose::NoTrans,
                n,
                n,
                n,
                1.0f32,
                &ad,
                n,
                &bd,
                n,
                0.0f32,
                &mut cd,
                n,
            )
            .expect("gemm");
            cuda.synchronize().expect("sync");
            best_ours = best_ours.min(t.elapsed().as_secs_f64());
            let t = std::time::Instant::now();
            unsafe { blas.gemm(cfg, &ra, &rb, &mut rc) }.expect("gemm");
            stream.synchronize().expect("sync");
            best_raw = best_raw.min(t.elapsed().as_secs_f64());
        }
        println!(
            "gpu_sgemm n={n}: ours_kernel {:8.1} GF/s | raw_cublas {:8.1}",
            flops / best_ours / 1e9,
            flops / best_raw / 1e9
        );
    }};
}

fn bench_dtrsm(m: usize, n: usize, reps: usize) {
    use cudarc::cublas::sys as bl;
    use sonium_compute::cpu::blas3::trsm::{TrsmSide, TrsmUplo};
    // Lower-triangular fixture with dominant diagonal, column-major.
    let mut a = vec![0.0f64; m * m];
    for i in 0..m {
        for j in 0..=i {
            a[j * m + i] = if i == j {
                2.0 + (i as f64 * 0.01).sin()
            } else {
                ((i * m + j) as f64 * 0.001) % 1.0
            };
        }
    }
    let b0: Vec<f64> = (0..m * n).map(|i| (i as f64 * 0.0123).sin()).collect();
    let flops = (m as f64) * (m as f64) * (n as f64);

    // ---- Our backend ----
    let cuda = CudaBackend::new_default().expect("cuda backend");
    let mut ad = cuda.raw_alloc::<f64>(a.len()).expect("alloc");
    let mut bd = cuda.raw_alloc::<f64>(b0.len()).expect("alloc");
    cuda.try_copy_h2d(&a, &mut ad).expect("h2d");
    cuda.try_copy_h2d(&b0, &mut bd).expect("h2d");
    cuda.try_trsm(
        TrsmSide::Left,
        TrsmUplo::Lower,
        Transpose::NoTrans,
        false,
        m,
        n,
        1.0,
        &ad,
        m,
        &mut bd,
        n,
    )
    .expect("warmup");
    cuda.synchronize().expect("sync");

    // ---- Raw cuBLAS ----
    let ctx = CudaContext::new(0).expect("ctx");
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
    let mut ra = stream.alloc_zeros::<f64>(a.len()).expect("alloc");
    let mut rb = stream.alloc_zeros::<f64>(b0.len()).expect("alloc");
    stream.memcpy_htod(&a, &mut ra).expect("h2d");
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let (ap, _g1) = ra.device_ptr(&stream);
    let alpha = 1.0f64;
    let run_raw = |bp: u64| unsafe {
        bl::cublasDtrsm_v2(
            handle,
            bl::cublasSideMode_t::CUBLAS_SIDE_LEFT,
            bl::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
            bl::cublasOperation_t::CUBLAS_OP_N,
            bl::cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
            m as i32,
            n as i32,
            &alpha,
            ap as *const f64,
            m as i32,
            bp as *mut f64,
            m as i32,
        )
    };
    stream.memcpy_htod(&b0, &mut rb).expect("h2d");
    let bp0 = rb.device_ptr_mut(&stream).0;
    assert_eq!(run_raw(bp0), bl::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    stream.synchronize().expect("sync");
    drop(rb);
    // Interleaved best-of (see bench_f64): clock drift hits both sides.
    let mut best_ours = f64::INFINITY;
    let mut best_raw = f64::INFINITY;
    for _ in 0..reps {
        cuda.try_copy_h2d(&b0, &mut bd).expect("h2d");
        let t = std::time::Instant::now();
        cuda.try_trsm(
            TrsmSide::Left,
            TrsmUplo::Lower,
            Transpose::NoTrans,
            false,
            m,
            n,
            1.0,
            &ad,
            m,
            &mut bd,
            n,
        )
        .expect("trsm");
        cuda.synchronize().expect("sync");
        best_ours = best_ours.min(t.elapsed().as_secs_f64());
        let mut rb = stream.alloc_zeros::<f64>(b0.len()).expect("alloc");
        stream.memcpy_htod(&b0, &mut rb).expect("h2d");
        let bp = rb.device_ptr_mut(&stream).0;
        let t = std::time::Instant::now();
        assert_eq!(run_raw(bp), bl::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        stream.synchronize().expect("sync");
        best_raw = best_raw.min(t.elapsed().as_secs_f64());
    }
    unsafe { bl::cublasDestroy_v2(handle) };

    println!(
        "gpu_dtrsm L m={m} n={n}: ours {:8.1} GF/s | raw {:8.1}",
        flops / best_ours / 1e9,
        flops / best_raw / 1e9
    );
}

fn main() {
    let reps = 10;
    for n in [1024, 2048, 4096, 8192] {
        bench_f32!(n, reps);
    }
    for n in [1024, 2048, 4096, 8192] {
        bench_f64(n, reps);
    }
    // Squares only: the row-major swap trick swaps m/n inside the wrapper,
    // so rectangular calls solve a different-shaped problem than raw
    // Left-side on the same buffers (same kernel class, incomparable flops).
    for (m, n) in [(1024, 1024), (2048, 2048), (4096, 4096)] {
        bench_dtrsm(m, n, reps);
    }
}
