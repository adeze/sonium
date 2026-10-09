use super::*;

pub unsafe fn axpy_f32(
    h: cublas_sys::cublasHandle_t,
    n: i32,
    alpha: f32,
    x: u64,
    y: u64,
) -> cublas_sys::cublasStatus_t {
    unsafe {
        cublas_sys::cublasSaxpy_v2(
            h,
            n,
            &alpha as *const f32,
            x as *const f32,
            1,
            y as *mut f32,
            1,
        )
    }
}

pub unsafe fn axpy_f64(
    h: cublas_sys::cublasHandle_t,
    n: i32,
    alpha: f64,
    x: u64,
    y: u64,
) -> cublas_sys::cublasStatus_t {
    unsafe {
        cublas_sys::cublasDaxpy_v2(
            h,
            n,
            &alpha as *const f64,
            x as *const f64,
            1,
            y as *mut f64,
            1,
        )
    }
}

pub unsafe fn axpy_c32(
    h: cublas_sys::cublasHandle_t,
    n: i32,
    alpha: Complex32,
    x: u64,
    y: u64,
) -> cublas_sys::cublasStatus_t {
    let alpha = cu_complex32(alpha);
    unsafe {
        cublas_sys::cublasCaxpy_v2(
            h,
            n,
            &alpha as *const cublas_sys::cuComplex,
            x as *const cublas_sys::cuComplex,
            1,
            y as *mut cublas_sys::cuComplex,
            1,
        )
    }
}

pub unsafe fn axpy_c64(
    h: cublas_sys::cublasHandle_t,
    n: i32,
    alpha: Complex64,
    x: u64,
    y: u64,
) -> cublas_sys::cublasStatus_t {
    let alpha = cu_complex64(alpha);
    unsafe {
        cublas_sys::cublasZaxpy_v2(
            h,
            n,
            &alpha as *const cublas_sys::cuDoubleComplex,
            x as *const cublas_sys::cuDoubleComplex,
            1,
            y as *mut cublas_sys::cuDoubleComplex,
            1,
        )
    }
}

pub unsafe fn dot_f32(
    h: cublas_sys::cublasHandle_t,
    n: i32,
    x: u64,
    y: u64,
) -> Result<f32, cublas_sys::cublasStatus_t> {
    let mut out: f32 = 0.0;
    let st = unsafe {
        cublas_sys::cublasSdot_v2(
            h,
            n,
            x as *const f32,
            1,
            y as *const f32,
            1,
            &mut out as *mut f32,
        )
    };
    ok(out, st)
}

pub unsafe fn dot_f64(
    h: cublas_sys::cublasHandle_t,
    n: i32,
    x: u64,
    y: u64,
) -> Result<f64, cublas_sys::cublasStatus_t> {
    let mut out: f64 = 0.0;
    let st = unsafe {
        cublas_sys::cublasDdot_v2(
            h,
            n,
            x as *const f64,
            1,
            y as *const f64,
            1,
            &mut out as *mut f64,
        )
    };
    ok(out, st)
}

/// Hermitian dot (conjugating in `x`). Matches `Backend::dot` contract.
pub unsafe fn dotc_c32(
    h: cublas_sys::cublasHandle_t,
    n: i32,
    x: u64,
    y: u64,
) -> Result<Complex32, cublas_sys::cublasStatus_t> {
    let mut out = cublas_sys::cuComplex { x: 0.0, y: 0.0 };
    let st = unsafe {
        cublas_sys::cublasCdotc_v2(
            h,
            n,
            x as *const cublas_sys::cuComplex,
            1,
            y as *const cublas_sys::cuComplex,
            1,
            &mut out as *mut cublas_sys::cuComplex,
        )
    };
    ok(complex32_from_cu(out), st)
}

pub unsafe fn dotc_c64(
    h: cublas_sys::cublasHandle_t,
    n: i32,
    x: u64,
    y: u64,
) -> Result<Complex64, cublas_sys::cublasStatus_t> {
    let mut out = cublas_sys::cuDoubleComplex { x: 0.0, y: 0.0 };
    let st = unsafe {
        cublas_sys::cublasZdotc_v2(
            h,
            n,
            x as *const cublas_sys::cuDoubleComplex,
            1,
            y as *const cublas_sys::cuDoubleComplex,
            1,
            &mut out as *mut cublas_sys::cuDoubleComplex,
        )
    };
    ok(complex64_from_cu(out), st)
}

pub unsafe fn nrm2_f32(
    h: cublas_sys::cublasHandle_t,
    n: i32,
    x: u64,
) -> Result<f32, cublas_sys::cublasStatus_t> {
    let mut out: f32 = 0.0;
    let st = unsafe { cublas_sys::cublasSnrm2_v2(h, n, x as *const f32, 1, &mut out as *mut f32) };
    ok(out, st)
}

pub unsafe fn nrm2_f64(
    h: cublas_sys::cublasHandle_t,
    n: i32,
    x: u64,
) -> Result<f64, cublas_sys::cublasStatus_t> {
    let mut out: f64 = 0.0;
    let st = unsafe { cublas_sys::cublasDnrm2_v2(h, n, x as *const f64, 1, &mut out as *mut f64) };
    ok(out, st)
}

pub unsafe fn nrm2_c32(
    h: cublas_sys::cublasHandle_t,
    n: i32,
    x: u64,
) -> Result<f32, cublas_sys::cublasStatus_t> {
    let mut out: f32 = 0.0;
    let st = unsafe {
        cublas_sys::cublasScnrm2_v2(
            h,
            n,
            x as *const cublas_sys::cuComplex,
            1,
            &mut out as *mut f32,
        )
    };
    ok(out, st)
}

pub unsafe fn nrm2_c64(
    h: cublas_sys::cublasHandle_t,
    n: i32,
    x: u64,
) -> Result<f64, cublas_sys::cublasStatus_t> {
    let mut out: f64 = 0.0;
    let st = unsafe {
        cublas_sys::cublasDznrm2_v2(
            h,
            n,
            x as *const cublas_sys::cuDoubleComplex,
            1,
            &mut out as *mut f64,
        )
    };
    ok(out, st)
}

pub unsafe fn scal_f32(
    h: cublas_sys::cublasHandle_t,
    n: i32,
    alpha: f32,
    x: u64,
) -> cublas_sys::cublasStatus_t {
    unsafe { cublas_sys::cublasSscal_v2(h, n, &alpha as *const f32, x as *mut f32, 1) }
}

pub unsafe fn scal_f64(
    h: cublas_sys::cublasHandle_t,
    n: i32,
    alpha: f64,
    x: u64,
) -> cublas_sys::cublasStatus_t {
    unsafe { cublas_sys::cublasDscal_v2(h, n, &alpha as *const f64, x as *mut f64, 1) }
}

pub unsafe fn scal_c32(
    h: cublas_sys::cublasHandle_t,
    n: i32,
    alpha: Complex32,
    x: u64,
) -> cublas_sys::cublasStatus_t {
    let alpha = cu_complex32(alpha);
    unsafe {
        cublas_sys::cublasCscal_v2(
            h,
            n,
            &alpha as *const cublas_sys::cuComplex,
            x as *mut cublas_sys::cuComplex,
            1,
        )
    }
}

pub unsafe fn scal_c64(
    h: cublas_sys::cublasHandle_t,
    n: i32,
    alpha: Complex64,
    x: u64,
) -> cublas_sys::cublasStatus_t {
    let alpha = cu_complex64(alpha);
    unsafe {
        cublas_sys::cublasZscal_v2(
            h,
            n,
            &alpha as *const cublas_sys::cuDoubleComplex,
            x as *mut cublas_sys::cuDoubleComplex,
            1,
        )
    }
}

fn ok<T>(out: T, st: cublas_sys::cublasStatus_t) -> Result<T, cublas_sys::cublasStatus_t> {
    if st == cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(out)
    } else {
        Err(st)
    }
}
