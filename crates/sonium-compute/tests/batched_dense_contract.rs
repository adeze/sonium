//! Checked grouped dense-operation contracts.
//!
//! These cases intentionally use two independent problems in one batch.  The
//! grouped path must produce the same values as the scalar reference while
//! preserving typed shape failures for caller-owned buffers.

use num_complex::{Complex32, Complex64};
use num_traits::ToPrimitive;
use sonium_compute::{
    Backend, BackendError, CpuBackend, DeviceScalar, GemmBatch, GemvBatch, MemoryPool,
    MemoryPoolConfig, Transpose,
};

fn assert_close<T>(actual: &[T], expected: &[T], tolerance: f64)
where
    T: DeviceScalar,
    T::Real: ToPrimitive,
{
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        let error = actual
            .sub(*expected)
            .norm_sqr()
            .to_f64()
            .expect("real norm converts to f64")
            .sqrt();
        let scale = expected
            .norm_sqr()
            .to_f64()
            .expect("real norm converts to f64")
            .sqrt()
            .max(1.0);
        assert!(
            error <= tolerance * scale,
            "error {error} > {}",
            tolerance * scale
        );
    }
}

fn run_gemv<T>(values: [T; 8], tolerance: f64)
where
    T: DeviceScalar,
    T::Real: ToPrimitive,
{
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 16)).expect("pool");
    let backend = pool.backend();
    let mut a0 = pool.try_alloc::<T>(4).expect("a0");
    let mut a1 = pool.try_alloc::<T>(4).expect("a1");
    let mut x0 = pool.try_alloc::<T>(2).expect("x0");
    let mut x1 = pool.try_alloc::<T>(2).expect("x1");
    let mut y0 = pool.try_alloc::<T>(2).expect("y0");
    let mut y1 = pool.try_alloc::<T>(2).expect("y1");
    backend.copy_h2d(&values[0..4], a0.raw_mut());
    backend.copy_h2d(&values[0..4], a1.raw_mut());
    backend.copy_h2d(&values[4..6], x0.raw_mut());
    backend.copy_h2d(&values[6..8], x1.raw_mut());
    backend.copy_h2d(&values[4..6], y0.raw_mut());
    backend.copy_h2d(&values[6..8], y1.raw_mut());

    let mut batch = [
        GemvBatch {
            op: Transpose::NoTrans,
            m: 2,
            n: 2,
            alpha: T::one(),
            a: a0.raw(),
            lda: 2,
            x: x0.raw(),
            beta: T::zero(),
            y: y0.raw_mut(),
        },
        GemvBatch {
            op: Transpose::NoTrans,
            m: 2,
            n: 2,
            alpha: T::one(),
            a: a1.raw(),
            lda: 2,
            x: x1.raw(),
            beta: T::zero(),
            y: y1.raw_mut(),
        },
    ];
    backend.try_gemv_batched(&mut batch).expect("batched GEMV");

    let mut actual0 = [T::zero(); 2];
    let mut actual1 = [T::zero(); 2];
    backend.copy_d2h(y0.raw(), &mut actual0);
    backend.copy_d2h(y1.raw(), &mut actual1);
    let expected0 = [
        values[0] * values[4] + values[1] * values[5],
        values[2] * values[4] + values[3] * values[5],
    ];
    let expected1 = [
        values[0] * values[6] + values[1] * values[7],
        values[2] * values[6] + values[3] * values[7],
    ];
    assert_close(&actual0, &expected0, tolerance);
    assert_close(&actual1, &expected1, tolerance);
}

fn run_gemm<T>(values: [T; 12], tolerance: f64)
where
    T: DeviceScalar,
    T::Real: ToPrimitive,
{
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 16)).expect("pool");
    let backend = pool.backend();
    let mut a0 = pool.try_alloc::<T>(4).expect("a0");
    let mut a1 = pool.try_alloc::<T>(4).expect("a1");
    let mut b0 = pool.try_alloc::<T>(4).expect("b0");
    let mut b1 = pool.try_alloc::<T>(4).expect("b1");
    let mut c0 = pool.try_alloc::<T>(4).expect("c0");
    let mut c1 = pool.try_alloc::<T>(4).expect("c1");
    backend.copy_h2d(&values[0..4], a0.raw_mut());
    backend.copy_h2d(&values[0..4], a1.raw_mut());
    backend.copy_h2d(&values[4..8], b0.raw_mut());
    backend.copy_h2d(&values[4..8], b1.raw_mut());
    backend.copy_h2d(&values[8..12], c0.raw_mut());
    backend.copy_h2d(&values[8..12], c1.raw_mut());

    let mut batch = [
        GemmBatch {
            transa: Transpose::NoTrans,
            transb: Transpose::NoTrans,
            m: 2,
            n: 2,
            k: 2,
            alpha: T::one(),
            a: a0.raw(),
            lda: 2,
            b: b0.raw(),
            ldb: 2,
            beta: T::zero(),
            c: c0.raw_mut(),
            ldc: 2,
        },
        GemmBatch {
            transa: Transpose::NoTrans,
            transb: Transpose::NoTrans,
            m: 2,
            n: 2,
            k: 2,
            alpha: T::one(),
            a: a1.raw(),
            lda: 2,
            b: b1.raw(),
            ldb: 2,
            beta: T::zero(),
            c: c1.raw_mut(),
            ldc: 2,
        },
    ];
    backend.try_gemm_batched(&mut batch).expect("batched GEMM");

    let mut actual0 = [T::zero(); 4];
    let mut actual1 = [T::zero(); 4];
    backend.copy_d2h(c0.raw(), &mut actual0);
    backend.copy_d2h(c1.raw(), &mut actual1);
    let expected = [
        values[0] * values[4] + values[1] * values[6],
        values[0] * values[5] + values[1] * values[7],
        values[2] * values[4] + values[3] * values[6],
        values[2] * values[5] + values[3] * values[7],
    ];
    assert_close(&actual0, &expected, tolerance);
    assert_close(&actual1, &expected, tolerance);
}

#[test]
fn cpu_batched_gemv_covers_all_scalar_families() {
    run_gemv([1.0_f32, 2.0, 3.0, 4.0, 0.5, -1.0, 2.0, 1.5], 2.0e-5);
    run_gemv([1.0_f64, 2.0, 3.0, 4.0, 0.5, -1.0, 2.0, 1.5], 2.0e-12);
    run_gemv(
        [
            Complex32::new(1.0, 0.5),
            Complex32::new(2.0, -0.5),
            Complex32::new(3.0, 0.25),
            Complex32::new(4.0, -0.25),
            Complex32::new(0.5, 0.1),
            Complex32::new(-1.0, 0.2),
            Complex32::new(2.0, -0.3),
            Complex32::new(1.5, 0.4),
        ],
        3.0e-5,
    );
    run_gemv(
        [
            Complex64::new(1.0, 0.5),
            Complex64::new(2.0, -0.5),
            Complex64::new(3.0, 0.25),
            Complex64::new(4.0, -0.25),
            Complex64::new(0.5, 0.1),
            Complex64::new(-1.0, 0.2),
            Complex64::new(2.0, -0.3),
            Complex64::new(1.5, 0.4),
        ],
        2.0e-12,
    );
}

#[test]
fn cpu_batched_dense_rejects_malformed_vector_lengths() {
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 12)).expect("pool");
    let backend = pool.backend();
    let a = pool.try_alloc::<f64>(4).expect("a");
    let x = pool.try_alloc::<f64>(1).expect("short x");
    let mut y = pool.try_alloc::<f64>(2).expect("y");
    let mut batch = [GemvBatch {
        op: Transpose::NoTrans,
        m: 2,
        n: 2,
        alpha: 1.0,
        a: a.raw(),
        lda: 2,
        x: x.raw(),
        beta: 0.0,
        y: y.raw_mut(),
    }];
    let error = backend
        .try_gemv_batched(&mut batch)
        .expect_err("short batched x must be rejected");
    assert!(matches!(error, BackendError::InvalidArgument { .. }));
}

#[test]
fn cpu_batched_gemm_covers_all_scalar_families() {
    run_gemm(
        [
            1.0_f32, 2.0, 3.0, 4.0, 0.5, -1.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0,
        ],
        2.0e-5,
    );
    run_gemm(
        [
            1.0_f64, 2.0, 3.0, 4.0, 0.5, -1.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0,
        ],
        2.0e-12,
    );
    run_gemm(
        [
            Complex32::new(1.0, 0.5),
            Complex32::new(2.0, -0.5),
            Complex32::new(3.0, 0.25),
            Complex32::new(4.0, -0.25),
            Complex32::new(0.5, 0.1),
            Complex32::new(-1.0, 0.2),
            Complex32::new(2.0, -0.3),
            Complex32::new(1.5, 0.4),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
        ],
        3.0e-5,
    );
    run_gemm(
        [
            Complex64::new(1.0, 0.5),
            Complex64::new(2.0, -0.5),
            Complex64::new(3.0, 0.25),
            Complex64::new(4.0, -0.25),
            Complex64::new(0.5, 0.1),
            Complex64::new(-1.0, 0.2),
            Complex64::new(2.0, -0.3),
            Complex64::new(1.5, 0.4),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        2.0e-12,
    );
}
