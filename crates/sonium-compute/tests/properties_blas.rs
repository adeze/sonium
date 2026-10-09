//! Seeded dense BLAS property checks.
//!
//! These tests deliberately compare the backend against a tiny row-major
//! reference implementation.  They cover dimensions that are easy for a
//! vectorized kernel to get wrong (tails, rectangular operands, leading
//! dimensions, and conjugate-transpose) without depending on a vendor BLAS.

use num_complex::Complex64;
use num_traits::{FromPrimitive, ToPrimitive};
use rand::{Rng, SeedableRng, rngs::StdRng};
use sonium_compute::{
    Backend, BackendError, CpuBackend, DeviceScalar, MemoryPool, MemoryPoolConfig, Transpose,
};

const SEED: u64 = 0xC0A4_2028_0805_0002;
const TOLERANCE: f64 = 2.0e-12;

fn scalar(rng: &mut StdRng) -> Complex64 {
    Complex64::new(rng.random_range(-2.0..=2.0), rng.random_range(-2.0..=2.0))
}

fn stored_shape(op: Transpose, m: usize, n: usize) -> (usize, usize) {
    match op {
        Transpose::NoTrans => (m, n),
        Transpose::Trans | Transpose::ConjTrans => (n, m),
    }
}

fn get_matrix(data: &[Complex64], lda: usize, op: Transpose, row: usize, col: usize) -> Complex64 {
    let (stored_row, stored_col) = match op {
        Transpose::NoTrans => (row, col),
        Transpose::Trans | Transpose::ConjTrans => (col, row),
    };
    let value = data[stored_row * lda + stored_col];
    if matches!(op, Transpose::ConjTrans) {
        value.conj()
    } else {
        value
    }
}

fn assert_close(actual: &[Complex64], expected: &[Complex64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let scale = expected.norm().max(1.0);
        let error = (*actual - *expected).norm();
        assert!(
            error <= TOLERANCE * scale,
            "{label}[{index}] mismatch: actual={actual}, expected={expected}, error={error}, tolerance={}",
            TOLERANCE * scale
        );
    }
}

trait ConjugateValue: Copy {
    fn conjugate(self) -> Self;
}

impl ConjugateValue for f32 {
    fn conjugate(self) -> Self {
        self
    }
}

impl ConjugateValue for f64 {
    fn conjugate(self) -> Self {
        self
    }
}

impl ConjugateValue for num_complex::Complex32 {
    fn conjugate(self) -> Self {
        self.conj()
    }
}

impl ConjugateValue for Complex64 {
    fn conjugate(self) -> Self {
        self.conj()
    }
}

fn get_matrix_generic<T: ConjugateValue>(
    data: &[T],
    lda: usize,
    op: Transpose,
    row: usize,
    col: usize,
) -> T {
    let (stored_row, stored_col) = match op {
        Transpose::NoTrans => (row, col),
        Transpose::Trans | Transpose::ConjTrans => (col, row),
    };
    let value = data[stored_row * lda + stored_col];
    if matches!(op, Transpose::ConjTrans) {
        value.conjugate()
    } else {
        value
    }
}

fn assert_close_generic<T>(actual: &[T], expected: &[T], tolerance: f64, label: &str)
where
    T: DeviceScalar,
    T::Real: ToPrimitive,
{
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (*actual - *expected)
            .norm_sqr()
            .to_f64()
            .expect("real scalar converts to f64")
            .sqrt();
        let scale = expected
            .norm_sqr()
            .to_f64()
            .expect("real scalar converts to f64")
            .sqrt()
            .max(1.0);
        assert!(
            error <= tolerance * scale,
            "{label}[{index}] mismatch: error={error}, tolerance={}",
            tolerance * scale
        );
    }
}

fn generic_scalar_matrix_cases<T, F>(make_scalar: F, tolerance: f64, label: &str)
where
    T: DeviceScalar + ConjugateValue,
    T::Real: ToPrimitive,
    F: Fn(usize) -> T + Copy,
{
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let backend = pool.backend();

    for case in 0..24 {
        let m = 1 + (case * 5 % 8);
        let n = 1 + (case * 7 % 7);
        let k = 1 + (case * 3 % 6);
        let transa = match case % 3 {
            0 => Transpose::NoTrans,
            1 => Transpose::Trans,
            _ => Transpose::ConjTrans,
        };
        let transb = match (case / 3) % 3 {
            0 => Transpose::NoTrans,
            1 => Transpose::Trans,
            _ => Transpose::ConjTrans,
        };

        let (a_rows, a_cols) = stored_shape(transa, m, k);
        let (b_rows, b_cols) = stored_shape(transb, k, n);
        let lda = a_cols + case % 2;
        let ldb = b_cols + (case + 1) % 2;
        let ldc = n + case % 3;
        let a_host: Vec<_> = (0..a_rows * lda).map(make_scalar).collect();
        let b_host: Vec<_> = (0..b_rows * ldb)
            .map(|index| make_scalar(index + 2000))
            .collect();
        let c_initial: Vec<_> = (0..m * ldc)
            .map(|index| make_scalar(index + 4000))
            .collect();
        let alpha = make_scalar(6000 + case);
        let beta = make_scalar(7000 + case);

        let mut a = pool.try_alloc::<T>(a_host.len()).unwrap();
        let mut b = pool.try_alloc::<T>(b_host.len()).unwrap();
        let mut c = pool.try_alloc::<T>(c_initial.len()).unwrap();
        backend.copy_h2d(&a_host, a.raw_mut());
        backend.copy_h2d(&b_host, b.raw_mut());
        backend.copy_h2d(&c_initial, c.raw_mut());
        backend
            .try_gemm(
                transa,
                transb,
                m,
                n,
                k,
                alpha,
                a.raw(),
                lda,
                b.raw(),
                ldb,
                beta,
                c.raw_mut(),
                ldc,
            )
            .unwrap_or_else(|error| panic!("{label} GEMM case {case} rejected: {error}"));

        let mut actual = vec![T::zero(); c_initial.len()];
        backend.copy_d2h(c.raw(), &mut actual);
        let mut expected = c_initial;
        for row in 0..m {
            for col in 0..n {
                let product = (0..k).fold(T::zero(), |sum, inner| {
                    sum + get_matrix_generic(&a_host, lda, transa, row, inner)
                        * get_matrix_generic(&b_host, ldb, transb, inner, col)
                });
                expected[row * ldc + col] = alpha * product + beta * expected[row * ldc + col];
            }
        }
        assert_close_generic(
            &actual,
            &expected,
            tolerance,
            &format!("{label} GEMM case {case}"),
        );

        let (stored_rows, stored_cols) = stored_shape(transa, m, n);
        let lda = stored_cols + case % 2;
        let a_host: Vec<_> = (0..stored_rows * lda).map(make_scalar).collect();
        let x_host: Vec<_> = (0..n).map(|index| make_scalar(index + 8000)).collect();
        let y_initial: Vec<_> = (0..m).map(|index| make_scalar(index + 9000)).collect();
        let alpha = make_scalar(10_000 + case);
        let beta = make_scalar(11_000 + case);
        let mut a = pool.try_alloc::<T>(a_host.len()).unwrap();
        let mut x = pool.try_alloc::<T>(x_host.len()).unwrap();
        let mut y = pool.try_alloc::<T>(y_initial.len()).unwrap();
        backend.copy_h2d(&a_host, a.raw_mut());
        backend.copy_h2d(&x_host, x.raw_mut());
        backend.copy_h2d(&y_initial, y.raw_mut());
        backend
            .try_gemv(
                transa,
                m,
                n,
                alpha,
                a.raw(),
                lda,
                x.raw(),
                beta,
                y.raw_mut(),
            )
            .unwrap_or_else(|error| panic!("{label} GEMV case {case} rejected: {error}"));
        let mut actual = vec![T::zero(); m];
        backend.copy_d2h(y.raw(), &mut actual);
        let expected: Vec<_> = (0..m)
            .map(|row| {
                let product = (0..n).fold(T::zero(), |sum, col| {
                    sum + get_matrix_generic(&a_host, lda, transa, row, col) * x_host[col]
                });
                alpha * product + beta * y_initial[row]
            })
            .collect();
        assert_close_generic(
            &actual,
            &expected,
            tolerance,
            &format!("{label} GEMV case {case}"),
        );
    }
}

fn generic_edge_shape_cases<T, F>(make_scalar: F, tolerance: f64, label: &str)
where
    T: DeviceScalar + ConjugateValue,
    T::Real: ToPrimitive,
    F: Fn(usize) -> T + Copy,
{
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let backend = pool.backend();
    // Include zero extents, unit/tail sizes, and both skinny orientations.
    // Every case is exercised under all transpose/conjugate-transpose modes.
    let shapes = [
        (0, 0, 0),
        (0, 1, 3),
        (1, 0, 4),
        (1, 4, 0),
        (1, 1, 1),
        (2, 3, 5),
        (3, 5, 2),
        (7, 1, 9),
        (5, 9, 3),
    ];
    let modes = [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans];

    for (case, &(m, n, k)) in shapes.iter().enumerate() {
        let alpha = match case % 3 {
            0 => T::zero(),
            1 => T::one(),
            _ => -T::one(),
        };
        let beta = match case % 3 {
            0 => T::one(),
            1 => T::zero(),
            _ => T::from_real(T::Real::from_f64(0.5).unwrap()),
        };

        for &transa in &modes {
            for &transb in &modes {
                let (a_rows, a_cols) = stored_shape(transa, m, k);
                let (b_rows, b_cols) = stored_shape(transb, k, n);
                let lda = a_cols + usize::from(a_cols != 0) * (case % 2);
                let ldb = b_cols + usize::from(b_cols != 0) * ((case + 1) % 2);
                let ldc = n + usize::from(n != 0) * (case % 3);
                let a_host: Vec<_> = (0..a_rows * lda).map(make_scalar).collect();
                let b_host: Vec<_> = (0..b_rows * ldb)
                    .map(|index| make_scalar(index + 20_000 + case * 101))
                    .collect();
                let c_initial: Vec<_> = (0..m * ldc)
                    .map(|index| make_scalar(index + 40_000 + case * 101))
                    .collect();

                let mut a = pool.try_alloc::<T>(a_host.len()).unwrap();
                let mut b = pool.try_alloc::<T>(b_host.len()).unwrap();
                let mut c = pool.try_alloc::<T>(c_initial.len()).unwrap();
                backend.copy_h2d(&a_host, a.raw_mut());
                backend.copy_h2d(&b_host, b.raw_mut());
                backend.copy_h2d(&c_initial, c.raw_mut());
                backend
                    .try_gemm(
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha,
                        a.raw(),
                        lda,
                        b.raw(),
                        ldb,
                        beta,
                        c.raw_mut(),
                        ldc,
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "{label} GEMM edge case {case} ({m},{n},{k}) {transa:?}/{transb:?} rejected: {error}"
                        )
                    });

                let mut actual = vec![T::zero(); c_initial.len()];
                backend.copy_d2h(c.raw(), &mut actual);
                let mut expected = c_initial.clone();
                for row in 0..m {
                    for col in 0..n {
                        let product = (0..k).fold(T::zero(), |sum, inner| {
                            sum + get_matrix_generic(&a_host, lda, transa, row, inner)
                                * get_matrix_generic(&b_host, ldb, transb, inner, col)
                        });
                        expected[row * ldc + col] =
                            alpha * product + beta * expected[row * ldc + col];
                    }
                }
                assert_close_generic(
                    &actual,
                    &expected,
                    tolerance,
                    &format!("{label} GEMM edge case {case} {transa:?}/{transb:?}"),
                );
            }
        }
    }
}

#[test]
fn generic_blas_property_matrix_covers_all_device_scalar_types() {
    generic_scalar_matrix_cases::<f32, _>(
        |index| (index as f32 * 0.17).sin() * 0.75,
        2.0e-5,
        "f32",
    );
    generic_scalar_matrix_cases::<f64, _>(
        |index| (index as f64 * 0.17).sin() * 0.75,
        2.0e-12,
        "f64",
    );
    generic_scalar_matrix_cases::<num_complex::Complex32, _>(
        |index| {
            num_complex::Complex32::new(
                (index as f32 * 0.17).sin() * 0.75,
                (index as f32 * 0.11).cos() * 0.5,
            )
        },
        3.0e-5,
        "complex32",
    );
    generic_scalar_matrix_cases::<Complex64, _>(
        |index| {
            Complex64::new(
                (index as f64 * 0.17).sin() * 0.75,
                (index as f64 * 0.11).cos() * 0.5,
            )
        },
        2.0e-12,
        "complex64",
    );
    generic_edge_shape_cases::<f32, _>(
        |index| (index as f32 * 0.13).sin() * 0.5,
        2.0e-5,
        "f32-edge",
    );
    generic_edge_shape_cases::<f64, _>(
        |index| (index as f64 * 0.13).sin() * 0.5,
        2.0e-12,
        "f64-edge",
    );
    generic_edge_shape_cases::<num_complex::Complex32, _>(
        |index| {
            num_complex::Complex32::new(
                (index as f32 * 0.13).sin() * 0.5,
                (index as f32 * 0.07).cos() * 0.25,
            )
        },
        3.0e-5,
        "complex32-edge",
    );
    generic_edge_shape_cases::<Complex64, _>(
        |index| {
            Complex64::new(
                (index as f64 * 0.13).sin() * 0.5,
                (index as f64 * 0.07).cos() * 0.25,
            )
        },
        2.0e-12,
        "complex64-edge",
    );
}

#[test]
fn complex_gemv_satisfies_hermitian_inner_product_identity() {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x00A1_1D01);
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let backend = pool.backend();
    for case in 0..20 {
        let m = 1 + (case * 5 % 9);
        let n = 1 + (case * 7 % 8);
        let lda = n + case % 3;
        let matrix: Vec<_> = (0..m * lda).map(|_| scalar(&mut rng)).collect();
        let x: Vec<_> = (0..n).map(|_| scalar(&mut rng)).collect();
        let y: Vec<_> = (0..m).map(|_| scalar(&mut rng)).collect();
        let mut a = pool.try_alloc::<Complex64>(matrix.len()).unwrap();
        let mut x_buffer = pool.try_alloc::<Complex64>(x.len()).unwrap();
        let mut y_buffer = pool.try_alloc::<Complex64>(y.len()).unwrap();
        backend.copy_h2d(&matrix, a.raw_mut());
        backend.copy_h2d(&x, x_buffer.raw_mut());
        backend.copy_h2d(&y, y_buffer.raw_mut());

        let mut ax_buffer = pool.try_alloc::<Complex64>(m).unwrap();
        backend
            .try_gemv(
                Transpose::NoTrans,
                m,
                n,
                Complex64::new(1.0, 0.0),
                a.raw(),
                lda,
                x_buffer.raw(),
                Complex64::new(0.0, 0.0),
                ax_buffer.raw_mut(),
            )
            .unwrap();
        let mut ahy_buffer = pool.try_alloc::<Complex64>(n).unwrap();
        backend
            .try_gemv(
                Transpose::ConjTrans,
                n,
                m,
                Complex64::new(1.0, 0.0),
                a.raw(),
                lda,
                y_buffer.raw(),
                Complex64::new(0.0, 0.0),
                ahy_buffer.raw_mut(),
            )
            .unwrap();
        let mut ax = vec![Complex64::default(); m];
        let mut ahy = vec![Complex64::default(); n];
        backend.copy_d2h(ax_buffer.raw(), &mut ax);
        backend.copy_d2h(ahy_buffer.raw(), &mut ahy);
        let lhs = ax
            .iter()
            .zip(&y)
            .map(|(value, test)| value.conj() * test)
            .sum::<Complex64>();
        let rhs = x
            .iter()
            .zip(&ahy)
            .map(|(value, test)| value.conj() * test)
            .sum::<Complex64>();
        assert!(
            (lhs - rhs).norm() <= 3.0e-12 * lhs.norm().max(rhs.norm()).max(1.0),
            "case {case}: <Ax,y>={lhs}, <x,Aᴴy>={rhs}"
        );
    }
}

#[test]
fn gemv_matches_reference_for_all_transpose_modes_and_tails() {
    let mut rng = StdRng::seed_from_u64(SEED);
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let backend = pool.backend();

    for case in 0..48 {
        let m = 1 + (case * 5 % 9);
        let n = 1 + (case * 7 % 8);
        let op = match case % 3 {
            0 => Transpose::NoTrans,
            1 => Transpose::Trans,
            _ => Transpose::ConjTrans,
        };
        let (stored_rows, stored_cols) = stored_shape(op, m, n);
        let lda = stored_cols + (case % 3);
        let a_host: Vec<_> = (0..stored_rows * lda).map(|_| scalar(&mut rng)).collect();
        let x_host: Vec<_> = (0..n).map(|_| scalar(&mut rng)).collect();
        let y_initial: Vec<_> = (0..m).map(|_| scalar(&mut rng)).collect();
        let alpha = scalar(&mut rng);
        let beta = scalar(&mut rng);

        let mut a = pool.try_alloc::<Complex64>(a_host.len()).unwrap();
        let mut x = pool.try_alloc::<Complex64>(x_host.len()).unwrap();
        let mut y = pool.try_alloc::<Complex64>(y_initial.len()).unwrap();
        backend.copy_h2d(&a_host, a.raw_mut());
        backend.copy_h2d(&x_host, x.raw_mut());
        backend.copy_h2d(&y_initial, y.raw_mut());
        backend
            .try_gemv(op, m, n, alpha, a.raw(), lda, x.raw(), beta, y.raw_mut())
            .unwrap_or_else(|error| panic!("case {case} rejected valid GEMV: {error}"));

        let mut actual = vec![Complex64::default(); m];
        backend.copy_d2h(y.raw(), &mut actual);
        let expected: Vec<_> = (0..m)
            .map(|row| {
                let product = (0..n)
                    .map(|col| get_matrix(&a_host, lda, op, row, col) * x_host[col])
                    .sum::<Complex64>();
                alpha * product + beta * y_initial[row]
            })
            .collect();
        assert_close(&actual, &expected, &format!("gemv case {case}"));
    }
}

#[test]
fn gemv_zero_columns_still_applies_beta_epilogue() {
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let backend = pool.backend();

    let mut a = pool.try_alloc::<f64>(0).unwrap();
    let mut x = pool.try_alloc::<f64>(0).unwrap();
    let mut y = pool.try_alloc::<f64>(3).unwrap();
    backend.copy_h2d(&[][..], a.raw_mut());
    backend.copy_h2d(&[][..], x.raw_mut());
    backend.copy_h2d(&[1.0, -2.0, 3.5], y.raw_mut());

    backend
        .try_gemv(
            Transpose::NoTrans,
            3,
            0,
            7.0,
            a.raw(),
            0,
            x.raw(),
            -2.0,
            y.raw_mut(),
        )
        .unwrap();

    let mut output = [0.0; 3];
    backend.copy_d2h(y.raw(), &mut output);
    assert_eq!(output, [-2.0, 4.0, -7.0]);

    let mut ac = pool.try_alloc::<Complex64>(0).unwrap();
    let mut xc = pool.try_alloc::<Complex64>(0).unwrap();
    let mut yc = pool.try_alloc::<Complex64>(2).unwrap();
    backend.copy_h2d(&[][..], ac.raw_mut());
    backend.copy_h2d(&[][..], xc.raw_mut());
    backend.copy_h2d(
        &[Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
        yc.raw_mut(),
    );
    backend
        .try_gemv(
            Transpose::NoTrans,
            2,
            0,
            Complex64::new(4.0, -1.0),
            ac.raw(),
            0,
            xc.raw(),
            Complex64::new(0.5, 2.0),
            yc.raw_mut(),
        )
        .unwrap();
    let mut output_complex = [Complex64::default(); 2];
    backend.copy_d2h(yc.raw(), &mut output_complex);
    assert_eq!(
        output_complex,
        [Complex64::new(-3.5, 3.0), Complex64::new(-2.5, -5.75),]
    );
}

#[test]
fn gemm_matches_reference_for_rectangular_transpose_combinations() {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x55AA);
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let backend = pool.backend();

    for case in 0..36 {
        let m = 1 + (case * 3 % 7);
        let n = 1 + (case * 5 % 8);
        let k = 1 + (case * 7 % 6);
        let transa = match case % 3 {
            0 => Transpose::NoTrans,
            1 => Transpose::Trans,
            _ => Transpose::ConjTrans,
        };
        let transb = match (case / 3) % 3 {
            0 => Transpose::NoTrans,
            1 => Transpose::Trans,
            _ => Transpose::ConjTrans,
        };
        let (a_rows, a_cols) = stored_shape(transa, m, k);
        let (b_rows, b_cols) = stored_shape(transb, k, n);
        let lda = a_cols + case % 2;
        let ldb = b_cols + (case + 1) % 2;
        let ldc = n + case % 3;
        let a_host: Vec<_> = (0..a_rows * lda).map(|_| scalar(&mut rng)).collect();
        let b_host: Vec<_> = (0..b_rows * ldb).map(|_| scalar(&mut rng)).collect();
        let c_initial: Vec<_> = (0..m * ldc).map(|_| scalar(&mut rng)).collect();
        let alpha = scalar(&mut rng);
        let beta = scalar(&mut rng);

        let mut a = pool.try_alloc::<Complex64>(a_host.len()).unwrap();
        let mut b = pool.try_alloc::<Complex64>(b_host.len()).unwrap();
        let mut c = pool.try_alloc::<Complex64>(c_initial.len()).unwrap();
        backend.copy_h2d(&a_host, a.raw_mut());
        backend.copy_h2d(&b_host, b.raw_mut());
        backend.copy_h2d(&c_initial, c.raw_mut());
        backend
            .try_gemm(
                transa,
                transb,
                m,
                n,
                k,
                alpha,
                a.raw(),
                lda,
                b.raw(),
                ldb,
                beta,
                c.raw_mut(),
                ldc,
            )
            .unwrap_or_else(|error| panic!("case {case} rejected valid GEMM: {error}"));

        let mut actual = vec![Complex64::default(); c_initial.len()];
        backend.copy_d2h(c.raw(), &mut actual);
        let mut expected = c_initial;
        for row in 0..m {
            for col in 0..n {
                let product = (0..k)
                    .map(|inner| {
                        get_matrix(&a_host, lda, transa, row, inner)
                            * get_matrix(&b_host, ldb, transb, inner, col)
                    })
                    .sum::<Complex64>();
                expected[row * ldc + col] = alpha * product + beta * expected[row * ldc + col];
            }
        }
        assert_close(&actual, &expected, &format!("gemm case {case}"));
    }
}

#[test]
fn zero_extent_dense_products_are_noops() {
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let backend = pool.backend();
    let a = pool.try_alloc::<Complex64>(0).unwrap();
    let x = pool.try_alloc::<Complex64>(0).unwrap();
    let mut y = pool.try_alloc::<Complex64>(0).unwrap();
    backend
        .try_gemv(
            Transpose::NoTrans,
            0,
            0,
            Complex64::new(2.0, 0.0),
            a.raw(),
            0,
            x.raw(),
            Complex64::new(1.0, 0.0),
            y.raw_mut(),
        )
        .unwrap();
    assert!(y.is_empty());
}

#[test]
fn checked_dense_contract_rejects_invalid_layout_and_nonfinite_scalars() {
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let backend = pool.backend();
    let a = pool.try_alloc::<f64>(6).unwrap();
    let x = pool.try_alloc::<f64>(3).unwrap();
    let mut y = pool.try_alloc::<f64>(2).unwrap();

    let invalid_layout = backend.try_gemv(
        Transpose::NoTrans,
        2,
        3,
        1.0,
        a.raw(),
        2,
        x.raw(),
        0.0,
        y.raw_mut(),
    );
    assert!(matches!(
        invalid_layout,
        Err(BackendError::InvalidArgument {
            operation: "gemv",
            ..
        })
    ));

    let nonfinite_scalar = backend.try_gemv(
        Transpose::NoTrans,
        2,
        3,
        f64::NAN,
        a.raw(),
        3,
        x.raw(),
        0.0,
        y.raw_mut(),
    );
    assert!(matches!(
        nonfinite_scalar,
        Err(BackendError::OperationFailed {
            operation: "gemv",
            ..
        })
    ));
}

#[test]
fn blas1_reference_checks_cover_axpy_scale_dot_and_nrm2() {
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let backend = pool.backend();
    let x_host = vec![
        Complex64::new(1.0, -2.0),
        Complex64::new(-0.5, 0.25),
        Complex64::new(2.0, 1.5),
        Complex64::new(-1.0, 0.75),
    ];
    let y_initial = vec![
        Complex64::new(0.25, 1.0),
        Complex64::new(1.0, -0.5),
        Complex64::new(-2.0, 0.5),
        Complex64::new(0.0, -1.0),
    ];
    let alpha = Complex64::new(0.75, -0.25);
    let beta = Complex64::new(-0.5, 0.125);
    let mut x = pool.try_alloc::<Complex64>(x_host.len()).unwrap();
    let mut y = pool.try_alloc::<Complex64>(y_initial.len()).unwrap();
    backend.copy_h2d(&x_host, x.raw_mut());
    backend.copy_h2d(&y_initial, y.raw_mut());

    backend.try_axpy(alpha, x.raw(), y.raw_mut()).unwrap();
    let mut actual = vec![Complex64::new(0.0, 0.0); y_initial.len()];
    backend.copy_d2h(y.raw(), &mut actual);
    let expected: Vec<_> = x_host
        .iter()
        .zip(&y_initial)
        .map(|(x, y)| alpha * *x + *y)
        .collect();
    assert_close(&actual, &expected, "complex axpy");

    backend.try_scale(beta, x.raw_mut()).unwrap();
    let mut scaled = vec![Complex64::new(0.0, 0.0); x_host.len()];
    backend.copy_d2h(x.raw(), &mut scaled);
    let expected_scaled: Vec<_> = x_host.iter().map(|value| beta * *value).collect();
    assert_close(&scaled, &expected_scaled, "complex scale");

    let dot = backend.try_dot_deterministic(x.raw(), y.raw()).unwrap();
    let expected_dot = expected_scaled
        .iter()
        .zip(&expected)
        .map(|(x, y)| x.conj() * *y)
        .sum::<Complex64>();
    assert!((dot - expected_dot).norm() <= TOLERANCE);

    let norm = backend.try_nrm2_deterministic(x.raw()).unwrap();
    let expected_norm = expected_scaled
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!((norm - expected_norm).abs() <= TOLERANCE);
}
