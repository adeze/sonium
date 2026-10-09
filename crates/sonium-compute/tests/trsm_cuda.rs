//! Native CUDA acceptance coverage for device-resident triangular solves.
//!
//! Skip-capable on hosts without an NVIDIA driver, mirroring
//! `tests/device_preconditioner_cuda.rs`; `SONIUM_REQUIRE_CUDA=1` turns
//! the skip into a failure.

#![cfg(feature = "cuda")]

use num_complex::{Complex32, Complex64};
use sonium_compute::cpu::blas3::trsm::{
    TrsmSide, TrsmUplo, trsm_c32, trsm_c64, trsm_f32, trsm_f64,
};
use sonium_compute::{Backend, CudaBackend, Transpose};

fn cuda_or_skip() -> Option<CudaBackend> {
    match CudaBackend::new_default() {
        Ok(backend) => Some(backend),
        Err(error) if std::env::var("SONIUM_REQUIRE_CUDA").is_ok() => {
            panic!("CUDA required by SONIUM_REQUIRE_CUDA but unavailable: {error}")
        }
        Err(error) => {
            eprintln!("CUDA unavailable, skipping native TRSM acceptance: {error}");
            None
        }
    }
}

/// Triangular fixture with a dominant diagonal (nonsingular); only the
/// referenced triangle is filled, matching what each solve reads.
fn tri_fixture_f64(uplo: TrsmUplo, adim: usize, lda: usize) -> Vec<f64> {
    let mut a = vec![0.0; adim * lda];
    for i in 0..adim {
        for j in 0..adim {
            let in_tri = match uplo {
                TrsmUplo::Lower => j <= i,
                TrsmUplo::Upper => j >= i,
            };
            if in_tri {
                a[i * lda + j] = if i == j {
                    2.0 + (i as f64 * 0.01).sin()
                } else {
                    ((i * lda + j) as f64 * 0.001) % 1.0
                };
            }
        }
    }
    a
}

fn ramp_f64(len: usize) -> Vec<f64> {
    (0..len).map(|i| (i as f64 * 0.0123).sin()).collect()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn native_cuda_dtrsm_matches_cpu_all_variants() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    // (m, n, unit, alpha): Left shapes plus one Right shape (A is n x n).
    for (m, n, unit, alpha) in [
        (70usize, 65usize, false, 1.0),
        (65, 70, false, 1.0),
        (70, 65, true, 0.5),
        (65, 70, true, 2.0),
    ] {
        for side in [TrsmSide::Left, TrsmSide::Right] {
            for uplo in [TrsmUplo::Lower, TrsmUplo::Upper] {
                for trans in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
                    let adim = match side {
                        TrsmSide::Left => m,
                        TrsmSide::Right => n,
                    };
                    let a = tri_fixture_f64(uplo, adim, adim);
                    let b0 = ramp_f64(m * n);
                    let mut want = b0.clone();
                    trsm_f64(side, uplo, trans, unit, m, n, alpha, &a, adim, &mut want, n);

                    let mut a_dev = cuda.raw_alloc::<f64>(a.len()).expect("a alloc");
                    let mut b_dev = cuda.raw_alloc::<f64>(b0.len()).expect("b alloc");
                    cuda.try_copy_h2d(&a, &mut a_dev).expect("a upload");
                    cuda.try_copy_h2d(&b0, &mut b_dev).expect("b upload");
                    cuda.try_trsm(
                        side, uplo, trans, unit, m, n, alpha, &a_dev, adim, &mut b_dev, n,
                    )
                    .expect("native dtrsm");
                    let mut got = vec![0.0; b0.len()];
                    cuda.try_copy_d2h(&b_dev, &mut got).expect("b download");

                    let err = max_abs_diff(&got, &want);
                    let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1.0);
                    assert!(
                        err < 1e-9 * scale,
                        "dtrsm mismatch side={side:?} uplo={uplo:?} trans={trans:?} unit={unit} \
                         m={m} n={n} alpha={alpha}: {err:e} (scale {scale:e})"
                    );
                }
            }
        }
    }
}

#[test]
fn native_cuda_trsm_spot_checks_other_types() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    // f32, one variant per axis is enough to prove the TypeId dispatch
    // reaches the right wrapper with the right swap mapping.
    {
        let (m, n) = (70usize, 65usize);
        let a: Vec<f32> = tri_fixture_f64(TrsmUplo::Lower, m, m)
            .iter()
            .map(|&v| v as f32)
            .collect();
        let b0: Vec<f32> = ramp_f64(m * n).iter().map(|&v| v as f32).collect();
        let mut want = b0.clone();
        trsm_f32(
            TrsmSide::Left,
            TrsmUplo::Lower,
            Transpose::Trans,
            false,
            m,
            n,
            0.5,
            &a,
            m,
            &mut want,
            n,
        );
        let mut a_dev = cuda.raw_alloc::<f32>(a.len()).expect("a alloc");
        let mut b_dev = cuda.raw_alloc::<f32>(b0.len()).expect("b alloc");
        cuda.try_copy_h2d(&a, &mut a_dev).expect("upload");
        cuda.try_copy_h2d(&b0, &mut b_dev).expect("upload");
        cuda.try_trsm(
            TrsmSide::Left,
            TrsmUplo::Lower,
            Transpose::Trans,
            false,
            m,
            n,
            0.5,
            &a_dev,
            m,
            &mut b_dev,
            n,
        )
        .expect("native strsm");
        let mut got = vec![0.0f32; b0.len()];
        cuda.try_copy_d2h(&b_dev, &mut got).expect("download");
        let err = got
            .iter()
            .zip(want.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(err < 1e-3, "strsm mismatch: {err:e}");
    }
    // Complex64 with ConjTrans exercises the conjugation fold.
    {
        let (m, n) = (40usize, 37usize);
        let a: Vec<Complex64> = (0..m * m)
            .map(|k| Complex64::new((k as f64 * 0.011).sin(), (k as f64 * 0.017).cos()))
            .collect();
        let mut a_tri = vec![Complex64::new(0.0, 0.0); m * m];
        for i in 0..m {
            for j in 0..=i {
                a_tri[i * m + j] = a[i * m + j]
                    + if i == j {
                        Complex64::new(2.0, 0.0)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
            }
        }
        let b0: Vec<Complex64> = (0..m * n)
            .map(|k| Complex64::new((k as f64 * 0.013).cos(), (k as f64 * 0.019).sin()))
            .collect();
        let mut want = b0.clone();
        trsm_c64(
            TrsmSide::Left,
            TrsmUplo::Lower,
            Transpose::ConjTrans,
            false,
            m,
            n,
            Complex64::new(1.0, 0.0),
            &a_tri,
            m,
            &mut want,
            n,
        );
        let mut a_dev = cuda.raw_alloc::<Complex64>(a_tri.len()).expect("a alloc");
        let mut b_dev = cuda.raw_alloc::<Complex64>(b0.len()).expect("b alloc");
        cuda.try_copy_h2d(&a_tri, &mut a_dev).expect("upload");
        cuda.try_copy_h2d(&b0, &mut b_dev).expect("upload");
        cuda.try_trsm(
            TrsmSide::Left,
            TrsmUplo::Lower,
            Transpose::ConjTrans,
            false,
            m,
            n,
            Complex64::new(1.0, 0.0),
            &a_dev,
            m,
            &mut b_dev,
            n,
        )
        .expect("native ztrsm");
        let mut got = vec![Complex64::new(0.0, 0.0); b0.len()];
        cuda.try_copy_d2h(&b_dev, &mut got).expect("download");
        let err = got
            .iter()
            .zip(want.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max);
        assert!(err < 1e-9, "ztrsm mismatch: {err:e}");
    }
    // Complex32 right-side solve proves the Right mapping on complex.
    {
        let (m, n) = (37usize, 40usize);
        let a: Vec<Complex32> = (0..n * n)
            .map(|k| Complex32::new((k as f32 * 0.011).sin(), (k as f32 * 0.017).cos()))
            .collect();
        let mut a_tri = vec![Complex32::new(0.0, 0.0); n * n];
        for i in 0..n {
            for j in i..n {
                a_tri[i * n + j] = a[i * n + j]
                    + if i == j {
                        Complex32::new(2.0, 0.0)
                    } else {
                        Complex32::new(0.0, 0.0)
                    };
            }
        }
        let b0: Vec<Complex32> = (0..m * n)
            .map(|k| Complex32::new((k as f32 * 0.013).cos(), (k as f32 * 0.019).sin()))
            .collect();
        let mut want = b0.clone();
        trsm_c32(
            TrsmSide::Right,
            TrsmUplo::Upper,
            Transpose::NoTrans,
            false,
            m,
            n,
            Complex32::new(1.0, 0.0),
            &a_tri,
            n,
            &mut want,
            n,
        );
        let mut a_dev = cuda.raw_alloc::<Complex32>(a_tri.len()).expect("a alloc");
        let mut b_dev = cuda.raw_alloc::<Complex32>(b0.len()).expect("b alloc");
        cuda.try_copy_h2d(&a_tri, &mut a_dev).expect("upload");
        cuda.try_copy_h2d(&b0, &mut b_dev).expect("upload");
        cuda.try_trsm(
            TrsmSide::Right,
            TrsmUplo::Upper,
            Transpose::NoTrans,
            false,
            m,
            n,
            Complex32::new(1.0, 0.0),
            &a_dev,
            n,
            &mut b_dev,
            n,
        )
        .expect("native ctrsm");
        let mut got = vec![Complex32::new(0.0, 0.0); b0.len()];
        cuda.try_copy_d2h(&b_dev, &mut got).expect("download");
        let err = got
            .iter()
            .zip(want.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f32::max);
        assert!(err < 1e-3, "ctrsm mismatch: {err:e}");
    }
}

#[test]
fn native_cuda_trsm_rejects_bad_dims() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let a_dev = cuda.raw_alloc::<f64>(70 * 70).expect("a alloc");
    let mut b_dev = cuda.raw_alloc::<f64>(70 * 65).expect("b alloc");
    let err = cuda
        .try_trsm(
            TrsmSide::Left,
            TrsmUplo::Lower,
            Transpose::NoTrans,
            false,
            70,
            65,
            1.0,
            &a_dev,
            32,
            &mut b_dev,
            65,
        )
        .expect_err("lda < adim must be rejected");
    assert!(
        format!("{err}").contains("leading dimension"),
        "unexpected error: {err}"
    );
}
