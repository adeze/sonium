//! Native CUDA acceptance coverage for device-resident symmetric rank-k
//! updates.
//!
//! Skip-capable on hosts without an NVIDIA driver, mirroring
//! `tests/device_preconditioner_cuda.rs`; `SONIUM_REQUIRE_CUDA=1` turns
//! the skip into a failure.

#![cfg(feature = "cuda")]

use num_complex::{Complex32, Complex64};
use sonium_compute::cpu::blas3::syrk::{syrk_c32, syrk_c64, syrk_f32, syrk_f64};
use sonium_compute::cpu::blas3::trsm::TrsmUplo;
use sonium_compute::{Backend, CudaBackend, Transpose};

fn cuda_or_skip() -> Option<CudaBackend> {
    match CudaBackend::new_default() {
        Ok(backend) => Some(backend),
        Err(error) if std::env::var("SONIUM_REQUIRE_CUDA").is_ok() => {
            panic!("CUDA required by SONIUM_REQUIRE_CUDA but unavailable: {error}")
        }
        Err(error) => {
            eprintln!("CUDA unavailable, skipping native SYRK acceptance: {error}");
            None
        }
    }
}

fn ramp_f64(len: usize) -> Vec<f64> {
    (0..len).map(|i| (i as f64 * 0.0123).sin()).collect()
}

#[test]
fn native_cuda_dsyrk_matches_cpu_all_variants() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    // (trans, n, k, alpha, beta): NoTrans (A n×k) and Trans (A k×n),
    // beta 0/1 to cover both the zero-fill and accumulate paths.
    for (trans, n, k, alpha, beta) in [
        (Transpose::NoTrans, 70usize, 65usize, 1.0, 0.0),
        (Transpose::NoTrans, 70, 65, 0.5, 1.0),
        (Transpose::Trans, 65, 70, 1.0, 0.0),
        (Transpose::Trans, 65, 70, 2.0, 0.25),
        (Transpose::ConjTrans, 65, 70, 1.0, 0.0),
    ] {
        for uplo in [TrsmUplo::Lower, TrsmUplo::Upper] {
            let (a_rows, a_cols) = match trans {
                Transpose::NoTrans => (n, k),
                _ => (k, n),
            };
            let a = ramp_f64(a_rows * a_cols);
            let c0 = ramp_f64(n * n);
            let mut want = c0.clone();
            syrk_f64(uplo, trans, n, k, alpha, &a, a_cols, beta, &mut want, n);

            let mut a_dev = cuda.raw_alloc::<f64>(a.len()).expect("a alloc");
            let mut c_dev = cuda.raw_alloc::<f64>(c0.len()).expect("c alloc");
            cuda.try_copy_h2d(&a, &mut a_dev).expect("a upload");
            cuda.try_copy_h2d(&c0, &mut c_dev).expect("c upload");
            cuda.try_syrk(
                uplo, trans, n, k, alpha, &a_dev, a_cols, beta, &mut c_dev, n,
            )
            .expect("native dsyrk");
            let mut got = vec![0.0; c0.len()];
            cuda.try_copy_d2h(&c_dev, &mut got).expect("c download");

            // Only the referenced triangle is defined; the other may
            // differ (cuBLAS leaves it untouched, same as the CPU kernel).
            let mut err = 0.0f64;
            let mut scale = 1.0f64;
            for i in 0..n {
                for j in 0..n {
                    let in_tri = match uplo {
                        TrsmUplo::Lower => j <= i,
                        TrsmUplo::Upper => j >= i,
                    };
                    if in_tri {
                        err = err.max((got[i * n + j] - want[i * n + j]).abs());
                        scale = scale.max(want[i * n + j].abs());
                    }
                }
            }
            assert!(
                err < 1e-9 * scale,
                "dsyrk mismatch uplo={uplo:?} trans={trans:?} n={n} k={k}: {err:e} (scale {scale:e})"
            );
        }
    }
}

#[test]
fn native_cuda_syrk_spot_checks_other_types() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    // f32 Trans exercises the flip mapping in single precision.
    {
        let (n, k) = (65usize, 70usize);
        let a: Vec<f32> = ramp_f64(k * n).iter().map(|&v| v as f32).collect();
        let c0: Vec<f32> = ramp_f64(n * n).iter().map(|&v| v as f32).collect();
        let mut want = c0.clone();
        syrk_f32(
            TrsmUplo::Upper,
            Transpose::Trans,
            n,
            k,
            0.5,
            &a,
            n,
            0.25,
            &mut want,
            n,
        );
        let mut a_dev = cuda.raw_alloc::<f32>(a.len()).expect("alloc");
        let mut c_dev = cuda.raw_alloc::<f32>(c0.len()).expect("alloc");
        cuda.try_copy_h2d(&a, &mut a_dev).expect("upload");
        cuda.try_copy_h2d(&c0, &mut c_dev).expect("upload");
        cuda.try_syrk(
            TrsmUplo::Upper,
            Transpose::Trans,
            n,
            k,
            0.5,
            &a_dev,
            n,
            0.25,
            &mut c_dev,
            n,
        )
        .expect("native ssyrk");
        let mut got = vec![0.0f32; c0.len()];
        cuda.try_copy_d2h(&c_dev, &mut got).expect("download");
        let mut err = 0.0f32;
        for i in 0..n {
            for j in i..n {
                err = err.max((got[i * n + j] - want[i * n + j]).abs());
            }
        }
        assert!(err < 1e-3, "ssyrk mismatch: {err:e}");
    }
    // Complex64 NoTrans (symmetric, no conjugation).
    {
        let (n, k) = (40usize, 37usize);
        let a: Vec<Complex64> = (0..n * k)
            .map(|t| Complex64::new((t as f64 * 0.011).sin(), (t as f64 * 0.017).cos()))
            .collect();
        let c0: Vec<Complex64> = (0..n * n)
            .map(|t| Complex64::new((t as f64 * 0.013).cos(), (t as f64 * 0.019).sin()))
            .collect();
        let mut want = c0.clone();
        syrk_c64(
            TrsmUplo::Lower,
            Transpose::NoTrans,
            n,
            k,
            Complex64::new(0.5, 0.0),
            &a,
            k,
            Complex64::new(0.25, 0.0),
            &mut want,
            n,
        );
        let mut a_dev = cuda.raw_alloc::<Complex64>(a.len()).expect("alloc");
        let mut c_dev = cuda.raw_alloc::<Complex64>(c0.len()).expect("alloc");
        cuda.try_copy_h2d(&a, &mut a_dev).expect("upload");
        cuda.try_copy_h2d(&c0, &mut c_dev).expect("upload");
        cuda.try_syrk(
            TrsmUplo::Lower,
            Transpose::NoTrans,
            n,
            k,
            Complex64::new(0.5, 0.0),
            &a_dev,
            k,
            Complex64::new(0.25, 0.0),
            &mut c_dev,
            n,
        )
        .expect("native zsyrk");
        let mut got = vec![Complex64::new(0.0, 0.0); c0.len()];
        cuda.try_copy_d2h(&c_dev, &mut got).expect("download");
        let mut err = 0.0f64;
        for i in 0..n {
            for j in 0..=i {
                err = err.max((got[i * n + j] - want[i * n + j]).norm());
            }
        }
        assert!(err < 1e-9, "zsyrk mismatch: {err:e}");
    }
    // Complex32 Trans.
    {
        let (n, k) = (37usize, 40usize);
        let a: Vec<Complex32> = (0..k * n)
            .map(|t| Complex32::new((t as f32 * 0.011).sin(), (t as f32 * 0.017).cos()))
            .collect();
        let c0: Vec<Complex32> = (0..n * n)
            .map(|t| Complex32::new((t as f32 * 0.013).cos(), (t as f32 * 0.019).sin()))
            .collect();
        let mut want = c0.clone();
        syrk_c32(
            TrsmUplo::Upper,
            Transpose::Trans,
            n,
            k,
            Complex32::new(1.0, 0.0),
            &a,
            n,
            Complex32::new(0.0, 0.0),
            &mut want,
            n,
        );
        let mut a_dev = cuda.raw_alloc::<Complex32>(a.len()).expect("alloc");
        let mut c_dev = cuda.raw_alloc::<Complex32>(c0.len()).expect("alloc");
        cuda.try_copy_h2d(&a, &mut a_dev).expect("upload");
        cuda.try_copy_h2d(&c0, &mut c_dev).expect("upload");
        cuda.try_syrk(
            TrsmUplo::Upper,
            Transpose::Trans,
            n,
            k,
            Complex32::new(1.0, 0.0),
            &a_dev,
            n,
            Complex32::new(0.0, 0.0),
            &mut c_dev,
            n,
        )
        .expect("native csyrk");
        let mut got = vec![Complex32::new(0.0, 0.0); c0.len()];
        cuda.try_copy_d2h(&c_dev, &mut got).expect("download");
        let mut err = 0.0f32;
        for i in 0..n {
            for j in i..n {
                err = err.max((got[i * n + j] - want[i * n + j]).norm());
            }
        }
        assert!(err < 1e-3, "csyrk mismatch: {err:e}");
    }
}

#[test]
fn native_cuda_syrk_conjtrans_real_herk_matches_cpu() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    // Hermitian update with real scalars delegates to cuBLAS herk. `C0`
    // is Hermitian (real diagonal): herk follows cuBLAS diagonal
    // semantics (diagonal imaginary parts are set to zero), so only a
    // Hermitian `C` matches the CPU kernel there.
    for uplo in [TrsmUplo::Lower, TrsmUplo::Upper] {
        for (alpha, beta) in [(0.5, 0.25), (1.0, 0.0)] {
            let (n, k) = (40usize, 37usize);
            let a: Vec<Complex64> = (0..k * n)
                .map(|t| Complex64::new((t as f64 * 0.011).sin(), (t as f64 * 0.017).cos()))
                .collect();
            let mut c0: Vec<Complex64> = (0..n * n)
                .map(|t| Complex64::new((t as f64 * 0.013).cos(), (t as f64 * 0.019).sin()))
                .collect();
            for i in 0..n {
                c0[i * n + i].im = 0.0;
            }
            let mut want = c0.clone();
            syrk_c64(
                uplo,
                Transpose::ConjTrans,
                n,
                k,
                Complex64::new(alpha, 0.0),
                &a,
                n,
                Complex64::new(beta, 0.0),
                &mut want,
                n,
            );
            let mut a_dev = cuda.raw_alloc::<Complex64>(a.len()).expect("alloc");
            let mut c_dev = cuda.raw_alloc::<Complex64>(c0.len()).expect("alloc");
            cuda.try_copy_h2d(&a, &mut a_dev).expect("upload");
            cuda.try_copy_h2d(&c0, &mut c_dev).expect("upload");
            cuda.try_syrk(
                uplo,
                Transpose::ConjTrans,
                n,
                k,
                Complex64::new(alpha, 0.0),
                &a_dev,
                n,
                Complex64::new(beta, 0.0),
                &mut c_dev,
                n,
            )
            .expect("native zherk");
            let mut got = vec![Complex64::new(0.0, 0.0); c0.len()];
            cuda.try_copy_d2h(&c_dev, &mut got).expect("download");
            let mut err = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    let in_tri = match uplo {
                        TrsmUplo::Lower => j <= i,
                        TrsmUplo::Upper => j >= i,
                    };
                    if in_tri {
                        err = err.max((got[i * n + j] - want[i * n + j]).norm());
                    }
                }
                assert!(
                    got[i * n + i].im == 0.0,
                    "zherk diagonal imag must be zero: {:?}",
                    got[i * n + i]
                );
            }
            assert!(
                err < 1e-9,
                "zherk mismatch uplo={uplo:?} alpha={alpha} beta={beta}: {err:e}"
            );
        }
    }
    // Complex32, Upper, beta != 0 exercises the accumulate path.
    {
        let (n, k) = (37usize, 40usize);
        let a: Vec<Complex32> = (0..k * n)
            .map(|t| Complex32::new((t as f32 * 0.011).sin(), (t as f32 * 0.017).cos()))
            .collect();
        let mut c0: Vec<Complex32> = (0..n * n)
            .map(|t| Complex32::new((t as f32 * 0.013).cos(), (t as f32 * 0.019).sin()))
            .collect();
        for i in 0..n {
            c0[i * n + i].im = 0.0;
        }
        let mut want = c0.clone();
        syrk_c32(
            TrsmUplo::Upper,
            Transpose::ConjTrans,
            n,
            k,
            Complex32::new(0.5, 0.0),
            &a,
            n,
            Complex32::new(0.25, 0.0),
            &mut want,
            n,
        );
        let mut a_dev = cuda.raw_alloc::<Complex32>(a.len()).expect("alloc");
        let mut c_dev = cuda.raw_alloc::<Complex32>(c0.len()).expect("alloc");
        cuda.try_copy_h2d(&a, &mut a_dev).expect("upload");
        cuda.try_copy_h2d(&c0, &mut c_dev).expect("upload");
        cuda.try_syrk(
            TrsmUplo::Upper,
            Transpose::ConjTrans,
            n,
            k,
            Complex32::new(0.5, 0.0),
            &a_dev,
            n,
            Complex32::new(0.25, 0.0),
            &mut c_dev,
            n,
        )
        .expect("native cherk");
        let mut got = vec![Complex32::new(0.0, 0.0); c0.len()];
        cuda.try_copy_d2h(&c_dev, &mut got).expect("download");
        let mut err = 0.0f32;
        for i in 0..n {
            for j in i..n {
                err = err.max((got[i * n + j] - want[i * n + j]).norm());
            }
        }
        assert!(err < 1e-3, "cherk mismatch: {err:e}");
    }
}

#[test]
fn native_cuda_syrk_conjtrans_complex_alpha_uses_host_fallback() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    // Complex scalars make the Hermitian update non-Hermitian, so no
    // cuBLAS herk spelling applies: checked dispatch fails over to the
    // host core (still correct, just slower). Real scalars delegate to
    // herk (see the test above).
    let (n, k) = (32usize, 24usize);
    let a: Vec<Complex64> = (0..k * n)
        .map(|t| Complex64::new((t as f64 * 0.011).sin(), (t as f64 * 0.017).cos()))
        .collect();
    let c0: Vec<Complex64> = (0..n * n)
        .map(|t| Complex64::new((t as f64 * 0.013).cos(), (t as f64 * 0.019).sin()))
        .collect();
    let mut want = c0.clone();
    syrk_c64(
        TrsmUplo::Lower,
        Transpose::ConjTrans,
        n,
        k,
        Complex64::new(0.5, 0.25),
        &a,
        n,
        Complex64::new(0.5, 0.0),
        &mut want,
        n,
    );
    let mut a_dev = cuda.raw_alloc::<Complex64>(a.len()).expect("alloc");
    let mut c_dev = cuda.raw_alloc::<Complex64>(c0.len()).expect("alloc");
    cuda.try_copy_h2d(&a, &mut a_dev).expect("upload");
    cuda.try_copy_h2d(&c0, &mut c_dev).expect("upload");
    cuda.try_syrk(
        TrsmUplo::Lower,
        Transpose::ConjTrans,
        n,
        k,
        Complex64::new(0.5, 0.25),
        &a_dev,
        n,
        Complex64::new(0.5, 0.0),
        &mut c_dev,
        n,
    )
    .expect("ConjTrans syrk must succeed via host fallback");
    let mut got = vec![Complex64::new(0.0, 0.0); c0.len()];
    cuda.try_copy_d2h(&c_dev, &mut got).expect("download");
    let mut err = 0.0f64;
    for i in 0..n {
        for j in 0..=i {
            err = err.max((got[i * n + j] - want[i * n + j]).norm());
        }
    }
    assert!(err < 1e-9, "ConjTrans fallback mismatch: {err:e}");
}

#[test]
fn native_cuda_syrk_rejects_bad_dims() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let a_dev = cuda.raw_alloc::<f64>(70 * 65).expect("alloc");
    let mut c_dev = cuda.raw_alloc::<f64>(70 * 70).expect("alloc");
    let err = cuda
        .try_syrk(
            TrsmUplo::Lower,
            Transpose::NoTrans,
            70,
            65,
            1.0,
            &a_dev,
            32,
            0.0,
            &mut c_dev,
            70,
        )
        .expect_err("lda < k must be rejected");
    assert!(
        format!("{err}").contains("leading dimension"),
        "unexpected error: {err}"
    );
}
