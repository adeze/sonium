//! ARM SME2 micro-kernels for scientific computing.
//!
//! Low-level SME2-accelerated dense linear algebra operations:
//!
//! - [`gemm::sgemm`] / [`gemm::dgemm`] — dense GEMM with multi-tile FMOPA,
//!   4-Z grouped loads, Kc/Nc cache blocking, and A/B panel packing.
//! - [`gemv::sgemv`] / [`gemv::dgemv`] — dense GEMV with 4-Z grouped loads
//!   and 4-way accumulator unrolling.
//! - [`gemm_bf16::gemm_bf16`] — mixed-precision bf16 × bf16 → f32 GEMM.
//!   See its module docs for the Apple-M4 BFMOPA quirk that forces the
//!   widening to happen at pack time rather than inside the SME unit.
//! - [`detect::Sme2Caps`] — runtime detection of SME2, FEAT_SME_F64F64,
//!   and the streaming vector length (SVL).
//!
//! # Platform support
//!
//! - aarch64 with FEAT_SME2 — full acceleration.
//! - x86_64 with AVX2+FMA — register-blocked kernels in
//!   [`crate::cpu::blas3::avx2`] (runtime-detected).
//! - All other platforms — scalar fallbacks with identical numerical
//!   behaviour.
//!
//! # Safety
//!
//! SME2 operations enter streaming mode via `SMSTART`, which zeroes all
//! NEON/SVE register state. The assembly kernels save/restore the
//! callee-saved SIMD registers around mode transitions, so callers do
//! not need to handle streaming mode.

pub mod blas1;
pub mod detect;
pub mod gemm;
pub mod gemm_bf16;
pub mod gemv;

#[cfg(test)]
mod tests {
    #[test]
    fn test_detect_caps() {
        let caps = super::detect::Sme2Caps::detect();
        let l2 = super::detect::l2_budget_bytes();
        // On any machine, this should not panic
        println!(
            "SME2: {}, f64: {}, SVL: {} bits, L2 budget: {} KB",
            caps.has_sme2,
            caps.has_f64,
            caps.svl_bits,
            l2 / 1024,
        );
        assert!(l2 >= 256 * 1024, "L2 budget should be at least 256 KB");
    }

    #[test]
    fn test_sgemv_identity() {
        // 4x4 identity * 4x1 vector via gemv.
        let a = [
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut y = [0.0f32; 4];
        super::gemv::sgemv(4, 4, 1.0, &a, 4, &x, 1.0, &mut y);
        assert_eq!(y, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_checked_sgemv_rejects_short_output_before_dispatch() {
        let a = [1.0f32; 4];
        let x = [1.0f32; 2];
        let mut y = [0.0f32; 1];
        let error = super::gemv::try_sgemv(2, 2, 1.0, &a, 2, &x, 0.0, &mut y)
            .expect_err("short GEMV output must be rejected before dispatch");
        assert!(matches!(
            error,
            crate::BackendError::InvalidArgument {
                operation: "sgemv",
                ..
            }
        ));
    }

    #[test]
    fn test_checked_sgemv_validates_zero_row_metadata_and_input() {
        let mut y: [f32; 0] = [];
        let error = super::gemv::try_sgemv(0, 3, 1.0, &[], 2, &[1.0, 2.0, 3.0], 0.0, &mut y)
            .expect_err("zero-row GEMV must still validate lda");
        assert!(matches!(
            error,
            crate::BackendError::InvalidArgument {
                operation: "sgemv",
                ..
            }
        ));

        let error = super::gemv::try_sgemv(0, 3, 1.0, &[], 3, &[], 0.0, &mut y)
            .expect_err("zero-row GEMV must still validate x length");
        assert!(matches!(
            error,
            crate::BackendError::InvalidArgument {
                operation: "sgemv",
                ..
            }
        ));

        super::gemv::try_sgemv(0, 3, 1.0, &[], 3, &[1.0, 2.0, 3.0], 0.0, &mut y)
            .expect("a well-formed zero-row GEMV remains a no-op");

        let mut y64: [f64; 0] = [];
        let error = super::gemv::try_dgemv(0, 2, 1.0, &[], 2, &[1.0], 0.0, &mut y64)
            .expect_err("zero-row DGEMV must still validate x length");
        assert!(matches!(
            error,
            crate::BackendError::InvalidArgument {
                operation: "dgemv",
                ..
            }
        ));
        super::gemv::try_dgemv(0, 2, 1.0, &[], 2, &[1.0, 2.0], 0.0, &mut y64)
            .expect("a well-formed zero-row DGEMV remains a no-op");
    }

    #[test]
    fn test_sgemm_basic() {
        // 2x3 * 3x2 = 2x2
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = [0.0f32; 4];
        super::gemm::sgemm(2, 2, 3, 1.0, &a, 3, &b, 2, 1.0, &mut c, 2, false);
        // Row 0: 1*7+2*9+3*11 = 7+18+33 = 58, 1*8+2*10+3*12 = 8+20+36 = 64
        // Row 1: 4*7+5*9+6*11 = 28+45+66 = 139, 4*8+5*10+6*12 = 32+50+72 = 154
        assert!((c[0] - 58.0).abs() < 1e-6);
        assert!((c[1] - 64.0).abs() < 1e-6);
        assert!((c[2] - 139.0).abs() < 1e-6);
        assert!((c[3] - 154.0).abs() < 1e-6);
    }

    #[test]
    fn test_dgemm_basic() {
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [5.0f64, 6.0, 7.0, 8.0];
        let mut c = [0.0f64; 4];
        super::gemm::dgemm(2, 2, 2, 1.0, &a, 2, &b, 2, 1.0, &mut c, 2, false);
        // Row 0: 1*5+2*7 = 19, 1*6+2*8 = 22
        // Row 1: 3*5+4*7 = 43, 3*6+4*8 = 50
        assert!((c[0] - 19.0).abs() < 1e-12);
        assert!((c[1] - 22.0).abs() < 1e-12);
        assert!((c[2] - 43.0).abs() < 1e-12);
        assert!((c[3] - 50.0).abs() < 1e-12);
    }

    #[test]
    fn test_sgemm_alpha() {
        // Test that alpha scaling works
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let b = [3.0f32, 0.0, 0.0, 3.0];
        let mut c = [0.0f32; 4];
        super::gemm::sgemm(2, 2, 2, 2.0, &a, 2, &b, 2, 1.0, &mut c, 2, false);
        assert!((c[0] - 6.0).abs() < 1e-6);
        assert!((c[3] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_sgemm_accumulate() {
        // Test that C += alpha * A * B (accumulates into C)
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let mut c = [10.0f32, 0.0, 0.0, 10.0];
        super::gemm::sgemm(2, 2, 2, 1.0, &a, 2, &b, 2, 1.0, &mut c, 2, false);
        assert!((c[0] - 11.0).abs() < 1e-6);
        assert!((c[3] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_dgemv_basic() {
        let a = [2.0f64, 1.0, 1.0, 3.0];
        let x = [1.0f64, 2.0];
        let mut y = [0.0f64; 2];
        super::gemv::dgemv(2, 2, 1.0, &a, 2, &x, 1.0, &mut y);
        assert!((y[0] - 4.0).abs() < 1e-12);
        assert!((y[1] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_checked_dgemv_rejects_overflowing_matrix_extent() {
        let mut y = [0.0f64; 1];
        let error =
            super::gemv::try_dgemv(usize::MAX, 1, 1.0, &[], usize::MAX, &[0.0], 0.0, &mut y)
                .expect_err("overflowing GEMV extent must be rejected");
        assert!(matches!(
            error,
            crate::BackendError::InvalidArgument {
                operation: "dgemv",
                ..
            }
        ));
    }

    #[test]
    fn test_dgemv_alpha() {
        let a = [1.0f64, 0.0, 0.0, 1.0];
        let x = [3.0f64, 5.0];
        let mut y = [0.0f64; 2];
        super::gemv::dgemv(2, 2, 2.0, &a, 2, &x, 1.0, &mut y);
        assert!((y[0] - 6.0).abs() < 1e-12);
        assert!((y[1] - 10.0).abs() < 1e-12);
    }

    /// Exercises the M/N panel loop (m,n > SVL_S=16 on Apple M4) and the
    /// row/column tail predicates. Compares against a plain triple-loop ref.
    #[test]
    fn test_sgemm_multi_panel_with_tail() {
        let m = 35; // 2 full panels of 16 + tail of 3
        let n = 23; // 1 full panel + tail of 7
        let k = 11;
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = 0.001 + (i as f32) * 0.013;
        }
        for (j, v) in b.iter_mut().enumerate() {
            *v = -0.5 + (j as f32) * 0.007;
        }
        let mut c_ref = vec![0.0f32; m * n];
        let mut c_sme = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c_ref[i * n + j] = 0.5 * s; // alpha=0.5, beta=0
            }
        }
        super::gemm::sgemm(m, n, k, 0.5, &a, k, &b, n, 0.0, &mut c_sme, n, false);
        for (i, (r, s)) in c_ref.iter().zip(c_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 1e-4 * (1.0 + r.abs()),
                "mismatch at flat idx {i}: ref={r}, sme={s}"
            );
        }
    }

    /// Same shape check for f64 with non-trivial beta.
    #[test]
    fn test_dgemm_multi_panel_with_tail() {
        let m = 19; // 2 panels of 8 + tail of 3
        let n = 13; // 1 panel + tail of 5
        let k = 7;
        let mut a = vec![0.0f64; m * k];
        let mut b = vec![0.0f64; k * n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = 0.002 + (i as f64) * 0.011;
        }
        for (j, v) in b.iter_mut().enumerate() {
            *v = -0.3 + (j as f64) * 0.009;
        }
        let mut c_ref = vec![0.25f64; m * n];
        let mut c_sme = c_ref.clone();
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c_ref[i * n + j] = 1.5 * s + 0.7 * c_ref[i * n + j];
            }
        }
        super::gemm::dgemm(m, n, k, 1.5, &a, k, &b, n, 0.7, &mut c_sme, n, false);
        for (i, (r, s)) in c_ref.iter().zip(c_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 1e-10 * (1.0 + r.abs()),
                "mismatch at flat idx {i}: ref={r}, sme={s}"
            );
        }
    }

    /// Exercises the 4-tile fast path (full panel_nr=64 col blocks) plus
    /// a partial 4-tile span plus column tail. m=35 spans 2 row panels + tail.
    #[test]
    fn test_sgemm_4tile_with_mixed_tail() {
        let m = 35;
        let n = 85; // 1 full 4-tile (64) + remainder 21 (single-tile path)
        let k = 23;
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = 0.001 + (i as f32) * 0.003;
        }
        for (j, v) in b.iter_mut().enumerate() {
            *v = -0.2 + (j as f32) * 0.002;
        }
        let mut c_ref = vec![0.1f32; m * n];
        let mut c_sme = c_ref.clone();
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c_ref[i * n + j] = 0.7 * s + 1.3 * c_ref[i * n + j];
            }
        }
        super::gemm::sgemm(m, n, k, 0.7, &a, k, &b, n, 1.3, &mut c_sme, n, false);
        for (i, (r, s)) in c_ref.iter().zip(c_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 5e-4 * (1.0 + r.abs()),
                "mismatch at flat idx {i}: ref={r}, sme={s}"
            );
        }
    }

    /// f64 equivalent — 8×32 4-tile blocks plus column tail.
    #[test]
    fn test_dgemm_4tile_with_mixed_tail() {
        let m = 17;
        let n = 41; // 1 full 4-tile (32) + remainder 9
        let k = 13;
        let mut a = vec![0.0f64; m * k];
        let mut b = vec![0.0f64; k * n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = 0.003 + (i as f64) * 0.005;
        }
        for (j, v) in b.iter_mut().enumerate() {
            *v = -0.1 + (j as f64) * 0.004;
        }
        let mut c_ref = vec![0.0f64; m * n];
        let mut c_sme = c_ref.clone();
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c_ref[i * n + j] = -0.4 * s; // beta=0 path
            }
        }
        super::gemm::dgemm(m, n, k, -0.4, &a, k, &b, n, 0.0, &mut c_sme, n, false);
        for (i, (r, s)) in c_ref.iter().zip(c_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 1e-11 * (1.0 + r.abs()),
                "mismatch at flat idx {i}: ref={r}, sme={s}"
            );
        }
    }

    /// Basic bf16 GEMM correctness: 4×3 × 3×4 = 4×4 reference matches the
    /// scalar widened reference within bf16's ~3-decimal-digit accuracy.
    #[test]
    fn test_bfgemm_basic() {
        use half::bf16;
        let m = 4;
        let n = 4;
        let k = 3;
        let a_f32 = [
            1.0f32, 0.5, 0.25, 2.0, 1.5, 1.0, -1.0, 0.0, 0.5, 3.0, 2.5, 2.0,
        ];
        let b_f32 = [
            0.5f32, 1.0, 1.5, 2.0, -0.5, 0.5, 1.0, 1.5, 0.25, 0.5, 0.75, 1.0,
        ];
        let a: Vec<bf16> = a_f32.iter().map(|&v| bf16::from_f32(v)).collect();
        let b: Vec<bf16> = b_f32.iter().map(|&v| bf16::from_f32(v)).collect();
        let mut c_ref = vec![0.0f32; m * n];
        let mut c_sme = vec![0.0f32; m * n];
        // Reference uses the same bf16 quantisation so we compare like-for-like.
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p].to_f32() * b[p * n + j].to_f32();
                }
                c_ref[i * n + j] = 1.0 * s; // alpha=1, beta=0
            }
        }
        super::gemm_bf16::gemm_bf16(m, n, k, 1.0, &a, k, &b, n, 0.0, &mut c_sme, n);
        for (i, (r, s)) in c_ref.iter().zip(c_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 1e-3 * (1.0 + r.abs()),
                "bfgemm mismatch at idx {i}: ref={r}, sme={s}"
            );
        }
    }

    /// Odd-K (kernel pair-pads K), row tail, column tail all together.
    #[test]
    fn test_bfgemm_odd_k_with_tails() {
        use half::bf16;
        let m = 19; // 1 panel (16) + tail 3
        let n = 17; // 1 panel (16) + tail 1
        let k = 11; // odd → kernel pads to 12 internally
        let mut a_f32 = vec![0.0f32; m * k];
        let mut b_f32 = vec![0.0f32; k * n];
        for (i, v) in a_f32.iter_mut().enumerate() {
            *v = ((i as f32) * 0.013).sin();
        }
        for (j, v) in b_f32.iter_mut().enumerate() {
            *v = ((j as f32) * 0.011).cos();
        }
        let a: Vec<bf16> = a_f32.iter().map(|&v| bf16::from_f32(v)).collect();
        let b: Vec<bf16> = b_f32.iter().map(|&v| bf16::from_f32(v)).collect();
        let mut c_ref = vec![0.0f32; m * n];
        let mut c_sme = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p].to_f32() * b[p * n + j].to_f32();
                }
                c_ref[i * n + j] = 0.6 * s;
            }
        }
        super::gemm_bf16::gemm_bf16(m, n, k, 0.6, &a, k, &b, n, 0.0, &mut c_sme, n);
        for (i, (r, s)) in c_ref.iter().zip(c_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 5e-3 * (1.0 + r.abs()),
                "bfgemm odd-k mismatch at idx {i}: ref={r}, sme={s}"
            );
        }
    }

    /// Exercises the outer Nc block loop. NC_CAP=4096; K*N*4 must also
    /// exceed 8 MB for blocking to engage. K=500, N=4800 gives 9.6 MB and
    /// N>NC_CAP so we get 2 Nc iterations. Mixes a Kc tail and a column
    /// tail at the end of each Nc block — every edge case lit at once.
    #[test]
    fn test_sgemm_nc_multi_block() {
        let m = 6;
        let n = 4_800;
        let k = 500;
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i as f32) * 0.00021).sin();
        }
        for (j, v) in b.iter_mut().enumerate() {
            *v = ((j as f32) * 0.00017).cos();
        }
        let mut c_ref = vec![0.0f32; m * n];
        let mut c_sme = c_ref.clone();
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c_ref[i * n + j] = 0.9 * s;
            }
        }
        super::gemm::sgemm(m, n, k, 0.9, &a, k, &b, n, 0.0, &mut c_sme, n, false);
        for (idx, (r, s)) in c_ref.iter().zip(c_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 5e-4 * (1.0 + r.abs()),
                "nc-multi-block mismatch at idx {idx}: ref={r}, sme={s}"
            );
        }
    }

    /// Exercises the GEMV grouped 4-Z fast path (n >= 4*SVL_S = 64 for f32),
    /// plus the tail single-Z loop, plus a row count > 1 to ensure
    /// per-row state is reset cleanly between rows.
    #[test]
    fn test_sgemv_grouped_load_and_tail() {
        let m = 5;
        let n = 137; // 4 full 4×SVL_S chunks (=128) + tail 9
        let mut a = vec![0.0f32; m * n];
        let mut x = vec![0.0f32; n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = 0.01 + (i as f32) * 0.003;
        }
        for (j, v) in x.iter_mut().enumerate() {
            *v = -0.5 + (j as f32) * 0.007;
        }
        let mut y_ref = vec![0.25f32; m];
        let mut y_sme = y_ref.clone();
        for i in 0..m {
            let mut s = 0.0f32;
            for j in 0..n {
                s += a[i * n + j] * x[j];
            }
            y_ref[i] = 0.8 * s + 0.3 * y_ref[i];
        }
        super::gemv::sgemv(m, n, 0.8, &a, n, &x, 0.3, &mut y_sme);
        for (i, (r, s)) in y_ref.iter().zip(y_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 1e-4 * (1.0 + r.abs()),
                "sgemv mismatch at row {i}: ref={r}, sme={s}"
            );
        }
    }

    /// Exercises non-tight leading dimensions: A, B, and C are each
    /// embedded in larger storage with `ld* > {k,n}` so the stride
    /// arithmetic in the pack helpers, the 4-tile kernel call, and the
    /// column-tail call must not assume `lda == k` etc.
    #[test]
    fn test_sgemm_non_tight_leading_dims() {
        let m = 35;
        let n = 85; // exercises 4-tile fast path + 21-col tail
        let k = 23;
        // Strides chosen to be larger than the logical extent so any
        // "treat lda as k" bug surfaces as garbage from neighbouring rows.
        let lda = k + 7;
        let ldb = n + 11;
        let ldc = n + 5;
        let mut a = vec![0.0f32; m * lda];
        let mut b = vec![0.0f32; k * ldb];
        let mut c_ref = vec![0.0f32; m * ldc];
        // Fill A and B; pad columns get a sentinel value that *must not*
        // contaminate the result.
        for i in 0..m {
            for p in 0..lda {
                a[i * lda + p] = if p < k {
                    0.01 + (i * k + p) as f32 * 0.013
                } else {
                    9_999.0 // sentinel
                };
            }
        }
        for p in 0..k {
            for j in 0..ldb {
                b[p * ldb + j] = if j < n {
                    -0.3 + (p * n + j) as f32 * 0.007
                } else {
                    9_999.0 // sentinel
                };
            }
        }
        // Initialise C with a recognisable padding pattern in the gap.
        for i in 0..m {
            for j in 0..ldc {
                c_ref[i * ldc + j] = if j < n { 0.5 } else { -7_777.0 };
            }
        }
        let mut c_sme = c_ref.clone();
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * lda + p] * b[p * ldb + j];
                }
                c_ref[i * ldc + j] = 0.4 * s + 1.1 * c_ref[i * ldc + j];
            }
        }
        super::gemm::sgemm(m, n, k, 0.4, &a, lda, &b, ldb, 1.1, &mut c_sme, ldc, false);
        for i in 0..m {
            for j in 0..n {
                let r = c_ref[i * ldc + j];
                let s = c_sme[i * ldc + j];
                assert!(
                    (r - s).abs() < 1e-4 * (1.0 + r.abs()),
                    "mismatch at ({i},{j}): ref={r}, sme={s}"
                );
            }
            // Padding columns of C must be untouched.
            for j in n..ldc {
                assert_eq!(
                    c_sme[i * ldc + j],
                    -7_777.0,
                    "stride-pad column {j} of row {i} was clobbered"
                );
            }
        }
    }

    /// f64 grouped 4-Z fast path (n >= 4*SVL_D = 32) + tail.
    #[test]
    fn test_dgemv_grouped_load_and_tail() {
        let m = 3;
        let n = 73; // 2 full 4×SVL_D chunks (=64) + tail 9
        let mut a = vec![0.0f64; m * n];
        let mut x = vec![0.0f64; n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = 0.02 + (i as f64) * 0.004;
        }
        for (j, v) in x.iter_mut().enumerate() {
            *v = -0.3 + (j as f64) * 0.005;
        }
        let mut y_ref = vec![-0.7f64; m];
        let mut y_sme = y_ref.clone();
        for i in 0..m {
            let mut s = 0.0f64;
            for j in 0..n {
                s += a[i * n + j] * x[j];
            }
            y_ref[i] = 1.4 * s - 0.2 * y_ref[i];
        }
        super::gemv::dgemv(m, n, 1.4, &a, n, &x, -0.2, &mut y_sme);
        for (i, (r, s)) in y_ref.iter().zip(y_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 1e-11 * (1.0 + r.abs()),
                "dgemv mismatch at row {i}: ref={r}, sme={s}"
            );
        }
    }

    /// GEMV scalar fallback must only touch `y[..m]` even when the caller
    /// passes a longer backing slice. (The SME2 asm path already writes
    /// exactly m elements; this protects the fallback's behaviour.)
    #[test]
    fn test_sgemv_scalar_respects_y_len() {
        // Run the f32 fallback explicitly by bypassing the SME2 dispatch
        // — we just call sgemv with sizes that the public entry handles
        // correctly. The check is that y[m..] stays untouched.
        let m = 3;
        let n = 4;
        let a = vec![1.0f32; m * n];
        let x = vec![1.0f32; n];
        let mut y = vec![999.0f32; m + 5];
        super::gemv::sgemv(m, n, 1.0, &a, n, &x, 0.0, &mut y);
        // Active region was zeroed-then-FMA'd: 0 + 1*sum(x_j) = 4.0
        for v in &y[..m] {
            assert!((v - 4.0).abs() < 1e-6, "active y mismatch: {v}");
        }
        // Padding past m must be untouched.
        for v in &y[m..] {
            assert_eq!(*v, 999.0, "y stride pad was clobbered");
        }
    }

    /// Sanity: passing too-short A panics with a clear message instead of
    /// running unsafe asm past the slice end.
    #[test]
    #[should_panic(expected = "sgemm A: slice len")]
    fn test_sgemm_short_a_panics() {
        let m = 8;
        let n = 8;
        let k = 8;
        let a = vec![0.0f32; m * k - 1]; // one element too short
        let b = vec![0.0f32; k * n];
        let mut c = vec![0.0f32; m * n];
        super::gemm::sgemm(m, n, k, 1.0, &a, k, &b, n, 0.0, &mut c, n, false);
    }

    #[test]
    fn test_checked_sgemm_rejects_short_a_without_mutation() {
        let a = [0.0f32; 3];
        let b = [0.0f32; 4];
        let mut c = [7.0f32; 4];
        let error = super::gemm::try_sgemm(2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2, false)
            .expect_err("checked GEMM must reject a short input");
        assert!(matches!(
            error,
            crate::BackendError::InvalidArgument {
                operation: "sgemm",
                ..
            }
        ));
        assert_eq!(c, [7.0; 4]);
    }

    #[test]
    fn test_checked_dgemm_rejects_overflowing_span() {
        let mut c = [3.0f64; 1];
        let error = super::gemm::try_dgemm(
            usize::MAX,
            2,
            1,
            1.0,
            &[],
            usize::MAX,
            &[],
            2,
            0.0,
            &mut c,
            2,
            false,
        )
        .expect_err("checked GEMM must reject overflowing extents");
        assert!(matches!(
            error,
            crate::BackendError::InvalidArgument {
                operation: "dgemm",
                ..
            }
        ));
        assert_eq!(c, [3.0]);
    }

    #[test]
    fn test_checked_bf16_gemm_rejects_bad_leading_dimension() {
        let a = [half::bf16::from_f32(1.0); 4];
        let b = [half::bf16::from_f32(1.0); 4];
        let mut c = [5.0f32; 4];
        let error = super::gemm_bf16::try_gemm_bf16(2, 2, 2, 1.0, &a, 1, &b, 2, 0.0, &mut c, 2)
            .expect_err("checked bf16 GEMM must reject lda < columns");
        assert!(matches!(
            error,
            crate::BackendError::InvalidArgument {
                operation: "gemm_bf16",
                ..
            }
        ));
        assert_eq!(c, [5.0; 4]);
    }

    #[test]
    fn test_sgemv_short_y_is_rejected_without_mutation() {
        let m = 4;
        let n = 4;
        let a = vec![0.0f32; m * n];
        let x = vec![0.0f32; n];
        let mut y = vec![0.0f32; m - 1];
        let before = y.clone();
        super::gemv::sgemv(m, n, 1.0, &a, n, &x, 0.0, &mut y);
        assert_eq!(y, before);
    }

    /// Exercises the Kc outer-block loop. Adaptive blocking only engages
    /// when K*N*sizeof > 8 MB, so we size the test to push past that
    /// threshold (K*N*4 = 13 MB > 8 MB → KC=512 cap kicks in, 3 passes).
    /// The partial-sum chain (first pass uses caller's beta, subsequent
    /// passes use beta=1) is what we want to validate against single-pass.
    #[test]
    fn test_sgemm_kc_multi_pass() {
        let m = 8;
        let n = 1700;
        let k = 1900; // K*N*4 = ~13 MB → 4 Kc passes of 512+512+512+364
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        // Bounded magnitudes so partial-sum FP error stays small.
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i as f32) * 0.00037).sin();
        }
        for (j, v) in b.iter_mut().enumerate() {
            *v = ((j as f32) * 0.00029).cos();
        }
        let mut c_ref = vec![0.5f32; m * n];
        let mut c_sme = c_ref.clone();
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c_ref[i * n + j] = 0.3 * s + 0.6 * c_ref[i * n + j];
            }
        }
        super::gemm::sgemm(m, n, k, 0.3, &a, k, &b, n, 0.6, &mut c_sme, n, false);
        for (i, (r, s)) in c_ref.iter().zip(c_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 5e-4 * (1.0 + r.abs()),
                "kc-multi-pass mismatch at idx {i}: ref={r}, sme={s}"
            );
        }
    }

    /// f64 Kc multi-pass — K*N*8 > 8 MB needed to engage blocking.
    #[test]
    fn test_dgemm_kc_multi_pass() {
        let m = 7;
        let n = 1100;
        let k = 1300; // K*N*8 = ~11 MB → DGEMM_KC=256 → 6 passes
        let mut a = vec![0.0f64; m * k];
        let mut b = vec![0.0f64; k * n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i as f64) * 0.00041).sin();
        }
        for (j, v) in b.iter_mut().enumerate() {
            *v = ((j as f64) * 0.00033).cos();
        }
        let mut c_ref = vec![1.25f64; m * n];
        let mut c_sme = c_ref.clone();
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c_ref[i * n + j] = 1.7 * s - 0.4 * c_ref[i * n + j];
            }
        }
        super::gemm::dgemm(m, n, k, 1.7, &a, k, &b, n, -0.4, &mut c_sme, n, false);
        for (i, (r, s)) in c_ref.iter().zip(c_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 1e-10 * (1.0 + r.abs()),
                "kc-multi-pass mismatch at idx {i}: ref={r}, sme={s}"
            );
        }
    }

    #[test]
    fn test_sgemv_accumulate() {
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let x = [1.0f32, 1.0];
        let mut y = [5.0f32, 7.0];
        super::gemv::sgemv(2, 2, 1.0, &a, 2, &x, 1.0, &mut y);
        assert!((y[0] - 6.0).abs() < 1e-6);
        assert!((y[1] - 8.0).abs() < 1e-6);
    }

    /// Exercises the rayon parallel-Mr path (m >= PARALLEL_M_THRESHOLD=64).
    /// Uses n=128 so multiple 4-tile panels fire, and non-trivial alpha/beta.
    #[test]
    fn test_sgemm_parallel_mr() {
        let m = 128; // 8 full row-panels at SVL_S=16
        let n = 128;
        let k = 64;
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i as f32) * 0.00043).sin();
        }
        for (j, v) in b.iter_mut().enumerate() {
            *v = ((j as f32) * 0.00037).cos();
        }
        let mut c_ref = vec![0.5f32; m * n];
        let mut c_sme = c_ref.clone();
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c_ref[i * n + j] = 0.6 * s + 0.8 * c_ref[i * n + j];
            }
        }
        super::gemm::sgemm(m, n, k, 0.6, &a, k, &b, n, 0.8, &mut c_sme, n, false);
        for (idx, (r, s)) in c_ref.iter().zip(c_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 5e-4 * (1.0 + r.abs()),
                "parallel sgemm mismatch at {idx}: ref={r}, sme={s}"
            );
        }
    }

    /// f64 parallel-Mr path. m=64 >= threshold; n=64 gives 2 full 4-tile panels.
    #[test]
    fn test_dgemm_parallel_mr() {
        let m = 64;
        let n = 64;
        let k = 32;
        let mut a = vec![0.0f64; m * k];
        let mut b = vec![0.0f64; k * n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i as f64) * 0.00051).sin();
        }
        for (j, v) in b.iter_mut().enumerate() {
            *v = ((j as f64) * 0.00047).cos();
        }
        let mut c_ref = vec![1.0f64; m * n];
        let mut c_sme = c_ref.clone();
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c_ref[i * n + j] = 1.2 * s - 0.5 * c_ref[i * n + j];
            }
        }
        super::gemm::dgemm(m, n, k, 1.2, &a, k, &b, n, -0.5, &mut c_sme, n, false);
        for (idx, (r, s)) in c_ref.iter().zip(c_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 1e-10 * (1.0 + r.abs()),
                "parallel dgemm mismatch at {idx}: ref={r}, sme={s}"
            );
        }
    }

    /// Parallel-Mr (m >= 64) crossed with Nc multi-block (N > 4096 cap).
    /// Forces packed_B to be re-built across multiple Nc tiles while
    /// workers in each tile race the row-panels — the most adversarial
    /// shape for the rayon scope / pack-buffer interaction.
    #[test]
    fn test_sgemm_parallel_mr_with_nc_blocking() {
        let m = 128; // 8 row-panels at svl_s=16
        let n = 4_800; // > SGEMM_NC_CAP=2048, 3 Nc passes
        let k = 500;
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i as f32) * 0.00019).sin();
        }
        for (j, v) in b.iter_mut().enumerate() {
            *v = ((j as f32) * 0.00021).cos();
        }
        let mut c_ref = vec![0.25f32; m * n];
        let mut c_sme = c_ref.clone();
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c_ref[i * n + j] = 0.7 * s + 0.4 * c_ref[i * n + j];
            }
        }
        super::gemm::sgemm(m, n, k, 0.7, &a, k, &b, n, 0.4, &mut c_sme, n, false);
        for (idx, (r, s)) in c_ref.iter().zip(c_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 5e-4 * (1.0 + r.abs()),
                "parallel sgemm × Nc-block mismatch at {idx}: ref={r}, sme={s}"
            );
        }
    }

    /// Parallel-Mr crossed with Kc multi-pass. K*N*4 > 8 MB engages Kc
    /// blocking, which means the C row-band is re-touched on every Kc
    /// pass with beta=1 — has to interleave correctly with the per-task
    /// packed_A in the parallel path.
    #[test]
    fn test_sgemm_parallel_mr_with_kc_multi_pass() {
        let m = 128;
        let n = 1_700;
        let k = 1_900; // K*N*4 ~ 13 MB → Kc=512, 4 passes
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i as f32) * 0.00033).sin();
        }
        for (j, v) in b.iter_mut().enumerate() {
            *v = ((j as f32) * 0.00027).cos();
        }
        let mut c_ref = vec![0.5f32; m * n];
        let mut c_sme = c_ref.clone();
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c_ref[i * n + j] = 0.3 * s + 0.6 * c_ref[i * n + j];
            }
        }
        super::gemm::sgemm(m, n, k, 0.3, &a, k, &b, n, 0.6, &mut c_sme, n, false);
        for (idx, (r, s)) in c_ref.iter().zip(c_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 5e-4 * (1.0 + r.abs()),
                "parallel sgemm × Kc-pass mismatch at {idx}: ref={r}, sme={s}"
            );
        }
    }

    /// f64 equivalent: parallel-Mr × Kc multi-pass. K*N*8 > 8 MB → Kc=256
    /// engaged with several passes; m=64 hits PARALLEL_M_THRESHOLD.
    #[test]
    fn test_dgemm_parallel_mr_with_kc_multi_pass() {
        let m = 64;
        let n = 1_100;
        let k = 1_300; // K*N*8 ~ 11 MB → Kc=256, ~6 passes
        let mut a = vec![0.0f64; m * k];
        let mut b = vec![0.0f64; k * n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i as f64) * 0.00041).sin();
        }
        for (j, v) in b.iter_mut().enumerate() {
            *v = ((j as f64) * 0.00033).cos();
        }
        let mut c_ref = vec![1.25f64; m * n];
        let mut c_sme = c_ref.clone();
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c_ref[i * n + j] = 1.7 * s - 0.4 * c_ref[i * n + j];
            }
        }
        super::gemm::dgemm(m, n, k, 1.7, &a, k, &b, n, -0.4, &mut c_sme, n, false);
        for (idx, (r, s)) in c_ref.iter().zip(c_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 1e-10 * (1.0 + r.abs()),
                "parallel dgemm × Kc-pass mismatch at {idx}: ref={r}, sme={s}"
            );
        }
    }

    /// Exercises the f64 three-tier dispatch (8-tile + 4-tile + single-tile)
    /// in one call. At SVL=512: panel_nr_8 = 64, panel_nr_4 = 32, svl_d = 8.
    /// n = 104 ⇒ nc_full_8 = 64 (one panel8 panel), nc_full_4 = 32 (one
    /// panel4 panel), column tail = 8 (one single-tile call). Hits every
    /// kernel and every transition.
    #[test]
    fn test_dgemm_panel8_three_tier_crossover() {
        let m = 8; // exactly one row-panel at svl_d=8
        let n = 104; // = 64 + 32 + 8
        let k = 17;
        let mut a = vec![0.0f64; m * k];
        let mut b = vec![0.0f64; k * n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i as f64) * 0.00029).sin();
        }
        for (j, v) in b.iter_mut().enumerate() {
            *v = ((j as f64) * 0.00037).cos();
        }
        let mut c_ref = vec![0.5f64; m * n];
        let mut c_sme = c_ref.clone();
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c_ref[i * n + j] = 0.9 * s + 0.3 * c_ref[i * n + j];
            }
        }
        super::gemm::dgemm(m, n, k, 0.9, &a, k, &b, n, 0.3, &mut c_sme, n, false);
        for (idx, (r, s)) in c_ref.iter().zip(c_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 1e-11 * (1.0 + r.abs()),
                "panel8 three-tier mismatch at {idx}: ref={r}, sme={s}"
            );
        }
    }

    /// Larger 3-tier shape with multi-row-panel m and parallel-Mr engaged.
    /// m=128 ⇒ 16 row-panels at svl_d=8, well above PARALLEL_M_THRESHOLD.
    /// n=144 = 2*64 + 16 ⇒ 2 panel8 + 0 panel4 + 2 tail (16 = 2*svl_d).
    /// Confirms the parallel scope correctly partitions C across both packed_B
    /// regions and the column tail.
    #[test]
    fn test_dgemm_panel8_parallel_with_tail() {
        let m = 128;
        let n = 144;
        let k = 23;
        let mut a = vec![0.0f64; m * k];
        let mut b = vec![0.0f64; k * n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i as f64) * 0.00041).sin();
        }
        for (j, v) in b.iter_mut().enumerate() {
            *v = ((j as f64) * 0.00033).cos();
        }
        let mut c_ref = vec![1.25f64; m * n];
        let mut c_sme = c_ref.clone();
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c_ref[i * n + j] = 0.6 * s + 0.7 * c_ref[i * n + j];
            }
        }
        super::gemm::dgemm(m, n, k, 0.6, &a, k, &b, n, 0.7, &mut c_sme, n, false);
        for (idx, (r, s)) in c_ref.iter().zip(c_sme.iter()).enumerate() {
            assert!(
                (r - s).abs() < 1e-10 * (1.0 + r.abs()),
                "parallel panel8 mismatch at {idx}: ref={r}, sme={s}"
            );
        }
    }

    /// Trans-A path: stored A is k×m row-major and we apply A^T (= m×k).
    /// Picks a small known shape so the math is verifiable by hand and the
    /// transposed-pack indexing is exercised end-to-end.
    /// A = [[1,2,3],[4,5,6]] (k=2, m=3 stored), A^T = [[1,4],[2,5],[3,6]]
    /// (m=3, k=2 applied). B = I (k=2, n=2). Expect C = A^T = [[1,4],[2,5],[3,6]].
    #[test]
    fn test_sgemm_trans_a_known_answer() {
        let m = 3;
        let n = 2;
        let k = 2;
        // Stored as k×m = 2×3 row-major, lda=3.
        let a: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // B = identity 2×2, ldb=2.
        let b: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 6]; // 3×2, ldc=2
        super::gemm::sgemm(m, n, k, 1.0, &a, 3, &b, 2, 0.0, &mut c, 2, true);
        let want = [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        for (g, w) in c.iter().zip(want.iter()) {
            assert!(
                (g - w).abs() < 1e-6,
                "trans_a mismatch: got {c:?}, want {want:?}"
            );
        }
    }

    /// Larger Trans-A shape: m=64, n=48, k=80. Exercises the panel4 fast
    /// path on a transposed source. Reference is the same-shape NoTrans
    /// computation against an explicitly transposed B-side matrix.
    #[test]
    fn test_sgemm_trans_a_parity_vs_explicit_transpose() {
        let m = 64;
        let n = 48;
        let k = 80;
        // Stored A as k×m (= 80×64) row-major.
        let mut a_stored = vec![0.0f32; k * m];
        for (i, v) in a_stored.iter_mut().enumerate() {
            *v = ((i as f32) * 0.00017).sin();
        }
        let mut b = vec![0.0f32; k * n];
        for (j, v) in b.iter_mut().enumerate() {
            *v = ((j as f32) * 0.00023).cos();
        }
        let mut c_trans = vec![0.5f32; m * n];
        let mut c_ref = c_trans.clone();

        // Reference: explicitly transpose A into row-major m×k, then NoTrans.
        let mut a_explicit = vec![0.0f32; m * k];
        for p in 0..k {
            for i in 0..m {
                a_explicit[i * k + p] = a_stored[p * m + i];
            }
        }
        super::gemm::sgemm(
            m,
            n,
            k,
            0.7,
            &a_explicit,
            k,
            &b,
            n,
            0.3,
            &mut c_ref,
            n,
            false,
        );
        super::gemm::sgemm(
            m,
            n,
            k,
            0.7,
            &a_stored,
            m,
            &b,
            n,
            0.3,
            &mut c_trans,
            n,
            true,
        );
        for (idx, (g, w)) in c_trans.iter().zip(c_ref.iter()).enumerate() {
            assert!(
                (g - w).abs() < 5e-4 * (1.0 + w.abs()),
                "trans_a parity mismatch at {idx}: got {g}, want {w}"
            );
        }
    }

    /// f64 Trans-A: small known-answer to catch any indexing regression
    /// specific to the f64 pack helper.
    #[test]
    fn test_dgemm_trans_a_known_answer() {
        let m = 3;
        let n = 2;
        let k = 2;
        let a: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3 stored
        let b: [f64; 4] = [1.0, 0.0, 0.0, 1.0];
        let mut c = [0.0f64; 6];
        super::gemm::dgemm(m, n, k, 1.0, &a, 3, &b, 2, 0.0, &mut c, 2, true);
        let want = [1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0];
        for (g, w) in c.iter().zip(want.iter()) {
            assert!(
                (g - w).abs() < 1e-12,
                "dgemm trans_a: got {c:?}, want {want:?}"
            );
        }
    }

    /// f64 Trans-A at sizes that hit the new 8-tile path: m=128 (parallel),
    /// n=128 (= 2 panel_nr_8 + 0 panel_nr_4), k=40.
    #[test]
    fn test_dgemm_trans_a_panel8_parity() {
        let m = 128;
        let n = 128;
        let k = 40;
        let mut a_stored = vec![0.0f64; k * m]; // k×m stored
        for (i, v) in a_stored.iter_mut().enumerate() {
            *v = ((i as f64) * 0.00023).sin();
        }
        let mut b = vec![0.0f64; k * n];
        for (j, v) in b.iter_mut().enumerate() {
            *v = ((j as f64) * 0.00019).cos();
        }
        let mut c_trans = vec![1.25f64; m * n];
        let mut c_ref = c_trans.clone();
        let mut a_explicit = vec![0.0f64; m * k];
        for p in 0..k {
            for i in 0..m {
                a_explicit[i * k + p] = a_stored[p * m + i];
            }
        }
        super::gemm::dgemm(
            m,
            n,
            k,
            0.8,
            &a_explicit,
            k,
            &b,
            n,
            -0.4,
            &mut c_ref,
            n,
            false,
        );
        super::gemm::dgemm(
            m,
            n,
            k,
            0.8,
            &a_stored,
            m,
            &b,
            n,
            -0.4,
            &mut c_trans,
            n,
            true,
        );
        for (idx, (g, w)) in c_trans.iter().zip(c_ref.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-10 * (1.0 + w.abs()),
                "dgemm trans_a × panel8 mismatch at {idx}: got {g}, want {w}"
            );
        }
    }
}
