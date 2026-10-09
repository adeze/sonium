use super::*;

#[test]
#[cfg(feature = "cpu")]
fn host_trsm_compute_round_trips_all_variants() {
    use crate::cpu::blas3::trsm::{TrsmSide, TrsmUplo};
    // Independent oracle: build B = op(A) * X (or X * op(A)) with a
    // direct triple loop, then the substitution core must recover X.
    for side in [TrsmSide::Left, TrsmSide::Right] {
        for uplo in [TrsmUplo::Lower, TrsmUplo::Upper] {
            for trans in [Transpose::NoTrans, Transpose::Trans] {
                for unit in [false, true] {
                    let (m, n) = (7usize, 5usize);
                    let adim = match side {
                        TrsmSide::Left => m,
                        TrsmSide::Right => n,
                    };
                    let mut a = vec![0.0f64; adim * adim];
                    for i in 0..adim {
                        for j in 0..adim {
                            let in_tri = match uplo {
                                TrsmUplo::Lower => j <= i,
                                TrsmUplo::Upper => j >= i,
                            };
                            if in_tri {
                                a[i * adim + j] = if i == j {
                                    2.0 + 0.1 * i as f64
                                } else {
                                    0.01 * ((i * adim + j) as f64)
                                };
                            }
                        }
                        if unit {
                            a[i * adim + i] = 0.0;
                        }
                    }
                    let x0: Vec<f64> = (0..m * n).map(|k| (k as f64 * 0.37).sin()).collect();
                    let ae = |r: usize, c: usize| -> f64 {
                        match trans {
                            Transpose::NoTrans => a[r * adim + c],
                            _ => a[c * adim + r],
                        }
                    };
                    let mut b = vec![0.0f64; m * n];
                    match side {
                        TrsmSide::Left => {
                            for i in 0..m {
                                for j in 0..n {
                                    let mut s = 0.0;
                                    for p in 0..adim {
                                        let ap = if unit && p == i { 1.0 } else { ae(i, p) };
                                        s += ap * x0[p * n + j];
                                    }
                                    b[i * n + j] = s;
                                }
                            }
                        }
                        TrsmSide::Right => {
                            for i in 0..m {
                                for j in 0..n {
                                    let mut s = 0.0;
                                    for p in 0..adim {
                                        let ap = if unit && p == j { 1.0 } else { ae(p, j) };
                                        s += x0[i * n + p] * ap;
                                    }
                                    b[i * n + j] = s;
                                }
                            }
                        }
                    }
                    let mut got = b.clone();
                    host_trsm_compute(side, uplo, trans, unit, m, n, 2.0, &a, adim, &mut got, n);
                    for (g, x) in got.iter().zip(x0.iter()) {
                        assert!(
                            (g - 2.0 * x).abs() < 1e-9,
                            "variant {side:?}/{uplo:?}/{trans:?}/unit={unit}: {g} vs {}",
                            2.0 * x
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[cfg(feature = "cpu")]
fn host_syrk_compute_round_trips_all_variants() {
    use crate::cpu::blas3::trsm::TrsmUplo;
    // Independent oracle: build C = alpha*op(A)*op(A)^T + beta*C0 with
    // a direct triple loop over the full matrix, then the triangle core
    // must agree on the referenced triangle (plus beta scaling with a
    // zero inner dimension).
    for uplo in [TrsmUplo::Lower, TrsmUplo::Upper] {
        for trans in [Transpose::NoTrans, Transpose::Trans] {
            for (n, k) in [(7usize, 5usize), (5, 7)] {
                let (a_rows, a_cols) = match trans {
                    Transpose::NoTrans => (n, k),
                    _ => (k, n),
                };
                let a: Vec<f64> = (0..a_rows * a_cols)
                    .map(|t| (t as f64 * 0.31).sin())
                    .collect();
                let c0: Vec<f64> = (0..n * n).map(|t| (t as f64 * 0.17).cos()).collect();
                let (alpha, beta) = (0.5, 0.25);
                let mut full = vec![0.0f64; n * n];
                for i in 0..n {
                    for j in 0..n {
                        let mut s = 0.0;
                        for p in 0..k {
                            let (f, g) = match trans {
                                Transpose::NoTrans => (a[i * k + p], a[j * k + p]),
                                _ => (a[p * n + i], a[p * n + j]),
                            };
                            s += f * g;
                        }
                        full[i * n + j] = alpha * s + beta * c0[i * n + j];
                    }
                }
                let mut got = c0.clone();
                host_syrk_compute(uplo, trans, n, k, alpha, &a, a_cols, beta, &mut got, n);
                for i in 0..n {
                    for j in 0..n {
                        let in_tri = match uplo {
                            TrsmUplo::Lower => j <= i,
                            TrsmUplo::Upper => j >= i,
                        };
                        if in_tri {
                            assert!(
                                (got[i * n + j] - full[i * n + j]).abs() < 1e-9,
                                "variant {uplo:?}/{trans:?} n={n} k={k} at ({i},{j})"
                            );
                        } else {
                            assert_eq!(
                                got[i * n + j],
                                c0[i * n + j],
                                "unreferenced triangle must stay untouched"
                            );
                        }
                    }
                }
            }
        }
    }
    // Zero inner dimension still scales the referenced triangle by
    // beta; the other triangle stays untouched.
    let c0 = vec![1.0f64; 4 * 4];
    let mut got = c0.clone();
    host_syrk_compute(
        TrsmUplo::Lower,
        Transpose::NoTrans,
        4,
        0,
        2.0,
        &[],
        1,
        0.5,
        &mut got,
        4,
    );
    for i in 0..4 {
        for j in 0..4 {
            let want = if j <= i { 0.5 } else { 1.0 };
            assert!((got[i * 4 + j] - want).abs() < 1e-12);
        }
    }
}

#[test]
fn legacy_precision_cast_wrappers_delegate_to_checked_paths() {
    let source = include_str!("mod.rs");

    let cast_to_start = source
        .rfind("fn cast_to_lower<Upper>(&self")
        .expect("legacy cast_to_lower wrapper should exist");
    let cast_to_end = source[cast_to_start..]
        .find("/// Checked upper-to-lower device cast.")
        .map(|offset| cast_to_start + offset)
        .expect("checked cast_to_lower marker should follow wrapper");
    let cast_to_wrapper = &source[cast_to_start..cast_to_end];
    assert!(cast_to_wrapper.contains("self.try_cast_to_lower(src, dst)"));
    assert!(!cast_to_wrapper.contains("assert_eq!("));
    assert!(!cast_to_wrapper.contains("vec!["));

    let cast_from_start = source
        .rfind("fn cast_from_lower<Upper>(")
        .expect("legacy cast_from_lower wrapper should exist");
    let cast_from_end = source[cast_from_start..]
        .find("/// Checked lower-to-upper device cast.")
        .map(|offset| cast_from_start + offset)
        .expect("checked cast_from_lower marker should follow wrapper");
    let cast_from_wrapper = &source[cast_from_start..cast_from_end];
    assert!(cast_from_wrapper.contains("self.try_cast_from_lower(src, dst)"));
    assert!(!cast_from_wrapper.contains("assert_eq!("));
    assert!(!cast_from_wrapper.contains("vec!["));

    let copy_h2d_start = source
        .rfind("fn copy_h2d_cast<Upper>")
        .expect("legacy copy_h2d_cast wrapper should exist");
    let copy_h2d_end = source[copy_h2d_start..]
        .find("/// Checked host-to-device precision cast.")
        .map(|offset| copy_h2d_start + offset)
        .expect("checked copy_h2d_cast marker should follow wrapper");
    let copy_h2d_wrapper = &source[copy_h2d_start..copy_h2d_end];
    assert!(copy_h2d_wrapper.contains("host_copy_h2d_cast(self, host, dev)"));
    assert!(!copy_h2d_wrapper.contains("assert_eq!("));
    assert!(!copy_h2d_wrapper.contains("vec!["));
}

#[test]
fn checked_host_precision_cast_rejects_mismatch_without_mutation() {
    let backend = crate::cpu::CpuBackend::new();
    let mut destination = backend.raw_alloc::<f32>(2).expect("destination allocation");
    backend.copy_h2d(&[3.0_f32, -4.0_f32], &mut destination);

    let error = backend
        .try_copy_h2d_cast(&[9.0_f64], &mut destination)
        .expect_err("length mismatch must be rejected before conversion");
    assert!(matches!(error, BackendError::InvalidArgument { .. }));

    let mut observed = [0.0_f32; 2];
    backend.copy_d2h(&destination, &mut observed);
    assert_eq!(observed, [3.0, -4.0]);

    backend.copy_h2d_cast(&[9.0_f64], &mut destination);
    backend.copy_d2h(&destination, &mut observed);
    assert_eq!(observed, [3.0, -4.0]);
}

#[test]
fn default_checked_csr_boundaries_poll_pending_backend_state() {
    let source = include_str!("mod.rs");
    let prebake_start = source
        .rfind("fn try_prebake_csr<'a, T: DeviceScalar>")
        .expect("default checked CSR prebake should exist");
    let prebake = &source[prebake_start..];
    assert!(prebake.contains("self.check_pending_error()?"));

    let spmv_start = source
        .rfind("fn try_spmv_prebaked<T: DeviceScalar>")
        .expect("default checked CSR SpMV should exist");
    let spmv = &source[spmv_start..];
    assert!(spmv.contains("self.check_pending_error()"));
}

#[test]
fn checked_host_staging_reports_capacity_overflow_without_panicking() {
    let error = try_zeroed_host::<u8>(usize::MAX, "overflow-test")
        .expect_err("an impossible host staging request must fail closed");
    assert!(
        matches!(error, BackendError::Allocation { reason } if reason.contains("host staging allocation failed"))
    );
}

#[test]
fn checked_host_staging_capacity_reserves_quota_before_vector_growth() {
    let ledger = Arc::new(BackendAllocationLedger::with_cap(8));
    let (values, guard) = try_host_staging_capacity::<u64>(&ledger, 1, "capacity-test")
        .expect("one u64 fits the staging quota");
    assert_eq!(values.capacity(), 1);
    assert_eq!(ledger.live_bytes.load(Ordering::Acquire), 8);

    let rejected = try_host_staging_capacity::<u64>(&ledger, 2, "capacity-test")
        .expect_err("quota must reject before attempting vector growth");
    assert!(matches!(rejected, BackendError::Allocation { .. }));
    drop(guard);
    assert_eq!(ledger.live_bytes.load(Ordering::Acquire), 0);

    let overflow = try_host_staging_capacity::<u64>(&ledger, usize::MAX, "capacity-test")
        .expect_err("element-byte multiplication must be checked");
    assert!(matches!(
        overflow,
        BackendError::InvalidArgument { reason, .. } if reason.contains("overflows")
    ));
    assert_eq!(ledger.live_bytes.load(Ordering::Acquire), 0);
}

#[test]
fn dense_validation_rejects_matrix_span_overflow() {
    let gemv = validate_gemv_args(Transpose::NoTrans, 2, 1, usize::MAX, usize::MAX, 1, 2)
        .expect_err("overflowing GEMV span must be rejected");
    assert!(matches!(
        gemv,
        BackendError::InvalidArgument { operation: "gemv", reason }
            if reason.contains("overflows")
    ));

    let gemm = validate_gemm_args(
        Transpose::NoTrans,
        Transpose::NoTrans,
        2,
        1,
        1,
        usize::MAX,
        usize::MAX,
        1,
        1,
        2,
        1,
    )
    .expect_err("overflowing GEMM span must be rejected");
    assert!(matches!(
        gemm,
        BackendError::InvalidArgument { operation: "gemm", reason }
            if reason.contains("overflows")
    ));
}

#[test]
fn checked_result_collection_is_fallible_and_shape_checked() {
    let values = try_collect_backend([Ok(1_u32), Ok(2_u32)], 2, "result-test")
        .expect("matching checked results");
    assert_eq!(values, [1, 2]);

    let short = try_collect_backend([Ok(1_u32)], 2, "result-test")
        .expect_err("short result iterators must fail closed");
    assert!(matches!(short, BackendError::OperationFailed { .. }));

    let overflow = try_collect_backend::<u8, _>(std::iter::empty(), usize::MAX, "result-test")
        .expect_err("impossible result capacity must return allocation error");
    assert!(matches!(overflow, BackendError::Allocation { .. }));
}

#[test]
fn allocation_ledger_rejects_before_driver_and_refunds_failed_attempts() {
    let ledger = Arc::new(BackendAllocationLedger::with_cap(16));
    let first = ledger.try_reserve(16).expect("reservation within the cap");
    let rejected = ledger
        .try_reserve(1)
        .expect_err("quota must reject before a driver allocation");
    assert!(matches!(rejected, OomError::QuotaExceeded { .. }));
    drop(first);

    let retry = ledger
        .try_reserve(16)
        .expect("failed allocation reservation must be refundable");
    drop(retry);
    assert_eq!(ledger.live_bytes.load(Ordering::Acquire), 0);
}

#[test]
fn allocation_ledger_injected_driver_failure_is_retryable() {
    let ledger = Arc::new(BackendAllocationLedger::with_cap(16));
    ledger.inject_driver_failure();
    let error = ledger
        .try_reserve(8)
        .expect_err("injected driver failure must be surfaced");
    assert!(matches!(error, OomError::DriverOom(reason) if reason.contains("injected")));
    assert_eq!(ledger.live_bytes.load(Ordering::Acquire), 0);

    let retry = ledger
        .try_reserve(8)
        .expect("a failed driver reservation must remain retryable");
    assert_eq!(ledger.live_bytes.load(Ordering::Acquire), 8);
    drop(retry);
    assert_eq!(ledger.live_bytes.load(Ordering::Acquire), 0);
}

#[test]
fn allocation_ledger_keeps_quota_until_last_deferred_owner_drops() {
    let ledger = Arc::new(BackendAllocationLedger::with_cap(32));
    let guard = Arc::new(ledger.try_reserve(24).expect("initial reservation"));
    let deferred = Arc::clone(&guard);
    drop(guard);
    assert_eq!(ledger.live_bytes.load(Ordering::Acquire), 24);
    assert_eq!(ledger.high_water_bytes.load(Ordering::Acquire), 24);
    assert!(ledger.try_reserve(9).is_err());
    drop(deferred);
    assert_eq!(ledger.live_bytes.load(Ordering::Acquire), 0);
    let final_retry = ledger
        .try_reserve(32)
        .expect("deferred owner release must restore the quota");
    drop(final_retry);
    assert_eq!(ledger.high_water_bytes.load(Ordering::Acquire), 32);
}

#[test]
fn host_staging_vec_is_accounted_and_refunded() {
    let ledger = Arc::new(BackendAllocationLedger::with_cap(16));
    let (values, reservation) =
        try_host_staging_vec::<f32>(&ledger, 4, "test_staging").expect("staging allocation");
    assert_eq!(values, vec![0.0; 4]);
    assert_eq!(ledger.live_bytes.load(Ordering::Acquire), 16);
    drop(values);
    drop(reservation);
    assert_eq!(ledger.live_bytes.load(Ordering::Acquire), 0);
    assert_eq!(ledger.high_water_bytes.load(Ordering::Acquire), 16);
}

#[test]
fn host_staging_vec_rejects_quota_and_size_overflow() {
    let ledger = Arc::new(BackendAllocationLedger::with_cap(4));
    assert!(matches!(
        try_host_staging_vec::<f64>(&ledger, 1, "test_staging"),
        Err(BackendError::Allocation { .. })
    ));
    assert_eq!(ledger.live_bytes.load(Ordering::Acquire), 0);
    assert!(matches!(
        try_host_staging_vec::<f64>(&ledger, usize::MAX, "test_staging"),
        Err(BackendError::InvalidArgument { .. })
    ));
    assert_eq!(ledger.live_bytes.load(Ordering::Acquire), 0);
}

#[test]
fn checked_legacy_backend_panic_becomes_operation_error() {
    let error = catch_backend_panic("test_operation", || -> () {
        panic!("simulated backend failure");
    })
    .expect_err("checked backend wrapper must not propagate the panic");

    assert!(matches!(
        error,
        BackendError::OperationFailed {
            operation: "test_operation",
            ..
        }
    ));
}

#[test]
fn checked_reductions_validate_shapes_before_dispatch() {
    let backend = crate::cpu::CpuBackend::new();
    let x = backend.raw_alloc::<f64>(2).expect("x allocation");
    let y = backend.raw_alloc::<f64>(3).expect("y allocation");

    let dot_error = backend
        .try_dot_with_mode(ReductionMode::Fast, &x, &y)
        .expect_err("checked dot must reject mismatched vectors");
    assert!(matches!(dot_error, BackendError::InvalidArgument { .. }));

    let multi_error = backend
        .try_multi_dot_with_mode(ReductionMode::Fast, &[&x], &y)
        .expect_err("checked multi-dot must reject mismatched vectors");
    assert!(matches!(multi_error, BackendError::InvalidArgument { .. }));
}

#[test]
fn checked_multi_dot_into_reuses_caller_storage_and_rejects_wrong_shape() {
    let backend = crate::cpu::CpuBackend::new();
    let mut x0 = backend.raw_alloc::<f64>(3).expect("x0 allocation");
    let mut x1 = backend.raw_alloc::<f64>(3).expect("x1 allocation");
    let mut y = backend.raw_alloc::<f64>(3).expect("y allocation");
    backend.copy_h2d(&[1.0, 2.0, 3.0], &mut x0);
    backend.copy_h2d(&[-2.0, 1.0, 0.5], &mut x1);
    backend.copy_h2d(&[0.5, -1.0, 2.0], &mut y);

    let basis = [&x0, &x1];
    let mut output = [f64::NAN; 2];
    backend
        .try_multi_dot_with_mode_into(ReductionMode::Deterministic, &basis, &y, &mut output)
        .expect("caller-owned output must be populated");
    assert_eq!(output, [4.5, -1.0]);

    let error = backend
        .try_multi_dot_with_mode_into(ReductionMode::Fast, &basis, &y, &mut [0.0; 1])
        .expect_err("wrong output shape must be rejected before dispatch");
    assert!(matches!(error, BackendError::InvalidArgument { .. }));
}

#[test]
fn checked_multi_dot_into_preserves_output_when_a_later_basis_fails() {
    let backend = crate::cpu::CpuBackend::new();
    let mut first = backend.raw_alloc::<f64>(2).expect("first allocation");
    let mut short = backend.raw_alloc::<f64>(1).expect("short allocation");
    let mut y = backend.raw_alloc::<f64>(2).expect("rhs allocation");
    backend.copy_h2d(&[2.0, 3.0], &mut first);
    backend.copy_h2d(&[4.0], &mut short);
    backend.copy_h2d(&[5.0, 7.0], &mut y);

    let mut output = [11.0, 13.0];
    let error = backend
        .try_multi_dot_with_mode_into(
            ReductionMode::Deterministic,
            &[&first, &short],
            &y,
            &mut output,
        )
        .expect_err("a mismatched later basis vector must fail");
    assert!(matches!(
        error,
        BackendError::InvalidArgument {
            operation: "dot",
            ..
        }
    ));
    assert_eq!(
        output,
        [11.0, 13.0],
        "failed reductions must not partially mutate output"
    );
}

#[test]
fn checked_multi_dot_into_covers_all_device_scalar_families() {
    fn check<T>(
        backend: &crate::cpu::CpuBackend,
        x0_host: &[T],
        x1_host: &[T],
        y_host: &[T],
        expected: [T; 2],
    ) where
        T: DeviceScalar + PartialEq + std::fmt::Debug,
    {
        let mut x0 = backend
            .raw_alloc::<T>(x0_host.len())
            .expect("x0 allocation");
        let mut x1 = backend
            .raw_alloc::<T>(x1_host.len())
            .expect("x1 allocation");
        let mut y = backend.raw_alloc::<T>(y_host.len()).expect("y allocation");
        backend.copy_h2d(x0_host, &mut x0);
        backend.copy_h2d(x1_host, &mut x1);
        backend.copy_h2d(y_host, &mut y);
        let basis = [&x0, &x1];
        let mut output = [T::zero(); 2];
        backend
            .try_multi_dot_with_mode_into(ReductionMode::Deterministic, &basis, &y, &mut output)
            .expect("caller-owned output must be populated");
        assert_eq!(output, expected);
    }

    let backend = crate::cpu::CpuBackend::new();
    check(
        &backend,
        &[1.0_f32, 2.0, 3.0],
        &[-2.0, 1.0, 0.5],
        &[0.5, -1.0, 2.0],
        [4.5, -1.0],
    );
    check(
        &backend,
        &[1.0_f64, 2.0, 3.0],
        &[-2.0, 1.0, 0.5],
        &[0.5, -1.0, 2.0],
        [4.5, -1.0],
    );
    check(
        &backend,
        &[
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
        ],
        &[
            Complex32::new(-2.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(0.5, 0.0),
        ],
        &[
            Complex32::new(0.5, 0.0),
            Complex32::new(-1.0, 0.0),
            Complex32::new(2.0, 0.0),
        ],
        [Complex32::new(4.5, 0.0), Complex32::new(-1.0, 0.0)],
    );
    check(
        &backend,
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ],
        &[
            Complex64::new(-2.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.0),
        ],
        &[
            Complex64::new(0.5, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(2.0, 0.0),
        ],
        [Complex64::new(4.5, 0.0), Complex64::new(-1.0, 0.0)],
    );
}

#[test]
fn checked_axpy_many_validates_coefficients_and_basis_shapes() {
    let backend = crate::cpu::CpuBackend::new();
    let x = backend.raw_alloc::<f64>(2).expect("x allocation");
    let short = backend.raw_alloc::<f64>(1).expect("short allocation");
    let mut y = backend.raw_alloc::<f64>(2).expect("y allocation");

    let error = backend
        .try_axpy_many(&[1.0, 2.0], &[&x], &mut y)
        .expect_err("coefficient/basis mismatch must be rejected");
    assert!(matches!(error, BackendError::InvalidArgument { .. }));

    let error = backend
        .try_axpy_many(&[1.0], &[&short], &mut y)
        .expect_err("basis/output mismatch must be rejected");
    assert!(matches!(error, BackendError::InvalidArgument { .. }));
}

#[test]
fn checked_cpu_axpy_many_matches_independent_linear_combination() {
    let backend = crate::cpu::CpuBackend::new();
    let mut x0 = backend.raw_alloc::<f64>(3).expect("x0 allocation");
    let mut x1 = backend.raw_alloc::<f64>(3).expect("x1 allocation");
    let mut y = backend.raw_alloc::<f64>(3).expect("y allocation");
    backend.copy_h2d(&[1.0, -2.0, 3.0], &mut x0);
    backend.copy_h2d(&[4.0, 5.0, -6.0], &mut x1);
    backend.copy_h2d(&[0.5, 1.5, -2.0], &mut y);

    backend
        .try_axpy_many(&[2.0, -0.25], &[&x0, &x1], &mut y)
        .expect("CPU checked axpy_many");

    let mut actual = [0.0; 3];
    backend.copy_d2h(&y, &mut actual);
    assert_eq!(actual, [1.5, -3.75, 5.5]);

    let error = backend
        .try_axpy_many(&[f64::NAN], &[&x0], &mut y)
        .expect_err("non-finite axpy_many coefficients must be rejected");
    assert!(matches!(
        error,
        BackendError::OperationFailed {
            operation: "axpy_many",
            ..
        }
    ));
}

#[test]
fn checked_batched_blas_validates_every_item_before_dispatch() {
    let backend = crate::cpu::CpuBackend::new();
    let a = backend.raw_alloc::<f64>(4).expect("A allocation");
    let x = backend.raw_alloc::<f64>(2).expect("x allocation");
    let mut y = backend.raw_alloc::<f64>(2).expect("y allocation");
    let mut gemv_batch = [GemvBatch {
        op: Transpose::NoTrans,
        m: 3,
        n: 2,
        alpha: 1.0,
        a: &a,
        lda: 2,
        x: &x,
        beta: 0.0,
        y: &mut y,
    }];
    assert!(matches!(
        backend.try_gemv_batched(&mut gemv_batch),
        Err(BackendError::InvalidArgument {
            operation: "gemv",
            ..
        })
    ));

    let b = backend.raw_alloc::<f64>(4).expect("B allocation");
    let mut c = backend.raw_alloc::<f64>(4).expect("C allocation");
    let mut gemm_batch = [GemmBatch {
        transa: Transpose::NoTrans,
        transb: Transpose::NoTrans,
        m: 3,
        n: 2,
        k: 2,
        alpha: 1.0,
        a: &a,
        lda: 2,
        b: &b,
        ldb: 2,
        beta: 0.0,
        c: &mut c,
        ldc: 2,
    }];
    assert!(matches!(
        backend.try_gemm_batched(&mut gemm_batch),
        Err(BackendError::InvalidArgument {
            operation: "gemm",
            ..
        })
    ));
}

#[test]
fn checked_spmv_csr_rejects_malformed_metadata_before_prebake() {
    let backend = crate::cpu::CpuBackend::new();
    let mut row_ptr = backend.raw_alloc::<i32>(3).expect("row pointer");
    let mut col_idx = backend.raw_alloc::<i32>(1).expect("column indices");
    let mut values = backend.raw_alloc::<f64>(1).expect("values");
    let x = backend.raw_alloc::<f64>(2).expect("x");
    let mut y = backend.raw_alloc::<f64>(2).expect("y");
    backend.copy_h2d(&[0, 2, 1], &mut row_ptr);
    backend.copy_h2d(&[0], &mut col_idx);
    backend.copy_h2d(&[3.0], &mut values);
    backend.copy_h2d(&[1.0, 2.0], &mut y);
    let error = backend
        .try_spmv_csr(2, 2, &row_ptr, &col_idx, &values, &x, &mut y)
        .expect_err("non-monotone row pointers must be rejected");
    assert!(matches!(error, BackendError::InvalidArgument { .. }));
    let mut unchanged = [0.0; 2];
    backend.copy_d2h(&y, &mut unchanged);
    assert_eq!(unchanged, [1.0, 2.0]);

    backend.copy_h2d(&[0, 1, 1], &mut row_ptr);
    backend.copy_h2d(&[f64::NAN], &mut values);
    let error = backend
        .try_spmv_csr(2, 2, &row_ptr, &col_idx, &values, &x, &mut y)
        .expect_err("non-finite matrix values must be rejected");
    assert!(matches!(
        error,
        BackendError::OperationFailed {
            operation: "spmv_csr",
            ..
        }
    ));
}

#[test]
fn checked_spmv_csr_validates_dimensions_and_computes_result() {
    let backend = crate::cpu::CpuBackend::new();
    let mut row_ptr = backend.raw_alloc::<i32>(3).expect("row pointer");
    let mut col_idx = backend.raw_alloc::<i32>(2).expect("column indices");
    let mut values = backend.raw_alloc::<f64>(2).expect("values");
    let mut x = backend.raw_alloc::<f64>(2).expect("x");
    let mut y = backend.raw_alloc::<f64>(2).expect("y");
    backend.copy_h2d(&[0, 1, 2], &mut row_ptr);
    backend.copy_h2d(&[0, 1], &mut col_idx);
    backend.copy_h2d(&[2.0, 4.0], &mut values);
    backend.copy_h2d(&[3.0, 5.0], &mut x);
    backend.copy_h2d(&[-1.0, -1.0], &mut y);

    backend
        .try_spmv_csr(2, 2, &row_ptr, &col_idx, &values, &x, &mut y)
        .expect("valid checked SpMV");
    let mut result = [0.0; 2];
    backend.copy_d2h(&y, &mut result);
    assert_eq!(result, [6.0, 20.0]);

    let short_x = backend.raw_alloc::<f64>(1).expect("short x");
    assert!(matches!(
        backend.try_spmv_csr(2, 2, &row_ptr, &col_idx, &values, &short_x, &mut y),
        Err(BackendError::InvalidArgument { .. })
    ));
}

#[test]
fn capability_table_declares_new_checked_primitives() {
    let capabilities = crate::cpu::CpuBackend::new().capabilities();
    assert_eq!(
        capabilities.axpy_many.state_for::<f64>(),
        CapabilityState::Native
    );
    assert_eq!(
        capabilities.fft.state_for::<f64>(),
        CapabilityState::Unsupported
    );
    assert_eq!(
        capabilities.segmented_reduce.state_for::<f64>(),
        CapabilityState::Native
    );
    assert_eq!(
        capabilities.batched_local_contract.state_for::<f64>(),
        CapabilityState::Native
    );
    assert_eq!(
        capabilities.bem.near_pair.c64,
        CapabilityState::Native,
        "CPU BEM near-pair reference must be advertised as native"
    );
    assert_eq!(
        capabilities.bem.field_evaluation.c64,
        CapabilityState::Native,
        "CPU BEM field reference must be advertised as native"
    );
}

#[test]
fn multi_dot_capability_includes_reduction_mode_residency() {
    let cpu = crate::cpu::CpuBackend::new().capabilities();
    for mode in [
        ReductionMode::Fast,
        ReductionMode::Reproducible,
        ReductionMode::Deterministic,
    ] {
        assert_eq!(
            cpu.multi_dot_capability_for::<f64>(mode),
            CapabilityState::Native
        );
    }

    let mixed = BackendCapabilities {
        multi_dot: ScalarCapabilities::splat(CapabilityState::Native),
        reduction: ReductionCapabilities::new(
            ScalarCapabilities::splat(CapabilityState::Native),
            ScalarCapabilities::splat(CapabilityState::HostStagedFallback),
        ),
        ..BackendCapabilities::unsupported("mixed")
    };
    assert_eq!(
        mixed.multi_dot_capability_for::<f64>(ReductionMode::Fast),
        CapabilityState::Native
    );
    assert_eq!(
        mixed.multi_dot_capability_for::<f64>(ReductionMode::Reproducible),
        CapabilityState::HostStagedFallback
    );
    assert_eq!(
        mixed.multi_dot_capability_for::<f64>(ReductionMode::Deterministic),
        CapabilityState::HostStagedFallback
    );
}

#[test]
fn capability_states_distinguish_native_vendor_and_host_fallbacks() {
    assert!(CapabilityState::Native.is_native());
    assert!(CapabilityState::VendorNative.is_native());
    assert!(CapabilityState::Native.is_available());
    assert!(CapabilityState::VendorNative.is_available());
    assert!(!CapabilityState::Native.uses_host_fallback());
    assert!(!CapabilityState::VendorNative.uses_host_fallback());

    for state in [
        CapabilityState::HostSharedFallback,
        CapabilityState::HostStagedFallback,
    ] {
        assert!(!state.is_native());
        assert!(state.is_available());
        assert!(state.uses_host_fallback());
    }
    assert!(!CapabilityState::Unsupported.is_native());
    assert!(!CapabilityState::Unsupported.is_available());
    assert!(!CapabilityState::Unsupported.uses_host_fallback());
}

#[test]
fn checked_strided_span_validates_padded_layouts_and_overflow() {
    assert_eq!(checked_strided_span("test", 0, 0, 0).unwrap(), 0);
    assert_eq!(checked_strided_span("test", 3, 10, 4).unwrap(), 24);

    assert!(matches!(
        checked_strided_span("test", 2, 3, 4),
        Err(BackendError::InvalidArgument {
            operation: "test",
            ..
        })
    ));
    assert!(matches!(
        checked_strided_span("test", 2, usize::MAX, 1),
        Err(BackendError::InvalidArgument {
            operation: "test",
            ..
        })
    ));
}

#[test]
fn checked_axpy_many_rejects_unsupported_mock_capability() {
    let capabilities = BackendCapabilities::unsupported("test-backend");
    let error = require_scalar_capability::<f64>(
        capabilities.backend_name,
        "axpy_many",
        capabilities.axpy_many,
    )
    .expect_err("unsupported axpy_many capability must fail before dispatch");
    assert!(matches!(
        error,
        BackendError::Unsupported {
            backend: "test-backend",
            operation: "axpy_many",
            ..
        }
    ));
}

#[test]
fn reduction_capabilities_expose_order_and_accumulation_policy() {
    let capabilities = crate::cpu::CpuBackend::new().capabilities().reduction;
    assert_eq!(
        capabilities.policy_for(ReductionMode::Fast),
        ReductionPolicy::FAST
    );
    assert_eq!(
        capabilities.policy_for(ReductionMode::Reproducible),
        ReductionPolicy::REPRODUCIBLE
    );
    assert_eq!(
        capabilities.policy_for(ReductionMode::Deterministic),
        ReductionPolicy::DETERMINISTIC
    );
    assert_eq!(
        capabilities
            .for_mode(ReductionMode::Reproducible)
            .state_for::<f64>(),
        capabilities
            .for_mode(ReductionMode::Deterministic)
            .state_for::<f64>()
    );
    assert_eq!(
        capabilities
            .policy_for(ReductionMode::Deterministic)
            .ordering,
        ReductionOrdering::FixedInputOrder
    );
    assert_eq!(
        capabilities
            .policy_for(ReductionMode::Deterministic)
            .accumulation,
        ReductionAccumulation::HostScalar
    );
    assert_eq!(
        capabilities.policy_for(ReductionMode::Fast).synchronization,
        ReductionSynchronization::BackendDefined
    );
    assert_eq!(
        capabilities
            .policy_for(ReductionMode::Deterministic)
            .synchronization,
        ReductionSynchronization::HostTransfer
    );
    assert_eq!(
        capabilities
            .policy_for(ReductionMode::Deterministic)
            .failure,
        ReductionFailureGuarantee::TypedError
    );
}

#[test]
fn unsupported_scalar_capability_is_typed_before_dispatch() {
    let capabilities = BackendCapabilities::unsupported("test-backend");
    let error =
        require_scalar_capability::<f64>(capabilities.backend_name, "gemv", capabilities.gemv)
            .expect_err("unsupported capability must fail before dispatch");
    assert!(matches!(
        error,
        BackendError::Unsupported {
            backend: "test-backend",
            operation: "gemv",
            ..
        }
    ));
}

#[test]
fn checked_bem_dispatch_rejects_host_fallback() {
    let capabilities = BackendCapabilities {
        bem_regular_pair_c32: CapabilityState::HostStagedFallback,
        ..BackendCapabilities::unsupported("test-backend")
    };
    let error = capabilities
        .require_native_bem_regular_pair()
        .expect_err("host fallback must not satisfy native BEM dispatch");
    assert!(matches!(error, BackendError::HostFallback { .. }));
}

#[test]
fn checked_bem_dispatch_accepts_native_capability() {
    let capabilities = BackendCapabilities {
        bem_regular_pair_c32: CapabilityState::Native,
        ..BackendCapabilities::unsupported("test-backend")
    };
    capabilities
        .require_native_bem_regular_pair()
        .expect("native BEM dispatch is available");
}

#[test]
fn checked_bem_batch_rejects_nonfinite_inputs_and_preserves_shape() {
    let backend = crate::cpu::CpuBackend::new();
    let pair = BemRegularPairIn::from_triangle(
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.0, 0.5, 0.0],
    );
    let output = backend
        .try_bem_regular_pair_c32_batch(&[pair], 2.0, -1.0)
        .expect("finite CPU BEM batch should succeed");
    assert_eq!(output.len(), 1);

    let mut invalid = pair;
    invalid.v1[0] = f32::NAN;
    let error = backend
        .try_bem_regular_pair_c32_batch(&[invalid], 2.0, -1.0)
        .expect_err("non-finite pair coordinates must be rejected");
    assert!(matches!(
        error,
        BackendError::OperationFailed {
            operation: "bem_regular_pair_c32_batch",
            ..
        }
    ));

    let error = backend
        .try_bem_regular_pair_c32_batch(&[pair], f32::INFINITY, -1.0)
        .expect_err("non-finite wave number must be rejected");
    assert!(matches!(
        error,
        BackendError::OperationFailed {
            operation: "bem_regular_pair_c32_batch",
            ..
        }
    ));
}

#[test]
fn checked_strided_views_copy_only_validated_elements() {
    let backend = crate::cpu::CpuBackend::new();
    let mut buffer = backend.raw_alloc::<f64>(8).expect("allocation");
    backend.copy_h2d(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &mut buffer);
    let view = backend
        .checked_view(&buffer, 1, 3, 2)
        .expect("valid strided view");
    let mut selected = [0.0; 3];
    backend
        .try_copy_view_d2h(&buffer, view, &mut selected)
        .expect("view download");
    assert_eq!(selected, [1.0, 3.0, 5.0]);

    backend
        .try_copy_view_h2d(&[10.0, 30.0, 50.0], &mut buffer, view)
        .expect("view upload");
    let mut full = [0.0; 8];
    backend.copy_d2h(&buffer, &mut full);
    assert_eq!(full, [0.0, 10.0, 2.0, 30.0, 4.0, 50.0, 6.0, 7.0]);
    assert!(matches!(
        backend.try_copy_view_d2h(&buffer, view, &mut [0.0; 2]),
        Err(BackendError::InvalidArgument {
            operation: "copy_view_d2h",
            ..
        })
    ));
}

#[test]
fn empty_views_are_valid_and_touch_no_storage() {
    let backend = crate::cpu::CpuBackend::new();
    let mut buffer = backend.raw_alloc::<f64>(4).expect("allocation");
    backend.copy_h2d(&[1.0, 2.0, 3.0, 4.0], &mut buffer);
    let view = BufferView::new(4, 4, 0, 7).expect("empty view at buffer end");
    let descriptor = view.descriptor::<f64>("cpu").expect("descriptor");
    assert_eq!(descriptor.rank, 1);
    assert_eq!(descriptor.shape, vec![0]);
    assert_eq!(descriptor.byte_offset, 4 * std::mem::size_of::<f64>());
    assert_eq!(descriptor.element_strides, vec![7]);
    assert_eq!(descriptor.layout, ViewLayout::Strided);
    assert_eq!(descriptor.dtype, "f64");
    assert_eq!(descriptor.device, "cpu");
    assert_eq!(descriptor.aliasing, ViewAliasing::ReadOnly);

    backend
        .try_copy_view_d2h(&buffer, view, &mut [])
        .expect("empty view download");
    backend
        .try_copy_view_h2d(&[], &mut buffer, view)
        .expect("empty view upload");
    let mut values = [0.0; 4];
    backend.copy_d2h(&buffer, &mut values);
    assert_eq!(values, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn view_descriptor_rejects_invalid_public_metadata() {
    let valid = BufferView::contiguous(0, 4)
        .descriptor::<f64>("cpu")
        .expect("valid descriptor");

    let mut wrong_rank = valid.clone();
    wrong_rank.rank = 2;
    assert!(matches!(
        wrong_rank.validate(),
        Err(BackendError::InvalidArgument {
            operation: "view_descriptor",
            ..
        })
    ));

    let mut wrong_stride_rank = valid.clone();
    wrong_stride_rank.element_strides.clear();
    assert!(matches!(
        wrong_stride_rank.validate(),
        Err(BackendError::InvalidArgument {
            operation: "view_descriptor",
            ..
        })
    ));

    let mut zero_stride = valid.clone();
    zero_stride.element_strides[0] = 0;
    assert!(matches!(
        zero_stride.validate(),
        Err(BackendError::InvalidArgument {
            operation: "view_descriptor",
            ..
        })
    ));

    let mut wrong_dtype = valid.clone();
    wrong_dtype.dtype = "f32";
    assert!(matches!(
        wrong_dtype.validate_for::<f64>("cpu"),
        Err(BackendError::InvalidArgument {
            operation: "view_descriptor",
            ..
        })
    ));

    let mut wrong_device = valid.clone();
    wrong_device.device = "metal";
    assert!(matches!(
        wrong_device.validate_for::<f64>("cpu"),
        Err(BackendError::InvalidArgument {
            operation: "view_descriptor",
            ..
        })
    ));

    let mut misaligned = valid;
    misaligned.byte_offset = 1;
    assert!(matches!(
        misaligned.validate_for::<f64>("cpu"),
        Err(BackendError::InvalidArgument {
            operation: "view_descriptor",
            ..
        })
    ));

    let matrix = ViewDescriptor {
        rank: 2,
        shape: vec![2, 3],
        byte_offset: 8,
        element_strides: vec![3, 1],
        layout: ViewLayout::RowMajor,
        dtype: "f64",
        device: "cpu",
        aliasing: ViewAliasing::ReadOnly,
    };
    assert_eq!(
        matrix
            .checked_byte_end_exclusive::<f64>("cpu")
            .expect("matrix span"),
        8 + 5 * std::mem::size_of::<f64>() + std::mem::size_of::<f64>()
    );

    let overflowing_matrix = ViewDescriptor {
        rank: 2,
        shape: vec![usize::MAX, 2],
        byte_offset: 0,
        element_strides: vec![2, 1],
        layout: ViewLayout::RowMajor,
        dtype: "f64",
        device: "cpu",
        aliasing: ViewAliasing::ReadOnly,
    };
    assert!(matches!(
        overflowing_matrix.checked_byte_end_exclusive::<f64>("cpu"),
        Err(BackendError::InvalidArgument {
            operation: "view_descriptor",
            ..
        })
    ));
}

#[test]
fn view_layout_metadata_is_checked_before_binding() {
    let row_major = ViewDescriptor {
        rank: 2,
        shape: vec![2, 3],
        byte_offset: 0,
        element_strides: vec![3, 1],
        layout: ViewLayout::RowMajor,
        dtype: "f64",
        device: "cpu",
        aliasing: ViewAliasing::ReadOnly,
    };
    assert!(row_major.validate().is_ok());

    let column_major = ViewDescriptor {
        element_strides: vec![1, 2],
        layout: ViewLayout::ColumnMajor,
        ..row_major.clone()
    };
    assert!(column_major.validate().is_ok());

    let mut mislabeled = row_major;
    mislabeled.layout = ViewLayout::ColumnMajor;
    assert!(matches!(
        mislabeled.validate(),
        Err(BackendError::InvalidArgument {
            operation: "view_descriptor",
            ..
        })
    ));

    let backend = crate::cpu::CpuBackend::new();
    let buffer = backend.raw_alloc::<f64>(6).expect("allocation");
    let descriptor = backend
        .checked_nd_view_descriptor_with_layout(
            &buffer,
            &[2, 3],
            0,
            &[1, 2],
            ViewLayout::ColumnMajor,
            ViewAliasing::ReadOnly,
        )
        .expect("column-major descriptor");
    assert_eq!(descriptor.layout, ViewLayout::ColumnMajor);
}

#[test]
fn checked_fft_view_descriptor_covers_batch_layout_and_rejection_contract() {
    let descriptor =
        checked_fft_view_descriptor::<f64>(32, 8, "cuda").expect("valid batched FFT descriptor");
    assert_eq!(descriptor.rank, 2);
    assert_eq!(descriptor.shape, vec![4, 8]);
    assert_eq!(descriptor.element_strides, vec![8, 1]);
    assert_eq!(descriptor.layout, ViewLayout::RowMajor);
    assert_eq!(descriptor.aliasing, ViewAliasing::OverlapSafe);
    assert_eq!(descriptor.device, "cuda");

    for (buffer_len, fft_len) in [(0, 8), (32, 1), (15, 8)] {
        let result = checked_fft_view_descriptor::<f64>(buffer_len, fft_len, "metal");
        if buffer_len == 15 {
            assert!(matches!(
                result,
                Err(BackendError::InvalidArgument {
                    operation: "fft_view_descriptor",
                    ..
                })
            ));
        } else {
            assert!(result.is_ok());
        }
    }

    assert!(matches!(
        checked_fft_view_descriptor::<f64>(16, 0, "cuda"),
        Err(BackendError::InvalidArgument {
            operation: "fft_view_descriptor",
            ..
        })
    ));

    let mut wrong_device =
        checked_fft_view_descriptor::<f64>(16, 4, "cuda").expect("descriptor for device mutation");
    wrong_device.device = "metal";
    assert!(matches!(
        wrong_device.validate_for_buffer::<f64>(16, "cuda"),
        Err(BackendError::InvalidArgument {
            operation: "view_descriptor",
            ..
        })
    ));
}

#[test]
fn checked_fft_strided_view_descriptor_covers_padding_and_overflow() {
    let descriptor = checked_fft_strided_view_descriptor::<f64>(20, 4, 3, 8, "cuda")
        .expect("valid strided FFT descriptor");
    assert_eq!(descriptor.rank, 2);
    assert_eq!(descriptor.shape, vec![3, 4]);
    assert_eq!(descriptor.element_strides, vec![8, 1]);
    assert_eq!(descriptor.layout, ViewLayout::Strided);
    assert_eq!(descriptor.device, "cuda");

    assert!(checked_fft_strided_view_descriptor::<f64>(12, 4, 2, 4, "cuda").is_ok());
    assert!(matches!(
        checked_fft_strided_view_descriptor::<f64>(7, 4, 2, 4, "cuda"),
        Err(BackendError::InvalidArgument {
            operation: "fft_strided_view_descriptor",
            ..
        })
    ));
    assert!(matches!(
        checked_fft_strided_view_descriptor::<f64>(usize::MAX, usize::MAX, 2, usize::MAX, "cuda"),
        Err(BackendError::InvalidArgument {
            operation: "fft_strided_view_descriptor",
            ..
        })
    ));
}

#[test]
fn view_descriptor_try_new_validates_owned_metadata() {
    let descriptor = ViewDescriptor::try_new(
        &[2, 3],
        0,
        &[3, 1],
        ViewLayout::RowMajor,
        "f64",
        "cpu",
        ViewAliasing::ReadOnly,
    )
    .expect("valid owned descriptor");
    assert_eq!(descriptor.rank, 2);
    assert_eq!(descriptor.shape, vec![2, 3]);
    assert_eq!(descriptor.element_strides, vec![3, 1]);

    let error = ViewDescriptor::try_new(
        &[2, 3],
        0,
        &[3],
        ViewLayout::RowMajor,
        "f64",
        "cpu",
        ViewAliasing::ReadOnly,
    )
    .expect_err("mismatched stride rank must be rejected");
    assert!(matches!(
        error,
        BackendError::InvalidArgument {
            operation: "view_descriptor",
            ..
        }
    ));
}

#[test]
fn multidimensional_view_descriptor_checks_rank_and_allocation_bounds() {
    let backend = crate::cpu::CpuBackend::new();
    let buffer = backend.raw_alloc::<f64>(6).expect("allocation");

    let descriptor = backend
        .checked_nd_view_descriptor(&buffer, &[2, 3], 0, &[3, 1], ViewAliasing::ReadOnly)
        .expect("valid matrix descriptor");
    assert_eq!(descriptor.rank, 2);
    assert_eq!(descriptor.shape, vec![2, 3]);
    assert_eq!(descriptor.element_strides, vec![3, 1]);

    let error = backend
        .checked_nd_view_descriptor(&buffer, &[2, 3], 0, &[3, 2], ViewAliasing::ReadOnly)
        .expect_err("descriptor span beyond allocation must be rejected");
    assert!(matches!(
        error,
        BackendError::InvalidArgument {
            operation: "view_descriptor",
            ..
        }
    ));

    let error = backend
        .checked_nd_view_descriptor(&buffer, &[2, 3], 0, &[3], ViewAliasing::Disjoint)
        .expect_err("rank/stride mismatch must be rejected");
    assert!(matches!(
        error,
        BackendError::InvalidArgument {
            operation: "view_descriptor",
            ..
        }
    ));
}

#[test]
fn checked_nd_byte_strided_views_copy_in_logical_row_major_order() {
    let backend = crate::cpu::CpuBackend::new();
    let mut buffer = backend.raw_alloc::<f64>(12).expect("allocation");
    backend.copy_h2d(
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        &mut buffer,
    );
    let element_size = std::mem::size_of::<f64>();
    let read_view = backend
        .checked_nd_byte_strided_view_descriptor(
            &buffer,
            &[2, 3],
            element_size,
            &[4 * element_size, element_size],
            ViewAliasing::ReadOnly,
        )
        .expect("padded matrix view");
    assert_eq!(read_view.checked_element_count().expect("element count"), 6);
    assert_eq!(
        read_view
            .checked_element_index::<f64>(4, "cpu")
            .expect("logical index"),
        6
    );
    let mut selected = [0.0; 6];
    backend
        .try_copy_nd_view_d2h(&buffer, &read_view, &mut selected)
        .expect("N-D download");
    assert_eq!(selected, [1.0, 2.0, 3.0, 5.0, 6.0, 7.0]);

    let write_view = backend
        .checked_nd_byte_strided_view_descriptor(
            &buffer,
            &[2, 3],
            element_size,
            &[4 * element_size, element_size],
            ViewAliasing::Disjoint,
        )
        .expect("writable padded matrix view");
    backend
        .try_copy_nd_view_h2d(
            &[10.0, 20.0, 30.0, 50.0, 60.0, 70.0],
            &mut buffer,
            &write_view,
        )
        .expect("N-D upload");
    let mut full = [0.0; 12];
    backend.copy_d2h(&buffer, &mut full);
    assert_eq!(
        full,
        [
            0.0, 10.0, 20.0, 30.0, 4.0, 50.0, 60.0, 70.0, 8.0, 9.0, 10.0, 11.0
        ]
    );

    let mut destination = backend
        .raw_alloc::<f64>(10)
        .expect("destination allocation");
    backend.copy_h2d(&[-1.0; 10], &mut destination);
    let destination_view = backend
        .checked_nd_view_descriptor(
            &destination,
            &[3, 2],
            2 * element_size,
            &[2, 1],
            ViewAliasing::Disjoint,
        )
        .expect("contiguous destination matrix view");
    backend
        .try_copy_nd_views_d2d(&buffer, &read_view, &mut destination, &destination_view)
        .expect("N-D device copy");
    let mut copied = [-1.0; 10];
    backend.copy_d2h(&destination, &mut copied);
    assert_eq!(
        copied,
        [-1.0, -1.0, 10.0, 20.0, 30.0, 50.0, 60.0, 70.0, -1.0, -1.0]
    );

    let transposed = backend
        .checked_nd_byte_strided_view_descriptor(
            &buffer,
            &[3, 2],
            element_size,
            &[element_size, 4 * element_size],
            ViewAliasing::ReadOnly,
        )
        .expect("transposed matrix view");
    let mut selected = [0.0; 6];
    backend
        .try_copy_nd_view_d2h(&buffer, &transposed, &mut selected)
        .expect("transposed N-D download");
    assert_eq!(selected, [10.0, 50.0, 20.0, 60.0, 30.0, 70.0]);
}

#[test]
fn checked_nd_views_reject_bad_byte_strides_and_define_overlap_semantics() {
    let backend = crate::cpu::CpuBackend::new();
    let mut buffer = backend.raw_alloc::<f64>(6).expect("allocation");
    backend.copy_h2d(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &mut buffer);
    let element_size = std::mem::size_of::<f64>();

    assert!(matches!(
        backend.checked_nd_byte_strided_view_descriptor(
            &buffer,
            &[2, 2],
            0,
            &[2 * element_size, element_size + 1],
            ViewAliasing::ReadOnly,
        ),
        Err(BackendError::InvalidArgument {
            operation: "view_descriptor",
            ..
        })
    ));

    let read_only = backend
        .checked_nd_view_descriptor(&buffer, &[2, 2], 0, &[2, 1], ViewAliasing::ReadOnly)
        .expect("read-only matrix view");
    assert!(matches!(
        backend.try_copy_nd_view_h2d(&[9.0; 4], &mut buffer, &read_only),
        Err(BackendError::InvalidArgument {
            operation: "copy_nd_view_h2d",
            ..
        })
    ));

    let destination = backend
        .checked_nd_view_descriptor(
            &buffer,
            &[2, 2],
            element_size,
            &[2, 1],
            ViewAliasing::OverlapSafe,
        )
        .expect("overlap-safe destination");
    backend
        .try_copy_nd_view_in_place(&mut buffer, &read_only, &destination)
        .expect("overlapping N-D copy");
    let mut values = [0.0; 6];
    backend.copy_d2h(&buffer, &mut values);
    assert_eq!(values, [0.0, 0.0, 1.0, 2.0, 3.0, 5.0]);

    let empty = backend
        .checked_nd_view_descriptor(
            &buffer,
            &[2, 0, 3],
            6 * element_size,
            &[3, 3, 1],
            ViewAliasing::ReadOnly,
        )
        .expect("empty descriptor at allocation end");
    backend
        .try_copy_nd_view_d2h(&buffer, &empty, &mut [])
        .expect("empty N-D copy");
}

#[test]
fn checked_nd_view_copies_reject_count_mismatch_without_mutation() {
    let backend = crate::cpu::CpuBackend::new();
    let mut buffer = backend.raw_alloc::<f64>(4).expect("allocation");
    backend.copy_h2d(&[1.0, 2.0, 3.0, 4.0], &mut buffer);
    let descriptor = backend
        .checked_nd_view_descriptor(&buffer, &[2, 2], 0, &[2, 1], ViewAliasing::OverlapSafe)
        .expect("valid matrix descriptor");

    let mut host = [9.0; 3];
    let error = backend
        .try_copy_nd_view_d2h(&buffer, &descriptor, &mut host)
        .expect_err("short host destination must be rejected");
    assert!(matches!(
        error,
        BackendError::InvalidArgument {
            operation: "copy_nd_view_d2h",
            ..
        }
    ));
    assert_eq!(host, [9.0; 3]);

    let before = [1.0, 2.0, 3.0, 4.0];
    let error = backend
        .try_copy_nd_view_h2d(&[8.0; 3], &mut buffer, &descriptor)
        .expect_err("short host source must be rejected");
    assert!(matches!(
        error,
        BackendError::InvalidArgument {
            operation: "copy_nd_view_h2d",
            ..
        }
    ));
    let mut after = [0.0; 4];
    backend.copy_d2h(&buffer, &mut after);
    assert_eq!(after, before);
}

#[test]
fn views_report_overflow_and_exact_overlap_without_index_access() {
    let strided = BufferView::new(16, 1, 3, 2).expect("valid strided view");
    let overlapping = BufferView::new(16, 5, 1, 1).expect("valid overlap");
    let disjoint = BufferView::new(16, 8, 2, 1).expect("valid disjoint view");
    assert!(strided.may_overlap(&overlapping).expect("overlap check"));
    assert!(!strided.may_overlap(&disjoint).expect("disjoint check"));
    assert_eq!(strided.checked_element_index(2).expect("last element"), 5);
    assert!(matches!(
        strided.checked_element_index(3),
        Err(BackendError::InvalidArgument {
            operation: "buffer_view",
            ..
        })
    ));
    let invalid_public_view = BufferView {
        offset: usize::MAX,
        len: 2,
        stride: usize::MAX,
    };
    assert!(matches!(
        invalid_public_view.checked_end_exclusive(),
        Err(BackendError::InvalidArgument {
            operation: "buffer_view",
            ..
        })
    ));
    assert!(matches!(
        (BufferView {
            offset: 0,
            len: 1,
            stride: 0,
        })
        .descriptor::<f64>("cpu"),
        Err(BackendError::InvalidArgument {
            operation: "buffer_view",
            ..
        })
    ));
}

#[test]
fn strided_views_report_exact_element_overlap() {
    let even = BufferView::new(16, 0, 4, 2).expect("valid even view");
    let odd = BufferView::new(16, 1, 4, 2).expect("valid odd view");
    assert!(!even.may_overlap(&odd).expect("even/odd overlap check"));

    let sparse = BufferView::new(32, 0, 4, 4).expect("valid sparse view");
    let disjoint = BufferView::new(32, 1, 3, 6).expect("valid disjoint view");
    assert!(
        !sparse
            .may_overlap(&disjoint)
            .expect("different-stride disjoint check")
    );

    let overlapping = BufferView::new(32, 2, 3, 6).expect("valid overlapping view");
    assert!(
        sparse
            .may_overlap(&overlapping)
            .expect("different-stride overlap check")
    );
    assert!(sparse.may_overlap(&sparse).expect("identical-view check"));
}

#[test]
fn overlapping_in_place_view_copy_has_memmove_semantics() {
    let backend = crate::cpu::CpuBackend::new();
    let mut buffer = backend.raw_alloc::<f64>(6).expect("allocation");
    backend.copy_h2d(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &mut buffer);
    backend
        .try_copy_view_in_place(
            &mut buffer,
            BufferView::contiguous(0, 4),
            BufferView::contiguous(1, 4),
        )
        .expect("overlapping view copy");
    let mut values = [0.0; 6];
    backend.copy_d2h(&buffer, &mut values);
    assert_eq!(values, [0.0, 0.0, 1.0, 2.0, 3.0, 5.0]);
}

#[test]
fn strided_device_view_copy_stages_separate_buffers() {
    let backend = crate::cpu::CpuBackend::new();
    let mut source = backend.raw_alloc::<f64>(8).expect("source allocation");
    let mut destination = backend.raw_alloc::<f64>(8).expect("destination allocation");
    backend.copy_h2d(
        &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        &mut source,
    );
    backend.copy_h2d(&[99.0; 8], &mut destination);

    backend
        .try_copy_views_d2d(
            &source,
            BufferView::new(8, 1, 3, 2).expect("source view"),
            &mut destination,
            BufferView::new(8, 0, 3, 2).expect("destination view"),
        )
        .expect("strided view copy");

    let mut values = [0.0; 8];
    backend.copy_d2h(&destination, &mut values);
    assert_eq!(values, [11.0, 99.0, 13.0, 99.0, 15.0, 99.0, 99.0, 99.0]);
}

#[test]
fn buffer_view_rejects_index_arithmetic_overflow() {
    let error = BufferView::new(usize::MAX, usize::MAX - 1, 2, usize::MAX)
        .expect_err("overflowing view coordinates must be rejected");
    assert!(matches!(
        error,
        BackendError::InvalidArgument {
            operation: "buffer_view",
            ..
        }
    ));
}

#[test]
fn checked_batched_local_contract_matches_host_reference() {
    let backend = crate::cpu::CpuBackend::new();
    let mut lhs = backend.raw_alloc::<f64>(8).expect("lhs allocation");
    let mut rhs = backend.raw_alloc::<f64>(8).expect("rhs allocation");
    let mut output = backend.raw_alloc::<f64>(8).expect("output allocation");
    backend.copy_h2d(&[1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0, 3.0], &mut lhs);
    backend.copy_h2d(&[5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0], &mut rhs);
    backend.copy_h2d(&[99.0; 8], &mut output);

    backend
        .try_batched_local_contract(&lhs, &rhs, &mut output, 2, 2, 2, 2)
        .expect("valid local contraction");
    let mut host_output = [0.0; 8];
    backend.copy_d2h(&output, &mut host_output);
    assert_eq!(host_output, [19.0, 22.0, 43.0, 50.0, 2.0, 4.0, 10.0, 14.0]);

    let error = backend
        .try_batched_local_contract(&lhs, &rhs, &mut output, 1, 2, 2, 2)
        .expect_err("shape mismatch must be typed");
    assert!(matches!(
        error,
        BackendError::InvalidArgument {
            operation: "batched_local_contract",
            ..
        }
    ));
}

#[test]
fn checked_segmented_sum_has_deterministic_overwrite_and_validation() {
    let backend = crate::cpu::CpuBackend::new();
    let mut values = backend.raw_alloc::<f64>(4).expect("values allocation");
    let mut offsets = backend.raw_alloc::<i32>(5).expect("offset allocation");
    let mut output = backend.raw_alloc::<f64>(4).expect("output allocation");
    backend.copy_h2d(&[1.0, 2.0, 3.0, 4.0], &mut values);
    backend.copy_h2d(&[0, 2, 2, 3, 4], &mut offsets);
    backend.copy_h2d(&[99.0; 4], &mut output);

    backend
        .try_segmented_sum(&values, &offsets, &mut output)
        .expect("valid segmented reduction");
    let mut host_output = [0.0; 4];
    backend.copy_d2h(&output, &mut host_output);
    assert_eq!(host_output, [3.0, 0.0, 3.0, 4.0]);

    backend.copy_h2d(&[0, 3, 2, 4, 4], &mut offsets);
    assert!(matches!(
        backend.try_segmented_sum(&values, &offsets, &mut output),
        Err(BackendError::InvalidArgument {
            operation: "segmented_sum",
            ..
        })
    ));
}

#[test]
fn deterministic_segmented_sum_is_reproducible_and_preserves_output_on_failure() {
    let backend = crate::cpu::CpuBackend::new();
    let mut values = backend.raw_alloc::<f64>(3).expect("values allocation");
    let mut offsets = backend.raw_alloc::<i32>(2).expect("offset allocation");
    let mut output = backend.raw_alloc::<f64>(1).expect("output allocation");
    backend.copy_h2d(&[1.0e16, 1.0, -1.0e16], &mut values);
    backend.copy_h2d(&[0, 3], &mut offsets);

    let mut expected_bits = None;
    for _ in 0..16 {
        backend
            .try_segmented_sum_with_mode(
                ReductionMode::Deterministic,
                &values,
                &offsets,
                &mut output,
            )
            .expect("deterministic segmented reduction");
        let mut host_output = [0.0];
        backend.copy_d2h(&output, &mut host_output);
        let bits = host_output[0].to_bits();
        assert_eq!(*expected_bits.get_or_insert(bits), bits);
    }
    backend
        .try_segmented_sum_with_mode(ReductionMode::Reproducible, &values, &offsets, &mut output)
        .expect("reproducible segmented reduction");
    let mut reproducible_output = [0.0];
    backend.copy_d2h(&output, &mut reproducible_output);
    assert_eq!(Some(reproducible_output[0].to_bits()), expected_bits);

    backend.copy_h2d(&[42.0], &mut output);
    backend.copy_h2d(&[f64::NAN, 1.0, 2.0], &mut values);
    assert!(matches!(
        backend.try_segmented_sum_with_mode(
            ReductionMode::Deterministic,
            &values,
            &offsets,
            &mut output,
        ),
        Err(BackendError::OperationFailed {
            operation: "segmented_sum",
            ..
        })
    ));
    let mut host_output = [0.0];
    backend.copy_d2h(&output, &mut host_output);
    assert_eq!(host_output, [42.0]);

    let mut overflow_values = backend
        .raw_alloc::<f64>(2)
        .expect("overflow values allocation");
    backend.copy_h2d(&[f64::MAX, f64::MAX], &mut overflow_values);
    backend.copy_h2d(&[0, 2], &mut offsets);
    assert!(matches!(
        backend.try_segmented_sum_with_mode(
            ReductionMode::Deterministic,
            &overflow_values,
            &offsets,
            &mut output,
        ),
        Err(BackendError::OperationFailed {
            operation: "segmented_sum",
            ..
        })
    ));
    backend.copy_d2h(&output, &mut host_output);
    assert_eq!(host_output, [42.0]);
}

#[test]
fn csr_row_bucket_stats_reject_malformed_row_pointers_without_panicking() {
    assert_eq!(
        CsrRowBucketStats::try_from_row_ptr(&[]).unwrap(),
        CsrRowBucketStats::default()
    );
    for row_ptr in [&[1, 1][..], &[0, 3, 2][..], &[0, i32::MAX, i32::MIN][..]] {
        let error = CsrRowBucketStats::try_from_row_ptr(row_ptr)
            .expect_err("malformed row pointers must be rejected");
        assert!(matches!(error, BackendError::InvalidArgument { .. }));
        assert_eq!(
            CsrRowBucketStats::from_row_ptr(row_ptr),
            CsrRowBucketStats::default()
        );
    }

    let stats = CsrRowBucketStats::try_from_row_ptr(&[0, 0, 3, 11, 80]).unwrap();
    assert_eq!(stats.rows, 4);
    assert_eq!(stats.nnz, 80);
    assert_eq!(stats.empty_rows, 1);
    assert_eq!(stats.short_rows, 1);
    assert_eq!(stats.simd_rows, 1);
    assert_eq!(stats.cta_rows, 1);
}

#[test]
fn cpu_compatibility_multi_dot_paths_preserve_results_and_reject_bad_shapes() {
    let backend = crate::cpu::CpuBackend::new();
    let mut y = backend.raw_alloc::<f64>(2).expect("y allocation");
    backend.copy_h2d(&[5.0, 6.0], &mut y);

    let mut first = backend.raw_alloc::<f64>(2).expect("first basis allocation");
    let mut second = backend
        .raw_alloc::<f64>(2)
        .expect("second basis allocation");
    backend.copy_h2d(&[1.0, 2.0], &mut first);
    backend.copy_h2d(&[3.0, 4.0], &mut second);
    let basis_refs = [&first, &second];
    assert_eq!(backend.multi_dot(&basis_refs, &y), vec![17.0, 39.0]);
    assert_eq!(
        backend.multi_dot_with_mode(ReductionMode::Deterministic, &basis_refs, &y),
        vec![17.0, 39.0]
    );

    let mut flat = backend.raw_alloc::<f64>(4).expect("flat basis allocation");
    backend.copy_h2d(&[1.0, 2.0, 3.0, 4.0], &mut flat);
    assert_eq!(backend.multi_dot_strided(&flat, 2, 2, &y), vec![17.0, 39.0]);

    let mut bad = backend.raw_alloc::<f64>(1).expect("bad basis allocation");
    backend.copy_h2d(&[1.0], &mut bad);
    let bad_refs = [&bad];
    assert!(backend.multi_dot(&bad_refs, &y).is_empty());

    backend.copy_h2d(&[1.0, 2.0, 3.0, 4.0], &mut flat);
    backend.copy_h2d(&[2.0, 3.0], &mut first);
    backend
        .try_basis_axpy_strided_device(&flat, 2, 2, &first, &mut y)
        .expect("device coefficient update should use the checked fallback");
    let mut updated = [0.0; 2];
    backend.copy_d2h(&y, &mut updated);
    assert_eq!(updated, [16.0, 22.0]);
}

#[test]
fn cpu_checked_csr_prebake_rejects_row_bucket_structure_before_dispatch() {
    let backend = crate::cpu::CpuBackend::new();
    let mut row_ptr = backend.raw_alloc::<i32>(3).unwrap();
    let mut col_idx = backend.raw_alloc::<i32>(1).unwrap();
    let mut values = backend.raw_alloc::<f64>(1).unwrap();
    backend.copy_h2d(&[0, 1, 0], &mut row_ptr);
    backend.copy_h2d(&[0], &mut col_idx);
    backend.copy_h2d(&[1.0], &mut values);

    assert!(matches!(
        backend.try_prebake_csr(2, 1, &row_ptr, &col_idx, &values),
        Err(BackendError::InvalidArgument { .. })
    ));
}

#[test]
fn checked_csr_transpose_preserves_stable_row_order_and_output_on_error() {
    let backend = crate::cpu::CpuBackend::new();
    assert_eq!(
        backend.capabilities().csr_transpose.f64,
        CapabilityState::Native
    );
    let mut row_ptr = backend.raw_alloc::<i32>(3).expect("row pointer");
    let mut col_idx = backend.raw_alloc::<i32>(3).expect("column indices");
    let mut values = backend.raw_alloc::<f64>(3).expect("values");
    let mut out_row_ptr = backend.raw_alloc::<i32>(4).expect("output row pointer");
    let mut out_col_idx = backend.raw_alloc::<i32>(3).expect("output row indices");
    let mut out_values = backend.raw_alloc::<f64>(3).expect("output values");
    backend.copy_h2d(&[0, 2, 3], &mut row_ptr);
    backend.copy_h2d(&[2, 0, 1], &mut col_idx);
    backend.copy_h2d(&[10.0, 20.0, 30.0], &mut values);
    backend.copy_h2d(&[91, 92, 93, 94], &mut out_row_ptr);
    backend.copy_h2d(&[81, 82, 83], &mut out_col_idx);
    backend.copy_h2d(&[71.0, 72.0, 73.0], &mut out_values);

    backend
        .try_csr_transpose(
            2,
            3,
            &row_ptr,
            &col_idx,
            &values,
            &mut out_row_ptr,
            &mut out_col_idx,
            &mut out_values,
        )
        .expect("valid CSR transpose");
    let mut host_out_row_ptr = [0; 4];
    let mut host_out_col_idx = [0; 3];
    let mut host_out_values = [0.0; 3];
    backend.copy_d2h(&out_row_ptr, &mut host_out_row_ptr);
    backend.copy_d2h(&out_col_idx, &mut host_out_col_idx);
    backend.copy_d2h(&out_values, &mut host_out_values);
    assert_eq!(host_out_row_ptr, [0, 1, 2, 3]);
    assert_eq!(host_out_col_idx, [0, 1, 0]);
    assert_eq!(host_out_values, [20.0, 30.0, 10.0]);

    backend.copy_h2d(&[0, 3, 2], &mut row_ptr);
    let error = backend
        .try_csr_transpose(
            2,
            3,
            &row_ptr,
            &col_idx,
            &values,
            &mut out_row_ptr,
            &mut out_col_idx,
            &mut out_values,
        )
        .expect_err("non-monotone CSR metadata must be rejected");
    assert!(matches!(error, BackendError::InvalidArgument { .. }));
    backend.copy_d2h(&out_row_ptr, &mut host_out_row_ptr);
    assert_eq!(host_out_row_ptr, [0, 1, 2, 3]);
}
