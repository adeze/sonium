use super::*;

#[test]
fn deterministic_dot_rejects_mismatched_lengths() {
    let backend = crate::cpu::CpuBackend::new();
    let x = backend.raw_alloc::<f64>(2).expect("x allocation");
    let y = backend.raw_alloc::<f64>(3).expect("y allocation");
    let error = backend
        .try_dot_deterministic(&x, &y)
        .expect_err("mismatched deterministic dot must be rejected");
    assert!(matches!(error, BackendError::InvalidArgument { .. }));
}

#[test]
fn checked_norm_rejects_non_finite_reduction_results() {
    let backend = crate::cpu::CpuBackend::new();
    let mut values = backend.raw_alloc::<f64>(1).expect("allocation");
    backend.copy_h2d(&[f64::NAN], &mut values);

    for mode in [
        ReductionMode::Fast,
        ReductionMode::Reproducible,
        ReductionMode::Deterministic,
    ] {
        assert!(matches!(
            backend.try_nrm2_with_mode(mode, &values),
            Err(BackendError::OperationFailed {
                operation: "nrm2",
                ..
            })
        ));
    }
}

#[test]
fn checked_dot_rejects_non_finite_reduction_results() {
    let backend = crate::cpu::CpuBackend::new();
    let mut x = backend.raw_alloc::<f64>(1).expect("x allocation");
    let mut y = backend.raw_alloc::<f64>(1).expect("y allocation");
    backend.copy_h2d(&[f64::NAN], &mut x);
    backend.copy_h2d(&[1.0], &mut y);
    for mode in [
        ReductionMode::Fast,
        ReductionMode::Reproducible,
        ReductionMode::Deterministic,
    ] {
        assert!(matches!(
            backend.try_dot_with_mode(mode, &x, &y),
            Err(BackendError::OperationFailed {
                operation: "dot" | "dot_deterministic",
                ..
            })
        ));
    }
}

#[test]
fn checked_reductions_accept_large_finite_components() {
    let backend = crate::cpu::CpuBackend::new();
    let mut x = backend.raw_alloc::<f64>(2).expect("x allocation");
    let mut y = backend.raw_alloc::<f64>(2).expect("y allocation");
    backend.copy_h2d(&[f64::MAX, 0.0], &mut x);
    backend.copy_h2d(&[0.0, 1.0], &mut y);

    for mode in [
        ReductionMode::Fast,
        ReductionMode::Reproducible,
        ReductionMode::Deterministic,
    ] {
        let norm = backend
            .try_nrm2_with_mode(mode, &x)
            .expect("finite large norm must not overflow its intermediate square");
        assert_eq!(norm, f64::MAX);
    }

    assert_eq!(
        backend
            .try_dot_deterministic(&x, &y)
            .expect("finite inputs with a finite result"),
        0.0
    );

    let mut basis = backend.raw_alloc::<f64>(1).expect("basis allocation");
    let mut output = backend.raw_alloc::<f64>(1).expect("output allocation");
    backend.copy_h2d(&[f64::MAX], &mut basis);
    backend.copy_h2d(&[1.0], &mut output);
    backend
        .try_basis_axpy_strided(f64::MAX, &basis, 1, 1, &[0.0], &mut output)
        .expect("finite update coefficients with a finite result");
    let mut host_output = [0.0];
    backend.copy_d2h(&output, &mut host_output);
    assert_eq!(host_output, [1.0]);
}

#[test]
fn checked_strided_multi_dot_validates_shape_and_fixed_order() {
    let backend = crate::cpu::CpuBackend::new();
    let mut basis = backend.raw_alloc::<f64>(6).expect("basis allocation");
    let mut y = backend.raw_alloc::<f64>(3).expect("y allocation");
    backend.copy_h2d(&[1.0, 2.0, 3.0, -1.0, 0.5, 4.0], &mut basis);
    backend.copy_h2d(&[2.0, -1.0, 0.5], &mut y);

    let result = backend
        .try_multi_dot_strided_with_mode(ReductionMode::Deterministic, &basis, 3, 2, &y)
        .expect("valid strided multi-dot");
    assert_eq!(result, vec![1.5, -0.5]);
    let reproducible = backend
        .try_multi_dot_strided_with_mode(ReductionMode::Reproducible, &basis, 3, 2, &y)
        .expect("reproducible strided multi-dot");
    assert_eq!(reproducible, result);
    let mut result_into = [0.0; 2];
    backend
        .try_multi_dot_strided_with_mode_into(
            ReductionMode::Fast,
            &basis,
            3,
            2,
            &y,
            &mut result_into,
        )
        .expect("caller-owned strided multi-dot output");
    assert_eq!(result_into, [1.5, -0.5]);
    assert!(matches!(
        backend.try_multi_dot_strided_with_mode_into(
            ReductionMode::Fast,
            &basis,
            3,
            2,
            &y,
            &mut [0.0; 1],
        ),
        Err(BackendError::InvalidArgument { .. })
    ));

    let mut projected = backend.raw_alloc::<f64>(3).expect("projection vector");
    backend.copy_h2d(&[2.0, -1.0, 0.5], &mut projected);
    let coefficients = backend
        .try_basis_project_out_strided_with_mode(ReductionMode::Fast, &basis, 3, 2, &mut projected)
        .expect("checked projection");
    assert_eq!(coefficients, vec![1.5, -0.5]);
    let mut projected_host = [0.0; 3];
    backend.copy_d2h(&projected, &mut projected_host);
    assert_eq!(projected_host, [0.0, -3.75, -2.0]);

    let mut projected_deterministic = backend
        .raw_alloc::<f64>(3)
        .expect("deterministic projection vector");
    backend.copy_h2d(&[2.0, -1.0, 0.5], &mut projected_deterministic);
    let deterministic_coefficients = backend
        .try_basis_project_out_strided_with_mode(
            ReductionMode::Deterministic,
            &basis,
            3,
            2,
            &mut projected_deterministic,
        )
        .expect("deterministic projection");
    assert_eq!(deterministic_coefficients, coefficients);
    backend.copy_d2h(&projected_deterministic, &mut projected_host);
    assert_eq!(projected_host, [0.0, -3.75, -2.0]);

    let short_basis = backend.raw_alloc::<f64>(5).expect("short basis allocation");
    assert!(matches!(
        backend.try_multi_dot_strided_with_mode(ReductionMode::Fast, &short_basis, 3, 2, &y,),
        Err(BackendError::InvalidArgument { .. })
    ));

    let mut invalid_projection = backend
        .raw_alloc::<f64>(3)
        .expect("invalid projection vector");
    backend.copy_h2d(&[2.0, -1.0, 0.5], &mut invalid_projection);
    assert!(matches!(
        backend.try_basis_project_out_strided_with_mode(
            ReductionMode::Fast,
            &short_basis,
            3,
            2,
            &mut invalid_projection,
        ),
        Err(BackendError::InvalidArgument { .. })
    ));
    let mut invalid_projection_host = [0.0; 3];
    backend.copy_d2h(&invalid_projection, &mut invalid_projection_host);
    assert_eq!(invalid_projection_host, [2.0, -1.0, 0.5]);

    let wrong_y = backend.raw_alloc::<f64>(2).expect("wrong y allocation");
    assert!(matches!(
        backend.try_multi_dot_strided_with_mode(
            ReductionMode::Deterministic,
            &basis,
            3,
            2,
            &wrong_y,
        ),
        Err(BackendError::InvalidArgument { .. })
    ));

    let mut destination = backend.raw_alloc::<f64>(3).expect("destination allocation");
    assert!(matches!(
        backend.try_basis_axpy_strided(1.0, &basis, 3, 2, &[1.0], &mut destination,),
        Err(BackendError::InvalidArgument { .. })
    ));
    assert!(matches!(
        backend.try_basis_axpy_strided(f64::NAN, &basis, 3, 2, &[1.0, 1.0], &mut destination,),
        Err(BackendError::OperationFailed { .. })
    ));
}

#[test]
fn checked_strided_multi_dot_rejects_nonfinite_inputs_in_fixed_order_modes() {
    let backend = crate::cpu::CpuBackend::new();
    let mut basis = backend.raw_alloc::<f64>(2).expect("basis allocation");
    let mut y = backend.raw_alloc::<f64>(2).expect("rhs allocation");
    backend.copy_h2d(&[1.0, f64::NAN], &mut basis);
    backend.copy_h2d(&[2.0, 3.0], &mut y);

    for mode in [ReductionMode::Reproducible, ReductionMode::Deterministic] {
        let error = backend
            .try_multi_dot_strided_with_mode(mode, &basis, 2, 1, &y)
            .expect_err("non-finite reduction input must fail closed");
        assert!(matches!(error, BackendError::OperationFailed { .. }));
    }
}

#[test]
fn reproducible_reductions_match_deterministic_for_all_public_scalars() {
    use num_complex::{Complex32, Complex64};

    fn check<T>(backend: &crate::cpu::CpuBackend, x_values: &[T], y_values: &[T])
    where
        T: DeviceScalar + PartialEq + std::fmt::Debug,
        T::Real: std::fmt::Debug,
    {
        let mut x = backend
            .raw_alloc::<T>(x_values.len())
            .expect("x allocation");
        let mut y = backend
            .raw_alloc::<T>(y_values.len())
            .expect("y allocation");
        backend.copy_h2d(x_values, &mut x);
        backend.copy_h2d(y_values, &mut y);

        assert_eq!(
            backend
                .try_dot_with_mode(ReductionMode::Reproducible, &x, &y)
                .expect("reproducible dot"),
            backend
                .try_dot_with_mode(ReductionMode::Deterministic, &x, &y)
                .expect("deterministic dot")
        );
        assert_eq!(
            backend
                .try_nrm2_with_mode(ReductionMode::Reproducible, &x)
                .expect("reproducible norm"),
            backend
                .try_nrm2_with_mode(ReductionMode::Deterministic, &x)
                .expect("deterministic norm")
        );
        assert_eq!(
            backend
                .try_multi_dot_with_mode(ReductionMode::Reproducible, &[&x], &y)
                .expect("reproducible multi-dot"),
            backend
                .try_multi_dot_with_mode(ReductionMode::Deterministic, &[&x], &y)
                .expect("deterministic multi-dot")
        );
    }

    let backend = crate::cpu::CpuBackend::new();
    check(
        &backend,
        &[1.0_f32, -2.0, 3.0, 0.5, -0.25],
        &[0.5_f32, 4.0, -1.0, 2.0, 8.0],
    );
    check(
        &backend,
        &[1.0_f64, -2.0, 3.0, 0.5, -0.25],
        &[0.5_f64, 4.0, -1.0, 2.0, 8.0],
    );
    check(
        &backend,
        &[
            Complex32::new(1.0, 2.0),
            Complex32::new(-2.0, 0.5),
            Complex32::new(3.0, -1.0),
            Complex32::new(0.5, 4.0),
            Complex32::new(-0.25, 0.0),
        ],
        &[
            Complex32::new(0.5, -1.0),
            Complex32::new(4.0, 2.0),
            Complex32::new(-1.0, 0.25),
            Complex32::new(2.0, -3.0),
            Complex32::new(8.0, 1.0),
        ],
    );
    check(
        &backend,
        &[
            Complex64::new(1.0, 2.0),
            Complex64::new(-2.0, 0.5),
            Complex64::new(3.0, -1.0),
            Complex64::new(0.5, 4.0),
            Complex64::new(-0.25, 0.0),
        ],
        &[
            Complex64::new(0.5, -1.0),
            Complex64::new(4.0, 2.0),
            Complex64::new(-1.0, 0.25),
            Complex64::new(2.0, -3.0),
            Complex64::new(8.0, 1.0),
        ],
    );
}

#[test]
fn legacy_strided_reductions_reject_bad_shapes_without_panicking() {
    let backend = crate::cpu::CpuBackend::new();
    let basis = backend.raw_alloc::<f64>(1).expect("basis allocation");
    let mut y = backend.raw_alloc::<f64>(2).expect("vector allocation");
    backend.copy_h2d(&[3.0, 4.0], &mut y);

    assert!(backend.multi_dot_strided(&basis, 2, 2, &y).is_empty());
    backend.basis_axpy_strided(1.0, &basis, 2, 2, &[1.0, 1.0], &mut y);

    let mut result = [0.0; 2];
    backend.copy_d2h(&y, &mut result);
    assert_eq!(result, [3.0, 4.0]);
}

#[test]
fn legacy_strided_reductions_reject_nonfinite_inputs_without_mutation() {
    let backend = crate::cpu::CpuBackend::new();
    let mut basis = backend.raw_alloc::<f64>(2).expect("basis allocation");
    let mut y = backend.raw_alloc::<f64>(2).expect("vector allocation");
    backend.copy_h2d(&[f64::NAN, 2.0], &mut basis);
    backend.copy_h2d(&[3.0, 4.0], &mut y);

    assert!(backend.multi_dot_strided(&basis, 2, 1, &y).is_empty());

    backend.copy_h2d(&[1.0, 2.0], &mut basis);
    backend.basis_axpy_strided(f64::NAN, &basis, 2, 1, &[1.0], &mut y);

    let mut result = [0.0; 2];
    backend.copy_d2h(&y, &mut result);
    assert_eq!(result, [3.0, 4.0]);
}

#[test]
fn deterministic_norm_uses_checked_transfer_and_finite_input_contract() {
    let backend = crate::cpu::CpuBackend::new();
    let mut values = backend.raw_alloc::<f64>(2).expect("allocation");
    backend.copy_h2d(&[3.0, 4.0], &mut values);

    assert_eq!(
        backend
            .try_nrm2_deterministic(&values)
            .expect("finite deterministic norm"),
        5.0
    );

    backend.copy_h2d(&[f64::NAN, 0.0], &mut values);
    assert!(matches!(
        backend.try_nrm2_deterministic(&values),
        Err(BackendError::OperationFailed {
            operation: "nrm2",
            ..
        })
    ));
}

#[test]
fn checked_axpy_rejects_mismatched_lengths_and_updates_valid_vectors() {
    let backend = crate::cpu::CpuBackend::new();
    let x = backend.raw_alloc::<f64>(2).expect("x allocation");
    let mut y = backend.raw_alloc::<f64>(3).expect("y allocation");
    let error = backend
        .try_axpy(2.0, &x, &mut y)
        .expect_err("checked axpy must reject mismatched vectors");
    assert!(matches!(error, BackendError::InvalidArgument { .. }));

    let mut x = backend.raw_alloc::<f64>(2).expect("x allocation");
    let mut y = backend.raw_alloc::<f64>(2).expect("y allocation");
    backend.copy_h2d(&[1.0, -2.0], &mut x);
    backend.copy_h2d(&[3.0, 4.0], &mut y);
    backend
        .try_axpy(0.5, &x, &mut y)
        .expect("checked axpy should dispatch on CPU");
    let mut result = [0.0; 2];
    backend.copy_d2h(&y, &mut result);
    assert_eq!(result, [3.5, 3.0]);
}

#[test]
fn checked_blas_rejects_non_finite_scalar_parameters() {
    let backend = crate::cpu::CpuBackend::new();
    let x = backend.raw_alloc::<f64>(2).expect("x allocation");
    let mut y = backend.raw_alloc::<f64>(2).expect("y allocation");

    assert!(require_finite_scalar("test", "value", f64::MAX).is_ok());
    assert!(matches!(
        backend.try_axpy(f64::NAN, &x, &mut y),
        Err(BackendError::OperationFailed {
            operation: "axpy",
            ..
        })
    ));
    assert!(matches!(
        backend.try_scale(f64::INFINITY, &mut y),
        Err(BackendError::OperationFailed {
            operation: "scale",
            ..
        })
    ));

    let a = backend.raw_alloc::<f64>(4).expect("A allocation");
    assert!(matches!(
        backend.try_gemv(Transpose::NoTrans, 2, 2, f64::NAN, &a, 2, &x, 1.0, &mut y,),
        Err(BackendError::OperationFailed {
            operation: "gemv",
            ..
        })
    ));

    let b = backend.raw_alloc::<f64>(4).expect("B allocation");
    let mut c = backend.raw_alloc::<f64>(4).expect("C allocation");
    assert!(matches!(
        backend.try_gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            1.0,
            &a,
            2,
            &b,
            2,
            f64::INFINITY,
            &mut c,
            2,
        ),
        Err(BackendError::OperationFailed {
            operation: "gemm",
            ..
        })
    ));

    let mut gemv_y = backend.raw_alloc::<f64>(2).expect("batched GEMV output");
    let mut gemv_batch = [GemvBatch {
        op: Transpose::NoTrans,
        m: 2,
        n: 2,
        alpha: f64::NAN,
        a: &a,
        lda: 2,
        x: &x,
        beta: 1.0,
        y: &mut gemv_y,
    }];
    assert!(matches!(
        backend.try_gemv_batched(&mut gemv_batch),
        Err(BackendError::OperationFailed {
            operation: "gemv_batched",
            ..
        })
    ));

    let mut gemm_c = backend.raw_alloc::<f64>(4).expect("batched GEMM output");
    let mut gemm_batch = [GemmBatch {
        transa: Transpose::NoTrans,
        transb: Transpose::NoTrans,
        m: 2,
        n: 2,
        k: 2,
        alpha: 1.0,
        a: &a,
        lda: 2,
        b: &b,
        ldb: 2,
        beta: f64::INFINITY,
        c: &mut gemm_c,
        ldc: 2,
    }];
    assert!(matches!(
        backend.try_gemm_batched(&mut gemm_batch),
        Err(BackendError::OperationFailed {
            operation: "gemm_batched",
            ..
        })
    ));
}

#[test]
fn checked_scale_and_norm_use_the_checked_backend_boundary() {
    let backend = crate::cpu::CpuBackend::new();
    let mut values = backend.raw_alloc::<f64>(2).expect("values allocation");
    backend.copy_h2d(&[3.0, 4.0], &mut values);

    backend
        .try_scale(2.0, &mut values)
        .expect("CPU checked scale");
    let norm = backend
        .try_nrm2_with_mode(ReductionMode::Fast, &values)
        .expect("CPU checked norm");

    assert_eq!(norm, 10.0);
}

#[test]
fn checked_spmv_converts_dimension_assertions_to_typed_errors() {
    let backend = crate::cpu::CpuBackend::new();
    let mut row_ptr = backend.raw_alloc::<i32>(3).expect("row pointer allocation");
    let mut col_idx = backend.raw_alloc::<i32>(2).expect("column allocation");
    let mut values = backend.raw_alloc::<f64>(2).expect("value allocation");
    backend.copy_h2d(&[0, 1, 2], &mut row_ptr);
    backend.copy_h2d(&[0, 1], &mut col_idx);
    backend.copy_h2d(&[2.0, 3.0], &mut values);
    let prebaked = backend.prebake_csr(2, 2, &row_ptr, &col_idx, &values);
    let x = backend.raw_alloc::<f64>(1).expect("x allocation");
    let mut y = backend.raw_alloc::<f64>(2).expect("y allocation");

    let error = backend
        .try_spmv_prebaked(&prebaked, &x, &mut y)
        .expect_err("checked SpMV must reject a short input");
    assert!(matches!(error, BackendError::InvalidArgument { .. }));
}

#[test]
fn checked_spmm_rejects_short_strided_rhs_buffers() {
    let backend = crate::cpu::CpuBackend::new();
    let mut row_ptr = backend.raw_alloc::<i32>(3).expect("row pointer allocation");
    let mut col_idx = backend.raw_alloc::<i32>(2).expect("column allocation");
    let mut values = backend.raw_alloc::<f64>(2).expect("value allocation");
    backend.copy_h2d(&[0, 1, 2], &mut row_ptr);
    backend.copy_h2d(&[0, 1], &mut col_idx);
    backend.copy_h2d(&[2.0, 3.0], &mut values);
    let prebaked = backend.prebake_csr(2, 2, &row_ptr, &col_idx, &values);
    let x = backend.raw_alloc::<f64>(1).expect("x allocation");
    let mut y = backend.raw_alloc::<f64>(2).expect("y allocation");

    let error = backend
        .try_spmm_prebaked(&prebaked, 1, &x, 2, &mut y, 2)
        .expect_err("checked SpMM must reject a short RHS buffer");
    assert!(matches!(error, BackendError::InvalidArgument { .. }));
}

#[test]
fn checked_spmm_rejects_overlapping_leading_dimensions() {
    let backend = crate::cpu::CpuBackend::new();
    let mut row_ptr = backend.raw_alloc::<i32>(3).expect("row pointer allocation");
    let mut col_idx = backend.raw_alloc::<i32>(2).expect("column allocation");
    let mut values = backend.raw_alloc::<f64>(2).expect("value allocation");
    backend.copy_h2d(&[0, 1, 2], &mut row_ptr);
    backend.copy_h2d(&[0, 1], &mut col_idx);
    backend.copy_h2d(&[2.0, 3.0], &mut values);
    let prebaked = backend.prebake_csr(2, 2, &row_ptr, &col_idx, &values);
    let mut x = backend.raw_alloc::<f64>(4).expect("x allocation");
    let mut y = backend.raw_alloc::<f64>(4).expect("y allocation");
    backend.copy_h2d(&[1.0, 2.0, 3.0, 4.0], &mut x);
    backend.copy_h2d(&[9.0, 9.0, 9.0, 9.0], &mut y);

    let error = backend
        .try_spmm_prebaked(&prebaked, 2, &x, 1, &mut y, 2)
        .expect_err("leading dimension smaller than matrix width must be rejected");
    assert!(matches!(
        error,
        BackendError::InvalidArgument {
            operation: "spmm_prebaked",
            ..
        }
    ));

    let mut host_y = [0.0; 4];
    backend.copy_d2h(&y, &mut host_y);
    assert_eq!(host_y, [9.0; 4]);
}

#[test]
fn checked_gather_rejects_negative_and_out_of_range_indices() {
    let backend = crate::cpu::CpuBackend::new();
    let mut values = backend.raw_alloc::<f64>(2).expect("values allocation");
    let mut indices = backend.raw_alloc::<i32>(4).expect("indices allocation");
    let mut weights = backend.raw_alloc::<f64>(4).expect("weights allocation");
    let mut output = backend.raw_alloc::<f64>(1).expect("output allocation");
    backend.copy_h2d(&[1.0, 2.0], &mut values);
    backend.copy_h2d(&[-1, 0, 1, 2], &mut indices);
    backend.copy_h2d(&[0.25; 4], &mut weights);
    let error = backend
        .try_gather4_weighted(&values, &indices, &weights, &mut output)
        .expect_err("invalid gather index must be rejected");
    assert!(matches!(error, BackendError::InvalidArgument { .. }));

    backend.copy_h2d(&[0, 1, 0, 1], &mut indices);
    backend.copy_h2d(&[0.25; 4], &mut weights);
    backend.copy_h2d(&[f64::NAN, 2.0], &mut values);
    backend.copy_h2d(&[17.0], &mut output);
    let error = backend
        .try_gather4_weighted(&values, &indices, &weights, &mut output)
        .expect_err("non-finite gather values must be rejected");
    assert!(matches!(
        error,
        BackendError::OperationFailed {
            operation: "gather4_weighted",
            ..
        }
    ));
    let mut output_host = [0.0];
    backend.copy_d2h(&output, &mut output_host);
    assert_eq!(output_host, [17.0]);

    backend.copy_h2d(&[1.0, 2.0], &mut values);
    backend.copy_h2d(&[f64::INFINITY, 0.25, 0.25, 0.25], &mut weights);
    let error = backend
        .try_gather4_weighted(&values, &indices, &weights, &mut output)
        .expect_err("non-finite gather weights must be rejected");
    assert!(matches!(
        error,
        BackendError::OperationFailed {
            operation: "gather4_weighted",
            ..
        }
    ));
    backend.copy_d2h(&output, &mut output_host);
    assert_eq!(output_host, [17.0]);
}
