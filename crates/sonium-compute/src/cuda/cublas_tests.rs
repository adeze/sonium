use super::*;

#[test]
fn nvrtc_arch_maps_known_generations() {
    assert_eq!(nvrtc_arch(5, 2), Some("compute_50"));
    assert_eq!(nvrtc_arch(6, 1), Some("compute_60"));
    assert_eq!(nvrtc_arch(7, 0), Some("compute_70"));
    assert_eq!(nvrtc_arch(7, 5), Some("compute_75"));
    assert_eq!(nvrtc_arch(8, 0), Some("compute_80"));
    assert_eq!(nvrtc_arch(8, 6), Some("compute_86"));
    assert_eq!(nvrtc_arch(8, 9), Some("compute_89"));
    assert_eq!(nvrtc_arch(9, 0), Some("compute_90"));
    assert_eq!(nvrtc_arch(10, 0), Some("compute_100"));
    assert_eq!(nvrtc_arch(12, 0), Some("compute_120"));
    assert_eq!(nvrtc_arch(4, 0), None);
    assert_eq!(nvrtc_arch(11, 0), None);
}

#[test]
fn cuda_strided_view_copy_is_device_to_device() {
    let source = include_str!("cublas.rs");
    let body = source
        .split("fn try_copy_views_d2d<T: BufferElement>")
        .nth(1)
        .expect("CUDA checked view-copy implementation should exist")
        .split("fn copy_h2d<T: BufferElement>")
        .next()
        .expect("CUDA view-copy implementation should precede compatibility copy");
    assert!(body.contains("slice_mut"));
    assert!(body.contains("memcpy_dtod"));
    assert!(body.contains("checked_view_descriptor"));
    assert!(!body.contains("try_copy_view_d2h"));
}

#[test]
fn cuda_complex_gemv_helpers_reject_conjugate_transpose_without_asserting() {
    let source = include_str!("cublas_blas23.rs");
    for name in [
        "gemv_c32_nt",
        "gemv_c64_nt",
        "gemv_batched_c32_nt",
        "gemv_batched_c64_nt",
    ] {
        let body = source
            .split(&format!("pub unsafe fn {name}"))
            .nth(1)
            .expect("complex GEMV helper should exist")
            .split("pub unsafe fn")
            .next()
            .expect("complex GEMV helper body should be bounded");
        assert!(body.contains("CUBLAS_STATUS_INVALID_VALUE"));
        assert!(!body.contains("debug_assert!"));
    }
}

#[test]
fn cuda_scalar_parameter_copy_has_runtime_size_guard() {
    let source = include_str!("cublas.rs");
    let body = source
        .split("unsafe fn try_write<U: Copy>")
        .nth(1)
        .expect("CUDA scalar parameter writer should exist")
        .split("fn as_ptr")
        .next()
        .expect("CUDA scalar parameter writer should precede pointer access");
    assert!(body.contains("if size > self.bytes.len()"));
    assert!(body.contains("CUDA scalar parameter size"));
    assert!(!body.contains("debug_assert!"));
}

#[test]
fn cuda_nrm2_fallback_uses_checked_host_staging() {
    let source = include_str!("cublas.rs");
    let body = source
        .split("fn try_nrm2_with_mode<T: DeviceScalar>")
        .nth(1)
        .expect("CUDA nrm2 implementation should exist")
        .split("fn try_scale<T: DeviceScalar>")
        .next()
        .expect("CUDA nrm2 implementation should precede scale");
    assert!(body.contains("nrm2 fallback host input"));
    assert!(!body.contains("vec![T::zero(); x.len]"));
}

#[test]
fn cuda_multi_dot_checked_zero_shapes_use_fallible_results() {
    let source = include_str!("cublas.rs");
    assert!(source.contains("return try_zeroed_host(x_basis.len(), \"multi_dot\")"));
    assert!(source.contains("return try_zeroed_host(basis_count, \"multi_dot_strided\")"));
    let compatibility_helper = ["fn ", "zeroed_compatibility_result", "<"].concat();
    assert!(!source.contains(&compatibility_helper));

    for (start, end) in [
        (
            "fn try_multi_dot_with_mode<T: DeviceScalar>",
            "fn try_multi_dot_with_mode_into<T: DeviceScalar>",
        ),
        (
            "fn try_multi_dot_strided_real<T: DeviceScalar>",
            "fn multi_dot_strided_host<T: DeviceScalar>",
        ),
    ] {
        let body = source
            .split(start)
            .nth(1)
            .expect("CUDA checked multi-dot helper should exist")
            .split(end)
            .next()
            .expect("CUDA checked multi-dot helper should precede its next helper");
        assert!(!body.contains("vec![T::zero();"));
    }
}

#[test]
fn cuda_native_pointer_tables_use_checked_staging() {
    let source = include_str!("cublas.rs");
    for (name, next) in [
        ("fn try_gemv_batched_native", "fn try_gemm_batched_native"),
        ("fn try_gemm_batched_native", "fn axpy_many_cuda"),
        ("fn axpy_many_cuda", "fn try_multi_dot_strided_real"),
        ("fn try_multi_dot_real_impl", "fn try_multi_dot_real_into"),
    ] {
        let body = source
            .split(name)
            .nth(1)
            .expect("CUDA pointer-table helper should exist")
            .split(next)
            .next()
            .expect("CUDA pointer-table helper should be bounded");
        assert!(body.contains("try_host_staging_capacity"));
        assert!(!body.contains("Vec::with_capacity"));
    }
}

#[test]
fn cuda_rank_n_view_copy_is_device_to_device() {
    let source = include_str!("cublas.rs");
    let body = source
        .split("fn try_copy_nd_views_d2d<T: BufferElement>")
        .nth(1)
        .expect("CUDA rank-N checked view-copy implementation should exist")
        .split("fn copy_h2d<T: BufferElement>")
        .next()
        .expect("CUDA rank-N view-copy implementation should precede compatibility copy");
    assert!(body.contains("checked_element_index"));
    assert!(body.contains("memcpy_dtod"));
    assert!(body.contains("ViewAliasing::OverlapSafe"));
    assert!(body.contains("try_copy_nd_view_d2h"));
}

#[test]
fn cuda_checked_copy_paths_record_transfer_stats() {
    let source = include_str!("cublas.rs");
    assert!(source.contains("transfer_stats: Arc<BackendTransferStats>"));
    assert!(source.contains("self.transfer_stats.record_h2d(bytes)"));
    assert!(source.contains("self.transfer_stats.record_d2h(bytes)"));
    assert!(source.contains("self.transfer_stats.record_d2d("));
    assert!(source.contains("fn transfer_stats(&self) -> TransferStatsSnapshot"));
}

#[test]
fn cuda_legacy_error_state_preserves_first_failure() {
    let pending_error = Mutex::new(None);
    record_first_pending_error(
        &pending_error,
        BackendError::Transfer {
            backend: "cuda",
            operation: "copy_h2d",
            reason: "first transfer failed".into(),
        },
    );
    record_first_pending_error(
        &pending_error,
        BackendError::Transfer {
            backend: "cuda",
            operation: "copy_d2h",
            reason: "later transfer failed".into(),
        },
    );

    assert!(matches!(
        pending_error.lock().unwrap().clone(),
        Some(BackendError::Transfer {
            operation: "copy_h2d",
            reason,
            ..
        }) if reason == "first transfer failed"
    ));
}

#[test]
fn cuda_injected_launch_and_completion_failures_poison_only_terminal_contexts() {
    let launch_error = Mutex::new(None);
    let ordinary_launch = backend_driver_launch_error(
        &launch_error,
        "test_kernel",
        "CUDA_ERROR_INVALID_VALUE".into(),
    );
    assert!(matches!(ordinary_launch, BackendError::Launch { .. }));
    assert!(launch_error.lock().unwrap().is_none());

    let device_error = Mutex::new(None);
    let lost = backend_driver_launch_error(
        &device_error,
        "test_kernel",
        "CUDA_ERROR_DEVICE_LOST".into(),
    );
    assert!(matches!(lost, BackendError::DeviceLost { .. }));
    assert_eq!(device_error.lock().unwrap().as_ref(), Some(&lost));

    let synchronization_error = backend_driver_synchronization_error(
        &device_error,
        "completion_event",
        "later synchronization failure".into(),
    );
    assert!(matches!(
        synchronization_error,
        BackendError::Synchronization { .. }
    ));
    assert_eq!(
        device_error.lock().unwrap().as_ref(),
        Some(&lost),
        "the first asynchronous failure must remain authoritative"
    );
}

#[test]
fn cuda_legacy_copy_wrappers_retain_typed_failures() {
    let source = include_str!("cublas.rs");
    for (method, next_method) in [
        ("fn copy_h2d", "fn copy_d2h"),
        ("fn copy_d2h", "fn copy_d2d"),
        ("fn copy_d2d", "fn try_gather4_weighted"),
    ] {
        let body = source
            .split(method)
            .nth(1)
            .expect("CUDA legacy copy wrapper should exist")
            .split(next_method)
            .next()
            .expect("CUDA legacy copy wrapper should precede its next method");
        assert!(body.contains("self.record_pending_error(error)"));
        assert!(!body.contains("log::error!"));
    }
    for (method, next_method) in [
        ("fn try_copy_h2d", "fn try_copy_d2h"),
        ("fn try_copy_d2h", "fn try_copy_d2d"),
        ("fn try_copy_d2d", "fn try_gather4_weighted"),
    ] {
        let body = source
            .split(method)
            .nth(1)
            .expect("CUDA checked copy wrapper should exist")
            .split(next_method)
            .next()
            .expect("CUDA checked copy wrapper should precede its next method");
        assert!(body.contains("self.check_pending_error()?"));
    }
}

#[test]
fn dense_lu_uses_checked_transfer_boundaries() {
    let source = include_str!("cublas.rs");
    // (constructor, constructor end marker, solve) per scalar type.
    for (constructor_name, constructor_end, solve_name) in [
        (
            "pub fn dense_lu_c32_from_column_major(",
            "/// `LU right-hand",
            "pub fn dense_lu_c32_solve(",
        ),
        (
            "pub fn dense_lu_f64_from_column_major(",
            "/// Solve `op(A)",
            "pub fn dense_lu_f64_solve(",
        ),
    ] {
        let constructor = source
            .split(constructor_name)
            .nth(1)
            .expect("CUDA dense LU constructor should exist")
            .split(constructor_end)
            .next()
            .expect("CUDA dense LU constructor should precede solve");
        assert!(constructor.contains("self.try_copy_h2d(values, &mut factors)"));
        assert!(!constructor.contains("self.copy_h2d(values, &mut factors)"));

        let solve = source
            .split(solve_name)
            .nth(1)
            .expect("CUDA dense LU solve should exist")
            .split("fn check_dense_lu_info")
            .next()
            .expect("CUDA dense LU solve should precede info check");
        assert!(solve.contains("self.try_copy_d2d(rhs, out)"));
        assert!(!solve.contains("self.copy_d2d(rhs, out)"));
    }

    let info = source
        .split("fn check_dense_lu_info")
        .nth(1)
        .expect("CUDA dense LU info check should exist")
        .split("fn cuda_multi_dot_capability")
        .next()
        .expect("CUDA dense LU info check should precede capability helpers");
    assert!(info.contains("self.try_copy_d2h(info, &mut host)"));
    assert!(!info.contains("self.copy_d2h(info, &mut host)"));
}

#[test]
fn cuda_legacy_native_wrappers_retain_typed_failures() {
    let source = include_str!("cublas.rs");
    for (method, next_method) in [
        ("fn gather4_weighted", "fn try_cast_to_lower"),
        ("fn cast_to_lower", "fn try_cast_from_lower"),
        ("fn cast_from_lower", "fn try_synchronize"),
        (
            "fn multi_dot<T: DeviceScalar>",
            "fn try_multi_dot_strided_with_mode",
        ),
        (
            "fn multi_dot_strided<T: DeviceScalar>",
            "fn try_spmv_prebaked<T: DeviceScalar>",
        ),
    ] {
        let body = source
            .split(method)
            .nth(1)
            .expect("CUDA legacy native wrapper should exist")
            .split(next_method)
            .next()
            .expect("CUDA legacy native wrapper should precede its checked successor");
        assert!(
            body.contains("self.record_pending_error(error"),
            "{method} must retain its first typed failure"
        );
    }

    let host_fallback = source
        .split("fn multi_dot_strided_host<T: DeviceScalar>")
        .nth(1)
        .expect("CUDA host-staged strided multi-dot helper should exist")
        .split("fn try_multi_dot_strided_host<T: DeviceScalar>")
        .next()
        .expect("checked host-staged helper should follow legacy helper");
    assert!(host_fallback.contains("-> BackendResult<Vec<T>>"));
    assert!(!host_fallback.contains("Vec::new()"));
}

#[test]
fn cuda_allocation_guard_tracks_live_and_high_water_bytes() {
    let stats = Arc::new(CudaAllocationStats::default());
    {
        let first = CudaAllocationGuard::new(stats.clone(), 16);
        assert_eq!(stats.live_bytes.load(Ordering::Relaxed), 16);
        assert_eq!(stats.high_water_bytes.load(Ordering::Relaxed), 16);
        let second = CudaAllocationGuard::new(stats.clone(), 24);
        assert_eq!(stats.live_bytes.load(Ordering::Relaxed), 40);
        assert_eq!(stats.high_water_bytes.load(Ordering::Relaxed), 40);
        drop(second);
        assert_eq!(stats.live_bytes.load(Ordering::Relaxed), 16);
        drop(first);
    }
    assert_eq!(stats.live_bytes.load(Ordering::Relaxed), 0);
    assert_eq!(stats.high_water_bytes.load(Ordering::Relaxed), 40);
}

#[test]
fn cuda_device_loss_is_classified_before_transfer_or_allocation_fallbacks() {
    assert!(cuda_device_lost_reason("CUDA_ERROR_DEVICE_LOST"));
    assert!(cuda_device_lost_reason("DriverError::DeviceLost"));
    assert!(cuda_device_lost_reason("device_lost"));
    assert!(!cuda_device_lost_reason("CUDA_ERROR_OUT_OF_MEMORY"));

    let transfer = cuda_device_lost_error(
        "copy_d2h",
        "CUDA_ERROR_DEVICE_LOST while reading output".into(),
    );
    assert!(matches!(
        transfer,
        Some(BackendError::DeviceLost {
            backend: "cuda",
            operation: "copy_d2h",
            ..
        })
    ));

    let synchronize = cuda_device_lost_error("synchronize", "CUDA_ERROR_DEVICE_LOST".into());
    assert!(matches!(
        synchronize,
        Some(BackendError::DeviceLost {
            backend: "cuda",
            operation: "synchronize",
            ..
        })
    ));
}

#[test]
fn cuda_checked_completion_paths_preserve_failure_contracts() {
    let source = include_str!("cublas.rs");

    let try_capture = source
        .split("pub fn try_capture_graph<F>")
        .nth(1)
        .expect("CUDA checked graph capture should exist")
        .split("/// Whether CUDA BEM")
        .next()
        .expect("CUDA checked graph capture should precede BEM helpers");
    assert!(!try_capture.contains("let _ = self.stream.end_capture"));
    assert!(try_capture.contains(".map_err(driver_to_oom)"));

    let capture_guard = source
        .split("struct CudaCaptureGuard")
        .nth(1)
        .expect("CUDA capture guard should exist")
        .split("impl CudaGraphReplay")
        .next()
        .expect("CUDA capture guard should precede graph replay");
    assert!(capture_guard.contains("pending_error"));
    assert!(capture_guard.contains("failure_injection"));
    assert!(capture_guard.contains("CudaFailureInjectionPoint::Teardown"));
    assert!(capture_guard.contains("take_cuda_failure"));
    assert!(capture_guard.contains("record_teardown_error"));
    assert!(!capture_guard.contains("let _ = self.stream.end_capture"));

    let try_synchronize = source
        .split("fn try_synchronize(&self) -> BackendResult<()>")
        .nth(1)
        .expect("CUDA checked synchronize should exist")
        .split("fn try_submit_completion")
        .next()
        .expect("CUDA checked synchronize should precede completion submission");
    assert!(try_synchronize.contains("self.check_pending_error()?"));
    assert!(try_synchronize.contains("backend_driver_synchronization_error"));
    assert!(source.contains("cuda_device_lost_error(operation, reason.clone())"));

    let try_submit = source
        .split("fn try_submit_completion(&self) -> BackendResult<Box<dyn BackendCompletion>>")
        .nth(1)
        .expect("CUDA checked completion submission should exist")
        .split("fn synchronize(&self) -> BackendResult<()>")
        .next()
        .expect("CUDA checked completion submission should precede legacy synchronize");
    assert!(try_submit.contains("self.check_pending_error()?"));

    let completion = source
        .split("impl BackendCompletion for CudaEventCompletion")
        .nth(1)
        .expect("CUDA event completion implementation should exist")
        .split("fn record_first_pending_error")
        .next()
        .expect("CUDA event completion should precede pending-error helpers");
    assert!(completion.contains("&self.pending_error"));
    assert!(completion.contains("check_pending_error_slot(&self.pending_error)?"));
    assert!(completion.contains("CudaFailureInjectionPoint::DeviceLoss"));
    assert!(completion.contains("backend_driver_synchronization_error"));
    assert!(completion.contains("cuda_error_state"));

    assert!(try_submit.contains("pending_error: self.pending_error.clone()"));
    assert!(try_submit.contains("backend_driver_launch_error"));
}

#[test]
fn cuda_strict_initialization_boundary_does_not_catch_panics() {
    let source = include_str!("cublas.rs");
    let compatibility = source
        .split("pub fn new(ordinal: usize)")
        .nth(1)
        .expect("CUDA compatibility constructor should exist")
        .split("pub fn try_new(ordinal: usize)")
        .next()
        .expect("strict constructor should follow compatibility constructor");
    assert!(compatibility.contains("Self::try_new(ordinal)"));
    assert!(compatibility.contains("catch_unwind"));

    let strict = source
        .split("pub fn try_new(ordinal: usize)")
        .nth(1)
        .expect("CUDA strict constructor should exist")
        .split("fn new_inner(ordinal: usize)")
        .next()
        .expect("strict constructor should precede initialization implementation");
    assert!(strict.contains("Self::new_inner(ordinal)"));
    assert!(!strict.contains("catch_unwind"));

    let default = source
        .split("pub fn try_new_default()")
        .nth(1)
        .expect("CUDA strict default constructor should exist");
    assert!(default.contains("Self::try_new(0)"));

    let module = include_str!("mod.rs");
    let strict_probe = module
        .split("pub fn try_cuda_device_count()")
        .nth(1)
        .expect("CUDA strict device-count probe should exist");
    assert!(!strict_probe.contains("catch_unwind"));
    let compatibility_probe = module
        .split("pub fn cuda_device_count()")
        .nth(1)
        .expect("CUDA compatibility device-count probe should exist")
        .split("pub fn try_cuda_device_count()")
        .next()
        .expect("strict probe should follow compatibility probe");
    assert!(compatibility_probe.contains("catch_unwind"));
}

#[test]
fn cuda_bem_legacy_failure_is_retained() {
    let source = include_str!("cublas.rs");
    let public_checked = source
        .split("pub fn try_bem_regular_pair_c32_batch")
        .nth(1)
        .expect("CUDA public checked BEM path should exist")
        .split("pub fn bem_regular_pair_c32_batch")
        .next()
        .expect("CUDA public legacy BEM path should follow checked path");
    assert!(public_checked.contains("self.check_pending_error()?"));

    let public_legacy = source
        .split("pub fn bem_regular_pair_c32_batch")
        .nth(1)
        .expect("CUDA public legacy BEM path should exist")
        .split("fn try_gemv_batched_native")
        .next()
        .expect("CUDA batched GEMV helper should follow public BEM path");
    assert!(public_legacy.contains("self.record_pending_error(error.clone())"));

    let trait_checked = source
        .split("fn try_bem_regular_pair_c32_batch")
        .nth(1)
        .expect("CUDA trait checked BEM path should exist")
        .split("fn bem_regular_pair_c32_batch")
        .next()
        .expect("CUDA trait legacy BEM path should follow checked path");
    assert!(trait_checked.contains("self.check_pending_error()?"));
}

#[test]
fn checked_dense_statuses_preserve_cublas_failure_class() {
    let allocation = backend_cublas_error(
        "gemm",
        cublas_sys::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED,
    );
    assert!(matches!(allocation, BackendError::Allocation { .. }));

    let execution = backend_cublas_error(
        "gemv",
        cublas_sys::cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED,
    );
    assert!(matches!(execution, BackendError::OperationFailed { .. }));

    let sparse_allocation = backend_cusparse_error(
        "spmv",
        cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_ALLOC_FAILED,
    );
    assert!(matches!(sparse_allocation, BackendError::Allocation { .. }));

    let sparse_execution = backend_cusparse_error(
        "spmm",
        cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_EXECUTION_FAILED,
    );
    assert!(matches!(
        sparse_execution,
        BackendError::OperationFailed { .. }
    ));
}

#[test]
fn cuda_failure_injection_points_are_typed_and_one_shot() {
    let cases = [
        CudaFailureInjectionPoint::Allocation,
        CudaFailureInjectionPoint::DescriptorCreation,
        CudaFailureInjectionPoint::Launch,
        CudaFailureInjectionPoint::TransferH2D,
        CudaFailureInjectionPoint::TransferD2H,
        CudaFailureInjectionPoint::TransferD2D,
        CudaFailureInjectionPoint::Synchronization,
        CudaFailureInjectionPoint::EventRecord,
        CudaFailureInjectionPoint::EventWait,
        CudaFailureInjectionPoint::DeviceLoss,
        CudaFailureInjectionPoint::Teardown,
    ];
    for point in cases {
        let slot = Mutex::new(Some(point));
        let error = take_cuda_failure(&slot, point).expect_err("injection should fire");
        let expected = match point {
            CudaFailureInjectionPoint::Allocation => {
                matches!(error, BackendError::Allocation { .. })
            }
            CudaFailureInjectionPoint::DescriptorCreation | CudaFailureInjectionPoint::Teardown => {
                matches!(error, BackendError::OperationFailed { .. })
            }
            CudaFailureInjectionPoint::Launch | CudaFailureInjectionPoint::EventRecord => {
                matches!(error, BackendError::Launch { .. })
            }
            CudaFailureInjectionPoint::TransferH2D
            | CudaFailureInjectionPoint::TransferD2H
            | CudaFailureInjectionPoint::TransferD2D => {
                matches!(error, BackendError::Transfer { .. })
            }
            CudaFailureInjectionPoint::Synchronization | CudaFailureInjectionPoint::EventWait => {
                matches!(error, BackendError::Synchronization { .. })
            }
            CudaFailureInjectionPoint::DeviceLoss => {
                matches!(error, BackendError::DeviceLost { .. })
            }
        };
        assert!(expected, "unexpected error class for {point:?}");
        assert!(
            take_cuda_failure(&slot, point).is_ok(),
            "injection must be one-shot"
        );
    }
}

#[test]
fn cuda_prebake_failed_stub_preserves_checked_error() {
    let mut prebaked = CudaCsrPrebaked::<f32>::failed_stub(
        4,
        4,
        cusparse_sys::cudaDataType_t::CUDA_R_32F,
        CsrRowBucketStats::from_row_ptr(&[]),
        BackendError::Allocation {
            reason: "injected cuSPARSE workspace failure".into(),
        },
    );
    assert!(matches!(
        prebaked.initialization_error.take(),
        Some(BackendError::Allocation { .. })
    ));
}

#[test]
fn cuda_prebake_invalid_stub_preserves_checked_error() {
    let mut prebaked = CudaCsrPrebaked::<f32>::failed_invalid_stub(
        4,
        4,
        cusparse_sys::cudaDataType_t::CUDA_R_32F,
        CsrRowBucketStats::from_row_ptr(&[]),
        "inconsistent CSR structure",
    );
    assert!(matches!(
        prebaked.initialization_error.take(),
        Some(BackendError::InvalidArgument {
            operation: "prebake_csr",
            reason,
        }) if reason == "inconsistent CSR structure"
    ));

    let source = include_str!("cublas.rs");
    let try_prebake = source
        .split("fn try_prebake_csr<'a, T: DeviceScalar>")
        .nth(1)
        .expect("CUDA checked CSR prebake should exist")
        .split("fn prebake_csr<'a, T: DeviceScalar>")
        .next()
        .expect("CUDA checked CSR prebake should precede legacy CSR prebake");
    assert!(try_prebake.contains("self.check_pending_error()?"));
    assert!(
        !try_prebake.contains("self.prebake_csr("),
        "checked CSR construction must not delegate to the log-and-stub legacy path"
    );

    let spmv = source
        .split("fn try_spmv_prebaked<T: DeviceScalar>")
        .nth(1)
        .expect("CUDA checked SpMV should exist")
        .split("fn spmv_prebaked<T: DeviceScalar>")
        .next()
        .expect("CUDA legacy SpMV should follow checked SpMV");
    assert!(spmv.contains("self.check_pending_error()?"));
    let spmv_legacy = source
        .split("fn spmv_prebaked<T: DeviceScalar>")
        .nth(1)
        .expect("CUDA legacy SpMV should exist")
        .split("fn try_spmm_prebaked<T: DeviceScalar>")
        .next()
        .expect("CUDA checked SpMM should follow legacy SpMV");
    assert!(spmv_legacy.contains("self.record_pending_error(error.clone())"));
    let spmm = source
        .split("fn try_spmm_prebaked<T: DeviceScalar>")
        .nth(1)
        .expect("CUDA checked SpMM should exist")
        .split("fn spmm_prebaked<T: DeviceScalar>")
        .next()
        .expect("CUDA legacy SpMM should follow checked SpMM");
    assert!(spmm.contains("self.check_pending_error()?"));
    let spmm_legacy = source
        .split("fn spmm_prebaked<T: DeviceScalar>")
        .nth(1)
        .expect("CUDA legacy SpMM should exist")
        .split("fn try_spmm_prebaked_colmajor<T: DeviceScalar>")
        .next()
        .expect("CUDA checked column-major SpMM should follow legacy SpMM");
    assert!(spmm_legacy.contains("self.record_pending_error(error.clone())"));
    let prebake_legacy = source
        .split("fn prebake_csr<'a, T: DeviceScalar>")
        .nth(1)
        .expect("CUDA CSR prebake should exist")
        .split("fn try_prebake_csr_native<'a, T: DeviceScalar>")
        .next()
        .expect("CUDA legacy CSR prebake should precede checked native CSR prebake");
    assert!(prebake_legacy.contains("CudaCsrPrebaked::failed_stub"));

    let prebake_native = source
        .split("fn try_prebake_csr_native<'a, T: DeviceScalar>")
        .nth(1)
        .expect("CUDA checked native CSR prebake should exist")
        .split("fn try_spmv_prebaked<T: DeviceScalar>")
        .next()
        .expect("CUDA native CSR prebake should precede checked SpMV");
    assert!(!prebake_native.contains("log::error!"));
    assert!(!prebake_native.contains("CudaCsrPrebaked::failed_stub"));
}

#[test]
fn cuda_sparse_checked_dispatch_does_not_turn_failed_stubs_into_success() {
    let source = include_str!("cublas.rs");
    let spmv_start = source
        .find("fn try_spmv_prebaked<T: DeviceScalar>")
        .expect("CUDA checked SpMV should exist");
    let spmv = &source[spmv_start..][..source[spmv_start..]
        .find("\n    fn spmv_prebaked<T: DeviceScalar>")
        .expect("CUDA legacy SpMV should follow checked SpMV")];
    assert!(spmv.contains("return Err(error.clone())"));
    assert!(spmv.find("initialization_error") < spmv.find("if prebaked.desc.is_null()"));

    let spmm_start = source
        .find("fn try_spmm_prebaked<T: DeviceScalar>")
        .expect("CUDA checked SpMM should exist");
    let spmm = &source[spmm_start..][..source[spmm_start..]
        .find("\n    fn spmm_prebaked<T: DeviceScalar>")
        .expect("CUDA legacy SpMM should follow checked SpMM")];
    assert!(spmm.contains("return Err(error.clone())"));
    assert!(spmm.find("initialization_error") < spmm.find("prebaked.desc.is_null()"));
}

#[test]
fn cuda_spmm_workspace_allocation_failure_cleans_descriptors() {
    let source = include_str!("cublas.rs");
    let spmm = source
        .split("fn try_spmm_prebaked<T: DeviceScalar>")
        .nth(1)
        .expect("CUDA checked SpMM should exist")
        .split("fn spmm_prebaked<T: DeviceScalar>")
        .next()
        .expect("CUDA checked SpMM should precede compatibility SpMM");
    assert!(
        !spmm.contains("Some(self.alloc_tracked(buffer_size)?)"),
        "workspace allocation errors must not bypass descriptor teardown"
    );
    let allocation = spmm
        .find("self.alloc_tracked(buffer_size)")
        .expect("checked SpMM should account workspace allocation");
    let allocation_branch = &spmm[allocation..][..spmm[allocation..]
        .find("workspace.bytes = buffer_size")
        .expect("workspace byte count should follow allocation")];
    assert!(allocation_branch.contains("cusparseDestroyDnMat(y_desc"));
    assert!(allocation_branch.contains("cusparseDestroyDnMat(x_desc"));
}

#[test]
fn cuda_checked_kernel_paths_preflight_sticky_errors() {
    let source = include_str!("cublas.rs");
    for (method, next_method) in [
        (
            "fn try_segmented_sum<T>",
            "fn try_batched_local_contract<T>",
        ),
        (
            "fn try_batched_local_contract<T>",
            "type Buffer<T: BufferElement>",
        ),
        (
            "fn try_fft_in_place<T: BufferElement>",
            "fn try_bem_regular_pair_c32_batch",
        ),
        (
            "fn try_multi_dot_with_mode<T: DeviceScalar>",
            "fn multi_dot<T: DeviceScalar>",
        ),
        (
            "fn try_multi_dot_strided_with_mode<T: DeviceScalar>",
            "fn multi_dot_strided<T: DeviceScalar>",
        ),
        (
            "fn try_basis_axpy_strided<T: DeviceScalar>",
            "fn basis_axpy_strided<T: DeviceScalar>",
        ),
    ] {
        let body = source
            .split(method)
            .nth(1)
            .expect("CUDA checked kernel method should exist")
            .split(next_method)
            .next()
            .expect("CUDA checked kernel method should precede its next method");
        assert!(body.contains("self.check_pending_error()?"));
    }
}

#[test]
fn local_kernel_module_has_all_capability_family_entry_points() {
    let source = include_str!("kernels/local_kernels.cu");
    for name in [
        "sonium_segmented_sum_f32",
        "sonium_segmented_sum_f64",
        "sonium_segmented_sum_c32",
        "sonium_segmented_sum_c64",
        "sonium_batched_local_contract_f32",
        "sonium_batched_local_contract_f64",
        "sonium_batched_local_contract_c32",
        "sonium_batched_local_contract_c64",
    ] {
        assert!(source.contains(name), "missing CUDA local kernel {name}");
    }
    let backend = include_str!("cublas.rs");
    assert!(backend.contains("fn try_segmented_sum<T>"));
    assert!(backend.contains("fn try_batched_local_contract<T>"));
    assert!(backend.contains("local_kernels.is_some()"));
}

#[test]
fn cuda_legacy_bem_wrapper_has_cpu_fallback() {
    let source = include_str!("cublas.rs");
    let public_methods = source
        .split("impl Backend for CudaBackend")
        .next()
        .expect("CUDA backend implementation should exist");
    assert!(public_methods.contains("using CPU fallback"));
    assert!(!public_methods.contains("CUDA BEM regular-pair evaluation failed: {error}"));
}

#[test]
fn spmv_prebaked_reuses_dense_vector_descriptors() {
    let source = include_str!("cublas.rs");
    let body = source
            .split("fn try_spmv_prebaked<T: DeviceScalar>")
            .nth(1)
            .expect("spmv_prebaked implementation should exist")
            .split("// ---------------------------------------------------------------------------\n// Type-id casting helper")
            .next()
            .expect("spmv_prebaked should precede type-id casting helper");

    assert!(
        !body.contains("cusparseCreateDnVec"),
        "spmv_prebaked should not recreate dense vector descriptors per call"
    );
    assert!(
        !body.contains("cusparseDestroyDnVec"),
        "spmv_prebaked should not destroy dense vector descriptors per call"
    );
    assert!(
        body.contains("cusparseDnVecSetValues"),
        "spmv_prebaked should retarget prebaked dense vector descriptors"
    );
}

#[test]
fn spmv_prebaked_exposes_inspector_record() {
    let source = include_str!("cublas.rs");
    let struct_body = source
        .split("pub struct CudaCsrPrebaked")
        .nth(1)
        .expect("CudaCsrPrebaked should exist")
        .split("impl<T: DeviceScalar> CudaCsrPrebaked")
        .next()
        .expect("CudaCsrPrebaked impl should follow the struct");
    let prebake_body = source
        .split("fn try_prebake_csr_native<'a, T: DeviceScalar>")
        .nth(1)
        .expect("checked native CSR prebake implementation should exist")
        .split("fn spmv_prebaked<T: DeviceScalar>")
        .next()
        .expect("checked native CSR prebake should precede spmv_prebaked");

    assert!(
        struct_body.contains("inspector: CsrSpmvInspector")
            && struct_body.contains("workspace_bytes: usize"),
        "CUDA prebaked CSR should retain inspector and workspace metadata"
    );
    assert!(
        prebake_body.contains("CsrRowBucketStats::try_from_row_ptr")
            && prebake_body.contains("CusparseGenericCsr")
            && prebake_body.contains("CUSPARSE_SPMV_ALG_DEFAULT"),
        "prebake_csr should cache row-bucket stats and cuSPARSE algorithm metadata"
    );
}

#[test]
fn cuda_overrides_multi_dot_with_fused_kernel_path() {
    let source = include_str!("cublas.rs");
    let impl_body = source
            .split("impl Backend for CudaBackend")
            .nth(1)
            .expect("CudaBackend Backend impl should exist")
            .split("// ---------------------------------------------------------------------------\n// Type-id casting helper")
            .next()
            .expect("Backend impl should precede type-id helper");

    assert!(
        impl_body.contains("fn try_multi_dot_with_mode<T: DeviceScalar>"),
        "CUDA backend should expose a checked multi-dot override"
    );
    assert!(
        impl_body.contains("try_multi_dot_real(")
            && impl_body.contains("&kernels.f32_kernel")
            && impl_body.contains("&kernels.f64_kernel"),
        "CUDA multi_dot should use fused real-valued kernels"
    );
    assert!(
        impl_body.contains("&kernels.c32_kernel") && impl_body.contains("&kernels.c64_kernel"),
        "CUDA multi_dot should use fused complex-valued kernels"
    );
}

#[test]
fn cuda_multi_dot_capability_matches_kernel_availability() {
    assert_eq!(cuda_multi_dot_capability(true), CapabilityState::Native);
    assert_eq!(
        cuda_multi_dot_capability(false),
        CapabilityState::HostStagedFallback
    );
}

#[test]
fn cuda_fft_capability_matches_complex_only_dispatch() {
    let fft = cuda_fft_capabilities(CapabilityState::VendorNative, CapabilityState::VendorNative);
    assert_eq!(fft.f32, CapabilityState::Unsupported);
    assert_eq!(fft.f64, CapabilityState::Unsupported);
    assert_eq!(fft.c32, CapabilityState::VendorNative);
    assert_eq!(fft.c64, CapabilityState::VendorNative);
}

#[test]
fn cuda_axpy_many_capability_distinguishes_fused_and_cublas_paths() {
    assert_eq!(cuda_axpy_many_capability(true), CapabilityState::Native);
    assert_eq!(
        cuda_axpy_many_capability(false),
        CapabilityState::VendorNative
    );
}

#[test]
fn cuda_runtime_stats_since_reports_per_workload_deltas() {
    let earlier = CudaRuntimeStatsSnapshot {
        kernel_launches: 4,
        library_operations: 7,
        synchronization_calls: 2,
        completion_events: 3,
        completion_waits: 1,
        transfer: TransferStatsSnapshot {
            h2d_operations: 2,
            h2d_bytes: 16,
            d2h_operations: 1,
            d2h_bytes: 8,
            d2d_operations: 5,
            d2d_bytes: 40,
        },
        ..CudaRuntimeStatsSnapshot::default()
    };
    let later = CudaRuntimeStatsSnapshot {
        kernel_launches: 9,
        library_operations: 11,
        synchronization_calls: 5,
        completion_events: 4,
        completion_waits: 3,
        transfer: TransferStatsSnapshot {
            h2d_operations: 3,
            h2d_bytes: 24,
            d2h_operations: 3,
            d2h_bytes: 24,
            d2d_operations: 8,
            d2d_bytes: 64,
        },
        live_allocation_bytes: 100,
        allocation_high_water_bytes: 200,
        live_staging_bytes: 12,
        staging_high_water_bytes: 32,
        driver_resources: CudaDriverResourceStatsSnapshot {
            live_contexts: 1,
            live_streams: 1,
            live_modules: 7,
            live_dynamic_libraries: 5,
            live_library_handles: 3,
            live_fft_plans: 0,
            live_completion_events: 2,
            initialization_baseline_free_bytes: 8192,
            initialization_free_bytes: 8192 - 4096,
            measured_driver_bytes: 4096,
        },
        ..CudaRuntimeStatsSnapshot::default()
    };
    let delta = later.since(earlier);
    assert_eq!(delta.kernel_launches, 5);
    assert_eq!(delta.library_operations, 4);
    assert_eq!(delta.synchronization_calls, 3);
    assert_eq!(delta.completion_events, 1);
    assert_eq!(delta.completion_waits, 2);
    assert_eq!(delta.transfer.h2d_bytes, 8);
    assert_eq!(delta.transfer.d2h_operations, 2);
    assert_eq!(delta.transfer.d2d_bytes, 24);
    assert_eq!(delta.live_allocation_bytes, 100);
    assert_eq!(delta.staging_high_water_bytes, 32);
    assert_eq!(delta.driver_resources.live_contexts, 1);
    assert_eq!(delta.driver_resources.live_modules, 7);
    assert_eq!(delta.driver_resources.live_library_handles, 3);
    assert_eq!(delta.driver_resources.live_completion_events, 2);
    assert_eq!(delta.driver_resources.live_fft_plans, 0);
    assert_eq!(
        delta.driver_resources.initialization_baseline_free_bytes,
        8192
    );
    assert_eq!(delta.driver_resources.initialization_free_bytes, 4096);
    assert_eq!(delta.driver_resources.measured_driver_bytes, 4096);
}

#[test]
fn cuda_cached_scratch_release_is_synchronized_and_complete() {
    let source = include_str!("cublas.rs");
    let body = source
        .split("pub fn try_release_cached_scratch")
        .nth(1)
        .expect("CUDA backend should expose explicit cached-scratch cleanup")
        .split("/// Bytes currently retained by temporary host staging vectors.")
        .next()
        .expect("cached-scratch cleanup should precede staging statistics");
    assert!(body.contains("self.try_synchronize()?"));
    assert!(body.contains("CudaMultiDotWorkspace::default()"));
    assert!(body.contains("CudaRealTransformWorkspace::default()"));
}

#[test]
fn cuda_checked_multi_dot_into_commits_transactionally() {
    let source = include_str!("cublas.rs");
    let body = source
        .split("fn try_multi_dot_with_mode_into<T: DeviceScalar>")
        .nth(1)
        .expect("CUDA checked multi-dot output path should exist")
        .split("fn multi_dot<T: DeviceScalar>")
        .next()
        .expect("legacy CUDA multi-dot should follow checked output path");

    assert!(body.contains("try_host_staging_vec(&self.staging_stats"));
    assert!(body.contains("out.copy_from_slice(&staged)"));
    assert!(
        !body.contains("out.iter_mut().zip"),
        "checked CUDA multi-dot must not publish per-basis results"
    );
    assert!(
        !body.contains("out.as_mut_ptr().cast"),
        "fused CUDA multi-dot must write staged output"
    );
}

#[test]
fn cuda_legacy_multi_dot_retains_typed_failure_without_silent_fallback() {
    let source = include_str!("cublas.rs");
    let impl_body = source
        .split("impl Backend for CudaBackend")
        .nth(1)
        .expect("CudaBackend Backend impl should exist");
    let legacy = impl_body
        .split("fn multi_dot<T: DeviceScalar>")
        .nth(1)
        .expect("legacy CUDA multi-dot should exist")
        .split("fn try_multi_dot_strided_with_mode<T: DeviceScalar>")
        .next()
        .expect("legacy CUDA multi-dot should precede strided checked multi-dot");

    assert!(legacy.contains("ReductionMode::Fast"));
    assert!(legacy.contains("try_multi_dot_with_mode(ReductionMode::Fast"));
    assert!(legacy.contains("self.record_pending_error(error)"));
    assert!(legacy.contains("Vec::new()"));
    assert!(!legacy.contains("zeroed_compatibility_result"));
    assert!(
        !legacy.contains("dot_deterministic"),
        "legacy failure handling must not read the device once per basis vector"
    );
}

#[test]
fn cuda_overrides_strided_multi_dot_with_flat_basis_kernel_path() {
    let source = include_str!("cublas.rs");
    let impl_body = source
            .split("impl Backend for CudaBackend")
            .nth(1)
            .expect("CudaBackend Backend impl should exist")
            .split("// ---------------------------------------------------------------------------\n// Type-id casting helper")
            .next()
            .expect("Backend impl should precede type-id helper");

    assert!(
        impl_body.contains("fn try_multi_dot_strided_with_mode<T: DeviceScalar>")
            && impl_body.contains("fn try_multi_dot_strided_with_mode_into<T: DeviceScalar>")
            && impl_body.contains("try_multi_dot_strided_real(")
            && impl_body.contains("try_multi_dot_strided_real_into(")
            && impl_body.contains("kernels.strided_f32_kernel")
            && impl_body.contains("kernels.strided_f64_kernel")
            && impl_body.contains("kernels.strided_c32_kernel")
            && impl_body.contains("kernels.strided_c64_kernel"),
        "CUDA should expose checked flat-basis multi-dot for all supported scalar types"
    );
    assert!(
        source.contains("multi_dot_strided: ScalarCapabilities::splat(multi_dot_strided)"),
        "CUDA capabilities should expose the strided multi-dot implementation state"
    );
    assert!(
        source.contains("sonium_multi_dot_strided_f32")
            && source.contains("sonium_multi_dot_strided_f64")
            && source.contains("sonium_multi_dot_strided_c32")
            && source.contains("sonium_multi_dot_strided_c64"),
        "CUDA should load all flat-basis multi-dot kernels"
    );
}

#[test]
fn cuda_checked_multi_dot_kernel_paths_are_fallible() {
    let source = include_str!("cublas.rs");
    let impl_body = source
            .split("impl Backend for CudaBackend")
            .nth(1)
            .expect("CudaBackend Backend impl should exist")
            .split("// ---------------------------------------------------------------------------\n// Type-id casting helper")
            .next()
            .expect("Backend impl should precede type-id helper");
    assert!(impl_body.contains("fn try_multi_dot_with_mode<T: DeviceScalar>"));
    assert!(impl_body.contains("fn try_multi_dot_with_mode_into<T: DeviceScalar>"));
    assert!(impl_body.contains("fn try_multi_dot_strided_with_mode<T: DeviceScalar>"));

    let helpers = source
        .split("fn try_multi_dot_strided_real<T: DeviceScalar>")
        .nth(1)
        .expect("checked strided multi-dot helper should exist")
        .split("impl Drop for CudaBackend")
        .next()
        .expect("multi-dot helpers should precede backend Drop");
    assert!(helpers.contains("fn try_multi_dot_real<T: DeviceScalar>"));
    assert!(helpers.contains("fn try_multi_dot_real_into<T: DeviceScalar>"));
    assert!(helpers.contains("backend_driver_launch_error"));
    assert!(helpers.contains("backend_driver_transfer_error"));
    assert!(!helpers.contains("expect("));
}

#[test]
fn cuda_strided_kernels_bind_checked_view_offsets_and_strides() {
    let source = include_str!("cublas.rs");
    assert!(source.contains("checked_nd_view_descriptor"));
    assert!(source.contains(".arg(&basis_offset)"));
    assert!(source.contains(".arg(&basis_row_stride)"));
    assert!(source.contains(".arg(&basis_col_stride)"));
    assert!(source.contains(".arg(&y_offset)"));
    assert!(source.contains(".arg(&y_stride)"));

    let kernels = include_str!("kernels/multi_dot_kernels.cu");
    assert!(kernels.contains("basis + basis_offset + basis_index * basis_row_stride"));
    assert!(kernels.contains("i * basis_col_stride"));
    assert!(kernels.contains("y_offset + i * y_stride"));
}

#[test]
fn cuda_strided_axpy_writes_back_through_output_view() {
    let kernels = include_str!("kernels/multi_dot_kernels.cu");
    for (name, next_name) in [
        (
            "sonium_basis_axpy_strided_f32",
            "sonium_basis_axpy_strided_f64",
        ),
        (
            "sonium_basis_axpy_strided_f64",
            "sonium_basis_axpy_strided_c32",
        ),
        (
            "sonium_basis_axpy_strided_c32",
            "sonium_basis_axpy_strided_c64",
        ),
        ("sonium_basis_axpy_strided_c64", "\n\n"),
    ] {
        let body = kernels
            .split(&format!("void {name}"))
            .nth(1)
            .expect("CUDA strided AXPY kernel should exist")
            .split(next_name)
            .next()
            .expect("CUDA strided AXPY kernel should precede the next kernel");
        assert!(
            body.contains("y[y_offset + i * y_stride]"),
            "{name} must honor the output view offset and stride"
        );
        assert!(
            !body.contains("y[i] =") && !body.contains("y[i].re ="),
            "{name} must not write the unstrided output base"
        );
    }
}

#[test]
fn cuda_fused_kernel_arguments_use_declared_widths() {
    let source = include_str!("cublas.rs");
    let axpy_many = source
        .split("fn axpy_many_cuda")
        .nth(1)
        .expect("CUDA fused AXPY helper should exist")
        .split("fn try_basis_axpy_strided_real")
        .next()
        .expect("CUDA fused AXPY helper should precede strided AXPY");
    assert!(!axpy_many.contains("debug_assert_eq!"));
    assert!(axpy_many.contains("checked_mul(std::mem::size_of::<usize>())"));
    assert!(axpy_many.contains("checked_mul(std::mem::size_of::<T>())"));

    let multi_dot = source
        .split("fn try_multi_dot_real_impl")
        .nth(1)
        .expect("CUDA fused multi-dot helper should exist")
        .split("fn try_multi_dot_real_into")
        .next()
        .expect("CUDA fused multi-dot helper should precede backend Drop");
    assert!(!multi_dot.contains("debug_assert_eq!"));
    assert!(multi_dot.contains("checked_mul(std::mem::size_of::<usize>())"));
    assert!(multi_dot.contains("checked_mul(std::mem::size_of::<T>()"));
    let cast = source
        .split("fn try_launch_cast_kernel(")
        .nth(1)
        .expect("CUDA cast helper should exist")
        .split("fn axpy_many_cuda")
        .next()
        .expect("CUDA cast helper should precede fused AXPY");
    assert!(cast.contains("let kernel_len = u64::try_from(len)"));
    assert!(cast.contains(".arg(&kernel_len)"));
    assert!(!cast.contains(".arg(&len)"));

    for (helper, next_helper, arg) in [
        (
            "fn try_basis_axpy_strided_real<T: DeviceScalar + PartialEq>",
            "fn try_basis_axpy_strided_host<T: DeviceScalar>",
            ".arg(&vector_len_u64)",
        ),
        (
            "fn try_multi_dot_strided_real<T: DeviceScalar>",
            "fn try_multi_dot_strided_host<T: DeviceScalar>",
            ".arg(&vector_len_u64)",
        ),
    ] {
        let body = source
            .split(helper)
            .nth(1)
            .expect("CUDA fused helper should exist")
            .split(next_helper)
            .next()
            .expect("CUDA fused helper should precede host fallback");
        assert!(body.contains(arg));
        assert!(body.contains(".arg(&basis_count_u64)"));
        assert!(!body.contains(".arg(&vector_len)"));
        assert!(!body.contains(".arg(&basis_count)"));
    }
}

#[test]
fn cuda_overrides_strided_basis_axpy_with_flat_basis_kernel_path() {
    let source = include_str!("cublas.rs");
    let impl_body = source
            .split("impl Backend for CudaBackend")
            .nth(1)
            .expect("CudaBackend Backend impl should exist")
            .split("// ---------------------------------------------------------------------------\n// Type-id casting helper")
            .next()
            .expect("Backend impl should precede type-id helper");
    assert!(
        impl_body.contains("fn basis_axpy_strided<T: DeviceScalar>")
            && impl_body.contains("basis_axpy_strided_f32")
            && impl_body.contains("basis_axpy_strided_f64")
            && impl_body.contains("basis_axpy_strided_c32")
            && impl_body.contains("basis_axpy_strided_c64"),
        "CUDA flat-basis basis_axpy_strided should cover all scalar types"
    );
    assert!(
        source.contains("sonium_basis_axpy_strided_f32")
            && source.contains("sonium_basis_axpy_strided_f64")
            && source.contains("sonium_basis_axpy_strided_c32")
            && source.contains("sonium_basis_axpy_strided_c64"),
        "CUDA should load all flat-basis AXPY kernels"
    );
    let kernels = include_str!("kernels/multi_dot_kernels.cu");
    assert!(
            kernels.contains(
                "sonium_basis_axpy_strided_f32(\n    const float* basis,\n    const float* alphas,\n    float* y,"
            ) && kernels.contains(
                "sonium_basis_axpy_strided_f64(\n    const double* basis,\n    const double* alphas,\n    double* y,"
            ) && kernels.contains(
                "sonium_basis_axpy_strided_c32(\n    const Complex32Kernel* basis,\n    const Complex32Kernel* alphas,\n    Complex32Kernel* y,"
            ) && kernels.contains(
                "sonium_basis_axpy_strided_c64(\n    const Complex64Kernel* basis,\n    const Complex64Kernel* alphas,\n    Complex64Kernel* y,"
            ),
            "CUDA flat-basis AXPY kernels should consume scaled coefficients"
        );
    assert!(
        source.contains("basis_axpy_strided: ScalarCapabilities::splat(multi_dot_strided)"),
        "CUDA capabilities expose the native/fallback basis AXPY state"
    );
}

#[test]
fn cuda_overrides_axpy_many_with_fused_kernel_path() {
    let source = include_str!("cublas.rs");
    let impl_body = source
            .split("impl Backend for CudaBackend")
            .nth(1)
            .expect("CudaBackend Backend impl should exist")
            .split("// ---------------------------------------------------------------------------\n// Type-id casting helper")
            .next()
            .expect("Backend impl should precede type-id helper");

    assert!(
        impl_body.contains("fn axpy_many<T: DeviceScalar>") && impl_body.contains("axpy_many_cuda"),
        "CUDA backend should not inherit Backend::axpy_many's serial default"
    );
    let checked = impl_body
        .split("fn try_axpy_many<T: DeviceScalar>")
        .nth(1)
        .expect("CUDA checked axpy_many should exist")
        .split("fn axpy_many<T: DeviceScalar>")
        .next()
        .expect("CUDA legacy axpy_many should follow checked path");
    assert!(
        checked.contains("require_scalar_capability") && checked.contains("capabilities.axpy_many"),
        "CUDA checked axpy_many must honor capability metadata"
    );
    assert!(
        source.contains("sonium_axpy_many_f32") && source.contains("sonium_axpy_many_f64"),
        "CUDA axpy_many should use fused real-valued kernels"
    );
    assert!(
        source.contains("sonium_axpy_many_c32") && source.contains("sonium_axpy_many_c64"),
        "CUDA axpy_many should use fused complex-valued kernels"
    );
}

#[test]
fn cuda_overrides_dense_batches_with_cublas_batched_calls() {
    let source = include_str!("cublas.rs");
    let impl_body = source
            .split("impl Backend for CudaBackend")
            .nth(1)
            .expect("CudaBackend Backend impl should exist")
            .split("// ---------------------------------------------------------------------------\n// Type-id casting helper")
            .next()
            .expect("Backend impl should precede type-id helper");

    assert!(
        impl_body.contains("fn gemv_batched<T: DeviceScalar>")
            && impl_body.contains("try_gemv_batched"),
        "CUDA backend should not inherit Backend::gemv_batched's serial default"
    );
    assert!(
        impl_body.contains("fn gemm_batched<T: DeviceScalar>")
            && impl_body.contains("try_gemm_batched"),
        "CUDA backend should not inherit Backend::gemm_batched's serial default"
    );
    // The batched entry points live in the extracted BLAS-2/3 submodule.
    let blas23 = include_str!("cublas_blas23.rs");
    assert!(
        blas23.contains("cublasSgemvBatched")
            && blas23.contains("cublasDgemvBatched")
            && blas23.contains("cublasCgemvBatched")
            && blas23.contains("cublasZgemvBatched"),
        "CUDA GEMV batches should call cuBLAS pointer-array batched entry points"
    );
    assert!(
        blas23.contains("cublasSgemmBatched")
            && blas23.contains("cublasDgemmBatched")
            && blas23.contains("cublasCgemmBatched")
            && blas23.contains("cublasZgemmBatched"),
        "CUDA GEMM batches should call cuBLAS pointer-array batched entry points"
    );
}

#[test]
fn cuda_batched_dense_helpers_return_status_on_count_overflow() {
    let source = include_str!("cublas_blas23.rs");
    for name in [
        "gemv_batched_f32",
        "gemv_batched_f64",
        "gemv_batched_c32_nt",
        "gemv_batched_c64_nt",
        "gemm_batched_f32",
        "gemm_batched_f64",
        "gemm_batched_c32",
        "gemm_batched_c64",
    ] {
        let body = source
            .split(&format!("pub unsafe fn {name}"))
            .nth(1)
            .expect("batched cuBLAS helper should exist")
            .split("pub unsafe fn")
            .next()
            .expect("batched cuBLAS helper body should be bounded");
        assert!(
            body.contains("CUBLAS_STATUS_INVALID_VALUE"),
            "{name} should return a cuBLAS status when the batch count cannot fit"
        );
        assert!(
            !body.contains("batch count exceeds i32") && !body.contains("expect("),
            "{name} should not panic on batch-count conversion"
        );
    }
}

#[test]
fn cuda_gemv_batched_validates_each_vector_length_before_dispatch() {
    let source = include_str!("cublas.rs");
    let start = source
        .find("fn try_gemv_batched_native<T: DeviceScalar>")
        .expect("checked batched GEMV helper should exist");
    let body = source[start..]
        .split("fn try_gemm_batched_native")
        .next()
        .expect("batched GEMV helper should precede batched GEMM");
    assert!(
        body.contains("item.x.len != item.n")
            && body.contains("item.y.len != item.m")
            && body.contains("BackendError::InvalidArgument"),
        "batched GEMV must reject malformed vector lengths with typed errors"
    );
    assert!(
        !body.contains("debug_assert_eq!(item.x.len")
            && !body.contains("debug_assert_eq!(item.y.len"),
        "batched GEMV must not rely on debug-only assertions for caller input"
    );
}

#[test]
fn checked_dense_dispatches_bind_all_tensor_operands() {
    let backend = include_str!("../backend/mod.rs");
    let backend_gemv = backend
        .split("fn try_gemv<T: DeviceScalar>")
        .nth(1)
        .expect("default checked GEMV should exist")
        .split("fn try_gemm<T: DeviceScalar>")
        .next()
        .expect("default checked GEMV should precede checked GEMM");
    assert!(backend_gemv.contains("checked_dense_matrix_descriptor"));
    assert!(backend_gemv.contains("checked_dense_vector_descriptor(x"));
    assert!(backend_gemv.contains("checked_dense_vector_descriptor(y"));

    let backend_gemm = backend
        .split("fn try_gemm<T: DeviceScalar>")
        .nth(1)
        .expect("default checked GEMM should exist")
        .split("/// Allocate buffer")
        .next()
        .expect("default checked GEMM should precede allocation");
    for operand in [
        "checked_dense_matrix_descriptor(a",
        "checked_dense_matrix_descriptor(b",
        "checked_dense_matrix_descriptor(c",
    ] {
        assert!(
            backend_gemm.contains(operand),
            "missing default GEMM binding: {operand}"
        );
    }

    let backend_batched_gemv = backend
        .split("fn try_gemv_batched<T: DeviceScalar>")
        .nth(1)
        .expect("default checked batched GEMV should exist")
        .split("fn try_gemm_batched<T: DeviceScalar>")
        .next()
        .expect("default checked batched GEMV should precede batched GEMM");
    assert!(backend_batched_gemv.contains("checked_dense_matrix_descriptor"));
    assert!(backend_batched_gemv.contains("checked_dense_vector_descriptor(item.x"));
    assert!(backend_batched_gemv.contains("checked_dense_vector_descriptor(item.y"));

    let backend_batched_gemm = backend
        .split("fn try_gemm_batched<T: DeviceScalar>")
        .nth(1)
        .expect("default checked batched GEMM should exist")
        .split("/// Build a [`Self::CsrPrebaked`")
        .next()
        .expect("default checked batched GEMM should precede sparse dispatch");
    for operand in [
        "checked_dense_matrix_descriptor(\n                item.a,",
        "checked_dense_matrix_descriptor(\n                item.b,",
        "checked_dense_matrix_descriptor(\n                item.c,",
    ] {
        assert!(
            backend_batched_gemm.contains(operand),
            "missing default batched GEMM binding: {operand}"
        );
    }

    let source = include_str!("cublas.rs");
    let cuda_gemv = source
        .split("fn try_gemv<T: DeviceScalar>")
        .nth(1)
        .expect("CUDA checked GEMV should exist")
        .split("fn try_gemm<T: DeviceScalar>")
        .next()
        .expect("CUDA checked GEMV should precede checked GEMM");
    assert!(cuda_gemv.contains("checked_dense_matrix_descriptor"));
    assert!(cuda_gemv.contains("checked_dense_vector_descriptor(x"));
    assert!(cuda_gemv.contains("checked_dense_vector_descriptor(y"));
    assert!(cuda_gemv.contains("require_scalar_capability::<T>"));

    let cuda_gemm = source
        .split("fn try_gemm<T: DeviceScalar>")
        .nth(1)
        .expect("CUDA checked GEMM should exist")
        .split("fn gemv<T: DeviceScalar>")
        .next()
        .expect("CUDA checked GEMM should precede compatibility GEMV");
    for operand in [
        "checked_dense_matrix_descriptor(a",
        "checked_dense_matrix_descriptor(b",
        "checked_dense_matrix_descriptor(c",
    ] {
        assert!(
            cuda_gemm.contains(operand),
            "missing CUDA GEMM binding: {operand}"
        );
    }
    assert!(cuda_gemm.contains("require_scalar_capability::<T>"));

    let cuda_batched_gemv = source
        .split("fn try_gemv_batched<T: DeviceScalar>")
        .nth(1)
        .expect("CUDA checked batched GEMV should exist")
        .split("fn gemv<T: DeviceScalar>")
        .next()
        .expect("CUDA checked batched GEMV should precede compatibility GEMV");
    assert!(cuda_batched_gemv.contains("checked_dense_matrix_descriptor"));
    assert!(cuda_batched_gemv.contains("checked_dense_vector_descriptor(item.x"));
    assert!(cuda_batched_gemv.contains("checked_dense_vector_descriptor(item.y"));

    let cuda_batched_gemm = source
        .split("fn try_gemm_batched<T: DeviceScalar>")
        .nth(1)
        .expect("CUDA checked batched GEMM should exist")
        .split("fn gemm<T: DeviceScalar>")
        .next()
        .expect("CUDA checked batched GEMM should precede compatibility GEMM");
    for operand in [
        "checked_dense_matrix_descriptor(\n                item.a,",
        "checked_dense_matrix_descriptor(\n                item.b,",
        "checked_dense_matrix_descriptor(\n                item.c,",
    ] {
        assert!(
            cuda_batched_gemm.contains(operand),
            "missing CUDA batched GEMM binding: {operand}"
        );
    }
}

#[test]
fn cuda_checked_sparse_dispatches_bind_capability_contract() {
    let source = include_str!("cublas.rs");
    let spmv = source
        .split("fn try_spmv_prebaked<T: DeviceScalar>")
        .nth(1)
        .expect("CUDA checked SpMV should exist")
        .split("fn spmv_prebaked<T: DeviceScalar>")
        .next()
        .expect("CUDA checked SpMV should precede compatibility SpMV");
    assert!(spmv.contains("require_scalar_capability::<T>"));
    assert!(spmv.contains("capabilities.csr_spmv"));

    let spmm = source
        .split("fn try_spmm_prebaked<T: DeviceScalar>")
        .nth(1)
        .expect("CUDA checked SpMM should exist")
        .split("fn spmm_prebaked<T: DeviceScalar>")
        .next()
        .expect("CUDA checked SpMM should precede compatibility SpMM");
    assert!(spmm.contains("require_scalar_capability::<T>"));
    assert!(spmm.contains("capabilities.csr_spmm"));
}

#[test]
fn cuda_csr_transpose_uses_checked_cusparse_conversion() {
    let source = include_str!("cublas.rs");
    let transpose = source
        .split("fn try_csr_transpose<T: DeviceScalar>")
        .nth(1)
        .expect("CUDA checked CSR transpose should exist")
        .split("fn prebake_csr<'a, T: DeviceScalar>")
        .next()
        .expect("CSR transpose should precede CSR prebake");
    assert!(transpose.contains("validate_host_csr_structure"));
    assert!(transpose.contains("cusparseCsr2cscEx2_bufferSize"));
    assert!(transpose.contains("cusparseCsr2cscEx2("));
    assert!(transpose.contains("self.cusparse_error(\"cusparseCsr2cscEx2\""));
    assert!(transpose.contains("record_library_operation"));
    assert!(!transpose.contains("catch_unwind"));
}

#[test]
fn cuda_legacy_dense_wrappers_delegate_to_checked_paths() {
    let source = include_str!("cublas.rs");
    let impl_body = source
            .split("impl Backend for CudaBackend")
            .nth(1)
            .expect("CudaBackend Backend impl should exist")
            .split("// ---------------------------------------------------------------------------\n// Type-id casting helper")
            .next()
            .expect("Backend impl should precede type-id helper");
    let gemv = impl_body
        .split("fn gemv<T: DeviceScalar>")
        .nth(1)
        .expect("legacy CUDA GEMV should exist")
        .split("fn gemv_batched<T: DeviceScalar>")
        .next()
        .expect("legacy CUDA GEMV should precede batched GEMV");
    let gemm = impl_body
        .split("fn gemm<T: DeviceScalar>")
        .nth(1)
        .expect("legacy CUDA GEMM should exist")
        .split("fn gemm_batched<T: DeviceScalar>")
        .next()
        .expect("legacy CUDA GEMM should precede batched GEMM");
    assert!(gemv.contains("self.try_gemv("));
    assert!(gemm.contains("self.try_gemm("));
    assert!(gemv.contains("self.record_pending_error(error)"));
    assert!(gemm.contains("self.record_pending_error(error)"));
    assert!(!gemv.contains("expect(") && !gemm.contains("expect("));

    let checked_gemv = impl_body
        .split("fn try_gemv<T: DeviceScalar>")
        .nth(1)
        .expect("checked CUDA GEMV should exist")
        .split("fn try_gemm<T: DeviceScalar>")
        .next()
        .expect("checked CUDA GEMV should precede checked GEMM");
    assert!(
        checked_gemv.contains("Transpose::ConjTrans")
            && checked_gemv.contains("return self.try_gemm(")
    );
}

#[test]
fn cuda_fft_adapter_is_wired_and_caches_batched_plans() {
    let backend = include_str!("cublas.rs");
    let cufft = include_str!("cufft.rs");
    let impl_body = backend
            .split("impl Backend for CudaBackend")
            .nth(1)
            .expect("CudaBackend Backend impl should exist")
            .split("// ---------------------------------------------------------------------------\n// Type-id casting helper")
            .next()
            .expect("Backend impl should precede type-id helper");

    assert!(
            backend.contains("#[path = \"cufft.rs\"]")
                && backend.contains("use cufft::CudaFft;")
                && backend.contains("fft: Option<CudaFft>")
                && backend.contains(
                    "CudaFft::new(\n            stream.clone(),\n            allocation_stats.clone(),\n            pending_error.clone(),\n        )",
                ),
            "CUDA backend should construct and retain the optional cuFFT adapter"
        );
    assert!(
        backend.contains("fft: cuda_fft_capabilities(fft_c32, fft_c64)")
            && backend.contains("CapabilityState::VendorNative"),
        "CUDA capabilities should expose cuFFT only when it is available"
    );
    assert!(
        impl_body.contains("fn try_fft_in_place<T: BufferElement>")
            && impl_body.contains("fft.exec_c32(buffer_c32, fft_len, inverse)")
            && impl_body.contains("fft.exec_c64(buffer_c64, fft_len, inverse)"),
        "CUDA Backend::try_fft_in_place should dispatch both complex precisions to cuFFT"
    );
    assert!(
        impl_body.contains("fn try_fft_strided_in_place<T: BufferElement>")
            && impl_body
                .contains("fft.exec_c32_strided(buffer_c32, fft_len, batch, stride, inverse)")
            && impl_body
                .contains("fft.exec_c64_strided(buffer_c64, fft_len, batch, stride, inverse)"),
        "CUDA Backend::try_fft_strided_in_place should dispatch both complex precisions to cuFFT"
    );
    for symbol in [
        "cufftPlanMany",
        "cufftGetSizeMany",
        "cufftSetStream",
        "cufftSetAutoAllocation",
        "cufftSetWorkArea",
        "cufftExecC2C",
        "cufftExecZ2Z",
        "cufftDestroy",
    ] {
        assert!(cufft.contains(symbol), "cuFFT adapter is missing {symbol}");
    }
    assert!(
        cufft.contains("Mutex<HashMap<CufftPlanLayout, CudaFftPlan>>")
            && cufft.contains("let key = CufftPlanLayout {")
            && cufft.contains("plans.get(&key)")
            && cufft.contains("plans.insert(")
            && cufft.contains("get_size_many")
            && cufft.contains("allocation_stats.try_reserve(work_size)")
            && cufft.contains("self.stream.alloc::<u8>(work_size)")
            && cufft.contains("set_work_area"),
        "cuFFT adapter should cache plans by precision, FFT length, batch, and stride"
    );
    assert!(
        backend.contains("let fft_c64 = fft_c32") && backend.contains("fn cuda_fft_capabilities"),
        "CUDA capabilities should advertise double-complex cuFFT"
    );
}
