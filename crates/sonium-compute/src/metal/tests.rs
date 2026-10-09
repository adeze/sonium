use super::*;
use approx::assert_relative_eq;

#[test]
fn metal_strided_view_copy_is_native_blit() {
    let source = include_str!("mod.rs");
    let body = source
        .split("fn try_copy_views_d2d<T: BufferElement>")
        .nth(1)
        .expect("Metal checked view-copy implementation should exist")
        .split("fn copy_d2d<T: BufferElement>")
        .next()
        .expect("Metal view-copy implementation should precede compatibility copy");
    assert!(body.contains("copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size"));
    assert!(body.contains("checked_view_descriptor"));
    assert!(!body.contains("try_copy_view_d2h"));
}

#[test]
fn metal_fmm_legacy_entry_points_reject_bad_shapes_without_panicking() {
    let source = include_str!("mod.rs");
    let phase = source
        .split("pub fn fmm_phase_mul_accumulate_c32(")
        .nth(1)
        .expect("Metal FMM phase entry point should exist")
        .split("/// FMM helper: gather")
        .next()
        .expect("Metal FMM phase entry point should precede gather");
    assert!(phase.contains("record_pending_error"));
    assert!(!phase.contains("assert_eq!("));

    let indexed = source
        .split("pub fn try_fmm_indexed_dense_matvec_accumulate_c32(")
        .nth(1)
        .expect("Metal indexed FMM entry point should exist")
        .split("// Native f32 op implementations")
        .next()
        .expect("Metal indexed FMM entry point should precede native ops");
    assert!(indexed.contains("checked_mul"));
    assert!(!indexed.contains("assert!("));

    let legacy = source
        .split("pub fn fmm_indexed_dense_matvec_accumulate_c32(")
        .nth(1)
        .expect("Metal indexed FMM compatibility wrapper should exist");
    assert!(legacy.contains("try_fmm_indexed_dense_matvec_accumulate_c32"));
    assert!(legacy.contains("record_pending_error"));
}

#[test]
fn metal_rank_n_view_copy_is_native_blit() {
    let source = include_str!("mod.rs");
    let body = source
        .split("fn try_copy_nd_views_d2d<T: BufferElement>")
        .nth(1)
        .expect("Metal rank-N checked view-copy implementation should exist")
        .split("fn copy_d2d<T: BufferElement>")
        .next()
        .expect("Metal rank-N view-copy implementation should precede compatibility copy");
    assert!(body.contains("checked_element_index"));
    assert!(body.contains("copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size"));
    assert!(body.contains("ViewAliasing::OverlapSafe"));
    assert!(body.contains("try_copy_nd_view_d2h"));
}

#[test]
fn metal_checked_copy_paths_record_transfer_stats() {
    let source = include_str!("mod.rs");
    assert!(source.contains("transfer_stats: Arc<BackendTransferStats>"));
    assert!(source.contains("record_h2d(std::mem::size_of_val(host))"));
    assert!(source.contains("record_d2h(std::mem::size_of_val(host))"));
    assert!(source.contains("record_d2d(bytes as usize)"));
    assert!(source.contains("fn transfer_stats(&self) -> TransferStatsSnapshot"));
}

#[test]
fn metal_allocation_accounting_tracks_clone_safe_live_and_high_water_bytes() {
    let stats = Arc::new(MetalAllocationStats::default());
    let allocation = Arc::new(MetalAllocationGuard::new(stats.clone(), 24));
    assert_eq!(stats.live_bytes.load(Ordering::Relaxed), 24);
    assert_eq!(stats.high_water_bytes.load(Ordering::Relaxed), 24);

    let clone = allocation.clone();
    drop(allocation);
    assert_eq!(stats.live_bytes.load(Ordering::Relaxed), 24);

    let second = MetalAllocationGuard::new(stats.clone(), 8);
    assert_eq!(stats.live_bytes.load(Ordering::Relaxed), 32);
    assert_eq!(stats.high_water_bytes.load(Ordering::Relaxed), 32);

    drop(second);
    drop(clone);
    assert_eq!(stats.live_bytes.load(Ordering::Relaxed), 0);
    assert_eq!(stats.high_water_bytes.load(Ordering::Relaxed), 32);
}

#[test]
fn metal_completion_error_poisoning_preserves_first_failure() {
    let pending_error = Mutex::new(None);
    record_first_pending_error(
        &pending_error,
        BackendError::CommandBuffer {
            backend: "metal",
            operation: "first",
            reason: "device lost".into(),
        },
    );
    record_first_pending_error(
        &pending_error,
        BackendError::CommandBuffer {
            backend: "metal",
            operation: "second",
            reason: "later failure".into(),
        },
    );

    let stored = pending_error.lock().unwrap().clone().unwrap();
    assert!(matches!(
        stored,
        BackendError::CommandBuffer {
            operation: "first",
            reason,
            ..
        } if reason == "device lost"
    ));
}

#[test]
fn metal_completion_wait_checks_sticky_error_after_command_status() {
    let source = include_str!("mod.rs");
    let completion = source
        .split("impl BackendCompletion for MetalCommandCompletion")
        .nth(1)
        .expect("Metal completion implementation should exist")
        .split("// ---------------------------------------------------------------------------")
        .next()
        .expect("Metal completion should precede encoder helpers");
    assert!(completion.contains("metal_error_state"));
    assert!(completion.contains("self.pending_error"));
}

#[test]
fn metal_flush_drains_scratch_before_returning_sticky_failure() {
    let source = include_str!("mod.rs");
    let flush = source
        .split("fn flush_pending_checked(&self) -> BackendResult<()> {")
        .nth(1)
        .expect("checked Metal flush implementation should exist")
        .split("fn flush_pending(&self)")
        .next()
        .expect("checked Metal flush should precede compatibility flush");
    assert!(flush.contains("let sticky_error ="));
    assert!(flush.contains("pending_scratch"));
    assert!(flush.contains("return_scratch_checked"));
    let drain = flush
        .find("let completed: Vec<_>")
        .expect("pending scratch should be drained");
    let first_return = flush
        .find("return Err(error)")
        .expect("flush should return typed errors");
    assert!(drain < first_return);
}

#[test]
fn metal_mps_kernel_cache_mutex_failure_is_typed() {
    let Some(backend) = try_metal_or_skip("metal_mps_kernel_cache_mutex_failure_is_typed") else {
        return;
    };

    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = backend
            .mps_kernels
            .lock()
            .expect("MPS kernel cache should start healthy");
        panic!("inject MPS kernel cache mutex poisoning");
    }));
    assert!(poisoned.is_err());

    let error = backend
        .mps_kernel_for(MpsMmKey {
            trans_left: false,
            trans_right: false,
            m: 1,
            n: 1,
            k: 1,
            alpha_bits: 1.0_f64.to_bits(),
            beta_bits: 0.0_f64.to_bits(),
        })
        .expect_err("poisoned MPS cache must return a typed error");
    assert!(matches!(
        error,
        BackendError::OperationFailed {
            operation: "metal_mps_kernel_cache",
            ..
        }
    ));
}

#[test]
fn metal_checked_dot_surfaces_sticky_completion_error() {
    let Some(backend) = try_metal_or_skip("metal_checked_dot_surfaces_sticky_completion_error")
    else {
        return;
    };
    let x = backend.raw_alloc::<f32>(1).expect("x allocation");
    let y = backend.raw_alloc::<f32>(1).expect("y allocation");
    backend.record_pending_error(BackendError::CommandBuffer {
        backend: "metal",
        operation: "injected_checked_dot",
        reason: "injected command-buffer failure".into(),
    });

    let error = backend
        .try_dot_with_mode(crate::backend::ReductionMode::Fast, &x, &y)
        .expect_err("checked dot must not return a neutral success after failure");
    assert!(matches!(
        error,
        BackendError::CommandBuffer {
            operation: "injected_checked_dot",
            ..
        }
    ));
}

#[test]
fn metal_legacy_scratch_mutex_failure_becomes_sticky_error() {
    let Some(backend) =
        try_metal_or_skip("metal_legacy_scratch_mutex_failure_becomes_sticky_error")
    else {
        return;
    };
    let buffer = backend
        .raw_alloc::<f32>(1)
        .expect("scratch test allocation");
    let raw = buffer.raw.0.clone();

    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = backend
            .scratch
            .lock()
            .expect("scratch mutex should start healthy");
        panic!("inject scratch mutex poisoning");
    }));
    assert!(poisoned.is_err());

    backend.return_scratch(4, raw);
    assert!(matches!(
        backend.check_pending_error(),
        Err(BackendError::OperationFailed {
            operation: "metal_scratch_pool",
            ..
        })
    ));
}

#[test]
fn metal_pending_mutex_failure_becomes_sticky_error() {
    let Some(backend) = try_metal_or_skip("metal_pending_mutex_failure_becomes_sticky_error")
    else {
        return;
    };
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = backend
            .pending
            .lock()
            .expect("pending mutex should start healthy");
        panic!("inject pending mutex poisoning");
    }));
    assert!(poisoned.is_err());
    assert!(backend.pending_command_buffer().is_none());
    assert!(matches!(
        backend.check_pending_error(),
        Err(BackendError::OperationFailed {
            operation: "metal_pending_queue",
            ..
        })
    ));
}

#[test]
fn metal_strided_limits_reject_native_u32_overflow() {
    assert!(MetalBackend::check_strided_backend_limits("test", 1024, 4).is_ok());
    if usize::BITS > u32::BITS {
        let over_u32 = (u32::MAX as usize)
            .checked_add(1)
            .expect("64-bit usize should represent u32::MAX + 1");
        assert!(matches!(
            MetalBackend::check_strided_backend_limits("test", over_u32, 1),
            Err(BackendError::InvalidArgument { .. })
        ));
        assert!(matches!(
            MetalBackend::check_strided_backend_limits("test", 1, over_u32),
            Err(BackendError::InvalidArgument { .. })
        ));
    }
}

#[test]
fn metal_gemv_limits_only_reject_native_u32_overflow() {
    assert!(MetalBackend::check_gemv_backend_limits::<f32>("test", 32, 32, 32).is_ok());
    assert!(
        MetalBackend::check_gemv_backend_limits::<f64>("test", usize::MAX, usize::MAX, usize::MAX,)
            .is_ok()
    );
    if usize::BITS > u32::BITS {
        let over_u32 = (u32::MAX as usize)
            .checked_add(1)
            .expect("64-bit usize should represent u32::MAX + 1");
        assert!(matches!(
            MetalBackend::check_gemv_backend_limits::<f32>("test", over_u32, 1, 1),
            Err(BackendError::InvalidArgument { .. })
        ));
        assert!(matches!(
            MetalBackend::check_gemv_backend_limits::<f32>("test", 1, 1, over_u32),
            Err(BackendError::InvalidArgument { .. })
        ));
    }
}

#[test]
fn metal_spmv_limits_only_reject_native_u32_overflow() {
    assert!(MetalBackend::check_spmv_backend_limits::<f32>("test", 32).is_ok());
    assert!(MetalBackend::check_spmv_backend_limits::<Complex32>("test", 32).is_ok());
    assert!(MetalBackend::check_spmv_backend_limits::<f64>("test", usize::MAX).is_ok());

    if usize::BITS > u32::BITS {
        let over_u32 = (u32::MAX as usize)
            .checked_add(1)
            .expect("64-bit usize should represent u32::MAX + 1");
        assert!(matches!(
            MetalBackend::check_spmv_backend_limits::<f32>("test", over_u32),
            Err(BackendError::InvalidArgument { .. })
        ));
        assert!(matches!(
            MetalBackend::check_spmv_backend_limits::<Complex32>("test", over_u32),
            Err(BackendError::InvalidArgument { .. })
        ));
    }
}

#[test]
fn metal_axpy_many_limits_reject_native_u32_and_byte_overflow() {
    assert!(MetalBackend::check_axpy_many_backend_limits::<f32>("test", 32, 4).is_ok());
    assert!(MetalBackend::check_axpy_many_backend_limits::<Complex32>("test", 32, 4).is_ok());
    assert!(
        MetalBackend::check_axpy_many_backend_limits::<f64>("test", usize::MAX, usize::MAX).is_ok()
    );

    if usize::BITS > u32::BITS {
        let over_u32 = (u32::MAX as usize)
            .checked_add(1)
            .expect("64-bit usize should represent u32::MAX + 1");
        assert!(matches!(
            MetalBackend::check_axpy_many_backend_limits::<f32>("test", over_u32, 1),
            Err(BackendError::InvalidArgument { .. })
        ));
        assert!(matches!(
            MetalBackend::check_axpy_many_backend_limits::<Complex32>("test", 1, over_u32),
            Err(BackendError::InvalidArgument { .. })
        ));

        // Each dimension is individually representable by the native
        // u32 ABI, but packing the full basis would overflow the host
        // byte-count arithmetic.  Reject it before scratch allocation.
        let max_native = u32::MAX as usize;
        assert!(matches!(
            MetalBackend::check_axpy_many_backend_limits::<f32>("test", max_native, max_native,),
            Err(BackendError::InvalidArgument { .. })
        ));
        assert!(matches!(
            MetalBackend::check_axpy_many_backend_limits::<Complex32>(
                "test", max_native, max_native,
            ),
            Err(BackendError::InvalidArgument { .. })
        ));
    }
}

#[test]
fn metal_native_arithmetic_paths_use_typed_shape_validation() {
    let source = include_str!("mod.rs");
    for (name, next) in [
        ("fn axpy_many_f32_native", "fn scale_f32_native"),
        ("fn axpy_many_c32_native", "fn scale_c32_native"),
        ("fn multi_dot_f32_native", "fn spmv_csr_f32_native"),
        ("fn multi_dot_c32_native", "fn spmv_csr_c32_native"),
    ] {
        let body = source
            .split(name)
            .nth(1)
            .expect("native arithmetic method should exist")
            .split(next)
            .next()
            .expect("native arithmetic method should have a following method");
        assert!(
            !body.contains("assert_eq!("),
            "{name} must not panic on shapes"
        );
        assert!(
            body.contains("checked_axpy_many_byte_sizes")
                || body.contains("check_axpy_many_backend_limits"),
            "{name} must validate dimensions and byte sizes"
        );
    }
}

#[test]
fn metal_mps_gemv_experimental_has_checked_and_legacy_paths() {
    let source = include_str!("mod.rs");
    let checked = source
        .split("pub fn try_gemv_f32_mps_experimental(")
        .nth(1)
        .expect("checked MPS GEMV entry point should exist")
        .split("/// Compatibility wrapper for the experimental MPS GEMV")
        .next()
        .expect("checked MPS GEMV should precede its compatibility wrapper");
    assert!(checked.contains("-> BackendResult<()>"));
    assert!(checked.contains("BackendError::InvalidArgument"));
    assert!(checked.contains("try_pending_command_buffer()?"));
    assert!(!checked.contains("assert!("));

    let wrapper = source
        .split("pub fn gemv_f32_mps_experimental(")
        .nth(1)
        .expect("legacy MPS GEMV entry point should exist")
        .split("/// `C := alpha * op(A) * op(B) + beta * C` on f32 via MPSMatrixMultiplication.")
        .next()
        .expect("legacy MPS GEMV should precede the next MPS operation");
    assert!(wrapper.contains("try_gemv_f32_mps_experimental"));
    assert!(wrapper.contains("record_pending_error"));
    assert!(!wrapper.contains("assert!("));
}

#[test]
fn metal_strided_basis_axpy_scales_with_checked_staging() {
    let source = include_str!("mod.rs");
    let body = source
        .split("fn basis_axpy_strided_native")
        .nth(1)
        .expect("Metal strided basis AXPY helper should exist")
        .split("/// Native MSL GEMV")
        .next()
        .expect("Metal strided basis AXPY helper should precede GEMV");
    assert!(body.contains("try_host_staging_vec"));
    assert!(body.contains("basis_axpy_strided coefficient staging"));
    assert!(!body.contains("collect::<Vec<_>>()"));
}

#[test]
fn metal_vector_limits_only_reject_native_u32_overflow() {
    assert!(MetalBackend::check_vector_backend_limits::<f32>("test", 32).is_ok());
    assert!(MetalBackend::check_vector_backend_limits::<Complex32>("test", 32).is_ok());
    assert!(MetalBackend::check_vector_backend_limits::<f64>("test", usize::MAX).is_ok());

    if usize::BITS > u32::BITS {
        let over_u32 = (u32::MAX as usize)
            .checked_add(1)
            .expect("64-bit usize should represent u32::MAX + 1");
        assert!(matches!(
            MetalBackend::check_vector_backend_limits::<f32>("test", over_u32),
            Err(BackendError::InvalidArgument { .. })
        ));
        assert!(matches!(
            MetalBackend::check_vector_backend_limits::<Complex32>("test", over_u32),
            Err(BackendError::InvalidArgument { .. })
        ));
    }
}

#[test]
fn metal_gather4_limits_only_reject_native_u32_overflow() {
    assert!(MetalBackend::check_gather4_backend_limits::<f32>("test", 32, 16, 4).is_ok());
    assert!(MetalBackend::check_gather4_backend_limits::<Complex32>("test", 32, 16, 4).is_ok());
    assert!(
        MetalBackend::check_gather4_backend_limits::<f64>(
            "test",
            usize::MAX,
            usize::MAX,
            usize::MAX,
        )
        .is_ok()
    );

    if usize::BITS > u32::BITS {
        let over_u32 = (u32::MAX as usize)
            .checked_add(1)
            .expect("64-bit usize should represent u32::MAX + 1");
        assert!(matches!(
            MetalBackend::check_gather4_backend_limits::<f32>("test", 1, 4, over_u32),
            Err(BackendError::InvalidArgument { .. })
        ));
        assert!(matches!(
            MetalBackend::check_gather4_backend_limits::<Complex32>("test", 1, over_u32, 1),
            Err(BackendError::InvalidArgument { .. })
        ));
    }
}

#[test]
fn metal_segmented_limits_only_reject_native_u32_overflow() {
    assert!(MetalBackend::check_segmented_backend_limits::<f32>("test", 32, 5, 4).is_ok());
    assert!(MetalBackend::check_segmented_backend_limits::<Complex32>("test", 32, 5, 4).is_ok());
    assert!(
        MetalBackend::check_segmented_backend_limits::<f64>(
            "test",
            usize::MAX,
            usize::MAX,
            usize::MAX,
        )
        .is_ok()
    );

    if usize::BITS > u32::BITS {
        let over_u32 = (u32::MAX as usize)
            .checked_add(1)
            .expect("64-bit usize should represent u32::MAX + 1");
        assert!(matches!(
            MetalBackend::check_segmented_backend_limits::<f32>("test", 1, 2, over_u32),
            Err(BackendError::InvalidArgument { .. })
        ));
        assert!(matches!(
            MetalBackend::check_segmented_backend_limits::<Complex32>("test", 1, 2, over_u32),
            Err(BackendError::InvalidArgument { .. })
        ));
    }
}

#[test]
fn strided_multi_dot_uses_flat_native_kernels_for_supported_types() {
    let source = include_str!("mod.rs");
    assert!(source.contains("fn multi_dot_strided<T: DeviceScalar>"));
    assert!(source.contains("multi_dot_strided_f32_native"));
    assert!(source.contains("multi_dot_strided_c32_native"));
    assert!(source.contains("multi_dot_strided: ScalarCapabilities::new"));
    let kernels = include_str!("kernels.metal");
    assert!(kernels.contains("sonium_multi_dot_f32"));
    assert!(kernels.contains("sonium_multi_dot_c32"));
}

#[test]
fn paired_metal_scratch_failures_return_prior_allocations() {
    let source = include_str!("mod.rs");

    let strided = source
        .split("fn multi_dot_strided_native<T: BufferElement>")
        .nth(1)
        .expect("strided multi-dot implementation should exist")
        .split("fn multi_dot_strided_f32_native")
        .next()
        .expect("strided multi-dot implementation should have an end");
    let strided_result_failure = strided
        .split("let Some(result_buf) = self.take_scratch(result_bytes) else {")
        .nth(1)
        .expect("strided multi-dot should acquire a result scratch buffer")
        .split("};")
        .next()
        .expect("strided result allocation branch should have an end");
    assert!(
        strided_result_failure.contains("self.return_scratch(partials_bytes, partials_buf)"),
        "strided multi-dot must return partials when result allocation fails"
    );

    let bem = source
        .split("pub fn bem_regular_pair_c32_batch(")
        .nth(1)
        .expect("BEM compatibility wrapper should exist")
        .split("// Host-staged f64 fallback")
        .next()
        .expect("BEM compatibility wrapper should have an end");
    let bem_output_failure = bem
        .split("let Some(out_buf) = self.take_scratch(out_bytes) else {")
        .nth(1)
        .expect("BEM wrapper should acquire an output scratch buffer")
        .split("};")
        .next()
        .expect("BEM output allocation branch should have an end");
    assert!(
        bem_output_failure.contains("self.return_scratch(in_bytes, in_buf)"),
        "BEM wrapper must return input scratch when output allocation fails"
    );
}

#[test]
fn native_multi_dot_zero_shapes_use_fallible_results() {
    let source = include_str!("mod.rs");
    for (helper, next) in [
        ("fn multi_dot_f32_native", "fn dot_c32_native"),
        ("fn multi_dot_c32_native", "fn multi_dot_strided_native"),
        (
            "fn multi_dot_strided_native",
            "fn multi_dot_strided_f32_native",
        ),
    ] {
        let body = source
            .split(helper)
            .nth(1)
            .expect("Metal native multi-dot helper should exist")
            .split(next)
            .next()
            .expect("Metal native multi-dot helper should be bounded");
        assert!(body.contains("try_zeroed_host"));
        assert!(!body.contains("vec![0.0_f32;"));
        assert!(!body.contains("vec![Complex32::new(0.0, 0.0);"));
        assert!(!body.contains("vec![T::default();"));
    }
}

#[test]
fn checked_multi_dot_into_preserves_caller_owned_output_contract() {
    let source = include_str!("mod.rs");
    let body = source
        .split("fn try_multi_dot_with_mode_into<T: DeviceScalar>")
        .nth(1)
        .expect("Metal checked multi-dot output helper should exist")
        .split("fn validate_strided_backend_limits")
        .next()
        .expect("Metal checked multi-dot helper should precede strided limits");
    assert!(body.contains("out.len() != x_basis.len()"));
    assert!(body.contains("try_host_staging_vec(&self.staging_stats"));
    assert!(body.contains("out.copy_from_slice(&result)"));
    assert!(body.contains("out.copy_from_slice(&staged)"));
    assert!(
        !body.contains("out.as_mut_ptr()"),
        "Metal checked multi-dot must write native results into staging"
    );
    assert!(body.contains("self.check_pending_error()?"));
    assert!(body.contains("ReductionMode::Fast") || body.contains("ReductionMode::Fast)"));
    assert!(body.contains("multi_dot_f32_native(&x_f32, y_f32, Some(out_f32))"));
    assert!(body.contains("multi_dot_c32_native(&x_c32, y_c32, Some(out_c32))"));
}

#[test]
fn checked_multi_dot_into_pending_error_preserves_destination() {
    let Some(backend) = try_metal_or_skip("checked_multi_dot_into_pending_error") else {
        return;
    };
    let x = backend.raw_alloc::<f32>(1).expect("x allocation");
    let y = backend.raw_alloc::<f32>(1).expect("y allocation");
    backend.record_pending_error(BackendError::CommandBuffer {
        backend: "metal",
        operation: "multi_dot_into",
        reason: "injected device failure".into(),
    });
    let mut output = [17.0_f32];
    let result = backend.try_multi_dot_with_mode_into(
        crate::backend::ReductionMode::Fast,
        &[&x],
        &y,
        &mut output,
    );
    assert!(result.is_err());
    assert_eq!(output, [17.0]);
}

#[test]
fn metal_strided_kernels_bind_checked_view_offsets_and_strides() {
    let source = include_str!("mod.rs");
    assert!(source.contains("checked_nd_view_descriptor"));
    assert!(source.contains("set_scalar(&encoder, 5, &basis_offset)"));
    assert!(source.contains("set_scalar(&encoder, 6, &basis_row_stride)"));
    assert!(source.contains("set_scalar(&encoder, 7, &basis_col_stride)"));
    assert!(source.contains("set_scalar(&encoder, 8, &y_offset)"));
    assert!(source.contains("set_scalar(&encoder, 9, &y_stride)"));

    let kernels = include_str!("kernels.metal");
    assert!(kernels.contains("basis_offset + b * basis_row_stride + i * basis_col_stride"));
    assert!(kernels.contains("y_offset + i * y_stride"));
    assert!(kernels.contains("basis_offset + basis * basis_row_stride"));
}

#[test]
fn metal_gather_binds_checked_views_before_native_dispatch() {
    let source = include_str!("mod.rs");
    let gather = source
        .split("fn gather4_weighted<T>")
        .nth(1)
        .expect("Metal gather implementation should exist")
        .split("fn synchronize")
        .next()
        .expect("Metal gather implementation should precede synchronization");
    assert!(gather.contains("checked_nd_view_descriptor"));
    assert!(gather.contains("ViewAliasing::Disjoint"));
    assert!(gather.contains("ViewAliasing::ReadOnly"));
}

#[test]
fn metal_gather_host_fallback_uses_checked_transfers() {
    let source = include_str!("mod.rs");
    let gather = source
        .split("fn gather4_weighted<T>")
        .nth(1)
        .expect("Metal gather implementation should exist")
        .split("fn synchronize")
        .next()
        .expect("Metal gather implementation should precede synchronization");
    assert!(gather.contains("self.try_copy_d2h(values, &mut values_host)"));
    assert!(gather.contains("self.try_copy_d2h(indices, &mut indices_host)"));
    assert!(gather.contains("self.try_copy_d2h(weights, &mut weights_host)"));
    assert!(gather.contains("self.try_copy_h2d(&out_host, out)"));
    assert!(!gather.contains("self.copy_d2h(values, &mut values_host)"));
    assert!(!gather.contains("self.copy_h2d(&out_host, out)"));
}

#[test]
fn metal_legacy_wrappers_retain_typed_argument_failures() {
    let source = include_str!("mod.rs");
    for (method, next_method) in [
        (
            "fn copy_h2d<T: BufferElement>",
            "fn copy_d2h<T: BufferElement>",
        ),
        (
            "fn copy_d2h<T: BufferElement>",
            "fn copy_d2d<T: BufferElement>",
        ),
        ("fn copy_d2d<T: BufferElement>", "fn copy_h2d_cast"),
        ("fn cast_to_lower<Upper>", "fn cast_from_lower<Upper>"),
        (
            "fn cast_from_lower<Upper>",
            "fn validate_gather4_backend_limits",
        ),
        ("fn gather4_weighted<T>", "fn synchronize"),
        (
            "fn axpy<T: DeviceScalar>",
            "fn validate_axpy_many_backend_limits",
        ),
        ("fn dot<T: DeviceScalar>", "fn nrm2<T: DeviceScalar>"),
        ("fn nrm2<T: DeviceScalar>", "fn multi_dot<T: DeviceScalar>"),
        (
            "fn multi_dot<T: DeviceScalar>",
            "fn validate_strided_backend_limits",
        ),
        (
            "fn multi_dot_strided<T: DeviceScalar>",
            "fn basis_axpy_strided<T: DeviceScalar>",
        ),
        (
            "fn basis_axpy_strided<T: DeviceScalar>",
            "fn scale<T: DeviceScalar>",
        ),
        (
            "fn scale<T: DeviceScalar>",
            "fn validate_gemv_backend_limits",
        ),
        ("fn gemv<T: DeviceScalar>", "fn gemm<T: DeviceScalar>"),
        (
            "fn gemm<T: DeviceScalar>",
            "fn prebake_csr<'a, T: DeviceScalar>",
        ),
    ] {
        let body = source
            .split(method)
            .nth(1)
            .expect("Metal legacy wrapper should exist")
            .split(next_method)
            .next()
            .expect("Metal legacy wrapper should precede its successor");
        assert!(
            body.contains("record_invalid_argument") || body.contains("record_pending_error(error"),
            "{method} must retain typed argument or execution failures"
        );
    }
}

#[test]
fn native_multi_dot_failure_after_blit_defers_concat_scratch() {
    let source = include_str!("mod.rs");
    for (name, next) in [
        ("fn multi_dot_f32_native", "fn multi_dot_strided_f32_native"),
        ("fn multi_dot_c32_native", "fn multi_dot_strided_c32_native"),
    ] {
        let body = source
            .split(name)
            .nth(1)
            .expect("native multi-dot helper should exist")
            .split(next)
            .next()
            .expect("native multi-dot helper should precede strided helper");
        let failure = body
            .split("let Some(partials_buf) = self.take_scratch(partials_bytes) else {")
            .nth(1)
            .expect("native multi-dot must handle partial allocation failure");
        assert!(
            failure.contains("self.defer_scratch_until_flush(concat_bytes, concat_buf)"),
            "{name} must keep the blit destination alive when reduction scratch allocation fails"
        );
    }
}

#[test]
fn strided_multi_dot_uses_checked_completion_flush() {
    let source = include_str!("mod.rs");
    let body = source
        .split("fn multi_dot_strided_native<T: BufferElement>")
        .nth(1)
        .expect("Metal strided multi-dot native helper should exist")
        .split("fn multi_dot_strided_f32_native")
        .next()
        .expect("Metal strided multi-dot helper should precede typed wrappers");

    assert!(
        body.contains("self.flush_pending_checked()"),
        "strided multi-dot must surface command-buffer completion failures"
    );
    assert!(
        body.contains("self.record_pending_error(error)"),
        "legacy Vec-returning compatibility must retain the typed failure"
    );

    let legacy_flush = source
        .split("fn flush_pending(&self)")
        .nth(1)
        .expect("Metal legacy flush helper should exist")
        .split("/// Look up a cached pipeline state")
        .next()
        .expect("Metal legacy flush helper should precede pipeline lookup");
    assert!(
        legacy_flush.contains("self.flush_pending_checked()")
            && legacy_flush.contains("self.record_pending_error(error)"),
        "legacy flush must delegate to the checked completion path"
    );
    assert!(
        !legacy_flush.contains(".expect("),
        "legacy flush must retain mutex failures instead of panicking"
    );
    let return_scratch = source
        .split("fn return_scratch(&self,")
        .nth(1)
        .expect("Metal legacy scratch return helper should exist")
        .split("fn return_scratch_checked")
        .next()
        .expect("Metal legacy scratch return helper should precede checked helper");
    assert!(
        return_scratch.contains("self.return_scratch_checked(bytes, buf)")
            && !return_scratch.contains("expect("),
        "legacy scratch return must retain pool failures"
    );
    let defer_scratch = source
        .split("fn defer_scratch_until_flush")
        .nth(1)
        .expect("Metal deferred scratch helper should exist")
        .split("fn try_pending_or_new(")
        .next()
        .expect("Metal deferred scratch helper should precede command-buffer helper");
    assert!(
        defer_scratch.contains("record_pending_error") && !defer_scratch.contains("expect("),
        "deferred scratch mutex failures must be sticky"
    );

    let pending_creation = source
        .split("fn try_pending_command_buffer(")
        .nth(1)
        .expect("Metal checked command-buffer wrapper should exist")
        .split("fn pending_command_buffer(")
        .next()
        .expect("Metal sticky command-buffer wrapper should follow checked helper");
    assert!(
        pending_creation.contains("try_pending_or_new")
            && pending_creation.contains("pending mutex poisoned"),
        "command-buffer creation must retain typed mutex and queue failures"
    );

    let pipeline_lookup = source
        .split("fn try_pso(")
        .nth(1)
        .expect("Metal checked pipeline lookup helper should exist")
        .split("fn pso(")
        .next()
        .expect("Metal legacy pipeline lookup should follow checked helper");
    assert!(
        pipeline_lookup.contains("operation: \"pipeline_lookup\"")
            && pipeline_lookup.contains("ok_or_else"),
        "pipeline lookup must expose typed failures"
    );
    let legacy_pipeline_lookup = source
        .split("fn pso(")
        .nth(1)
        .expect("Metal legacy pipeline lookup helper should exist")
        .split("/// Dispatch a 1-D kernel")
        .next()
        .expect("Metal pipeline lookup should precede dispatch helper");
    assert!(
        legacy_pipeline_lookup.contains("self.record_pending_error(error)")
            && !legacy_pipeline_lookup.contains("panic!("),
        "legacy pipeline lookup must retain typed failures without panicking"
    );
}

#[test]
fn strided_basis_axpy_uses_contiguous_native_kernels_for_supported_types() {
    let source = include_str!("mod.rs");
    assert!(source.contains("fn basis_axpy_strided<T: DeviceScalar>"));
    assert!(source.contains("basis_axpy_strided_native"));
    assert!(source.contains("kernel_names::AXPY_MANY_F32"));
    assert!(source.contains("kernel_names::AXPY_MANY_C32"));
    assert!(source.contains("basis_axpy_strided: ScalarCapabilities::new"));
    let checked = source
        .split("fn try_axpy_many<T: DeviceScalar>")
        .nth(1)
        .expect("Metal checked axpy_many should exist")
        .split("fn axpy_many<T: DeviceScalar>")
        .next()
        .expect("Metal legacy axpy_many should follow checked path");
    assert!(
        checked.contains("require_scalar_capability") && checked.contains("capabilities.axpy_many"),
        "Metal checked axpy_many must honor capability metadata"
    );
    let kernels = include_str!("kernels.metal");
    assert!(kernels.contains("sonium_axpy_many_f32"));
    assert!(kernels.contains("sonium_axpy_many_c32"));
}

#[test]
fn local_kernel_registration_matches_capability_table() {
    let source = include_str!("kernels.metal");
    for name in [
        "sonium_segmented_sum_f32",
        "sonium_segmented_sum_c32",
        "sonium_batched_local_contract_f32",
        "sonium_batched_local_contract_c32",
        "sonium_spmm_csr_f32",
        "sonium_spmm_csr_c32",
        "sonium_fft_stockham_radix2_stage_c32",
        "sonium_fft_stockham_radix4_stage_c32",
    ] {
        assert!(source.contains(name), "missing Metal local kernel {name}");
    }
    let backend = include_str!("mod.rs");
    assert!(backend.contains("kernel_names::SEGMENTED_SUM_F32"));
    assert!(backend.contains("kernel_names::BATCHED_LOCAL_CONTRACT_C32"));
    assert!(backend.contains("fn try_segmented_sum<T>"));
    assert!(backend.contains("fn try_batched_local_contract<T>"));
    assert!(backend.contains("csr_spmm: native_f32_c32,"));
    assert!(backend.contains("kernel_names::FFT_STOCKHAM_RADIX2_STAGE_C32"));
    assert!(backend.contains("kernel_names::FFT_STOCKHAM_RADIX4_STAGE_C32"));
}

#[test]
fn raw_allocations_report_overflow_and_support_empty_buffers() {
    let Some(backend) =
        try_metal_or_skip("raw_allocations_report_overflow_and_support_empty_buffers")
    else {
        return;
    };

    backend.set_allocation_cap_bytes(0);
    let empty = backend.raw_alloc::<f32>(0).expect("empty allocation");
    assert_eq!(empty.len, 0);
    assert_eq!(backend.live_allocation_bytes(), 0);
    drop(empty);
    assert_eq!(backend.live_allocation_bytes(), 0);
    assert!(matches!(
        backend.raw_alloc::<f32>(usize::MAX),
        Err(OomError::Invalid(reason)) if reason.contains("overflow")
    ));
    assert!(matches!(
        // SAFETY: this call only checks the allocation-size contract; no
        // value is read from the returned buffer.
        unsafe { backend.raw_alloc_uninitialized::<f32>(usize::MAX) },
        Err(OomError::Invalid(reason)) if reason.contains("overflow")
    ));
}

#[test]
fn metal_prebaked_legacy_failures_are_sticky() {
    let source = include_str!("mod.rs");
    let try_prebake = source
        .split("fn try_prebake_csr<'a, T: DeviceScalar>")
        .nth(1)
        .expect("Metal checked CSR prebake should exist")
        .split("fn try_spmv_prebaked<T: DeviceScalar>")
        .next()
        .expect("Metal checked SpMV should follow checked CSR prebake");
    assert!(try_prebake.contains("self.check_pending_error()?"));

    let try_spmv = source
        .split("fn try_spmv_prebaked<T: DeviceScalar>")
        .nth(1)
        .expect("Metal checked SpMV should exist")
        .split("fn try_spmm_prebaked<T: DeviceScalar>")
        .next()
        .expect("Metal checked SpMM should follow checked SpMV");
    assert!(try_spmv.contains("self.check_pending_error()?"));
    assert!(try_spmv.contains("self.check_pending_error()"));
    assert!(try_spmv.contains("require_scalar_capability::<T>"));
    assert!(try_spmv.contains("capabilities.csr_spmv"));

    let legacy_spmv = source
        .split("fn spmv_prebaked<T: DeviceScalar>")
        .nth(1)
        .expect("Metal legacy SpMV should exist")
        .split("fn spmm_prebaked<T: DeviceScalar>")
        .next()
        .expect("Metal legacy SpMM should follow legacy SpMV");
    assert!(legacy_spmv.contains("self.record_pending_error(error.clone())"));

    let try_spmm = source
        .split("fn try_spmm_prebaked<T: DeviceScalar>")
        .nth(1)
        .expect("Metal checked SpMM should exist")
        .split("fn try_spmm_prebaked_colmajor<T: DeviceScalar>")
        .next()
        .expect("Metal checked column-major SpMM should follow checked SpMM");
    assert!(try_spmm.contains("self.check_pending_error()?"));
    assert!(try_spmm.contains("self.check_pending_error()"));
    assert!(try_spmm.contains("require_scalar_capability::<T>"));
    assert!(try_spmm.contains("capabilities.csr_spmm"));

    let legacy_spmm = source
        .split("fn spmm_prebaked<T: DeviceScalar>")
        .nth(1)
        .expect("Metal legacy SpMM should exist")
        .split("fn spmm_prebaked_colmajor<T: DeviceScalar>")
        .next()
        .expect("Metal column-major SpMM should follow legacy SpMM");
    assert!(legacy_spmm.contains("self.record_pending_error(error.clone())"));
    assert!(!legacy_spmm.contains("assert!("));

    let direct_spmv = source
        .split("fn spmv_csr<T: DeviceScalar>")
        .nth(1)
        .expect("Metal direct CSR SpMV should exist")
        .split("fn spmv_csr_f32_native")
        .next()
        .expect("Metal direct CSR SpMV should precede native helper");
    assert!(direct_spmv.contains("self.record_pending_error(error)"));
    assert!(direct_spmv.contains("buffer lengths"));
}

#[test]
fn checked_bem_dispatch_surfaces_native_failures() {
    let source = include_str!("mod.rs");
    let checked = source
        .split("pub fn try_bem_regular_pair_c32_batch(")
        .nth(1)
        .expect("Metal checked BEM helper should exist")
        .split("pub fn bem_regular_pair_c32_batch(")
        .next()
        .expect("Metal legacy BEM wrapper should follow checked helper");
    assert!(checked.contains("self.check_pending_error()?"));
    assert!(checked.contains("take_scratch_checked"));
    assert!(checked.contains("try_pending_command_buffer"));
    assert!(checked.contains("flush_pending_checked"));
    assert!(checked.contains("u32::try_from(n)"));

    let backend_impl = source
        .split("impl Backend for MetalBackend")
        .nth(1)
        .expect("Metal backend implementation should exist")
        .split("fn transfer_stats")
        .next()
        .expect("Metal transfer statistics should follow backend operations");
    assert!(backend_impl.contains("fn try_bem_regular_pair_c32_batch"));
    assert!(backend_impl.contains("MetalBackend::try_bem_regular_pair_c32_batch"));
}

#[test]
fn checked_csr_prebake_rejects_malformed_structure_before_native_setup() {
    let Some(backend) =
        try_metal_or_skip("checked_csr_prebake_rejects_malformed_structure_before_native_setup")
    else {
        return;
    };
    let row_ptr = upload(&backend, &[0_i32, 2]);
    let col_idx = upload(&backend, &[0_i32]);
    let values = upload(&backend, &[1.0_f32]);

    assert!(matches!(
        backend.try_prebake_csr(1, 1, &row_ptr, &col_idx, &values),
        Err(BackendError::InvalidArgument { operation: "prebake_csr", reason })
            if reason.contains("terminal row pointer")
    ));
}

#[test]
fn fft_stockham_keeps_batches_device_resident() {
    let source = include_str!("mod.rs");
    let body = source
        .split("fn fft_stockham_c32(")
        .nth(1)
        .expect("Metal Stockham FFT implementation should exist")
        .split("    /// FMM helper:")
        .next()
        .expect("Metal Stockham FFT should precede the next helper");

    assert!(
        body.contains("FFT_STOCKHAM_RADIX2_STAGE_C32")
            && body.contains("FFT_STOCKHAM_RADIX4_STAGE_C32")
            && body.contains("let mut src_is_buffer = true")
            && body.contains("dispatch_1d_lazy")
            && body.contains("copy_d2d"),
        "Metal FFT should use device-resident radix-2/radix-4 Stockham stages and ping-pong copies"
    );
    assert!(body.contains("self.try_copy_d2d(&scratch_buffer, buffer)"));
    assert!(!body.contains("self.copy_d2d(&scratch_buffer, buffer)"));
    assert!(
        !body.contains("FFT_BIT_REVERSE_C32")
            && !body.contains("copy_d2h")
            && !body.contains("copy_h2d"),
        "Metal FFT must not stage intermediate batches through the host"
    );

    let kernels = include_str!("kernels.metal");
    assert!(
        kernels.contains("kernel void sonium_fft_stockham_radix2_stage_c32")
            && kernels.contains("kernel void sonium_fft_stockham_radix4_stage_c32")
            && !kernels.contains("kernel void sonium_fft_bit_reverse_c32"),
        "Metal FFT must register device-resident Stockham stages without bit reversal"
    );
    let radix4 = kernels
        .split("kernel void sonium_fft_stockham_radix4_stage_c32(")
        .nth(1)
        .expect("radix-4 Metal Stockham kernel should exist");
    assert!(
        radix4.contains("uint stride = fft_len / 4")
            && radix4.contains("base_out = batch_base + group * (m * 4) + lane")
            && radix4.contains("float base_angle")
            && radix4.contains("sign_i_b"),
        "radix-4 Metal Stockham kernel must use N/4 input stride, packed output groups, and signed twiddles"
    );
}

/// Try-or-skip wrapper that *prints* the skip reason so a green test
/// run that ran zero Metal kernels is visible in the output. The
/// previous pattern `let Some(b) = try_metal() else { return };`
/// caused every test to count as "passed" in sandboxed CI even though
/// nothing was actually exercised — that masked a row-major /
/// column-major bug in the gemv kernels for a full development cycle.
///
/// Honours `SONIUM_REQUIRE_METAL=1` (env): when set, the absence of a
/// Metal device upgrades the skip into a hard panic. Use it in CI on
/// Metal hosts to guarantee the suite actually ran.
fn try_metal_or_skip(test_name: &str) -> Option<MetalBackend> {
    match MetalBackend::new() {
        Ok(b) => Some(b),
        Err(e) => {
            if std::env::var_os("SONIUM_REQUIRE_METAL").is_some() {
                panic!("{test_name}: SONIUM_REQUIRE_METAL is set but no Metal device: {e:?}");
            }
            eprintln!("[metal::tests::{test_name}] SKIPPED: no Metal device ({e:?})");
            None
        }
    }
}

#[test]
fn stockham_fft_native_matches_direct_dft_for_batched_complex32() {
    let Some(backend) =
        try_metal_or_skip("stockham_fft_native_matches_direct_dft_for_batched_complex32")
    else {
        return;
    };
    let n = 16usize;
    let batch = 2usize;
    let input: Vec<Complex32> = (0..n * batch)
        .map(|index| Complex32::new((index as f32 * 0.17).sin(), (index as f32 * 0.11).cos()))
        .collect();
    let mut device = upload(&backend, &input);
    backend
        .try_fft_in_place(&mut device, n, false)
        .expect("native Metal FFT dispatch failed");
    let actual = download(&backend, &device);

    for batch_index in 0..batch {
        for k in 0..n {
            let mut expected = Complex32::new(0.0, 0.0);
            for j in 0..n {
                let phase = -std::f32::consts::TAU * j as f32 * k as f32 / n as f32;
                let (sin, cos) = phase.sin_cos();
                expected += input[batch_index * n + j] * Complex32::new(cos, sin);
            }
            assert_relative_eq!(
                actual[batch_index * n + k].re,
                expected.re,
                epsilon = 3.0e-4
            );
            assert_relative_eq!(
                actual[batch_index * n + k].im,
                expected.im,
                epsilon = 3.0e-4
            );
        }
    }

    backend
        .try_fft_in_place(&mut device, n, true)
        .expect("native Metal inverse FFT dispatch failed");
    let recovered = download(&backend, &device);
    for (actual, expected) in recovered.iter().zip(input.iter()) {
        assert_relative_eq!(actual.re / n as f32, expected.re, epsilon = 4.0e-4);
        assert_relative_eq!(actual.im / n as f32, expected.im, epsilon = 4.0e-4);
    }
}

fn upload<T: BufferElement>(b: &MetalBackend, host: &[T]) -> MetalBuffer<T> {
    let mut buf = b.raw_alloc::<T>(host.len()).unwrap();
    b.copy_h2d(host, &mut buf);
    buf
}

fn download<T: BufferElement>(b: &MetalBackend, buf: &MetalBuffer<T>) -> Vec<T> {
    let mut out = vec![unsafe { std::mem::zeroed::<T>() }; buf.len];
    b.copy_d2h(buf, &mut out);
    out
}

#[test]
fn copy_d2d_uses_metal_blit_encoder() {
    let source = include_str!("mod.rs");
    let body = source
        .split("fn copy_d2d<T: BufferElement>")
        .nth(1)
        .expect("copy_d2d implementation should exist")
        .split("fn copy_h2d_cast")
        .next()
        .expect("copy_d2d should precede copy_h2d_cast");

    assert!(
        !body.contains("flush_pending"),
        "Metal copy_d2d should enqueue a blit instead of forcing a host sync"
    );
    assert!(
        !body.contains("host_slice"),
        "Metal copy_d2d should not copy through host-visible slices"
    );
    assert!(
        body.contains("blit_encoder")
            && body.contains("copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size"),
        "Metal copy_d2d should use a Metal blit encoder"
    );
}

#[test]
fn write_only_inner_loop_ops_remain_batched_until_sync() {
    let Some(b) = try_metal_or_skip("write_only_inner_loop_ops_remain_batched_until_sync") else {
        return;
    };
    let x_host: Vec<f32> = (0..32).map(|i| i as f32 * 0.125).collect();
    let y_host: Vec<f32> = (0..32).map(|i| (i as f32 * 0.25).sin()).collect();
    let a_host: Vec<f32> = (0..16).map(|i| (i as f32 * 0.1).cos()).collect();
    let x = upload(&b, &x_host);
    let mut y = upload(&b, &y_host);
    let a = upload(&b, &a_host);
    let gemv_x = upload(&b, &[1.0_f32, -0.5, 0.25, 0.75]);
    let mut gemv_y = upload(&b, &[0.0_f32; 4]);

    assert!(
        b.pending.lock().expect("pending mutex poisoned").is_none(),
        "fresh Metal backend should not start with queued work"
    );
    b.axpy(0.5_f32, &x, &mut y);
    assert!(
        b.pending.lock().expect("pending mutex poisoned").is_some(),
        "axpy should enqueue without synchronizing"
    );
    b.scale(0.25_f32, &mut y);
    assert!(
        b.pending.lock().expect("pending mutex poisoned").is_some(),
        "scale should stay in the same pending batch"
    );
    b.gemv(
        Transpose::NoTrans,
        4,
        4,
        1.0_f32,
        &a,
        4,
        &gemv_x,
        0.0_f32,
        &mut gemv_y,
    );
    assert!(
        b.pending.lock().expect("pending mutex poisoned").is_some(),
        "native GEMV should not force an inner-loop sync"
    );

    b.synchronize().expect("backend synchronization failed");
    assert!(
        b.pending.lock().expect("pending mutex poisoned").is_none(),
        "explicit synchronize should clear the pending batch"
    );
}

#[test]
fn axpy_many_native_remains_batched_until_sync() {
    let Some(b) = try_metal_or_skip("axpy_many_native_remains_batched_until_sync") else {
        return;
    };
    let x0 = upload(&b, &[1.0_f32, 2.0, 3.0, 4.0]);
    let x1 = upload(&b, &[0.5_f32, 1.5, 2.5, 3.5]);
    let mut y = upload(&b, &[10.0_f32, 20.0, 30.0, 40.0]);

    b.axpy_many(&[2.0_f32, -0.5], &[&x0, &x1], &mut y);
    assert!(
        b.pending.lock().expect("pending mutex poisoned").is_some(),
        "axpy_many should enqueue without synchronizing"
    );
    assert!(
        !b.pending_scratch
            .lock()
            .expect("pending_scratch mutex poisoned")
            .is_empty(),
        "axpy_many staging buffers must stay alive until the batch completes"
    );

    b.synchronize().expect("backend synchronization failed");
    assert!(
        b.pending.lock().expect("pending mutex poisoned").is_none(),
        "explicit synchronize should clear the pending batch"
    );
    assert!(
        b.pending_scratch
            .lock()
            .expect("pending_scratch mutex poisoned")
            .is_empty(),
        "completed axpy_many staging buffers should return to scratch"
    );
}

#[test]
fn axpy_f32_native_matches_host_compute() {
    let Some(b) = try_metal_or_skip("axpy_f32_native_matches_host_compute") else {
        return;
    };
    let x_host: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.1).collect();
    let mut y_host: Vec<f32> = (0..1024).map(|i| (i as f32 + 1.0).sin()).collect();
    let x = upload(&b, &x_host);
    let mut y = upload(&b, &y_host);
    let alpha: f32 = 0.7;
    b.axpy(alpha, &x, &mut y);
    for (xi, yi) in x_host.iter().zip(y_host.iter_mut()) {
        *yi += alpha * xi;
    }
    let got = download(&b, &y);
    for (g, w) in got.iter().zip(y_host.iter()) {
        assert_relative_eq!(g, w, max_relative = 1e-6);
    }
}

#[test]
fn axpy_many_f32_native_matches_host_compute() {
    let Some(b) = try_metal_or_skip("axpy_many_f32_native_matches_host_compute") else {
        return;
    };
    let n = 1024;
    let x0_host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).sin()).collect();
    let x1_host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos()).collect();
    let x2_host: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0).ln()).collect();
    let mut y_host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).sin()).collect();
    let x0 = upload(&b, &x0_host);
    let x1 = upload(&b, &x1_host);
    let x2 = upload(&b, &x2_host);
    let mut y = upload(&b, &y_host);
    let alphas = [0.7_f32, -0.25, 1.1];

    b.axpy_many(&alphas, &[&x0, &x1, &x2], &mut y);
    for (((yi, x0), x1), x2) in y_host
        .iter_mut()
        .zip(x0_host.iter())
        .zip(x1_host.iter())
        .zip(x2_host.iter())
    {
        *yi += alphas[0] * *x0 + alphas[1] * *x1 + alphas[2] * *x2;
    }
    let got = download(&b, &y);
    for (g, w) in got.iter().zip(y_host.iter()) {
        assert_relative_eq!(g, w, max_relative = 1e-6);
    }
}

#[test]
fn scale_f32_native_matches_host_compute() {
    let Some(b) = try_metal_or_skip("scale_f32_native_matches_host_compute") else {
        return;
    };
    let mut x_host: Vec<f32> = (0..513).map(|i| (i as f32 * 0.13).cos()).collect();
    let mut x = upload(&b, &x_host);
    let alpha: f32 = -1.5;
    b.scale(alpha, &mut x);
    for v in x_host.iter_mut() {
        *v *= alpha;
    }
    let got = download(&b, &x);
    for (g, w) in got.iter().zip(x_host.iter()) {
        assert_relative_eq!(g, w, max_relative = 1e-6);
    }
}

#[test]
fn dot_f32_native_matches_host_compute() {
    let Some(b) = try_metal_or_skip("dot_f32_native_matches_host_compute") else {
        return;
    };
    let x_host: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.07).sin()).collect();
    let y_host: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.11).cos()).collect();
    let x = upload(&b, &x_host);
    let y = upload(&b, &y_host);
    let got = b.dot(&x, &y);
    let want: f32 = x_host.iter().zip(y_host.iter()).map(|(a, b)| a * b).sum();
    // Reduction order differs between scalar host sum and tree
    // reduction on the GPU; allow ~1e-4 relative tolerance.
    assert_relative_eq!(got, want, max_relative = 1e-4);
}

#[test]
fn nrm2_f32_native_matches_host_compute() {
    let Some(b) = try_metal_or_skip("nrm2_f32_native_matches_host_compute") else {
        return;
    };
    let x_host: Vec<f32> = (0..1500).map(|i| (i as f32 + 1.0).ln()).collect();
    let x = upload(&b, &x_host);
    let got = b.nrm2(&x);
    let want: f32 = x_host.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert_relative_eq!(got, want, max_relative = 1e-4);
}

#[test]
fn spmv_csr_f32_native_matches_host_compute() {
    let Some(b) = try_metal_or_skip("spmv_csr_f32_native_matches_host_compute") else {
        return;
    };
    // 1-D Laplacian `[−1, 2, −1]` of size n.
    let n = 256;
    let mut row_ptr_h = vec![0_i32; n + 1];
    let mut col_idx_h: Vec<i32> = Vec::new();
    let mut values_h: Vec<f32> = Vec::new();
    for i in 0..n {
        if i > 0 {
            col_idx_h.push(i as i32 - 1);
            values_h.push(-1.0);
        }
        col_idx_h.push(i as i32);
        values_h.push(2.0);
        if i + 1 < n {
            col_idx_h.push(i as i32 + 1);
            values_h.push(-1.0);
        }
        row_ptr_h[i + 1] = values_h.len() as i32;
    }
    let x_h: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
    let row_ptr = upload(&b, &row_ptr_h);
    let col_idx = upload(&b, &col_idx_h);
    let values = upload(&b, &values_h);
    let x = upload(&b, &x_h);
    let mut y = b.raw_alloc::<f32>(n).unwrap();
    b.spmv_csr::<f32>(n, n, &row_ptr, &col_idx, &values, &x, &mut y);
    let got = download(&b, &y);

    let mut want = vec![0.0_f32; n];
    for i in 0..n {
        let lo = row_ptr_h[i] as usize;
        let hi = row_ptr_h[i + 1] as usize;
        for k in lo..hi {
            want[i] += values_h[k] * x_h[col_idx_h[k] as usize];
        }
    }
    for (g, w) in got.iter().zip(want.iter()) {
        assert_relative_eq!(g, w, max_relative = 1e-5);
    }
}

#[test]
fn multi_dot_f32_native_matches_serial_dots() {
    let Some(b) = try_metal_or_skip("multi_dot_f32_native_matches_serial_dots") else {
        return;
    };
    let n = 1024;
    let k = 8;
    let y_host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin()).collect();
    let basis_host: Vec<Vec<f32>> = (0..k)
        .map(|j| {
            (0..n)
                .map(|i| (i as f32 + 1.0).ln() + j as f32 * 0.1)
                .collect()
        })
        .collect();

    let y_buf = upload(&b, &y_host);
    let basis_bufs: Vec<MetalBuffer<f32>> = basis_host.iter().map(|h| upload(&b, h)).collect();
    let basis_refs: Vec<&MetalBuffer<f32>> = basis_bufs.iter().collect();

    let got = b.multi_dot::<f32>(&basis_refs, &y_buf);

    // Reference: serial dots (which on Metal goes via dot_f32_native).
    let want: Vec<f32> = basis_refs.iter().map(|x| b.dot(*x, &y_buf)).collect();

    assert_eq!(got.len(), k);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        // Both compute the same partial-sum tree; small float-ordering
        // differences are possible. 1e-4 relative is generous.
        assert_relative_eq!(*g, *w, max_relative = 1e-4);
        // Also check against host dot.
        let host_want: f32 = basis_host[i]
            .iter()
            .zip(y_host.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert_relative_eq!(*g, host_want, max_relative = 1e-4);
    }
}

// -----------------------------------------------------------------------
// Complex32 parity tests
// -----------------------------------------------------------------------

#[test]
fn basis_axpy_strided_f32_native_matches_host_compute() {
    use crate::backend::Backend as _;

    let Some(b) = try_metal_or_skip("basis_axpy_strided_f32_native_matches_host_compute") else {
        return;
    };
    let vector_len = 257;
    let basis_count = 3;
    let basis_host: Vec<f32> = (0..vector_len * basis_count)
        .map(|i| (i as f32 * 0.037).sin())
        .collect();
    let coeffs = [0.7_f32, -0.25, 1.1];
    let alpha = -0.6_f32;
    let mut y_host: Vec<f32> = (0..vector_len).map(|i| (i as f32 * 0.071).cos()).collect();
    let basis = upload(&b, &basis_host);
    let mut y = upload(&b, &y_host);

    b.basis_axpy_strided(alpha, &basis, vector_len, basis_count, &coeffs, &mut y);
    for (i, value) in y_host.iter_mut().enumerate() {
        for basis_index in 0..basis_count {
            *value += alpha * coeffs[basis_index] * basis_host[basis_index * vector_len + i];
        }
    }

    let got = download(&b, &y);
    for (actual, expected) in got.iter().zip(y_host.iter()) {
        assert_relative_eq!(actual, expected, max_relative = 1e-5, epsilon = 1e-6);
    }
}

#[test]
fn axpy_c32_native_matches_host_compute() {
    let Some(b) = try_metal_or_skip("axpy_c32_native_matches_host_compute") else {
        return;
    };
    let x_host: Vec<Complex32> = (0..1024)
        .map(|i| Complex32::new((i as f32) * 0.1, ((i as f32) + 1.0).cos()))
        .collect();
    let mut y_host: Vec<Complex32> = (0..1024)
        .map(|i| Complex32::new(((i as f32) + 1.0).sin(), (i as f32) * 0.05))
        .collect();
    let x = upload(&b, &x_host);
    let mut y = upload(&b, &y_host);
    let alpha = Complex32::new(0.7, -0.3);
    b.axpy(alpha, &x, &mut y);
    for (xi, yi) in x_host.iter().zip(y_host.iter_mut()) {
        *yi += alpha * *xi;
    }
    let got = download(&b, &y);
    for (g, w) in got.iter().zip(y_host.iter()) {
        assert_relative_eq!(g.re, w.re, max_relative = 1e-6);
        assert_relative_eq!(g.im, w.im, max_relative = 1e-6);
    }
}

#[test]
fn axpy_many_c32_native_matches_host_compute() {
    let Some(b) = try_metal_or_skip("axpy_many_c32_native_matches_host_compute") else {
        return;
    };
    let n = 512;
    let x0_host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i as f32 * 0.09).sin(), (i as f32 * 0.03).cos()))
        .collect();
    let x1_host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i as f32 * 0.05).cos(), (i as f32 * 0.13).sin()))
        .collect();
    let mut y_host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i as f32 * 0.17).sin(), (i as f32 * 0.19).cos()))
        .collect();
    let x0 = upload(&b, &x0_host);
    let x1 = upload(&b, &x1_host);
    let mut y = upload(&b, &y_host);
    let alphas = [Complex32::new(0.7, -0.2), Complex32::new(-0.4, 0.3)];

    b.axpy_many(&alphas, &[&x0, &x1], &mut y);
    for ((yi, x0), x1) in y_host.iter_mut().zip(x0_host.iter()).zip(x1_host.iter()) {
        *yi += alphas[0] * *x0 + alphas[1] * *x1;
    }
    let got = download(&b, &y);
    for (g, w) in got.iter().zip(y_host.iter()) {
        assert_relative_eq!(g.re, w.re, max_relative = 1e-6);
        assert_relative_eq!(g.im, w.im, max_relative = 1e-6);
    }
}

#[test]
fn basis_axpy_strided_c32_native_matches_host_compute() {
    use crate::backend::Backend as _;

    let Some(b) = try_metal_or_skip("basis_axpy_strided_c32_native_matches_host_compute") else {
        return;
    };
    let vector_len = 193;
    let basis_count = 2;
    let basis_host: Vec<Complex32> = (0..vector_len * basis_count)
        .map(|i| Complex32::new((i as f32 * 0.041).sin(), (i as f32 * 0.023).cos()))
        .collect();
    let coeffs = [Complex32::new(0.7, -0.2), Complex32::new(-0.4, 0.3)];
    let alpha = Complex32::new(-0.6, 0.15);
    let mut y_host: Vec<Complex32> = (0..vector_len)
        .map(|i| Complex32::new((i as f32 * 0.071).cos(), (i as f32 * 0.053).sin()))
        .collect();
    let basis = upload(&b, &basis_host);
    let mut y = upload(&b, &y_host);

    b.basis_axpy_strided(alpha, &basis, vector_len, basis_count, &coeffs, &mut y);
    for (i, value) in y_host.iter_mut().enumerate() {
        for basis_index in 0..basis_count {
            *value += alpha * coeffs[basis_index] * basis_host[basis_index * vector_len + i];
        }
    }

    let got = download(&b, &y);
    for (actual, expected) in got.iter().zip(y_host.iter()) {
        assert_relative_eq!(actual.re, expected.re, max_relative = 1e-5, epsilon = 1e-6);
        assert_relative_eq!(actual.im, expected.im, max_relative = 1e-5, epsilon = 1e-6);
    }
}

#[test]
fn scale_c32_native_matches_host_compute() {
    let Some(b) = try_metal_or_skip("scale_c32_native_matches_host_compute") else {
        return;
    };
    let mut x_host: Vec<Complex32> = (0..513)
        .map(|i| Complex32::new(((i as f32) * 0.13).cos(), ((i as f32) * 0.17).sin()))
        .collect();
    let mut x = upload(&b, &x_host);
    let alpha = Complex32::new(-1.5, 0.4);
    b.scale(alpha, &mut x);
    for v in x_host.iter_mut() {
        *v = alpha * *v;
    }
    let got = download(&b, &x);
    for (g, w) in got.iter().zip(x_host.iter()) {
        assert_relative_eq!(g.re, w.re, max_relative = 1e-6);
        assert_relative_eq!(g.im, w.im, max_relative = 1e-6);
    }
}

#[test]
fn dot_c32_native_matches_host_hermitian() {
    let Some(b) = try_metal_or_skip("dot_c32_native_matches_host_hermitian") else {
        return;
    };
    let x_host: Vec<Complex32> = (0..2048)
        .map(|i| Complex32::new(((i as f32) * 0.07).sin(), ((i as f32) * 0.09).cos()))
        .collect();
    let y_host: Vec<Complex32> = (0..2048)
        .map(|i| Complex32::new(((i as f32) * 0.11).cos(), ((i as f32) * 0.13).sin()))
        .collect();
    let x = upload(&b, &x_host);
    let y = upload(&b, &y_host);
    let got: Complex32 = b.dot(&x, &y);
    // Reference: Σ conj(x_i) * y_i.
    let mut want = Complex32::new(0.0, 0.0);
    for (xi, yi) in x_host.iter().zip(y_host.iter()) {
        want += xi.conj() * yi;
    }
    // Tree reduction order on GPU vs scalar host order: ~1e-4 rel.
    assert_relative_eq!(got.re, want.re, max_relative = 1e-4);
    assert_relative_eq!(got.im, want.im, max_relative = 1e-4);
}

#[test]
fn nrm2_c32_native_matches_host_compute() {
    let Some(b) = try_metal_or_skip("nrm2_c32_native_matches_host_compute") else {
        return;
    };
    let x_host: Vec<Complex32> = (0..1500)
        .map(|i| Complex32::new(((i as f32) + 1.0).ln(), ((i as f32) * 0.5).sin()))
        .collect();
    let x = upload(&b, &x_host);
    let got: f32 = b.nrm2(&x);
    let want_sq: f32 = x_host.iter().map(|c| c.re * c.re + c.im * c.im).sum();
    let want = want_sq.sqrt();
    assert_relative_eq!(got, want, max_relative = 1e-4);
}

#[test]
fn spmv_csr_c32_native_matches_host_compute() {
    let Some(b) = try_metal_or_skip("spmv_csr_c32_native_matches_host_compute") else {
        return;
    };
    // Complex tridiagonal — diagonal is purely real, off-diagonals
    // include a small imaginary part to exercise the complex multiply.
    let n = 256;
    let mut row_ptr_h = vec![0_i32; n + 1];
    let mut col_idx_h: Vec<i32> = Vec::new();
    let mut values_h: Vec<Complex32> = Vec::new();
    for i in 0..n {
        if i > 0 {
            col_idx_h.push(i as i32 - 1);
            values_h.push(Complex32::new(-1.0, 0.05));
        }
        col_idx_h.push(i as i32);
        values_h.push(Complex32::new(2.0, 0.0));
        if i + 1 < n {
            col_idx_h.push(i as i32 + 1);
            values_h.push(Complex32::new(-1.0, -0.05));
        }
        row_ptr_h[i + 1] = values_h.len() as i32;
    }
    let x_h: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(((i as f32) * 0.1).sin(), ((i as f32) * 0.07).cos()))
        .collect();
    let row_ptr = upload(&b, &row_ptr_h);
    let col_idx = upload(&b, &col_idx_h);
    let values = upload(&b, &values_h);
    let x = upload(&b, &x_h);
    let mut y = b.raw_alloc::<Complex32>(n).unwrap();
    b.spmv_csr::<Complex32>(n, n, &row_ptr, &col_idx, &values, &x, &mut y);
    let got = download(&b, &y);

    let mut want = vec![Complex32::new(0.0, 0.0); n];
    for i in 0..n {
        let lo = row_ptr_h[i] as usize;
        let hi = row_ptr_h[i + 1] as usize;
        for k in lo..hi {
            want[i] += values_h[k] * x_h[col_idx_h[k] as usize];
        }
    }
    for (g, w) in got.iter().zip(want.iter()) {
        assert_relative_eq!(g.re, w.re, max_relative = 1e-5);
        assert_relative_eq!(g.im, w.im, max_relative = 1e-5);
    }
}

#[test]
fn multi_dot_c32_native_matches_serial_dots() {
    let Some(b) = try_metal_or_skip("multi_dot_c32_native_matches_serial_dots") else {
        return;
    };
    let n = 1024;
    let k = 8;
    let y_host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(((i as f32) * 0.07).sin(), ((i as f32) * 0.13).cos()))
        .collect();
    let basis_host: Vec<Vec<Complex32>> = (0..k)
        .map(|j| {
            (0..n)
                .map(|i| {
                    Complex32::new(
                        ((i as f32) + 1.0).ln() + (j as f32) * 0.1,
                        ((i as f32) * 0.05).cos() + (j as f32) * 0.07,
                    )
                })
                .collect()
        })
        .collect();

    let y_buf = upload(&b, &y_host);
    let basis_bufs: Vec<MetalBuffer<Complex32>> =
        basis_host.iter().map(|h| upload(&b, h)).collect();
    let basis_refs: Vec<&MetalBuffer<Complex32>> = basis_bufs.iter().collect();

    let got = b.multi_dot::<Complex32>(&basis_refs, &y_buf);
    let want: Vec<Complex32> = basis_refs.iter().map(|x| b.dot(*x, &y_buf)).collect();

    assert_eq!(got.len(), k);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_relative_eq!(g.re, w.re, max_relative = 1e-4);
        assert_relative_eq!(g.im, w.im, max_relative = 1e-4);
        // Cross-check against host conj(x) * y dot.
        let mut host_want = Complex32::new(0.0, 0.0);
        for (xi, yi) in basis_host[i].iter().zip(y_host.iter()) {
            host_want += xi.conj() * yi;
        }
        assert_relative_eq!(g.re, host_want.re, max_relative = 1e-4);
        assert_relative_eq!(g.im, host_want.im, max_relative = 1e-4);
    }
}

/// Known-answer sanity check that pinpoints row-major / column-major
/// confusion before the bigger sin/cos parity test tries to mask it.
/// Row-major A = [[1,2,3], [10,20,30]] with lda=3, x = [1,1,1], beta=0
/// must give y = [6, 60].
#[test]
fn gemv_f32_known_answer_row_major() {
    let Some(b) = try_metal_or_skip("gemv_f32_known_answer_row_major") else {
        return;
    };
    let a_host: Vec<f32> = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
    let x_host: Vec<f32> = vec![1.0, 1.0, 1.0];
    let y0: Vec<f32> = vec![0.0, 0.0];
    let a = upload(&b, &a_host);
    let x = upload(&b, &x_host);
    let mut y = upload(&b, &y0);
    b.gemv(Transpose::NoTrans, 2, 3, 1.0_f32, &a, 3, &x, 0.0, &mut y);
    let got = download(&b, &y);
    assert_relative_eq!(got[0], 6.0, max_relative = 1e-4);
    assert_relative_eq!(got[1], 60.0, max_relative = 1e-4);
}

/// Known-answer Trans GEMV. Stored A = [[1,2,3],[4,5,6]] (n=2 rows × m=3
/// cols, lda=3). A^T applied is 3×2. With x = [1, 1] (length n=2),
/// y = A^T * x = [1+4, 2+5, 3+6] = [5, 7, 9].
#[test]
fn gemv_f32_trans_known_answer() {
    let Some(b) = try_metal_or_skip("gemv_f32_trans_known_answer") else {
        return;
    };
    let a_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x_host: Vec<f32> = vec![1.0, 1.0];
    let y0: Vec<f32> = vec![0.0, 0.0, 0.0];
    let a = upload(&b, &a_host);
    let x = upload(&b, &x_host);
    let mut y = upload(&b, &y0);
    // m=3 (output rows = A^T rows), n=2 (sum dimension = stored rows of A)
    b.gemv(Transpose::Trans, 3, 2, 1.0_f32, &a, 3, &x, 0.0, &mut y);
    let got = download(&b, &y);
    assert_relative_eq!(got[0], 5.0, max_relative = 1e-4);
    assert_relative_eq!(got[1], 7.0, max_relative = 1e-4);
    assert_relative_eq!(got[2], 9.0, max_relative = 1e-4);
}

/// Larger Trans parity vs CPU. Stored A is n×m row-major with lda ≥ m.
#[test]
fn gemv_f32_trans_matches_cpu() {
    let Some(b) = try_metal_or_skip("gemv_f32_trans_matches_cpu") else {
        return;
    };
    let m = 64;
    let n = 80;
    let lda = m + 5;
    let mut a_host = vec![0.0f32; n * lda];
    for j in 0..n {
        for i in 0..m {
            a_host[j * lda + i] = ((i * 31 + j * 17 + 1) as f32 * 0.001).sin();
        }
    }
    let x_host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.09).cos()).collect();
    let y0_host: Vec<f32> = (0..m).map(|i| (i as f32 * 0.11).sin()).collect();

    let a = upload(&b, &a_host);
    let x = upload(&b, &x_host);
    let mut y = upload(&b, &y0_host);
    let alpha: f32 = 0.6;
    let beta: f32 = 0.3;
    b.gemv(Transpose::Trans, m, n, alpha, &a, lda, &x, beta, &mut y);
    let got = download(&b, &y);

    let mut want = y0_host.clone();
    cpu_blas2::gemv_f32(
        Transpose::Trans,
        m,
        n,
        alpha,
        &a_host,
        lda,
        &x_host,
        beta,
        &mut want,
    );
    for (g, w) in got.iter().zip(want.iter()) {
        assert_relative_eq!(g, w, max_relative = 1e-4, epsilon = 1e-5);
    }
}

/// ConjTrans for Complex32 — flips the conjugate flag in the MSL kernel.
#[test]
fn gemv_c32_conj_trans_matches_cpu() {
    let Some(b) = try_metal_or_skip("gemv_c32_conj_trans_matches_cpu") else {
        return;
    };
    let m = 48;
    let n = 56;
    let lda = m + 7;
    let mut a_host = vec![Complex32::new(0.0, 0.0); n * lda];
    for j in 0..n {
        for i in 0..m {
            let p = ((i * 13 + j * 23 + 1) as f32) * 0.001;
            a_host[j * lda + i] = Complex32::new(p.sin(), p.cos());
        }
    }
    let x_host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i as f32 * 0.07).cos(), (i as f32 * 0.09).sin()))
        .collect();
    let y0_host: Vec<Complex32> = (0..m)
        .map(|i| Complex32::new((i as f32 * 0.11).sin(), (i as f32 * 0.13).cos()))
        .collect();
    let a = upload(&b, &a_host);
    let x = upload(&b, &x_host);
    let mut y = upload(&b, &y0_host);
    let alpha = Complex32::new(0.7, -0.3);
    let beta = Complex32::new(-0.4, 0.2);
    b.gemv(Transpose::ConjTrans, m, n, alpha, &a, lda, &x, beta, &mut y);
    let got = download(&b, &y);

    let mut want = y0_host.clone();
    cpu_blas2::gemv_c32(
        Transpose::ConjTrans,
        m,
        n,
        alpha,
        &a_host,
        lda,
        &x_host,
        beta,
        &mut want,
    );
    for (g, w) in got.iter().zip(want.iter()) {
        assert_relative_eq!(g.re, w.re, max_relative = 1e-4, epsilon = 1e-5);
        assert_relative_eq!(g.im, w.im, max_relative = 1e-4, epsilon = 1e-5);
    }
}

#[test]
fn gemv_f32_native_notrans_matches_cpu() {
    let Some(b) = try_metal_or_skip("gemv_f32_native_notrans_matches_cpu") else {
        return;
    };
    // Row-major NoTrans: A is m×n with lda >= n. Non-tight stride to
    // exercise the kernel's `row*lda + col` arithmetic.
    let m = 64;
    let n = 96;
    let lda = n + 7;
    let mut a_host: Vec<f32> = vec![0.0; m * lda];
    for i in 0..m {
        for j in 0..n {
            a_host[i * lda + j] = ((i * 31 + j * 17 + 1) as f32 * 0.001).sin();
        }
    }
    let x_host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos()).collect();
    let y0_host: Vec<f32> = (0..m).map(|i| (i as f32 * 0.13).sin()).collect();

    let a = upload(&b, &a_host);
    let x = upload(&b, &x_host);
    let mut y = upload(&b, &y0_host);
    let alpha: f32 = 0.7;
    let beta: f32 = -0.4;

    b.gemv(Transpose::NoTrans, m, n, alpha, &a, lda, &x, beta, &mut y);
    let got = download(&b, &y);

    // Reference: CPU path over the same row-major buffer.
    let mut want = y0_host.clone();
    cpu_blas2::gemv_f32(
        Transpose::NoTrans,
        m,
        n,
        alpha,
        &a_host,
        lda,
        &x_host,
        beta,
        &mut want,
    );
    for (g, w) in got.iter().zip(want.iter()) {
        assert_relative_eq!(g, w, max_relative = 1e-4, epsilon = 1e-5);
    }
}

#[test]
fn gemv_f32_mps_experimental_matches_cpu() {
    let Some(b) = try_metal_or_skip("gemv_f32_mps_experimental_matches_cpu") else {
        return;
    };
    let m = 32;
    let n = 40;
    let lda = n + 3;
    let mut a_host = vec![0.0_f32; m * lda];
    for i in 0..m {
        for j in 0..n {
            a_host[i * lda + j] = ((i * 19 + j * 7 + 3) as f32 * 0.01).sin();
        }
    }
    let x_host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).cos()).collect();
    let y0_host: Vec<f32> = (0..m).map(|i| (i as f32 * 0.09).sin()).collect();
    let a = upload(&b, &a_host);
    let x = upload(&b, &x_host);
    let mut y = upload(&b, &y0_host);
    let alpha = 0.75_f32;
    let beta = -0.25_f32;

    b.gemv_f32_mps_experimental(Transpose::NoTrans, m, n, alpha, &a, lda, &x, beta, &mut y);
    let got = download(&b, &y);

    let mut want = y0_host;
    cpu_blas2::gemv_f32(
        Transpose::NoTrans,
        m,
        n,
        alpha,
        &a_host,
        lda,
        &x_host,
        beta,
        &mut want,
    );
    for (g, w) in got.iter().zip(want.iter()) {
        assert_relative_eq!(g, w, max_relative = 1e-4, epsilon = 1e-5);
    }
}

#[test]
fn gemv_f32_mps_experimental_packed_trans_matches_cpu() {
    let Some(b) = try_metal_or_skip("gemv_f32_mps_experimental_packed_trans_matches_cpu") else {
        return;
    };
    let m = 37;
    let n = 29;
    let lda = m + 5;
    let mut a_host = vec![0.0_f32; n * lda];
    for j in 0..n {
        for i in 0..m {
            a_host[j * lda + i] = ((i * 13 + j * 17 + 11) as f32 * 0.011).cos();
        }
    }
    let trans_lda = n;
    let mut a_trans = vec![0.0_f32; m * trans_lda];
    for j in 0..n {
        for i in 0..m {
            a_trans[i * trans_lda + j] = a_host[j * lda + i];
        }
    }
    let x_host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin()).collect();
    let y0_host: Vec<f32> = (0..m).map(|i| (i as f32 * 0.03).cos()).collect();
    let a = upload(&b, &a_trans);
    let x = upload(&b, &x_host);
    let mut y = upload(&b, &y0_host);
    let alpha = -0.65_f32;
    let beta = 0.35_f32;

    b.gemv_f32_mps_experimental(
        Transpose::NoTrans,
        m,
        n,
        alpha,
        &a,
        trans_lda,
        &x,
        beta,
        &mut y,
    );
    let got = download(&b, &y);

    let mut want = y0_host;
    cpu_blas2::gemv_f32(
        Transpose::Trans,
        m,
        n,
        alpha,
        &a_host,
        lda,
        &x_host,
        beta,
        &mut want,
    );
    for (g, w) in got.iter().zip(want.iter()) {
        assert_relative_eq!(g, w, max_relative = 1e-4, epsilon = 1e-5);
    }
}

#[test]
fn gemv_c32_native_notrans_matches_cpu() {
    let Some(b) = try_metal_or_skip("gemv_c32_native_notrans_matches_cpu") else {
        return;
    };
    let m = 48;
    let n = 73;
    let lda = n + 5; // row-major NoTrans requires lda >= n
    let mut a_host: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); m * lda];
    for i in 0..m {
        for j in 0..n {
            let p = ((i * 13 + j * 23 + 1) as f32) * 0.001;
            a_host[i * lda + j] = Complex32::new(p.sin(), p.cos());
        }
    }
    let x_host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i as f32 * 0.07).cos(), (i as f32 * 0.09).sin()))
        .collect();
    let y0_host: Vec<Complex32> = (0..m)
        .map(|i| Complex32::new((i as f32 * 0.11).sin(), (i as f32 * 0.13).cos()))
        .collect();

    let a = upload(&b, &a_host);
    let x = upload(&b, &x_host);
    let mut y = upload(&b, &y0_host);
    let alpha = Complex32::new(0.7, -0.3);
    let beta = Complex32::new(-0.4, 0.2);

    b.gemv(Transpose::NoTrans, m, n, alpha, &a, lda, &x, beta, &mut y);
    let got = download(&b, &y);

    let mut want = y0_host.clone();
    cpu_blas2::gemv_c32(
        Transpose::NoTrans,
        m,
        n,
        alpha,
        &a_host,
        lda,
        &x_host,
        beta,
        &mut want,
    );
    for (g, w) in got.iter().zip(want.iter()) {
        assert_relative_eq!(g.re, w.re, max_relative = 1e-4, epsilon = 1e-5);
        assert_relative_eq!(g.im, w.im, max_relative = 1e-4, epsilon = 1e-5);
    }
}

#[test]
fn gemm_f32_policy_keeps_small_on_cpu_and_large_on_mps() {
    assert!(!MetalBackend::f32_gemm_uses_mps(256, 256, 256));
    assert!(MetalBackend::f32_gemm_uses_mps(257, 257, 257));
}

#[test]
fn gemm_capability_is_conservative_for_shape_dependent_dispatch() {
    let gemm = MetalBackend::gemm_capabilities();
    assert_eq!(gemm.f32, CapabilityState::HostSharedFallback);
    assert_eq!(gemm.f64, CapabilityState::HostSharedFallback);
    assert_eq!(gemm.c32, CapabilityState::HostSharedFallback);
    assert_eq!(gemm.c64, CapabilityState::HostSharedFallback);
    assert_ne!(gemm.f32, CapabilityState::VendorNative);
    assert_ne!(gemm.c32, CapabilityState::VendorNative);
}

#[test]
fn axpy_many_capability_matches_metal_scalar_paths() {
    let axpy_many = MetalBackend::native_f32_c32_capabilities();
    assert_eq!(axpy_many.f32, CapabilityState::Native);
    assert_eq!(axpy_many.c32, CapabilityState::Native);
    assert_eq!(axpy_many.f64, CapabilityState::HostSharedFallback);
    assert_eq!(axpy_many.c64, CapabilityState::HostSharedFallback);
}

#[test]
fn fft_capability_matches_complex32_only_dispatch() {
    let fft = MetalBackend::fft_capabilities();
    assert_eq!(fft.f32, CapabilityState::Unsupported);
    assert_eq!(fft.f64, CapabilityState::Unsupported);
    assert_eq!(fft.c32, CapabilityState::Native);
    assert_eq!(fft.c64, CapabilityState::Unsupported);
}

#[test]
fn gemm_c32_policy_keeps_small_on_cpu_and_large_on_mps() {
    assert!(!MetalBackend::c32_gemm_uses_mps(256, 256, 256));
    assert!(MetalBackend::c32_gemm_uses_mps(257, 257, 257));
}

/// Known-answer tiny GEMM goes through the shared-memory CPU/SME2 policy.
/// Row-major A = [[1,2,3],[4,5,6]] (m=2, k=3), B = identity (k=3, n=3),
/// expects C = A.
#[test]
fn gemm_f32_small_shared_cpu_known_answer() {
    let Some(b) = try_metal_or_skip("gemm_f32_small_shared_cpu_known_answer") else {
        return;
    };
    assert!(!MetalBackend::f32_gemm_uses_mps(2, 3, 3));
    let a_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_host: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let c0: Vec<f32> = vec![0.0; 6];
    let a = upload(&b, &a_host);
    let bm = upload(&b, &b_host);
    let mut c = upload(&b, &c0);
    b.gemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        2,
        3,
        3,
        1.0_f32,
        &a,
        3,
        &bm,
        3,
        0.0,
        &mut c,
        3,
    );
    let got = download(&b, &c);
    for (i, (g, w)) in got.iter().zip(a_host.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "mps gemm known-answer mismatch at {i}: got {g}, want {w}"
        );
    }
}

#[test]
fn gemm_f32_small_shared_cpu_flushes_pending_gpu_writes() {
    let Some(b) = try_metal_or_skip("gemm_f32_small_shared_cpu_flushes_pending_gpu_writes") else {
        return;
    };
    let m = 2;
    let n = 3;
    let k = 3;
    assert!(!MetalBackend::f32_gemm_uses_mps(m, n, k));

    let a_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_host: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let c0: Vec<f32> = vec![0.25, -0.5, 0.75, 1.0, -1.25, 1.5];
    let pending_add: Vec<f32> = vec![0.5, 1.0, -0.25, 2.0, -1.5, 0.25];

    let a = upload(&b, &a_host);
    let bm = upload(&b, &b_host);
    let add = upload(&b, &pending_add);
    let mut c = upload(&b, &c0);

    b.axpy(2.0_f32, &add, &mut c);
    b.gemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        m,
        n,
        k,
        1.0_f32,
        &a,
        3,
        &bm,
        3,
        1.0,
        &mut c,
        3,
    );
    let got = download(&b, &c);

    let mut want = c0.clone();
    for (w, x) in want.iter_mut().zip(pending_add.iter()) {
        *w += 2.0 * x;
    }
    cpu_blas3::gemm_f32(
        Transpose::NoTrans,
        Transpose::NoTrans,
        m,
        n,
        k,
        1.0,
        &a_host,
        3,
        &b_host,
        3,
        1.0,
        &mut want,
        3,
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "small CPU GEMM pending-flush mismatch at {i}: got {g}, want {w}"
        );
    }
}

#[test]
fn gemm_c32_small_shared_cpu_flushes_pending_gpu_writes() {
    let Some(b) = try_metal_or_skip("gemm_c32_small_shared_cpu_flushes_pending_gpu_writes") else {
        return;
    };
    let m = 2;
    let n = 3;
    let k = 3;
    assert!(!MetalBackend::c32_gemm_uses_mps(m, n, k));

    let a_host: Vec<Complex32> = vec![
        Complex32::new(1.0, 0.5),
        Complex32::new(2.0, -0.25),
        Complex32::new(3.0, 0.75),
        Complex32::new(4.0, -0.5),
        Complex32::new(5.0, 0.25),
        Complex32::new(6.0, -0.75),
    ];
    let b_host: Vec<Complex32> = vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(1.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(1.0, 0.0),
    ];
    let c0: Vec<Complex32> = vec![
        Complex32::new(0.25, -0.5),
        Complex32::new(-0.5, 0.75),
        Complex32::new(0.75, 0.25),
        Complex32::new(1.0, -1.25),
        Complex32::new(-1.25, 1.5),
        Complex32::new(1.5, -0.25),
    ];
    let pending_add: Vec<Complex32> = vec![
        Complex32::new(0.5, 0.25),
        Complex32::new(1.0, -0.5),
        Complex32::new(-0.25, 0.75),
        Complex32::new(2.0, 0.5),
        Complex32::new(-1.5, -1.0),
        Complex32::new(0.25, 1.25),
    ];

    let a = upload(&b, &a_host);
    let bm = upload(&b, &b_host);
    let add = upload(&b, &pending_add);
    let mut c = upload(&b, &c0);

    let add_alpha = Complex32::new(0.5, -0.25);
    b.axpy(add_alpha, &add, &mut c);
    b.gemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        m,
        n,
        k,
        Complex32::new(0.75, 0.25),
        &a,
        3,
        &bm,
        3,
        Complex32::new(1.0, 0.0),
        &mut c,
        3,
    );
    let got = download(&b, &c);

    let mut want = c0.clone();
    for (w, x) in want.iter_mut().zip(pending_add.iter()) {
        *w += add_alpha * *x;
    }
    cpu_blas3::gemm_c32(
        Transpose::NoTrans,
        Transpose::NoTrans,
        m,
        n,
        k,
        Complex32::new(0.75, 0.25),
        &a_host,
        3,
        &b_host,
        3,
        Complex32::new(1.0, 0.0),
        &mut want,
        3,
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g.re - w.re).abs() < 1e-4 * (1.0 + w.re.abs()),
            "small c32 CPU GEMM re[{i}]: got {}, want {}",
            g.re,
            w.re
        );
        assert!(
            (g.im - w.im).abs() < 1e-4 * (1.0 + w.im.abs()),
            "small c32 CPU GEMM im[{i}]: got {}, want {}",
            g.im,
            w.im
        );
    }
}

/// MPS gemm parity vs CPU SGEMM on a non-trivial shape with
/// non-tight lda/ldb/ldc and alpha != 1, beta != 0.
#[test]
fn gemm_f32_mps_matches_cpu() {
    let Some(b) = try_metal_or_skip("gemm_f32_mps_matches_cpu") else {
        return;
    };
    let m = 257;
    let n = 257;
    let k = 257;
    assert!(MetalBackend::f32_gemm_uses_mps(m, n, k));
    // Row-major with non-tight strides.
    let lda = k + 5;
    let ldb = n + 7;
    let ldc = n + 3;
    let mut a_host = vec![0.0f32; m * lda];
    let mut b_host = vec![0.0f32; k * ldb];
    for i in 0..m {
        for j in 0..k {
            a_host[i * lda + j] = ((i * 31 + j * 17 + 1) as f32 * 0.001).sin();
        }
    }
    for i in 0..k {
        for j in 0..n {
            b_host[i * ldb + j] = ((i * 13 + j * 23 + 1) as f32 * 0.001).cos();
        }
    }
    let c0: Vec<f32> = (0..m * ldc).map(|i| (i as f32) * 0.01).collect();
    let alpha: f32 = 0.7;
    let beta: f32 = -0.4;

    let a = upload(&b, &a_host);
    let bm = upload(&b, &b_host);
    let mut c = upload(&b, &c0);
    b.gemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        m,
        n,
        k,
        alpha,
        &a,
        lda,
        &bm,
        ldb,
        beta,
        &mut c,
        ldc,
    );
    let got = download(&b, &c);

    let mut want = c0.clone();
    cpu_blas3::gemm_f32(
        Transpose::NoTrans,
        Transpose::NoTrans,
        m,
        n,
        k,
        alpha,
        &a_host,
        lda,
        &b_host,
        ldb,
        beta,
        &mut want,
        ldc,
    );
    for i in 0..m {
        for j in 0..n {
            let g = got[i * ldc + j];
            let w = want[i * ldc + j];
            assert!(
                (g - w).abs() < 1e-3 * (1.0 + w.abs()),
                "mps gemm parity ({i},{j}): got {g}, want {w}"
            );
        }
    }
}

#[test]
fn gemm_c32_mps_prototype_matches_cpu() {
    let Some(b) = try_metal_or_skip("gemm_c32_mps_prototype_matches_cpu") else {
        return;
    };
    let m = 257;
    let n = 257;
    let k = 257;
    assert!(MetalBackend::c32_gemm_uses_mps(m, n, k));
    let lda = k + 2;
    let ldb = n + 3;
    let ldc = n + 1;
    let mut a_host = vec![Complex32::new(0.0, 0.0); m * lda];
    let mut b_host = vec![Complex32::new(0.0, 0.0); k * ldb];
    for i in 0..m {
        for j in 0..k {
            let p = ((i * 31 + j * 17 + 1) as f32 * 0.01).sin();
            a_host[i * lda + j] = Complex32::new(p, 0.5 * p.cos());
        }
    }
    for i in 0..k {
        for j in 0..n {
            let p = ((i * 13 + j * 23 + 1) as f32 * 0.01).cos();
            b_host[i * ldb + j] = Complex32::new(0.25 * p.sin(), p);
        }
    }
    let c0: Vec<Complex32> = (0..m * ldc)
        .map(|i| Complex32::new((i as f32 * 0.01).sin(), (i as f32 * 0.02).cos()))
        .collect();
    let alpha = Complex32::new(0.7, -0.2);
    let beta = Complex32::new(-0.3, 0.1);

    let a = upload(&b, &a_host);
    let bm = upload(&b, &b_host);
    let mut c = upload(&b, &c0);
    b.gemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        m,
        n,
        k,
        alpha,
        &a,
        lda,
        &bm,
        ldb,
        beta,
        &mut c,
        ldc,
    );
    let got = download(&b, &c);

    let mut want = c0.clone();
    cpu_blas3::gemm_c32(
        Transpose::NoTrans,
        Transpose::NoTrans,
        m,
        n,
        k,
        alpha,
        &a_host,
        lda,
        &b_host,
        ldb,
        beta,
        &mut want,
        ldc,
    );
    for i in 0..m {
        for j in 0..n {
            let idx = i * ldc + j;
            assert_relative_eq!(
                got[idx].re,
                want[idx].re,
                max_relative = 1e-4,
                epsilon = 1e-4
            );
            assert_relative_eq!(
                got[idx].im,
                want[idx].im,
                max_relative = 1e-4,
                epsilon = 1e-4
            );
        }
    }
}

#[test]
fn gemm_c32_mps_prototype_uses_uninitialized_scratch() {
    let source = include_str!("mod.rs");
    let body = source
        .split("fn mps_gemm_c32_prototype")
        .nth(1)
        .expect("mps_gemm_c32_prototype implementation should exist")
        .split("fn bem_regular_pair_c32_batch")
        .next()
        .expect("Complex32 GEMM prototype should precede BEM kernel");
    assert!(
        body.contains("raw_alloc_uninitialized::<f32>"),
        "Metal Complex32 GEMM scratch should avoid zero-fill allocation"
    );
    assert!(
        !body.contains("self.raw_alloc::<f32>"),
        "Metal Complex32 GEMM scratch should not use zero-filled raw_alloc"
    );
    assert!(
        !body.contains("expect(\"metal c32 gemm"),
        "Metal Complex32 GEMM scratch failures must not panic"
    );
    assert!(
        body.contains("map_err(BackendError::from)"),
        "Metal Complex32 GEMM scratch failures must remain typed"
    );
    assert!(
        body.contains("self.flush_pending_checked()?"),
        "Metal Complex32 GEMM must surface completion failures"
    );
}

#[test]
fn gemm_c32_mps_prototype_conjtrans_matches_cpu() {
    let Some(b) = try_metal_or_skip("gemm_c32_mps_prototype_conjtrans_matches_cpu") else {
        return;
    };
    let m = 257;
    let n = 257;
    let k = 257;
    assert!(MetalBackend::c32_gemm_uses_mps(m, n, k));
    let lda = m + 2;
    let ldb = k + 3;
    let ldc = n + 1;
    let mut a_host = vec![Complex32::new(0.0, 0.0); k * lda];
    let mut b_host = vec![Complex32::new(0.0, 0.0); n * ldb];
    for row in 0..k {
        for col in 0..m {
            let p = ((row * 11 + col * 17 + 1) as f32 * 0.02).sin();
            a_host[row * lda + col] = Complex32::new(p, 0.3 * p.cos());
        }
    }
    for row in 0..n {
        for col in 0..k {
            let p = ((row * 19 + col * 13 + 2) as f32 * 0.015).cos();
            b_host[row * ldb + col] = Complex32::new(0.4 * p.sin(), -0.2 * p);
        }
    }
    let c0: Vec<Complex32> = (0..m * ldc)
        .map(|i| Complex32::new((i as f32 * 0.03).sin(), (i as f32 * 0.04).cos()))
        .collect();
    let alpha = Complex32::new(-0.2, 0.6);
    let beta = Complex32::new(0.25, -0.35);

    let a = upload(&b, &a_host);
    let bm = upload(&b, &b_host);
    let mut c = upload(&b, &c0);
    b.gemm(
        Transpose::ConjTrans,
        Transpose::ConjTrans,
        m,
        n,
        k,
        alpha,
        &a,
        lda,
        &bm,
        ldb,
        beta,
        &mut c,
        ldc,
    );
    let got = download(&b, &c);

    let mut want = c0.clone();
    cpu_blas3::gemm_c32(
        Transpose::ConjTrans,
        Transpose::ConjTrans,
        m,
        n,
        k,
        alpha,
        &a_host,
        lda,
        &b_host,
        ldb,
        beta,
        &mut want,
        ldc,
    );
    for i in 0..m {
        for j in 0..n {
            let idx = i * ldc + j;
            assert_relative_eq!(
                got[idx].re,
                want[idx].re,
                max_relative = 1e-4,
                epsilon = 1e-4
            );
            assert_relative_eq!(
                got[idx].im,
                want[idx].im,
                max_relative = 1e-4,
                epsilon = 1e-4
            );
        }
    }
}

/// TransA parity: stored A is k×m row-major, applied as A^T.
#[test]
fn gemm_f32_mps_trans_a_matches_cpu() {
    let Some(b) = try_metal_or_skip("gemm_f32_mps_trans_a_matches_cpu") else {
        return;
    };
    let m = 257;
    let n = 257;
    let k = 257;
    assert!(MetalBackend::f32_gemm_uses_mps(m, n, k));
    let lda = m + 3;
    let ldb = n + 5;
    let ldc = n + 7;
    // Stored A as k×m.
    let mut a_host = vec![0.0f32; k * lda];
    for p in 0..k {
        for i in 0..m {
            a_host[p * lda + i] = ((p * 19 + i * 29 + 1) as f32 * 0.001).sin();
        }
    }
    let mut b_host = vec![0.0f32; k * ldb];
    for i in 0..k {
        for j in 0..n {
            b_host[i * ldb + j] = ((i * 23 + j * 11 + 1) as f32 * 0.001).cos();
        }
    }
    let c0: Vec<f32> = (0..m * ldc).map(|i| (i as f32) * 0.005).collect();
    let alpha: f32 = 0.6;
    let beta: f32 = 0.3;

    let a = upload(&b, &a_host);
    let bm = upload(&b, &b_host);
    let mut c = upload(&b, &c0);
    b.gemm(
        Transpose::Trans,
        Transpose::NoTrans,
        m,
        n,
        k,
        alpha,
        &a,
        lda,
        &bm,
        ldb,
        beta,
        &mut c,
        ldc,
    );
    let got = download(&b, &c);

    let mut want = c0.clone();
    cpu_blas3::gemm_f32(
        Transpose::Trans,
        Transpose::NoTrans,
        m,
        n,
        k,
        alpha,
        &a_host,
        lda,
        &b_host,
        ldb,
        beta,
        &mut want,
        ldc,
    );
    for i in 0..m {
        for j in 0..n {
            let g = got[i * ldc + j];
            let w = want[i * ldc + j];
            assert!(
                (g - w).abs() < 1e-3 * (1.0 + w.abs()),
                "mps gemm trans_a ({i},{j}): got {g}, want {w}"
            );
        }
    }
}

#[test]
fn spmv_csr_adaptive_matches_cpu_variable_nnz() {
    use crate::backend::Backend as _;
    let Some(b) = try_metal_or_skip("spmv_csr_adaptive_matches_cpu_variable_nnz") else {
        return;
    };
    // Deliberately variable nnz/row: row 0 has 1 entry (short), row 1
    // has 16 entries (simd), row 2 has 80 entries (cta), row 3 has 3
    // entries (short), etc., scaling to ~200 rows so we exercise all
    // three buckets.
    let n = 256_usize;
    let mut row_ptr_h: Vec<i32> = Vec::with_capacity(n + 1);
    let mut col_idx_h: Vec<i32> = Vec::new();
    let mut values_h: Vec<f32> = Vec::new();
    row_ptr_h.push(0);
    for i in 0..n {
        // Cycle through { short, simd, cta, short, simd } patterns.
        let nnz = match i % 5 {
            0 => 1,
            1 => 16,
            2 => 80,
            3 => 3,
            _ => 24,
        };
        for k in 0..nnz {
            let col = (i + k * 3) % n;
            col_idx_h.push(col as i32);
            values_h.push(((i * 31 + k * 13 + 1) as f32 * 0.01).sin());
        }
        row_ptr_h.push(col_idx_h.len() as i32);
    }
    let x_h: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos()).collect();

    let row_ptr = upload(&b, &row_ptr_h);
    let col_idx = upload(&b, &col_idx_h);
    let values = upload(&b, &values_h);
    let x = upload(&b, &x_h);
    let mut y = b.raw_alloc::<f32>(n).unwrap();

    let prebaked = b.prebake_csr::<f32>(n, n, &row_ptr, &col_idx, &values);
    b.spmv_prebaked::<f32>(&prebaked, &x, &mut y);
    let got = download(&b, &y);

    let mut want = vec![0.0_f32; n];
    for i in 0..n {
        let lo = row_ptr_h[i] as usize;
        let hi = row_ptr_h[i + 1] as usize;
        let mut acc = 0.0_f32;
        for k in lo..hi {
            acc += values_h[k] * x_h[col_idx_h[k] as usize];
        }
        want[i] = acc;
    }
    for (g, w) in got.iter().zip(want.iter()) {
        assert_relative_eq!(g, w, max_relative = 1e-4, epsilon = 1e-5);
    }
}

#[test]
fn spmm_csr_native_matches_cpu_for_strided_rhs() {
    use crate::backend::Backend as _;
    let Some(b) = try_metal_or_skip("spmm_csr_native_matches_cpu_for_strided_rhs") else {
        return;
    };
    let row_ptr_host = [0_i32, 2, 3];
    let col_idx_host = [0_i32, 2, 1];
    let values_host = [2.0_f32, -1.0, 3.0];
    let x_host = [1.0_f32, 2.0, 3.0, 99.0, 98.0, 4.0, 5.0, 6.0, 97.0, 96.0];
    let row_ptr = upload(&b, &row_ptr_host);
    let col_idx = upload(&b, &col_idx_host);
    let values = upload(&b, &values_host);
    let x = upload(&b, &x_host);
    let mut y = upload(&b, &[77.0_f32; 8]);
    let prebaked = b.prebake_csr::<f32>(2, 3, &row_ptr, &col_idx, &values);
    b.spmm_prebaked::<f32>(&prebaked, 2, &x, 5, &mut y, 4);
    let got = download(&b, &y);
    assert_relative_eq!(got[0], -1.0, epsilon = 1.0e-5);
    assert_relative_eq!(got[1], 6.0, epsilon = 1.0e-5);
    assert_relative_eq!(got[4], 2.0, epsilon = 1.0e-5);
    assert_relative_eq!(got[5], 15.0, epsilon = 1.0e-5);
}

#[test]
fn f64_ops_still_host_staged() {
    // f64 path remains correct (host-staged). We exercise axpy
    // over f64 and verify the result.
    let Some(b) = try_metal_or_skip("f64_ops_still_host_staged") else {
        return;
    };
    let x_host: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let y_host: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0];
    let x = upload(&b, &x_host);
    let mut y = upload(&b, &y_host);
    b.axpy(0.5_f64, &x, &mut y);
    let got = download(&b, &y);
    let expected: Vec<f64> = x_host
        .iter()
        .zip(y_host.iter())
        .map(|(xi, yi)| 0.5 * xi + yi)
        .collect();
    for (g, w) in got.iter().zip(expected.iter()) {
        assert_relative_eq!(g, w, max_relative = 1e-12);
    }
}
