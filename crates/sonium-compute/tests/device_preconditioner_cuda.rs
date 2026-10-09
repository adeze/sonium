//! Native CUDA acceptance coverage for device-resident preconditioners.
//!
//! The test is intentionally skip-capable on hosts without an NVIDIA driver;
//! CI jobs with `SONIUM_REQUIRE_CUDA=1` turn that skip into a failure.

#![cfg(feature = "cuda")]
// CUDA device CSR fixtures use the checked `try_*` constructors. The
// saddle-point `from_csr_host_parts` fixtures below still use the panicking
// compatibility constructors; production code must use the `try_*` variants.
#![allow(deprecated)]

use sonium_compute::preconditioner::{
    PressureSchurBackendKind, SaddlePointJacobiSchurBackend, SaddlePointJacobiSchurBackendConfig,
};
use sonium_compute::{
    Backend, BackendCsr, BackendError, ChebyshevBackend, CpuBackend, CudaBackend,
    DeviceAdjointCapability, DevicePreconditioner, HostCsrRef, PointJacobiBackend,
    PolynomialJacobiBackend,
};

fn cuda_or_skip() -> Option<CudaBackend> {
    match CudaBackend::new_default() {
        Ok(backend) => Some(backend),
        Err(error) if std::env::var("SONIUM_REQUIRE_CUDA").is_ok() => {
            panic!("CUDA required by SONIUM_REQUIRE_CUDA but unavailable: {error}")
        }
        Err(error) => {
            eprintln!("CUDA unavailable, skipping native preconditioner acceptance: {error}");
            None
        }
    }
}

#[test]
fn native_cuda_preconditioners_match_cpu_without_action_staging() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let cpu = CpuBackend::new();
    let row_ptr = vec![0, 2, 5, 8, 11, 13];
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
    let values = vec![
        10.0, -1.0, -1.0, 20.0, -1.0, -1.0, 30.0, -1.0, -1.0, 40.0, -1.0, -1.0, 50.0,
    ];

    let a_cpu = BackendCsr::try_from_host(&cpu, 5, 5, &row_ptr, &col_idx, &values)
        .expect("fixture CSR upload");
    let a_cuda = BackendCsr::try_from_host(&cuda, 5, 5, &row_ptr, &col_idx, &values)
        .expect("fixture CSR upload");
    let point_cpu = PointJacobiBackend::from_csr_host(&cpu, 5, &row_ptr, &col_idx, &values);
    let point_cuda = PointJacobiBackend::from_csr_host(&cuda, 5, &row_ptr, &col_idx, &values);
    let polynomial_cpu =
        PolynomialJacobiBackend::from_csr_host(&cpu, &a_cpu, &row_ptr, &col_idx, &values, 3, 0.75);
    let polynomial_cuda = PolynomialJacobiBackend::from_csr_host(
        &cuda, &a_cuda, &row_ptr, &col_idx, &values, 3, 0.75,
    );
    let chebyshev_cpu =
        ChebyshevBackend::from_csr_host(&cpu, &a_cpu, &row_ptr, &col_idx, &values, 3, 30.0);
    let chebyshev_cuda =
        ChebyshevBackend::from_csr_host(&cuda, &a_cuda, &row_ptr, &col_idx, &values, 3, 30.0);

    let rhs_host = [1.0_f64, -2.0, 3.0, -4.0, 5.0];
    let block_rhs_host = [1.0_f64, -2.0, 3.0, -4.0, 5.0, -5.0, 4.0, -3.0, 2.0, -1.0];
    let mut rhs_cpu = cpu.raw_alloc::<f64>(5).unwrap();
    let mut rhs_cuda = cuda.raw_alloc::<f64>(5).unwrap();
    let mut block_rhs_cpu = cpu.raw_alloc::<f64>(10).unwrap();
    let mut block_rhs_cuda = cuda.raw_alloc::<f64>(10).unwrap();
    cpu.copy_h2d(&rhs_host, &mut rhs_cpu);
    cuda.copy_h2d(&rhs_host, &mut rhs_cuda);
    cpu.copy_h2d(&block_rhs_host, &mut block_rhs_cpu);
    cuda.copy_h2d(&block_rhs_host, &mut block_rhs_cuda);

    macro_rules! exercise {
        ($cpu_pre:expr, $cuda_pre:expr) => {{
            let mut cpu_out = cpu.raw_alloc::<f64>(5).unwrap();
            let mut cuda_out = cuda.raw_alloc::<f64>(5).unwrap();
            $cpu_pre.try_apply(&cpu, &rhs_cpu, &mut cpu_out).unwrap();
            let before = cuda.runtime_stats();
            $cuda_pre
                .try_apply(&cuda, &rhs_cuda, &mut cuda_out)
                .unwrap();
            cuda.try_synchronize().unwrap();
            let action_stats = cuda.runtime_stats_since(before);
            assert_eq!(action_stats.transfer.h2d_operations, 0);
            assert_eq!(action_stats.transfer.d2h_operations, 0);
            assert_eq!(action_stats.staging_allocation_operations, 0);
            assert_eq!(action_stats.live_staging_bytes, 0);

            let mut cpu_values = [0.0_f64; 5];
            let mut cuda_values = [0.0_f64; 5];
            cpu.copy_d2h(&cpu_out, &mut cpu_values);
            cuda.copy_d2h(&cuda_out, &mut cuda_values);
            for (cpu_value, cuda_value) in cpu_values.iter().zip(cuda_values) {
                assert!((cpu_value - cuda_value).abs() < 2.0e-10);
            }

            let mut cpu_block_out = cpu.raw_alloc::<f64>(10).unwrap();
            let mut cuda_block_out = cuda.raw_alloc::<f64>(10).unwrap();
            $cpu_pre
                .try_apply_colmajor(&cpu, 2, &block_rhs_cpu, &mut cpu_block_out)
                .unwrap();
            let before = cuda.runtime_stats();
            $cuda_pre
                .try_apply_colmajor(&cuda, 2, &block_rhs_cuda, &mut cuda_block_out)
                .unwrap();
            cuda.try_synchronize().unwrap();
            let action_stats = cuda.runtime_stats_since(before);
            assert_eq!(action_stats.transfer.h2d_operations, 0);
            assert_eq!(action_stats.transfer.d2h_operations, 0);
            assert_eq!(action_stats.staging_allocation_operations, 0);
            assert_eq!(action_stats.live_staging_bytes, 0);

            let mut cpu_block_values = [0.0_f64; 10];
            let mut cuda_block_values = [0.0_f64; 10];
            cpu.copy_d2h(&cpu_block_out, &mut cpu_block_values);
            cuda.copy_d2h(&cuda_block_out, &mut cuda_block_values);
            for (cpu_value, cuda_value) in cpu_block_values.iter().zip(cuda_block_values) {
                assert!((cpu_value - cuda_value).abs() < 2.0e-10);
            }
            assert_eq!($cuda_pre.diagnostics().apply_count, 2);
        }};
    }

    exercise!(&point_cpu, &point_cuda);
    exercise!(&polynomial_cpu, &polynomial_cuda);
    exercise!(&chebyshev_cpu, &chebyshev_cuda);

    assert_eq!(
        point_cuda.transpose_capability(),
        DeviceAdjointCapability::Exact
    );
    assert_eq!(
        polynomial_cuda.transpose_capability(),
        DeviceAdjointCapability::Unsupported
    );
    let mut transpose_out_cpu = cpu.raw_alloc::<f64>(5).unwrap();
    let mut transpose_out_cuda = cuda.raw_alloc::<f64>(5).unwrap();
    point_cpu
        .try_apply_transpose(&cpu, &rhs_cpu, &mut transpose_out_cpu)
        .unwrap();
    let before = cuda.runtime_stats();
    point_cuda
        .try_apply_transpose(&cuda, &rhs_cuda, &mut transpose_out_cuda)
        .unwrap();
    cuda.try_synchronize().unwrap();
    let action_stats = cuda.runtime_stats_since(before);
    assert_eq!(action_stats.transfer.h2d_operations, 0);
    assert_eq!(action_stats.transfer.d2h_operations, 0);
    assert_eq!(action_stats.staging_allocation_operations, 0);
    assert_eq!(action_stats.live_staging_bytes, 0);

    let mut transpose_cpu = [0.0_f64; 5];
    let mut transpose_cuda = [0.0_f64; 5];
    cpu.copy_d2h(&transpose_out_cpu, &mut transpose_cpu);
    cuda.copy_d2h(&transpose_out_cuda, &mut transpose_cuda);
    for (cpu_value, cuda_value) in transpose_cpu.iter().zip(transpose_cuda) {
        assert!((cpu_value - cuda_value).abs() < 2.0e-10);
    }

    macro_rules! unsupported_adjoint {
        ($pre:expr, $method:ident) => {{
            let before = cuda.runtime_stats();
            let error = $pre
                .$method(&cuda, &rhs_cuda, &mut transpose_out_cuda)
                .expect_err("unsupported CUDA adjoint must fail before dispatch");
            assert!(matches!(
                error,
                sonium_compute::BackendError::Unsupported { .. }
            ));
            let action_stats = cuda.runtime_stats_since(before);
            assert_eq!(action_stats.transfer.h2d_operations, 0);
            assert_eq!(action_stats.transfer.d2h_operations, 0);
            assert_eq!(action_stats.staging_allocation_operations, 0);
            assert_eq!(action_stats.live_staging_bytes, 0);
        }};
    }

    unsupported_adjoint!(&point_cuda, try_apply_hermitian);
    unsupported_adjoint!(&polynomial_cuda, try_apply_transpose);
    unsupported_adjoint!(&polynomial_cuda, try_apply_hermitian);
    unsupported_adjoint!(&chebyshev_cuda, try_apply_transpose);
    unsupported_adjoint!(&chebyshev_cuda, try_apply_hermitian);
    assert_eq!(polynomial_cuda.diagnostics().hermitian_failures, 1);

    let mut fault_output = cuda.raw_alloc::<f64>(5).unwrap();
    cuda.copy_h2d(&[91.0_f64; 5], &mut fault_output);
    cuda.inject_preconditioner_failure(sonium_compute::PreconditionerFailureInjectionPoint::Launch);
    let error = point_cuda
        .try_apply(&cuda, &rhs_cuda, &mut fault_output)
        .expect_err("native CUDA launch fault must be typed");
    assert!(matches!(error, sonium_compute::BackendError::Launch { .. }));
    cuda.recover_after_error()
        .expect("recoverable native CUDA launch fault");
    point_cuda
        .try_apply(&cuda, &rhs_cuda, &mut fault_output)
        .expect("native CUDA retry after launch fault");

    cuda.inject_preconditioner_failure(
        sonium_compute::PreconditionerFailureInjectionPoint::NonFiniteGeneration,
    );
    let error = point_cuda
        .try_apply(&cuda, &rhs_cuda, &mut fault_output)
        .expect_err("native CUDA non-finite fault must be typed");
    assert!(matches!(
        error,
        sonium_compute::BackendError::InvalidArgument { .. }
    ));

    cuda.inject_preconditioner_failure(
        sonium_compute::PreconditionerFailureInjectionPoint::Synchronization,
    );
    let error = cuda
        .try_synchronize()
        .expect_err("native CUDA synchronization fault must be typed");
    assert!(matches!(
        error,
        sonium_compute::BackendError::Synchronization { .. }
    ));
}

#[test]
fn native_cuda_complex32_preconditioners_match_cpu_without_action_staging() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let cpu = CpuBackend::new();
    let row_ptr = vec![0, 2, 5, 8, 11, 13];
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
    let values = vec![
        num_complex::Complex32::new(10.0, 0.5),
        num_complex::Complex32::new(-1.0, 0.1),
        num_complex::Complex32::new(-1.0, -0.1),
        num_complex::Complex32::new(20.0, 0.25),
        num_complex::Complex32::new(-1.0, 0.1),
        num_complex::Complex32::new(-1.0, -0.1),
        num_complex::Complex32::new(30.0, 0.5),
        num_complex::Complex32::new(-1.0, 0.1),
        num_complex::Complex32::new(-1.0, -0.1),
        num_complex::Complex32::new(40.0, 0.25),
        num_complex::Complex32::new(-1.0, 0.1),
        num_complex::Complex32::new(-1.0, -0.1),
        num_complex::Complex32::new(50.0, 0.5),
    ];
    let a_cpu = BackendCsr::try_from_host(&cpu, 5, 5, &row_ptr, &col_idx, &values)
        .expect("fixture CSR upload");
    let a_cuda = BackendCsr::try_from_host(&cuda, 5, 5, &row_ptr, &col_idx, &values)
        .expect("fixture CSR upload");
    let point_cpu = PointJacobiBackend::from_csr_host(&cpu, 5, &row_ptr, &col_idx, &values);
    let point_cuda = PointJacobiBackend::from_csr_host(&cuda, 5, &row_ptr, &col_idx, &values);
    let polynomial_cpu = PolynomialJacobiBackend::from_csr_host(
        &cpu,
        &a_cpu,
        &row_ptr,
        &col_idx,
        &values,
        2,
        num_complex::Complex32::new(0.75, 0.0),
    );
    let polynomial_cuda = PolynomialJacobiBackend::from_csr_host(
        &cuda,
        &a_cuda,
        &row_ptr,
        &col_idx,
        &values,
        2,
        num_complex::Complex32::new(0.75, 0.0),
    );
    let chebyshev_cpu =
        ChebyshevBackend::from_csr_host(&cpu, &a_cpu, &row_ptr, &col_idx, &values, 2, 30.0);
    let chebyshev_cuda =
        ChebyshevBackend::from_csr_host(&cuda, &a_cuda, &row_ptr, &col_idx, &values, 2, 30.0);

    let rhs_host = [
        num_complex::Complex32::new(1.0, -0.25),
        num_complex::Complex32::new(-2.0, 0.5),
        num_complex::Complex32::new(3.0, -0.75),
        num_complex::Complex32::new(-4.0, 1.0),
        num_complex::Complex32::new(5.0, -1.25),
    ];
    let block_rhs_host = [
        rhs_host[0],
        rhs_host[1],
        rhs_host[2],
        rhs_host[3],
        rhs_host[4],
        rhs_host[4].conj(),
        rhs_host[3].conj(),
        rhs_host[2].conj(),
        rhs_host[1].conj(),
        rhs_host[0].conj(),
    ];
    let mut rhs_cpu = cpu.raw_alloc::<num_complex::Complex32>(5).unwrap();
    let mut rhs_cuda = cuda.raw_alloc::<num_complex::Complex32>(5).unwrap();
    let mut block_rhs_cpu = cpu.raw_alloc::<num_complex::Complex32>(10).unwrap();
    let mut block_rhs_cuda = cuda.raw_alloc::<num_complex::Complex32>(10).unwrap();
    cpu.copy_h2d(&rhs_host, &mut rhs_cpu);
    cuda.copy_h2d(&rhs_host, &mut rhs_cuda);
    cpu.copy_h2d(&block_rhs_host, &mut block_rhs_cpu);
    cuda.copy_h2d(&block_rhs_host, &mut block_rhs_cuda);

    macro_rules! exercise_complex {
        ($cpu_pre:expr, $cuda_pre:expr) => {{
            let mut cpu_out = cpu.raw_alloc::<num_complex::Complex32>(5).unwrap();
            let mut cuda_out = cuda.raw_alloc::<num_complex::Complex32>(5).unwrap();
            $cpu_pre.try_apply(&cpu, &rhs_cpu, &mut cpu_out).unwrap();
            let before = cuda.runtime_stats();
            $cuda_pre
                .try_apply(&cuda, &rhs_cuda, &mut cuda_out)
                .unwrap();
            cuda.try_synchronize().unwrap();
            let action_stats = cuda.runtime_stats_since(before);
            assert_eq!(action_stats.transfer.h2d_operations, 0);
            assert_eq!(action_stats.transfer.d2h_operations, 0);
            assert_eq!(action_stats.staging_allocation_operations, 0);
            assert_eq!(action_stats.live_staging_bytes, 0);
            let mut cpu_values = [num_complex::Complex32::new(0.0, 0.0); 5];
            let mut cuda_values = [num_complex::Complex32::new(0.0, 0.0); 5];
            cpu.copy_d2h(&cpu_out, &mut cpu_values);
            cuda.copy_d2h(&cuda_out, &mut cuda_values);
            for (cpu_value, cuda_value) in cpu_values.iter().zip(cuda_values) {
                assert!((*cpu_value - cuda_value).norm() < 2.0e-4);
            }

            let mut cpu_block_out = cpu.raw_alloc::<num_complex::Complex32>(10).unwrap();
            let mut cuda_block_out = cuda.raw_alloc::<num_complex::Complex32>(10).unwrap();
            $cpu_pre
                .try_apply_colmajor(&cpu, 2, &block_rhs_cpu, &mut cpu_block_out)
                .unwrap();
            let before = cuda.runtime_stats();
            $cuda_pre
                .try_apply_colmajor(&cuda, 2, &block_rhs_cuda, &mut cuda_block_out)
                .unwrap();
            cuda.try_synchronize().unwrap();
            let action_stats = cuda.runtime_stats_since(before);
            assert_eq!(action_stats.transfer.h2d_operations, 0);
            assert_eq!(action_stats.transfer.d2h_operations, 0);
            let mut cpu_block_values = [num_complex::Complex32::new(0.0, 0.0); 10];
            let mut cuda_block_values = [num_complex::Complex32::new(0.0, 0.0); 10];
            cpu.copy_d2h(&cpu_block_out, &mut cpu_block_values);
            cuda.copy_d2h(&cuda_block_out, &mut cuda_block_values);
            for (cpu_value, cuda_value) in cpu_block_values.iter().zip(cuda_block_values) {
                assert!((*cpu_value - cuda_value).norm() < 2.0e-4);
            }
        }};
    }

    exercise_complex!(&point_cpu, &point_cuda);
    exercise_complex!(&polynomial_cpu, &polynomial_cuda);
    exercise_complex!(&chebyshev_cpu, &chebyshev_cuda);
}

#[test]
fn native_cuda_device_loss_injection_is_typed() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let point = PointJacobiBackend::from_diagonal(&cuda, &[2.0_f64, 3.0]);
    let mut rhs = cuda.raw_alloc::<f64>(2).unwrap();
    let mut output = cuda.raw_alloc::<f64>(2).unwrap();
    cuda.copy_h2d(&[1.0, 2.0], &mut rhs);
    cuda.inject_preconditioner_failure(
        sonium_compute::PreconditionerFailureInjectionPoint::DeviceLoss,
    );
    let error = point
        .try_apply(&cuda, &rhs, &mut output)
        .expect_err("native CUDA device-loss fault must be typed");
    assert!(matches!(
        error,
        sonium_compute::BackendError::DeviceLost { .. }
    ));
}

#[test]
fn native_cuda_saddle_preconditioner_matches_cpu_without_action_staging() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let cpu = CpuBackend::new();
    let row_ptr = vec![0, 3, 6, 7, 8];
    let col_idx = vec![0, 1, 2, 0, 1, 3, 2, 3];
    let values = vec![4.0_f64, 0.5, 1.0, 0.25, 3.0, -0.5, 0.1, 0.1];
    let schur_row_ptr = vec![0_usize, 2, 4];
    let schur_col_idx = vec![0_usize, 1, 0, 1];
    let schur_values = vec![2.0_f64, 0.5, 0.5, 2.0];
    let config = SaddlePointJacobiSchurBackendConfig {
        velocity_block_size: 2,
        velocity_degree: 2,
        velocity_damping: 0.75,
        pressure_degree: 2,
        pressure_damping: 0.75,
        pressure_kind: PressureSchurBackendKind::Polynomial,
        pressure_chebyshev_degree: 3,
        pressure_chebyshev_eigenvalue_ratio: 30.0,
    };
    let a_cpu = BackendCsr::try_from_host(&cpu, 4, 4, &row_ptr, &col_idx, &values)
        .expect("fixture CSR upload");
    let a_cuda = BackendCsr::try_from_host(&cuda, 4, 4, &row_ptr, &col_idx, &values)
        .expect("fixture CSR upload");
    let cpu_pre = SaddlePointJacobiSchurBackend::from_csr_host_parts(
        &cpu,
        &a_cpu,
        &row_ptr,
        &col_idx,
        &values,
        HostCsrRef::new(2, 2, &schur_row_ptr, &schur_col_idx, &schur_values),
        config,
    );
    let cuda_pre = SaddlePointJacobiSchurBackend::from_csr_host_parts(
        &cuda,
        &a_cuda,
        &row_ptr,
        &col_idx,
        &values,
        HostCsrRef::new(2, 2, &schur_row_ptr, &schur_col_idx, &schur_values),
        config,
    );
    assert_eq!(
        cuda_pre.transpose_capability(),
        DeviceAdjointCapability::Unsupported
    );
    assert_eq!(
        cuda_pre.hermitian_capability(),
        DeviceAdjointCapability::Unsupported
    );

    let rhs_host = [1.0_f64, -2.0, 0.5, 3.0];
    let block_rhs_host = [1.0_f64, -2.0, 0.5, 3.0, -1.5, 0.25, 2.0, -0.75];
    let mut rhs_cpu = cpu.raw_alloc::<f64>(4).unwrap();
    let mut rhs_cuda = cuda.raw_alloc::<f64>(4).unwrap();
    let mut block_rhs_cpu = cpu.raw_alloc::<f64>(8).unwrap();
    let mut block_rhs_cuda = cuda.raw_alloc::<f64>(8).unwrap();
    cpu.copy_h2d(&rhs_host, &mut rhs_cpu);
    cuda.copy_h2d(&rhs_host, &mut rhs_cuda);
    cpu.copy_h2d(&block_rhs_host, &mut block_rhs_cpu);
    cuda.copy_h2d(&block_rhs_host, &mut block_rhs_cuda);
    let mut cpu_out = cpu.raw_alloc::<f64>(4).unwrap();
    let mut cuda_out = cuda.raw_alloc::<f64>(4).unwrap();
    cpu_pre.try_apply(&cpu, &rhs_cpu, &mut cpu_out).unwrap();
    let before = cuda.runtime_stats();
    cuda_pre.try_apply(&cuda, &rhs_cuda, &mut cuda_out).unwrap();
    cuda.try_synchronize().unwrap();
    let action_stats = cuda.runtime_stats_since(before);
    assert_eq!(action_stats.transfer.h2d_operations, 0);
    assert_eq!(action_stats.transfer.d2h_operations, 0);
    assert_eq!(action_stats.staging_allocation_operations, 0);
    assert_eq!(action_stats.live_staging_bytes, 0);
    let mut cpu_values = [0.0_f64; 4];
    let mut cuda_values = [0.0_f64; 4];
    cpu.copy_d2h(&cpu_out, &mut cpu_values);
    cuda.copy_d2h(&cuda_out, &mut cuda_values);
    for (cpu_value, cuda_value) in cpu_values.iter().zip(cuda_values) {
        assert!((cpu_value - cuda_value).abs() < 2.0e-10);
    }

    let mut cpu_block_out = cpu.raw_alloc::<f64>(8).unwrap();
    let mut cuda_block_out = cuda.raw_alloc::<f64>(8).unwrap();
    cpu_pre
        .try_apply_colmajor(&cpu, 2, &block_rhs_cpu, &mut cpu_block_out)
        .unwrap();
    let before = cuda.runtime_stats();
    cuda_pre
        .try_apply_colmajor(&cuda, 2, &block_rhs_cuda, &mut cuda_block_out)
        .unwrap();
    cuda.try_synchronize().unwrap();
    let action_stats = cuda.runtime_stats_since(before);
    assert_eq!(action_stats.transfer.h2d_operations, 0);
    assert_eq!(action_stats.transfer.d2h_operations, 0);
    let mut cpu_block_values = [0.0_f64; 8];
    let mut cuda_block_values = [0.0_f64; 8];
    cpu.copy_d2h(&cpu_block_out, &mut cpu_block_values);
    cuda.copy_d2h(&cuda_block_out, &mut cuda_block_values);
    for (cpu_value, cuda_value) in cpu_block_values.iter().zip(cuda_block_values) {
        assert!((cpu_value - cuda_value).abs() < 2.0e-10);
    }
    assert_eq!(cuda_pre.diagnostics().apply_count, 2);
}

#[test]
fn native_cuda_preconditioner_setup_quota_failure_is_retryable() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };

    // The live allocation count is a real lower bound for this backend.  A
    // cap equal to it must reject the first setup allocation before entering
    // the driver, while retaining the existing allocations.  This exercises
    // the same setup transaction used by the SP2 failure producer rather than
    // a detached raw-buffer quota probe.
    let total_memory = cuda
        .total_device_memory_bytes()
        .expect("CUDA backend must report total memory");
    let live_before = cuda.live_allocation_bytes();
    cuda.set_allocation_cap_bytes(live_before);
    let error = PointJacobiBackend::try_from_diagonal(&cuda, &[2.0_f64; 9])
        .err()
        .expect("over-cap preconditioner setup must fail");
    assert!(
        matches!(error, BackendError::Allocation { .. }),
        "setup quota failure must remain a typed allocation error: {error}"
    );
    assert_eq!(
        cuda.live_allocation_bytes(),
        live_before,
        "failed setup must not publish or retain partial device state"
    );

    cuda.set_allocation_cap_bytes(total_memory);
    let retried = PointJacobiBackend::try_from_diagonal(&cuda, &[2.0_f64; 9])
        .expect("under-cap preconditioner setup retry");
    assert!(
        cuda.live_allocation_bytes() > live_before,
        "successful retry must own resident setup allocations"
    );
    drop(retried);
    assert_eq!(
        cuda.live_allocation_bytes(),
        live_before,
        "dropping the retried preconditioner must release all setup allocations"
    );
}
