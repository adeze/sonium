// Fixture uses the checked `try_*` constructors on a 1x1 matrix, matching
// the production contract.

use sonium_compute::IdentityBackendPreconditioner;
use sonium_compute::backend::Backend;
use sonium_compute::cpu::CpuBackend;
use sonium_compute::gmres::BackendCsr;
use sonium_compute::gmres_ir::{BackendCsrF32, GmresIrConfig, gmres_ir_backend_preconditioned};

#[test]
fn gmres_ir_rejects_dimension_mismatch_without_panicking() {
    let backend = CpuBackend::new();
    let row_ptrs = [0_i32, 1];
    let col_indices = [0_i32];
    let values = [1.0_f64];
    let values_f32 = [1.0_f32];
    let outer = BackendCsr::try_from_host(&backend, 1, 1, &row_ptrs, &col_indices, &values)
        .expect("fixture CSR upload");
    let inner = BackendCsrF32::try_from_host(&backend, 1, 1, &row_ptrs, &col_indices, &values_f32)
        .expect("fixture CSR upload");
    let b = backend.raw_alloc::<f64>(2).expect("RHS allocation");
    let mut x = backend.raw_alloc::<f64>(1).expect("solution allocation");
    let config = GmresIrConfig {
        max_outer_iterations: 2,
        max_inner_iterations: 2,
        ..Default::default()
    };

    let error = gmres_ir_backend_preconditioned::<CpuBackend, _, _, _, f64>(
        &backend,
        &outer,
        &inner,
        &IdentityBackendPreconditioner,
        &b,
        None,
        &mut x,
        &config,
    )
    .expect_err("dimension mismatch must be a typed error");
    assert!(matches!(
        error,
        sonium_compute::GmresIrError::InvalidInput { .. }
    ));

    let b_valid = backend.raw_alloc::<f64>(1).expect("RHS allocation");
    let mut x_valid = backend.raw_alloc::<f64>(1).expect("solution allocation");
    let invalid_configs = [
        GmresIrConfig {
            max_outer_iterations: 0,
            ..Default::default()
        },
        GmresIrConfig {
            max_inner_iterations: 0,
            ..Default::default()
        },
        GmresIrConfig {
            outer_tolerance: f64::NAN,
            ..Default::default()
        },
        GmresIrConfig {
            inner_tolerance: f32::NAN,
            ..Default::default()
        },
    ];
    for config in invalid_configs {
        let error = gmres_ir_backend_preconditioned::<CpuBackend, _, _, _, f64>(
            &backend,
            &outer,
            &inner,
            &IdentityBackendPreconditioner,
            &b_valid,
            None,
            &mut x_valid,
            &config,
        )
        .expect_err("invalid GMRES-IR configuration must be typed");
        assert!(matches!(
            error,
            sonium_compute::GmresIrError::InvalidInput { .. }
        ));
    }
}
