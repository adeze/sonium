#![cfg(feature = "cuda")]

use sonium_sim_structural_mechanics::{
    BracketStudy, CudaStaticBracketConfig, StructuralBackend, StructuralCudaError,
    StructuralExecutionMode, StructuralPrecisionPolicy, run_static_bracket_cuda,
};

#[test]
fn cuda_static_configuration_and_cancellation_fail_before_device_work() {
    let study = BracketStudy::introductory_bracket();
    let invalid = CudaStaticBracketConfig {
        max_iterations: 0,
        ..CudaStaticBracketConfig::default()
    };
    assert!(matches!(
        run_static_bracket_cuda(&study, &invalid),
        Err(StructuralCudaError::InvalidConfiguration {
            field: "max_iterations",
            ..
        })
    ));

    let cancelled = CudaStaticBracketConfig::default();
    cancelled.cancellation.cancel();
    assert!(matches!(
        run_static_bracket_cuda(&study, &cancelled),
        Err(StructuralCudaError::Cancelled {
            stage: "before CPU reference"
        })
    ));
}

#[test]
fn real_cuda_static_forward_adjoint_and_product_evidence() {
    let mut study = BracketStudy::introductory_bracket();
    // The same production family and gates as the default study, with a
    // bounded mesh suitable for routine hardware evidence.
    study.geometry.nx = 12;
    study.geometry.ny = 3;
    let config = CudaStaticBracketConfig::default();
    let result = match run_static_bracket_cuda(&study, &config) {
        Ok(result) => result,
        Err(StructuralCudaError::BackendUnavailable { message, .. })
            if std::env::var("SONIUM_REQUIRE_CUDA_STRUCTURAL").as_deref() != Ok("1") =>
        {
            eprintln!("skipping CUDA structural hardware evidence: {message}");
            return;
        }
        Err(error) => panic!("CUDA structural evidence failed: {error}"),
    };

    assert_eq!(result.execution.backend, StructuralBackend::Cuda);
    assert_eq!(
        result.execution.execution_mode,
        StructuralExecutionMode::Accelerator
    );
    assert_eq!(
        result.execution.precision_policy,
        StructuralPrecisionPolicy::F64
    );
    assert!(result.execution.capability_verified);
    assert!(result.result.converged);
    assert!(result.evidence.physical_gates_passed);
    assert!(!result.evidence.fallback_used);
    assert_eq!(result.evidence.backend, "cuda");
    assert_eq!(result.evidence.precision, "f64");
    assert_eq!(result.evidence.polynomial_order, "P1");
    assert!(result.evidence.forward_true_relative_residual <= config.relative_residual_tolerance);
    assert!(
        result.evidence.displacement_cpu_relative_error <= config.displacement_relative_tolerance
    );
    assert!(result.evidence.stress_cpu_relative_error <= config.stress_relative_tolerance);
    assert!(result.evidence.energy_cpu_relative_error <= config.energy_relative_tolerance);
    assert!(result.evidence.derivative.transpose_adjoint_executed);
    assert!(result.evidence.derivative.accepted);
    assert!(
        result.evidence.derivative.gradient_relative_error <= config.gradient_relative_tolerance
    );
    assert!(result.evidence.h2d_operations > 0);
    assert!(result.evidence.d2h_operations > 0);
    assert!(result.evidence.allocation_high_water_bytes > 0);
    assert_eq!(result.evidence.live_allocation_bytes_after_cleanup, 0);
    if let Ok(path) = std::env::var("SONIUM_CUDA_STRUCTURAL_EVIDENCE_OUTPUT") {
        let json = serde_json::to_vec_pretty(&result.evidence)
            .expect("serialize CUDA structural evidence artifact");
        std::fs::write(path, json).expect("write CUDA structural evidence artifact");
    }
    println!(
        "STRUCTURAL_CUDA_EVIDENCE {}",
        serde_json::to_string(&result.evidence).expect("serialize CUDA evidence")
    );
}
