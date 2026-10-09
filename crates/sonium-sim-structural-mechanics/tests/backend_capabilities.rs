use sonium_sim_structural_mechanics::{
    STRUCTURAL_CAPABILITY_SCHEMA_VERSION, StructuralAnalysisFamily, StructuralBackend,
    StructuralExecutionMode, StructuralPrecisionPolicy, structural_backend_capabilities,
};

#[test]
fn public_backend_catalogue_promotes_cpu_reference_only() {
    let catalogue = structural_backend_capabilities();
    assert_eq!(catalogue.len(), 4);
    assert!(
        catalogue
            .iter()
            .all(|capability| capability.schema_version == STRUCTURAL_CAPABILITY_SCHEMA_VERSION)
    );

    let cpu = catalogue
        .iter()
        .find(|capability| capability.backend == StructuralBackend::Cpu)
        .expect("CPU capability should be declared");
    assert!(cpu.integrated);
    assert_eq!(cpu.execution_mode, StructuralExecutionMode::CpuReference);
    assert!(cpu.supports(
        StructuralAnalysisFamily::TopologyOptimization,
        StructuralPrecisionPolicy::F64,
    ));
    assert!(cpu.supports(
        StructuralAnalysisFamily::Modal,
        StructuralPrecisionPolicy::F64,
    ));
    assert!(!cpu.supports(
        StructuralAnalysisFamily::Modal,
        StructuralPrecisionPolicy::F32,
    ));
}

#[test]
fn public_backend_catalogue_rejects_unwired_accelerator_claims() {
    let catalogue = structural_backend_capabilities();
    for backend in [StructuralBackend::Metal, StructuralBackend::Wasm] {
        let capability = catalogue
            .iter()
            .find(|capability| capability.backend == backend)
            .expect("declared backend should have a capability row");
        assert!(!capability.integrated);
        assert_eq!(
            capability.execution_mode,
            StructuralExecutionMode::Unsupported
        );
        assert!(capability.analysis_families.is_empty());
        assert!(capability.precision_policies.is_empty());
        assert!(!capability.limitations.is_empty());
        assert!(!capability.supports(
            StructuralAnalysisFamily::Static,
            StructuralPrecisionPolicy::F64,
        ));
    }

    let cuda = catalogue
        .iter()
        .find(|capability| capability.backend == StructuralBackend::Cuda)
        .expect("CUDA capability should be declared");
    #[cfg(feature = "cuda")]
    {
        assert!(cuda.integrated);
        assert_eq!(cuda.execution_mode, StructuralExecutionMode::Accelerator);
        assert_eq!(
            cuda.analysis_families,
            vec![StructuralAnalysisFamily::Static]
        );
        assert_eq!(
            cuda.precision_policies,
            vec![StructuralPrecisionPolicy::F64]
        );
        assert!(cuda.supports(
            StructuralAnalysisFamily::Static,
            StructuralPrecisionPolicy::F64,
        ));
        assert!(!cuda.supports(
            StructuralAnalysisFamily::Modal,
            StructuralPrecisionPolicy::F64,
        ));
        assert!(!cuda.supports(
            StructuralAnalysisFamily::Static,
            StructuralPrecisionPolicy::F32,
        ));
    }
    #[cfg(not(feature = "cuda"))]
    {
        assert!(!cuda.integrated);
        assert_eq!(cuda.execution_mode, StructuralExecutionMode::Unsupported);
        assert!(cuda.analysis_families.is_empty());
        assert!(cuda.precision_policies.is_empty());
    }
}
