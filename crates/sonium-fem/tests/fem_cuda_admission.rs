use sonium_fem::{
    FemAction, FemActualBackend, FemAdmissionError, FemAssemblyMode, FemBackend, FemBackendRequest,
    FemElement, FemOrder, FemPhysics, FemScalar, FemSpace, FemWorkload, admit_fem_cuda,
    admit_fem_workload,
};

fn workload() -> FemWorkload {
    FemWorkload::native_cuda(FemScalar::F64, FemAction::Forward)
}

fn assert_rejected(workload: FemWorkload) {
    let error = admit_fem_cuda(workload).expect_err("unsupported CUDA workload must be rejected");
    let FemAdmissionError::Unsupported {
        requested_backend,
        actual_backend,
        fallback,
        fallback_reason,
        ..
    } = error
    else {
        panic!("unsupported workload returned a different error");
    };
    assert_eq!(requested_backend, FemBackend::Cuda);
    assert_eq!(actual_backend, FemActualBackend::NotExecuted);
    assert_eq!(fallback, "none");
    assert!(fallback_reason.contains("rejected before assembly"));
}

#[test]
fn unsupported_element_rejects_before_assembly() {
    let unsupported = FemWorkload {
        element: FemElement::Tetrahedron,
        ..workload()
    };
    let error = admit_fem_cuda(unsupported).expect_err("tetrahedron CUDA must be rejected");
    let FemAdmissionError::Unsupported {
        requested_backend,
        actual_backend,
        fallback,
        fallback_reason,
        ..
    } = error
    else {
        panic!("unsupported element returned a different error");
    };
    assert_eq!(requested_backend, FemBackend::Cuda);
    assert_eq!(actual_backend, FemActualBackend::NotExecuted);
    assert_eq!(fallback, "none");
    assert!(fallback_reason.contains("before assembly"));
}

#[test]
fn unsupported_physics_and_order_reject_without_dispatch() {
    let other_physics = FemWorkload {
        physics: FemPhysics::Other,
        ..workload()
    };
    assert_rejected(other_physics);

    let higher_order = FemWorkload {
        order: FemOrder::P2OrHigher,
        ..workload()
    };
    assert_rejected(higher_order);

    for scalar in [FemScalar::F32, FemScalar::Complex32] {
        assert_rejected(FemWorkload {
            scalar,
            ..workload()
        });
    }
    assert_rejected(FemWorkload {
        scalar: FemScalar::Complex64,
        action: FemAction::Transpose,
        ..workload()
    });
    for element in [
        FemElement::Quad,
        FemElement::Tetrahedron,
        FemElement::Hexahedron,
    ] {
        assert_rejected(FemWorkload {
            element,
            ..workload()
        });
    }
    for space in [FemSpace::Hcurl, FemSpace::Hdiv, FemSpace::Dg] {
        assert_rejected(FemWorkload {
            space,
            ..workload()
        });
    }
    for mode in [FemAssemblyMode::PartialAssembly] {
        assert_rejected(FemWorkload { mode, ..workload() });
    }
}

#[test]
fn p2_helmholtz_is_admitted_as_an_explicit_bounded_cell() {
    let admission = admit_fem_cuda(FemWorkload::native_cuda_p2(
        FemScalar::Complex64,
        FemAction::Hermitian,
    ))
    .expect("P2/Tri3 Helmholtz CUDA cell should be admitted");
    assert_eq!(admission.provenance.actual_backend, FemActualBackend::Cuda);
    assert_eq!(admission.provenance.fallback, "none");
    assert_eq!(admission.workload.order, FemOrder::P2);
}

#[test]
fn fallback_provenance_records_requested_and_actual_backend() {
    let unsupported = FemWorkload {
        scalar: FemScalar::Complex32,
        ..workload()
    };
    let admission = admit_fem_workload(unsupported, FemBackendRequest::with_cpu_fallback())
        .expect("explicit CPU fallback should be admitted");
    assert_eq!(admission.provenance.requested_backend, FemBackend::Cuda);
    assert_eq!(admission.provenance.actual_backend, FemActualBackend::Cpu);
    assert_eq!(admission.provenance.fallback, "cpu");
    assert!(
        admission
            .provenance
            .fallback_reason
            .contains("CUDA workload unsupported")
    );
}
