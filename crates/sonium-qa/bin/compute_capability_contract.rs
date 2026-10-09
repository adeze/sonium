//! CPU capability-table contract for the checked compute surface.
//!
//! This is deliberately small and deterministic. It does not launch a
//! simulator or require accelerator hardware; it verifies that the CPU table
//! advertises every checked scalar operation as native, FFT as unsupported,
//! and that a representative checked operation agrees with that metadata.

use num_complex::Complex32;
use sonium_compute::{
    Backend, BackendCapabilities, BackendError, CapabilityState, CpuBackend, DeviceScalar,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.capability-contract.v1";

#[cfg(test)]
mod tests {
    use super::run;

    #[test]
    fn cpu_fft_probe_requires_the_typed_unsupported_error() {
        let (_, unsupported) = run().expect("CPU capability contract should pass");
        assert!(
            unsupported,
            "unsupported FFT must be measured as a passed probe"
        );
    }
}

fn check_scalar<T: DeviceScalar>(
    backend: &CpuBackend,
    capabilities: &BackendCapabilities,
) -> Result<usize, String> {
    let entries = [
        ("blas1", capabilities.blas1),
        ("gemv", capabilities.gemv),
        ("gemm", capabilities.gemm),
        ("csr_spmv", capabilities.csr_spmv),
        ("csr_spmm", capabilities.csr_spmm),
        ("axpy_many", capabilities.axpy_many),
        ("multi_dot", capabilities.multi_dot),
        ("multi_dot_strided", capabilities.multi_dot_strided),
        ("basis_axpy_strided", capabilities.basis_axpy_strided),
        (
            "basis_project_out_strided",
            capabilities.basis_project_out_strided,
        ),
        ("gather4_weighted", capabilities.gather4_weighted),
        ("segmented_reduce", capabilities.segmented_reduce),
        (
            "batched_local_contract",
            capabilities.batched_local_contract,
        ),
    ];
    for (operation, scalar_capability) in entries {
        if scalar_capability.state_for::<T>() != CapabilityState::Native {
            return Err(format!(
                "CPU capability for {operation}/{} is not native: {:?}",
                std::any::type_name::<T>(),
                scalar_capability.state_for::<T>()
            ));
        }
    }
    for mode in [
        sonium_compute::ReductionMode::Fast,
        sonium_compute::ReductionMode::Reproducible,
        sonium_compute::ReductionMode::Deterministic,
    ] {
        if capabilities.reduction.for_mode(mode).state_for::<T>() != CapabilityState::Native {
            return Err(format!(
                "CPU reduction capability for {}/{} is not native",
                std::any::type_name::<T>(),
                mode as u8
            ));
        }
    }

    let mut buffer = backend
        .raw_alloc::<T>(4)
        .map_err(|error| format!("{} allocation failed: {error}", std::any::type_name::<T>()))?;
    backend
        .try_copy_h2d(&[T::one(); 4], &mut buffer)
        .map_err(|error| format!("{} setup failed: {error}", std::any::type_name::<T>()))?;
    backend.try_scale(T::one(), &mut buffer).map_err(|error| {
        format!(
            "{} checked scale failed: {error}",
            std::any::type_name::<T>()
        )
    })?;
    Ok(entries.len() + 3)
}

fn run() -> Result<(usize, bool), String> {
    let backend = CpuBackend::new();
    let capabilities = backend.capabilities();
    if capabilities.backend_name != "cpu" {
        return Err(format!(
            "unexpected backend name: {}",
            capabilities.backend_name
        ));
    }
    if capabilities.fft.state_for::<Complex32>() != CapabilityState::Unsupported {
        return Err("CPU FFT capability must be unsupported".into());
    }
    let mut checked_cells = 0;
    checked_cells += check_scalar::<f32>(&backend, &capabilities)?;
    checked_cells += check_scalar::<f64>(&backend, &capabilities)?;
    checked_cells += check_scalar::<num_complex::Complex32>(&backend, &capabilities)?;
    checked_cells += check_scalar::<num_complex::Complex64>(&backend, &capabilities)?;

    let mut fft_buffer = backend
        .raw_alloc::<Complex32>(4)
        .map_err(|error| format!("FFT probe allocation failed: {error}"))?;
    // `BackendCapabilities::backend_name` is intentionally a stable
    // capability label, while the concrete `CpuBackend` type name is an
    // implementation detail.  Only the operation/scalar contract belongs in
    // this probe; matching a literal backend type made a valid CPU
    // `Unsupported { operation: "fft" }` error report as a false negative.
    let unsupported = matches!(
        backend.try_fft_in_place(&mut fft_buffer, 4, false),
        Err(BackendError::Unsupported {
            operation: "fft",
            ..
        })
    );
    Ok((checked_cells, unsupported))
}

fn main() {
    match run() {
        Ok((checked_cells, unsupported_fft)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("checked_capability_cells", checked_cells as f64)
                .metric("unsupported_fft_probe", unsupported_fft as u8 as f64)
                .minimum("checked_capability_cells", 64.0)
                .minimum("unsupported_fft_probe", 1.0)
                .reference("CPU checked capability table and typed FFT rejection")
                .reference_checksum("analytic/deterministic-fixture:compute-capability-contract-v1")
                .dimension("four scalar families; 13 operation classes; three reduction modes")
                .seed("compute-capability-contract:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("capability QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("capability QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
