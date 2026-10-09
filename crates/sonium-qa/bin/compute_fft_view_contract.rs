//! CPU-only QA for the shared native FFT view descriptor contract.
//!
//! The binary does not require CUDA or Metal. It exercises the exact
//! descriptor constructor consumed by both native adapters, including empty
//! batches, row-major strides, in-place aliasing, and malformed lengths.

use num_complex::{Complex32, Complex64};
use sonium_compute::{
    BackendError, BufferElement, ViewAliasing, ViewLayout, checked_fft_view_descriptor,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.fft-view-contract.v1";

fn check_scalar<T: BufferElement>(device: &'static str) -> Result<(usize, usize), String> {
    let mut valid = 0usize;
    for (buffer_len, fft_len) in [(0usize, 8usize), (32, 8), (64, 16)] {
        let descriptor = checked_fft_view_descriptor::<T>(buffer_len, fft_len, device)
            .map_err(|error| format!("valid FFT descriptor rejected: {error}"))?;
        if descriptor.rank != 2
            || descriptor.shape != vec![buffer_len / fft_len, fft_len]
            || descriptor.element_strides != vec![fft_len, 1]
            || descriptor.layout != ViewLayout::RowMajor
            || descriptor.aliasing != ViewAliasing::OverlapSafe
            || descriptor.device != device
        {
            return Err("FFT descriptor metadata does not match native binding contract".into());
        }
        valid += 1;
    }

    let mut rejected = 0usize;
    for (buffer_len, fft_len) in [(15usize, 8usize), (16, 0), (usize::MAX, 2)] {
        let error = checked_fft_view_descriptor::<T>(buffer_len, fft_len, device)
            .expect_err("malformed FFT descriptor must be rejected");
        if !matches!(error, BackendError::InvalidArgument { .. }) {
            return Err(format!(
                "malformed FFT descriptor returned unexpected error: {error}"
            ));
        }
        rejected += 1;
    }

    let mut descriptor = checked_fft_view_descriptor::<T>(32, 8, device)
        .map_err(|error| format!("descriptor setup failed: {error}"))?;
    descriptor.element_strides[1] = 2;
    if !matches!(
        descriptor.validate_for_buffer::<T>(32, device),
        Err(BackendError::InvalidArgument { .. })
    ) {
        return Err("mislabeled FFT strides were not rejected".into());
    }
    rejected += 1;

    Ok((valid, rejected))
}

fn run() -> Result<(usize, usize), String> {
    let mut valid = 0usize;
    let mut rejected = 0usize;
    for (scalar_valid, scalar_rejected) in [
        check_scalar::<f32>("cuda")?,
        check_scalar::<f64>("cuda")?,
        check_scalar::<Complex32>("metal")?,
        check_scalar::<Complex64>("metal")?,
    ] {
        valid += scalar_valid;
        rejected += scalar_rejected;
    }
    Ok((valid, rejected))
}

fn main() {
    match run() {
        Ok((valid, rejected)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("valid_descriptor_cases", valid as f64)
                .metric("rejection_cases", rejected as f64)
                .minimum("valid_descriptor_cases", 12.0)
                .minimum("rejection_cases", 16.0)
                .reference("shared checked rank-2 FFT view descriptor contract")
                .reference_checksum("analytic/deterministic-fixture:compute-fft-view-contract-v1")
                .dimension("four scalar families; empty and batched row-major FFT views")
                .seed("compute-fft-view-contract:descriptor-matrix-v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("FFT view QA result must serialize")
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
                    .expect("FFT view QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
