//! Native CUDA acceptance coverage for batched local contractions.
//!
//! Skip-capable on hosts without an NVIDIA driver, mirroring
//! `tests/syrk_cuda.rs`; `SONIUM_REQUIRE_CUDA=1` turns the skip into a
//! failure.

#![cfg(feature = "cuda")]

use sonium_compute::tensor_contract::batched_local_contract;
use sonium_compute::{Backend, CudaBackend};

fn cuda_or_skip() -> Option<CudaBackend> {
    match CudaBackend::new_default() {
        Ok(backend) => Some(backend),
        Err(error) if std::env::var("SONIUM_REQUIRE_CUDA").is_ok() => {
            panic!("CUDA required by SONIUM_REQUIRE_CUDA but unavailable: {error}")
        }
        Err(error) => {
            eprintln!("CUDA unavailable, skipping batched acceptance: {error}");
            None
        }
    }
}

fn ramp32(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
        .collect()
}

fn ramp64(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| ((i * 37 % 101) as f64 - 50.0) / 50.0)
        .collect()
}

/// f32 cuBLAS fast path vs the host reference (tolerance: different
/// accumulation order), plus fast-vs-native cross-check under the
/// `SONIUM_CUDA_BATCH_BLAS=0` kill-switch.
#[test]
fn native_cuda_batched_f32_matches_host() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let (batches, rows, inner, cols) = (4usize, 9usize, 7usize, 11usize);
    let lhs = ramp32(batches * rows * inner);
    let rhs = ramp32(batches * inner * cols);
    let mut want = vec![0.0f32; batches * rows * cols];
    batched_local_contract(&lhs, &rhs, &mut want, batches, rows, inner, cols)
        .expect("host reference");
    for gated in [false, true] {
        // SAFETY: only writer of the switch among tests.
        unsafe {
            if gated {
                std::env::set_var("SONIUM_CUDA_BATCH_BLAS", "0");
            } else {
                std::env::remove_var("SONIUM_CUDA_BATCH_BLAS");
            }
        }
        let mut dl = cuda.raw_alloc::<f32>(lhs.len()).expect("lhs alloc");
        let mut dr = cuda.raw_alloc::<f32>(rhs.len()).expect("rhs alloc");
        let mut dout = cuda.raw_alloc::<f32>(want.len()).expect("out alloc");
        cuda.try_copy_h2d(&lhs, &mut dl).expect("lhs upload");
        cuda.try_copy_h2d(&rhs, &mut dr).expect("rhs upload");
        cuda.try_batched_local_contract(&dl, &dr, &mut dout, batches, rows, inner, cols)
            .expect("device contract");
        let mut got = vec![0.0f32; want.len()];
        cuda.try_copy_d2h(&dout, &mut got).expect("download");
        let err = got
            .iter()
            .zip(want.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            err < 1e-4,
            "f32 batched (native-gated={gated}) maxdiff {err:e} over host"
        );
    }
    unsafe {
        std::env::remove_var("SONIUM_CUDA_BATCH_BLAS");
    }
}

/// f64 native kernel vs the host reference (stays native: beats cuBLAS
/// ~8% at every probed size).
#[test]
fn native_cuda_batched_f64_matches_host() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let (batches, rows, inner, cols) = (4usize, 9usize, 7usize, 11usize);
    let lhs = ramp64(batches * rows * inner);
    let rhs = ramp64(batches * inner * cols);
    let mut want = vec![0.0f64; batches * rows * cols];
    batched_local_contract(&lhs, &rhs, &mut want, batches, rows, inner, cols)
        .expect("host reference");
    let mut dl = cuda.raw_alloc::<f64>(lhs.len()).expect("lhs alloc");
    let mut dr = cuda.raw_alloc::<f64>(rhs.len()).expect("rhs alloc");
    let mut dout = cuda.raw_alloc::<f64>(want.len()).expect("out alloc");
    cuda.try_copy_h2d(&lhs, &mut dl).expect("lhs upload");
    cuda.try_copy_h2d(&rhs, &mut dr).expect("rhs upload");
    cuda.try_batched_local_contract(&dl, &dr, &mut dout, batches, rows, inner, cols)
        .expect("device contract");
    let mut got = vec![0.0f64; want.len()];
    cuda.try_copy_d2h(&dout, &mut got).expect("download");
    let err = got
        .iter()
        .zip(want.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    assert!(err < 1e-10, "f64 batched maxdiff {err:e} over host");
}
