//! Tiled CUDA multi-dot correctness: the 2D-grid (`tiles` blocks per dot)
//! kernels must agree with the CPU reference to reduction tolerance.
//!
//! Timing under a shared GPU is not meaningful, so these tests assert
//! numerics only; performance gating happens on a quiet box.

#![cfg(feature = "cuda")]

use num_complex::{Complex32, Complex64};
use sonium_compute::{Backend, CudaBackend, MemoryPool, MemoryPoolConfig, ReductionMode};

const REL_TOL: f64 = 1e-5;

fn skip_or_require_cuda<T>(label: &str, result: Result<T, String>) -> T {
    match result {
        Ok(value) => value,
        Err(err) => {
            if std::env::var("SONIUM_REQUIRE_CUDA")
                .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
                .unwrap_or(false)
            {
                panic!("CUDA required by SONIUM_REQUIRE_CUDA=1 but unavailable for {label}: {err}");
            }
            eprintln!("skipping {label}: CUDA unavailable ({err})");
            std::process::exit(0);
        }
    }
}

fn assert_vec_close_f64(label: &str, actual: &[f64], expected: &[f64]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch ({} vs {})",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let scale = e.abs().max(1.0);
        let diff = (a - e).abs();
        assert!(
            diff <= REL_TOL * scale,
            "{label}[{i}]: {a} vs {e} (diff {diff})"
        );
    }
}

fn assert_vec_close_f32(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch ({} vs {})",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let scale = e.abs().max(1.0);
        let diff = (a - e).abs();
        assert!(
            diff <= (REL_TOL as f32) * scale,
            "{label}[{i}]: {a} vs {e} (diff {diff})"
        );
    }
}

#[test]
fn cuda_tiled_multi_dot_f64_matches_cpu() {
    let backend = skip_or_require_cuda(
        "tiled multi_dot f64",
        CudaBackend::new_default().map_err(|e| format!("{e:?}")),
    );
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    // Non-multiple of block (256) and of tiles*block to exercise tails.
    let n = 65537usize;
    let k = 6usize;
    let tiles = 8usize;
    let y_host: Vec<f64> = (0..n).map(|i| (i as f64 * 0.07).sin()).collect();
    // Flat basis-major layout: basis[j * n + i].
    let basis_host: Vec<f64> = (0..k)
        .flat_map(|j| (0..n).map(move |i| ((i + 1) as f64).ln() + j as f64 * 0.125))
        .collect();

    let mut y = pool.try_alloc::<f64>(n).unwrap();
    pool.backend().copy_h2d(&y_host, y.raw_mut());
    let mut basis = pool.try_alloc::<f64>(k * n).unwrap();
    pool.backend().copy_h2d(&basis_host, basis.raw_mut());

    let mut got = vec![0.0f64; k];
    pool.backend()
        .try_multi_dot_tiled_with_mode_into(
            ReductionMode::Fast,
            basis.raw(),
            n,
            k,
            tiles,
            y.raw(),
            &mut got,
        )
        .unwrap();

    let want: Vec<f64> = (0..k)
        .map(|j| {
            basis_host[j * n..(j + 1) * n]
                .iter()
                .zip(y_host.iter())
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();
    assert_vec_close_f64("cuda/tiled_multi_dot_f64", &got, &want);
}

#[test]
fn cuda_tiled_multi_dot_c64_matches_cpu_dotc() {
    let backend = skip_or_require_cuda(
        "tiled multi_dot c64",
        CudaBackend::new_default().map_err(|e| format!("{e:?}")),
    );
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let n = 32769usize;
    let k = 4usize;
    let tiles = 4usize;
    let y_host: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 * 0.11).sin(), (i as f64 * 0.05).cos()))
        .collect();
    let basis_host: Vec<Complex64> = (0..k)
        .flat_map(|j| {
            (0..n).map(move |i| {
                Complex64::new(
                    ((i + 1) as f64).ln() + j as f64 * 0.2,
                    (i as f64 * 0.03 + j as f64).sin(),
                )
            })
        })
        .collect();

    let mut y = pool.try_alloc::<Complex64>(n).unwrap();
    pool.backend().copy_h2d(&y_host, y.raw_mut());
    let mut basis = pool.try_alloc::<Complex64>(k * n).unwrap();
    pool.backend().copy_h2d(&basis_host, basis.raw_mut());

    let mut got = vec![Complex64::new(0.0, 0.0); k];
    pool.backend()
        .try_multi_dot_tiled_with_mode_into(
            ReductionMode::Fast,
            basis.raw(),
            n,
            k,
            tiles,
            y.raw(),
            &mut got,
        )
        .unwrap();

    for (j, (g, _)) in got.iter().zip(0..k).enumerate() {
        let want: Complex64 = basis_host[j * n..(j + 1) * n]
            .iter()
            .zip(y_host.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        let scale = (want.re.abs() + want.im.abs()).max(1.0);
        assert!(
            (g.re - want.re).abs() <= REL_TOL * scale && (g.im - want.im).abs() <= REL_TOL * scale,
            "cuda/tiled_multi_dot_c64[{j}]: {g:?} vs {want:?}"
        );
    }
}

#[test]
fn cuda_tiled_multi_dot_f32_matches_cpu() {
    let backend = skip_or_require_cuda(
        "tiled multi_dot f32",
        CudaBackend::new_default().map_err(|e| format!("{e:?}")),
    );
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    // More tiles than data per block: most blocks only add +0.0.
    let n = 257usize;
    let k = 3usize;
    let tiles = 16usize;
    let y_host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin()).collect();
    let basis_host: Vec<f32> = (0..k)
        .flat_map(|j| (0..n).map(move |i| ((i + 1) as f32).ln() + j as f32 * 0.125))
        .collect();

    let mut y = pool.try_alloc::<f32>(n).unwrap();
    pool.backend().copy_h2d(&y_host, y.raw_mut());
    let mut basis = pool.try_alloc::<f32>(k * n).unwrap();
    pool.backend().copy_h2d(&basis_host, basis.raw_mut());

    let mut got = vec![0.0f32; k];
    pool.backend()
        .try_multi_dot_tiled_with_mode_into(
            ReductionMode::Fast,
            basis.raw(),
            n,
            k,
            tiles,
            y.raw(),
            &mut got,
        )
        .unwrap();

    let want: Vec<f32> = (0..k)
        .map(|j| {
            basis_host[j * n..(j + 1) * n]
                .iter()
                .zip(y_host.iter())
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();
    assert_vec_close_f32("cuda/tiled_multi_dot_f32", &got, &want);
}

#[test]
fn cuda_tiled_multi_dot_c32_matches_cpu_dotc() {
    let backend = skip_or_require_cuda(
        "tiled multi_dot c32",
        CudaBackend::new_default().map_err(|e| format!("{e:?}")),
    );
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let n = 8193usize;
    let k = 5usize;
    let tiles = 8usize;
    let y_host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i as f32 * 0.11).sin(), (i as f32 * 0.05).cos()))
        .collect();
    let basis_host: Vec<Complex32> = (0..k)
        .flat_map(|j| {
            (0..n).map(move |i| {
                Complex32::new(
                    ((i + 1) as f32).ln() + j as f32 * 0.2,
                    (i as f32 * 0.03 + j as f32).sin(),
                )
            })
        })
        .collect();

    let mut y = pool.try_alloc::<Complex32>(n).unwrap();
    pool.backend().copy_h2d(&y_host, y.raw_mut());
    let mut basis = pool.try_alloc::<Complex32>(k * n).unwrap();
    pool.backend().copy_h2d(&basis_host, basis.raw_mut());

    let mut got = vec![Complex32::new(0.0, 0.0); k];
    pool.backend()
        .try_multi_dot_tiled_with_mode_into(
            ReductionMode::Fast,
            basis.raw(),
            n,
            k,
            tiles,
            y.raw(),
            &mut got,
        )
        .unwrap();

    for (j, g) in got.iter().enumerate() {
        let want: Complex32 = basis_host[j * n..(j + 1) * n]
            .iter()
            .zip(y_host.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        let scale = (want.re.abs() + want.im.abs()).max(1.0);
        assert!(
            (g.re - want.re).abs() <= (REL_TOL as f32) * scale
                && (g.im - want.im).abs() <= (REL_TOL as f32) * scale,
            "cuda/tiled_multi_dot_c32[{j}]: {g:?} vs {want:?}"
        );
    }
}

/// Pointer-array path through the public entry: at these sizes the default
/// selection uses the tiled pointer kernels (n >= MIN_N), proving the
/// Arnoldi hot path end to end.
#[test]
fn cuda_tiled_ptr_multi_dot_f64_matches_cpu() {
    let backend = skip_or_require_cuda(
        "tiled ptr multi_dot f64",
        CudaBackend::new_default().map_err(|e| format!("{e:?}")),
    );
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let n = 200_000usize;
    let k = 8usize;
    let y_host: Vec<f64> = (0..n).map(|i| (i as f64 * 0.07).sin()).collect();
    let basis_host: Vec<Vec<f64>> = (0..k)
        .map(|j| {
            (0..n)
                .map(|i| ((i + 1) as f64).ln() + j as f64 * 0.125)
                .collect()
        })
        .collect();

    let mut y = pool.try_alloc::<f64>(n).unwrap();
    pool.backend().copy_h2d(&y_host, y.raw_mut());
    let mut basis = Vec::with_capacity(k);
    for host in &basis_host {
        let mut buf = pool.try_alloc::<f64>(n).unwrap();
        pool.backend().copy_h2d(host, buf.raw_mut());
        basis.push(buf);
    }
    let basis_refs: Vec<_> = basis.iter().map(|b| b.raw()).collect();
    let got = pool.backend().multi_dot(&basis_refs, y.raw());

    let want: Vec<f64> = basis_host
        .iter()
        .map(|x| x.iter().zip(y_host.iter()).map(|(a, b)| a * b).sum())
        .collect();
    assert_vec_close_f64("cuda/tiled_ptr_multi_dot_f64", &got, &want);
}

/// Device-output path: routes through the tiled device branch when the
/// switch is on, else the proven 1D device path. Numerics must match either
/// way, so this test is meaningful with and without the switch.
#[test]
fn cuda_tiled_multi_dot_device_output_matches_cpu() {
    let backend = skip_or_require_cuda(
        "multi_dot device output",
        CudaBackend::new_default().map_err(|e| format!("{e:?}")),
    );
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let n = 32771usize;
    let k = 6usize;
    let y_host: Vec<f64> = (0..n).map(|i| (i as f64 * 0.07).sin()).collect();
    let basis_host: Vec<f64> = (0..k)
        .flat_map(|j| (0..n).map(move |i| ((i + 1) as f64).ln() + j as f64 * 0.125))
        .collect();

    let mut y = pool.try_alloc::<f64>(n).unwrap();
    pool.backend().copy_h2d(&y_host, y.raw_mut());
    let mut basis = pool.try_alloc::<f64>(k * n).unwrap();
    pool.backend().copy_h2d(&basis_host, basis.raw_mut());
    let mut out = pool.try_alloc::<f64>(k).unwrap();
    pool.backend()
        .try_multi_dot_strided_with_mode_device_into(
            ReductionMode::Fast,
            basis.raw(),
            n,
            k,
            y.raw(),
            out.raw_mut(),
        )
        .unwrap();
    let mut got = vec![0.0f64; k];
    pool.backend().copy_d2h(out.raw(), &mut got);

    let want: Vec<f64> = (0..k)
        .map(|j| {
            basis_host[j * n..(j + 1) * n]
                .iter()
                .zip(y_host.iter())
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();
    assert_vec_close_f64("cuda/multi_dot_device_output", &got, &want);
}

#[test]
fn cuda_tiled_multi_dot_rejects_zero_tiles() {
    let backend = skip_or_require_cuda(
        "tiled multi_dot zero tiles",
        CudaBackend::new_default().map_err(|e| format!("{e:?}")),
    );
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let n = 257usize;
    let k = 3usize;
    let y = pool.try_alloc::<f64>(n).unwrap();
    let basis = pool.try_alloc::<f64>(k * n).unwrap();
    let mut out = vec![0.0f64; k];
    let result = pool.backend().try_multi_dot_tiled_with_mode_into(
        ReductionMode::Fast,
        basis.raw(),
        n,
        k,
        0,
        y.raw(),
        &mut out,
    );
    assert!(result.is_err(), "tiles=0 must be rejected, got {out:?}");
}
