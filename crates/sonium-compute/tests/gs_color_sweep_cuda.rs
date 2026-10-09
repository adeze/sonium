//! CUDA acceptance for the Gauss-Seidel color sweep kernel: the native
//! kernel path agrees with the staged CPU reference, and the launch is a
//! real device kernel (not the host-staged fallback).
//!
//! Skip-capable on hosts without an NVIDIA driver; `SONIUM_REQUIRE_CUDA=1`
//! turns the skip into a failure.

#![cfg(feature = "cuda")]

use sonium_compute::operator::DeviceCsr;
use sonium_compute::{Backend, CpuBackend, CudaBackend, MemoryPool, MemoryPoolConfig};

fn cuda_or_skip() -> Option<CudaBackend> {
    match CudaBackend::new_default() {
        Ok(backend) => Some(backend),
        Err(error) if std::env::var("SONIUM_REQUIRE_CUDA").is_ok() => {
            panic!("CUDA required by SONIUM_REQUIRE_CUDA but unavailable: {error}")
        }
        Err(error) => {
            eprintln!("CUDA unavailable, skipping GS sweep acceptance: {error}");
            None
        }
    }
}

/// Tridiagonal -1/2/-1 (SPD) with `i32` indices and a red-black coloring.
type TridiagFixture = (Vec<i32>, Vec<i32>, Vec<f64>, Vec<i32>, Vec<i32>);
fn fixture(n: usize) -> TridiagFixture {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        row_ptr.push(col_idx.len() as i32);
        if i > 0 {
            col_idx.push(i as i32 - 1);
            values.push(-1.0);
        }
        col_idx.push(i as i32);
        values.push(2.0);
        if i + 1 < n {
            col_idx.push(i as i32 + 1);
            values.push(-1.0);
        }
    }
    row_ptr.push(col_idx.len() as i32);
    let red: Vec<i32> = (0..n).step_by(2).map(|i| i as i32).collect();
    let black: Vec<i32> = (1..n).step_by(2).map(|i| i as i32).collect();
    (row_ptr, col_idx, values, red, black)
}

fn sweep<B, T>(
    backend: &B,
    a: &DeviceCsr<B, T>,
    colors: &B::Buffer<i32>,
    omega: T::Real,
    b: &B::Buffer<T>,
    x: &mut B::Buffer<T>,
) where
    B: Backend,
    T: sonium_compute::DeviceScalar,
{
    backend
        .try_gs_color_sweep(&a.row_ptr, &a.col_idx, &a.values, colors, omega, b, x)
        .unwrap();
}

#[test]
fn cuda_gs_color_sweep_matches_cpu_and_launches_kernel() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let n = 64usize;
    let (rp, ci, vals, red, black) = fixture(n);
    // Checkerboard right-hand side: the highest-frequency mode, which
    // Gauss-Seidel damps in a handful of sweeps (a smooth ramp would be
    // dominated by slow low modes and prove nothing).
    let rhs: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    let cpu_pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let cuda_pool = MemoryPool::new(cuda, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let cpu = cpu_pool.backend();
    let cuda = cuda_pool.backend();

    let a_cpu = DeviceCsr::try_from_host(cpu, n, n, &rp, &ci, &vals).unwrap();
    let a_cuda = DeviceCsr::try_from_host(cuda, n, n, &rp, &ci, &vals).unwrap();
    let mut red_cpu = cpu_pool.try_alloc::<i32>(red.len()).unwrap();
    let mut black_cpu = cpu_pool.try_alloc::<i32>(black.len()).unwrap();
    let mut red_cuda = cuda_pool.try_alloc::<i32>(red.len()).unwrap();
    let mut black_cuda = cuda_pool.try_alloc::<i32>(black.len()).unwrap();
    cpu.copy_h2d(&red, red_cpu.raw_mut());
    cpu.copy_h2d(&black, black_cpu.raw_mut());
    cuda.copy_h2d(&red, red_cuda.raw_mut());
    cuda.copy_h2d(&black, black_cuda.raw_mut());

    let mut b_cpu = cpu_pool.try_alloc::<f64>(n).unwrap();
    let mut b_cuda = cuda_pool.try_alloc::<f64>(n).unwrap();
    let mut x_cpu = cpu_pool.try_alloc::<f64>(n).unwrap();
    let mut x_cuda = cuda_pool.try_alloc::<f64>(n).unwrap();
    cpu.copy_h2d(&rhs, b_cpu.raw_mut());
    cuda.copy_h2d(&rhs, b_cuda.raw_mut());

    // Symmetric multicolor: forward then reverse, four times. CPU takes the
    // staged reference path, CUDA the native kernel.
    let before = cuda.runtime_stats();
    for _ in 0..4 {
        sweep(
            cpu,
            &a_cpu,
            red_cpu.raw(),
            1.0,
            b_cpu.raw(),
            x_cpu.raw_mut(),
        );
        sweep(
            cpu,
            &a_cpu,
            black_cpu.raw(),
            1.0,
            b_cpu.raw(),
            x_cpu.raw_mut(),
        );
        sweep(
            cpu,
            &a_cpu,
            black_cpu.raw(),
            1.0,
            b_cpu.raw(),
            x_cpu.raw_mut(),
        );
        sweep(
            cpu,
            &a_cpu,
            red_cpu.raw(),
            1.0,
            b_cpu.raw(),
            x_cpu.raw_mut(),
        );
        sweep(
            cuda,
            &a_cuda,
            red_cuda.raw(),
            1.0,
            b_cuda.raw(),
            x_cuda.raw_mut(),
        );
        sweep(
            cuda,
            &a_cuda,
            black_cuda.raw(),
            1.0,
            b_cuda.raw(),
            x_cuda.raw_mut(),
        );
        sweep(
            cuda,
            &a_cuda,
            black_cuda.raw(),
            1.0,
            b_cuda.raw(),
            x_cuda.raw_mut(),
        );
        sweep(
            cuda,
            &a_cuda,
            red_cuda.raw(),
            1.0,
            b_cuda.raw(),
            x_cuda.raw_mut(),
        );
    }
    cuda.try_synchronize().unwrap();
    let after = cuda.runtime_stats();
    assert!(
        after.kernel_launches > before.kernel_launches,
        "GS sweep did not launch a device kernel"
    );

    let mut x_cpu_host = vec![0.0; n];
    let mut x_cuda_host = vec![0.0; n];
    cpu.try_copy_d2h(x_cpu.raw(), &mut x_cpu_host).unwrap();
    cuda.try_copy_d2h(x_cuda.raw(), &mut x_cuda_host).unwrap();
    let max_diff = x_cpu_host
        .iter()
        .zip(&x_cuda_host)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(max_diff < 1e-12, "CPU/CUDA sweep mismatch {max_diff:.2e}");

    // Sanity: sweeps actually smooth (residual well below the zero guess).
    let res: f64 = x_cpu_host
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let mut ax = 2.0 * xi;
            if i > 0 {
                ax -= x_cpu_host[i - 1];
            }
            if i + 1 < n {
                ax -= x_cpu_host[i + 1];
            }
            (rhs[i] - ax).powi(2)
        })
        .sum::<f64>()
        .sqrt();
    let b_norm: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(res / b_norm < 0.05, "sweeps did not smooth");
}

#[test]
fn cuda_gs_color_sweep_complex_matches_cpu() {
    use num_complex::Complex64;
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    // Complex-shifted tridiagonal: diagonal 4+2i, off-diagonal -1.
    let n = 32usize;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        row_ptr.push(col_idx.len() as i32);
        if i > 0 {
            col_idx.push(i as i32 - 1);
            values.push(Complex64::new(-1.0, 0.0));
        }
        col_idx.push(i as i32);
        values.push(Complex64::new(4.0, 2.0));
        if i + 1 < n {
            col_idx.push(i as i32 + 1);
            values.push(Complex64::new(-1.0, 0.0));
        }
    }
    row_ptr.push(col_idx.len() as i32);
    let red: Vec<i32> = (0..n).step_by(2).map(|i| i as i32).collect();
    let black: Vec<i32> = (1..n).step_by(2).map(|i| i as i32).collect();
    let rhs: Vec<Complex64> = (0..n)
        .map(|i| {
            if i % 2 == 0 {
                Complex64::new(1.0, 0.5)
            } else {
                Complex64::new(-1.0, -0.5)
            }
        })
        .collect();

    let cpu_pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let cuda_pool = MemoryPool::new(cuda, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let cpu = cpu_pool.backend();
    let cuda = cuda_pool.backend();

    let a_cpu = DeviceCsr::try_from_host(cpu, n, n, &row_ptr, &col_idx, &values).unwrap();
    let a_cuda = DeviceCsr::try_from_host(cuda, n, n, &row_ptr, &col_idx, &values).unwrap();
    let mut red_cuda = cuda_pool.try_alloc::<i32>(red.len()).unwrap();
    let mut black_cuda = cuda_pool.try_alloc::<i32>(black.len()).unwrap();
    cuda.copy_h2d(&red, red_cuda.raw_mut());
    cuda.copy_h2d(&black, black_cuda.raw_mut());
    let mut red_cpu = cpu_pool.try_alloc::<i32>(red.len()).unwrap();
    let mut black_cpu = cpu_pool.try_alloc::<i32>(black.len()).unwrap();
    cpu.copy_h2d(&red, red_cpu.raw_mut());
    cpu.copy_h2d(&black, black_cpu.raw_mut());

    let mut b_cpu = cpu_pool.try_alloc::<Complex64>(n).unwrap();
    let mut b_cuda = cuda_pool.try_alloc::<Complex64>(n).unwrap();
    let mut x_cpu = cpu_pool.try_alloc::<Complex64>(n).unwrap();
    let mut x_cuda = cuda_pool.try_alloc::<Complex64>(n).unwrap();
    cpu.copy_h2d(&rhs, b_cpu.raw_mut());
    cuda.copy_h2d(&rhs, b_cuda.raw_mut());

    for _ in 0..4 {
        sweep(
            cpu,
            &a_cpu,
            red_cpu.raw(),
            1.0,
            b_cpu.raw(),
            x_cpu.raw_mut(),
        );
        sweep(
            cpu,
            &a_cpu,
            black_cpu.raw(),
            1.0,
            b_cpu.raw(),
            x_cpu.raw_mut(),
        );
        sweep(
            cuda,
            &a_cuda,
            red_cuda.raw(),
            1.0,
            b_cuda.raw(),
            x_cuda.raw_mut(),
        );
        sweep(
            cuda,
            &a_cuda,
            black_cuda.raw(),
            1.0,
            b_cuda.raw(),
            x_cuda.raw_mut(),
        );
    }
    cuda.try_synchronize().unwrap();

    let mut x_cpu_host = vec![Complex64::new(0.0, 0.0); n];
    let mut x_cuda_host = vec![Complex64::new(0.0, 0.0); n];
    cpu.try_copy_d2h(x_cpu.raw(), &mut x_cpu_host).unwrap();
    cuda.try_copy_d2h(x_cuda.raw(), &mut x_cuda_host).unwrap();
    let max_diff = x_cpu_host
        .iter()
        .zip(&x_cuda_host)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(
        max_diff < 1e-12,
        "complex CPU/CUDA sweep mismatch {max_diff:.2e}"
    );
}

#[test]
fn cuda_gs_color_sweep_f32_matches_cpu() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let n = 64usize;
    let (rp, ci, vals_f64, red, black) = fixture(n);
    let vals: Vec<f32> = vals_f64.iter().map(|&v| v as f32).collect();
    let rhs: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    let cpu_pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let cuda_pool = MemoryPool::new(cuda, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let cpu = cpu_pool.backend();
    let cuda = cuda_pool.backend();

    let a_cpu = DeviceCsr::try_from_host(cpu, n, n, &rp, &ci, &vals).unwrap();
    let a_cuda = DeviceCsr::try_from_host(cuda, n, n, &rp, &ci, &vals).unwrap();
    let mut red_cpu = cpu_pool.try_alloc::<i32>(red.len()).unwrap();
    let mut black_cpu = cpu_pool.try_alloc::<i32>(black.len()).unwrap();
    let mut red_cuda = cuda_pool.try_alloc::<i32>(red.len()).unwrap();
    let mut black_cuda = cuda_pool.try_alloc::<i32>(black.len()).unwrap();
    cpu.copy_h2d(&red, red_cpu.raw_mut());
    cpu.copy_h2d(&black, black_cpu.raw_mut());
    cuda.copy_h2d(&red, red_cuda.raw_mut());
    cuda.copy_h2d(&black, black_cuda.raw_mut());

    let mut b_cpu = cpu_pool.try_alloc::<f32>(n).unwrap();
    let mut b_cuda = cuda_pool.try_alloc::<f32>(n).unwrap();
    let mut x_cpu = cpu_pool.try_alloc::<f32>(n).unwrap();
    let mut x_cuda = cuda_pool.try_alloc::<f32>(n).unwrap();
    cpu.copy_h2d(&rhs, b_cpu.raw_mut());
    cuda.copy_h2d(&rhs, b_cuda.raw_mut());

    for _ in 0..4 {
        sweep(
            cpu,
            &a_cpu,
            red_cpu.raw(),
            1.0,
            b_cpu.raw(),
            x_cpu.raw_mut(),
        );
        sweep(
            cpu,
            &a_cpu,
            black_cpu.raw(),
            1.0,
            b_cpu.raw(),
            x_cpu.raw_mut(),
        );
        sweep(
            cuda,
            &a_cuda,
            red_cuda.raw(),
            1.0,
            b_cuda.raw(),
            x_cuda.raw_mut(),
        );
        sweep(
            cuda,
            &a_cuda,
            black_cuda.raw(),
            1.0,
            b_cuda.raw(),
            x_cuda.raw_mut(),
        );
    }
    cuda.try_synchronize().unwrap();

    let mut x_cpu_host = vec![0.0f32; n];
    let mut x_cuda_host = vec![0.0f32; n];
    cpu.try_copy_d2h(x_cpu.raw(), &mut x_cpu_host).unwrap();
    cuda.try_copy_d2h(x_cuda.raw(), &mut x_cuda_host).unwrap();
    let max_diff = x_cpu_host
        .iter()
        .zip(&x_cuda_host)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        max_diff < 1e-5,
        "f32 CPU/CUDA sweep mismatch {max_diff:.2e}"
    );
}

#[test]
fn cuda_gs_color_sweep_c32_matches_cpu() {
    use num_complex::Complex32;
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let n = 32usize;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        row_ptr.push(col_idx.len() as i32);
        if i > 0 {
            col_idx.push(i as i32 - 1);
            values.push(Complex32::new(-1.0, 0.0));
        }
        col_idx.push(i as i32);
        values.push(Complex32::new(4.0, 2.0));
        if i + 1 < n {
            col_idx.push(i as i32 + 1);
            values.push(Complex32::new(-1.0, 0.0));
        }
    }
    row_ptr.push(col_idx.len() as i32);
    let red: Vec<i32> = (0..n).step_by(2).map(|i| i as i32).collect();
    let black: Vec<i32> = (1..n).step_by(2).map(|i| i as i32).collect();
    let rhs: Vec<Complex32> = (0..n)
        .map(|i| {
            if i % 2 == 0 {
                Complex32::new(1.0, 0.5)
            } else {
                Complex32::new(-1.0, -0.5)
            }
        })
        .collect();

    let cpu_pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let cuda_pool = MemoryPool::new(cuda, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let cpu = cpu_pool.backend();
    let cuda = cuda_pool.backend();

    let a_cpu = DeviceCsr::try_from_host(cpu, n, n, &row_ptr, &col_idx, &values).unwrap();
    let a_cuda = DeviceCsr::try_from_host(cuda, n, n, &row_ptr, &col_idx, &values).unwrap();
    let mut red_cpu = cpu_pool.try_alloc::<i32>(red.len()).unwrap();
    let mut black_cpu = cpu_pool.try_alloc::<i32>(black.len()).unwrap();
    let mut red_cuda = cuda_pool.try_alloc::<i32>(red.len()).unwrap();
    let mut black_cuda = cuda_pool.try_alloc::<i32>(black.len()).unwrap();
    cpu.copy_h2d(&red, red_cpu.raw_mut());
    cpu.copy_h2d(&black, black_cpu.raw_mut());
    cuda.copy_h2d(&red, red_cuda.raw_mut());
    cuda.copy_h2d(&black, black_cuda.raw_mut());

    let mut b_cpu = cpu_pool.try_alloc::<Complex32>(n).unwrap();
    let mut b_cuda = cuda_pool.try_alloc::<Complex32>(n).unwrap();
    let mut x_cpu = cpu_pool.try_alloc::<Complex32>(n).unwrap();
    let mut x_cuda = cuda_pool.try_alloc::<Complex32>(n).unwrap();
    cpu.copy_h2d(&rhs, b_cpu.raw_mut());
    cuda.copy_h2d(&rhs, b_cuda.raw_mut());

    for _ in 0..4 {
        sweep(
            cpu,
            &a_cpu,
            red_cpu.raw(),
            1.0,
            b_cpu.raw(),
            x_cpu.raw_mut(),
        );
        sweep(
            cpu,
            &a_cpu,
            black_cpu.raw(),
            1.0,
            b_cpu.raw(),
            x_cpu.raw_mut(),
        );
        sweep(
            cuda,
            &a_cuda,
            red_cuda.raw(),
            1.0,
            b_cuda.raw(),
            x_cuda.raw_mut(),
        );
        sweep(
            cuda,
            &a_cuda,
            black_cuda.raw(),
            1.0,
            b_cuda.raw(),
            x_cuda.raw_mut(),
        );
    }
    cuda.try_synchronize().unwrap();

    let mut x_cpu_host = vec![Complex32::new(0.0, 0.0); n];
    let mut x_cuda_host = vec![Complex32::new(0.0, 0.0); n];
    cpu.try_copy_d2h(x_cpu.raw(), &mut x_cpu_host).unwrap();
    cuda.try_copy_d2h(x_cuda.raw(), &mut x_cuda_host).unwrap();
    let max_diff = x_cpu_host
        .iter()
        .zip(&x_cuda_host)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f32::max);
    assert!(
        max_diff < 1e-5,
        "c32 CPU/CUDA sweep mismatch {max_diff:.2e}"
    );
}
