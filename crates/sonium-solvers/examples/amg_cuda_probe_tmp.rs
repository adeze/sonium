//! TEMPORARY GPU probe: upload the mirror AMG hierarchy to CUDA and compare
//! one V-cycle application against the CPU path (correctness + timing).
//!
//! Run: `cargo run --release -p sonium-solvers --features cuda --example amg_cuda_probe_tmp`

use sonium_compute::{Backend, CudaBackend, DevicePreconditioner};
use sonium_solvers::{
    AmgConfig, AmgPreconditioner, BackendAmgPreconditioner, CsrMatrix, Preconditioner,
};

fn poisson_3d(n: usize) -> CsrMatrix<f64> {
    let nn = n * n * n;
    let mut triplets = Vec::with_capacity(7 * nn);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let row = i * n * n + j * n + k;
                triplets.push((row, row, 6.0));
                if i > 0 {
                    triplets.push((row, (i - 1) * n * n + j * n + k, -1.0));
                }
                if i + 1 < n {
                    triplets.push((row, (i + 1) * n * n + j * n + k, -1.0));
                }
                if j > 0 {
                    triplets.push((row, i * n * n + (j - 1) * n + k, -1.0));
                }
                if j + 1 < n {
                    triplets.push((row, i * n * n + (j + 1) * n + k, -1.0));
                }
                if k > 0 {
                    triplets.push((row, i * n * n + j * n + k - 1, -1.0));
                }
                if k + 1 < n {
                    triplets.push((row, i * n * n + j * n + k + 1, -1.0));
                }
            }
        }
    }
    CsrMatrix::from_triplets(nn, nn, triplets)
}

fn main() {
    let cuda = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("CUDA unavailable, skipping: {e}");
            return;
        }
    };
    for &n in &[16usize, 24, 32, 48] {
        let matrix = poisson_3d(n);
        let nn = n * n * n;
        let cfg = AmgConfig::for_hypre_boomeramg_comparison();
        let t0 = std::time::Instant::now();
        let host = AmgPreconditioner::from_csr(&matrix, cfg);
        let host_setup = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = std::time::Instant::now();
        let dev = BackendAmgPreconditioner::from_host_amg(&cuda, &host);
        cuda.try_synchronize().unwrap();
        let upload_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let b: Vec<f64> = (0..nn).map(|i| i as f64).collect();
        // CPU apply: min over reps (same rigor as GPU side).
        let mut cpu_ms = f64::INFINITY;
        let mut z_cpu = vec![0.0f64; nn];
        for _ in 0..5 {
            let t2 = std::time::Instant::now();
            z_cpu = host.apply(&b);
            cpu_ms = cpu_ms.min(t2.elapsed().as_secs_f64() * 1000.0);
        }

        // CPU Richardson with Jacobi smoothing (matches the device
        // fallback smoother) for an apples-to-apples factor.
        let mut jcfg = AmgConfig::for_hypre_boomeramg_comparison();
        jcfg.smoother = sonium_solvers::AmgSmoother::L1Jacobi;
        let host_j = AmgPreconditioner::from_csr(&matrix, jcfg);
        let bnorm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut xj = vec![0.0f64; nn];
        let mut cpu_factors = Vec::new();
        for _ in 0..30 {
            let ax = matrix.matvec(&xj);
            let r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, ai)| bi - ai).collect();
            let rn: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt() / bnorm;
            cpu_factors.push(rn);
            if rn < 1e-6 {
                break;
            }
            let c = host_j.apply(&r);
            for (xi, ci) in xj.iter_mut().zip(c.iter()) {
                *xi += ci;
            }
        }

        let mut r_dev = cuda.raw_alloc::<f64>(nn).unwrap();
        let mut z_dev = cuda.raw_alloc::<f64>(nn).unwrap();
        cuda.copy_h2d(&b, &mut r_dev);
        // Warmup, then min over reps of 20 synced applies.
        for _ in 0..10 {
            dev.try_apply(&cuda, &r_dev, &mut z_dev).unwrap();
        }
        cuda.try_synchronize().unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t3 = std::time::Instant::now();
            for _ in 0..20 {
                dev.try_apply(&cuda, &r_dev, &mut z_dev).unwrap();
            }
            cuda.try_synchronize().unwrap();
            best = best.min(t3.elapsed().as_secs_f64() * 1000.0 / 20.0);
        }
        let mut z_gpu = vec![0.0f64; nn];
        cuda.copy_d2h(&z_dev, &mut z_gpu);

        // GPU Richardson to 1e-6 (device SpMV + device AMG, host norm).
        let a_dev = sonium_compute::PersistentDeviceCsr::try_from_host_csr_ref(
            &cuda,
            matrix.as_host_csr_ref(),
        )
        .unwrap();
        let mut x_dev = cuda.raw_alloc::<f64>(nn).unwrap();
        let mut ax_dev = cuda.raw_alloc::<f64>(nn).unwrap();
        let mut rr_dev = cuda.raw_alloc::<f64>(nn).unwrap();
        let mut c_dev = cuda.raw_alloc::<f64>(nn).unwrap();
        let mut b_dev = cuda.raw_alloc::<f64>(nn).unwrap();
        cuda.copy_h2d(&b, &mut b_dev);
        cuda.copy_h2d(&vec![0.0f64; nn], &mut x_dev);
        let mut gpu_factors = Vec::new();
        let t4 = std::time::Instant::now();
        for _ in 0..60 {
            a_dev.try_apply(&cuda, &x_dev, &mut ax_dev).unwrap();
            cuda.copy_h2d(&b, &mut rr_dev);
            cuda.try_axpy(-1.0, &ax_dev, &mut rr_dev).unwrap();
            let mut rr_host = vec![0.0f64; nn];
            cuda.copy_d2h(&rr_dev, &mut rr_host);
            let rn: f64 = rr_host.iter().map(|v| v * v).sum::<f64>().sqrt() / bnorm;
            gpu_factors.push(rn);
            if rn < 1e-6 {
                break;
            }
            dev.try_apply(&cuda, &rr_dev, &mut c_dev).unwrap();
            cuda.try_axpy(1.0, &c_dev, &mut x_dev).unwrap();
        }
        cuda.try_synchronize().unwrap();
        let gpu_ms = t4.elapsed().as_secs_f64() * 1000.0;

        let num: f64 = z_cpu
            .iter()
            .zip(z_gpu.iter())
            .map(|(a, c)| (a - c) * (a - c))
            .sum::<f64>()
            .sqrt();
        let den: f64 = z_cpu.iter().map(|a| a * a).sum::<f64>().sqrt();
        println!(
            "n={n} dofs={nn} levels={} host_setup={host_setup:.1}ms upload={upload_ms:.1}ms cpu_apply={cpu_ms:.2}ms gpu_apply~{best:.2}ms rel_diff={:.2e}",
            host.num_levels(),
            num / den.max(1e-300),
        );
        let cf = |f: &[f64]| {
            f.iter()
                .map(|v| format!("{v:.1e}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        println!(
            "  cpu-jacobi richardson ({} iters): {}",
            cpu_factors.len(),
            cf(&cpu_factors)
        );
        println!(
            "  gpu richardson ({} iters, {gpu_ms:.1}ms): {}",
            gpu_factors.len(),
            cf(&gpu_factors)
        );
    }
}
