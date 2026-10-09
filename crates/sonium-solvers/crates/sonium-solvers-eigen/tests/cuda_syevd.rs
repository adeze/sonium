//! cuSOLVER-backed symmetric eigendecomposition: accuracy against the CPU
//! path and vendor-quality residuals. Compiled with the crate's `cuda`
//! feature; skips gracefully without a CUDA device.

#![cfg(feature = "cuda")]

use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_eigen::{cuda::symmetric_eigen_cuda, symmetric_eigen};

fn random_symmetric(n: usize, seed: u64) -> DenseMatrix<f64> {
    let mut s = seed;
    let mut next = move || {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    let mut raw = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let v = (next() as f64) / (u64::MAX as f64) * 2.0 - 1.0;
            raw[i * n + j] = v;
            raw[j * n + i] = v;
        }
    }
    DenseMatrix::from_row_major_vec(n, n, raw)
}

fn require_cuda(n: usize) -> Option<DenseMatrix<f64>> {
    let a = random_symmetric(n, 7000 + n as u64);
    match symmetric_eigen_cuda(&a, n) {
        Ok(_) => Some(a),
        Err(e) => {
            eprintln!("CUDA unavailable, skipping cuda_syevd n={n}: {e}");
            None
        }
    }
}

#[test]
fn cuda_syevd_matches_cpu() {
    for &n in &[2usize, 17, 64, 150] {
        let Some(a) = require_cuda(n) else { continue };
        let an = a.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();
        let (cevals, cvecs) = symmetric_eigen_cuda(&a, n).expect("GPU solve converges");
        let (evals, _) = symmetric_eigen(&a, n).expect("CPU solve converges");
        // cuSOLVER returns ascending; CPU order is unspecified: sort both.
        let mut cs = cevals.clone();
        let mut es = evals.clone();
        cs.sort_by(|x, y| x.total_cmp(y));
        es.sort_by(|x, y| x.total_cmp(y));
        let ed = cs
            .iter()
            .zip(es.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(ed < 1e-12 * an, "n={n} cpu/gpu eig diff {ed:.3e}");
        // GPU residual + orthogonality at vendor level.
        let z = cvecs.as_slice();
        let mut resid = 0.0f64;
        for j in 0..n {
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..n {
                let mut ax = 0.0;
                for k2 in 0..n {
                    ax += a.as_slice()[i * n + k2] * z[k2 * n + j];
                }
                let r = ax - cevals[j] * z[i * n + j];
                num += r * r;
                den += z[i * n + j] * z[i * n + j];
            }
            resid = resid.max((num.sqrt() / den.sqrt().max(1e-300)) / an);
        }
        let mut orth = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut dot = 0.0;
                for k2 in 0..n {
                    dot += z[k2 * n + i] * z[k2 * n + j];
                }
                orth = orth.max(dot.abs());
            }
        }
        assert!(resid < 1e-13, "n={n} gpu resid {resid:.3e}");
        assert!(orth < 1e-13, "n={n} gpu orth {orth:.3e}");
    }
}

#[test]
fn auto_dispatch_routes_by_size() {
    use sonium_solvers_eigen::cuda::{
        AUTO_GPU_MIN_N, CudaBackend, symmetric_eigen_auto_on, symmetric_eigen_cuda_on,
    };
    // Routing assertions assume the default threshold, not a local override.
    unsafe { std::env::remove_var("SONIUM_EIGEN_GPU_MIN_N") };
    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("CUDA unavailable, skipping auto dispatch: {e}");
            return;
        }
    };
    // Below the crossover the auto entry must reproduce the CPU path.
    let n_small = (AUTO_GPU_MIN_N / 4).max(8);
    let a_small = random_symmetric(n_small, 4242);
    let (auto_evals, _) =
        symmetric_eigen_auto_on(&backend, &a_small, n_small).expect("auto small-n solves");
    let (cpu_evals, _) = symmetric_eigen(&a_small, n_small).expect("CPU solves");
    let mut cs = auto_evals.clone();
    let mut es = cpu_evals.clone();
    cs.sort_by(|x, y| x.total_cmp(y));
    es.sort_by(|x, y| x.total_cmp(y));
    let ed = cs
        .iter()
        .zip(es.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(
        ed == 0.0,
        "auto below crossover must be the CPU path, diff {ed:.3e}"
    );
    // At/above the crossover the auto entry must reproduce the GPU path.
    let n_big = AUTO_GPU_MIN_N + 60;
    let a_big = random_symmetric(n_big, 8188);
    let (auto_bevals, auto_bvecs) =
        symmetric_eigen_auto_on(&backend, &a_big, n_big).expect("auto large-n solves");
    let (gpu_evals, gpu_vecs) =
        symmetric_eigen_cuda_on(&backend, &a_big, n_big).expect("GPU solves");
    assert_eq!(
        auto_bevals, gpu_evals,
        "auto above crossover must be the GPU path"
    );
    assert_eq!(
        auto_bvecs.as_slice(),
        gpu_vecs.as_slice(),
        "auto above crossover must be the GPU path"
    );
}
