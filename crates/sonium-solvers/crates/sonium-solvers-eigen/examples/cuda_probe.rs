//! CUDA crossover probe: times `symmetric_eigen_cuda` (cuSOLVER Dsyevd)
//! against the CPU path and dumps matrix + spectrum + vectors to
//! `/tmp/cuda_n{n}.bin` for the `/tmp/cuda_compare.py` accuracy check.
//!
//! Run: `cargo run --release -p sonium-solvers-eigen --features cuda
//! --example cuda_probe`

#[cfg(feature = "cuda")]
fn main() {
    use sonium_solvers_core::dense::DenseMatrix;
    use sonium_solvers_eigen::symmetric_eigen;
    use std::time::Instant;

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

    use sonium_solvers_eigen::cuda::{CudaBackend, symmetric_eigen_cuda_on};
    let t_setup = Instant::now();
    let backend = CudaBackend::new_default().expect("CUDA device present");
    println!(
        "backend setup {:.1} ms",
        t_setup.elapsed().as_secs_f64() * 1000.0
    );
    println!(
        "{:>5} {:>10} {:>10} {:>8}",
        "n", "cpu_ms", "gpu_ms", "speedup"
    );
    for &n in &[64usize, 150, 300, 1000, 2000, 4000] {
        let a = random_symmetric(n, 1000 + n as u64);
        // CPU reference only where tractable (n <= 300, already baselined).
        let cpu_ms = if n <= 300 {
            symmetric_eigen(&a, n).expect("cpu warmup converges");
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t0 = Instant::now();
                symmetric_eigen(&a, n).expect("cpu converges");
                best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
            }
            best
        } else {
            f64::NAN
        };
        symmetric_eigen_cuda_on(&backend, &a, n).expect("gpu warmup converges");
        let mut gbest = f64::INFINITY;
        let mut out = None;
        for _ in 0..5 {
            let t0 = Instant::now();
            let r = symmetric_eigen_cuda_on(&backend, &a, n).expect("gpu converges");
            gbest = gbest.min(t0.elapsed().as_secs_f64() * 1000.0);
            out = Some(r);
        }
        let (evals, evecs) = out.unwrap();
        println!(
            "{n:>5} {:>10.1} {gbest:>10.1} {:>8.2}",
            cpu_ms,
            cpu_ms / gbest,
        );
        let mut buf = Vec::with_capacity(8 + 8 * (2 * n * n + n));
        buf.extend_from_slice(&(n as u64).to_le_bytes());
        for x in a.as_slice() {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        for x in &evals {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        for x in evecs.as_slice() {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(format!("/tmp/cuda_n{n}.bin"), &buf).expect("dump");
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("cuda_probe requires --features cuda");
    std::process::exit(2);
}
