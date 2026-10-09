//! Vendor-parity probe: times `symmetric_eigen` end to end and dumps
//! matrix + spectrum + vectors to `/tmp/vendor_n{n}.bin` for the
//! `/tmp/vendor_compare.py` scipy check.
//!
//! Run: `cargo run --release -p sonium-solvers-eigen --example vendor_probe`

use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_eigen::symmetric_eigen;
use std::time::Instant;

fn random_symmetric(n: usize, seed: u64) -> DenseMatrix<f64> {
    // SplitMix64-style deterministic stream (matches crate test helper).
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

fn main() {
    for &n in &[64usize, 100, 150, 200, 300] {
        let a = random_symmetric(n, 1000 + n as u64);
        // Warmup.
        symmetric_eigen(&a, n).expect("warmup converges");
        let mut best = f64::INFINITY;
        let mut out = None;
        for _ in 0..5 {
            let t0 = Instant::now();
            let r = symmetric_eigen(&a, n).expect("converges");
            best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
            out = Some(r);
        }
        let (evals, evecs) = out.unwrap();
        println!("n={n} sonium {best:.1} ms");
        // Dump: u64 n, n*n f64 A (row-major), n f64 evals, n*n f64 vecs.
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
        std::fs::write(format!("/tmp/vendor_n{n}.bin"), &buf).expect("dump");
    }
}
