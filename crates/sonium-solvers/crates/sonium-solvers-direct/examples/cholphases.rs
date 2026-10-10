// Rust guideline compliant 2026-02-21
//! Fast Cholesky wall + phase probe.
//!
//! `SONIUM_CHOL_PHASES=1 cholphases 1024 3` times the default geometry;
//! `cholphases 1024 6 32 48 64` interleaves `SONIUM_CHOL_NB`
//! candidates (order-rotated) for drift-robust A/B. Input is cheap
//! diagonally dominant SPD (O(n^2) build); Cholesky has no pivoting so
//! the op count matches the bench's X·X^T input.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_direct::{
    CholeskyFactorization, cholesky_factorize_in_place, cholesky_factorize_in_place_unchecked,
};

/// `SONIUM_CHOLPHASES_UNCHECKED=1` runs the unchecked twin (skips entry
/// validation + output screen) to quantify that overhead.
fn factorize(
    m: DenseMatrix<f64>,
) -> Result<CholeskyFactorization<f64>, sonium_solvers_direct::CholeskyError> {
    if std::env::var("SONIUM_CHOLPHASES_UNCHECKED").is_ok() {
        cholesky_factorize_in_place_unchecked(m)
    } else {
        cholesky_factorize_in_place(m)
    }
}

fn spd_input(n: usize) -> DenseMatrix<f64> {
    let mut rng = StdRng::seed_from_u64(0x50_74_72_66);
    let mut m = DenseMatrix::from_fn(n, n, |_, _| rng.random_range(-1.0..1.0));
    let s = m.as_mut_slice();
    for i in 0..n {
        for j in 0..i {
            let v = 0.5 * (s[i * n + j] + s[j * n + i]);
            s[i * n + j] = v;
            s[j * n + i] = v;
        }
        s[i * n + i] += n as f64;
    }
    m
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let nbs: Vec<usize> = std::env::args()
        .skip(3)
        .filter_map(|v| v.parse().ok())
        .collect();
    let base = spd_input(n);
    factorize(base.clone()).expect("chol warmup");
    if nbs.is_empty() {
        let inputs: Vec<DenseMatrix<f64>> = (0..reps).map(|_| base.clone()).collect();
        let t = std::time::Instant::now();
        for m in inputs {
            let _ = factorize(m).expect("chol");
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        println!("chol n={n}: {ms:.2}ms wall-avg ({reps} reps)");
        return;
    }
    let total = reps * nbs.len();
    let inputs: Vec<DenseMatrix<f64>> = (0..total).map(|_| base.clone()).collect();
    let mut walls = vec![Vec::with_capacity(reps); nbs.len()];
    for (round, chunk) in inputs.into_iter().enumerate() {
        let arm = (round + round / nbs.len()) % nbs.len();
        // SAFETY: this thread is the only env writer and no worker
        // reads env between iterations (each factorization joins first).
        // Arm 0 means "shipped default" (unset the override).
        if nbs[arm] == 0 {
            unsafe { std::env::remove_var("SONIUM_CHOL_NB") };
        } else {
            unsafe { std::env::set_var("SONIUM_CHOL_NB", nbs[arm].to_string()) };
        }
        let t = std::time::Instant::now();
        let _ = factorize(chunk).expect("chol");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        walls[arm].push(ms);
        println!(
            "round {:>2} nb={:<3} {ms:8.2}ms",
            round / nbs.len(),
            nbs[arm]
        );
    }
    for (arm, nb) in nbs.iter().enumerate() {
        let mut s = walls[arm].clone();
        s.sort_by(|a, b| a.total_cmp(b));
        let med = s[s.len() / 2];
        println!("nb={nb:<3} med {med:8.2}ms over {} iters", s.len());
    }
}
