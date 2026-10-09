// Rust guideline compliant 2026-02-21
//! Fast LU wall + phase probe.
//!
//! `SONIUM_LU_PHASES=1 luphases 2048 3` times one geometry; `luphases
//! 4096 6 32 48 64` interleaves nb candidates in one process (6 rounds,
//! rotating 32/48/64) for drift-robust A/B. Input matches the
//! `lapack_compare` bench exactly (seed + uniform range).

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_direct::{LuWorkspace, lu_factorize_in_place};

fn bench_input(n: usize) -> DenseMatrix<f64> {
    let mut rng = StdRng::seed_from_u64(0x6E_74_72_66);
    DenseMatrix::from_fn(n, n, |_, _| rng.random_range(-1.0..1.0))
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let nbs: Vec<usize> = std::env::args()
        .skip(3)
        .filter_map(|v| v.parse().ok())
        .collect();
    let base = bench_input(n);
    let mut ws = LuWorkspace::new();
    lu_factorize_in_place(base.clone(), &mut ws).expect("lu warmup");
    if nbs.is_empty() {
        let inputs: Vec<DenseMatrix<f64>> = (0..reps).map(|_| base.clone()).collect();
        let t = std::time::Instant::now();
        for m in inputs {
            let _ = lu_factorize_in_place(m, &mut ws).expect("lu");
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        println!("lu n={n}: {ms:.2}ms wall-avg ({reps} reps)");
        return;
    }
    // Interleaved A/B: rotate candidates every iteration so each round
    // shares one box moment; per-iteration walls print for pair counts.
    let total = reps * nbs.len();
    let inputs: Vec<DenseMatrix<f64>> = (0..total).map(|_| base.clone()).collect();
    let mut walls = vec![Vec::with_capacity(reps); nbs.len()];
    for (round, chunk) in inputs.into_iter().enumerate() {
        // Rotate the starting arm each round so no candidate always
        // runs first (cold-order bias).
        let arm = (round + round / nbs.len()) % nbs.len();
        // SAFETY: this thread is the only env writer and no worker
        // reads env between iterations (each factorization joins first).
        // Arm 0 means "shipped default" (unset the override).
        if nbs[arm] == 0 {
            unsafe { std::env::remove_var("SONIUM_LU_NB") };
        } else {
            unsafe { std::env::set_var("SONIUM_LU_NB", nbs[arm].to_string()) };
        }
        let t = std::time::Instant::now();
        let _ = lu_factorize_in_place(chunk, &mut ws).expect("lu");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        walls[arm].push(ms);
        println!("round {:>2} nb={:<3} {ms:8.2}ms", round / nbs.len(), nbs[arm]);
    }
    for (arm, nb) in nbs.iter().enumerate() {
        let mut s = walls[arm].clone();
        s.sort_by(|a, b| a.total_cmp(b));
        let med = s[s.len() / 2];
        println!("nb={nb:<3} med {med:8.2}ms over {} iters", s.len());
    }
}
