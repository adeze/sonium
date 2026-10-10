// Rust guideline compliant 2026-02-21
//! Standalone SYRK probe: times `syrk_f64` at Cholesky shapes, sweeping
//! sizes in one process (order-rotated) so all sizes share each box
//! moment. `syrkprobe 48 6 976 512 256` sweeps three sizes, 6 rounds.

use sonium_compute::backend::Transpose;
use sonium_compute::cpu::blas3::syrk::syrk_f64;
use sonium_compute::cpu::blas3::trsm::TrsmUplo;

fn main() {
    let k: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);
    let rounds: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let sizes: Vec<usize> = std::env::args()
        .skip(3)
        .filter_map(|v| v.parse().ok())
        .collect();
    let sizes = if sizes.is_empty() { vec![976] } else { sizes };
    // One buffer set per size (reused across rounds, like in-situ panels
    // reuse the factor).
    let mut bufs: Vec<(Vec<f64>, Vec<f64>)> = sizes
        .iter()
        .map(|&n| {
            let a: Vec<f64> = (0..n * k).map(|i| (i as f64 * 0.0137).sin()).collect();
            (a, vec![0.0f64; n * n])
        })
        .collect();
    for (n, (a, c)) in sizes.iter().zip(bufs.iter_mut()) {
        syrk_f64(
            TrsmUplo::Lower,
            Transpose::NoTrans,
            *n,
            k,
            -1.0,
            a,
            k,
            1.0,
            c,
            *n,
        );
    }
    let mut walls = vec![Vec::with_capacity(rounds); sizes.len()];
    let total = rounds * sizes.len();
    for round in 0..total {
        let arm = (round + round / sizes.len()) % sizes.len();
        let n = sizes[arm];
        let (a, c) = &mut bufs[arm];
        let t = std::time::Instant::now();
        syrk_f64(
            TrsmUplo::Lower,
            Transpose::NoTrans,
            n,
            k,
            -1.0,
            a,
            k,
            1.0,
            c,
            n,
        );
        walls[arm].push(t.elapsed().as_secs_f64() * 1e6);
    }
    let mut sum = 0.0;
    for (arm, n) in sizes.iter().enumerate() {
        let mut s = walls[arm].clone();
        s.sort_by(|a, b| a.total_cmp(b));
        let med = s[s.len() / 2];
        sum += med;
        let gflops = 2.0 * (*n as f64) * (*n as f64) * (k as f64) / 2.0 / med / 1e3;
        println!("n={n:<4} med {med:8.1}us {gflops:6.1} GFLOPS ({rounds} rounds)");
    }
    println!("SUM {sum:.1}us over {} sizes", sizes.len());
}
