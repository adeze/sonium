// Rust guideline compliant 2026-02-21
//! Fast QR wall + phase probe: `SONIUM_QR_PHASES=1 qrphases 2048 2`.
//! Input matches the `lapack_compare` bench (seed + uniform range).

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_direct::QrFactorization;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let mut rng = StdRng::seed_from_u64(0x71_72_66);
    let base = DenseMatrix::from_fn(n, n, |_, _| rng.random_range(-1.0..1.0));
    QrFactorization::factorize(&base).expect("qr warmup");
    let t = std::time::Instant::now();
    for _ in 0..reps {
        let _ = QrFactorization::factorize(&base).expect("qr");
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    println!("qr n={n}: {ms:.2}ms wall-avg ({reps} reps)");
}
