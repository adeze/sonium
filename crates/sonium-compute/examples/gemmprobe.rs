use sonium_compute::Transpose;
use sonium_compute::cpu::blas3::gemm_f64;
/// Transpose mode from env (`TA`/`TB` = "N" or "T", default "N").
/// Buffer extents follow the OP dimensions: A is m×k (NoTrans) or k×m
/// (Trans) with lda = columns; B is k×n (NoTrans) or n×k (Trans).
fn bench(m: usize, n: usize, k: usize, reps: usize) {
    let ta = std::env::var("TA").map(|v| v == "T").unwrap_or(false);
    let tb = std::env::var("TB").map(|v| v == "T").unwrap_or(false);
    let (ar, ac, lda) = if ta { (k, m, m) } else { (m, k, k) };
    let (br, bc, ldb) = if tb { (n, k, k) } else { (k, n, n) };
    let a: Vec<f64> = (0..ar * ac).map(|i| 1.0 / ((i % 13) + 1) as f64).collect();
    let b: Vec<f64> = (0..br * bc).map(|i| 1.0 / ((i % 7) + 1) as f64).collect();
    let c0: Vec<f64> = (0..m * n).map(|i| (i % 5) as f64).collect();
    let mut c = c0.clone();
    let ta_e = if ta {
        Transpose::Trans
    } else {
        Transpose::NoTrans
    };
    let tb_e = if tb {
        Transpose::Trans
    } else {
        Transpose::NoTrans
    };
    for _ in 0..2 {
        c.copy_from_slice(&c0);
        gemm_f64(ta_e, tb_e, m, n, k, -1.0, &a, lda, &b, ldb, 1.0, &mut c, n);
    }
    // NOTE: the input refresh sits outside the timed region — an earlier
    // revision timed it too, understating compute throughput by up to 2x
    // on large-C shapes (a 31 MB refresh hid inside a 1.4 ms GEMM).
    let mut compute_s = 0.0;
    for _ in 0..reps {
        c.copy_from_slice(&c0);
        let rt = std::time::Instant::now();
        gemm_f64(ta_e, tb_e, m, n, k, -1.0, &a, lda, &b, ldb, 1.0, &mut c, n);
        compute_s += rt.elapsed().as_secs_f64();
    }
    let ms = compute_s * 1000.0 / reps as f64;
    println!(
        "{m}x{n}x{k}: {ms:.3}ms {:.1} GF/s",
        2.0 * (m as f64) * (n as f64) * (k as f64) / ms / 1e6
    );
    std::hint::black_box(&c);
}
fn main() {
    let (m, n, k): (usize, usize, usize) = (
        std::env::var("M")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(384),
        std::env::var("NN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512),
        std::env::var("K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128),
    );
    bench(m, n, k, 10);
}
