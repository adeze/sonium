use sonium_compute::Transpose;
use sonium_compute::cpu::blas3::gemm_f64;
fn main() {
    let (m, n, k) = (384usize, 512usize, 128usize);
    let a: Vec<f64> = (0..m * k).map(|i| 1.0 / ((i % 13) + 1) as f64).collect();
    let b: Vec<f64> = (0..k * n).map(|i| 1.0 / ((i % 7) + 1) as f64).collect();
    let c0: Vec<f64> = (0..m * n).map(|i| (i % 5) as f64).collect();
    let mut c = c0.clone();
    // warm pool + caches
    for _ in 0..3 {
        c.copy_from_slice(&c0);
        gemm_f64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            -1.0,
            &a,
            k,
            &b,
            n,
            1.0,
            &mut c,
            n,
        );
    }
    for r in 0..6 {
        c.copy_from_slice(&c0);
        let t = std::time::Instant::now();
        gemm_f64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            -1.0,
            &a,
            k,
            &b,
            n,
            1.0,
            &mut c,
            n,
        );
        println!("rep{r}: {:.3}ms", t.elapsed().as_secs_f64() * 1000.0);
    }
}
