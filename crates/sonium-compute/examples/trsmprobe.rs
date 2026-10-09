use sonium_compute::Transpose;
use sonium_compute::cpu::blas3::trsm::{TrsmSide, TrsmUplo, trsm_f64};
fn main() {
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let side = if std::env::var("RIGHT").is_ok() {
        TrsmSide::Right
    } else {
        TrsmSide::Left
    };
    let uplo = if std::env::var("UP").is_ok() {
        TrsmUplo::Upper
    } else {
        TrsmUplo::Lower
    };
    let adim = match side {
        TrsmSide::Left => n,
        TrsmSide::Right => n,
    };
    let mut a = vec![0.0f64; adim * adim];
    match uplo {
        TrsmUplo::Lower => {
            for i in 0..adim {
                for j in 0..=i {
                    a[i * adim + j] = 1.0 / (1.0 + ((i + j) % 7) as f64);
                }
                a[i * adim + i] += adim as f64;
            }
        }
        TrsmUplo::Upper => {
            for i in 0..adim {
                for j in i..adim {
                    a[i * adim + j] = 1.0 / (1.0 + ((i + j) % 7) as f64);
                }
                a[i * adim + i] += adim as f64;
            }
        }
    }
    let b0: Vec<f64> = (0..n * n).map(|i| 1.0 / ((i % 13) + 1) as f64).collect();
    for _ in 0..3 {
        let mut bb = b0.clone();
        trsm_f64(
            side,
            uplo,
            Transpose::NoTrans,
            false,
            n,
            n,
            1.0,
            &a,
            adim,
            &mut bb,
            n,
        );
    }
    let mut best = f64::INFINITY;
    for _ in 0..8 {
        let mut bb = b0.clone();
        let t0 = std::time::Instant::now();
        trsm_f64(
            side,
            uplo,
            Transpose::NoTrans,
            false,
            n,
            n,
            1.0,
            &a,
            adim,
            &mut bb,
            n,
        );
        best = best.min(t0.elapsed().as_secs_f64());
    }
    println!(
        "{side:?} {uplo:?} m={n}: {:.2}ms {:.1} GF/s",
        best * 1000.0,
        (n as f64).powi(3) / 2.0 / best / 1e9
    );
}
