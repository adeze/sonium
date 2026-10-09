use sonium_solvers_direct::lu_factorize;
use sonium_solvers_core::dense::DenseMatrix;
fn bench(n: usize) {
    let mut rm: Vec<f64> = (0..n*n).map(|i| (i as f64 * 0.0137).sin()).collect();
    for i in 0..n { rm[i*n+i] += n as f64; }
    let mat = DenseMatrix::from_row_major_vec(n, n, rm);
    for _ in 0..2 { let _ = lu_factorize(&mat).expect("lu"); }
    let t = std::time::Instant::now(); let reps = 3;
    for _ in 0..reps { let _ = lu_factorize(&mat).expect("lu"); }
    let ms = t.elapsed().as_secs_f64()*1000.0/reps as f64;
    println!("lu n={n}: {ms:.1}ms {:.1} GF/s", 2.0*(n as f64).powi(3)/3.0/ms/1e6);
}
fn main() { bench(512); bench(1024); }
