// Right-lower band on PACKED tile (contiguous rows) + ab hoist.
fn kernel_packed(xb: *mut f64, ldb: usize, rows: usize, nb: usize, ab: *const f64, di: *const f64) {
    unsafe {
        let mut col = [0.0f64; 128];
        for j in (0..nb).rev() {
            for k in (j + 1)..nb {
                col[k] = *ab.add(k * nb + j);
            }
            let mut i = 0;
            while i + 4 <= rows {
                let mut s0 = *xb.add(i * ldb + j);
                let mut s1 = *xb.add((i + 1) * ldb + j);
                let mut s2 = *xb.add((i + 2) * ldb + j);
                let mut s3 = *xb.add((i + 3) * ldb + j);
                for k in (j + 1)..nb {
                    let akj = col[k];
                    s0 -= *xb.add(i * ldb + k) * akj;
                    s1 -= *xb.add((i + 1) * ldb + k) * akj;
                    s2 -= *xb.add((i + 2) * ldb + k) * akj;
                    s3 -= *xb.add((i + 3) * ldb + k) * akj;
                }
                let d = *di.add(j);
                *xb.add(i * ldb + j) = s0 * d;
                *xb.add((i + 1) * ldb + j) = s1 * d;
                *xb.add((i + 2) * ldb + j) = s2 * d;
                *xb.add((i + 3) * ldb + j) = s3 * d;
                i += 4;
            }
            for i in i..rows {
                let mut s = *xb.add(i * ldb + j);
                for k in (j + 1)..nb {
                    s -= *xb.add(i * ldb + k) * col[k];
                }
                *xb.add(i * ldb + j) = s * *di.add(j);
            }
        }
    }
}
fn main() {
    let (nb, rows) = (128usize, 32usize);
    let ldb = nb; // packed: contiguous rows x nb
    let mut ab = vec![0.0f64; nb * nb];
    for i in 0..nb {
        for j in 0..=i {
            ab[i * nb + j] = 1.0 / (1.0 + ((i + j) % 7) as f64);
        }
        ab[i * nb + i] += nb as f64;
    }
    let di: Vec<f64> = (0..nb).map(|i| 1.0 / ab[i * nb + i]).collect();
    let x0: Vec<f64> = (0..rows * ldb)
        .map(|i| 1.0 / ((i % 13) + 1) as f64)
        .collect();
    let macs = (rows * nb * nb / 2) as f64;
    let mut xb = x0.clone();
    kernel_packed(xb.as_mut_ptr(), ldb, rows, nb, ab.as_ptr(), di.as_ptr());
    let t = std::time::Instant::now();
    let reps = 200;
    for _ in 0..reps {
        xb.copy_from_slice(&x0);
        kernel_packed(xb.as_mut_ptr(), ldb, rows, nb, ab.as_ptr(), di.as_ptr());
    }
    println!(
        "packed+hoist: {:.2} GF/s/core",
        macs / (t.elapsed().as_secs_f64() * 1000.0 / reps as f64) / 1e6
    );
    std::hint::black_box(&xb);
}
