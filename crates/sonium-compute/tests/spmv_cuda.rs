//! Native CUDA acceptance for the unchecked CSR SpMV fast path.
//!
//! Skip-capable on hosts without an NVIDIA driver, mirroring
//! `tests/trsm_cuda.rs`; `SONIUM_REQUIRE_CUDA=1` turns the skip into a
//! failure.

#![cfg(feature = "cuda")]

use sonium_compute::{Backend, CudaBackend};

fn cuda_or_skip() -> Option<CudaBackend> {
    match CudaBackend::new_default() {
        Ok(backend) => Some(backend),
        Err(error) if std::env::var("SONIUM_REQUIRE_CUDA").is_ok() => {
            panic!("CUDA required by SONIUM_REQUIRE_CUDA but unavailable: {error}")
        }
        Err(error) => {
            eprintln!("CUDA unavailable, skipping native SpMV acceptance: {error}");
            None
        }
    }
}

/// Tridiagonal SPD fixture (2 on the diagonal, -1 off): n rows,
/// `3n - 2` nonzeros, sorted columns.
fn tridiag_f64(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i as i32 - 1);
            values.push(-1.0);
        }
        col_idx.push(i as i32);
        values.push(2.0);
        if i + 1 < n {
            col_idx.push(i as i32 + 1);
            values.push(-1.0);
        }
        row_ptr.push(col_idx.len() as i32);
    }
    (row_ptr, col_idx, values)
}

fn host_spmv(n: usize, row_ptr: &[i32], col_idx: &[i32], values: &[f64], x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut acc = 0.0;
        for k in row_ptr[i] as usize..row_ptr[i + 1] as usize {
            acc += values[k] * x[col_idx[k] as usize];
        }
        y[i] = acc;
    }
    y
}

#[test]
fn native_cuda_spmv_unchecked_matches_checked() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let n = 64;
    let (row_ptr, col_idx, values) = tridiag_f64(n);
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.031).sin()).collect();
    let want = host_spmv(n, &row_ptr, &col_idx, &values, &x);

    let mut rp_dev = cuda.raw_alloc::<i32>(row_ptr.len()).expect("rp alloc");
    let mut ci_dev = cuda.raw_alloc::<i32>(col_idx.len()).expect("ci alloc");
    let mut v_dev = cuda.raw_alloc::<f64>(values.len()).expect("v alloc");
    let mut x_dev = cuda.raw_alloc::<f64>(x.len()).expect("x alloc");
    let mut y_checked = cuda.raw_alloc::<f64>(n).expect("y alloc");
    let mut y_unchecked = cuda.raw_alloc::<f64>(n).expect("y alloc");
    cuda.try_copy_h2d(&row_ptr, &mut rp_dev).expect("rp upload");
    cuda.try_copy_h2d(&col_idx, &mut ci_dev).expect("ci upload");
    cuda.try_copy_h2d(&values, &mut v_dev).expect("v upload");
    cuda.try_copy_h2d(&x, &mut x_dev).expect("x upload");
    cuda.try_spmv_csr(n, n, &rp_dev, &ci_dev, &v_dev, &x_dev, &mut y_checked)
        .expect("checked spmv");
    cuda.try_spmv_csr_unchecked(n, n, &rp_dev, &ci_dev, &v_dev, &x_dev, &mut y_unchecked)
        .expect("unchecked spmv");

    let mut got_checked = vec![0.0; n];
    let mut got_unchecked = vec![0.0; n];
    cuda.try_copy_d2h(&y_checked, &mut got_checked)
        .expect("dl checked");
    cuda.try_copy_d2h(&y_unchecked, &mut got_unchecked)
        .expect("dl unchecked");

    // Same descriptors + kernel: bitwise identical.
    assert_eq!(got_checked, got_unchecked);
    let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1.0);
    for (g, w) in got_unchecked.iter().zip(&want) {
        assert!(
            (g - w).abs() < 1e-12 * scale,
            "spmv mismatch vs host: {g} vs {w}"
        );
    }
}

#[test]
fn native_cuda_spmv_unchecked_keeps_dimension_checks() {
    let Some(cuda) = cuda_or_skip() else {
        return;
    };
    let n = 16;
    let (row_ptr, col_idx, values) = tridiag_f64(n);
    let x: Vec<f64> = vec![1.0; n];
    let mut rp_dev = cuda.raw_alloc::<i32>(row_ptr.len()).expect("rp alloc");
    let mut ci_dev = cuda.raw_alloc::<i32>(col_idx.len()).expect("ci alloc");
    let mut v_dev = cuda.raw_alloc::<f64>(values.len()).expect("v alloc");
    let mut x_dev = cuda.raw_alloc::<f64>(x.len()).expect("x alloc");
    // Wrong-length y: the O(1) checks must still fire.
    let mut y_short = cuda.raw_alloc::<f64>(n - 1).expect("y alloc");
    cuda.try_copy_h2d(&row_ptr, &mut rp_dev).expect("rp upload");
    cuda.try_copy_h2d(&col_idx, &mut ci_dev).expect("ci upload");
    cuda.try_copy_h2d(&values, &mut v_dev).expect("v upload");
    cuda.try_copy_h2d(&x, &mut x_dev).expect("x upload");
    let err = cuda
        .try_spmv_csr_unchecked(n, n, &rp_dev, &ci_dev, &v_dev, &x_dev, &mut y_short)
        .expect_err("short y must be rejected");
    assert!(
        format!("{err:?}").contains("spmv_csr_unchecked"),
        "unexpected error origin: {err:?}"
    );
}
