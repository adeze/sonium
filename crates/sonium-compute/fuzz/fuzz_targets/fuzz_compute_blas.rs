#![no_main]

use libfuzzer_sys::fuzz_target;
use num_complex::Complex64;
use sonium_compute::{Backend, CpuBackend, MemoryPool, MemoryPoolConfig, Transpose};

fn value(bytes: &[u8], index: usize) -> Complex64 {
    let re = bytes.get(index).copied().unwrap_or(0) as f64 / 32.0 - 4.0;
    let im = bytes.get(index + 1).copied().unwrap_or(0) as f64 / 32.0 - 4.0;
    Complex64::new(re, im)
}

fuzz_target!(|bytes: &[u8]| {
    let m = usize::from(bytes.first().copied().unwrap_or(1) % 9);
    let n = usize::from(bytes.get(1).copied().unwrap_or(1) % 9);
    let k = usize::from(bytes.get(2).copied().unwrap_or(1) % 9);
    let m = m.max(1);
    let n = n.max(1);
    let k = k.max(1);
    let trans = match bytes.get(3).copied().unwrap_or(0) % 3 {
        0 => Transpose::NoTrans,
        1 => Transpose::Trans,
        _ => Transpose::ConjTrans,
    };

    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20))
        .expect("bounded fuzz pool must construct");
    let backend = pool.backend();

    let x_len = n;
    let a_len = m.saturating_mul(n).saturating_add(8);
    let y_len = m;
    let mut a = pool
        .try_alloc::<Complex64>(a_len)
        .expect("small allocation");
    let mut x = pool
        .try_alloc::<Complex64>(x_len)
        .expect("small allocation");
    let mut y = pool
        .try_alloc::<Complex64>(y_len)
        .expect("small allocation");
    let a_values: Vec<_> = (0..a_len).map(|i| value(bytes, 4 + i * 2)).collect();
    let x_values: Vec<_> = (0..x_len).map(|i| value(bytes, 4 + i * 3)).collect();
    let y_values: Vec<_> = (0..y_len).map(|i| value(bytes, 4 + i * 5)).collect();
    backend.copy_h2d(&a_values, a.raw_mut());
    backend.copy_h2d(&x_values, x.raw_mut());
    backend.copy_h2d(&y_values, y.raw_mut());
    let lda = match trans {
        Transpose::NoTrans => n,
        Transpose::Trans | Transpose::ConjTrans => m,
    };
    let _ = backend.try_gemv(
        trans,
        m,
        n,
        value(bytes, 0),
        a.raw(),
        lda,
        x.raw(),
        value(bytes, 2),
        y.raw_mut(),
    );

    // Exercise rectangular GEMM with exactly-sized buffers.  `copy_h2d` is
    // intentionally an infallible compatibility primitive, so the fuzzer
    // must establish its length precondition before calling it; malformed
    // BLAS shapes are then handled by the checked `try_gemm` API.
    let a_gemm_len = m.saturating_mul(k);
    let b_gemm_len = k.saturating_mul(n);
    let c_gemm_len = m.saturating_mul(n);
    let mut a_gemm = pool
        .try_alloc::<Complex64>(a_gemm_len)
        .expect("small allocation");
    let mut b = pool
        .try_alloc::<Complex64>(b_gemm_len)
        .expect("small allocation");
    let mut c = pool
        .try_alloc::<Complex64>(c_gemm_len)
        .expect("small allocation");
    let a_gemm_values: Vec<_> = (0..a_gemm_len).map(|i| value(bytes, 8 + i * 2)).collect();
    let b_values: Vec<_> = (0..b_gemm_len).map(|i| value(bytes, 9 + i * 3)).collect();
    let c_values: Vec<_> = (0..c_gemm_len).map(|i| value(bytes, 10 + i * 5)).collect();
    backend.copy_h2d(&a_gemm_values, a_gemm.raw_mut());
    backend.copy_h2d(&b_values, b.raw_mut());
    backend.copy_h2d(&c_values, c.raw_mut());
    let _ = backend.try_gemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        m,
        n,
        k,
        value(bytes, 1),
        a_gemm.raw(),
        k,
        b.raw(),
        n,
        value(bytes, 3),
        c.raw_mut(),
        n,
    );
});
