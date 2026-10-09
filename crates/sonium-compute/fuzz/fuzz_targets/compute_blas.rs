#![no_main]

use libfuzzer_sys::fuzz_target;
use sonium_compute::{Backend, CpuBackend, MemoryPool, MemoryPoolConfig, Transpose};

fuzz_target!(|bytes: &[u8]| {
    let byte = |index: usize| bytes.get(index).copied().unwrap_or_default() as usize;
    let m = byte(0) % 9;
    let n = byte(1) % 9;
    let lda = n.saturating_add(byte(2) % 3);
    let matrix_len = m.saturating_mul(lda.max(n));
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20))
        .expect("CPU fuzz pool must be available");
    let mut a = pool.try_alloc::<f64>(matrix_len).expect("matrix allocation");
    let mut x = pool.try_alloc::<f64>(n).expect("vector allocation");
    let mut y = pool.try_alloc::<f64>(m).expect("output allocation");
    let a_host = vec![0.25; matrix_len];
    let x_host = vec![1.0; n];
    let y_host = vec![0.0; m];
    pool.backend().copy_h2d(&a_host, a.raw_mut());
    pool.backend().copy_h2d(&x_host, x.raw_mut());
    pool.backend().copy_h2d(&y_host, y.raw_mut());

    let op = match byte(3) % 3 {
        0 => Transpose::NoTrans,
        1 => Transpose::Trans,
        _ => Transpose::ConjTrans,
    };
    let _ = pool.backend().try_gemv(
        op,
        m,
        n,
        1.0,
        a.raw(),
        lda,
        x.raw(),
        0.0,
        y.raw_mut(),
    );
});
