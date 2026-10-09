//! Complex `gemm` via 3 real `gemm` calls (the "3M" decomposition).

#![allow(unsafe_op_in_unsafe_fn)]

//!
//! For row-major complex matrices `A` (m×k), `B` (k×n), `C` (m×n) and
//! scalars `α, β ∈ ℂ`, with `S1 = Ar+Ai`, `S2 = Br+Bi`:
//!
//! ```text
//! R = Ar·Br            (real gemm 1)
//! Q = Ai·Bi            (real gemm 2)
//! I = S1·S2            (real gemm 3)
//! Re(A·B) = R - Q
//! Im(A·B) = I - R - Q
//! C := α·(Re + i·Im) + β·C
//! ```
//!
//! Implementation:
//!
//! 1. Fused deinterleave + add (`A` / `B` → `Ar`, `Ai`, `S1` / `Br`,
//!    `Bi`, `S2`) into split real-arrays held in **thread-local pooled
//!    buffers** — repeat callers (BEM frequency sweeps, Hessenberg
//!    multiplies inside GMRES) pay no per-call allocation. AVX2
//!    shuffle/add kernel on x86, scalar fallback elsewhere.
//! 2. Three real `gemm`s through the cross-arch `sme2::gemm` shim
//!    (AVX2 on x86) into pooled `R`, `Q`, `I` tiles.
//! 3. Fused 3-input epilogue (`apply_ab3_*_row`): forms `Re`/`Im`
//!    ahead of the `α`/`β` math in one pass (no separate subtract).
//!
//! Total work: 3 real `gemm`s (25% fewer GEMM flops than 4M) plus
//! vectorised O(n²) passes. Replaces the 4M decomposition 2026-09-28
//! (4M measured 2.3-3.0x vs OpenBLAS: scalar splits + a 4th GEMM).
//!
//! Accuracy note: `Im = I - R - Q` cancels when `|Ar·Bi + Ai·Br| <<
//! max(|terms|)` (the known 3M caveat); the suite's reference tests
//! (1e-12 f64, 1e-5 f32) and downstream tolerance-based consumers
//! (iterative solvers) accept it. Native complex kernels would remove
//! the cancellation and the split traffic — the remaining lever if 3M
//! still trails the vendor (which runs complex at peak-real rate).

#![allow(clippy::too_many_arguments)]

use crate::Transpose;
use crate::sme2;
use num_complex::{Complex32, Complex64};
use std::cell::RefCell;
use std::sync::OnceLock;

// Thread-local scratch pools. Each Vec grows monotonically to the largest
// (m*k), (k*n), or (m*n) seen on this thread and is reused thereafter.
// The three real-typed gemm callees pack their own A/B internally, so we
// only hold the deinterleaved source halves, the two pre-adds, and the
// three `m × n` real intermediates.
thread_local! {
    static C32_AR: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static C32_AI: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static C32_S1: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static C32_BR: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static C32_BI: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static C32_S2: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static C32_R:  RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static C32_Q:  RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static C32_I:  RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static C64_AR: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static C64_AI: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static C64_S1: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static C64_BR: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static C64_BI: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static C64_S2: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static C64_R:  RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static C64_Q:  RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static C64_I:  RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Borrow a thread-local buffer, growing to `need` on first call.
///
/// Reentrancy: the borrow is held while `f` runs (including the three real
/// `gemm` calls and their internal `rayon::scope`s), and a rayon worker
/// waiting in one of those scopes may steal another complex GEMM onto the
/// same thread, re-entering this pool slot. `try_borrow_mut` detects that
/// case and falls back to a freshly allocated buffer owned by the nested
/// call — one allocation on the nested path instead of panicking with
/// "RefCell already borrowed". The hot non-nested path is unchanged.
fn with_buf<T, F, R>(cell: &'static std::thread::LocalKey<RefCell<Vec<T>>>, need: usize, f: F) -> R
where
    T: Copy + num_traits::Zero + 'static,
    F: FnOnce(&mut [T]) -> R,
{
    cell.with(|c| match c.try_borrow_mut() {
        Ok(mut buf) => {
            if buf.len() < need {
                buf.resize(need, T::zero());
            }
            f(&mut buf[..need])
        }
        Err(_) => {
            let mut buf = vec![T::zero(); need];
            f(&mut buf)
        }
    })
}

/// Scalar fused deinterleave + add over one linear run of `n` complex
/// elements: `re[i] = z[i].re`, `im[i] = z[i].im`, `sum[i] = re+im`.
/// Reference for the AVX2 kernels below (bitwise-identical: same op
/// order per lane — one add, no FMA).
fn split_add_linear_scalar_f32(z: &[Complex32], re: &mut [f32], im: &mut [f32], sum: &mut [f32]) {
    for (i, zij) in z.iter().enumerate() {
        re[i] = zij.re;
        im[i] = zij.im;
        sum[i] = zij.re + zij.im;
    }
}

/// f64 twin of [`split_add_linear_scalar_f32`].
fn split_add_linear_scalar_f64(z: &[Complex64], re: &mut [f64], im: &mut [f64], sum: &mut [f64]) {
    for (i, zij) in z.iter().enumerate() {
        re[i] = zij.re;
        im[i] = zij.im;
        sum[i] = zij.re + zij.im;
    }
}

/// AVX2 fused deinterleave + add over one linear run (8 complex per
/// iteration): two loads, lane shuffles (`0x88`/`0xDD` pick re/im),
/// one cross-lane permute each, one add, three stores. `sum = re+im`
/// with the same rounding as scalar (one `_mm256_add_ps`, no FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn split_add_linear_avx2_f32(
    zp: *const f32,
    re: *mut f32,
    im: *mut f32,
    sum: *mut f32,
    n: usize,
) {
    use core::arch::x86_64::*;
    // [r0,r1,r4,r5,r2,r3,r6,r7] -> [r0..r7].
    let perm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    let mut i = 0;
    while i + 8 <= n {
        let a = _mm256_loadu_ps(zp.add(i * 2));
        let b = _mm256_loadu_ps(zp.add(i * 2 + 8));
        let rv = _mm256_permutevar8x32_ps(_mm256_shuffle_ps(a, b, 0x88), perm);
        let iv = _mm256_permutevar8x32_ps(_mm256_shuffle_ps(a, b, 0xDD), perm);
        _mm256_storeu_ps(re.add(i), rv);
        _mm256_storeu_ps(im.add(i), iv);
        _mm256_storeu_ps(sum.add(i), _mm256_add_ps(rv, iv));
        i += 8;
    }
    while i < n {
        let r = *zp.add(i * 2);
        let v = *zp.add(i * 2 + 1);
        *re.add(i) = r;
        *im.add(i) = v;
        *sum.add(i) = r + v;
        i += 1;
    }
}

/// f64 twin of [`split_add_linear_avx2_f32`] (4 complex per iteration;
/// unpack + 4x64 permute instead of shuffle + 8x32 permute).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn split_add_linear_avx2_f64(
    zp: *const f64,
    re: *mut f64,
    im: *mut f64,
    sum: *mut f64,
    n: usize,
) {
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 4 <= n {
        let a = _mm256_loadu_pd(zp.add(i * 2));
        let b = _mm256_loadu_pd(zp.add(i * 2 + 4));
        // unpacklo/hi give [r0,r2,r1,r3] lane-split; [0,2,1,3] sequences.
        let rv = _mm256_permute4x64_pd(_mm256_unpacklo_pd(a, b), 0b11_01_10_00);
        let iv = _mm256_permute4x64_pd(_mm256_unpackhi_pd(a, b), 0b11_01_10_00);
        _mm256_storeu_pd(re.add(i), rv);
        _mm256_storeu_pd(im.add(i), iv);
        _mm256_storeu_pd(sum.add(i), _mm256_add_pd(rv, iv));
        i += 4;
    }
    while i < n {
        let r = *zp.add(i * 2);
        let v = *zp.add(i * 2 + 1);
        *re.add(i) = r;
        *im.add(i) = v;
        *sum.add(i) = r + v;
        i += 1;
    }
}

/// Whether the AVX2 split/add kernels apply (x86-64 with AVX2, unless
/// scalar dispatch is forced — mirrors the blas1 convention).
#[cfg(target_arch = "x86_64")]
#[inline]
fn split_avx2_on() -> bool {
    !crate::cpu::force_scalar_dispatch() && std::is_x86_feature_detected!("avx2")
}

/// Fused deinterleave + add over a `rows × cols` row-major complex
/// matrix with leading dimension `ld`: `re = Re Z`, `im = Im Z`,
/// `sum = re + im` (the 3M pre-adds `S1`/`S2`). One linear AVX2 run
/// per row (single run when `ld == cols`); scalar fallback off-x86
/// or when AVX2 is unavailable.
fn split_add_f32(
    z: &[Complex32],
    rows: usize,
    cols: usize,
    ld: usize,
    re: &mut [f32],
    im: &mut [f32],
    sum: &mut [f32],
) {
    if want_mt(rows, cols) {
        split_add_mt_f32(z, rows, cols, ld, re, im, sum);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if split_avx2_on() {
        // SAFETY: row runs are in-bounds (`rows*ld <= z.len()` by the
        // caller's slice contract, `rows*cols <= re/im/sum.len()` by
        // the pool sizing); AVX2 gated by `split_avx2_on`.
        unsafe {
            if ld == cols {
                split_add_linear_avx2_f32(
                    z.as_ptr() as *const f32,
                    re.as_mut_ptr(),
                    im.as_mut_ptr(),
                    sum.as_mut_ptr(),
                    rows * cols,
                );
                return;
            }
            for i in 0..rows {
                split_add_linear_avx2_f32(
                    z.as_ptr().add(i * ld) as *const f32,
                    re.as_mut_ptr().add(i * cols),
                    im.as_mut_ptr().add(i * cols),
                    sum.as_mut_ptr().add(i * cols),
                    cols,
                );
            }
            return;
        }
    }
    if ld == cols {
        split_add_linear_scalar_f32(z, re, im, sum);
        return;
    }
    for i in 0..rows {
        split_add_linear_scalar_f32(
            &z[i * ld..i * ld + cols],
            &mut re[i * cols..(i + 1) * cols],
            &mut im[i * cols..(i + 1) * cols],
            &mut sum[i * cols..(i + 1) * cols],
        );
    }
}

/// Pointer wrappers that are `Send + Sync` for `rayon::scope` worker
/// dispatch (house pattern, cf. `avx2::SendConstPtr`). Each task owns a
/// disjoint row band; the complex input is read-only.
#[derive(Clone, Copy)]
struct SendConstPtr<T>(*const T);
unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}
#[derive(Clone, Copy)]
struct SendMutPtr<T>(*mut T);
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

/// Minimum elements (`rows * cols`) before the splits and the epilogue
/// pay `rayon::scope` dispatch: measured 2026-09-28 (16T quiet, c3mab
/// gate sweep) — sq128 (16k) MT loses 1.34-1.76x (each MT op pays
/// ~45 µs fixed: scope install + 16-task fan-out dwarfs the ~4 µs
/// serial split), sq192 (37k) wins 0.82x, sq256+ wins 0.57-0.82x.
/// 32k elems ≈ 43 µs serial (f32) vs ~45 µs MT fixed + ~3 µs
/// parallel work — the breakeven, with margin on both sides.
const SPLIT_MT_MIN_ELEMS: usize = 32_768;

/// Kill-switch `SONIUM_C3M_MT=0` forces the serial splits/epilogue
/// (default on).
fn split_mt_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SONIUM_C3M_MT").as_deref() != Ok("0"))
}

fn want_mt(rows: usize, cols: usize) -> bool {
    rows > 1
        && rows.saturating_mul(cols) >= SPLIT_MT_MIN_ELEMS
        && rayon::current_num_threads() > 1
        && split_mt_on()
}

/// Row-band-parallel [`split_add_f32`]: one band per worker over the
/// `rows` input rows (input stride `ld`, outputs contiguous `cols`
/// rows). Bitwise-identical to serial: every element is written once
/// by exactly one band with the same op order.
fn split_add_mt_f32(
    z: &[Complex32],
    rows: usize,
    cols: usize,
    ld: usize,
    re: &mut [f32],
    im: &mut [f32],
    sum: &mut [f32],
) {
    let bands = rows.min(rayon::current_num_threads()).max(1);
    let rows_per = rows.div_ceil(bands);
    let zp = SendConstPtr(z.as_ptr());
    let rp = SendMutPtr(re.as_mut_ptr());
    let ip = SendMutPtr(im.as_mut_ptr());
    let sp = SendMutPtr(sum.as_mut_ptr());
    // SAFETY: bands partition the rows disjointly; task `b` touches
    // input rows `[r0, r1)` at stride `ld` and output rows `[r0, r1)`
    // at stride `cols`, all in-bounds by the caller contract.
    super::scope_pinned(|s| {
        for b in 0..bands {
            let (r0, r1) = (b * rows_per, ((b + 1) * rows_per).min(rows));
            if r0 >= r1 {
                continue;
            }
            s.spawn(move |_| {
                // Rebind whole wrappers (not bare `.0` fields) so the
                // closure captures the `Send` structs (cf. avx2 pack).
                let zp: SendConstPtr<Complex32> = zp;
                let rp: SendMutPtr<f32> = rp;
                let ip: SendMutPtr<f32> = ip;
                let sp: SendMutPtr<f32> = sp;
                #[cfg(target_arch = "x86_64")]
                if split_avx2_on() {
                    unsafe {
                        for i in r0..r1 {
                            split_add_linear_avx2_f32(
                                zp.0.add(i * ld) as *const f32,
                                rp.0.add(i * cols),
                                ip.0.add(i * cols),
                                sp.0.add(i * cols),
                                cols,
                            );
                        }
                    }
                    return;
                }
                for i in r0..r1 {
                    unsafe {
                        split_add_linear_scalar_f32(
                            std::slice::from_raw_parts(zp.0.add(i * ld), cols),
                            std::slice::from_raw_parts_mut(rp.0.add(i * cols), cols),
                            std::slice::from_raw_parts_mut(ip.0.add(i * cols), cols),
                            std::slice::from_raw_parts_mut(sp.0.add(i * cols), cols),
                        );
                    }
                }
            });
        }
    });
}

/// f64 twin of [`split_add_mt_f32`].
fn split_add_mt_f64(
    z: &[Complex64],
    rows: usize,
    cols: usize,
    ld: usize,
    re: &mut [f64],
    im: &mut [f64],
    sum: &mut [f64],
) {
    let bands = rows.min(rayon::current_num_threads()).max(1);
    let rows_per = rows.div_ceil(bands);
    let zp = SendConstPtr(z.as_ptr());
    let rp = SendMutPtr(re.as_mut_ptr());
    let ip = SendMutPtr(im.as_mut_ptr());
    let sp = SendMutPtr(sum.as_mut_ptr());
    // SAFETY: as in [`split_add_mt_f32`].
    super::scope_pinned(|s| {
        for b in 0..bands {
            let (r0, r1) = (b * rows_per, ((b + 1) * rows_per).min(rows));
            if r0 >= r1 {
                continue;
            }
            s.spawn(move |_| {
                // Rebind whole wrappers (not bare `.0` fields) so the
                // closure captures the `Send` structs (cf. avx2 pack).
                let zp: SendConstPtr<Complex64> = zp;
                let rp: SendMutPtr<f64> = rp;
                let ip: SendMutPtr<f64> = ip;
                let sp: SendMutPtr<f64> = sp;
                #[cfg(target_arch = "x86_64")]
                if split_avx2_on() {
                    unsafe {
                        for i in r0..r1 {
                            split_add_linear_avx2_f64(
                                zp.0.add(i * ld) as *const f64,
                                rp.0.add(i * cols),
                                ip.0.add(i * cols),
                                sp.0.add(i * cols),
                                cols,
                            );
                        }
                    }
                    return;
                }
                for i in r0..r1 {
                    unsafe {
                        split_add_linear_scalar_f64(
                            std::slice::from_raw_parts(zp.0.add(i * ld), cols),
                            std::slice::from_raw_parts_mut(rp.0.add(i * cols), cols),
                            std::slice::from_raw_parts_mut(ip.0.add(i * cols), cols),
                            std::slice::from_raw_parts_mut(sp.0.add(i * cols), cols),
                        );
                    }
                }
            });
        }
    });
}

/// f64 twin of [`split_add_f32`].
fn split_add_f64(
    z: &[Complex64],
    rows: usize,
    cols: usize,
    ld: usize,
    re: &mut [f64],
    im: &mut [f64],
    sum: &mut [f64],
) {
    if want_mt(rows, cols) {
        split_add_mt_f64(z, rows, cols, ld, re, im, sum);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if split_avx2_on() {
        // SAFETY: as in [`split_add_f32`].
        unsafe {
            if ld == cols {
                split_add_linear_avx2_f64(
                    z.as_ptr() as *const f64,
                    re.as_mut_ptr(),
                    im.as_mut_ptr(),
                    sum.as_mut_ptr(),
                    rows * cols,
                );
                return;
            }
            for i in 0..rows {
                split_add_linear_avx2_f64(
                    z.as_ptr().add(i * ld) as *const f64,
                    re.as_mut_ptr().add(i * cols),
                    im.as_mut_ptr().add(i * cols),
                    sum.as_mut_ptr().add(i * cols),
                    cols,
                );
            }
            return;
        }
    }
    if ld == cols {
        split_add_linear_scalar_f64(z, re, im, sum);
        return;
    }
    for i in 0..rows {
        split_add_linear_scalar_f64(
            &z[i * ld..i * ld + cols],
            &mut re[i * cols..(i + 1) * cols],
            &mut im[i * cols..(i + 1) * cols],
            &mut sum[i * cols..(i + 1) * cols],
        );
    }
}

/// Row-band-parallel 3M epilogue: `C[i] = apply_ab3(R[i], Q[i], I[i])`
/// over disjoint row bands. Bitwise-identical to the serial loop.
fn epilogue_mt_f32(
    r: &[f32],
    q: &[f32],
    imag: &[f32],
    c: &mut [Complex32],
    m: usize,
    n: usize,
    ldc: usize,
    alpha: Complex32,
    beta: Complex32,
    beta_zero: bool,
) {
    let bands = m.min(rayon::current_num_threads()).max(1);
    let rows_per = m.div_ceil(bands);
    let rp = SendConstPtr(r.as_ptr());
    let qp = SendConstPtr(q.as_ptr());
    let ip = SendConstPtr(imag.as_ptr());
    let cp = SendMutPtr(c.as_mut_ptr());
    // SAFETY: as in [`split_add_mt_f32`]: task `b` owns output C rows
    // `[r0, r1)` at stride `ldc` and reads the matching contiguous
    // `n`-wide rows of the three input planes.
    super::scope_pinned(|s| {
        for b in 0..bands {
            let (r0, r1) = (b * rows_per, ((b + 1) * rows_per).min(m));
            if r0 >= r1 {
                continue;
            }
            s.spawn(move |_| {
                // Rebind whole wrappers so the closure captures the
                // `Send` structs rather than bare pointer fields.
                let rp: SendConstPtr<f32> = rp;
                let qp: SendConstPtr<f32> = qp;
                let ip: SendConstPtr<f32> = ip;
                let cp: SendMutPtr<Complex32> = cp;
                for i in r0..r1 {
                    unsafe {
                        crate::cpu::blas1::apply_ab3_c32_row(
                            std::slice::from_raw_parts(rp.0.add(i * n), n),
                            std::slice::from_raw_parts(qp.0.add(i * n), n),
                            std::slice::from_raw_parts(ip.0.add(i * n), n),
                            std::slice::from_raw_parts_mut(cp.0.add(i * ldc), n),
                            alpha,
                            beta,
                            beta_zero,
                        );
                    }
                }
            });
        }
    });
}

/// f64 twin of [`epilogue_mt_f32`].
fn epilogue_mt_f64(
    r: &[f64],
    q: &[f64],
    imag: &[f64],
    c: &mut [Complex64],
    m: usize,
    n: usize,
    ldc: usize,
    alpha: Complex64,
    beta: Complex64,
    beta_zero: bool,
) {
    let bands = m.min(rayon::current_num_threads()).max(1);
    let rows_per = m.div_ceil(bands);
    let rp = SendConstPtr(r.as_ptr());
    let qp = SendConstPtr(q.as_ptr());
    let ip = SendConstPtr(imag.as_ptr());
    let cp = SendMutPtr(c.as_mut_ptr());
    // SAFETY: as in [`epilogue_mt_f32`].
    super::scope_pinned(|s| {
        for b in 0..bands {
            let (r0, r1) = (b * rows_per, ((b + 1) * rows_per).min(m));
            if r0 >= r1 {
                continue;
            }
            s.spawn(move |_| {
                // Rebind whole wrappers so the closure captures the
                // `Send` structs rather than bare pointer fields.
                let rp: SendConstPtr<f64> = rp;
                let qp: SendConstPtr<f64> = qp;
                let ip: SendConstPtr<f64> = ip;
                let cp: SendMutPtr<Complex64> = cp;
                for i in r0..r1 {
                    unsafe {
                        crate::cpu::blas1::apply_ab3_c64_row(
                            std::slice::from_raw_parts(rp.0.add(i * n), n),
                            std::slice::from_raw_parts(qp.0.add(i * n), n),
                            std::slice::from_raw_parts(ip.0.add(i * n), n),
                            std::slice::from_raw_parts_mut(cp.0.add(i * ldc), n),
                            alpha,
                            beta,
                            beta_zero,
                        );
                    }
                }
            });
        }
    });
}

/// Complex `gemm` for `(NoTrans, NoTrans)` on `Complex32` (3M: three
/// real GEMMs — `R = Ar·Br`, `Q = Ai·Bi`, `I = S1·S2` — plus a fused
/// 3-input epilogue; see the module docs).
/// Band-fused 3M gate for `gemm_c32_nn` (shipped default-on,
/// `SONIUM_C3M_FUSE=0` restores phased): engage at `n, k <= 1024`,
/// `m <= 2048`, all dims `>= 64`. The phased driver pays full-DRAM
/// split + epilogue passes (the 3 GEMMs evict the TLS scratch from
/// L3 between phases: 24-34% overhead at c512/c1k) and its MT inner
/// GEMMs stream whole-A per N-task from L3 (~31 GF/s per thread).
/// The fused driver splits B once (shared, L3-resident), then runs
/// one MT scope over row bands; each band (serially) splits its A
/// rows, runs 3 serial real GEMMs on L2-resident band tiles (~84
/// GF/s per band at c512), and folds its epilogue rows — ~8 MB DRAM
/// (A+B+C once) vs ~25 MB phased.
/// Win domain (same-process interleaved fused-vs-phased, 20 reps,
/// reproduced): c128 0.84x, c256 0.92x, c512 0.87x, c1k 0.92x,
/// c2k1k 0.89x, c1kn56 0.53x, c1kn12 0.83x — all in-gate. Losses:
/// c1536/c2k 1.01-1.09x and c1k2k 1.04-1.07x (B over 1 MB: per-band
/// B-repacks exceed the overhead saved), c56n1k 1.24x and c512n1
/// 1.10x (thin bands under 64 rows with B over 256 KB: repack
/// ratio dominates). Rule: B (`k*n` els) at most 1 MB, and bands
/// (at pool width) at least 64 rows unless B fits 256 KB; the dims
/// box keeps untested tails (below 64) and heights (m over 2048)
/// out. Bitwise vs phased where kc matches (k < 768: MT kc =
/// serial kc = 256); pass boundaries differ above (f32 wide-Kc),
/// tolerance-level. No f64 twin: the z-window is non-monotonic
/// (z128 wins 0.38x on dispatch, z256 loses 1.07x to 2D inners,
/// z512/z1536 win, z1k/z2k tie) — it needs an inner-path-aware
/// predicate, parked. Read per call for same-process A/B.
fn fuse_on(m: usize, n: usize, k: usize) -> bool {
    if std::env::var("SONIUM_C3M_FUSE")
        .map(|v| v == "0")
        .unwrap_or(false)
    {
        return false;
    }
    if !(m >= 64 && n >= 64 && k >= 64 && m <= 2048 && n <= 1024 && k <= 1024) {
        return false;
    }
    let b_els = k.saturating_mul(n);
    if b_els > 1_048_576 {
        return false;
    }
    if b_els <= 262_144 {
        return true;
    }
    let nbands = rayon::current_num_threads().max(1).min(m.max(1));
    m.div_ceil(nbands) >= 64
}

fn gemm_c32_nn_fused(
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    b: &[Complex32],
    ldb: usize,
    beta: Complex32,
    c: &mut [Complex32],
    ldc: usize,
) {
    let beta_zero = beta.re == 0.0 && beta.im == 0.0;
    with_buf(&C32_BR, k * n, |br| {
        with_buf(&C32_BI, k * n, |bi| {
            with_buf(&C32_S2, k * n, |s2| {
                split_add_f32(b, k, n, ldb, br, bi, s2);
                let nbands = rayon::current_num_threads().max(1).min(m.max(1));
                let band = m.div_ceil(nbands);
                // SAFETY: bands own disjoint C row ranges; A/B/splits
                // read-only shared. Raw pointers because borrows cannot
                // move into one spawn closure per band.
                let a_send = SendConstPtr(a.as_ptr());
                let br_send = SendConstPtr(br.as_ptr());
                let bi_send = SendConstPtr(bi.as_ptr());
                let s2_send = SendConstPtr(s2.as_ptr());
                let c_send = SendMutPtr(c.as_mut_ptr());
                let kn = k * n;
                super::scope_for_macs(m.saturating_mul(n).saturating_mul(k), |s| {
                    for bd in 0..nbands {
                        let r0 = bd * band;
                        let r1 = (r0 + band).min(m);
                        if r0 >= r1 {
                            break;
                        }
                        s.spawn(move |_| {
                            let a_send: SendConstPtr<Complex32> = a_send;
                            let br_send: SendConstPtr<f32> = br_send;
                            let bi_send: SendConstPtr<f32> = bi_send;
                            let s2_send: SendConstPtr<f32> = s2_send;
                            let c_send: SendMutPtr<Complex32> = c_send;
                            let bm = r1 - r0;
                            // Uninit band bufs (no `vec!` zeroing: 6 MB/call
                            // of memset at c512): every element is
                            // overwritten before read (splits cover
                            // ar/ai/s1; beta=0 GEMMs cover r/q/imag).
                            // SAFETY: `set_len` after reserve; all reads
                            // below follow their covering writes. The
                            // lint is allowed here (not `MaybeUninit`
                            // ceremony): the covering writes are
                            // unconditional and total, reviewed at the
                            // call sites, and locked by the fused-vs-
                            // phased bitwise test.
                            #[allow(clippy::uninit_vec)]
                            unsafe fn uninit_buf(n: usize) -> Vec<f32> {
                                let mut v = Vec::with_capacity(n);
                                v.set_len(n);
                                v
                            }
                            let mut ar = unsafe { uninit_buf(bm * k) };
                            let mut ai = unsafe { uninit_buf(bm * k) };
                            let mut s1 = unsafe { uninit_buf(bm * k) };
                            let mut r = unsafe { uninit_buf(bm * n) };
                            let mut q = unsafe { uninit_buf(bm * n) };
                            let mut imag = unsafe { uninit_buf(bm * n) };
                            // SAFETY: band slices in-bounds by construction
                            // (`r1 <= m`, `bm*k`/`bm*n` bufs); AVX2 gated.
                            unsafe {
                                #[cfg(target_arch = "x86_64")]
                                if split_avx2_on() {
                                    for i in 0..bm {
                                        split_add_linear_avx2_f32(
                                            a_send.0.add(r0 * lda + i * lda) as *const f32,
                                            ar.as_mut_ptr().add(i * k),
                                            ai.as_mut_ptr().add(i * k),
                                            s1.as_mut_ptr().add(i * k),
                                            k,
                                        );
                                    }
                                } else {
                                    for i in 0..bm {
                                        let z = core::slice::from_raw_parts(
                                            a_send.0.add(r0 * lda + i * lda),
                                            k,
                                        );
                                        split_add_linear_scalar_f32(
                                            z,
                                            &mut ar[i * k..(i + 1) * k],
                                            &mut ai[i * k..(i + 1) * k],
                                            &mut s1[i * k..(i + 1) * k],
                                        );
                                    }
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                for i in 0..bm {
                                    let z = core::slice::from_raw_parts(
                                        a_send.0.add(r0 * lda + i * lda),
                                        k,
                                    );
                                    split_add_linear_scalar_f32(
                                        z,
                                        &mut ar[i * k..(i + 1) * k],
                                        &mut ai[i * k..(i + 1) * k],
                                        &mut s1[i * k..(i + 1) * k],
                                    );
                                }
                                let br = core::slice::from_raw_parts(br_send.0, kn);
                                let bi = core::slice::from_raw_parts(bi_send.0, kn);
                                let s2 = core::slice::from_raw_parts(s2_send.0, kn);
                                super::gemm_f32_serial(
                                    Transpose::NoTrans,
                                    Transpose::NoTrans,
                                    bm,
                                    n,
                                    k,
                                    1.0,
                                    &ar,
                                    k,
                                    br,
                                    n,
                                    0.0,
                                    &mut r,
                                    n,
                                );
                                super::gemm_f32_serial(
                                    Transpose::NoTrans,
                                    Transpose::NoTrans,
                                    bm,
                                    n,
                                    k,
                                    1.0,
                                    &ai,
                                    k,
                                    bi,
                                    n,
                                    0.0,
                                    &mut q,
                                    n,
                                );
                                super::gemm_f32_serial(
                                    Transpose::NoTrans,
                                    Transpose::NoTrans,
                                    bm,
                                    n,
                                    k,
                                    1.0,
                                    &s1,
                                    k,
                                    s2,
                                    n,
                                    0.0,
                                    &mut imag,
                                    n,
                                );
                                for i in 0..bm {
                                    let crow = core::slice::from_raw_parts_mut(
                                        c_send.0.add((r0 + i) * ldc),
                                        n,
                                    );
                                    crate::cpu::blas1::apply_ab3_c32_row(
                                        &r[i * n..(i + 1) * n],
                                        &q[i * n..(i + 1) * n],
                                        &imag[i * n..(i + 1) * n],
                                        crow,
                                        alpha,
                                        beta,
                                        beta_zero,
                                    );
                                }
                            }
                        });
                    }
                });
            });
        });
    });
}

pub fn gemm_c32_nn(
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    b: &[Complex32],
    ldb: usize,
    beta: Complex32,
    c: &mut [Complex32],
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if fuse_on(m, n, k) {
        gemm_c32_nn_fused(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    }

    let ar_s = alpha.re;
    let ai_s = alpha.im;
    let br_s = beta.re;
    let bi_s = beta.im;
    let beta_zero = br_s == 0.0 && bi_s == 0.0;

    with_buf(&C32_AR, m * k, |ar| {
        with_buf(&C32_AI, m * k, |ai| {
            with_buf(&C32_S1, m * k, |s1| {
                with_buf(&C32_BR, k * n, |br| {
                    with_buf(&C32_BI, k * n, |bi| {
                        with_buf(&C32_S2, k * n, |s2| {
                            split_add_f32(a, m, k, lda, ar, ai, s1);
                            split_add_f32(b, k, n, ldb, br, bi, s2);
                            with_buf(&C32_R, m * n, |r| {
                                with_buf(&C32_Q, m * n, |q| {
                                    with_buf(&C32_I, m * n, |imag| {
                                        sme2::gemm::sgemm(
                                            m, n, k, 1.0, ar, k, br, n, 0.0, r, n, false,
                                        );
                                        sme2::gemm::sgemm(
                                            m, n, k, 1.0, ai, k, bi, n, 0.0, q, n, false,
                                        );
                                        sme2::gemm::sgemm(
                                            m, n, k, 1.0, s1, k, s2, n, 0.0, imag, n, false,
                                        );

                                        // Fused 3M epilogue into the
                                        // interleaved C, one contiguous row
                                        // at a time (vectorized when the row
                                        // is contiguous in C, i.e. ldc == n).
                                        let alpha = Complex32::new(ar_s, ai_s);
                                        let beta = Complex32::new(br_s, bi_s);
                                        if want_mt(m, n) {
                                            epilogue_mt_f32(
                                                r, q, imag, c, m, n, ldc, alpha, beta, beta_zero,
                                            );
                                        } else {
                                            for i in 0..m {
                                                crate::cpu::blas1::apply_ab3_c32_row(
                                                    &r[i * n..(i + 1) * n],
                                                    &q[i * n..(i + 1) * n],
                                                    &imag[i * n..(i + 1) * n],
                                                    &mut c[i * ldc..i * ldc + n],
                                                    alpha,
                                                    beta,
                                                    beta_zero,
                                                );
                                            }
                                        }
                                    });
                                });
                            });
                        });
                    });
                });
            });
        });
    });
}

/// Complex `gemm` for `(NoTrans, NoTrans)` on `Complex64` (3M twin of
/// [`gemm_c32_nn`]).
pub fn gemm_c64_nn(
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    b: &[Complex64],
    ldb: usize,
    beta: Complex64,
    c: &mut [Complex64],
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }

    let ar_s = alpha.re;
    let ai_s = alpha.im;
    let br_s = beta.re;
    let bi_s = beta.im;
    let beta_zero = br_s == 0.0 && bi_s == 0.0;

    with_buf(&C64_AR, m * k, |ar| {
        with_buf(&C64_AI, m * k, |ai| {
            with_buf(&C64_S1, m * k, |s1| {
                with_buf(&C64_BR, k * n, |br| {
                    with_buf(&C64_BI, k * n, |bi| {
                        with_buf(&C64_S2, k * n, |s2| {
                            split_add_f64(a, m, k, lda, ar, ai, s1);
                            split_add_f64(b, k, n, ldb, br, bi, s2);
                            with_buf(&C64_R, m * n, |r| {
                                with_buf(&C64_Q, m * n, |q| {
                                    with_buf(&C64_I, m * n, |imag| {
                                        sme2::gemm::dgemm(
                                            m, n, k, 1.0, ar, k, br, n, 0.0, r, n, false,
                                        );
                                        sme2::gemm::dgemm(
                                            m, n, k, 1.0, ai, k, bi, n, 0.0, q, n, false,
                                        );
                                        sme2::gemm::dgemm(
                                            m, n, k, 1.0, s1, k, s2, n, 0.0, imag, n, false,
                                        );

                                        // Fused 3M epilogue (see the c32
                                        // twin for the layout note).
                                        let alpha = Complex64::new(ar_s, ai_s);
                                        let beta = Complex64::new(br_s, bi_s);
                                        if want_mt(m, n) {
                                            epilogue_mt_f64(
                                                r, q, imag, c, m, n, ldc, alpha, beta, beta_zero,
                                            );
                                        } else {
                                            for i in 0..m {
                                                crate::cpu::blas1::apply_ab3_c64_row(
                                                    &r[i * n..(i + 1) * n],
                                                    &q[i * n..(i + 1) * n],
                                                    &imag[i * n..(i + 1) * n],
                                                    &mut c[i * ldc..i * ldc + n],
                                                    alpha,
                                                    beta,
                                                    beta_zero,
                                                );
                                            }
                                        }
                                    });
                                });
                            });
                        });
                    });
                });
            });
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn ref_gemm_c64(
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex64,
        a: &[Complex64],
        b: &[Complex64],
        beta: Complex64,
        c0: &[Complex64],
    ) -> Vec<Complex64> {
        let mut c = c0.to_vec();
        for i in 0..m {
            for j in 0..n {
                let mut s = Complex64::new(0.0, 0.0);
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = alpha * s + beta * c0[i * n + j];
            }
        }
        c
    }

    #[test]
    fn gemm_c64_3m_matches_reference() {
        let m = 5;
        let n = 4;
        let k = 3;
        let alpha = Complex64::new(0.3, -1.1);
        let beta = Complex64::new(0.5, 0.7);
        let a: Vec<Complex64> = (0..m * k)
            .map(|i| Complex64::new(0.1 * i as f64, 1.0 - 0.2 * i as f64))
            .collect();
        let b: Vec<Complex64> = (0..k * n)
            .map(|i| Complex64::new(-0.7 + 0.13 * i as f64, 0.25 * i as f64))
            .collect();
        let c0: Vec<Complex64> = (0..m * n)
            .map(|i| Complex64::new(1.0 + i as f64 * 0.05, 0.5 - i as f64 * 0.02))
            .collect();
        let mut c = c0.clone();
        gemm_c64_nn(m, n, k, alpha, &a, k, &b, n, beta, &mut c, n);
        let want = ref_gemm_c64(m, n, k, alpha, &a, &b, beta, &c0);
        for (a, b) in c.iter().zip(want.iter()) {
            assert_relative_eq!(a.re, b.re, max_relative = 1e-12);
            assert_relative_eq!(a.im, b.im, max_relative = 1e-12);
        }
    }

    /// Hammer the thread-local pool with 128 back-to-back complex GEMMs of
    /// varying size. The point isn't speed (we don't bench here) — it's to
    /// confirm the pool grows monotonically without over-allocating or
    /// corrupting state across calls. Pre-pool this would allocate
    /// 128 * 6 = 768 fresh Vecs; with the pool we hit the allocator at
    /// most once per Vec slot (when a new high-water size is requested).
    #[test]
    fn gemm_c64_pool_stable_across_repeated_calls() {
        // Vary shape so the pool sees several size requests.
        let shapes = [(5usize, 4, 3), (7, 6, 5), (5, 4, 3), (3, 3, 3), (7, 6, 5)];
        let alpha = Complex64::new(0.3, -1.1);
        let beta = Complex64::new(0.5, 0.7);
        for round in 0..128 {
            let (m, n, k) = shapes[round % shapes.len()];
            let a: Vec<Complex64> = (0..m * k)
                .map(|i| Complex64::new(0.01 * (i + round) as f64, 1.0 - 0.02 * i as f64))
                .collect();
            let b: Vec<Complex64> = (0..k * n)
                .map(|i| Complex64::new(-0.7 + 0.13 * i as f64, 0.25 * i as f64))
                .collect();
            let c0: Vec<Complex64> = (0..m * n)
                .map(|i| Complex64::new(1.0 + i as f64 * 0.05, 0.5 - i as f64 * 0.02))
                .collect();
            let mut c = c0.clone();
            gemm_c64_nn(m, n, k, alpha, &a, k, &b, n, beta, &mut c, n);
            let want = ref_gemm_c64(m, n, k, alpha, &a, &b, beta, &c0);
            for (g, w) in c.iter().zip(want.iter()) {
                // Absolute floor: 3M's R - Q form cancels like any split
                // method when the residual is small vs the k-dot
                // intermediates (same epsilon as the sibling test above).
                assert_relative_eq!(g.re, w.re, max_relative = 1e-12, epsilon = 1e-12);
                assert_relative_eq!(g.im, w.im, max_relative = 1e-12, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn gemm_c32_3m_matches_reference_beta_zero() {
        let m = 3;
        let n = 5;
        let k = 4;
        let alpha = Complex32::new(0.7, 0.0);
        let beta = Complex32::new(0.0, 0.0);
        let a: Vec<Complex32> = (0..m * k)
            .map(|i| Complex32::new(0.1 * i as f32, 1.0 - 0.2 * i as f32))
            .collect();
        let b: Vec<Complex32> = (0..k * n)
            .map(|i| Complex32::new(-0.7 + 0.13 * i as f32, 0.25 * i as f32))
            .collect();
        let mut c = vec![Complex32::new(0.0, 0.0); m * n];
        gemm_c32_nn(m, n, k, alpha, &a, k, &b, n, beta, &mut c, n);
        // Reference.
        let mut want = vec![Complex32::new(0.0, 0.0); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = Complex32::new(0.0, 0.0);
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                want[i * n + j] = alpha * s;
            }
        }
        for (a, b) in c.iter().zip(want.iter()) {
            assert_relative_eq!(a.re, b.re, max_relative = 1e-5, epsilon = 1e-6);
            assert_relative_eq!(a.im, b.im, max_relative = 1e-5, epsilon = 1e-6);
        }
    }

    /// Thin Schur-shaped update (`α = −1`, `β = 1`, `k = 32`) with both
    /// contiguous and strided `C`: exercises the vectorized epilogue rows
    /// (vector body + scalar tail at a non-multiple width) against the
    /// naive reference.
    #[test]
    fn gemm_c64_schur_thin_strided_matches_reference() {
        let (m, n, k) = (130usize, 129usize, 32usize);
        let alpha = Complex64::new(-1.0, 0.0);
        let beta = Complex64::new(1.0, 0.0);
        let a: Vec<Complex64> = (0..m * k)
            .map(|i| Complex64::new(0.01 * (i % 97) as f64, -0.02 * (i % 53) as f64))
            .collect();
        let b: Vec<Complex64> = (0..k * n)
            .map(|i| Complex64::new(-0.03 * (i % 61) as f64, 0.015 * (i % 79) as f64))
            .collect();
        // Reference with contiguous C.
        let c0: Vec<Complex64> = (0..m * n)
            .map(|i| Complex64::new(0.05 * (i % 37) as f64, 0.04 * (i % 41) as f64))
            .collect();
        let mut c = c0.clone();
        gemm_c64_nn(m, n, k, alpha, &a, k, &b, n, beta, &mut c, n);
        let want = ref_gemm_c64(m, n, k, alpha, &a, &b, beta, &c0);
        for (g, w) in c.iter().zip(want.iter()) {
            assert_relative_eq!(g.re, w.re, max_relative = 1e-12, epsilon = 1e-12);
            assert_relative_eq!(g.im, w.im, max_relative = 1e-12, epsilon = 1e-12);
        }
        // Strided C (ldc > n): same math, row-wise epilogue.
        let ldc = n + 17;
        let mut cs = vec![Complex64::new(0.0, 0.0); m * ldc];
        for i in 0..m {
            cs[i * ldc..i * ldc + n].copy_from_slice(&c0[i * n..(i + 1) * n]);
        }
        gemm_c64_nn(m, n, k, alpha, &a, k, &b, n, beta, &mut cs, ldc);
        for i in 0..m {
            for (g, w) in cs[i * ldc..i * ldc + n]
                .iter()
                .zip(want[i * n..(i + 1) * n].iter())
            {
                assert_relative_eq!(g.re, w.re, max_relative = 1e-12, epsilon = 1e-12);
                assert_relative_eq!(g.im, w.im, max_relative = 1e-12, epsilon = 1e-12);
            }
        }
    }

    /// AVX2 fused split+add matches the scalar reference bitwise
    /// (same op order, no FMA): vector bodies + scalar tails at
    /// non-multiple widths (7/8/9/15/16/17 f32, 3/4/5/7/8 f64),
    /// contiguous and strided (`ld > cols`) layouts.
    #[test]
    fn split_add_matches_scalar_bitwise() {
        for cols in [1usize, 3, 4, 5, 7, 8, 9, 15, 16, 17, 64] {
            for ld_extra in [0usize, 3] {
                let (rows, ld) = (5usize, cols + ld_extra);
                let z32: Vec<Complex32> = (0..rows * ld)
                    .map(|i| Complex32::new(0.13 * i as f32, -0.29 * i as f32 + 1.0))
                    .collect();
                let (mut re, mut im, mut sum) = (
                    vec![0.0f32; rows * cols],
                    vec![0.0f32; rows * cols],
                    vec![0.0f32; rows * cols],
                );
                let (mut re_s, mut im_s, mut sum_s) = (
                    vec![0.0f32; rows * cols],
                    vec![0.0f32; rows * cols],
                    vec![0.0f32; rows * cols],
                );
                split_add_f32(&z32, rows, cols, ld, &mut re, &mut im, &mut sum);
                for i in 0..rows {
                    split_add_linear_scalar_f32(
                        &z32[i * ld..i * ld + cols],
                        &mut re_s[i * cols..(i + 1) * cols],
                        &mut im_s[i * cols..(i + 1) * cols],
                        &mut sum_s[i * cols..(i + 1) * cols],
                    );
                }
                assert_eq!(re, re_s, "f32 re cols={cols} ld={ld}");
                assert_eq!(im, im_s, "f32 im cols={cols} ld={ld}");
                assert_eq!(sum, sum_s, "f32 sum cols={cols} ld={ld}");

                let z64: Vec<Complex64> = (0..rows * ld)
                    .map(|i| Complex64::new(0.13 * i as f64, -0.29 * i as f64 + 1.0))
                    .collect();
                let (mut re, mut im, mut sum) = (
                    vec![0.0f64; rows * cols],
                    vec![0.0f64; rows * cols],
                    vec![0.0f64; rows * cols],
                );
                let (mut re_s, mut im_s, mut sum_s) = (
                    vec![0.0f64; rows * cols],
                    vec![0.0f64; rows * cols],
                    vec![0.0f64; rows * cols],
                );
                split_add_f64(&z64, rows, cols, ld, &mut re, &mut im, &mut sum);
                for i in 0..rows {
                    split_add_linear_scalar_f64(
                        &z64[i * ld..i * ld + cols],
                        &mut re_s[i * cols..(i + 1) * cols],
                        &mut im_s[i * cols..(i + 1) * cols],
                        &mut sum_s[i * cols..(i + 1) * cols],
                    );
                }
                assert_eq!(re, re_s, "f64 re cols={cols} ld={ld}");
                assert_eq!(im, im_s, "f64 im cols={cols} ld={ld}");
                assert_eq!(sum, sum_s, "f64 sum cols={cols} ld={ld}");
            }
        }
    }

    /// Vectorised 3M epilogue rows match the scalar twins bitwise
    /// (scalar association, no FMA): bodies + tails, both beta arms.
    #[test]
    fn apply_ab3_rows_match_scalar_bitwise() {
        use crate::cpu::blas1::{apply_ab3_c32_row, apply_ab3_c64_row, scalar};
        for n in [1usize, 5, 8, 9, 17, 64] {
            for beta_zero in [false, true] {
                let r32: Vec<f32> = (0..n).map(|i| 0.31 * i as f32 - 1.0).collect();
                let q32: Vec<f32> = (0..n).map(|i| -0.17 * i as f32 + 0.5).collect();
                let i32: Vec<f32> = (0..n).map(|i| 0.11 * i as f32 + 2.0).collect();
                let c0: Vec<Complex32> = (0..n)
                    .map(|i| Complex32::new(0.07 * i as f32, -0.03 * i as f32))
                    .collect();
                let (alpha, beta) = (Complex32::new(0.7, -0.4), Complex32::new(1.5, 0.5));
                let (mut got, mut want) = (c0.clone(), c0.clone());
                apply_ab3_c32_row(&r32, &q32, &i32, &mut got, alpha, beta, beta_zero);
                scalar::apply_ab3_c32_row(&r32, &q32, &i32, &mut want, alpha, beta, beta_zero);
                assert_eq!(got, want, "c32 n={n} beta_zero={beta_zero}");

                let r64: Vec<f64> = (0..n).map(|i| 0.31 * i as f64 - 1.0).collect();
                let q64: Vec<f64> = (0..n).map(|i| -0.17 * i as f64 + 0.5).collect();
                let i64: Vec<f64> = (0..n).map(|i| 0.11 * i as f64 + 2.0).collect();
                let c0: Vec<Complex64> = (0..n)
                    .map(|i| Complex64::new(0.07 * i as f64, -0.03 * i as f64))
                    .collect();
                let (alpha, beta) = (Complex64::new(0.7, -0.4), Complex64::new(1.5, 0.5));
                let (mut got, mut want) = (c0.clone(), c0.clone());
                apply_ab3_c64_row(&r64, &q64, &i64, &mut got, alpha, beta, beta_zero);
                scalar::apply_ab3_c64_row(&r64, &q64, &i64, &mut want, alpha, beta, beta_zero);
                assert_eq!(got, want, "c64 n={n} beta_zero={beta_zero}");
            }
        }
    }

    /// 3M vs naive reference above the MT gate (all three splits and
    /// the epilogue take the row-band-parallel path under the default
    /// multi-threaded test pool): 1e-4 relative at k-dot depth 128.
    #[test]
    fn gemm_c32_3m_mt_shapes_match_reference() {
        let (m, n, k) = (200usize, 192usize, 128usize);
        let alpha = Complex32::new(0.7, -0.4);
        let beta = Complex32::new(0.5, 0.25);
        let a: Vec<Complex32> = (0..m * k)
            .map(|i| Complex32::new(0.01 * (i % 97) as f32, -0.02 * (i % 53) as f32))
            .collect();
        let b: Vec<Complex32> = (0..k * n)
            .map(|i| Complex32::new(-0.03 * (i % 61) as f32, 0.015 * (i % 79) as f32))
            .collect();
        let c0: Vec<Complex32> = (0..m * n)
            .map(|i| Complex32::new(0.05 * (i % 37) as f32, 0.04 * (i % 41) as f32))
            .collect();
        let mut c = c0.clone();
        gemm_c32_nn(m, n, k, alpha, &a, k, &b, n, beta, &mut c, n);
        for i in 0..m {
            for j in 0..n {
                let mut s = Complex32::new(0.0, 0.0);
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                let w = alpha * s + beta * c0[i * n + j];
                let g = c[i * n + j];
                assert_relative_eq!(g.re, w.re, max_relative = 1e-4, epsilon = 1e-4);
                assert_relative_eq!(g.im, w.im, max_relative = 1e-4, epsilon = 1e-4);
            }
        }
    }

    /// c64 twin of the MT-path coverage test: 1e-12 relative.
    #[test]
    fn gemm_c64_3m_mt_shapes_match_reference() {
        let (m, n, k) = (200usize, 192usize, 128usize);
        let alpha = Complex64::new(0.7, -0.4);
        let beta = Complex64::new(0.5, 0.25);
        let a: Vec<Complex64> = (0..m * k)
            .map(|i| Complex64::new(0.01 * (i % 97) as f64, -0.02 * (i % 53) as f64))
            .collect();
        let b: Vec<Complex64> = (0..k * n)
            .map(|i| Complex64::new(-0.03 * (i % 61) as f64, 0.015 * (i % 79) as f64))
            .collect();
        let c0: Vec<Complex64> = (0..m * n)
            .map(|i| Complex64::new(0.05 * (i % 37) as f64, 0.04 * (i % 41) as f64))
            .collect();
        let mut c = c0.clone();
        gemm_c64_nn(m, n, k, alpha, &a, k, &b, n, beta, &mut c, n);
        let want = ref_gemm_c64(m, n, k, alpha, &a, &b, beta, &c0);
        for (g, w) in c.iter().zip(want.iter()) {
            assert_relative_eq!(g.re, w.re, max_relative = 1e-12, epsilon = 1e-12);
            assert_relative_eq!(g.im, w.im, max_relative = 1e-12, epsilon = 1e-12);
        }
    }

    /// 3M vs naive reference on a larger f32 shape (48×40×32, k-dot
    /// depth where 3M cancellation could show): 1e-5 relative.
    #[test]
    fn gemm_c32_3m_larger_matches_reference() {
        let (m, n, k) = (48usize, 40usize, 32usize);
        let alpha = Complex32::new(0.7, -0.4);
        let beta = Complex32::new(0.5, 0.25);
        let a: Vec<Complex32> = (0..m * k)
            .map(|i| Complex32::new(0.01 * (i % 97) as f32, -0.02 * (i % 53) as f32))
            .collect();
        let b: Vec<Complex32> = (0..k * n)
            .map(|i| Complex32::new(-0.03 * (i % 61) as f32, 0.015 * (i % 79) as f32))
            .collect();
        let c0: Vec<Complex32> = (0..m * n)
            .map(|i| Complex32::new(0.05 * (i % 37) as f32, 0.04 * (i % 41) as f32))
            .collect();
        let mut c = c0.clone();
        gemm_c32_nn(m, n, k, alpha, &a, k, &b, n, beta, &mut c, n);
        for i in 0..m {
            for j in 0..n {
                let mut s = Complex32::new(0.0, 0.0);
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                let w = alpha * s + beta * c0[i * n + j];
                let g = c[i * n + j];
                assert_relative_eq!(g.re, w.re, max_relative = 1e-5, epsilon = 1e-5);
                assert_relative_eq!(g.im, w.im, max_relative = 1e-5, epsilon = 1e-5);
            }
        }
    }

    /// Band-fused vs phased 3M (f32): 512^3 (in-gate, kc matches so
    /// serial-vs-MT inners share pass order) must agree bit-for-bit
    /// with non-tight leading dims and non-trivial alpha/beta;
    /// 1024^3 (f32 wide-Kc moves the MT pass boundaries) within
    /// tolerance. Arms selected by explicit default/`=0` — the gate
    /// reads env per call. Engagement asserted before the lock (see
    /// the pack-vs-gather test: asserting under the guard poisons
    /// the crate-wide env lock on failure).
    #[test]
    fn gemm_c32_fused_matches_phased() {
        use crate::cpu::parallelism::env_test_lock;
        // Fixed 16-pool: the band rule keys off pool width (1024^3
        // makes 64-row bands on 16 workers, 8-row bands on the 128
        // ambient test pool), so both the gate asserts and the
        // comparison below run installed.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap();
        pool.install(|| {
            assert!(fuse_on(512, 512, 512));
            assert!(fuse_on(1024, 1024, 1024));
            assert!(!fuse_on(2048, 2048, 2048));
            assert!(!fuse_on(32, 512, 512));
            assert!(!fuse_on(256, 1024, 1024));
            assert!(!fuse_on(512, 1024, 1024));
            assert!(fuse_on(1024, 256, 1024));
            let _guard = env_test_lock();
            for (m, n, k, bitwise) in [
                (512usize, 512usize, 512usize, true),
                (1024, 1024, 1024, false),
            ] {
                let (lda, ldb, ldc) = (k + 7, n + 5, n + 3);
                let alpha = Complex32::new(0.9, -0.2);
                let beta = Complex32::new(0.35, 0.1);
                let a: Vec<Complex32> = (0..m * lda)
                    .map(|i| Complex32::new(0.01 * (i % 97) as f32, -0.02 * (i % 53) as f32))
                    .collect();
                let b: Vec<Complex32> = (0..k * ldb)
                    .map(|i| Complex32::new(-0.03 * (i % 61) as f32, 0.015 * (i % 79) as f32))
                    .collect();
                let c0: Vec<Complex32> = (0..m * ldc)
                    .map(|i| Complex32::new(0.05 * (i % 37) as f32, 0.04 * (i % 41) as f32))
                    .collect();
                // SAFETY: only writer of `SONIUM_C3M_FUSE` among tests;
                // the lock serializes env-mutating tests.
                let mut cf = c0.clone();
                unsafe {
                    std::env::remove_var("SONIUM_C3M_FUSE");
                }
                gemm_c32_nn(m, n, k, alpha, &a, lda, &b, ldb, beta, &mut cf, ldc);
                let mut cp = c0.clone();
                unsafe {
                    std::env::set_var("SONIUM_C3M_FUSE", "0");
                }
                gemm_c32_nn(m, n, k, alpha, &a, lda, &b, ldb, beta, &mut cp, ldc);
                unsafe {
                    std::env::remove_var("SONIUM_C3M_FUSE");
                }
                if bitwise {
                    assert_eq!(cf, cp, "fused must be bitwise vs phased at {m}x{n}x{k}");
                } else {
                    for (g, w) in cf.iter().zip(cp.iter()) {
                        assert_relative_eq!(g.re, w.re, max_relative = 1e-4, epsilon = 1e-4);
                        assert_relative_eq!(g.im, w.im, max_relative = 1e-4, epsilon = 1e-4);
                    }
                }
            }
        });
    }
}
