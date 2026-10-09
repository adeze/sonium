//! CPU BEM regular-pair Helmholtz kernel.
//!
//! Numerically equivalent to the GPU `bem_regular_pair_c32_batch` kernels
//! in [`crate::metal`] and [`crate::cuda`], but in f64 throughout. Uses the
//! same fixed 4-point Gauss-Tri quadrature (`TRI4_QUAD` below — must match
//! `SONIUM_BEM_TRI4_QUAD` in `kernels.metal`, the CUDA PTX kernel, and
//! `GAUCORWEI_TR4` in `sonium-bem`'s `gauss.rs`).
//!
//! Two implementations live behind one entry point ([`regular_pair_batch`]):
//!
//! - **NEON (aarch64)**: lane-parallel across 2 pairs, all vec3 / sqrt /
//!   division work as `float64x2_t`. `sin`/`cos` stay scalar per lane
//!   (libm sincos already pairs them) and are pulled into SIMD registers
//!   via `vsetq_lane_f64`. Tail pairs run the scalar path.
//! - **scalar**: a single-pair reference; the canonical f64 baseline.
//!
//! AVX2 (x86_64) and SVE (aarch64 server) follow the same lane-parallel
//! pattern when added — slot a `#[cfg(...)] mod avx2` next to `mod neon`.

use num_complex::{Complex32, Complex64};
use std::f64::consts::PI;

use crate::bem::{BemRegularPairIn, BemRegularPairOut};

/// 4-point Tri3 quadrature with weights pre-scaled by 0.5 (reference
/// triangle area). Identical to the constant baked into the Metal and
/// CUDA kernels.
const TRI4_QUAD: [(f64, f64, f64); 4] = [
    (0.333_333_333_333_333, 0.333_333_333_333_333, -0.281_250),
    (0.6, 0.2, 0.260_416_666_666_667),
    (0.2, 0.6, 0.260_416_666_666_667),
    (0.2, 0.2, 0.260_416_666_666_667),
];

/// Per-pair f64 input. CPU counterpart to [`crate::bem::BemRegularPairIn`],
/// which carries f32 to match the GPU buffer layout.
#[derive(Clone, Copy, Debug)]
pub struct RegularPairInF64 {
    pub source_point: [f64; 3],
    pub source_normal: [f64; 3],
    pub v0: [f64; 3],
    pub v1: [f64; 3],
    pub v2: [f64; 3],
}

/// Per-pair f64 output: `(G, H, H^T, E)`.
pub type RegularPairOutF64 = (Complex64, Complex64, Complex64, Complex64);

/// Batch-evaluate the four Helmholtz regular-pair integrals for `pairs`.
/// On aarch64 with NEON, runs the 2-lane SIMD kernel for pair pairs and
/// scalar for the tail; elsewhere scalar throughout.
///
/// Large batches band over threads through the machine-adaptive gate in
/// [`crate::cpu::parallelism`]. Each pair's integrals depend only on that
/// pair's data, so band boundaries never change any output value: the
/// serial and parallel paths are bitwise identical.
pub fn regular_pair_batch(
    pairs: &[RegularPairInF64],
    wave_number: f64,
    harmonic_factor: f64,
) -> Vec<RegularPairOutF64> {
    use crate::cpu::parallelism::{BEM_PAIRS_PER_THREAD, use_compute_parallel};
    use rayon::prelude::*;
    if use_compute_parallel(pairs.len(), BEM_PAIRS_PER_THREAD) {
        let threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .max(2);
        let band = pairs.len().div_ceil(threads).max(1);
        pairs
            .par_chunks(band)
            .map(|chunk| regular_pair_batch_serial(chunk, wave_number, harmonic_factor))
            .collect::<Vec<Vec<RegularPairOutF64>>>()
            .into_iter()
            .flatten()
            .collect()
    } else {
        regular_pair_batch_serial(pairs, wave_number, harmonic_factor)
    }
}

fn regular_pair_batch_serial(
    pairs: &[RegularPairInF64],
    wave_number: f64,
    harmonic_factor: f64,
) -> Vec<RegularPairOutF64> {
    let mut out = Vec::with_capacity(pairs.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        while i + 2 <= pairs.len() {
            // SAFETY: target_arch = aarch64 implies NEON is available
            // (mandatory on AArch64). All loads are from the caller's
            // slice; no aliasing.
            let (a, b) = unsafe {
                neon::regular_pair_x2(&pairs[i], &pairs[i + 1], wave_number, harmonic_factor)
            };
            out.extend([a, b]);
            i += 2;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Runtime feature gate: AVX2+FMA isn't mandatory on x86_64 like
        // NEON is on aarch64. Older Westmere/Sandy Bridge CPUs lack it.
        if !crate::cpu::force_scalar_dispatch()
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
        {
            while i + 4 <= pairs.len() {
                // SAFETY: runtime AVX2+FMA feature check above; no aliasing.
                let (a, b, c, d) = unsafe {
                    avx2::regular_pair_x4(
                        &pairs[i],
                        &pairs[i + 1],
                        &pairs[i + 2],
                        &pairs[i + 3],
                        wave_number,
                        harmonic_factor,
                    )
                };
                out.push(a);
                out.push(b);
                out.push(c);
                out.push(d);
                i += 4;
            }
        }
    }

    while i < pairs.len() {
        out.push(regular_pair_scalar(&pairs[i], wave_number, harmonic_factor));
        i += 1;
    }

    out
}

/// Cutoff below which a quadrature point is skipped as singular. Shared by
/// [`regular_pair_scalar`] and [`regular_pair_batch_with_skipped_count`] so
/// the diagnostic count matches the integration path exactly.
pub const SINGULAR_SKIP_CUTOFF: f64 = 1e-15;

/// Like [`regular_pair_batch`], but also returns the number of quadrature
/// points skipped by the singular cutoff (`r < 1e-15`). A nonzero count
/// means touching/adjacent triangles were integrated with dropped points
/// and the hypersingular `E` output is biased — route such pairs to a
/// singular-quadrature path instead of trusting the regular kernel.
pub fn regular_pair_batch_with_skipped_count(
    pairs: &[RegularPairInF64],
    wave_number: f64,
    harmonic_factor: f64,
) -> (Vec<RegularPairOutF64>, usize) {
    let out = regular_pair_batch(pairs, wave_number, harmonic_factor);
    let mut skipped = 0usize;
    for p in pairs {
        for &(xi, eta, _) in TRI4_QUAD.iter() {
            let n0 = 1.0 - xi - eta;
            let crd = [
                p.v0[0] * n0 + xi * p.v1[0] + eta * p.v2[0],
                p.v0[1] * n0 + xi * p.v1[1] + eta * p.v2[1],
                p.v0[2] * n0 + xi * p.v1[2] + eta * p.v2[2],
            ];
            let dx = crd[0] - p.source_point[0];
            let dy = crd[1] - p.source_point[1];
            let dz = crd[2] - p.source_point[2];
            if (dx * dx + dy * dy + dz * dz).sqrt() < SINGULAR_SKIP_CUTOFF {
                skipped += 1;
            }
        }
    }
    (out, skipped)
}

/// CPU/reference implementation of the double-precision BEM backend ABI.
///
/// Keeping this adapter in the compute crate gives every backend the same
/// checked shape and finite-value contract.  It is also useful for parity
/// tests: callers can compare a native CUDA result with the exact same f64
/// scalar kernel without rebuilding BEM-specific geometry code.
pub fn regular_pair_f64_batch(
    pairs: &[crate::bem::BemRegularPairInF64],
    wave_number: f64,
    harmonic_factor: f64,
) -> Vec<crate::bem::BemRegularPairOutF64> {
    pairs
        .iter()
        .map(|pair| {
            let (g, h, ht, e) = regular_pair_scalar(
                &RegularPairInF64 {
                    source_point: pair.source_point,
                    source_normal: pair.source_normal,
                    v0: pair.v0,
                    v1: pair.v1,
                    v2: pair.v2,
                },
                wave_number,
                harmonic_factor,
            );
            crate::bem::BemRegularPairOutF64 { g, h, ht, e }
        })
        .collect()
}

/// CPU fallback for the cross-backend f32 BEM regular-pair ABI.
///
/// The GPU ABI stores geometry as f32 and returns Complex32. The CPU path
/// evaluates the same fixed quadrature in f64, then narrows the result to the
/// accelerator ABI so callers get identical shape and precision semantics on
/// CPU, Metal, and CUDA.
pub fn regular_pair_c32_batch_from_gpu_abi(
    pairs: &[BemRegularPairIn],
    wave_number: f32,
    harmonic_factor: f32,
) -> Vec<BemRegularPairOut> {
    let pairs_f64: Vec<RegularPairInF64> = pairs
        .iter()
        .map(|p| RegularPairInF64 {
            source_point: f32x3_to_f64(p.source_point),
            source_normal: f32x3_to_f64(p.source_normal),
            v0: f32x3_to_f64(p.v0),
            v1: f32x3_to_f64(p.v1),
            v2: f32x3_to_f64(p.v2),
        })
        .collect();

    regular_pair_batch(&pairs_f64, wave_number as f64, harmonic_factor as f64)
        .into_iter()
        .map(|(g, h, ht, e)| BemRegularPairOut {
            g: c64_to_c32(g),
            h: c64_to_c32(h),
            ht: c64_to_c32(ht),
            e: c64_to_c32(e),
        })
        .collect()
}

/// Scalar reference: numerically identical to the Metal/CUDA kernel in
/// f64. Hot loop inside the SIMD tail and the canonical baseline.
#[inline]
pub fn regular_pair_scalar(
    p: &RegularPairInF64,
    wave_number: f64,
    harmonic_factor: f64,
) -> RegularPairOutF64 {
    let wavruim = harmonic_factor * wave_number;
    let k2 = wave_number * wave_number;

    let e1 = sub3(p.v1, p.v0);
    let e2 = sub3(p.v2, p.v0);
    let raw_n = cross3(e1, e2);
    let jac = norm3(raw_n);
    let n_y = if jac > 1e-15 {
        scale3(raw_n, 1.0 / jac)
    } else {
        [0.0; 3]
    };

    let mut g_re = 0.0;
    let mut g_im = 0.0;
    let mut h_re = 0.0;
    let mut h_im = 0.0;
    let mut ht_re = 0.0;
    let mut ht_im = 0.0;
    let mut e_re = 0.0;
    let mut e_im = 0.0;

    for &(xi, eta, w) in TRI4_QUAD.iter() {
        let n0 = 1.0 - xi - eta;
        let crd = axpy3(axpy3(scale3(p.v0, n0), xi, p.v1), eta, p.v2);
        let wga = w * jac;
        let diff = sub3(crd, p.source_point);
        let r = norm3(diff);
        if r < SINGULAR_SKIP_CUTOFF {
            continue;
        }
        let ur = scale3(diff, 1.0 / r);
        let re1 = wavruim * r;
        let re2 = wga / (4.0 * PI * r);
        let phase = sonium_fastmath::exp_i(re1);
        let zg_re = phase.re * re2;
        let zg_im = phase.im * re2;
        let z1_re = -1.0 / r;
        let zhh_base_re = zg_re * z1_re - zg_im * wavruim;
        let zhh_base_im = zg_re * wavruim + zg_im * z1_re;
        let r1h = dot3(ur, n_y);
        let r2h = -dot3(ur, p.source_normal);
        let zhh_re = zhh_base_re * r1h;
        let zhh_im = zhh_base_im * r1h;
        let zht_re = zhh_base_re * r2h;
        let zht_im = zhh_base_im * r2h;
        let rq = r1h * r2h;
        let nx_dot_ny = dot3(p.source_normal, n_y);
        let dq = r * r;
        let ze_factor_re = (3.0 / dq - k2) * rq + nx_dot_ny / dq;
        let ze_factor_im = -wavruim / r * (3.0 * rq + nx_dot_ny);
        let ze_re = zg_re * ze_factor_re - zg_im * ze_factor_im;
        let ze_im = zg_re * ze_factor_im + zg_im * ze_factor_re;
        g_re += zg_re;
        g_im += zg_im;
        h_re += zhh_re;
        h_im += zhh_im;
        ht_re += zht_re;
        ht_im += zht_im;
        e_re += ze_re;
        e_im += ze_im;
    }
    (
        Complex64::new(g_re, g_im),
        Complex64::new(h_re, h_im),
        Complex64::new(ht_re, ht_im),
        Complex64::new(e_re, e_im),
    )
}

// ---------------------------------------------------------------------------
// scalar vec3 helpers (private)
// ---------------------------------------------------------------------------

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn norm3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
fn axpy3(a: [f64; 3], t: f64, b: [f64; 3]) -> [f64; 3] {
    [a[0] + t * b[0], a[1] + t * b[1], a[2] + t * b[2]]
}

#[inline]
fn f32x3_to_f64(v: [f32; 3]) -> [f64; 3] {
    [v[0] as f64, v[1] as f64, v[2] as f64]
}

#[inline]
fn c64_to_c32(z: Complex64) -> Complex32 {
    Complex32::new(z.re as f32, z.im as f32)
}

// ---------------------------------------------------------------------------
// NEON 2-pair SIMD
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{RegularPairInF64, RegularPairOutF64, TRI4_QUAD};
    use num_complex::Complex64;
    use std::arch::aarch64::*;
    use std::f64::consts::PI;

    /// Pack scalar `a, b` into a `float64x2_t` with lane 0 = a, lane 1 = b.
    #[inline]
    unsafe fn pack(a: f64, b: f64) -> float64x2_t {
        // SAFETY: vsetq_lane_f64 is safe for any in-range lane index (1 here).
        unsafe { vsetq_lane_f64(b, vdupq_n_f64(a), 1) }
    }

    /// Process pairs `a, b` in lane-parallel fashion. Lane 0 holds `a`'s
    /// state, lane 1 holds `b`'s. The scalar code path is replicated
    /// faithfully — including the `r < 1e-15` skip — by masking the
    /// per-quad-point contribution rather than branching on lane state.
    #[target_feature(enable = "neon")]
    pub unsafe fn regular_pair_x2(
        a: &RegularPairInF64,
        b: &RegularPairInF64,
        wave_number: f64,
        harmonic_factor: f64,
    ) -> (RegularPairOutF64, RegularPairOutF64) {
        // SAFETY: this whole block executes inside a `target_feature = "neon"`
        // function, so every NEON intrinsic is sound under the calling
        // contract for aarch64.
        unsafe {
            let wavruim = harmonic_factor * wave_number;
            let k2 = wave_number * wave_number;

            // Source point + normal: 3× f64x2 each
            let sp_x = pack(a.source_point[0], b.source_point[0]);
            let sp_y = pack(a.source_point[1], b.source_point[1]);
            let sp_z = pack(a.source_point[2], b.source_point[2]);
            let sn_x = pack(a.source_normal[0], b.source_normal[0]);
            let sn_y = pack(a.source_normal[1], b.source_normal[1]);
            let sn_z = pack(a.source_normal[2], b.source_normal[2]);

            // Triangle vertices
            let v0_x = pack(a.v0[0], b.v0[0]);
            let v0_y = pack(a.v0[1], b.v0[1]);
            let v0_z = pack(a.v0[2], b.v0[2]);
            let v1_x = pack(a.v1[0], b.v1[0]);
            let v1_y = pack(a.v1[1], b.v1[1]);
            let v1_z = pack(a.v1[2], b.v1[2]);
            let v2_x = pack(a.v2[0], b.v2[0]);
            let v2_y = pack(a.v2[1], b.v2[1]);
            let v2_z = pack(a.v2[2], b.v2[2]);

            // e1 = v1 - v0,  e2 = v2 - v0
            let e1_x = vsubq_f64(v1_x, v0_x);
            let e1_y = vsubq_f64(v1_y, v0_y);
            let e1_z = vsubq_f64(v1_z, v0_z);
            let e2_x = vsubq_f64(v2_x, v0_x);
            let e2_y = vsubq_f64(v2_y, v0_y);
            let e2_z = vsubq_f64(v2_z, v0_z);

            // raw_n = cross(e1, e2)
            let nx = vfmsq_f64(vmulq_f64(e1_y, e2_z), e1_z, e2_y);
            let ny = vfmsq_f64(vmulq_f64(e1_z, e2_x), e1_x, e2_z);
            let nz = vfmsq_f64(vmulq_f64(e1_x, e2_y), e1_y, e2_x);
            let eps_v = vdupq_n_f64(1e-15);
            let one_v = vdupq_n_f64(1.0);
            let zero_v = vdupq_n_f64(0.0);

            // jac = |raw_n|; n_y = raw_n / jac (per-lane mask on jac > eps)
            let jac_sq = vfmaq_f64(vfmaq_f64(vmulq_f64(nx, nx), ny, ny), nz, nz);
            let jac = vsqrtq_f64(jac_sq);
            let jac_ok = vcgtq_f64(jac, eps_v);
            let inv_jac = vdivq_f64(one_v, jac);
            let inv_jac_masked = vbslq_f64(jac_ok, inv_jac, zero_v);
            let ny_x = vmulq_f64(nx, inv_jac_masked);
            let ny_y = vmulq_f64(ny, inv_jac_masked);
            let ny_z = vmulq_f64(nz, inv_jac_masked);

            // Cross-quadrature constant
            let nx_dot_ny = vfmaq_f64(vfmaq_f64(vmulq_f64(sn_x, ny_x), sn_y, ny_y), sn_z, ny_z);

            // Complex accumulators: 4 outputs × {re, im} = 8 lane-parallel sums
            let mut g_re = vdupq_n_f64(0.0);
            let mut g_im = vdupq_n_f64(0.0);
            let mut h_re = vdupq_n_f64(0.0);
            let mut h_im = vdupq_n_f64(0.0);
            let mut ht_re = vdupq_n_f64(0.0);
            let mut ht_im = vdupq_n_f64(0.0);
            let mut e_re = vdupq_n_f64(0.0);
            let mut e_im = vdupq_n_f64(0.0);

            let wavruim_v = vdupq_n_f64(wavruim);
            let neg_wavruim_v = vnegq_f64(wavruim_v);
            let k2_v = vdupq_n_f64(k2);
            let three_v = vdupq_n_f64(3.0);
            let one_over_4pi = vdupq_n_f64(1.0 / (4.0 * PI));

            for &(xi, eta, w) in TRI4_QUAD.iter() {
                let n0 = 1.0 - xi - eta;
                let xi_v = vdupq_n_f64(xi);
                let eta_v = vdupq_n_f64(eta);
                let n0_v = vdupq_n_f64(n0);
                let w_v = vdupq_n_f64(w);

                // crd = n0 * v0 + xi * v1 + eta * v2
                let crd_x = vfmaq_f64(vfmaq_f64(vmulq_f64(v0_x, n0_v), v1_x, xi_v), v2_x, eta_v);
                let crd_y = vfmaq_f64(vfmaq_f64(vmulq_f64(v0_y, n0_v), v1_y, xi_v), v2_y, eta_v);
                let crd_z = vfmaq_f64(vfmaq_f64(vmulq_f64(v0_z, n0_v), v1_z, xi_v), v2_z, eta_v);

                let wga = vmulq_f64(w_v, jac);
                let diff_x = vsubq_f64(crd_x, sp_x);
                let diff_y = vsubq_f64(crd_y, sp_y);
                let diff_z = vsubq_f64(crd_z, sp_z);

                let r_sq = vfmaq_f64(
                    vfmaq_f64(vmulq_f64(diff_x, diff_x), diff_y, diff_y),
                    diff_z,
                    diff_z,
                );
                let r = vsqrtq_f64(r_sq);

                // Per-lane "r > eps" mask. Below eps we contribute nothing,
                // matching the scalar `continue`. Achieved by zeroing inv_r,
                // which factors through every downstream term.
                let r_ok = vcgtq_f64(r, eps_v);
                let inv_r = vdivq_f64(one_v, r);
                let inv_r_m = vbslq_f64(r_ok, inv_r, zero_v);

                let ur_x = vmulq_f64(diff_x, inv_r_m);
                let ur_y = vmulq_f64(diff_y, inv_r_m);
                let ur_z = vmulq_f64(diff_z, inv_r_m);

                // re1 = wavruim * r;  re2 = wga / (4π r) (masked)
                let re1 = vmulq_f64(wavruim_v, r);
                let re2 = vmulq_f64(vmulq_f64(wga, one_over_4pi), inv_r_m);

                // Lane-parallel sin/cos via the sonium-fastmath NEON kernel:
                // Cody–Waite range reduction + Sleef u10 minimax polynomials,
                // entirely in `float64x2_t`. Replaces the prior two scalar
                // libm calls per quadrature point. The kernel is accurate to
                // within ~8 ULP of `f64::sin_cos` over `|x| ≤ 1e6`, well below
                // the BEM quadrature error.
                let (sin_v, cos_v) = sonium_fastmath::sincos_f64x2(re1);

                // zg = re2 * (cos + i sin)
                let zg_re = vmulq_f64(re2, cos_v);
                let zg_im = vmulq_f64(re2, sin_v);

                // z1 = -1/r + i wavruim
                // zhh_base = zg * z1
                //   re = zg_re*(-inv_r) - zg_im*wavruim
                //   im = zg_re*wavruim + zg_im*(-inv_r)
                let neg_inv_r = vnegq_f64(inv_r_m);
                let zhh_base_re = vfmsq_f64(vmulq_f64(zg_re, neg_inv_r), zg_im, wavruim_v);
                let zhh_base_im = vfmaq_f64(vmulq_f64(zg_re, wavruim_v), zg_im, neg_inv_r);

                // r1h = dot(ur, n_y);  r2h = -dot(ur, source_normal)
                let r1h = vfmaq_f64(vfmaq_f64(vmulq_f64(ur_x, ny_x), ur_y, ny_y), ur_z, ny_z);
                let r2h_pos = vfmaq_f64(vfmaq_f64(vmulq_f64(ur_x, sn_x), ur_y, sn_y), ur_z, sn_z);
                let r2h = vnegq_f64(r2h_pos);

                // zhh = zhh_base * r1h  (real scalar)
                let zhh_re = vmulq_f64(zhh_base_re, r1h);
                let zhh_im = vmulq_f64(zhh_base_im, r1h);
                let zht_re = vmulq_f64(zhh_base_re, r2h);
                let zht_im = vmulq_f64(zhh_base_im, r2h);

                let rq = vmulq_f64(r1h, r2h);
                let inv_dq = vmulq_f64(inv_r_m, inv_r_m); // 1 / r² (masked)

                // ze_factor_re = (3/dq - k²) * rq + nx_dot_ny/dq
                let three_over_dq_minus_k2 = vfmsq_f64(k2_v, three_v, inv_dq); // k² - 3/dq
                // We have (k² - 3/dq); negate to get (3/dq - k²)
                let three_over_dq_minus_k2 = vnegq_f64(three_over_dq_minus_k2);
                let ze_factor_re =
                    vfmaq_f64(vmulq_f64(nx_dot_ny, inv_dq), three_over_dq_minus_k2, rq);

                // ze_factor_im = -wavruim/r * (3*rq + nx_dot_ny)
                let three_rq_plus_nxdotny = vfmaq_f64(nx_dot_ny, three_v, rq);
                let neg_wavruim_over_r = vmulq_f64(neg_wavruim_v, inv_r_m);
                let ze_factor_im = vmulq_f64(neg_wavruim_over_r, three_rq_plus_nxdotny);

                // ze = zg * (ze_factor_re + i ze_factor_im)
                let ze_re = vfmsq_f64(vmulq_f64(zg_re, ze_factor_re), zg_im, ze_factor_im);
                let ze_im = vfmaq_f64(vmulq_f64(zg_re, ze_factor_im), zg_im, ze_factor_re);

                g_re = vaddq_f64(g_re, zg_re);
                g_im = vaddq_f64(g_im, zg_im);
                h_re = vaddq_f64(h_re, zhh_re);
                h_im = vaddq_f64(h_im, zhh_im);
                ht_re = vaddq_f64(ht_re, zht_re);
                ht_im = vaddq_f64(ht_im, zht_im);
                e_re = vaddq_f64(e_re, ze_re);
                e_im = vaddq_f64(e_im, ze_im);
            }

            (
                (
                    Complex64::new(vgetq_lane_f64(g_re, 0), vgetq_lane_f64(g_im, 0)),
                    Complex64::new(vgetq_lane_f64(h_re, 0), vgetq_lane_f64(h_im, 0)),
                    Complex64::new(vgetq_lane_f64(ht_re, 0), vgetq_lane_f64(ht_im, 0)),
                    Complex64::new(vgetq_lane_f64(e_re, 0), vgetq_lane_f64(e_im, 0)),
                ),
                (
                    Complex64::new(vgetq_lane_f64(g_re, 1), vgetq_lane_f64(g_im, 1)),
                    Complex64::new(vgetq_lane_f64(h_re, 1), vgetq_lane_f64(h_im, 1)),
                    Complex64::new(vgetq_lane_f64(ht_re, 1), vgetq_lane_f64(ht_im, 1)),
                    Complex64::new(vgetq_lane_f64(e_re, 1), vgetq_lane_f64(e_im, 1)),
                ),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 4-pair SIMD
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{RegularPairInF64, RegularPairOutF64, TRI4_QUAD};
    use num_complex::Complex64;
    use std::arch::x86_64::*;
    use std::f64::consts::PI;

    /// Process pairs `(a, b, c, d)` in lane-parallel fashion. Lane 0 = a,
    /// lane 1 = b, lane 2 = c, lane 3 = d. Same algorithm as the NEON 2-lane
    /// kernel; doubles the throughput per call by going 4-wide.
    ///
    /// # Safety
    ///
    /// Caller must verify the host CPU has AVX2 + FMA. Use
    /// `std::is_x86_feature_detected!("avx2")` and
    /// `std::is_x86_feature_detected!("fma")` for runtime dispatch.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn regular_pair_x4(
        a: &RegularPairInF64,
        b: &RegularPairInF64,
        c: &RegularPairInF64,
        d: &RegularPairInF64,
        wave_number: f64,
        harmonic_factor: f64,
    ) -> (
        RegularPairOutF64,
        RegularPairOutF64,
        RegularPairOutF64,
        RegularPairOutF64,
    ) {
        // SAFETY: target_feature = "avx2,fma" guarantees every intrinsic
        // is sound under the caller's contract.
        unsafe {
            let wavruim = harmonic_factor * wave_number;
            let k2 = wave_number * wave_number;

            // Pack lane 0..3 — note _mm256_set_pd reverses argument order:
            // lane 0 = last arg, lane 3 = first arg.
            #[inline]
            unsafe fn pack4(a: f64, b: f64, c: f64, d: f64) -> __m256d {
                // SAFETY: AVX is available under the parent target_feature.
                unsafe { _mm256_set_pd(d, c, b, a) }
            }

            let sp_x = pack4(
                a.source_point[0],
                b.source_point[0],
                c.source_point[0],
                d.source_point[0],
            );
            let sp_y = pack4(
                a.source_point[1],
                b.source_point[1],
                c.source_point[1],
                d.source_point[1],
            );
            let sp_z = pack4(
                a.source_point[2],
                b.source_point[2],
                c.source_point[2],
                d.source_point[2],
            );
            let sn_x = pack4(
                a.source_normal[0],
                b.source_normal[0],
                c.source_normal[0],
                d.source_normal[0],
            );
            let sn_y = pack4(
                a.source_normal[1],
                b.source_normal[1],
                c.source_normal[1],
                d.source_normal[1],
            );
            let sn_z = pack4(
                a.source_normal[2],
                b.source_normal[2],
                c.source_normal[2],
                d.source_normal[2],
            );

            let v0_x = pack4(a.v0[0], b.v0[0], c.v0[0], d.v0[0]);
            let v0_y = pack4(a.v0[1], b.v0[1], c.v0[1], d.v0[1]);
            let v0_z = pack4(a.v0[2], b.v0[2], c.v0[2], d.v0[2]);
            let v1_x = pack4(a.v1[0], b.v1[0], c.v1[0], d.v1[0]);
            let v1_y = pack4(a.v1[1], b.v1[1], c.v1[1], d.v1[1]);
            let v1_z = pack4(a.v1[2], b.v1[2], c.v1[2], d.v1[2]);
            let v2_x = pack4(a.v2[0], b.v2[0], c.v2[0], d.v2[0]);
            let v2_y = pack4(a.v2[1], b.v2[1], c.v2[1], d.v2[1]);
            let v2_z = pack4(a.v2[2], b.v2[2], c.v2[2], d.v2[2]);

            // e1 = v1 - v0, e2 = v2 - v0
            let e1_x = _mm256_sub_pd(v1_x, v0_x);
            let e1_y = _mm256_sub_pd(v1_y, v0_y);
            let e1_z = _mm256_sub_pd(v1_z, v0_z);
            let e2_x = _mm256_sub_pd(v2_x, v0_x);
            let e2_y = _mm256_sub_pd(v2_y, v0_y);
            let e2_z = _mm256_sub_pd(v2_z, v0_z);

            // raw_n = cross(e1, e2)
            // fnmadd(a, b, c) = -a*b + c = c - a*b
            let nx = _mm256_fnmadd_pd(e1_z, e2_y, _mm256_mul_pd(e1_y, e2_z));
            let ny = _mm256_fnmadd_pd(e1_x, e2_z, _mm256_mul_pd(e1_z, e2_x));
            let nz = _mm256_fnmadd_pd(e1_y, e2_x, _mm256_mul_pd(e1_x, e2_y));

            // jac = |raw_n|; n_y = raw_n / jac with per-lane mask
            // jac² = nx² + ny² + nz² via FMA chain
            let jac_sq = _mm256_fmadd_pd(nz, nz, _mm256_fmadd_pd(ny, ny, _mm256_mul_pd(nx, nx)));
            let jac = _mm256_sqrt_pd(jac_sq);
            let eps = _mm256_set1_pd(1e-15);
            let one = _mm256_set1_pd(1.0);
            let zero = _mm256_setzero_pd();
            let jac_ok = _mm256_cmp_pd(jac, eps, _CMP_GT_OQ);
            let inv_jac = _mm256_div_pd(one, jac);
            let inv_jac_m = _mm256_blendv_pd(zero, inv_jac, jac_ok);
            let ny_x = _mm256_mul_pd(nx, inv_jac_m);
            let ny_y = _mm256_mul_pd(ny, inv_jac_m);
            let ny_z = _mm256_mul_pd(nz, inv_jac_m);

            // nx_dot_ny = dot(source_normal, n_y)
            let nx_dot_ny = _mm256_fmadd_pd(
                sn_z,
                ny_z,
                _mm256_fmadd_pd(sn_y, ny_y, _mm256_mul_pd(sn_x, ny_x)),
            );

            // 8 complex accumulators (re, im) × {G, H, H^T, E}
            let mut g_re = zero;
            let mut g_im = zero;
            let mut h_re = zero;
            let mut h_im = zero;
            let mut ht_re = zero;
            let mut ht_im = zero;
            let mut e_re = zero;
            let mut e_im = zero;

            let wavruim_v = _mm256_set1_pd(wavruim);
            let neg_wavruim_v = _mm256_xor_pd(wavruim_v, _mm256_set1_pd(-0.0));
            let k2_v = _mm256_set1_pd(k2);
            let three_v = _mm256_set1_pd(3.0);
            let one_over_4pi = _mm256_set1_pd(1.0 / (4.0 * PI));

            for &(xi, eta, w) in TRI4_QUAD.iter() {
                let n0 = 1.0 - xi - eta;
                let xi_v = _mm256_set1_pd(xi);
                let eta_v = _mm256_set1_pd(eta);
                let n0_v = _mm256_set1_pd(n0);
                let w_v = _mm256_set1_pd(w);

                // crd = n0 * v0 + xi * v1 + eta * v2
                let crd_x = _mm256_fmadd_pd(
                    v2_x,
                    eta_v,
                    _mm256_fmadd_pd(v1_x, xi_v, _mm256_mul_pd(v0_x, n0_v)),
                );
                let crd_y = _mm256_fmadd_pd(
                    v2_y,
                    eta_v,
                    _mm256_fmadd_pd(v1_y, xi_v, _mm256_mul_pd(v0_y, n0_v)),
                );
                let crd_z = _mm256_fmadd_pd(
                    v2_z,
                    eta_v,
                    _mm256_fmadd_pd(v1_z, xi_v, _mm256_mul_pd(v0_z, n0_v)),
                );

                let wga = _mm256_mul_pd(w_v, jac);
                let diff_x = _mm256_sub_pd(crd_x, sp_x);
                let diff_y = _mm256_sub_pd(crd_y, sp_y);
                let diff_z = _mm256_sub_pd(crd_z, sp_z);

                let r_sq = _mm256_fmadd_pd(
                    diff_z,
                    diff_z,
                    _mm256_fmadd_pd(diff_y, diff_y, _mm256_mul_pd(diff_x, diff_x)),
                );
                let r = _mm256_sqrt_pd(r_sq);

                // Per-lane "r > eps" mask (matches scalar `continue` semantics)
                let r_ok = _mm256_cmp_pd(r, eps, _CMP_GT_OQ);
                let inv_r = _mm256_div_pd(one, r);
                let inv_r_m = _mm256_blendv_pd(zero, inv_r, r_ok);

                let ur_x = _mm256_mul_pd(diff_x, inv_r_m);
                let ur_y = _mm256_mul_pd(diff_y, inv_r_m);
                let ur_z = _mm256_mul_pd(diff_z, inv_r_m);

                let re1 = _mm256_mul_pd(wavruim_v, r);
                let re2 = _mm256_mul_pd(_mm256_mul_pd(wga, one_over_4pi), inv_r_m);

                // Lane-parallel sin/cos via the sonium-fastmath AVX2 kernel.
                // Replaces 4 scalar libm calls per quadrature point with one
                // 4-lane SIMD evaluation.
                let (sin_v, cos_v) = sonium_fastmath::sincos_f64x4(re1);

                // zg = re2 * (cos + i sin)
                let zg_re = _mm256_mul_pd(re2, cos_v);
                let zg_im = _mm256_mul_pd(re2, sin_v);

                // zhh_base = zg * (-1/r + i wavruim)
                let neg_inv_r = _mm256_xor_pd(inv_r_m, _mm256_set1_pd(-0.0));
                let zhh_base_re =
                    _mm256_fnmadd_pd(zg_im, wavruim_v, _mm256_mul_pd(zg_re, neg_inv_r));
                let zhh_base_im =
                    _mm256_fmadd_pd(zg_im, neg_inv_r, _mm256_mul_pd(zg_re, wavruim_v));

                // r1h = dot(ur, n_y); r2h = -dot(ur, source_normal)
                let r1h = _mm256_fmadd_pd(
                    ur_z,
                    ny_z,
                    _mm256_fmadd_pd(ur_y, ny_y, _mm256_mul_pd(ur_x, ny_x)),
                );
                let r2h_pos = _mm256_fmadd_pd(
                    ur_z,
                    sn_z,
                    _mm256_fmadd_pd(ur_y, sn_y, _mm256_mul_pd(ur_x, sn_x)),
                );
                let r2h = _mm256_xor_pd(r2h_pos, _mm256_set1_pd(-0.0));

                let zhh_re = _mm256_mul_pd(zhh_base_re, r1h);
                let zhh_im = _mm256_mul_pd(zhh_base_im, r1h);
                let zht_re = _mm256_mul_pd(zhh_base_re, r2h);
                let zht_im = _mm256_mul_pd(zhh_base_im, r2h);

                let rq = _mm256_mul_pd(r1h, r2h);
                let inv_dq = _mm256_mul_pd(inv_r_m, inv_r_m);

                // ze_factor_re = (3/dq - k²) * rq + nx_dot_ny/dq
                // fnmadd(a,b,c) = c - a*b, so 3/dq - k² = fnmadd(k², 1, 3·inv_dq)
                // Easier: compute (k² - 3/dq) via fnmadd then negate.
                let k2_minus_3_over_dq = _mm256_fnmadd_pd(three_v, inv_dq, k2_v);
                let three_over_dq_minus_k2 =
                    _mm256_xor_pd(k2_minus_3_over_dq, _mm256_set1_pd(-0.0));
                let ze_factor_re =
                    _mm256_fmadd_pd(three_over_dq_minus_k2, rq, _mm256_mul_pd(nx_dot_ny, inv_dq));

                // ze_factor_im = -wavruim/r * (3*rq + nx_dot_ny)
                let three_rq_plus_nxdotny = _mm256_fmadd_pd(three_v, rq, nx_dot_ny);
                let neg_wavruim_over_r = _mm256_mul_pd(neg_wavruim_v, inv_r_m);
                let ze_factor_im = _mm256_mul_pd(neg_wavruim_over_r, three_rq_plus_nxdotny);

                // ze = zg * (ze_factor_re + i ze_factor_im)
                let ze_re =
                    _mm256_fnmadd_pd(zg_im, ze_factor_im, _mm256_mul_pd(zg_re, ze_factor_re));
                let ze_im =
                    _mm256_fmadd_pd(zg_im, ze_factor_re, _mm256_mul_pd(zg_re, ze_factor_im));

                g_re = _mm256_add_pd(g_re, zg_re);
                g_im = _mm256_add_pd(g_im, zg_im);
                h_re = _mm256_add_pd(h_re, zhh_re);
                h_im = _mm256_add_pd(h_im, zhh_im);
                ht_re = _mm256_add_pd(ht_re, zht_re);
                ht_im = _mm256_add_pd(ht_im, zht_im);
                e_re = _mm256_add_pd(e_re, ze_re);
                e_im = _mm256_add_pd(e_im, ze_im);
            }

            // Lane extract via _mm256_storeu_pd into a 4-element array.
            let extract = |re: __m256d, im: __m256d| -> [Complex64; 4] {
                let mut r_buf = [0.0_f64; 4];
                let mut i_buf = [0.0_f64; 4];
                // SAFETY: 4×f64 aligned-or-not store with matching length.
                _mm256_storeu_pd(r_buf.as_mut_ptr(), re);
                _mm256_storeu_pd(i_buf.as_mut_ptr(), im);
                [
                    Complex64::new(r_buf[0], i_buf[0]),
                    Complex64::new(r_buf[1], i_buf[1]),
                    Complex64::new(r_buf[2], i_buf[2]),
                    Complex64::new(r_buf[3], i_buf[3]),
                ]
            };
            let gs = extract(g_re, g_im);
            let hs = extract(h_re, h_im);
            let hts = extract(ht_re, ht_im);
            let es = extract(e_re, e_im);
            (
                (gs[0], hs[0], hts[0], es[0]),
                (gs[1], hs[1], hts[1], es[1]),
                (gs[2], hs[2], hts[2], es[2]),
                (gs[3], hs[3], hts[3], es[3]),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_pair(seed: u64) -> RegularPairInF64 {
        let frac = |salt: u64| -> f64 {
            let x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(salt);
            let x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            let x = x ^ (x >> 31);
            (x as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        let sp = [5.0 * frac(1), 5.0 * frac(2), 5.0 * frac(3)];
        let mut sn = [frac(4), frac(5), frac(6)];
        let snorm = norm3(sn).max(1e-9);
        sn = scale3(sn, 1.0 / snorm);
        let base = [
            5.0 * frac(7) + if frac(7) > 0.0 { -8.0 } else { 8.0 },
            5.0 * frac(8) + if frac(8) > 0.0 { -8.0 } else { 8.0 },
            5.0 * frac(9) + if frac(9) > 0.0 { -8.0 } else { 8.0 },
        ];
        let v0 = base;
        let v1 = [base[0] + 0.3, base[1] + 0.05 * frac(10), base[2]];
        let v2 = [base[0] + 0.05 * frac(11), base[1] + 0.3, base[2]];
        RegularPairInF64 {
            source_point: sp,
            source_normal: sn,
            v0,
            v1,
            v2,
        }
    }

    fn rel_err(a: Complex64, b: Complex64) -> f64 {
        let d = (a - b).norm();
        let denom = a.norm().max(1e-30);
        d / denom
    }

    #[test]
    fn batch_matches_scalar_on_fixture() {
        let pairs: Vec<RegularPairInF64> = (0..17).map(fixture_pair).collect();
        let k = 1.5;
        let hf = 1.0;
        let batch = regular_pair_batch(&pairs, k, hf);
        for (i, p) in pairs.iter().enumerate() {
            let s = regular_pair_scalar(p, k, hf);
            assert!(
                rel_err(batch[i].0, s.0) < 1e-12,
                "pair {i}: G diff {:.3e}",
                rel_err(batch[i].0, s.0)
            );
            assert!(rel_err(batch[i].1, s.1) < 1e-12, "pair {i}: H diff");
            assert!(rel_err(batch[i].2, s.2) < 1e-12, "pair {i}: H^T diff");
            assert!(rel_err(batch[i].3, s.3) < 1e-12, "pair {i}: E diff");
        }
    }

    #[test]
    fn batch_empty_is_empty() {
        let out = regular_pair_batch(&[], 1.0, 1.0);
        assert!(out.is_empty());
    }

    #[test]
    fn skipped_count_flags_touching_pairs() {
        // Well-separated pair: no quadrature point near the source.
        let far = fixture_pair(7);
        let (_, skipped) = regular_pair_batch_with_skipped_count(&[far], 1.5, 1.0);
        assert_eq!(skipped, 0);
        // Self-interaction: source point coincides bit-exactly with the
        // first quadrature point (the centroid). The naive
        // `(v0 + v1 + v2) / 3` differs from the kernel's own
        // `v0*n0 + xi*v1 + eta*v2` evaluation order by ~1 ulp (~1e-15 at
        // these coordinates), which can land just outside the absolute
        // cutoff — so the fixture replicates the kernel interpolation
        // exactly, giving `r == 0.0` by construction.
        let (xi, eta, _) = TRI4_QUAD[0];
        let n0 = 1.0 - xi - eta;
        let centroid = [
            far.v0[0] * n0 + xi * far.v1[0] + eta * far.v2[0],
            far.v0[1] * n0 + xi * far.v1[1] + eta * far.v2[1],
            far.v0[2] * n0 + xi * far.v1[2] + eta * far.v2[2],
        ];
        let touching = RegularPairInF64 {
            source_point: centroid,
            source_normal: far.source_normal,
            v0: far.v0,
            v1: far.v1,
            v2: far.v2,
        };
        let (out, skipped) = regular_pair_batch_with_skipped_count(&[touching], 1.5, 1.0);
        assert_eq!(out.len(), 1);
        assert!(skipped > 0, "touching pair must report skipped points");
    }
}
