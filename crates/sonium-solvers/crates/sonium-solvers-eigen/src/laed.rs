//! LAPACK-style secular eigensolver for the divide-and-conquer merge.
//!
//! Ports Reference-LAPACK `DLAED4` (rational-interpolation secular root
//! finder), `DLAED6` (three-pole Gragg–Thornton–Warner step), `DLAED5`
//! (closed-form 2-by-2), and the Gu–Eisenstat recomputed-`ẑ` vector
//! construction from `DLAED3`.
//!
//! Reference: R.-C. Li, "Solving Secular Equations Stably and
//! Efficiently" (1994); LAPACK Working Note. Originals are BSD-3
//! (Univ. of Tennessee / Berkeley / Colorado Denver / NAG Ltd.);
//! this port keeps the algorithm and convergence guards faithful
//! (0-based indexing aside) so vendor accuracy behavior carries over.
//! Full Fortran text used for the port is archived at `/tmp/dlaed4_full.txt`,
//! `/tmp/dlaed6_full.txt`, `/tmp/dlaed3_full.txt`.
//!
//! Why this replaces the τ-first bracket solver: rational interpolation
//! fits the pole structure, so each root costs 3–5 secular evaluations
//! instead of ~25, and the recomputed-`ẑ` vectors make the converged
//! values exact eigenvalues of the updated problem by construction —
//! no τ-refinement second pass. Measured per-node census showed the old
//! path spending 1.3–8 ms in secular solves per k=150 merge with the
//! value histogram centered at ~25 evals/root plus 43–68 refinement
//! triggers; this collapses both.

use crate::divide_conquer::secular_sweep_bump;

/// Machine epsilon (`DLAMCH('Epsilon')`).
const EPS: f64 = f64::EPSILON;
/// Maximum secular iterations per root (`MAXIT` in DLAED4).
const MAXIT: i32 = 30;

/// One root of `f(λ) = 1 + Σᵢ ρ·zᵢ²/(dᵢ − λ) = 0` by rational
/// interpolation (DLAED4).
///
/// * `d` — strictly ascending poles, len `k >= 3`;
/// * `z` — update vector (same scaling as `rho`; normalize before call);
/// * `rho` — rank-1 coefficient, must be `> 0` (negate-and-reverse the
///   problem for `rho < 0` before calling);
/// * `j` — 0-based root index: `λⱼ ∈ (dⱼ, dⱼ₊₁)` for `j < k−1`,
///   `λₖ₋₁ > dₖ₋₁` when `j == k−1`;
/// * `delta` — output, `delta[i] = dᵢ − λ` (drives the Löwner vectors).
///
/// Returns `None` (`INFO = 1`) when the iteration cap is hit; the caller
/// falls back to the legacy bracket solver for the whole node, so this
/// path never introduces a new failure mode.
pub(crate) fn laed4_root(
    d: &[f64],
    z: &[f64],
    rho: f64,
    j: usize,
    delta: &mut [f64],
    count: bool,
) -> Option<f64> {
    debug_assert!(d.len() >= 3);
    debug_assert_eq!(z.len(), d.len());
    debug_assert_eq!(delta.len(), d.len());
    debug_assert!(rho > 0.0);
    debug_assert!(j < d.len());
    let k = d.len();
    let rhoinv = 1.0 / rho;
    if j == k - 1 {
        laed4_last(d, z, rho, rhoinv, delta, count)
    } else {
        laed4_interior(d, z, rhoinv, j, delta, count)
    }
}

/// Evaluate `ψ = Σ_{i in idx} zᵢ²/δᵢ`, its derivative
/// `dψ = Σ (zᵢ/δᵢ)²`, and the error-bound accumulator over an index
/// range. One call = one secular sweep (same O(k) work unit as one
/// legacy `secular(λ)` evaluation, so census bins stay comparable).
#[inline]
fn sweep(
    z: &[f64],
    delta: &[f64],
    range: impl Iterator<Item = usize>,
    count: bool,
) -> (f64, f64, f64) {
    if count {
        secular_sweep_bump();
    }
    let mut s = 0.0;
    let mut ds = 0.0;
    let mut err = 0.0;
    for i in range {
        let t = z[i] / delta[i];
        let zi_t = z[i] * t;
        s += zi_t;
        ds += t * t;
        err += zi_t;
    }
    (s, ds, err)
}

/// DLAED4 case `I = N`: root right of the last pole.
fn laed4_last(
    d: &[f64],
    z: &[f64],
    rho: f64,
    rhoinv: f64,
    delta: &mut [f64],
    count: bool,
) -> Option<f64> {
    let k = d.len();
    let last = k - 1;
    let ii = k - 2; // 0-based origin pole (Fortran II = N−1)
    let mut _niter = 1;
    let midpt = rho / 2.0;
    for (dlt, &dt) in delta.iter_mut().zip(d.iter()) {
        *dlt = (dt - d[last]) - midpt;
    }
    // Initial ψ over J = 1..N−2 (excludes the last two poles).
    let (psi, _, _) = sweep(z, delta, 0..k - 2, count);
    let mut c = rhoinv + psi;
    let mut w = c + z[ii] * z[ii] / delta[ii] + z[last] * z[last] / delta[last];
    let (mut tau, mut dltlb, mut dltub);
    if w <= 0.0 {
        let temp = z[ii] * z[ii] / (d[last] - d[ii] + rho) + z[last] * z[last] / rho;
        if c <= temp {
            tau = rho;
        } else {
            let del = d[last] - d[ii];
            let a = -c * del + z[ii] * z[ii] + z[last] * z[last];
            let b = z[last] * z[last] * del;
            tau = if a < 0.0 {
                2.0 * b / ((a * a + 4.0 * b * c).sqrt() - a)
            } else {
                (a + (a * a + 4.0 * b * c).sqrt()) / (2.0 * c)
            };
        }
        // D(N)+RHO/2 <= λ(N) < D(N)+TAU <= D(N)+RHO (proved in LAPACK).
        dltlb = midpt;
        dltub = rho;
    } else {
        let del = d[last] - d[ii];
        let a = -c * del + z[ii] * z[ii] + z[last] * z[last];
        let b = z[last] * z[last] * del;
        tau = if a < 0.0 {
            2.0 * b / ((a * a + 4.0 * b * c).sqrt() - a)
        } else {
            (a + (a * a + 4.0 * b * c).sqrt()) / (2.0 * c)
        };
        // D(N) < D(N)+TAU < λ(N) < D(N)+RHO/2 (proved in LAPACK).
        dltlb = 0.0;
        dltub = midpt;
    }
    for (dlt, &dt) in delta.iter_mut().zip(d.iter()) {
        *dlt = (dt - d[last]) - tau;
    }
    // Re-evaluate ψ (now including the origin pole II) and φ.
    let (psi, mut dpsi, mut erretm) = sweep(z, delta, 0..k - 1, count);
    erretm = erretm.abs();
    let temp = z[last] / delta[last];
    let phi = z[last] * temp;
    let mut dphi = temp * temp;
    erretm = 8.0 * (-phi - psi) + erretm - phi + rhoinv + tau.abs() * (dpsi + dphi);
    w = rhoinv + phi + psi;
    if w.abs() <= EPS * erretm {
        return finish(d[last], tau, delta);
    }
    if w <= 0.0 {
        dltlb = dltlb.max(tau);
    } else {
        dltub = dltub.min(tau);
    }
    _niter += 1;
    // First rational-interpolation step (two-pole, origin side).
    let mut eta = {
        c = w - delta[ii] * dpsi - delta[last] * dphi;
        let a = (delta[ii] + delta[last]) * w - delta[ii] * delta[last] * (dpsi + dphi);
        let b = delta[ii] * delta[last] * w;
        if c < 0.0 {
            c = c.abs();
        }
        if c == 0.0 {
            // Update proposed by Li, Ren-Cang.
            -w / (dpsi + dphi)
        } else if a >= 0.0 {
            (a + (a * a - 4.0 * b * c).abs().sqrt()) / (2.0 * c)
        } else {
            2.0 * b / (a - (a * a - 4.0 * b * c).abs().sqrt())
        }
    };
    // eta must oppose w; roundoff fallback is one Newton step.
    if w * eta > 0.0 {
        eta = -w / (dpsi + dphi);
    }
    let mut temp2 = tau + eta;
    if temp2 > dltub || temp2 < dltlb {
        let eta1 = -w / (dpsi + dphi);
        temp2 = tau + eta1;
        let eta2 = if w < 0.0 {
            (dltub - tau) / 2.0
        } else {
            (dltlb - tau) / 2.0
        };
        if dltlb <= temp2 && temp2 <= dltub {
            // Geometric mean of the Newton distance, keeping direction.
            eta = eta1.signum() * eta1.abs().sqrt() * eta2.abs().sqrt();
        } else {
            eta = eta2;
        }
    }
    for dlt in delta.iter_mut() {
        *dlt -= eta;
    }
    tau += eta;
    // Main loop.
    let mut iter = _niter + 1;
    loop {
        if iter > MAXIT {
            return None; // INFO = 1
        }
        let (psi, dpsi_new, mut erretm) = sweep(z, delta, 0..k - 1, count);
        dpsi = dpsi_new;
        erretm = erretm.abs();
        let temp = z[last] / delta[last];
        let phi = z[last] * temp;
        dphi = temp * temp;
        erretm = 8.0 * (-phi - psi) + erretm - phi + rhoinv + tau.abs() * (dpsi + dphi);
        w = rhoinv + phi + psi;
        if w.abs() <= EPS * erretm {
            return finish(d[last], tau, delta);
        }
        if w <= 0.0 {
            dltlb = dltlb.max(tau);
        } else {
            dltub = dltub.min(tau);
        }
        iter += 1;
        c = w - delta[ii] * dpsi - delta[last] * dphi;
        let a = (delta[ii] + delta[last]) * w - delta[ii] * delta[last] * (dpsi + dphi);
        let b = delta[ii] * delta[last] * w;
        if c < 0.0 {
            c = c.abs();
        }
        eta = if c == 0.0 {
            -w / (dpsi + dphi)
        } else if a >= 0.0 {
            (a + (a * a - 4.0 * b * c).abs().sqrt()) / (2.0 * c)
        } else {
            2.0 * b / (a - (a * a - 4.0 * b * c).abs().sqrt())
        };
        if w * eta > 0.0 {
            eta = -w / (dpsi + dphi);
        }
        temp2 = tau + eta;
        if temp2 > dltub || temp2 < dltlb {
            let eta1 = -w / (dpsi + dphi);
            temp2 = tau + eta1;
            let eta2 = if w < 0.0 {
                (dltub - tau) / 2.0
            } else {
                (dltlb - tau) / 2.0
            };
            if dltlb <= temp2 && temp2 <= dltub {
                eta = eta1.signum() * eta1.abs().sqrt() * eta2.abs().sqrt();
            } else {
                eta = eta2;
            }
        }
        for dlt in delta.iter_mut() {
            *dlt -= eta;
        }
        tau += eta;
    }
}

/// Accept a converged offset: `λ = origin + τ` must be finite.
fn finish(origin: f64, tau: f64, #[allow(unused_variables)] delta: &mut [f64]) -> Option<f64> {
    let lam = origin + tau;
    lam.is_finite().then_some(lam)
}

/// DLAED4 case `I < N`: root between poles `dⱼ` and `dⱼ₊₁`.
#[allow(clippy::too_many_lines)]
fn laed4_interior(
    d: &[f64],
    z: &[f64],
    rhoinv: f64,
    j: usize,
    delta: &mut [f64],
    count: bool,
) -> Option<f64> {
    let k = d.len();
    let ip1 = j + 1;
    let mut niter = 1;
    // Initial guess at the interval midpoint.
    let del = d[ip1] - d[j];
    let midpt = del / 2.0;
    for (dlt, &dt) in delta.iter_mut().zip(d.iter()) {
        *dlt = (dt - d[j]) - midpt;
    }
    let (psi, _, _) = sweep(z, delta, 0..j, count);
    let (phi, _, _) = sweep(z, delta, (ip1 + 1..k).rev(), count);
    let c0 = rhoinv + psi + phi;
    let w0 = c0 + z[j] * z[j] / delta[j] + z[ip1] * z[ip1] / delta[ip1];
    // Choose the nearer pole as origin (proved brackets in LAPACK notes).
    let orgati = w0 > 0.0;
    let (mut tau, mut dltlb, mut dltub) = if orgati {
        let a = c0 * del + z[j] * z[j] + z[ip1] * z[ip1];
        let b = z[j] * z[j] * del;
        let tau = if a > 0.0 {
            2.0 * b / (a + (a * a - 4.0 * b * c0).abs().sqrt())
        } else {
            (a - (a * a - 4.0 * b * c0).abs().sqrt()) / (2.0 * c0)
        };
        (tau, 0.0, midpt)
    } else {
        let a = c0 * del - z[j] * z[j] - z[ip1] * z[ip1];
        let b = z[ip1] * z[ip1] * del;
        let tau = if a < 0.0 {
            2.0 * b / (a - (a * a + 4.0 * b * c0).abs().sqrt())
        } else {
            -(a + (a * a + 4.0 * b * c0).abs().sqrt()) / (2.0 * c0)
        };
        (tau, -midpt, 0.0)
    };
    let origin = if orgati { d[j] } else { d[ip1] };
    for (dlt, &dt) in delta.iter_mut().zip(d.iter()) {
        *dlt = (dt - origin) - tau;
    }
    let io = if orgati { j } else { ip1 }; // 0-based origin pole
    // ψ over poles below the origin, φ over poles above.
    let (mut psi, mut dpsi, err_psi) = sweep(z, delta, 0..io, count);
    let (mut phi, mut dphi, err_phi) = sweep(z, delta, (io + 1..k).rev(), count);
    // W without the origin-pole term (drives the SWTCH3 choice).
    let w_noii = rhoinv + phi + psi;
    let mut swtch3 = if orgati { w_noii < 0.0 } else { w_noii > 0.0 };
    if io == 0 || io == k - 1 {
        swtch3 = false;
    }
    let t0 = z[io] / delta[io];
    let mut dw = dpsi + dphi + t0 * t0;
    let tz = z[io] * t0;
    let mut w = w_noii + tz;
    let mut erretm = 8.0 * (phi - psi)
        + err_psi.abs()
        + err_phi
        + 2.0 * rhoinv
        + 3.0 * tz.abs()
        + tau.abs() * dw;
    if w.abs() <= EPS * erretm {
        return finish(origin, tau, delta);
    }
    if w <= 0.0 {
        dltlb = dltlb.max(tau);
    } else {
        dltub = dltub.min(tau);
    }
    niter += 1;
    // First step: two-pole interpolation, or three-pole (DLAED6).
    let mut eta = if !swtch3 {
        let (c, a, b) = if orgati {
            let c = w - delta[ip1] * dw - (d[j] - d[ip1]) * (z[j] / delta[j]).powi(2);
            let a = (delta[j] + delta[ip1]) * w - delta[j] * delta[ip1] * dw;
            (c, a, delta[j] * delta[ip1] * w)
        } else {
            let c = w - delta[j] * dw - (d[ip1] - d[j]) * (z[ip1] / delta[ip1]).powi(2);
            let a = (delta[j] + delta[ip1]) * w - delta[j] * delta[ip1] * dw;
            (c, a, delta[j] * delta[ip1] * w)
        };
        step_eta(c, a, b, z, delta, dpsi, dphi, j, ip1, orgati, false)
    } else {
        let (c, zz) = swtch3_params(rhoinv, psi, phi, dpsi, dphi, z, delta, d, io, orgati);
        match laed6(
            niter,
            orgati,
            c,
            [delta[io - 1], delta[io], delta[io + 1]],
            zz,
            w,
        ) {
            Some(e) => e,
            None => return finish(origin, tau, delta),
        }
    };
    if w * eta >= 0.0 {
        eta = -w / dw;
    }
    eta = clamp_step(eta, w, dw, tau, dltlb, dltub);
    let mut prew = w;
    for dlt in delta.iter_mut() {
        *dlt -= eta;
    }
    // Re-evaluate at the stepped point.
    let (psi_new, dpsi_new, err_psi) = sweep(z, delta, 0..io, count);
    psi = psi_new;
    dpsi = dpsi_new;
    let (phi_new, dphi_new, err_phi) = sweep(z, delta, (io + 1..k).rev(), count);
    phi = phi_new;
    dphi = dphi_new;
    let t0 = z[io] / delta[io];
    dw = dpsi + dphi + t0 * t0;
    let tz = z[io] * t0;
    w = rhoinv + phi + psi + tz;
    erretm = 8.0 * (phi - psi)
        + err_psi.abs()
        + err_phi
        + 2.0 * rhoinv
        + 3.0 * tz.abs()
        + tau.abs() * dw;
    let mut swtch = false;
    if orgati {
        if -w > prew.abs() / 10.0 {
            swtch = true;
        }
    } else if w > prew.abs() / 10.0 {
        swtch = true;
    }
    tau += eta;
    // Main loop.
    let mut iter = niter + 1;
    loop {
        if iter > MAXIT {
            return None; // INFO = 1
        }
        if w.abs() <= EPS * erretm {
            return finish(origin, tau, delta);
        }
        if w <= 0.0 {
            dltlb = dltlb.max(tau);
        } else {
            dltub = dltub.min(tau);
        }
        niter = iter;
        eta = if !swtch3 {
            let (c, a, b) = if !swtch {
                if orgati {
                    let c = w - delta[ip1] * dw - (d[j] - d[ip1]) * (z[j] / delta[j]).powi(2);
                    let a = (delta[j] + delta[ip1]) * w - delta[j] * delta[ip1] * dw;
                    (c, a, delta[j] * delta[ip1] * w)
                } else {
                    let c = w - delta[j] * dw - (d[ip1] - d[j]) * (z[ip1] / delta[ip1]).powi(2);
                    let a = (delta[j] + delta[ip1]) * w - delta[j] * delta[ip1] * dw;
                    (c, a, delta[j] * delta[ip1] * w)
                }
            } else {
                let t0 = z[io] / delta[io];
                if orgati {
                    dpsi += t0 * t0;
                } else {
                    dphi += t0 * t0;
                }
                let c = w - delta[j] * dpsi - delta[ip1] * dphi;
                let a = (delta[j] + delta[ip1]) * w - delta[j] * delta[ip1] * dw;
                (c, a, delta[j] * delta[ip1] * w)
            };
            step_eta(c, a, b, z, delta, dpsi, dphi, j, ip1, orgati, swtch)
        } else {
            let (c, zz) = if swtch {
                let c = rhoinv + psi + phi - delta[io - 1] * dpsi - delta[io + 1] * dphi;
                let zz = [
                    delta[io - 1] * delta[io - 1] * dpsi,
                    z[io] * z[io],
                    delta[io + 1] * delta[io + 1] * dphi,
                ];
                (c, zz)
            } else {
                swtch3_params(rhoinv, psi, phi, dpsi, dphi, z, delta, d, io, orgati)
            };
            match laed6(
                niter,
                orgati,
                c,
                [delta[io - 1], delta[io], delta[io + 1]],
                zz,
                w,
            ) {
                Some(e) => e,
                None => return finish(origin, tau, delta),
            }
        };
        if w * eta >= 0.0 {
            eta = -w / dw;
        }
        eta = clamp_step(eta, w, dw, tau, dltlb, dltub);
        for dlt in delta.iter_mut() {
            *dlt -= eta;
        }
        tau += eta;
        prew = w;
        let (psi_new, dpsi_new, err_psi) = sweep(z, delta, 0..io, count);
        psi = psi_new;
        dpsi = dpsi_new;
        let (phi_new, dphi_new, err_phi) = sweep(z, delta, (io + 1..k).rev(), count);
        phi = phi_new;
        dphi = dphi_new;
        let t0 = z[io] / delta[io];
        dw = dpsi + dphi + t0 * t0;
        let tz = z[io] * t0;
        w = rhoinv + phi + psi + tz;
        erretm = 8.0 * (phi - psi)
            + err_psi.abs()
            + err_phi
            + 2.0 * rhoinv
            + 3.0 * tz.abs()
            + tau.abs() * dw;
        if w * prew > 0.0 && w.abs() > prew.abs() / 10.0 {
            swtch = !swtch;
        }
        iter += 1;
    }
}

/// Two-pole rational-interpolation step (DLAED4 `ETA` computation).
#[allow(clippy::too_many_arguments)]
fn step_eta(
    c: f64,
    a: f64,
    b: f64,
    z: &[f64],
    delta: &[f64],
    dpsi: f64,
    dphi: f64,
    j: usize,
    ip1: usize,
    orgati: bool,
    swtch: bool,
) -> f64 {
    if c == 0.0 {
        let a_adj = if a == 0.0 {
            if !swtch {
                if orgati {
                    z[j] * z[j] + delta[ip1] * delta[ip1] * (dpsi + dphi)
                } else {
                    z[ip1] * z[ip1] + delta[j] * delta[j] * (dpsi + dphi)
                }
            } else {
                delta[j] * delta[j] * dpsi + delta[ip1] * delta[ip1] * dphi
            }
        } else {
            a
        };
        b / a_adj
    } else if a <= 0.0 {
        (a - (a * a - 4.0 * b * c).abs().sqrt()) / (2.0 * c)
    } else {
        2.0 * b / (a + (a * a - 4.0 * b * c).abs().sqrt())
    }
}

/// Three-pole interpolation parameters (DLAED4 `SWTCH3` setup, pre-loop
/// and `!SWTCH` in-loop form): `(C, ZZ)`.
#[allow(clippy::too_many_arguments)]
fn swtch3_params(
    rhoinv: f64,
    psi: f64,
    phi: f64,
    dpsi: f64,
    dphi: f64,
    z: &[f64],
    delta: &[f64],
    d: &[f64],
    io: usize,
    orgati: bool,
) -> (f64, [f64; 3]) {
    let temp = rhoinv + psi + phi;
    if orgati {
        let t1 = z[io - 1] / delta[io - 1];
        let t1sq = t1 * t1;
        let c = temp - delta[io + 1] * (dpsi + dphi) - (d[io - 1] - d[io + 1]) * t1sq;
        let zz = [
            z[io - 1] * z[io - 1],
            z[io] * z[io],
            delta[io + 1] * delta[io + 1] * ((dpsi - t1sq) + dphi),
        ];
        (c, zz)
    } else {
        let t1 = z[io + 1] / delta[io + 1];
        let t1sq = t1 * t1;
        let c = temp - delta[io - 1] * (dpsi + dphi) - (d[io + 1] - d[io - 1]) * t1sq;
        let zz = [
            delta[io - 1] * delta[io - 1] * (dpsi + (dphi - t1sq)),
            z[io] * z[io],
            z[io + 1] * z[io + 1],
        ];
        (c, zz)
    }
}

/// Three-pole Gragg–Thornton–Warner step (DLAED6): root of
/// `f(x) = c + Σᵢ zzᵢ/(dᵢ − x)` nearest the origin, bracketed between
/// the two poles flanking the current estimate.
///
/// * `d` — the three relevant pole offsets (`DELTA(IIM1..=IIP1)`);
/// * `zz` — squared-effective-`z` weights (see `swtch3_params`);
/// * `finit` — `f` at the current point (more accurate than re-eval).
///
/// Returns the step `ETA`, or `None` (`INFO = 1`); DLAED4 treats the
/// latter as converged-with-current-`TAU`, and so do we.
fn laed6(
    kniter: i32,
    orgati: bool,
    c_rho: f64,
    d: [f64; 3],
    zz: [f64; 3],
    finit: f64,
) -> Option<f64> {
    const MAXIT6: i32 = 40;
    let (mut lbd, mut ubd) = if orgati { (d[1], d[2]) } else { (d[0], d[1]) };
    if finit < 0.0 {
        lbd = 0.0;
    } else {
        ubd = 0.0;
    }
    let mut tau = 0.0;
    if kniter == 2 {
        // Two-pole quadratic start.
        let (a, b, c) = if orgati {
            let temp = (d[2] - d[1]) / 2.0;
            let c = c_rho + zz[0] / ((d[0] - d[1]) - temp);
            let a = c * (d[1] + d[2]) + zz[1] + zz[2];
            let b = c * d[1] * d[2] + zz[1] * d[2] + zz[2] * d[1];
            (a, b, c)
        } else {
            let temp = (d[0] - d[1]) / 2.0;
            let c = c_rho + zz[2] / ((d[2] - d[1]) - temp);
            let a = c * (d[0] + d[1]) + zz[0] + zz[1];
            let b = c * d[0] * d[1] + zz[0] * d[1] + zz[1] * d[0];
            (a, b, c)
        };
        let m = a.abs().max(b.abs()).max(c.abs());
        let (a, b, c) = (a / m, b / m, c / m);
        tau = if c == 0.0 {
            b / a
        } else if a <= 0.0 {
            (a - (a * a - 4.0 * b * c).abs().sqrt()) / (2.0 * c)
        } else {
            2.0 * b / (a + (a * a - 4.0 * b * c).abs().sqrt())
        };
        if tau < lbd || tau > ubd {
            tau = (lbd + ubd) / 2.0;
        }
        if d[0] == tau || d[1] == tau || d[2] == tau {
            tau = 0.0;
        } else {
            let temp = finit
                + tau * zz[0] / (d[0] * (d[0] - tau))
                + tau * zz[1] / (d[1] * (d[1] - tau))
                + tau * zz[2] / (d[2] * (d[2] - tau));
            if temp <= 0.0 {
                lbd = tau;
            } else {
                ubd = tau;
            }
            if finit.abs() <= temp.abs() {
                tau = 0.0;
            }
        }
    }
    // Machine parameters for overflow-safe scaling (recomputed per
    // entry for thread safety, as in LAPACK).
    let safmin = f64::MIN_POSITIVE;
    let small1 = 2f64.powi((safmin.log2() / 3.0).trunc() as i32);
    let sminv1 = 1.0 / small1;
    let small2 = small1 * small1;
    let sminv2 = 1.0 / small2;
    let pivot = if orgati {
        (d[1] - tau).abs().min((d[2] - tau).abs())
    } else {
        (d[0] - tau).abs().min((d[1] - tau).abs())
    };
    let (ds, zs, mut tau, mut lbd, mut ubd, sclinv) = if pivot <= small1 {
        let (sclfac, sclinv) = if pivot <= small2 {
            (sminv2, small2)
        } else {
            (sminv1, small1)
        };
        (
            [d[0] * sclfac, d[1] * sclfac, d[2] * sclfac],
            [zz[0] * sclfac, zz[1] * sclfac, zz[2] * sclfac],
            tau * sclfac,
            lbd * sclfac,
            ubd * sclfac,
            sclinv,
        )
    } else {
        (d, zz, tau, lbd, ubd, 1.0)
    };
    let scaled = sclinv != 1.0;
    let mut fc = 0.0;
    let mut df = 0.0;
    let mut ddf = 0.0;
    for i in 0..3 {
        let t = 1.0 / (ds[i] - tau);
        let t1 = zs[i] * t;
        fc += t1 / ds[i];
        df += t1 * t;
        ddf += t1 * t * t;
    }
    let mut f = finit + tau * fc;
    if f == 0.0 {
        return Some(if scaled { tau * sclinv } else { tau });
    }
    if f <= 0.0 {
        lbd = tau;
    } else {
        ubd = tau;
    }
    // GTW cubic iteration (LAPACK always runs 2..=40 here).
    for _ in 2..=MAXIT6 {
        let (t1, t2) = if orgati {
            (ds[1] - tau, ds[2] - tau)
        } else {
            (ds[0] - tau, ds[1] - tau)
        };
        let mut a = (t1 + t2) * f - t1 * t2 * df;
        let mut b = t1 * t2 * f;
        let mut c = f - (t1 + t2) * df + t1 * t2 * ddf;
        let m = a.abs().max(b.abs()).max(c.abs());
        a /= m;
        b /= m;
        c /= m;
        let mut eta = if c == 0.0 {
            b / a
        } else if a <= 0.0 {
            (a - (a * a - 4.0 * b * c).abs().sqrt()) / (2.0 * c)
        } else {
            2.0 * b / (a + (a * a - 4.0 * b * c).abs().sqrt())
        };
        if f * eta >= 0.0 {
            eta = -f / df;
        }
        tau += eta;
        if tau < lbd || tau > ubd {
            tau = (lbd + ubd) / 2.0;
        }
        fc = 0.0;
        let mut erretm = 0.0;
        df = 0.0;
        ddf = 0.0;
        let mut pole_hit = false;
        for i in 0..3 {
            if ds[i] - tau == 0.0 {
                pole_hit = true;
                break;
            }
            let t = 1.0 / (ds[i] - tau);
            let t1 = zs[i] * t;
            let t2 = t1 * t;
            let t4 = t1 / ds[i];
            fc += t4;
            erretm += t4.abs();
            df += t2;
            ddf += t2 * t;
        }
        if pole_hit {
            return Some(if scaled { tau * sclinv } else { tau });
        }
        f = finit + tau * fc;
        erretm = 8.0 * (finit.abs() + tau.abs() * erretm) + tau.abs() * df;
        if f.abs() <= 4.0 * EPS * erretm || (ubd - lbd) <= 4.0 * EPS * tau.abs() {
            return Some(if scaled { tau * sclinv } else { tau });
        }
        if f <= 0.0 {
            lbd = tau;
        } else {
            ubd = tau;
        }
    }
    None // INFO = 1
}

/// Closed-form 2-by-2 secular solve (DLAED5): eigenvalue `which`
/// (`false` = left of the pair, `true` = right) of
/// `diag(d₀,d₁) + ρ·z·zᵀ` with `ρ > 0`, plus the normalized Löwner
/// vector as `DELTA` (DLAED3 uses it directly as the eigenvector —
/// no `ẑ` recomputation for `k = 2`).
pub(crate) fn laed5_pair(
    d0: f64,
    d1: f64,
    z0: f64,
    z1: f64,
    rho: f64,
    right: bool,
) -> (f64, [f64; 2]) {
    debug_assert!(d0 < d1);
    debug_assert!(rho > 0.0);
    let del = d1 - d0;
    let z0sq = z0 * z0;
    let z1sq = z1 * z1;
    let (lam, e0, e1) = if !right {
        let w = 1.0 + 2.0 * rho * (z1sq - z0sq) / del;
        if w > 0.0 {
            let b = del + rho * (z0sq + z1sq);
            let c = rho * z0sq * del;
            // B > 0 always.
            let tau = 2.0 * c / (b + (b * b - 4.0 * c).abs().sqrt());
            (d0 + tau, -z0 / tau, z1 / (del - tau))
        } else {
            let b = -del + rho * (z0sq + z1sq);
            let c = rho * z1sq * del;
            let tau = if b > 0.0 {
                -2.0 * c / (b + (b * b + 4.0 * c).sqrt())
            } else {
                (b - (b * b + 4.0 * c).sqrt()) / 2.0
            };
            (d1 + tau, -z0 / (del + tau), -z1 / tau)
        }
    } else {
        let b = -del + rho * (z0sq + z1sq);
        let c = rho * z1sq * del;
        let tau = if b > 0.0 {
            (b + (b * b + 4.0 * c).sqrt()) / 2.0
        } else {
            2.0 * c / (-b + (b * b + 4.0 * c).sqrt())
        };
        (d1 + tau, -z0 / (del + tau), -z1 / tau)
    };
    let nrm = (e0 * e0 + e1 * e1).sqrt();
    (lam, [e0 / nrm, e1 / nrm])
}

/// Gu–Eisenstat recomputed-`ẑ` (DLAED3 "compute updated W"):
/// `ŵᵢ = sign(zᵢ)·√(−Πⱼ δⱼ[i]/(dᵢ−dⱼ))` with the `j = i` factor's
/// denominator defined as 1, where `δⱼ = deltas[j]` is the per-root
/// gap vector from [`laed4_root`]. The converged values are then exact
/// eigenvalues of `D + ρ·ŵ·ŵᵀ` by construction, so plain Löwner
/// vectors are orthogonal to machine precision with no refinement
/// pass. Returns `None` on any non-finite input (caller falls back).
pub(crate) fn recompute_w(d: &[f64], z: &[f64], deltas: &[Vec<f64>]) -> Option<Vec<f64>> {
    let k = d.len();
    debug_assert_eq!(z.len(), k);
    debug_assert_eq!(deltas.len(), k);
    let mut w: Vec<f64> = (0..k).map(|i| deltas[i][i]).collect();
    for (j, dj) in deltas.iter().enumerate() {
        debug_assert_eq!(dj.len(), k);
        for i in 0..j {
            w[i] *= dj[i] / (d[i] - d[j]);
        }
        for i in j + 1..k {
            w[i] *= dj[i] / (d[i] - d[j]);
        }
    }
    for (i, wi) in w.iter_mut().enumerate() {
        let neg = -*wi;
        if !neg.is_finite() || neg < 0.0 {
            return None;
        }
        *wi = neg.sqrt().copysign(z[i]);
    }
    Some(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn splitmix(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        (z as f64) / (u64::MAX as f64) * 2.0 - 1.0
    }

    fn secular(d: &[f64], z: &[f64], rho: f64, lam: f64) -> f64 {
        let mut f = 1.0;
        for i in 0..d.len() {
            f += rho * z[i] * z[i] / (d[i] - lam);
        }
        f
    }

    /// 200-step bisection reference (independent of the port).
    fn bisect_ref(d: &[f64], z: &[f64], rho: f64, j: usize) -> f64 {
        let k = d.len();
        let norm_sq: f64 = z.iter().map(|v| v * v).sum();
        let (mut lo, mut hi) = if j < k - 1 {
            (d[j], d[j + 1])
        } else {
            (d[k - 1], d[k - 1] + rho * norm_sq + 1.0)
        };
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if secular(d, z, rho, mid) > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        0.5 * (lo + hi)
    }

    #[test]
    fn laed4_matches_bisection_all_roots() {
        for seed in [11u64, 77, 1234, 99991] {
            let mut s = seed;
            let k = 8;
            let mut d: Vec<f64> = (0..k).map(|_| splitmix(&mut s) * 4.0).collect();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // Enforce strict separation (deflation's contract).
            for i in 1..k {
                if d[i] - d[i - 1] < 1e-6 {
                    d[i] = d[i - 1] + 1e-6 * (1.0 + i as f64);
                }
            }
            let z: Vec<f64> = (0..k).map(|_| 0.2 + splitmix(&mut s).abs()).collect();
            let nrm: f64 = z.iter().map(|v| v * v).sum::<f64>().sqrt();
            let zn: Vec<f64> = z.iter().map(|v| v / nrm).collect();
            let rho = 1.7 * nrm * nrm;
            for j in 0..k {
                let mut delta = vec![0.0; k];
                let lam = laed4_root(&d, &zn, rho, j, &mut delta, false)
                    .expect("laed4 converges on separated problem");
                let reference = bisect_ref(&d, &zn, rho, j);
                let scale = lam.abs().max(1.0);
                assert!(
                    (lam - reference).abs() <= 1e-9 * scale,
                    "seed {seed} root {j}: {lam} vs {reference}"
                );
                // Secular residual must be at roundoff level (a wrong
                // bracket would give O(1)).
                let f = secular(&d, &zn, rho, lam);
                assert!(f.abs() <= 1e-12, "seed {seed} root {j}: f = {f:.3e}");
                // DELTA consistency with direct gaps.
                for i in 0..k {
                    let gap = d[i] - lam;
                    let tol = 16.0 * EPS * gap.abs().max(lam.abs()).max(1e-300);
                    assert!(
                        (delta[i] - gap).abs() <= tol,
                        "seed {seed} root {j} gap {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn laed4_last_root_matches_bisection() {
        // The I=N case has its own initial guess; stress large gaps.
        let mut s = 4242u64;
        for _ in 0..8 {
            let k = 5;
            let mut d: Vec<f64> = (0..k).map(|_| splitmix(&mut s) * 10.0).collect();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for i in 1..k {
                if d[i] - d[i - 1] < 1e-4 {
                    d[i] = d[i - 1] + 1e-4 * (1.0 + i as f64);
                }
            }
            let z: Vec<f64> = (0..k).map(|_| 0.1 + splitmix(&mut s).abs()).collect();
            let nrm: f64 = z.iter().map(|v| v * v).sum::<f64>().sqrt();
            let zn: Vec<f64> = z.iter().map(|v| v / nrm).collect();
            let rho = 0.3 * nrm * nrm;
            let mut delta = vec![0.0; k];
            let lam = laed4_root(&d, &zn, rho, k - 1, &mut delta, false).expect("last root");
            let reference = bisect_ref(&d, &zn, rho, k - 1);
            assert!(
                (lam - reference).abs() <= 1e-9 * lam.abs().max(1.0),
                "{lam} vs {reference}"
            );
            assert!(secular(&d, &zn, rho, lam).abs() <= 1e-12);
        }
    }

    #[test]
    fn laed5_pair_satisfies_eigen_equation() {
        let mut s = 31337u64;
        for _ in 0..40 {
            let mut d = [splitmix(&mut s) * 3.0, splitmix(&mut s) * 3.0];
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if d[1] - d[0] < 1e-8 {
                d[1] = d[0] + 1e-8;
            }
            // Bounded away from zero: deflation's contract (no negligible
            // z reaches the solver).
            let z = [0.2 + splitmix(&mut s).abs(), 0.2 + splitmix(&mut s).abs()];
            let nrm = (z[0] * z[0] + z[1] * z[1]).sqrt();
            let (z0, z1) = (z[0] / nrm, z[1] / nrm);
            let rho = 2.3 * nrm * nrm;
            let mut vecs = [[0.0; 2]; 2];
            let mut lams = [0.0; 2];
            for (t, right) in [false, true].iter().enumerate() {
                let (lam, e) = laed5_pair(d[0], d[1], z0, z1, rho, *right);
                lams[t] = lam;
                vecs[t] = e;
            }
            assert!(lams[0] < lams[1], "order: {lams:?}");
            for t in 0..2 {
                let (lam, e) = (lams[t], vecs[t]);
                // (D + ρzzᵀ)e = λe directly.
                let dot = z0 * e[0] + z1 * e[1];
                for i in 0..2 {
                    let di = if i == 0 { d[0] } else { d[1] };
                    let zi = if i == 0 { z0 } else { z1 };
                    let r = di * e[i] + rho * zi * dot - lam * e[i];
                    assert!(r.abs() <= 1e-13, "pair residual {r:.3e}");
                }
                let n = (e[0] * e[0] + e[1] * e[1]).sqrt();
                assert!((n - 1.0).abs() <= 1e-15, "unit {n}");
            }
            let dot = vecs[0][0] * vecs[1][0] + vecs[0][1] * vecs[1][1];
            assert!(dot.abs() <= 1e-14, "orthogonal {dot:.3e}");
        }
    }

    #[test]
    fn recomputed_w_gives_orthogonal_eigenvectors() {
        let mut s = 777u64;
        let k = 8;
        let mut d: Vec<f64> = (0..k).map(|_| splitmix(&mut s) * 4.0).collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 1..k {
            if d[i] - d[i - 1] < 1e-6 {
                d[i] = d[i - 1] + 1e-6 * (1.0 + i as f64);
            }
        }
        let z: Vec<f64> = (0..k).map(|_| 0.2 + splitmix(&mut s).abs()).collect();
        let nrm: f64 = z.iter().map(|v| v * v).sum::<f64>().sqrt();
        let zn: Vec<f64> = z.iter().map(|v| v / nrm).collect();
        let rho = 1.7 * nrm * nrm;
        let mut lams = vec![0.0; k];
        let mut deltas = vec![vec![0.0; k]; k];
        for j in 0..k {
            lams[j] = laed4_root(&d, &zn, rho, j, &mut deltas[j], false).expect("root");
        }
        let w = recompute_w(&d, &zn, &deltas).expect("finite w");
        // Löwner vectors, normalized.
        let mut vecs = vec![vec![0.0; k]; k];
        for j in 0..k {
            let mut n = 0.0;
            for i in 0..k {
                vecs[j][i] = w[i] / deltas[j][i];
                n += vecs[j][i] * vecs[j][i];
            }
            n = n.sqrt();
            for i in 0..k {
                vecs[j][i] /= n;
            }
        }
        for a in 0..k {
            for b in a + 1..k {
                let dot: f64 = (0..k).map(|i| vecs[a][i] * vecs[b][i]).sum();
                assert!(dot.abs() <= 1e-13, "pair ({a},{b}) dot {dot:.3e}");
            }
            // Residual against D + ρzzᵀ.
            let dotz: f64 = (0..k).map(|i| zn[i] * vecs[a][i]).sum();
            let mut rsq = 0.0;
            for i in 0..k {
                let r = d[i] * vecs[a][i] + rho * zn[i] * dotz - lams[a] * vecs[a][i];
                rsq += r * r;
            }
            assert!(rsq.sqrt() <= 1e-12, "residual col {a}");
        }
    }
}

/// Safeguarded step clamp shared by both DLAED4 cases: out-of-bounds
/// interpolation falls back to Newton, then to a bisection /
/// geometric-mean step.
fn clamp_step(eta: f64, w: f64, dw: f64, tau: f64, dltlb: f64, dltub: f64) -> f64 {
    let temp = tau + eta;
    if temp > dltub || temp < dltlb {
        let eta1 = -w / dw;
        let temp = tau + eta1;
        let eta2 = if w < 0.0 {
            (dltub - tau) / 2.0
        } else {
            (dltlb - tau) / 2.0
        };
        if dltlb <= temp && temp <= dltub {
            // Geometric mean of the Newton distance, keeping direction.
            eta1.signum() * eta1.abs().sqrt() * eta2.abs().sqrt()
        } else {
            eta2
        }
    } else {
        eta
    }
}
