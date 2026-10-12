//! Thick-cylinder autofrettage: references and native spatial pressure-cycle accuracy.
//!
//! An internally pressurized thick cylinder (plane strain, von Mises,
//! elastic-perfectly plastic) is loaded past first yield and unloaded
//! elastically. The overload leaves beneficial compressive hoop residual
//! stress at the bore. This binary validates:
//! - first-yield and limit pressures against closed forms (with a Tresca
//!   cross-check of the 2/sqrt(3) von Mises factor),
//! - the residual field: traction-free bore and outer wall, compressive
//!   bore hoop stress, radial equilibrium, and no reverse yielding,
//! - a finite-strain J2 material point driven along the bore overstrain
//!   path, against the small-strain model.
//!
//! The C3 Radioss comparison case lives in `bin/fem_opencourant_plasticity_thick_cylinder.rs`.
//! The separate spatial gate uses a near-incompressible radial B-bar FEM mesh
//! with native J2 updates and pressure loading/unloading; reference and bore
//! material-point assertions remain separate from this numerical evidence.

// Rust guideline compliant 2026-02-21

#[path = "support/autofrettage.rs"]
mod autofrettage;

use sonium_fem::physics::solid::plasticity::{
    FiniteStrainJ2Material, FiniteStrainState, HardeningLaw, J2Material, PlasticState,
    StressUpdateModel, update::von_mises_c1,
};

const A_M: f64 = 0.05;
const B_M: f64 = 0.10;
const SIGMA_Y: f64 = 250.0e6;
const SQRT3: f64 = 1.7320508075688772;

/// Lame elastic solution under internal pressure p at radius r.
fn lame(a: f64, b: f64, p: f64, r: f64) -> (f64, f64) {
    let a2 = a * a;
    let b2 = b * b;
    let amp = p * a2 / (b2 - a2);
    let shape = b2 / (r * r);
    (amp * (1.0 - shape), amp * (1.0 + shape))
}

/// First-yield pressure, von Mises plane strain: p_y = sy (1-(a/b)^2)/sqrt(3).
fn yield_pressure(a: f64, b: f64, sy: f64) -> f64 {
    sy * (1.0 - (a / b).powi(2)) / SQRT3
}

/// Limit pressure, von Mises plane strain: p_lim = (2/sqrt(3)) sy ln(b/a).
fn limit_pressure(a: f64, b: f64, sy: f64) -> f64 {
    (2.0 / SQRT3) * sy * (b / a).ln()
}

/// Elastic-plastic interface radius c for overload pressure p_star.
fn interface_radius(a: f64, b: f64, sy: f64, p_star: f64) -> f64 {
    let target = p_star * SQRT3 / (2.0 * sy);
    let g = |c: f64| (c / a).ln() + (b * b - c * c) / (2.0 * b * b) - target;
    let mut lo = a;
    let mut hi = b;
    assert!(
        g(lo) <= 0.0 && g(hi) >= 0.0,
        "overload outside (p_y, p_lim)"
    );
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if g(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Loaded stresses (sigma_r, sigma_theta) at radius r for overload p_star.
fn loaded_stress(a: f64, b: f64, c: f64, sy: f64, p_star: f64, r: f64) -> (f64, f64) {
    let flow = 2.0 * sy / SQRT3;
    if r <= c {
        let sr = -p_star + flow * (r / a).ln();
        (sr, sr + flow)
    } else {
        let qc = p_star - flow * (c / a).ln();
        let c2 = c * c;
        let b2 = b * b;
        let amp = qc * c2 / (b2 - c2);
        let shape = b2 / (r * r);
        (amp * (1.0 - shape), amp * (1.0 + shape))
    }
}

/// Residual stresses after elastic unloading of p_star.
fn residual_stress(a: f64, b: f64, c: f64, sy: f64, p_star: f64, r: f64) -> (f64, f64) {
    let (lr, lt) = loaded_stress(a, b, c, sy, p_star, r);
    let (ur, ut) = lame(a, b, p_star, r);
    (lr - ur, lt - ut)
}

/// Von Mises stress under plane strain with sigma_z = (sr+st)/2.
fn equiv_plane_strain(sr: f64, st: f64) -> f64 {
    (SQRT3 / 2.0) * (st - sr).abs()
}

fn test_pressures() {
    let py = yield_pressure(A_M, B_M, SIGMA_Y);
    let plim = limit_pressure(A_M, B_M, SIGMA_Y);
    println!("p_y = {py:.10e}, p_lim = {plim:.10e}");
    // Closed-form regression values (SI): guard against formula drift.
    assert!(
        (py - 108.2531754731e6).abs() < 1.0e3,
        "p_y drifted: {py:.6e}"
    );
    assert!(
        (plim - 200.09435564e6).abs() < 1.0e3,
        "p_lim drifted: {plim:.6e}"
    );
    // Tresca cross-check: von Mises limit = (2/sqrt(3)) * Tresca limit.
    let tresca_lim = SIGMA_Y * (B_M / A_M).ln();
    assert!(
        ((plim / tresca_lim) - 2.0 / SQRT3).abs() < 1e-12,
        "von Mises / Tresca factor wrong"
    );
    assert!(py < plim, "ordering");
    // Consistency with the legacy QA helper (Tresca-style sy*ln(b/a)).
    let legacy = sonium_qa::analytical::plasticity_thick_cylinder_limit_pressure(A_M, B_M, SIGMA_Y);
    assert!(
        (legacy - tresca_lim).abs() / tresca_lim < 1e-12,
        "legacy helper mismatch"
    );
}

fn test_residual_field() {
    let p_star = 150.0e6;
    let py = yield_pressure(A_M, B_M, SIGMA_Y);
    let plim = limit_pressure(A_M, B_M, SIGMA_Y);
    assert!(
        p_star > py && p_star < plim,
        "overload must partially yield"
    );

    let c = interface_radius(A_M, B_M, SIGMA_Y, p_star);
    println!("interface radius c = {c:.10e} for p* = {p_star:.6e}");
    assert!(
        (c - 6.1670010320e-2).abs() < 1.0e-9,
        "interface drifted: {c:.6e}"
    );
    assert!(c > A_M && c < B_M, "partial autofrettage needs a < c < b");

    // Traction-free residuals at bore and outer wall.
    let (sr_a, st_a) = residual_stress(A_M, B_M, c, SIGMA_Y, p_star, A_M);
    let (sr_b, st_b) = residual_stress(A_M, B_M, c, SIGMA_Y, p_star, B_M);
    println!("bore residual: sr = {sr_a:.6e}, st = {st_a:.6e}");
    println!("outer residual: sr = {sr_b:.6e}, st = {st_b:.6e}");
    assert!(sr_a.abs() < 1.0, "bore radial residual must vanish");
    assert!(sr_b.abs() < 1.0, "outer radial residual must vanish");
    // Beneficial compressive hoop residual at the bore (golden value).
    assert!(
        (st_a + 1.1132486541e8).abs() < 1.0e3,
        "bore hoop drifted: {st_a:.6e}"
    );
    assert!(st_a < 0.0, "bore hoop residual must be compressive");
    assert!(
        (st_b - 9.7886435030e6).abs() < 1.0e3,
        "outer hoop drifted: {st_b:.6e}"
    );

    // Radial equilibrium of the residual field on a grid, skipping the
    // interface kink: d sr/dr + (sr - st)/r = 0.
    let n = 41;
    let mut max_violation: f64 = 0.0;
    for i in 0..n {
        let r = A_M + (B_M - A_M) * i as f64 / (n - 1) as f64;
        if (r - c).abs() < 1.5 * (B_M - A_M) / (n - 1) as f64 {
            continue;
        }
        let h = 1.0e-7;
        let (sr_p, _) = residual_stress(A_M, B_M, c, SIGMA_Y, p_star, r + h);
        let (sr_m, _) = residual_stress(A_M, B_M, c, SIGMA_Y, p_star, r - h);
        let (sr, st) = residual_stress(A_M, B_M, c, SIGMA_Y, p_star, r);
        let violation = ((sr_p - sr_m) / (2.0 * h) + (sr - st) / r).abs() / SIGMA_Y * A_M;
        max_violation = max_violation.max(violation);
    }
    println!("max equilibrium violation (normalized) = {max_violation:.3e}");
    assert!(
        max_violation < 1.0e-6,
        "residual field violates equilibrium"
    );

    // No reverse yielding anywhere: residual equivalent below yield.
    let mut max_q: f64 = 0.0;
    for i in 0..=200 {
        let r = A_M + (B_M - A_M) * i as f64 / 200.0;
        let (sr, st) = residual_stress(A_M, B_M, c, SIGMA_Y, p_star, r);
        max_q = max_q.max(equiv_plane_strain(sr, st));
    }
    println!("max residual equivalent = {max_q:.6e} vs sy = {SIGMA_Y:.6e}");
    assert!(
        max_q < SIGMA_Y,
        "reverse yielding invalidates elastic unloading"
    );
}

fn test_finite_strain_bore_overstrain() {
    // Bore material point under plane-strain hoop overstretch:
    // F = diag(1/lam, lam, 1), lam: 1 -> 1.01 -> 1.
    let finite = FiniteStrainJ2Material::new(200e9, 0.3, SIGMA_Y, HardeningLaw::Perfect);
    let small = J2Material::new(200e9, 0.3, SIGMA_Y, HardeningLaw::Perfect);
    let mut fs = FiniteStrainState::new();
    let mut ss = PlasticState::new_3d();
    let mut lam = 1.0;
    let mut peak_hoop = 0.0_f64;
    for target in [1.01, 1.0] {
        let step = (target - lam) / 50.0;
        for _ in 0..50 {
            let l2 = lam + step;
            let f = [lam / l2, 0.0, 0.0, 0.0, l2 / lam, 0.0, 0.0, 0.0, 1.0];
            let update = finite.update(&fs, &f);
            finite.commit(&mut fs, &update);
            peak_hoop = peak_hoop.max(fs.stress[1]);
            // Same logarithmic path for the small-strain model (C1 order).
            let e = (l2 / lam).ln();
            let d = [-e, e, 0.0, 0.0, 0.0, 0.0];
            let us = small.update(&ss, &d);
            small.commit(&mut ss, &us);
            lam = l2;
        }
    }
    println!(
        "bore overstrain: peak hoop = {peak_hoop:.6e}, residual hoop = {:.6e}",
        fs.stress[1]
    );
    // The isochoric hoop path yields at sigma_theta = sy/sqrt(3) (pure
    // shear-like with sigma_z = 0), so the yield check uses q at peak load.
    assert!(
        (peak_hoop - SIGMA_Y / 3.0_f64.sqrt()).abs() / SIGMA_Y < 0.02,
        "peak hoop should sit at the shear yield, got {peak_hoop:.3e}"
    );
    assert!(
        fs.stress[1] < -1.0e6,
        "unloaded bore hoop must be compressive, got {:.3e}",
        fs.stress[1]
    );
    let q = von_mises_c1(&fs.stress);
    // Reverse yielding during unloading can leave the point on the surface.
    assert!(
        q <= SIGMA_Y * (1.0 + 1e-9),
        "unloaded point must respect the yield surface"
    );
    let det_fp = fs.fp[0] * (fs.fp[4] * fs.fp[8] - fs.fp[5] * fs.fp[7])
        - fs.fp[1] * (fs.fp[3] * fs.fp[8] - fs.fp[5] * fs.fp[6])
        + fs.fp[2] * (fs.fp[3] * fs.fp[7] - fs.fp[4] * fs.fp[6]);
    assert!((det_fp - 1.0).abs() < 1e-12, "det(Fp) = {det_fp}");
    // Peak hoop agreement between the two models (recomputed small-strain
    // peak is implicit in ss; compare residual hoop instead, robustly).
    let small_resid = ss.stress[1];
    println!("small-strain residual hoop = {small_resid:.6e}");
    assert!(
        (fs.stress[1] - small_resid).abs() / small_resid.abs().max(1.0) < 0.05,
        "finite/small residual mismatch"
    );
}
fn main() {
    let result = std::panic::catch_unwind(|| {
        test_pressures();
        test_residual_field();
        test_finite_strain_bore_overstrain();
    });
    match result {
        Ok(()) => {
            match autofrettage::run()
                .and_then(|result| result.to_json_line().map_err(|e| e.to_string()))
            {
                Ok(json) => {
                    println!("QA_RESULT: {json}");
                    println!("SUCCESS");
                }
                Err(error) => {
                    eprintln!("FAILED: {error}");
                    std::process::exit(1);
                }
            }
        }
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
