//! Simple shear with superposed rotation: objectivity trap test.
//!
//! Drives the finite-strain J2 model through simple shear to gamma = 1 and
//! through the same shear with a superposed rigid rotation. A Jaumann-rate
//! implementation fails this test: its shear stress oscillates spuriously
//! with accumulated shear. The Green-Naghdi rate used here grows the shear
//! stress monotonically toward saturation and rotates the Cauchy stress
//! exactly with superposed rotations.
//!
//! Expected behavior (documented for the golden record):
//! - sigma_xy(gamma) rises monotonically and saturates near the shear yield
//!   k = sigma_y / sqrt(3) (plus isotropic hardening growth); the total
//!   negative variation stays below 1% of the peak (numerical noise only).
//! - A shear / rigid-rotation-hold / shear-in-rotated-frame path reproduces
//!   the rotated uninterrupted response exactly: q, plastic strain, and Fp
//!   freeze during the hold while sigma rotates by exactly R (1e-9).
//! - A constant rotation Q applied to the whole loading path rotates the
//!   Cauchy stress by exactly Q (frame indifference, 1e-6).

use sonium_fem::physics::solid::plasticity::{
    FiniteStrainJ2Material, FiniteStrainState, HardeningLaw, StressUpdateModel,
    update::von_mises_c1,
};

const SHEAR_YIELD_TOL: f64 = 0.02;
const OBJECTIVITY_TOL: f64 = 1.0e-6;

fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut c = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            c[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        }
    }
    c
}

fn rot_z(theta: f64) -> [f64; 9] {
    let (s, c) = theta.sin_cos();
    [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]
}

/// Runs simple shear to `gamma_max` in `n_steps`, returning per-step history.
fn run_shear(
    material: &FiniteStrainJ2Material,
    gamma_max: f64,
    n_steps: usize,
) -> (FiniteStrainState, Vec<[f64; 6]>, Vec<f64>, Vec<f64>) {
    let mut state = FiniteStrainState::new();
    let dg = gamma_max / n_steps as f64;
    let mut stresses = Vec::with_capacity(n_steps + 1);
    let mut gammas = Vec::with_capacity(n_steps + 1);
    let mut eps = Vec::with_capacity(n_steps + 1);
    stresses.push(state.stress);
    gammas.push(0.0);
    eps.push(0.0);
    for step in 1..=n_steps {
        // Incremental gradient of simple shear: f = F_new F_old^-1.
        let f = [1.0, dg, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let update = material.update(&state, &f);
        material.commit(&mut state, &update);
        stresses.push(state.stress);
        gammas.push(dg * step as f64);
        eps.push(state.equiv_plastic_strain);
    }
    (state, stresses, gammas, eps)
}

fn test_shear_no_oscillation() {
    let material = FiniteStrainJ2Material::mild_steel();
    let (state, stresses, gammas, eps) = run_shear(&material, 1.0, 200);
    assert!(state.yielding, "shear to gamma=1 must yield");

    // Monotonicity: total negative variation below 1% of the peak value.
    let peak = stresses.iter().map(|s| s[3]).fold(0.0, f64::max);
    assert!(peak > 0.0, "shear stress must grow positive");
    let mut negative_variation = 0.0;
    for pair in stresses.windows(2) {
        let step = pair[1][3] - pair[0][3];
        if step < 0.0 {
            negative_variation -= step;
        }
    }
    println!("peak sigma_xy = {peak:.6e}, negative variation = {negative_variation:.6e}");
    assert!(
        negative_variation < 0.01 * peak,
        "shear stress oscillates: negative variation {negative_variation:.3e} vs peak {peak:.3e}"
    );

    // Consistency: once plastic, von Mises stress stays on the yield surface.
    for ((s, g), ep) in stresses.iter().zip(gammas.iter()).zip(eps.iter()) {
        if *ep > 1.0e-6 {
            let q = von_mises_c1(s);
            let yield_now = material.base.current_yield_stress(*ep);
            assert!(
                (q - yield_now).abs() / yield_now < SHEAR_YIELD_TOL,
                "off yield surface at gamma={g:.3}: q={q:.3e} vs {yield_now:.3e}"
            );
        }
    }
    let q_final = von_mises_c1(&state.stress);
    let yield_now = material
        .base
        .current_yield_stress(state.equiv_plastic_strain);
    println!("q_final = {q_final:.6e}, hardened yield = {yield_now:.6e}");
    assert!(
        (q_final - yield_now).abs() / yield_now < SHEAR_YIELD_TOL,
        "final stress off the yield surface"
    );

    // Shear yield k = sigma_y / sqrt(3) bounds the saturated shear stress
    // from below (isotropic hardening raises it above k).
    let k = material.base.yield_stress / 3.0_f64.sqrt();
    assert!(
        peak > 0.9 * k,
        "saturated shear {peak:.3e} below shear yield {k:.3e}"
    );

    // Curve samples for the golden record.
    for target in [0.25_f64, 0.5, 0.75, 1.0] {
        let idx = (target * 200.0).round() as usize;
        println!(
            "gamma = {target:.2}: sigma_xy = {:.10e}, q = {:.10e}",
            stresses[idx][3],
            von_mises_c1(&stresses[idx])
        );
    }
}

fn test_shear_rotate_shear_path() {
    let material = FiniteStrainJ2Material::mild_steel();
    let dg = 0.005;

    // Reference: uninterrupted shear to gamma = 1 in 200 steps.
    let (_, stresses_plain, _, _) = run_shear(&material, 1.0, 200);

    // Phase A: shear to gamma = 0.5 in 100 steps.
    let mut state = FiniteStrainState::new();
    for _ in 0..100 {
        let f = [1.0, dg, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let update = material.update(&state, &f);
        material.commit(&mut state, &update);
    }
    let stress_a = state.stress;
    let ep_a = state.equiv_plastic_strain;
    let fp_a = state.fp;
    let q_a = von_mises_c1(&stress_a);

    // Phase B: pure 90-degree rotation in 90 steps. Equivalent stress,
    // plastic strain, and Fp must freeze; sigma must rotate exactly.
    let mut rot_total = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let step_rot = rot_z(std::f64::consts::FRAC_PI_2 / 90.0);
    let mut max_q_drift: f64 = 0.0;
    let mut max_rot_err: f64 = 0.0;
    for _ in 0..90 {
        let update = material.update(&state, &step_rot);
        assert!(!update.yielding, "pure rotation must not yield");
        material.commit(&mut state, &update);
        rot_total = mat3_mul(&step_rot, &rot_total);
        max_q_drift = max_q_drift.max((von_mises_c1(&state.stress) - q_a).abs() / q_a);
        let expected = rotate_stress(&stress_a, &rot_total);
        for i in 0..6 {
            max_rot_err = max_rot_err.max((state.stress[i] - expected[i]).abs() / q_a);
        }
    }
    println!("rotation hold: q drift = {max_q_drift:.3e}, rotation error = {max_rot_err:.3e}");
    assert!(max_q_drift < 1.0e-9, "q drifted during rigid rotation");
    assert!(max_rot_err < 1.0e-9, "stress did not rotate exactly");
    assert!(
        (state.equiv_plastic_strain - ep_a).abs() == 0.0,
        "plastic strain changed during rotation"
    );
    for i in 0..9 {
        assert!(
            (state.fp[i] - fp_a[i]).abs() < 1e-15,
            "Fp changed during rotation"
        );
    }

    // Phase C: continue shearing in the rotated frame. By frame indifference
    // this must reproduce the rotated uninterrupted response exactly.
    let r90 = rot_z(std::f64::consts::FRAC_PI_2);
    let r90t = [
        r90[0], r90[3], r90[6], r90[1], r90[4], r90[7], r90[2], r90[5], r90[8],
    ];
    let mut max_q_diff: f64 = 0.0;
    let mut max_s_diff: f64 = 0.0;
    for k in 1..=100 {
        let f = [1.0, dg, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let f_rot = mat3_mul(&mat3_mul(&r90, &f), &r90t);
        let update = material.update(&state, &f_rot);
        material.commit(&mut state, &update);

        let expected = rotate_stress(&stresses_plain[100 + k], &r90);
        let scale = von_mises_c1(&expected).max(1.0);
        max_q_diff =
            max_q_diff.max((von_mises_c1(&state.stress) - von_mises_c1(&expected)).abs() / scale);
        for i in 0..6 {
            max_s_diff = max_s_diff.max((state.stress[i] - expected[i]).abs() / scale);
        }
    }
    println!(
        "shear-in-rotated-frame: q deviation = {max_q_diff:.3e}, stress deviation = {max_s_diff:.3e}"
    );
    assert!(max_q_diff < 1.0e-6, "rotated shear changed q");
    assert!(
        max_s_diff < 1.0e-6,
        "rotated shear broke frame indifference"
    );
}

fn rotate_stress(s: &[f64; 6], q: &[f64; 9]) -> [f64; 6] {
    let t = [s[0], s[3], s[5], s[3], s[1], s[4], s[5], s[4], s[2]];
    let qt = [q[0], q[3], q[6], q[1], q[4], q[7], q[2], q[5], q[8]];
    let r = mat3_mul(&mat3_mul(q, &t), &qt);
    [
        r[0],
        r[4],
        r[8],
        0.5 * (r[1] + r[3]),
        0.5 * (r[5] + r[7]),
        0.5 * (r[6] + r[2]),
    ]
}

fn test_fixed_rotation_frame_indifference() {
    let material = FiniteStrainJ2Material::mild_steel();
    // Constant 37-degree rotation of the whole loading path.
    let q = rot_z(37.0_f64.to_radians());
    let q_inv = [q[0], q[3], q[6], q[1], q[4], q[7], q[2], q[5], q[8]];

    let (_, stresses_plain, _, _) = run_shear(&material, 1.0, 100);

    // Rotated increments: f_rot = Q f Q^T for the shear increments.
    let mut state = FiniteStrainState::new();
    let dg = 0.01;
    let mut max_diff: f64 = 0.0;
    for step in 1..=100 {
        let f = [1.0, dg, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let f_rot = mat3_mul(&mat3_mul(&q, &f), &q_inv);
        let update = material.update(&state, &f_rot);
        material.commit(&mut state, &update);

        let expected = rotate_stress(&stresses_plain[step], &q);
        let scale = von_mises_c1(&expected).max(1.0);
        for i in 0..6 {
            max_diff = max_diff.max((state.stress[i] - expected[i]).abs() / scale);
        }
    }
    println!("frame-indifference max relative deviation = {max_diff:.3e}");
    assert!(
        max_diff < OBJECTIVITY_TOL,
        "fixed rotation not transported exactly: {max_diff:.3e}"
    );
}

fn test_perfect_plasticity_saturates() {
    let material = FiniteStrainJ2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);
    let (state, stresses, _, _) = run_shear(&material, 1.0, 200);
    let k = 250e6 / 3.0_f64.sqrt();
    // Late-path shear stress plateaus at the shear yield k.
    let late = stresses[150..].iter().map(|s| s[3]).fold(0.0, f64::max);
    let early = stresses[100];
    println!(
        "perfect plasticity: sigma_xy(gamma=0.5) = {:.6e}, late peak = {late:.6e}, k = {k:.6e}",
        early[3]
    );
    assert!(
        (late - k).abs() / k < 0.05,
        "shear stress {late:.3e} did not saturate at k={k:.3e}"
    );
    let q = von_mises_c1(&state.stress);
    assert!((q - 250e6).abs() / 250e6 < SHEAR_YIELD_TOL);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_shear_no_oscillation();
        test_shear_rotate_shear_path();
        test_fixed_rotation_frame_indifference();
        test_perfect_plasticity_saturates();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
