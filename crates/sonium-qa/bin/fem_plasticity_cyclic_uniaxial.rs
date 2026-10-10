//! Cyclic uniaxial loading of the finite-strain J2 model.
//!
//! Drives a material point through strain-controlled triangle-wave cycles
//! with F = diag(lam, 1/sqrt(lam), 1/sqrt(lam)) and checks the stabilized
//! hysteresis: Bauschinger reverse yielding under kinematic hardening,
//! cyclic hardening saturation under isotropic hardening, positive plastic
//! dissipation per cycle, isochoric plastic flow (det Fp = 1), and agreement
//! with the small-strain model at small amplitudes.

use sonium_fem::physics::solid::plasticity::{
    FiniteStrainJ2Material, FiniteStrainState, HardeningLaw, J2Material, PlasticState,
    StressUpdateModel, update::von_mises_c1,
};

fn diag_f(lam: f64) -> [f64; 9] {
    let mu = 1.0 / lam.sqrt();
    [lam, 0.0, 0.0, 0.0, mu, 0.0, 0.0, 0.0, mu]
}

/// Incremental gradient from two total diagonal stretches.
fn inc_f(lam_new: f64, lam_old: f64) -> [f64; 9] {
    let f_new = diag_f(lam_new);
    let mu_old = lam_old.sqrt();
    [
        f_new[0] / lam_old,
        0.0,
        0.0,
        0.0,
        f_new[4] * mu_old,
        0.0,
        0.0,
        0.0,
        f_new[8] * mu_old,
    ]
}

fn mat3_det(a: &[f64; 9]) -> f64 {
    a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6])
}

/// Runs triangle cycles between lam_min and lam_max, recording sigma_xx.
fn run_cycles(
    material: &FiniteStrainJ2Material,
    lam_min: f64,
    lam_max: f64,
    steps_per_leg: usize,
    n_cycles: usize,
) -> (FiniteStrainState, Vec<f64>, Vec<f64>) {
    let mut state = FiniteStrainState::new();
    let mut lam = 1.0;
    let mut sigmas = Vec::new();
    let mut lams = Vec::new();
    // First quarter: 1 -> lam_max, then full triangle legs.
    let mut targets = vec![lam_max];
    for _ in 0..n_cycles {
        targets.push(lam_min);
        targets.push(lam_max);
    }
    for target in targets {
        let dlam = (target - lam) / steps_per_leg as f64;
        for _ in 0..steps_per_leg {
            let lam_new = lam + dlam;
            let f = inc_f(lam_new, lam);
            let update = material.update(&state, &f);
            material.commit(&mut state, &update);
            lam = lam_new;
            sigmas.push(state.stress[0]);
            lams.push(lam);
        }
    }
    (state, sigmas, lams)
}

fn test_kinematic_stabilized_loop() {
    let material = FiniteStrainJ2Material::new(
        200e9,
        0.3,
        250e6,
        HardeningLaw::LinearKinematic { modulus: 5e9 },
    );
    let (state, sigmas, _) = run_cycles(&material, 0.985, 1.015, 30, 3);
    let leg = 30;
    // Peak stresses of cycles 2 and 3 (each full cycle spans 2 legs after
    // the initial quarter leg).
    let peak = |cycle: usize| {
        let start = leg + (cycle - 1) * 2 * leg;
        sigmas[start..start + 2 * leg]
            .iter()
            .copied()
            .fold(f64::MIN, f64::max)
    };
    let valley = |cycle: usize| {
        let start = leg + (cycle - 1) * 2 * leg;
        sigmas[start..start + 2 * leg]
            .iter()
            .copied()
            .fold(f64::MAX, f64::min)
    };
    let p2 = peak(2);
    let p3 = peak(3);
    let v2 = valley(2);
    let v3 = valley(3);
    println!(
        "kinematic peaks: c2 = {p2:.6e}, c3 = {p3:.6e}; valleys: c2 = {v2:.6e}, c3 = {v3:.6e}"
    );
    assert!(
        p2 > 0.0 && v2 < 0.0,
        "loop must span tension and compression"
    );
    // Linear kinematic hardening shakes down after the first cycle.
    assert!(
        (p3 - p2).abs() / p2 < 0.005,
        "tensile peak not stabilized: {p2:.3e} vs {p3:.3e}"
    );
    assert!(
        (v3 - v2).abs() / v2.abs() < 0.005,
        "compressive valley not stabilized"
    );
    // The stabilized symmetric loop is symmetric; Bauschinger shows on the
    // first reversal, checked in test_bauschinger_first_reversal below.
    // Isochoric plastic flow through the cycles.
    assert!(
        (mat3_det(&state.fp) - 1.0).abs() < 1e-12,
        "det(Fp) = {}",
        mat3_det(&state.fp)
    );
    println!(
        "equiv plastic strain after 3 cycles = {:.6e}",
        state.equiv_plastic_strain
    );
}

fn test_bauschinger_first_reversal() {
    let material = FiniteStrainJ2Material::new(
        200e9,
        0.3,
        250e6,
        HardeningLaw::LinearKinematic { modulus: 5e9 },
    );
    let mut state = FiniteStrainState::new();
    // Forward load to +1.5%.
    let mut lam = 1.0;
    for _ in 0..30 {
        let lam_new = lam + 0.015 / 30.0;
        let f = inc_f(lam_new, lam);
        let update = material.update(&state, &f);
        material.commit(&mut state, &update);
        lam = lam_new;
    }
    let forward_peak = state.stress[0];
    // The isochoric path constrains the transverse stress to -sigma_xx/2,
    // so yielding (q = 250 MPa) happens at sigma_xx ~ 197 MPa, below the
    // uniaxial-stress yield. Assert the yielded state via q, not sigma_xx.
    let q_forward = von_mises_c1(&state.stress);
    assert!(
        q_forward > 0.99 * 250e6,
        "forward load must reach the yield surface, q = {q_forward:.3e}"
    );

    // Unload toward compression; find the first reverse-yielding step after
    // the elastic gap. The back-stress pulls reverse yield above -forward.
    let mut saw_elastic = false;
    let mut reverse_yield = f64::NAN;
    for _ in 0..60 {
        let lam_new = lam - 0.03 / 60.0;
        let f = inc_f(lam_new, lam);
        let update = material.update(&state, &f);
        if !update.yielding {
            saw_elastic = true;
        }
        if saw_elastic && update.yielding {
            reverse_yield = update.stress[0];
            material.commit(&mut state, &update);
            break;
        }
        material.commit(&mut state, &update);
        lam = lam_new;
    }
    println!("forward peak = {forward_peak:.6e}, reverse yield = {reverse_yield:.6e}");
    assert!(saw_elastic, "unloading must start elastic");
    assert!(
        reverse_yield.is_finite(),
        "reverse yielding must resume within the leg"
    );
    assert!(
        reverse_yield.abs() < 0.95 * forward_peak,
        "Bauschinger: |{reverse_yield:.3e}| should be well below {forward_peak:.3e}"
    );
}

fn test_isotropic_cyclic_hardening() {
    let material = FiniteStrainJ2Material::new(
        200e9,
        0.3,
        250e6,
        HardeningLaw::LinearIsotropic { modulus: 2e9 },
    );
    let (_, sigmas, _) = run_cycles(&material, 0.985, 1.015, 30, 3);
    let leg = 30;
    let peak = |cycle: usize| {
        let start = leg + (cycle - 1) * 2 * leg;
        sigmas[start..start + 2 * leg]
            .iter()
            .copied()
            .fold(f64::MIN, f64::max)
    };
    let p1 = peak(1);
    let p2 = peak(2);
    let p3 = peak(3);
    println!("isotropic peaks: c1 = {p1:.6e}, c2 = {p2:.6e}, c3 = {p3:.6e}");
    // Isotropic hardening grows the yield surface every cycle.
    assert!(p2 > p1, "cycle 2 should harden over cycle 1");
    assert!(p3 > p2, "cycle 3 should harden over cycle 2");
}

fn test_cycle_dissipation_positive() {
    let material = FiniteStrainJ2Material::mild_steel();
    let mut state = FiniteStrainState::new();
    // Closed triangle cycle: 1 -> 1.02 -> 0.98 -> 1.02 -> 1.0.
    // Plastic work W = sum sigma : d with d = ln U per step (diagonal here).
    let mut lam = 1.0;
    let mut work = 0.0;
    for target in [1.02, 0.98, 1.02, 1.0] {
        let n = 40;
        let dlam = (target - lam) / f64::from(n);
        for _ in 0..n {
            let lam_new = lam + dlam;
            let f = inc_f(lam_new, lam);
            let update = material.update(&state, &f);
            // Logarithmic strain increment of the diagonal step.
            let d = [f[0].ln(), f[4].ln(), f[8].ln()];
            work += update.stress[0] * d[0] + update.stress[1] * d[1] + update.stress[2] * d[2];
            material.commit(&mut state, &update);
            lam = lam_new;
        }
    }
    println!("net work over closed cycle = {work:.6e} J/m^3");
    // Dissipation over the ~0.12 plastic strain path is O(10 MJ/m^3), far
    // above any elastic end-point residue, so the sign is a strict check.
    assert!(work > 1.0e6, "closed cycle must dissipate positive work");
}

fn test_small_amplitude_matches_small_strain() {
    let finite = FiniteStrainJ2Material::mild_steel();
    let small = J2Material::mild_steel();
    // Gentle cycle to 0.2% strain: both models must agree closely.
    let (f_state, f_sigmas, _) = run_cycles(&finite, 0.998, 1.002, 20, 1);
    let mut s_state = PlasticState::new_3d();
    let mut s_peak: f64 = 0.0;
    let mut lam: f64 = 1.0;
    for target in [1.002, 0.998, 1.002] {
        let dlam = (target - lam) / 20.0;
        for _ in 0..20 {
            // Small-strain engineering increment matching ln(lam) to 1e-6.
            let d = [dlam, -0.5 * dlam, -0.5 * dlam, 0.0, 0.0, 0.0];
            let update = small.update(&s_state, &d);
            small.commit(&mut s_state, &update);
            lam += dlam;
            s_peak = s_peak.max(s_state.stress[0]);
        }
    }
    let f_peak = f_sigmas.iter().copied().fold(f64::MIN, f64::max);
    println!("small-amplitude peaks: finite = {f_peak:.6e}, small-strain = {s_peak:.6e}");
    assert!(
        (f_peak - s_peak).abs() / s_peak.abs().max(1.0) < 0.02,
        "small-amplitude mismatch"
    );
    assert_eq!(f_state.yielding, s_state.yielding);
    let q_f = von_mises_c1(&f_state.stress);
    assert!(q_f >= 0.0);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_kinematic_stabilized_loop();
        test_bauschinger_first_reversal();
        test_isotropic_cyclic_hardening();
        test_cycle_dissipation_positive();
        test_small_amplitude_matches_small_strain();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
