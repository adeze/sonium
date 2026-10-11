//! QA: cohesive zone patch tests vs analytic peaks and energies.
//!
//! Opens single cohesive points monotonically and checks peak tractions,
//! failure openings, dissipated energies, secant unloading, and the
//! mixed-mode Benzeggagh-Kenane interpolation. Prints PASS; exits 1 on
//! any mismatch.

// Rust guideline compliant 2026-02-21

use sonium_fem::physics::solid::fracture::cohesive::{
    BilinearCohesiveLaw, CohesiveState, ExponentialCohesiveLaw, MixedModeCohesiveLaw,
    cohesive_patch_response, patch_dissipated_energy,
};

fn main() {
    // Bilinear epoxy: peak traction and dissipated energy.
    let law = BilinearCohesiveLaw::epoxy();
    let mut state = CohesiveState::new();
    let curve =
        cohesive_patch_response(2000, law.failure_opening(), |d| law.traction(d, &mut state));
    let peak = curve.iter().map(|(_, t)| *t).fold(0.0, f64::max);
    println!(
        "bilinear: peak {peak:.6e}, dissipated {:.6e}",
        patch_dissipated_energy(&curve)
    );
    assert!((peak - law.strength).abs() / law.strength < 1e-3);
    assert!(
        (patch_dissipated_energy(&curve) - law.fracture_energy).abs() / law.fracture_energy < 1e-6
    );
    assert!((state.damage - 1.0).abs() < 1e-12);
    // Bilinear unloading follows the damaged secant.
    let mut secant = CohesiveState::new();
    let half = 0.5 * (law.onset_opening() + law.failure_opening());
    let loading = law.traction(half, &mut secant);
    let partial = law.traction(0.5 * half, &mut secant);
    assert!((partial - 0.5 * loading).abs() / loading < 1e-9);
    println!("bilinear-secant: damage {:.4}", secant.damage);
    // Exponential law: peak at the characteristic opening, full energy.
    let exp_law = ExponentialCohesiveLaw::new(30.0e6, 500.0);
    let mut exp_state = CohesiveState::new();
    let dc = exp_law.characteristic_opening();
    assert!((exp_law.traction(dc, &mut exp_state) - 30.0e6).abs() / 30.0e6 < 1e-12);
    let mut tail = CohesiveState::new();
    let exp_curve = cohesive_patch_response(20000, 20.0 * dc, |d| exp_law.traction(d, &mut tail));
    println!(
        "exponential: dissipated {:.6e}",
        patch_dissipated_energy(&exp_curve)
    );
    assert!((patch_dissipated_energy(&exp_curve) - 500.0).abs() / 500.0 < 1e-4);
    // Mixed mode recovers pure modes and the BK interpolation.
    let mixed = MixedModeCohesiveLaw::epoxy();
    assert!((mixed.mixed_fracture_energy(0.0) - 500.0).abs() < 1e-9);
    let mixed_g = mixed.mixed_fracture_energy(1.0);
    println!("mixed-bk: G(1.0) {mixed_g:.6e}");
    assert!(mixed_g > 500.0 && mixed_g < 800.0);
    let mut pure = CohesiveState::new();
    let t = mixed.traction_3d([mixed.normal.onset_opening(), 0.0, 0.0], &mut pure);
    assert!((t[0] - 30.0e6).abs() / 30.0e6 < 1e-9);
    println!("PASS: qa-fem-cohesive-patch");
}
