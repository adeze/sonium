//! Phase-field fracture validations: homogeneous AT2 law, through-crack
//! profile plus Griffith energy, and tension/compression asymmetry.
//!
//! All three run the staggered phase-field damage study on fixed hermetic
//! meshes and compare against closed forms of the AT2 model with the Miehe
//! spectral split (no fitted constants).

// Rust guideline compliant 2026-09-30

use sonium_sim_structural_mechanics::{PhaseFieldDamageStudy, run_phase_field_damage};

/// Lame constants from Young's modulus and Poisson's ratio.
fn lame(youngs_modulus: f64, poissons_ratio: f64) -> (f64, f64) {
    let nu = poissons_ratio;
    (
        youngs_modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
        youngs_modulus / (2.0 * (1.0 + nu)),
    )
}

/// AT2 homogeneous damage for driving energy `h`: `(Gc/l + 2H) d = 2H`.
fn homogeneous_damage(driving: f64, gc: f64, length_scale: f64) -> f64 {
    2.0 * driving / (gc / length_scale + 2.0 * driving)
}

/// Invert the AT2 law: recover the driving energy from homogeneous damage.
fn driving_from_damage(damage: f64, gc: f64, length_scale: f64) -> f64 {
    gc / length_scale * damage / (2.0 * (1.0 - damage))
}

/// Unseeded homogeneous study at the requested extension (positive tension,
/// negative compression) on the fixed 8x4 introductory mesh.
fn homogeneous_study(extension_m: f64) -> PhaseFieldDamageStudy {
    let mut study = PhaseFieldDamageStudy::introductory();
    study.seed_history_energy_density_j_per_m3 = 0.0;
    study.prescribed_extension_m = extension_m;
    study
}

#[test]
fn phase_field_homogeneous_tension_matches_at2_law() {
    // Uniaxial tension fixes every principal strain, so the spectral
    // driving energy and the AT2 damage are both closed-form (measured
    // exact to 1e-12 relative; gate 1e-9). The crack energy carries only
    // the bulk term (uniform damage, no gradient) and the elastic energy
    // is the degraded plane-stress density (both measured exact to
    // print precision; gate 1e-9).
    let study = homogeneous_study(0.05);
    let solved = run_phase_field_damage(&study).expect("homogeneous tension run");
    assert!(solved.converged);
    assert_eq!(solved.damage_factorizations, study.load_steps);
    let (lambda, mu) = lame(study.youngs_modulus_pa, study.poissons_ratio);
    let nu = study.poissons_ratio;
    let strain = study.prescribed_extension_m / study.width_m;
    // Uniaxial principals (e, -nu e, -nu e), trace e(1-2nu) > 0.
    let driving = strain * strain * (0.5 * lambda * (1.0 - 2.0 * nu).powi(2) + mu);
    let gc = study.critical_energy_release_rate_j_per_m2;
    let length = study.length_scale_m;
    let expected = homogeneous_damage(driving, gc, length);
    assert!(
        ((solved.max_damage - expected) / expected).abs() < 1.0e-9,
        "AT2 law mismatch: {} vs {expected}",
        solved.max_damage
    );
    assert!(
        (solved.max_damage - solved.mean_damage).abs() < 1.0e-12,
        "damage not homogeneous: max {} mean {}",
        solved.max_damage,
        solved.mean_damage
    );
    assert!(
        (solved.damage_localization_ratio - 1.0).abs() < 1.0e-12,
        "localization ratio {} != 1",
        solved.damage_localization_ratio
    );
    let area = study.width_m * study.height_m;
    let expected_crack = gc * expected * expected / (2.0 * length) * area;
    assert!(
        ((solved.final_crack_surface_energy_j - expected_crack) / expected_crack).abs() < 1.0e-9,
        "crack energy {} vs {expected_crack}",
        solved.final_crack_surface_energy_j
    );
    let residual = study.residual_stiffness;
    let degradation = (1.0 - residual) * (1.0 - expected).powi(2) + residual;
    let expected_elastic = degradation * 0.5 * study.youngs_modulus_pa * strain * strain * area;
    assert!(
        ((solved.final_elastic_energy_j - expected_elastic) / expected_elastic).abs() < 1.0e-9,
        "elastic energy {} vs {expected_elastic}",
        solved.final_elastic_energy_j
    );
}

#[test]
fn phase_field_through_crack_matches_profile_and_griffith() {
    // A full-width seeded notch (no tip) develops the 1D AT2 optimal
    // profile: exponential tails with characteristic length l about a
    // fully damaged core (measured within 0.01 absolute of
    // exp(-(r-h)/l) over r in [2h, 2l]; gate 0.02, plus a 5% gate on
    // the fitted decay rate vs 1/l). The crack energy is Griffith's
    // Gc per unit length plus the seeded-band bulk term Gc*a*h/l
    // (measured ratio 1.103 vs predicted 1.125; gate [1.0, 1.2]).
    // l/h = 8 resolves the profile; the far tails past 2l feel the
    // plate edges and are excluded from the fit.
    let mut study = PhaseFieldDamageStudy::introductory();
    study.nx = 128;
    study.ny = 64;
    study.length_scale_m = 0.0625;
    study.seed_x_m = study.width_m;
    study.seed_half_width_m = 0.004;
    study.seed_history_energy_density_j_per_m3 = 1.0e8;
    study.prescribed_extension_m = 1.0e-4;
    study.load_steps = 2;
    let solved = run_phase_field_damage(&study).expect("through-crack run");
    assert_eq!(solved.damage_factorizations, study.load_steps);
    assert!(solved.history_monotone);
    assert!(solved.damage_monotone);
    assert!(solved.final_mechanics_residual < 1.0e-6);
    assert!(solved.final_free_equilibrium_residual < 1.0e-6);
    assert!(solved.final_damage_residual < 1.0e-6);
    // The discontinuous seed-history jump overshoots d = 1 by 3e-3 at
    // the P1 level (measured); the clamped damage used downstream is
    // unaffected, so this gate admits it instead of `converged`.
    assert!(solved.max_bound_violation < 5.0e-3);
    assert!(solved.max_damage > 0.999);
    let nx = study.nx;
    let ny = study.ny;
    let column = nx / 2;
    let dy = study.height_m / ny as f64;
    let mid = 0.5 * study.height_m;
    let length = study.length_scale_m;
    let damage_at = |j: usize| solved.nodal_damage[j * (nx + 1) + column];
    let radius_at = |j: usize| ((j as f64 * dy) - mid).abs();
    for j in 0..=ny {
        let radius = radius_at(j);
        if radius >= 2.0 * dy && radius <= 2.0 * length {
            let predicted = (-(radius - dy) / length).exp();
            let damage = damage_at(j);
            assert!(
                (damage - predicted).abs() < 0.02,
                "profile mismatch at r={radius}: {damage} vs {predicted}"
            );
        }
        // Symmetric seed and mesh give a symmetric profile.
        assert!(
            (damage_at(j) - damage_at(ny - j)).abs() < 1.0e-9,
            "asymmetric profile at row {j}"
        );
    }
    // Fitted exponential decay rate over [4h, 2l] vs 1/l.
    let (mut r0, mut d0) = (0.0, 0.0);
    let (mut r1, mut d1) = (0.0, 0.0);
    for j in 0..=ny / 2 {
        let radius = radius_at(j);
        if radius >= 4.0 * dy && radius <= 2.0 * length {
            if r0 == 0.0 {
                (r0, d0) = (radius, damage_at(j));
            }
            (r1, d1) = (radius, damage_at(j));
        }
    }
    let rate = -(d1 / d0).ln() / (r1 - r0);
    assert!(
        ((rate - 1.0 / length) * length).abs() < 0.05,
        "decay rate {rate} vs {}",
        1.0 / length
    );
    let ratio = solved.final_crack_surface_energy_j
        / (study.critical_energy_release_rate_j_per_m2 * study.width_m);
    assert!(
        (1.0..=1.2).contains(&ratio),
        "Griffith ratio {ratio} outside [1.0, 1.2]"
    );
}

#[test]
fn phase_field_compression_cracks_less_than_tension() {
    // The spectral split sends only positive principal strains to the
    // history: uniaxial compression still cracks (Poisson expansion is
    // tensile) but far less than tension. Both branches match their
    // closed form (measured exact to 1e-12 relative; gate 1e-9), the
    // inverted driving-energy ratio matches the split ratio 6.22
    // (measured exact to 1e-12; gate 1e-6), and tension damage exceeds
    // 4x compression damage (measured 5.1x).
    let tension = homogeneous_study(0.05);
    let solved_tension = run_phase_field_damage(&tension).expect("tension run");
    assert!(solved_tension.converged);
    let compression = homogeneous_study(-0.05);
    let solved_compression = run_phase_field_damage(&compression).expect("compression run");
    assert!(solved_compression.converged);
    let (lambda, mu) = lame(tension.youngs_modulus_pa, tension.poissons_ratio);
    let nu = tension.poissons_ratio;
    let strain = tension.prescribed_extension_m / tension.width_m;
    let gc = tension.critical_energy_release_rate_j_per_m2;
    let length = tension.length_scale_m;
    // Compression principals (-e, nu e, nu e), trace negative: only the
    // transverse Poisson terms drive damage.
    let expected_compression =
        homogeneous_damage(strain * strain * (2.0 * mu * nu * nu), gc, length);
    assert!(
        ((solved_compression.max_damage - expected_compression) / expected_compression).abs()
            < 1.0e-9,
        "compression law mismatch: {} vs {expected_compression}",
        solved_compression.max_damage
    );
    let expected_ratio = (0.5 * lambda * (1.0 - 2.0 * nu).powi(2) + mu) / (2.0 * mu * nu * nu);
    let measured_ratio = driving_from_damage(solved_tension.max_damage, gc, length)
        / driving_from_damage(solved_compression.max_damage, gc, length);
    assert!(
        ((measured_ratio - expected_ratio) / expected_ratio).abs() < 1.0e-6,
        "split ratio {measured_ratio} vs {expected_ratio}"
    );
    assert!(
        solved_tension.max_damage > 4.0 * solved_compression.max_damage,
        "weak asymmetry: tension {} vs compression {}",
        solved_tension.max_damage,
        solved_compression.max_damage
    );
}
