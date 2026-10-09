//! Lumped moving-assembly surrogate validations: force-energy consistency,
//! contact/endstop penalty specs, and damage/softening closed forms.
//!
//! These pin the Stage 4 lumped speaker surrogate (fast, hermetic, exact):
//! the constitutive law behind the large-signal transient path, including
//! the voice-coil rub/endstop contact terms that mirror the 3D bracket
//! contact study at lumped scale.

// Rust guideline compliant 2026-09-28

use sonium_sim_speaker_modeling::{
    ModelingError, MovingAssemblyBreakupSpec, MovingAssemblyContactSpec, MovingAssemblyFatigueSpec,
    MovingAssemblyNonlinearity, MovingAssemblyViscoelasticSpec,
};

/// Representative hardening assembly: asymmetric suspension plus quadratic
/// hard stop, linear rub contact, and mild breakup softening.
fn sample_assembly() -> MovingAssemblyNonlinearity {
    MovingAssemblyNonlinearity {
        suspension_quadratic_n_per_m2: 4.0e4,
        suspension_cubic_n_per_m3: 2.5e6,
        hard_stop_excursion_m: Some(8.0e-3),
        hard_stop_stiffness_n_per_m: 1.2e6,
        viscoelastic: Some(MovingAssemblyViscoelasticSpec {
            damping_n_s_per_m2: 15.0,
        }),
        contact: Some(MovingAssemblyContactSpec {
            clearance_m: 6.0e-3,
            stiffness_n_per_m: 8.0e5,
            damping_n_s_per_m: 40.0,
        }),
        breakup: Some(MovingAssemblyBreakupSpec {
            onset_excursion_m: 9.0e-3,
            softening_stiffness_n_per_m: 3.0e4,
        }),
        fatigue: Some(MovingAssemblyFatigueSpec {
            reference_excursion_m: 5.0e-3,
            cycles_to_failure: 1.0e7,
            damage_exponent: 3.0,
        }),
    }
}

#[test]
fn speaker_lumped_force_is_energy_gradient() {
    // Conservative core: dE/dx must equal the static force everywhere it
    // is smooth (checked by central differences, 1e-6 relative) and the
    // energy must vanish at rest. (No positivity claim: the asymmetric
    // quadratic term makes the energy odd-indefinite near rest, which is
    // the correct physics of an asymmetric suspension.) Kink points
    // (clearances/onsets) are excluded: the force is C0 there.
    let assembly = sample_assembly();
    assembly.validate().expect("sample assembly valid");
    assert_eq!(assembly.potential_energy(0.0), 0.0);
    assert_eq!(assembly.nonlinear_force(0.0), 0.0);
    // Exact closed form at a suspension-only point (3 mm: stops clear).
    let x_check = 3.0e-3f64;
    let expected_energy = assembly.suspension_quadratic_n_per_m2 * x_check.powi(3) / 3.0
        + assembly.suspension_cubic_n_per_m3 * x_check.powi(4) / 4.0;
    assert!(
        (assembly.potential_energy(x_check) - expected_energy).abs()
            < 1.0e-12 * expected_energy.abs()
    );
    assert!(
        (assembly.potential_energy(-x_check) + expected_energy
            - 2.0 * assembly.suspension_cubic_n_per_m3 * x_check.powi(4) / 4.0)
            .abs()
            < 1.0e-12 * expected_energy.abs()
    );
    let step = 1.0e-7;
    for micrometers in [500, 1500, 3000, 5000, 6500, 7500, 8500, 9500, 11000] {
        for sign in [1.0, -1.0] {
            let x = sign * micrometers as f64 * 1e-6;
            let fd = (assembly.potential_energy(x + step) - assembly.potential_energy(x - step))
                / (2.0 * step);
            let force = assembly.nonlinear_force(x);
            let scale = fd.abs().max(force.abs()).max(1.0e-9);
            assert!(
                (fd - force).abs() < 1.0e-6 * scale,
                "dE/dx mismatch at x = {x:e}: {fd:e} vs {force:e}"
            );
            // Odd symmetry of the symmetric terms is broken only by the
            // quadratic suspension term, which is even: F(-x) + F(x) must
            // equal twice the quadratic part exactly.
            let mirror = assembly.nonlinear_force(-x);
            let expected_even = 2.0 * assembly.suspension_quadratic_n_per_m2 * x.powi(2);
            assert!(
                ((force + mirror) - expected_even).abs() < 1.0e-9 * expected_even.abs().max(1.0),
                "even/odd split wrong at x = {x:e}"
            );
        }
    }
}

#[test]
fn speaker_contact_endstop_matches_penalty_spec() {
    // Lumped contact/rub: exact deadband inside the clearance, linear
    // penalty outside (doubling checked exactly), velocity damping gated
    // by the same clearance, and C0 continuity at the clearance edge.
    // The hard stop is the quadratic counterpart (4x at double depth).
    let assembly = sample_assembly();
    let contact = assembly.contact.expect("sample has contact");
    // Deadband: identically zero force and penetration inside.
    for x in [0.0, 1.0e-3, 5.999e-3] {
        assert_eq!(assembly.contact_penetration_m(x), 0.0);
        assert_eq!(assembly.contact_penetration_m(-x), 0.0);
    }
    let contact_only = MovingAssemblyNonlinearity {
        suspension_quadratic_n_per_m2: 0.0,
        suspension_cubic_n_per_m3: 0.0,
        hard_stop_excursion_m: None,
        hard_stop_stiffness_n_per_m: 0.0,
        viscoelastic: None,
        contact: Some(contact),
        breakup: None,
        fatigue: None,
    };
    for x in [0.0, 1.0e-3, 5.999e-3] {
        assert_eq!(contact_only.nonlinear_force(x), 0.0);
        assert_eq!(contact_only.nonlinear_force(-x), 0.0);
        assert_eq!(contact_only.nonlinear_damping_force(x, 2.5), 0.0);
    }
    // Linear penalty outside: exact values and doubling.
    let depth = 1.5e-3;
    let x0 = contact.clearance_m + depth;
    let expected = contact.stiffness_n_per_m * depth;
    assert!(
        (contact_only.nonlinear_force(x0) - expected).abs() < 1.0e-9 * expected,
        "contact force value"
    );
    assert!(
        (contact_only.nonlinear_force(-x0) + expected).abs() < 1.0e-9 * expected,
        "contact force must be odd"
    );
    let double = contact_only.nonlinear_force(contact.clearance_m + 2.0 * depth);
    assert!(
        (double - 2.0 * expected).abs() < 1.0e-9 * expected,
        "contact penalty must be linear: {double} vs {}",
        2.0 * expected
    );
    assert!(
        (assembly.contact_penetration_m(x0) - depth).abs() < 1.0e-15,
        "penetration value"
    );
    // Damping gated by the clearance.
    assert_eq!(
        contact_only.nonlinear_damping_force(x0, 2.5),
        contact.damping_n_s_per_m * 2.5
    );
    // C0 continuity at the edge (no force jump on touch-down).
    let edge = contact_only.nonlinear_force(contact.clearance_m + 1.0e-12);
    assert!(
        edge.abs() < contact.stiffness_n_per_m * 2.0e-12,
        "contact force must touch down continuously: {edge:e}"
    );
    // Hard stop: quadratic penalty, odd, exact doubling law (4x).
    let stop_only = MovingAssemblyNonlinearity {
        suspension_quadratic_n_per_m2: 0.0,
        suspension_cubic_n_per_m3: 0.0,
        hard_stop_excursion_m: assembly.hard_stop_excursion_m,
        hard_stop_stiffness_n_per_m: assembly.hard_stop_stiffness_n_per_m,
        viscoelastic: None,
        contact: None,
        breakup: None,
        fatigue: None,
    };
    let stop = assembly.hard_stop_excursion_m.unwrap();
    assert_eq!(stop_only.nonlinear_force(stop * 0.99), 0.0);
    let f1 = stop_only.nonlinear_force(stop + depth);
    let expected_stop = assembly.hard_stop_stiffness_n_per_m * depth.powi(2);
    assert!((f1 - expected_stop).abs() < 1.0e-9 * expected_stop);
    let f2 = stop_only.nonlinear_force(stop + 2.0 * depth);
    assert!(
        (f2 - 4.0 * expected_stop).abs() < 1.0e-9 * expected_stop,
        "hard stop must be quadratic: {f2} vs {}",
        4.0 * expected_stop
    );
    assert_eq!(f1 + stop_only.nonlinear_force(-(stop + depth)), 0.0);
}

#[test]
fn speaker_lumped_damage_and_softening_laws() {
    // Miner fatigue: exact closed form (unity at the reference life),
    // cubic scaling with excursion, linearity in cycles. Breakup:
    // linear softening past onset, strict activation boundary, and
    // energy consistency of the softening term. Guards lock the
    // hardening/clearance assumptions.
    let assembly = sample_assembly();
    let fatigue = assembly.fatigue.expect("sample has fatigue");
    assert_eq!(
        assembly.fatigue_damage(fatigue.reference_excursion_m, fatigue.cycles_to_failure),
        1.0,
        "reference life must give unit damage"
    );
    let base = assembly.fatigue_damage(2.5e-3, 1.0e5);
    let doubled = assembly.fatigue_damage(5.0e-3, 1.0e5);
    assert!(
        (doubled / base - 8.0).abs() < 1.0e-12,
        "cubic excursion scaling: {doubled} vs {base}"
    );
    let more_cycles = assembly.fatigue_damage(2.5e-3, 3.0e5);
    assert!(
        (more_cycles / base - 3.0).abs() < 1.0e-12,
        "linear cycle scaling"
    );
    assert_eq!(assembly.fatigue_damage(0.0, 1.0e5), 0.0);
    // Breakup softening: exact linear law past onset, inactive at it.
    let breakup = assembly.breakup.expect("sample has breakup");
    assert!(!assembly.breakup_active(breakup.onset_excursion_m));
    assert!(assembly.breakup_active(breakup.onset_excursion_m + 1.0e-12));
    let soft_only = MovingAssemblyNonlinearity {
        suspension_quadratic_n_per_m2: 0.0,
        suspension_cubic_n_per_m3: 0.0,
        hard_stop_excursion_m: None,
        hard_stop_stiffness_n_per_m: 0.0,
        viscoelastic: None,
        contact: None,
        breakup: Some(breakup),
        fatigue: None,
    };
    let depth = 2.0e-3;
    let expected = -breakup.softening_stiffness_n_per_m * depth;
    let soft = soft_only.nonlinear_force(breakup.onset_excursion_m + depth);
    assert!(
        (soft - expected).abs() < 1.0e-12 * expected.abs(),
        "softening force value: {soft} vs {expected}"
    );
    // Softening energy consistent (FD in the active interior).
    let x = breakup.onset_excursion_m + depth;
    let step = 1.0e-8;
    let fd = (soft_only.potential_energy(x + step) - soft_only.potential_energy(x - step))
        / (2.0 * step);
    assert!(
        (fd - soft).abs() < 1.0e-6 * soft.abs(),
        "softening dE/dx: {fd:e} vs {soft:e}"
    );
    // Viscoelastic damping: exact c*|x|*v law.
    let viscoelastic = assembly.viscoelastic.expect("sample has viscoelastic");
    assert_eq!(
        assembly.nonlinear_damping_force(4.0e-3, 1.5),
        viscoelastic.damping_n_s_per_m2 * 4.0e-3 * 1.5
    );
    // Guards: negative cubic (softening), dead contact, bad clearance.
    let mut bad = sample_assembly();
    bad.suspension_cubic_n_per_m3 = -1.0;
    assert!(matches!(
        bad.validate(),
        Err(ModelingError::OutOfRange { .. })
    ));
    let mut dead = sample_assembly();
    dead.contact = Some(MovingAssemblyContactSpec {
        clearance_m: 1.0e-3,
        stiffness_n_per_m: 0.0,
        damping_n_s_per_m: 0.0,
    });
    assert!(matches!(
        dead.validate(),
        Err(ModelingError::OutOfRange { .. })
    ));
    let mut clearance = sample_assembly();
    clearance.contact = Some(MovingAssemblyContactSpec {
        clearance_m: 0.0,
        stiffness_n_per_m: 1.0,
        damping_n_s_per_m: 0.0,
    });
    assert!(matches!(
        clearance.validate(),
        Err(ModelingError::OutOfRange { .. })
    ));
}
