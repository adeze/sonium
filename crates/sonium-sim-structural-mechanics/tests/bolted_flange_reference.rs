//! Bolted-flange 3D contact validations: preload linearity and clamp
//! equilibrium, tensile joint-diagram separation, and shear stick response
//! with a rigorous Coulomb-knee bracket.
//!
//! All three run the full-3D augmented-Lagrangian contact study (one-sided
//! node-to-surface pairing, regularized Coulomb friction) on a fixed
//! hermetic mesh and compare against exact joint theory plus an
//! independently measured contact resultant (not the bolt bookkeeping).

// Rust guideline compliant 2026-09-28

use sonium_sim_structural_mechanics::{BoltedFlangeStudy, run_bolted_flange};

/// Fixed hermetic study: 8x4 flange/base blocks (270 nodes), 2 preload steps.
fn flange_probe_study() -> BoltedFlangeStudy {
    let mut study = BoltedFlangeStudy::introductory();
    study.geometry.nx = 8;
    study.geometry.ny = 4;
    study.geometry.nz_flange = 2;
    study.geometry.nz_base = 2;
    study.preload_steps = 2;
    study.load_steps = 2;
    study
}

#[test]
fn bolted_flange_preload_is_linear_and_equilibrated() {
    // Augmented-Lagrangian contact of a linear elastic body is positively
    // homogeneous: doubling the pretension shortening must double the bolt
    // force exactly (measured 2e-7 relative; gate 1e-4). Zero external
    // load must leave the preload state untouched (measured exact to print
    // precision; gate 1e-5). The 1.6% deficit vs nominal preload is real
    // elastic sharing (plate bending on the discrete contact foundation);
    // the 0.025 gate bounds it without claiming a closed form.
    let study = flange_probe_study();
    let solved = run_bolted_flange(&study).expect("converged preload run");
    assert!(solved.converged);
    assert_eq!(solved.steps.len(), 4);
    let bolt1 = solved.steps[0].bolt_force_n;
    let bolt2 = solved.steps[1].bolt_force_n;
    assert!(
        ((2.0 * bolt1 - bolt2) / bolt2).abs() < 1.0e-4,
        "preload not linear: bolt1 = {bolt1}, bolt2 = {bolt2}"
    );
    for step in &solved.steps[2..] {
        assert!(
            ((step.bolt_force_n - bolt2) / bolt2).abs() < 1.0e-5,
            "zero load changed preload: {} vs {bolt2}",
            step.bolt_force_n
        );
    }
    assert!(
        ((bolt2 - study.preload_force_n) / study.preload_force_n).abs() < 0.025,
        "preload sharing out of bound: {bolt2} vs nominal {}",
        study.preload_force_n
    );
    for step in &solved.steps {
        // One-sided pairing on this mesh detects 72 active pairs; a
        // two-sided regression would read back 144.
        assert_eq!(step.active_pairs, 72, "step {} pair count", step.step);
        assert!(
            step.max_penetration_m < 1.0e-9,
            "step {} penetration {}",
            step.step,
            step.max_penetration_m
        );
        // Independent equilibrium: measured contact resultant vs bookkeeping.
        let clamp_err = (step.contact_clamp_n - step.clamp_force_n).abs();
        assert!(
            clamp_err < 0.5 + 2.0e-3 * step.clamp_force_n.abs(),
            "step {} clamp mismatch: {} vs {}",
            step.step,
            step.contact_clamp_n,
            step.clamp_force_n
        );
        assert!(
            step.flange_ux_m.abs() < 1.0e-6,
            "step {} lateral drift {}",
            step.step,
            step.flange_ux_m
        );
    }
}

#[test]
fn bolted_flange_joint_diagram_separates_at_predicted_load() {
    // Stiff-clamp joint: the solid-block stack (~8e10 N/m) dwarfs the bolt
    // (1e8 N/m), so the load factor Phi = dBolt/dTensile is ~0 (measured
    // -0.002; gate +-0.02) and separation lands at F_pre / (1 - Phi),
    // bracketed here by the 15 kN (attached) and 20 kN (lifted) steps.
    // Past separation the bolt must carry the full tensile load exactly
    // (measured 0.04 N off at 20 kN; gate 1 N): only two vertical forces
    // remain on the flange.
    let mut study = flange_probe_study();
    study.load_steps = 4;
    study.external_tensile_n = 20.0e3;
    let solved = run_bolted_flange(&study).expect("converged tensile run");
    assert!(solved.converged);
    assert_eq!(solved.steps.len(), 6);
    let preload_bolt = solved.steps[1].bolt_force_n;
    let phi = (solved.steps[2].bolt_force_n - preload_bolt) / 5.0e3;
    assert!(
        phi.abs() < 0.02,
        "joint load factor out of stiff-clamp bound: {phi}"
    );
    let predicted_separation = preload_bolt / (1.0 - phi);
    assert!(
        predicted_separation > 15.0e3 && predicted_separation <= 20.0e3,
        "separation load outside the measured bracket: {predicted_separation}"
    );
    let attached = &solved.steps[4];
    assert!(
        attached.clamp_force_n > 1.0e3 && attached.active_pairs > 0,
        "15 kN step should still be attached: clamp = {}, pairs = {}",
        attached.clamp_force_n,
        attached.active_pairs
    );
    let lifted = &solved.steps[5];
    assert_eq!(lifted.active_pairs, 0, "20 kN step should be lifted");
    assert!(
        lifted.clamp_force_n.abs() < 5.0,
        "lifted clamp should vanish: {}",
        lifted.clamp_force_n
    );
    assert!(
        lifted.contact_clamp_n.abs() < 0.5,
        "measured lifted clamp should vanish: {}",
        lifted.contact_clamp_n
    );
    assert!(
        (lifted.bolt_force_n - 20.0e3).abs() < 1.0,
        "lifted bolt must carry full tensile: {}",
        lifted.bolt_force_n
    );
    for step in &solved.steps {
        let clamp_err = (step.contact_clamp_n - step.clamp_force_n).abs();
        assert!(
            clamp_err < 0.5 + 2.0e-3 * step.clamp_force_n.abs(),
            "step {} clamp mismatch: {} vs {}",
            step.step,
            step.contact_clamp_n,
            step.clamp_force_n
        );
        assert!(
            step.max_penetration_m < 1.0e-9,
            "step {} penetration {}",
            step.step,
            step.max_penetration_m
        );
    }
}

#[test]
fn bolted_flange_shear_stick_and_coulomb_knee() {
    // Stick regime (U <= 0.95): friction carries the full shear (measured
    // within 13.3 N at 5.6 kN; gate 25 N + 2e-3 of load for spring/residual
    // slack), transverse drift stays sub-micron (measured <= 0.86 um;
    // gate 1.5 um), and the incremental compliance grows toward onset
    // (partial-slip softening: measured window ratio 1.54; gate 1.2).
    // The knee itself is bracketed rigorously: no sub-5 um equilibrium
    // can exist at the next (U = 1.11) step because friction caps at mu*N
    // and the lateral springs are the only other x load path (elastic
    // forces cancel globally), so any U = 1.11 equilibrium needs
    // >= 0.6 mm drift (gate 0.5 mm) while U = 0.949 holds <= 0.9 um.
    let mut study = flange_probe_study();
    study.load_steps = 6;
    study.transverse_shear_n = 5.6e3;
    let solved = run_bolted_flange(&study).expect("converged shear run");
    assert!(solved.converged);
    assert_eq!(solved.steps.len(), 8);
    let mu = study.friction_coefficient;
    for step in &solved.steps {
        let shear = if step.preload_phase {
            0.0
        } else {
            study.transverse_shear_n * step.load_factor
        };
        assert!(
            (step.contact_shear_x_n + shear).abs() < 25.0 + 2.0e-3 * shear,
            "step {} shear mismatch: {} vs {}",
            step.step,
            step.contact_shear_x_n,
            -shear
        );
        assert!(
            step.contact_shear_y_n.abs() < 2.0,
            "step {} transverse shear {}",
            step.step,
            step.contact_shear_y_n
        );
        assert!(
            step.flange_ux_m.abs() < 1.5e-6,
            "step {} drift {}",
            step.step,
            step.flange_ux_m
        );
        if !step.preload_phase {
            let utilization = shear / (mu * step.clamp_force_n);
            assert!(
                (step.coulomb_utilization - utilization).abs() < 1.0e-9,
                "step {} utilization formula: {} vs {utilization}",
                step.step,
                step.coulomb_utilization
            );
        }
        let clamp_err = (step.contact_clamp_n - step.clamp_force_n).abs();
        assert!(
            clamp_err < 0.5 + 2.0e-3 * step.clamp_force_n.abs(),
            "step {} clamp mismatch: {} vs {}",
            step.step,
            step.contact_clamp_n,
            step.clamp_force_n
        );
        assert!(
            step.max_penetration_m < 1.0e-9,
            "step {} penetration {}",
            step.step,
            step.max_penetration_m
        );
        assert!(step.active_pairs > 0, "step {} lost contact", step.step);
    }
    // Superlinearity over equal load windows (baseline-free differences).
    let ux = |index: usize| solved.steps[index].flange_ux_m;
    let early = ux(4) - ux(2);
    let late = ux(7) - ux(5);
    assert!(
        early > 0.0 && late / early > 1.2,
        "stick compliance should soften toward onset: {early:e} vs {late:e}"
    );
    // Analytic knee bracket (see header comment for the proof).
    let last = &solved.steps[7];
    assert!(last.flange_ux_m.abs() < 1.5e-6);
    let lateral_spring = study.material.bolt_stiffness_n_per_m * 1.0e-2;
    let next_shear = study.transverse_shear_n * 7.0 / 6.0;
    let required_drift = (next_shear - mu * last.clamp_force_n) / lateral_spring;
    assert!(
        required_drift > 5.0e-4,
        "U = 1.11 equilibrium must sit past 0.5 mm drift: {required_drift:e}"
    );
}
