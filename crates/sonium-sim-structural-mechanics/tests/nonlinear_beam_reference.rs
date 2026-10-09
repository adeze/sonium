//! Nonlinear beam validations: elastica cantilever, beam-column response
//! with Euler-load extraction, and the low-deformation linear limit.
//!
//! All three compare the corotational study against exact theories, and a
//! fourth test pins the references themselves to their linear limits so a
//! sign or scale slip in a reference cannot hide behind a passing suite.

// Rust guideline compliant 2026-02-21

use sonium_sim_structural_mechanics::{
    Beam2DStudy, NonlinearBeamStudy, beam_column_cantilever_tip, elastica_cantilever_tip,
    euler_cantilever_buckling_load, run_beam2d_static, run_nonlinear_beam,
};

fn cantilever_fixture() -> NonlinearBeamStudy {
    NonlinearBeamStudy::introductory()
}

#[test]
fn elastica_cantilever_matches_shooting_reference() {
    // Large-deflection tip loads at PL²/EI = 1 and 3 (tip slope up to ~57°).
    // The 1e-3 gate sits ~6x above the measured 1.5e-4 discretization error.
    for load_parameter in [1.0, 3.0] {
        let proto = cantilever_fixture();
        let ei = proto.bending_stiffness_nm2();
        let length = proto.geometry.length_m;
        let load = load_parameter * ei / (length * length);
        let exact = elastica_cantilever_tip(load, length, ei).expect("valid elastica load");

        let mut study = cantilever_fixture();
        study.tip_transverse_load_n = -load;
        study.load_steps = 20;
        let result = run_nonlinear_beam(&study).expect("converged elastica run");
        assert!(result.converged);

        let tip_x = length + result.tip_ux_m;
        let x_error = (tip_x - exact.x_m).abs() / length;
        let y_error = (result.tip_uy_m - exact.y_m).abs() / exact.y_m.abs();
        let slope_error =
            (result.tip_rotation_rad - exact.tip_slope_rad).abs() / exact.tip_slope_rad.abs();
        assert!(
            x_error < 1.0e-3,
            "p = {load_parameter}: x error = {x_error}"
        );
        assert!(
            y_error < 1.0e-3,
            "p = {load_parameter}: y error = {y_error}"
        );
        assert!(
            slope_error < 1.0e-3,
            "p = {load_parameter}: slope error = {slope_error}"
        );
    }
}

#[test]
fn beam_column_matches_closed_form_and_extracts_euler_load() {
    // Subcritical beam-column response plus a closed-form SDOF extraction
    // of the Euler load: δ = δ₀/(1 − P/C) through two load levels gives
    // C = (δ₁P₁ − δ₂P₂)/(δ₁ − δ₂). Southwell's δ/P-vs-δ plot is unusable
    // here (wrong-signed slope even on the exact beam-column curve for tip
    // loading), hence the direct two-point fit.
    let proto = cantilever_fixture();
    let ei = proto.bending_stiffness_nm2();
    let length = proto.geometry.length_m;
    let critical = euler_cantilever_buckling_load(length, ei);
    // Transverse load with linear dy/L ~ 3e-4: small-rotation regime where
    // the linearized beam-column theory applies.
    let transverse = 1.0e-3 * ei / length.powi(2);

    let tip_at = |axial: f64| -> f64 {
        let mut study = cantilever_fixture();
        study.tip_axial_load_n = -axial;
        study.tip_transverse_load_n = -transverse;
        study.load_steps = 10;
        run_nonlinear_beam(&study)
            .expect("converged beam-column run")
            .tip_uy_m
    };

    for ratio in [0.3, 0.6] {
        let axial = ratio * critical;
        let exact = beam_column_cantilever_tip(axial, transverse, length, ei)
            .expect("subcritical beam-column load");
        let fem = tip_at(axial);
        let error = (fem - exact).abs() / exact.abs();
        assert!(
            error < 1.0e-3,
            "P/Pcr = {ratio}: beam-column error = {error}"
        );
    }

    let first = 0.5 * critical;
    let second = 0.8 * critical;
    let first_tip = tip_at(first);
    let second_tip = tip_at(second);
    let extracted = (first_tip * first - second_tip * second) / (first_tip - second_tip);
    let error = (extracted - critical).abs() / critical;
    assert!(
        error < 0.02,
        "extracted Euler load {extracted} vs {critical}: error = {error}"
    );
}

#[test]
fn low_deformation_matches_linear_theory() {
    // At dy/L ~ 3e-4 geometric nonlinearity is O((dy/L)²) ~ 1e-7, so the
    // corotational tip must reproduce both the closed-form PL³/3EI and the
    // linear beam study on the same mesh. Gate 5e-6 is ~40x the measured
    // 1.1e-7 deviation.
    let proto = cantilever_fixture();
    let ei = proto.bending_stiffness_nm2();
    let length = proto.geometry.length_m;
    let load = 1.0e-3 * ei / length.powi(2);

    let mut nonlinear = cantilever_fixture();
    nonlinear.tip_transverse_load_n = -load;
    nonlinear.load_steps = 2;
    let result = run_nonlinear_beam(&nonlinear).expect("converged small-load run");
    assert!(result.converged);

    let linear_closed_form = -load * length.powi(3) / (3.0 * ei);
    let closed_form_error = (result.tip_uy_m - linear_closed_form).abs() / linear_closed_form.abs();
    assert!(
        closed_form_error < 5.0e-6,
        "closed-form error = {closed_form_error}"
    );

    // The linear leg runs on 8 elements: its CG solver stalls on the
    // slender 32-element mesh (axial/bending conditioning), and Hermite
    // tip values are mesh-exact for point-loaded Euler-Bernoulli beams,
    // so the comparison stays valid.
    let mut linear_geometry = nonlinear.geometry;
    linear_geometry.elements = 8;
    let linear_study = Beam2DStudy {
        name: "linear-match-fixture".to_string(),
        geometry: linear_geometry,
        material: nonlinear.material,
        vertical_tip_force_n: -load,
        timoshenko: false,
        modal_modes: 3,
    };
    let linear = run_beam2d_static(&linear_study).expect("converged linear run");
    let study_error = (result.tip_uy_m - linear.tip_deflection_m).abs() / linear_closed_form.abs();
    assert!(study_error < 5.0e-6, "linear-study error = {study_error}");
}

#[test]
fn references_recover_linear_limits() {
    // The references themselves must collapse to linear theory: the
    // elastica carries a genuine O(p) nonlinear correction (~1e-5 at
    // p = 0.01), while the beam-column P = 0 branch is exactly linear.
    let proto = cantilever_fixture();
    let ei = proto.bending_stiffness_nm2();
    let length = proto.geometry.length_m;

    let small = 0.01 * ei / (length * length);
    let tip = elastica_cantilever_tip(small, length, ei).expect("valid elastica load");
    let linear = -small * length.powi(3) / (3.0 * ei);
    assert!(
        (tip.y_m - linear).abs() / linear.abs() < 5.0e-5,
        "elastica small-load deviation = {}",
        (tip.y_m - linear).abs() / linear.abs()
    );

    let transverse = 1.0;
    let zero_axial =
        beam_column_cantilever_tip(0.0, transverse, length, ei).expect("zero axial load is linear");
    let linear_tip = -transverse * length.powi(3) / (3.0 * ei);
    assert!(
        (zero_axial - linear_tip).abs() / linear_tip.abs() < 1.0e-12,
        "P = 0 must be exactly linear"
    );
}

#[test]
fn invalid_nonlinear_beam_inputs_are_rejected() {
    let mut study = cantilever_fixture();
    study.geometry.elements = 0;
    assert!(run_nonlinear_beam(&study).is_err());
    study.geometry.elements = 65;
    assert!(run_nonlinear_beam(&study).is_err());
    study.geometry.elements = 8;
    study.geometry.length_m = -1.0;
    assert!(run_nonlinear_beam(&study).is_err());
    study.geometry.length_m = 1.0;
    study.tip_transverse_load_n = f64::NAN;
    assert!(run_nonlinear_beam(&study).is_err());
    study.tip_transverse_load_n = 0.0;
    study.load_steps = 0;
    assert!(run_nonlinear_beam(&study).is_err());
    study.load_steps = 2;
    study.newton_residual_tolerance = 0.0;
    assert!(run_nonlinear_beam(&study).is_err());

    let proto = cantilever_fixture();
    let ei = proto.bending_stiffness_nm2();
    assert!(elastica_cantilever_tip(-1.0, 1.0, ei).is_err());
    assert!(elastica_cantilever_tip(1.0, 0.0, ei).is_err());
    let critical = euler_cantilever_buckling_load(1.0, ei);
    assert!(beam_column_cantilever_tip(critical, 1.0, 1.0, ei).is_err());
    assert!(beam_column_cantilever_tip(2.0 * critical, 1.0, 1.0, ei).is_err());
    assert!(beam_column_cantilever_tip(1.0, 1.0, 1.0, 0.0).is_err());
}
