//! Nonlinear shell validations: low-deformation linear limit with quadratic
//! membrane-onset scaling, large-deflection strip against the beam elastica,
//! and Southwell extraction of the simply-supported plate buckling load.
//!
//! All three compare the corotational shell study against exact theories, and
//! a fourth test pins the plate references themselves to closed forms so a
//! sign or scale slip in a reference cannot hide behind a passing suite.

// Rust guideline compliant 2026-09-28

use sonium_sim_structural_mechanics::{
    NonlinearShellBucklingStudy, NonlinearShellPressureStudy, NonlinearShellStripStudy,
    ShellStaticStudy, elastica_cantilever_tip, plate_bending_stiffness_nm,
    run_nonlinear_shell_buckling, run_nonlinear_shell_pressure, run_nonlinear_shell_strip,
    run_shell_static, simply_supported_plate_buckling_load_n_per_m,
};

/// Max-norm field difference between the nonlinear and linear pressure
/// solutions, relative to the linear field scale.
fn pressure_field_difference(pressure_pa: f64) -> (f64, usize) {
    let mut nonlinear = NonlinearShellPressureStudy::introductory();
    nonlinear.pressure_pa = pressure_pa;
    nonlinear.load_steps = 3;
    let solved = run_nonlinear_shell_pressure(&nonlinear).expect("converged pressure run");
    assert!(solved.converged);
    let linear_study = ShellStaticStudy {
        name: "nonlinear-shell-linear-match".to_string(),
        geometry: nonlinear.geometry,
        material: nonlinear.material,
        pressure_pa,
    };
    let linear = run_shell_static(&linear_study).expect("linear pressure run");
    let mut max_diff = 0.0f64;
    let mut max_scale = 0.0f64;
    for (a, b) in solved.displacement.iter().zip(linear.displacement.iter()) {
        for c in 0..6 {
            max_diff = max_diff.max((a[c] - b[c]).abs());
            max_scale = max_scale.max(b[c].abs());
        }
    }
    (max_diff / max_scale, solved.total_newton_iterations)
}

#[test]
fn nonlinear_shell_matches_linear_plate_at_low_pressure() {
    // At w/t ~ 0.2% the von Karman membrane correction is O((w/t)^2) ~ 1e-6,
    // so the corotational solve must reproduce the linear DKT field. The
    // 1e-4 gate sits ~2.6x above the measured 3.8e-5 deviation. The deviation
    // itself must scale quadratically with load (field_rel/p constant),
    // proving it is genuine membrane stiffening rather than a linear bias.
    let mut slope = Vec::new();
    for pressure in [0.05, 0.1, 0.2] {
        let (field_rel, _) = pressure_field_difference(pressure);
        slope.push(field_rel / pressure);
        if pressure <= 0.1 {
            assert!(
                field_rel < 1.0e-4,
                "p = {pressure}: field deviation = {field_rel:e}"
            );
        }
    }
    for window in slope.windows(2) {
        let ratio = window[1] / window[0];
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "membrane deviation is not quadratic in load: {slope:?}"
        );
    }
}

#[test]
fn nonlinear_shell_strip_matches_elastica() {
    // Cantilever strip at PL^2/EI = 1 and 2 (tip deflection L/3 and L/2).
    // The tip-center node is the beam comparison point (anticlastic warping
    // vanishes on the centerline). The 2% gate covers the O((b/L)^2)
    // plate-vs-beam model gap at p = 1 (measured 1.58%; an L = 2 probe drops
    // it to 0.36%, confirming slender convergence to beam theory); p = 2
    // exercises ~40-degree tip slopes at 0.06%.
    for load_parameter in [1.0, 2.0] {
        let proto = NonlinearShellStripStudy::introductory();
        let width = proto.geometry.width_m;
        let length = proto.geometry.length_m;
        let ei =
            proto.material.youngs_modulus_pa * width * proto.material.thickness_m.powi(3) / 12.0;
        let total = load_parameter * ei / (length * length);
        let exact = elastica_cantilever_tip(total, length, ei).expect("valid elastica load");

        let mut study = NonlinearShellStripStudy::introductory();
        study.geometry.nx = 40;
        study.geometry.ny = 4;
        study.tip_line_load_n_per_m = -total / width;
        study.load_steps = 10;
        let result = run_nonlinear_shell_strip(&study).expect("converged strip run");
        assert!(result.converged);

        let error = (result.tip_center_w_m - exact.y_m).abs() / exact.y_m.abs();
        assert!(
            error < 2.0e-2,
            "p = {load_parameter}: tip error = {error:e}"
        );
    }
}

#[test]
fn nonlinear_shell_southwell_recovers_plate_buckling_load() {
    // Imperfect simply-supported plate (3 Pa pressure seeds w/t ~ 0.04):
    // Southwell plot of center deflection w against N*w is linear with
    // slope 1/Ncr. The 5% gate sits ~5x above the measured 1.0% deviation
    // from the classical k = 4 load; R^2 guards the imperfection regime
    // (membrane stiffening would curve the line).
    let proto = NonlinearShellBucklingStudy::introductory();
    let d = plate_bending_stiffness_nm(
        proto.material.youngs_modulus_pa,
        proto.material.poisson_ratio,
        proto.material.thickness_m,
    );
    let classical = simply_supported_plate_buckling_load_n_per_m(
        d,
        proto.geometry.length_m,
        proto.geometry.width_m,
    );
    let mut points = Vec::new();
    for ratio in [0.4, 0.6, 0.8] {
        let mut study = NonlinearShellBucklingStudy::introductory();
        study.compression_n_per_m = ratio * classical;
        study.pressure_pa = 3.0;
        study.load_steps = 10;
        let result = run_nonlinear_shell_buckling(&study).expect("converged buckling run");
        assert!(result.converged);
        points.push((ratio * classical * result.center_w_m, result.center_w_m));
    }
    let n = points.len() as f64;
    let mean_x = points.iter().map(|(x, _)| x).sum::<f64>() / n;
    let mean_y = points.iter().map(|(_, y)| y).sum::<f64>() / n;
    let mut sxx = 0.0f64;
    let mut sxy = 0.0f64;
    let mut syy = 0.0f64;
    for (x, y) in &points {
        sxx += (x - mean_x) * (x - mean_x);
        sxy += (x - mean_x) * (y - mean_y);
        syy += (y - mean_y) * (y - mean_y);
    }
    let slope = sxy / sxx;
    let r_squared = sxy * sxy / (sxx * syy);
    let extracted = 1.0 / slope;
    let error = (extracted - classical).abs() / classical;
    assert!(
        error < 5.0e-2,
        "Southwell Ncr = {extracted:.6e}, classical = {classical:.6e}"
    );
    assert!(r_squared > 0.999, "Southwell R^2 = {r_squared:.6}");
}

#[test]
fn nonlinear_shell_plate_references_match_closed_forms() {
    // Pin the V3 references: D = Et^3/12(1-nu^2) and the square-plate
    // k = 4 buckling load Ncr = 4 pi^2 D / b^2.
    let e = 200.0e9;
    let nu = 0.3;
    let t = 0.002;
    let d = plate_bending_stiffness_nm(e, nu, t);
    let closed_d = e * t.powi(3) / (12.0 * (1.0 - nu * nu));
    assert!(
        (d - closed_d).abs() / closed_d < 1.0e-12,
        "bending stiffness mismatch"
    );
    let ncr = simply_supported_plate_buckling_load_n_per_m(d, 1.0, 1.0);
    let closed_ncr = 4.0 * std::f64::consts::PI.powi(2) * d / 1.0f64.powi(2);
    assert!(
        (ncr - closed_ncr).abs() / closed_ncr < 1.0e-12,
        "buckling load mismatch"
    );
}

#[test]
fn invalid_nonlinear_shell_inputs_are_rejected() {
    let mut pressure = NonlinearShellPressureStudy::introductory();
    pressure.pressure_pa = f64::NAN;
    assert!(run_nonlinear_shell_pressure(&pressure).is_err());
    pressure.pressure_pa = 0.1;
    pressure.load_steps = 0;
    assert!(run_nonlinear_shell_pressure(&pressure).is_err());
    pressure.load_steps = 3;
    pressure.newton_residual_tolerance = 0.0;
    assert!(run_nonlinear_shell_pressure(&pressure).is_err());

    let mut strip = NonlinearShellStripStudy::introductory();
    strip.tip_line_load_n_per_m = f64::NAN;
    assert!(run_nonlinear_shell_strip(&strip).is_err());
    strip.tip_line_load_n_per_m = -1.0;
    strip.geometry.nx = 0;
    assert!(run_nonlinear_shell_strip(&strip).is_err());

    let mut buckling = NonlinearShellBucklingStudy::introductory();
    buckling.compression_n_per_m = -1.0;
    assert!(run_nonlinear_shell_buckling(&buckling).is_err());
    buckling.compression_n_per_m = 100.0;
    buckling.pressure_pa = f64::NAN;
    assert!(run_nonlinear_shell_buckling(&buckling).is_err());
}
