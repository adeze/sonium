//! Mixed u-p validations: manufactured exactness, Lamé ladder at
//! near-incompressibility, and locking-free Poisson sweep.

// Rust guideline compliant 2026-02-21

use sonium_sim_structural_mechanics::{
    MixedUpAnnulusStudy, MixedUpMmsStudy, run_mixed_up_annulus, run_mixed_up_mms,
};

#[test]
fn manufactured_quadratic_fields_are_exact() {
    // P2 displacement reproduces the quadratic u and P1 the linear p, so
    // the discrete solution must match to solver tolerance — at both a
    // compressible and a nearly incompressible Poisson ratio.
    for nu in [0.3, 0.4999] {
        let mut study = MixedUpMmsStudy::introductory();
        study.material.poisson_ratio = nu;
        let result = run_mixed_up_mms(&study).expect("valid MMS run");
        assert!(result.converged);
        assert!(
            result.max_displacement_relative_error < 1.0e-9,
            "nu = {nu}: u error = {}",
            result.max_displacement_relative_error
        );
        assert!(
            result.max_pressure_relative_error < 1.0e-7,
            "nu = {nu}: p error = {}",
            result.max_pressure_relative_error
        );
    }
}

#[test]
fn lame_annulus_converges_at_nu_04999() {
    let grids = [(4, 32), (8, 64), (16, 128)];
    let mut errors = Vec::new();
    for (n_radial, n_angular) in grids {
        let mut study = MixedUpAnnulusStudy::introductory();
        study.geometry.n_radial = n_radial;
        study.geometry.n_angular = n_angular;
        study.material.poisson_ratio = 0.4999;
        let result = run_mixed_up_annulus(&study).expect("valid annulus run");
        assert!(result.converged);
        errors.push(result.max_displacement_relative_error);
    }
    for pair in errors.windows(2) {
        assert!(pair[1] < pair[0], "errors must decrease: {errors:?}");
    }
    let logs: Vec<f64> = errors.iter().map(|error| error.log2()).collect();
    let mean = logs.iter().sum::<f64>() / logs.len() as f64;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (index, &value) in logs.iter().enumerate() {
        numerator += (index as f64 - 1.0) * (value - mean);
        denominator += (index as f64 - 1.0).powi(2);
    }
    let slope = -numerator / denominator;
    assert!(slope >= 1.8, "slope = {slope}, errors = {errors:?}");
    assert!(errors[2] < 1.0e-3, "fine error = {}", errors[2]);
}

#[test]
fn lame_annulus_outer_pressure_matches_closed_form() {
    // Inner-pressure loading is covered by the ladder above; this exercises
    // the outer-pressure branch of the Lamé coefficients (a different exact
    // field: c1/c2 change sign structure) on the ladder's fine grid, so the
    // same 1e-3 accuracy gate transfers (smooth load, same regularity).
    let mut study = MixedUpAnnulusStudy::introductory();
    study.geometry.n_radial = 16;
    study.geometry.n_angular = 128;
    study.material.poisson_ratio = 0.4999;
    study.inner_pressure_pa = 0.0;
    study.outer_pressure_pa = 1.0e5;
    let result = run_mixed_up_annulus(&study).expect("valid annulus run");
    assert!(result.converged);
    assert!(
        result.max_displacement_relative_error < 1.0e-3,
        "outer-pressure error = {}",
        result.max_displacement_relative_error
    );
}

#[test]
fn poisson_sweep_stays_locking_free_through_incompressibility() {
    // The P1 displacement path degrades ~48x from nu = 0.3 to 0.4999 and
    // its solver fails on refined grids at nu >= 0.495. The mixed path
    // must stay flat — including exactly nu = 1/2 with pressure pinning.
    let mut errors = Vec::new();
    for nu in [0.3, 0.49, 0.499, 0.4999, 0.5] {
        let mut study = MixedUpAnnulusStudy::introductory();
        study.geometry.n_radial = 4;
        study.geometry.n_angular = 32;
        study.material.poisson_ratio = nu;
        let result = run_mixed_up_annulus(&study).expect("valid annulus run");
        assert!(result.converged, "nu = {nu} must converge");
        errors.push((nu, result.max_displacement_relative_error));
    }
    let min_error = errors.iter().map(|&(_, e)| e).fold(f64::INFINITY, f64::min);
    let max_error = errors.iter().map(|&(_, e)| e).fold(0.0f64, f64::max);
    assert!(
        max_error < 3.0 * min_error,
        "locking-free sweep must stay flat: {errors:?}"
    );
}

#[test]
fn invalid_mixed_up_inputs_are_rejected() {
    let mut study = MixedUpAnnulusStudy::introductory();
    study.material.poisson_ratio = 0.0;
    assert!(run_mixed_up_annulus(&study).is_err());
    study.material.poisson_ratio = 0.51;
    assert!(run_mixed_up_annulus(&study).is_err());
    study.material.poisson_ratio = 0.3;
    study.geometry.n_radial = 0;
    assert!(run_mixed_up_annulus(&study).is_err());

    let mut mms = MixedUpMmsStudy::introductory();
    mms.material.poisson_ratio = 0.5;
    assert!(run_mixed_up_mms(&mms).is_err());
    mms.material.poisson_ratio = 0.3;
    mms.n = 0;
    assert!(run_mixed_up_mms(&mms).is_err());
}
