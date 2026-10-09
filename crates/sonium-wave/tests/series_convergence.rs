//! Audit `reviews/audit-sonium-wave-20260921.md`: truncation diagnostics.
//!
//! The audit requires stable, error-controlled series truncation with useful
//! validity domains before any batched-evaluation benchmarking. These tests
//! prove that [`suggested_sphere_terms`] and [`suggested_cylinder_terms`]
//! bound the series tail below `1e-9` relative error for `ka <= 20` on the
//! reference grid, and that the reference truncation itself sits on a
//! converged plateau (so the comparison cannot pass with two equally
//! under-resolved truncations). The `observation` bench may only cite
//! throughputs against this bounded-error regime.
//!
//! Measurement note: the pre-existing `ka + 10` sphere heuristic is
//! unconverged for `ka >= 2` at forward angles (`theta = 0`, where every
//! partial wave adds in phase): at `ka = 5` it leaves an O(1) relative tail,
//! and at `ka = 20` even 63 terms leave a 0.6 relative error. The cylinder
//! `2*ka + 10` rule converges everywhere measured (tails `<= 1e-13`).

use num_complex::Complex64;
use sonium_wave::analytical::{
    cylinder_scattering_2d, sphere_scattering_3d, suggested_cylinder_terms, suggested_sphere_terms,
    try_cylinder_scattering_2d, try_sphere_scattering_3d,
};
use std::f64::consts::PI;

const GRID_R: [f64; 3] = [1.5, 2.0, 3.0];
const GRID_THETA: [f64; 4] = [0.0, PI / 4.0, PI / 2.0, PI];
const KAS: [f64; 7] = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0];
const TAIL_TOLERANCE: f64 = 1e-9;
const PLATEAU_TOLERANCE: f64 = 1e-12;
const REFINEMENT: usize = 32;

fn worst_relative_difference(coarse: &[Complex64], fine: &[Complex64]) -> f64 {
    assert_eq!(coarse.len(), fine.len());
    let scale = fine
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max)
        .max(1e-30);
    coarse
        .iter()
        .zip(fine)
        .map(|(a, b)| (a - b).norm() / scale)
        .fold(0.0_f64, f64::max)
}

#[test]
fn suggested_terms_match_validated_rules() {
    assert_eq!(suggested_sphere_terms(0.5), 20);
    assert_eq!(suggested_sphere_terms(1.0), 20);
    assert_eq!(suggested_sphere_terms(2.0), 24);
    assert_eq!(suggested_sphere_terms(5.0), 36);
    assert_eq!(suggested_sphere_terms(20.0), 96);
    assert_eq!(suggested_cylinder_terms(0.5), 11);
    assert_eq!(suggested_cylinder_terms(1.0), 12);
    assert_eq!(suggested_cylinder_terms(2.0), 14);
    assert_eq!(suggested_cylinder_terms(5.0), 20);
    assert_eq!(suggested_cylinder_terms(20.0), 50);

    // Degenerate inputs saturate to the floor instead of truncating to zero.
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(suggested_sphere_terms(bad), 16, "ka={bad}");
        assert_eq!(suggested_cylinder_terms(bad), 10, "ka={bad}");
    }
}

#[test]
fn sphere_tail_is_bounded_for_validated_ka_range() {
    for ka in KAS {
        let suggested = suggested_sphere_terms(ka);
        let coarse =
            try_sphere_scattering_3d(ka, 1.0, suggested, GRID_R.to_vec(), GRID_THETA.to_vec())
                .expect("suggested truncation must evaluate");
        let fine = sphere_scattering_3d(
            ka,
            1.0,
            suggested + REFINEMENT,
            GRID_R.to_vec(),
            GRID_THETA.to_vec(),
        );
        let plateau = sphere_scattering_3d(
            ka,
            1.0,
            suggested + 2 * REFINEMENT,
            GRID_R.to_vec(),
            GRID_THETA.to_vec(),
        );
        assert_eq!(coarse.pressure.len(), GRID_R.len() * GRID_THETA.len());
        let tail = worst_relative_difference(&coarse.pressure, &fine.pressure);
        assert!(
            tail < TAIL_TOLERANCE,
            "sphere ka={ka}: suggested tail {tail:.3e} exceeds {TAIL_TOLERANCE:.0e}"
        );
        let drift = worst_relative_difference(&fine.pressure, &plateau.pressure);
        assert!(
            drift < PLATEAU_TOLERANCE,
            "sphere ka={ka}: reference not converged (drift {drift:.3e}); \
             widen REFINEMENT before trusting the tail bound"
        );
    }
}

#[test]
fn cylinder_tail_is_bounded_for_validated_ka_range() {
    for ka in KAS {
        let suggested = suggested_cylinder_terms(ka);
        let coarse =
            try_cylinder_scattering_2d(ka, 1.0, suggested, GRID_R.to_vec(), GRID_THETA.to_vec())
                .expect("suggested truncation must evaluate");
        let fine = cylinder_scattering_2d(
            ka,
            1.0,
            suggested + REFINEMENT,
            GRID_R.to_vec(),
            GRID_THETA.to_vec(),
        );
        let plateau = cylinder_scattering_2d(
            ka,
            1.0,
            suggested + 2 * REFINEMENT,
            GRID_R.to_vec(),
            GRID_THETA.to_vec(),
        );
        assert_eq!(coarse.pressure.len(), GRID_R.len() * GRID_THETA.len());
        let tail = worst_relative_difference(&coarse.pressure, &fine.pressure);
        assert!(
            tail < TAIL_TOLERANCE,
            "cylinder ka={ka}: suggested tail {tail:.3e} exceeds {TAIL_TOLERANCE:.0e}"
        );
        let drift = worst_relative_difference(&fine.pressure, &plateau.pressure);
        assert!(
            drift < PLATEAU_TOLERANCE,
            "cylinder ka={ka}: reference not converged (drift {drift:.3e}); \
             widen REFINEMENT before trusting the tail bound"
        );
    }
}
