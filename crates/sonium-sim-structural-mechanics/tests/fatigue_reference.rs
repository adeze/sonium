//! Calibrated fatigue validations: published curve points, Miner plus
//! exponent scaling, and knee/cut-off shape end to end.
//!
//! The S-N calibration is EN 1993-1-9 detail categories (reference life
//! 2e6 cycles, knee at 5e6 with slope change 3 -> 5, cut-off at 1e8).
//! All runs use the fixed 8x2 cantilever mesh with pulsating (zero to
//! maximum) transverse tip loads; expected values are closed forms of
//! the code curve and the Miner rule (no fitted constants).

// Rust guideline compliant 2026-09-30

use sonium_sim_structural_mechanics::{
    FemFatigue2DLoadBlock, FemFatigue2DStudy, PublishedSnCurve, run_fem_fatigue_2d,
};

/// Pulsating block: zero to a transverse tip force in N.
fn pulsating_block(force_n: f64, applied_cycles: f64) -> FemFatigue2DLoadBlock {
    FemFatigue2DLoadBlock {
        minimum_force_n: [0.0, 0.0],
        maximum_force_n: [0.0, -force_n],
        applied_cycles,
    }
}

/// Fixed hermetic study on category 36 (cruciform joints).
fn category36_study(blocks: Vec<FemFatigue2DLoadBlock>) -> FemFatigue2DStudy {
    let mut study = FemFatigue2DStudy::introductory();
    study.geometry.nx = 8;
    study.geometry.ny = 2;
    study.sn_curve = Some(PublishedSnCurve::eurocode3(36.0).unwrap());
    study.load_blocks = blocks;
    study
}

#[test]
fn published_curve_hits_eurocode3_defining_points() {
    // Category 160 (plain material): the reference point is exact by
    // construction, the knee and cut-off stresses match their closed
    // forms (measured exact to 1e-15; gate 1e-12), both branches meet
    // at the knee life 5e6 (measured within 1e-8; gate 1e-7), the
    // cut-off life is 1e8, and anything below cut-off (or
    // non-positive) has infinite life.
    let curve = PublishedSnCurve::eurocode3(160.0).unwrap();
    assert_eq!(curve.allowable_cycles(160.0e6), 2.0e6);
    let knee = curve.knee_stress_range_pa();
    let expected_knee = 160.0e6 * (2.0e6_f64 / 5.0e6).powf(1.0 / 3.0);
    assert!(
        ((knee - expected_knee) / expected_knee).abs() < 1.0e-12,
        "knee stress {knee} vs {expected_knee}"
    );
    assert!(
        ((curve.allowable_cycles(knee) - 5.0e6) / 5.0e6).abs() < 1.0e-12,
        "knee life {} vs 5e6",
        curve.allowable_cycles(knee)
    );
    for side in [1.0 - 1.0e-9, 1.0 + 1.0e-9] {
        let near = curve.allowable_cycles(knee * side);
        assert!(
            ((near - 5.0e6) / 5.0e6).abs() < 1.0e-7,
            "branch discontinuity at knee: {near} vs 5e6"
        );
    }
    let cutoff = curve.cutoff_stress_range_pa();
    let expected_cutoff = knee * (5.0e6_f64 / 1.0e8).powf(1.0 / 5.0);
    assert!(
        ((cutoff - expected_cutoff) / expected_cutoff).abs() < 1.0e-12,
        "cut-off stress {cutoff} vs {expected_cutoff}"
    );
    assert!(
        ((curve.allowable_cycles(cutoff) - 1.0e8) / 1.0e8).abs() < 1.0e-12,
        "cut-off life {} vs 1e8",
        curve.allowable_cycles(cutoff)
    );
    assert!(
        curve
            .allowable_cycles(cutoff * (1.0 - 1.0e-9))
            .is_infinite()
    );
    assert!(curve.allowable_cycles(0.0).is_infinite());
    assert!(curve.allowable_cycles(-5.0).is_infinite());
    for bad in [0.0, -10.0, f64::NAN, f64::INFINITY] {
        assert!(
            PublishedSnCurve::eurocode3(bad).is_err(),
            "category {bad} accepted"
        );
    }
}

#[test]
fn published_miner_accumulation_and_exponent_scaling() {
    // Both blocks sit above the category-36 knee (measured ranges 35.4
    // and 70.8 MPa vs knee 26.5 MPa), so the allowable ratio is exactly
    // 2^3 = 8 from linear FEM plus the m = 3 branch (measured exact to
    // 1e-13; gate 1e-9). The Miner sum and the failure-cycle
    // interpolation match their formulas (measured exact to 1e-15;
    // gate 1e-12); damage crosses 1.0 inside the second block.
    let study = category36_study(vec![
        pulsating_block(500.0, 1.0e6),
        pulsating_block(1000.0, 200_000.0),
    ]);
    let curve = study.sn_curve.unwrap();
    let solved = run_fem_fatigue_2d(&study).expect("two-block fatigue run");
    assert!(solved.converged);
    assert_eq!(solved.blocks.len(), 2);
    for block in &solved.blocks {
        assert!(block.solver_residual <= 1.0e-7);
        assert!(block.equilibrium_residual_n <= 1.0e-7);
        assert!(
            2.0 * block.stress_amplitude_pa > curve.knee_stress_range_pa(),
            "block below knee: range {}",
            2.0 * block.stress_amplitude_pa
        );
    }
    let ratio = solved.blocks[0].allowable_cycles / solved.blocks[1].allowable_cycles;
    assert!(
        ((ratio - 8.0) / 8.0).abs() < 1.0e-9,
        "allowable ratio {ratio} vs 8"
    );
    let expected_total: f64 = solved
        .blocks
        .iter()
        .map(|block| block.applied_cycles / block.allowable_cycles)
        .sum();
    assert!(
        ((solved.total_damage - expected_total) / expected_total).abs() < 1.0e-12,
        "Miner sum {} vs {expected_total}",
        solved.total_damage
    );
    assert!(solved.blocks[0].damage_increment < 1.0);
    assert!(solved.failed);
    let failure = solved.failure_cycle.expect("failure cycle recorded");
    let expected_failure = 1.0e6
        + 200_000.0 * (1.0 - solved.blocks[0].damage_increment) / solved.blocks[1].damage_increment;
    assert!(
        ((failure - expected_failure) / expected_failure).abs() < 1.0e-12,
        "failure cycle {failure} vs {expected_failure}"
    );
    assert!(
        failure > 1.0e6 && failure < 1.2e6,
        "failure cycle {failure} outside block 2"
    );
}

#[test]
fn published_knee_and_cutoff_shape_end_to_end() {
    // The high block (1000 N, range 70.8 MPa) rides the m = 3 branch
    // and the mid block (350 N, range 24.8 MPa) the m = 5 branch
    // between knee 26.5 and cut-off 14.6 MPa; each allowable matches
    // its branch closed form from the reported amplitude (measured
    // exact to 1e-14; gate 1e-9). The tiny block (100 N, range 7.1
    // MPa) is below cut-off: infinite life, exactly zero damage.
    let study = category36_study(vec![
        pulsating_block(1000.0, 1_000.0),
        pulsating_block(350.0, 50_000.0),
        pulsating_block(100.0, 10_000_000.0),
    ]);
    let curve = study.sn_curve.unwrap();
    let solved = run_fem_fatigue_2d(&study).expect("three-block fatigue run");
    assert!(solved.converged);
    assert_eq!(solved.blocks.len(), 3);
    let high_range = 2.0 * solved.blocks[0].stress_amplitude_pa;
    let mid_range = 2.0 * solved.blocks[1].stress_amplitude_pa;
    let tiny_range = 2.0 * solved.blocks[2].stress_amplitude_pa;
    assert!(high_range > curve.knee_stress_range_pa());
    assert!(mid_range < curve.knee_stress_range_pa() && mid_range > curve.cutoff_stress_range_pa());
    assert!(tiny_range < curve.cutoff_stress_range_pa());
    let expected_high = 2.0e6 * (36.0e6 / high_range).powf(3.0);
    assert!(
        ((solved.blocks[0].allowable_cycles - expected_high) / expected_high).abs() < 1.0e-9,
        "m=3 branch {} vs {expected_high}",
        solved.blocks[0].allowable_cycles
    );
    let expected_mid = 5.0e6 * (curve.knee_stress_range_pa() / mid_range).powf(5.0);
    assert!(
        ((solved.blocks[1].allowable_cycles - expected_mid) / expected_mid).abs() < 1.0e-9,
        "m=5 branch {} vs {expected_mid}",
        solved.blocks[1].allowable_cycles
    );
    assert!(solved.blocks[2].allowable_cycles.is_infinite());
    assert_eq!(solved.blocks[2].damage_increment, 0.0);
    assert_eq!(
        solved.blocks[2].cumulative_damage,
        solved.blocks[1].cumulative_damage
    );
    assert!(!solved.failed);
    assert!(solved.failure_cycle.is_none());
}
