//! Focused independent checks for RoomSim's backend-neutral spatial metrics.

use num_complex::Complex64;
use sonium_sim_room::fast_room::run_fast_hybrid;
use sonium_sim_room::room_metrics::{
    ACOUSTIC_REFERENCE_PRESSURE_PA, SpatialMetricSeries, peak_pressure_to_spl,
};
use sonium_xem::{Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source};

fn peak_for_spl(spl_db: f64) -> f64 {
    std::f64::consts::SQRT_2 * ACOUSTIC_REFERENCE_PRESSURE_PA * 10.0_f64.powf(spl_db / 20.0)
}

pub fn known_listener_levels_reproduce_spatial_rms_and_spread_independently() {
    let peak_60 = peak_for_spl(60.0);
    let peak_80 = peak_for_spl(80.0);
    let series = SpatialMetricSeries::from_listener_pressures(
        &[100.0],
        &[
            vec![Complex64::from_polar(peak_60, 0.25)],
            vec![Complex64::from_polar(peak_80, -1.1)],
        ],
    )
    .expect("known metric fixture should be valid");
    let sample = series.samples[0];

    let expected_rms_pa = ((peak_60 * peak_60 + peak_80 * peak_80) / 4.0).sqrt();
    let expected_rms_spl = 20.0 * (expected_rms_pa / ACOUSTIC_REFERENCE_PRESSURE_PA).log10();
    assert!((sample.rms_pressure_pa - expected_rms_pa).abs() < 1.0e-15);
    assert!((sample.rms_spl_db - expected_rms_spl).abs() < 1.0e-12);
    assert!((sample.mean_listener_spl_db - 70.0).abs() < 1.0e-12);
    assert!((sample.min_listener_spl_db - 60.0).abs() < 1.0e-12);
    assert!((sample.max_listener_spl_db - 80.0).abs() < 1.0e-12);
    assert!((sample.listener_spread_db - 20.0).abs() < 1.0e-12);
}

pub fn metric_reproduction_is_invariant_to_listener_order_and_phase() {
    let magnitude = |real: f64, imag: f64| (real * real + imag * imag).sqrt();
    let a0 = magnitude(0.002, 0.003);
    let a1 = magnitude(0.004, -0.001);
    let b0 = magnitude(-0.01, 0.002);
    let b1 = magnitude(0.002, 0.009);
    let c0 = magnitude(0.0004, -0.0002);
    let c1 = magnitude(-0.003, -0.004);
    let original = SpatialMetricSeries::from_listener_pressures(
        &[100.0, 200.0],
        &[
            vec![Complex64::new(0.002, 0.003), Complex64::new(0.004, -0.001)],
            vec![Complex64::new(-0.01, 0.002), Complex64::new(0.002, 0.009)],
            vec![
                Complex64::new(0.0004, -0.0002),
                Complex64::new(-0.003, -0.004),
            ],
        ],
    )
    .unwrap();
    let permuted = SpatialMetricSeries::from_listener_pressures(
        &[100.0, 200.0],
        &[
            vec![
                Complex64::from_polar(c0, 0.9),
                Complex64::from_polar(c1, -0.5),
            ],
            vec![
                Complex64::from_polar(a0, -0.3),
                Complex64::from_polar(a1, 1.2),
            ],
            vec![
                Complex64::from_polar(b0, 2.4),
                Complex64::from_polar(b1, -2.2),
            ],
        ],
    )
    .unwrap();

    // The second fixture uses the same magnitudes but arbitrary phases and a
    // different listener-row order.  The metric is intentionally phase-blind
    // and listener-order independent.
    for (a, b) in original.samples.iter().zip(permuted.samples.iter()) {
        assert!((a.rms_pressure_pa - b.rms_pressure_pa).abs() < 1.0e-12);
        assert!((a.rms_spl_db - b.rms_spl_db).abs() < 1.0e-12);
        assert!((a.mean_listener_spl_db - b.mean_listener_spl_db).abs() < 1.0e-12);
        assert!((a.listener_spread_db - b.listener_spread_db).abs() < 1.0e-12);
    }
}

pub fn fast_hybrid_output_reproduces_spatial_rms_independently() {
    let simulation = RoomSimulation::with_frequencies(
        RoomGeometry::Rectangular(RectangularRoom::new(2.4, 2.1, 1.9)),
        vec![Source::omnidirectional(Point3D::new(0.55, 0.62, 0.71), 1.0)],
        vec![
            Point3D::new(1.55, 1.22, 0.88),
            Point3D::new(1.78, 0.96, 1.03),
            Point3D::new(1.32, 1.48, 0.79),
        ],
        80.0,
        320.0,
        3,
    );
    let output = run_fast_hybrid(&simulation).expect("deterministic fast fixture should solve");
    let metrics = SpatialMetricSeries::for_source(&output, 0).unwrap();
    assert_eq!(metrics.listener_count, 3);

    for (frequency_index, sample) in metrics.samples.iter().enumerate() {
        let pressures = &output.pressures[0];
        let p0 = pressures[0][frequency_index];
        let p1 = pressures[1][frequency_index];
        let p2 = pressures[2][frequency_index];
        let expected_rms_pa = ((p0.norm_sqr() + p1.norm_sqr() + p2.norm_sqr()) / 6.0).sqrt();
        assert!((sample.rms_pressure_pa - expected_rms_pa).abs() < 1.0e-12);

        let expected_spl = 20.0 * (expected_rms_pa / ACOUSTIC_REFERENCE_PRESSURE_PA).log10();
        assert!((sample.rms_spl_db - expected_spl).abs() < 1.0e-10);
        assert!(sample.listener_spread_db.is_finite());
        assert!(peak_pressure_to_spl(p0).is_finite());
    }
}

#[test]
fn spatial_metrics_known_levels_test() {
    known_listener_levels_reproduce_spatial_rms_and_spread_independently();
}

#[test]
fn spatial_metrics_listener_invariance_test() {
    metric_reproduction_is_invariant_to_listener_order_and_phase();
}

#[test]
fn spatial_metrics_fast_hybrid_test() {
    fast_hybrid_output_reproduces_spatial_rms_independently();
}
