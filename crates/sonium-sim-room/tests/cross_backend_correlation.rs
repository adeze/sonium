//! Deterministic CPU fast-hybrid/FEM spatial-pattern correlation reference.

use num_complex::Complex64;
use sonium_sim_room::backend_parity::compare_normalized_listener_patterns;
use sonium_sim_room::fast_room::run_fast_hybrid;
use sonium_sim_room::fem_runner::{FemMeshConfig, FemSourceModel, run_fem_with_mesh};
use sonium_sim_room::room_metrics::SpatialMetricSeries;
use sonium_xem::{
    BoundaryConfig, Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source, SurfaceConfig,
};

fn fixture() -> RoomSimulation {
    let mut simulation = RoomSimulation::new(
        RoomGeometry::Rectangular(RectangularRoom::new(1.6, 1.5, 1.4)),
        vec![Source::omnidirectional(Point3D::new(0.4, 0.35, 0.35), 1.0)],
        vec![Point3D::new(1.2, 1.1, 1.0), Point3D::new(1.0, 0.55, 1.05)],
    );
    let wall = SurfaceConfig::Admittance {
        real: 1.0,
        imag: 0.0,
    };
    simulation.boundaries = BoundaryConfig {
        floor: wall.clone(),
        ceiling: wall.clone(),
        walls: wall,
        front_wall: None,
        back_wall: None,
        left_wall: None,
        right_wall: None,
    };
    simulation.frequencies = vec![200.0, 400.0, 800.0];
    simulation
}

fn normalized_listener_patterns(output: &sonium_sim_room::SimulationOutput) -> Vec<f64> {
    let pressures = &output.pressures[0];
    let mut patterns = Vec::with_capacity(output.frequencies.len() * pressures.len());
    for frequency_index in 0..output.frequencies.len() {
        let magnitudes: Vec<f64> = pressures
            .iter()
            .map(|listener| listener[frequency_index].norm())
            .collect();
        let reference = (magnitudes.iter().map(|value| value * value).sum::<f64>()
            / magnitudes.len() as f64)
            .sqrt()
            .max(1.0e-20);
        patterns.extend(magnitudes.into_iter().map(|value| value / reference));
    }
    patterns
}

fn correlation(lhs: &[f64], rhs: &[f64]) -> f64 {
    assert_eq!(lhs.len(), rhs.len());
    let lhs_mean = lhs.iter().sum::<f64>() / lhs.len() as f64;
    let rhs_mean = rhs.iter().sum::<f64>() / rhs.len() as f64;
    let mut numerator = 0.0;
    let mut lhs_norm = 0.0;
    let mut rhs_norm = 0.0;
    for (&left, &right) in lhs.iter().zip(rhs) {
        let left = left - lhs_mean;
        let right = right - rhs_mean;
        numerator += left * right;
        lhs_norm += left * left;
        rhs_norm += right * right;
    }
    numerator / (lhs_norm * rhs_norm).sqrt().max(1.0e-20)
}

pub fn run_fast_hybrid_and_fem_correlation() -> f64 {
    let simulation = fixture();
    let fast = run_fast_hybrid(&simulation).expect("fast-hybrid fixture should solve");
    let fem = run_fem_with_mesh(
        &simulation,
        &FemMeshConfig {
            fine_radius: 0.4,
            modal_damping_ratio: 0.0,
            source_model: FemSourceModel::PointLoad,
            ..FemMeshConfig::default()
        },
    )
    .expect("FEM fixture should solve");

    fast.validate_layout().expect("fast output layout");
    fem.validate_layout().expect("FEM output layout");
    assert_eq!(fast.frequencies, fem.frequencies);
    assert_eq!(fast.pressures[0].len(), fem.pressures[0].len());
    assert!(
        fast.pressures
            .iter()
            .flatten()
            .flatten()
            .all(|pressure: &Complex64| pressure.re.is_finite() && pressure.im.is_finite())
    );
    assert!(
        fem.pressures
            .iter()
            .flatten()
            .flatten()
            .all(|pressure: &Complex64| pressure.re.is_finite() && pressure.im.is_finite())
    );

    let fast_metrics = SpatialMetricSeries::for_source(&fast, 0).expect("fast metrics");
    let fem_metrics = SpatialMetricSeries::for_source(&fem, 0).expect("FEM metrics");
    assert!(
        fast_metrics
            .samples
            .iter()
            .chain(&fem_metrics.samples)
            .all(|sample| sample.rms_spl_db.is_finite() && sample.listener_spread_db.is_finite())
    );

    let fast_pattern = normalized_listener_patterns(&fast);
    let fem_pattern = normalized_listener_patterns(&fem);
    let checked_report =
        compare_normalized_listener_patterns(&fast, &fem, 0, 0.0).expect("checked correlation");
    checked_report
        .require_within(0.75, 2.0, 2.0)
        .expect("checked fast-hybrid/FEM spatial correlation");
    let pattern_correlation = correlation(&fast_pattern, &fem_pattern);
    assert!(
        pattern_correlation >= 0.75,
        "normalized fast-hybrid/FEM listener correlation {pattern_correlation:.4} < 0.75"
    );
    pattern_correlation
}

#[test]
fn fast_hybrid_and_fem_correlate_on_normalized_listener_patterns() {
    run_fast_hybrid_and_fem_correlation();
}
