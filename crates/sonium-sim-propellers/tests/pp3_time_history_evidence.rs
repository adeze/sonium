//! Native PP3 moving FW-H time-history convergence evidence.
//!
//! This packet exercises the bounded retarded-time reference with a stationary
//! surface whose prescribed source strengths are quadratic in source time.
//! Piecewise-linear histories therefore have a measurable first-order
//! time-step error in the source-strength derivatives.  The packet is an
//! analytic convergence check for the moving-surface reference; it does not
//! promote the result to a resolved rotor or CFD time-history claim.

use serde::Serialize;
use sonium_sim_propellers::{
    MovingFwhElementHistory, MovingFwhError, MovingFwhOptions, MovingFwhSample, Point3,
    evaluate_moving_fwh_reference, sha256_hex,
};
use std::fs;
use std::path::Path;

const SCHEMA: &str = "sonium.propellers.time-history-evidence.v1";
const REFINEMENTS: [usize; 5] = [16, 32, 64, 128, 256];
const DENSITY_KG_M3: f64 = 1.2;
const SOUND_SPEED_M_S: f64 = 343.0;
const SURFACE_AREA_M2: f64 = 0.2;
const OBSERVER_DISTANCE_M: f64 = 10.0;
const TARGET_SOURCE_TIME_S: f64 = 0.5;
const PRESSURE_OFFSET_PA: f64 = 1.0;
const PRESSURE_QUADRATIC_PA: f64 = 2.0;
const VELOCITY_OFFSET_M_S: f64 = 0.5;
const VELOCITY_QUADRATIC_M_S: f64 = 0.75;
const CONVERGENCE_TOLERANCE: f64 = 5.0e-3;

#[derive(Debug, Serialize)]
struct TimeHistoryLevel {
    refinement_intervals: usize,
    step_s: f64,
    sample_count: usize,
    retarded_time_s: f64,
    source_observer_distance_m: f64,
    doppler_factor: f64,
    observed_thickness_pressure_pa: f64,
    observed_loading_pressure_pa: f64,
    observed_total_pressure_pa: f64,
    expected_total_pressure_pa: f64,
    absolute_error_pa: f64,
    relative_error: f64,
}

#[derive(Debug, Serialize)]
struct InvalidProbe {
    id: &'static str,
    status: &'static str,
    typed_error: String,
}

#[derive(Debug, Serialize)]
struct TimeHistoryArtifact {
    schema: &'static str,
    status: &'static str,
    scope: &'static str,
    claim_class: &'static str,
    reference: &'static str,
    canonical_input: String,
    canonical_input_sha256: String,
    dimensions: Vec<&'static str>,
    levels: Vec<TimeHistoryLevel>,
    exact_convergence_order: f64,
    observed_convergence_orders: Vec<f64>,
    observed_convergence_order: f64,
    max_relative_error: f64,
    final_relative_error: f64,
    tolerance: f64,
    converged: bool,
    invalid_probe: InvalidProbe,
    canonical_output: String,
    output_sha256: String,
}

fn point(x: f64, y: f64, z: f64) -> Point3 {
    Point3 { x, y, z }
}

fn source_sample(time_s: f64, pressure_pa: f64, normal_velocity_m_s: f64) -> MovingFwhSample {
    MovingFwhSample {
        time_s,
        position: point(0.0, 0.0, 0.0),
        normal: point(0.0, 0.0, 1.0),
        area_m2: SURFACE_AREA_M2,
        pressure_pa,
        normal_velocity_m_s,
    }
}

fn history(intervals: usize) -> MovingFwhElementHistory {
    let step_s = 1.0 / intervals as f64;
    let samples = (0..=intervals)
        .map(|index| {
            let time_s = index as f64 * step_s;
            source_sample(
                time_s,
                PRESSURE_OFFSET_PA + PRESSURE_QUADRATIC_PA * time_s * time_s,
                VELOCITY_OFFSET_M_S + VELOCITY_QUADRATIC_M_S * time_s * time_s,
            )
        })
        .collect();
    MovingFwhElementHistory { samples }
}

fn options(observer_time_s: f64) -> MovingFwhOptions {
    MovingFwhOptions {
        observer_time_s,
        observer_position: point(0.0, 0.0, OBSERVER_DISTANCE_M),
        density_kg_m3: DENSITY_KG_M3,
        sound_speed_m_s: SOUND_SPEED_M_S,
        minimum_observer_distance_m: 0.5,
        retarded_time_tolerance_s: 1.0e-12,
        maximum_retarded_iterations: 128,
        ..MovingFwhOptions::default()
    }
}

fn expected_total_pressure_pa() -> f64 {
    let pressure_derivative_pa_s = 2.0 * PRESSURE_QUADRATIC_PA * TARGET_SOURCE_TIME_S;
    let velocity_derivative_m_s2 = 2.0 * VELOCITY_QUADRATIC_M_S * TARGET_SOURCE_TIME_S;
    let thickness = DENSITY_KG_M3 * SURFACE_AREA_M2 * velocity_derivative_m_s2
        / (4.0 * std::f64::consts::PI * OBSERVER_DISTANCE_M);
    let loading = SURFACE_AREA_M2 * pressure_derivative_pa_s
        / (4.0 * std::f64::consts::PI * SOUND_SPEED_M_S * OBSERVER_DISTANCE_M);
    thickness + loading
}

fn evidence(output_dir: Option<&Path>) -> TimeHistoryArtifact {
    assert_eq!(std::env::consts::OS, "linux", "PP3 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "PP3 requires native x86_64"
    );

    let observer_time_s = TARGET_SOURCE_TIME_S + OBSERVER_DISTANCE_M / SOUND_SPEED_M_S;
    let expected_total = expected_total_pressure_pa();
    let mut levels = Vec::with_capacity(REFINEMENTS.len());
    for intervals in REFINEMENTS {
        let result = evaluate_moving_fwh_reference(&[history(intervals)], options(observer_time_s))
            .expect("analytic moving FW-H time-history reference");
        assert!(result.is_finite());
        let element = result.elements[0];
        let absolute_error = (result.total_pressure_pa - expected_total).abs();
        let relative_error = absolute_error / expected_total.abs();
        assert!(relative_error.is_finite());
        assert!((element.retarded_time_s - TARGET_SOURCE_TIME_S).abs() <= 1.0e-11);
        assert!((element.source_observer_distance_m - OBSERVER_DISTANCE_M).abs() <= 1.0e-12);
        assert!((element.doppler_factor - 1.0).abs() <= 1.0e-14);
        levels.push(TimeHistoryLevel {
            refinement_intervals: intervals,
            step_s: 1.0 / intervals as f64,
            sample_count: intervals + 1,
            retarded_time_s: element.retarded_time_s,
            source_observer_distance_m: element.source_observer_distance_m,
            doppler_factor: element.doppler_factor,
            observed_thickness_pressure_pa: result.thickness_pressure_pa,
            observed_loading_pressure_pa: result.loading_pressure_pa,
            observed_total_pressure_pa: result.total_pressure_pa,
            expected_total_pressure_pa: expected_total,
            absolute_error_pa: absolute_error,
            relative_error,
        });
    }

    let convergence_orders = levels
        .windows(2)
        .map(|pair| (pair[0].absolute_error_pa / pair[1].absolute_error_pa).log2())
        .collect::<Vec<_>>();
    assert!(
        convergence_orders
            .iter()
            .all(|order| order.is_finite() && *order > 0.8 && *order < 1.2)
    );
    let observed_convergence_order =
        convergence_orders.iter().sum::<f64>() / convergence_orders.len() as f64;
    let max_relative_error = levels
        .iter()
        .map(|level| level.relative_error)
        .fold(0.0, f64::max);
    let final_relative_error = levels
        .last()
        .expect("time-history ladder has levels")
        .relative_error;
    assert!(final_relative_error <= CONVERGENCE_TOLERANCE);

    let invalid_probe = match evaluate_moving_fwh_reference(
        &[history(REFINEMENTS[0])],
        options(OBSERVER_DISTANCE_M / SOUND_SPEED_M_S - 1.0e-3),
    ) {
        Err(error @ MovingFwhError::RetardedTimeOutsideHistory { .. }) => InvalidProbe {
            id: "observer-time-before-history-window",
            status: "typed_rejected",
            typed_error: format!("{error:?}"),
        },
        other => panic!("invalid observer time was not typed-rejected: {other:?}"),
    };

    let canonical_input = format!(
        "schema={SCHEMA};density_kg_m3={DENSITY_KG_M3};sound_speed_m_s={SOUND_SPEED_M_S};surface_area_m2={SURFACE_AREA_M2};observer_distance_m={OBSERVER_DISTANCE_M};target_source_time_s={TARGET_SOURCE_TIME_S};pressure_offset_pa={PRESSURE_OFFSET_PA};pressure_quadratic_pa={PRESSURE_QUADRATIC_PA};velocity_offset_m_s={VELOCITY_OFFSET_M_S};velocity_quadratic_m_s={VELOCITY_QUADRATIC_M_S};refinements={REFINEMENTS:?};tolerance={CONVERGENCE_TOLERANCE}"
    );
    let canonical_output = serde_json::to_string(&levels).expect("serialize time-history levels");
    let artifact = TimeHistoryArtifact {
        schema: SCHEMA,
        status: "pass",
        scope: "native Linux x86_64 CPU/f64; Windows/Arm64 waived",
        claim_class: "analytic stationary moving FW-H time-history reference",
        reference: "quadratic source strengths with piecewise-linear histories converge at first order in time step",
        canonical_input_sha256: sha256_hex(canonical_input.as_bytes()),
        canonical_input,
        dimensions: vec![
            "source_time_s",
            "retarded_time_s",
            "time_step_s",
            "surface_area_m2",
            "observer_distance_m",
        ],
        levels,
        exact_convergence_order: 1.0,
        observed_convergence_orders: convergence_orders,
        observed_convergence_order,
        max_relative_error,
        final_relative_error,
        tolerance: CONVERGENCE_TOLERANCE,
        converged: true,
        invalid_probe,
        output_sha256: sha256_hex(canonical_output.as_bytes()),
        canonical_output,
    };

    if let Some(output_dir) = output_dir {
        fs::create_dir_all(output_dir).expect("create PP3 time-history output directory");
        fs::write(
            output_dir.join("native-x86-pp3-time-history-v1.json"),
            serde_json::to_vec_pretty(&artifact).expect("serialize PP3 time-history artifact"),
        )
        .expect("write PP3 time-history artifact");
    }
    artifact
}

#[test]
fn native_pp3_time_history_evidence_v1() {
    let artifact = evidence(
        std::env::var_os("SONIUM_OUTPUT_DIR")
            .map(|path| Path::new(&path).to_path_buf())
            .as_deref(),
    );
    assert_eq!(artifact.status, "pass");
    assert_eq!(artifact.levels.len(), REFINEMENTS.len());
    assert!(artifact.converged);
    assert_eq!(artifact.invalid_probe.status, "typed_rejected");
}
