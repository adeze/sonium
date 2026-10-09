//! Native PP3 observer-distance convergence evidence.
//!
//! This packet is deliberately independent of the reduced propeller ladder:
//! it exercises the stationary compact FW-H reference against its analytic
//! free-field inverse-distance identity.  It does not promote the result to a
//! resolved moving-surface FW-H or CFD claim.

use serde::Serialize;
use sonium_sim_propellers::{
    FwhError, FwhOptions, FwhSurfaceElement, Point3, evaluate_fwh_reference, sha256_hex,
};
use std::fs;
use std::path::Path;

const SCHEMA: &str = "sonium.propellers.observer-distance-evidence.v1";
const DISTANCES_M: [f64; 4] = [2.0, 4.0, 8.0, 16.0];
const RELATIVE_TOLERANCE: f64 = 1.0e-12;
const FREQUENCY_HZ: f64 = 1_000.0;
const DENSITY_KG_M3: f64 = 1.2;
const SOUND_SPEED_M_S: f64 = 343.0;
const SURFACE_AREA_M2: f64 = 0.2;
const SURFACE_PRESSURE_PA: f64 = 2.0;

#[derive(Debug, Serialize)]
struct DistanceLevel {
    distance_m: f64,
    expected_pressure_pa: f64,
    observed_pressure_pa: f64,
    scaled_pressure_pa_m: f64,
    relative_error: f64,
    source_observer_distance_m: f64,
}

#[derive(Debug, Serialize)]
struct InvalidProbe {
    id: &'static str,
    status: &'static str,
    typed_error: String,
}

#[derive(Debug, Serialize)]
struct ObserverDistanceArtifact {
    schema: &'static str,
    status: &'static str,
    scope: &'static str,
    claim_class: &'static str,
    reference: &'static str,
    canonical_input: String,
    canonical_input_sha256: String,
    dimensions: Vec<&'static str>,
    levels: Vec<DistanceLevel>,
    inverse_distance_exponent: f64,
    max_relative_error: f64,
    max_scaled_relative_change: f64,
    tolerance: f64,
    converged: bool,
    invalid_probe: InvalidProbe,
    canonical_output: String,
    output_sha256: String,
}

fn source_element() -> FwhSurfaceElement {
    FwhSurfaceElement {
        position: Point3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        normal: Point3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        area_m2: SURFACE_AREA_M2,
        pressure_real_pa: SURFACE_PRESSURE_PA,
        pressure_imaginary_pa: 0.0,
        normal_velocity_real_m_s: 0.0,
        normal_velocity_imaginary_m_s: 0.0,
    }
}

fn options(distance_m: f64) -> FwhOptions {
    FwhOptions {
        frequency_hz: FREQUENCY_HZ,
        density_kg_m3: DENSITY_KG_M3,
        sound_speed_m_s: SOUND_SPEED_M_S,
        observer_position: Point3 {
            x: 0.0,
            y: 0.0,
            z: distance_m,
        },
        minimum_observer_distance_m: 0.5,
        ..FwhOptions::default()
    }
}

fn expected_pressure_pa(distance_m: f64) -> f64 {
    let omega_rad_s = 2.0 * std::f64::consts::PI * FREQUENCY_HZ;
    omega_rad_s * SURFACE_AREA_M2 * SURFACE_PRESSURE_PA
        / (4.0 * std::f64::consts::PI * SOUND_SPEED_M_S * distance_m)
}

fn evidence(output_dir: Option<&Path>) -> ObserverDistanceArtifact {
    assert_eq!(std::env::consts::OS, "linux", "PP3 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "PP3 requires native x86_64"
    );

    let element = source_element();
    let mut levels = Vec::with_capacity(DISTANCES_M.len());
    for distance_m in DISTANCES_M {
        let expected = expected_pressure_pa(distance_m);
        let result = evaluate_fwh_reference(&[element], options(distance_m))
            .expect("analytic observer-distance reference");
        assert!(result.is_finite());
        let observed = result.total_pressure_pa.magnitude_pa();
        let relative_error = (observed - expected).abs() / expected;
        assert!(observed.is_finite() && relative_error <= RELATIVE_TOLERANCE);
        assert!((result.minimum_observer_distance_m - distance_m).abs() <= 1.0e-14);
        levels.push(DistanceLevel {
            distance_m,
            expected_pressure_pa: expected,
            observed_pressure_pa: observed,
            scaled_pressure_pa_m: observed * distance_m,
            relative_error,
            source_observer_distance_m: result.minimum_observer_distance_m,
        });
    }

    let scaled_changes = levels
        .windows(2)
        .map(|pair| {
            (pair[1].scaled_pressure_pa_m - pair[0].scaled_pressure_pa_m).abs()
                / pair[1].scaled_pressure_pa_m
        })
        .collect::<Vec<_>>();
    let max_relative_error = levels
        .iter()
        .map(|level| level.relative_error)
        .fold(0.0, f64::max);
    let max_scaled_relative_change = scaled_changes.iter().copied().fold(0.0, f64::max);
    assert!(max_scaled_relative_change <= RELATIVE_TOLERANCE);

    let invalid_probe = match evaluate_fwh_reference(&[element], options(0.25)) {
        Err(error @ FwhError::ObserverTooClose { .. }) => InvalidProbe {
            id: "observer-distance-below-minimum",
            status: "typed_rejected",
            typed_error: format!("{error:?}"),
        },
        other => panic!("invalid observer distance was not typed-rejected: {other:?}"),
    };

    let canonical_input = format!(
        "schema={SCHEMA};frequency_hz={FREQUENCY_HZ};density_kg_m3={DENSITY_KG_M3};sound_speed_m_s={SOUND_SPEED_M_S};surface_area_m2={SURFACE_AREA_M2};surface_pressure_pa={SURFACE_PRESSURE_PA};distances_m={DISTANCES_M:?};minimum_observer_distance_m=0.5"
    );
    let canonical_output = serde_json::to_string(&levels).expect("serialize distance levels");
    let artifact = ObserverDistanceArtifact {
        schema: SCHEMA,
        status: "pass",
        scope: "native Linux x86_64 CPU/f64; Windows/Arm64 waived",
        claim_class: "analytic stationary compact FW-H observer reference",
        reference: "free-field dipole pressure magnitude follows the exact inverse-distance identity",
        canonical_input_sha256: sha256_hex(canonical_input.as_bytes()),
        canonical_input,
        dimensions: vec![
            "observer_distance_m",
            "harmonic_frequency_hz",
            "surface_area_m2",
            "source_observer_phase",
        ],
        levels,
        inverse_distance_exponent: 1.0,
        max_relative_error,
        max_scaled_relative_change,
        tolerance: RELATIVE_TOLERANCE,
        converged: true,
        invalid_probe,
        output_sha256: sha256_hex(canonical_output.as_bytes()),
        canonical_output,
    };

    if let Some(output_dir) = output_dir {
        fs::create_dir_all(output_dir).expect("create PP3 observer-distance output directory");
        fs::write(
            output_dir.join("native-x86-pp3-observer-distance-v1.json"),
            serde_json::to_vec_pretty(&artifact).expect("serialize PP3 observer-distance artifact"),
        )
        .expect("write PP3 observer-distance artifact");
    }
    artifact
}

#[test]
fn native_pp3_observer_distance_evidence_v1() {
    let artifact = evidence(
        std::env::var_os("SONIUM_OUTPUT_DIR")
            .map(|path| Path::new(&path).to_path_buf())
            .as_deref(),
    );
    assert_eq!(artifact.status, "pass");
    assert_eq!(artifact.levels.len(), DISTANCES_M.len());
    assert!(artifact.converged);
    assert_eq!(artifact.invalid_probe.status, "typed_rejected");
}
