//! Native PP3 compact rotating-dipole observer-direction evidence.
//!
//! This packet exercises the phase-aware compact rotating-dipole reference
//! against independent angular, propagation, and sphere-power identities. It
//! records an observer-direction/phase contract only; it does not promote the
//! result to resolved propeller directivity, wake, or installation physics.

use serde::Serialize;
use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{RotatingDipoleReference, RotatingDipoleReferenceError, sha256_hex};
use std::f64::consts::PI;
use std::fs;
use std::path::Path;

const SCHEMA: &str = "sonium.propellers.observer-direction-evidence.v1";
const HARMONIC: usize = 1;
const REFERENCE_DISTANCE_M: f64 = 1.0;
const ANGLES_RAD: [f64; 7] = [
    0.0,
    PI / 6.0,
    PI / 4.0,
    PI / 2.0,
    3.0 * PI / 4.0,
    5.0 * PI / 6.0,
    PI,
];
const PROPAGATION_DISTANCES_M: [f64; 3] = [1.0, 2.0, 4.0];
const AZIMUTHS_RAD: [f64; 4] = [0.0, PI / 2.0, PI, -PI / 2.0];
const TEST_POLAR_ANGLE_RAD: f64 = PI / 3.0;
const TEST_AZIMUTH_RAD: f64 = PI / 4.0;
const TEST_DISTANCE_M: f64 = 2.0;
const RELATIVE_TOLERANCE: f64 = 1.0e-12;
const PHASE_TOLERANCE_RAD: f64 = 1.0e-12;

#[derive(Debug, Serialize)]
struct AngularLevel {
    polar_angle_rad: f64,
    expected_pressure_pa: f64,
    observed_pressure_pa: f64,
    relative_error: f64,
}

#[derive(Debug, Serialize)]
struct PhaseLevel {
    observer_distance_m: f64,
    polar_angle_rad: f64,
    azimuth_angle_rad: f64,
    expected_real_pa: f64,
    observed_real_pa: f64,
    expected_imaginary_pa: f64,
    observed_imaginary_pa: f64,
    expected_phase_rad: f64,
    observed_phase_rad: f64,
    phase_residual_rad: f64,
    relative_complex_error: f64,
    scaled_magnitude_pa_m: f64,
}

#[derive(Debug, Serialize)]
struct InvalidProbe {
    id: &'static str,
    status: &'static str,
    typed_error: String,
}

#[derive(Debug, Serialize)]
struct CanonicalOutput<'a> {
    angular: &'a [AngularLevel],
    propagation: &'a [PhaseLevel],
    azimuth: &'a [PhaseLevel],
}

#[derive(Debug, Serialize)]
struct ObserverDirectionArtifact {
    schema: &'static str,
    status: &'static str,
    scope: &'static str,
    claim_class: &'static str,
    reference: &'static str,
    canonical_input: String,
    canonical_input_sha256: String,
    harmonic: usize,
    frequency_hz: f64,
    reference_distance_m: f64,
    equatorial_peak_pressure_pa: f64,
    density_kg_m3: f64,
    speed_of_sound_m_s: f64,
    dimensions: Vec<&'static str>,
    angular_levels: Vec<AngularLevel>,
    propagation_levels: Vec<PhaseLevel>,
    azimuth_levels: Vec<PhaseLevel>,
    expected_power_ratio_to_isotropic: f64,
    observed_power_ratio_to_isotropic: f64,
    max_angular_relative_error: f64,
    max_complex_relative_error: f64,
    max_phase_residual_rad: f64,
    max_scaled_magnitude_relative_change: f64,
    tolerance: f64,
    phase_tolerance_rad: f64,
    converged: bool,
    invalid_probes: Vec<InvalidProbe>,
    canonical_output: String,
    output_sha256: String,
}

fn wrapped_phase_residual(observed: f64, expected: f64) -> f64 {
    (observed - expected + PI).rem_euclid(2.0 * PI) - PI
}

fn phase_level(
    reference: RotatingDipoleReference,
    distance_m: f64,
    polar_angle_rad: f64,
    azimuth_angle_rad: f64,
) -> PhaseLevel {
    let wavenumber_rad_m = 2.0 * PI * reference.frequency_hz / reference.speed_of_sound_m_s;
    let expected_phase_rad = -(wavenumber_rad_m * distance_m + azimuth_angle_rad);
    let amplitude_pa = reference.equatorial_peak_pressure_pa * reference.reference_distance_m
        / distance_m
        * polar_angle_rad.sin();
    let expected_real_pa = amplitude_pa * expected_phase_rad.cos();
    let expected_imaginary_pa = amplitude_pa * expected_phase_rad.sin();
    let observed = reference
        .complex_pressure_pa(distance_m, polar_angle_rad, azimuth_angle_rad)
        .expect("compact rotating-dipole phase reference");
    let difference_real_pa = observed.real_pa - expected_real_pa;
    let difference_imaginary_pa = observed.imaginary_pa - expected_imaginary_pa;
    let relative_complex_error = difference_real_pa.hypot(difference_imaginary_pa)
        / amplitude_pa.abs().max(f64::MIN_POSITIVE);
    PhaseLevel {
        observer_distance_m: distance_m,
        polar_angle_rad,
        azimuth_angle_rad,
        expected_real_pa,
        observed_real_pa: observed.real_pa,
        expected_imaginary_pa,
        observed_imaginary_pa: observed.imaginary_pa,
        expected_phase_rad,
        observed_phase_rad: observed.phase_rad(),
        phase_residual_rad: wrapped_phase_residual(observed.phase_rad(), expected_phase_rad),
        relative_complex_error,
        scaled_magnitude_pa_m: observed.magnitude_pa() * distance_m,
    }
}

fn evidence(output_dir: Option<&Path>) -> ObserverDirectionArtifact {
    assert_eq!(std::env::consts::OS, "linux", "PP3 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "PP3 requires native x86_64"
    );

    let source = PropellerSource::drone_5inch(18_000.0);
    let reference =
        RotatingDipoleReference::try_from_source(&source, HARMONIC, REFERENCE_DISTANCE_M)
            .expect("checked compact rotating-dipole reference");
    assert!(reference.equatorial_peak_pressure_pa > 0.0);

    let mut angular_levels = Vec::with_capacity(ANGLES_RAD.len());
    for polar_angle_rad in ANGLES_RAD {
        let expected_pressure_pa =
            reference.equatorial_peak_pressure_pa * polar_angle_rad.sin().abs();
        let observed_pressure_pa = reference
            .peak_pressure_pa(polar_angle_rad)
            .expect("valid dipole polar angle");
        let relative_error = (observed_pressure_pa - expected_pressure_pa).abs()
            / reference.equatorial_peak_pressure_pa;
        assert!(relative_error <= RELATIVE_TOLERANCE);
        angular_levels.push(AngularLevel {
            polar_angle_rad,
            expected_pressure_pa,
            observed_pressure_pa,
            relative_error,
        });
    }
    let max_angular_relative_error = angular_levels
        .iter()
        .map(|level| level.relative_error)
        .fold(0.0, f64::max);
    assert!(max_angular_relative_error <= RELATIVE_TOLERANCE);

    let propagation_levels = PROPAGATION_DISTANCES_M
        .into_iter()
        .map(|distance_m| {
            phase_level(
                reference,
                distance_m,
                TEST_POLAR_ANGLE_RAD,
                TEST_AZIMUTH_RAD,
            )
        })
        .collect::<Vec<_>>();
    let azimuth_levels = AZIMUTHS_RAD
        .into_iter()
        .map(|azimuth_angle_rad| {
            phase_level(
                reference,
                TEST_DISTANCE_M,
                TEST_POLAR_ANGLE_RAD,
                azimuth_angle_rad,
            )
        })
        .collect::<Vec<_>>();
    let max_complex_relative_error = propagation_levels
        .iter()
        .chain(azimuth_levels.iter())
        .map(|level| level.relative_complex_error)
        .fold(0.0, f64::max);
    let max_phase_residual_rad = propagation_levels
        .iter()
        .chain(azimuth_levels.iter())
        .map(|level| level.phase_residual_rad.abs())
        .fold(0.0, f64::max);
    assert!(max_complex_relative_error <= RELATIVE_TOLERANCE);
    assert!(max_phase_residual_rad <= PHASE_TOLERANCE_RAD);
    let scaled_magnitudes = propagation_levels
        .iter()
        .map(|level| level.scaled_magnitude_pa_m)
        .collect::<Vec<_>>();
    let max_scaled_magnitude_relative_change = scaled_magnitudes
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs() / pair[0].abs().max(f64::MIN_POSITIVE))
        .fold(0.0, f64::max);
    assert!(max_scaled_magnitude_relative_change <= RELATIVE_TOLERANCE);

    let expected_power_ratio_to_isotropic = RotatingDipoleReference::DIRECTIVITY_POWER_FACTOR;
    let observed_power_ratio_to_isotropic =
        reference.radiated_sound_power_w() / reference.isotropic_reference_power_w();
    assert!(
        (observed_power_ratio_to_isotropic - expected_power_ratio_to_isotropic).abs() <= 1.0e-15
    );

    let invalid_probes = vec![
        match reference.peak_pressure_pa(-1.0e-3) {
            Err(error @ RotatingDipoleReferenceError::InvalidPolarAngle) => InvalidProbe {
                id: "polar-angle-outside-spherical-domain",
                status: "typed_rejected",
                typed_error: format!("{error:?}"),
            },
            other => panic!("invalid polar angle was not typed-rejected: {other:?}"),
        },
        match reference.complex_pressure_pa(TEST_DISTANCE_M, TEST_POLAR_ANGLE_RAD, f64::NAN) {
            Err(error @ RotatingDipoleReferenceError::InvalidInput { .. }) => InvalidProbe {
                id: "nonfinite-azimuth-angle",
                status: "typed_rejected",
                typed_error: format!("{error:?}"),
            },
            other => panic!("invalid azimuth angle was not typed-rejected: {other:?}"),
        },
    ];

    let canonical_input = format!(
        "schema={SCHEMA};source_n_blades={};source_rpm={};source_density={};harmonic={HARMONIC};reference_distance_m={REFERENCE_DISTANCE_M};angles_rad={ANGLES_RAD:?};propagation_distances_m={PROPAGATION_DISTANCES_M:?};test_polar_angle_rad={TEST_POLAR_ANGLE_RAD};test_azimuth_rad={TEST_AZIMUTH_RAD};azimuths_rad={AZIMUTHS_RAD:?};test_distance_m={TEST_DISTANCE_M}",
        source.n_blades, source.rpm, source.density,
    );
    let canonical_output = serde_json::to_string(&CanonicalOutput {
        angular: &angular_levels,
        propagation: &propagation_levels,
        azimuth: &azimuth_levels,
    })
    .expect("serialize observer-direction levels");
    let artifact = ObserverDirectionArtifact {
        schema: SCHEMA,
        status: "pass",
        scope: "native Linux x86_64 CPU/f64; Windows/Arm64 waived",
        claim_class: "analytic compact rotating-dipole observer direction and phase reference",
        reference: "compact dipole pressure follows sin(theta), exp(-i(k r + phi)), 1/r, and a 2/3 sphere-power factor",
        canonical_input_sha256: sha256_hex(canonical_input.as_bytes()),
        canonical_input,
        harmonic: reference.harmonic,
        frequency_hz: reference.frequency_hz,
        reference_distance_m: reference.reference_distance_m,
        equatorial_peak_pressure_pa: reference.equatorial_peak_pressure_pa,
        density_kg_m3: reference.density_kg_m3,
        speed_of_sound_m_s: reference.speed_of_sound_m_s,
        dimensions: vec![
            "polar_angle_rad",
            "observer_distance_m",
            "azimuth_angle_rad",
            "complex_pressure_pa",
            "sphere_integrated_power",
        ],
        angular_levels,
        propagation_levels,
        azimuth_levels,
        expected_power_ratio_to_isotropic,
        observed_power_ratio_to_isotropic,
        max_angular_relative_error,
        max_complex_relative_error,
        max_phase_residual_rad,
        max_scaled_magnitude_relative_change,
        tolerance: RELATIVE_TOLERANCE,
        phase_tolerance_rad: PHASE_TOLERANCE_RAD,
        converged: true,
        invalid_probes,
        output_sha256: sha256_hex(canonical_output.as_bytes()),
        canonical_output,
    };

    if let Some(output_dir) = output_dir {
        fs::create_dir_all(output_dir).expect("create PP3 observer-direction output directory");
        fs::write(
            output_dir.join("native-x86-pp3-observer-direction-v1.json"),
            serde_json::to_vec_pretty(&artifact)
                .expect("serialize PP3 observer-direction artifact"),
        )
        .expect("write PP3 observer-direction artifact");
    }
    artifact
}

#[test]
fn native_pp3_observer_direction_evidence_v1() {
    let artifact = evidence(
        std::env::var_os("SONIUM_OUTPUT_DIR")
            .map(|path| Path::new(&path).to_path_buf())
            .as_deref(),
    );
    assert_eq!(artifact.status, "pass");
    assert_eq!(artifact.angular_levels.len(), ANGLES_RAD.len());
    assert_eq!(
        artifact.propagation_levels.len(),
        PROPAGATION_DISTANCES_M.len()
    );
    assert_eq!(artifact.azimuth_levels.len(), AZIMUTHS_RAD.len());
    assert!(artifact.converged);
    assert!(
        artifact
            .invalid_probes
            .iter()
            .all(|probe| probe.status == "typed_rejected")
    );
}
