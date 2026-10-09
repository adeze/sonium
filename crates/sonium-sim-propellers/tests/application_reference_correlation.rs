//! W1-08 release-backed application/reference correlation.
//!
//! This is intentionally a bounded propeller slice. It checks that the
//! application-level spectrum wrapper preserves the FEM aeroacoustic source
//! reference and that the combined design remains inside its published
//! empirical reduction prior. It is not a validation of the full 3-D blade
//! solve or of measured propeller data.

use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    COMBINED_REDUCTION_PRIOR_DB, EvaluationMode, PropellerDesign, SpectrumDirectivityConvention,
    SpectrumPhaseConvention, SpectrumPressureConvention,
};

const CASE_ID: &str = "w1-08.application-reference-correlation.propeller.v1";
const RPM: f64 = 18_000.0;
const N_HARMONICS: usize = 8;
const BAND_HZ: (f64, f64) = (1_000.0, 5_000.0);
const RELATIVE_TOLERANCE: f64 = 1.0e-12;

fn assert_close(label: &str, actual: f64, expected: f64) {
    let tolerance = expected.abs().max(1.0) * RELATIVE_TOLERANCE;
    assert!(
        (actual - expected).abs() <= tolerance,
        "{CASE_ID}: {label} actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
    );
}

#[test]
fn w1_08_propeller_application_reference_correlation_v1() {
    let source = PropellerSource::drone_5inch(RPM);
    let reference_pressures = source.harmonic_amplitudes(N_HARMONICS);

    // The source reference is a two-blade, 18,000 RPM rotor, so the BPF is
    // 600 Hz. Keep this explicit to catch unit or blade-count regressions.
    assert_close(
        "blade-passing frequency",
        source.blade_passing_frequency(),
        600.0,
    );

    for mode in [
        EvaluationMode::FastSurrogate,
        EvaluationMode::ResearchControl,
    ] {
        let spectrum = PropellerDesign::conventional(source.clone()).spectrum(N_HARMONICS, mode);

        assert_eq!(
            spectrum.provenance.evaluation_mode, mode,
            "{CASE_ID}: application did not publish the requested evaluation mode"
        );
        assert_eq!(
            spectrum.provenance.pressure,
            SpectrumPressureConvention::PeakMagnitudeAtOneMetre,
            "{CASE_ID}: unexpected pressure convention"
        );
        assert_eq!(
            spectrum.provenance.directivity,
            SpectrumDirectivityConvention::UnitMeanPowerIsotropicReference,
            "{CASE_ID}: unexpected directivity convention"
        );
        assert_eq!(
            spectrum.provenance.phase,
            SpectrumPhaseConvention::MagnitudeOnly,
            "{CASE_ID}: unexpected phase convention"
        );
        assert_eq!(
            spectrum.harmonics.len(),
            N_HARMONICS,
            "{CASE_ID}: application/reference harmonic count mismatch"
        );

        for (index, harmonic) in spectrum.harmonics.iter().enumerate() {
            let harmonic_number = index + 1;
            assert_eq!(
                harmonic.harmonic, harmonic_number,
                "{CASE_ID}: harmonic index mismatch at position {index}"
            );
            assert_close(
                &format!("harmonic {harmonic_number} frequency"),
                harmonic.frequency_hz,
                source.harmonic_frequency(harmonic_number),
            );
            assert_close(
                &format!("harmonic {harmonic_number} source pressure"),
                harmonic.base_pressure_pa,
                reference_pressures[index],
            );
            assert_close(
                &format!("harmonic {harmonic_number} modified pressure"),
                harmonic.pressure_pa,
                harmonic.base_pressure_pa * harmonic.amplitude_reduction,
            );
            assert!(
                (0.0..=1.0).contains(&harmonic.amplitude_reduction),
                "{CASE_ID}: harmonic {harmonic_number} reduction out of range: {}",
                harmonic.amplitude_reduction
            );
        }
    }

    // This is the application/reference correlation point: the combined
    // application design must remain within the repository's explicitly
    // published empirical prior for this operating point and band.
    let baseline = PropellerDesign::conventional(source.clone())
        .spectrum(N_HARMONICS, EvaluationMode::ResearchControl);
    let combined = PropellerDesign::toroidal_serrated(source)
        .spectrum(N_HARMONICS, EvaluationMode::ResearchControl);
    let reduction_db = combined
        .try_band_reduction_db_against(&baseline, BAND_HZ.0, BAND_HZ.1)
        .unwrap_or_else(|error| panic!("{CASE_ID}: invalid application/reference band: {error:?}"));

    assert!(
        COMBINED_REDUCTION_PRIOR_DB.contains(reduction_db),
        "{CASE_ID}: combined-design reduction {reduction_db:.6} dB is outside [{:.6}, {:.6}] dB",
        COMBINED_REDUCTION_PRIOR_DB.min_db,
        COMBINED_REDUCTION_PRIOR_DB.max_db
    );
}
