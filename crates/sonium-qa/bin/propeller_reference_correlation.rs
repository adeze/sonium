//! Release-backed propeller application/reference correlation.
//!
//! The source model supplies the independent harmonic reference amplitudes;
//! the product wrapper must preserve those values, units, conventions, and
//! the published reduction prior for the combined design.

use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_qa::QaResult;
use sonium_sim_propellers::{
    COMBINED_REDUCTION_PRIOR_DB, EvaluationMode, PropellerDesign, SpectrumDirectivityConvention,
    SpectrumPhaseConvention, SpectrumPressureConvention,
};

const CASE_ID: &str = "propeller.application-reference-correlation.v1";
const RPM: f64 = 18_000.0;
const N_HARMONICS: usize = 8;
const BAND_HZ: (f64, f64) = (1_000.0, 5_000.0);

fn run() -> Result<(f64, f64), String> {
    let source = PropellerSource::drone_5inch(RPM);
    let reference_pressures = source.harmonic_amplitudes(N_HARMONICS);
    if (source.blade_passing_frequency() - 600.0).abs() > 1.0e-12 {
        return Err("two-blade 18,000 RPM source did not produce a 600 Hz BPF".into());
    }

    for mode in [
        EvaluationMode::FastSurrogate,
        EvaluationMode::ResearchControl,
    ] {
        let spectrum = PropellerDesign::conventional(source.clone()).spectrum(N_HARMONICS, mode);
        if spectrum.provenance.evaluation_mode != mode
            || spectrum.provenance.pressure != SpectrumPressureConvention::PeakMagnitudeAtOneMetre
            || spectrum.provenance.directivity
                != SpectrumDirectivityConvention::UnitMeanPowerIsotropicReference
            || spectrum.provenance.phase != SpectrumPhaseConvention::MagnitudeOnly
            || spectrum.harmonics.len() != N_HARMONICS
        {
            return Err("propeller application/reference convention metadata mismatch".into());
        }
        for (index, harmonic) in spectrum.harmonics.iter().enumerate() {
            let expected_frequency = source.harmonic_frequency(index + 1);
            let expected_pressure = reference_pressures[index];
            let pressure = harmonic.base_pressure_pa * harmonic.amplitude_reduction;
            let frequency_error = (harmonic.frequency_hz - expected_frequency).abs();
            let pressure_error = (harmonic.base_pressure_pa - expected_pressure).abs();
            if frequency_error > 1.0e-12
                || pressure_error > expected_pressure.abs().max(1.0) * 1.0e-12
                || (harmonic.pressure_pa - pressure).abs()
                    > harmonic.base_pressure_pa.abs().max(1.0) * 1.0e-12
                || !(0.0..=1.0).contains(&harmonic.amplitude_reduction)
            {
                return Err(format!(
                    "propeller harmonic {} failed correlation",
                    index + 1
                ));
            }
        }
    }

    let baseline = PropellerDesign::conventional(source.clone())
        .spectrum(N_HARMONICS, EvaluationMode::ResearchControl);
    let combined = PropellerDesign::toroidal_serrated(source)
        .spectrum(N_HARMONICS, EvaluationMode::ResearchControl);
    let reduction_db = combined
        .try_band_reduction_db_against(&baseline, BAND_HZ.0, BAND_HZ.1)
        .map_err(|error| format!("invalid propeller correlation band: {error:?}"))?;
    if !COMBINED_REDUCTION_PRIOR_DB.contains(reduction_db) {
        return Err(format!(
            "combined-design reduction {reduction_db:.6} dB is outside [{:.6}, {:.6}] dB",
            COMBINED_REDUCTION_PRIOR_DB.min_db, COMBINED_REDUCTION_PRIOR_DB.max_db
        ));
    }
    Ok((reduction_db, 2.0))
}

fn main() {
    match run() {
        Ok((reduction_db, mode_count)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("blade_passing_frequency_hz", 600.0)
                .metric("evaluation_mode_count", mode_count)
                .metric("combined_reduction_db", reduction_db)
                .minimum("evaluation_mode_count", 2.0)
                .input_checksum("analytic/deterministic-fixture:propeller-reference-correlation-v1")
                .reference("independent PropellerSource harmonics plus published combined-design reduction prior")
                .reference_checksum("analytic/deterministic-fixture:propeller-reference-correlation-v1")
                .dimension("two-blade 18,000 RPM source; eight harmonics; 1-5 kHz band")
                .formulation("production propeller spectrum wrapper against independent source harmonics")
                .tolerance("frequency/pressure/convention correlation <= 1e-12; reduction inside published prior")
                .seed("propeller-reference-correlation-v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("propeller result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("propeller failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
