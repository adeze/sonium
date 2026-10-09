//! Compare complete speaker observations with independent measurements.
//!
//! Input values use RMS phasors with exp(+i omega t), SI units, and the
//! speaker frame x=right, y=up, z=forward. BEM producers must conjugate and
//! convert peak amplitudes before supplying predictions. Comparison never
//! fits parameters, rescales pressure, shifts phase, or extrapolates data.

// Rust guideline compliant 2026-02-21
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_sim_speaker_modeling::IndependentMeasurementMetadata;

/// Observable and its canonical unit in a whole-speaker measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerObservable {
    PressureRmsPa,
    ElectricalImpedanceOhm,
    DisplacementRmsM,
    PressureThdFraction,
    CompressionDb,
    CoilTemperatureK,
}

const REQUIRED: [SpeakerObservable; 6] = [
    SpeakerObservable::PressureRmsPa,
    SpeakerObservable::ElectricalImpedanceOhm,
    SpeakerObservable::DisplacementRmsM,
    SpeakerObservable::PressureThdFraction,
    SpeakerObservable::CompressionDb,
    SpeakerObservable::CoilTemperatureK,
];

/// Convert a peak exp(-i omega t) BEM pressure to the RMS exp(+i omega t)
/// convention used by measured speaker references.
pub fn bem_peak_to_measurement_rms(pressure: Complex64) -> Complex64 {
    pressure.conj() / std::f64::consts::SQRT_2
}

/// Calibration data is reported separately from held-out validation data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleRole {
    Calibration,
    Validation,
}

/// Identity of one independently acquired observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerObservation {
    pub id: String,
    pub role: SampleRole,
    pub observable: SpeakerObservable,
    pub frequency_hz: f64,
    pub drive_voltage_rms_v: f64,
    /// Receiver location for pressure; probe location for other observables.
    pub position_m: [f64; 3],
    pub value: Complex64,
    /// Allowed absolute error in this observable's canonical units.
    pub absolute_tolerance: f64,
    /// Allowed relative complex error, added to the absolute allowance.
    pub relative_tolerance: f64,
}

/// Versioned measured-speaker data with acquisition traceability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerMeasurementSet {
    pub schema_version: u32,
    pub geometry_id: String,
    pub metadata: IndependentMeasurementMetadata,
    pub observations: Vec<SpeakerObservation>,
}

/// Predictions keyed to exact measurement conditions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerPredictionSet {
    pub schema_version: u32,
    pub geometry_id: String,
    pub provenance: String,
    pub observations: Vec<SpeakerPrediction>,
}

/// Predicted observable at an exact drive, frequency, and probe location.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerPrediction {
    pub id: String,
    pub observable: SpeakerObservable,
    pub frequency_hz: f64,
    pub drive_voltage_rms_v: f64,
    pub position_m: [f64; 3],
    pub value: Complex64,
}

/// One independently recomputed error and acceptance decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerComparison {
    pub id: String,
    pub role: SampleRole,
    pub observable: SpeakerObservable,
    pub absolute_error: f64,
    pub allowed_error: f64,
    pub passed: bool,
}

/// Held-out validation result, including missing complete-speaker evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerMeasurementReport {
    pub schema_version: u32,
    pub dataset_id: String,
    pub geometry_id: String,
    pub prediction_provenance: String,
    pub comparisons: Vec<SpeakerComparison>,
    pub missing_validation_observables: Vec<SpeakerObservable>,
    /// True only when every held-out sample passes and all channels exist.
    pub complete_speaker_passed: bool,
}

/// Validate exact measurement conditions and compare independent observations.
///
/// # Errors
/// Returns an error for invalid metadata, samples, duplicate IDs, missing
/// predictions, incompatible conditions, or a missing held-out sample.
pub fn compare_speaker_measurements(
    measured: &SpeakerMeasurementSet,
    predicted: &SpeakerPredictionSet,
) -> Result<SpeakerMeasurementReport, String> {
    measured.metadata.validate().map_err(|e| e.to_string())?;
    if measured.schema_version != 1
        || predicted.schema_version != 1
        || measured.geometry_id.trim().is_empty()
        || measured.geometry_id != predicted.geometry_id
        || predicted.provenance.trim().is_empty()
        || measured.observations.is_empty()
    {
        return Err("invalid or mismatched speaker-reference metadata".into());
    }
    let mut ids = std::collections::BTreeSet::new();
    for prediction in &predicted.observations {
        validate_sample(
            &prediction.id,
            prediction.observable,
            prediction.frequency_hz,
            prediction.drive_voltage_rms_v,
            prediction.position_m,
            prediction.value,
        )?;
        if !ids.insert(&prediction.id) {
            return Err("duplicate prediction id".into());
        }
    }
    ids.clear();
    let mut comparisons = Vec::with_capacity(measured.observations.len());
    for sample in &measured.observations {
        validate_sample(
            &sample.id,
            sample.observable,
            sample.frequency_hz,
            sample.drive_voltage_rms_v,
            sample.position_m,
            sample.value,
        )?;
        if !ids.insert(&sample.id) {
            return Err("duplicate measurement id".into());
        }
        if !sample.absolute_tolerance.is_finite()
            || sample.absolute_tolerance < 0.0
            || !sample.relative_tolerance.is_finite()
            || sample.relative_tolerance < 0.0
            || sample.absolute_tolerance + sample.relative_tolerance <= 0.0
        {
            return Err(format!("invalid tolerance for {}", sample.id));
        }
        let prediction = predicted
            .observations
            .iter()
            .find(|p| p.id == sample.id)
            .ok_or_else(|| format!("missing prediction for {}", sample.id))?;
        if prediction.observable != sample.observable
            || prediction.frequency_hz != sample.frequency_hz
            || prediction.drive_voltage_rms_v != sample.drive_voltage_rms_v
            || prediction.position_m != sample.position_m
        {
            return Err(format!("prediction conditions differ for {}", sample.id));
        }
        let error = (prediction.value - sample.value).norm();
        let allowance = sample.absolute_tolerance + sample.relative_tolerance * sample.value.norm();
        if !error.is_finite() || !allowance.is_finite() {
            return Err("comparison overflow".into());
        }
        comparisons.push(SpeakerComparison {
            id: sample.id.clone(),
            role: sample.role,
            observable: sample.observable,
            absolute_error: error,
            allowed_error: allowance,
            passed: error <= allowance,
        });
    }
    if !comparisons.iter().any(|p| p.role == SampleRole::Validation) {
        return Err("at least one held-out validation sample is required".into());
    }
    let missing: Vec<_> = REQUIRED
        .into_iter()
        .filter(|observable| {
            !comparisons
                .iter()
                .any(|p| p.role == SampleRole::Validation && p.observable == *observable)
        })
        .collect();
    let complete_speaker_passed = missing.is_empty()
        && comparisons
            .iter()
            .filter(|p| p.role == SampleRole::Validation)
            .all(|p| p.passed);
    Ok(SpeakerMeasurementReport {
        schema_version: 1,
        dataset_id: measured.metadata.dataset_id.clone(),
        geometry_id: measured.geometry_id.clone(),
        prediction_provenance: predicted.provenance.clone(),
        comparisons,
        missing_validation_observables: missing,
        complete_speaker_passed,
    })
}

fn validate_sample(
    id: &str,
    observable: SpeakerObservable,
    frequency: f64,
    voltage: f64,
    position: [f64; 3],
    value: Complex64,
) -> Result<(), String> {
    if id.trim().is_empty()
        || !frequency.is_finite()
        || frequency <= 0.0
        || !voltage.is_finite()
        || voltage <= 0.0
        || position.iter().any(|v| !v.is_finite())
        || !value.re.is_finite()
        || !value.im.is_finite()
    {
        return Err(format!("invalid speaker sample {id}"));
    }
    if matches!(
        observable,
        SpeakerObservable::PressureThdFraction
            | SpeakerObservable::CompressionDb
            | SpeakerObservable::CoilTemperatureK
    ) && value.im != 0.0
    {
        return Err(format!(
            "scalar speaker sample {id} has an imaginary component"
        ));
    }
    if (observable == SpeakerObservable::PressureThdFraction && value.re < 0.0)
        || (observable == SpeakerObservable::CoilTemperatureK && value.re <= 0.0)
    {
        return Err(format!(
            "speaker sample {id} is outside its physical domain"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_observables_reject_complex_or_unphysical_values() {
        for (observable, value) in [
            (SpeakerObservable::CompressionDb, Complex64::new(-2.0, 1.0)),
            (
                SpeakerObservable::PressureThdFraction,
                Complex64::new(-0.1, 0.0),
            ),
            (
                SpeakerObservable::CoilTemperatureK,
                Complex64::new(0.0, 0.0),
            ),
        ] {
            assert!(validate_sample("invalid", observable, 100.0, 1.0, [0.0; 3], value).is_err());
        }
        assert!(
            validate_sample(
                "compression",
                SpeakerObservable::CompressionDb,
                100.0,
                1.0,
                [0.0; 3],
                Complex64::new(-2.0, 0.0)
            )
            .is_ok()
        );
    }

    #[test]
    fn bem_conversion_preserves_real_waveform_and_rms() {
        let peak = Complex64::new(3.0, 4.0);
        let rms = bem_peak_to_measurement_rms(peak);
        assert!((rms.norm() - 5.0 / std::f64::consts::SQRT_2).abs() < 1e-12);
        assert!(rms.im < 0.0);
        assert!((rms.re * std::f64::consts::SQRT_2 - peak.re).abs() < 1e-12);
    }

    fn fixture() -> (SpeakerMeasurementSet, SpeakerPredictionSet) {
        let observations: Vec<_> = REQUIRED
            .into_iter()
            .enumerate()
            .map(|(i, observable)| SpeakerObservation {
                id: i.to_string(),
                role: SampleRole::Validation,
                observable,
                frequency_hz: 100.0,
                drive_voltage_rms_v: 2.83,
                position_m: [0.0, 0.0, 1.0],
                value: Complex64::new(1.0, 0.0),
                absolute_tolerance: 0.01,
                relative_tolerance: 0.02,
            })
            .collect();
        let predictions = observations
            .iter()
            .map(|o| SpeakerPrediction {
                id: o.id.clone(),
                observable: o.observable,
                frequency_hz: o.frequency_hz,
                drive_voltage_rms_v: o.drive_voltage_rms_v,
                position_m: o.position_m,
                value: o.value,
            })
            .collect();
        (
            SpeakerMeasurementSet {
                schema_version: 1,
                geometry_id: "fixture".into(),
                metadata: IndependentMeasurementMetadata {
                    dataset_id: "synthetic-test-only".into(),
                    source: "unit test".into(),
                    acquisition_method: "constructed values".into(),
                },
                observations,
            },
            SpeakerPredictionSet {
                schema_version: 1,
                geometry_id: "fixture".into(),
                provenance: "unit test".into(),
                observations: predictions,
            },
        )
    }

    #[test]
    fn held_out_failure_cannot_be_hidden_by_calibration_fit() {
        let (mut measured, mut predicted) = fixture();
        assert!(
            compare_speaker_measurements(&measured, &predicted)
                .unwrap()
                .complete_speaker_passed
        );
        measured.observations[0].role = SampleRole::Calibration;
        let report = compare_speaker_measurements(&measured, &predicted).unwrap();
        assert!(!report.complete_speaker_passed);
        predicted.observations[1].value *= 2.0;
        assert!(
            !compare_speaker_measurements(&measured, &predicted)
                .unwrap()
                .comparisons[1]
                .passed
        );
    }

    #[test]
    fn reference_conditions_and_missing_predictions_fail_closed() {
        let (measured, mut predicted) = fixture();
        predicted.observations[0].drive_voltage_rms_v = 1.0;
        assert!(compare_speaker_measurements(&measured, &predicted).is_err());
        predicted.observations.remove(0);
        assert!(compare_speaker_measurements(&measured, &predicted).is_err());
    }
}
