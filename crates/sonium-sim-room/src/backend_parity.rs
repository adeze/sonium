//! Checked backend-parity evidence for solved RoomSim responses.
//!
//! Backend comparisons operate on already-produced responses.  They do not
//! start an accelerator or simulator workload; callers must provide CPU and
//! accelerator outputs on the same source/listener/frequency topology.

use crate::{FrequencyProvenance, SimulationOutput};
use num_complex::Complex64;

/// Complex transfer-function error statistics between two solved backends.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendParityReport {
    /// Human-readable reference backend label.
    pub reference_backend: String,
    /// Human-readable candidate backend label.
    pub candidate_backend: String,
    /// Number of source/listener/frequency samples compared.
    pub sample_count: usize,
    /// RMS relative complex-pressure error.
    pub rms_relative_error: f64,
    /// Maximum relative complex-pressure error.
    pub max_relative_error: f64,
}

impl BackendParityReport {
    /// Require both aggregate parity metrics to remain within budget.
    pub fn require_within(
        &self,
        max_rms_relative_error: f64,
        max_max_relative_error: f64,
    ) -> Result<(), String> {
        if !max_rms_relative_error.is_finite()
            || max_rms_relative_error < 0.0
            || !max_max_relative_error.is_finite()
            || max_max_relative_error < 0.0
        {
            return Err("backend-parity error budgets must be finite non-negative".into());
        }
        if self.sample_count == 0
            || !self.rms_relative_error.is_finite()
            || self.rms_relative_error < 0.0
            || !self.max_relative_error.is_finite()
            || self.max_relative_error < 0.0
        {
            return Err("backend-parity report contains invalid metrics".into());
        }
        if self.rms_relative_error > max_rms_relative_error
            || self.max_relative_error > max_max_relative_error
        {
            return Err(format!(
                "backend-parity error exceeds budget: RMS={} (limit {}), max={} (limit {})",
                self.rms_relative_error,
                max_rms_relative_error,
                self.max_relative_error,
                max_max_relative_error,
            ));
        }
        Ok(())
    }
}

/// Compare fully solved complex responses from two backends.
pub fn compare_backend_outputs(
    reference_backend: &str,
    reference: &SimulationOutput,
    candidate_backend: &str,
    candidate: &SimulationOutput,
    frequency_tolerance_hz: f64,
    relative_floor_pa: f64,
) -> Result<BackendParityReport, String> {
    if reference_backend.trim().is_empty() || candidate_backend.trim().is_empty() {
        return Err("backend-parity backend labels must not be empty".into());
    }
    if !frequency_tolerance_hz.is_finite() || frequency_tolerance_hz < 0.0 {
        return Err("backend-parity frequency tolerance must be finite non-negative".into());
    }
    if !relative_floor_pa.is_finite() || relative_floor_pa <= 0.0 {
        return Err("backend-parity relative pressure floor must be finite positive".into());
    }
    reference.require_solved("backend-parity reference")?;
    candidate.require_solved("backend-parity candidate")?;
    validate_topology(reference, candidate, frequency_tolerance_hz)?;

    let mut sample_count = 0usize;
    let mut sum_squared = 0.0f64;
    let mut max_relative_error = 0.0f64;
    for (reference_source, candidate_source) in reference.pressures.iter().zip(&candidate.pressures)
    {
        for (reference_listener, candidate_listener) in
            reference_source.iter().zip(candidate_source)
        {
            for (&reference_pressure, &candidate_pressure) in
                reference_listener.iter().zip(candidate_listener)
            {
                let error = relative_complex_error(
                    reference_pressure,
                    candidate_pressure,
                    relative_floor_pa,
                );
                if !error.is_finite() {
                    return Err("backend-parity relative error is non-finite".into());
                }
                sample_count = sample_count
                    .checked_add(1)
                    .ok_or_else(|| "backend-parity sample counter overflowed".to_string())?;
                sum_squared += error * error;
                if !sum_squared.is_finite() {
                    return Err("backend-parity RMS accumulation is non-finite".into());
                }
                max_relative_error = max_relative_error.max(error);
            }
        }
    }
    if sample_count == 0 {
        return Err("backend-parity response contains no samples".into());
    }
    let rms_relative_error = (sum_squared / sample_count as f64).sqrt();
    if !rms_relative_error.is_finite() || !max_relative_error.is_finite() {
        return Err("backend-parity report is non-finite".into());
    }
    Ok(BackendParityReport {
        reference_backend: reference_backend.to_string(),
        candidate_backend: candidate_backend.to_string(),
        sample_count,
        rms_relative_error,
        max_relative_error,
    })
}

fn validate_topology(
    reference: &SimulationOutput,
    candidate: &SimulationOutput,
    frequency_tolerance_hz: f64,
) -> Result<(), String> {
    if reference.source_names != candidate.source_names {
        return Err("backend-parity source names differ".into());
    }
    if reference.frequencies.len() != candidate.frequencies.len()
        || reference
            .frequencies
            .iter()
            .zip(&candidate.frequencies)
            .any(|(reference, candidate)| (reference - candidate).abs() > frequency_tolerance_hz)
    {
        return Err("backend-parity frequency grids differ".into());
    }
    if reference.provenance != candidate.provenance
        || reference
            .provenance
            .iter()
            .any(|provenance| *provenance != FrequencyProvenance::Solved)
    {
        return Err("backend-parity requires matching solved provenance".into());
    }
    if reference.pressures.len() != candidate.pressures.len() {
        return Err("backend-parity source axis differs".into());
    }
    for (reference_source, candidate_source) in reference.pressures.iter().zip(&candidate.pressures)
    {
        if reference_source.len() != candidate_source.len() {
            return Err("backend-parity listener axis differs".into());
        }
        for (reference_listener, candidate_listener) in
            reference_source.iter().zip(candidate_source)
        {
            if reference_listener.len() != candidate_listener.len() {
                return Err("backend-parity frequency axis differs".into());
            }
        }
    }
    Ok(())
}

fn relative_complex_error(
    reference: Complex64,
    candidate: Complex64,
    relative_floor_pa: f64,
) -> f64 {
    let difference = (candidate - reference).norm();
    if difference == 0.0 {
        return 0.0;
    }
    difference / reference.norm().max(relative_floor_pa)
}

/// Normalized spatial-pattern agreement between two response backends.
///
/// Each frequency is normalized independently over listener magnitudes before
/// the flattened Pearson correlation and normalized residuals are computed.
/// This deliberately measures spatial distribution rather than absolute source
/// calibration; callers must use [`compare_backend_outputs`] for complex
/// transfer-function parity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialPatternCorrelationReport {
    /// Source axis compared.
    pub source_index: usize,
    /// Number of compared frequency bins.
    pub frequency_count: usize,
    /// Number of listeners per frequency.
    pub listener_count: usize,
    /// Total normalized listener samples.
    pub sample_count: usize,
    /// Pearson correlation of normalized listener magnitudes.
    pub pearson_correlation: f64,
    /// RMS difference of normalized listener magnitudes.
    pub rms_normalized_error: f64,
    /// Maximum absolute normalized listener difference.
    pub max_abs_normalized_error: f64,
}

impl SpatialPatternCorrelationReport {
    /// Require spatial correlation and normalized residuals to meet budgets.
    pub fn require_within(
        &self,
        minimum_correlation: f64,
        maximum_rms_normalized_error: f64,
        maximum_abs_normalized_error: f64,
    ) -> Result<(), String> {
        if !minimum_correlation.is_finite()
            || !(-1.0..=1.0).contains(&minimum_correlation)
            || !maximum_rms_normalized_error.is_finite()
            || maximum_rms_normalized_error < 0.0
            || !maximum_abs_normalized_error.is_finite()
            || maximum_abs_normalized_error < 0.0
            || self.sample_count == 0
            || self.listener_count == 0
            || self.frequency_count == 0
            || !self.pearson_correlation.is_finite()
            || !self.rms_normalized_error.is_finite()
            || self.rms_normalized_error < 0.0
            || !self.max_abs_normalized_error.is_finite()
            || self.max_abs_normalized_error < 0.0
        {
            return Err("spatial-pattern correlation report or budgets are invalid".into());
        }
        if self.pearson_correlation < minimum_correlation {
            return Err(format!(
                "spatial-pattern correlation {:.6} is below {:.6} budget",
                self.pearson_correlation, minimum_correlation
            ));
        }
        if self.rms_normalized_error > maximum_rms_normalized_error {
            return Err(format!(
                "spatial-pattern RMS error {:.6} exceeds {:.6} budget",
                self.rms_normalized_error, maximum_rms_normalized_error
            ));
        }
        if self.max_abs_normalized_error > maximum_abs_normalized_error {
            return Err(format!(
                "spatial-pattern maximum error {:.6} exceeds {:.6} budget",
                self.max_abs_normalized_error, maximum_abs_normalized_error
            ));
        }
        Ok(())
    }
}

/// Compare normalized listener-magnitude patterns for one source.
pub fn compare_normalized_listener_patterns(
    reference: &SimulationOutput,
    candidate: &SimulationOutput,
    source_index: usize,
    frequency_tolerance_hz: f64,
) -> Result<SpatialPatternCorrelationReport, String> {
    reference.validate_layout()?;
    candidate.validate_layout()?;
    if !frequency_tolerance_hz.is_finite() || frequency_tolerance_hz < 0.0 {
        return Err("spatial-pattern frequency tolerance must be finite and non-negative".into());
    }
    if reference.source_names != candidate.source_names {
        return Err("spatial-pattern source names differ".into());
    }
    if reference.frequencies.len() != candidate.frequencies.len()
        || reference
            .frequencies
            .iter()
            .zip(&candidate.frequencies)
            .any(|(reference_frequency, candidate_frequency)| {
                (reference_frequency - candidate_frequency).abs() > frequency_tolerance_hz
            })
    {
        return Err("spatial-pattern frequency grids differ".into());
    }
    let reference_source = reference
        .pressures
        .get(source_index)
        .ok_or_else(|| "spatial-pattern reference source index is out of range".to_string())?;
    let candidate_source = candidate
        .pressures
        .get(source_index)
        .ok_or_else(|| "spatial-pattern candidate source index is out of range".to_string())?;
    if reference_source.is_empty() || candidate_source.is_empty() {
        return Err("spatial-pattern listener axis is empty".into());
    }
    if reference_source.len() != candidate_source.len() {
        return Err("spatial-pattern listener axes differ".into());
    }

    let sample_capacity = reference.frequencies.len() * reference_source.len();
    let mut reference_values = Vec::with_capacity(sample_capacity);
    let mut candidate_values = Vec::with_capacity(sample_capacity);
    for frequency_index in 0..reference.frequencies.len() {
        let reference_magnitudes = reference_source
            .iter()
            .map(|listener| listener[frequency_index].norm())
            .collect::<Vec<_>>();
        let candidate_magnitudes = candidate_source
            .iter()
            .map(|listener| listener[frequency_index].norm())
            .collect::<Vec<_>>();
        if reference_magnitudes
            .iter()
            .chain(&candidate_magnitudes)
            .any(|value| !value.is_finite())
        {
            return Err("spatial-pattern response contains non-finite pressure".into());
        }
        let reference_scale = (reference_magnitudes
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            / reference_magnitudes.len() as f64)
            .sqrt()
            .max(1.0e-20);
        let candidate_scale = (candidate_magnitudes
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            / candidate_magnitudes.len() as f64)
            .sqrt()
            .max(1.0e-20);
        reference_values.extend(
            reference_magnitudes
                .into_iter()
                .map(|value| value / reference_scale),
        );
        candidate_values.extend(
            candidate_magnitudes
                .into_iter()
                .map(|value| value / candidate_scale),
        );
    }

    let sample_count = reference_values.len();
    let reference_mean = reference_values.iter().sum::<f64>() / sample_count as f64;
    let candidate_mean = candidate_values.iter().sum::<f64>() / sample_count as f64;
    let mut covariance = 0.0;
    let mut reference_variance = 0.0;
    let mut candidate_variance = 0.0;
    let mut squared_error = 0.0;
    let mut max_abs_error: f64 = 0.0;
    for (&reference_value, &candidate_value) in reference_values.iter().zip(&candidate_values) {
        covariance += (reference_value - reference_mean) * (candidate_value - candidate_mean);
        reference_variance += (reference_value - reference_mean).powi(2);
        candidate_variance += (candidate_value - candidate_mean).powi(2);
        let error = (candidate_value - reference_value).abs();
        squared_error += error * error;
        max_abs_error = max_abs_error.max(error);
    }
    let denominator = (reference_variance * candidate_variance).sqrt();
    let pearson_correlation = if denominator <= 1.0e-24 {
        if max_abs_error <= 1.0e-12 { 1.0 } else { 0.0 }
    } else {
        (covariance / denominator).clamp(-1.0, 1.0)
    };
    let report = SpatialPatternCorrelationReport {
        source_index,
        frequency_count: reference.frequencies.len(),
        listener_count: reference_source.len(),
        sample_count,
        pearson_correlation,
        rms_normalized_error: (squared_error / sample_count as f64).sqrt(),
        max_abs_normalized_error: max_abs_error,
    };
    if !report.pearson_correlation.is_finite()
        || !report.rms_normalized_error.is_finite()
        || !report.max_abs_normalized_error.is_finite()
    {
        return Err("spatial-pattern correlation produced non-finite metrics".into());
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(values: &[f64], provenance: FrequencyProvenance) -> SimulationOutput {
        SimulationOutput::with_uniform_provenance(
            vec![100.0, 200.0],
            vec![vec![
                values
                    .iter()
                    .copied()
                    .map(|value| Complex64::new(value, value * 0.1))
                    .collect(),
            ]],
            vec!["source".into()],
            provenance,
        )
    }

    fn spatial_response(values: [[f64; 2]; 2]) -> SimulationOutput {
        SimulationOutput::solved(
            vec![100.0, 200.0],
            vec![vec![
                vec![
                    Complex64::new(values[0][0], 0.0),
                    Complex64::new(values[0][1], 0.0),
                ],
                vec![
                    Complex64::new(values[1][0], 0.0),
                    Complex64::new(values[1][1], 0.0),
                ],
            ]],
            vec!["source".into()],
        )
    }

    #[test]
    fn normalized_spatial_patterns_ignore_absolute_scale() {
        let reference = spatial_response([[1.0, 2.0], [2.0, 4.0]]);
        let candidate = spatial_response([[10.0, 20.0], [20.0, 40.0]]);
        let report = compare_normalized_listener_patterns(&reference, &candidate, 0, 0.0).unwrap();
        assert_eq!(report.source_index, 0);
        assert_eq!(report.frequency_count, 2);
        assert_eq!(report.listener_count, 2);
        assert_eq!(report.sample_count, 4);
        assert!((report.pearson_correlation - 1.0).abs() < 1.0e-12);
        assert!(report.rms_normalized_error < 1.0e-12);
        report.require_within(0.99, 1.0e-12, 1.0e-12).unwrap();
    }

    #[test]
    fn normalized_spatial_patterns_reject_wrong_distribution_and_bad_budget() {
        let reference = spatial_response([[1.0, 2.0], [2.0, 4.0]]);
        let candidate = spatial_response([[2.0, 4.0], [1.0, 2.0]]);
        let report = compare_normalized_listener_patterns(&reference, &candidate, 0, 0.0).unwrap();
        assert!(report.pearson_correlation < -0.9);
        assert!(report.require_within(0.5, 2.0, 2.0).is_err());
        assert!(report.require_within(f64::NAN, 1.0, 1.0).is_err());
    }

    #[test]
    fn normalized_spatial_patterns_reject_topology_and_non_finite_inputs() {
        let reference = spatial_response([[1.0, 2.0], [2.0, 4.0]]);
        assert!(
            compare_normalized_listener_patterns(&reference, &reference, 1, 0.0)
                .unwrap_err()
                .contains("source index")
        );
        let wrong_frequency = SimulationOutput::solved(
            vec![100.0, 250.0],
            reference.pressures.clone(),
            reference.source_names.clone(),
        );
        assert!(
            compare_normalized_listener_patterns(&reference, &wrong_frequency, 0, 0.0)
                .unwrap_err()
                .contains("frequency grids")
        );
        let mut non_finite = reference.clone();
        non_finite.pressures[0][0][0] = Complex64::new(f64::NAN, 0.0);
        assert!(
            compare_normalized_listener_patterns(&reference, &non_finite, 0, 0.0)
                .unwrap_err()
                .contains("non-finite")
        );
    }

    #[test]
    fn exact_solved_backends_have_zero_complex_error() {
        let output = response(&[1.0, 2.0], FrequencyProvenance::Solved);
        let report =
            compare_backend_outputs("cpu", &output, "cuda", &output, 0.0, 1.0e-12).unwrap();
        assert_eq!(report.reference_backend, "cpu");
        assert_eq!(report.candidate_backend, "cuda");
        assert_eq!(report.sample_count, 2);
        assert_eq!(report.rms_relative_error, 0.0);
        assert_eq!(report.max_relative_error, 0.0);
        report.require_within(0.0, 0.0).unwrap();
    }

    #[test]
    fn parity_reports_error_and_rejects_budget_overrun() {
        let reference = response(&[1.0, 2.0], FrequencyProvenance::Solved);
        let candidate = response(&[1.01, 2.0], FrequencyProvenance::Solved);
        let report =
            compare_backend_outputs("cpu", &reference, "metal", &candidate, 0.0, 1.0e-12).unwrap();
        assert!(report.max_relative_error > 0.009);
        let error = report.require_within(0.001, 0.001).unwrap_err();
        assert!(error.contains("exceeds budget"));
        report.require_within(0.02, 0.02).unwrap();
    }

    #[test]
    fn parity_rejects_topology_provenance_and_non_finite_inputs() {
        let reference = response(&[1.0, 2.0], FrequencyProvenance::Solved);
        let mut source_mismatch = reference.clone();
        source_mismatch.source_names[0] = "other".into();
        assert!(
            compare_backend_outputs("cpu", &reference, "cuda", &source_mismatch, 0.0, 1.0e-12)
                .unwrap_err()
                .contains("source names")
        );

        let frequency_mismatch = SimulationOutput::with_uniform_provenance(
            vec![100.0, 250.0],
            reference.pressures.clone(),
            reference.source_names.clone(),
            FrequencyProvenance::Solved,
        );
        assert!(
            compare_backend_outputs("cpu", &reference, "cuda", &frequency_mismatch, 0.0, 1.0e-12,)
                .unwrap_err()
                .contains("frequency grids")
        );

        let reduced = response(&[1.0, 2.0], FrequencyProvenance::ReducedOrder);
        assert!(
            compare_backend_outputs("cpu", &reference, "cuda", &reduced, 0.0, 1.0e-12)
                .unwrap_err()
                .contains("reduced-order")
        );

        let mut non_finite = reference.clone();
        non_finite.pressures[0][0][0] = Complex64::new(f64::NAN, 0.0);
        assert!(
            compare_backend_outputs("cpu", &reference, "cuda", &non_finite, 0.0, 1.0e-12)
                .unwrap_err()
                .contains("non-finite")
        );
    }

    #[test]
    fn parity_rejects_invalid_labels_tolerances_and_budgets() {
        let output = response(&[1.0, 2.0], FrequencyProvenance::Solved);
        assert!(
            compare_backend_outputs("", &output, "cuda", &output, 0.0, 1.0e-12)
                .unwrap_err()
                .contains("labels")
        );
        assert!(
            compare_backend_outputs("cpu", &output, "cuda", &output, -1.0, 1.0e-12)
                .unwrap_err()
                .contains("frequency tolerance")
        );
        assert!(
            compare_backend_outputs("cpu", &output, "cuda", &output, 0.0, 0.0)
                .unwrap_err()
                .contains("pressure floor")
        );
        let report =
            compare_backend_outputs("cpu", &output, "cuda", &output, 0.0, 1.0e-12).unwrap();
        assert!(
            report
                .require_within(f64::NAN, 0.1)
                .unwrap_err()
                .contains("budgets")
        );
    }
}
