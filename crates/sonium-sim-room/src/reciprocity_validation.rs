//! Checked reciprocity evidence for solved RoomSim transfer responses.
//!
//! Reciprocity is a property of a transfer-normalized linear acoustic model:
//! the pressure from source `i` at listener `j` must equal the pressure from
//! source `j` at listener `i`, at the same frequency.  `SimulationOutput`
//! intentionally does not store listener identities, so this boundary accepts
//! explicit listener labels and validates the complete source/listener matrix
//! before comparing it.

use crate::SimulationOutput;
use num_complex::Complex64;
use std::collections::HashSet;

/// Aggregate complex-pressure reciprocity error over a source/listener matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReciprocityComparisonReport {
    /// Number of source/listener/frequency pairs compared.
    pub sample_count: usize,
    /// RMS symmetric relative complex-pressure error.
    pub rms_relative_error: f64,
    /// Maximum symmetric relative complex-pressure error.
    pub max_relative_error: f64,
}

impl ReciprocityComparisonReport {
    /// Require finite non-negative errors within explicit RMS and maximum
    /// relative budgets.
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
            return Err("reciprocity error budgets must be finite non-negative values".into());
        }
        if self.sample_count == 0 {
            return Err("reciprocity report requires at least one sample".into());
        }
        if !self.rms_relative_error.is_finite()
            || self.rms_relative_error < 0.0
            || !self.max_relative_error.is_finite()
            || self.max_relative_error < 0.0
        {
            return Err("reciprocity report contains invalid error metrics".into());
        }
        if self.rms_relative_error > max_rms_relative_error
            || self.max_relative_error > max_max_relative_error
        {
            return Err(format!(
                "reciprocity error exceeds budget: RMS={} (limit {}), max={} (limit {})",
                self.rms_relative_error,
                max_rms_relative_error,
                self.max_relative_error,
                max_max_relative_error,
            ));
        }
        Ok(())
    }
}

/// Compare two transfer-normalized solved responses in reciprocal directions.
///
/// `forward_listener_names` must be the listener-axis labels of `forward`.
/// `reverse` must use those labels as its source axis and use the forward
/// source labels as its listener axis.  The symmetric denominator is the
/// larger of the two pressure magnitudes and `relative_floor_pa`, so a small
/// response cannot create an undefined relative error.  The caller is
/// responsible for producing transfer-normalized responses with identical
/// source excitation conventions in both directions.
pub fn compare_reciprocal_transfer_outputs(
    forward: &SimulationOutput,
    forward_listener_names: &[String],
    reverse: &SimulationOutput,
    reverse_listener_names: &[String],
    frequency_tolerance_hz: f64,
    relative_floor_pa: f64,
) -> Result<ReciprocityComparisonReport, String> {
    if !frequency_tolerance_hz.is_finite() || frequency_tolerance_hz < 0.0 {
        return Err("reciprocity frequency tolerance must be finite non-negative".into());
    }
    if !relative_floor_pa.is_finite() || relative_floor_pa <= 0.0 {
        return Err("reciprocity relative pressure floor must be finite positive".into());
    }

    forward.require_solved("reciprocity forward response")?;
    reverse.require_solved("reciprocity reverse response")?;
    validate_listener_names(
        forward_listener_names,
        forward.pressures.first().map_or(0, Vec::len),
        "forward",
    )?;
    validate_listener_names(
        reverse_listener_names,
        reverse.pressures.first().map_or(0, Vec::len),
        "reverse",
    )?;

    if reverse.source_names != forward_listener_names {
        return Err("reciprocity reverse source labels must equal forward listener labels".into());
    }
    if reverse_listener_names != forward.source_names {
        return Err("reciprocity reverse listener labels must equal forward source labels".into());
    }
    if forward.frequencies.len() != reverse.frequencies.len() {
        return Err("reciprocity frequency grids differ in length".into());
    }
    for (index, (forward_frequency, reverse_frequency)) in forward
        .frequencies
        .iter()
        .zip(&reverse.frequencies)
        .enumerate()
    {
        if (forward_frequency - reverse_frequency).abs() > frequency_tolerance_hz {
            return Err(format!(
                "reciprocity frequency mismatch at index {index}: forward={forward_frequency}, reverse={reverse_frequency}"
            ));
        }
    }
    validate_matrix_shape(forward, forward_listener_names.len(), "forward")?;
    validate_matrix_shape(reverse, reverse_listener_names.len(), "reverse")?;

    let mut sample_count = 0usize;
    let mut sum_squared = 0.0f64;
    let mut maximum = 0.0f64;
    for (source_index, forward_source) in forward.pressures.iter().enumerate() {
        for (listener_index, forward_response) in forward_source.iter().enumerate() {
            let reverse_response = &reverse.pressures[listener_index][source_index];
            for (frequency_index, (&forward_pressure, &reverse_pressure)) in
                forward_response.iter().zip(reverse_response).enumerate()
            {
                let error =
                    symmetric_relative_error(forward_pressure, reverse_pressure, relative_floor_pa);
                if !error.is_finite() {
                    return Err(format!(
                        "reciprocity relative error is non-finite at source {source_index}, listener {listener_index}, frequency {frequency_index}"
                    ));
                }
                sample_count = sample_count
                    .checked_add(1)
                    .ok_or_else(|| "reciprocity sample count overflowed".to_string())?;
                sum_squared += error * error;
                if !sum_squared.is_finite() {
                    return Err("reciprocity RMS accumulation is non-finite".into());
                }
                maximum = maximum.max(error);
            }
        }
    }
    if sample_count == 0 {
        return Err("reciprocity comparison contains no samples".into());
    }
    let rms = (sum_squared / sample_count as f64).sqrt();
    if !rms.is_finite() || !maximum.is_finite() {
        return Err("reciprocity report is non-finite".into());
    }
    Ok(ReciprocityComparisonReport {
        sample_count,
        rms_relative_error: rms,
        max_relative_error: maximum,
    })
}

fn validate_listener_names(
    names: &[String],
    expected_count: usize,
    direction: &str,
) -> Result<(), String> {
    if names.len() != expected_count {
        return Err(format!(
            "reciprocity {direction} listener labels contain {}, expected {expected_count}",
            names.len()
        ));
    }
    if names.is_empty() {
        return Err(format!("reciprocity {direction} listener labels are empty"));
    }
    let mut unique = HashSet::with_capacity(names.len());
    for (index, name) in names.iter().enumerate() {
        if name.trim().is_empty() {
            return Err(format!(
                "reciprocity {direction} listener label {index} is empty"
            ));
        }
        if !unique.insert(name.to_ascii_lowercase()) {
            return Err(format!(
                "reciprocity {direction} listener labels are not unique"
            ));
        }
    }
    Ok(())
}

fn validate_matrix_shape(
    output: &SimulationOutput,
    listener_count: usize,
    direction: &str,
) -> Result<(), String> {
    if output
        .pressures
        .iter()
        .any(|source| source.len() != listener_count)
    {
        return Err(format!(
            "reciprocity {direction} response matrix has inconsistent listener axes"
        ));
    }
    Ok(())
}

fn symmetric_relative_error(forward: Complex64, reverse: Complex64, relative_floor_pa: f64) -> f64 {
    let difference = (forward - reverse).norm();
    if difference == 0.0 {
        return 0.0;
    }
    difference / forward.norm().max(reverse.norm()).max(relative_floor_pa)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrequencyProvenance;

    fn output(
        source_names: &[&str],
        pressures: Vec<Vec<Vec<Complex64>>>,
        provenance: FrequencyProvenance,
    ) -> SimulationOutput {
        SimulationOutput::with_uniform_provenance(
            vec![100.0, 200.0],
            pressures,
            source_names
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            provenance,
        )
    }

    fn reciprocal_fixture() -> (SimulationOutput, Vec<String>, SimulationOutput, Vec<String>) {
        let a = Complex64::new(1.0, 0.2);
        let b = Complex64::new(3.0, -0.5);
        let c = Complex64::new(5.0, 0.7);
        let forward = output(
            &["a", "b"],
            vec![
                vec![vec![a, a * 2.0], vec![b, b * 2.0]],
                vec![vec![b, b * 2.0], vec![c, c * 2.0]],
            ],
            FrequencyProvenance::Solved,
        );
        let reverse = output(
            &["a", "b"],
            vec![
                vec![vec![a, a * 2.0], vec![b, b * 2.0]],
                vec![vec![b, b * 2.0], vec![c, c * 2.0]],
            ],
            FrequencyProvenance::Solved,
        );
        (
            forward,
            vec!["a".into(), "b".into()],
            reverse,
            vec!["a".into(), "b".into()],
        )
    }

    #[test]
    fn exact_reciprocal_matrix_passes() {
        let (forward, forward_listeners, reverse, reverse_listeners) = reciprocal_fixture();
        let report = compare_reciprocal_transfer_outputs(
            &forward,
            &forward_listeners,
            &reverse,
            &reverse_listeners,
            0.0,
            1.0e-12,
        )
        .unwrap();
        assert_eq!(report.sample_count, 8);
        assert_eq!(report.rms_relative_error, 0.0);
        assert_eq!(report.max_relative_error, 0.0);
        report.require_within(0.0, 0.0).unwrap();
    }

    #[test]
    fn asymmetric_pair_fails_explicit_budget() {
        let (forward, forward_listeners, mut reverse, reverse_listeners) = reciprocal_fixture();
        reverse.pressures[0][1][0] = Complex64::new(3.5, -0.5);
        let report = compare_reciprocal_transfer_outputs(
            &forward,
            &forward_listeners,
            &reverse,
            &reverse_listeners,
            0.0,
            1.0e-12,
        )
        .unwrap();
        assert!(report.max_relative_error > 0.1);
        assert!(
            report
                .require_within(0.1, 0.1)
                .unwrap_err()
                .contains("exceeds budget")
        );
    }

    #[test]
    fn reciprocal_axes_are_checked_by_label_and_shape() {
        let (forward, mut forward_listeners, reverse, reverse_listeners) = reciprocal_fixture();
        forward_listeners[1] = "other".into();
        let error = compare_reciprocal_transfer_outputs(
            &forward,
            &forward_listeners,
            &reverse,
            &reverse_listeners,
            0.0,
            1.0e-12,
        )
        .unwrap_err();
        assert!(error.contains("reverse source labels"));

        let (forward, forward_listeners, reverse, reverse_listeners) = reciprocal_fixture();
        let mut malformed = reverse;
        malformed.pressures[0].pop();
        let error = compare_reciprocal_transfer_outputs(
            &forward,
            &forward_listeners,
            &malformed,
            &reverse_listeners,
            0.0,
            1.0e-12,
        )
        .unwrap_err();
        assert!(error.contains("listener labels contain") || error.contains("listener axes"));

        let (mut malformed_forward, forward_listeners, reverse, reverse_listeners) =
            reciprocal_fixture();
        malformed_forward.pressures[1].pop();
        let error = compare_reciprocal_transfer_outputs(
            &malformed_forward,
            &forward_listeners,
            &reverse,
            &reverse_listeners,
            0.0,
            1.0e-12,
        )
        .unwrap_err();
        assert!(error.contains("inconsistent listener axes"));
    }

    #[test]
    fn frequency_grid_mismatch_is_rejected() {
        let (forward, forward_listeners, mut reverse, reverse_listeners) = reciprocal_fixture();
        reverse.frequencies[1] = 201.0;
        let error = compare_reciprocal_transfer_outputs(
            &forward,
            &forward_listeners,
            &reverse,
            &reverse_listeners,
            0.0,
            1.0e-12,
        )
        .unwrap_err();
        assert!(error.contains("frequency mismatch"));
        compare_reciprocal_transfer_outputs(
            &forward,
            &forward_listeners,
            &reverse,
            &reverse_listeners,
            1.0,
            1.0e-12,
        )
        .unwrap();
    }

    #[test]
    fn non_solved_and_non_finite_outputs_are_rejected() {
        let (forward, forward_listeners, mut reverse, reverse_listeners) = reciprocal_fixture();
        reverse.provenance[0] = FrequencyProvenance::ReducedOrder;
        let error = compare_reciprocal_transfer_outputs(
            &forward,
            &forward_listeners,
            &reverse,
            &reverse_listeners,
            0.0,
            1.0e-12,
        )
        .unwrap_err();
        assert!(error.contains("requires solved samples"));

        let (forward, forward_listeners, mut reverse, reverse_listeners) = reciprocal_fixture();
        reverse.pressures[1][1][0] = Complex64::new(f64::NAN, 0.0);
        let error = compare_reciprocal_transfer_outputs(
            &forward,
            &forward_listeners,
            &reverse,
            &reverse_listeners,
            0.0,
            1.0e-12,
        )
        .unwrap_err();
        assert!(error.contains("non-finite pressure"));
    }

    #[test]
    fn listener_labels_reject_empty_duplicate_and_wrong_count() {
        let (forward, _, reverse, reverse_listeners) = reciprocal_fixture();
        let error = compare_reciprocal_transfer_outputs(
            &forward,
            &["a".into(), "a".into()],
            &reverse,
            &reverse_listeners,
            0.0,
            1.0e-12,
        )
        .unwrap_err();
        assert!(error.contains("not unique"));

        let (forward, _, reverse, reverse_listeners) = reciprocal_fixture();
        let error = compare_reciprocal_transfer_outputs(
            &forward,
            &["a".into(), " ".into()],
            &reverse,
            &reverse_listeners,
            0.0,
            1.0e-12,
        )
        .unwrap_err();
        assert!(error.contains("empty"));

        let (forward, _, reverse, reverse_listeners) = reciprocal_fixture();
        let error = compare_reciprocal_transfer_outputs(
            &forward,
            &["a".into()],
            &reverse,
            &reverse_listeners,
            0.0,
            1.0e-12,
        )
        .unwrap_err();
        assert!(error.contains("contain 1"));
    }

    #[test]
    fn invalid_tolerances_and_budgets_are_rejected() {
        let (forward, forward_listeners, reverse, reverse_listeners) = reciprocal_fixture();
        assert!(
            compare_reciprocal_transfer_outputs(
                &forward,
                &forward_listeners,
                &reverse,
                &reverse_listeners,
                -1.0,
                1.0e-12,
            )
            .unwrap_err()
            .contains("frequency tolerance")
        );
        assert!(
            compare_reciprocal_transfer_outputs(
                &forward,
                &forward_listeners,
                &reverse,
                &reverse_listeners,
                0.0,
                0.0,
            )
            .unwrap_err()
            .contains("pressure floor")
        );
        let report = compare_reciprocal_transfer_outputs(
            &forward,
            &forward_listeners,
            &reverse,
            &reverse_listeners,
            0.0,
            1.0e-12,
        )
        .unwrap();
        assert!(
            report
                .require_within(f64::NAN, 0.1)
                .unwrap_err()
                .contains("budgets")
        );

        let invalid = ReciprocityComparisonReport {
            sample_count: 0,
            rms_relative_error: 0.0,
            max_relative_error: 0.0,
        };
        assert!(
            invalid
                .require_within(0.1, 0.1)
                .unwrap_err()
                .contains("at least one sample")
        );
        let invalid = ReciprocityComparisonReport {
            sample_count: 1,
            rms_relative_error: f64::NAN,
            max_relative_error: 0.0,
        };
        assert!(
            invalid
                .require_within(0.1, 0.1)
                .unwrap_err()
                .contains("invalid error metrics")
        );
    }
}
