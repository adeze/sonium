//! AutoEQ data generation using acoustic room solvers
//!
//! Generates CSV measurement files and roomeq JSON configs from simulated
//! room acoustics for testing the RoomEQ optimizer.
//!
//! FEM is the default path for closed-room interior Helmholtz problems
//! (sparse matrices, frequency-dependent local impedance BCs, easy symmetry
//! exploitation). The `generate-roomeq-data` binary can also splice in a
//! dense BLR TBEM solve for the low-frequency modal band.

use num_complex::Complex64;
use std::cmp::Ordering;
use std::collections::HashSet;

pub mod backend_parity;
pub mod csv_export;
pub mod energy_validation;
pub mod external_reference;
pub mod fast_room;
pub mod fem_runner;
pub mod gallery;
pub mod hf_extension;
pub mod measured_acceptance;
pub mod measured_io;
pub mod measured_validation;
pub mod mesh_convergence;
pub mod reciprocity_validation;
pub mod room_metrics;
pub mod roomeq_config_gen;
pub mod runtime;
pub mod scenarios;
pub mod source_calibration;
pub mod spatial_field;
pub mod treatment_optimization;
pub mod treatment_projection;
pub mod wav_export;

/// How a frequency sample of a [`SimulationOutput`] was produced.
///
/// Consumers must be able to tell physically solved samples apart from
/// reduced-order, interpolated, geometric, and synthesized ones: only
/// `Solved` samples may be used to optimize room geometry, treatments,
/// directivity, or crossovers, or to claim acoustic power / full-range
/// predictions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyProvenance {
    /// Solved by the full numerical model (FEM or dense TBEM).
    Solved,
    /// Produced by a reduced-order model (fast hybrid modal/image method).
    ReducedOrder,
    /// Interpolated between solved or reduced samples; this is not a direct
    /// physical solve at the labelled frequency.
    Interpolated,
    /// Produced by a geometric-acoustics method such as image-source, ray, or
    /// diffusion tracing.
    Geometric,
    /// Synthesized (high-frequency extension); no physical room or
    /// loudspeaker solution exists at this frequency.
    Synthetic,
}

impl FrequencyProvenance {
    /// Stable lowercase label used in exported artifacts.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Solved => "solved",
            Self::ReducedOrder => "reduced-order",
            Self::Interpolated => "interpolated",
            Self::Geometric => "geometric",
            Self::Synthetic => "synthetic",
        }
    }

    /// Convert this simulation label to the RoomEQ scoring provenance domain.
    ///
    /// Simulation data cannot be a calibrated measurement, so this conversion
    /// only covers the five origins shared by generated room responses and
    /// RoomEQ's CEA-2034 policy input.
    pub const fn to_roomeq_score_provenance(self) -> sonium_roomeq::cea2034::ScoreSampleProvenance {
        match self {
            Self::Solved => sonium_roomeq::cea2034::ScoreSampleProvenance::Solved,
            Self::ReducedOrder => sonium_roomeq::cea2034::ScoreSampleProvenance::ReducedOrder,
            Self::Interpolated => sonium_roomeq::cea2034::ScoreSampleProvenance::Interpolated,
            Self::Geometric => sonium_roomeq::cea2034::ScoreSampleProvenance::Geometric,
            Self::Synthetic => sonium_roomeq::cea2034::ScoreSampleProvenance::Synthetic,
        }
    }
}

/// Output from a room simulation.
///
/// Pressures are indexed as `[source_idx][lp_idx][freq_idx]`.
#[derive(Debug, Clone)]
pub struct SimulationOutput {
    /// Frequency points in Hz
    pub frequencies: Vec<f64>,
    /// Complex pressure at each (source, listening position, frequency)
    pub pressures: Vec<Vec<Vec<Complex64>>>,
    /// Source names matching the source index
    pub source_names: Vec<String>,
    /// Per-frequency provenance mask, aligned with `frequencies`.
    pub provenance: Vec<FrequencyProvenance>,
}

impl SimulationOutput {
    /// Validate that the provenance mask and pressure tensor are aligned with
    /// the frequency grid. Exporters and physical consumers should call this
    /// before indexing the output; it prevents a stale synthetic tail from
    /// being silently paired with a different frequency grid.
    pub fn validate_layout(&self) -> Result<(), String> {
        if self.frequencies.is_empty() {
            return Err("frequency grid is empty".to_string());
        }
        if self
            .frequencies
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        {
            return Err("frequency grid must contain finite positive values".to_string());
        }
        if self
            .frequencies
            .windows(2)
            .any(|window| window[1] <= window[0])
        {
            return Err("frequency grid must be strictly increasing".to_string());
        }
        if self.provenance.len() != self.frequencies.len() {
            return Err(format!(
                "provenance/frequency length mismatch: {} entries for {} frequencies",
                self.provenance.len(),
                self.frequencies.len()
            ));
        }
        if self.source_names.len() != self.pressures.len() {
            return Err(format!(
                "source-name/pressure length mismatch: {} names for {} sources",
                self.source_names.len(),
                self.pressures.len()
            ));
        }
        if self.pressures.is_empty() {
            return Err("pressure tensor has no sources".to_string());
        }
        if self
            .source_names
            .iter()
            .any(|name| !is_portable_filename_stem(name))
        {
            return Err(
                "source names must be non-empty, printable, and safe for output filenames"
                    .to_string(),
            );
        }
        // WAV/CSV exporters use source names as filename stems.  Compare a
        // case-folded key so `Left` and `left` cannot collide after transfer
        // to a case-insensitive desktop filesystem.
        let mut unique_source_names = HashSet::with_capacity(self.source_names.len());
        if self
            .source_names
            .iter()
            .any(|name| !unique_source_names.insert(name.to_lowercase()))
        {
            return Err(
                "source names must be unique case-insensitively for deterministic exports"
                    .to_string(),
            );
        }
        for (source_index, source) in self.pressures.iter().enumerate() {
            if source.is_empty() {
                return Err(format!(
                    "pressure tensor source {source_index} has no listening positions"
                ));
            }
            for (listener_index, pressures) in source.iter().enumerate() {
                if pressures.len() != self.frequencies.len() {
                    return Err(format!(
                        "pressure/frequency length mismatch for source {source_index}, listener {listener_index}: {} pressures for {} frequencies",
                        pressures.len(),
                        self.frequencies.len()
                    ));
                }
                if pressures
                    .iter()
                    .any(|pressure| !pressure.re.is_finite() || !pressure.im.is_finite())
                {
                    return Err(format!(
                        "non-finite pressure for source {source_index}, listener {listener_index}"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Require a fully solved frequency grid for a consumer that makes
    /// physical claims (geometry optimization, treatment fitting, or power
    /// prediction). Reduced-order and synthetic samples remain exportable for
    /// QA, but must not silently enter those workflows.
    pub fn require_solved(&self, operation: &str) -> Result<(), String> {
        self.validate_layout()?;
        if let Some((index, provenance)) = self
            .provenance
            .iter()
            .enumerate()
            .find(|(_, provenance)| **provenance != FrequencyProvenance::Solved)
        {
            return Err(format!(
                "{operation} requires solved samples; frequency index {index} is {}",
                provenance.as_str()
            ));
        }
        Ok(())
    }

    /// Return RoomEQ score labels exactly aligned with this response grid.
    ///
    /// This checked conversion prevents a stale provenance sidecar from being
    /// used with a different frequency grid when a RoomEQ score policy decides
    /// whether a generated response is eligible for optimization.
    pub fn roomeq_score_provenance(
        &self,
    ) -> Result<Vec<sonium_roomeq::cea2034::ScoreSampleProvenance>, String> {
        self.validate_layout()?;
        Ok(self
            .provenance
            .iter()
            .copied()
            .map(FrequencyProvenance::to_roomeq_score_provenance)
            .collect())
    }

    /// Build an output whose whole frequency grid shares one provenance.
    pub fn with_uniform_provenance(
        frequencies: Vec<f64>,
        pressures: Vec<Vec<Vec<Complex64>>>,
        source_names: Vec<String>,
        provenance: FrequencyProvenance,
    ) -> Self {
        let mask = vec![provenance; frequencies.len()];
        Self {
            frequencies,
            pressures,
            source_names,
            provenance: mask,
        }
    }

    /// All-solved output (full numerical model across the grid).
    pub fn solved(
        frequencies: Vec<f64>,
        pressures: Vec<Vec<Vec<Complex64>>>,
        source_names: Vec<String>,
    ) -> Self {
        Self::with_uniform_provenance(
            frequencies,
            pressures,
            source_names,
            FrequencyProvenance::Solved,
        )
    }

    /// Merge two already validated, ordered frequency segments.
    ///
    /// Dense low-frequency and FEM high-frequency runners use this boundary
    /// to build one response grid. Validate both segments before touching the
    /// nested pressure tensor so malformed runner output is reported as a
    /// typed error instead of causing an indexing panic or silently pairing a
    /// stale provenance mask with the wrong frequency.
    pub fn merge_frequency_segments(left: Self, right: Self) -> Result<Self, String> {
        left.validate_layout()
            .map_err(|error| format!("left frequency segment is invalid: {error}"))?;
        right
            .validate_layout()
            .map_err(|error| format!("right frequency segment is invalid: {error}"))?;

        if left.source_names != right.source_names {
            return Err("frequency segments have different source names".into());
        }
        if left.frequencies.last() >= right.frequencies.first() {
            return Err("frequency segments must be strictly ordered and non-overlapping".into());
        }
        if left.pressures.len() != right.pressures.len() {
            return Err("frequency segments have different source counts".into());
        }

        for (source_index, (left_source, right_source)) in
            left.pressures.iter().zip(&right.pressures).enumerate()
        {
            if left_source.len() != right_source.len() {
                return Err(format!(
                    "frequency segments have different listener counts for source {source_index}"
                ));
            }
            if left_source
                .iter()
                .zip(right_source)
                .any(|(left_listener, right_listener)| {
                    left_listener.is_empty() || right_listener.is_empty()
                })
            {
                return Err(format!(
                    "frequency segments contain an empty listener response for source {source_index}"
                ));
            }
        }

        let mut frequencies = left.frequencies;
        frequencies.extend(right.frequencies);
        let mut provenance = left.provenance;
        provenance.extend(right.provenance);
        let mut pressures = left.pressures;
        for (left_source, right_source) in pressures.iter_mut().zip(right.pressures) {
            for (left_listener, right_listener) in left_source.iter_mut().zip(right_source) {
                left_listener.extend(right_listener);
            }
        }

        let merged = Self {
            frequencies,
            pressures,
            source_names: left.source_names,
            provenance,
        };
        merged
            .validate_layout()
            .map_err(|error| format!("merged frequency segment is invalid: {error}"))?;
        Ok(merged)
    }

    /// Merge two solved frequency segments with a deterministic complex
    /// crossfade over their exact shared frequency points.
    ///
    /// The overlap must be represented by the same frequency values in both
    /// segments. The left response has weight one at the first shared point
    /// and the right response has weight one at the last shared point; a
    /// single shared point is averaged. This preserves complex phase while
    /// preventing a hard method switch at the dense-BLR/FEM boundary.
    ///
    /// Overlap samples must carry the same provenance. In particular, this
    /// prevents a solved/reduced-order or solved/synthetic blend from being
    /// accidentally promoted to `Solved` by a generic merge helper.
    pub fn merge_frequency_segments_with_overlap(left: Self, right: Self) -> Result<Self, String> {
        left.validate_layout()
            .map_err(|error| format!("left frequency segment is invalid: {error}"))?;
        right
            .validate_layout()
            .map_err(|error| format!("right frequency segment is invalid: {error}"))?;

        if left.source_names != right.source_names {
            return Err("frequency segments have different source names".into());
        }
        if left.pressures.len() != right.pressures.len() {
            return Err("frequency segments have different source counts".into());
        }
        for (source_index, (left_source, right_source)) in
            left.pressures.iter().zip(&right.pressures).enumerate()
        {
            if left_source.len() != right_source.len() {
                return Err(format!(
                    "frequency segments have different listener counts for source {source_index}"
                ));
            }
            if left_source
                .iter()
                .zip(right_source)
                .any(|(left_listener, right_listener)| {
                    left_listener.is_empty() || right_listener.is_empty()
                })
            {
                return Err(format!(
                    "frequency segments contain an empty listener response for source {source_index}"
                ));
            }
        }

        let mut shared = Vec::new();
        let (mut left_index, mut right_index) = (0, 0);
        while left_index < left.frequencies.len() && right_index < right.frequencies.len() {
            match left.frequencies[left_index].total_cmp(&right.frequencies[right_index]) {
                Ordering::Less => left_index += 1,
                Ordering::Greater => right_index += 1,
                Ordering::Equal => {
                    shared.push((left_index, right_index));
                    left_index += 1;
                    right_index += 1;
                }
            }
        }
        if shared.is_empty() {
            return Err("overlap merge requires at least one exact shared frequency point".into());
        }
        for &(left_frequency, right_frequency) in &shared {
            if left.provenance[left_frequency] != right.provenance[right_frequency] {
                return Err(format!(
                    "overlap provenance differs at {:.6} Hz",
                    left.frequencies[left_frequency]
                ));
            }
        }

        let first_shared = left.frequencies[shared[0].0];
        let last_shared = left.frequencies[shared.last().unwrap().0];
        let output_capacity = left.frequencies.len() + right.frequencies.len();
        let mut frequencies = Vec::with_capacity(output_capacity);
        let mut provenance = Vec::with_capacity(output_capacity);
        let mut pressures = left
            .pressures
            .iter()
            .map(|source| {
                source
                    .iter()
                    .map(|_| Vec::with_capacity(output_capacity))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let (mut left_index, mut right_index) = (0, 0);
        while left_index < left.frequencies.len() || right_index < right.frequencies.len() {
            if right_index == right.frequencies.len()
                || (left_index < left.frequencies.len()
                    && left.frequencies[left_index].total_cmp(&right.frequencies[right_index])
                        == Ordering::Less)
            {
                frequencies.push(left.frequencies[left_index]);
                provenance.push(left.provenance[left_index]);
                for (output_source, left_source) in pressures.iter_mut().zip(&left.pressures) {
                    for (output_listener, left_listener) in
                        output_source.iter_mut().zip(left_source)
                    {
                        output_listener.push(left_listener[left_index]);
                    }
                }
                left_index += 1;
            } else if left_index == left.frequencies.len()
                || right.frequencies[right_index].total_cmp(&left.frequencies[left_index])
                    == Ordering::Less
            {
                frequencies.push(right.frequencies[right_index]);
                provenance.push(right.provenance[right_index]);
                for (output_source, right_source) in pressures.iter_mut().zip(&right.pressures) {
                    for (output_listener, right_listener) in
                        output_source.iter_mut().zip(right_source)
                    {
                        output_listener.push(right_listener[right_index]);
                    }
                }
                right_index += 1;
            } else {
                debug_assert_eq!(left.frequencies[left_index], right.frequencies[right_index]);
                let frequency = left.frequencies[left_index];
                let left_weight = if first_shared == last_shared {
                    0.5
                } else {
                    ((last_shared - frequency) / (last_shared - first_shared)).clamp(0.0, 1.0)
                };
                frequencies.push(frequency);
                provenance.push(left.provenance[left_index]);
                for ((output_source, left_source), right_source) in pressures
                    .iter_mut()
                    .zip(&left.pressures)
                    .zip(&right.pressures)
                {
                    for ((output_listener, left_listener), right_listener) in
                        output_source.iter_mut().zip(left_source).zip(right_source)
                    {
                        output_listener.push(
                            left_listener[left_index] * left_weight
                                + right_listener[right_index] * (1.0 - left_weight),
                        );
                    }
                }
                left_index += 1;
                right_index += 1;
            }
        }

        let merged = Self {
            frequencies,
            pressures,
            source_names: left.source_names,
            provenance,
        };
        merged
            .validate_layout()
            .map_err(|error| format!("overlap merge result is invalid: {error}"))?;
        Ok(merged)
    }
}

/// Return whether `name` can be used unchanged as a source-export filename
/// stem on the supported desktop platforms.
///
/// The WAV exporter derives names directly from this value, so admission must
/// follow the strictest common filename rules rather than just the host's.
pub(crate) fn is_portable_filename_stem(name: &str) -> bool {
    if name.trim().is_empty()
        || name == "."
        || name == ".."
        || name.ends_with(['.', ' '])
        || name.chars().any(|character| {
            character.is_control()
                || matches!(
                    character,
                    '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*'
                )
        })
    {
        return false;
    }

    let uppercase = name.to_ascii_uppercase();
    if matches!(uppercase.as_str(), "CON" | "PRN" | "AUX" | "NUL") {
        return false;
    }
    if let Some(number) = uppercase
        .strip_prefix("COM")
        .or_else(|| uppercase.strip_prefix("LPT"))
    {
        return !matches!(number, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9");
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_guard_rejects_misaligned_and_non_solved_consumers() {
        let mut output = SimulationOutput::solved(
            vec![100.0, 200.0],
            vec![vec![vec![Complex64::new(1.0, 0.0); 2]]],
            vec!["left".to_string()],
        );
        output.provenance[1] = FrequencyProvenance::Synthetic;
        let error = output.require_solved("physical optimization").unwrap_err();
        assert!(error.contains("synthetic"));

        output.provenance.clear();
        assert!(output.validate_layout().unwrap_err().contains("provenance"));
    }

    #[test]
    fn all_non_solved_provenance_categories_are_rejected_by_physical_consumers() {
        for provenance in [
            FrequencyProvenance::ReducedOrder,
            FrequencyProvenance::Interpolated,
            FrequencyProvenance::Geometric,
            FrequencyProvenance::Synthetic,
        ] {
            let mut output = SimulationOutput::solved(
                vec![100.0],
                vec![vec![vec![Complex64::new(1.0, 0.0)]]],
                vec!["left".to_string()],
            );
            output.provenance[0] = provenance;
            let error = output.require_solved("physical optimization").unwrap_err();
            assert!(error.contains(provenance.as_str()));
        }
    }

    #[test]
    fn roomeq_provenance_bridge_is_exact_and_layout_checked() {
        use sonium_roomeq::cea2034::ScoreSampleProvenance;

        let output = SimulationOutput {
            frequencies: vec![40.0, 80.0, 160.0, 320.0, 640.0],
            pressures: vec![vec![vec![Complex64::new(1.0, 0.0); 5]]],
            source_names: vec!["speaker".to_string()],
            provenance: vec![
                FrequencyProvenance::Solved,
                FrequencyProvenance::ReducedOrder,
                FrequencyProvenance::Interpolated,
                FrequencyProvenance::Geometric,
                FrequencyProvenance::Synthetic,
            ],
        };
        assert_eq!(
            output.roomeq_score_provenance().unwrap(),
            vec![
                ScoreSampleProvenance::Solved,
                ScoreSampleProvenance::ReducedOrder,
                ScoreSampleProvenance::Interpolated,
                ScoreSampleProvenance::Geometric,
                ScoreSampleProvenance::Synthetic,
            ]
        );

        let mut stale = output;
        stale.provenance.pop();
        assert!(stale.roomeq_score_provenance().is_err());
    }

    #[test]
    fn layout_rejects_non_monotone_or_empty_pressure_axes() {
        let base =
            |frequencies, pressures, names| SimulationOutput::solved(frequencies, pressures, names);
        assert!(
            base(
                vec![200.0, 100.0],
                vec![vec![vec![Complex64::new(1.0, 0.0); 2]]],
                vec!["source".into()]
            )
            .validate_layout()
            .unwrap_err()
            .contains("strictly increasing")
        );
        assert!(
            base(vec![100.0], Vec::new(), Vec::new())
                .validate_layout()
                .unwrap_err()
                .contains("no sources")
        );
        assert!(
            base(vec![100.0], vec![Vec::new()], vec!["source".into()])
                .validate_layout()
                .unwrap_err()
                .contains("listening positions")
        );
        assert!(
            base(
                vec![100.0],
                vec![vec![vec![Complex64::new(1.0, 0.0)]]],
                vec!["../outside".into()]
            )
            .validate_layout()
            .unwrap_err()
            .contains("safe for output")
        );
        assert!(
            base(
                vec![100.0],
                vec![
                    vec![vec![Complex64::new(1.0, 0.0)]],
                    vec![vec![Complex64::new(1.0, 0.0)]]
                ],
                vec!["duplicate".into(), "duplicate".into()]
            )
            .validate_layout()
            .unwrap_err()
            .contains("unique")
        );
        assert!(
            base(
                vec![100.0],
                vec![
                    vec![vec![Complex64::new(1.0, 0.0)]],
                    vec![vec![Complex64::new(1.0, 0.0)]],
                ],
                vec!["Left".into(), "left".into()]
            )
            .validate_layout()
            .unwrap_err()
            .contains("case-insensitively")
        );
    }

    #[test]
    fn layout_rejects_non_portable_source_filename_stems() {
        let output_for = |name: &str| {
            SimulationOutput::solved(
                vec![100.0],
                vec![vec![vec![Complex64::new(1.0, 0.0)]]],
                vec![name.to_owned()],
            )
        };
        for name in [
            "colon:name",
            "quote\"name",
            "wildcard*name",
            "trailing-dot.",
            "trailing-space ",
            "CON",
            "lpt9",
        ] {
            assert!(
                output_for(name)
                    .validate_layout()
                    .unwrap_err()
                    .contains("safe for output"),
                "expected {name:?} to be rejected"
            );
        }
        assert!(output_for("front-left").validate_layout().is_ok());
    }

    #[test]
    fn frequency_segment_merge_preserves_ordered_provenance_and_pressure_axes() {
        let left = SimulationOutput::with_uniform_provenance(
            vec![40.0, 80.0],
            vec![vec![
                vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
                vec![Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
            ]],
            vec!["left".into()],
            FrequencyProvenance::Solved,
        );
        let right = SimulationOutput::with_uniform_provenance(
            vec![160.0],
            vec![vec![
                vec![Complex64::new(5.0, 0.0)],
                vec![Complex64::new(6.0, 0.0)],
            ]],
            vec!["left".into()],
            FrequencyProvenance::Synthetic,
        );

        let merged = SimulationOutput::merge_frequency_segments(left, right).unwrap();
        assert_eq!(merged.frequencies, vec![40.0, 80.0, 160.0]);
        assert_eq!(
            merged.provenance,
            vec![
                FrequencyProvenance::Solved,
                FrequencyProvenance::Solved,
                FrequencyProvenance::Synthetic,
            ]
        );
        assert_eq!(
            merged.pressures[0][1],
            vec![
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(6.0, 0.0),
            ]
        );
    }

    #[test]
    fn frequency_segment_merge_rejects_overlap_order_and_malformed_axes() {
        let segment = |frequencies: Vec<f64>| {
            SimulationOutput::solved(
                frequencies.clone(),
                vec![vec![vec![Complex64::new(1.0, 0.0); frequencies.len()]]],
                vec!["source".into()],
            )
        };
        assert!(
            SimulationOutput::merge_frequency_segments(segment(vec![100.0]), segment(vec![100.0]))
                .unwrap_err()
                .contains("non-overlapping")
        );
        assert!(
            SimulationOutput::merge_frequency_segments(segment(vec![200.0]), segment(vec![100.0]))
                .unwrap_err()
                .contains("non-overlapping")
        );

        let malformed = SimulationOutput {
            frequencies: vec![100.0],
            pressures: vec![Vec::new()],
            source_names: vec!["source".into()],
            provenance: vec![FrequencyProvenance::Solved],
        };
        assert!(
            SimulationOutput::merge_frequency_segments(malformed, segment(vec![200.0]))
                .unwrap_err()
                .contains("left frequency segment is invalid")
        );

        let different_sources = SimulationOutput::with_uniform_provenance(
            vec![200.0],
            vec![vec![vec![Complex64::new(1.0, 0.0)]]],
            vec!["other".into()],
            FrequencyProvenance::Solved,
        );
        assert!(
            SimulationOutput::merge_frequency_segments(segment(vec![100.0]), different_sources)
                .unwrap_err()
                .contains("source names")
        );

        let listener_mismatch = SimulationOutput::with_uniform_provenance(
            vec![200.0],
            vec![vec![
                vec![Complex64::new(1.0, 0.0)],
                vec![Complex64::new(2.0, 0.0)],
            ]],
            vec!["source".into()],
            FrequencyProvenance::Solved,
        );
        assert!(
            SimulationOutput::merge_frequency_segments(segment(vec![100.0]), listener_mismatch)
                .unwrap_err()
                .contains("listener counts")
        );
    }

    #[test]
    fn overlap_merge_crossfades_complex_pressure_on_exact_shared_grid() {
        let left = SimulationOutput::with_uniform_provenance(
            vec![100.0, 200.0, 300.0],
            vec![vec![vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, 2.0),
                Complex64::new(3.0, 3.0),
            ]]],
            vec!["source".into()],
            FrequencyProvenance::Solved,
        );
        let right = SimulationOutput::with_uniform_provenance(
            vec![200.0, 300.0, 400.0],
            vec![vec![vec![
                Complex64::new(20.0, 20.0),
                Complex64::new(30.0, 30.0),
                Complex64::new(40.0, 40.0),
            ]]],
            vec!["source".into()],
            FrequencyProvenance::Solved,
        );

        let merged = SimulationOutput::merge_frequency_segments_with_overlap(left, right).unwrap();
        assert_eq!(merged.frequencies, vec![100.0, 200.0, 300.0, 400.0]);
        assert_eq!(merged.provenance, vec![FrequencyProvenance::Solved; 4]);
        assert_eq!(merged.pressures[0][0][0], Complex64::new(1.0, 1.0));
        assert_eq!(merged.pressures[0][0][1], Complex64::new(2.0, 2.0));
        assert_eq!(merged.pressures[0][0][2], Complex64::new(30.0, 30.0));
        assert_eq!(merged.pressures[0][0][3], Complex64::new(40.0, 40.0));
    }

    #[test]
    fn overlap_merge_averages_single_shared_point_and_rejects_unsafe_inputs() {
        let segment = |frequencies: Vec<f64>, value: f64, provenance| {
            SimulationOutput::with_uniform_provenance(
                frequencies.clone(),
                vec![vec![vec![Complex64::new(value, 0.0); frequencies.len()]]],
                vec!["source".into()],
                provenance,
            )
        };
        let merged = SimulationOutput::merge_frequency_segments_with_overlap(
            segment(vec![100.0, 200.0], 2.0, FrequencyProvenance::Solved),
            segment(vec![200.0, 300.0], 10.0, FrequencyProvenance::Solved),
        )
        .unwrap();
        assert_eq!(merged.frequencies, vec![100.0, 200.0, 300.0]);
        assert_eq!(merged.pressures[0][0][1], Complex64::new(6.0, 0.0));

        assert!(
            SimulationOutput::merge_frequency_segments_with_overlap(
                segment(vec![100.0], 1.0, FrequencyProvenance::Solved),
                segment(vec![200.0], 1.0, FrequencyProvenance::Solved),
            )
            .unwrap_err()
            .contains("exact shared frequency")
        );
        assert!(
            SimulationOutput::merge_frequency_segments_with_overlap(
                segment(vec![100.0, 200.0], 1.0, FrequencyProvenance::Solved),
                segment(vec![200.0, 300.0], 1.0, FrequencyProvenance::Synthetic),
            )
            .unwrap_err()
            .contains("overlap provenance")
        );

        let empty_listener = SimulationOutput {
            frequencies: vec![100.0, 200.0],
            pressures: vec![vec![Vec::new()]],
            source_names: vec!["source".into()],
            provenance: vec![FrequencyProvenance::Solved; 2],
        };
        assert!(
            SimulationOutput::merge_frequency_segments_with_overlap(
                empty_listener,
                segment(vec![100.0, 200.0], 1.0, FrequencyProvenance::Solved),
            )
            .unwrap_err()
            .contains("invalid")
        );
    }
}
