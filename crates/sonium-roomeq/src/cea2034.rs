//! CEA-2034 (Spinorama) speaker measurement metrics.
//!
//! Re-implementation of the autoeq CEA-2034 scoring logic, using `Vec<f64>`
//! instead of `ndarray` so that the sonium workspace can drop the ndarray
//! dependency.

use serde::{Deserialize, Deserializer, Serialize};

// ---------------------------------------------------------------------------
// Core spinorama computation
// ---------------------------------------------------------------------------

/// Weighted mean of finite SPL values, evaluated entirely in the log domain.
/// Zero-weight samples are excluded before choosing the scale. Scaling both
/// SPL and weights avoids underflow even for subnormal nonzero weights.
fn energy_mean_db(samples: impl Iterator<Item = (f64, f64)> + Clone) -> f64 {
    let samples = samples.filter(|(_, weight)| *weight > 0.0);
    let base = samples
        .clone()
        .map(|(db, _)| db)
        .fold(f64::NEG_INFINITY, f64::max);
    let term = |(db, weight): (f64, f64)| (db - base) / 10.0 + weight.log10();
    let log_scale = samples.clone().map(term).fold(f64::NEG_INFINITY, f64::max);
    let sum = samples
        .clone()
        .map(|sample| 10f64.powf(term(sample) - log_scale))
        .sum::<f64>();
    let weights = samples.map(|(_, weight)| weight).sum::<f64>();
    // A weighted average cannot exceed its largest input; suppress only
    // positive roundoff in the logarithmic correction.
    base + (10.0 * (log_scale + sum.log10() - weights.log10())).min(0.0)
}

/// Shared nested/contiguous aggregation without exponentiating absolute SPL.
fn cea2034_from_spl(
    n_cols: usize,
    sample: impl Fn(usize, usize) -> f64,
    idx: &[Vec<usize>],
    weights: &[f64],
) -> Vec<Vec<f64>> {
    let mut cea = vec![vec![0.0; n_cols]; 9];
    for col in 0..n_cols {
        for (group, rows) in idx.iter().take(7).enumerate() {
            cea[group][col] = energy_mean_db(rows.iter().map(|&row| (sample(row, col), 1.0)));
        }
        cea[1][col] = energy_mean_db(cea[2..=6].iter().map(|row| (row[col], 1.0)));
        cea[7][col] = energy_mean_db(idx[7].iter().map(|&row| (sample(row, col), weights[row])));
        cea[8][col] = energy_mean_db(
            [
                (cea[0][col], 0.12),
                (cea[1][col], 0.44),
                (cea[7][col], 0.44),
            ]
            .into_iter(),
        );
    }
    cea
}

#[cfg(test)]
fn spl2pressure(spl: &[f64]) -> Vec<f64> {
    spl.iter()
        .map(|&v| 10f64.powf((v - 105.0) / 20.0))
        .collect()
}

#[cfg(test)]
fn pressure2spl(p: Vec<f64>) -> Vec<f64> {
    p.into_iter().map(|v| 20.0 * v.log10() + 105.0).collect()
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Compute Mean Absolute Deviation (MAD) for a slice of SPL values.
fn mad(spl: &[f64], imin: usize, imax: usize) -> f64 {
    if imin >= imax {
        return 0.0;
    }
    let slice = &spl[imin..imax];
    let m = slice.iter().sum::<f64>() / slice.len() as f64;
    let diffs: f64 = slice.iter().map(|&v| (v - m).abs()).sum();
    diffs / slice.len() as f64
}

/// Compute the coefficient of determination (R-squared) between two arrays.
fn r_squared(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() {
        return f64::NAN;
    }
    let n = x.len() as f64;
    if n == 0.0 {
        return f64::NAN;
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;

    let num: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xv, &yv)| (xv - mx) * (yv - my))
        .sum();
    let sxx: f64 = x.iter().map(|&xv| (xv - mx) * (xv - mx)).sum();
    let syy: f64 = y.iter().map(|&yv| (yv - my) * (yv - my)).sum();

    if sxx == 0.0 || syy == 0.0 {
        return f64::NAN;
    }
    let r = num / (sxx.sqrt() * syy.sqrt());
    r * r
}

/// Generate octave band frequencies.
///
/// Fallible octave-band table; `None` when `count < 2`.
pub fn try_octave(count: usize) -> Option<Vec<(f64, f64, f64)>> {
    if count < 2 {
        return None;
    }
    let reference = 1290.0_f64;
    let p = 2.0_f64.powf(1.0 / count as f64);
    let p_band = 2.0_f64.powf(1.0 / (2.0 * count as f64));
    let o_iter: i32 = (count as i32 * 10 + 1) / 2;
    let mut centers: Vec<f64> = Vec::with_capacity((o_iter as usize) * 2 + 1);
    for i in (1..=o_iter).rev() {
        centers.push(reference / p.powi(i));
    }
    centers.push(reference);
    for i in 1..=o_iter {
        let center = reference * p.powi(i);
        if (center / p_band) <= 20000.0 {
            centers.push(reference * p.powi(i));
        }
    }
    let bands: Vec<(f64, f64, f64)> = centers
        .into_iter()
        .map(|c| (c / p_band, c, c * p_band))
        .collect();
    Some(bands)
}

/// Generate octave band frequencies.
///
/// # Panics
/// * If count is less than 2 (use [`try_octave`] for a fallible variant)
pub fn octave(count: usize) -> Vec<(f64, f64, f64)> {
    try_octave(count).expect("count (N) must be >= 2")
}

/// Compute octave band intervals for a given frequency array.
pub fn octave_intervals(count: usize, freq: &[f64]) -> Vec<(usize, usize)> {
    let bands = octave(count);

    let min_freq = freq.first().copied().unwrap_or(0.0);
    let band_min_freq = 100.0_f64.max(min_freq);

    let mut out = Vec::new();
    for (low, center, high) in bands.into_iter() {
        if center < band_min_freq || center > 12000.0 {
            continue;
        }
        let imin = freq.iter().position(|&f| f >= low).unwrap_or(freq.len());
        let imax = freq.iter().position(|&f| f > high).unwrap_or(freq.len());
        if imin < imax {
            out.push((imin, imax));
        }
    }
    out
}

/// Compute the Narrow Band Deviation (NBD) metric.
pub fn nbd(intervals: &[(usize, usize)], spl: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut cnt = 0.0;
    for &(imin, imax) in intervals.iter() {
        if imin >= imax {
            continue;
        }
        let v = mad(spl, imin, imax);
        if v.is_finite() {
            sum += v;
            cnt += 1.0;
        }
    }
    if cnt == 0.0 { f64::NAN } else { sum / cnt }
}

/// Compute the Low Frequency Extension (LFX) metric.
pub fn lfx(freq: &[f64], lw: &[f64], sp: &[f64]) -> f64 {
    let lw_min = freq.iter().position(|&f| f >= 300.0).unwrap_or(freq.len());
    let lw_max = freq.iter().position(|&f| f > 10000.0).unwrap_or(freq.len());
    if lw_min >= lw_max {
        return 300.0_f64.log10();
    }
    let lw_ref = lw[lw_min..lw_max].iter().sum::<f64>() / (lw_max - lw_min) as f64 - 6.0;

    let mut indices: Vec<usize> = Vec::new();
    for (i, (&f, &spv)) in freq.iter().zip(sp.iter()).enumerate() {
        if f <= 300.0 && spv <= lw_ref {
            indices.push(i);
        }
    }
    if indices.is_empty() {
        return freq.first().copied().unwrap_or(50.0).log10();
    }

    let mut last_idx = indices[0];
    for &idx in indices.iter().skip(1) {
        if idx == last_idx + 1 {
            last_idx = idx;
        } else {
            break;
        }
    }

    let next_idx = last_idx + 1;
    if next_idx < freq.len() {
        freq[next_idx].log10()
    } else {
        300.0_f64.log10()
    }
}

/// Compute the Smoothness Metric (SM).
pub fn sm(freq: &[f64], spl: &[f64]) -> f64 {
    let f_min = freq.iter().position(|&f| f > 100.0).unwrap_or(freq.len());
    let f_max = freq
        .iter()
        .position(|&f| f >= 16000.0)
        .unwrap_or(freq.len());
    if f_min >= f_max {
        return f64::NAN;
    }
    let x: Vec<f64> = freq[f_min..f_max].iter().map(|&v| v.log10()).collect();
    let y = &spl[f_min..f_max];
    r_squared(&x, y)
}

// ---------------------------------------------------------------------------
// Public scoring API
// ---------------------------------------------------------------------------

/// Metrics computed for the CEA2034 preference score.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScoreMetrics {
    /// Narrow Band Deviation for on-axis response.
    pub nbd_on: f64,
    /// Narrow Band Deviation for predicted in-room response.
    pub nbd_pir: f64,
    /// Low Frequency Extension metric.
    pub lfx: f64,
    /// Smoothness Metric for predicted in-room response.
    pub sm_pir: f64,
    /// Overall preference score.
    pub pref_score: f64,
}

/// Origin of one frequency sample consumed by a RoomEQ score.
///
/// The values deliberately distinguish a full physical solve from reduced,
/// interpolated, geometric, and synthetic data.  A preference score is only
/// meaningful when the caller makes that fidelity choice explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScoreSampleProvenance {
    /// A calibrated measurement.
    Measured,
    /// A full physical simulation.
    Solved,
    /// A reduced-order-model result.
    ReducedOrder,
    /// An interpolation between evaluated samples.
    Interpolated,
    /// A geometric-acoustics approximation.
    Geometric,
    /// A generated or extrapolated sample without a physical solve.
    Synthetic,
}

impl ScoreSampleProvenance {
    /// Stable label used in score-policy diagnostics and artifacts.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::Solved => "solved",
            Self::ReducedOrder => "reduced-order",
            Self::Interpolated => "interpolated",
            Self::Geometric => "geometric",
            Self::Synthetic => "synthetic",
        }
    }

    /// Whether this sample is independently measured or produced by the full
    /// physical model rather than an approximation or extrapolation.
    pub fn is_measured_or_solved(self) -> bool {
        matches!(self, Self::Measured | Self::Solved)
    }
}

/// Policy governing whether non-physical synthetic bands may be scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScoreProvenancePolicy {
    /// Reject synthetic samples so they cannot silently influence optimization.
    #[default]
    RejectSynthetic,
    /// Permit synthetic samples, retaining their explicit output provenance.
    AllowSynthetic,
    /// Accept only measured or full-solver samples. Reduced, interpolated,
    /// geometric, and synthetic bands must be promoted explicitly before they
    /// can influence a high-confidence optimization objective.
    RequireMeasuredOrSolved,
}

/// CEA-2034 score accompanied by the fidelity of every input frequency.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProvenancedScoreMetrics {
    /// Numerical CEA-2034 metrics.
    pub metrics: ScoreMetrics,
    /// Per-frequency origin in the same order as the scored frequency grid.
    pub frequency_provenance: Vec<ScoreSampleProvenance>,
    /// Whether the scalar metrics include any synthetic sample.
    pub contains_synthetic: bool,
    /// Whether the scalar metrics include any reduced, interpolated,
    /// geometric, or synthetic sample.
    pub contains_non_authoritative: bool,
}

#[derive(Deserialize)]
struct ProvenancedScoreMetricsWire {
    metrics: ScoreMetricsWire,
    frequency_provenance: Vec<ScoreSampleProvenance>,
    contains_synthetic: bool,
    contains_non_authoritative: bool,
}

#[derive(Deserialize)]
struct ScoreMetricsWire {
    nbd_on: f64,
    nbd_pir: f64,
    lfx: f64,
    sm_pir: f64,
    pref_score: f64,
}

impl ProvenancedScoreMetrics {
    /// Validate a serialized score artifact before it is retained or promoted.
    ///
    /// The disclosure flags are derived evidence, not caller-controlled
    /// annotations. Recomputing them here prevents a JSON report from hiding a
    /// synthetic or reduced-order band behind stale `false` flags.
    pub fn validate(&self) -> Result<(), Cea2034InputError> {
        if self.frequency_provenance.is_empty() {
            return Err(Cea2034InputError(
                "CEA-2034 provenance score must retain at least one frequency label".into(),
            ));
        }
        if [
            self.metrics.nbd_on,
            self.metrics.nbd_pir,
            self.metrics.lfx,
            self.metrics.sm_pir,
            self.metrics.pref_score,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err(Cea2034InputError(
                "CEA-2034 provenance score contains non-finite metrics".into(),
            ));
        }
        let contains_synthetic = self
            .frequency_provenance
            .contains(&ScoreSampleProvenance::Synthetic);
        let contains_non_authoritative = self
            .frequency_provenance
            .iter()
            .any(|provenance| !provenance.is_measured_or_solved());
        if self.contains_synthetic != contains_synthetic
            || self.contains_non_authoritative != contains_non_authoritative
        {
            return Err(Cea2034InputError(
                "CEA-2034 provenance disclosure flags do not match frequency labels".into(),
            ));
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for ProvenancedScoreMetrics {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProvenancedScoreMetricsWire::deserialize(deserializer)?;
        let score = Self {
            metrics: ScoreMetrics {
                nbd_on: wire.metrics.nbd_on,
                nbd_pir: wire.metrics.nbd_pir,
                lfx: wire.metrics.lfx,
                sm_pir: wire.metrics.sm_pir,
                pref_score: wire.metrics.pref_score,
            },
            frequency_provenance: wire.frequency_provenance,
            contains_synthetic: wire.contains_synthetic,
            contains_non_authoritative: wire.contains_non_authoritative,
        };
        score.validate().map_err(serde::de::Error::custom)?;
        Ok(score)
    }
}

/// Invalid matrix dimensions or angle mapping for CEA-2034 scoring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cea2034InputError(pub String);

impl std::fmt::Display for Cea2034InputError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Cea2034InputError {}

/// Borrowed row-major SPL matrix for large CEA-2034 data sets.
///
/// `spl_db` contains `row_count * frequency_count` values, ordered by row
/// then frequency. The checked constructor keeps the source storage contiguous
/// and avoids requiring callers to allocate one `Vec` per angle or seat.
#[derive(Debug, Clone, Copy)]
pub struct ContiguousCea2034InputRef<'a> {
    spl_db: &'a [f64],
    row_count: usize,
    frequency_count: usize,
}

impl<'a> ContiguousCea2034InputRef<'a> {
    /// Validate a non-empty finite row-major SPL matrix.
    pub fn new(
        spl_db: &'a [f64],
        row_count: usize,
        frequency_count: usize,
    ) -> Result<Self, Cea2034InputError> {
        if row_count == 0 || frequency_count == 0 {
            return Err(Cea2034InputError(
                "CEA-2034 contiguous SPL matrix must have non-zero row and frequency counts".into(),
            ));
        }
        let expected = row_count.checked_mul(frequency_count).ok_or_else(|| {
            Cea2034InputError("CEA-2034 contiguous SPL matrix dimensions overflow".into())
        })?;
        if spl_db.len() != expected {
            return Err(Cea2034InputError(format!(
                "CEA-2034 contiguous SPL matrix has {} values, expected {expected}",
                spl_db.len()
            )));
        }
        if spl_db.iter().any(|value| !value.is_finite()) {
            return Err(Cea2034InputError(
                "CEA-2034 contiguous SPL values must be finite".into(),
            ));
        }
        Ok(Self {
            spl_db,
            row_count,
            frequency_count,
        })
    }

    /// Number of SPL rows (for example, angles or seats).
    pub const fn row_count(self) -> usize {
        self.row_count
    }

    /// Number of frequency samples in every row.
    pub const fn frequency_count(self) -> usize {
        self.frequency_count
    }

    fn row(self, row: usize) -> &'a [f64] {
        let start = row * self.frequency_count;
        &self.spl_db[start..start + self.frequency_count]
    }
}

fn validate_cea2034_inputs(
    spl: &[Vec<f64>],
    idx: &[Vec<usize>],
    weights: &[f64],
) -> Result<(), Cea2034InputError> {
    let Some(first_row) = spl.first() else {
        return Err(Cea2034InputError(
            "CEA-2034 SPL matrix must contain at least one row".into(),
        ));
    };
    if first_row.is_empty() {
        return Err(Cea2034InputError(
            "CEA-2034 SPL rows must contain at least one frequency".into(),
        ));
    }
    if spl.iter().any(|row| row.len() != first_row.len()) {
        return Err(Cea2034InputError(
            "CEA-2034 SPL rows must be rectangular".into(),
        ));
    }
    if spl.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Cea2034InputError(
            "CEA-2034 SPL values must be finite".into(),
        ));
    }
    // CEA-2034 carries LW, an overwritten ER slot, five ER component rows,
    // and the weighted SP group. The ER aggregate reads component rows 2..=6,
    // so seven groups would leave one component at the zero-initialized SP
    // slot and silently bias the result.
    if idx.len() != 8 {
        return Err(Cea2034InputError(
            "CEA-2034 index groups must contain exactly eight groups (LW, five ER components, and SP)".into(),
        ));
    }
    if weights.len() < spl.len()
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(Cea2034InputError(
            "CEA-2034 weights must be finite, non-negative, and cover every SPL row".into(),
        ));
    }
    for (group, rows) in idx.iter().enumerate() {
        if rows.is_empty() {
            return Err(Cea2034InputError(format!(
                "CEA-2034 index group {group} must not be empty"
            )));
        }
        if rows.iter().any(|&row| row >= spl.len()) {
            return Err(Cea2034InputError(format!(
                "CEA-2034 index group {group} contains a row outside the SPL matrix"
            )));
        }
    }
    let weighted_rows = &idx[idx.len() - 1];
    let weight_sum: f64 = weighted_rows.iter().map(|&row| weights[row]).sum();
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(Cea2034InputError(
            "CEA-2034 SP weights must have a positive finite sum".into(),
        ));
    }
    Ok(())
}

/// Checked CEA-2034 spinorama computation.
///
/// `idx` follows the CTA/CEA layout: listening window at group 0, the
/// overwritten early-reflections slot at group 1, five early-reflection
/// component groups at groups 2..=6, and weighted sound power as the final
/// group (therefore at least eight groups total).
///
/// Curve semantics follow CTA-2034-A ("Standard Method of Measurement for
/// In-Home Loudspeakers") and the preference model of Olive et al. (2013),
/// "A listening test versus measurements / A statistical model that predicts
/// listener preference": listening window (LW), early reflections (ER),
/// sound power (SP), predicted in-room response (PIR), and the directivity
/// index DI = LW − SP (see [`try_directivity_index`).
pub fn try_cea2034(
    spl: &[Vec<f64>],
    idx: &[Vec<usize>],
    weights: &[f64],
) -> Result<Vec<Vec<f64>>, Cea2034InputError> {
    validate_cea2034_inputs(spl, idx, weights)?;
    let result = cea2034_from_spl(spl[0].len(), |row, col| spl[row][col], idx, weights);
    if result.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Cea2034InputError(
            "CEA-2034 computation produced non-finite values".into(),
        ));
    }
    Ok(result)
}

/// Directivity-index (DI) curve: listening-window SPL minus sound-power SPL
/// per frequency bin, per CTA-2034-A.
///
/// Both curves must be finite, non-empty, and the same length; a length
/// mismatch or non-finite sample is a [`Cea2034InputError`], not a silent
/// truncation, so caller-side LW/SP mix-ups fail loudly.
pub fn try_directivity_index(lw_db: &[f64], sp_db: &[f64]) -> Result<Vec<f64>, Cea2034InputError> {
    if lw_db.is_empty() || lw_db.len() != sp_db.len() {
        return Err(Cea2034InputError(
            "directivity index requires non-empty LW and SP curves of equal length".into(),
        ));
    }
    if lw_db
        .iter()
        .chain(sp_db.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Cea2034InputError(
            "directivity index curves must be finite".into(),
        ));
    }
    let result: Vec<_> = lw_db
        .iter()
        .zip(sp_db.iter())
        .map(|(lw, sp)| lw - sp)
        .collect();
    if result.iter().any(|value| !value.is_finite()) {
        return Err(Cea2034InputError("directivity index overflowed".into()));
    }
    Ok(result)
}

/// Checked CEA-2034 computation for a borrowed contiguous SPL matrix.
///
/// The source matrix is row-major and is never expanded into nested input
/// vectors. Aggregates retain the historical `Vec<Vec<f64>>` output shape for
/// compatibility with downstream spinorama consumers.
pub fn try_cea2034_contiguous(
    input: ContiguousCea2034InputRef<'_>,
    idx: &[Vec<usize>],
    weights: &[f64],
) -> Result<Vec<Vec<f64>>, Cea2034InputError> {
    if idx.len() != 8 {
        return Err(Cea2034InputError(
            "CEA-2034 index groups must contain exactly eight groups (LW, five ER components, and SP)".into(),
        ));
    }
    if weights.len() < input.row_count
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(Cea2034InputError(
            "CEA-2034 weights must be finite, non-negative, and cover every SPL row".into(),
        ));
    }
    for (group, rows) in idx.iter().enumerate() {
        if rows.is_empty() {
            return Err(Cea2034InputError(format!(
                "CEA-2034 index group {group} must not be empty"
            )));
        }
        if rows.iter().any(|&row| row >= input.row_count) {
            return Err(Cea2034InputError(format!(
                "CEA-2034 index group {group} contains a row outside the SPL matrix"
            )));
        }
    }
    let weighted_rows = &idx[idx.len() - 1];
    let weight_sum: f64 = weighted_rows.iter().map(|&row| weights[row]).sum();
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(Cea2034InputError(
            "CEA-2034 SP weights must have a positive finite sum".into(),
        ));
    }

    let result = cea2034_from_spl(
        input.frequency_count,
        |row, col| input.row(row)[col],
        idx,
        weights,
    );
    if result.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Cea2034InputError(
            "CEA-2034 computation produced non-finite values".into(),
        ));
    }
    Ok(result)
}

/// Backward-compatible CEA-2034 wrapper. New callers should use
/// [`try_cea2034`] to reject malformed matrices without a panic.
pub fn cea2034(spl: &[Vec<f64>], idx: &[Vec<usize>], weights: &[f64]) -> Vec<Vec<f64>> {
    try_cea2034(spl, idx, weights).unwrap_or_else(|error| panic!("invalid CEA-2034 input: {error}"))
}

/// Measurement plane carrying a labelled spinorama angle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpinoramaPlane {
    Horizontal,
    Vertical,
}

/// Explicit location of a semantically significant spinorama curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpinoramaAngle {
    pub plane: SpinoramaPlane,
    pub degrees: f64,
}

/// Rectangular, finite spinorama measurements with explicit angle labels.
#[derive(Debug)]
pub struct ValidatedSpinorama<'a> {
    frequencies: &'a [f64],
    horizontal_spl: &'a [Vec<f64>],
    vertical_spl: &'a [Vec<f64>],
    on_axis: SpinoramaAngle,
    on_axis_row: usize,
}

impl<'a> ValidatedSpinorama<'a> {
    pub fn new(
        frequencies: &'a [f64],
        horizontal_angles: &[f64],
        horizontal_spl: &'a [Vec<f64>],
        vertical_angles: &[f64],
        vertical_spl: &'a [Vec<f64>],
        on_axis: SpinoramaAngle,
    ) -> Result<Self, Cea2034InputError> {
        if frequencies.len() < 2
            || frequencies
                .iter()
                .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
            || frequencies.windows(2).any(|pair| pair[1] <= pair[0])
        {
            return Err(Cea2034InputError(
                "frequencies must contain at least two finite, positive, strictly increasing samples"
                    .into(),
            ));
        }

        let validate_plane =
            |name: &str, angles: &[f64], spl: &[Vec<f64>]| -> Result<(), Cea2034InputError> {
                if angles.is_empty() || angles.len() != spl.len() {
                    return Err(Cea2034InputError(format!(
                        "{name} angle labels and SPL rows must be non-empty and have equal length"
                    )));
                }
                if angles.iter().any(|angle| !angle.is_finite()) {
                    return Err(Cea2034InputError(format!(
                        "{name} angle labels must be finite"
                    )));
                }
                for (index, angle) in angles.iter().enumerate() {
                    if angles[..index]
                        .iter()
                        .any(|previous| (previous - angle).abs() <= 1e-9)
                    {
                        return Err(Cea2034InputError(format!(
                            "{name} angle label {angle} is duplicated"
                        )));
                    }
                }
                if spl.iter().any(|row| row.len() != frequencies.len()) {
                    return Err(Cea2034InputError(format!(
                        "{name} SPL rows must match the frequency count"
                    )));
                }
                if spl.iter().flatten().any(|value| !value.is_finite()) {
                    return Err(Cea2034InputError(format!(
                        "{name} SPL rows must contain only finite values"
                    )));
                }
                Ok(())
            };

        validate_plane("horizontal", horizontal_angles, horizontal_spl)?;
        validate_plane("vertical", vertical_angles, vertical_spl)?;
        if !on_axis.degrees.is_finite() {
            return Err(Cea2034InputError(
                "on-axis angle label must be finite".into(),
            ));
        }
        let on_axis_angles = match on_axis.plane {
            SpinoramaPlane::Horizontal => horizontal_angles,
            SpinoramaPlane::Vertical => vertical_angles,
        };
        let matches: Vec<usize> = on_axis_angles
            .iter()
            .enumerate()
            .filter_map(|(row, &angle)| ((angle - on_axis.degrees).abs() <= 1e-9).then_some(row))
            .collect();
        if matches.len() != 1 {
            return Err(Cea2034InputError(format!(
                "on-axis {:?} angle {} must identify exactly one row",
                on_axis.plane, on_axis.degrees
            )));
        }

        Ok(Self {
            frequencies,
            horizontal_spl,
            vertical_spl,
            on_axis,
            on_axis_row: matches[0],
        })
    }

    fn on_axis_curve(&self) -> &[f64] {
        match self.on_axis.plane {
            SpinoramaPlane::Horizontal => &self.horizontal_spl[self.on_axis_row],
            SpinoramaPlane::Vertical => &self.vertical_spl[self.on_axis_row],
        }
    }
}

/// Compute all CEA2034 metrics and preference score without input checks.
fn score_unchecked(
    freq: &[f64],
    intervals: &[(usize, usize)],
    on: &[f64],
    lw: &[f64],
    sp: &[f64],
    pir: &[f64],
) -> ScoreMetrics {
    let nbd_on = nbd(intervals, on);
    let nbd_pir = nbd(intervals, pir);
    let sm_pir = sm(freq, pir);
    let lfx_val = lfx(freq, lw, sp);
    let pref = 12.69 - 2.49 * nbd_on - 2.99 * nbd_pir - 4.31 * lfx_val + 2.32 * sm_pir;
    ScoreMetrics {
        nbd_on,
        nbd_pir,
        lfx: lfx_val,
        sm_pir,
        pref_score: pref,
    }
}

/// Compute CEA-2034 metrics after validating the frequency grid and curves.
///
/// This is the fallible entry point for callers that receive measurements or
/// generated data at runtime. It rejects ragged/misaligned curves, malformed
/// intervals, non-finite values, and undefined metrics before any slicing is
/// attempted. The legacy [`score`] wrapper remains available for callers that
/// intentionally rely on panic-compatible behaviour.
pub fn try_score(
    freq: &[f64],
    intervals: &[(usize, usize)],
    on: &[f64],
    lw: &[f64],
    sp: &[f64],
    pir: &[f64],
) -> Result<ScoreMetrics, Cea2034InputError> {
    if freq.len() < 2
        || freq
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        || freq.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err(Cea2034InputError(
            "CEA-2034 score frequency grid must contain at least two finite, positive, strictly increasing samples"
                .into(),
        ));
    }
    let curves = [on, lw, sp, pir];
    if curves
        .iter()
        .any(|curve| curve.len() != freq.len() || curve.iter().any(|value| !value.is_finite()))
    {
        return Err(Cea2034InputError(
            "CEA-2034 score curves must be finite and match the frequency grid".into(),
        ));
    }
    if intervals.is_empty()
        || intervals
            .iter()
            .any(|&(start, end)| start >= end || end > freq.len())
    {
        return Err(Cea2034InputError(
            "CEA-2034 score intervals must be non-empty and within the frequency grid".into(),
        ));
    }

    let metrics = score_unchecked(freq, intervals, on, lw, sp, pir);
    if [
        metrics.nbd_on,
        metrics.nbd_pir,
        metrics.lfx,
        metrics.sm_pir,
        metrics.pref_score,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err(Cea2034InputError(
            "CEA-2034 score metrics are undefined for the supplied curves".into(),
        ));
    }
    Ok(metrics)
}

/// Borrowed CEA-2034 curves used for one preference-score evaluation.
///
/// This view avoids copying a candidate's frequency responses when an
/// optimizer requests provenance-aware diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct ScoreInputRef<'a> {
    /// Strictly increasing positive frequency grid.
    pub freq: &'a [f64],
    /// Octave-band intervals over [`Self::freq`].
    pub intervals: &'a [(usize, usize)],
    /// On-axis SPL response.
    pub on: &'a [f64],
    /// Listening-window SPL response.
    pub lw: &'a [f64],
    /// Sound-power SPL response.
    pub sp: &'a [f64],
    /// Predicted in-room SPL response.
    pub pir: &'a [f64],
}

/// Score CEA-2034 curves while carrying their per-frequency fidelity labels.
///
/// Synthetic bands are rejected unless the caller explicitly selects
/// [`ScoreProvenancePolicy::AllowSynthetic`]. This makes it impossible for an
/// optimization caller to treat extrapolated high-frequency data as solved or
/// measured data by omission.
pub fn try_score_with_provenance(
    input: ScoreInputRef<'_>,
    provenance: &[ScoreSampleProvenance],
    policy: ScoreProvenancePolicy,
) -> Result<ProvenancedScoreMetrics, Cea2034InputError> {
    if provenance.len() != input.freq.len() {
        return Err(Cea2034InputError(
            "CEA-2034 score provenance must match the frequency grid".into(),
        ));
    }
    let contains_synthetic = provenance.contains(&ScoreSampleProvenance::Synthetic);
    let first_non_authoritative = provenance
        .iter()
        .enumerate()
        .find(|(_, provenance)| !provenance.is_measured_or_solved());
    let contains_non_authoritative = first_non_authoritative.is_some();
    match policy {
        ScoreProvenancePolicy::RejectSynthetic if contains_synthetic => {
            return Err(Cea2034InputError(
                "CEA-2034 score contains synthetic samples; select an explicit synthetic-band policy"
                    .into(),
            ));
        }
        ScoreProvenancePolicy::RequireMeasuredOrSolved
            if let Some((index, provenance)) = first_non_authoritative =>
        {
            return Err(Cea2034InputError(format!(
                "CEA-2034 score requires measured or solved samples; frequency index {index} is {}",
                provenance.as_str()
            )));
        }
        _ => {}
    }

    let metrics = try_score(
        input.freq,
        input.intervals,
        input.on,
        input.lw,
        input.sp,
        input.pir,
    )?;
    let result = ProvenancedScoreMetrics {
        metrics,
        frequency_provenance: provenance.to_vec(),
        contains_synthetic,
        contains_non_authoritative,
    };
    result.validate()?;
    Ok(result)
}

/// Compute all CEA2034 metrics and preference score.
///
/// New data-boundary callers should prefer [`try_score`]. This wrapper keeps
/// the historical infallible API and therefore panics when the input violates
/// the score contract.
pub fn score(
    freq: &[f64],
    intervals: &[(usize, usize)],
    on: &[f64],
    lw: &[f64],
    sp: &[f64],
    pir: &[f64],
) -> ScoreMetrics {
    score_unchecked(freq, intervals, on, lw, sp, pir)
}

/// Compute CEA2034 metrics and preference score for a PEQ filter.
///
/// Returns the spinorama data (2-D array) and the score metrics.
pub fn score_peq(
    input: &ValidatedSpinorama<'_>,
    idx: &[Vec<usize>],
    intervals: &[(usize, usize)],
    weights: &[f64],
    peq: &[f64],
) -> Result<(Vec<Vec<f64>>, ScoreMetrics), Cea2034InputError> {
    let frequencies = input.frequencies;
    if frequencies.len() != peq.len()
        || peq.is_empty()
        || peq.iter().any(|value| !value.is_finite())
    {
        return Err(Cea2034InputError(
            "PEQ values must be finite and match the non-empty frequency array".into(),
        ));
    }

    let total_rows = input.horizontal_spl.len() + input.vertical_spl.len();
    if weights.len() < total_rows
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        || idx.len() < 8
        || idx.iter().flatten().any(|&row| row >= total_rows)
    {
        return Err(Cea2034InputError(
            "spinorama index groups and finite weights must cover all SPL rows".into(),
        ));
    }
    if intervals
        .iter()
        .any(|&(start, end)| start >= end || end > peq.len())
    {
        return Err(Cea2034InputError(
            "octave intervals must be ordered and within the frequency array".into(),
        ));
    }

    let spl_h_peq: Vec<Vec<f64>> = input
        .horizontal_spl
        .iter()
        .map(|row| {
            row.iter()
                .zip(peq)
                .map(|(&spl, &filter)| spl + filter)
                .collect()
        })
        .collect();
    let spl_v_peq: Vec<Vec<f64>> = input
        .vertical_spl
        .iter()
        .map(|row| {
            row.iter()
                .zip(peq)
                .map(|(&spl, &filter)| spl + filter)
                .collect()
        })
        .collect();
    let on: Vec<f64> = input
        .on_axis_curve()
        .iter()
        .zip(peq)
        .map(|(&spl, &filter)| spl + filter)
        .collect();

    let mut spl_full = spl_h_peq;
    spl_full.extend(spl_v_peq);
    let spin = try_cea2034(&spl_full, idx, weights)?;
    let lw = &spin[0];
    let sp_row = &spin[spin.len() - 2];
    let pir = &spin[spin.len() - 1];
    let metrics = try_score(frequencies, intervals, &on, lw, sp_row, pir)?;
    Ok((spin, metrics))
}

/// Borrowed curves for one approximate PEQ score evaluation.
#[derive(Debug, Clone, Copy)]
pub struct ApproxPeqScoreInputRef<'a> {
    /// Strictly increasing positive frequency grid.
    pub freq: &'a [f64],
    /// Octave-band intervals over [`Self::freq`].
    pub intervals: &'a [(usize, usize)],
    /// Listening-window SPL response.
    pub lw: &'a [f64],
    /// Sound-power SPL response.
    pub sp: &'a [f64],
    /// Predicted in-room SPL response.
    pub pir: &'a [f64],
    /// On-axis SPL response.
    pub on: &'a [f64],
    /// PEQ adjustment in dB.
    pub peq: &'a [f64],
}

/// Reusable owned work buffers for [`score_peq_approx_with_scratch`].
///
/// An optimization loop should retain one scratch value per worker. It avoids
/// reallocating four frequency-length curves for every candidate while keeping
/// all score inputs borrowed from the caller.
#[derive(Debug, Default, Clone)]
pub struct PeqScoreScratch {
    on: Vec<f64>,
    lw: Vec<f64>,
    sp: Vec<f64>,
    pir: Vec<f64>,
}

impl PeqScoreScratch {
    /// Allocate empty scratch buffers with at least `frequency_count` capacity.
    pub fn with_capacity(frequency_count: usize) -> Self {
        Self {
            on: Vec::with_capacity(frequency_count),
            lw: Vec::with_capacity(frequency_count),
            sp: Vec::with_capacity(frequency_count),
            pir: Vec::with_capacity(frequency_count),
        }
    }

    /// Smallest backing-buffer capacity, useful for allocation-regression tests.
    pub fn minimum_capacity(&self) -> usize {
        [
            self.on.capacity(),
            self.lw.capacity(),
            self.sp.capacity(),
            self.pir.capacity(),
        ]
        .into_iter()
        .min()
        .unwrap_or(0)
    }

    fn fill(&mut self, on: &[f64], lw: &[f64], sp: &[f64], pir: &[f64], peq: &[f64]) {
        self.on.clear();
        self.lw.clear();
        self.sp.clear();
        self.pir.clear();
        self.on
            .extend(on.iter().zip(peq).map(|(&value, &filter)| value + filter));
        self.lw
            .extend(lw.iter().zip(peq).map(|(&value, &filter)| value + filter));
        self.sp
            .extend(sp.iter().zip(peq).map(|(&value, &filter)| value + filter));
        self.pir
            .extend(pir.iter().zip(peq).map(|(&value, &filter)| value + filter));
    }
}

/// Compute approximate CEA2034 metrics with caller-owned reusable work buffers.
///
/// Works directly with pre-computed LW, SP, and PIR curves rather than raw
/// measurements. The scratch buffers are overwritten on every successful call.
pub fn score_peq_approx_with_scratch(
    input: ApproxPeqScoreInputRef<'_>,
    scratch: &mut PeqScoreScratch,
) -> Result<ScoreMetrics, Cea2034InputError> {
    let curves = [input.lw, input.sp, input.pir, input.on, input.peq];
    if input.freq.len() < 2
        || curves.iter().any(|curve| curve.len() != input.freq.len())
        || input
            .freq
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        || input.freq.windows(2).any(|pair| pair[1] <= pair[0])
        || curves
            .iter()
            .any(|curve| curve.iter().any(|value| !value.is_finite()))
        || input
            .intervals
            .iter()
            .any(|&(start, end)| start >= end || end > input.freq.len())
    {
        return Err(Cea2034InputError(
            "approximate CEA-2034 curves must be finite, rectangular, and aligned to an ordered frequency grid"
                .into(),
        ));
    }
    scratch.fill(input.on, input.lw, input.sp, input.pir, input.peq);
    try_score(
        input.freq,
        input.intervals,
        &scratch.on,
        &scratch.lw,
        &scratch.sp,
        &scratch.pir,
    )
}

/// Compute approximate CEA2034 metrics and preference score for a PEQ filter.
///
/// This compatible one-shot wrapper allocates ephemeral buffers. Candidate
/// loops should use [`score_peq_approx_with_scratch`] instead.
pub fn score_peq_approx(
    freq: &[f64],
    intervals: &[(usize, usize)],
    lw: &[f64],
    sp: &[f64],
    pir: &[f64],
    on: &[f64],
    peq: &[f64],
) -> Result<ScoreMetrics, Cea2034InputError> {
    let mut scratch = PeqScoreScratch::with_capacity(freq.len());
    score_peq_approx_with_scratch(
        ApproxPeqScoreInputRef {
            freq,
            intervals,
            lw,
            sp,
            pir,
            on,
            peq,
        },
        &mut scratch,
    )
}

/// Compute Predicted In-Room (PIR) response from LW, ER, and SP measurements.
pub fn compute_pir_from_lw_er_sp(lw: &[f64], er: &[f64], sp: &[f64]) -> Vec<f64> {
    assert_eq!(lw.len(), er.len(), "PIR curve length mismatch");
    assert_eq!(lw.len(), sp.len(), "PIR curve length mismatch");
    assert!(
        lw.iter().chain(er).chain(sp).all(|v| v.is_finite()),
        "PIR curves must be finite"
    );
    lw.iter()
        .zip(er.iter())
        .zip(sp.iter())
        .map(|((&lw, &er), &sp)| energy_mean_db([(lw, 0.12), (er, 0.44), (sp, 0.44)].into_iter()))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octave_count_2_includes_reference_center() {
        let bands = octave(2);
        assert!(bands.iter().any(|&(_l, c, _h)| (c - 1290.0).abs() < 1e-9));
    }

    #[test]
    fn try_octave_rejects_count_below_2() {
        assert!(try_octave(0).is_none());
        assert!(try_octave(1).is_none());
        assert_eq!(try_octave(2), Some(octave(2)));
    }

    #[test]
    fn nbd_simple_mean_of_mads() {
        let spl = vec![0.0, 1.0, 2.0, 1.0, 0.0];
        let intervals = vec![(0, 3), (2, 5)];
        let v = nbd(&intervals, &spl);
        assert!(v.is_finite());
    }

    #[test]
    fn score_peq_approx_matches_score_when_peq_zero() {
        let freq = vec![100.0, 1000.0, 10000.0];
        let intervals = vec![(0, 3)];
        let on = vec![80.0, 85.0, 82.0];
        let lw = vec![81.0, 84.0, 83.0];
        let sp = vec![79.0, 83.0, 81.0];
        let pir = vec![80.5, 84.0, 82.0];
        let zero = vec![0.0; freq.len()];

        let m1 = score(&freq, &intervals, &on, &lw, &sp, &pir);
        let m2 = score_peq_approx(&freq, &intervals, &lw, &sp, &pir, &on, &zero).unwrap();

        assert!((m1.nbd_on - m2.nbd_on).abs() < 1e-12);
        assert!((m1.nbd_pir - m2.nbd_pir).abs() < 1e-12);
        assert!((m1.lfx - m2.lfx).abs() < 1e-12);
        assert!((m1.sm_pir - m2.sm_pir).abs() < 1e-12);
        assert!((m1.pref_score - m2.pref_score).abs() < 1e-12);
    }

    #[test]
    fn checked_cea2034_rejects_malformed_groups_and_weights() {
        let valid_spl = vec![vec![80.0, 81.0]; 8];
        let valid_idx = vec![vec![0]; 8];
        let valid_weights = vec![1.0; 8];
        let result = try_cea2034(&valid_spl, &valid_idx, &valid_weights)
            .expect("valid CEA matrix should be accepted");
        assert!(result.iter().flatten().all(|value| value.is_finite()));

        let mut ragged = valid_spl.clone();
        ragged[3].pop();
        assert!(try_cea2034(&ragged, &valid_idx, &valid_weights).is_err());

        let mut empty_group = valid_idx.clone();
        empty_group[0].clear();
        assert!(try_cea2034(&valid_spl, &empty_group, &valid_weights).is_err());

        let mut out_of_range = valid_idx.clone();
        out_of_range[2] = vec![valid_spl.len()];
        assert!(try_cea2034(&valid_spl, &out_of_range, &valid_weights).is_err());

        let mut extra_group = valid_idx.clone();
        extra_group.push(vec![0]);
        assert!(try_cea2034(&valid_spl, &extra_group, &valid_weights).is_err());

        assert!(try_cea2034(&valid_spl, &valid_idx, &[0.0; 7]).is_err());
    }

    #[test]
    fn contiguous_cea2034_matches_nested_rows_without_input_repacking() {
        let nested = vec![
            vec![80.0, 81.0, 82.0],
            vec![81.0, 82.0, 83.0],
            vec![82.0, 83.0, 84.0],
            vec![83.0, 84.0, 85.0],
            vec![84.0, 85.0, 86.0],
            vec![85.0, 86.0, 87.0],
            vec![86.0, 87.0, 88.0],
            vec![87.0, 88.0, 89.0],
        ];
        let flat: Vec<f64> = nested.iter().flatten().copied().collect();
        let indices = vec![
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![4],
            vec![5],
            vec![6],
            vec![7],
        ];
        let weights = vec![1.0; nested.len()];

        let contiguous = ContiguousCea2034InputRef::new(&flat, nested.len(), 3)
            .expect("row-major source must be admitted");
        assert_eq!(contiguous.row_count(), nested.len());
        assert_eq!(contiguous.frequency_count(), 3);
        assert_eq!(contiguous.row(4), nested[4]);

        let nested_result = try_cea2034(&nested, &indices, &weights).unwrap();
        let contiguous_result = try_cea2034_contiguous(contiguous, &indices, &weights).unwrap();
        assert_eq!(contiguous_result, nested_result);
    }

    #[test]
    fn contiguous_cea2034_rejects_bad_layout_values_and_groups() {
        assert!(ContiguousCea2034InputRef::new(&[80.0; 7], 2, 4).is_err());
        assert!(ContiguousCea2034InputRef::new(&[80.0; 8], 0, 4).is_err());
        assert!(ContiguousCea2034InputRef::new(&[80.0, f64::NAN], 1, 2).is_err());

        let flat = vec![80.0; 16];
        let input = ContiguousCea2034InputRef::new(&flat, 8, 2).unwrap();
        let valid_indices = vec![vec![0]; 8];
        let valid_weights = vec![1.0; 8];
        let mut invalid_indices = valid_indices.clone();
        invalid_indices[3] = vec![8];
        assert!(try_cea2034_contiguous(input, &invalid_indices, &valid_weights).is_err());
        assert!(try_cea2034_contiguous(input, &valid_indices[..7], &valid_weights).is_err());
        let mut extra_group_indices = valid_indices;
        extra_group_indices.push(vec![0]);
        assert!(try_cea2034_contiguous(input, &extra_group_indices, &valid_weights).is_err());
    }

    #[test]
    fn lfx_next_bin_after_first_block() {
        let freq = vec![
            50.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 5000.0, 10000.0, 12000.0,
        ];
        let lw = vec![80.0; 9];
        let sp = vec![70.0, 73.0, 75.0, 76.0, 80.0, 80.0, 80.0, 80.0, 80.0];
        let val = lfx(&freq, &lw, &sp);
        assert!((val - 200.0_f64.log10()).abs() < 1e-12);
    }

    #[test]
    fn lfx_no_indices_falls_back_to_first_freq() {
        let freq = vec![
            50.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 5000.0, 10000.0, 12000.0,
        ];
        let lw = vec![80.0; 9];
        let sp = vec![75.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0];
        let val = lfx(&freq, &lw, &sp);
        assert!((val - 50.0_f64.log10()).abs() < 1e-12);
    }

    #[test]
    fn lfx_next_index_oob_defaults_to_300() {
        let freq = vec![100.0, 200.0, 300.0];
        let lw = vec![80.0, 80.0, 80.0];
        let sp = vec![70.0, 70.0, 70.0];
        let val = lfx(&freq, &lw, &sp);
        assert!((val - 300.0_f64.log10()).abs() < 1e-12);
    }

    #[test]
    fn mad_empty_slice_returns_zero_not_nan() {
        let spl = vec![1.0, 2.0, 3.0];
        let result = mad(&spl, 2, 2);
        assert_eq!(result, 0.0, "mad() on empty slice must return 0.0, not NaN");
    }

    #[test]
    fn octave_intervals_skips_empty_bands() {
        let freq = vec![15000.0, 16000.0, 17000.0];
        let intervals = octave_intervals(3, &freq);
        for &(imin, imax) in &intervals {
            assert!(
                imin < imax,
                "Empty band ({}, {}) should have been skipped",
                imin,
                imax
            );
        }
    }

    #[test]
    fn nbd_with_empty_bands_is_finite() {
        let spl = vec![80.0; 5];
        let intervals = vec![(0, 3), (3, 3), (2, 5)];
        let result = nbd(&intervals, &spl);
        assert!(
            result.is_finite(),
            "nbd must be finite even with empty bands, got {}",
            result
        );
    }

    #[test]
    fn spl_pressure_roundtrip_is_identity() {
        let spl = vec![60.0, 80.0, 100.0];
        let p = spl2pressure(&spl);
        let spl2 = pressure2spl(p);
        for (a, b) in spl.iter().zip(spl2.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn pir_equals_input_when_all_equal() {
        let lw = vec![80.0, 80.0, 80.0];
        let er = vec![80.0, 80.0, 80.0];
        let sp = vec![80.0, 80.0, 80.0];
        let pir = compute_pir_from_lw_er_sp(&lw, &er, &sp);
        for v in &pir {
            assert!((*v - 80.0).abs() < 1e-12);
        }
    }

    #[test]
    fn validated_spinorama_rejects_short_ragged_and_mislabeled_inputs() {
        let frequencies = [100.0, 1000.0, 10_000.0];
        let horizontal_angles = [0.0];
        let vertical_angles = [90.0];
        let horizontal = vec![vec![80.0, 81.0, 82.0]];
        let vertical = vec![vec![79.0, 80.0, 81.0]];
        let on_axis = SpinoramaAngle {
            plane: SpinoramaPlane::Horizontal,
            degrees: 0.0,
        };

        assert!(
            ValidatedSpinorama::new(
                &[1000.0],
                &horizontal_angles,
                &[vec![80.0]],
                &vertical_angles,
                &[vec![79.0]],
                on_axis,
            )
            .is_err()
        );
        assert!(
            ValidatedSpinorama::new(
                &frequencies,
                &horizontal_angles,
                &[vec![80.0, 81.0]],
                &vertical_angles,
                &vertical,
                on_axis,
            )
            .is_err()
        );
        assert!(
            ValidatedSpinorama::new(
                &frequencies,
                &[0.0, 0.0],
                &[horizontal[0].clone(), horizontal[0].clone()],
                &vertical_angles,
                &vertical,
                on_axis,
            )
            .is_err()
        );
        assert!(
            ValidatedSpinorama::new(
                &frequencies,
                &horizontal_angles,
                &horizontal,
                &vertical_angles,
                &vertical,
                SpinoramaAngle {
                    plane: SpinoramaPlane::Horizontal,
                    degrees: 10.0,
                },
            )
            .is_err()
        );
        assert!(
            ValidatedSpinorama::new(
                &frequencies,
                &horizontal_angles,
                &[vec![80.0, f64::NAN, 82.0]],
                &vertical_angles,
                &vertical,
                on_axis,
            )
            .is_err()
        );
    }

    #[test]
    fn score_peq_rejects_zero_variance_metrics() {
        let frequencies = [100.0, 1000.0, 10_000.0];
        let angles = [0.0];
        let horizontal = vec![vec![80.0; 3]];
        let vertical = vec![vec![80.0; 3]];
        let input = ValidatedSpinorama::new(
            &frequencies,
            &angles,
            &horizontal,
            &angles,
            &vertical,
            SpinoramaAngle {
                plane: SpinoramaPlane::Horizontal,
                degrees: 0.0,
            },
        )
        .unwrap();
        let result = score_peq(
            &input,
            &vec![vec![0]; 7],
            &[(0, frequencies.len())],
            &[1.0, 1.0],
            &[0.0; 3],
        );
        assert!(result.is_err());
    }

    #[test]
    fn validated_spinorama_selects_on_axis_by_plane_and_angle() {
        let frequencies = [100.0, 1000.0, 10_000.0];
        let horizontal_angles = [-10.0, 0.0];
        let vertical_angles = [0.0, 10.0];
        let horizontal = vec![vec![70.0; 3], vec![71.0; 3]];
        let vertical = vec![vec![80.0; 3], vec![81.0; 3]];
        let input = ValidatedSpinorama::new(
            &frequencies,
            &horizontal_angles,
            &horizontal,
            &vertical_angles,
            &vertical,
            SpinoramaAngle {
                plane: SpinoramaPlane::Vertical,
                degrees: 10.0,
            },
        )
        .unwrap();
        assert_eq!(input.on_axis_curve(), vertical[1]);
    }

    #[test]
    fn score_peq_approx_rejects_misaligned_curves() {
        let frequencies = [100.0, 1000.0, 10_000.0];
        let curve = [80.0, 81.0, 82.0];
        assert!(
            score_peq_approx(
                &frequencies,
                &[(0, 3)],
                &curve[..2],
                &curve,
                &curve,
                &curve,
                &[0.0; 3],
            )
            .is_err()
        );
    }

    #[test]
    fn provenance_aware_score_rejects_synthetic_bands_without_explicit_policy() {
        let freq = [100.0, 1000.0, 10_000.0];
        let intervals = [(0, 3)];
        let on = [80.0, 85.0, 82.0];
        let lw = [81.0, 84.0, 83.0];
        let sp = [79.0, 83.0, 81.0];
        let pir = [80.5, 84.0, 82.0];
        let provenance = [
            ScoreSampleProvenance::Solved,
            ScoreSampleProvenance::Solved,
            ScoreSampleProvenance::Synthetic,
        ];

        assert!(
            try_score_with_provenance(
                ScoreInputRef {
                    freq: &freq,
                    intervals: &intervals,
                    on: &on,
                    lw: &lw,
                    sp: &sp,
                    pir: &pir,
                },
                &provenance,
                ScoreProvenancePolicy::RejectSynthetic,
            )
            .is_err()
        );

        let disclosed = try_score_with_provenance(
            ScoreInputRef {
                freq: &freq,
                intervals: &intervals,
                on: &on,
                lw: &lw,
                sp: &sp,
                pir: &pir,
            },
            &provenance,
            ScoreProvenancePolicy::AllowSynthetic,
        )
        .unwrap();
        assert!(disclosed.contains_synthetic);
        assert!(disclosed.contains_non_authoritative);
        assert_eq!(disclosed.frequency_provenance, provenance);
        assert!(disclosed.metrics.pref_score.is_finite());
    }

    #[test]
    fn strict_provenance_policy_rejects_every_approximate_band() {
        let freq = [100.0, 1000.0, 10_000.0];
        let intervals = [(0, 3)];
        let on = [80.0, 85.0, 82.0];
        let lw = [81.0, 84.0, 83.0];
        let sp = [79.0, 83.0, 81.0];
        let pir = [80.5, 84.0, 82.0];
        let input = ScoreInputRef {
            freq: &freq,
            intervals: &intervals,
            on: &on,
            lw: &lw,
            sp: &sp,
            pir: &pir,
        };

        for provenance in [
            ScoreSampleProvenance::ReducedOrder,
            ScoreSampleProvenance::Interpolated,
            ScoreSampleProvenance::Geometric,
            ScoreSampleProvenance::Synthetic,
        ] {
            let labels = [
                ScoreSampleProvenance::Measured,
                provenance,
                ScoreSampleProvenance::Solved,
            ];
            let error = try_score_with_provenance(
                input,
                &labels,
                ScoreProvenancePolicy::RequireMeasuredOrSolved,
            )
            .unwrap_err();
            assert!(error.0.contains(provenance.as_str()));
        }

        let authoritative = [
            ScoreSampleProvenance::Measured,
            ScoreSampleProvenance::Solved,
            ScoreSampleProvenance::Measured,
        ];
        let result = try_score_with_provenance(
            input,
            &authoritative,
            ScoreProvenancePolicy::RequireMeasuredOrSolved,
        )
        .unwrap();
        assert!(!result.contains_synthetic);
        assert!(!result.contains_non_authoritative);
        assert!(result.metrics.pref_score.is_finite());
    }

    #[test]
    fn provenance_score_artifacts_round_trip_and_reject_stale_disclosures() {
        let freq = [100.0, 1_000.0, 10_000.0];
        let intervals = [(0, 3)];
        let input = ScoreInputRef {
            freq: &freq,
            intervals: &intervals,
            on: &[80.0, 85.0, 82.0],
            lw: &[81.0, 84.0, 83.0],
            sp: &[79.0, 83.0, 81.0],
            pir: &[80.5, 84.0, 82.0],
        };
        let scored = try_score_with_provenance(
            input,
            &[
                ScoreSampleProvenance::Measured,
                ScoreSampleProvenance::ReducedOrder,
                ScoreSampleProvenance::Solved,
            ],
            ScoreProvenancePolicy::AllowSynthetic,
        )
        .unwrap();
        let value = serde_json::to_value(&scored).unwrap();
        let restored: ProvenancedScoreMetrics = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(restored, scored);

        let mut stale_flag = value.clone();
        stale_flag["contains_non_authoritative"] = serde_json::Value::Bool(false);
        assert!(serde_json::from_value::<ProvenancedScoreMetrics>(stale_flag).is_err());

        let mut unknown_label = value;
        unknown_label["frequency_provenance"][1] = serde_json::Value::String("guessed".into());
        assert!(serde_json::from_value::<ProvenancedScoreMetrics>(unknown_label).is_err());
    }

    #[test]
    fn provenance_aware_score_rejects_misaligned_labels_and_preserves_physical_labels() {
        let freq = [100.0, 1000.0, 10_000.0];
        let intervals = [(0, 3)];
        let curve = [80.0, 85.0, 82.0];
        assert!(
            try_score_with_provenance(
                ScoreInputRef {
                    freq: &freq,
                    intervals: &intervals,
                    on: &curve,
                    lw: &curve,
                    sp: &curve,
                    pir: &curve,
                },
                &[ScoreSampleProvenance::Measured],
                ScoreProvenancePolicy::RejectSynthetic,
            )
            .is_err()
        );

        let provenance = [
            ScoreSampleProvenance::Measured,
            ScoreSampleProvenance::ReducedOrder,
            ScoreSampleProvenance::Geometric,
        ];
        let scored = try_score_with_provenance(
            ScoreInputRef {
                freq: &freq,
                intervals: &intervals,
                on: &curve,
                lw: &[81.0, 84.0, 83.0],
                sp: &[79.0, 83.0, 81.0],
                pir: &[80.5, 84.0, 82.0],
            },
            &provenance,
            ScoreProvenancePolicy::RejectSynthetic,
        )
        .unwrap();
        assert!(!scored.contains_synthetic);
        assert_eq!(scored.frequency_provenance, provenance);
    }
}
