//! Chroma feature extraction.
//!
//! Ported from bliss-audio chroma.rs — already pure Rust, no aubio dependency.
//! Computes 13 interval/triad features from the chromagram.

use super::utils::{hz_to_octs_inplace, normalize, stft};
use crate::dsp::stft::generate_hann_window;
use num_complex::Complex;
use sonium_fft::{RealFftPlan, RealFftPlanner};
use sonium_solvers_core::dense::DenseMatrix;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Cache key for chroma filters: `(sample_rate, n_fft, n_chroma, quantized
/// tuning)` — same keying as `ChromaFeatureExtractor`'s per-instance cache.
type ChromaFilterKey = (u32, usize, u32, i64);

/// Process-wide cache of transposed single-precision chroma filters for the
/// standalone `compute_chroma_features` path.
static CHROMA_FILTER_CACHE: OnceLock<Mutex<HashMap<ChromaFilterKey, Arc<DenseMatrix<f32>>>>> =
    OnceLock::new();

/// Linearly spaced grid from `start` to `end` (inclusive).
fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + step * i as f64).collect()
}

/// Euclidean norm of a slice.
fn l2_norm(xs: &[f64]) -> f64 {
    xs.iter().map(|x| x * x).sum::<f64>().sqrt()
}

const WINDOW_SIZE: usize = 8192;
const MAX_VALUE: f32 = 1.0;
const MIN_VALUE: f32 = 0.0;
const MAX_L2_INTERVAL: f32 = 0.25;
const MAX_L2_TRIAD: f32 = 0.025;
const MAX_TRIAD_INTERVAL_RATIO: f32 = std::f32::consts::FRAC_PI_2;

/// Error type for chroma analysis.
#[derive(Debug, Clone)]
pub struct ChromaError(pub String);

impl std::fmt::Display for ChromaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "chroma error: {}", self.0)
    }
}

impl std::error::Error for ChromaError {}

/// Reusable chroma feature extractor.
///
/// Caches the Hann window, FFT plan, STFT scratch buffers, and the chroma
/// filter matrix so repeated analysis on the same window size does not
/// re-allocate or re-compute them on every call.
///
/// The cached filter is stored in transposed form `(n_bins, n_chroma)` as
/// `f32` so the bin-major matmul loop reads half the data and accumulates in
/// the faster single-precision path.
pub struct ChromaFeatureExtractor {
    window_size: usize,
    hop_length: usize,
    hann_window: Vec<f32>,
    fft: Arc<RealFftPlan<f32>>,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    spectrum_buf: Option<DenseMatrix<f64>>,
    padded_buf: Vec<f32>,
    filter_cache: HashMap<(u32, usize, u32, i64), DenseMatrix<f32>>,
    chroma_buf: Vec<f32>,
}

impl ChromaFeatureExtractor {
    /// Create a new extractor with the default chroma window size (8192 samples).
    pub fn new() -> Self {
        Self::with_window_size(WINDOW_SIZE, 2205)
    }

    /// Create a new extractor with a custom window and hop size.
    pub fn with_window_size(window_size: usize, hop_length: usize) -> Self {
        let planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(window_size);
        let hann_window = generate_hann_window(window_size);
        let n_bins = window_size / 2 + 1;
        let fft_scratch = vec![Complex::new(0.0, 0.0); fft.scratch_len()];
        Self {
            window_size,
            hop_length,
            hann_window,
            fft,
            fft_input: vec![0.0; window_size],
            fft_output: vec![Complex::new(0.0, 0.0); n_bins],
            fft_scratch,
            spectrum_buf: None,
            padded_buf: Vec::new(),
            filter_cache: HashMap::new(),
            chroma_buf: Vec::new(),
        }
    }

    /// Reset internal state (currently a no-op; buffers are overwritten each analysis).
    pub fn reset(&mut self) {}

    /// Compute the STFT magnitude spectrum for the configured window/hop,
    /// reusing internal buffers.
    fn compute_stft(&mut self, signal: &[f32]) -> DenseMatrix<f64> {
        let window_length = self.window_size;
        let hop_length = self.hop_length;
        let n_frames = (signal.len() as f32 / hop_length as f32).ceil() as usize;
        let n_bins = window_length / 2 + 1;

        let mut spectrum = match self.spectrum_buf.take() {
            Some(mut s) => {
                if s.shape() == (n_bins, n_frames) {
                    s.fill(0.0);
                    s
                } else {
                    DenseMatrix::zeros(n_bins, n_frames)
                }
            }
            None => DenseMatrix::zeros(n_bins, n_frames),
        };

        // Reuse the reflect-pad buffer instead of allocating each call.
        let pad = window_length / 2;
        let padded_len = signal.len() + 2 * pad;
        self.padded_buf.resize(padded_len, 0.0);
        for i in 0..pad {
            self.padded_buf[i] = signal[pad - i];
        }
        self.padded_buf[pad..pad + signal.len()].copy_from_slice(signal);
        let suffix_start = pad + signal.len();
        for i in 0..pad {
            self.padded_buf[suffix_start + i] = signal[signal.len() - 2 - i];
        }

        for (frame, window) in self
            .padded_buf
            .windows(window_length)
            .step_by(hop_length)
            .enumerate()
            .take(n_frames)
        {
            for (i, &w) in window.iter().enumerate() {
                self.fft_input[i] = w * self.hann_window[i];
            }
            self.fft.forward_with_scratch(
                &self.fft_input,
                &mut self.fft_output,
                &mut self.fft_scratch,
            );
            for i in 0..n_bins {
                let x = self.fft_output[i];
                // Store squared magnitude; chroma_stft_with_filter expects it,
                // and pip_track works on squared values with an adjusted threshold.
                spectrum[(i, frame)] = (x.re * x.re + x.im * x.im) as f64;
            }
        }

        spectrum
    }

    /// Compute 13 chroma interval features from the full song samples.
    ///
    /// Returns a Vec of 13 normalized features (6 interval classes + 4 triads + 2 L2 norms + 1 ratio).
    pub fn compute(&mut self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>, ChromaError> {
        let n_chroma = 12u32;

        // Reflect-padding in compute_stft indexes samples[pad - i] and
        // samples[len - 2 - i] with pad = window_size/2; too-short input
        // would panic.
        if samples.len() <= self.window_size / 2 {
            return Err(ChromaError(format!(
                "signal too short for chroma analysis: {} samples, need more than {}",
                samples.len(),
                self.window_size / 2
            )));
        }

        let spectrum = self.compute_stft(samples);
        let tuning = estimate_tuning(sample_rate, &spectrum, self.window_size, 0.01, n_chroma)?;
        let cache_key = (
            sample_rate,
            self.window_size,
            n_chroma,
            quantize_tuning(tuning),
        );

        let filter = match self.filter_cache.get(&cache_key) {
            Some(f) => f,
            None => {
                let f = chroma_filter(sample_rate, self.window_size, n_chroma, tuning)?;
                // Store transposed in row-major layout so the flat-offset
                // indexing of the f32 matmul fast path reads whole rows.
                let f_t = DenseMatrix::from_fn(f.ncols(), f.nrows(), |i, j| f[(j, i)] as f32);
                self.filter_cache.insert(cache_key, f_t);
                self.filter_cache.get(&cache_key).unwrap()
            }
        };

        let n_frames = spectrum.ncols();
        self.chroma_buf.resize(n_chroma as usize * n_frames, 0.0);
        let chroma = chroma_stft_with_filter(filter, &spectrum, &mut self.chroma_buf)?;
        self.spectrum_buf = Some(spectrum);

        let mut raw_features = chroma_interval_features(chroma)?;

        let (interval_class, interval_class_mode) = raw_features.split_at_mut(6);

        let l2_norm_interval_class = l2_norm(interval_class);
        let l2_norm_interval_class_mode = l2_norm(interval_class_mode);

        if l2_norm_interval_class > 0. {
            for x in interval_class.iter_mut() {
                *x /= l2_norm_interval_class;
            }
        }
        if l2_norm_interval_class_mode > 0. {
            for x in interval_class_mode.iter_mut() {
                *x /= l2_norm_interval_class_mode;
            }
        }

        let mut features: Vec<f32> = raw_features
            .iter()
            .map(|&x| normalize(x as f32, MIN_VALUE, MAX_VALUE))
            .collect();

        let normalized_l2_norm_interval_class =
            (2. * (l2_norm_interval_class as f32 - 0.) / (MAX_L2_INTERVAL - 0.) - 1.).min(1.);
        features.push(normalized_l2_norm_interval_class);

        let normalized_l2_norm_interval_class_mode =
            (2. * (l2_norm_interval_class_mode as f32 - 0.) / (MAX_L2_TRIAD - 0.) - 1.).min(1.);
        features.push(normalized_l2_norm_interval_class_mode);

        let angle = (20. * l2_norm_interval_class_mode).atan2(l2_norm_interval_class + 1e-12_f64);
        let normalized_ratio = 2. * (angle as f32 - 0.) / (MAX_TRIAD_INTERVAL_RATIO - 0.) - 1.;
        features.push(normalized_ratio);

        Ok(features)
    }
}

impl Default for ChromaFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute chroma STFT using a pre-computed, transposed, single-precision
/// filter matrix.
///
/// `filter` must have shape `(n_bins, n_chroma)` (the transpose of the
/// canonical `(n_chroma, n_bins)` filter). `spectrum` must contain squared
/// magnitudes (as produced by `ChromaFeatureExtractor::compute_stft`).
/// `chroma_buf` is a reusable scratch buffer large enough to hold
/// `n_chroma * n_frames` `f32` values.
fn chroma_stft_with_filter(
    filter: &DenseMatrix<f32>,
    spectrum: &DenseMatrix<f64>,
    chroma_buf: &mut [f32],
) -> Result<DenseMatrix<f64>, ChromaError> {
    chroma_matmul(filter, spectrum, chroma_buf);

    let n_chroma = filter.ncols();
    let n_frames = spectrum.ncols();

    // Normalize columns while still in f32, then convert to f64 for the
    // downstream pipeline. This keeps the normalization pass in the faster
    // precision and avoids allocating a second copy of the chromagram.
    let mut sums = vec![0.0_f32; n_frames];
    for r in 0..n_chroma {
        let row = &chroma_buf[r * n_frames..(r + 1) * n_frames];
        for j in 0..n_frames {
            sums[j] += row[j].abs();
        }
    }
    for sum in &mut sums {
        if *sum < f32::MIN_POSITIVE {
            *sum = 1.0;
        }
    }

    let mut raw_chroma = DenseMatrix::zeros(n_chroma, n_frames);
    // Fuse column-normalization and f32->f64 conversion into one pass.
    let out = raw_chroma.as_mut_slice();
    for r in 0..n_chroma {
        let row = &chroma_buf[r * n_frames..(r + 1) * n_frames];
        let out_row = &mut out[r * n_frames..(r + 1) * n_frames];
        for j in 0..n_frames {
            out_row[j] = f64::from(row[j] / sums[j]);
        }
    }

    Ok(raw_chroma)
}

/// Cache-friendly dense multiply for the small chroma filter shape.
///
/// `filter` is passed in already-transposed form with shape `(n_bins, n_chroma)`
/// and contiguous rows. `spectrum` has shape `(n_bins, n_frames)`. The result
/// is written into `chroma_buf` as `n_chroma * n_frames` `f32` values. The
/// bin-major loop reads both operands contiguously and accumulates into a small
/// result that stays in L1 cache.
fn chroma_matmul(filter: &DenseMatrix<f32>, spectrum: &DenseMatrix<f64>, chroma_buf: &mut [f32]) {
    let n_bins = filter.nrows();
    let n_chroma = filter.ncols();
    let n_frames = spectrum.ncols();
    let out_len = n_chroma * n_frames;

    chroma_buf[..out_len].fill(0.0);

    // Flat slices: `DenseMatrix` is always row-major contiguous, so the
    // flat-offset indexing below matches logical rows. Process four
    // chroma rows per inner pass so the spectrum row (the largest operand) is
    // read once per quadruple instead of once per row.
    let f = filter.as_slice();
    let s = spectrum.as_slice();
    for i in 0..n_bins {
        let f_row = &f[i * n_chroma..(i + 1) * n_chroma];
        let s_row = &s[i * n_frames..(i + 1) * n_frames];
        let n_quads = n_chroma / 4 * 4;
        for r in (0..n_quads).step_by(4) {
            let fc0 = f_row[r];
            let fc1 = f_row[r + 1];
            let fc2 = f_row[r + 2];
            let fc3 = f_row[r + 3];
            if fc0 == 0.0 && fc1 == 0.0 && fc2 == 0.0 && fc3 == 0.0 {
                continue;
            }
            let (chunk, _) = chroma_buf.split_at_mut((r + 4) * n_frames);
            let (a, b) = chunk.split_at_mut((r + 2) * n_frames);
            let (out0_part, out1) = a.split_at_mut((r + 1) * n_frames);
            let out0 = &mut out0_part[r * n_frames..];
            let out1 = &mut out1[..n_frames];
            let (out2, out3) = b.split_at_mut(n_frames);
            for j in 0..n_frames {
                let v = s_row[j] as f32;
                out0[j] += fc0 * v;
                out1[j] += fc1 * v;
                out2[j] += fc2 * v;
                out3[j] += fc3 * v;
            }
        }
        // Remainder rows when n_chroma is not a multiple of 4.
        for r in n_quads..n_chroma {
            let fc = f_row[r];
            if fc == 0.0 {
                continue;
            }
            let out = &mut chroma_buf[r * n_frames..(r + 1) * n_frames];
            for j in 0..n_frames {
                out[j] += fc * s_row[j] as f32;
            }
        }
    }
}

/// Compute 13 chroma interval features from the full song samples.
///
/// Returns a Vec of 13 normalized features (6 interval classes + 4 triads + 2 L2 norms + 1 ratio).
pub fn compute_chroma_features(samples: &[f32], sample_rate: u32) -> Result<Vec<f32>, ChromaError> {
    let n_chroma = 12u32;

    // utils::stft reflect-pads by WINDOW_SIZE/2, which needs
    // samples.len() > WINDOW_SIZE/2; too-short input would panic.
    if samples.len() <= WINDOW_SIZE / 2 {
        return Err(ChromaError(format!(
            "signal too short for chroma analysis: {} samples, need more than {}",
            samples.len(),
            WINDOW_SIZE / 2
        )));
    }

    let mut spectrum = stft(samples, WINDOW_SIZE, 2205);
    // Square the linear magnitudes from utils::stft up front: pip_track's
    // threshold and `chroma_stft_with_filter` both operate on squared
    // magnitudes, matching `ChromaFeatureExtractor::compute`.
    for x in spectrum.as_mut_slice() {
        *x *= *x;
    }
    let tuning = estimate_tuning(sample_rate, &spectrum, WINDOW_SIZE, 0.01, 12)?;
    let chroma = chroma_stft(sample_rate, &mut spectrum, WINDOW_SIZE, n_chroma, tuning)?;

    let mut raw_features = chroma_interval_features(chroma)?;

    let (interval_class, interval_class_mode) = raw_features.split_at_mut(6);

    let l2_norm_interval_class = l2_norm(interval_class);
    let l2_norm_interval_class_mode = l2_norm(interval_class_mode);

    if l2_norm_interval_class > 0. {
        for x in interval_class.iter_mut() {
            *x /= l2_norm_interval_class;
        }
    }
    if l2_norm_interval_class_mode > 0. {
        for x in interval_class_mode.iter_mut() {
            *x /= l2_norm_interval_class_mode;
        }
    }

    let mut features: Vec<f32> = raw_features
        .iter()
        .map(|&x| normalize(x as f32, MIN_VALUE, MAX_VALUE))
        .collect();

    let normalized_l2_norm_interval_class =
        (2. * (l2_norm_interval_class as f32 - 0.) / (MAX_L2_INTERVAL - 0.) - 1.).min(1.);
    features.push(normalized_l2_norm_interval_class);

    let normalized_l2_norm_interval_class_mode =
        (2. * (l2_norm_interval_class_mode as f32 - 0.) / (MAX_L2_TRIAD - 0.) - 1.).min(1.);
    features.push(normalized_l2_norm_interval_class_mode);

    let angle = (20. * l2_norm_interval_class_mode).atan2(l2_norm_interval_class + 1e-12_f64);
    let normalized_ratio = 2. * (angle as f32 - 0.) / (MAX_TRIAD_INTERVAL_RATIO - 0.) - 1.;
    features.push(normalized_ratio);

    Ok(features)
}

fn chroma_interval_features(mut chroma: DenseMatrix<f64>) -> Result<Vec<f64>, ChromaError> {
    // Apply the exponential transform and normalize each frame in-place.
    for x in chroma.as_mut_slice() {
        *x = (*x * 15.).exp();
    }
    let (n_rows, n_frames) = chroma.shape();
    for j in 0..n_frames {
        let mut sum: f64 = (0..n_rows).map(|i| chroma[(i, j)].abs()).sum();
        if sum < 0.0001 {
            sum = 1.;
        }
        for i in 0..n_rows {
            chroma[(i, j)] /= sum;
        }
    }

    let interval_feature_matrix = extract_interval_features(&chroma);
    if n_frames == 0 {
        return Err(ChromaError(
            "Tried to run chroma on empty array. Need at least one sample.".to_string(),
        ));
    }
    // Mean over frames (rows of the template/frame matrix).
    let mut means = vec![0.0; interval_feature_matrix.nrows()];
    for (r, mean) in means.iter_mut().enumerate() {
        let mut sum = 0.0;
        for j in 0..n_frames {
            sum += interval_feature_matrix[(r, j)];
        }
        *mean = sum / n_frames as f64;
    }
    Ok(means)
}

/// 12-bit masks for the 10 interval templates (columns of the template matrix).
const TEMPLATE_MASKS: [u16; 10] = [
    0x003, // minor 2nd {0,1}
    0x005, // major 2nd {0,2}
    0x009, // minor 3rd {0,3}
    0x011, // major 3rd {0,4}
    0x021, // perfect 4th {0,5}
    0x041, // tritone {0,6}
    0x091, // major triad {0,4,7}
    0x089, // minor triad {0,3,7}
    0x049, // diminished triad {0,3,6}
    0x111, // augmented triad {0,4,8}
];

fn extract_interval_features(chroma: &DenseMatrix<f64>) -> DenseMatrix<f64> {
    let n_frames = chroma.ncols();
    let mut result = DenseMatrix::zeros(TEMPLATE_MASKS.len(), n_frames);
    for (t_idx, &mask) in TEMPLATE_MASKS.iter().enumerate() {
        for shift in 0..12 {
            let rotated = ((mask << shift) | (mask >> (12 - shift))) & 0xFFF;
            for frame in 0..n_frames {
                let mut prod = 1.0_f64;
                let mut bits = rotated;
                while bits != 0 {
                    let bin = bits.trailing_zeros() as usize;
                    prod *= chroma[(bin, frame)];
                    bits &= bits - 1;
                }
                result[(t_idx, frame)] += prod;
            }
        }
    }
    result
}

#[cfg(test)]
fn normalize_feature_sequence(feature: &DenseMatrix<f64>) -> DenseMatrix<f64> {
    let mut normalized_sequence = feature.clone();
    let (n_rows, n_frames) = normalized_sequence.shape();
    for j in 0..n_frames {
        let mut sum: f64 = (0..n_rows).map(|i| normalized_sequence[(i, j)].abs()).sum();
        if sum < 0.0001 {
            sum = 1.;
        }
        for i in 0..n_rows {
            normalized_sequence[(i, j)] /= sum;
        }
    }
    normalized_sequence
}

fn chroma_filter(
    sample_rate: u32,
    n_fft: usize,
    n_chroma: u32,
    tuning: f64,
) -> Result<DenseMatrix<f64>, ChromaError> {
    let ctroct = 5.0;
    let octwidth = 2.;
    let n_chroma_float = f64::from(n_chroma);
    let n_chroma2 = (n_chroma_float / 2.0).round() as u32;
    let n_chroma2_float = f64::from(n_chroma2);

    let frequencies = linspace(0., f64::from(sample_rate), n_fft + 1);

    let mut freq_bins = frequencies;
    hz_to_octs_inplace(&mut freq_bins, tuning, n_chroma);
    for x in freq_bins.iter_mut() {
        *x *= n_chroma_float;
    }
    freq_bins[0] = freq_bins[1] - 1.5 * n_chroma_float;

    let n_freq = freq_bins.len();
    let n_rows = n_chroma as usize;
    let mut binwidth_bins = vec![1.0_f64; n_freq];
    for i in 0..n_freq - 1 {
        let width = freq_bins[i + 1] - freq_bins[i];
        binwidth_bins[i] = if width <= 1. { 1. } else { width };
    }

    let mut d = DenseMatrix::zeros(n_rows, n_freq);
    for idx in 0..n_rows {
        for j in 0..n_freq {
            d[(idx, j)] = idx as f64;
        }
    }
    // Broadcast: subtract each row index from the shared frequency grid.
    for i in 0..n_rows {
        for j in 0..n_freq {
            d[(i, j)] = freq_bins[j] - d[(i, j)];
        }
    }

    for x in d.as_mut_slice() {
        *x = (*x + n_chroma2_float + 10. * n_chroma_float) % n_chroma_float - n_chroma2_float;
    }
    for i in 0..n_rows {
        for j in 0..n_freq {
            d[(i, j)] /= binwidth_bins[j];
        }
    }
    for x in d.as_mut_slice() {
        *x = (-0.5 * (2. * *x) * (2. * *x)).exp();
    }

    let mut wts = d;
    for j in 0..n_freq {
        let mut sum: f64 = (0..n_rows)
            .map(|i| wts[(i, j)] * wts[(i, j)])
            .sum::<f64>()
            .sqrt();
        if sum < f64::MIN_POSITIVE {
            sum = 1.;
        }
        for i in 0..n_rows {
            wts[(i, j)] /= sum;
        }
    }

    for x in freq_bins.iter_mut() {
        *x = (-0.5 * ((*x / n_chroma_float - ctroct) / octwidth).powi(2)).exp();
    }
    for i in 0..n_rows {
        for j in 0..n_freq {
            wts[(i, j)] *= freq_bins[j];
        }
    }

    // np.roll by -3 along the chroma axis.
    let mut b = DenseMatrix::zeros(n_rows, n_freq);
    for j in 0..n_freq {
        for i in 0..3 {
            b[(n_rows - 3 + i, j)] = wts[(i, j)];
        }
        for i in 0..n_rows - 3 {
            b[(i, j)] = wts[(i + 3, j)];
        }
    }

    wts = b;
    let non_aliased = 1 + n_fft / 2;
    Ok(DenseMatrix::from_fn(n_rows, non_aliased, |i, j| {
        wts[(i, j)]
    }))
}

fn pip_track(
    sample_rate: u32,
    spectrum: &DenseMatrix<f64>,
    n_fft: usize,
) -> Result<(Vec<f64>, Vec<f64>), ChromaError> {
    let sample_rate_float = f64::from(sample_rate);
    let fmin = 150.0_f64;
    let fmax = 4000.0_f64.min(sample_rate_float / 2.0);
    // Spectrum contains squared magnitudes; keep the same effective threshold
    // in the squared domain: (0.1 * max_magnitude)^2 = 0.01 * max_sq.
    let threshold = 0.01;

    let fft_freqs = linspace(0., sample_rate_float / 2., 1 + n_fft / 2);

    let n_bins = spectrum.nrows();
    let n_frames = spectrum.ncols();

    let freq_mask: Vec<bool> = fft_freqs
        .iter()
        .map(|&f| (fmin <= f) && (f < fmax))
        .collect();

    // Per-column peak (over bins) scaled by the threshold.
    let mut ref_value = vec![0.0; n_frames];
    for j in 0..n_frames {
        let mut peak = spectrum[(0, j)];
        for i in 1..n_bins {
            if spectrum[(i, j)] > peak {
                peak = spectrum[(i, j)];
            }
        }
        ref_value[j] = peak * threshold;
    }

    let taken_columns = freq_mask.iter().filter(|&&x| x).count();
    let mut pitches = Vec::with_capacity(taken_columns * n_bins);
    let mut mags = Vec::with_capacity(taken_columns * n_bins);

    // Issue #5: guard against empty freq mask (e.g. very low sample rates
    // where the first FFT bin is already above fmax).
    let beginning = match freq_mask.iter().position(|&b| b) {
        Some(b) => b,
        None => return Ok((pitches, mags)),
    };
    let end = match freq_mask.iter().rposition(|&b| b) {
        Some(e) => e,
        None => return Ok((pitches, mags)),
    };

    // Guard against tiny freq masks: the three shifted rows below need at
    // least 4 in-range bins, otherwise `end - 3` underflows (debug panic) or
    // the range inverts (empty after clamping). No peaks to find either way.
    if end < beginning + 4 {
        return Ok((pitches, mags));
    }

    // Parabolic peak picking over (before, center, after) bin triples.
    // Row-major order matches the previous zipped traversal.
    for i in 0..end - 3 - beginning {
        for j in 0..n_frames {
            let before_elem = spectrum[(beginning + i, j)];
            let elem = spectrum[(beginning + 1 + i, j)];
            let after_elem = spectrum[(beginning + 2 + i, j)];
            if elem > ref_value[j] && after_elem <= elem && before_elem < elem {
                let avg = 0.5 * (after_elem - before_elem);
                let mut shift = 2. * elem - after_elem - before_elem;
                if shift.abs() < f64::MIN_POSITIVE {
                    shift += 1.;
                }
                shift = avg / shift;
                pitches
                    .push(((i + beginning + 1) as f64 + shift) * sample_rate_float / n_fft as f64);
                // Parabolically interpolated peak value, kept in the
                // squared-magnitude domain of the spectrum (the downstream median
                // filter in estimate_tuning is scale-invariant).
                mags.push(elem + 0.5 * avg * shift);
            }
        }
    }

    Ok((pitches, mags))
}

fn pitch_tuning(
    frequencies: &mut [f64],
    resolution: f64,
    bins_per_octave: u32,
) -> Result<f64, ChromaError> {
    if frequencies.is_empty() {
        return Ok(0.0);
    }
    hz_to_octs_inplace(frequencies, 0.0, 12);
    for x in frequencies.iter_mut() {
        *x = f64::from(bins_per_octave) * *x % 1.0;
    }
    for x in frequencies.iter_mut() {
        *x = if *x >= 0.5 { *x - 1. } else { *x };
    }

    let indexes: Vec<usize> = frequencies
        .iter()
        .map(|&x| ((x + 0.5) / resolution) as usize)
        .collect();
    let mut counts: Vec<usize> = vec![0; ((0.5 - -0.5) / resolution) as usize];
    for &idx in indexes.iter() {
        if idx < counts.len() {
            counts[idx] += 1;
        }
    }
    let max_index = counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, v)| *v)
        .map(|(i, _)| i)
        .ok_or_else(|| ChromaError("empty counts in pitch_tuning".to_string()))?;

    Ok((-50. + (100. * resolution * max_index as f64)) / 100.)
}

fn estimate_tuning(
    sample_rate: u32,
    spectrum: &DenseMatrix<f64>,
    n_fft: usize,
    resolution: f64,
    bins_per_octave: u32,
) -> Result<f64, ChromaError> {
    let (pitch, mag) = pip_track(sample_rate, spectrum, n_fft)?;

    let (filtered_pitch, filtered_mag): (Vec<f64>, Vec<f64>) = pitch
        .iter()
        .zip(&mag)
        .filter(|&(&p, _)| p > 0.)
        .map(|(x, y)| (*x, *y))
        .unzip();

    if filtered_pitch.is_empty() {
        return Ok(0.);
    }

    // Compute median of magnitudes
    let mut sorted_mags = filtered_mag.clone();
    sorted_mags.sort_by(|a, b| a.total_cmp(b));
    let threshold = if sorted_mags.len() % 2 == 0 {
        (sorted_mags[sorted_mags.len() / 2 - 1] + sorted_mags[sorted_mags.len() / 2]) / 2.0
    } else {
        sorted_mags[sorted_mags.len() / 2]
    };

    let mut pitch_arr: Vec<f64> = filtered_pitch
        .iter()
        .zip(&filtered_mag)
        .filter_map(|(&p, &m)| if m >= threshold { Some(p) } else { None })
        .collect();

    pitch_tuning(&mut pitch_arr, resolution, bins_per_octave)
}

fn chroma_stft(
    sample_rate: u32,
    spectrum: &mut DenseMatrix<f64>,
    n_fft: usize,
    n_chroma: u32,
    tuning: f64,
) -> Result<DenseMatrix<f64>, ChromaError> {
    // Standalone path: `spectrum` already holds squared magnitudes (squared
    // by `compute_chroma_features` before tuning estimation).
    //
    // The filter depends only on (sample_rate, n_fft, n_chroma, tuning);
    // cache it process-wide (keyed by quantized tuning, same as the
    // extractor's per-instance cache) instead of rebuilding it per call.
    let key: ChromaFilterKey = (sample_rate, n_fft, n_chroma, quantize_tuning(tuning));
    let cache = CHROMA_FILTER_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let filter_t = {
        let mut guard = cache.lock().expect("chroma filter cache poisoned");
        match guard.get(&key) {
            Some(f) => Arc::clone(f),
            None => {
                let filter = chroma_filter(sample_rate, n_fft, n_chroma, tuning)?;
                // Transpose to a row-major, contiguous
                // `(n_bins, n_chroma)` buffer in single precision: the
                // matmul fast path indexes rows by flat offset.
                let f_t = DenseMatrix::from_fn(filter.ncols(), filter.nrows(), |i, j| {
                    filter[(j, i)] as f32
                });
                let f_t = Arc::new(f_t);
                guard.insert(key, Arc::clone(&f_t));
                f_t
            }
        }
    };
    let n_frames = spectrum.ncols();
    let mut chroma_buf = vec![0.0_f32; n_chroma as usize * n_frames];
    chroma_stft_with_filter(&filter_t, spectrum, &mut chroma_buf)
}

/// Tuning is in the range [-0.5, 0.5) bins; quantize to 1e-4 bins so the
/// filter caches key on an integer.
fn quantize_tuning(tuning: f64) -> i64 {
    (tuning * 10_000.0).round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chroma_matmul_non_multiple_of_4_rows() {
        // n_chroma = 6 exercises the remainder path of the fast loop
        // (previously the quad loop would index past the row end).
        let n_bins = 3;
        let n_chroma = 6;
        let n_frames = 5;
        let filter = DenseMatrix::from_fn(n_bins, n_chroma, |i, r| (i * n_chroma + r + 1) as f32);
        let spectrum = DenseMatrix::from_fn(n_bins, n_frames, |i, j| (i * n_frames + j + 1) as f64);
        let mut buf = vec![0.0f32; n_chroma * n_frames];
        chroma_matmul(&filter, &spectrum, &mut buf);
        // Reference: straightforward per-row accumulation (same order).
        for r in 0..n_chroma {
            for j in 0..n_frames {
                let expected: f32 = (0..n_bins)
                    .map(|i| filter[[i, r]] * spectrum[[i, j]] as f32)
                    .sum();
                assert!(
                    (buf[r * n_frames + j] - expected).abs() < 1e-6,
                    "mismatch at ({r}, {j}): got {}, expected {expected}",
                    buf[r * n_frames + j]
                );
            }
        }
    }

    #[test]
    fn test_chroma_features_length() {
        // Generate a simple tone
        let sr = 22050u32;
        let duration = 5.0;
        let n = (sr as f32 * duration) as usize;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();

        let features = compute_chroma_features(&signal, sr).unwrap();
        assert_eq!(features.len(), 13);
    }

    #[test]
    fn test_normalize_feature_sequence_basic() {
        let array =
            DenseMatrix::from_row_major_vec(2, 4, vec![0.1, 0.3, 0.4, 0., 1.1, 0.53, 1.01, 0.]);
        let expected = [
            0.08333333, 0.36144578, 0.28368794, 0., 0.91666667, 0.63855422, 0.71631206, 0.,
        ];

        let normalized = normalize_feature_sequence(&array);

        for (expected, actual) in expected.iter().zip(normalized.as_slice().iter()) {
            assert!((expected - actual).abs() < 1e-6);
        }
    }

    fn reference_template_matrix() -> [[i32; 10]; 12] {
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    }

    fn synth_tones(freqs: &[f32], sr: u32, secs: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        let mut out = vec![0.0f32; n];
        let scale = 1.0 / freqs.len() as f32;
        for &f in freqs {
            for (i, s) in out.iter_mut().enumerate() {
                *s += scale * (2.0 * std::f32::consts::PI * f * i as f32 / sr as f32).sin();
            }
        }
        out
    }

    #[test]
    fn test_template_masks_match_template_columns() {
        // TEMPLATE_MASKS must encode the COLUMNS of the template matrix
        // (bit i set <=> row i of that column is 1).
        let templates = reference_template_matrix();
        for col in 0..templates[0].len() {
            let mut mask = 0u16;
            for row in 0..templates.len() {
                if templates[row][col] != 0 {
                    mask |= 1 << row;
                }
            }
            assert_eq!(
                TEMPLATE_MASKS[col], mask,
                "mask for template column {col} does not match template matrix"
            );
        }
    }

    #[test]
    fn test_chroma_major_triad_feature_dominates() {
        // C4-E4-G4 major triad -> feature 6 (major triad template {0,4,7})
        // must dominate the triad group [6..10].
        let sr = 22050u32;
        let signal = synth_tones(&[261.63, 329.63, 392.00], sr, 5.0);
        let features = compute_chroma_features(&signal, sr).unwrap();
        let triads = &features[6..10];
        assert!(
            triads[0] > triads[1] && triads[0] > triads[2] && triads[0] > triads[3],
            "major triad feature should dominate: {triads:?}"
        );
    }

    #[test]
    fn test_chroma_minor_triad_feature_dominates() {
        // C4-Eb4-G4 minor triad -> feature 7 (minor triad template {0,3,7})
        // must dominate the triad group [6..10].
        let sr = 22050u32;
        let signal = synth_tones(&[261.63, 311.13, 392.00], sr, 5.0);
        let features = compute_chroma_features(&signal, sr).unwrap();
        let triads = &features[6..10];
        assert!(
            triads[1] > triads[0] && triads[1] > triads[2] && triads[1] > triads[3],
            "minor triad feature should dominate: {triads:?}"
        );
    }

    #[test]
    fn test_chroma_dyad_major_third_feature_dominates() {
        // C4-E4 dyad -> feature 3 (major third template {0,4}) must dominate
        // the interval group [0..6].
        let sr = 22050u32;
        let signal = synth_tones(&[261.63, 329.63], sr, 5.0);
        let features = compute_chroma_features(&signal, sr).unwrap();
        let intervals = &features[0..6];
        let argmax = intervals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(argmax, 3, "interval features: {intervals:?}");
    }

    #[test]
    fn test_chroma_short_signal_returns_err() {
        // Shorter than half the window: reflect-padding would panic.
        let signal = vec![0.5f32; 100];
        let result = compute_chroma_features(&signal, 22050);
        assert!(result.is_err(), "short signal should error, not panic");
        let result = ChromaFeatureExtractor::new().compute(&signal, 22050);
        assert!(result.is_err(), "short signal should error, not panic");
    }

    #[test]
    fn test_chroma_exactly_half_window_returns_err() {
        let signal = vec![0.5f32; WINDOW_SIZE / 2];
        assert!(compute_chroma_features(&signal, 22050).is_err());
        assert!(
            ChromaFeatureExtractor::new()
                .compute(&signal, 22050)
                .is_err()
        );
    }

    #[test]
    fn test_pip_track_few_bins_no_panic() {
        // n_fft=4 at sr=600: the [150, 300) Hz freq mask selects only bin 1,
        // so `end - 3` would underflow and `beginning..end - 3` would panic.
        let spectrum = DenseMatrix::from_elem(3, 4, 1.0);
        let result = pip_track(600, &spectrum, 4);
        assert!(result.is_ok(), "pip_track should not panic: {result:?}");
    }

    #[test]
    fn test_chroma_both_paths_agree() {
        // compute_chroma_features (utils::stft) and ChromaFeatureExtractor::compute
        // must produce matching features for the same input.
        let sr = 22050u32;
        let signal = synth_tones(&[442.0, 554.4, 659.3], sr, 5.0);
        let a = compute_chroma_features(&signal, sr).unwrap();
        let b = ChromaFeatureExtractor::new().compute(&signal, sr).unwrap();
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
            assert!(
                (x - y).abs() < 1e-3,
                "feature {i} differs between paths: {x} vs {y}"
            );
        }
    }

    #[test]
    fn test_chroma_empty_freq_mask_low_sr() {
        // Issue #5: very low sample rates can produce an empty freq mask
        // because Nyquist is below fmin (150 Hz).  This should not panic.
        let sr = 200u32; // Nyquist = 100 Hz < fmin = 150 Hz
        // Need at least WINDOW_SIZE (8192) + 1 samples for reflect_pad.
        let signal = vec![0.5f32; 8200];
        let result = compute_chroma_features(&signal, sr);
        assert!(
            result.is_ok(),
            "low-SR chroma should not error: {:?}",
            result
        );
    }
}

#[test]
fn test_pitch_tuning_empty_returns_zero() {
    let mut freqs: Vec<f64> = Vec::new();
    let tuning = pitch_tuning(&mut freqs, 0.01, 12).unwrap();
    assert_eq!(tuning, 0.0);
}
