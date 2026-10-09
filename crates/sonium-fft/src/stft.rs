//! Windowed real STFT/ISTFT with explicit framing and padding semantics.
//!
//! Spectra are stored frame-major, with each frame containing the real FFT
//! half-spectrum `fft_size / 2 + 1`. Analysis applies the configured window;
//! synthesis performs weighted overlap-add and divides by the accumulated
//! squared window, giving perfect reconstruction wherever the window cover is
//! nonzero.

use num_complex::Complex;
use std::fmt;

use crate::factor::PlanLengthError;
use crate::real::RealFftPlan;
use crate::scalar::FftScalar;

/// Window family used by [`StftPlan`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowFunction {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
}

/// Errors returned by checked STFT construction and execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StftError {
    /// FFT or hop sizes violate the framing contract.
    InvalidConfig { fft_size: usize, hop_size: usize },
    /// The underlying real FFT planner rejected the transform length.
    Plan(PlanLengthError),
    /// A frame count or output length overflowed `usize`.
    LengthOverflow,
    /// A caller-provided output or scratch buffer is too short.
    BufferTooShort {
        buffer: &'static str,
        required: usize,
        actual: usize,
    },
}

impl fmt::Display for StftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { fft_size, hop_size } => {
                write!(
                    f,
                    "invalid STFT config: fft_size={fft_size}, hop_size={hop_size}"
                )
            }
            Self::Plan(error) => write!(f, "STFT FFT plan failed: {error}"),
            Self::LengthOverflow => f.write_str("STFT length arithmetic overflowed usize"),
            Self::BufferTooShort {
                buffer,
                required,
                actual,
            } => write!(
                f,
                "STFT {buffer} buffer is too short: need {required}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for StftError {}

impl From<PlanLengthError> for StftError {
    fn from(error: PlanLengthError) -> Self {
        Self::Plan(error)
    }
}

/// Framing and boundary configuration for an STFT.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StftConfig {
    pub fft_size: usize,
    pub hop_size: usize,
    pub window: WindowFunction,
    /// Add `fft_size / 2` implicit zeros on both sides of the signal in the
    /// analysis coordinate system. Synthesis trims those coordinates back to
    /// the requested output length.
    pub center: bool,
    /// Include a final zero-padded frame when the available signal does not
    /// end on an exact hop.
    pub pad_end: bool,
}

impl StftConfig {
    pub fn new(fft_size: usize, hop_size: usize, window: WindowFunction) -> Self {
        Self {
            fft_size,
            hop_size,
            window,
            center: false,
            pad_end: true,
        }
    }
}

/// Reusable real-input STFT plan.
pub struct StftPlan<T: FftScalar> {
    config: StftConfig,
    fft: RealFftPlan<T>,
    window: Vec<T>,
}

impl<T: FftScalar> StftPlan<T> {
    pub fn new(config: StftConfig) -> Self {
        Self::try_new(config).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Checked construction for externally supplied framing parameters.
    pub fn try_new(config: StftConfig) -> Result<Self, StftError> {
        if config.fft_size == 0 || config.hop_size == 0 || config.hop_size > config.fft_size {
            return Err(StftError::InvalidConfig {
                fft_size: config.fft_size,
                hop_size: config.hop_size,
            });
        }
        let fft = RealFftPlan::try_new(config.fft_size)?;
        let window = (0..config.fft_size)
            .map(|i| window_value(config.window, i, config.fft_size))
            .collect();
        Ok(Self {
            config,
            fft,
            window,
        })
    }

    pub fn config(&self) -> StftConfig {
        self.config
    }

    pub fn fft_size(&self) -> usize {
        self.config.fft_size
    }

    pub fn hop_size(&self) -> usize {
        self.config.hop_size
    }

    pub fn spectrum_len(&self) -> usize {
        self.fft.spectrum_len()
    }

    /// Number of frames emitted for a signal of `signal_len` samples.
    pub fn frame_count(&self, signal_len: usize) -> usize {
        self.try_frame_count(signal_len)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Checked frame-count calculation that never wraps `usize`.
    pub fn try_frame_count(&self, signal_len: usize) -> Result<usize, StftError> {
        let padding = usize::from(self.config.center)
            .checked_mul(self.config.fft_size / 2)
            .ok_or(StftError::LengthOverflow)?;
        let available = signal_len
            .checked_add(padding.checked_mul(2).ok_or(StftError::LengthOverflow)?)
            .ok_or(StftError::LengthOverflow)?;
        if available < self.fft_size() {
            return Ok(usize::from(self.config.pad_end && available > 0));
        }
        let remaining = available - self.fft_size();
        if self.config.pad_end {
            remaining
                .checked_div(self.config.hop_size)
                .and_then(|frames| {
                    frames.checked_add(1).and_then(|frames| {
                        frames.checked_add(usize::from(
                            !remaining.is_multiple_of(self.config.hop_size),
                        ))
                    })
                })
                .ok_or(StftError::LengthOverflow)
        } else {
            remaining
                .checked_div(self.config.hop_size)
                .and_then(|frames| frames.checked_add(1))
                .ok_or(StftError::LengthOverflow)
        }
    }

    /// Analyze a signal, allocating only the returned frame-major spectrum.
    pub fn analyze(&self, input: &[T]) -> Vec<Complex<T>> {
        let frames = self.frame_count(input.len());
        let mut output = vec![Complex::new(T::zero(), T::zero()); frames * self.spectrum_len()];
        let mut fft_scratch = vec![Complex::new(T::zero(), T::zero()); self.fft.scratch_len()];
        let mut frame = vec![T::zero(); self.fft_size()];
        self.analyze_with_scratch(input, &mut output, &mut fft_scratch, &mut frame);
        output
    }

    /// Allocation-free STFT analysis. Returns the number of frames written.
    pub fn analyze_with_scratch(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        fft_scratch: &mut [Complex<T>],
        frame: &mut [T],
    ) -> usize {
        let frames = self.frame_count(input.len());
        assert_eq!(
            output.len(),
            frames * self.spectrum_len(),
            "STFT output length mismatch"
        );
        assert!(
            frame.len() >= self.fft_size(),
            "STFT frame scratch is too short"
        );
        assert!(
            fft_scratch.len() >= self.fft.scratch_len(),
            "STFT FFT scratch is too short"
        );
        let padding = if self.config.center {
            self.config.fft_size / 2
        } else {
            0
        };
        for frame_index in 0..frames {
            let start = frame_index * self.config.hop_size;
            for (i, sample) in frame[..self.fft_size()].iter_mut().enumerate() {
                let source = start as isize + i as isize - padding as isize;
                *sample = if source >= 0 && (source as usize) < input.len() {
                    input[source as usize] * self.window[i]
                } else {
                    T::zero()
                };
            }
            let spectrum = &mut output
                [frame_index * self.spectrum_len()..(frame_index + 1) * self.spectrum_len()];
            self.fft
                .forward_with_scratch(&frame[..self.fft_size()], spectrum, fft_scratch);
        }
        frames
    }

    /// Checked allocation-free STFT analysis.
    pub fn try_analyze_with_scratch(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        fft_scratch: &mut [Complex<T>],
        frame: &mut [T],
    ) -> Result<usize, StftError> {
        let frames = self.try_frame_count(input.len())?;
        let output_len = frames
            .checked_mul(self.spectrum_len())
            .ok_or(StftError::LengthOverflow)?;
        if output.len() != output_len {
            return Err(StftError::BufferTooShort {
                buffer: "output",
                required: output_len,
                actual: output.len(),
            });
        }
        if frame.len() < self.fft_size() {
            return Err(StftError::BufferTooShort {
                buffer: "frame",
                required: self.fft_size(),
                actual: frame.len(),
            });
        }
        if fft_scratch.len() < self.fft.scratch_len() {
            return Err(StftError::BufferTooShort {
                buffer: "FFT scratch",
                required: self.fft.scratch_len(),
                actual: fft_scratch.len(),
            });
        }
        Ok(self.analyze_with_scratch(input, output, fft_scratch, frame))
    }

    /// Synthesize a signal of `signal_len` samples, allocating the workspaces
    /// needed by weighted overlap-add.
    pub fn synthesize(&self, spectra: &[Complex<T>], signal_len: usize) -> Vec<T> {
        let mut output = vec![T::zero(); signal_len];
        let mut fft_scratch = vec![Complex::new(T::zero(), T::zero()); self.fft.scratch_len()];
        let mut frame = vec![T::zero(); self.fft_size()];
        let mut norm = vec![T::zero(); signal_len];
        self.synthesize_with_scratch(
            spectra,
            &mut output,
            &mut fft_scratch,
            &mut frame,
            &mut norm,
        );
        output
    }

    /// Allocation-free weighted overlap-add synthesis. `norm` must be a
    /// zeroable buffer with the same length as `output`; it is overwritten.
    pub fn synthesize_with_scratch(
        &self,
        spectra: &[Complex<T>],
        output: &mut [T],
        fft_scratch: &mut [Complex<T>],
        frame: &mut [T],
        norm: &mut [T],
    ) {
        assert_eq!(
            spectra.len() % self.spectrum_len(),
            0,
            "STFT spectrum is not frame-aligned"
        );
        assert!(
            frame.len() >= self.fft_size(),
            "STFT frame scratch is too short"
        );
        assert!(
            fft_scratch.len() >= self.fft.scratch_len(),
            "STFT FFT scratch is too short"
        );
        assert_eq!(
            norm.len(),
            output.len(),
            "STFT normalization length mismatch"
        );
        output.fill(T::zero());
        norm.fill(T::zero());
        let frames = spectra.len() / self.spectrum_len();
        let padding = if self.config.center {
            self.config.fft_size / 2
        } else {
            0
        };
        for frame_index in 0..frames {
            let spectrum = &spectra
                [frame_index * self.spectrum_len()..(frame_index + 1) * self.spectrum_len()];
            self.fft
                .inverse_with_scratch(spectrum, &mut frame[..self.fft_size()], fft_scratch);
            let start = frame_index * self.config.hop_size;
            for (i, &frame_sample) in frame[..self.fft_size()].iter().enumerate() {
                let destination = start as isize + i as isize - padding as isize;
                if destination >= 0 && (destination as usize) < output.len() {
                    let index = destination as usize;
                    let window = self.window[i];
                    output[index] = output[index] + frame_sample * window;
                    norm[index] = norm[index] + window * window;
                }
            }
        }
        for (sample, weight) in output.iter_mut().zip(norm.iter()) {
            if *weight > T::epsilon() {
                *sample = *sample / *weight;
            }
        }
    }

    /// Checked weighted overlap-add synthesis.
    pub fn try_synthesize_with_scratch(
        &self,
        spectra: &[Complex<T>],
        output: &mut [T],
        fft_scratch: &mut [Complex<T>],
        frame: &mut [T],
        norm: &mut [T],
    ) -> Result<(), StftError> {
        if !spectra.len().is_multiple_of(self.spectrum_len()) {
            return Err(StftError::BufferTooShort {
                buffer: "spectrum",
                required: spectra
                    .len()
                    .saturating_add(self.spectrum_len() - spectra.len() % self.spectrum_len()),
                actual: spectra.len(),
            });
        }
        if frame.len() < self.fft_size() {
            return Err(StftError::BufferTooShort {
                buffer: "frame",
                required: self.fft_size(),
                actual: frame.len(),
            });
        }
        if fft_scratch.len() < self.fft.scratch_len() {
            return Err(StftError::BufferTooShort {
                buffer: "FFT scratch",
                required: self.fft.scratch_len(),
                actual: fft_scratch.len(),
            });
        }
        if norm.len() != output.len() {
            return Err(StftError::BufferTooShort {
                buffer: "normalization",
                required: output.len(),
                actual: norm.len(),
            });
        }
        self.synthesize_with_scratch(spectra, output, fft_scratch, frame, norm);
        Ok(())
    }
}

fn window_value<T: FftScalar>(window: WindowFunction, index: usize, n: usize) -> T {
    let phase = if n <= 1 {
        0.0
    } else {
        2.0 * std::f64::consts::PI * index as f64 / (n - 1) as f64
    };
    let value = match window {
        WindowFunction::Rectangular => 1.0,
        WindowFunction::Hann => 0.5 * (1.0 - phase.cos()),
        WindowFunction::Hamming => 0.54 - 0.46 * phase.cos(),
        WindowFunction::Blackman => 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos(),
    };
    T::from_f64(value).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hann_centered_stft_reconstructs() {
        let mut config = StftConfig::new(32, 8, WindowFunction::Hann);
        config.center = true;
        let plan = StftPlan::<f64>::new(config);
        let input: Vec<f64> = (0..257)
            .map(|i| (i as f64 * 0.07).sin() + 0.2 * (i as f64 * 0.013).cos())
            .collect();
        let spectra = plan.analyze(&input);
        let output = plan.synthesize(&spectra, input.len());
        for (actual, expected) in output.iter().zip(input.iter()) {
            assert_relative_eq!(actual, expected, max_relative = 1e-9, epsilon = 1e-9);
        }
    }

    #[test]
    fn frame_count_and_padding_are_explicit() {
        let mut config = StftConfig::new(8, 4, WindowFunction::Rectangular);
        config.pad_end = false;
        let plan = StftPlan::<f32>::new(config);
        assert_eq!(plan.frame_count(0), 0);
        assert_eq!(plan.frame_count(8), 1);
        assert_eq!(plan.frame_count(9), 1);
        config.pad_end = true;
        let plan = StftPlan::<f32>::new(config);
        assert_eq!(plan.frame_count(9), 2);
    }

    #[test]
    fn checked_constructor_and_scratch_boundary_are_typed() {
        assert!(matches!(
            StftPlan::<f64>::try_new(StftConfig::new(0, 1, WindowFunction::Rectangular)),
            Err(StftError::InvalidConfig { .. })
        ));
        let plan = StftPlan::<f64>::try_new(StftConfig::new(8, 4, WindowFunction::Hann)).unwrap();
        let frames = plan.try_frame_count(16).unwrap();
        let mut output = vec![Complex::new(0.0, 0.0); frames * plan.spectrum_len()];
        let mut fft_scratch = Vec::new();
        let mut frame = vec![0.0; plan.fft_size()];
        let error = plan
            .try_analyze_with_scratch(&[0.0; 16], &mut output, &mut fft_scratch, &mut frame)
            .unwrap_err();
        assert!(matches!(
            error,
            StftError::BufferTooShort {
                buffer: "FFT scratch",
                ..
            }
        ));
    }

    #[test]
    fn checked_constructor_rejects_unrepresentable_lengths() {
        let config = StftConfig::new(usize::MAX, usize::MAX, WindowFunction::Rectangular);
        assert!(matches!(
            StftPlan::<f64>::try_new(config),
            Err(StftError::Plan(PlanLengthError::LengthOverflow { .. }))
                | Err(StftError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn checked_synthesis_rejects_unaligned_spectrum_and_norm() {
        let plan =
            StftPlan::<f64>::try_new(StftConfig::new(8, 4, WindowFunction::Rectangular)).unwrap();
        let mut output = vec![0.0; 8];
        let mut frame = vec![0.0; 8];
        let mut scratch = vec![Complex::new(0.0, 0.0); plan.fft_size()];
        let mut norm = vec![0.0; 7];
        let error = plan
            .try_synthesize_with_scratch(
                &[Complex::new(0.0, 0.0)],
                &mut output,
                &mut scratch,
                &mut frame,
                &mut norm,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            StftError::BufferTooShort {
                buffer: "spectrum",
                ..
            }
        ));
    }

    #[test]
    fn analysis_frames_satisfy_windowed_parseval_identity() {
        let mut config = StftConfig::new(16, 4, WindowFunction::Blackman);
        config.pad_end = false;
        let plan = StftPlan::<f64>::new(config);
        let input: Vec<f64> = (0..48)
            .map(|index| {
                let x = index as f64;
                0.7 * (0.21 * x).sin() - 0.3 * (0.047 * x).cos() + 0.1 * (0.13 * x).sin()
            })
            .collect();
        let spectra = plan.analyze(&input);
        let frames = plan.frame_count(input.len());

        for frame_index in 0..frames {
            let time_energy: f64 = (0..plan.fft_size())
                .map(|index| {
                    let sample = input[frame_index * plan.hop_size() + index] * plan.window[index];
                    sample * sample
                })
                .sum();
            let spectrum = &spectra
                [frame_index * plan.spectrum_len()..(frame_index + 1) * plan.spectrum_len()];
            let frequency_energy: f64 = spectrum
                .iter()
                .enumerate()
                .map(|(bin, value)| {
                    let edge = bin == 0 || bin == plan.fft_size() / 2;
                    let weight = if edge { 1.0 } else { 2.0 };
                    weight * value.norm_sqr() / plan.fft_size() as f64
                })
                .sum();
            assert_relative_eq!(frequency_energy, time_energy, epsilon = 1.0e-11);
        }
    }
}
