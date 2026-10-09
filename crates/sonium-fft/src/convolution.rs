//! FFT-based linear convolution with an explicit padding and scratch contract.
//!
//! The forward and inverse transforms use the crate's unnormalised DFT
//! convention. A linear convolution is therefore performed with
//! `N >= lhs.len() + rhs.len() - 1` zero-padding and an inverse scale of
//! `1/N`. [`LinearConvolutionPlan`] keeps the FFT plans and workspaces
//! reusable so steady-state calls do not allocate.

use num_complex::Complex;
use std::sync::Arc;

use crate::CpuFftDispatch;
use crate::FftDirection;
use crate::FftPlan;
use crate::FftScalar;
use crate::cpu::planner::CpuFftPlanner;
use crate::factor::{PlanLengthError, try_plan_length};

/// Validation failures from checked convolution planning/execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvolutionError {
    /// A convolution operand has zero length.
    EmptyInput,
    /// The requested linear-convolution output length overflows `usize`.
    LengthOverflow,
    /// The FFT plan cannot represent the requested transform length.
    Plan(PlanLengthError),
    /// An execution buffer does not match the plan's shape contract.
    LengthMismatch {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    /// The supplied scratch buffer is too short.
    ScratchTooShort { required: usize, actual: usize },
    /// A complex sample contains a non-finite component.
    NonFiniteInput { buffer: &'static str, index: usize },
}

impl std::fmt::Display for ConvolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => f.write_str("convolution operands must be non-empty"),
            Self::LengthOverflow => f.write_str("convolution output length overflows usize"),
            Self::Plan(error) => write!(f, "invalid convolution FFT plan: {error}"),
            Self::LengthMismatch {
                buffer,
                expected,
                actual,
            } => write!(
                f,
                "{buffer} length mismatch: expected {expected}, got {actual}"
            ),
            Self::ScratchTooShort { required, actual } => {
                write!(
                    f,
                    "convolution scratch is too short: need {required}, got {actual}"
                )
            }
            Self::NonFiniteInput { buffer, index } => {
                write!(f, "{buffer}[{index}] contains a non-finite component")
            }
        }
    }
}

impl std::error::Error for ConvolutionError {}

fn validate_samples<T: FftScalar>(
    buffer: &[Complex<T>],
    name: &'static str,
) -> Result<(), ConvolutionError> {
    buffer
        .iter()
        .enumerate()
        .find(|(_, value)| !value.re.is_finite() || !value.im.is_finite())
        .map_or(Ok(()), |(index, _)| {
            Err(ConvolutionError::NonFiniteInput {
                buffer: name,
                index,
            })
        })
}

/// Reusable plan for the linear convolution of fixed-size complex vectors.
pub struct LinearConvolutionPlan<T: FftScalar> {
    lhs_len: usize,
    rhs_len: usize,
    output_len: usize,
    fft_len: usize,
    forward: Arc<dyn FftPlan<T>>,
    inverse: Arc<dyn FftPlan<T>>,
}

impl<T: FftScalar> LinearConvolutionPlan<T> {
    /// Build a convolution plan for vectors of the supplied lengths.
    ///
    /// Empty vectors are rejected because there is no positive FFT length
    /// associated with an empty convolution. The free [`linear_convolution`]
    /// helper handles empty inputs by returning an empty vector.
    pub fn new(lhs_len: usize, rhs_len: usize) -> Self {
        Self::try_new(lhs_len, rhs_len)
            .unwrap_or_else(|error| panic!("cannot build linear convolution plan: {error}"))
    }

    /// Checked plan construction for externally supplied operand lengths.
    pub fn try_new(lhs_len: usize, rhs_len: usize) -> Result<Self, ConvolutionError> {
        Self::try_new_with_dispatch(lhs_len, rhs_len, CpuFftDispatch::Auto)
    }

    /// Construct a linear convolution plan with an explicit CPU FFT dispatch
    /// policy. This is primarily useful for scalar/AVX2 parity evidence.
    pub fn new_with_dispatch(lhs_len: usize, rhs_len: usize, dispatch: CpuFftDispatch) -> Self {
        Self::try_new_with_dispatch(lhs_len, rhs_len, dispatch)
            .unwrap_or_else(|error| panic!("cannot build linear convolution plan: {error}"))
    }

    /// Checked counterpart to [`Self::new_with_dispatch`].
    pub fn try_new_with_dispatch(
        lhs_len: usize,
        rhs_len: usize,
        dispatch: CpuFftDispatch,
    ) -> Result<Self, ConvolutionError> {
        if lhs_len == 0 || rhs_len == 0 {
            return Err(ConvolutionError::EmptyInput);
        }
        let output_len = lhs_len
            .checked_add(rhs_len)
            .and_then(|length| length.checked_sub(1))
            .ok_or(ConvolutionError::LengthOverflow)?;
        let fft_len = output_len
            .checked_next_power_of_two()
            .ok_or(ConvolutionError::LengthOverflow)?;
        try_plan_length(fft_len).map_err(ConvolutionError::Plan)?;
        let planner = CpuFftPlanner::<T>::with_dispatch(dispatch);
        Ok(Self {
            lhs_len,
            rhs_len,
            output_len,
            fft_len,
            forward: planner
                .try_plan(fft_len, FftDirection::Forward)
                .map_err(ConvolutionError::Plan)?,
            inverse: planner
                .try_plan(fft_len, FftDirection::Inverse)
                .map_err(ConvolutionError::Plan)?,
        })
    }

    pub fn lhs_len(&self) -> usize {
        self.lhs_len
    }

    pub fn rhs_len(&self) -> usize {
        self.rhs_len
    }

    pub fn output_len(&self) -> usize {
        self.output_len
    }

    /// Padded FFT length used by the plan.
    pub fn fft_len(&self) -> usize {
        self.fft_len
    }

    /// Number of complex elements required by [`Self::process_with_scratch`].
    ///
    /// The first two `fft_len` elements hold the padded operands. The
    /// remaining elements are scratch for the forward/inverse FFT plans.
    pub fn scratch_len(&self) -> usize {
        self.try_scratch_len()
            .expect("linear convolution scratch length overflows usize")
    }

    /// Checked scratch-size query.
    pub fn try_scratch_len(&self) -> Result<usize, ConvolutionError> {
        self.fft_len
            .checked_mul(2)
            .and_then(|operands| {
                operands.checked_add(self.forward.scratch_len().max(self.inverse.scratch_len()))
            })
            .ok_or(ConvolutionError::LengthOverflow)
    }

    /// Compute the convolution, allocating its output and workspaces.
    pub fn process(&self, lhs: &[Complex<T>], rhs: &[Complex<T>]) -> Vec<Complex<T>> {
        let mut output = vec![Complex::new(T::zero(), T::zero()); self.output_len];
        let mut scratch = vec![Complex::new(T::zero(), T::zero()); self.scratch_len()];
        self.process_with_scratch(lhs, rhs, &mut output, &mut scratch);
        output
    }

    /// Compute the convolution without allocating.
    ///
    /// `lhs` and `rhs` must match the lengths used to build the plan, and
    /// `output` must have exactly [`Self::output_len`]. The output is the
    /// ordinary linear convolution, not circular convolution.
    pub fn process_with_scratch(
        &self,
        lhs: &[Complex<T>],
        rhs: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        assert_eq!(
            lhs.len(),
            self.lhs_len,
            "linear convolution lhs length mismatch"
        );
        assert_eq!(
            rhs.len(),
            self.rhs_len,
            "linear convolution rhs length mismatch"
        );
        assert_eq!(
            output.len(),
            self.output_len,
            "linear convolution output length mismatch"
        );
        assert!(
            scratch.len() >= self.scratch_len(),
            "linear convolution scratch is too short"
        );

        let (operands, fft_scratch) = scratch.split_at_mut(2 * self.fft_len);
        let (lhs_fft, rhs_fft) = operands.split_at_mut(self.fft_len);
        lhs_fft.fill(Complex::new(T::zero(), T::zero()));
        rhs_fft.fill(Complex::new(T::zero(), T::zero()));
        lhs_fft[..self.lhs_len].copy_from_slice(lhs);
        rhs_fft[..self.rhs_len].copy_from_slice(rhs);

        self.forward
            .process_with_scratch(lhs_fft, &mut fft_scratch[..self.forward.scratch_len()]);
        self.forward
            .process_with_scratch(rhs_fft, &mut fft_scratch[..self.forward.scratch_len()]);
        for (lhs_value, rhs_value) in lhs_fft.iter_mut().zip(rhs_fft.iter()) {
            *lhs_value = *lhs_value * *rhs_value;
        }
        self.inverse
            .process_with_scratch(lhs_fft, &mut fft_scratch[..self.inverse.scratch_len()]);

        let scale = T::from_usize(self.fft_len)
            .expect("FFT length must be representable by the convolution scalar")
            .recip();
        for (dst, &value) in output.iter_mut().zip(lhs_fft.iter()) {
            *dst = value * scale;
        }
    }

    /// Checked allocation-free linear convolution.
    pub fn try_process_with_scratch(
        &self,
        lhs: &[Complex<T>],
        rhs: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ConvolutionError> {
        validate_samples(lhs, "lhs")?;
        validate_samples(rhs, "rhs")?;
        if lhs.len() != self.lhs_len {
            return Err(ConvolutionError::LengthMismatch {
                buffer: "lhs",
                expected: self.lhs_len,
                actual: lhs.len(),
            });
        }
        if rhs.len() != self.rhs_len {
            return Err(ConvolutionError::LengthMismatch {
                buffer: "rhs",
                expected: self.rhs_len,
                actual: rhs.len(),
            });
        }
        if output.len() != self.output_len {
            return Err(ConvolutionError::LengthMismatch {
                buffer: "output",
                expected: self.output_len,
                actual: output.len(),
            });
        }
        let required = self.try_scratch_len()?;
        if scratch.len() < required {
            return Err(ConvolutionError::ScratchTooShort {
                required,
                actual: scratch.len(),
            });
        }
        self.process_with_scratch(lhs, rhs, output, scratch);
        Ok(())
    }
}

/// Compute a complex linear convolution using a temporary reusable plan.
pub fn linear_convolution<T: FftScalar>(lhs: &[Complex<T>], rhs: &[Complex<T>]) -> Vec<Complex<T>> {
    if lhs.is_empty() || rhs.is_empty() {
        return Vec::new();
    }
    LinearConvolutionPlan::new(lhs.len(), rhs.len()).process(lhs, rhs)
}

/// Reusable plan for an `N`-point circular convolution.
pub struct CircularConvolutionPlan<T: FftScalar> {
    n: usize,
    forward: Arc<dyn FftPlan<T>>,
    inverse: Arc<dyn FftPlan<T>>,
}

impl<T: FftScalar> CircularConvolutionPlan<T> {
    /// Build a plan for two equal-length vectors of length `n`.
    pub fn new(n: usize) -> Self {
        Self::try_new(n)
            .unwrap_or_else(|error| panic!("cannot build circular convolution plan: {error}"))
    }

    /// Checked plan construction for an externally supplied transform length.
    pub fn try_new(n: usize) -> Result<Self, ConvolutionError> {
        Self::try_new_with_dispatch(n, CpuFftDispatch::Auto)
    }

    /// Construct a circular convolution plan with an explicit CPU FFT
    /// dispatch policy.
    pub fn new_with_dispatch(n: usize, dispatch: CpuFftDispatch) -> Self {
        Self::try_new_with_dispatch(n, dispatch)
            .unwrap_or_else(|error| panic!("cannot build circular convolution plan: {error}"))
    }

    /// Checked counterpart to [`Self::new_with_dispatch`].
    pub fn try_new_with_dispatch(
        n: usize,
        dispatch: CpuFftDispatch,
    ) -> Result<Self, ConvolutionError> {
        if n == 0 {
            return Err(ConvolutionError::EmptyInput);
        }
        try_plan_length(n).map_err(ConvolutionError::Plan)?;
        let planner = CpuFftPlanner::<T>::with_dispatch(dispatch);
        Ok(Self {
            n,
            forward: planner
                .try_plan(n, FftDirection::Forward)
                .map_err(ConvolutionError::Plan)?,
            inverse: planner
                .try_plan(n, FftDirection::Inverse)
                .map_err(ConvolutionError::Plan)?,
        })
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Number of complex elements required by [`Self::process_with_scratch`].
    pub fn scratch_len(&self) -> usize {
        self.try_scratch_len()
            .expect("circular convolution scratch length overflows usize")
    }

    /// Checked scratch-size query.
    pub fn try_scratch_len(&self) -> Result<usize, ConvolutionError> {
        self.n
            .checked_mul(2)
            .and_then(|operands| {
                operands.checked_add(self.forward.scratch_len().max(self.inverse.scratch_len()))
            })
            .ok_or(ConvolutionError::LengthOverflow)
    }

    /// Compute the circular convolution, allocating output and workspaces.
    pub fn process(&self, lhs: &[Complex<T>], rhs: &[Complex<T>]) -> Vec<Complex<T>> {
        let mut output = vec![Complex::new(T::zero(), T::zero()); self.n];
        let mut scratch = vec![Complex::new(T::zero(), T::zero()); self.scratch_len()];
        self.process_with_scratch(lhs, rhs, &mut output, &mut scratch);
        output
    }

    /// Compute `y[k] = Σ_j lhs[j] · rhs[(k-j) mod N]` without allocating.
    pub fn process_with_scratch(
        &self,
        lhs: &[Complex<T>],
        rhs: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        assert_eq!(
            lhs.len(),
            self.n,
            "circular convolution lhs length mismatch"
        );
        assert_eq!(
            rhs.len(),
            self.n,
            "circular convolution rhs length mismatch"
        );
        assert_eq!(
            output.len(),
            self.n,
            "circular convolution output length mismatch"
        );
        assert!(
            scratch.len() >= self.scratch_len(),
            "circular convolution scratch is too short"
        );

        let (operands, fft_scratch) = scratch.split_at_mut(2 * self.n);
        let (lhs_fft, rhs_fft) = operands.split_at_mut(self.n);
        lhs_fft.copy_from_slice(lhs);
        rhs_fft.copy_from_slice(rhs);
        self.forward
            .process_with_scratch(lhs_fft, &mut fft_scratch[..self.forward.scratch_len()]);
        self.forward
            .process_with_scratch(rhs_fft, &mut fft_scratch[..self.forward.scratch_len()]);
        for (lhs_value, rhs_value) in lhs_fft.iter_mut().zip(rhs_fft.iter()) {
            *lhs_value = *lhs_value * *rhs_value;
        }
        self.inverse
            .process_with_scratch(lhs_fft, &mut fft_scratch[..self.inverse.scratch_len()]);

        let scale = T::from_usize(self.n)
            .expect("FFT length must be representable by the convolution scalar")
            .recip();
        for (dst, &value) in output.iter_mut().zip(lhs_fft.iter()) {
            *dst = value * scale;
        }
    }

    /// Checked allocation-free circular convolution.
    pub fn try_process_with_scratch(
        &self,
        lhs: &[Complex<T>],
        rhs: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ConvolutionError> {
        validate_samples(lhs, "lhs")?;
        validate_samples(rhs, "rhs")?;
        if lhs.len() != self.n {
            return Err(ConvolutionError::LengthMismatch {
                buffer: "lhs",
                expected: self.n,
                actual: lhs.len(),
            });
        }
        if rhs.len() != self.n {
            return Err(ConvolutionError::LengthMismatch {
                buffer: "rhs",
                expected: self.n,
                actual: rhs.len(),
            });
        }
        if output.len() != self.n {
            return Err(ConvolutionError::LengthMismatch {
                buffer: "output",
                expected: self.n,
                actual: output.len(),
            });
        }
        let required = self.try_scratch_len()?;
        if scratch.len() < required {
            return Err(ConvolutionError::ScratchTooShort {
                required,
                actual: scratch.len(),
            });
        }
        self.process_with_scratch(lhs, rhs, output, scratch);
        Ok(())
    }
}

/// Compute a circular convolution of two equal-length vectors.
pub fn circular_convolution<T: FftScalar>(
    lhs: &[Complex<T>],
    rhs: &[Complex<T>],
) -> Vec<Complex<T>> {
    if lhs.is_empty() || rhs.is_empty() {
        return Vec::new();
    }
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "circular convolution lengths must match"
    );
    CircularConvolutionPlan::new(lhs.len()).process(lhs, rhs)
}

/// Checked circular convolution with explicit equal-length validation.
pub fn try_circular_convolution<T: FftScalar>(
    lhs: &[Complex<T>],
    rhs: &[Complex<T>],
) -> Result<Vec<Complex<T>>, ConvolutionError> {
    if lhs.is_empty() || rhs.is_empty() {
        return Err(ConvolutionError::EmptyInput);
    }
    if lhs.len() != rhs.len() {
        return Err(ConvolutionError::LengthMismatch {
            buffer: "rhs",
            expected: lhs.len(),
            actual: rhs.len(),
        });
    }
    validate_samples(lhs, "lhs")?;
    validate_samples(rhs, "rhs")?;
    Ok(CircularConvolutionPlan::try_new(lhs.len())?.process(lhs, rhs))
}

/// Checked linear convolution with explicit non-empty and finite-input validation.
pub fn try_linear_convolution<T: FftScalar>(
    lhs: &[Complex<T>],
    rhs: &[Complex<T>],
) -> Result<Vec<Complex<T>>, ConvolutionError> {
    if lhs.is_empty() || rhs.is_empty() {
        return Err(ConvolutionError::EmptyInput);
    }
    validate_samples(lhs, "lhs")?;
    validate_samples(rhs, "rhs")?;
    Ok(LinearConvolutionPlan::try_new(lhs.len(), rhs.len())?.process(lhs, rhs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn direct(lhs: &[Complex64], rhs: &[Complex64]) -> Vec<Complex64> {
        let mut result = vec![Complex64::new(0.0, 0.0); lhs.len() + rhs.len() - 1];
        for (i, &a) in lhs.iter().enumerate() {
            for (j, &b) in rhs.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        result
    }

    #[test]
    fn linear_convolution_matches_direct_complex_convolution() {
        let lhs: Vec<_> = (0..11)
            .map(|i| Complex64::new((i as f64 * 0.17).sin(), i as f64 * 0.03))
            .collect();
        let rhs: Vec<_> = (0..7)
            .map(|i| Complex64::new((i as f64 * 0.29).cos(), -(i as f64) * 0.05))
            .collect();
        let actual = linear_convolution(&lhs, &rhs);
        let expected = direct(&lhs, &rhs);
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((*actual - *expected).norm() < 2.0e-11);
        }
    }

    #[test]
    fn reusable_plan_honours_scratch_and_padding_contract() {
        let plan = LinearConvolutionPlan::<f32>::new(5, 13);
        assert!(plan.fft_len() >= plan.output_len());
        assert!(plan.fft_len().is_power_of_two());
        let lhs = vec![Complex::new(0.25_f32, -0.1); 5];
        let rhs = vec![Complex::new(-0.3_f32, 0.2); 13];
        let mut output = vec![Complex::new(0.0, 0.0); plan.output_len()];
        let mut scratch = vec![Complex::new(0.0, 0.0); plan.scratch_len()];
        plan.process_with_scratch(&lhs, &rhs, &mut output, &mut scratch);
        let expected = direct(
            &lhs.iter()
                .map(|v| Complex64::new(v.re as f64, v.im as f64))
                .collect::<Vec<_>>(),
            &rhs.iter()
                .map(|v| Complex64::new(v.re as f64, v.im as f64))
                .collect::<Vec<_>>(),
        );
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!((actual.re as f64 - expected.re).abs() < 2.0e-5);
            assert!((actual.im as f64 - expected.im).abs() < 2.0e-5);
        }
    }

    #[test]
    fn empty_free_function_is_an_empty_convolution() {
        assert!(linear_convolution::<f64>(&[], &[Complex64::new(1.0, 0.0)]).is_empty());
        assert!(linear_convolution::<f64>(&[Complex64::new(1.0, 0.0)], &[]).is_empty());
        assert!(circular_convolution::<f64>(&[], &[]).is_empty());
    }

    #[test]
    fn checked_convolution_api_rejects_invalid_shapes_and_nonfinite_samples() {
        assert!(matches!(
            LinearConvolutionPlan::<f64>::try_new(0, 2),
            Err(ConvolutionError::EmptyInput)
        ));
        assert!(matches!(
            CircularConvolutionPlan::<f64>::try_new(0),
            Err(ConvolutionError::EmptyInput)
        ));
        assert_eq!(
            try_circular_convolution::<f64>(&[Complex64::new(1.0, 0.0)], &[]),
            Err(ConvolutionError::EmptyInput)
        );
        assert_eq!(
            try_circular_convolution::<f64>(
                &[Complex64::new(1.0, 0.0)],
                &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]
            ),
            Err(ConvolutionError::LengthMismatch {
                buffer: "rhs",
                expected: 1,
                actual: 2,
            })
        );
        assert_eq!(
            try_linear_convolution::<f64>(
                &[Complex64::new(f64::NAN, 0.0)],
                &[Complex64::new(1.0, 0.0)]
            ),
            Err(ConvolutionError::NonFiniteInput {
                buffer: "lhs",
                index: 0,
            })
        );
    }

    #[test]
    fn checked_convolution_api_rejects_short_scratch_and_length_overflow() {
        let plan = LinearConvolutionPlan::<f64>::try_new(2, 2).unwrap();
        let lhs = [Complex64::new(1.0, 0.0); 2];
        let rhs = [Complex64::new(1.0, 0.0); 2];
        let mut output = vec![Complex64::new(0.0, 0.0); plan.output_len()];
        let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len() - 1];
        assert!(matches!(
            plan.try_process_with_scratch(&lhs, &rhs, &mut output, &mut scratch),
            Err(ConvolutionError::ScratchTooShort { .. })
        ));
        assert!(matches!(
            LinearConvolutionPlan::<f64>::try_new(usize::MAX, 2),
            Err(ConvolutionError::LengthOverflow)
        ));
    }

    #[test]
    fn checked_plan_execution_matches_legacy_convolution() {
        let lhs = [Complex64::new(1.0, 0.5), Complex64::new(-0.2, 0.3)];
        let rhs = [Complex64::new(0.4, -0.1), Complex64::new(0.7, 0.2)];
        let plan = LinearConvolutionPlan::<f64>::try_new(lhs.len(), rhs.len()).unwrap();
        assert_eq!(plan.try_scratch_len().unwrap(), plan.scratch_len());
        let mut output = vec![Complex64::new(0.0, 0.0); plan.output_len()];
        let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];
        plan.try_process_with_scratch(&lhs, &rhs, &mut output, &mut scratch)
            .unwrap();
        let expected = linear_convolution(&lhs, &rhs);
        assert!(
            output
                .iter()
                .zip(expected.iter())
                .all(|(actual, expected)| (*actual - *expected).norm() < 1e-12)
        );

        let circular = CircularConvolutionPlan::<f64>::try_new(lhs.len()).unwrap();
        let mut circular_output = vec![Complex64::new(0.0, 0.0); circular.len()];
        let mut circular_scratch = vec![Complex64::new(0.0, 0.0); circular.scratch_len()];
        circular
            .try_process_with_scratch(&lhs, &rhs, &mut circular_output, &mut circular_scratch)
            .unwrap();
        let expected_circular = circular_convolution(&lhs, &rhs);
        assert!(
            circular_output
                .iter()
                .zip(expected_circular.iter())
                .all(|(actual, expected)| (*actual - *expected).norm() < 1e-12)
        );
    }

    #[test]
    fn circular_convolution_matches_direct_periodic_convolution() {
        let lhs: Vec<_> = (0..9)
            .map(|i| Complex64::new((i as f64 * 0.11).sin(), i as f64 * 0.07))
            .collect();
        let rhs: Vec<_> = (0..9)
            .map(|i| Complex64::new((i as f64 * 0.23).cos(), -(i as f64) * 0.02))
            .collect();
        let actual = circular_convolution(&lhs, &rhs);
        let mut expected = vec![Complex64::new(0.0, 0.0); lhs.len()];
        for k in 0..lhs.len() {
            for j in 0..lhs.len() {
                expected[k] += lhs[j] * rhs[(k + lhs.len() - j) % lhs.len()];
            }
        }
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((*actual - *expected).norm() < 2.0e-11);
        }
    }
}
