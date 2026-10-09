//! Two- and three-dimensional CPU transforms and strided execution helpers.
//!
//! Plans use row-major contiguous storage. Each axis is transformed with the
//! same unnormalised complex DFT convention as one-dimensional plans. The
//! caller-provided gather buffer and one-dimensional work region make the
//! steady-state path allocation-free, including axes backed by Bluestein.

use num_complex::Complex;
use std::sync::Arc;

use crate::FftDirection;
use crate::FftPlan;
use crate::cpu::planner::CpuFftPlanner;
use crate::factor::PlanLengthError;
use crate::scalar::FftScalar;
use sonium_compute::transpose_2d;

/// Checked execution failure for strided and multidimensional FFT helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NdFftError {
    InvalidDimension,
    LengthOverflow,
    BufferTooShort { required: usize, actual: usize },
    ScratchTooShort { required: usize, actual: usize },
    AliasedBuffers,
    Plan(PlanLengthError),
}

impl std::fmt::Display for NdFftError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimension => formatter.write_str("FFT dimensions are invalid"),
            Self::LengthOverflow => formatter.write_str("FFT buffer length overflows usize"),
            Self::BufferTooShort { required, actual } => {
                write!(
                    formatter,
                    "FFT buffer has {actual} elements, needs {required}"
                )
            }
            Self::ScratchTooShort { required, actual } => {
                write!(
                    formatter,
                    "FFT scratch has {actual} elements, needs {required}"
                )
            }
            Self::AliasedBuffers => formatter.write_str("FFT input and scratch buffers overlap"),
            Self::Plan(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for NdFftError {}

impl From<PlanLengthError> for NdFftError {
    fn from(error: PlanLengthError) -> Self {
        Self::Plan(error)
    }
}

/// Apply a one-dimensional transform to regularly strided, independent
/// vectors. `stride` and `plan.len()` are measured in complex elements.
///
/// `scratch[..plan.len()]` is the gather buffer and the remaining region is
/// forwarded to [`FftPlan::process_with_scratch`]. Therefore callers must
/// provide at least `plan.len() + plan.scratch_len()` elements. The buffer
/// and scratch slices must be disjoint; padding between transforms is left
/// untouched.
pub fn process_strided<T: FftScalar>(
    plan: &dyn FftPlan<T>,
    buffer: &mut [Complex<T>],
    batch: usize,
    stride: usize,
    scratch: &mut [Complex<T>],
) {
    try_process_strided(plan, buffer, batch, stride, scratch)
        .expect("invalid strided FFT execution buffers");
}

/// Checked counterpart of [`process_strided`].
pub fn try_process_strided<T: FftScalar>(
    plan: &dyn FftPlan<T>,
    buffer: &mut [Complex<T>],
    batch: usize,
    stride: usize,
    scratch: &mut [Complex<T>],
) -> Result<(), NdFftError> {
    let n = plan.len();
    if n == 0 || stride < n {
        return Err(NdFftError::InvalidDimension);
    }
    let fft_scratch_len = plan.scratch_len();
    let required_scratch = n
        .checked_add(fft_scratch_len)
        .ok_or(NdFftError::LengthOverflow)?;
    if scratch.len() < required_scratch {
        return Err(NdFftError::ScratchTooShort {
            required: required_scratch,
            actual: scratch.len(),
        });
    }
    if batch == 0 {
        return Ok(());
    }
    let required = (batch - 1)
        .checked_mul(stride)
        .and_then(|v| v.checked_add(n))
        .ok_or(NdFftError::LengthOverflow)?;
    if buffer.len() < required {
        return Err(NdFftError::BufferTooShort {
            required,
            actual: buffer.len(),
        });
    }
    let (transform, fft_scratch) = scratch.split_at_mut(n);
    for b in 0..batch {
        let base = b * stride;
        for (j, dst) in transform.iter_mut().enumerate() {
            *dst = buffer[base + j];
        }
        plan.process_with_scratch(transform, fft_scratch);
        for (j, src) in transform.iter().enumerate() {
            buffer[base + j] = *src;
        }
    }
    Ok(())
}

/// Row-major two-dimensional FFT plan.
pub struct Fft2dPlan<T: FftScalar> {
    rows: usize,
    cols: usize,
    total_len: usize,
    direction: FftDirection,
    row_plan: Arc<dyn FftPlan<T>>,
    col_plan: Arc<dyn FftPlan<T>>,
}

impl<T: FftScalar> Fft2dPlan<T> {
    pub fn new(rows: usize, cols: usize, direction: FftDirection) -> Self {
        Self::try_new(rows, cols, direction).expect("invalid 2D FFT dimensions")
    }

    pub fn try_new(rows: usize, cols: usize, direction: FftDirection) -> Result<Self, NdFftError> {
        let planner = CpuFftPlanner::<T>::new();
        Self::try_with_planner(rows, cols, direction, &planner)
    }

    pub fn with_planner(
        rows: usize,
        cols: usize,
        direction: FftDirection,
        planner: &CpuFftPlanner<T>,
    ) -> Self {
        Self::try_with_planner(rows, cols, direction, planner).expect("invalid 2D FFT dimensions")
    }

    pub fn try_with_planner(
        rows: usize,
        cols: usize,
        direction: FftDirection,
        planner: &CpuFftPlanner<T>,
    ) -> Result<Self, NdFftError> {
        if rows == 0 || cols == 0 {
            return Err(NdFftError::InvalidDimension);
        }
        let total_len = rows.checked_mul(cols).ok_or(NdFftError::LengthOverflow)?;
        Ok(Self {
            rows,
            cols,
            total_len,
            direction,
            row_plan: planner.try_plan(cols, direction)?,
            col_plan: planner.try_plan(rows, direction)?,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn direction(&self) -> FftDirection {
        self.direction
    }

    pub fn scratch_len(&self) -> usize {
        self.try_scratch_len()
            .expect("2D FFT scratch size overflow")
    }

    pub fn try_scratch_len(&self) -> Result<usize, NdFftError> {
        self.total_len
            .checked_add(self.row_plan.scratch_len().max(self.col_plan.scratch_len()))
            .ok_or(NdFftError::LengthOverflow)
    }

    pub fn process(&self, buffer: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::new(T::zero(), T::zero()); self.scratch_len()];
        self.process_with_scratch(buffer, &mut scratch);
    }

    pub fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        self.try_process_with_scratch(buffer, scratch)
            .expect("invalid 2D FFT execution buffers");
    }

    pub fn try_process_with_scratch(
        &self,
        buffer: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), NdFftError> {
        if buffer.len() != self.total_len {
            return Err(NdFftError::BufferTooShort {
                required: self.total_len,
                actual: buffer.len(),
            });
        }
        let required_scratch = self.try_scratch_len()?;
        if scratch.len() < required_scratch {
            return Err(NdFftError::ScratchTooShort {
                required: required_scratch,
                actual: scratch.len(),
            });
        }

        let (transpose_scratch, fft_scratch) = scratch.split_at_mut(self.total_len);

        for row in buffer.chunks_exact_mut(self.cols) {
            self.row_plan.process_with_scratch(row, fft_scratch);
        }

        transpose_2d(buffer, transpose_scratch, self.rows, self.cols).map_err(
            |error| match error {
                sonium_compute::TransposeError::LengthOverflow => NdFftError::LengthOverflow,
                sonium_compute::TransposeError::OverlappingBuffers => NdFftError::AliasedBuffers,
                _ => NdFftError::BufferTooShort {
                    required: self.total_len,
                    actual: transpose_scratch.len(),
                },
            },
        )?;
        for col in 0..self.cols {
            self.col_plan.process_with_scratch(
                &mut transpose_scratch[col * self.rows..(col + 1) * self.rows],
                fft_scratch,
            );
        }
        transpose_2d(transpose_scratch, buffer, self.cols, self.rows).map_err(
            |error| match error {
                sonium_compute::TransposeError::LengthOverflow => NdFftError::LengthOverflow,
                sonium_compute::TransposeError::OverlappingBuffers => NdFftError::AliasedBuffers,
                _ => NdFftError::BufferTooShort {
                    required: self.total_len,
                    actual: buffer.len(),
                },
            },
        )?;
        Ok(())
    }
}

/// Row-major three-dimensional FFT plan with dimensions `(depth, rows, cols)`.
pub struct Fft3dPlan<T: FftScalar> {
    depth: usize,
    rows: usize,
    cols: usize,
    plane_len: usize,
    total_len: usize,
    direction: FftDirection,
    depth_plan: Arc<dyn FftPlan<T>>,
    row_plan: Arc<dyn FftPlan<T>>,
    col_plan: Arc<dyn FftPlan<T>>,
}

impl<T: FftScalar> Fft3dPlan<T> {
    pub fn new(depth: usize, rows: usize, cols: usize, direction: FftDirection) -> Self {
        Self::try_new(depth, rows, cols, direction).expect("invalid 3D FFT dimensions")
    }

    pub fn try_new(
        depth: usize,
        rows: usize,
        cols: usize,
        direction: FftDirection,
    ) -> Result<Self, NdFftError> {
        let planner = CpuFftPlanner::<T>::new();
        Self::try_with_planner(depth, rows, cols, direction, &planner)
    }

    pub fn with_planner(
        depth: usize,
        rows: usize,
        cols: usize,
        direction: FftDirection,
        planner: &CpuFftPlanner<T>,
    ) -> Self {
        Self::try_with_planner(depth, rows, cols, direction, planner)
            .expect("invalid 3D FFT dimensions")
    }

    pub fn try_with_planner(
        depth: usize,
        rows: usize,
        cols: usize,
        direction: FftDirection,
        planner: &CpuFftPlanner<T>,
    ) -> Result<Self, NdFftError> {
        if depth == 0 || rows == 0 || cols == 0 {
            return Err(NdFftError::InvalidDimension);
        }
        let plane_len = rows.checked_mul(cols).ok_or(NdFftError::LengthOverflow)?;
        let total_len = depth
            .checked_mul(plane_len)
            .ok_or(NdFftError::LengthOverflow)?;
        Ok(Self {
            depth,
            rows,
            cols,
            plane_len,
            total_len,
            direction,
            depth_plan: planner.try_plan(depth, direction)?,
            row_plan: planner.try_plan(rows, direction)?,
            col_plan: planner.try_plan(cols, direction)?,
        })
    }

    pub fn dims(&self) -> (usize, usize, usize) {
        (self.depth, self.rows, self.cols)
    }

    pub fn direction(&self) -> FftDirection {
        self.direction
    }

    pub fn scratch_len(&self) -> usize {
        self.try_scratch_len()
            .expect("3D FFT scratch size overflow")
    }

    pub fn try_scratch_len(&self) -> Result<usize, NdFftError> {
        self.depth
            .max(self.rows)
            .max(self.cols)
            .checked_add(
                self.depth_plan
                    .scratch_len()
                    .max(self.row_plan.scratch_len())
                    .max(self.col_plan.scratch_len()),
            )
            .ok_or(NdFftError::LengthOverflow)
    }

    pub fn process(&self, buffer: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::new(T::zero(), T::zero()); self.scratch_len()];
        self.process_with_scratch(buffer, &mut scratch);
    }

    pub fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        self.try_process_with_scratch(buffer, scratch)
            .expect("invalid 3D FFT execution buffers");
    }

    pub fn try_process_with_scratch(
        &self,
        buffer: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), NdFftError> {
        let plane = self.plane_len;
        if buffer.len() != self.total_len {
            return Err(NdFftError::BufferTooShort {
                required: self.total_len,
                actual: buffer.len(),
            });
        }
        let required_scratch = self.try_scratch_len()?;
        if scratch.len() < required_scratch {
            return Err(NdFftError::ScratchTooShort {
                required: required_scratch,
                actual: scratch.len(),
            });
        }

        let gather_len = self.depth.max(self.rows).max(self.cols);
        let (gather, fft_scratch) = scratch.split_at_mut(gather_len);

        // Fastest axis: contiguous rows.
        for row in buffer.chunks_exact_mut(self.cols) {
            self.col_plan.process_with_scratch(row, fft_scratch);
        }

        // Middle axis: gather one column from each row of each depth plane.
        for z in 0..self.depth {
            let plane_base = z * plane;
            for x in 0..self.cols {
                for y in 0..self.rows {
                    gather[y] = buffer[plane_base + y * self.cols + x];
                }
                self.row_plan
                    .process_with_scratch(&mut gather[..self.rows], fft_scratch);
                for y in 0..self.rows {
                    buffer[plane_base + y * self.cols + x] = gather[y];
                }
            }
        }

        // Slowest axis: gather depth values at each in-plane coordinate.
        for y in 0..self.rows {
            for x in 0..self.cols {
                for z in 0..self.depth {
                    gather[z] = buffer[z * plane + y * self.cols + x];
                }
                self.depth_plan
                    .process_with_scratch(&mut gather[..self.depth], fft_scratch);
                for z in 0..self.depth {
                    buffer[z * plane + y * self.cols + x] = gather[z];
                }
            }
        }
        Ok(())
    }
}

/// Return a plan for the adjoint of a one-dimensional transform.
///
/// With the unnormalised DFT convention, the adjoint of a forward transform
/// is the unnormalised inverse transform, and vice versa. No extra `1/N` is
/// applied here; callers needing a unitary adjoint can scale separately.
pub fn plan_adjoint<T: FftScalar>(
    planner: &CpuFftPlanner<T>,
    n: usize,
    direction: FftDirection,
) -> Arc<dyn FftPlan<T>> {
    planner.plan(n, direction.opposite())
}

/// Apply the linear FFT derivative rule to a tangent vector.
///
/// The DFT is linear, therefore its Jacobian-vector product is the same FFT
/// operator applied to the tangent. This named helper makes that rule
/// explicit for optimization and adjoint call sites.
pub fn apply_derivative<T: FftScalar>(plan: &dyn FftPlan<T>, tangent: &mut [Complex<T>]) {
    plan.process(tangent);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    #[test]
    fn checked_nd_boundaries_reject_invalid_dimensions_and_buffers() {
        let planner = CpuFftPlanner::<f64>::new();
        assert!(matches!(
            Fft2dPlan::try_with_planner(0, 4, FftDirection::Forward, &planner),
            Err(NdFftError::InvalidDimension)
        ));
        let plan = Fft2dPlan::try_with_planner(2, 3, FftDirection::Forward, &planner)
            .expect("valid 2D plan");
        let mut short_buffer = vec![Complex64::new(0.0, 0.0); 5];
        let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];
        assert!(matches!(
            plan.try_process_with_scratch(&mut short_buffer, &mut scratch),
            Err(NdFftError::BufferTooShort { .. })
        ));
        let mut buffer = vec![Complex64::new(0.0, 0.0); 6];
        let mut short_scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len() - 1];
        assert!(matches!(
            plan.try_process_with_scratch(&mut buffer, &mut short_scratch),
            Err(NdFftError::ScratchTooShort { .. })
        ));

        let mut strided_buffer = vec![Complex64::new(0.0, 0.0); 6];
        assert!(matches!(
            try_process_strided(
                plan.row_plan.as_ref(),
                &mut strided_buffer,
                2,
                2,
                &mut scratch,
            ),
            Err(NdFftError::InvalidDimension)
        ));
    }

    fn assert_round_trip_2d(rows: usize, cols: usize) {
        let planner = CpuFftPlanner::<f64>::new();
        let forward = Fft2dPlan::with_planner(rows, cols, FftDirection::Forward, &planner);
        let inverse = Fft2dPlan::with_planner(rows, cols, FftDirection::Inverse, &planner);
        let input: Vec<Complex64> = (0..rows * cols)
            .map(|i| Complex64::new((i as f64 * 0.2).sin(), (i as f64 * 0.07).cos()))
            .collect();
        let mut actual = input.clone();
        let mut scratch = vec![Complex64::new(0.0, 0.0); forward.scratch_len()];
        forward.process_with_scratch(&mut actual, &mut scratch);
        inverse.process_with_scratch(&mut actual, &mut scratch);
        for (got, want) in actual.iter().zip(input.iter()) {
            assert_relative_eq!(got.re / (rows * cols) as f64, want.re, epsilon = 1e-10);
            assert_relative_eq!(got.im / (rows * cols) as f64, want.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn two_and_three_dimensional_round_trips() {
        assert_round_trip_2d(3, 5);
        // A prime axis forces Bluestein inside the multidimensional plan and
        // verifies that the advertised scratch region is forwarded to it.
        assert_round_trip_2d(7, 5);
        let dims = (2, 3, 4);
        let planner = CpuFftPlanner::<f64>::new();
        let forward =
            Fft3dPlan::with_planner(dims.0, dims.1, dims.2, FftDirection::Forward, &planner);
        let inverse =
            Fft3dPlan::with_planner(dims.0, dims.1, dims.2, FftDirection::Inverse, &planner);
        let n = dims.0 * dims.1 * dims.2;
        let input: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(i as f64 * 0.1, (i as f64).cos()))
            .collect();
        let mut actual = input.clone();
        let mut scratch = vec![Complex64::new(0.0, 0.0); forward.scratch_len()];
        forward.process_with_scratch(&mut actual, &mut scratch);
        inverse.process_with_scratch(&mut actual, &mut scratch);
        for (got, want) in actual.iter().zip(input.iter()) {
            assert_relative_eq!(got.re / n as f64, want.re, epsilon = 1e-10);
            assert_relative_eq!(got.im / n as f64, want.im, epsilon = 1e-10);
        }

        let prime_dims = (7, 2, 2);
        let forward = Fft3dPlan::with_planner(
            prime_dims.0,
            prime_dims.1,
            prime_dims.2,
            FftDirection::Forward,
            &planner,
        );
        let inverse = Fft3dPlan::with_planner(
            prime_dims.0,
            prime_dims.1,
            prime_dims.2,
            FftDirection::Inverse,
            &planner,
        );
        let n = prime_dims.0 * prime_dims.1 * prime_dims.2;
        let input: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.11).sin(), (i as f64 * 0.17).cos()))
            .collect();
        let mut actual = input.clone();
        let mut scratch = vec![Complex64::new(0.0, 0.0); forward.scratch_len()];
        forward.process_with_scratch(&mut actual, &mut scratch);
        inverse.process_with_scratch(&mut actual, &mut scratch);
        for (got, want) in actual.iter().zip(input.iter()) {
            assert_relative_eq!(got.re / n as f64, want.re, epsilon = 1e-10);
            assert_relative_eq!(got.im / n as f64, want.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn forward_and_adjoint_plans_satisfy_inner_product_identity() {
        let n = 7;
        let planner = CpuFftPlanner::<f64>::new();
        let forward = planner.plan(n, FftDirection::Forward);
        let adjoint = plan_adjoint(&planner, n, FftDirection::Forward);
        let x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.17).sin(), (i as f64 * 0.23).cos()))
            .collect();
        let y: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.31).cos(), (i as f64 * 0.07).sin()))
            .collect();

        let mut forward_x = x.clone();
        forward.process(&mut forward_x);
        let mut adjoint_y = y.clone();
        adjoint.process(&mut adjoint_y);

        let lhs: Complex64 = forward_x.iter().zip(&y).map(|(fx, y)| fx.conj() * y).sum();
        let rhs: Complex64 = x.iter().zip(&adjoint_y).map(|(x, ay)| x.conj() * ay).sum();
        assert_relative_eq!(lhs.re, rhs.re, epsilon = 1e-9);
        assert_relative_eq!(lhs.im, rhs.im, epsilon = 1e-9);
    }
}
