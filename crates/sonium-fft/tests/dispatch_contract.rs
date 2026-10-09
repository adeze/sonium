//! Independent scalar/AVX2 dispatch and transform-identity coverage.

use num_complex::Complex;
use sonium_fft::cpu::planner::CpuFftPlanner;
use sonium_fft::{CpuFftDispatch, Fft2dPlan, FftDirection, LinearConvolutionPlan, process_strided};

fn input<T: sonium_fft::FftScalar>(len: usize, seed: u64) -> Vec<Complex<T>> {
    let mut state = 0xD1_5A_7C_u64 ^ seed ^ len as u64;
    (0..len)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let re = (state as f64 / u64::MAX as f64) - 0.5 + index as f64 * 1.0e-7;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let im = (state as f64 / u64::MAX as f64) - 0.5;
            Complex::new(
                T::from_f64(re).expect("test value is representable"),
                T::from_f64(im).expect("test value is representable"),
            )
        })
        .collect()
}

fn dft<T: sonium_fft::FftScalar>(input: &[Complex<T>], direction: FftDirection) -> Vec<Complex<T>> {
    let n = input.len();
    let sign = match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };
    (0..n)
        .map(|k| {
            input
                .iter()
                .enumerate()
                .map(|(j, &value)| {
                    let angle = sign * std::f64::consts::TAU * (j * k) as f64 / n as f64;
                    value
                        * Complex::new(
                            T::from_f64(angle.cos()).expect("test cosine is representable"),
                            T::from_f64(angle.sin()).expect("test sine is representable"),
                        )
                })
                .sum()
        })
        .collect()
}

fn max_error<T: sonium_fft::FftScalar>(left: &[Complex<T>], right: &[Complex<T>]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let re = (left.re - right.re)
                .abs()
                .to_f64()
                .expect("test scalar converts to f64");
            let im = (left.im - right.im)
                .abs()
                .to_f64()
                .expect("test scalar converts to f64");
            re.max(im)
        })
        .fold(0.0, f64::max)
}

fn inner_product<T: sonium_fft::FftScalar>(
    left: &[Complex<T>],
    right: &[Complex<T>],
) -> Complex<T> {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.conj() * right)
        .sum()
}

fn run_transform_case<T: sonium_fft::FftScalar>(
    dispatch: CpuFftDispatch,
    n: usize,
    batch: usize,
    stride: Option<usize>,
    direction: FftDirection,
) -> (f64, f64, usize) {
    let buffer_len = stride.map_or(n * batch, |stride| (batch - 1) * stride + n);
    let original = input::<T>(buffer_len, n as u64 ^ batch as u64);
    let planner = CpuFftPlanner::<T>::with_dispatch(dispatch);
    let plan = planner.plan(n, direction);
    let mut actual = original.clone();
    let mut scratch = vec![Complex::new(T::zero(), T::zero()); n + plan.scratch_len()];
    if let Some(stride) = stride {
        process_strided(&*plan, &mut actual, batch, stride, &mut scratch);
    } else {
        plan.process_with_scratch(&mut actual, &mut scratch[n..]);
    }

    let mut expected = original.clone();
    if let Some(stride) = stride {
        for batch_index in 0..batch {
            let start = batch_index * stride;
            expected[start..start + n]
                .copy_from_slice(&dft(&original[start..start + n], direction));
        }
    } else {
        for batch_index in 0..batch {
            let start = batch_index * n;
            expected[start..start + n]
                .copy_from_slice(&dft(&original[start..start + n], direction));
        }
    }
    let reference_error = max_error(&actual, &expected);

    let comparison_planner =
        CpuFftPlanner::<T>::with_dispatch(if dispatch == CpuFftDispatch::Scalar {
            CpuFftDispatch::Avx2
        } else {
            CpuFftDispatch::Scalar
        });
    let comparison_plan = comparison_planner.plan(n, direction);
    let mut comparison = original.clone();
    let mut comparison_scratch =
        vec![Complex::new(T::zero(), T::zero()); n + comparison_plan.scratch_len()];
    if let Some(stride) = stride {
        process_strided(
            &*comparison_plan,
            &mut comparison,
            batch,
            stride,
            &mut comparison_scratch,
        );
    } else {
        comparison_plan.process_with_scratch(&mut comparison, &mut comparison_scratch[n..]);
    }

    let dispatch_error = max_error(&actual, &comparison);
    let padding_error = stride.map_or(0.0, |stride| {
        (0..batch)
            .flat_map(|batch_index| {
                let start = batch_index * stride + n;
                let end = buffer_len.min((batch_index + 1) * stride);
                actual[start..end]
                    .iter()
                    .zip(&original[start..end])
                    .map(|(actual, original)| (*actual - *original).norm().to_f64().unwrap())
            })
            .fold(0.0, f64::max)
    });
    (
        reference_error.max(padding_error),
        dispatch_error,
        plan.scratch_len(),
    )
}

fn tolerance<T: sonium_fft::FftScalar>(n: usize) -> f64 {
    if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
        8.0e-4 * (n as f64).sqrt().max(1.0)
    } else {
        3.0e-10 * (n as f64).sqrt().max(1.0)
    }
}

#[test]
fn scalar_and_avx2_cover_all_transform_shapes_and_directions() {
    for &dispatch in &[CpuFftDispatch::Scalar, CpuFftDispatch::Avx2] {
        for &direction in &[FftDirection::Forward, FftDirection::Inverse] {
            for &(n, batch, stride) in &[
                (1usize, 1usize, None),
                (2, 3, None),
                (3, 2, None),
                (16, 3, None),
                (45, 2, None),
                (63, 3, None),
                (257, 1, None),
                (15, 3, Some(19)),
            ] {
                let (f64_reference, f64_dispatch, scratch) =
                    run_transform_case::<f64>(dispatch, n, batch, stride, direction);
                assert!(
                    f64_reference <= tolerance::<f64>(n),
                    "f64 n={n} {direction:?}"
                );
                assert!(f64_dispatch <= tolerance::<f64>(n), "f64 dispatch n={n}");
                assert!(scratch >= n);

                let (f32_reference, f32_dispatch, _) =
                    run_transform_case::<f32>(dispatch, n, batch, stride, direction);
                assert!(
                    f32_reference <= tolerance::<f32>(n),
                    "f32 n={n} {direction:?}"
                );
                assert!(f32_dispatch <= tolerance::<f32>(n), "f32 dispatch n={n}");
            }
        }

        // Empty buffers are a documented no-op even though plans themselves
        // have positive lengths.
        let planner = CpuFftPlanner::<f64>::with_dispatch(dispatch);
        let plan = planner.plan(8, FftDirection::Forward);
        let mut empty = Vec::new();
        plan.process(&mut empty);
    }
}

#[test]
fn selected_paths_satisfy_round_trip_parseval_and_hermitian_adjoint_identities() {
    for &dispatch in &[CpuFftDispatch::Scalar, CpuFftDispatch::Avx2] {
        let planner = CpuFftPlanner::<f64>::with_dispatch(dispatch);
        let pair = planner.plan_with_adjoint(63, FftDirection::Forward);
        let original = input::<f64>(63, 0xA11D);
        let mut transformed = original.clone();
        let mut scratch = vec![Complex::new(0.0, 0.0); pair.scratch_len()];
        pair.try_process_with_scratch(&mut transformed, &mut scratch)
            .unwrap();
        let mut round_trip = transformed.clone();
        pair.try_process_adjoint_with_scratch(&mut round_trip, &mut scratch)
            .unwrap();
        assert!(
            max_error(
                &round_trip
                    .iter()
                    .map(|value| *value / 63.0)
                    .collect::<Vec<_>>(),
                &original,
            ) < tolerance::<f64>(63)
        );

        let input_energy: f64 = original.iter().map(|value| value.norm_sqr()).sum();
        let output_energy: f64 = transformed.iter().map(|value| value.norm_sqr()).sum();
        assert!((output_energy - 63.0 * input_energy).abs() < 1.0e-8 * input_energy);

        let y = input::<f64>(63, 0xAD10);
        let left = inner_product(&transformed, &y);
        let mut adjoint_y = y.clone();
        pair.process_adjoint_with_scratch(&mut adjoint_y, &mut scratch);
        let right = inner_product(&original, &adjoint_y);
        assert!((left - right).norm() < 1.0e-8);
    }
}

fn direct_linear_convolution(left: &[Complex<f64>], right: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let mut output = vec![Complex::new(0.0, 0.0); left.len() + right.len() - 1];
    for (i, &left_value) in left.iter().enumerate() {
        for (j, &right_value) in right.iter().enumerate() {
            output[i + j] += left_value * right_value;
        }
    }
    output
}

#[test]
fn avx2_selected_convolution_and_multidimensional_paths_match_references() {
    for &dispatch in &[CpuFftDispatch::Scalar, CpuFftDispatch::Avx2] {
        let left = input::<f64>(17, 1);
        let right = input::<f64>(11, 2);
        let plan = LinearConvolutionPlan::<f64>::new_with_dispatch(17, 11, dispatch);
        let mut output = vec![Complex::new(0.0, 0.0); plan.output_len()];
        let mut scratch = vec![Complex::new(0.0, 0.0); plan.scratch_len()];
        plan.try_process_with_scratch(&left, &right, &mut output, &mut scratch)
            .unwrap();
        assert!(max_error(&output, &direct_linear_convolution(&left, &right)) < 2.0e-9);

        let planner = CpuFftPlanner::<f64>::with_dispatch(dispatch);
        let plan_2d = Fft2dPlan::with_planner(5, 7, FftDirection::Forward, &planner);
        let mut buffer = input::<f64>(35, 3);
        let reference = buffer.clone();
        let mut scratch = vec![Complex::new(0.0, 0.0); plan_2d.scratch_len()];
        plan_2d.process_with_scratch(&mut buffer, &mut scratch);
        assert!(
            buffer
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
        assert_ne!(buffer, reference, "non-trivial 2D path was not exercised");
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_request_has_a_tested_scalar_fallback_when_features_are_absent() {
    let planner = CpuFftPlanner::<f64>::with_dispatch(CpuFftDispatch::Avx2);
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        assert_eq!(planner.dispatch(), CpuFftDispatch::Avx2);
    } else {
        assert_eq!(planner.dispatch(), CpuFftDispatch::Scalar);
    }
}
