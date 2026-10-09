//! Cross-family contracts for the public CPU transform APIs.
//!
//! The one-dimensional complex tests have a separate high-precision oracle.
//! These tests cover the public real, cosine/sine, multidimensional, strided,
//! and STFT surfaces so their normalization, layout, scratch, and boundary
//! promises are checked independently of the implementation's internal tests.

use num_complex::Complex64;
use sonium_fft::{
    DctPlan, DstPlan, Fft2dPlan, Fft3dPlan, FftDirection, FftPlanner, RealFftPlan, StftConfig,
    StftPlan, WindowFunction, apply_derivative, dct2, dct3, dst2, dst3, process_strided,
};
use std::f64::consts::{PI, TAU};

fn assert_complex_close(actual: Complex64, expected: Complex64, tolerance: f64, label: &str) {
    let scale = actual.norm().max(expected.norm()).max(1.0);
    assert!(
        (actual - expected).norm() <= tolerance * scale,
        "{label}: actual={actual:?}, expected={expected:?}, tolerance={tolerance:e}"
    );
}

fn direct_dft_2d(
    input: &[Complex64],
    rows: usize,
    cols: usize,
    direction: FftDirection,
) -> Vec<Complex64> {
    let sign = match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };
    let mut output = vec![Complex64::new(0.0, 0.0); rows * cols];
    for kr in 0..rows {
        for kc in 0..cols {
            let mut sum = Complex64::new(0.0, 0.0);
            for r in 0..rows {
                for c in 0..cols {
                    let phase = sign
                        * TAU
                        * (r as f64 * kr as f64 / rows as f64 + c as f64 * kc as f64 / cols as f64);
                    let (sin, cos) = phase.sin_cos();
                    sum += input[r * cols + c] * Complex64::new(cos, sin);
                }
            }
            output[kr * cols + kc] = sum;
        }
    }
    output
}

fn direct_dft_3d(
    input: &[Complex64],
    depth: usize,
    rows: usize,
    cols: usize,
    direction: FftDirection,
) -> Vec<Complex64> {
    let sign = match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };
    let mut output = vec![Complex64::new(0.0, 0.0); depth * rows * cols];
    for kz in 0..depth {
        for ky in 0..rows {
            for kx in 0..cols {
                let mut sum = Complex64::new(0.0, 0.0);
                for z in 0..depth {
                    for y in 0..rows {
                        for x in 0..cols {
                            let phase = sign
                                * TAU
                                * (z as f64 * kz as f64 / depth as f64
                                    + y as f64 * ky as f64 / rows as f64
                                    + x as f64 * kx as f64 / cols as f64);
                            let (sin, cos) = phase.sin_cos();
                            let index = (z * rows + y) * cols + x;
                            sum += input[index] * Complex64::new(cos, sin);
                        }
                    }
                }
                output[(kz * rows + ky) * cols + kx] = sum;
            }
        }
    }
    output
}

fn direct_dct2(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    (0..n)
        .map(|k| {
            input
                .iter()
                .enumerate()
                .map(|(index, &sample)| {
                    sample * (PI * (index as f64 + 0.5) * k as f64 / n as f64).cos()
                })
                .sum()
        })
        .collect()
}

fn direct_dct3(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    (0..n)
        .map(|index| {
            let sum = input[0] * 0.5
                + input
                    .iter()
                    .enumerate()
                    .skip(1)
                    .map(|(k, &value)| {
                        value * (PI * k as f64 * (index as f64 + 0.5) / n as f64).cos()
                    })
                    .sum::<f64>();
            sum * (2.0 / n as f64)
        })
        .collect()
}

fn direct_dst2(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    (0..n)
        .map(|k| {
            input
                .iter()
                .enumerate()
                .map(|(index, &sample)| {
                    sample * (PI * (index as f64 + 0.5) * (k as f64 + 1.0) / n as f64).sin()
                })
                .sum()
        })
        .collect()
}

fn direct_dst3(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    (0..n)
        .map(|index| {
            input
                .iter()
                .enumerate()
                .map(|(k, &value)| {
                    let scale = if k + 1 == n {
                        1.0 / n as f64
                    } else {
                        2.0 / n as f64
                    };
                    value * (PI * (index as f64 + 0.5) * (k as f64 + 1.0) / n as f64).sin() * scale
                })
                .sum()
        })
        .collect()
}

fn direct_real_spectrum(frame: &[f64]) -> Vec<Complex64> {
    let n = frame.len();
    (0..=n / 2)
        .map(|k| {
            frame
                .iter()
                .enumerate()
                .map(|(index, &sample)| {
                    let phase = -TAU * index as f64 * k as f64 / n as f64;
                    let (sin, cos) = phase.sin_cos();
                    Complex64::new(sample * cos, sample * sin)
                })
                .sum()
        })
        .collect()
}

fn inner_product(lhs: &[Complex64], rhs: &[Complex64]) -> Complex64 {
    lhs.iter().zip(rhs).map(|(a, b)| a.conj() * b).sum()
}

#[test]
fn complex_fft_properties_cover_linearity_impulse_constant_and_tone() {
    for &n in &[3usize, 8, 15] {
        let planner = FftPlanner::<f64>::new();
        let plan = planner.plan_forward(n);
        let x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.13).sin(), (i as f64 * 0.19).cos()))
            .collect();
        let y: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.23).cos(), -(i as f64 * 0.07).sin()))
            .collect();
        let mut fx = x.clone();
        let mut fy = y.clone();
        plan.process(&mut fx);
        plan.process(&mut fy);
        let mut combined: Vec<Complex64> = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| *x * 1.7 + *y * Complex64::new(-0.4, 0.8))
            .collect();
        plan.process(&mut combined);
        for i in 0..n {
            assert_complex_close(
                combined[i],
                fx[i] * 1.7 + fy[i] * Complex64::new(-0.4, 0.8),
                2.0e-11,
                "complex linearity",
            );
        }

        let mut impulse = vec![Complex64::new(0.0, 0.0); n];
        impulse[0] = Complex64::new(1.0, 0.0);
        plan.process(&mut impulse);
        assert!(impulse.iter().all(|value| (value.re - 1.0).abs() < 1.0e-12));

        let mut constant = vec![Complex64::new(1.0, 0.0); n];
        plan.process(&mut constant);
        assert!((constant[0].re - n as f64).abs() < 1.0e-12);
        assert!(constant[1..].iter().all(|value| value.norm() < 1.0e-11));

        let tone_bin = n / 3;
        let mut tone: Vec<Complex64> = (0..n)
            .map(|i| {
                let phase = TAU * tone_bin as f64 * i as f64 / n as f64;
                let (sin, cos) = phase.sin_cos();
                Complex64::new(cos, sin)
            })
            .collect();
        plan.process(&mut tone);
        assert!((tone[tone_bin].re - n as f64).abs() < 1.0e-10);
        assert!(
            tone.iter()
                .enumerate()
                .filter(|(index, _)| *index != tone_bin)
                .all(|(_, value)| value.norm() < 1.0e-10)
        );
    }
}

#[test]
fn real_fft_round_trip_and_parseval_hold_for_even_and_odd_lengths() {
    for &n in &[1usize, 2, 3, 8, 9, 16, 31] {
        let plan = RealFftPlan::<f64>::new(n);
        let input: Vec<f64> = (0..n)
            .map(|i| (0.13 * i as f64).sin() + 0.2 * (0.07 * i as f64).cos())
            .collect();
        let spectrum = plan.forward(&input);
        let recovered = plan.inverse(&spectrum);
        for (i, (&actual, &expected)) in recovered.iter().zip(&input).enumerate() {
            assert!((actual - expected).abs() < 2.0e-10, "n={n}, i={i}");
        }

        if n > 1 {
            let mut spectral_energy = spectrum[0].norm_sqr();
            let interior_end = if n.is_multiple_of(2) {
                n / 2
            } else {
                n / 2 + 1
            };
            for value in &spectrum[1..interior_end] {
                spectral_energy += 2.0 * value.norm_sqr();
            }
            if n.is_multiple_of(2) {
                spectral_energy += spectrum[n / 2].norm_sqr();
            }
            let time_energy: f64 = input.iter().map(|value| value * value).sum();
            assert!((spectral_energy / n as f64 - time_energy).abs() < 2.0e-9);
        }
    }
}

#[test]
fn cosine_and_sine_transforms_have_declared_inverse_scaling() {
    for &n in &[0usize, 1, 2, 7, 16] {
        let input: Vec<f64> = (0..n).map(|i| (0.17 * i as f64).sin()).collect();
        let mut transformed = vec![0.0; n];
        let mut recovered = vec![0.0; n];
        dct2(&input, &mut transformed);
        dct3(&transformed, &mut recovered);
        for (actual, expected) in recovered.iter().zip(&input) {
            assert!((actual - expected).abs() < 2.0e-10);
        }
        dst2(&input, &mut transformed);
        dst3(&transformed, &mut recovered);
        for (actual, expected) in recovered.iter().zip(&input) {
            assert!((actual - expected).abs() < 2.0e-10);
        }
    }

    let n = 8;
    let constant = vec![1.0; n];
    let mut dct = vec![0.0; n];
    dct2(&constant, &mut dct);
    assert!((dct[0] - n as f64).abs() < 1.0e-12);
    assert!(dct[1..].iter().all(|value| value.abs() < 1.0e-12));

    let dct_plan = DctPlan::<f64>::new(n);
    let dst_plan = DstPlan::<f64>::new(n);
    let mut dct_plan_output = vec![0.0; n];
    let mut dst_plan_output = vec![0.0; n];
    dct_plan.forward(&constant, &mut dct_plan_output);
    dst_plan.forward(&constant, &mut dst_plan_output);
    assert_eq!(dct_plan.len(), n);
    assert_eq!(dst_plan.len(), n);
    assert!((dct_plan_output[0] - n as f64).abs() < 1.0e-12);
    assert!(dst_plan_output.iter().all(|value| value.is_finite()));
}

#[test]
fn cosine_and_sine_transforms_match_independent_references() {
    let input: Vec<f64> = (0..7)
        .map(|i| (0.17 * i as f64).sin() - 0.13 * (i as f64).cos())
        .collect();
    let mut actual = vec![0.0; input.len()];

    dct2(&input, &mut actual);
    for (index, (&got, &expected)) in actual.iter().zip(direct_dct2(&input).iter()).enumerate() {
        assert!((got - expected).abs() < 2.0e-12, "DCT-II index {index}");
    }
    dct3(&input, &mut actual);
    for (index, (&got, &expected)) in actual.iter().zip(direct_dct3(&input).iter()).enumerate() {
        assert!((got - expected).abs() < 2.0e-12, "DCT-III index {index}");
    }
    dst2(&input, &mut actual);
    for (index, (&got, &expected)) in actual.iter().zip(direct_dst2(&input).iter()).enumerate() {
        assert!((got - expected).abs() < 2.0e-12, "DST-II index {index}");
    }
    dst3(&input, &mut actual);
    for (index, (&got, &expected)) in actual.iter().zip(direct_dst3(&input).iter()).enumerate() {
        assert!((got - expected).abs() < 2.0e-12, "DST-III index {index}");
    }
}

#[test]
fn multidimensional_layout_matches_separable_direct_dft() {
    let rows = 3;
    let cols = 4;
    let input: Vec<Complex64> = (0..rows * cols)
        .map(|i| Complex64::new((i as f64 * 0.11).sin(), (i as f64 * 0.07).cos()))
        .collect();
    let expected = direct_dft_2d(&input, rows, cols, FftDirection::Forward);
    let plan = Fft2dPlan::<f64>::new(rows, cols, FftDirection::Forward);
    let mut actual = input.clone();
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];
    plan.process_with_scratch(&mut actual, &mut scratch);
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert_complex_close(got, want, 2.0e-11, &format!("2D index {index}"));
    }

    let dims = (2, 3, 4);
    let n = dims.0 * dims.1 * dims.2;
    let input_3d: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(i as f64 * 0.03, (i as f64 * 0.19).cos()))
        .collect();
    let forward = Fft3dPlan::<f64>::new(dims.0, dims.1, dims.2, FftDirection::Forward);
    let inverse = Fft3dPlan::<f64>::new(dims.0, dims.1, dims.2, FftDirection::Inverse);
    let mut round_trip = input_3d.clone();
    let mut scratch = vec![Complex64::new(0.0, 0.0); forward.scratch_len()];
    forward.process_with_scratch(&mut round_trip, &mut scratch);
    let direct = direct_dft_3d(&input_3d, dims.0, dims.1, dims.2, FftDirection::Forward);
    let mut direct_actual = input_3d.clone();
    forward.process_with_scratch(&mut direct_actual, &mut scratch);
    for (actual, expected) in direct_actual.iter().zip(&direct) {
        assert_complex_close(*actual, *expected, 2.0e-10, "3D direct reference");
    }
    inverse.process_with_scratch(&mut round_trip, &mut scratch);
    for (actual, expected) in round_trip.iter().zip(&input_3d) {
        assert_complex_close(*actual / n as f64, *expected, 2.0e-10, "3D round trip");
    }
}

#[test]
fn strided_and_adjoint_contracts_hold() {
    let n = 7;
    let batch = 3;
    let stride = 8;
    let planner = FftPlanner::<f64>::new();
    let plan = planner.plan_forward(n);
    let mut buffer = vec![Complex64::new(-7.0, 3.0); (batch - 1) * stride + n];
    let original: Vec<Vec<Complex64>> = (0..batch)
        .map(|b| {
            (0..n)
                .map(|i| Complex64::new((b * n + i) as f64 * 0.13, -(i as f64) * 0.07))
                .collect()
        })
        .collect();
    for (b, values) in original.iter().enumerate() {
        buffer[b * stride..b * stride + n].copy_from_slice(values);
    }
    let mut scratch = vec![Complex64::new(0.0, 0.0); n + plan.scratch_len()];
    process_strided(&*plan, &mut buffer, batch, stride, &mut scratch);
    for (b, values) in original.iter().enumerate() {
        let mut expected = values.clone();
        let direct = planner.plan_forward(n);
        direct.process(&mut expected);
        for (actual, expected) in buffer[b * stride..b * stride + n].iter().zip(expected) {
            assert_complex_close(*actual, expected, 2.0e-12, "strided transform");
        }
        if b + 1 < batch {
            assert_eq!(buffer[b * stride + n], Complex64::new(-7.0, 3.0));
        }
    }

    let x: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 * 0.4).sin(), i as f64 * 0.1))
        .collect();
    let y: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 * 0.2).cos(), -(i as f64) * 0.05))
        .collect();
    let mut fx = x.clone();
    let mut inverse_y = y.clone();
    plan.process(&mut fx);
    planner.plan_inverse(n).process(&mut inverse_y);
    assert_complex_close(
        inner_product(&fx, &y),
        inner_product(&x, &inverse_y),
        2.0e-10,
        "FFT adjoint",
    );

    let mut tangent = x.clone();
    let mut expected_tangent = x;
    plan.process(&mut expected_tangent);
    apply_derivative(&*plan, &mut tangent);
    assert_eq!(tangent, expected_tangent);
}

#[test]
fn stft_reconstructs_all_published_windows_with_centered_padding() {
    let input: Vec<f64> = (0..257)
        .map(|i| (i as f64 * 0.07).sin() + 0.2 * (i as f64 * 0.013).cos())
        .collect();
    for window in [
        WindowFunction::Rectangular,
        WindowFunction::Hann,
        WindowFunction::Hamming,
        WindowFunction::Blackman,
    ] {
        let mut config = StftConfig::new(32, 8, window);
        config.center = true;
        let plan = StftPlan::<f64>::new(config);
        let spectra = plan.analyze(&input);
        let output = plan.synthesize(&spectra, input.len());
        for (index, (&actual, &expected)) in output.iter().zip(&input).enumerate() {
            assert!(
                (actual - expected).abs() < 2.0e-9,
                "window={window:?}, index={index}, actual={actual:e}, expected={expected:e}"
            );
        }
    }
}

#[test]
fn rectangular_stft_matches_independent_frame_dfts() {
    let mut config = StftConfig::new(4, 2, WindowFunction::Rectangular);
    config.center = false;
    config.pad_end = false;
    let plan = StftPlan::<f64>::new(config);
    let input: Vec<f64> = (0..6).map(|i| (0.21 * i as f64).sin()).collect();
    let spectra = plan.analyze(&input);

    assert_eq!(plan.frame_count(input.len()), 2);
    assert_eq!(spectra.len(), 2 * plan.spectrum_len());
    for frame_index in 0..2 {
        let expected = direct_real_spectrum(&input[frame_index * 2..frame_index * 2 + 4]);
        let actual =
            &spectra[frame_index * plan.spectrum_len()..(frame_index + 1) * plan.spectrum_len()];
        for (bin, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            assert_complex_close(
                got,
                want,
                2.0e-12,
                &format!("STFT frame={frame_index}, bin={bin}"),
            );
        }
    }
}

#[test]
#[should_panic(expected = "real FFT scratch is too short")]
fn real_fft_rejects_short_scratch() {
    let plan = RealFftPlan::<f64>::new(8);
    let input = vec![0.0; 8];
    let mut output = vec![Complex64::new(0.0, 0.0); plan.spectrum_len()];
    let mut scratch = vec![];
    plan.forward_with_scratch(&input, &mut output, &mut scratch);
}
