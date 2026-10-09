//! Seeded property QA for checked sonium-compute boundary APIs.
//!
//! The default seed is printed in the replay string.  Set `SONIUM_QA_SEED`
//! to replay a failing generated case exactly.

use num_complex::Complex64;
use rand::{Rng, SeedableRng, rngs::StdRng};
use sonium_compute::{
    Backend, BufferElement, BufferView, CpuBackend, DeviceBufferLike, DeviceScalar, ReductionMode,
    ViewAliasing, pointwise_mul, scale_real, segmented_sum, transpose_2d,
};

const DEFAULT_SEED: u64 = 0xC0A4_2028_0805_0001;
const CASES: usize = 512;

fn seed() -> Result<u64, String> {
    match std::env::var("SONIUM_QA_SEED") {
        Ok(value) => value
            .parse()
            .map_err(|error| format!("invalid SONIUM_QA_SEED={value:?}: {error}")),
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_SEED),
        Err(error) => Err(format!("could not read SONIUM_QA_SEED: {error}")),
    }
}

fn expected_valid(buffer_len: usize, offset: usize, len: usize, stride: usize) -> bool {
    if stride == 0 || offset > buffer_len {
        return false;
    }
    len == 0
        || len
            .checked_sub(1)
            .and_then(|last| last.checked_mul(stride))
            .and_then(|last| offset.checked_add(last))
            .and_then(|last| last.checked_add(1))
            .is_some_and(|end| end <= buffer_len)
}

fn check_views(seed: u64) -> Result<(), String> {
    let backend = CpuBackend::new();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut buffer = backend
        .raw_alloc::<f64>(257)
        .map_err(|error| error.to_string())?;

    for case in 0..CASES {
        let offset = rng.random_range(0..=300);
        let len = match case % 8 {
            0 => 0,
            1 => 1,
            _ => rng.random_range(0..=260),
        };
        let stride = match case % 9 {
            0 => 0,
            1 => 1,
            _ => rng.random_range(1..=300),
        };
        let got = backend.checked_view(&buffer, offset, len, stride);
        if got.is_ok() != expected_valid(buffer.len(), offset, len, stride) {
            return Err(format!(
                "case {case}: checked view disagrees with reference: offset={offset}, len={len}, stride={stride}, got={got:?}"
            ));
        }
    }

    // Negative strides are intentionally unsupported by the unsigned public
    // contract.  Exercise their raw representation so future signed-view work
    // cannot accidentally reinterpret `-1` as a valid giant stride.
    let negative_stride = usize::MAX;
    if BufferView::new(buffer.len(), 8, 2, negative_stride).is_ok() {
        return Err("negative-stride representation was accepted".into());
    }

    let descriptor = backend
        .checked_view_descriptor_with_aliasing(&buffer, 1, 16, 3, ViewAliasing::OverlapSafe)
        .map_err(|error| error.to_string())?;
    descriptor
        .validate_for_buffer::<f64>(buffer.len(), backend.capabilities().backend_name)
        .map_err(|error| error.to_string())?;

    // Aliasing must retain memmove semantics, including overlapping views.
    let source: Vec<f64> = (0..buffer.len()).map(|i| i as f64 - 16.0).collect();
    backend
        .try_copy_h2d(&source, &mut buffer)
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_view_in_place(
            &mut buffer,
            BufferView::contiguous(0, 32),
            BufferView::contiguous(8, 32),
        )
        .map_err(|error| error.to_string())?;
    let mut copied = vec![0.0; buffer.len()];
    backend
        .try_copy_d2h(&buffer, &mut copied)
        .map_err(|error| error.to_string())?;
    if copied[8..40] != source[..32] {
        return Err("overlapping copy did not preserve memmove semantics".into());
    }
    Ok(())
}

fn check_numeric_edges() -> Result<(), String> {
    let backend = CpuBackend::new();
    let mut x = backend
        .raw_alloc::<f64>(1)
        .map_err(|error| error.to_string())?;
    let mut y = backend
        .raw_alloc::<f64>(1)
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&[1.0], &mut x)
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&[2.0], &mut y)
        .map_err(|error| error.to_string())?;
    if backend.try_axpy(f64::NAN, &x, &mut y).is_ok() {
        return Err("NaN alpha was accepted by checked axpy".into());
    }
    if backend.try_axpy(f64::INFINITY, &x, &mut y).is_ok() {
        return Err("infinite alpha was accepted by checked axpy".into());
    }
    if backend
        .try_dot_deterministic(&x, &y)
        .map_err(|error| error.to_string())?
        != 2.0
    {
        return Err("deterministic dot lost the scalar reference result".into());
    }
    Ok(())
}

fn check_scalar_descriptor<T: BufferElement>(
    backend: &CpuBackend,
    label: &str,
) -> Result<(), String> {
    let buffer = backend
        .raw_alloc::<T>(16)
        .map_err(|error| format!("{label}: {error}"))?;
    backend
        .checked_view_descriptor_with_aliasing(&buffer, 1, 5, 3, ViewAliasing::ReadOnly)
        .and_then(|descriptor| {
            descriptor.validate_for_buffer::<T>(buffer.len(), backend.capabilities().backend_name)
        })
        .map_err(|error| format!("{label}: {error}"))
}

fn check_all_scalar_descriptors() -> Result<(), String> {
    let backend = CpuBackend::new();
    check_scalar_descriptor::<f32>(&backend, "f32")?;
    check_scalar_descriptor::<f64>(&backend, "f64")?;
    check_scalar_descriptor::<num_complex::Complex32>(&backend, "complex32")?;
    check_scalar_descriptor::<num_complex::Complex64>(&backend, "complex64")?;
    check_scalar_descriptor::<half::f16>(&backend, "f16")?;
    check_scalar_descriptor::<half::bf16>(&backend, "bf16")?;
    check_scalar_descriptor::<i32>(&backend, "i32")?;
    check_scalar_descriptor::<u32>(&backend, "u32")?;
    check_scalar_descriptor::<usize>(&backend, "usize")
}

fn check_transpose() -> Result<(), String> {
    let rows = 37;
    let cols = 29;
    let input: Vec<f64> = (0..rows * cols)
        .map(|index| (index as f64 * 0.125).sin())
        .collect();
    let mut output = vec![0.0; rows * cols];
    transpose_2d(&input, &mut output, rows, cols).map_err(|error| error.to_string())?;
    for row in 0..rows {
        for col in 0..cols {
            let expected = input[row * cols + col];
            let actual = output[col * rows + row];
            if actual != expected {
                return Err(format!(
                    "transpose mismatch at ({row},{col}): actual={actual}, expected={expected}"
                ));
            }
        }
    }
    Ok(())
}

fn check_elementwise() -> Result<(), String> {
    let mut product = [0.0_f64; 3];
    pointwise_mul(&[1.0, -2.0, 3.0], &[4.0, 0.5, -1.0], &mut product)
        .map_err(|error| error.to_string())?;
    if product != [4.0, -1.0, -3.0] {
        return Err(format!("pointwise product mismatch: {product:?}"));
    }
    scale_real(&mut product, 0.25);
    if product != [1.0, -0.25, -0.75] {
        return Err(format!("real scaling mismatch: {product:?}"));
    }
    Ok(())
}

fn check_segmented_reduction(seed: u64) -> Result<usize, String> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5E6D_0001);
    for case in 0..128usize {
        let value_count = case % 23;
        let segment_count = 1 + (case * 7 % 17);
        let values: Vec<Complex64> = (0..value_count)
            .map(|_| Complex64::new(rng.random_range(-3.0..=3.0), rng.random_range(-3.0..=3.0)))
            .collect();
        let mut offsets = vec![0_i32; segment_count + 1];
        for segment in 0..segment_count {
            let remaining = value_count - offsets[segment] as usize;
            let take = if remaining == 0 {
                0
            } else {
                rng.random_range(0..=remaining)
            };
            offsets[segment + 1] = offsets[segment] + take as i32;
        }
        offsets[segment_count] = value_count as i32;
        let mut actual = vec![Complex64::new(99.0, -77.0); segment_count];
        let mut repeated = vec![Complex64::new(0.0, 0.0); segment_count];
        segmented_sum(&values, &offsets, &mut actual)
            .map_err(|error| format!("segmented reduction case {case}: {error}"))?;
        segmented_sum(&values, &offsets, &mut repeated)
            .map_err(|error| format!("segmented reduction repeat {case}: {error}"))?;
        if actual != repeated {
            return Err(format!(
                "segmented reduction case {case} is not deterministic"
            ));
        }
        for segment in 0..segment_count {
            let start = offsets[segment] as usize;
            let end = offsets[segment + 1] as usize;
            let expected = values[start..end]
                .iter()
                .copied()
                .fold(Complex64::default(), |sum, value| sum + value);
            if actual[segment] != expected {
                return Err(format!(
                    "segmented reduction case {case}, segment {segment}: actual={}, expected={expected}",
                    actual[segment]
                ));
            }
        }
    }
    Ok(128)
}

fn check_backend_segmented_type<T: DeviceScalar + PartialEq>(
    backend: &CpuBackend,
    label: &str,
    values: &[T],
    offsets: &[i32],
    expected: &[T],
) -> Result<usize, String> {
    let mut device_values = backend
        .raw_alloc::<T>(values.len())
        .map_err(|error| format!("{label}: values allocation: {error}"))?;
    let mut device_offsets = backend
        .raw_alloc::<i32>(offsets.len())
        .map_err(|error| format!("{label}: offsets allocation: {error}"))?;
    backend
        .try_copy_h2d(values, &mut device_values)
        .map_err(|error| format!("{label}: values upload: {error}"))?;
    backend
        .try_copy_h2d(offsets, &mut device_offsets)
        .map_err(|error| format!("{label}: offsets upload: {error}"))?;
    for mode in [
        ReductionMode::Fast,
        ReductionMode::Reproducible,
        ReductionMode::Deterministic,
    ] {
        let mut output = backend
            .raw_alloc::<T>(expected.len())
            .map_err(|error| format!("{label} {mode:?}: output allocation: {error}"))?;
        backend
            .try_segmented_sum_with_mode(mode, &device_values, &device_offsets, &mut output)
            .map_err(|error| format!("{label} {mode:?}: reduction: {error}"))?;
        let mut actual = vec![T::zero(); expected.len()];
        backend
            .try_copy_d2h(&output, &mut actual)
            .map_err(|error| format!("{label} {mode:?}: result download: {error}"))?;
        if actual != expected {
            return Err(format!("{label} {mode:?}: result differs from reference"));
        }
    }
    Ok(3)
}

fn check_backend_segmented(seed: u64) -> Result<usize, String> {
    let backend = CpuBackend::new();
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5E6D_0002);
    let offsets = [0_i32, 2, 2, 5, 8, 8];
    let values_f64: Vec<f64> = (0..8).map(|_| rng.random_range(-3.0..=3.0)).collect();
    let expected_f64: Vec<f64> = offsets
        .windows(2)
        .map(|window| {
            values_f64[window[0] as usize..window[1] as usize]
                .iter()
                .sum()
        })
        .collect();
    let values_c64: Vec<Complex64> = (0..8)
        .map(|_| Complex64::new(rng.random_range(-3.0..=3.0), rng.random_range(-3.0..=3.0)))
        .collect();
    let expected_c64: Vec<Complex64> = offsets
        .windows(2)
        .map(|window| {
            values_c64[window[0] as usize..window[1] as usize]
                .iter()
                .copied()
                .sum()
        })
        .collect();
    let f64_cases = check_backend_segmented_type(
        &backend,
        "f64 segmented reduction",
        &values_f64,
        &offsets,
        &expected_f64,
    )?;
    let c64_cases = check_backend_segmented_type(
        &backend,
        "Complex64 segmented reduction",
        &values_c64,
        &offsets,
        &expected_c64,
    )?;
    Ok(f64_cases + c64_cases)
}

fn run() -> Result<usize, String> {
    let seed = seed()?;
    println!(
        "seed={seed} replay=SONIUM_QA_SEED={seed} cargo run -p sonium-qa --bin qa-compute-properties"
    );
    check_views(seed)?;
    check_numeric_edges()?;
    check_all_scalar_descriptors()?;
    check_transpose()?;
    check_elementwise()?;
    let host_segmented_cases = check_segmented_reduction(seed)?;
    let backend_segmented_cases = check_backend_segmented(seed)?;
    Ok(host_segmented_cases + backend_segmented_cases)
}

fn main() {
    match run() {
        Ok(segmented_layout_cases) => {
            let record = sonium_qa::QaResult::passed("compute.properties.v1")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("segmented_layout_cases", segmented_layout_cases as f64)
                .metric("segmented_backend_modes", 6.0)
                .seed(format!("SONIUM_QA_SEED={}", seed().unwrap_or_default()));
            if let Ok(record) = record.to_json_line() {
                println!("QA_RESULT: {record}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(record) =
                sonium_qa::QaResult::failed("compute.properties.v1", error.to_string())
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
            {
                println!("QA_RESULT: {record}");
            }
            eprintln!("COMPUTE PROPERTY QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
