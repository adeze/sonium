//! Seeded kernel-level invariants independent of the backend implementation.
//!
//! These checks exercise the scalar reference and the dispatched CPU path on
//! odd/tail batch sizes, plus deterministic segmented reductions. They are
//! intentionally small enough for every commit while still covering the
//! contracts consumed by BEM and FEM assembly.

use approx::assert_relative_eq;
use num_complex::{Complex32, Complex64};
use num_traits::ToPrimitive;
use rand::{Rng, SeedableRng, rngs::StdRng};
use sonium_compute::bem_cpu::{RegularPairInF64, regular_pair_batch, regular_pair_scalar};
use sonium_compute::{
    Backend, CpuBackend, DeviceCsr, DeviceOperator, DeviceScalar, ReductionMode, ViewAliasing,
    fmm_cpu::{
        GatherDotScatterC64, compact_indexed_gather_c64, compact_indexed_scatter_accumulate_c64,
        dense_matvec_scatter_accumulate_c64, gather_dot_scatter_batch_c64,
        indexed_dense_matvec_c64, try_compact_indexed_gather_c64,
        try_compact_indexed_scatter_accumulate_c64, try_gather_dot_scatter_batch_c64,
        try_indexed_dense_matvec_accumulate_c64, try_indexed_dense_matvec_c64,
        try_phase_mul_accumulate_c64,
    },
    segmented_sum,
    tensor_contract::batched_local_contract,
};

#[test]
fn device_csr_spmv_matches_host_reference_and_rejects_shapes() {
    let backend = CpuBackend::new();
    let row_ptr = [0_i32, 2, 3, 5];
    let col_idx = [0_i32, 2, 1, 0, 2];
    let values = [2.0_f64, -1.0, 4.0, 0.5, 3.0];
    let csr = DeviceCsr::try_from_host(&backend, 3, 3, &row_ptr, &col_idx, &values)
        .expect("valid CSR upload");

    let mut x = backend.raw_alloc::<f64>(3).expect("input allocation");
    let mut y = backend.raw_alloc::<f64>(3).expect("output allocation");
    backend.copy_h2d(&[1.0, 2.0, 3.0], &mut x);
    backend.copy_h2d(&[91.0, 92.0, 93.0], &mut y);

    DeviceOperator::try_apply(&csr, &backend, &x, &mut y).expect("checked CSR SpMV");
    let mut actual = [0.0; 3];
    backend.copy_d2h(&y, &mut actual);
    assert_eq!(actual, [-1.0, 8.0, 9.5]);

    let mut before = [0.0; 3];
    backend.copy_d2h(&y, &mut before);
    let short_x = backend.raw_alloc::<f64>(2).expect("short input allocation");
    let error = DeviceOperator::try_apply(&csr, &backend, &short_x, &mut y)
        .expect_err("malformed CSR SpMV input must be rejected");
    assert!(matches!(
        error,
        sonium_compute::BackendError::InvalidArgument { .. }
    ));
    let mut after = [0.0; 3];
    backend.copy_d2h(&y, &mut after);
    assert_eq!(after, before, "rejected CSR SpMV must not mutate output");
}

#[test]
fn device_csr_seeded_rectangular_multirhs_matches_scalar_reference() {
    let backend = CpuBackend::new();
    let mut rng = StdRng::seed_from_u64(SEED ^ 0xC5A5_0001);

    for case in 0..64 {
        let rows = 1 + (case * 5 % 8);
        let cols = 1 + (case * 7 % 9);
        let rhs = 1 + (case * 3 % 4);
        let mut row_ptr = Vec::with_capacity(rows + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0_i32);

        for row in 0..rows {
            let mut columns: Vec<usize> =
                (0..cols).filter(|_| rng.random_range(0..4) != 0).collect();
            if case % 3 != 0 && columns.is_empty() {
                columns.push((case + row) % cols);
            }
            columns.sort_unstable();
            columns.dedup();
            for column in columns {
                col_idx.push(column as i32);
                values.push(rng.random_range(-2.0..=2.0));
            }
            row_ptr.push(col_idx.len() as i32);
        }

        let csr = DeviceCsr::try_from_host(&backend, rows, cols, &row_ptr, &col_idx, &values)
            .unwrap_or_else(|error| panic!("seeded CSR case {case} rejected: {error}"));

        let x_host: Vec<f64> = (0..cols * rhs)
            .map(|_| rng.random_range(-1.5..=1.5))
            .collect();
        let mut expected = vec![0.0_f64; rows * rhs];
        for column_rhs in 0..rhs {
            for row in 0..rows {
                let start = row_ptr[row] as usize;
                let end = row_ptr[row + 1] as usize;
                expected[column_rhs * rows + row] = (start..end)
                    .map(|entry| {
                        values[entry] * x_host[column_rhs * cols + col_idx[entry] as usize]
                    })
                    .sum();
            }
        }

        let mut x = backend.raw_alloc::<f64>(x_host.len()).unwrap();
        let mut y = backend.raw_alloc::<f64>(expected.len()).unwrap();
        backend.copy_h2d(&x_host, &mut x);
        backend.copy_h2d(&vec![91.0; expected.len()], &mut y);
        DeviceOperator::try_apply_colmajor(&csr, &backend, rhs, &x, &mut y)
            .unwrap_or_else(|error| panic!("seeded CSR case {case} failed: {error}"));

        let mut actual = vec![0.0_f64; expected.len()];
        backend.copy_d2h(&y, &mut actual);
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            let tolerance = 2.0e-12 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "seeded CSR case {case}, output {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }
}

mod common;

const SEED: u64 = 0xC0A4_2028_0805_0003;

fn point(rng: &mut StdRng) -> [f64; 3] {
    [
        rng.random_range(-1.0..=1.0),
        rng.random_range(-1.0..=1.0),
        rng.random_range(-1.0..=1.0),
    ]
}

fn pair(rng: &mut StdRng) -> RegularPairInF64 {
    let v0 = point(rng);
    let mut v1 = point(rng);
    let mut v2 = point(rng);
    // Keep a non-degenerate triangle without making the generator depend on
    // the implementation's geometric tolerances.
    v1[0] += 1.5;
    v2[1] += 1.5;
    RegularPairInF64 {
        source_point: point(rng),
        source_normal: [0.0, 0.0, 1.0],
        v0,
        v1,
        v2,
    }
}

#[test]
fn bem_batch_matches_scalar_reference_for_tail_sizes() {
    let mut rng = StdRng::seed_from_u64(SEED);
    for length in 0..=9 {
        let pairs: Vec<_> = (0..length).map(|_| pair(&mut rng)).collect();
        let batch = regular_pair_batch(&pairs, 2.75, 1.0);
        let scalar: Vec<_> = pairs
            .iter()
            .map(|value| regular_pair_scalar(value, 2.75, 1.0))
            .collect();
        assert_eq!(batch.len(), scalar.len(), "batch length {length}");
        for (index, (actual, expected)) in batch.iter().zip(&scalar).enumerate() {
            for (actual, expected) in [actual.0, actual.1, actual.2, actual.3]
                .into_iter()
                .zip([expected.0, expected.1, expected.2, expected.3])
            {
                let scale = expected.norm().max(1.0);
                assert!(
                    (actual - expected).norm() <= 2.0e-12 * scale,
                    "pair {index} mismatch: actual={actual:?}, expected={expected:?}"
                );
            }
        }
    }
}

#[test]
fn phase_accumulation_matches_independent_complex_loop() {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0xA5A5);
    let mut dst = vec![Complex64::new(0.0, 0.0); 17];
    let phase: Vec<_> = (0..17)
        .map(|_| Complex64::new(rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0)))
        .collect();
    let src: Vec<_> = (0..17)
        .map(|_| Complex64::new(rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0)))
        .collect();
    let mut expected = dst.clone();
    for ((out, phase), src) in expected.iter_mut().zip(&phase).zip(&src) {
        *out += *phase * *src;
    }
    try_phase_mul_accumulate_c64(&mut dst, &phase, &src).unwrap();
    assert_eq!(dst, expected);
}

#[test]
fn segmented_sum_is_deterministic_and_overwrites_output() {
    let values = [1.0_f64, -2.0, 3.5, 4.0, -1.5, 8.0];
    let offsets = [0, 2, 2, 5, 6];
    let mut first = [99.0; 4];
    let mut second = [-77.0; 4];
    segmented_sum(&values, &offsets, &mut first).unwrap();
    segmented_sum(&values, &offsets, &mut second).unwrap();
    assert_eq!(first, [-1.0, 0.0, 6.0, 8.0,]);
    assert_eq!(first, second);
}

#[test]
fn seeded_segmented_sum_matches_independent_reference() {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x5E6D_0001);
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
        // Assign any values not consumed by random empty segments to the final
        // segment so the independent reference covers the complete input.
        offsets[segment_count] = value_count as i32;

        let mut actual = vec![Complex64::new(99.0, -77.0); segment_count];
        let mut repeated = vec![Complex64::new(0.0, 0.0); segment_count];
        segmented_sum(&values, &offsets, &mut actual).expect("generated offsets are valid");
        segmented_sum(&values, &offsets, &mut repeated).expect("generated offsets are valid");
        assert_eq!(actual, repeated, "case {case} is not deterministic");
        for segment in 0..segment_count {
            let start = offsets[segment] as usize;
            let end = offsets[segment + 1] as usize;
            let expected = values[start..end]
                .iter()
                .copied()
                .fold(Complex64::default(), |sum, value| sum + value);
            assert_eq!(actual[segment], expected, "case {case}, segment {segment}");
        }
    }
}

fn assert_backend_segmented_modes<T: DeviceScalar + PartialEq>(
    backend: &CpuBackend,
    values: &[T],
    offsets: &[i32],
    expected: &[T],
) {
    let mut device_values = backend.raw_alloc::<T>(values.len()).unwrap();
    let mut device_offsets = backend.raw_alloc::<i32>(offsets.len()).unwrap();
    backend.try_copy_h2d(values, &mut device_values).unwrap();
    backend.try_copy_h2d(offsets, &mut device_offsets).unwrap();
    for mode in [
        ReductionMode::Fast,
        ReductionMode::Reproducible,
        ReductionMode::Deterministic,
    ] {
        let mut device_output = backend.raw_alloc::<T>(expected.len()).unwrap();
        backend
            .try_segmented_sum_with_mode(mode, &device_values, &device_offsets, &mut device_output)
            .unwrap();
        let mut actual = vec![T::zero(); expected.len()];
        backend.try_copy_d2h(&device_output, &mut actual).unwrap();
        assert_eq!(actual, expected, "reduction mode {mode:?}");
    }
}

#[test]
fn backend_segmented_reduction_modes_match_reference_for_real_and_complex() {
    let backend = CpuBackend::new();
    let offsets = [0_i32, 2, 2, 5, 8, 8];
    let values_f64 = [0.5, -1.25, 3.0, 2.5, -4.0, 1.0, 0.25, 2.0];
    let expected_f64 = [
        values_f64[0] + values_f64[1],
        0.0,
        values_f64[2] + values_f64[3] + values_f64[4],
        values_f64[5] + values_f64[6] + values_f64[7],
        0.0,
    ];
    assert_backend_segmented_modes(&backend, &values_f64, &offsets, &expected_f64);

    let values_c64 = [
        Complex64::new(0.5, 1.0),
        Complex64::new(-1.25, 0.25),
        Complex64::new(3.0, -0.5),
        Complex64::new(2.5, 1.5),
        Complex64::new(-4.0, 2.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(0.25, 0.5),
        Complex64::new(2.0, -0.25),
    ];
    let expected_c64 = [
        values_c64[0] + values_c64[1],
        Complex64::default(),
        values_c64[2] + values_c64[3] + values_c64[4],
        values_c64[5] + values_c64[6] + values_c64[7],
        Complex64::default(),
    ];
    assert_backend_segmented_modes(&backend, &values_c64, &offsets, &expected_c64);
}

#[test]
fn indexed_matvec_matches_independent_reference_and_scatter() {
    let matrix = [
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.25, -0.75),
        Complex64::new(3.0, -0.5),
        Complex64::new(0.5, 2.0),
        Complex64::new(-1.5, 0.25),
    ];
    let src = common::random_complex64(SEED ^ 0x11, 7);
    let src_indices = common::unique_indices(SEED ^ 0x12, 3, src.len());
    let mut gathered_result = vec![Complex64::new(99.0, -2.0); 2];
    indexed_dense_matvec_c64(&matrix, 2, 3, 3, &src, &src_indices, &mut gathered_result);

    let expected: Vec<_> = (0..2)
        .map(|row| {
            (0..3).fold(Complex64::default(), |sum, col| {
                sum + matrix[row * 3 + col] * src[src_indices[col]]
            })
        })
        .collect();
    for (actual, expected) in gathered_result.iter().zip(&expected) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 2.0e-14);
        assert_relative_eq!(actual.im, expected.im, epsilon = 2.0e-14);
    }

    let dst_indices = [5, 1];
    let mut scattered = vec![Complex64::new(4.0, 1.0); src.len()];
    dense_matvec_scatter_accumulate_c64(&matrix, 2, 3, 3, &src, &mut scattered, &dst_indices);
    let contiguous_expected: Vec<_> = (0..2)
        .map(|row| {
            (0..3).fold(Complex64::default(), |sum, col| {
                sum + matrix[row * 3 + col] * src[col]
            })
        })
        .collect();
    for (index, &dst) in dst_indices.iter().enumerate() {
        let expected = Complex64::new(4.0, 1.0) + contiguous_expected[index];
        assert_relative_eq!(scattered[dst].re, expected.re, epsilon = 2.0e-14);
        assert_relative_eq!(scattered[dst].im, expected.im, epsilon = 2.0e-14);
    }
    for (index, value) in scattered.iter().enumerate() {
        if !dst_indices.contains(&index) {
            assert_eq!(*value, Complex64::new(4.0, 1.0));
        }
    }
}

#[test]
fn compact_gather_scatter_round_trip_preserves_selected_directions() {
    let source = common::random_complex64(SEED ^ 0x21, 12);
    let indices = common::unique_indices(SEED ^ 0x22, 7, source.len());
    let mut compact = vec![Complex64::new(0.0, 0.0); indices.len()];
    compact_indexed_gather_c64(&source, &indices, &mut compact);
    let mut restored = vec![Complex64::new(0.0, 0.0); source.len()];
    compact_indexed_scatter_accumulate_c64(&compact, &indices, &mut restored);
    for (index, value) in restored.iter().enumerate() {
        if indices.binary_search(&index).is_ok() {
            assert_eq!(*value, source[index]);
        } else {
            assert_eq!(*value, Complex64::new(0.0, 0.0));
        }
    }
}

#[test]
fn batched_gather_dot_scatter_matches_independent_reference() {
    let source = common::random_complex64(SEED ^ 0x31, 9);
    let mut actual = vec![Complex64::new(0.0, 0.0); source.len()];
    let matrix_a = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(-1.0, 0.5),
        Complex64::new(2.0, -0.25),
    ];
    let matrix_b = [
        Complex64::new(-0.5, 1.0),
        Complex64::new(1.5, 0.0),
        Complex64::new(0.25, -0.5),
        Complex64::new(0.0, -2.0),
    ];
    let batches = [
        GatherDotScatterC64 {
            matrix: &matrix_a,
            rows: 2,
            cols: 2,
            lda: 2,
            src_indices: &[0, 4],
            dst_indices: &[2, 7],
        },
        GatherDotScatterC64 {
            matrix: &matrix_b,
            rows: 2,
            cols: 2,
            lda: 2,
            src_indices: &[3, 8],
            dst_indices: &[1, 6],
        },
    ];
    gather_dot_scatter_batch_c64(&batches, &source, &mut actual);

    let mut expected = vec![Complex64::new(0.0, 0.0); source.len()];
    for batch in &batches {
        for row in 0..batch.rows {
            let value = (0..batch.cols).fold(Complex64::default(), |sum, col| {
                sum + batch.matrix[row * batch.lda + col] * source[batch.src_indices[col]]
            });
            expected[batch.dst_indices[row]] += value;
        }
    }
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 2.0e-14);
        assert_relative_eq!(actual.im, expected.im, epsilon = 2.0e-14);
    }
}

#[test]
fn fmm_indexed_helpers_seeded_rectangular_and_repeated_indices() {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0xF5AA_0001);

    for case in 0..48 {
        let rows = 1 + (case * 5 % 7);
        let cols = 1 + (case * 7 % 8);
        let leading_dimension = cols + case % 3;
        let source_len = cols + 3 + (case % 5);
        let matrix: Vec<Complex64> = (0..rows * leading_dimension)
            .map(|_| Complex64::new(rng.random_range(-2.0..=2.0), rng.random_range(-2.0..=2.0)))
            .collect();
        let source: Vec<Complex64> = (0..source_len)
            .map(|_| Complex64::new(rng.random_range(-2.0..=2.0), rng.random_range(-2.0..=2.0)))
            .collect();
        let src_indices: Vec<usize> = (0..cols).map(|_| rng.random_range(0..source_len)).collect();
        let dst_indices: Vec<usize> = (0..rows).map(|_| rng.random_range(0..source_len)).collect();

        let mut gathered = vec![Complex64::new(0.0, 0.0); rows];
        try_indexed_dense_matvec_c64(
            &matrix,
            rows,
            cols,
            leading_dimension,
            &source,
            &src_indices,
            &mut gathered,
        )
        .unwrap_or_else(|error| panic!("seeded FMM gather case {case} failed: {error}"));
        for row in 0..rows {
            let expected = (0..cols).fold(Complex64::default(), |sum, col| {
                sum + matrix[row * leading_dimension + col] * source[src_indices[col]]
            });
            assert_relative_eq!(gathered[row].re, expected.re, epsilon = 2.0e-12);
            assert_relative_eq!(gathered[row].im, expected.im, epsilon = 2.0e-12);
        }

        let initial: Vec<Complex64> = (0..source_len)
            .map(|_| Complex64::new(rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0)))
            .collect();
        let mut scattered = initial.clone();
        try_indexed_dense_matvec_accumulate_c64(
            &matrix,
            rows,
            cols,
            leading_dimension,
            &source,
            &src_indices,
            &mut scattered,
            &dst_indices,
        )
        .unwrap_or_else(|error| panic!("seeded FMM scatter case {case} failed: {error}"));
        let mut expected_scattered = initial;
        for row in 0..rows {
            let value = (0..cols).fold(Complex64::default(), |sum, col| {
                sum + matrix[row * leading_dimension + col] * source[src_indices[col]]
            });
            expected_scattered[dst_indices[row]] += value;
        }
        for (actual, expected) in scattered.iter().zip(expected_scattered) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 2.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 2.0e-12);
        }
    }
}

#[test]
fn checked_fmm_helpers_reject_invalid_shape_and_indices_without_mutation() {
    let matrix = [Complex64::new(1.0, 0.0)];
    let source = [Complex64::new(2.0, 0.0)];
    let mut output = vec![Complex64::new(7.0, 0.0)];
    assert!(try_indexed_dense_matvec_c64(&matrix, 2, 1, 1, &source, &[0], &mut output).is_err());
    assert_eq!(output, [Complex64::new(7.0, 0.0)]);
    assert!(try_indexed_dense_matvec_c64(&matrix, 1, 1, 1, &source, &[2], &mut output).is_err());
    assert_eq!(output, [Complex64::new(7.0, 0.0)]);
    assert!(try_indexed_dense_matvec_accumulate_c64(
        &matrix,
        1,
        1,
        1,
        &source,
        &[0],
        &mut output,
        &[2],
    )
    .is_err());
    assert_eq!(output, [Complex64::new(7.0, 0.0)]);
    assert!(try_compact_indexed_gather_c64(&source, &[2], &mut output).is_err());
    assert_eq!(output, [Complex64::new(7.0, 0.0)]);
    assert!(try_compact_indexed_scatter_accumulate_c64(&source, &[2], &mut output).is_err());
    assert_eq!(output, [Complex64::new(7.0, 0.0)]);
    let batch = [GatherDotScatterC64 {
        matrix: &matrix,
        rows: 1,
        cols: 1,
        lda: 1,
        src_indices: &[2],
        dst_indices: &[0],
    }];
    assert!(try_gather_dot_scatter_batch_c64(&batch, &source, &mut output).is_err());
    assert_eq!(output, [Complex64::new(7.0, 0.0)]);
}

#[test]
fn batched_local_contract_matches_independent_reference_for_scalar_families() {
    fn check<T>(lhs: &[T], rhs: &[T], expected: &[T])
    where
        T: DeviceScalar + PartialEq + std::fmt::Debug,
    {
        let mut output = vec![T::zero(); expected.len()];
        batched_local_contract(lhs, rhs, &mut output, 2, 2, 2, 2)
            .expect("valid batched local contraction");
        assert_eq!(output, expected);
    }

    check(
        &[1.0_f64, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0, 3.0],
        &[5.0_f64, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0],
        &[19.0_f64, 22.0, 43.0, 50.0, 2.0, 4.0, 10.0, 14.0],
    );
    check(
        &[
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(3.0, 0.5),
            Complex32::new(-1.0, 2.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(0.0, 1.0),
            Complex32::new(1.0, -1.0),
            Complex32::new(3.0, 0.0),
        ],
        &[
            Complex32::new(5.0, 0.0),
            Complex32::new(6.0, -1.0),
            Complex32::new(7.0, 1.0),
            Complex32::new(8.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 1.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, -1.0),
        ],
        &[
            Complex32::new(20.0, 0.0),
            Complex32::new(23.0, -3.0),
            Complex32::new(6.0, 15.5),
            Complex32::new(10.5, 16.0),
            Complex32::new(2.0, 3.0),
            Complex32::new(5.0, 6.0),
            Complex32::new(10.0, -1.0),
            Complex32::new(15.0, -4.0),
        ],
    );
    check(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(3.0, 0.0),
        ],
        &[
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, -1.0),
            Complex64::new(7.0, 1.0),
            Complex64::new(8.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, -1.0),
        ],
        &[
            Complex64::new(20.0, 0.0),
            Complex64::new(23.0, -3.0),
            Complex64::new(6.0, 15.5),
            Complex64::new(10.5, 16.0),
            Complex64::new(2.0, 3.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(10.0, -1.0),
            Complex64::new(15.0, -4.0),
        ],
    );

    let mut sentinel = [Complex64::new(9.0, -2.0)];
    assert!(
        batched_local_contract(&[Complex64::new(1.0, 0.0)], &[], &mut sentinel, 1, 1, 1, 1)
            .is_err()
    );
    assert_eq!(sentinel, [Complex64::new(9.0, -2.0)]);
}

#[test]
fn checked_view_contracts_cover_strided_rank_n_empty_and_overlap_cases() {
    fn check<T>(values: &[T])
    where
        T: DeviceScalar + PartialEq + std::fmt::Debug,
    {
        let backend = CpuBackend::new();
        let mut source = backend.raw_alloc::<T>(values.len()).unwrap();
        backend.try_copy_h2d(values, &mut source).unwrap();

        let view = backend.checked_view(&source, 1, 4, 2).unwrap();
        let mut selected = vec![T::zero(); 4];
        backend
            .try_copy_view_d2h(&source, view, &mut selected)
            .unwrap();
        assert_eq!(selected, [values[1], values[3], values[5], values[7]]);

        let empty = backend.checked_view(&source, values.len(), 0, 3).unwrap();
        backend.try_copy_view_d2h(&source, empty, &mut []).unwrap();

        let descriptor = backend
            .checked_nd_view_descriptor(
                &source,
                &[2, 3],
                std::mem::size_of::<T>(),
                &[4, 1],
                ViewAliasing::ReadOnly,
            )
            .unwrap();
        let mut rank_values = vec![T::zero(); 6];
        backend
            .try_copy_nd_view_d2h(&source, &descriptor, &mut rank_values)
            .unwrap();
        assert_eq!(
            rank_values,
            [
                values[1], values[2], values[3], values[5], values[6], values[7]
            ]
        );

        let mut destination = backend.raw_alloc::<T>(values.len()).unwrap();
        let destination_descriptor = backend
            .checked_nd_view_descriptor(
                &destination,
                &[2, 3],
                std::mem::size_of::<T>(),
                &[4, 1],
                ViewAliasing::Disjoint,
            )
            .unwrap();
        backend
            .try_copy_nd_view_h2d(&rank_values, &mut destination, &destination_descriptor)
            .unwrap();
        let mut round_trip = vec![T::zero(); 6];
        backend
            .try_copy_nd_view_d2h(&destination, &destination_descriptor, &mut round_trip)
            .unwrap();
        assert_eq!(round_trip, rank_values);

        let source_view = backend.checked_view(&destination, 0, 4, 1).unwrap();
        let destination_view = backend.checked_view(&destination, 1, 4, 1).unwrap();
        backend
            .try_copy_view_in_place(&mut destination, source_view, destination_view)
            .unwrap();
        let mut destination_host = vec![T::zero(); values.len()];
        backend
            .try_copy_d2h(&destination, &mut destination_host)
            .unwrap();
        assert_eq!(
            destination_host,
            [
                T::zero(),
                T::zero(),
                values[1],
                values[2],
                values[3],
                values[5],
                values[6],
                values[7],
            ]
            .as_slice()
        );

        assert!(
            backend
                .checked_nd_view_descriptor(&source, &[2, 3], 0, &[0, 1], ViewAliasing::ReadOnly,)
                .is_err()
        );
        let mut readonly_destination = backend.raw_alloc::<T>(values.len()).unwrap();
        let readonly_descriptor = backend
            .checked_nd_view_descriptor(
                &readonly_destination,
                &[2, 3],
                std::mem::size_of::<T>(),
                &[4, 1],
                ViewAliasing::ReadOnly,
            )
            .unwrap();
        assert!(
            backend
                .try_copy_nd_view_h2d(
                    &rank_values,
                    &mut readonly_destination,
                    &readonly_descriptor
                )
                .is_err()
        );
    }

    check(&[0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    check(&[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    check(&[
        Complex64::new(0.0, 0.5),
        Complex64::new(1.0, -0.5),
        Complex64::new(2.0, 1.5),
        Complex64::new(3.0, -1.5),
        Complex64::new(4.0, 2.5),
        Complex64::new(5.0, -2.5),
        Complex64::new(6.0, 3.5),
        Complex64::new(7.0, -3.5),
    ]);
    check(&[
        Complex32::new(0.0, 0.5),
        Complex32::new(1.0, -0.5),
        Complex32::new(2.0, 1.5),
        Complex32::new(3.0, -1.5),
        Complex32::new(4.0, 2.5),
        Complex32::new(5.0, -2.5),
        Complex32::new(6.0, 3.5),
        Complex32::new(7.0, -3.5),
    ]);
}

#[test]
fn reduction_policy_modes_match_conjugating_reference_for_all_scalar_families() {
    fn check<T, F>(make_scalar: F)
    where
        T: DeviceScalar,
        T::Real: ToPrimitive,
        F: Fn(usize) -> T + Copy,
    {
        let backend = CpuBackend::new();
        for (fixture, length) in [0usize, 1, 7, 10].into_iter().enumerate() {
            let x_host: Vec<_> = (0..length)
                .map(|index| make_scalar(index + fixture * 101))
                .collect();
            let y_host: Vec<_> = (0..length)
                .map(|index| make_scalar(index + fixture * 211 + 10_000))
                .collect();
            let basis_count = 3;
            let basis_host: Vec<_> = (0..basis_count * length)
                .map(|index| make_scalar(index + fixture * 307 + 20_000))
                .collect();
            let mut x = backend.raw_alloc::<T>(length).unwrap();
            let mut y = backend.raw_alloc::<T>(length).unwrap();
            let mut basis = backend.raw_alloc::<T>(basis_host.len()).unwrap();
            backend.try_copy_h2d(&x_host, &mut x).unwrap();
            backend.try_copy_h2d(&y_host, &mut y).unwrap();
            backend.try_copy_h2d(&basis_host, &mut basis).unwrap();
            let expected_dot = x_host
                .iter()
                .zip(&y_host)
                .fold(T::zero(), |sum, (&lhs, &rhs)| sum + lhs.conj_mul(rhs));
            let expected_norm = x_host
                .iter()
                .map(|value| value.norm_sqr().to_f64().unwrap())
                .sum::<f64>()
                .sqrt();
            for mode in [
                ReductionMode::Fast,
                ReductionMode::Reproducible,
                ReductionMode::Deterministic,
            ] {
                let dot = backend.try_dot_with_mode(mode, &x, &y).unwrap();
                let dot_error = dot.sub(expected_dot).norm_sqr().to_f64().unwrap().sqrt();
                assert!(dot_error <= 2.0e-5, "mode {mode:?}, dot error {dot_error}");
                let norm = backend.try_nrm2_with_mode(mode, &x).unwrap();
                assert!((norm.to_f64().unwrap() - expected_norm).abs() <= 2.0e-5);
                let multi = backend
                    .try_multi_dot_strided_with_mode(mode, &basis, length, basis_count, &y)
                    .unwrap();
                assert_eq!(multi.len(), basis_count);
                for basis_index in 0..basis_count {
                    let expected = (0..length).fold(T::zero(), |sum, column| {
                        sum + basis_host[basis_index * length + column].conj_mul(y_host[column])
                    });
                    let error = multi[basis_index]
                        .sub(expected)
                        .norm_sqr()
                        .to_f64()
                        .unwrap()
                        .sqrt();
                    assert!(error <= 2.0e-5, "mode {mode:?}, multi-dot error {error}");
                }
            }
        }
    }

    check::<f32, _>(|index| (index as f32 * 0.23).sin() * 0.35);
    check::<f64, _>(|index| (index as f64 * 0.23).sin() * 0.35);
    check::<Complex32, _>(|index| {
        Complex32::new(
            (index as f32 * 0.23).sin() * 0.35,
            (index as f32 * 0.17).cos() * 0.21,
        )
    });
    check::<Complex64, _>(|index| {
        Complex64::new(
            (index as f64 * 0.23).sin() * 0.35,
            (index as f64 * 0.17).cos() * 0.21,
        )
    });
}
