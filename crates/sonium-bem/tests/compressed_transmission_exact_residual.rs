use num_complex::Complex64;
use sonium_bem::core::assembly::{
    CompressedTransmissionTbemSystem, build_transmission_tbem_system,
};
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::types::PhysicsParams;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::GmresConfig;
use sonium_solvers::traits::LinearOperator;

fn source_matrix() -> DenseMatrix<Complex64> {
    let mut matrix = DenseMatrix::zeros(4, 4);
    let values = [
        [5.0, 0.0, 1.0, 2.0],
        [0.0, 6.0, 3.0, 5.0],
        [2.0, 1.0, 7.0, 0.0],
        [4.0, 3.0, 0.0, 8.0],
    ];
    for row in 0..4 {
        for column in 0..4 {
            matrix[[row, column]] = Complex64::new(values[row][column], 0.0);
        }
    }
    matrix
}

fn relative_residual(
    matrix: &DenseMatrix<Complex64>,
    values: &[Complex64],
    rhs: &[Complex64],
    hermitian: bool,
) -> f64 {
    let mut action = vec![Complex64::default(); matrix.nrows()];
    if hermitian {
        for row in 0..matrix.nrows() {
            for column in 0..matrix.ncols() {
                action[column] += matrix[[row, column]].conj() * values[row];
            }
        }
    } else {
        for row in 0..matrix.nrows() {
            for column in 0..matrix.ncols() {
                action[row] += matrix[[row, column]] * values[column];
            }
        }
    }
    let residual_squared = action
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>();
    let rhs_squared = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>();
    residual_squared.sqrt() / rhs_squared.sqrt().max(f64::MIN_POSITIVE)
}

fn solve_config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 100,
        restart: 8,
        tolerance: 1.0e-12,
        exact_residual_check_interval: 1,
        ..GmresConfig::default()
    }
}

#[test]
fn dense_backed_compressed_solves_report_exact_source_residuals() {
    let source = source_matrix();
    let dense_backed = CompressedTransmissionTbemSystem::from_dense_matrix(2, &source, 2, 0.5, 1)
        .expect("valid dense-backed compression");
    assert_eq!(dense_backed.compressed_block_count(), 2);

    // The same blocks without an exact source matrix provide the deliberately
    // weaker compressed-action certificate used as the control case.
    let compressed_action_only =
        CompressedTransmissionTbemSystem::from_entry(2, 4, 2, 0.5, 1, |row, column| {
            source[[row, column]]
        })
        .expect("valid on-demand compression");

    let mut nearfield = DenseMatrix::zeros(4, 4);
    for index in 0..4 {
        nearfield[[index, index]] = Complex64::new(1.0, 0.0);
    }
    let config = solve_config();
    let rhs = vec![
        Complex64::new(1.0, 0.2),
        Complex64::new(-2.0, 0.1),
        Complex64::new(0.5, -0.3),
        Complex64::new(3.0, 0.4),
    ];

    let compressed_forward = compressed_action_only
        .solve_with_ilu(&rhs, &nearfield, &config, 1.0e-10)
        .expect("compressed forward solve");
    let compressed_exact_residual =
        relative_residual(&source, compressed_forward.values(), &rhs, false);
    assert!(
        compressed_forward.relative_residual() <= 1.0e-10,
        "compressed-action residual {:.3e}",
        compressed_forward.relative_residual()
    );
    assert!(
        compressed_exact_residual > 1.0e-4,
        "rank-truncated iterate unexpectedly satisfies exact residual {:.3e}",
        compressed_exact_residual
    );

    let exact_forward = dense_backed
        .solve_with_ilu(&rhs, &nearfield, &config, 1.0e-10)
        .expect("dense-backed exact-certified forward solve");
    let independently_certified_forward =
        relative_residual(&source, exact_forward.values(), &rhs, false);
    assert!(
        independently_certified_forward <= 1.0e-10,
        "exact forward residual {:.3e}",
        independently_certified_forward
    );
    assert!((exact_forward.relative_residual() - independently_certified_forward).abs() <= 1.0e-14);

    let adjoint_rhs = vec![
        Complex64::new(-0.4, 0.6),
        Complex64::new(1.5, -0.2),
        Complex64::new(2.2, 0.3),
        Complex64::new(-1.0, -0.5),
    ];
    let compressed_adjoint = compressed_action_only
        .solve_hermitian_with_ilu(&adjoint_rhs, &nearfield, &config, 1.0e-10)
        .expect("compressed Hermitian solve");
    let compressed_adjoint_exact_residual =
        relative_residual(&source, compressed_adjoint.values(), &adjoint_rhs, true);
    assert!(compressed_adjoint_exact_residual > 1.0e-4);

    let exact_adjoint = dense_backed
        .solve_hermitian_with_ilu(&adjoint_rhs, &nearfield, &config, 1.0e-10)
        .expect("dense-backed exact-certified Hermitian solve");
    let independently_certified_adjoint =
        relative_residual(&source, exact_adjoint.values(), &adjoint_rhs, true);
    assert!(independently_certified_adjoint <= 1.0e-10);
    assert!((exact_adjoint.relative_residual() - independently_certified_adjoint).abs() <= 1.0e-14);
}

#[test]
fn kernel_native_value_compression_certifies_against_dense_source() {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let exterior =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid exterior physics");
    let interior =
        PhysicsParams::try_new(420.0, 1_480.0, 1_000.0, true).expect("valid interior physics");
    let system = build_transmission_tbem_system(&mesh.elements, &mesh.nodes, exterior, interior)
        .expect("dense transmission system");

    // Force several ACA tiles and a deliberately small rank cap.  Some
    // near-field tiles may correctly fall back to dense storage when the
    // physical block is not low rank.  The kernel-native action is still
    // checked against the dense source retained by the production system, so
    // a compressed-only iterate cannot be published as an exact solve.
    let compressed = system
        .compress_kernel_native(16, 0.5, 4)
        .expect("kernel-native compression");
    assert!(compressed.dense_block_count() > 0);
    assert!(compressed.maximum_block_relative_error().is_finite());

    let rhs = (0..system.total_dofs())
        .map(|index| Complex64::new(0.2 + index as f64 * 0.001, -0.1))
        .collect::<Vec<_>>();
    let nearfield = system.matrix().clone();
    let mut config = solve_config();
    config.max_iterations = 80;
    config.restart = 16;
    config.tolerance = 1.0e-9;

    let solution = compressed
        .solve_with_ilu(&rhs, &nearfield, &config, 1.0e-9)
        .expect("exact-source-certified kernel-native solve");
    let exact_residual = relative_residual(system.matrix(), solution.values(), &rhs, false);
    assert!(
        exact_residual <= 1.0e-9,
        "kernel-native exact residual {exact_residual:.3e}"
    );
    assert!((solution.relative_residual() - exact_residual).abs() <= 1.0e-12);
}

#[test]
fn compressed_transmission_block_action_matches_column_actions() {
    let source = source_matrix();
    let compressed = CompressedTransmissionTbemSystem::from_dense_matrix(2, &source, 2, 0.5, 1)
        .expect("valid compressed transmission matrix");
    let mut input = DenseMatrix::zeros(4, 3);
    for column in 0..input.ncols() {
        for row in 0..input.nrows() {
            input[[row, column]] = Complex64::new(
                0.2 + 0.11 * row as f64 + 0.07 * column as f64,
                -0.3 + 0.05 * row as f64 - 0.09 * column as f64,
            );
        }
    }

    let mut block_action = DenseMatrix::zeros(4, 3);
    <CompressedTransmissionTbemSystem as LinearOperator<Complex64>>::apply_block(
        &compressed,
        &input,
        &mut block_action,
    );
    for column in 0..input.ncols() {
        let vector = (0..input.nrows())
            .map(|row| input[[row, column]])
            .collect::<Vec<_>>();
        let expected = compressed.apply(&vector);
        for row in 0..input.nrows() {
            assert_eq!(block_action[[row, column]], expected[row]);
        }
    }
}

#[test]
fn compressed_transmission_hermitian_block_action_matches_column_actions() {
    let source = source_matrix();
    let compressed = CompressedTransmissionTbemSystem::from_dense_matrix(2, &source, 2, 0.5, 1)
        .expect("valid compressed transmission matrix");
    let mut input = DenseMatrix::zeros(4, 3);
    for column in 0..input.ncols() {
        for row in 0..input.nrows() {
            input[[row, column]] = Complex64::new(
                -0.17 + 0.08 * row as f64 + 0.13 * column as f64,
                0.29 - 0.04 * row as f64 + 0.06 * column as f64,
            );
        }
    }

    let block_action = compressed
        .apply_hermitian_block_checked(&input)
        .expect("finite Hermitian block action");
    for column in 0..input.ncols() {
        let vector = (0..input.nrows())
            .map(|row| input[[row, column]])
            .collect::<Vec<_>>();
        let expected = compressed.apply_hermitian(&vector);
        for row in 0..input.nrows() {
            assert_eq!(block_action[[row, column]], expected[row]);
        }
    }

    let wrong_rows = DenseMatrix::zeros(3, 3);
    assert!(matches!(
        compressed.apply_hermitian_block_checked(&wrong_rows),
        Err(sonium_bem::core::assembly::TransmissionCompressionError::Dimension { .. })
    ));
    input[[1, 2]] = Complex64::new(f64::NAN, 0.0);
    assert!(matches!(
        compressed.apply_hermitian_block_checked(&input),
        Err(sonium_bem::core::assembly::TransmissionCompressionError::NonFiniteInput { .. })
    ));
}

#[test]
fn compressed_transmission_block_row_actions_match_column_actions() {
    let source = source_matrix();
    let compressed = CompressedTransmissionTbemSystem::from_dense_matrix(2, &source, 2, 0.5, 1)
        .expect("valid compressed transmission matrix");
    let mut input = DenseMatrix::zeros(4, 3);
    for column in 0..input.ncols() {
        for row in 0..input.nrows() {
            input[[row, column]] = Complex64::new(
                0.03 + 0.07 * row as f64 - 0.05 * column as f64,
                -0.21 + 0.09 * row as f64 + 0.02 * column as f64,
            );
        }
    }
    let rows = [3, 1, 3];
    let forward = compressed
        .apply_rows_block_checked(&input, &rows)
        .expect("forward block rows");
    let transpose = compressed
        .apply_transpose_rows_block_checked(&input, &rows)
        .expect("transpose block rows");
    for column in 0..input.ncols() {
        let vector = (0..input.nrows())
            .map(|row| input[[row, column]])
            .collect::<Vec<_>>();
        let expected_forward = compressed
            .apply_rows_checked(&vector, &rows)
            .expect("forward rows");
        let expected_transpose = compressed
            .apply_transpose_rows_checked(&vector, &rows)
            .expect("transpose rows");
        for (local_row, _) in rows.iter().enumerate() {
            assert_eq!(forward[[local_row, column]], expected_forward[local_row]);
            assert_eq!(
                transpose[[local_row, column]],
                expected_transpose[local_row]
            );
        }
    }

    let wrong_rows = [4];
    assert!(matches!(
        compressed.apply_rows_block_checked(&input, &wrong_rows),
        Err(sonium_bem::core::assembly::TransmissionCompressionError::InvalidRow { .. })
    ));
}
