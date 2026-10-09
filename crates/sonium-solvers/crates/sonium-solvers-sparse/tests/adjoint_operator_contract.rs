use num_complex::Complex64;
use sonium_solvers_core::{
    dense::DenseMatrix,
    traits::{AdjointCapability, LinearOperator, SolverError},
};
use sonium_solvers_sparse::{
    BlockLayout, BlockOperator, CompactCsrMatrix, CsrMatrix, DenseBlockOperator, FnOperator,
    RowBucketedCsrMatrix, SellCSigmaMatrix,
};

fn fixture() -> CsrMatrix<Complex64> {
    CsrMatrix::from_raw_parts(
        3,
        2,
        vec![0, 2, 3, 5],
        vec![0, 1, 1, 0, 1],
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(-2.0, 0.5),
            Complex64::new(0.25, -3.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-0.75, 2.5),
        ],
    )
}

fn property_vector(len: usize, seed: u64) -> Vec<Complex64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let re = ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let im = ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0;
            Complex64::new(re, im)
        })
        .collect()
}

fn assert_randomized_adjoint_contract(
    operator_name: &str,
    operator: &dyn LinearOperator<Complex64>,
) {
    assert_eq!(
        operator.transpose_capability(),
        AdjointCapability::Exact,
        "{operator_name}: Complex64 transpose capability"
    );
    assert_eq!(
        operator.hermitian_capability(),
        AdjointCapability::Exact,
        "{operator_name}: Complex64 Hermitian capability"
    );

    for seed in 0..32_u64 {
        let x = property_vector(operator.num_cols(), 0x000A_11CE_0000_u64.wrapping_add(seed));
        let y = property_vector(operator.num_rows(), 0x000B_00B5_0000_u64.wrapping_add(seed));
        let forward = operator
            .try_apply(&x)
            .unwrap_or_else(|error| panic!("{operator_name}: forward seed {seed}: {error:?}"));
        let transpose = operator
            .try_apply_transpose(&y)
            .unwrap_or_else(|error| panic!("{operator_name}: transpose seed {seed}: {error:?}"));
        let hermitian = operator
            .try_apply_hermitian(&y)
            .unwrap_or_else(|error| panic!("{operator_name}: Hermitian seed {seed}: {error:?}"));

        let forward_inner: Complex64 = forward.iter().zip(&y).map(|(a, b)| a.conj() * *b).sum();
        let hermitian_inner: Complex64 = x.iter().zip(&hermitian).map(|(a, b)| a.conj() * *b).sum();
        let transpose_left: Complex64 = forward.iter().zip(&y).map(|(a, b)| *a * *b).sum();
        let transpose_right: Complex64 = x.iter().zip(&transpose).map(|(a, b)| *a * *b).sum();
        let scale = 1.0 + forward_inner.norm().max(hermitian_inner.norm());
        assert!(
            (forward_inner - hermitian_inner).norm() <= 5.0e-12 * scale,
            "{operator_name}: Hermitian identity failed for Complex64 shape {}x{}, seed {seed}: lhs={forward_inner:?}, rhs={hermitian_inner:?}",
            operator.num_rows(),
            operator.num_cols()
        );
        let transpose_scale = 1.0 + transpose_left.norm().max(transpose_right.norm());
        assert!(
            (transpose_left - transpose_right).norm() <= 5.0e-12 * transpose_scale,
            "{operator_name}: transpose identity failed for Complex64 shape {}x{}, seed {seed}: lhs={transpose_left:?}, rhs={transpose_right:?}",
            operator.num_rows(),
            operator.num_cols()
        );
    }
}

fn assert_complex_block_matches_vectors(
    operator: &dyn LinearOperator<Complex64>,
    x: &DenseMatrix<Complex64>,
    adjoint_input: &DenseMatrix<Complex64>,
) {
    assert_eq!(x.nrows(), operator.num_cols());
    assert_eq!(adjoint_input.nrows(), operator.num_rows());
    assert_eq!(x.ncols(), adjoint_input.ncols());

    let mut forward = DenseMatrix::zeros(operator.num_rows(), x.ncols());
    operator
        .try_apply_block(x, &mut forward)
        .expect("checked forward block action");
    let mut transpose = DenseMatrix::zeros(operator.num_cols(), adjoint_input.ncols());
    operator
        .try_apply_transpose_block(adjoint_input, &mut transpose)
        .expect("checked transpose block action");
    let mut hermitian = DenseMatrix::zeros(operator.num_cols(), adjoint_input.ncols());
    operator
        .try_apply_hermitian_block(adjoint_input, &mut hermitian)
        .expect("checked Hermitian block action");
    for column in 0..x.ncols() {
        let expected = operator
            .try_apply(&x.column_to_vec(column))
            .expect("checked forward action");
        for row in 0..operator.num_rows() {
            assert!((forward[[row, column]] - expected[row]).norm() < 1.0e-12);
        }
    }
    for column in 0..adjoint_input.ncols() {
        let forward_column = forward.column_to_vec(column);
        let x_column = x.column_to_vec(column);
        let adjoint_column = adjoint_input.column_to_vec(column);
        let lhs_hermitian: Complex64 = forward_column
            .iter()
            .zip(adjoint_column.iter())
            .map(|(a, b)| a.conj() * *b)
            .sum();
        let rhs_hermitian: Complex64 = x_column
            .iter()
            .zip(hermitian.column_to_vec(column).iter())
            .map(|(a, b)| a.conj() * *b)
            .sum();
        assert!((lhs_hermitian - rhs_hermitian).norm() < 1.0e-12);
        let lhs_transpose: Complex64 = forward_column
            .iter()
            .zip(adjoint_column.iter())
            .map(|(a, b)| *a * *b)
            .sum();
        let rhs_transpose: Complex64 = x_column
            .iter()
            .zip(transpose.column_to_vec(column).iter())
            .map(|(a, b)| *a * *b)
            .sum();
        assert!((lhs_transpose - rhs_transpose).norm() < 1.0e-12);

        let expected_transpose = operator
            .try_apply_transpose(&adjoint_column)
            .expect("checked transpose action");
        let expected_hermitian = operator
            .try_apply_hermitian(&adjoint_column)
            .expect("checked Hermitian action");
        for row in 0..operator.num_cols() {
            assert!((transpose[[row, column]] - expected_transpose[row]).norm() < 1.0e-12);
            assert!((hermitian[[row, column]] - expected_hermitian[row]).norm() < 1.0e-12);
        }
    }
}

struct UnknownAdjointOperator;

impl LinearOperator<f64> for UnknownAdjointOperator {
    fn num_rows(&self) -> usize {
        1
    }

    fn num_cols(&self) -> usize {
        1
    }

    fn apply_into(&self, _x: &[f64], y: &mut [f64]) {
        y[0] = 1.0;
    }

    fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
        panic!("unknown adjoint callback must not be invoked by checked dispatch");
    }
}

fn assert_inner_product_adjoint(operator: &dyn LinearOperator<Complex64>) {
    let x = [Complex64::new(0.5, -1.0), Complex64::new(-2.0, 0.25)];
    let y = [
        Complex64::new(1.0, 0.5),
        Complex64::new(-0.25, 2.0),
        Complex64::new(1.5, -0.75),
    ];
    assert_inner_product_adjoint_for_vectors(operator, &x, &y);
}

fn assert_inner_product_adjoint_for_vectors(
    operator: &dyn LinearOperator<Complex64>,
    x: &[Complex64],
    y: &[Complex64],
) {
    assert_eq!(operator.transpose_capability(), AdjointCapability::Exact);
    assert_eq!(operator.hermitian_capability(), AdjointCapability::Exact);

    let forward = operator.try_apply(x).expect("forward action");
    let transpose = operator
        .try_apply_transpose(y)
        .expect("exact transpose action");
    let hermitian = operator
        .try_apply_hermitian(y)
        .expect("exact Hermitian action");

    let lhs = forward
        .iter()
        .zip(y.iter())
        .map(|(a, b)| a.conj() * *b)
        .fold(Complex64::new(0.0, 0.0), |sum, value| sum + value);
    let rhs = x
        .iter()
        .zip(hermitian.iter())
        .map(|(a, b)| a.conj() * *b)
        .fold(Complex64::new(0.0, 0.0), |sum, value| sum + value);
    assert!((lhs - rhs).norm() < 1.0e-12, "{lhs:?} != {rhs:?}");

    let transpose_lhs = forward
        .iter()
        .zip(y.iter())
        .map(|(a, b)| *a * *b)
        .fold(Complex64::new(0.0, 0.0), |sum, value| sum + value);
    let transpose_rhs = x
        .iter()
        .zip(transpose.iter())
        .map(|(a, b)| *a * *b)
        .fold(Complex64::new(0.0, 0.0), |sum, value| sum + value);
    assert!(
        (transpose_lhs - transpose_rhs).norm() < 1.0e-12,
        "{transpose_lhs:?} != {transpose_rhs:?}"
    );
}

#[test]
fn reordered_sparse_formats_keep_exact_real_and_complex_adjoint_contracts() {
    let csr = fixture();
    let compact = CompactCsrMatrix::from_csr(&csr).expect("u32-compatible CSR");
    let bucketed = RowBucketedCsrMatrix::from_csr(&csr);
    let sell = SellCSigmaMatrix::from_csr(&csr, 2, 8);

    let operators: [&dyn LinearOperator<Complex64>; 4] = [&csr, &compact, &bucketed, &sell];
    for operator in operators {
        assert_inner_product_adjoint(operator);
    }

    let real = CsrMatrix::from_raw_parts(
        2,
        3,
        vec![0, 2, 4],
        vec![0, 2, 1, 2],
        vec![1.0, -2.0, 3.0, 4.0],
    );
    let real_compact = CompactCsrMatrix::from_csr(&real).expect("u32-compatible CSR");
    let real_bucketed = RowBucketedCsrMatrix::from_csr(&real);
    let real_sell = SellCSigmaMatrix::from_csr(&real, 2, 8);
    let real_formats: [&dyn LinearOperator<f64>; 4] =
        [&real, &real_compact, &real_bucketed, &real_sell];
    let x = [1.0, -0.5, 2.0];
    let y = [0.25, -1.5];
    for operator in real_formats {
        assert_eq!(operator.transpose_capability(), AdjointCapability::Exact);
        assert_eq!(operator.hermitian_capability(), AdjointCapability::Exact);
        assert_eq!(
            operator.try_apply_transpose(&y).expect("transpose"),
            operator.try_apply_hermitian(&y).expect("Hermitian")
        );
        assert_eq!(operator.try_apply(&x).expect("forward").len(), y.len());
    }
}

fn square_complex_csr(values: [Complex64; 4]) -> CsrMatrix<Complex64> {
    CsrMatrix::from_raw_parts(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], values.to_vec())
}

#[test]
fn csr_block_adjoint_dispatch_accepts_multiple_right_hand_sides() {
    let csr = fixture();
    let x = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(0.5, -1.0),
            Complex64::new(1.0, 0.25),
            Complex64::new(-1.25, 0.75),
            Complex64::new(0.5, -0.5),
        ],
    );
    let adjoint_input = DenseMatrix::from_row_major_vec(
        3,
        2,
        vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(-0.25, 2.0),
            Complex64::new(1.5, -0.75),
            Complex64::new(0.75, 0.25),
            Complex64::new(-0.5, 0.125),
            Complex64::new(0.25, -1.25),
        ],
    );
    assert_complex_block_matches_vectors(&csr, &x, &adjoint_input);
}

#[test]
fn operator_combinators_preserve_complex_adjoint_contracts() {
    let a = square_complex_csr([
        Complex64::new(1.0, 2.0),
        Complex64::new(-2.0, 0.5),
        Complex64::new(0.25, -1.0),
        Complex64::new(3.0, -0.75),
    ]);
    let b = square_complex_csr([
        Complex64::new(-1.0, 0.25),
        Complex64::new(0.5, 1.5),
        Complex64::new(2.0, -0.5),
        Complex64::new(1.0, 1.0),
    ]);
    let sum = sonium_solvers_sparse::SumOperator::new(&a, &b);
    let scaled = sonium_solvers_sparse::ScaledOperator::new(&sum, Complex64::new(0.5, -1.0));
    let composite = sonium_solvers_sparse::CompositeOperator::new(&scaled, &a);
    let x = [Complex64::new(0.5, -1.0), Complex64::new(-1.25, 0.75)];
    let y = [Complex64::new(1.0, 0.5), Complex64::new(-0.25, 2.0)];

    for operator in [&sum as &dyn LinearOperator<Complex64>, &scaled, &composite] {
        assert_inner_product_adjoint_for_vectors(operator, &x, &y);
        assert_complex_block_matches_vectors(
            operator,
            &DenseMatrix::from_row_major_vec(
                2,
                2,
                vec![
                    Complex64::new(0.5, -1.0),
                    Complex64::new(1.0, 0.25),
                    Complex64::new(-1.25, 0.75),
                    Complex64::new(0.5, -0.5),
                ],
            ),
            &DenseMatrix::from_row_major_vec(
                2,
                2,
                vec![
                    Complex64::new(1.0, 0.5),
                    Complex64::new(-0.25, 2.0),
                    Complex64::new(1.5, -0.75),
                    Complex64::new(0.75, 0.25),
                ],
            ),
        );
    }

    let exact_matrix_free = FnOperator::new_into_with_adjoint(
        2,
        2,
        |x, y| {
            y[0] = x[0] + Complex64::new(2.0, 0.0) * x[1];
            y[1] = Complex64::new(-1.0, 0.5) * x[0] + x[1];
        },
        |x, y| {
            y[0] = x[0] + Complex64::new(-1.0, 0.5) * x[1];
            y[1] = Complex64::new(2.0, 0.0) * x[0] + x[1];
        },
        |x, y| {
            y[0] = x[0] + Complex64::new(-1.0, -0.5) * x[1];
            y[1] = Complex64::new(2.0, 0.0) * x[0] + x[1];
        },
    );
    assert_complex_block_matches_vectors(
        &exact_matrix_free,
        &DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(0.5, -1.0),
                Complex64::new(1.0, 0.25),
                Complex64::new(-1.25, 0.75),
                Complex64::new(0.5, -0.5),
            ],
        ),
        &DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(1.0, 0.5),
                Complex64::new(-0.25, 2.0),
                Complex64::new(1.5, -0.75),
                Complex64::new(0.75, 0.25),
            ],
        ),
    );

    let real = DenseBlockOperator::new(DenseMatrix::from_row_major_vec(
        2,
        3,
        vec![1.0, 2.0, 0.0, -1.0, 0.0, 3.0],
    ));
    let real_x = DenseMatrix::from_row_major_vec(3, 2, vec![1.0, -1.0, 2.0, 0.0, 3.0, 2.0]);
    let real_adjoint_input = DenseMatrix::from_row_major_vec(2, 2, vec![1.0, -2.0, 4.0, 1.0]);
    let mut real_forward = DenseMatrix::zeros(2, 2);
    real.try_apply_block(&real_x, &mut real_forward)
        .expect("real forward block");
    assert_eq!(real_forward.as_slice(), &[5.0, -1.0, 8.0, 7.0]);
    let mut real_transpose = DenseMatrix::zeros(3, 2);
    real.try_apply_transpose_block(&real_adjoint_input, &mut real_transpose)
        .expect("real transpose block");
    assert_eq!(
        real_transpose.as_slice(),
        &[-3.0, -3.0, 2.0, -4.0, 12.0, 3.0]
    );
    let mut real_hermitian = DenseMatrix::zeros(3, 2);
    real.try_apply_hermitian_block(&real_adjoint_input, &mut real_hermitian)
        .expect("real Hermitian block");
    assert_eq!(real_hermitian.as_slice(), real_transpose.as_slice());
}

#[test]
fn checked_block_adjoint_dispatch_rejects_bad_shapes_unknowns_panics_and_non_finite_outputs() {
    let operator = DenseBlockOperator::new(DenseMatrix::from_row_major_vec(
        2,
        3,
        vec![1.0, 2.0, 0.0, -1.0, 0.0, 3.0],
    ));
    let mut output = DenseMatrix::zeros(3, 2);
    assert!(matches!(
        operator.try_apply_transpose_block(&DenseMatrix::zeros(1, 2), &mut output),
        Err(SolverError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        operator
            .try_apply_transpose_block(&DenseMatrix::zeros(2, 2), &mut DenseMatrix::zeros(2, 2)),
        Err(SolverError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        operator
            .try_apply_hermitian_block(&DenseMatrix::zeros(2, 2), &mut DenseMatrix::zeros(3, 1)),
        Err(SolverError::DimensionMismatch { .. })
    ));

    let unknown = UnknownAdjointOperator;
    let mut unknown_output = DenseMatrix::zeros(1, 2);
    assert!(matches!(
        unknown.try_apply_transpose_block(&DenseMatrix::zeros(1, 2), &mut unknown_output),
        Err(SolverError::UnsupportedOperatorOperation { .. })
    ));

    let forward_only = FnOperator::<f64>::new_into(1, 1, |_x, _y| {
        panic!("unsupported adjoint callback must not be invoked")
    });
    let mut forward_only_output = DenseMatrix::zeros(1, 1);
    assert!(
        forward_only
            .try_apply_hermitian_block(&DenseMatrix::zeros(1, 1), &mut forward_only_output)
            .is_err()
    );

    let non_finite = FnOperator::new_into_with_adjoint(
        1,
        1,
        |_x, y| y[0] = 1.0,
        |_x, y| y[0] = f64::NAN,
        |_x, y| y[0] = 1.0,
    );
    let mut non_finite_output = DenseMatrix::zeros(1, 1);
    assert!(matches!(
        non_finite.try_apply_transpose_block(&DenseMatrix::zeros(1, 1), &mut non_finite_output),
        Err(SolverError::OperationFailed { .. })
    ));

    let panicking = FnOperator::new_into_with_adjoint(
        1,
        1,
        |_x, y| y[0] = 1.0,
        |_x, _y| panic!("transpose callback failure"),
        |_x, y| y[0] = 1.0,
    );
    let mut panicking_output = DenseMatrix::zeros(1, 1);
    assert!(matches!(
        panicking.try_apply_transpose_block(&DenseMatrix::zeros(1, 1), &mut panicking_output),
        Err(SolverError::OperationFailed { .. })
    ));
}

#[test]
fn block_operator_preserves_complex_adjoint_contract_and_rejects_unknown_blocks() {
    let layout = BlockLayout::new(vec![2, 1], vec![1, 2]);
    let block = BlockOperator::new(layout)
        .with_block(
            0,
            0,
            Box::new(CsrMatrix::from_raw_parts(
                2,
                1,
                vec![0, 1, 2],
                vec![0, 0],
                vec![Complex64::new(1.0, 0.5), Complex64::new(-2.0, 1.0)],
            )),
        )
        .with_block(
            0,
            1,
            Box::new(CsrMatrix::from_raw_parts(
                2,
                2,
                vec![0, 2, 4],
                vec![0, 1, 0, 1],
                vec![
                    Complex64::new(0.5, -1.0),
                    Complex64::new(2.0, 0.25),
                    Complex64::new(-1.0, 0.5),
                    Complex64::new(1.5, -0.75),
                ],
            )),
        )
        .with_block(
            1,
            0,
            Box::new(CsrMatrix::from_raw_parts(
                1,
                1,
                vec![0, 1],
                vec![0],
                vec![Complex64::new(3.0, -0.5)],
            )),
        )
        .with_block(
            1,
            1,
            Box::new(CsrMatrix::from_raw_parts(
                1,
                2,
                vec![0, 2],
                vec![0, 1],
                vec![Complex64::new(-0.25, 1.0), Complex64::new(0.75, -2.0)],
            )),
        );
    let x = [
        Complex64::new(0.5, -1.0),
        Complex64::new(-1.25, 0.75),
        Complex64::new(2.0, 0.25),
    ];
    let y = [
        Complex64::new(1.0, 0.5),
        Complex64::new(-0.25, 2.0),
        Complex64::new(1.5, -0.75),
    ];
    assert_inner_product_adjoint_for_vectors(&block, &x, &y);

    let forward_only = sonium_solvers_sparse::FnOperator::new(2, 1, |input: &[Complex64]| {
        vec![input[0], input[0]]
    });
    let incomplete = BlockOperator::new(BlockLayout::new(vec![2], vec![1])).with_block(
        0,
        0,
        Box::new(forward_only),
    );
    assert_eq!(
        incomplete.transpose_capability(),
        AdjointCapability::Unsupported
    );
    assert!(
        incomplete
            .try_apply_transpose(&[Complex64::new(1.0, 0.0); 2])
            .is_err()
    );
}

#[test]
fn randomized_complex_adjoint_matrix_covers_reordered_sparse_layouts() {
    let csr = fixture();
    let compact = CompactCsrMatrix::from_csr(&csr).expect("u32-compatible CSR");
    let bucketed = RowBucketedCsrMatrix::from_csr(&csr);
    let sell = SellCSigmaMatrix::from_csr(&csr, 2, 8);
    let operators: [(&str, &dyn LinearOperator<Complex64>); 4] = [
        ("CSR", &csr),
        ("compact CSR", &compact),
        ("row-bucketed CSR", &bucketed),
        ("SELL-C-sigma", &sell),
    ];

    for (name, operator) in operators {
        assert_randomized_adjoint_contract(name, operator);
    }
}

#[test]
fn randomized_complex_adjoint_matrix_covers_composed_and_matrix_free_operators() {
    let a = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(-2.0, 0.5),
            Complex64::new(0.25, -1.0),
            Complex64::new(3.0, -0.75),
        ],
    );
    let b = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            Complex64::new(-1.0, 0.25),
            Complex64::new(0.5, 1.5),
            Complex64::new(2.0, -0.5),
            Complex64::new(1.0, 1.0),
        ],
    );
    let sum = sonium_solvers_sparse::SumOperator::new(&a, &b);
    let scaled = sonium_solvers_sparse::ScaledOperator::new(&sum, Complex64::new(0.5, -1.0));
    let composite = sonium_solvers_sparse::CompositeOperator::new(&scaled, &a);
    let matrix_free = FnOperator::new_into_with_adjoint(
        2,
        2,
        |x, y| {
            y[0] = x[0] + Complex64::new(2.0, 0.0) * x[1];
            y[1] = Complex64::new(-1.0, 0.5) * x[0] + x[1];
        },
        |x, y| {
            y[0] = x[0] + Complex64::new(-1.0, 0.5) * x[1];
            y[1] = Complex64::new(2.0, 0.0) * x[0] + x[1];
        },
        |x, y| {
            y[0] = x[0] + Complex64::new(-1.0, -0.5) * x[1];
            y[1] = Complex64::new(2.0, 0.0) * x[0] + x[1];
        },
    );
    let dense = DenseBlockOperator::new(DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.25),
            Complex64::new(-0.5, 1.0),
            Complex64::new(2.0, -0.75),
            Complex64::new(0.25, 0.5),
        ],
    ));
    let operators: [(&str, &dyn LinearOperator<Complex64>); 5] = [
        ("sum", &sum),
        ("scaled sum", &scaled),
        ("composite", &composite),
        ("matrix-free", &matrix_free),
        ("dense block", &dense),
    ];
    for (name, operator) in operators {
        assert_randomized_adjoint_contract(name, operator);
    }
}
