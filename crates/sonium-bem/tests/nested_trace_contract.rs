//! Nested and multi-region matrix-free trace-operator contracts.

use num_complex::Complex64;
use sonium_bem::{
    ConservativeTraceTransfer, CouplingConvention, CouplingError, NestedTraceEdge,
    NestedTraceOperator, NestedTracePreconditioner, NestedTracePreconditionerKind,
    NestedTraceRegion,
};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::{BlockGmresConfig, GmresConfig};
use sonium_solvers::sparse::DenseBlockOperator;
use sonium_solvers::traits::{AdjointCapability, LinearOperator, Preconditioner};

fn matrix(rows: usize, cols: usize, values: &[Complex64]) -> DenseMatrix<Complex64> {
    DenseMatrix::from_row_major_vec(rows, cols, values.to_vec())
}

fn transfer_a() -> ConservativeTraceTransfer {
    ConservativeTraceTransfer::try_new(
        vec![vec![(0, 1.0 / 3.0), (1, 2.0 / 3.0)]],
        vec![1.0, 2.0],
        vec![3.0],
    )
    .expect("conservative two-to-one transfer")
}

fn transfer_b() -> ConservativeTraceTransfer {
    ConservativeTraceTransfer::try_new(
        vec![vec![(0, 1.0)], vec![(0, 1.0)]],
        vec![3.0],
        vec![1.0, 2.0],
    )
    .expect("conservative one-to-two transfer")
}

fn dense_transpose(matrix: &DenseMatrix<Complex64>, input: &[Complex64]) -> Vec<Complex64> {
    (0..matrix.ncols())
        .map(|column| {
            (0..matrix.nrows())
                .map(|row| matrix[[row, column]] * input[row])
                .sum()
        })
        .collect()
}

fn dense_hermitian(matrix: &DenseMatrix<Complex64>, input: &[Complex64]) -> Vec<Complex64> {
    (0..matrix.ncols())
        .map(|column| {
            (0..matrix.nrows())
                .map(|row| matrix[[row, column]].conj() * input[row])
                .sum()
        })
        .collect()
}

fn assert_close(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    let error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max);
    assert!(error < 1.0e-12, "maximum error {error:e}");
}

struct ScalePreconditioner {
    scale: Complex64,
}

impl Preconditioner<Complex64> for ScalePreconditioner {
    fn apply(&self, rhs: &[Complex64]) -> Vec<Complex64> {
        rhs.iter().map(|value| *value * self.scale).collect()
    }

    fn apply_transpose(&self, rhs: &[Complex64]) -> Vec<Complex64> {
        self.apply(rhs)
    }

    fn apply_hermitian(&self, rhs: &[Complex64]) -> Vec<Complex64> {
        rhs.iter().map(|value| *value * self.scale.conj()).collect()
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

fn operator() -> NestedTraceOperator {
    let regions = vec![
        NestedTraceRegion::new(
            "core",
            Box::new(DenseBlockOperator::new(matrix(
                2,
                2,
                &[
                    Complex64::new(2.0, 0.2),
                    Complex64::new(-0.3, 0.1),
                    Complex64::new(0.7, -0.4),
                    Complex64::new(1.4, 0.3),
                ],
            ))),
        ),
        NestedTraceRegion::new(
            "shell",
            Box::new(DenseBlockOperator::new(matrix(
                1,
                1,
                &[Complex64::new(1.2, -0.5)],
            ))),
        ),
        NestedTraceRegion::new(
            "exterior",
            Box::new(DenseBlockOperator::new(matrix(
                2,
                2,
                &[
                    Complex64::new(1.1, 0.0),
                    Complex64::new(0.2, -0.2),
                    Complex64::new(-0.4, 0.6),
                    Complex64::new(0.8, 0.1),
                ],
            ))),
        ),
    ];
    let edges = vec![
        NestedTraceEdge::new("core-shell", 0, 1, transfer_a(), Complex64::new(0.4, -0.1)),
        NestedTraceEdge::new(
            "shell-exterior",
            1,
            2,
            transfer_b(),
            Complex64::new(-0.2, 0.3),
        ),
    ];
    NestedTraceOperator::try_new(
        regions,
        edges,
        CouplingConvention {
            bem: sonium_bem::ACOUSTIC_CONVENTION,
            normal_points_out_of_interior: true,
        },
    )
    .expect("nested operator")
}

#[test]
fn nested_actions_match_materialized_dense_reference() {
    let operator = operator();
    assert_eq!(operator.region_count(), 3);
    assert_eq!(operator.edge_count(), 2);
    assert_eq!(operator.dimension(), 5);
    assert_eq!(operator.region_offsets(), &[0, 2, 3, 5]);
    assert_eq!(
        operator.region_names().collect::<Vec<_>>(),
        ["core", "shell", "exterior"]
    );

    let state = vec![
        Complex64::new(0.2, -0.5),
        Complex64::new(-0.7, 0.3),
        Complex64::new(0.4, 0.9),
        Complex64::new(-0.1, 0.2),
        Complex64::new(0.8, -0.4),
    ];
    let dense = operator.to_dense_matrix().expect("dense reference");
    assert_close(
        &operator.try_apply(&state).expect("forward"),
        &dense.matvec(&state),
    );
    assert_close(
        &operator.try_apply_transpose(&state).expect("transpose"),
        &dense_transpose(&dense, &state),
    );
    assert_close(
        &operator.try_apply_hermitian(&state).expect("Hermitian"),
        &dense_hermitian(&dense, &state),
    );

    let block = DenseMatrix::from_row_major_vec(
        5,
        2,
        vec![
            state[0],
            Complex64::new(-0.2, 0.1),
            state[1],
            Complex64::new(0.5, -0.8),
            state[2],
            Complex64::new(0.3, 0.4),
            state[3],
            Complex64::new(-0.9, 0.2),
            state[4],
            Complex64::new(0.1, 0.7),
        ],
    );
    let applied = operator.try_apply_block(&block).expect("forward block");
    for column in 0..block.ncols() {
        assert_close(
            &applied.column_to_vec(column),
            &operator
                .try_apply(&block.column_to_vec(column))
                .expect("scalar forward"),
        );
    }
}

#[test]
fn nested_row_actions_preserve_selected_order_and_block_columns() {
    let operator = operator();
    let state = vec![
        Complex64::new(0.2, -0.5),
        Complex64::new(-0.7, 0.3),
        Complex64::new(0.4, 0.9),
        Complex64::new(-0.1, 0.2),
        Complex64::new(0.8, -0.4),
    ];
    let rows = [4, 0, 4, 2];
    let forward = operator.try_apply(&state).expect("forward");
    let transpose = operator.try_apply_transpose(&state).expect("transpose");
    let hermitian = operator.try_apply_hermitian(&state).expect("Hermitian");
    assert_close(
        &operator
            .try_apply_rows(&state, &rows)
            .expect("forward rows"),
        &rows.iter().map(|&row| forward[row]).collect::<Vec<_>>(),
    );
    assert_close(
        &operator
            .try_apply_transpose_rows(&state, &rows)
            .expect("transpose rows"),
        &rows.iter().map(|&row| transpose[row]).collect::<Vec<_>>(),
    );
    assert_close(
        &operator
            .try_apply_hermitian_rows(&state, &rows)
            .expect("Hermitian rows"),
        &rows.iter().map(|&row| hermitian[row]).collect::<Vec<_>>(),
    );

    let block = DenseMatrix::from_row_major_vec(
        5,
        2,
        vec![
            state[0],
            Complex64::new(-0.2, 0.1),
            state[1],
            Complex64::new(0.5, -0.8),
            state[2],
            Complex64::new(0.3, 0.4),
            state[3],
            Complex64::new(-0.9, 0.2),
            state[4],
            Complex64::new(0.1, 0.7),
        ],
    );
    let selected = operator
        .try_apply_rows_block(&block, &rows)
        .expect("forward row block");
    let full = operator.try_apply_block(&block).expect("forward block");
    for (selected_row, &global_row) in rows.iter().enumerate() {
        for column in 0..block.ncols() {
            assert_eq!(selected[[selected_row, column]], full[[global_row, column]]);
        }
    }
    for (selected, full) in [
        (
            operator
                .try_apply_transpose_rows_block(&block, &rows)
                .expect("transpose row block"),
            operator
                .try_apply_transpose_block(&block)
                .expect("transpose block"),
        ),
        (
            operator
                .try_apply_hermitian_rows_block(&block, &rows)
                .expect("Hermitian row block"),
            operator
                .try_apply_hermitian_block(&block)
                .expect("Hermitian block"),
        ),
    ] {
        for (selected_row, &global_row) in rows.iter().enumerate() {
            for column in 0..block.ncols() {
                assert_eq!(selected[[selected_row, column]], full[[global_row, column]]);
            }
        }
    }
    assert!(
        operator
            .try_apply_rows(&state, &[operator.dimension()])
            .is_err()
    );
    assert!(
        operator
            .try_apply_rows_block(&block, &[operator.dimension()])
            .is_err()
    );
}

#[test]
fn nested_constructor_rejects_bad_graph_edges() {
    let region = |name: &str| {
        NestedTraceRegion::new(
            name,
            Box::new(DenseBlockOperator::new(matrix(
                1,
                1,
                &[Complex64::new(1.0, 0.0)],
            ))),
        )
    };
    let bad_transfer =
        ConservativeTraceTransfer::try_new(vec![vec![(0, 1.0)]], vec![1.0], vec![1.0])
            .expect("transfer");
    let self_edge =
        NestedTraceEdge::new("self", 0, 0, bad_transfer.clone(), Complex64::new(1.0, 0.0));
    assert!(matches!(
        NestedTraceOperator::try_new(
            vec![region("single")],
            vec![self_edge],
            CouplingConvention {
                bem: sonium_bem::ACOUSTIC_CONVENTION,
                normal_points_out_of_interior: true,
            },
        ),
        Err(CouplingError::InvalidEdge { .. })
    ));

    let out_of_range =
        NestedTraceEdge::new("missing", 0, 2, bad_transfer, Complex64::new(1.0, 0.0));
    assert!(matches!(
        NestedTraceOperator::try_new(
            vec![region("first"), region("second")],
            vec![out_of_range],
            CouplingConvention {
                bem: sonium_bem::ACOUSTIC_CONVENTION,
                normal_points_out_of_interior: true,
            },
        ),
        Err(CouplingError::InvalidEdge { .. })
    ));
}

#[test]
fn nested_operator_exposes_exact_adjoint_capabilities() {
    let operator = operator();
    assert_eq!(
        operator.transpose_capability(),
        sonium_solvers::traits::AdjointCapability::Exact
    );
    assert_eq!(
        operator.hermitian_capability(),
        sonium_solvers::traits::AdjointCapability::Exact
    );
    let input = vec![Complex64::new(1.0, 0.0); 5];
    let mut output = vec![Complex64::default(); 5];
    LinearOperator::apply_into(&operator, &input, &mut output);
    assert_close(&output, &operator.try_apply(&input).expect("forward"));
}

#[test]
fn nested_preconditioner_sweeps_match_materialized_dense_actions() {
    let state = vec![
        Complex64::new(0.2, -0.5),
        Complex64::new(-0.7, 0.3),
        Complex64::new(0.4, 0.9),
        Complex64::new(-0.1, 0.2),
        Complex64::new(0.8, -0.4),
    ];
    let block = DenseMatrix::from_row_major_vec(
        5,
        2,
        vec![
            state[0],
            Complex64::new(-0.2, 0.1),
            state[1],
            Complex64::new(0.5, -0.8),
            state[2],
            Complex64::new(0.3, 0.4),
            state[3],
            Complex64::new(-0.9, 0.2),
            state[4],
            Complex64::new(0.1, 0.7),
        ],
    );
    for kind in [
        NestedTracePreconditionerKind::BlockDiagonal,
        NestedTracePreconditionerKind::ForwardSweep,
        NestedTracePreconditionerKind::ReverseSweep,
    ] {
        let preconditioner = NestedTracePreconditioner::try_new_with_kind(
            vec![
                Box::new(ScalePreconditioner {
                    scale: Complex64::new(0.5, 0.1),
                }),
                Box::new(ScalePreconditioner {
                    scale: Complex64::new(0.25, -0.2),
                }),
                Box::new(ScalePreconditioner {
                    scale: Complex64::new(0.75, 0.05),
                }),
            ],
            vec![2, 1, 2],
            vec![
                NestedTraceEdge::new("core-shell", 0, 1, transfer_a(), Complex64::new(0.4, -0.1)),
                NestedTraceEdge::new(
                    "shell-exterior",
                    1,
                    2,
                    transfer_b(),
                    Complex64::new(-0.2, 0.3),
                ),
            ],
            kind,
            CouplingConvention {
                bem: sonium_bem::ACOUSTIC_CONVENTION,
                normal_points_out_of_interior: true,
            },
        )
        .expect("nested preconditioner");

        let mut dense = DenseMatrix::from_fn(5, 5, |_, _| Complex64::default());
        for column in 0..5 {
            let mut basis = vec![Complex64::default(); 5];
            basis[column] = Complex64::new(1.0, 0.0);
            dense.set_column(
                column,
                &preconditioner
                    .try_apply_checked(&basis)
                    .expect("preconditioner basis action"),
            );
        }
        assert_close(
            &preconditioner.try_apply_checked(&state).expect("forward"),
            &dense.matvec(&state),
        );
        assert_close(
            &preconditioner
                .try_apply_transpose_checked(&state)
                .expect("transpose"),
            &dense_transpose(&dense, &state),
        );
        assert_close(
            &preconditioner
                .try_apply_hermitian_checked(&state)
                .expect("Hermitian"),
            &dense_hermitian(&dense, &state),
        );
        let output = preconditioner.try_apply_block(&block).expect("block");
        for column in 0..block.ncols() {
            assert_close(
                &output.column_to_vec(column),
                &preconditioner
                    .try_apply_checked(&block.column_to_vec(column))
                    .expect("scalar block action"),
            );
        }
    }
}

#[test]
fn nested_direct_solves_round_trip_forward_hermitian_and_block_rhs() {
    let operator = operator();
    let state = vec![
        Complex64::new(0.2, -0.5),
        Complex64::new(-0.7, 0.3),
        Complex64::new(0.4, 0.9),
        Complex64::new(-0.1, 0.2),
        Complex64::new(0.8, -0.4),
    ];
    let rhs = operator.try_apply(&state).expect("forward rhs");
    let solved = operator.solve(&rhs).expect("forward direct solve");
    assert_close(&solved, &state);
    assert!(operator.relative_residual(&solved, &rhs).unwrap() < 1.0e-12);

    let hermitian_rhs = operator.try_apply_hermitian(&state).expect("Hermitian rhs");
    let hermitian_solved = operator
        .solve_hermitian(&hermitian_rhs)
        .expect("Hermitian direct solve");
    assert_close(&hermitian_solved, &state);
    assert!(
        operator
            .relative_hermitian_residual(&hermitian_solved, &hermitian_rhs)
            .unwrap()
            < 1.0e-12
    );

    let transpose_rhs = operator.try_apply_transpose(&state).expect("transpose rhs");
    let transpose_solved = operator
        .solve_transpose(&transpose_rhs)
        .expect("transpose direct solve");
    assert_close(&transpose_solved, &state);
    assert!(
        operator
            .relative_transpose_residual(&transpose_solved, &transpose_rhs)
            .unwrap()
            < 1.0e-12
    );

    let rhs_block = DenseMatrix::from_row_major_vec(
        5,
        2,
        vec![
            rhs[0],
            hermitian_rhs[0],
            rhs[1],
            hermitian_rhs[1],
            rhs[2],
            hermitian_rhs[2],
            rhs[3],
            hermitian_rhs[3],
            rhs[4],
            hermitian_rhs[4],
        ],
    );
    let solved_block = operator
        .solve_block(&rhs_block)
        .expect("forward block solve");
    let solved_hermitian_block = operator
        .solve_hermitian_block(&rhs_block)
        .expect("Hermitian block solve");
    let transpose_rhs_block = DenseMatrix::from_row_major_vec(5, 1, transpose_rhs.clone());
    let solved_transpose_block = operator
        .solve_transpose_block(&transpose_rhs_block)
        .expect("transpose block solve");
    assert_close(&solved_block.column_to_vec(0), &state);
    assert_close(&solved_hermitian_block.column_to_vec(1), &state);
    assert_close(&solved_transpose_block.column_to_vec(0), &state);
}

#[test]
fn nested_matrix_free_gmres_round_trips_all_adjoint_actions_and_blocks() {
    let operator = operator();
    let preconditioner = ScalePreconditioner {
        scale: Complex64::new(1.0, 0.0),
    };
    let config = GmresConfig {
        max_iterations: 8,
        restart: 5,
        tolerance: 1.0e-12,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
        min_restart: None,
        max_restart: None,
    };
    let state = vec![
        Complex64::new(0.2, -0.5),
        Complex64::new(-0.7, 0.3),
        Complex64::new(0.4, 0.9),
        Complex64::new(-0.1, 0.2),
        Complex64::new(0.8, -0.4),
    ];

    let rhs = operator.try_apply(&state).expect("forward rhs");
    let solved = operator
        .try_gmres_solve(&rhs, &preconditioner, &config)
        .expect("matrix-free forward GMRES");
    assert!(solved.converged);
    assert!(operator.relative_residual(&solved.x, &rhs).unwrap() < 1.0e-10);

    let transpose_rhs = operator.try_apply_transpose(&state).expect("transpose rhs");
    let transpose_solved = operator
        .try_gmres_solve_transpose(&transpose_rhs, &preconditioner, &config)
        .expect("matrix-free transpose GMRES");
    assert!(transpose_solved.converged);
    assert!(
        operator
            .relative_transpose_residual(&transpose_solved.x, &transpose_rhs)
            .unwrap()
            < 1.0e-10
    );

    let hermitian_rhs = operator.try_apply_hermitian(&state).expect("Hermitian rhs");
    let hermitian_solved = operator
        .try_gmres_solve_hermitian(&hermitian_rhs, &preconditioner, &config)
        .expect("matrix-free Hermitian GMRES");
    assert!(hermitian_solved.converged);
    assert!(
        operator
            .relative_hermitian_residual(&hermitian_solved.x, &hermitian_rhs)
            .unwrap()
            < 1.0e-10
    );

    let rhs_block = DenseMatrix::from_row_major_vec(
        operator.dimension(),
        2,
        vec![
            rhs[0],
            Complex64::new(0.3, 0.1),
            rhs[1],
            Complex64::new(-0.2, 0.7),
            rhs[2],
            Complex64::new(0.8, -0.4),
            rhs[3],
            Complex64::new(-0.5, 0.2),
            rhs[4],
            Complex64::new(0.1, -0.9),
        ],
    );
    let block_config = BlockGmresConfig {
        max_iterations: 8,
        restart: 5,
        tolerance: 1.0e-12,
        true_residual_check: true,
        exact_residual_check_interval: 1,
    };
    let block_solved = operator
        .try_block_gmres_solve(&rhs_block, &preconditioner, &block_config)
        .expect("matrix-free block GMRES");
    assert!(block_solved.converged);
    for column in 0..rhs_block.ncols() {
        assert!(
            operator
                .relative_residual(
                    &block_solved.x.column_to_vec(column),
                    &rhs_block.column_to_vec(column)
                )
                .unwrap()
                < 1.0e-10
        );
    }
}

#[test]
fn nested_matrix_free_gmres_rejects_bad_rhs_and_configuration() {
    let operator = operator();
    let preconditioner = ScalePreconditioner {
        scale: Complex64::new(1.0, 0.0),
    };
    let config = GmresConfig::default();
    assert!(matches!(
        operator.try_gmres_solve(&[Complex64::new(1.0, 0.0)], &preconditioner, &config),
        Err(CouplingError::Dimension { .. })
    ));

    let mut rhs = vec![Complex64::new(0.0, 0.0); operator.dimension()];
    rhs[1] = Complex64::new(f64::NAN, 0.0);
    assert!(matches!(
        operator.try_gmres_solve(&rhs, &preconditioner, &config),
        Err(CouplingError::NonFinite { .. })
    ));

    let invalid_config = GmresConfig {
        max_iterations: 0,
        ..config
    };
    let finite_rhs = vec![Complex64::new(0.0, 0.0); operator.dimension()];
    assert!(matches!(
        operator.try_gmres_solve(&finite_rhs, &preconditioner, &invalid_config),
        Err(CouplingError::OperatorFailure { .. })
    ));
}

#[test]
fn nested_direct_solves_reject_bad_rhs_and_singular_operator() {
    let operator = operator();
    assert!(matches!(
        operator.solve(&[Complex64::new(1.0, 0.0)]),
        Err(CouplingError::Dimension { .. })
    ));
    let mut non_finite_rhs = vec![Complex64::new(0.0, 0.0); 5];
    non_finite_rhs[2] = Complex64::new(f64::NAN, 0.0);
    assert!(matches!(
        operator.solve(&non_finite_rhs),
        Err(CouplingError::NonFinite { .. })
    ));

    let singular = NestedTraceOperator::try_new(
        vec![NestedTraceRegion::new(
            "singular",
            Box::new(DenseBlockOperator::new(matrix(
                1,
                1,
                &[Complex64::new(0.0, 0.0)],
            ))),
        )],
        Vec::new(),
        CouplingConvention {
            bem: sonium_bem::ACOUSTIC_CONVENTION,
            normal_points_out_of_interior: true,
        },
    )
    .expect("singular operator construction");
    assert!(matches!(
        singular.solve(&[Complex64::new(1.0, 0.0)]),
        Err(CouplingError::OperatorFailure { .. })
    ));
}
