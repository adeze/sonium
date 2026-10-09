//! Matrix-free transmission reciprocity and power contracts.

use num_complex::Complex64;
use sonium_bem::core::assembly::build_transmission_tbem_system;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::{
    ConservativeTraceTransfer, CoupledPreconditionerKind, CoupledTraceOperator,
    CoupledTracePreconditioner, CoupledTraceSystem, CouplingError,
    check_transmission_passivity_contract, check_transmission_power_balance,
    check_transmission_reciprocity_operator,
};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::sparse::{CsrMatrix, DenseBlockOperator};
use sonium_solvers::traits::{AdjointCapability, LinearOperator, Preconditioner};

fn matrix_free_fixture() -> CoupledTraceOperator {
    let transfer = ConservativeTraceTransfer::try_new(
        vec![vec![(0, 1.0)], vec![(1, 1.0)]],
        vec![1.0, 1.0],
        vec![1.0, 1.0],
    )
    .expect("identity conservative transfer");
    let interior = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, Complex64::new(2.0, 0.15)),
            (1, 1, Complex64::new(2.7, -0.2)),
        ],
    );
    let exterior = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(1.6, -0.1),
            Complex64::new(0.12, 0.04),
            Complex64::new(-0.08, 0.03),
            Complex64::new(2.4, 0.25),
        ],
    );
    CoupledTraceOperator::try_new(
        Box::new(interior),
        Box::new(DenseBlockOperator::new(exterior)),
        transfer,
        Complex64::new(0.2, -0.08),
    )
    .expect("matrix-free coupled fixture")
}

fn probes(dimension: usize) -> Vec<Vec<Complex64>> {
    (0..3)
        .map(|seed| {
            (0..dimension)
                .map(|index| {
                    let phase = (index + 1 + seed * 5) as f64;
                    Complex64::new(0.15 + 0.013 * phase, -0.2 + 0.009 * phase)
                })
                .collect()
        })
        .collect()
}

#[test]
fn matrix_free_coupled_operator_block_action_matches_scalar_columns() {
    let operator = matrix_free_fixture();
    let states = DenseMatrix::from_row_major_vec(
        4,
        2,
        vec![
            Complex64::new(0.4, -0.6),
            Complex64::new(-0.2, 0.8),
            Complex64::new(0.9, 0.1),
            Complex64::new(-0.7, -0.3),
            Complex64::new(-0.15, 0.25),
            Complex64::new(0.55, -0.4),
            Complex64::new(0.3, 0.75),
            Complex64::new(-0.8, 0.1),
        ],
    );
    let actual = operator
        .try_apply_block(&states)
        .expect("checked coupled block action");
    let expected = DenseMatrix::from_fn(4, 2, |row, column| {
        operator
            .try_apply(&states.column_to_vec(column))
            .expect("checked scalar action")[row]
    });
    assert_close(actual.as_slice(), expected.as_slice());

    let wrong_rows = DenseMatrix::zeros(3, 2);
    assert!(matches!(
        operator.try_apply_block(&wrong_rows),
        Err(CouplingError::Dimension { .. })
    ));
    let non_finite = DenseMatrix::from_row_major_vec(
        4,
        1,
        vec![
            Complex64::new(f64::NAN, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );
    assert!(matches!(
        operator.try_apply_block(&non_finite),
        Err(CouplingError::NonFinite { .. })
    ));
}

#[test]
fn matrix_free_coupled_operator_block_gmres_reports_true_residuals() {
    let operator = matrix_free_fixture();
    let expected = DenseMatrix::from_row_major_vec(
        4,
        2,
        vec![
            Complex64::new(0.4, -0.6),
            Complex64::new(-0.2, 0.8),
            Complex64::new(0.9, 0.1),
            Complex64::new(-0.7, -0.3),
            Complex64::new(-0.15, 0.25),
            Complex64::new(0.55, -0.4),
            Complex64::new(0.3, 0.75),
            Complex64::new(-0.8, 0.1),
        ],
    );
    let rhs = operator
        .try_apply_block(&expected)
        .expect("fixture right-hand side");
    let config = sonium_solvers::iterative::BlockGmresConfig {
        max_iterations: 8,
        restart: 4,
        tolerance: 1.0e-12,
        true_residual_check: true,
        exact_residual_check_interval: 1,
    };
    let solution = operator
        .try_block_gmres(&rhs, &config)
        .expect("matrix-free coupled block GMRES");
    assert!(solution.converged);
    assert!(
        solution
            .true_residuals
            .iter()
            .all(|residual| *residual < 1.0e-10)
    );
    for column in 0..expected.ncols() {
        assert_close(
            &solution.x.column_to_vec(column),
            &expected.column_to_vec(column),
        );
    }

    let preconditioner = sonium_solvers::traits::IdentityPreconditioner;
    let preconditioned = operator
        .try_block_gmres_preconditioned(&rhs, &preconditioner, &config)
        .expect("matrix-free preconditioned block GMRES");
    assert!(preconditioned.converged);
    assert!(
        preconditioned
            .true_residuals
            .iter()
            .all(|residual| *residual < 1.0e-10)
    );
}

#[test]
fn matrix_free_coupled_operator_block_gmres_rejects_bad_rhs() {
    let operator = matrix_free_fixture();
    let config = sonium_solvers::iterative::BlockGmresConfig::default();
    let wrong_rows = DenseMatrix::zeros(3, 2);
    assert!(matches!(
        operator.try_block_gmres(&wrong_rows, &config),
        Err(CouplingError::Dimension { .. })
    ));
    let non_finite = DenseMatrix::from_row_major_vec(
        4,
        1,
        vec![
            Complex64::new(f64::NAN, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );
    assert!(matches!(
        operator.try_block_gmres(&non_finite, &config),
        Err(CouplingError::NonFinite { .. })
    ));
}

#[test]
fn compressed_transmission_satisfies_weighted_reciprocity_without_dense_action() {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let exterior = sonium_bem::PhysicsParams::new(420.0, 343.0, 1.21, false);
    let interior = sonium_bem::PhysicsParams::new(420.0, 1_480.0, 1_000.0, true);
    let system =
        build_transmission_tbem_system(&mesh.elements, &mesh.nodes, exterior.clone(), interior)
            .expect("production transmission assembly");
    let compressed = system
        .compress(8, 1.0e-6, 8)
        .expect("matrix-free transmission compression");
    assert!(compressed.compressed_block_count() > 0);

    let report = check_transmission_reciprocity_operator(
        &compressed,
        system.interface_dofs(),
        exterior.omega,
        exterior.harmonic_factor,
        &probes(system.total_dofs()),
        5.0e-4,
    )
    .expect("compressed weighted reciprocity");
    assert_eq!(report.pairs, 3);
    assert!(report.relative_error <= 5.0e-4, "{report:?}");
}

#[test]
fn transmission_power_balance_rejects_nonconservative_trace() {
    let areas = [0.5, 0.75];
    let pressure = [Complex64::new(2.0, 1.0), Complex64::new(1.0, -0.5)];
    let velocity = [Complex64::new(0.1, -0.2), Complex64::new(0.2, 0.1)];
    let balanced = check_transmission_power_balance(
        &areas, &pressure, &velocity, &pressure, &velocity, 1.0e-12,
    )
    .expect("identical traces should conserve power");
    assert!(balanced.relative_error <= 1.0e-12);

    let mismatched_velocity = [velocity[0], velocity[1] + Complex64::new(0.25, 0.0)];
    assert!(
        check_transmission_power_balance(
            &areas,
            &pressure,
            &velocity,
            &pressure,
            &mismatched_velocity,
            1.0e-12,
        )
        .is_err()
    );
}

#[test]
fn matrix_free_coupled_operator_matches_dense_forward_and_adjoint_actions() {
    let transfer = ConservativeTraceTransfer::try_new(
        vec![vec![(0, 0.25), (1, 0.75)], vec![(0, 0.75), (1, 0.25)]],
        vec![1.25, 1.75],
        vec![2.0, 1.0],
    )
    .expect("conservative transfer");
    let interior = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(2.0, 0.5),
            Complex64::new(-0.2, 0.1),
            Complex64::new(0.4, -0.3),
            Complex64::new(1.5, -0.25),
        ],
    );
    let exterior = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(0.8, -0.1),
            Complex64::new(0.3, 0.2),
            Complex64::new(-0.1, 0.4),
            Complex64::new(1.2, 0.6),
        ],
    );
    let dense = CoupledTraceSystem::try_new(
        interior.clone(),
        exterior.clone(),
        transfer.clone(),
        Complex64::new(0.7, -0.2),
    )
    .expect("dense coupled system");
    let fem = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, Complex64::new(2.0, 0.5)),
            (0, 1, Complex64::new(-0.2, 0.1)),
            (1, 0, Complex64::new(0.4, -0.3)),
            (1, 1, Complex64::new(1.5, -0.25)),
        ],
    );
    let operator = CoupledTraceOperator::try_new(
        Box::new(fem),
        Box::new(DenseBlockOperator::new(exterior)),
        transfer,
        Complex64::new(0.7, -0.2),
    )
    .expect("matrix-free coupled operator");
    let state = vec![
        Complex64::new(0.4, -0.6),
        Complex64::new(-0.2, 0.8),
        Complex64::new(0.9, 0.1),
        Complex64::new(-0.7, -0.3),
    ];
    let matrix = dense.to_dense_matrix().expect("dense materialization");
    let expected_forward = matrix.matvec(&state);
    let expected_transpose = dense_matrix_transpose_matvec(&matrix, &state);
    let expected_hermitian = dense_matrix_hermitian_matvec(&matrix, &state);
    assert_close(
        &operator.try_apply(&state).expect("forward"),
        &expected_forward,
    );
    assert_close(
        &operator.try_apply_transpose(&state).expect("transpose"),
        &expected_transpose,
    );
    assert_close(
        &operator.try_apply_hermitian(&state).expect("Hermitian"),
        &expected_hermitian,
    );
    assert_eq!(
        operator.transpose_capability(),
        sonium_solvers::traits::AdjointCapability::Exact
    );
    assert_eq!(
        operator.hermitian_capability(),
        sonium_solvers::traits::AdjointCapability::Exact
    );
}

#[test]
fn matrix_free_coupled_operator_rejects_unknown_diagonal_adjoint() {
    struct UnknownAdjointOperator;

    impl LinearOperator<Complex64> for UnknownAdjointOperator {
        fn num_rows(&self) -> usize {
            1
        }

        fn num_cols(&self) -> usize {
            1
        }

        fn apply_into(&self, input: &[Complex64], output: &mut [Complex64]) {
            output[0] = input[0];
        }

        fn apply_transpose_into(&self, input: &[Complex64], output: &mut [Complex64]) {
            output[0] = input[0];
        }
    }

    let transfer = ConservativeTraceTransfer::try_new(vec![vec![(0, 1.0)]], vec![1.0], vec![1.0])
        .expect("conservative transfer");
    let result = CoupledTraceOperator::try_new(
        Box::new(UnknownAdjointOperator),
        Box::new(UnknownAdjointOperator),
        transfer,
        Complex64::new(1.0, 0.0),
    );
    assert!(matches!(
        result,
        Err(CouplingError::UnsupportedAdjoint {
            block: "interior",
            operation: "transpose",
        })
    ));
}

#[test]
fn coupled_trace_preconditioner_variants_match_dense_adjoint_actions() {
    let transfer = ConservativeTraceTransfer::try_new(
        vec![vec![(0, 0.25), (1, 0.75)], vec![(0, 0.75), (1, 0.25)]],
        vec![1.25, 1.75],
        vec![2.0, 1.0],
    )
    .expect("conservative transfer");
    let state = vec![
        Complex64::new(0.25, -0.5),
        Complex64::new(-1.0, 0.75),
        Complex64::new(0.5, 0.125),
        Complex64::new(-0.25, 1.0),
    ];
    let block = DenseMatrix::from_row_major_vec(
        4,
        2,
        vec![
            state[0],
            Complex64::new(-0.5, 0.25),
            state[1],
            Complex64::new(1.25, -0.75),
            state[2],
            Complex64::new(-1.0, 0.5),
            state[3],
            Complex64::new(0.75, 0.0),
        ],
    );

    for kind in [
        CoupledPreconditionerKind::BlockDiagonal,
        CoupledPreconditionerKind::LowerTriangular,
        CoupledPreconditionerKind::UpperTriangular,
    ] {
        let preconditioner = CoupledTracePreconditioner::try_new_with_kind(
            Box::new(ScalePreconditioner {
                scale: Complex64::new(0.5, 0.1),
            }),
            Box::new(ScalePreconditioner {
                scale: Complex64::new(0.25, -0.2),
            }),
            transfer.clone(),
            Complex64::new(0.3, -0.15),
            kind,
        )
        .expect("checked coupled preconditioner");

        let dense = (0..4)
            .map(|column| {
                let mut basis = vec![Complex64::new(0.0, 0.0); 4];
                basis[column] = Complex64::new(1.0, 0.0);
                preconditioner.apply_checked(&basis).expect("basis action")
            })
            .collect::<Vec<_>>();
        let dense_matrix = DenseMatrix::from_fn(4, 4, |row, column| dense[column][row]);
        assert_close(
            &preconditioner
                .apply_checked(&state)
                .expect("forward action"),
            &dense_matrix.matvec(&state),
        );
        assert_close(
            &preconditioner
                .apply_transpose_checked(&state)
                .expect("transpose action"),
            &dense_matrix_transpose_matvec(&dense_matrix, &state),
        );
        assert_close(
            &preconditioner
                .apply_hermitian_checked(&state)
                .expect("Hermitian action"),
            &dense_matrix_hermitian_matvec(&dense_matrix, &state),
        );

        let mut output = DenseMatrix::zeros(4, 2);
        preconditioner
            .try_apply_block(&block, &mut output)
            .expect("forward block action");
        let expected_block = DenseMatrix::from_fn(4, 2, |row, column| {
            dense_matrix.matvec(&block.column_to_vec(column))[row]
        });
        assert_close(output.as_slice(), expected_block.as_slice());
    }
}

#[test]
fn coupled_trace_preconditioner_rejects_unknown_adjoint_and_bad_state() {
    struct ForwardOnly;
    impl Preconditioner<Complex64> for ForwardOnly {
        fn apply(&self, rhs: &[Complex64]) -> Vec<Complex64> {
            rhs.to_vec()
        }
    }
    let transfer = ConservativeTraceTransfer::try_new(vec![vec![(0, 1.0)]], vec![1.0], vec![1.0])
        .expect("identity transfer");
    let rejected = CoupledTracePreconditioner::try_new(
        Box::new(ForwardOnly),
        Box::new(ScalePreconditioner {
            scale: Complex64::new(1.0, 0.0),
        }),
        transfer.clone(),
        Complex64::new(0.0, 0.0),
    );
    assert!(matches!(
        rejected,
        Err(CouplingError::UnsupportedAdjoint {
            block: "interior",
            operation: "transpose preconditioner"
        })
    ));

    let preconditioner = CoupledTracePreconditioner::try_new(
        Box::new(ScalePreconditioner {
            scale: Complex64::new(1.0, 0.0),
        }),
        Box::new(ScalePreconditioner {
            scale: Complex64::new(1.0, 0.0),
        }),
        transfer,
        Complex64::new(0.0, 0.0),
    )
    .expect("identity preconditioner");
    assert!(matches!(
        preconditioner.apply_checked(&[Complex64::new(1.0, 0.0)]),
        Err(CouplingError::Dimension { .. })
    ));
    assert!(matches!(
        preconditioner.apply_checked(&[Complex64::new(f64::NAN, 0.0), Complex64::new(0.0, 0.0),]),
        Err(CouplingError::NonFinite { .. })
    ));
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

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
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

fn dense_matrix_transpose_matvec(
    matrix: &DenseMatrix<Complex64>,
    state: &[Complex64],
) -> Vec<Complex64> {
    (0..matrix.ncols())
        .map(|column| {
            (0..matrix.nrows())
                .map(|row| matrix[[row, column]] * state[row])
                .sum()
        })
        .collect()
}

fn dense_matrix_hermitian_matvec(
    matrix: &DenseMatrix<Complex64>,
    state: &[Complex64],
) -> Vec<Complex64> {
    (0..matrix.ncols())
        .map(|column| {
            (0..matrix.nrows())
                .map(|row| matrix[[row, column]].conj() * state[row])
                .sum()
        })
        .collect()
}

#[test]
fn coupled_trace_system_bounded_solves_cover_forward_adjoint_and_blocks() {
    let transfer = ConservativeTraceTransfer::try_new(vec![vec![(0, 1.0)]], vec![1.0], vec![1.0])
        .expect("conservative transfer");
    let system = CoupledTraceSystem::try_new(
        DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(2.0, 0.3)]),
        DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(1.4, -0.2)]),
        transfer,
        Complex64::new(0.25, -0.1),
    )
    .expect("dense coupled system");
    let state = vec![Complex64::new(0.4, -0.6), Complex64::new(-0.2, 0.8)];
    let rhs = system.apply(&state).expect("forward action");
    let solved = system.solve(&rhs).expect("forward solve");
    assert_close(&solved, &state);
    assert!(system.relative_residual(&solved, &rhs).unwrap() < 1.0e-12);

    let adjoint_rhs = system.apply_adjoint(&state).expect("adjoint action");
    let adjoint_solved = system.solve_adjoint(&adjoint_rhs).expect("adjoint solve");
    assert_close(&adjoint_solved, &state);
    assert!(
        system
            .relative_adjoint_residual(&adjoint_solved, &adjoint_rhs)
            .unwrap()
            < 1.0e-12
    );

    let rhs_block =
        DenseMatrix::from_row_major_vec(2, 2, vec![rhs[0], adjoint_rhs[0], rhs[1], adjoint_rhs[1]]);
    let solved_block = system.solve_block(&rhs_block).expect("forward block solve");
    assert_close(&solved_block.column_to_vec(0), &state);
    let adjoint_block = system
        .solve_adjoint_block(&rhs_block)
        .expect("adjoint block solve");
    assert_close(&adjoint_block.column_to_vec(1), &state);
}

#[test]
fn transmission_passivity_contract_rejects_active_matrix_free_operator() {
    // A = J^{-1} for interface_dofs=1, omega=1, harmonic sign=+1, so J A=I.
    struct PassiveOperator {
        sign: f64,
    }

    impl LinearOperator<Complex64> for PassiveOperator {
        fn num_rows(&self) -> usize {
            2
        }

        fn num_cols(&self) -> usize {
            2
        }

        fn apply_into(&self, input: &[Complex64], output: &mut [Complex64]) {
            output[0] = self.sign * input[1];
            output[1] = self.sign * Complex64::new(0.0, -1.0) * input[0];
        }

        fn apply_transpose_into(&self, input: &[Complex64], output: &mut [Complex64]) {
            output[0] = self.sign * Complex64::new(0.0, -1.0) * input[1];
            output[1] = self.sign * input[0];
        }
    }

    let probes = probes(2);
    let report = check_transmission_passivity_contract(
        &PassiveOperator { sign: 1.0 },
        1,
        1.0,
        1.0,
        &probes,
        1.0e-12,
    )
    .expect("J A=I is passive");
    assert!(report.minimum_power > 0.0);
    assert!(
        check_transmission_passivity_contract(
            &PassiveOperator { sign: -1.0 },
            1,
            1.0,
            1.0,
            &probes,
            1.0e-12,
        )
        .is_err()
    );
}

#[test]
fn coupled_trace_system_transpose_solves_cover_scalar_and_blocks() {
    let transfer = ConservativeTraceTransfer::try_new(vec![vec![(0, 1.0)]], vec![1.0], vec![1.0])
        .expect("conservative transfer");
    let system = CoupledTraceSystem::try_new(
        DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(2.0, 0.3)]),
        DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(1.4, -0.2)]),
        transfer,
        Complex64::new(0.25, -0.1),
    )
    .expect("dense coupled system");
    let state = vec![Complex64::new(0.4, -0.6), Complex64::new(-0.2, 0.8)];
    let rhs = system.apply_transpose(&state).expect("transpose action");
    let solved = system.solve_transpose(&rhs).expect("transpose solve");
    assert_close(&solved, &state);
    assert!(system.relative_transpose_residual(&solved, &rhs).unwrap() < 1.0e-12);

    let scaled_state = vec![state[0] * 0.5, state[1] * -0.25];
    let second_rhs = system
        .apply_transpose(&scaled_state)
        .expect("second transpose action");
    let rhs_block =
        DenseMatrix::from_row_major_vec(2, 2, vec![rhs[0], second_rhs[0], rhs[1], second_rhs[1]]);
    let solved_block = system
        .solve_transpose_block(&rhs_block)
        .expect("transpose block solve");
    assert_close(&solved_block.column_to_vec(0), &state);
    assert_close(&solved_block.column_to_vec(1), &scaled_state);
    assert!(
        system
            .relative_transpose_residual(&solved_block.column_to_vec(1), &second_rhs)
            .unwrap()
            < 1.0e-12
    );
}
