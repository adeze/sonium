use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectMethod, DirectFactorization, MatrixClass, Ordering,
    SparseDirectConfig, SparseDirectMethod, factorize_dense, factorize_sparse,
};
use sonium_solvers::preconditioners::IluPreconditioner;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::validation_corpus::{CorpusClass, SOLVER_VALIDATION_CORPUS};
use sonium_solvers::{GmresConfig, gmres, gmres_preconditioned};

fn real_matvec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn csr(matrix: &[Vec<f64>]) -> CsrMatrix<f64> {
    let triplets = matrix
        .iter()
        .enumerate()
        .flat_map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .filter_map(move |(column, &value)| (value != 0.0).then_some((row, column, value)))
        })
        .collect();
    CsrMatrix::from_triplets(matrix.len(), matrix[0].len(), triplets)
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn corpus_metadata_covers_all_required_solver_classes() {
    assert_eq!(SOLVER_VALIDATION_CORPUS.len(), 12);
    for class in [
        CorpusClass::Spd,
        CorpusClass::Hpd,
        CorpusClass::SymmetricIndefinite,
        CorpusClass::Nonsymmetric,
        CorpusClass::SaddlePoint,
        CorpusClass::IllScaled,
        CorpusClass::NearSingular,
        CorpusClass::RankDeficient,
        CorpusClass::Helmholtz,
        CorpusClass::ConvectionDiffusion,
    ] {
        assert!(
            SOLVER_VALIDATION_CORPUS
                .iter()
                .any(|case| case.class == class)
        );
    }
}

#[test]
fn direct_corpus_covers_spd_indefinite_and_saddle_point() {
    let spd = vec![
        vec![4.0, 1.0, 0.0],
        vec![1.0, 3.0, 1.0],
        vec![0.0, 1.0, 2.0],
    ];
    let x_spd = [1.0, -2.0, 0.75];
    let spd_dense = DenseMatrix::from_row_major_vec(3, 3, spd.iter().flatten().copied().collect());
    let spd_factor = factorize_dense(
        &spd_dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            method: DenseDirectMethod::Cholesky,
            ..Default::default()
        },
    )
    .expect("SPD corpus factorization");
    assert_close(
        &spd_factor.solve(&real_matvec(&spd, &x_spd)).unwrap(),
        &x_spd,
        1.0e-12,
    );

    let indefinite = DenseMatrix::from_row_major_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
    let indefinite_factor = factorize_dense(
        &indefinite,
        &DenseDirectConfig {
            matrix_class: MatrixClass::SymmetricIndefinite,
            method: DenseDirectMethod::Ldlt,
            ..Default::default()
        },
    )
    .expect("symmetric-indefinite corpus factorization");
    assert_close(
        &indefinite_factor.solve(&[2.0, 1.0]).unwrap(),
        &[1.0, 2.0],
        1.0e-12,
    );

    let saddle =
        DenseMatrix::from_row_major_vec(3, 3, vec![2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.0]);
    let saddle_factor = factorize_dense(
        &saddle,
        &DenseDirectConfig {
            matrix_class: MatrixClass::SymmetricIndefinite,
            method: DenseDirectMethod::Ldlt,
            ..Default::default()
        },
    )
    .expect("saddle-point corpus factorization");
    assert_close(
        &saddle_factor.solve(&[5.0, 7.0, 3.0]).unwrap(),
        &[1.0, 2.0, 3.0],
        1.0e-12,
    );
}

#[test]
fn sparse_corpus_covers_nonsymmetric_scaling_and_convection_diffusion() {
    let nonsymmetric = vec![
        vec![4.0, 1.0, 0.5],
        vec![2.0, 5.0, 1.0],
        vec![0.25, 1.5, 3.0],
    ];
    let x = [1.0, -0.5, 2.0];
    let factor = factorize_sparse(
        &csr(&nonsymmetric),
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .expect("nonsymmetric corpus factorization");
    assert_close(
        &factor.solve(&real_matvec(&nonsymmetric, &x)).unwrap(),
        &x,
        1.0e-12,
    );

    let ill_scaled = vec![
        vec![1.0e-6, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0e6],
    ];
    let ill_factor = factorize_sparse(
        &csr(&ill_scaled),
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .expect("ill-scaled corpus factorization");
    assert_close(
        &ill_factor
            .solve(&real_matvec(&ill_scaled, &[2.0, -3.0, 4.0]))
            .unwrap(),
        &[2.0, -3.0, 4.0],
        1.0e-10,
    );

    let near_singular = vec![vec![1.0, 1.0], vec![1.0, 1.0 + 1.0e-8]];
    let near_factor = factorize_sparse(
        &csr(&near_singular),
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .expect("near-singular corpus factorization");
    assert_close(
        &near_factor
            .solve(&real_matvec(&near_singular, &[1.0, -1.0]))
            .unwrap(),
        &[1.0, -1.0],
        1.0e-6,
    );

    let convection_diffusion = vec![
        vec![2.0, -1.0, 0.0],
        vec![-0.25, 2.0, -1.0],
        vec![0.0, -0.25, 2.0],
    ];
    let convection_factor = factorize_sparse(
        &csr(&convection_diffusion),
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .expect("convection-diffusion corpus factorization");
    assert_close(
        &convection_factor
            .solve(&real_matvec(&convection_diffusion, &[1.0, 2.0, 3.0]))
            .unwrap(),
        &[1.0, 2.0, 3.0],
        1.0e-10,
    );
}

#[test]
fn rank_deficient_corpus_is_rejected_with_typed_failure() {
    let rank_deficient = csr(&[vec![1.0, 1.0], vec![1.0, 1.0]]);
    let outcome = factorize_sparse(
        &rank_deficient,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .and_then(|factor| factor.solve(&[2.0, 2.0]));
    assert!(
        outcome.is_err(),
        "rank-deficient corpus must not report a successful direct solve"
    );
}

#[test]
fn generated_16_dof_pde_fixtures_match_metadata_and_solve() {
    let poisson = sonium_solvers::poisson_1d_fixture(16).expect("Poisson fixture");
    assert_eq!(poisson.num_rows, 16);
    assert_eq!(poisson.num_cols, 16);
    assert_eq!(poisson.values.len(), 46);
    let expected: Vec<f64> = (0..16).map(|index| 1.0 + index as f64 / 16.0).collect();
    let rhs = poisson.matvec(&expected);
    let poisson_solution = sonium_solvers::gmres(
        &poisson,
        &rhs,
        &sonium_solvers::GmresConfig {
            max_iterations: 200,
            restart: 16,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            ..Default::default()
        },
    );
    assert!(poisson_solution.converged);
    assert!(poisson_solution.residual < 1.0e-10);

    let convection =
        sonium_solvers::convection_diffusion_1d_fixture(16).expect("convection-diffusion fixture");
    assert_eq!(convection.values.len(), 46);
    let convection_rhs = convection.matvec(&expected);
    let convection_solution = sonium_solvers::gmres(
        &convection,
        &convection_rhs,
        &sonium_solvers::GmresConfig {
            max_iterations: 200,
            restart: 16,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            ..Default::default()
        },
    );
    assert!(convection_solution.converged);
    assert!(convection_solution.residual < 1.0e-10);
    assert!(
        convection_solution
            .x
            .iter()
            .zip(&expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-8)
    );
}

#[test]
fn generated_pde_fixtures_reject_zero_dimension() {
    assert!(sonium_solvers::poisson_1d_fixture(0).is_err());
    assert!(sonium_solvers::convection_diffusion_1d_fixture(0).is_err());
}

#[test]
fn complex_corpus_covers_hpd_and_helmholtz_residuals() {
    let hpd = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(4.0, 0.0),
        ],
    );
    let hpd_expected = [Complex64::new(0.5, -1.0), Complex64::new(2.0, 0.25)];
    let hpd_rhs = [
        hpd[(0, 0)] * hpd_expected[0] + hpd[(0, 1)] * hpd_expected[1],
        hpd[(1, 0)] * hpd_expected[0] + hpd[(1, 1)] * hpd_expected[1],
    ];
    let hpd_factor = factorize_dense(
        &hpd,
        &DenseDirectConfig {
            matrix_class: MatrixClass::HermitianPositiveDefinite,
            method: DenseDirectMethod::Cholesky,
            ..Default::default()
        },
    )
    .expect("HPD corpus factorization");
    let hpd_solution = hpd_factor.solve(&hpd_rhs).unwrap();
    assert!(
        hpd_solution
            .iter()
            .zip(hpd_expected)
            .all(|(actual, expected)| (*actual - expected).norm() <= 1.0e-12)
    );

    let helmholtz = DenseMatrix::from_row_major_vec(
        3,
        3,
        vec![
            Complex64::new(2.0, 0.2),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(2.0, 0.2),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(2.0, 0.2),
        ],
    );
    let helmholtz_expected = [
        Complex64::new(1.0, 0.5),
        Complex64::new(-0.25, 1.0),
        Complex64::new(2.0, -0.75),
    ];
    let helmholtz_rhs = (0..3)
        .map(|row| {
            (0..3)
                .map(|column| helmholtz[(row, column)] * helmholtz_expected[column])
                .sum::<Complex64>()
        })
        .collect::<Vec<_>>();
    let helmholtz_factor = factorize_dense(
        &helmholtz,
        &DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Lu,
            ..Default::default()
        },
    )
    .expect("Helmholtz corpus factorization");
    let solution = helmholtz_factor.solve(&helmholtz_rhs).unwrap();
    let residual = (0..3)
        .map(|row| {
            (0..3)
                .map(|column| helmholtz[(row, column)] * solution[column])
                .sum::<Complex64>()
                - helmholtz_rhs[row]
        })
        .map(|value| value.norm())
        .fold(0.0, f64::max);
    assert!(residual <= 1.0e-10, "Helmholtz residual={residual}");
}

fn one_dimensional_stencil(n: usize, diagonal: f64, lower: f64, upper: f64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|row| {
            (0..n)
                .map(|column| {
                    if row == column {
                        diagonal
                    } else if column + 1 == row {
                        lower
                    } else if row + 1 == column {
                        upper
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

#[test]
fn larger_generated_pde_fixtures_cover_direct_and_iterative_workloads() {
    let poisson = one_dimensional_stencil(16, 2.0, -1.0, -1.0);
    let expected = (0..16)
        .map(|index| 1.0 + index as f64 * 0.1)
        .collect::<Vec<_>>();
    let rhs = real_matvec(&poisson, &expected);
    let factor = factorize_sparse(
        &csr(&poisson),
        &SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            method: SparseDirectMethod::SimplicialCholesky,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .expect("16x16 Poisson factorization");
    let solution = factor.solve(&rhs).expect("16x16 Poisson solve");
    assert_close(&solution, &expected, 1.0e-10);
    assert!(factor.factor_nonzeros() >= 16);
    assert!(factor.factor_memory_bytes() > 0);

    let convection = one_dimensional_stencil(16, 2.0, -1.25, -0.75);
    let convection_rhs = real_matvec(&convection, &expected);
    let convection_solution = gmres(
        &csr(&convection),
        &convection_rhs,
        &GmresConfig {
            max_iterations: 64,
            tolerance: 1.0e-12,
            print_interval: 0,
            ..Default::default()
        },
    );
    assert!(convection_solution.converged);
    assert!(convection_solution.residual <= 1.0e-12);
    assert!(convection_solution.backward_error.is_finite());
}

#[test]
fn preconditioned_corpus_covers_ilu_and_independent_residuals() {
    let poisson = one_dimensional_stencil(16, 2.0, -1.0, -1.0);
    let expected = (0..16)
        .map(|index| 1.0 + index as f64 * 0.1)
        .collect::<Vec<_>>();
    let rhs = real_matvec(&poisson, &expected);
    let matrix = csr(&poisson);
    let preconditioner =
        IluPreconditioner::try_from_csr(&matrix).expect("SPD corpus ILU construction");
    let solution = gmres_preconditioned(
        &matrix,
        &preconditioner,
        &rhs,
        &GmresConfig {
            max_iterations: 80,
            tolerance: 1.0e-12,
            print_interval: 0,
            ..Default::default()
        },
    );
    assert!(solution.converged, "ILU-preconditioned GMRES: {solution:?}");
    assert_close(&solution.x, &expected, 1.0e-10);
    let residual = real_matvec(&poisson, &solution.x)
        .iter()
        .zip(&rhs)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f64>()
        .sqrt()
        / rhs
            .iter()
            .map(|value| value.powi(2))
            .sum::<f64>()
            .sqrt()
            .max(1.0);
    assert!(residual <= 1.0e-10, "independent residual={residual:e}");
}

#[test]
fn preconditioned_corpus_covers_complex_helmholtz_gmres() {
    let matrix = CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, Complex64::new(2.0, 0.2)),
            (0, 1, Complex64::new(-1.0, 0.0)),
            (1, 0, Complex64::new(-1.0, 0.0)),
            (1, 1, Complex64::new(2.0, 0.2)),
            (1, 2, Complex64::new(-1.0, 0.0)),
            (2, 1, Complex64::new(-1.0, 0.0)),
            (2, 2, Complex64::new(2.0, 0.2)),
        ],
    );
    let expected = [
        Complex64::new(1.0, 0.5),
        Complex64::new(-0.25, 1.0),
        Complex64::new(2.0, -0.75),
    ];
    let rhs = matrix.matvec(&expected);
    let preconditioner =
        IluPreconditioner::try_from_csr(&matrix).expect("complex Helmholtz ILU construction");
    let solution = gmres_preconditioned(
        &matrix,
        &preconditioner,
        &rhs,
        &GmresConfig {
            max_iterations: 30,
            tolerance: 1.0e-12,
            print_interval: 0,
            ..Default::default()
        },
    );
    assert!(
        solution.converged,
        "complex Helmholtz ILU-preconditioned GMRES: {solution:?}"
    );
    let solution_error = solution
        .x
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - expected).norm())
        .fold(0.0, f64::max);
    assert!(
        solution_error <= 1.0e-10,
        "solution error={solution_error:e}"
    );
    let residual = matrix
        .matvec(&solution.x)
        .iter()
        .zip(&rhs)
        .map(|(actual, expected)| (*actual - expected).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / rhs
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0);
    assert!(residual <= 1.0e-10, "independent residual={residual:e}");
}
