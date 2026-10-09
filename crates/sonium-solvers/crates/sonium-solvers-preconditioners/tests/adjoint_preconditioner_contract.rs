use num_complex::Complex64;
use sonium_solvers_core::{
    dense::DenseMatrix,
    traits::{AdjointCapability, Preconditioner, SolverError},
};
use sonium_solvers_preconditioners::{
    AdditiveSchwarzPreconditioner, AmgConfig, AmgCycle, AmgPreconditioner, AmgRestriction,
    AmgSmoother, BlockJacobiPreconditioner, BlockPreconditionerConfig, ChebyshevConfig,
    ChebyshevPreconditioner, DiagonalPreconditioner, HiluPreconditioner, IdentityPreconditioner,
    IluColoringPreconditioner, IluFixedPointPreconditioner, IluPreconditioner, IlutConfig,
    IlutPreconditioner, ParIlutPreconditioner, SaddlePointConfig, SaddlePointPreconditioner,
    SchurStrategy,
};
use sonium_solvers_sparse::CsrMatrix;

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

fn assert_complex_block_matches_vectors<P: Preconditioner<Complex64>>(
    preconditioner: &P,
    input: &DenseMatrix<Complex64>,
) {
    let mut forward = DenseMatrix::zeros(input.nrows(), input.ncols());
    preconditioner
        .try_apply_block(input, &mut forward)
        .expect("checked forward block action");
    let mut transpose = DenseMatrix::zeros(input.nrows(), input.ncols());
    preconditioner
        .try_apply_transpose_block(input, &mut transpose)
        .expect("checked transpose block action");
    let mut hermitian = DenseMatrix::zeros(input.nrows(), input.ncols());
    preconditioner
        .try_apply_hermitian_block(input, &mut hermitian)
        .expect("checked Hermitian block action");
    for column in 0..input.ncols() {
        let vector = input.column_to_vec(column);
        let expected_forward = preconditioner
            .try_apply(&vector)
            .expect("checked forward action");
        let expected_transpose = preconditioner
            .try_apply_transpose(&vector)
            .expect("checked transpose action");
        let expected_hermitian = preconditioner
            .try_apply_hermitian(&vector)
            .expect("checked Hermitian action");
        for row in 0..input.nrows() {
            assert!((forward[[row, column]] - expected_forward[row]).norm() < 1.0e-11);
            assert!((transpose[[row, column]] - expected_transpose[row]).norm() < 1.0e-11);
            assert!((hermitian[[row, column]] - expected_hermitian[row]).norm() < 1.0e-11);
        }
        let lhs_hermitian: Complex64 = expected_forward
            .iter()
            .zip(vector.iter())
            .map(|(a, b)| a.conj() * *b)
            .sum();
        let rhs_hermitian: Complex64 = vector
            .iter()
            .zip(expected_hermitian.iter())
            .map(|(a, b)| a.conj() * *b)
            .sum();
        assert!((lhs_hermitian - rhs_hermitian).norm() < 1.0e-11);
        let lhs_transpose: Complex64 = expected_forward
            .iter()
            .zip(vector.iter())
            .map(|(a, b)| *a * *b)
            .sum();
        let rhs_transpose: Complex64 = vector
            .iter()
            .zip(expected_transpose.iter())
            .map(|(a, b)| *a * *b)
            .sum();
        assert!((lhs_transpose - rhs_transpose).norm() < 1.0e-11);
    }
}

fn assert_hermitian_inner_product_contract<P>(preconditioner: &P, r: &[Complex64], s: &[Complex64])
where
    P: Preconditioner<Complex64>,
{
    let mr = preconditioner.apply(r);
    let mh_s = preconditioner
        .try_apply_hermitian(s)
        .expect("exact Hermitian preconditioner action");
    let lhs: Complex64 = mr.iter().zip(s).map(|(a, b)| a.conj() * b).sum();
    let rhs: Complex64 = r.iter().zip(mh_s).map(|(a, b)| a.conj() * b).sum();
    assert!(
        (lhs - rhs).norm() < 1e-11,
        "adjoint mismatch: {lhs:?} vs {rhs:?}"
    );

    for seed in 0..16_u64 {
        let generated_r = property_vector(r.len(), seed.wrapping_add(1));
        let generated_s = property_vector(s.len(), seed.wrapping_add(101));
        let generated_mr = preconditioner.apply(&generated_r);
        let generated_mh_s = preconditioner
            .try_apply_hermitian(&generated_s)
            .expect("exact Hermitian preconditioner action");
        let generated_lhs: Complex64 = generated_mr
            .iter()
            .zip(&generated_s)
            .map(|(a, b)| a.conj() * b)
            .sum();
        let generated_rhs: Complex64 = generated_r
            .iter()
            .zip(generated_mh_s)
            .map(|(a, b)| a.conj() * b)
            .sum();
        assert!(
            (generated_lhs - generated_rhs).norm() < 1e-11,
            "randomized Hermitian adjoint mismatch for seed {seed}: {generated_lhs:?} vs {generated_rhs:?}"
        );
    }
}

fn assert_transpose_inner_product_contract<P>(preconditioner: &P, r: &[Complex64], s: &[Complex64])
where
    P: Preconditioner<Complex64>,
{
    let mr = preconditioner.apply(r);
    let mt_s = preconditioner
        .try_apply_transpose(s)
        .expect("exact transpose preconditioner action");
    let lhs: Complex64 = mr.iter().zip(s).map(|(a, b)| *a * b).sum();
    let rhs: Complex64 = r.iter().zip(mt_s).map(|(a, b)| *a * b).sum();
    assert!(
        (lhs - rhs).norm() < 1e-11,
        "transpose mismatch: {lhs:?} vs {rhs:?}"
    );

    for seed in 0..16_u64 {
        let generated_r = property_vector(r.len(), seed.wrapping_add(201));
        let generated_s = property_vector(s.len(), seed.wrapping_add(301));
        let generated_mr = preconditioner.apply(&generated_r);
        let generated_mt_s = preconditioner
            .try_apply_transpose(&generated_s)
            .expect("exact transpose preconditioner action");
        let generated_lhs: Complex64 = generated_mr
            .iter()
            .zip(&generated_s)
            .map(|(a, b)| *a * *b)
            .sum();
        let generated_rhs: Complex64 = generated_r
            .iter()
            .zip(generated_mt_s)
            .map(|(a, b)| *a * b)
            .sum();
        assert!(
            (generated_lhs - generated_rhs).norm() < 1e-11,
            "randomized transpose adjoint mismatch for seed {seed}: {generated_lhs:?} vs {generated_rhs:?}"
        );
    }
}

#[test]
fn ilu_hermitian_action_satisfies_the_adjoint_inner_product_contract() {
    let matrix = CsrMatrix::try_from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, 0.5),
            Complex64::new(1.0, -0.5),
            Complex64::new(3.0, 0.0),
        ],
    )
    .expect("canonical Hermitian CSR");
    let preconditioner = IluPreconditioner::try_from_csr(&matrix).expect("ILU setup");
    assert_eq!(
        preconditioner.hermitian_capability(),
        AdjointCapability::Exact
    );

    let r = [Complex64::new(1.0, -0.5), Complex64::new(-2.0, 1.0)];
    let s = [Complex64::new(0.25, 2.0), Complex64::new(1.5, -0.75)];
    let mr = preconditioner.apply(&r);
    let mh_s = preconditioner
        .try_apply_hermitian(&s)
        .expect("exact ILU Hermitian action");
    let lhs: Complex64 = mr.iter().zip(s).map(|(a, b)| a.conj() * b).sum();
    let rhs: Complex64 = r.iter().zip(mh_s).map(|(a, b)| a.conj() * b).sum();
    assert!(
        (lhs - rhs).norm() < 1e-12,
        "adjoint mismatch: {lhs:?} vs {rhs:?}"
    );
}

#[test]
fn diagonal_hermitian_action_satisfies_the_adjoint_inner_product_contract() {
    let preconditioner = DiagonalPreconditioner::try_from_diagonal(&[
        Complex64::new(2.0, 1.0),
        Complex64::new(-1.5, 0.25),
    ])
    .expect("finite diagonal setup");
    assert_eq!(
        preconditioner.transpose_capability(),
        AdjointCapability::Exact
    );
    assert_eq!(
        preconditioner.hermitian_capability(),
        AdjointCapability::Exact
    );

    let r = [Complex64::new(1.0, -0.5), Complex64::new(-2.0, 1.0)];
    let s = [Complex64::new(0.25, 2.0), Complex64::new(1.5, -0.75)];
    let mr = preconditioner.apply(&r);
    let mh_s = preconditioner
        .try_apply_hermitian(&s)
        .expect("exact diagonal Hermitian action");
    let lhs: Complex64 = mr.iter().zip(s).map(|(a, b)| a.conj() * b).sum();
    let rhs: Complex64 = r.iter().zip(mh_s).map(|(a, b)| a.conj() * b).sum();
    assert!((lhs - rhs).norm() < 1e-12);

    assert_complex_block_matches_vectors(
        &preconditioner,
        &DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(1.0, -0.5),
                Complex64::new(0.25, 2.0),
                Complex64::new(-2.0, 1.0),
                Complex64::new(1.5, -0.75),
            ],
        ),
    );
}

#[test]
fn ilut_transpose_and_hermitian_actions_satisfy_inner_product_contract() {
    let matrix = CsrMatrix::try_from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            Complex64::new(4.0, 0.5),
            Complex64::new(0.8, -0.2),
            Complex64::new(0.3, 0.1),
            Complex64::new(3.0, -0.75),
            Complex64::new(0.7, 0.2),
            Complex64::new(0.4, -0.15),
            Complex64::new(2.5, 0.3),
        ],
    )
    .expect("canonical ILUT matrix");
    let preconditioner = IlutPreconditioner::try_from_csr(&matrix).expect("ILUT setup");
    assert_eq!(
        preconditioner.transpose_capability(),
        AdjointCapability::Exact
    );
    assert_eq!(
        preconditioner.hermitian_capability(),
        AdjointCapability::Exact
    );

    let r = [
        Complex64::new(1.0, -0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.25, 0.75),
    ];
    let s = [
        Complex64::new(0.25, 2.0),
        Complex64::new(1.5, -0.75),
        Complex64::new(-0.5, 0.2),
    ];
    assert_hermitian_inner_product_contract(&preconditioner, &r, &s);

    let mr = preconditioner.apply(&r);
    let mt_s = preconditioner
        .try_apply_transpose(&s)
        .expect("exact transpose preconditioner action");
    let lhs_transpose: Complex64 = mr.iter().zip(s).map(|(a, b)| *a * b).sum();
    let rhs_transpose: Complex64 = r.iter().zip(mt_s).map(|(a, b)| *a * b).sum();
    assert!((lhs_transpose - rhs_transpose).norm() < 1e-11);
}

#[test]
fn block_jacobi_transpose_and_hermitian_actions_satisfy_inner_product_contract() {
    let matrix = CsrMatrix::try_from_raw_parts(
        4,
        4,
        vec![0, 2, 4, 6, 8],
        vec![0, 1, 0, 1, 2, 3, 2, 3],
        vec![
            Complex64::new(4.0, 0.5),
            Complex64::new(0.8, -0.2),
            Complex64::new(0.3, 0.1),
            Complex64::new(3.0, -0.75),
            Complex64::new(2.5, 0.3),
            Complex64::new(0.4, -0.15),
            Complex64::new(0.7, 0.2),
            Complex64::new(2.0, 0.4),
        ],
    )
    .expect("canonical block-Jacobi matrix");
    let preconditioner = BlockJacobiPreconditioner::try_from_csr(&matrix, 2, IlutConfig::default())
        .expect("block-Jacobi setup");
    assert_eq!(
        preconditioner.transpose_capability(),
        AdjointCapability::Exact
    );
    assert_eq!(
        preconditioner.hermitian_capability(),
        AdjointCapability::Exact
    );

    let r = [
        Complex64::new(1.0, -0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.25, 0.75),
        Complex64::new(-0.5, 0.2),
    ];
    let s = [
        Complex64::new(0.25, 2.0),
        Complex64::new(1.5, -0.75),
        Complex64::new(-0.5, 0.2),
        Complex64::new(0.75, -1.25),
    ];
    assert_hermitian_inner_product_contract(&preconditioner, &r, &s);

    let mr = preconditioner.apply(&r);
    let mt_s = preconditioner
        .try_apply_transpose(&s)
        .expect("exact transpose block-Jacobi action");
    let lhs: Complex64 = mr.iter().zip(s).map(|(a, b)| *a * b).sum();
    let rhs: Complex64 = r.iter().zip(mt_s).map(|(a, b)| *a * b).sum();
    assert!((lhs - rhs).norm() < 1e-11);

    assert_complex_block_matches_vectors(
        &preconditioner,
        &DenseMatrix::from_row_major_vec(
            4,
            2,
            vec![
                Complex64::new(1.0, -0.5),
                Complex64::new(0.25, 2.0),
                Complex64::new(-2.0, 1.0),
                Complex64::new(1.5, -0.75),
                Complex64::new(0.25, 0.75),
                Complex64::new(-0.5, 0.2),
                Complex64::new(-0.5, 0.2),
                Complex64::new(0.75, -1.25),
            ],
        ),
    );
}

#[test]
fn identity_preconditioner_has_real_checked_block_adjoint_contract() {
    let preconditioner = IdentityPreconditioner;
    let input = DenseMatrix::from_row_major_vec(3, 2, vec![1.0, -2.0, 0.5, 3.0, 4.0, -1.0]);
    let mut forward = DenseMatrix::zeros(3, 2);
    preconditioner
        .try_apply_block(&input, &mut forward)
        .expect("checked identity block action");
    assert_eq!(forward.as_slice(), input.as_slice());

    let mut transpose = DenseMatrix::zeros(3, 2);
    preconditioner
        .try_apply_transpose_block(&input, &mut transpose)
        .expect("checked identity transpose block action");
    assert_eq!(transpose.as_slice(), input.as_slice());

    let mut hermitian = DenseMatrix::zeros(3, 2);
    preconditioner
        .try_apply_hermitian_block(&input, &mut hermitian)
        .expect("checked identity Hermitian block action");
    assert_eq!(hermitian.as_slice(), input.as_slice());
}

#[test]
fn checked_preconditioner_block_dispatch_rejects_bad_shapes_and_unavailable_adjoint() {
    let preconditioner = IdentityPreconditioner;
    let input = DenseMatrix::<f64>::zeros(2, 2);
    let mut wrong_rows = DenseMatrix::zeros(3, 2);
    assert!(matches!(
        preconditioner.try_apply_transpose_block(&input, &mut wrong_rows),
        Err(SolverError::DimensionMismatch { .. })
    ));
    let mut wrong_columns = DenseMatrix::zeros(2, 1);
    assert!(matches!(
        preconditioner.try_apply_hermitian_block(&input, &mut wrong_columns),
        Err(SolverError::DimensionMismatch { .. })
    ));

    let matrix = CsrMatrix::try_from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 3.0])
        .expect("diagonal matrix");
    let unsupported =
        IluFixedPointPreconditioner::try_from_csr(&matrix, 2).expect("fixed-point ILU setup");
    let mut output = DenseMatrix::zeros(2, 2);
    assert!(matches!(
        unsupported.try_apply_transpose_block(&input, &mut output),
        Err(SolverError::UnsupportedOperation { .. })
    ));
    assert!(matches!(
        unsupported.try_apply_hermitian_block(&input, &mut output),
        Err(SolverError::UnsupportedOperation { .. })
    ));
}

#[test]
fn one_level_schwarz_transpose_and_hermitian_actions_satisfy_inner_product_contract() {
    let matrix = CsrMatrix::try_from_raw_parts(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![
            Complex64::new(4.0, 0.5),
            Complex64::new(0.8, -0.2),
            Complex64::new(0.3, 0.1),
            Complex64::new(3.0, -0.75),
            Complex64::new(0.7, 0.2),
            Complex64::new(0.4, -0.15),
            Complex64::new(2.5, 0.3),
            Complex64::new(0.6, -0.1),
            Complex64::new(0.2, 0.05),
            Complex64::new(2.0, 0.4),
        ],
    )
    .expect("canonical Schwarz matrix");
    let preconditioner = AdditiveSchwarzPreconditioner::try_from_csr(&matrix, 2, 1)
        .expect("one-level Schwarz setup");
    assert_eq!(
        preconditioner.transpose_capability(),
        AdjointCapability::Exact
    );
    assert_eq!(
        preconditioner.hermitian_capability(),
        AdjointCapability::Exact
    );

    let r = [
        Complex64::new(1.0, -0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.25, 0.75),
        Complex64::new(-0.5, 0.2),
    ];
    let s = [
        Complex64::new(0.25, 2.0),
        Complex64::new(1.5, -0.75),
        Complex64::new(-0.5, 0.2),
        Complex64::new(0.75, -1.25),
    ];
    assert_hermitian_inner_product_contract(&preconditioner, &r, &s);

    let mr = preconditioner.apply(&r);
    let mt_s = preconditioner
        .try_apply_transpose(&s)
        .expect("exact transpose Schwarz action");
    let lhs: Complex64 = mr.iter().zip(s).map(|(a, b)| *a * b).sum();
    let rhs: Complex64 = r.iter().zip(mt_s).map(|(a, b)| *a * b).sum();
    assert!((lhs - rhs).norm() < 1e-11);
}

#[test]
fn parallel_and_hierarchical_ilu_adjoint_actions_satisfy_inner_product_contract() {
    let matrix = CsrMatrix::try_from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            Complex64::new(4.0, 0.5),
            Complex64::new(0.8, -0.2),
            Complex64::new(0.3, 0.1),
            Complex64::new(3.0, -0.75),
            Complex64::new(0.7, 0.2),
            Complex64::new(0.4, -0.15),
            Complex64::new(2.5, 0.3),
        ],
    )
    .expect("canonical parallel ILU matrix");
    let r = [
        Complex64::new(1.0, -0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.25, 0.75),
    ];
    let s = [
        Complex64::new(0.25, 2.0),
        Complex64::new(1.5, -0.75),
        Complex64::new(-0.5, 0.2),
    ];

    let coloring = IluColoringPreconditioner::try_from_csr(&matrix).expect("coloring ILU setup");
    assert_eq!(coloring.transpose_capability(), AdjointCapability::Exact);
    assert_eq!(coloring.hermitian_capability(), AdjointCapability::Exact);
    assert_transpose_inner_product_contract(&coloring, &r, &s);
    assert_hermitian_inner_product_contract(&coloring, &r, &s);

    let parilut = ParIlutPreconditioner::try_from_csr(&matrix).expect("ParILUT setup");
    assert_eq!(parilut.transpose_capability(), AdjointCapability::Exact);
    assert_eq!(parilut.hermitian_capability(), AdjointCapability::Exact);
    assert_transpose_inner_product_contract(&parilut, &r, &s);
    assert_hermitian_inner_product_contract(&parilut, &r, &s);

    let hilu = HiluPreconditioner::try_from_csr_uncoupled(&matrix).expect("HILU setup");
    assert_eq!(hilu.transpose_capability(), AdjointCapability::Exact);
    assert_eq!(hilu.hermitian_capability(), AdjointCapability::Exact);
    assert_transpose_inner_product_contract(&hilu, &r, &s);
    assert_hermitian_inner_product_contract(&hilu, &r, &s);
}

#[test]
fn fixed_point_ilu_rejects_unavailable_adjoint_actions() {
    let matrix = CsrMatrix::try_from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            Complex64::new(4.0, 0.5),
            Complex64::new(0.8, -0.2),
            Complex64::new(0.3, 0.1),
            Complex64::new(3.0, -0.75),
            Complex64::new(0.7, 0.2),
            Complex64::new(0.4, -0.15),
            Complex64::new(2.5, 0.3),
        ],
    )
    .expect("canonical fixed-point ILU matrix");
    let preconditioner =
        IluFixedPointPreconditioner::try_from_csr(&matrix, 3).expect("fixed-point ILU setup");
    let r = [
        Complex64::new(1.0, -0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.25, 0.75),
    ];

    assert_eq!(
        preconditioner.transpose_capability(),
        AdjointCapability::Unsupported
    );
    assert_eq!(
        preconditioner.hermitian_capability(),
        AdjointCapability::Unsupported
    );
    assert!(preconditioner.try_apply_transpose(&r).is_err());
    assert!(preconditioner.try_apply_hermitian(&r).is_err());

    let coarse = AdditiveSchwarzPreconditioner::try_from_csr(&matrix, 2, 1)
        .expect("coarse Schwarz setup")
        .with_nicolaides_coarse_space(&matrix);
    assert!(coarse.has_coarse_space());
    assert_eq!(coarse.transpose_capability(), AdjointCapability::Exact);
    assert_eq!(coarse.hermitian_capability(), AdjointCapability::Exact);
    assert!(coarse.try_apply_transpose(&r).is_ok());
    assert!(coarse.try_apply_hermitian(&r).is_ok());
}
fn assert_real_adjoint_property<P: Preconditioner<f64>>(preconditioner: &P, r: &[f64], s: &[f64]) {
    assert_eq!(
        preconditioner.transpose_capability(),
        AdjointCapability::Exact
    );
    assert_eq!(
        preconditioner.hermitian_capability(),
        AdjointCapability::Exact
    );

    let mr = preconditioner.apply(r);
    let mt_s = preconditioner
        .try_apply_transpose(s)
        .expect("exact real transpose preconditioner action");
    let mh_s = preconditioner
        .try_apply_hermitian(s)
        .expect("exact real Hermitian preconditioner action");

    let lhs: f64 = mr.iter().zip(s).map(|(a, b)| a * b).sum();
    let rhs_transpose: f64 = r.iter().zip(&mt_s).map(|(a, b)| a * b).sum();
    let rhs_hermitian: f64 = r.iter().zip(&mh_s).map(|(a, b)| a * b).sum();
    assert!((lhs - rhs_transpose).abs() < 1e-11);
    assert!((lhs - rhs_hermitian).abs() < 1e-11);
    assert!(
        mt_s.iter()
            .zip(&mh_s)
            .all(|(transpose, hermitian)| (transpose - hermitian).abs() < 1e-12)
    );

    for seed in 0..16_u64 {
        let generated_r: Vec<f64> = property_vector(r.len(), seed.wrapping_add(401))
            .into_iter()
            .map(|value| value.re)
            .collect();
        let generated_s: Vec<f64> = property_vector(s.len(), seed.wrapping_add(501))
            .into_iter()
            .map(|value| value.re)
            .collect();
        let generated_mr = preconditioner.apply(&generated_r);
        let generated_mt_s = preconditioner
            .try_apply_transpose(&generated_s)
            .expect("exact real transpose preconditioner action");
        let generated_mh_s = preconditioner
            .try_apply_hermitian(&generated_s)
            .expect("exact real Hermitian preconditioner action");
        let generated_lhs: f64 = generated_mr
            .iter()
            .zip(&generated_s)
            .map(|(a, b)| a * b)
            .sum();
        let generated_rhs: f64 = generated_r
            .iter()
            .zip(&generated_mt_s)
            .map(|(a, b)| a * b)
            .sum();
        let generated_hermitian_rhs: f64 = generated_r
            .iter()
            .zip(&generated_mh_s)
            .map(|(a, b)| a * b)
            .sum();
        assert!((generated_lhs - generated_rhs).abs() < 1e-11);
        assert!((generated_lhs - generated_hermitian_rhs).abs() < 1e-11);
    }
}

#[test]
fn real_preconditioners_satisfy_transpose_and_hermitian_property_contracts() {
    let matrix = CsrMatrix::try_from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![4.0, 0.8, 0.3, 3.0, 0.7, 0.4, 2.5],
    )
    .expect("canonical real property matrix");
    let r = [1.0, -2.0, 0.25];
    let s = [0.25, 1.5, -0.5];

    let ilu = IluPreconditioner::try_from_csr(&matrix).expect("ILU setup");
    assert_real_adjoint_property(&ilu, &r, &s);

    let ilut = IlutPreconditioner::try_from_csr(&matrix).expect("ILUT setup");
    assert_real_adjoint_property(&ilut, &r, &s);

    let block_jacobi = BlockJacobiPreconditioner::try_from_csr(&matrix, 2, IlutConfig::default())
        .expect("Block-Jacobi setup");
    assert_real_adjoint_property(&block_jacobi, &r, &s);

    let diagonal =
        DiagonalPreconditioner::try_from_diagonal(&[4.0, 3.0, 2.5]).expect("diagonal setup");
    assert_real_adjoint_property(&diagonal, &r, &s);
}

#[test]
fn chebyshev_checked_spd_constructor_exposes_exact_complex_adjoint_actions() {
    let matrix = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, 0.0),
        ],
    );
    let preconditioner = ChebyshevPreconditioner::try_from_csr(
        &matrix,
        ChebyshevConfig {
            degree: 2,
            ..ChebyshevConfig::default()
        },
    )
    .expect("checked SPD Chebyshev setup");
    assert_eq!(
        preconditioner.transpose_capability(),
        AdjointCapability::Exact
    );
    assert_eq!(
        preconditioner.hermitian_capability(),
        AdjointCapability::Exact
    );
    let r = property_vector(2, 0x1234);
    let s = property_vector(2, 0x5678);
    assert_hermitian_inner_product_contract(&preconditioner, &r, &s);
    assert_transpose_inner_product_contract(&preconditioner, &r, &s);
}

#[test]
fn saddle_point_adjoint_contract_is_exact_only_for_block_diagonal_mode() {
    let matrix = CsrMatrix::try_from_raw_parts(
        4,
        4,
        vec![0, 1, 2, 3, 4],
        vec![0, 1, 2, 3],
        vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, -0.5),
            Complex64::new(4.0, 0.25),
            Complex64::new(5.0, -0.75),
        ],
    )
    .expect("canonical block-diagonal saddle matrix");

    let diagonal_config = SaddlePointConfig {
        block1_size: 2,
        block1_config: BlockPreconditionerConfig::Diagonal,
        block2_config: BlockPreconditionerConfig::Diagonal,
        schur_strategy: SchurStrategy::Diagonal,
        upper_triangular: false,
    };
    let diagonal = SaddlePointPreconditioner::from_matrix(&matrix, diagonal_config);
    assert_eq!(diagonal.transpose_capability(), AdjointCapability::Exact);
    assert_eq!(diagonal.hermitian_capability(), AdjointCapability::Exact);

    let input = vec![
        Complex64::new(1.0, -0.5),
        Complex64::new(-2.0, 0.25),
        Complex64::new(0.75, 1.5),
        Complex64::new(-1.25, -0.5),
    ];
    let transpose = diagonal
        .try_apply_transpose(&input)
        .expect("block-diagonal transpose is exact");
    let hermitian = diagonal
        .try_apply_hermitian(&input)
        .expect("block-diagonal Hermitian action is exact");
    for ((value, &entry), diagonal_value) in transpose.iter().zip(&input).zip([
        Complex64::new(2.0, 1.0),
        Complex64::new(3.0, -0.5),
        Complex64::new(4.0, 0.25),
        Complex64::new(5.0, -0.75),
    ]) {
        assert!((*value - entry / diagonal_value).norm() < 1.0e-12);
    }
    for ((value, &entry), diagonal_value) in hermitian.iter().zip(&input).zip([
        Complex64::new(2.0, 1.0),
        Complex64::new(3.0, -0.5),
        Complex64::new(4.0, 0.25),
        Complex64::new(5.0, -0.75),
    ]) {
        assert!((*value - entry / diagonal_value.conj()).norm() < 1.0e-12);
    }

    let triangular_config = SaddlePointConfig {
        block1_size: 2,
        block1_config: BlockPreconditionerConfig::Diagonal,
        block2_config: BlockPreconditionerConfig::Diagonal,
        schur_strategy: SchurStrategy::External,
        upper_triangular: true,
    };
    let schur = CsrMatrix::try_from_raw_parts(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
    )
    .expect("canonical Schur approximation");
    let triangular =
        SaddlePointPreconditioner::from_matrix_with_schur(&matrix, &schur, triangular_config);
    assert_eq!(triangular.transpose_capability(), AdjointCapability::Exact);
    assert_eq!(triangular.hermitian_capability(), AdjointCapability::Exact);
    triangular
        .try_apply_transpose(&input)
        .expect("triangular transpose is exact");
    triangular
        .try_apply_hermitian(&input)
        .expect("triangular Hermitian action is exact");
}

#[test]
fn amg_adjoint_capability_is_explicitly_unsupported_and_fails_closed() {
    let matrix = CsrMatrix::from_triplets(
        8,
        8,
        (0..8)
            .flat_map(|row| {
                let mut entries = vec![(row, row, Complex64::new(2.0, 0.0))];
                if row > 0 {
                    entries.push((row, row - 1, Complex64::new(-1.0, 0.0)));
                }
                if row + 1 < 8 {
                    entries.push((row, row + 1, Complex64::new(-1.0, 0.0)));
                }
                entries
            })
            .collect(),
    );
    let amg = AmgPreconditioner::try_from_csr(&matrix, AmgConfig::default())
        .expect("AMG setup for adjoint capability contract");

    assert_eq!(amg.transpose_capability(), AdjointCapability::Unsupported);
    assert_eq!(amg.hermitian_capability(), AdjointCapability::Unsupported);
    let input = vec![Complex64::new(0.25, -0.5); 8];
    assert!(matches!(
        amg.try_apply_transpose(&input),
        Err(SolverError::UnsupportedOperation { .. })
    ));
    assert!(matches!(
        amg.try_apply_hermitian(&input),
        Err(SolverError::UnsupportedOperation { .. })
    ));
    assert!(
        amg.try_apply(&input).is_ok(),
        "forward AMG remains available"
    );
}

#[test]
fn randomized_complex_adjoint_properties_cover_chebyshev_preconditioner() {
    let matrix = CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(0.75, 0.0),
            Complex64::new(0.75, 0.0),
            Complex64::new(2.5, 0.0),
        ],
    );
    let preconditioner = ChebyshevPreconditioner::try_from_csr(
        &matrix,
        ChebyshevConfig {
            degree: 3,
            ..ChebyshevConfig::default()
        },
    )
    .expect("checked SPD Chebyshev setup");
    assert_eq!(
        preconditioner.transpose_capability(),
        AdjointCapability::Exact
    );
    assert_eq!(
        preconditioner.hermitian_capability(),
        AdjointCapability::Exact
    );
    let r = property_vector(3, 0x000C_EB00_0000_u64);
    let s = property_vector(3, 0x000C_EB00_0100_u64);
    assert_hermitian_inner_product_contract(&preconditioner, &r, &s);
    assert_transpose_inner_product_contract(&preconditioner, &r, &s);
}

#[test]
fn randomized_complex_adjoint_properties_cover_composed_preconditioner_family() {
    let matrix = CsrMatrix::try_from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            Complex64::new(4.0, 0.5),
            Complex64::new(0.8, -0.2),
            Complex64::new(0.3, 0.1),
            Complex64::new(3.0, -0.75),
            Complex64::new(0.7, 0.2),
            Complex64::new(0.4, -0.15),
            Complex64::new(2.5, 0.3),
        ],
    )
    .expect("canonical complex property matrix");

    let ilu = IluPreconditioner::try_from_csr(&matrix).expect("ILU setup");
    let ilut = IlutPreconditioner::try_from_csr(&matrix).expect("ILUT setup");
    let block_jacobi = BlockJacobiPreconditioner::try_from_csr(&matrix, 2, IlutConfig::default())
        .expect("Block-Jacobi setup");
    let diagonal = DiagonalPreconditioner::try_from_diagonal(&[
        Complex64::new(4.0, 0.5),
        Complex64::new(3.0, -0.75),
        Complex64::new(2.5, 0.3),
    ])
    .expect("diagonal setup");
    let schwarz =
        AdditiveSchwarzPreconditioner::try_from_csr(&matrix, 2, 1).expect("Schwarz setup");
    let parilut = ParIlutPreconditioner::try_from_csr(&matrix).expect("ParILUT setup");
    let hilu = HiluPreconditioner::try_from_csr_uncoupled(&matrix).expect("HILU setup");

    for seed in 0..16_u64 {
        let r = property_vector(3, 0xA5A5_0000_u64.wrapping_add(seed));
        let s = property_vector(3, 0x5A5A_1000_u64.wrapping_add(seed * 17));
        assert_hermitian_inner_product_contract(&ilu, &r, &s);
        assert_transpose_inner_product_contract(&ilu, &r, &s);
        assert_hermitian_inner_product_contract(&ilut, &r, &s);
        assert_transpose_inner_product_contract(&ilut, &r, &s);
        assert_hermitian_inner_product_contract(&block_jacobi, &r, &s);
        assert_transpose_inner_product_contract(&block_jacobi, &r, &s);
        assert_hermitian_inner_product_contract(&diagonal, &r, &s);
        assert_transpose_inner_product_contract(&diagonal, &r, &s);
        assert_hermitian_inner_product_contract(&schwarz, &r, &s);
        assert_transpose_inner_product_contract(&schwarz, &r, &s);
        assert_hermitian_inner_product_contract(&parilut, &r, &s);
        assert_transpose_inner_product_contract(&parilut, &r, &s);
        assert_hermitian_inner_product_contract(&hilu, &r, &s);
        assert_transpose_inner_product_contract(&hilu, &r, &s);
    }
}

#[test]
fn amg_symmetric_v_cycle_exposes_exact_adjoint_actions() {
    let mut entries = Vec::new();
    for row in 0..8 {
        entries.push((row, row, Complex64::new(4.0, 0.0)));
        if row + 1 < 8 {
            let upper = Complex64::new(-0.8, 0.2);
            entries.push((row, row + 1, upper));
            entries.push((row + 1, row, upper.conj()));
        }
    }
    let matrix = CsrMatrix::from_triplets(8, 8, entries);
    let config = AmgConfig {
        restriction: AmgRestriction::ConjugateTranspose,
        smoother: AmgSmoother::SymmetricGaussSeidel,
        cycle: AmgCycle::VCycle,
        num_pre_smooth: 1,
        num_post_smooth: 1,
        k_cycle_iterations: 0,
        f_point_smoothing: false,
        ..AmgConfig::default()
    };
    let amg = AmgPreconditioner::try_from_csr(&matrix, config).expect("Hermitian AMG setup");

    assert_eq!(amg.transpose_capability(), AdjointCapability::Unsupported);
    assert_eq!(amg.hermitian_capability(), AdjointCapability::Exact);

    let r = property_vector(8, 0xA11CE);
    let s = property_vector(8, 0xB00B5);
    assert_hermitian_inner_product_contract(&amg, &r, &s);
    let forward = amg.try_apply(&r).expect("forward AMG action");
    let hermitian = amg
        .try_apply_hermitian(&r)
        .expect("exact Hermitian AMG action");
    assert!(
        forward
            .iter()
            .zip(&hermitian)
            .all(|(lhs, rhs)| (*lhs - *rhs).norm() < 1.0e-10),
        "self-adjoint AMG action must equal its Hermitian action"
    );
}
