use num_complex::Complex64;
use sonium_solvers_core::traits::{AdjointCapability, Preconditioner};
use sonium_solvers_preconditioners::{
    BlockPreconditionerConfig, LscBlocks, PcdBlocks, SaddlePointConfig, SaddlePointPreconditioner,
    SchurStrategy,
};
use sonium_solvers_sparse::CsrMatrix;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn saddle_matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::from_triplets(
        4,
        4,
        vec![
            (0, 0, c(2.0, 0.5)),
            (0, 1, c(0.2, -0.1)),
            (0, 2, c(0.7, 0.3)),
            (0, 3, c(-0.4, 0.2)),
            (1, 0, c(-0.1, 0.4)),
            (1, 1, c(3.0, -0.25)),
            (1, 2, c(0.5, -0.2)),
            (1, 3, c(0.6, 0.1)),
            (2, 0, c(0.3, -0.2)),
            (2, 1, c(-0.8, 0.1)),
            (2, 2, c(1.0, 0.2)),
            (3, 0, c(0.2, 0.5)),
            (3, 1, c(0.4, -0.3)),
            (3, 3, c(1.5, -0.1)),
        ],
    )
}

fn property_vector(seed: u64) -> Vec<Complex64> {
    let mut state = seed;
    (0..4)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let re = ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let im = ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0;
            c(re, im)
        })
        .collect()
}

fn diagonal_schur() -> CsrMatrix<Complex64> {
    CsrMatrix::from_triplets(2, 2, vec![(0, 0, c(1.5, 0.2)), (1, 1, c(2.2, -0.4))])
}

fn diagonal_block_config() -> SaddlePointConfig {
    SaddlePointConfig {
        block1_size: 2,
        block1_config: BlockPreconditionerConfig::Diagonal,
        block2_config: BlockPreconditionerConfig::Diagonal,
        schur_strategy: SchurStrategy::Diagonal,
        upper_triangular: true,
    }
}

fn assert_adjoint_contract<P: Preconditioner<Complex64>>(preconditioner: &P) {
    assert_eq!(
        preconditioner.transpose_capability(),
        AdjointCapability::Exact
    );
    assert_eq!(
        preconditioner.hermitian_capability(),
        AdjointCapability::Exact
    );

    for seed in 0..16_u64 {
        let r = property_vector(0x1234_0000 + seed);
        let s = property_vector(0x5678_0000 + seed);
        let mr = preconditioner.apply(&r);
        let mt_s = preconditioner
            .try_apply_transpose(&s)
            .expect("exact transpose action");
        let mh_s = preconditioner
            .try_apply_hermitian(&s)
            .expect("exact Hermitian action");

        let transpose_lhs: Complex64 = mr.iter().zip(&s).map(|(a, b)| *a * *b).sum();
        let transpose_rhs: Complex64 = r.iter().zip(mt_s).map(|(a, b)| *a * b).sum();
        assert!(
            (transpose_lhs - transpose_rhs).norm() < 1.0e-11,
            "transpose mismatch for seed {seed}: {transpose_lhs:?} vs {transpose_rhs:?}"
        );

        let hermitian_lhs: Complex64 = mr.iter().zip(&s).map(|(a, b)| a.conj() * *b).sum();
        let hermitian_rhs: Complex64 = r.iter().zip(mh_s).map(|(a, b)| a.conj() * b).sum();
        assert!(
            (hermitian_lhs - hermitian_rhs).norm() < 1.0e-11,
            "Hermitian mismatch for seed {seed}: {hermitian_lhs:?} vs {hermitian_rhs:?}"
        );
    }
}

#[test]
fn triangular_and_composite_saddle_adjoint_actions_are_exact() {
    let matrix = saddle_matrix();

    for upper_triangular in [true, false] {
        let mut external_config = diagonal_block_config();
        external_config.upper_triangular = upper_triangular;
        let external = SaddlePointPreconditioner::from_matrix_with_schur(
            &matrix,
            &diagonal_schur(),
            external_config,
        );
        assert_adjoint_contract(&external);

        let mut btdiagb_config = diagonal_block_config();
        btdiagb_config.schur_strategy = SchurStrategy::BtDiagB;
        btdiagb_config.upper_triangular = upper_triangular;
        let btdiagb = SaddlePointPreconditioner::from_matrix(&matrix, btdiagb_config);
        assert_adjoint_contract(&btdiagb);

        let mut simple_config = diagonal_block_config();
        simple_config.schur_strategy = SchurStrategy::Simple;
        simple_config.upper_triangular = upper_triangular;
        let simple = SaddlePointPreconditioner::from_matrix(&matrix, simple_config);
        assert_adjoint_contract(&simple);

        let mut pcd_config = diagonal_block_config();
        pcd_config.schur_strategy = SchurStrategy::Pcd;
        pcd_config.upper_triangular = upper_triangular;
        let pcd = SaddlePointPreconditioner::from_matrix_with_pcd(
            &matrix,
            &PcdBlocks {
                fp: CsrMatrix::from_triplets(
                    2,
                    2,
                    vec![
                        (0, 0, c(1.2, 0.3)),
                        (0, 1, c(-0.4, 0.2)),
                        (1, 0, c(0.5, -0.1)),
                        (1, 1, c(1.8, -0.25)),
                    ],
                ),
                ap: diagonal_schur(),
                mp: CsrMatrix::from_triplets(2, 2, vec![(0, 0, c(1.1, 0.1)), (1, 1, c(0.9, -0.2))]),
            },
            pcd_config,
        );
        assert_adjoint_contract(&pcd);

        let mut lsc_config = diagonal_block_config();
        lsc_config.schur_strategy = SchurStrategy::Lsc;
        lsc_config.upper_triangular = upper_triangular;
        let lsc = SaddlePointPreconditioner::from_matrix_with_lsc(
            &matrix,
            &LscBlocks {
                velocity_mass_diag_inv: vec![c(1.0, 0.1), c(0.8, -0.2)],
            },
            lsc_config,
        );
        assert_adjoint_contract(&lsc);
    }
}
