use num_complex::Complex64;
use sonium_solvers_core::traits::LinearOperator;
use sonium_solvers_sparse::CsrMatrix;
use sonium_solvers_sparse::spmv_plan::{SpmvPlan, SpmvPlanConfig, SpmvPlanPreference};

fn forced_bsr2_config() -> SpmvPlanConfig {
    SpmvPlanConfig {
        preference: SpmvPlanPreference::Bsr2,
        ..SpmvPlanConfig::default()
    }
}

#[test]
fn forced_bsr_preserves_duplicate_csr_entries() {
    // Duplicate CSR columns are valid sparse storage and must be summed.
    let matrix =
        CsrMatrix::from_raw_parts(2, 2, vec![0, 2, 3], vec![0, 0, 1], vec![1.0_f64, 2.0, 3.0]);
    let csr = matrix.try_apply(&[1.0, 1.0]).expect("CSR action");
    assert_eq!(csr, vec![3.0, 3.0]);
    let plan = SpmvPlan::try_from_csr(&matrix, forced_bsr2_config()).expect("BSR plan");
    assert_eq!(
        plan.format(),
        sonium_solvers_sparse::spmv_plan::SpmvPlanFormat::Bsr2
    );
    assert_eq!(plan.try_apply(&[1.0, 1.0]).expect("BSR action"), csr);
}

#[test]
fn forced_bsr_rectangular_transpose_and_hermitian_have_csr_shapes() {
    let matrix = CsrMatrix::from_raw_parts(
        4,
        6,
        vec![0, 3, 5, 8, 10],
        vec![0, 2, 5, 1, 4, 0, 3, 5, 2, 4],
        vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(-2.0, 1.0),
            Complex64::new(0.25, -0.75),
            Complex64::new(3.0, -1.0),
            Complex64::new(-0.5, 2.0),
            Complex64::new(0.75, 0.25),
            Complex64::new(2.0, -0.5),
            Complex64::new(-1.0, 1.5),
            Complex64::new(0.5, 0.0),
            Complex64::new(1.25, -0.25),
        ],
    );
    let plan = SpmvPlan::try_from_csr(&matrix, forced_bsr2_config()).expect("BSR plan");
    assert_eq!(plan.num_rows(), 4);
    assert_eq!(plan.num_cols(), 6);

    let rhs = vec![
        Complex64::new(0.5, -1.0),
        Complex64::new(1.0, 0.25),
        Complex64::new(-0.75, 0.5),
        Complex64::new(2.0, -0.5),
    ];
    let csr_transpose = matrix.try_apply_transpose(&rhs).expect("CSR transpose");
    let bsr_transpose = plan.try_apply_transpose(&rhs).expect("BSR transpose");
    assert_eq!(bsr_transpose.len(), 6);
    for (got, expected) in bsr_transpose.iter().zip(csr_transpose) {
        assert!((*got - expected).norm() < 1.0e-12);
    }

    let csr_hermitian = matrix.try_apply_hermitian(&rhs).expect("CSR Hermitian");
    let bsr_hermitian = plan.try_apply_hermitian(&rhs).expect("BSR Hermitian");
    assert_eq!(bsr_hermitian.len(), 6);
    for (got, expected) in bsr_hermitian.iter().zip(csr_hermitian) {
        assert!((*got - expected).norm() < 1.0e-12);
    }
}
