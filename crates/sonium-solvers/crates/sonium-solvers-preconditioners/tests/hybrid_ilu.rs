use num_complex::Complex64;
use sonium_solvers_core::traits::Preconditioner;
use sonium_solvers_iterative::{FgmresConfig, fgmres_preconditioned};
use sonium_solvers_preconditioners::{
    HybridIluConfig, HybridIluPreconditioner, IluPreconditioner, IlutConfig, IlutPreconditioner,
};
use sonium_solvers_sparse::CsrMatrix;

fn cyclic_matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::from_triplets(
        4,
        4,
        vec![
            (0, 0, 4.0.into()),
            (0, 1, 1.0.into()),
            (0, 3, 1.0.into()),
            (1, 0, 1.0.into()),
            (1, 1, 4.0.into()),
            (1, 2, 1.0.into()),
            (2, 1, 1.0.into()),
            (2, 2, 4.0.into()),
            (2, 3, 1.0.into()),
            (3, 0, 1.0.into()),
            (3, 2, 1.0.into()),
            (3, 3, 4.0.into()),
        ],
    )
}

#[test]
fn valid_low_rank_term_is_applied() {
    let matrix = cyclic_matrix();
    let basis = vec![vec![1.0.into(), 0.0.into(), 0.0.into(), 0.0.into()]];
    let correction = vec![Complex64::new(0.01, -0.02)];
    let hybrid = HybridIluPreconditioner::try_from_csr(
        &matrix,
        basis,
        correction,
        HybridIluConfig {
            max_probe_residual_ratio: 100.0,
        },
    )
    .expect("valid hybrid preconditioner");
    assert!(hybrid.augmentation_active());

    let rhs = vec![Complex64::new(2.0, 3.0), 1.0.into(), 0.0.into(), 0.0.into()];
    let baseline = IluPreconditioner::try_from_csr(&matrix)
        .expect("ILU")
        .apply(&rhs);
    let applied = hybrid.apply(&rhs);
    assert_eq!(applied[0], baseline[0] + Complex64::new(0.08, -0.01));
    assert_eq!(&applied[1..], &baseline[1..]);
}

#[test]
fn invalid_learned_shapes_fall_back_to_plain_ilu() {
    let matrix = cyclic_matrix();
    let hybrid = HybridIluPreconditioner::try_from_csr(
        &matrix,
        vec![vec![1.0.into(); 3]],
        vec![1.0.into()],
        HybridIluConfig::default(),
    )
    .expect("bad learned data must not discard valid ILU");
    assert!(!hybrid.augmentation_active());
    assert!(hybrid.diagnostics().rejection_reason.is_some());

    let rhs = vec![1.0.into(), 2.0.into(), 3.0.into(), 4.0.into()];
    let baseline = IluPreconditioner::try_from_csr(&matrix)
        .expect("ILU")
        .apply(&rhs);
    assert_eq!(hybrid.apply(&rhs), baseline);
}

#[test]
fn non_finite_and_worsening_terms_fall_back_without_panic() {
    let matrix = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.0.into()), (1, 1, 1.0.into())]);
    let non_finite = HybridIluPreconditioner::try_from_csr(
        &matrix,
        vec![vec![Complex64::new(f64::NAN, 0.0), 0.0.into()]],
        vec![1.0.into()],
        HybridIluConfig::default(),
    )
    .expect("non-finite augmentation falls back");
    assert!(!non_finite.augmentation_active());

    let worsening = HybridIluPreconditioner::try_from_csr(
        &matrix,
        vec![vec![1.0.into(), 0.0.into()]],
        vec![1.0.into()],
        HybridIluConfig::default(),
    )
    .expect("worsening augmentation falls back");
    assert!(!worsening.augmentation_active());
    assert_eq!(
        worsening.apply(&[1.0.into(), 2.0.into()]),
        vec![1.0.into(), 2.0.into()]
    );
}

#[test]
fn every_setup_probe_must_pass_and_rank_is_bounded() {
    let matrix = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.0.into()), (1, 1, 1.0.into())]);
    let probes = vec![vec![0.0.into(), 1.0.into()], vec![1.0.into(), 0.0.into()]];
    let hybrid = HybridIluPreconditioner::try_from_ilu_with_probes(
        &matrix,
        IluPreconditioner::try_from_csr(&matrix).unwrap(),
        vec![vec![1.0.into(), 0.0.into()]],
        vec![1.0.into()],
        &probes,
        HybridIluConfig::default(),
    )
    .unwrap();
    assert!(!hybrid.augmentation_active());
    assert_eq!(hybrid.diagnostics().probes_evaluated, 2);
    assert!(hybrid.diagnostics().worst_probe_residual_ratio.is_some());

    let too_many = vec![vec![Complex64::ZERO; 2]; 65];
    let rejected = HybridIluPreconditioner::try_from_csr(
        &matrix,
        too_many,
        vec![Complex64::ZERO; 65 * 65],
        HybridIluConfig::default(),
    )
    .unwrap();
    assert!(!rejected.augmentation_active());
    assert!(
        rejected
            .diagnostics()
            .rejection_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("supported maximum"))
    );
}

#[test]
fn hybrid_wrapper_accepts_ilut_as_its_classical_base() {
    let matrix = cyclic_matrix();
    let ilut = IlutPreconditioner::try_from_csr_with_config(&matrix, &IlutConfig::default())
        .expect("ILUT");
    let rhs = vec![1.0.into(), 2.0.into(), 3.0.into(), 4.0.into()];
    let baseline = ilut.apply(&rhs);
    let hybrid = HybridIluPreconditioner::try_from_ilut_with_probes(
        &matrix,
        ilut,
        Vec::new(),
        Vec::new(),
        &[rhs.clone()],
        HybridIluConfig::default(),
    )
    .expect("ILUT-backed hybrid preconditioner");

    assert!(!hybrid.augmentation_active());
    assert_eq!(hybrid.apply(&rhs), baseline);
}

#[test]
fn overshot_correction_is_damped_until_it_improves_the_probe() {
    let matrix = cyclic_matrix();
    let ilu = IluPreconditioner::try_from_csr(&matrix).expect("ILU");
    let rhs = vec![1.0.into(), 0.0.into(), 0.0.into(), 0.0.into()];
    let baseline = ilu.apply(&rhs);
    let baseline_applied = matrix.matvec(&baseline);
    let residual: Vec<_> = rhs
        .iter()
        .zip(&baseline_applied)
        .map(|(right, left)| *right - *left)
        .collect();
    let mode = vec![1.0.into(), 0.0.into(), 0.0.into(), 0.0.into()];
    let applied_mode = matrix.matvec(&mode);
    let numerator: Complex64 = applied_mode
        .iter()
        .zip(&residual)
        .map(|(left, right)| left.conj() * right)
        .sum();
    let denominator: f64 = applied_mode.iter().map(|value| value.norm_sqr()).sum();
    assert!(
        numerator.norm() > 1.0e-12,
        "test matrix must leave an ILU residual"
    );

    // Three times the least-squares step overshoots; a half step improves it.
    let hybrid = HybridIluPreconditioner::try_from_ilu_with_probes(
        &matrix,
        ilu,
        vec![mode],
        vec![3.0 * numerator / denominator],
        &[rhs.clone()],
        HybridIluConfig::default(),
    )
    .expect("damped hybrid");

    assert!(hybrid.augmentation_active());
    assert_eq!(hybrid.diagnostics().accepted_correction_scale, Some(0.5));
    assert!(hybrid.diagnostics().worst_probe_residual_ratio.unwrap() < 1.0);
    let hybrid_residual: f64 = matrix
        .matvec(&hybrid.apply(&rhs))
        .iter()
        .zip(&rhs)
        .map(|(left, right)| (*right - *left).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let baseline_residual: f64 = baseline_applied
        .iter()
        .zip(&rhs)
        .map(|(left, right)| (*right - *left).norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!(hybrid_residual < baseline_residual);
}

#[test]
fn hybrid_preconditioner_runs_through_fgmres_with_exact_residual_convergence() {
    let matrix = cyclic_matrix();
    let hybrid = HybridIluPreconditioner::try_from_csr(
        &matrix,
        vec![vec![1.0.into(), 0.0.into(), 0.0.into(), 0.0.into()]],
        vec![Complex64::ZERO],
        HybridIluConfig::default(),
    )
    .expect("valid low-rank wrapper");
    let rhs = vec![1.0.into(), 2.0.into(), 3.0.into(), 4.0.into()];
    let solved = fgmres_preconditioned(
        &matrix,
        &hybrid,
        &rhs,
        None,
        &FgmresConfig {
            max_iterations: 30,
            restart: 10,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
        },
    );
    assert!(solved.converged);
    let residual = matrix
        .matvec(&solved.x)
        .into_iter()
        .zip(&rhs)
        .map(|(actual, expected)| (actual - expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!(residual < 1.0e-10);
}
