use num_complex::Complex64;
use sonium_solvers_core::Preconditioner;
use sonium_solvers_preconditioners::{
    AmgCoarsening, AmgConfig, AmgCycle, AmgInterpolation, AmgPreconditioner, AmgRestriction,
    AmgSmoother, MaterialSocConfig,
};
use sonium_solvers_sparse::CsrMatrix;

fn diagonal_matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::from_raw_parts(
        8,
        8,
        (0..=8).collect(),
        (0..8).collect(),
        (0..8)
            .map(|index| Complex64::new(2.0 + index as f64, 0.0))
            .collect(),
    )
}

fn configured() -> AmgConfig {
    AmgConfig {
        coarsening: AmgCoarsening::RugeStuben,
        interpolation: AmgInterpolation::Extended,
        restriction: AmgRestriction::ConjugateTranspose,
        symmetrize_strength: true,
        smoother: AmgSmoother::Jacobi,
        f_point_smoothing: true,
        cycle: AmgCycle::VCycle,
        k_cycle_iterations: 2,
        strong_threshold: 0.3,
        max_levels: 4,
        coarse_size: 2,
        num_pre_smooth: 2,
        num_post_smooth: 3,
        jacobi_weight: 0.7,
        trunc_factor: 0.05,
        galerkin_drop_tol: 0.001,
        max_interp_elements: 6,
        aggressive_coarsening_levels: 1,
        mis_distance: 2,
        sa_strength_with_ata: false,
        material_soc: Some(MaterialSocConfig {
            tensors: (0..8).map(|_| vec![1.0]).collect(),
            dim: 1,
        }),
        energy_min_iterations: 3,
        energy_min_pattern_distance: 2,
        near_nullspace_dim: 1,
        block_size: 1,
        max_grid_complexity: Some(4.0),
        max_operator_complexity: Some(8.0),
        #[cfg(feature = "experimental-gnn")]
        gnn_weights: None,
    }
}

#[test]
fn amg_checkpoint_round_trip_preserves_actions_and_configuration() {
    let matrix = diagonal_matrix();
    let original = AmgPreconditioner::try_from_csr(&matrix, configured()).expect("AMG setup");
    let bytes = original.to_checkpoint_bytes().expect("AMG checkpoint");
    let restored =
        AmgPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes).expect("AMG restore");
    assert_eq!(restored.config().coarsening, AmgCoarsening::RugeStuben);
    assert_eq!(restored.config().cycle, AmgCycle::VCycle);
    assert_eq!(restored.config().restriction, configured().restriction);
    assert_eq!(restored.num_levels(), original.num_levels());

    let rhs: Vec<_> = (0..8)
        .map(|index| Complex64::new(index as f64 + 0.25, -0.5 * index as f64))
        .collect();
    assert_eq!(restored.apply(&rhs), original.apply(&rhs));
}

#[test]
fn amg_checkpoint_default_configuration_preserves_adjoint_actions() {
    let matrix = diagonal_matrix();
    let original = AmgPreconditioner::try_from_csr(&matrix, AmgConfig::default()).expect("AMG");
    let bytes = original.to_checkpoint_bytes().expect("AMG checkpoint");
    let restored =
        AmgPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes).expect("AMG restore");
    let rhs: Vec<_> = (0..8)
        .map(|index| Complex64::new(index as f64 + 0.25, -0.5 * index as f64))
        .collect();
    assert_eq!(restored.apply(&rhs), original.apply(&rhs));
    assert_eq!(
        restored.apply_transpose(&rhs),
        original.apply_transpose(&rhs)
    );
    assert_eq!(
        restored.apply_hermitian(&rhs),
        original.apply_hermitian(&rhs)
    );
}

#[test]
fn amg_checkpoint_rejects_scalar_mismatch_truncation_and_trailing_bytes() {
    let matrix = CsrMatrix::<f64>::identity(4);
    let original = AmgPreconditioner::try_from_csr(&matrix, AmgConfig::default()).expect("AMG");
    let bytes = original.to_checkpoint_bytes().expect("AMG checkpoint");
    assert!(AmgPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes).is_err());
    assert!(AmgPreconditioner::<f64>::from_checkpoint_bytes(&bytes[..bytes.len() - 1]).is_err());
    let mut trailing = bytes;
    trailing.push(0);
    assert!(AmgPreconditioner::<f64>::from_checkpoint_bytes(&trailing).is_err());
}

#[test]
fn amg_checkpoint_rejects_corrupt_header_and_reserved_flags() {
    let matrix = CsrMatrix::<f64>::identity(4);
    let original = AmgPreconditioner::try_from_csr(&matrix, AmgConfig::default()).expect("AMG");
    let bytes = original.to_checkpoint_bytes().expect("AMG checkpoint");
    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 0xff;
    assert!(AmgPreconditioner::<f64>::from_checkpoint_bytes(&bad_magic).is_err());
    let mut bad_flags = bytes;
    bad_flags[13] = 1;
    assert!(AmgPreconditioner::<f64>::from_checkpoint_bytes(&bad_flags).is_err());

    // Header (14 bytes), dimensions (16 bytes), enum/flag prefix (7 bytes),
    // and twelve u64 configuration values put `strong_threshold` at offset
    // 133 in format v1. A NaN must be rejected before hierarchy construction.
    let mut bad_finite = original.to_checkpoint_bytes().expect("AMG checkpoint");
    bad_finite[133..141].copy_from_slice(&f64::NAN.to_le_bytes());
    assert!(AmgPreconditioner::<f64>::from_checkpoint_bytes(&bad_finite).is_err());
}
