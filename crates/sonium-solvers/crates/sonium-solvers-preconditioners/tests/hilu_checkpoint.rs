use num_complex::Complex64;
use num_traits::{FromPrimitive, Zero};
use sonium_solvers_core::traits::Preconditioner;
use sonium_solvers_preconditioners::{HiluConfig, HiluPreconditioner};
use sonium_solvers_sparse::CsrMatrix;

fn matrix<T: sonium_solvers_core::traits::ComplexField>(n: usize) -> CsrMatrix<T> {
    let mut row_ptrs = vec![0];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n {
        if row > 0 {
            col_indices.push(row - 1);
            values.push(T::from_re_im(
                T::Real::from_f64(0.2).unwrap(),
                T::Real::zero(),
            ));
        }
        col_indices.push(row);
        values.push(T::from_re_im(
            T::Real::from_f64(4.0).unwrap(),
            T::Real::zero(),
        ));
        if row + 1 < n {
            col_indices.push(row + 1);
            values.push(T::from_re_im(
                T::Real::from_f64(0.2).unwrap(),
                T::Real::zero(),
            ));
        }
        row_ptrs.push(values.len());
    }
    CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
}

fn vector<T: sonium_solvers_core::traits::ComplexField>(n: usize) -> Vec<T> {
    (0..n)
        .map(|index| {
            T::from_re_im(
                T::Real::from_f64(1.0 + index as f64 * 0.03).unwrap(),
                T::Real::from_f64(0.1 - index as f64 * 0.01).unwrap(),
            )
        })
        .collect()
}

fn assert_close<T: sonium_solvers_core::traits::ComplexField>(left: &[T], right: &[T]) {
    assert_eq!(left.len(), right.len());
    let tolerance = T::Real::from_f64(1e-10).unwrap();
    for (a, b) in left.iter().zip(right) {
        assert!((*a - *b).norm() < tolerance);
    }
}

#[test]
fn uncoupled_real_round_trip_preserves_actions() {
    let source = HiluPreconditioner::<f64>::try_from_csr_uncoupled_with_config(
        &matrix(20),
        &HiluConfig {
            min_block_size: 4,
            max_levels: Some(3),
        },
    )
    .expect("HILU setup");
    let restored = HiluPreconditioner::<f64>::from_checkpoint_bytes(
        &source.to_checkpoint_bytes().expect("HILU serialization"),
    )
    .expect("HILU restore");
    let rhs = vector::<f64>(20);
    assert_close(&source.apply(&rhs), &restored.apply(&rhs));
    assert_close(
        &source.apply_transpose(&rhs),
        &restored.apply_transpose(&rhs),
    );
    assert_close(
        &source.apply_hermitian(&rhs),
        &restored.apply_hermitian(&rhs),
    );
}

#[test]
fn coupled_complex_round_trip_preserves_actions() {
    let source = HiluPreconditioner::<Complex64>::try_from_csr_coupled_with_config(
        &matrix(20),
        &HiluConfig {
            min_block_size: 4,
            max_levels: Some(3),
        },
        0.0,
    )
    .expect("HILU setup");
    let restored = HiluPreconditioner::<Complex64>::from_checkpoint_bytes(
        &source.to_checkpoint_bytes().expect("HILU serialization"),
    )
    .expect("HILU restore");
    let rhs = vector::<Complex64>(20);
    assert_close(&source.apply(&rhs), &restored.apply(&rhs));
    assert_close(
        &source.apply_transpose(&rhs),
        &restored.apply_transpose(&rhs),
    );
    assert_close(
        &source.apply_hermitian(&rhs),
        &restored.apply_hermitian(&rhs),
    );
}

#[test]
fn malformed_hilu_checkpoint_is_rejected() {
    let source = HiluPreconditioner::<f64>::try_from_csr_uncoupled(&matrix(8)).expect("HILU setup");
    let bytes = source.to_checkpoint_bytes().expect("HILU serialization");

    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 1;
    assert!(HiluPreconditioner::<f64>::from_checkpoint_bytes(&bad_magic).is_err());
    assert!(HiluPreconditioner::<f64>::from_checkpoint_bytes(&bytes[..bytes.len() - 1]).is_err());
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(HiluPreconditioner::<f64>::from_checkpoint_bytes(&trailing).is_err());
    assert!(HiluPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes).is_err());

    // Header dimension is at byte 14; changing it must invalidate the saved
    // block partition rather than silently accepting a mismatched topology.
    let mut bad_dimension = bytes;
    bad_dimension[14..22].copy_from_slice(&1_u64.to_le_bytes());
    assert!(HiluPreconditioner::<f64>::from_checkpoint_bytes(&bad_dimension).is_err());
}
