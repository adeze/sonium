use sonium_fem::assembly::{PrebakedAssembler, PrebakedAssemblyError, TripletMatrix};

fn diagonal(size: usize, value: f64) -> TripletMatrix<f64> {
    let mut matrix = TripletMatrix::new(size, size);
    for index in 0..size {
        matrix.push(index, index, value);
    }
    matrix
}

#[test]
fn checked_prebaked_constructor_accepts_valid_components() {
    let assembler =
        PrebakedAssembler::<2>::try_from_triplets([diagonal(2, 2.0), diagonal(2, 3.0)], &[])
            .expect("valid square component matrices");

    assert_eq!(assembler.num_rows, 2);
    assert_eq!(assembler.nnz(), 2);
}

#[test]
fn checked_prebaked_constructor_rejects_non_square_and_mismatched_components() {
    let nonsquare = TripletMatrix::new(2, 3);
    let error =
        PrebakedAssembler::<2>::try_from_triplets([nonsquare, TripletMatrix::new(2, 2)], &[])
            .err()
            .expect("non-square matrices must be rejected");
    assert!(matches!(error, PrebakedAssemblyError::NonSquare { .. }));

    let error =
        PrebakedAssembler::<2>::try_from_triplets([diagonal(2, 1.0), diagonal(3, 1.0)], &[])
            .err()
            .expect("component dimensions must agree");
    assert!(matches!(
        error,
        PrebakedAssemblyError::DimensionMismatch { .. }
    ));
}

#[test]
fn checked_prebaked_constructor_rejects_bad_triplets_before_merge() {
    let mut nonfinite = diagonal(2, 1.0);
    nonfinite.push(0, 1, f64::NAN);
    let error = PrebakedAssembler::<2>::try_from_triplets([nonfinite, diagonal(2, 1.0)], &[])
        .err()
        .expect("non-finite values must be rejected");
    assert!(matches!(
        error,
        PrebakedAssemblyError::NonFiniteValue { .. }
    ));

    let mut out_of_bounds = TripletMatrix::new(2, 2);
    out_of_bounds.push(2, 0, 1.0);
    let error = PrebakedAssembler::<2>::try_from_triplets([out_of_bounds, diagonal(2, 1.0)], &[])
        .err()
        .expect("out-of-bounds triplets must be rejected");
    assert!(matches!(
        error,
        PrebakedAssemblyError::IndexOutOfBounds { .. }
    ));
}

#[test]
fn checked_prebaked_constructor_rejects_duplicate_boundary_tags_and_zero_components() {
    let boundary = diagonal(2, 1.0);
    let error = PrebakedAssembler::<2>::try_from_triplets(
        [diagonal(2, 1.0), diagonal(2, 1.0)],
        &[(7, boundary.clone()), (7, boundary)],
    )
    .err()
    .expect("boundary tags must be unique");
    assert!(matches!(
        error,
        PrebakedAssemblyError::DuplicateBoundaryTag { tag: 7 }
    ));

    let error = PrebakedAssembler::<0>::try_from_triplets([], &[])
        .err()
        .expect("zero-component assemblers have no shared dimension");
    assert!(matches!(
        error,
        PrebakedAssemblyError::DimensionMismatch { .. }
    ));
}
