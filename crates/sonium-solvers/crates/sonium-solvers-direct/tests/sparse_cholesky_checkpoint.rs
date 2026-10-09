use num_complex::{Complex32, Complex64};
use sonium_solvers_direct::{CholeskyOrdering, SparseCholeskyError, SparseCholeskyFactorization};
use sonium_solvers_sparse::CsrMatrix;

fn real_fixture() -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 5.0),
            (1, 2, 0.75),
            (2, 1, 0.75),
            (2, 2, 3.0),
        ],
    )
}

fn hermitian_fixture() -> CsrMatrix<Complex64> {
    CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, Complex64::new(4.0, 0.0)),
            (0, 1, Complex64::new(1.0, 0.5)),
            (1, 0, Complex64::new(1.0, -0.5)),
            (1, 1, Complex64::new(3.0, 0.0)),
        ],
    )
}

fn assert_invalid<T: sonium_solvers_core::traits::ComplexField>(bytes: Vec<u8>) {
    assert!(matches!(
        SparseCholeskyFactorization::<T>::from_checkpoint_bytes(&bytes),
        Err(SparseCholeskyError::InvalidCheckpoint(_))
    ));
}

fn vector_value_offset(bytes: &[u8], vector_index: usize, element_index: usize) -> usize {
    let mut offset = 22; // magic + version + scalar flags + dimension
    for index in 0..=vector_index {
        let count = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        if index == vector_index {
            assert!(element_index < count);
            return offset + element_index * 8;
        }
        offset += count * 8;
    }
    unreachable!()
}

#[test]
fn real_simplicial_checkpoint_round_trip_preserves_solves() {
    let factor = SparseCholeskyFactorization::factorize_with_ordering(
        &real_fixture(),
        CholeskyOrdering::Natural,
        false,
    )
    .unwrap();
    let bytes = factor.to_checkpoint_bytes().unwrap();
    let restored = SparseCholeskyFactorization::<f64>::from_checkpoint_bytes(&bytes).unwrap();
    let rhs = [1.0, -2.0, 0.5];
    assert_eq!(restored.dimension(), factor.dimension());
    assert_eq!(restored.factor_nonzeros(), factor.factor_nonzeros());
    assert_eq!(restored.is_hermitian(), factor.is_hermitian());
    let expected = factor.solve(&rhs).unwrap();
    let actual = restored.solve(&rhs).unwrap();
    for (left, right) in expected.iter().zip(actual.iter()) {
        assert!((left - right).abs() < 1.0e-12);
    }
}

#[test]
fn real_supernodal_checkpoint_round_trip_preserves_solves() {
    let factor = SparseCholeskyFactorization::factorize_supernodal(&real_fixture()).unwrap();
    let restored = SparseCholeskyFactorization::<f64>::from_checkpoint_bytes(
        &factor.to_checkpoint_bytes().unwrap(),
    )
    .unwrap();
    let rhs = [0.25, 1.0, -0.75];
    let expected = factor.solve(&rhs).unwrap();
    let actual = restored.solve(&rhs).unwrap();
    assert!(
        expected
            .iter()
            .zip(actual.iter())
            .all(|(left, right)| (left - right).abs() < 1.0e-12)
    );
}

#[test]
fn real_checkpoint_round_trip_preserves_explicit_orderings() {
    let matrix = real_fixture();
    let expected = [1.0, -2.0, 0.5];
    let rhs = matrix.matvec(&expected);

    for ordering in [
        CholeskyOrdering::Natural,
        CholeskyOrdering::Colamd,
        CholeskyOrdering::NestedDissection,
    ] {
        let factor = SparseCholeskyFactorization::factorize_with_ordering(&matrix, ordering, false)
            .unwrap_or_else(|error| panic!("{ordering:?} factorization failed: {error}"));
        let bytes = factor
            .to_checkpoint_bytes()
            .unwrap_or_else(|error| panic!("{ordering:?} checkpoint failed: {error}"));
        let restored = SparseCholeskyFactorization::<f64>::from_checkpoint_bytes(&bytes)
            .unwrap_or_else(|error| panic!("{ordering:?} restore failed: {error}"));
        let actual = restored
            .solve(&rhs)
            .unwrap_or_else(|error| panic!("{ordering:?} restored solve failed: {error}"));
        let error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(error < 1.0e-12, "{ordering:?} solution error {error:e}");
    }
}

#[test]
fn complex_hermitian_checkpoint_round_trip_preserves_solves() {
    let factor = SparseCholeskyFactorization::factorize_hermitian(&hermitian_fixture()).unwrap();
    let restored = SparseCholeskyFactorization::<Complex64>::from_checkpoint_bytes(
        &factor.to_checkpoint_bytes().unwrap(),
    )
    .unwrap();
    let rhs = [Complex64::new(1.0, 0.25), Complex64::new(-0.5, 1.0)];
    let expected = factor.solve(&rhs).unwrap();
    let actual = restored.solve(&rhs).unwrap();
    assert!(
        expected
            .iter()
            .zip(actual.iter())
            .all(|(left, right)| (*left - *right).norm() < 1.0e-12)
    );
    assert!(restored.is_hermitian());
}

#[test]
fn zero_dimension_checkpoint_round_trip_is_supported() {
    let matrix = CsrMatrix::<f64>::from_triplets(0, 0, Vec::new());
    let factor = SparseCholeskyFactorization::factorize_with_ordering(
        &matrix,
        CholeskyOrdering::Natural,
        false,
    )
    .unwrap();
    let restored = SparseCholeskyFactorization::<f64>::from_checkpoint_bytes(
        &factor.to_checkpoint_bytes().unwrap(),
    )
    .unwrap();
    assert_eq!(restored.dimension(), 0);
    assert_eq!(restored.solve(&[]).unwrap(), Vec::<f64>::new());
}

#[test]
fn narrow_real_and_complex_checkpoint_scalar_families_round_trip() {
    let real = CsrMatrix::from_triplets(
        2,
        2,
        vec![(0, 0, 2.0_f32), (0, 1, 0.25), (1, 0, 0.25), (1, 1, 3.0)],
    );
    let real_factor = SparseCholeskyFactorization::factorize_with_ordering(
        &real,
        CholeskyOrdering::Natural,
        false,
    )
    .unwrap();
    let real_restored = SparseCholeskyFactorization::<f32>::from_checkpoint_bytes(
        &real_factor.to_checkpoint_bytes().unwrap(),
    )
    .unwrap();
    let real_rhs = [1.0_f32, -0.5];
    let real_expected = real_factor.solve(&real_rhs).unwrap();
    let real_actual = real_restored.solve(&real_rhs).unwrap();
    assert!(
        real_expected
            .iter()
            .zip(real_actual.iter())
            .all(|(left, right)| (left - right).abs() < 1.0e-5)
    );

    let complex = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, Complex32::new(2.0, 0.0)),
            (0, 1, Complex32::new(0.25, 0.125)),
            (1, 0, Complex32::new(0.25, -0.125)),
            (1, 1, Complex32::new(3.0, 0.0)),
        ],
    );
    let complex_factor = SparseCholeskyFactorization::factorize_hermitian(&complex).unwrap();
    let complex_restored = SparseCholeskyFactorization::<Complex32>::from_checkpoint_bytes(
        &complex_factor.to_checkpoint_bytes().unwrap(),
    )
    .unwrap();
    let complex_rhs = [Complex32::new(1.0, 0.25), Complex32::new(-0.5, 0.5)];
    let complex_expected = complex_factor.solve(&complex_rhs).unwrap();
    let complex_actual = complex_restored.solve(&complex_rhs).unwrap();
    assert!(
        complex_expected
            .iter()
            .zip(complex_actual.iter())
            .all(|(left, right)| (*left - *right).norm() < 1.0e-5)
    );
}

#[test]
fn checkpoint_rejects_header_shape_and_payload_corruption() {
    let factor = SparseCholeskyFactorization::factorize_with_ordering(
        &real_fixture(),
        CholeskyOrdering::Natural,
        false,
    )
    .unwrap();
    let bytes = factor.to_checkpoint_bytes().unwrap();

    let mut magic = bytes.clone();
    magic[0] ^= 1;
    assert_invalid::<f64>(magic);

    let mut version = bytes.clone();
    version[8] = 2;
    assert_invalid::<f64>(version);

    let mut reserved = bytes.clone();
    reserved[13] = 1;
    assert_invalid::<f64>(reserved);

    let mut trailing = bytes.clone();
    trailing.push(0);
    assert_invalid::<f64>(trailing);

    assert_invalid::<f64>(bytes[..bytes.len() - 1].to_vec());
    assert_invalid::<Complex64>(bytes.clone());

    let mut non_finite = bytes.clone();
    let non_finite_offset = non_finite.len() - 8;
    non_finite[non_finite_offset..].copy_from_slice(&f64::NAN.to_le_bytes());
    assert_invalid::<f64>(non_finite);
}

#[test]
fn checkpoint_rejects_invalid_permutations_pointers_and_rows() {
    let factor = SparseCholeskyFactorization::factorize_with_ordering(
        &real_fixture(),
        CholeskyOrdering::Natural,
        false,
    )
    .unwrap();
    let bytes = factor.to_checkpoint_bytes().unwrap();

    let mut permutation = bytes.clone();
    let permutation_offset = vector_value_offset(&permutation, 0, 0);
    permutation[permutation_offset..permutation_offset + 8]
        .copy_from_slice(&(factor.dimension() as u64).to_le_bytes());
    assert_invalid::<f64>(permutation);

    let mut pointers = bytes.clone();
    let pointer_offset = vector_value_offset(&pointers, 3, 0);
    pointers[pointer_offset..pointer_offset + 8].copy_from_slice(&1u64.to_le_bytes());
    assert_invalid::<f64>(pointers);

    let mut rows = bytes;
    let row_offset = vector_value_offset(&rows, 4, 0);
    rows[row_offset..row_offset + 8].copy_from_slice(&(factor.dimension() as u64).to_le_bytes());
    assert_invalid::<f64>(rows);
}
