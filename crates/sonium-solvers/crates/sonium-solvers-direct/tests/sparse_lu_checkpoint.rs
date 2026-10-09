use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::ComplexField;
use sonium_solvers_direct::{
    SPARSE_LU_CHECKPOINT_FORMAT_VERSION, SparseLuBackend, SparseLuError, SparseLuFactorization,
};
use sonium_solvers_sparse::CsrMatrix;

fn matrix<T: ComplexField>() -> CsrMatrix<T> {
    CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (
                0,
                0,
                T::from_re_im(T::Real::from_f64(5.0).unwrap(), T::Real::zero()),
            ),
            (
                0,
                1,
                T::from_re_im(T::Real::from_f64(1.0).unwrap(), T::Real::zero()),
            ),
            (
                1,
                0,
                T::from_re_im(T::Real::from_f64(2.0).unwrap(), T::Real::zero()),
            ),
            (
                1,
                1,
                T::from_re_im(T::Real::from_f64(4.0).unwrap(), T::Real::zero()),
            ),
            (
                1,
                2,
                T::from_re_im(T::Real::from_f64(1.0).unwrap(), T::Real::zero()),
            ),
            (
                2,
                1,
                T::from_re_im(T::Real::from_f64(1.0).unwrap(), T::Real::zero()),
            ),
            (
                2,
                2,
                T::from_re_im(T::Real::from_f64(3.0).unwrap(), T::Real::zero()),
            ),
        ],
    )
}

fn rhs<T: ComplexField>() -> Vec<T> {
    vec![
        T::from_re_im(T::Real::from_f64(7.0).unwrap(), T::Real::zero()),
        T::from_re_im(T::Real::from_f64(3.0).unwrap(), T::Real::zero()),
        T::from_re_im(T::Real::from_f64(5.0).unwrap(), T::Real::zero()),
    ]
}

fn assert_round_trip<T: ComplexField>(backend: SparseLuBackend, tolerance: f64)
where
    T::Real: ToPrimitive,
{
    let matrix = matrix::<T>();
    let rhs = rhs::<T>();
    let factor = SparseLuFactorization::factorize_with_backend(&matrix, backend, T::Real::one())
        .expect("factorization");
    let before = factor.solve(&rhs).expect("original solve");
    let bytes = factor.to_checkpoint_bytes().expect("serialize factor");
    assert!(bytes.starts_with(b"SNLUCP01"));
    assert_eq!(
        u16::from_le_bytes([bytes[8], bytes[9]]),
        SPARSE_LU_CHECKPOINT_FORMAT_VERSION
    );
    let restored =
        SparseLuFactorization::<T>::from_checkpoint_bytes(&bytes).expect("restore factor");
    assert_eq!(restored.backend(), backend);
    assert_eq!(restored.dimension(), factor.dimension());
    assert_eq!(
        restored.structure_fingerprint(),
        factor.structure_fingerprint()
    );
    assert_eq!(restored.factor_nonzeros(), factor.factor_nonzeros());
    let after = restored.solve(&rhs).expect("restored solve");
    for (lhs, rhs) in before.iter().zip(after.iter()) {
        assert!((*lhs - *rhs).norm().to_f64().unwrap() < tolerance);
    }
}

#[test]
fn checkpoint_round_trip_preserves_simplicial_scalar_families() {
    assert_round_trip::<f32>(SparseLuBackend::Simplicial, 2.0e-5);
    assert_round_trip::<f64>(SparseLuBackend::Simplicial, 1.0e-12);
    assert_round_trip::<Complex32>(SparseLuBackend::Simplicial, 2.0e-5);
    assert_round_trip::<Complex64>(SparseLuBackend::Simplicial, 1.0e-12);
}

#[test]
fn checkpoint_round_trip_preserves_supernodal_factor() {
    assert_round_trip::<f64>(SparseLuBackend::Supernodal, 1.0e-12);
    assert_round_trip::<Complex64>(SparseLuBackend::Supernodal, 1.0e-12);
}

#[test]
fn preprocessed_supernodal_checkpoint_preserves_scaled_solve() {
    let matrix = CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 1.0e-4),
            (0, 1, 2.0e-4),
            (1, 0, 3.0e3),
            (1, 1, 4.0e3),
            (1, 2, 1.0e3),
            (2, 1, 5.0e-2),
            (2, 2, 2.0e-2),
        ],
    );
    let factor = SparseLuFactorization::factorize_supernodal_with_preprocessing(&matrix, 1.0, true)
        .expect("preprocessed factorization");
    let rhs = [1.0, 2.0, 3.0];
    let expected = factor.solve(&rhs).expect("original preprocessed solve");
    let bytes = factor
        .to_checkpoint_bytes()
        .expect("serialize preprocessed factor");
    let restored = SparseLuFactorization::<f64>::from_checkpoint_bytes(&bytes)
        .expect("restore preprocessed factor");
    let actual = restored.solve(&rhs).expect("restored preprocessed solve");
    for (lhs, rhs) in expected.iter().zip(actual.iter()) {
        assert!((lhs - rhs).abs() < 1.0e-10);
    }
}

#[test]
fn checkpoint_rejects_truncation_magic_and_trailing_bytes() {
    let factor = SparseLuFactorization::factorize(&matrix::<f64>()).expect("factorization");
    let bytes = factor.to_checkpoint_bytes().expect("serialize factor");

    let truncated = &bytes[..bytes.len() - 1];
    assert!(matches!(
        SparseLuFactorization::<f64>::from_checkpoint_bytes(truncated),
        Err(SparseLuError::InvalidCheckpoint(_))
    ));

    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 0xff;
    assert!(matches!(
        SparseLuFactorization::<f64>::from_checkpoint_bytes(&bad_magic),
        Err(SparseLuError::InvalidCheckpoint(_))
    ));

    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(matches!(
        SparseLuFactorization::<f64>::from_checkpoint_bytes(&trailing),
        Err(SparseLuError::InvalidCheckpoint(_))
    ));

    assert!(matches!(
        SparseLuFactorization::<f32>::from_checkpoint_bytes(&bytes),
        Err(SparseLuError::InvalidCheckpoint(_))
    ));
}

/// 2D nonsymmetric convection-diffusion grid (the direct-solver perf
/// workload shape): exercises ordering, partial pivoting with row swaps,
/// and fill-in across a non-trivial elimination tree.
fn convection_diffusion_grid(grid: usize) -> CsrMatrix<f64> {
    let n = grid * grid;
    let mut triplets = Vec::with_capacity(5 * n);
    for row in 0..grid {
        for column in 0..grid {
            let index = row * grid + column;
            if row > 0 {
                triplets.push((index, index - grid, -1.1));
            }
            if column > 0 {
                triplets.push((index, index - 1, -1.2));
            }
            triplets.push((index, index, 4.01));
            if column + 1 < grid {
                triplets.push((index, index + 1, -0.8));
            }
            if row + 1 < grid {
                triplets.push((index, index + grid, -0.9));
            }
        }
    }
    CsrMatrix::from_triplets(n, n, triplets)
}

#[test]
fn simplicial_factorize_is_deterministic_byte_for_byte() {
    // The simplicial numeric kernel must be a pure function of its input:
    // same matrix, same factor bytes (ordering, pivots, L/U values and
    // storage order, schedules). Pins the fused touch-marking, batched
    // finite checks, and first-occurrence L-column store against any
    // accidental order/nondeterminism regression.
    for matrix in [
        convection_diffusion_grid(12),
        CsrMatrix::from_triplets(
            4,
            4,
            vec![
                (0, 1, 2.0),
                (0, 3, 1.0),
                (1, 0, -2.0),
                (1, 1, 1.0),
                (1, 2, 1.0),
                (2, 1, 1.0),
                (2, 2, 3.0),
                (2, 3, -2.0),
                (3, 0, 1.0),
                (3, 3, 2.0),
            ],
        ),
    ] {
        let first = SparseLuFactorization::factorize(&matrix).expect("factorization");
        let second = SparseLuFactorization::factorize(&matrix).expect("factorization");
        assert_eq!(first.factor_nonzeros(), second.factor_nonzeros());
        assert_eq!(
            first.to_checkpoint_bytes().expect("serialize"),
            second.to_checkpoint_bytes().expect("serialize"),
        );
    }
}

#[test]
fn empty_factor_checkpoint_round_trips_without_schedule_entries() {
    let matrix = CsrMatrix::<f64>::from_raw_parts(0, 0, vec![0], Vec::new(), Vec::new());
    let factor = SparseLuFactorization::factorize(&matrix).expect("empty factorization");
    let bytes = factor
        .to_checkpoint_bytes()
        .expect("serialize empty factor");
    let restored =
        SparseLuFactorization::<f64>::from_checkpoint_bytes(&bytes).expect("restore empty factor");
    assert_eq!(restored.dimension(), 0);
    assert!(restored.solve(&[]).expect("empty solve").is_empty());
}
