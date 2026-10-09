//! Thread-count determinism pins for sparse LU (ParNum Phase 4).
//!
//! The sparse numeric factorization is serial; thread-count-sensitive work
//! lives in the level-set triangular solve (gated by `n >= 1000` and
//! `SONIUM_SPARSE_PAR_MIN_NNZ`) and inside `ComplexField::mat_mul_nn`.
//! These tests pin exact checkpoint bytes, pivot sequences, and error kinds
//! so that running this suite at `RAYON_NUM_THREADS=1` and `=16` (both must
//! be green) proves bit-identical factorization across thread counts: any
//! thread-dependent bit would miss the golden under one of the counts.
//!
//! Goldens were captured from a scratch probe whose `RAYON=1` vs `RAYON=16`
//! outputs were verified byte-identical (`$HOME/tmp/parnum/raw/probe_*.log`).
//! Supernodal value-bit goldens re-pinned 2026-09-28 for the merged
//! BLAS/LAPACK kernels (last-ulp reassociation; simplicial goldens,
//! lengths, pivot sequences, and residual gates all unchanged).
//! Complex64 supernodal value bits re-pinned 2026-10-06 for the native
//! complex GEMM drivers (`opt/compute-x86` checkpoint): bytes FNV only;
//! length, pivot sequence, repeat determinism, and the 1e-10 residual
//! gate (observed 7.14e-16) all hold, and RAYON=1 vs =16 outputs are
//! byte-identical, so the drift is deterministic last-ulp reassociation.

use num_complex::Complex64;
use sonium_solvers_direct::{SparseLuBackend, SparseLuError, SparseLuFactorization};
use sonium_solvers_sparse::CsrMatrix;

fn fnv_bytes(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn fnv_usize(values: &[usize]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &value in values {
        for byte in (value as u64).to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Decode `row_perm` (the pivot sequence) from checkpoint bytes.
///
/// Layout (`sparse_lu.rs` checkpoint writer): 8 magic + u16 version +
/// u8 backend + u8 flags + u8 scalar + u8 real, then u64 n, u64
/// supernode count, pattern row-ptrs/col-indices vecs, L col-ptrs /
/// row-indices / values, U col-ptrs / row-indices / values, row_perm,
/// col_perm. A usize-vec is u64 len + u64 entries; a values-vec is u64
/// len + 16 bytes per entry (re/im as f64).
fn decode_row_perm(bytes: &[u8]) -> Vec<usize> {
    fn read_u64(bytes: &[u8], offset: &mut usize) -> u64 {
        let value = u64::from_le_bytes(bytes[*offset..*offset + 8].try_into().unwrap());
        *offset += 8;
        value
    }
    fn read_usize_vec(bytes: &[u8], offset: &mut usize) -> Vec<usize> {
        let len = read_u64(bytes, offset) as usize;
        (0..len).map(|_| read_u64(bytes, offset) as usize).collect()
    }
    let mut offset = 14usize; // magic + version + 4 tag bytes
    let _n = read_u64(bytes, &mut offset);
    let _supernode_count = read_u64(bytes, &mut offset);
    let _pattern_row_ptrs = read_usize_vec(bytes, &mut offset);
    let _pattern_col_indices = read_usize_vec(bytes, &mut offset);
    let _l_col_ptrs = read_usize_vec(bytes, &mut offset);
    let _l_row_indices = read_usize_vec(bytes, &mut offset);
    offset += read_u64(bytes, &mut offset) as usize * 16;
    let _u_col_ptrs = read_usize_vec(bytes, &mut offset);
    let _u_row_indices = read_usize_vec(bytes, &mut offset);
    offset += read_u64(bytes, &mut offset) as usize * 16;
    let row_perm = read_usize_vec(bytes, &mut offset);
    let _col_perm = read_usize_vec(bytes, &mut offset);
    assert_eq!(
        offset,
        bytes.len(),
        "row_perm decoder must consume the whole checkpoint"
    );
    row_perm
}

fn convection_diffusion(grid: usize) -> CsrMatrix<f64> {
    let n = grid * grid;
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut column_indices = Vec::with_capacity(5 * n);
    let mut values = Vec::with_capacity(5 * n);
    row_ptrs.push(0);
    for row in 0..grid {
        for column in 0..grid {
            let index = row * grid + column;
            if row > 0 {
                column_indices.push(index - grid);
                values.push(-1.1);
            }
            if column > 0 {
                column_indices.push(index - 1);
                values.push(-1.2);
            }
            column_indices.push(index);
            values.push(4.01);
            if column + 1 < grid {
                column_indices.push(index + 1);
                values.push(-0.8);
            }
            if row + 1 < grid {
                column_indices.push(index + grid);
                values.push(-0.9);
            }
            row_ptrs.push(column_indices.len());
        }
    }
    CsrMatrix::from_raw_parts(n, n, row_ptrs, column_indices, values)
}

/// Exact-tie adversarial matrix: every nonzero has magnitude 1, so every
/// pivot search faces exact ties and the pivot sequence is purely a
/// function of each backend's deterministic first-max-wins scan order.
/// (All-ones tridiagonals are singular iff `n % 3 == 2`; 4 and 7 are safe.)
fn exact_tie_tridiagonal(n: usize) -> CsrMatrix<f64> {
    assert!(n % 3 != 2, "all-ones tridiagonal is singular");
    let mut triplets = Vec::with_capacity(3 * n - 2);
    for row in 0..n {
        triplets.push((row, row, 1.0));
        if row > 0 {
            triplets.push((row, row - 1, 1.0));
        }
        if row + 1 < n {
            triplets.push((row, row + 1, 1.0));
        }
    }
    CsrMatrix::from_triplets(n, n, triplets)
}

fn relative_residual(matrix: &CsrMatrix<f64>, x: &[f64], rhs: &[f64]) -> f64 {
    let actual = matrix.matvec(x);
    let error = actual
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    let scale = rhs.iter().map(|value| value.abs()).fold(0.0, f64::max);
    error / scale.max(f64::MIN_POSITIVE)
}

fn backend_tag(backend: SparseLuBackend) -> &'static str {
    match backend {
        SparseLuBackend::Simplicial => "simplicial",
        SparseLuBackend::Supernodal => "supernodal",
    }
}

/// (name, matrix, backend, checkpoint len, checkpoint FNV, row_perm FNV).
type CorpusEntry = (
    &'static str,
    CsrMatrix<f64>,
    SparseLuBackend,
    usize,
    u64,
    u64,
);

fn determinism_corpus() -> Vec<CorpusEntry> {
    let mut corpus = Vec::new();
    for (name, matrix) in [
        ("conv8", convection_diffusion(8)),
        ("conv12", convection_diffusion(12)),
        ("tie4", exact_tie_tridiagonal(4)),
        ("tie7", exact_tie_tridiagonal(7)),
        // n=1024: large enough to engage the level-set parallel solve.
        ("conv32", convection_diffusion(32)),
    ] {
        for backend in [SparseLuBackend::Simplicial, SparseLuBackend::Supernodal] {
            let (len, fnv, row_perm_fnv) = match (name, backend) {
                ("conv8", SparseLuBackend::Simplicial) => {
                    (23718, 0x9d13560d22c2b304, 0x4f3c6c92c91581e5)
                }
                ("conv8", SparseLuBackend::Supernodal) => {
                    (23718, 0xd8f77adcc0b86342, 0x4f3c6c92c91581e5)
                }
                ("conv12", SparseLuBackend::Simplicial) => {
                    (70166, 0x20d1f349aff546bc, 0xc19ee2b2303c32e5)
                }
                ("conv12", SparseLuBackend::Supernodal) => {
                    (70166, 0x40d1f9dbef45dd33, 0xc19ee2b2303c32e5)
                }
                ("tie4", SparseLuBackend::Simplicial) => {
                    (614, 0x53ed93c69d26edd4, 0xd1910600cb1e4225)
                }
                ("tie4", SparseLuBackend::Supernodal) => {
                    (614, 0x0d88d087b2408c24, 0xd1910600cb1e4225)
                }
                ("tie7", SparseLuBackend::Simplicial) => {
                    (1070, 0x4140ed04861ba219, 0x9683ef06dedad262)
                }
                ("tie7", SparseLuBackend::Supernodal) => {
                    (1070, 0x35cc81c2ad6f7529, 0x9683ef06dedad262)
                }
                ("conv32", SparseLuBackend::Simplicial) => {
                    (992166, 0x950e809525fd4720, 0x29aeb9a27b84ef19)
                }
                ("conv32", SparseLuBackend::Supernodal) => {
                    (992166, 0xec4fa36bd4ab3adf, 0x29aeb9a27b84ef19)
                }
                _ => unreachable!(),
            };
            corpus.push((name, matrix.clone(), backend, len, fnv, row_perm_fnv));
        }
    }
    corpus
}

#[test]
fn checkpoint_bytes_deterministic_with_golden_pins() {
    // Same-backend repeat factorization must be byte-identical (checkpoint
    // bytes embed L/U structure + values + row_perm + col_perm), the
    // decoded pivot sequence must match, and both must hit the goldens
    // (which also pins cross-thread-count equality once this suite runs
    // green at RAYON_NUM_THREADS=1 and =16).
    for (name, matrix, backend, len, fnv, row_perm_fnv) in determinism_corpus() {
        let tag = backend_tag(backend);
        let first = SparseLuFactorization::factorize_with_backend(&matrix, backend, 1.0)
            .unwrap_or_else(|error| panic!("{name}/{tag} first factorization: {error:?}"));
        let second = SparseLuFactorization::factorize_with_backend(&matrix, backend, 1.0)
            .unwrap_or_else(|error| panic!("{name}/{tag} second factorization: {error:?}"));
        let first_bytes = first.to_checkpoint_bytes().expect("serialize first");
        let second_bytes = second.to_checkpoint_bytes().expect("serialize second");
        assert_eq!(
            first_bytes, second_bytes,
            "{name}/{tag}: repeat factorization changed checkpoint bytes"
        );
        assert_eq!(first_bytes.len(), len, "{name}/{tag}: checkpoint length");
        assert_eq!(
            fnv_bytes(&first_bytes),
            fnv,
            "{name}/{tag}: checkpoint bytes golden"
        );
        let row_perm = decode_row_perm(&first_bytes);
        assert_eq!(
            row_perm.len(),
            matrix.num_rows,
            "{name}/{tag}: row_perm length"
        );
        assert_eq!(
            fnv_usize(&row_perm),
            row_perm_fnv,
            "{name}/{tag}: pivot-sequence golden"
        );
        // Both backends must land in the same residual class (their bytes
        // legitimately differ: supernodal GEMM summation order vs scalar).
        let n = matrix.num_rows;
        let expected: Vec<f64> = (0..n).map(|i| (i as f64 * 0.013).sin()).collect();
        let rhs = matrix.matvec(&expected);
        let solution = first.solve(&rhs).expect("solve");
        let residual = relative_residual(&matrix, &solution, &rhs);
        assert!(
            residual <= 1.0e-10,
            "{name}/{tag}: residual {residual:.2e} exceeds 1e-10"
        );
    }
}

#[test]
fn symbolic_reuse_matches_oneshot_bytes() {
    // Reused-symbolic numeric with the one-shot pivot default must produce
    // byte-identical checkpoints (same ordering, same arithmetic).
    for (name, matrix, golden_fnv) in [
        ("conv8", convection_diffusion(8), 0xd8f77adcc0b86342_u64),
        ("conv12", convection_diffusion(12), 0x40d1f9dbef45dd33_u64),
        ("tie7", exact_tie_tridiagonal(7), 0x35cc81c2ad6f7529_u64),
    ] {
        let symbolic = SparseLuFactorization::analyze_supernodal(&matrix).expect("analyze");
        let reused =
            SparseLuFactorization::factorize_supernodal_with_symbolic(&matrix, &symbolic, 1.0)
                .expect("reused-symbolic numeric");
        let oneshot = SparseLuFactorization::factorize_supernodal(&matrix).expect("one-shot");
        let reused_bytes = reused.to_checkpoint_bytes().expect("serialize reused");
        let oneshot_bytes = oneshot.to_checkpoint_bytes().expect("serialize one-shot");
        assert_eq!(
            reused_bytes, oneshot_bytes,
            "{name}: symbolic-reuse diverged from one-shot bytes"
        );
        assert_eq!(
            fnv_bytes(&reused_bytes),
            golden_fnv,
            "{name}: reused-symbolic golden"
        );

        // Same pattern, scaled values: must validate, stay deterministic
        // across repeats, and actually differ (non-vacuous check).
        let mut scaled = matrix.clone();
        for value in scaled.values.iter_mut() {
            *value *= 1.5;
        }
        let first =
            SparseLuFactorization::factorize_supernodal_with_symbolic(&scaled, &symbolic, 1.0)
                .expect("scaled numeric");
        let second =
            SparseLuFactorization::factorize_supernodal_with_symbolic(&scaled, &symbolic, 1.0)
                .expect("scaled numeric repeat");
        let first_bytes = first.to_checkpoint_bytes().expect("serialize scaled");
        assert_eq!(
            first_bytes,
            second
                .to_checkpoint_bytes()
                .expect("serialize scaled repeat"),
            "{name}: scaled-values numeric not deterministic"
        );
        assert_ne!(
            first_bytes, reused_bytes,
            "{name}: scaled-values numeric unexpectedly identical to base"
        );
    }
}

/// Singular / ill-conditioned / non-finite fixtures.
fn error_corpus() -> Vec<(&'static str, CsrMatrix<f64>)> {
    vec![
        (
            // Structurally empty row 2: exact-zero pivot under any ordering.
            "zero-pivot",
            CsrMatrix::from_raw_parts(
                4,
                4,
                vec![0, 1, 2, 2, 3],
                vec![0, 1, 3],
                vec![1.0, 1.0, 1.0],
            ),
        ),
        (
            // Duplicate rows 0/1: rank-deficient.
            "dup-rows",
            CsrMatrix::from_raw_parts(
                3,
                3,
                vec![0, 2, 4, 6],
                vec![0, 1, 0, 1, 1, 2],
                vec![1.0, 2.0, 1.0, 2.0, 1.0, 1.0],
            ),
        ),
        (
            // 1e-16 pivot vs the eps*n*||A|| direct-solver guard (~9e-16):
            // strict paths must refuse, preconditioner mode must repair.
            "illcond-1e-16",
            CsrMatrix::from_raw_parts(
                4,
                4,
                vec![0, 1, 2, 3, 4],
                vec![0, 1, 2, 3],
                vec![1.0, 1.0, 1e-16, 1.0],
            ),
        ),
        (
            // Struct literal: from_raw_parts rejects non-finite values at
            // admission, so the NonFiniteMatrix path needs a direct build.
            "nonfinite",
            CsrMatrix {
                num_rows: 2,
                num_cols: 2,
                row_ptrs: vec![0, 2, 3],
                col_indices: vec![0, 1, 1],
                values: vec![1.0, f64::INFINITY, 1.0],
            },
        ),
    ]
}

fn assert_same_error(left: SparseLuError, right: SparseLuError, context: &str) {
    assert_eq!(
        format!("{left:?}"),
        format!("{right:?}"),
        "{context}: error mismatch"
    );
}

#[test]
fn singular_and_illconditioned_error_parity() {
    // All numeric paths (simplicial, supernodal one-shot, reused-symbolic)
    // must report the same error kind AND payload (column / row+col).
    for (name, matrix) in error_corpus() {
        let simplicial = SparseLuFactorization::factorize(&matrix);
        let supernodal = SparseLuFactorization::factorize_supernodal(&matrix);
        let symbolic_reuse =
            SparseLuFactorization::analyze_supernodal(&matrix).and_then(|symbolic| {
                SparseLuFactorization::factorize_supernodal_with_symbolic(&matrix, &symbolic, 1.0)
            });
        match (simplicial, supernodal, symbolic_reuse) {
            (Err(simp), Err(sup), Err(reused)) => {
                assert_same_error(simp, sup, &format!("{name} simplicial vs supernodal"));
                let simplicial = SparseLuFactorization::factorize(&matrix).unwrap_err();
                assert_same_error(
                    simplicial,
                    reused,
                    &format!("{name} simplicial vs symbolic-reuse"),
                );
            }
            (simp, sup, reused) => panic!(
                "{name}: expected all paths to fail, got \
                 simplicial={simp:?}, supernodal={sup:?}, reused={reused:?}"
            ),
        }
    }

    // Exact payloads (elimination-step columns / admission coordinates).
    let corpus = error_corpus();
    let payload = |name: &str| {
        let matrix = corpus
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .unwrap()
            .1
            .clone();
        format!(
            "{:?}",
            SparseLuFactorization::factorize(&matrix).unwrap_err()
        )
    };
    assert_eq!(payload("zero-pivot"), "SingularMatrix(1)");
    assert_eq!(payload("dup-rows"), "SingularMatrix(2)");
    assert_eq!(payload("illcond-1e-16"), "SingularMatrix(1)");
    assert_eq!(payload("nonfinite"), "NonFiniteMatrix { row: 0, col: 1 }");
}

#[test]
fn preconditioner_mode_spot_coverage() {
    let corpus = error_corpus();
    let matrix = |name: &str| {
        corpus
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .unwrap()
            .1
            .clone()
    };

    // Exact-zero pivots are still rejected in preconditioner mode.
    for name in ["zero-pivot", "dup-rows"] {
        let error = SparseLuFactorization::factorize_for_preconditioner(&matrix(name)).unwrap_err();
        assert!(
            matches!(error, SparseLuError::SingularMatrix(_)),
            "{name}: preconditioner mode must still reject exact-zero pivots, got {error:?}"
        );
    }

    // The 1e-16 pivot is repaired (clamped to the eps^2*||A|| floor) and
    // the factor solves the diagonal system to full expected accuracy.
    let ill = matrix("illcond-1e-16");
    let factor = SparseLuFactorization::factorize_for_preconditioner(&ill).expect("precond factor");
    let rhs = vec![1.0, 2.0, 3.0, 4.0];
    let solution = factor.solve(&rhs).expect("precond solve");
    let expected = vec![1.0, 2.0, 3.0e16, 4.0];
    for (index, (&got, &want)) in solution.iter().zip(&expected).enumerate() {
        let relative = ((got - want) / want).abs();
        assert!(
            relative < 1.0e-12,
            "precond solve[{index}]: {got:.6e} vs {want:.6e}"
        );
    }
    // Checkpoint round-trip preserves the preconditioner-mode factor
    // (flags bit) and its solve bits.
    let bytes = factor.to_checkpoint_bytes().expect("serialize precond");
    assert_eq!(bytes.len(), 422, "preconditioner checkpoint length");
    let restored =
        SparseLuFactorization::<f64>::from_checkpoint_bytes(&bytes).expect("restore precond");
    assert_eq!(
        restored.to_checkpoint_bytes().expect("re-serialize"),
        bytes,
        "preconditioner checkpoint round-trip changed bytes"
    );
    let restored_solution = restored.solve(&rhs).expect("restored precond solve");
    assert_eq!(
        solve_bits(&solution),
        solve_bits(&restored_solution),
        "preconditioner round-trip changed solve bits"
    );
}

/// Solve with the wave-dispatch gate forced to both extremes, restoring the
/// ambient value (same discipline as `sparse_lu_parallel.rs`: the gate only
/// routes between bitwise-identical paths).
fn solve_at_gate_extremes(
    factor: &SparseLuFactorization<f64>,
    rhs: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    const GATE: &str = "SONIUM_SPARSE_PAR_MIN_NNZ";
    // SAFETY: any value a concurrent test thread observes keeps its solve
    // correct (identical paths); restored below.
    let saved = std::env::var(GATE).ok();
    let guarded = |value: &str| {
        unsafe { std::env::set_var(GATE, value) };
        factor.solve(rhs).expect("gated solve")
    };
    let dispatched = guarded("0");
    let serial = guarded(&usize::MAX.to_string());
    match saved {
        Some(value) => unsafe { std::env::set_var(GATE, value) },
        None => unsafe { std::env::remove_var(GATE) },
    }
    (dispatched, serial)
}

fn solve_bits(solution: &[f64]) -> Vec<u64> {
    solution.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn checkpoint_roundtrip_through_parallel_solve_path() {
    // n=1024 engages the level-set parallel solve (n >= 1000 gate):
    // restore-from-checkpoint must preserve bytes, and the restored
    // factor's parallel (dispatched) and serial (inline) solves must be
    // bit-identical to the original factor's solve.
    let matrix = convection_diffusion(32);
    assert!(matrix.num_rows >= 1000);
    let n = matrix.num_rows;
    let expected: Vec<f64> = (0..n).map(|i| (i as f64 * 0.013).sin()).collect();
    let rhs = matrix.matvec(&expected);
    for backend in [SparseLuBackend::Simplicial, SparseLuBackend::Supernodal] {
        let tag = backend_tag(backend);
        let factor = SparseLuFactorization::factorize_with_backend(&matrix, backend, 1.0)
            .unwrap_or_else(|error| panic!("{tag} factorization: {error:?}"));
        let bytes = factor.to_checkpoint_bytes().expect("serialize");
        let restored =
            SparseLuFactorization::<f64>::from_checkpoint_bytes(&bytes).expect("restore");
        assert_eq!(
            restored.to_checkpoint_bytes().expect("re-serialize"),
            bytes,
            "{tag}: checkpoint round-trip changed bytes"
        );
        let original = solve_bits(&factor.solve(&rhs).expect("original solve"));
        let (dispatched, serial) = solve_at_gate_extremes(&restored, &rhs);
        assert_eq!(
            solve_bits(&dispatched),
            original,
            "{tag}: restored parallel-solve bits differ from original"
        );
        assert_eq!(
            solve_bits(&serial),
            original,
            "{tag}: restored serial-solve bits differ from original"
        );
    }
}

#[test]
fn scalar_family_spot_f32_and_complex64() {
    // f32 determinism + round-trip + goldens on the conv8 corpus.
    let base = convection_diffusion(8);
    let matrix_f32 = CsrMatrix::from_raw_parts(
        base.num_rows,
        base.num_cols,
        base.row_ptrs.clone(),
        base.col_indices.clone(),
        base.values.iter().map(|&value| value as f32).collect(),
    );
    for (backend, golden_fnv) in [
        (SparseLuBackend::Simplicial, 0xa2279a3402eedc30_u64),
        (SparseLuBackend::Supernodal, 0x8cb603ec36914cee_u64),
    ] {
        let tag = backend_tag(backend);
        let first = SparseLuFactorization::factorize_with_backend(&matrix_f32, backend, 1.0f32)
            .unwrap_or_else(|error| panic!("f32/{tag}: {error:?}"));
        let second = SparseLuFactorization::factorize_with_backend(&matrix_f32, backend, 1.0f32)
            .unwrap_or_else(|error| panic!("f32/{tag} repeat: {error:?}"));
        let bytes = first.to_checkpoint_bytes().expect("serialize f32");
        assert_eq!(
            bytes,
            second.to_checkpoint_bytes().expect("serialize f32 repeat"),
            "f32/{tag}: repeat factorization changed bytes"
        );
        assert_eq!(bytes.len(), 23718, "f32/{tag}: checkpoint length");
        assert_eq!(fnv_bytes(&bytes), golden_fnv, "f32/{tag}: golden");
        let restored =
            SparseLuFactorization::<f32>::from_checkpoint_bytes(&bytes).expect("restore f32");
        assert_eq!(
            restored.to_checkpoint_bytes().expect("re-serialize f32"),
            bytes,
            "f32/{tag}: round-trip changed bytes"
        );
    }

    // Complex64 determinism + pivot-sequence + residual gate.
    let matrix_c64 = CsrMatrix::from_raw_parts(
        base.num_rows,
        base.num_cols,
        base.row_ptrs.clone(),
        base.col_indices.clone(),
        base.values
            .iter()
            .map(|&value| Complex64::new(value, 0.25 * value))
            .collect(),
    );
    for (backend, golden_fnv) in [
        (SparseLuBackend::Simplicial, 0xcfdc936a2c5518d6_u64),
        (SparseLuBackend::Supernodal, 0x17e8b1a27e7e9843_u64),
    ] {
        let tag = backend_tag(backend);
        let first = SparseLuFactorization::factorize_with_backend(&matrix_c64, backend, 1.0)
            .unwrap_or_else(|error| panic!("c64/{tag}: {error:?}"));
        let second = SparseLuFactorization::factorize_with_backend(&matrix_c64, backend, 1.0)
            .unwrap_or_else(|error| panic!("c64/{tag} repeat: {error:?}"));
        let bytes = first.to_checkpoint_bytes().expect("serialize c64");
        assert_eq!(
            bytes,
            second.to_checkpoint_bytes().expect("serialize c64 repeat"),
            "c64/{tag}: repeat factorization changed bytes"
        );
        assert_eq!(bytes.len(), 23718, "c64/{tag}: checkpoint length");
        assert_eq!(fnv_bytes(&bytes), golden_fnv, "c64/{tag}: golden");
        let row_perm = decode_row_perm(&bytes);
        assert_eq!(row_perm.len(), matrix_c64.num_rows);
        assert_eq!(
            fnv_usize(&row_perm),
            0x4f3c6c92c91581e5,
            "c64/{tag}: pivot-sequence golden"
        );
        let restored =
            SparseLuFactorization::<Complex64>::from_checkpoint_bytes(&bytes).expect("restore c64");
        assert_eq!(
            restored.to_checkpoint_bytes().expect("re-serialize c64"),
            bytes,
            "c64/{tag}: round-trip changed bytes"
        );
        let n = matrix_c64.num_rows;
        let expected: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.013).sin(), (i as f64 * 0.007).cos()))
            .collect();
        let rhs = matrix_c64.matvec(&expected);
        let solution = first.solve(&rhs).expect("c64 solve");
        let actual = matrix_c64.matvec(&solution);
        let error = actual
            .iter()
            .zip(&rhs)
            .map(|(a, e)| (a - e).norm())
            .fold(0.0_f64, f64::max);
        let scale = rhs.iter().map(|value| value.norm()).fold(0.0, f64::max);
        let residual = error / scale.max(f64::MIN_POSITIVE);
        assert!(
            residual <= 1.0e-10,
            "c64/{tag}: residual {residual:.2e} exceeds 1e-10"
        );
    }
}
