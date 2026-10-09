use sonium_solvers_direct::SparseLuFactorization;
use sonium_solvers_sparse::CsrMatrix;

fn tridiagonal(n: usize) -> CsrMatrix<f64> {
    let mut triplets = Vec::with_capacity(3 * n.saturating_sub(1) + 1);
    for row in 0..n {
        triplets.push((row, row, 4.0 + 0.001 * row as f64));
        if row > 0 {
            triplets.push((row, row - 1, -1.0));
        }
        if row + 1 < n {
            triplets.push((row, row + 1, -1.0));
        }
    }
    CsrMatrix::from_triplets(n, n, triplets)
}

fn permuted_tridiagonal(n: usize) -> (CsrMatrix<f64>, Vec<usize>) {
    let permutation: Vec<usize> = (0..n).map(|index| (37 * index) % n).collect();
    let mut triplets = Vec::with_capacity(3 * n.saturating_sub(1) + 1);
    for row in 0..n {
        let permuted_row = permutation[row];
        triplets.push((permuted_row, permuted_row, 4.0 + 0.001 * row as f64));
        if row > 0 {
            triplets.push((permuted_row, permutation[row - 1], -1.0));
        }
        if row + 1 < n {
            triplets.push((permuted_row, permutation[row + 1], -1.0));
        }
    }
    (CsrMatrix::from_triplets(n, n, triplets), permutation)
}

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

#[test]
fn level_scheduled_sparse_lu_matches_serial_known_solution() {
    let expected: Vec<f64> = (0..1200)
        .map(|index| {
            let x = index as f64 / 1199.0;
            0.25 + 0.5 * x - 0.2 * (3.0 * x).cos()
        })
        .collect();

    let serial_matrix = tridiagonal(999);
    let serial_expected = &expected[..999];
    let serial_rhs = serial_matrix.matvec(serial_expected);
    let serial_factor = SparseLuFactorization::factorize(&serial_matrix).expect("serial factor");
    let serial_solution = serial_factor.solve(&serial_rhs).expect("serial solve");
    assert!(max_error(&serial_solution, serial_expected) < 1.0e-10);

    let parallel_matrix = tridiagonal(expected.len());
    let parallel_rhs = parallel_matrix.matvec(&expected);
    let parallel_factor =
        SparseLuFactorization::factorize(&parallel_matrix).expect("level-scheduled factorization");
    for _ in 0..3 {
        let parallel_solution = parallel_factor
            .solve(&parallel_rhs)
            .expect("level-scheduled solve");
        assert!(max_error(&parallel_solution, &expected) < 1.0e-10);
    }
}

#[test]
fn supernodal_large_sparse_lu_matches_known_solution() {
    let expected: Vec<f64> = (0..1200)
        .map(|index| {
            let x = index as f64 / 1199.0;
            0.25 + 0.5 * x - 0.2 * (3.0 * x).cos()
        })
        .collect();
    let matrix = tridiagonal(expected.len());
    let rhs = matrix.matvec(&expected);
    let factor =
        SparseLuFactorization::factorize_supernodal(&matrix).expect("supernodal factorization");

    for _ in 0..3 {
        let solution = factor.solve(&rhs).expect("supernodal solve");
        assert!(max_error(&solution, &expected) < 1.0e-10);
    }
}

/// Run `solve` with the wave-dispatch gate forced to both extremes:
/// `0` dispatches every wave (pre-gate behavior), `usize::MAX` runs every
/// wave serially inline. Returns both solutions with the ambient env value
/// restored. No other test reads this variable for correctness (it only
/// routes work between bitwise-identical paths), so the process-global set
/// cannot disturb parallel tests.
fn solve_at_gate_extremes(
    factor: &SparseLuFactorization<f64>,
    rhs: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    const GATE: &str = "SONIUM_SPARSE_PAR_MIN_NNZ";
    // SAFETY: the gate only routes work between bitwise-identical paths, so
    // any value a concurrent test thread observes keeps its solve correct;
    // the var is restored below and no other test depends on its value.
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

#[test]
fn wave_dispatch_gate_is_bitwise_identical() {
    // Serial-in-wave and parallel-in-wave write disjoint rows (each target
    // once), so the gate must move work between identical-execution paths:
    // dispatch-every-wave and serialize-every-wave agree bit-for-bit.
    let expected: Vec<f64> = (0..1200)
        .map(|index| {
            let x = index as f64 / 1199.0;
            0.25 + 0.5 * x - 0.2 * (3.0 * x).cos()
        })
        .collect();
    let matrix = tridiagonal(expected.len());
    let rhs = matrix.matvec(&expected);
    let ambient = std::env::var("SONIUM_SPARSE_PAR_MIN_NNZ").ok();
    for factor in [
        SparseLuFactorization::factorize(&matrix).expect("simplicial factor"),
        SparseLuFactorization::factorize_supernodal(&matrix).expect("supernodal factor"),
    ] {
        let (dispatched, serial) = solve_at_gate_extremes(&factor, &rhs);
        assert_eq!(dispatched.len(), serial.len());
        for (index, (&a, &b)) in dispatched.iter().zip(&serial).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "gate changed solution bits at index {index}"
            );
        }
        // The block path shares the prevalidated core: same bits there too.
        let block: Vec<f64> = rhs.iter().flat_map(|&v| [v, 2.0 * v]).collect();
        let mut out_dispatched = vec![0.0; block.len()];
        let mut out_serial = vec![0.0; block.len()];
        // SAFETY: same justification as `solve_at_gate_extremes`.
        unsafe { std::env::set_var("SONIUM_SPARSE_PAR_MIN_NNZ", "0") };
        factor
            .solve_many_into(&block, &mut out_dispatched, 2)
            .expect("dispatched block solve");
        unsafe { std::env::set_var("SONIUM_SPARSE_PAR_MIN_NNZ", &usize::MAX.to_string()) };
        factor
            .solve_many_into(&block, &mut out_serial, 2)
            .expect("serial block solve");
        let bits = |x: &[f64]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&out_dispatched), bits(&out_serial));
        // Block columns match the corresponding single-RHS solutions solved
        // under the same gate extreme.
        let single = factor.solve(&rhs).expect("single solve");
        let doubled: Vec<f64> = rhs.iter().map(|v| 2.0 * v).collect();
        let single2 = factor.solve(&doubled).expect("scaled solve");
        for row in 0..expected.len() {
            assert_eq!(out_serial[row * 2].to_bits(), single[row].to_bits());
            assert_eq!(out_serial[row * 2 + 1].to_bits(), single2[row].to_bits());
        }
    }
    // SAFETY: same justification as `solve_at_gate_extremes`.
    match ambient {
        Some(value) => unsafe { std::env::set_var("SONIUM_SPARSE_PAR_MIN_NNZ", value) },
        None => unsafe { std::env::remove_var("SONIUM_SPARSE_PAR_MIN_NNZ") },
    }
}

#[test]
fn permuted_large_sparse_lu_matches_known_solution() {
    let base_solution: Vec<f64> = (0..1200)
        .map(|index| {
            let x = index as f64 / 1199.0;
            0.25 + 0.5 * x - 0.2 * (3.0 * x).cos()
        })
        .collect();
    let (matrix, permutation) = permuted_tridiagonal(base_solution.len());
    let mut expected = vec![0.0; base_solution.len()];
    for (index, &permuted_index) in permutation.iter().enumerate() {
        expected[permuted_index] = base_solution[index];
    }
    let rhs = matrix.matvec(&expected);

    let simplicial = SparseLuFactorization::factorize(&matrix).expect("permuted factorization");
    let supernodal =
        SparseLuFactorization::factorize_supernodal(&matrix).expect("permuted supernodal factor");
    for factor in [&simplicial, &supernodal] {
        for _ in 0..3 {
            let solution = factor.solve(&rhs).expect("permuted solve");
            assert!(max_error(&solution, &expected) < 1.0e-10);
        }
    }
}
