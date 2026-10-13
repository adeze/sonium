//! Host SpMV throughput across the parallel-dispatch tiers.
//!
//! Times `CsrMatrix::matvec_into_buf` (the FGMRES hot loop) on banded
//! matrices shaped like the fine-3D Navier-Stokes monolith. Every case
//! gates on a scalar forward reference before timing, then reports
//! per-iteration latency and effective bandwidth.
//!
//! Usage: `cargo bench -p sonium-solvers-sparse --bench spmv_host_scale -- [all|small|medium|large] [unchecked]`
//!
//! The optional `unchecked` mode validates once via the correctness gate
//! and then times the unchecked kernel, isolating the per-call
//! validation overhead from the kernel cost.

// Rust guideline compliant 2026-02-21

use sonium_solvers_sparse::CsrMatrix;
use std::hint::black_box;
use std::time::Instant;

const TOLERANCE: f64 = 1.0e-9;
const WARMUP_ITERS: usize = 2;

struct Case {
    name: &'static str,
    rows: usize,
    half_bandwidth: usize,
    repeats: usize,
}

const CASES: [Case; 3] = [
    Case {
        name: "small",
        rows: 4_096,
        half_bandwidth: 12,
        repeats: 40,
    },
    Case {
        name: "medium",
        rows: 50_000,
        half_bandwidth: 30,
        repeats: 12,
    },
    Case {
        name: "large",
        rows: 361_591,
        half_bandwidth: 47,
        repeats: 6,
    },
];

fn banded_matrix(rows: usize, half_bandwidth: usize) -> CsrMatrix<f64> {
    let mut row_ptrs = Vec::with_capacity(rows + 1);
    let mut columns = Vec::new();
    let mut values = Vec::new();
    row_ptrs.push(0);
    for row in 0..rows {
        let lo = row.saturating_sub(half_bandwidth);
        let hi = (row + half_bandwidth + 1).min(rows);
        for col in lo..hi {
            columns.push(col);
            values.push(1.0 + ((row + col) % 19) as f64 * 0.01);
        }
        row_ptrs.push(columns.len());
    }
    CsrMatrix::from_raw_parts(rows, rows, row_ptrs, columns, values)
}

fn scalar_forward(matrix: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    (0..matrix.num_rows)
        .map(|row| {
            (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                .map(|index| matrix.values[index] * x[matrix.col_indices[index]])
                .sum()
        })
        .collect()
}

fn checksum(values: &[f64]) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| *value * (index + 1) as f64)
        .sum()
}

fn timed_iters(case: &Case, mut apply: impl FnMut()) -> f64 {
    for _ in 0..WARMUP_ITERS {
        apply();
    }
    let start = Instant::now();
    for _ in 0..case.repeats {
        apply();
    }
    start.elapsed().as_nanos() as f64 / case.repeats as f64
}

fn run_case(case: &Case, unchecked: bool) {
    let matrix = banded_matrix(case.rows, case.half_bandwidth);
    let nnz = matrix.values.len();
    let x: Vec<f64> = (0..matrix.num_cols)
        .map(|column| 1.0 + (column % 23) as f64 * 0.005)
        .collect();

    // Correctness gate before timing.
    let expected = scalar_forward(&matrix, &x);
    let mut y = vec![0.0; matrix.num_rows];
    matrix.matvec_into_buf(&x, &mut y);
    let error: f64 = y
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (*actual - *expected).abs())
        .fold(0.0, f64::max);
    assert!(
        error <= TOLERANCE,
        "SpMV error {error:e} exceeds {TOLERANCE:e}"
    );
    assert!(y.iter().all(|value| value.is_finite()));

    let ns_per_iter = if unchecked {
        // SAFETY: the correctness gate above ran the checked apply (which
        // validates the hot structure), and the matrix is not mutated
        // afterwards, so the unchecked contract holds for every iteration.
        timed_iters(case, || unsafe {
            matrix.matvec_into_buf_unchecked(black_box(&x), &mut y);
            black_box(&y);
        })
    } else {
        timed_iters(case, || {
            matrix.matvec_into_buf(black_box(&x), &mut y);
            black_box(&y);
        })
    };

    // Bytes per SpMV: values + column indices stream once; x is gathered
    // (counted once, L3-resident here) and y is written once.
    let bytes = nnz * 12 + (matrix.num_rows + matrix.num_cols) * 8;
    let gbps = bytes as f64 / ns_per_iter;
    let tier_threads = sonium_solvers_sys::runtime::bandwidth_threads_for_nnz(nnz);
    println!(
        "{:>6} rows={:>7} nnz={:>9} tier_threads={:>3} {:>10.3} ms/iter {:>8.1} GB/s checksum={:e}",
        case.name,
        matrix.num_rows,
        nnz,
        tier_threads,
        ns_per_iter / 1.0e6,
        gbps,
        checksum(&y),
    );
    println!(
        "{{\"bench\":\"spmv_host_scale\",\"case\":\"{}\",\"rows\":{},\"nnz\":{},\"tier_threads\":{},\"unchecked\":{},\"ns_per_iter\":{:.1},\"gbps\":{:.2}}}",
        case.name, matrix.num_rows, nnz, tier_threads, unchecked, ns_per_iter, gbps,
    );
}

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let rt = sonium_solvers_sys::runtime::runtime();
    println!(
        "spmv_host_scale parallelism={} compute_threads={} bandwidth_threads={} rayon_current_main={}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        rt.compute_threads,
        rt.bandwidth_threads,
        rayon::current_num_threads(),
    );
    let unchecked = std::env::args().nth(2).as_deref() == Some("unchecked");
    for case in CASES.iter().filter(|c| filter == "all" || filter == c.name) {
        run_case(case, unchecked);
    }
}
