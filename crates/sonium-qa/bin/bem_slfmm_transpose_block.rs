//! Checked SLFMM algebraic-transpose multi-RHS action evidence.

use num_complex::Complex64;
use sonium_bem::core::assembly::slfmm::{DMatrixEntry, NearFieldBlock, SlfmmSystem};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "bem.slfmm-transpose-block.v1";

fn fixture() -> SlfmmSystem {
    let mut system = SlfmmSystem::new(2, 2, 2, 0);
    system.cluster_dof_indices = vec![vec![0], vec![1]];
    system.near_matrix.push(NearFieldBlock {
        source_cluster: 0,
        field_cluster: 1,
        coefficients: DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(0.5, -0.2)]),
    });
    system.t_matrices = vec![
        DenseMatrix::from_row_major_vec(
            2,
            1,
            vec![Complex64::new(1.0, 0.1), Complex64::new(-0.3, 0.4)],
        ),
        DenseMatrix::from_row_major_vec(
            2,
            1,
            vec![Complex64::new(0.7, -0.2), Complex64::new(0.2, 0.6)],
        ),
    ];
    system.s_matrices = vec![
        DenseMatrix::from_row_major_vec(
            1,
            2,
            vec![Complex64::new(0.9, 0.3), Complex64::new(-0.1, 0.5)],
        ),
        DenseMatrix::from_row_major_vec(
            1,
            2,
            vec![Complex64::new(-0.4, 0.2), Complex64::new(0.8, -0.1)],
        ),
    ];
    system.d_matrices.push(DMatrixEntry {
        source_cluster: 0,
        field_cluster: 1,
        diagonal: vec![Complex64::new(0.6, -0.3), Complex64::new(-0.2, 0.7)],
    });
    system
}

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        let system = fixture();
        let input = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(0.2, -0.4),
                Complex64::new(-0.7, 0.1),
                Complex64::new(0.5, 0.8),
                Complex64::new(-0.1, -0.2),
            ],
        );
        let block = system.matvec_transpose_block(&input);
        let mut maximum_error: f64 = 0.0;
        for rhs in 0..input.ncols() {
            let column = (0..input.nrows())
                .map(|row| input[[row, rhs]])
                .collect::<Vec<_>>();
            let scalar = system.matvec_transpose(&column);
            for row in 0..input.nrows() {
                maximum_error = maximum_error.max((block[[row, rhs]] - scalar[row]).norm());
            }
        }
        assert!(
            maximum_error <= 1.0e-12,
            "block/scalar transpose error {maximum_error:e}"
        );
        maximum_error
    });
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(maximum_error) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .reference("scalar SLFMM algebraic-transpose action")
            .reference_checksum("synthetic/deterministic-fixture:bem-slfmm-transpose-block-v1")
            .input_checksum("synthetic/deterministic-fixture:bem-slfmm-transpose-block-v1")
            .dimension("2-DOF SLFMM near/far fixture; two transpose RHS columns")
            .formulation("batched SLFMM near-field and S-D-T transpose traversal")
            .tolerance("block/scalar transpose error <= 1e-12")
            .metric("maximum_block_scalar_error", maximum_error)
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(payload) => QaResult::failed(CASE_ID, format!("panic: {payload:?}")),
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if !passed {
        std::process::exit(1);
    }
}
