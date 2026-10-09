//! Checked MLFMM algebraic-transpose multi-RHS action evidence.

use num_complex::Complex64;
use sonium_bem::core::assembly::mlfmm::{DMatrixEntry, LevelMatrices, MlfmmSystem, NearFieldBlock};
use sonium_bem::core::types::{Cluster, ClusterLevel};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "bem.mlfmm-transpose-block.v1";

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn fixture() -> MlfmmSystem {
    let mut root = Cluster::new([0.0, 0.0, 0.0]);
    root.sons = vec![0, 1];
    let mut root_level = ClusterLevel::new(1);
    root_level.clusters.push(root);

    let mut leaf0 = Cluster::new([-1.0, 0.0, 0.0]);
    leaf0.father = Some(0);
    let mut leaf1 = Cluster::new([1.0, 0.0, 0.0]);
    leaf1.father = Some(0);
    let mut leaf_level = ClusterLevel::new(2);
    leaf_level.clusters.extend([leaf0, leaf1]);

    let mut system = MlfmmSystem::new(2, 2, vec![root_level, leaf_level], vec![2, 2]);
    system.leaf_dof_indices = vec![vec![0], vec![1]];
    system.t_matrices = vec![
        LevelMatrices {
            level: 0,
            num_clusters: 1,
            matrices: vec![DenseMatrix::from_row_major_vec(
                2,
                4,
                vec![
                    c(1.0, 0.0),
                    c(0.0, 0.0),
                    c(-0.3, 0.1),
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                    c(0.7, -0.4),
                    c(0.0, 0.0),
                    c(-0.1, 0.6),
                ],
            )],
        },
        LevelMatrices {
            level: 1,
            num_clusters: 2,
            matrices: vec![
                DenseMatrix::from_row_major_vec(2, 1, vec![c(0.8, 0.1), c(-0.2, 0.3)]),
                DenseMatrix::from_row_major_vec(2, 1, vec![c(0.4, -0.6), c(0.9, 0.2)]),
            ],
        },
    ];
    system.d_matrices = vec![
        vec![DMatrixEntry {
            source_cluster: 0,
            field_cluster: 0,
            level: 0,
            diagonal: vec![c(0.6, 0.2), c(-0.1, 0.5)],
            active_directions: None,
        }],
        vec![DMatrixEntry {
            source_cluster: 0,
            field_cluster: 1,
            level: 1,
            diagonal: vec![c(0.3, -0.7), c(0.2, 0.4)],
            active_directions: None,
        }],
    ];
    system.rebuild_translation_plans();
    system.s_matrices = vec![
        LevelMatrices {
            level: 0,
            num_clusters: 1,
            matrices: vec![DenseMatrix::from_row_major_vec(
                4,
                2,
                vec![
                    c(0.5, 0.0),
                    c(0.0, 0.0),
                    c(0.2, -0.1),
                    c(-0.4, 0.6),
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                    c(0.9, 0.1),
                    c(0.0, 0.0),
                ],
            )],
        },
        LevelMatrices {
            level: 1,
            num_clusters: 2,
            matrices: vec![
                DenseMatrix::from_row_major_vec(1, 2, vec![c(0.2, 0.9), c(-0.5, 0.4)]),
                DenseMatrix::from_row_major_vec(1, 2, vec![c(0.6, -0.2), c(0.3, 0.7)]),
            ],
        },
    ];
    system.near_matrix.push(NearFieldBlock {
        source_cluster: 0,
        field_cluster: 1,
        coefficients: DenseMatrix::from_row_major_vec(1, 1, vec![c(0.25, -0.15)]),
    });
    system
}

fn main() {
    let started = Instant::now();
    let system = fixture();
    let input = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![c(0.7, -0.3), c(-0.2, 0.5), c(0.4, 0.8), c(-0.6, 0.1)],
    );
    let block = system.matvec_transpose_block(&input);
    let rows = [1, 0, 1];
    let selected = system
        .apply_transpose_rows_block_checked(&input, &rows)
        .expect("checked MLFMM transpose rows");
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
    for (selected_row, &row) in rows.iter().enumerate() {
        for rhs in 0..input.ncols() {
            maximum_error =
                maximum_error.max((selected[[selected_row, rhs]] - block[[row, rhs]]).norm());
        }
    }

    let passed = maximum_error <= 1.0e-12;
    let result = if passed {
        QaResult::passed(CASE_ID)
            .reference_checksum("synthetic/deterministic-fixture:bem-mlfmm-transpose-block-v1")
            .input_checksum("synthetic/deterministic-fixture:bem-mlfmm-transpose-block-v1")
            .dimension("2-DOF two-level MLFMM near/far RHS block")
            .tolerance("block/scalar max error <= 1e-12")
            .metric("maximum_error", maximum_error)
            .elapsed_ms(started.elapsed().as_millis() as u64)
    } else {
        QaResult::failed(CASE_ID, format!("block/scalar error {maximum_error:.3e}"))
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if !passed {
        std::process::exit(1);
    }
}
