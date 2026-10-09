//! Deterministic CPU contract for nested/multi-region trace coupling.

use num_complex::Complex64;
use sonium_bem::{
    ConservativeTraceTransfer, CouplingConvention, CouplingError, NestedTraceEdge,
    NestedTraceOperator, NestedTraceRegion,
};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::sparse::DenseBlockOperator;
use std::time::Instant;

const CASE_ID: &str = "bem.nested-trace-operator.v1";

fn convention() -> CouplingConvention {
    CouplingConvention {
        bem: sonium_bem::ACOUSTIC_CONVENTION,
        normal_points_out_of_interior: true,
    }
}

fn transfer_a() -> ConservativeTraceTransfer {
    ConservativeTraceTransfer::try_new(
        vec![vec![(0, 1.0 / 3.0), (1, 2.0 / 3.0)]],
        vec![1.0, 2.0],
        vec![3.0],
    )
    .expect("transfer A")
}

fn transfer_b() -> ConservativeTraceTransfer {
    ConservativeTraceTransfer::try_new(
        vec![vec![(0, 1.0)], vec![(0, 1.0)]],
        vec![3.0],
        vec![1.0, 2.0],
    )
    .expect("transfer B")
}

fn region(name: &str, matrix: DenseMatrix<Complex64>) -> NestedTraceRegion {
    NestedTraceRegion::new(name, Box::new(DenseBlockOperator::new(matrix)))
}

fn max_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max)
}

fn selected_error(actual: &[Complex64], full: &[Complex64], rows: &[usize]) -> f64 {
    let expected = rows.iter().map(|&row| full[row]).collect::<Vec<_>>();
    max_error(actual, &expected)
}

fn selected_block_error(
    actual: &DenseMatrix<Complex64>,
    full: &DenseMatrix<Complex64>,
    rows: &[usize],
) -> f64 {
    rows.iter()
        .enumerate()
        .flat_map(|(selected_row, &global_row)| {
            (0..actual.ncols()).map(move |column| {
                (actual[[selected_row, column]] - full[[global_row, column]]).norm()
            })
        })
        .fold(0.0, f64::max)
}

fn transpose(matrix: &DenseMatrix<Complex64>, input: &[Complex64]) -> Vec<Complex64> {
    (0..matrix.ncols())
        .map(|column| {
            (0..matrix.nrows())
                .map(|row| matrix[[row, column]] * input[row])
                .sum()
        })
        .collect()
}

fn hermitian(matrix: &DenseMatrix<Complex64>, input: &[Complex64]) -> Vec<Complex64> {
    (0..matrix.ncols())
        .map(|column| {
            (0..matrix.nrows())
                .map(|row| matrix[[row, column]].conj() * input[row])
                .sum()
        })
        .collect()
}

type Metrics = (
    f64,
    f64,
    f64,
    f64,
    usize,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    usize,
);

fn run() -> Result<Metrics, String> {
    let regions = vec![
        region(
            "core",
            DenseMatrix::from_row_major_vec(
                2,
                2,
                vec![
                    Complex64::new(2.0, 0.2),
                    Complex64::new(-0.3, 0.1),
                    Complex64::new(0.7, -0.4),
                    Complex64::new(1.4, 0.3),
                ],
            ),
        ),
        region(
            "shell",
            DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(1.2, -0.5)]),
        ),
        region(
            "exterior",
            DenseMatrix::from_row_major_vec(
                2,
                2,
                vec![
                    Complex64::new(1.1, 0.0),
                    Complex64::new(0.2, -0.2),
                    Complex64::new(-0.4, 0.6),
                    Complex64::new(0.8, 0.1),
                ],
            ),
        ),
    ];
    let edges = vec![
        NestedTraceEdge::new("core-shell", 0, 1, transfer_a(), Complex64::new(0.4, -0.1)),
        NestedTraceEdge::new(
            "shell-exterior",
            1,
            2,
            transfer_b(),
            Complex64::new(-0.2, 0.3),
        ),
    ];
    let operator = NestedTraceOperator::try_new(regions, edges, convention())
        .map_err(|error| error.to_string())?;
    let state = vec![
        Complex64::new(0.2, -0.5),
        Complex64::new(-0.7, 0.3),
        Complex64::new(0.4, 0.9),
        Complex64::new(-0.1, 0.2),
        Complex64::new(0.8, -0.4),
    ];
    let dense = operator
        .to_dense_matrix()
        .map_err(|error| error.to_string())?;
    let forward = max_error(
        &operator
            .try_apply(&state)
            .map_err(|error| error.to_string())?,
        &dense.matvec(&state),
    );
    let transpose_error = max_error(
        &operator
            .try_apply_transpose(&state)
            .map_err(|error| error.to_string())?,
        &transpose(&dense, &state),
    );
    let hermitian_error = max_error(
        &operator
            .try_apply_hermitian(&state)
            .map_err(|error| error.to_string())?,
        &hermitian(&dense, &state),
    );
    let block = DenseMatrix::from_row_major_vec(
        5,
        2,
        vec![
            state[0],
            Complex64::new(-0.2, 0.1),
            state[1],
            Complex64::new(0.5, -0.8),
            state[2],
            Complex64::new(0.3, 0.4),
            state[3],
            Complex64::new(-0.9, 0.2),
            state[4],
            Complex64::new(0.1, 0.7),
        ],
    );
    let block_output = operator
        .try_apply_block(&block)
        .map_err(|error| error.to_string())?;
    let mut block_error: f64 = 0.0;
    for column in 0..block.ncols() {
        block_error = block_error.max(max_error(
            &block_output.column_to_vec(column),
            &operator
                .try_apply(&block.column_to_vec(column))
                .map_err(|error| error.to_string())?,
        ));
    }

    let rows = [4, 0, 4, 2];
    let forward_output = operator
        .try_apply(&state)
        .map_err(|error| error.to_string())?;
    let transpose_output = operator
        .try_apply_transpose(&state)
        .map_err(|error| error.to_string())?;
    let hermitian_output = operator
        .try_apply_hermitian(&state)
        .map_err(|error| error.to_string())?;
    let row_forward_error = selected_error(
        &operator
            .try_apply_rows(&state, &rows)
            .map_err(|error| error.to_string())?,
        &forward_output,
        &rows,
    );
    let row_transpose_error = selected_error(
        &operator
            .try_apply_transpose_rows(&state, &rows)
            .map_err(|error| error.to_string())?,
        &transpose_output,
        &rows,
    );
    let row_hermitian_error = selected_error(
        &operator
            .try_apply_hermitian_rows(&state, &rows)
            .map_err(|error| error.to_string())?,
        &hermitian_output,
        &rows,
    );
    let row_forward_block = operator
        .try_apply_rows_block(&block, &rows)
        .map_err(|error| error.to_string())?;
    let row_transpose_block = operator
        .try_apply_transpose_rows_block(&block, &rows)
        .map_err(|error| error.to_string())?;
    let row_hermitian_block = operator
        .try_apply_hermitian_rows_block(&block, &rows)
        .map_err(|error| error.to_string())?;
    let transpose_block = operator
        .try_apply_transpose_block(&block)
        .map_err(|error| error.to_string())?;
    let hermitian_block = operator
        .try_apply_hermitian_block(&block)
        .map_err(|error| error.to_string())?;
    let row_forward_block_error = selected_block_error(&row_forward_block, &block_output, &rows);
    let row_transpose_block_error =
        selected_block_error(&row_transpose_block, &transpose_block, &rows);
    let row_hermitian_block_error =
        selected_block_error(&row_hermitian_block, &hermitian_block, &rows);
    let invalid_row_rejections = usize::from(
        operator
            .try_apply_rows(&state, &[operator.dimension()])
            .is_err(),
    ) + usize::from(
        operator
            .try_apply_rows_block(&block, &[operator.dimension()])
            .is_err(),
    );

    let mut rejection_count = 0;
    let self_edge = NestedTraceEdge::new(
        "self",
        0,
        0,
        ConservativeTraceTransfer::try_new(vec![vec![(0, 1.0)]], vec![1.0], vec![1.0])
            .map_err(|error| error.to_string())?,
        Complex64::new(1.0, 0.0),
    );
    if matches!(
        NestedTraceOperator::try_new(
            vec![region(
                "single",
                DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(1.0, 0.0)]),
            )],
            vec![self_edge],
            convention(),
        ),
        Err(CouplingError::InvalidEdge { .. })
    ) {
        rejection_count += 1;
    }
    let out_of_range = NestedTraceEdge::new(
        "missing",
        0,
        2,
        ConservativeTraceTransfer::try_new(vec![vec![(0, 1.0)]], vec![1.0], vec![1.0])
            .map_err(|error| error.to_string())?,
        Complex64::new(1.0, 0.0),
    );
    if matches!(
        NestedTraceOperator::try_new(
            vec![
                region(
                    "first",
                    DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(1.0, 0.0)]),
                ),
                region(
                    "second",
                    DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(1.0, 0.0)]),
                ),
            ],
            vec![out_of_range],
            convention(),
        ),
        Err(CouplingError::InvalidEdge { .. })
    ) {
        rejection_count += 1;
    }

    Ok((
        forward,
        transpose_error,
        hermitian_error,
        block_error,
        rejection_count,
        row_forward_error,
        row_transpose_error,
        row_hermitian_error,
        row_forward_block_error,
        row_transpose_block_error,
        row_hermitian_block_error,
        invalid_row_rejections,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((
            forward,
            transpose,
            hermitian,
            block,
            rejections,
            row_forward,
            row_transpose,
            row_hermitian,
            row_forward_block,
            row_transpose_block,
            row_hermitian_block,
            invalid_rows,
        )) if forward <= 1.0e-12
            && transpose <= 1.0e-12
            && hermitian <= 1.0e-12
            && block <= 1.0e-12
            && row_forward <= 1.0e-12
            && row_transpose <= 1.0e-12
            && row_hermitian <= 1.0e-12
            && row_forward_block <= 1.0e-12
            && row_transpose_block <= 1.0e-12
            && row_hermitian_block <= 1.0e-12
            && rejections == 2
            && invalid_rows == 2 =>
        {
            QaResult::passed(CASE_ID)
                .metric("forward_error", forward)
                .metric("transpose_error", transpose)
                .metric("hermitian_error", hermitian)
                .metric("block_error", block)
                .metric("row_forward_error", row_forward)
                .metric("row_transpose_error", row_transpose)
                .metric("row_hermitian_error", row_hermitian)
                .metric("row_forward_block_error", row_forward_block)
                .metric("row_transpose_block_error", row_transpose_block)
                .metric("row_hermitian_block_error", row_hermitian_block)
                .metric("invalid_edge_rejections", rejections as f64)
                .metric("invalid_row_rejections", invalid_rows as f64)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent dense nested multi-region trace action")
                .dimension("2+1+2 region trace DOFs")
                .formulation("checked matrix-free nested FEM-BEM trace coupling")
                .seed("bem-nested-trace-operator:cpu:v1")
        }
        Ok((
            forward,
            transpose,
            hermitian,
            block,
            rejections,
            row_forward,
            row_transpose,
            row_hermitian,
            row_forward_block,
            row_transpose_block,
            row_hermitian_block,
            invalid_rows,
        )) => QaResult::failed(
            CASE_ID,
            format!(
                "contract tolerance failed: forward={forward:e}, transpose={transpose:e}, Hermitian={hermitian:e}, block={block:e}, row_forward={row_forward:e}, row_transpose={row_transpose:e}, row_hermitian={row_hermitian:e}, row_forward_block={row_forward_block:e}, row_transpose_block={row_transpose_block:e}, row_hermitian_block={row_hermitian_block:e}, rejections={rejections}, invalid_rows={invalid_rows}"
            ),
        ),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    let result = result.elapsed_ms(started.elapsed().as_millis() as u64);
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
