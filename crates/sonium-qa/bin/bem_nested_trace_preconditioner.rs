//! Deterministic CPU contract for nested/multi-region block preconditioning.

use num_complex::Complex64;
use sonium_bem::{
    ConservativeTraceTransfer, CouplingConvention, CouplingError, NestedTraceEdge,
    NestedTracePreconditioner, NestedTracePreconditionerKind,
};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::traits::{AdjointCapability, Preconditioner};
use std::time::Instant;

const CASE_ID: &str = "bem.nested-trace-preconditioner.v1";

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

struct ScalePreconditioner {
    scale: Complex64,
}

impl Preconditioner<Complex64> for ScalePreconditioner {
    fn apply(&self, rhs: &[Complex64]) -> Vec<Complex64> {
        rhs.iter().map(|value| *value * self.scale).collect()
    }

    fn apply_transpose(&self, rhs: &[Complex64]) -> Vec<Complex64> {
        self.apply(rhs)
    }

    fn apply_hermitian(&self, rhs: &[Complex64]) -> Vec<Complex64> {
        rhs.iter().map(|value| *value * self.scale.conj()).collect()
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

fn max_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
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

fn build(kind: NestedTracePreconditionerKind) -> Result<NestedTracePreconditioner, String> {
    NestedTracePreconditioner::try_new_with_kind(
        vec![
            Box::new(ScalePreconditioner {
                scale: Complex64::new(0.5, 0.1),
            }),
            Box::new(ScalePreconditioner {
                scale: Complex64::new(0.25, -0.2),
            }),
            Box::new(ScalePreconditioner {
                scale: Complex64::new(0.75, 0.05),
            }),
        ],
        vec![2, 1, 2],
        vec![
            NestedTraceEdge::new("core-shell", 0, 1, transfer_a(), Complex64::new(0.4, -0.1)),
            NestedTraceEdge::new(
                "shell-exterior",
                1,
                2,
                transfer_b(),
                Complex64::new(-0.2, 0.3),
            ),
        ],
        kind,
        convention(),
    )
    .map_err(|error| error.to_string())
}

fn run() -> Result<(f64, usize), String> {
    let state = vec![
        Complex64::new(0.2, -0.5),
        Complex64::new(-0.7, 0.3),
        Complex64::new(0.4, 0.9),
        Complex64::new(-0.1, 0.2),
        Complex64::new(0.8, -0.4),
    ];
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
    let mut maximum_error: f64 = 0.0;
    for kind in [
        NestedTracePreconditionerKind::BlockDiagonal,
        NestedTracePreconditionerKind::ForwardSweep,
        NestedTracePreconditionerKind::ReverseSweep,
    ] {
        let preconditioner = build(kind)?;
        let mut dense = DenseMatrix::from_fn(5, 5, |_, _| Complex64::default());
        for column in 0..5 {
            let mut basis = vec![Complex64::default(); 5];
            basis[column] = Complex64::new(1.0, 0.0);
            dense.set_column(
                column,
                &preconditioner
                    .try_apply_checked(&basis)
                    .map_err(|error| error.to_string())?,
            );
        }
        maximum_error = maximum_error.max(max_error(
            &preconditioner
                .try_apply_checked(&state)
                .map_err(|error| error.to_string())?,
            &dense.matvec(&state),
        ));
        maximum_error = maximum_error.max(max_error(
            &preconditioner
                .try_apply_transpose_checked(&state)
                .map_err(|error| error.to_string())?,
            &transpose(&dense, &state),
        ));
        maximum_error = maximum_error.max(max_error(
            &preconditioner
                .try_apply_hermitian_checked(&state)
                .map_err(|error| error.to_string())?,
            &hermitian(&dense, &state),
        ));
        let output = preconditioner
            .try_apply_block(&block)
            .map_err(|error| error.to_string())?;
        for column in 0..block.ncols() {
            maximum_error = maximum_error.max(max_error(
                &output.column_to_vec(column),
                &preconditioner
                    .try_apply_checked(&block.column_to_vec(column))
                    .map_err(|error| error.to_string())?,
            ));
        }
    }

    let mut rejections = 0;
    if matches!(
        NestedTracePreconditioner::try_new_with_kind(
            vec![Box::new(ScalePreconditioner {
                scale: Complex64::new(1.0, 0.0),
            })],
            vec![2, 1],
            Vec::new(),
            NestedTracePreconditionerKind::BlockDiagonal,
            convention(),
        ),
        Err(CouplingError::Dimension { .. })
    ) {
        rejections += 1;
    }
    if matches!(
        NestedTracePreconditioner::try_new_with_kind(
            vec![
                Box::new(ScalePreconditioner {
                    scale: Complex64::new(1.0, 0.0),
                }),
                Box::new(ScalePreconditioner {
                    scale: Complex64::new(1.0, 0.0),
                }),
            ],
            vec![1, 1],
            vec![NestedTraceEdge::new(
                "self",
                0,
                0,
                ConservativeTraceTransfer::try_new(vec![vec![(0, 1.0)]], vec![1.0], vec![1.0],)
                    .map_err(|error| error.to_string())?,
                Complex64::new(1.0, 0.0),
            )],
            NestedTracePreconditionerKind::BlockDiagonal,
            convention(),
        ),
        Err(CouplingError::InvalidEdge { .. })
    ) {
        rejections += 1;
    }
    Ok((maximum_error, rejections))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((maximum_error, rejections)) if maximum_error <= 1.0e-12 && rejections == 2 => {
            QaResult::passed(CASE_ID)
                .metric("maximum_action_error", maximum_error)
                .metric("preconditioner_variants", 3.0)
                .metric("invalid_input_rejections", rejections as f64)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent dense nested triangular/block preconditioner actions")
                .dimension("three regions with 2+1+2 trace DOFs")
                .formulation("checked FEM-BEM nested block preconditioning")
                .seed("bem-nested-trace-preconditioner:cpu:v1")
        }
        Ok((maximum_error, rejections)) => QaResult::failed(
            CASE_ID,
            format!(
                "contract tolerance failed: maximum_error={maximum_error:e}, rejections={rejections}"
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
