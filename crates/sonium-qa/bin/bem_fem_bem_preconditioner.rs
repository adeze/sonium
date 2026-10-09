//! Deterministic matrix-free FEM--BEM block-preconditioner evidence.

use num_complex::Complex64;
use sonium_bem::{
    ConservativeTraceTransfer, CoupledPreconditionerKind, CoupledTracePreconditioner, CouplingError,
};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::traits::{AdjointCapability, Preconditioner};

const CASE_ID: &str = "bem.fem-bem-preconditioner.v1";

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

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

fn transfer() -> Result<ConservativeTraceTransfer, CouplingError> {
    ConservativeTraceTransfer::try_new(
        vec![vec![(0, 0.25), (1, 0.75)], vec![(0, 0.75), (1, 0.25)]],
        vec![1.25, 1.75],
        vec![2.0, 1.0],
    )
}

fn max_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max)
}

fn transpose_matvec(matrix: &DenseMatrix<Complex64>, state: &[Complex64]) -> Vec<Complex64> {
    (0..matrix.ncols())
        .map(|column| {
            (0..matrix.nrows())
                .map(|row| matrix[[row, column]] * state[row])
                .sum()
        })
        .collect()
}

fn hermitian_matvec(matrix: &DenseMatrix<Complex64>, state: &[Complex64]) -> Vec<Complex64> {
    (0..matrix.ncols())
        .map(|column| {
            (0..matrix.nrows())
                .map(|row| matrix[[row, column]].conj() * state[row])
                .sum()
        })
        .collect()
}

fn run() -> Result<(usize, f64, f64, f64, f64, usize, usize), String> {
    let state = vec![
        Complex64::new(0.25, -0.5),
        Complex64::new(-1.0, 0.75),
        Complex64::new(0.5, 0.125),
        Complex64::new(-0.25, 1.0),
    ];
    let block = DenseMatrix::from_row_major_vec(
        4,
        2,
        vec![
            state[0],
            Complex64::new(-0.5, 0.25),
            state[1],
            Complex64::new(1.25, -0.75),
            state[2],
            Complex64::new(-1.0, 0.5),
            state[3],
            Complex64::new(0.75, 0.0),
        ],
    );
    let mut max_forward: f64 = 0.0;
    let mut max_transpose: f64 = 0.0;
    let mut max_hermitian: f64 = 0.0;
    let mut max_block: f64 = 0.0;
    let kinds = [
        CoupledPreconditionerKind::BlockDiagonal,
        CoupledPreconditionerKind::LowerTriangular,
        CoupledPreconditionerKind::UpperTriangular,
    ];
    for kind in kinds {
        let preconditioner = CoupledTracePreconditioner::try_new_with_kind(
            Box::new(ScalePreconditioner {
                scale: Complex64::new(0.5, 0.1),
            }),
            Box::new(ScalePreconditioner {
                scale: Complex64::new(0.25, -0.2),
            }),
            transfer().map_err(|error| error.to_string())?,
            Complex64::new(0.3, -0.15),
            kind,
        )
        .map_err(|error| error.to_string())?;
        let dense = (0..4)
            .map(|column| {
                let mut basis = vec![Complex64::new(0.0, 0.0); 4];
                basis[column] = Complex64::new(1.0, 0.0);
                preconditioner.apply_checked(&basis).expect("basis action")
            })
            .collect::<Vec<_>>();
        let matrix = DenseMatrix::from_fn(4, 4, |row, column| dense[column][row]);
        max_forward = max_forward.max(max_error(
            &preconditioner
                .apply_checked(&state)
                .map_err(|e| e.to_string())?,
            &matrix.matvec(&state),
        ));
        max_transpose = max_transpose.max(max_error(
            &preconditioner
                .apply_transpose_checked(&state)
                .map_err(|e| e.to_string())?,
            &transpose_matvec(&matrix, &state),
        ));
        max_hermitian = max_hermitian.max(max_error(
            &preconditioner
                .apply_hermitian_checked(&state)
                .map_err(|e| e.to_string())?,
            &hermitian_matvec(&matrix, &state),
        ));

        let mut output = DenseMatrix::zeros(4, 2);
        preconditioner
            .try_apply_block(&block, &mut output)
            .map_err(|e| e.to_string())?;
        let expected = DenseMatrix::from_fn(4, 2, |row, column| {
            matrix.matvec(&block.column_to_vec(column))[row]
        });
        max_block = max_block.max(max_error(output.as_slice(), expected.as_slice()));
    }

    let unknown = CoupledTracePreconditioner::try_new(
        Box::new(ForwardOnly),
        Box::new(ScalePreconditioner {
            scale: Complex64::new(1.0, 0.0),
        }),
        transfer().map_err(|error| error.to_string())?,
        Complex64::new(0.0, 0.0),
    );
    let unknown_rejections = usize::from(matches!(
        unknown,
        Err(CouplingError::UnsupportedAdjoint {
            block: "interior",
            operation: "transpose preconditioner"
        })
    ));

    let valid = CoupledTracePreconditioner::try_new(
        Box::new(ScalePreconditioner {
            scale: Complex64::new(1.0, 0.0),
        }),
        Box::new(ScalePreconditioner {
            scale: Complex64::new(1.0, 0.0),
        }),
        transfer().map_err(|error| error.to_string())?,
        Complex64::new(0.0, 0.0),
    )
    .map_err(|error| error.to_string())?;
    let bad_state_rejections = usize::from(matches!(
        valid.apply_checked(&[Complex64::new(1.0, 0.0)]),
        Err(CouplingError::Dimension { .. })
    ));
    if max_forward > 1.0e-12
        || max_transpose > 1.0e-12
        || max_hermitian > 1.0e-12
        || max_block > 1.0e-12
        || unknown_rejections != 1
        || bad_state_rejections != 1
    {
        return Err(format!(
            "coupled preconditioner contract mismatch: forward={max_forward}, transpose={max_transpose}, hermitian={max_hermitian}, block={max_block}, unknown={unknown_rejections}, bad_state={bad_state_rejections}"
        ));
    }
    Ok((
        kinds.len(),
        max_forward,
        max_transpose,
        max_hermitian,
        max_block,
        unknown_rejections,
        bad_state_rejections,
    ))
}

struct ForwardOnly;

impl Preconditioner<Complex64> for ForwardOnly {
    fn apply(&self, rhs: &[Complex64]) -> Vec<Complex64> {
        rhs.to_vec()
    }
}

fn main() {
    match run() {
        Ok((kinds, forward, transpose, hermitian, block, unknown, bad_state)) => {
            let result = QaResult::passed(CASE_ID)
                .metric("preconditioner_kinds", kinds as f64)
                .metric("max_forward_error", forward)
                .metric("max_transpose_error", transpose)
                .metric("max_hermitian_error", hermitian)
                .metric("max_block_error", block)
                .metric("unknown_adjoint_rejections", unknown as f64)
                .metric("bad_state_rejections", bad_state as f64)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent dense block-diagonal and triangular actions")
                .dimension("2+2 complex FEM--BEM trace DOFs")
                .formulation("checked matrix-free coupled block preconditioner")
                .seed("bem-fem-bem-preconditioner:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("checked matrix-free coupled block preconditioner")
                .dimension("2+2 complex FEM--BEM trace DOFs")
                .formulation("checked matrix-free coupled block preconditioner")
                .seed("bem-fem-bem-preconditioner:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            std::process::exit(1);
        }
    }
}
