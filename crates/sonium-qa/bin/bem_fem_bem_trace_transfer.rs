//! Deterministic FEM–BEM interface-transfer contract evidence.

use num_complex::Complex64;
use sonium_bem::core::coupling::{ConservativeTraceTransfer, CoupledTraceSystem, CouplingError};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "bem.fem-bem-trace-transfer.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    source_count: usize,
    target_count: usize,
    constant_error: f64,
    block_error: f64,
    adjoint_error: f64,
    adjoint_block_error: f64,
    coupled_dense_error: f64,
    coupled_block_error: f64,
    coupled_adjoint_error: f64,
    coupled_adjoint_block_error: f64,
    coupled_residual: f64,
    residual_mismatch: f64,
    typed_rejections: usize,
}

fn run() -> Result<Metrics, String> {
    let faces = [[10, 20, 30], [20, 30, 40]];
    let measures = [1.0, 2.0];
    let (transfer, source_nodes) =
        ConservativeTraceTransfer::try_from_tri3_vertex_faces(&faces, &measures)
            .map_err(|error| format!("valid Tri3 transfer rejected: {error}"))?;

    let source = vec![Complex64::new(2.0, -1.0); source_nodes.len()];
    let target = transfer
        .apply(&source)
        .map_err(|error| format!("constant trace transfer failed: {error}"))?;
    let constant_error = target
        .iter()
        .map(|value| (*value - source[0]).norm())
        .fold(0.0, f64::max);

    let source_block = DenseMatrix::from_fn(source_nodes.len(), 2, |row, column| {
        Complex64::new(
            (row + 1 + 3 * column) as f64,
            (column as f64) - 0.25 * (row as f64),
        )
    });
    let mapped_block = transfer
        .apply_block(&source_block)
        .map_err(|error| format!("block trace transfer failed: {error}"))?;
    let block_error = (0..source_block.ncols())
        .map(|column| {
            let expected = transfer
                .apply(&source_block.column_to_vec(column))
                .expect("validated column transfer");
            mapped_block
                .column_to_vec(column)
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (*actual - expected).norm())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    let adjoint_error = transfer
        .check_adjoint(
            &[
                Complex64::new(1.0, 2.0),
                Complex64::new(-3.0, 1.0),
                Complex64::new(0.25, -0.5),
                Complex64::new(2.0, 0.75),
            ],
            &[Complex64::new(2.0, -1.0), Complex64::new(-0.5, 0.25)],
        )
        .map_err(|error| format!("weighted adjoint check failed: {error}"))?;
    let target_block = DenseMatrix::from_fn(2, 2, |row, column| {
        Complex64::new((row + 2 * column + 1) as f64, -0.5 * column as f64)
    });
    let adjoint_block = transfer
        .apply_adjoint_block(&target_block)
        .map_err(|error| format!("block adjoint transfer failed: {error}"))?;
    let adjoint_block_error = (0..target_block.ncols())
        .map(|column| {
            let expected = transfer
                .apply_adjoint(&target_block.column_to_vec(column))
                .expect("validated column adjoint transfer");
            adjoint_block
                .column_to_vec(column)
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (*actual - expected).norm())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);

    let coupled = CoupledTraceSystem::try_new(
        DenseMatrix::from_fn(source_nodes.len(), source_nodes.len(), |row, column| {
            Complex64::new(
                if row == column { 2.0 } else { 0.1 },
                0.05 * (row + column) as f64,
            )
        }),
        DenseMatrix::from_fn(target.len(), target.len(), |row, column| {
            Complex64::new(
                if row == column { 1.5 } else { 0.2 },
                -0.03 * (row + column) as f64,
            )
        }),
        transfer.clone(),
        Complex64::new(0.25, -0.1),
    )
    .map_err(|error| format!("coupled trace system rejected: {error}"))?;
    let coupled_state =
        DenseMatrix::from_fn(source_nodes.len() + target.len(), 2, |row, column| {
            Complex64::new((row + 1 + 2 * column) as f64, -0.1 * (row + column) as f64)
        });
    let coupled_block = coupled
        .apply_block(&coupled_state)
        .map_err(|error| format!("coupled block action failed: {error}"))?;
    let coupled_block_error = (0..coupled_state.ncols())
        .map(|column| {
            let expected = coupled
                .apply(&coupled_state.column_to_vec(column))
                .expect("validated coupled column");
            coupled_block
                .column_to_vec(column)
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (*actual - expected).norm())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    let coupled_matrix = coupled
        .to_dense_matrix()
        .map_err(|error| format!("coupled dense materialization failed: {error}"))?;
    let coupled_probe = coupled_state.column_to_vec(0);
    let coupled_action = coupled
        .apply(&coupled_probe)
        .map_err(|error| format!("coupled action failed: {error}"))?;
    let coupled_dense_action = (0..coupled_matrix.nrows())
        .map(|row| {
            (0..coupled_matrix.ncols())
                .map(|column| coupled_matrix[[row, column]] * coupled_probe[column])
                .sum::<Complex64>()
        })
        .collect::<Vec<_>>();
    let coupled_dense_error = coupled_action
        .iter()
        .zip(&coupled_dense_action)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max);
    let coupled_residual = coupled
        .relative_residual(&coupled_probe, &coupled_action)
        .map_err(|error| format!("coupled residual failed: {error}"))?;
    let coupled_adjoint = coupled
        .apply_adjoint(&coupled_probe)
        .map_err(|error| format!("coupled adjoint action failed: {error}"))?;
    let coupled_adjoint_expected = (0..coupled_matrix.ncols())
        .map(|column| {
            (0..coupled_matrix.nrows())
                .map(|row| coupled_matrix[[row, column]].conj() * coupled_probe[row])
                .sum::<Complex64>()
        })
        .collect::<Vec<_>>();
    let coupled_adjoint_error = coupled_adjoint
        .iter()
        .zip(&coupled_adjoint_expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max);
    let coupled_adjoint_block = coupled
        .apply_adjoint_block(&coupled_state)
        .map_err(|error| format!("coupled adjoint block action failed: {error}"))?;
    let coupled_adjoint_block_error = (0..coupled_state.ncols())
        .map(|column| {
            let expected = coupled
                .apply_adjoint(&coupled_state.column_to_vec(column))
                .expect("validated coupled adjoint column");
            coupled_adjoint_block
                .column_to_vec(column)
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (*actual - expected).norm())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    let report = transfer
        .residual_report(
            &[Complex64::new(0.0, 0.0); 4],
            &[Complex64::new(1.0, 0.0); 4],
            &[Complex64::new(0.0, 0.0); 2],
            &[Complex64::new(1.0, 0.0); 2],
            &source,
            &target,
        )
        .map_err(|error| format!("residual report failed: {error}"))?;

    let mut typed_rejections = 0;
    if matches!(
        ConservativeTraceTransfer::try_from_tri3_vertex_faces(&[[1, 1, 2]], &[1.0]),
        Err(CouplingError::InvalidRow { .. })
    ) {
        typed_rejections += 1;
    }
    if matches!(
        transfer.apply(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(f64::NAN, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]),
        Err(CouplingError::NonFinite { .. })
    ) {
        typed_rejections += 1;
    }
    if typed_rejections != 2 {
        return Err(format!(
            "expected two typed rejections, got {typed_rejections}"
        ));
    }

    Ok(Metrics {
        source_count: source_nodes.len(),
        target_count: target.len(),
        constant_error,
        block_error,
        adjoint_error,
        adjoint_block_error,
        coupled_dense_error,
        coupled_block_error,
        coupled_adjoint_error,
        coupled_adjoint_block_error,
        coupled_residual,
        residual_mismatch: report.trace_relative_mismatch,
        typed_rejections,
    })
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("source_count", metrics.source_count as f64)
            .metric("target_count", metrics.target_count as f64)
            .metric("constant_error", metrics.constant_error)
            .metric("block_error", metrics.block_error)
            .metric("adjoint_error", metrics.adjoint_error)
            .metric("adjoint_block_error", metrics.adjoint_block_error)
            .metric("coupled_dense_error", metrics.coupled_dense_error)
            .metric("coupled_block_error", metrics.coupled_block_error)
            .metric(
                "coupled_adjoint_error",
                metrics.coupled_adjoint_error,
            )
            .metric(
                "coupled_adjoint_block_error",
                metrics.coupled_adjoint_block_error,
            )
            .metric("coupled_residual", metrics.coupled_residual)
            .metric("residual_mismatch", metrics.residual_mismatch)
            .metric("typed_rejections", metrics.typed_rejections as f64)
            .minimum("source_count", 4.0)
            .minimum("target_count", 2.0)
            .maximum("constant_error", 1.0e-14)
            .maximum("block_error", 1.0e-14)
            .maximum("adjoint_error", 1.0e-14)
            .maximum("adjoint_block_error", 1.0e-14)
            .maximum("coupled_dense_error", 1.0e-14)
            .maximum("coupled_block_error", 1.0e-14)
            .maximum("coupled_adjoint_error", 1.0e-14)
            .maximum("coupled_adjoint_block_error", 1.0e-14)
            .maximum("coupled_residual", 1.0e-14)
            .maximum("residual_mismatch", 1.0e-14)
            .minimum("typed_rejections", 2.0)
            .input_checksum("analytic/deterministic-fixture:bem-fem-bem-trace-transfer-v1")
            .reference("Tri3 vertex-face interpolation with lumped-measure conservation and weighted Hermitian adjoint")
            .reference_checksum("analytic/deterministic-fixture:bem-fem-bem-trace-transfer-v1")
            .dimension("compact four-vertex, two-face FEM-to-BEM trace map")
            .formulation("checked ConservativeTraceTransfer boundary contract")
        .tolerance(
            "constant/adjoint/block/coupled errors <= 1e-14; two typed rejections",
        )
            .seed("bem-fem-bem-trace-transfer:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("QA result must serialize")
    );
    if matches!(&result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
