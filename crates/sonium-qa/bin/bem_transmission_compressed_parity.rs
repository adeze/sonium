//! Bounded CPU parity evidence for compressed transmission actions.

use sonium_bem::core::assembly::build_transmission_tbem_system_with_continuous_quadratic;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::types::PhysicsParams;
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::GmresConfig;
use sonium_solvers::traits::LinearOperator;
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-compressed-parity.v1";

fn dense_relative_residual(
    matrix: &DenseMatrix<num_complex::Complex64>,
    values: &[num_complex::Complex64],
    rhs: &[num_complex::Complex64],
    hermitian: bool,
) -> f64 {
    let mut action = vec![num_complex::Complex64::default(); matrix.nrows()];
    if hermitian {
        for row in 0..matrix.nrows() {
            for column in 0..matrix.ncols() {
                action[column] += matrix[[row, column]].conj() * values[row];
            }
        }
    } else {
        for row in 0..matrix.nrows() {
            for column in 0..matrix.ncols() {
                action[row] += matrix[[row, column]] * values[column];
            }
        }
    }
    let residual_squared = action
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>();
    let rhs_squared = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>();
    residual_squared.sqrt() / rhs_squared.sqrt().max(f64::MIN_POSITIVE)
}

fn rank_truncated_far_block_probe() -> Result<(f64, usize), String> {
    let entry = |row: usize, column: usize| {
        let value = match (row, column) {
            (0, 0) | (1, 1) | (2, 2) | (3, 3) => 2.0,
            (0, 2) => 1.0,
            (1, 3) => 0.2,
            (2, 0) => 0.5,
            (3, 1) => 0.1,
            _ => 0.0,
        };
        num_complex::Complex64::new(value, 0.0)
    };
    let compressed = sonium_bem::core::assembly::CompressedTransmissionTbemSystem::from_entry(
        2, 4, 2, 0.75, 1, entry,
    )
    .map_err(|error| format!("rank-truncated far-block probe construction failed: {error}"))?;
    let input = [
        num_complex::Complex64::new(1.0, -0.2),
        num_complex::Complex64::new(-0.5, 0.7),
        num_complex::Complex64::new(0.8, 0.1),
        num_complex::Complex64::new(-1.2, -0.4),
    ];
    let mut dense_action = vec![num_complex::Complex64::default(); 4];
    for row in 0..4 {
        for column in 0..4 {
            dense_action[row] += entry(row, column) * input[column];
        }
    }
    let compressed_action = compressed.apply(&input);
    let error = compressed_action
        .iter()
        .zip(&dense_action)
        .map(|(actual, reference)| (*actual - *reference).norm())
        .fold(0.0, f64::max)
        / dense_action
            .iter()
            .map(|value| value.norm())
            .fold(0.0, f64::max)
            .max(f64::MIN_POSITIVE);
    if !error.is_finite() || compressed.compressed_block_count() < 1 {
        return Err(format!(
            "rank-truncated far-block probe invalid: error={error}, compressed_blocks={}",
            compressed.compressed_block_count()
        ));
    }
    Ok((error, compressed.compressed_block_count()))
}

fn run() -> Result<
    (
        usize,
        usize,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        usize,
        f64,
        f64,
        f64,
        usize,
    ),
    String,
> {
    let mesh = generate_icosphere_mesh(0.2, 1);
    let system = build_transmission_tbem_system_with_continuous_quadratic(
        &mesh.elements,
        &mesh.nodes,
        PhysicsParams::new(420.0, 343.0, 1.21, false),
        PhysicsParams::new(420.0, 1_480.0, 1_000.0, true),
    )
    .map_err(|error| error.to_string())?;
    let compressed = system
        .compress(48, 1.0e-2, 48)
        .map_err(|error| error.to_string())?;
    if compressed.compressed_block_count() == 0 {
        return Err("transmission compression produced no low-rank blocks".to_string());
    }

    let nonfinite_aca_rejection =
        match sonium_bem::core::assembly::CompressedTransmissionTbemSystem::from_entry(
            2,
            4,
            2,
            1.0e-2,
            1,
            |row, column| {
                if row < 2 && column >= 2 {
                    num_complex::Complex64::new(f64::MAX, 0.0)
                } else {
                    num_complex::Complex64::new(1.0 + (row + column) as f64, 0.0)
                }
            },
        ) {
            Err(sonium_bem::TransmissionCompressionError::NonFiniteBlock { .. }) => 1.0,
            Err(error) => {
                return Err(format!(
                    "finite ACA overflow probe returned an unexpected error: {error}"
                ));
            }
            Ok(_) => {
                return Err(
                    "finite ACA overflow probe unexpectedly accepted a non-finite block error"
                        .to_string(),
                );
            }
        };

    let nonfinite_action_output_rejection = {
        let overflow_operator =
            sonium_bem::core::assembly::CompressedTransmissionTbemSystem::from_entry(
                2,
                4,
                2,
                1.0e-2,
                2,
                |_row, _column| num_complex::Complex64::new(2.0, 0.0),
            )
            .map_err(|error| {
                format!("finite overflow-output probe construction failed: {error}")
            })?;
        let overflow_input = vec![num_complex::Complex64::new(f64::MAX, 0.0); 4];
        match overflow_operator.apply_checked(&overflow_input) {
            Err(sonium_bem::TransmissionCompressionError::NonFiniteOutput {
                operation: "forward",
                ..
            }) => 1.0,
            Err(error) => {
                return Err(format!(
                    "finite overflow-output probe returned an unexpected error: {error}"
                ));
            }
            Ok(_) => {
                return Err(
                    "finite overflow-output probe unexpectedly accepted a non-finite action"
                        .to_string(),
                );
            }
        }
    };

    let dimension = system.total_dofs();
    let trial = (0..dimension)
        .map(|index| {
            num_complex::Complex64::new(-0.13 + index as f64 / 47.0, 0.07 - index as f64 / 61.0)
        })
        .collect::<Vec<_>>();
    let on_demand = sonium_bem::core::assembly::CompressedTransmissionTbemSystem::from_entry(
        system.interface_dofs(),
        dimension,
        48,
        1.0e-2,
        48,
        |row, column| system.matrix()[[row, column]],
    )
    .map_err(|error| error.to_string())?;
    let dense_action = (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| system.matrix()[[row, column]] * trial[column])
                .sum::<num_complex::Complex64>()
        })
        .collect::<Vec<_>>();
    let compressed_action = compressed.apply(&trial);
    let on_demand_action = on_demand.apply(&trial);
    let forward_error = compressed_action
        .iter()
        .zip(&dense_action)
        .map(|(actual, reference)| (*actual - *reference).norm())
        .fold(0.0, f64::max)
        / dense_action
            .iter()
            .map(|value| value.norm())
            .fold(0.0, f64::max)
            .max(f64::MIN_POSITIVE);
    let on_demand_action_error = on_demand_action
        .iter()
        .zip(&compressed_action)
        .map(|(actual, reference)| (*actual - *reference).norm())
        .fold(0.0, f64::max)
        / compressed_action
            .iter()
            .map(|value| value.norm())
            .fold(0.0, f64::max)
            .max(f64::MIN_POSITIVE);

    let dense_adjoint = (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| system.matrix()[[column, row]].conj() * trial[column])
                .sum::<num_complex::Complex64>()
        })
        .collect::<Vec<_>>();
    let compressed_adjoint = compressed.apply_hermitian(&trial);
    let adjoint_error = compressed_adjoint
        .iter()
        .zip(&dense_adjoint)
        .map(|(actual, reference)| (*actual - *reference).norm())
        .fold(0.0, f64::max)
        / dense_adjoint
            .iter()
            .map(|value| value.norm())
            .fold(0.0, f64::max)
            .max(f64::MIN_POSITIVE);
    let dense_transpose = (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| system.matrix()[[column, row]] * trial[column])
                .sum::<num_complex::Complex64>()
        })
        .collect::<Vec<_>>();
    let compressed_transpose = compressed.apply_transpose(&trial);
    let transpose_error = compressed_transpose
        .iter()
        .zip(&dense_transpose)
        .map(|(actual, reference)| (*actual - *reference).norm())
        .fold(0.0, f64::max)
        / dense_transpose
            .iter()
            .map(|value| value.norm())
            .fold(0.0, f64::max)
            .max(f64::MIN_POSITIVE);
    if !forward_error.is_finite() || !adjoint_error.is_finite() || !transpose_error.is_finite() {
        return Err("compressed transmission action was non-finite".to_string());
    }
    let config = GmresConfig::<f64> {
        max_iterations: 120,
        restart: 32,
        tolerance: 1.0e-10,
        exact_residual_check_interval: 1,
        ..GmresConfig::default()
    };
    let mut nearfield = DenseMatrix::zeros(dimension, dimension);
    for index in 0..dimension {
        nearfield[[index, index]] = num_complex::Complex64::new(1.0, 0.0);
    }
    let forward_solution = compressed
        .solve_with_ilu(&trial, &nearfield, &config, 1.0e-8)
        .map_err(|error| error.to_string())?;
    let second_rhs = trial
        .iter()
        .enumerate()
        .map(|(index, value)| *value + num_complex::Complex64::new(0.05 * index as f64, -0.02))
        .collect::<Vec<_>>();
    let forward_batch = compressed
        .solve_many_with_ilu(&[trial.clone(), second_rhs], &nearfield, &config, 1.0e-8)
        .map_err(|error| error.to_string())?;
    let adjoint_solution = compressed
        .solve_hermitian_with_ilu(&trial, &nearfield, &config, 1.0e-8)
        .map_err(|error| error.to_string())?;
    let exact_forward_residual =
        dense_relative_residual(system.matrix(), forward_solution.values(), &trial, false);
    let exact_adjoint_residual =
        dense_relative_residual(system.matrix(), adjoint_solution.values(), &trial, true);
    if !exact_forward_residual.is_finite()
        || !exact_adjoint_residual.is_finite()
        || (exact_forward_residual - forward_solution.relative_residual()).abs() > 1.0e-12
        || (exact_adjoint_residual - adjoint_solution.relative_residual()).abs() > 1.0e-12
    {
        return Err(format!(
            "exact residual sentinel mismatch: forward={exact_forward_residual:.3e}/{:.3e}, adjoint={exact_adjoint_residual:.3e}/{:.3e}",
            forward_solution.relative_residual(),
            adjoint_solution.relative_residual(),
        ));
    }
    let adjoint_batch = compressed
        .solve_hermitian_many_with_ilu(
            &[trial.clone(), forward_solution.values().to_vec()],
            &nearfield,
            &config,
            1.0e-8,
        )
        .map_err(|error| error.to_string())?;
    let (rank_truncated_error, rank_truncated_blocks) = rank_truncated_far_block_probe()?;
    Ok((
        dimension,
        compressed.compressed_block_count(),
        forward_error,
        adjoint_error,
        transpose_error,
        on_demand_action_error,
        compressed.maximum_block_relative_error(),
        forward_solution.relative_residual(),
        adjoint_solution.relative_residual(),
        exact_forward_residual,
        exact_adjoint_residual,
        forward_batch
            .iter()
            .chain(adjoint_batch.iter())
            .map(|solution| solution.relative_residual())
            .fold(0.0, f64::max),
        adjoint_batch.len(),
        nonfinite_aca_rejection,
        nonfinite_action_output_rejection,
        rank_truncated_error,
        rank_truncated_blocks,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((
            dimension,
            compressed_blocks,
            forward_error,
            adjoint_error,
            transpose_error,
            on_demand_action_error,
            maximum_block_error,
            forward_solve_residual,
            adjoint_solve_residual,
            exact_forward_residual,
            exact_adjoint_residual,
            batch_solve_residual,
            adjoint_batch_count,
            nonfinite_aca_rejection,
            nonfinite_action_output_rejection,
            rank_truncated_error,
            rank_truncated_blocks,
        )) => QaResult::passed(CASE_ID)
            .metric("dimension", dimension as f64)
            .metric("compressed_blocks", compressed_blocks as f64)
            .metric("forward_action_error", forward_error)
            .metric("adjoint_action_error", adjoint_error)
            .metric("transpose_action_error", transpose_error)
            .metric("on_demand_action_error", on_demand_action_error)
            .metric("maximum_block_relative_error", maximum_block_error)
            .metric("forward_solve_residual", forward_solve_residual)
            .metric("adjoint_solve_residual", adjoint_solve_residual)
            .metric("exact_forward_source_residual", exact_forward_residual)
            .metric("exact_adjoint_source_residual", exact_adjoint_residual)
            .metric("batch_solve_residual", batch_solve_residual)
            .metric("adjoint_batch_count", adjoint_batch_count as f64)
            .metric("nonfinite_aca_rejection", nonfinite_aca_rejection)
            .metric(
                "nonfinite_action_output_rejection",
                nonfinite_action_output_rejection,
            )
            .metric("rank_truncated_far_block_error", rank_truncated_error)
            .metric("rank_truncated_far_block_count", rank_truncated_blocks as f64)
            .maximum("forward_action_error", 2.0e-2)
            .maximum("adjoint_action_error", 2.0e-2)
            .maximum("transpose_action_error", 2.0e-2)
            .maximum("on_demand_action_error", 1.0e-12)
            .maximum("forward_solve_residual", 1.0e-8)
            .maximum("adjoint_solve_residual", 1.0e-8)
            .maximum("exact_forward_source_residual", 1.0e-8)
            .maximum("exact_adjoint_source_residual", 1.0e-8)
            .maximum("batch_solve_residual", 1.0e-8)
            .minimum("compressed_blocks", 1.0)
            .minimum("adjoint_batch_count", 2.0)
            .minimum("nonfinite_aca_rejection", 1.0)
            .minimum("nonfinite_action_output_rejection", 1.0)
            .maximum("rank_truncated_far_block_error", 0.75)
            .minimum("rank_truncated_far_block_count", 1.0)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .input_checksum("bem:icosphere:subdivision-1:continuous-quadratic-transmission")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-transmission-compressed-parity-v1",
            )
            .tolerance(
                "action errors <= 2e-2; compressed solve residuals <= 1e-8; adjoint batch count = 2",
            )
            .dimension("3-D CPU dense-source block-ACA transmission operator")
            .seed("bem-transmission-compressed-parity:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(message) => QaResult::failed(CASE_ID, message)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .input_checksum("bem:icosphere:subdivision-1:continuous-quadratic-transmission")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-transmission-compressed-parity-v1",
            )
            .tolerance(
                "action errors <= 2e-2; compressed solve residuals <= 1e-8; adjoint batch count = 2",
            )
            .dimension("3-D CPU dense-source block-ACA transmission operator")
            .seed("bem-transmission-compressed-parity:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
