//! Checked compression accuracy and forward/adjoint field convergence.

// Rust guideline compliant 2026-02-21

use num_complex::Complex64;
use sonium_bem::core::assembly::{
    CompressedTransmissionTbemSystem, build_transmission_tbem_system_with_continuous_quadratic,
};
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::solver::transmission::{
    solve_incident_transmission, solve_transmission_adjoint_system,
};
use sonium_bem::core::types::PhysicsParams;
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::GmresConfig;
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-compressed-field-parity.v1";
// Keep algebraic solve accuracy fixed while refining the approximation.
const SOLVER_TOLERANCE: f64 = 1.0e-10;
// The corrected weighted kernel needs a fourth approximation level to
// resolve the independently measured dense adjoint residual below 1e-5.
const COMPRESSION_TOLERANCES: [f64; 4] = [1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8];
// The fine operator budget matches ACA's requested approximation accuracy.
// Field and dense-residual budgets allow ten times that error for conditioning.
const FINE_ACTION_TOLERANCE: f64 = 1.0e-6;
const FINE_FIELD_TOLERANCE: f64 = 1.0e-5;
// Errors below this floor are indistinguishable from roundoff in field sums.
const REFINEMENT_FLOOR: f64 = 1.0e-12;

fn relative_l2(actual: &[Complex64], reference: &[Complex64]) -> f64 {
    if actual.len() != reference.len() || actual.is_empty() {
        return f64::INFINITY;
    }
    let numerator = actual
        .iter()
        .zip(reference)
        .map(|(value, expected)| (*value - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = reference
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    numerator / denominator
}

fn matrix_action(
    matrix: &DenseMatrix<Complex64>,
    input: &[Complex64],
    adjoint: bool,
) -> Vec<Complex64> {
    (0..input.len())
        .map(|row| {
            (0..input.len())
                .map(|column| {
                    let entry = if adjoint {
                        matrix[[column, row]].conj()
                    } else {
                        matrix[[row, column]]
                    };
                    entry * input[column]
                })
                .sum()
        })
        .collect()
}

fn contraction(previous: f64, next: f64) -> f64 {
    (next - REFINEMENT_FLOOR).max(0.0) / (previous - REFINEMENT_FLOOR).max(REFINEMENT_FLOOR)
}

#[derive(Debug)]
struct Level {
    forward_error: f64,
    adjoint_error: f64,
    action_error: f64,
    adjoint_action_error: f64,
    forward_residual: f64,
    adjoint_residual: f64,
    dense_forward_residual: f64,
    dense_adjoint_residual: f64,
    compressed_blocks: usize,
}

fn run() -> Result<QaResult, String> {
    let mesh = generate_icosphere_mesh(0.2, 1);
    let exterior = PhysicsParams::new(42.0, 343.0, 1.21, false);
    let interior = PhysicsParams::new(42.0, 240.0, 2.4, true);
    let system = build_transmission_tbem_system_with_continuous_quadratic(
        &mesh.elements,
        &mesh.nodes,
        exterior,
        interior,
    )
    .map_err(|error| error.to_string())?;
    let dimension = system.total_dofs();
    let incident = IncidentField::plane_wave_z();
    let rhs = system
        .incident_rhs(&incident)
        .map_err(|error| error.to_string())?;
    let adjoint_rhs = (0..dimension)
        .map(|index| Complex64::new((0.31 * index as f64).sin(), (0.47 * index as f64).cos()))
        .collect::<Vec<_>>();
    let dense_forward = solve_incident_transmission(&system, &incident, SOLVER_TOLERANCE)
        .map_err(|error| error.to_string())?;
    let dense_adjoint = solve_transmission_adjoint_system(&system, &adjoint_rhs, SOLVER_TOLERANCE)
        .map_err(|error| error.to_string())?;
    let points = [0.3_f64, 0.9, 1.6, 2.4]
        .into_iter()
        .flat_map(|theta| {
            [0.21, 0.8].map(|radius| [radius * theta.sin(), 0.0, radius * theta.cos()])
        })
        .collect::<Vec<_>>();
    let dense_pressure = dense_forward
        .evaluate_exterior_field_checked(&system, &points, &incident)
        .map_err(|error| error.to_string())?
        .iter()
        .map(|field| field.p_total)
        .collect::<Vec<_>>();
    let dense_adjoint_pressure = dense_adjoint
        .evaluate_exterior_adjoint_pressure_checked(&system, &points)
        .map_err(|error| error.to_string())?;
    let mut nearfield = system.matrix().clone();
    let mut adjoint_nearfield = DenseMatrix::zeros(dimension, dimension);
    for row in 0..dimension {
        nearfield[[row, row]] += Complex64::new(1.0, 0.0);
        for column in 0..dimension {
            adjoint_nearfield[[row, column]] = system.matrix()[[column, row]].conj();
        }
        adjoint_nearfield[[row, row]] += Complex64::new(1.0, 0.0);
    }
    let config = GmresConfig::<f64> {
        max_iterations: 400,
        restart: 64,
        tolerance: SOLVER_TOLERANCE,
        exact_residual_check_interval: 1,
        ..GmresConfig::default()
    };
    let mut levels = Vec::new();
    let mut record = QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("dimension", dimension as f64)
        .metric("solver_tolerance", SOLVER_TOLERANCE)
        .maximum("solver_tolerance", SOLVER_TOLERANCE);
    for (index, tolerance) in COMPRESSION_TOLERANCES.into_iter().enumerate() {
        // The entry constructor retains no exact source matrix. Its bounded
        // direct fallback therefore solves the approximation, too. Supply
        // entries from the same dense source to isolate compression error.
        let compressed = CompressedTransmissionTbemSystem::from_entry(
            system.interface_dofs(),
            dimension,
            32,
            tolerance,
            32,
            |row, column| system.matrix()[[row, column]],
        )
        .map_err(|error| error.to_string())?;
        let forward = compressed
            .solve_with_ilu(&rhs, &nearfield, &config, SOLVER_TOLERANCE)
            .map_err(|error| error.to_string())?;
        let adjoint = compressed
            .solve_hermitian_with_ilu(&adjoint_rhs, &adjoint_nearfield, &config, SOLVER_TOLERANCE)
            .map_err(|error| error.to_string())?;
        let pressure = forward
            .evaluate_exterior_field_checked(&system, &points, &incident)
            .map_err(|error| error.to_string())?
            .iter()
            .map(|field| field.p_total)
            .collect::<Vec<_>>();
        let adjoint_pressure = adjoint
            .evaluate_exterior_adjoint_pressure_checked(&system, &points)
            .map_err(|error| error.to_string())?;
        let mut action_error: f64 = 0.0;
        let mut adjoint_action_error: f64 = 0.0;
        for probe in 0..3 {
            let input = (0..dimension)
                .map(|entry| {
                    Complex64::new(
                        ((entry + 1) as f64 * (0.13 + probe as f64)).sin(),
                        ((entry + 1) as f64 * (0.27 + probe as f64)).cos(),
                    )
                })
                .collect::<Vec<_>>();
            action_error = action_error.max(relative_l2(
                &compressed
                    .apply_checked(&input)
                    .map_err(|error| error.to_string())?,
                &matrix_action(system.matrix(), &input, false),
            ));
            adjoint_action_error = adjoint_action_error.max(relative_l2(
                &compressed
                    .apply_hermitian_checked(&input)
                    .map_err(|error| error.to_string())?,
                &matrix_action(system.matrix(), &input, true),
            ));
        }
        if forward
            .evaluate_exterior_field_checked(&system, &[[f64::NAN, 0.0, 0.8]], &incident)
            .is_ok()
            || adjoint
                .evaluate_exterior_adjoint_pressure_checked(&system, &[[f64::NAN, 0.0, 0.8]])
                .is_ok()
        {
            return Err("compressed field accepted a non-finite point".into());
        }
        let level = Level {
            forward_error: relative_l2(&pressure, &dense_pressure),
            adjoint_error: relative_l2(&adjoint_pressure, &dense_adjoint_pressure),
            action_error,
            adjoint_action_error,
            forward_residual: relative_l2(
                &compressed
                    .apply_checked(forward.values())
                    .map_err(|error| error.to_string())?,
                &rhs,
            ),
            adjoint_residual: relative_l2(
                &compressed
                    .apply_hermitian_checked(adjoint.values())
                    .map_err(|error| error.to_string())?,
                &adjoint_rhs,
            ),
            dense_forward_residual: relative_l2(
                &matrix_action(system.matrix(), forward.values(), false),
                &rhs,
            ),
            dense_adjoint_residual: relative_l2(
                &matrix_action(system.matrix(), adjoint.values(), true),
                &adjoint_rhs,
            ),
            compressed_blocks: compressed.compressed_block_count(),
        };
        println!("compression tolerance {tolerance:.1e}: {level:?}");
        record = record
            .metric(format!("level_{index}_compression_tolerance"), tolerance)
            .metric(
                format!("level_{index}_forward_field_error"),
                level.forward_error,
            )
            .metric(
                format!("level_{index}_adjoint_field_error"),
                level.adjoint_error,
            )
            .metric(format!("level_{index}_action_error"), level.action_error)
            .metric(
                format!("level_{index}_adjoint_action_error"),
                level.adjoint_action_error,
            )
            .metric(
                format!("level_{index}_compressed_blocks"),
                level.compressed_blocks as f64,
            )
            .minimum(format!("level_{index}_compressed_blocks"), 1.0)
            .metric(
                format!("level_{index}_forward_residual"),
                level.forward_residual,
            )
            .maximum(format!("level_{index}_forward_residual"), SOLVER_TOLERANCE)
            .metric(
                format!("level_{index}_adjoint_residual"),
                level.adjoint_residual,
            )
            .maximum(format!("level_{index}_adjoint_residual"), SOLVER_TOLERANCE)
            .metric(
                format!("level_{index}_dense_forward_residual"),
                level.dense_forward_residual,
            )
            .metric(
                format!("level_{index}_dense_adjoint_residual"),
                level.dense_adjoint_residual,
            );
        levels.push(level);
    }
    let fine = levels.last().expect("compression ladder is nonempty");
    let mut maximum_contraction: f64 = 0.0;
    for pair in levels.windows(2) {
        for (previous, next) in [
            (pair[0].forward_error, pair[1].forward_error),
            (pair[0].adjoint_error, pair[1].adjoint_error),
            (pair[0].action_error, pair[1].action_error),
            (pair[0].adjoint_action_error, pair[1].adjoint_action_error),
        ] {
            maximum_contraction = maximum_contraction.max(contraction(previous, next));
        }
    }
    let record = record.metric("fine_forward_field_error", fine.forward_error).maximum("fine_forward_field_error", FINE_FIELD_TOLERANCE)
        .metric("fine_adjoint_field_error", fine.adjoint_error).maximum("fine_adjoint_field_error", FINE_FIELD_TOLERANCE)
        .metric("fine_action_error", fine.action_error).maximum("fine_action_error", FINE_ACTION_TOLERANCE)
        .metric("fine_adjoint_action_error", fine.adjoint_action_error).maximum("fine_adjoint_action_error", FINE_ACTION_TOLERANCE)
        .metric("fine_dense_forward_residual", fine.dense_forward_residual).maximum("fine_dense_forward_residual", FINE_FIELD_TOLERANCE)
        .metric("fine_dense_adjoint_residual", fine.dense_adjoint_residual).maximum("fine_dense_adjoint_residual", FINE_FIELD_TOLERANCE)
        .metric("maximum_refinement_ratio", maximum_contraction).maximum("maximum_refinement_ratio", 0.25)
        .metric("coarse_action_error", levels[0].action_error).minimum("coarse_action_error", 1.0e-8)
        .metric("numerical_solve_executed", (2 + 2 * levels.len()) as f64).minimum("numerical_solve_executed", 10.0)
        .input_checksum("bem:icosphere:subdivision-1:continuous-quadratic:fluid-contrast:aca-1e-2-1e-4-1e-6-1e-8")
        .reference("independent dense source operator and forward/Hermitian solutions on the identical mesh")
        .reference_checksum("analytic/deterministic-fixture:bem-transmission-compressed-field-parity-v1")
        .dimension("3-D CPU continuous-quadratic 80-panel transmission; eight near/far field points")
        .formulation("entry-based block-ACA without retained exact source; independent dense action/residuals; four approximation levels with fixed algebraic tolerance")
        .tolerance("solver residuals <=1e-10; fine fields/dense residuals <=1e-5; fine operator actions <=1e-6; every adjacent field/action contraction <=0.25 above roundoff floor 1e-12")
        .seed("bem-transmission-compressed-field-parity:cpu:v1");
    Ok(record)
}

fn main() {
    let started = Instant::now();
    let mut result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    }
    .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
    if matches!(result.status, QaStatus::Passed)
        && let Err(error) = result.release_eligible()
    {
        result.status = QaStatus::Failed;
        result.message = Some(error);
    }
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
