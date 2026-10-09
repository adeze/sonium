//! Release-style general-profile axisymmetric transmission contract evidence.

use std::f64::consts::PI;
use std::time::Instant;

use num_complex::Complex64;
use sonium_bem::{
    AxisymmetricBemError, AxisymmetricPenetrableProfile, AxisymmetricPenetrableSphere,
    PhysicsParams,
};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "bem.axisymmetric-profile-transmission.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    sphere_matrix_error: f64,
    sphere_rhs_error: f64,
    deformed_residual: f64,
    pressure_continuity_error: f64,
    velocity_continuity_error: f64,
    adjoint_residual: f64,
    adjoint_batch_residual: f64,
    compressed_action_error: f64,
    compressed_block_count: usize,
    finite_field_count: usize,
    typed_rejections: usize,
}

fn media() -> (PhysicsParams, PhysicsParams) {
    (
        PhysicsParams::new(420.0, 343.0, 1.21, false),
        PhysicsParams::new(420.0, 240.0, 2.4, true),
    )
}

fn run() -> Result<Metrics, String> {
    let count = 8;
    let radius = 0.2;
    let ratio = 0.65;
    let profile = (0..count)
        .map(|index| {
            let theta = PI * (index as f64 + 0.5) / count as f64;
            [radius * theta.sin(), radius * theta.cos()]
        })
        .collect::<Vec<_>>();
    let normals = (0..count)
        .map(|index| {
            let theta = PI * (index as f64 + 0.5) / count as f64;
            [theta.sin(), theta.cos()]
        })
        .collect::<Vec<_>>();
    let (exterior, interior) = media();
    let profile_problem = AxisymmetricPenetrableProfile::try_new(
        profile,
        normals,
        radius * (1.0 - ratio),
        radius * (1.0 / ratio - 1.0),
        32,
        exterior.clone(),
        interior.clone(),
    )
    .map_err(|error| error.to_string())?;
    let sphere_problem = AxisymmetricPenetrableSphere::try_new(
        radius,
        count,
        ratio,
        32,
        exterior.clone(),
        interior.clone(),
    )
    .map_err(|error| error.to_string())?;
    let (profile_matrix, profile_rhs) = profile_problem
        .assemble_system(Complex64::new(1.0, 0.0))
        .map_err(|error| error.to_string())?;
    let (sphere_matrix, sphere_rhs) = sphere_problem
        .assemble_system(Complex64::new(1.0, 0.0))
        .map_err(|error| error.to_string())?;
    let sphere_matrix_error = profile_matrix
        .as_slice()
        .iter()
        .zip(sphere_matrix.as_slice())
        .map(|(actual, reference)| (*actual - *reference).norm())
        .fold(0.0, f64::max);
    let sphere_rhs_error = profile_rhs
        .iter()
        .zip(sphere_rhs.iter())
        .map(|(actual, reference)| (*actual - *reference).norm())
        .fold(0.0, f64::max);

    let deformed_count = 12;
    let (deformed_profile, deformed_normals): (Vec<_>, Vec<_>) = (0..deformed_count)
        .map(|index| {
            let theta = PI * (index as f64 + 0.5) / deformed_count as f64;
            let radial = radius * (1.0 + 0.12 * (2.0 * theta).cos());
            let radial_derivative = -0.24 * radius * (2.0 * theta).sin();
            let axial_derivative = -radius * theta.sin();
            (
                [radial, radius * theta.cos()],
                [-axial_derivative, radial_derivative],
            )
        })
        .unzip();
    let deformed_problem = AxisymmetricPenetrableProfile::try_new(
        deformed_profile,
        deformed_normals,
        0.025,
        0.035,
        32,
        exterior.clone(),
        interior.clone(),
    )
    .map_err(|error| error.to_string())?;
    let (deformed_matrix, _) = deformed_problem
        .assemble_system(Complex64::new(1.0, 0.0))
        .map_err(|error| error.to_string())?;
    let compressed = deformed_problem
        .compress(4, 1.0e-2, 24)
        .map_err(|error| error.to_string())?;
    let trial = (0..deformed_problem.num_dofs())
        .map(|index| Complex64::new(0.13 + index as f64 / 19.0, -0.07 + index as f64 / 31.0))
        .collect::<Vec<_>>();
    let dense_action = (0..deformed_problem.num_dofs())
        .map(|row| {
            (0..deformed_problem.num_dofs())
                .map(|column| deformed_matrix[[row, column]] * trial[column])
                .sum::<Complex64>()
        })
        .collect::<Vec<_>>();
    let compressed_action = sonium_solvers::traits::LinearOperator::apply(&compressed, &trial);
    let action_error = compressed_action
        .iter()
        .zip(&dense_action)
        .map(|(actual, reference)| (*actual - *reference).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let action_scale = dense_action
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(1.0e-15);
    let compressed_action_error = action_error / action_scale;
    let compressed_block_count = compressed.compressed_block_count();
    let solution = deformed_problem
        .solve(Complex64::new(1.0, 0.0), 1.0e-9)
        .map_err(|error| error.to_string())?;
    let pressure_continuity_error = solution
        .pressure_continuity_error()
        .map_err(|error| error.to_string())?;
    let velocity_continuity_error = solution
        .velocity_continuity_error()
        .map_err(|error| error.to_string())?;
    let fields = solution
        .evaluate_exterior_pressure(&[[0.0, 0.8], [0.2, 0.7]])
        .map_err(|error| error.to_string())?;
    let finite_field_count = fields
        .iter()
        .filter(|value| value.re.is_finite() && value.im.is_finite())
        .count();
    let rhs = (0..deformed_problem.num_dofs())
        .map(|index| Complex64::new(0.2 + index as f64 / 31.0, -0.1 + index as f64 / 47.0))
        .collect::<Vec<_>>();
    let adjoint_residual = deformed_problem
        .solve_adjoint(&rhs, 1.0e-9)
        .map_err(|error| error.to_string())?
        .relative_residual();
    let rhs_second = rhs
        .iter()
        .enumerate()
        .map(|(index, value)| *value + Complex64::new(0.11 * index as f64, -0.07))
        .collect::<Vec<_>>();
    let adjoint_batch = deformed_problem
        .solve_adjoint_many(&[rhs.clone(), rhs_second], 1.0e-9)
        .map_err(|error| error.to_string())?;
    if adjoint_batch.len() != 2 {
        return Err(format!(
            "adjoint batch returned {} solutions instead of 2",
            adjoint_batch.len()
        ));
    }
    let adjoint_batch_residual = adjoint_batch
        .iter()
        .map(|solution| solution.relative_residual())
        .fold(0.0, f64::max);

    let mut typed_rejections = 0;
    let mut bad_normals = vec![[1.0, 0.0]; count];
    bad_normals[2] = [f64::NAN, 0.0];
    if matches!(
        AxisymmetricPenetrableProfile::try_new(
            vec![[0.1, 0.0]; count],
            bad_normals,
            0.02,
            0.03,
            32,
            exterior.clone(),
            interior.clone(),
        ),
        Err(AxisymmetricBemError::InvalidProfileNormal { index: 2 })
    ) {
        typed_rejections += 1;
    }
    if solution
        .evaluate_exterior_pressure(&[[f64::NAN, 0.8]])
        .is_err()
    {
        typed_rejections += 1;
    }

    Ok(Metrics {
        sphere_matrix_error,
        sphere_rhs_error,
        deformed_residual: solution.relative_residual(),
        pressure_continuity_error,
        velocity_continuity_error,
        adjoint_residual,
        adjoint_batch_residual,
        compressed_action_error,
        compressed_block_count,
        finite_field_count,
        typed_rejections,
    })
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .metric("sphere_matrix_error", metrics.sphere_matrix_error)
            .metric("sphere_rhs_error", metrics.sphere_rhs_error)
            .metric("deformed_residual", metrics.deformed_residual)
            .metric(
                "pressure_continuity_error",
                metrics.pressure_continuity_error,
            )
            .metric(
                "velocity_continuity_error",
                metrics.velocity_continuity_error,
            )
            .metric("adjoint_residual", metrics.adjoint_residual)
            .metric("adjoint_batch_residual", metrics.adjoint_batch_residual)
            .metric("compressed_action_error", metrics.compressed_action_error)
            .metric(
                "compressed_block_count",
                metrics.compressed_block_count as f64,
            )
            .metric("finite_field_count", metrics.finite_field_count as f64)
            .metric("typed_rejections", metrics.typed_rejections as f64)
            .maximum("sphere_matrix_error", 1.0e-12)
            .maximum("sphere_rhs_error", 1.0e-12)
            .maximum("deformed_residual", 1.0e-9)
            .maximum("pressure_continuity_error", 1.0e-9)
            .maximum("velocity_continuity_error", 1.0e-9)
            .maximum("adjoint_residual", 1.0e-9)
            .maximum("adjoint_batch_residual", 1.0e-9)
            .maximum("compressed_action_error", 2.0e-2)
            .minimum("compressed_block_count", 1.0)
            .minimum("finite_field_count", 2.0)
            .minimum("typed_rejections", 2.0)
            .input_checksum(
                "analytic/deterministic-fixture:bem-axisymmetric-profile-transmission-v1",
            )
            .reference_checksum(
                "analytic/deterministic-fixture:bem-axisymmetric-profile-transmission-v1",
            )
            .reference("sphere matrix parity plus deformed-profile collocation/adjoint invariants")
            .dimension("general axisymmetric meridional profile; 8/12 points; 32 azimuthal samples")
            .formulation("checked CPU two-medium profile ring-source transmission collocation")
            .tolerance("sphere parity <= 1e-12; residual/continuity <= 1e-9")
            .seed("bem-axisymmetric-profile-transmission:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(error) => QaResult::failed(CASE_ID, error)
            .input_checksum(
                "analytic/deterministic-fixture:bem-axisymmetric-profile-transmission-v1",
            )
            .reference_checksum(
                "analytic/deterministic-fixture:bem-axisymmetric-profile-transmission-v1",
            )
            .requested_backend("cpu")
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
