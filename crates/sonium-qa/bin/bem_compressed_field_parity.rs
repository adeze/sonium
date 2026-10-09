//! Release-backed common rigid-sphere field parity across compressed BEM backends.

use num_complex::Complex64;
use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, SolverMethod};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.compressed-field-parity.v1";
const BACKENDS: [AssemblyMethod; 4] = [
    AssemblyMethod::Hmatrix,
    AssemblyMethod::Slfmm,
    AssemblyMethod::Mlfmm,
    AssemblyMethod::DirectionalMlfmm,
];
const POINTS: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.24],
    [0.12, 0.04, 0.22],
    [-0.16, 0.05, 0.28],
    [0.08, -0.14, 0.26],
];

#[derive(Debug, Clone, Copy)]
struct Metrics {
    dimension: usize,
    point_count: usize,
    backend_count: usize,
    max_relative_field_error: f64,
    checked_backend_count: usize,
    invalid_point_rejection_count: usize,
}

fn relative_l2(actual: &[Complex64], reference: &[Complex64]) -> f64 {
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
        .max(1.0e-12);
    numerator / denominator
}

fn run() -> Result<Metrics, String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let dense = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct)
        .solve(&problem)
        .map_err(|error| error.to_string())?;
    let dense_field = POINTS
        .iter()
        .map(|point| dense.evaluate_pressure(point))
        .collect::<Vec<_>>();
    let dimension = dense.num_dofs();
    let mut max_relative_field_error = 0.0_f64;
    let mut checked_backend_count = 0;
    let mut invalid_point_rejection_count = 0;
    for assembly in BACKENDS {
        let solver = BemSolver::new()
            .with_assembly_method(assembly)
            .with_solver_method(SolverMethod::Gmres)
            .with_tolerance(1.0e-7)
            .with_compressed_retries(0);
        let solution = solver
            .solve(&problem)
            .map_err(|error| format!("{assembly:?} solve: {error}"))?;
        let field = POINTS
            .iter()
            .map(|point| solution.evaluate_pressure(point))
            .collect::<Vec<_>>();
        let checked = solution
            .evaluate_pressure_field_checked(&POINTS)
            .map_err(|error| format!("{assembly:?} checked field: {error}"))?;
        if checked.len() != POINTS.len()
            || checked
                .iter()
                .any(|point| !point.p_total.re.is_finite() || !point.p_total.im.is_finite())
        {
            return Err(format!(
                "{assembly:?} checked field produced invalid values"
            ));
        }
        checked_backend_count += 1;
        if solution
            .evaluate_pressure_field_checked(&[[f64::NAN, 0.0, 0.24]])
            .is_err()
        {
            invalid_point_rejection_count += 1;
        }
        let error = relative_l2(&field, &dense_field);
        if !error.is_finite() || error > 5.0e-2 {
            return Err(format!(
                "{assembly:?} field parity error {error:.3e} exceeds 5%"
            ));
        }
        max_relative_field_error = max_relative_field_error.max(error);
    }
    Ok(Metrics {
        dimension,
        point_count: POINTS.len(),
        backend_count: BACKENDS.len(),
        max_relative_field_error,
        checked_backend_count,
        invalid_point_rejection_count,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("dimension", metrics.dimension as f64)
            .metric("point_count", metrics.point_count as f64)
                .metric("backend_count", metrics.backend_count as f64)
                .metric("max_relative_field_error", metrics.max_relative_field_error)
                .metric("checked_backend_count", metrics.checked_backend_count as f64)
                .metric(
                    "invalid_point_rejection_count",
                    metrics.invalid_point_rejection_count as f64,
                )
            .minimum("dimension", 1.0)
            .minimum("point_count", 4.0)
            .minimum("backend_count", 4.0)
            .minimum("checked_backend_count", 4.0)
            .minimum("invalid_point_rejection_count", 4.0)
                .maximum("max_relative_field_error", 5.0e-2)
            .input_checksum("analytic/deterministic-fixture:bem-compressed-field-parity-v1")
            .reference("independent dense TBEM pressure field plus checked compressed field evaluation")
            .reference_checksum("analytic/deterministic-fixture:bem-compressed-field-parity-v1")
            .dimension("3-D rigid sphere; constant-panel 48-DOF dense/H-matrix/SLFMM/MLFMM/directional-MLFMM")
            .formulation("common checked and legacy pressure-field contract across compressed backends")
            .tolerance("compressed-to-dense field error <= 5e-2; four checked backends and invalid-point rejections")
            .seed("bem-compressed-field-parity:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(_) => QaResult::failed(CASE_ID, "panic during compressed field parity validation")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
    };
    println!(
        "QA_RESULT: {}",
        serde_json::to_string(&result).expect("QA JSON serialization")
    );
    if matches!(&result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
