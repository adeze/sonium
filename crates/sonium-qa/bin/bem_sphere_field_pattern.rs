//! Release-backed CPU evidence for the production rigid-sphere far field.
//!
//! The BEM field is evaluated through the public production solution API and
//! compared as a complex scattered pressure pattern with the independent
//! `sonium-wave` rigid-sphere Mie series.  This case is deliberately limited
//! to an exterior rigid sphere with a Neumann boundary.

use num_complex::Complex64;
use serde_json::json;
use sonium_bem::core::{
    AssemblyMethod, BemProblem, BemSolution, BemSolver, IncidentField, SolverMethod,
};
use sonium_qa::QaResult;
use sonium_wave::analytical::sphere_scattering_3d;
use sonium_xem::io::binary_output_dir;
use std::f64::consts::PI;
use std::time::Instant;

const CASE_ID: &str = "bem.sphere-field-pattern-mie.v1";
const ARTIFACT_PATH: &str = "sonium-qa/qa-bem-sphere-field-pattern/field-pattern.json";
const RADIUS_M: f64 = 0.2;
const SPEED_OF_SOUND_M_PER_S: f64 = 343.0;
const DENSITY_KG_PER_M3: f64 = 1.21;
const KA: f64 = 0.5;
const MIE_TERMS: usize = 40;
const MAX_L2_RELATIVE_ERROR: f64 = 1.5e-1;
const MAX_POINTWISE_RELATIVE_ERROR: f64 = 2.5e-1;
const INPUT_CHECKSUM: &str =
    "analytic/deterministic-fixture:bem-sphere-field-pattern-mie-v1:radius=0.2m,ka=0.5";
const REFERENCE: &str =
    "sonium-wave::analytical::sphere_scattering_3d independent rigid-sphere Mie field pattern";
const REFERENCE_CHECKSUM: &str =
    "sonium-wave::analytical::sphere_scattering_3d:rigid-sphere-neumann:v1";
const DIMENSION: &str = "3D rigid sphere, 7 far-field polar samples at r=max(12R,R+1m), ka=0.5";
const FORMULATION: &str = "production dense TBEM/Burton-Miller rigid-sphere Neumann solve versus independent sonium-wave Mie series";
const TOLERANCE: &str = "l2_relative_error <= 1.5e-1; max_pointwise_relative_error <= 2.5e-1";
const THETA: [f64; 7] = [0.2, 0.65, 1.1, 1.55, 2.0, 2.45, 2.9];

#[derive(Debug)]
struct FieldPatternMetrics {
    l2_relative_error: f64,
    max_pointwise_relative_error: f64,
    eval_radius_m: f64,
    frequency_hz: f64,
    wave_number: f64,
    elapsed_ms: u64,
    dofs: usize,
    bem_scattered: Vec<Complex64>,
    reference_scattered: Vec<Complex64>,
}

fn frequency_hz() -> f64 {
    KA * SPEED_OF_SOUND_M_PER_S / (2.0 * PI * RADIUS_M)
}

fn relative_complex_error(actual: Complex64, reference: Complex64) -> f64 {
    (actual - reference).norm() / actual.norm().max(reference.norm()).max(f64::MIN_POSITIVE)
}

fn solve_rigid_sphere() -> Result<(BemSolution, u64), String> {
    let problem = BemProblem::try_rigid_sphere_scattering(
        RADIUS_M,
        frequency_hz(),
        SPEED_OF_SOUND_M_PER_S,
        DENSITY_KG_PER_M3,
    )
    .map_err(|error| format!("rigid-sphere problem admission failed: {error:?}"))?
    .with_incident_field(IncidentField::plane_wave([0.0, 0.0, 1.0], 1.0))
    .with_burton_miller(true);

    let started = Instant::now();
    let solution = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::BiCgStab)
        .with_max_iterations(500)
        .with_tolerance(1.0e-8)
        .solve(&problem)
        .map_err(|error| format!("rigid-sphere production solve failed: {error:?}"))?;
    Ok((solution, started.elapsed().as_millis() as u64))
}

fn evaluate_field_pattern() -> Result<FieldPatternMetrics, String> {
    let (solution, elapsed_ms) = solve_rigid_sphere()?;
    let eval_radius_m = (12.0 * RADIUS_M).max(RADIUS_M + 1.0);
    let points: Vec<[f64; 3]> = THETA
        .iter()
        .map(|&theta| {
            [
                eval_radius_m * theta.sin(),
                0.0,
                eval_radius_m * theta.cos(),
            ]
        })
        .collect();

    let bem_scattered: Vec<Complex64> = solution
        .evaluate_pressure_field(&points)
        .into_iter()
        .map(|field| field.p_scattered)
        .collect();
    let mie = sphere_scattering_3d(
        solution.physics.wave_number,
        RADIUS_M,
        MIE_TERMS,
        vec![eval_radius_m],
        THETA.to_vec(),
    );
    let reference_scattered: Vec<Complex64> = mie
        .pressure
        .iter()
        .zip(points.iter())
        .map(|(total, point)| {
            let phase = solution.physics.wave_number * point[2];
            *total - Complex64::from_polar(1.0, phase)
        })
        .collect();

    if bem_scattered.len() != reference_scattered.len()
        || bem_scattered
            .iter()
            .chain(reference_scattered.iter())
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("BEM/Mie field pattern contains a non-finite value".into());
    }

    let reference_norm = reference_scattered
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    if reference_norm <= f64::MIN_POSITIVE {
        return Err("Mie field pattern has no measurable scattered field".into());
    }
    let l2_relative_error = bem_scattered
        .iter()
        .zip(reference_scattered.iter())
        .map(|(actual, reference)| (*actual - *reference).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / reference_norm;
    let max_pointwise_relative_error = bem_scattered
        .iter()
        .zip(reference_scattered.iter())
        .map(|(actual, reference)| relative_complex_error(*actual, *reference))
        .fold(0.0, f64::max);

    if !l2_relative_error.is_finite()
        || l2_relative_error > MAX_L2_RELATIVE_ERROR
        || !max_pointwise_relative_error.is_finite()
        || max_pointwise_relative_error > MAX_POINTWISE_RELATIVE_ERROR
    {
        return Err(format!(
            "field-pattern thresholds exceeded: l2={l2_relative_error:.6e} (max {MAX_L2_RELATIVE_ERROR:.6e}), pointwise={max_pointwise_relative_error:.6e} (max {MAX_POINTWISE_RELATIVE_ERROR:.6e})"
        ));
    }

    Ok(FieldPatternMetrics {
        l2_relative_error,
        max_pointwise_relative_error,
        eval_radius_m,
        frequency_hz: frequency_hz(),
        wave_number: solution.physics.wave_number,
        elapsed_ms,
        dofs: solution.num_dofs(),
        bem_scattered,
        reference_scattered,
    })
}

fn complex_pairs(values: &[Complex64]) -> Vec<[f64; 2]> {
    values.iter().map(|value| [value.re, value.im]).collect()
}

fn write_artifact(metrics: &FieldPatternMetrics) -> Result<(), String> {
    let output_dir = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"));
    let artifact = json!({
        "case_id": CASE_ID,
        "status": "passed",
        "radius_m": RADIUS_M,
        "frequency_hz": metrics.frequency_hz,
        "wave_number": metrics.wave_number,
        "ka": KA,
        "eval_radius_m": metrics.eval_radius_m,
        "theta_rad": THETA,
        "bem_scattered_complex": complex_pairs(&metrics.bem_scattered),
        "reference_scattered_complex": complex_pairs(&metrics.reference_scattered),
        "l2_relative_error": metrics.l2_relative_error,
        "max_pointwise_relative_error": metrics.max_pointwise_relative_error,
        "max_l2_relative_error": MAX_L2_RELATIVE_ERROR,
        "max_pointwise_relative_error_threshold": MAX_POINTWISE_RELATIVE_ERROR,
    });
    let path = output_dir.join("field-pattern.json");
    let bytes = serde_json::to_vec_pretty(&artifact).map_err(|error| error.to_string())?;
    std::fs::write(path, bytes).map_err(|error| format!("write field-pattern artifact: {error}"))
}

fn main() {
    let started = Instant::now();
    let result = match evaluate_field_pattern() {
        Ok(metrics) => match write_artifact(&metrics) {
            Ok(()) => QaResult::passed(CASE_ID)
                .metric("l2_relative_error", metrics.l2_relative_error)
                .metric(
                    "max_pointwise_relative_error",
                    metrics.max_pointwise_relative_error,
                )
                .metric("sample_count", THETA.len() as f64)
                .metric("dofs", metrics.dofs as f64)
                .maximum("l2_relative_error", MAX_L2_RELATIVE_ERROR)
                .maximum("max_pointwise_relative_error", MAX_POINTWISE_RELATIVE_ERROR)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .backend_passed("cpu")
                .seed("bem-sphere-field-pattern-mie:v1")
                .input_checksum(INPUT_CHECKSUM)
                .reference(REFERENCE)
                .reference_checksum(REFERENCE_CHECKSUM)
                .dimension(DIMENSION)
                .formulation(FORMULATION)
                .tolerance(TOLERANCE)
                .artifact(ARTIFACT_PATH)
                .timing_ms("production_solve", metrics.elapsed_ms)
                .elapsed_ms(started.elapsed().as_millis() as u64),
            Err(error) => QaResult::failed(CASE_ID, error),
        },
        Err(error) => QaResult::failed(CASE_ID, error),
    };

    match result.to_json_line() {
        Ok(line) => println!("QA_RESULT: {line}"),
        Err(error) => {
            eprintln!("FAILED: could not emit QA result: {error}");
            std::process::exit(1);
        }
    }
    if result.status == sonium_qa::QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
