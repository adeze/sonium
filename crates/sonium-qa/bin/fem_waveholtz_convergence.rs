//! WaveHoltz direct-reference mesh and temporal study.
//!
//! This is not an h-convergence proof against an analytical solution. It
//! checks that the production WaveHoltz method tracks a separately assembled
//! direct Helmholtz solve on several mesh sizes, and that its temporal filter
//! error decreases at the expected second-order rate.

use num_complex::Complex64;
use sonium_fem::assembly::{HelmholtzAssembler, HelmholtzProblem};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::solver::{SolverConfig, SolverType, solve};
use sonium_fem::waveholtz::{WaveHoltzConfig, solve_waveholtz};
use sonium_qa::QaResult;
use std::f64::consts::PI;
use std::time::Instant;

const CASE_ID: &str = "fem.waveholtz-direct-reference.v1";
const TEMPORAL_ORDER_MINIMUM: f64 = 1.8;
const REFERENCE_ERROR_MAXIMUM: f64 = 1.0e-3;

#[derive(Debug)]
struct Metrics {
    temporal_error_20: f64,
    temporal_error_40: f64,
    temporal_error_80: f64,
    temporal_order_20_40: f64,
    temporal_order_40_80: f64,
    mesh_error_4: f64,
    mesh_error_8: f64,
    mesh_error_16: f64,
    mesh_iterations_4: usize,
    mesh_iterations_8: usize,
    mesh_iterations_16: usize,
}

fn waveholtz_direct_reference_error(n: usize, k: f64, steps_per_period: usize) -> (f64, usize) {
    let mesh = unit_square_triangles(n);
    let source = |x: f64, y: f64, _z: f64| Complex64::new((PI * x).sin() * (PI * y).sin(), 0.0);
    let problem =
        HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, Complex64::new(k, 0.0), source);

    let direct = solve(
        &problem,
        &SolverConfig {
            solver_type: SolverType::Direct,
            ..Default::default()
        },
    )
    .expect("direct Helmholtz solve should succeed");

    let assembler = HelmholtzAssembler::new(&mesh, PolynomialDegree::P1);
    let rhs_real: Vec<f64> = problem.rhs.iter().map(|value| value.re).collect();
    let waveholtz = solve_waveholtz(
        &assembler,
        &rhs_real,
        k,
        &WaveHoltzConfig {
            steps_per_period,
            tolerance: 1.0e-10,
            inner_tolerance: 1.0e-12,
            dispersion_correction: false,
            ..Default::default()
        },
    )
    .expect("WaveHoltz solve should succeed");

    let direct_norm = direct
        .values
        .iter()
        .map(|value| value.re * value.re)
        .sum::<f64>()
        .sqrt();
    let difference_norm = direct
        .values
        .iter()
        .zip(waveholtz.values.iter())
        .map(|(direct, waveholtz)| (direct.re - waveholtz.re).powi(2))
        .sum::<f64>()
        .sqrt();

    (
        difference_norm / direct_norm.max(1.0e-15),
        waveholtz.iterations,
    )
}

fn run() -> Result<Metrics, String> {
    let temporal_steps = [20_usize, 40, 80];
    let temporal_errors: Vec<f64> = temporal_steps
        .iter()
        .map(|&steps_per_period| {
            let (error, _) = waveholtz_direct_reference_error(8, 1.0, steps_per_period);
            println!("steps/period={steps_per_period:>3}: relative error={error:.4e}");
            error
        })
        .collect();

    if !temporal_errors.iter().all(|error| error.is_finite()) {
        return Err("WaveHoltz temporal reference produced a non-finite error".into());
    }
    if !(temporal_errors[2] < temporal_errors[1] && temporal_errors[1] < temporal_errors[0]) {
        return Err(format!(
            "WaveHoltz temporal error is not strictly decreasing: {temporal_errors:?}"
        ));
    }

    let temporal_order_20_40 = (temporal_errors[0] / temporal_errors[1]).log2();
    let temporal_order_40_80 = (temporal_errors[1] / temporal_errors[2]).log2();
    if temporal_order_20_40 < TEMPORAL_ORDER_MINIMUM
        || temporal_order_40_80 < TEMPORAL_ORDER_MINIMUM
    {
        return Err(format!(
            "WaveHoltz temporal convergence order below {TEMPORAL_ORDER_MINIMUM}: {temporal_order_20_40:.3}, {temporal_order_40_80:.3}"
        ));
    }
    if temporal_errors[2] > REFERENCE_ERROR_MAXIMUM {
        return Err(format!(
            "WaveHoltz/direct error at 80 steps/period exceeds {REFERENCE_ERROR_MAXIMUM}: {:.4e}",
            temporal_errors[2]
        ));
    }

    println!("=== WaveHoltz Direct-Reference Mesh Study ===");
    println!(
        "{:>8} {:>10} {:>16} {:>8}",
        "n", "DOFs", "rel_error", "iters"
    );
    println!("{}", "-".repeat(50));

    let mut mesh_results = Vec::new();
    for &n in &[4_usize, 8, 16] {
        let (error, iterations) = waveholtz_direct_reference_error(n, 1.0, 80);
        let ndofs = (n + 1) * (n + 1);
        println!("{n:>8} {ndofs:>10} {error:>16.4e} {iterations:>8}");
        if !error.is_finite() || error > REFERENCE_ERROR_MAXIMUM {
            return Err(format!(
                "WaveHoltz/direct relative error at n={n} exceeds {REFERENCE_ERROR_MAXIMUM}: {error:.4e}"
            ));
        }
        if iterations == 0 {
            return Err(format!("WaveHoltz performed no iterations at n={n}"));
        }
        mesh_results.push((error, iterations));
    }

    Ok(Metrics {
        temporal_error_20: temporal_errors[0],
        temporal_error_40: temporal_errors[1],
        temporal_error_80: temporal_errors[2],
        temporal_order_20_40,
        temporal_order_40_80,
        mesh_error_4: mesh_results[0].0,
        mesh_error_8: mesh_results[1].0,
        mesh_error_16: mesh_results[2].0,
        mesh_iterations_4: mesh_results[0].1,
        mesh_iterations_8: mesh_results[1].1,
        mesh_iterations_16: mesh_results[2].1,
    })
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("separate assembled direct Helmholtz solve")
                .reference_checksum(
                    "analytic/deterministic-fixture:fem-waveholtz-direct-reference-v1",
                )
                .input_checksum("analytic/deterministic-fixture:fem-waveholtz-direct-reference-v1")
                .dimension("2D unit-square P1 triangles; temporal steps 20/40/80; meshes n=4/8/16")
                .formulation(
                    "production WaveHoltz time/filter/GMRES path versus assembled direct FEM solve",
                )
                .tolerance("temporal order >= 1.8 and all relative reference errors <= 1e-3")
                .metric("temporal_error_20", metrics.temporal_error_20)
                .metric("temporal_error_40", metrics.temporal_error_40)
                .metric("temporal_error_80", metrics.temporal_error_80)
                .metric("temporal_order_20_40", metrics.temporal_order_20_40)
                .metric("temporal_order_40_80", metrics.temporal_order_40_80)
                .metric("mesh_error_4", metrics.mesh_error_4)
                .metric("mesh_error_8", metrics.mesh_error_8)
                .metric("mesh_error_16", metrics.mesh_error_16)
                .metric("mesh_iterations_4", metrics.mesh_iterations_4 as f64)
                .metric("mesh_iterations_8", metrics.mesh_iterations_8 as f64)
                .metric("mesh_iterations_16", metrics.mesh_iterations_16 as f64)
                .minimum("temporal_order_20_40", TEMPORAL_ORDER_MINIMUM)
                .minimum("temporal_order_40_80", TEMPORAL_ORDER_MINIMUM)
                .maximum("temporal_error_80", REFERENCE_ERROR_MAXIMUM)
                .maximum("mesh_error_4", REFERENCE_ERROR_MAXIMUM)
                .maximum("mesh_error_8", REFERENCE_ERROR_MAXIMUM)
                .maximum("mesh_error_16", REFERENCE_ERROR_MAXIMUM)
                .minimum("mesh_iterations_4", 1.0)
                .minimum("mesh_iterations_8", 1.0)
                .minimum("mesh_iterations_16", 1.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("WaveHoltz QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone());
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("WaveHoltz QA failure must serialize")
            );
            eprintln!("FEM WAVEHOLTZ DIRECT REFERENCE FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
