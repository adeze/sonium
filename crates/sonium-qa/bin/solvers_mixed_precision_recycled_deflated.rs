//! Independent CPU evidence for mixed-precision recycled/deflated continuation.

use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::preconditioners::DiagonalPreconditioner;
use sonium_solvers::sparse::CsrMatrix;
#[allow(deprecated)]
use sonium_solvers::{
    DeflationSubspace, GcroDrConfig, GcroDrSubspace, GmresConfig, RecycledSubspace, SolverStatus,
    try_mixed_precision_deflated_solve, try_mixed_precision_gcro_dr_solve_with_guess,
    try_mixed_precision_recycled_solve, try_mixed_precision_recycled_solve_with_guess,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.mixed-precision-recycled-deflated.v1";
const SEED: &str = "solvers-mixed-precision-recycled-deflated:cpu:v1";

fn operator() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 3.0])
}

fn config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 2,
        restart: 1,
        tolerance: 1.0e-20,
        exact_residual_check_interval: 0,
        ..GmresConfig::default()
    }
}

fn gcro_config() -> GcroDrConfig<f64> {
    GcroDrConfig {
        max_iterations: 2,
        restart: 1,
        tolerance: 1.0e-20,
        print_interval: 0,
    }
}

#[allow(deprecated)]
fn run() -> Result<(usize, usize, f64, f64), String> {
    let matrix = operator();
    let deflation = DeflationSubspace::try_new(vec![vec![1.0, 0.0]], &matrix)
        .map_err(|error| format!("deflation construction failed: {error}"))?;
    let low_preconditioner = || {
        Box::new(DiagonalPreconditioner::from_diagonal(&[1.0_f32, 1.0]))
            as Box<dyn sonium_solvers::Preconditioner<f32> + Send + Sync>
    };

    let deflated = try_mixed_precision_deflated_solve::<f64, f32, _>(
        &matrix,
        &deflation,
        &[1.0, 1.0],
        low_preconditioner(),
        &config(),
    )
    .map_err(|error| format!("deflated solve failed: {error}"))?;
    if !deflated.converged
        || deflated.status != SolverStatus::Converged
        || deflated.op_counts.exact_residual_checks == 0
    {
        return Err(format!(
            "deflated gate failed: status={:?}, converged={}, exact_checks={}",
            deflated.status, deflated.converged, deflated.op_counts.exact_residual_checks
        ));
    }

    let mut recycled = RecycledSubspace::new(1);
    let initial = try_mixed_precision_recycled_solve::<f64, f32, _>(
        &matrix,
        &[1.0, 1.0],
        &mut recycled,
        low_preconditioner(),
        &config(),
    )
    .map_err(|error| format!("recycled solve failed: {error}"))?;
    if initial.op_counts.exact_residual_checks == 0 {
        return Err("recycled solve did not perform an exact candidate check".into());
    }

    let resumed = try_mixed_precision_recycled_solve_with_guess::<f64, f32, _>(
        &matrix,
        &[1.0, 1.0],
        Some(&[0.5, 1.0 / 3.0]),
        &mut recycled,
        low_preconditioner(),
        &config(),
    )
    .map_err(|error| format!("recycled continuation failed: {error}"))?;
    if !resumed.converged || resumed.iterations != 0 || resumed.residual > 1.0e-14 {
        return Err(format!(
            "recycled continuation contract failed: converged={}, iterations={}, residual={:e}",
            resumed.converged, resumed.iterations, resumed.residual
        ));
    }

    let mut gcro_subspace = GcroDrSubspace::new(1);
    let gcro_resumed = try_mixed_precision_gcro_dr_solve_with_guess::<f64, f32, _>(
        &matrix,
        &[1.0, 1.0],
        Some(&[0.5, 1.0 / 3.0]),
        &mut gcro_subspace,
        low_preconditioner(),
        &gcro_config(),
    )
    .map_err(|error| format!("GCRO-DR continuation failed: {error}"))?;
    if !gcro_resumed.converged || gcro_resumed.iterations != 0 || gcro_resumed.residual > 1.0e-14 {
        return Err(format!(
            "GCRO-DR continuation contract failed: converged={}, iterations={}, residual={:e}",
            gcro_resumed.converged, gcro_resumed.iterations, gcro_resumed.residual
        ));
    }

    Ok((
        deflated.op_counts.exact_residual_checks,
        initial.op_counts.exact_residual_checks,
        resumed.residual,
        gcro_resumed.residual,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((deflated_checks, recycled_checks, continuation_residual, gcro_residual)) => {
            QaResult::passed(CASE_ID)
                .reference("independent diagonal CPU mixed-precision continuation fixture")
                .input_checksum(
                    "analytic/deterministic-fixture:mixed-precision-recycled-deflated-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:mixed-precision-recycled-deflated-v1",
                )
                .dimension("2x2 diagonal; f64 operator/f32 preconditioner")
                .seed(SEED)
                .metric("deflated_exact_residual_checks", deflated_checks as f64)
                .metric("recycled_exact_residual_checks", recycled_checks as f64)
                .metric("recycled_continuation_residual", continuation_residual)
                .metric("gcro_dr_continuation_residual", gcro_residual)
                .elapsed_ms(started.elapsed().as_millis() as u64)
        }
        Err(error) => QaResult::failed(CASE_ID, error)
            .reference("independent diagonal CPU mixed-precision continuation fixture")
            .input_checksum("analytic/deterministic-fixture:mixed-precision-recycled-deflated-v1")
            .reference_checksum(
                "analytic/deterministic-fixture:mixed-precision-recycled-deflated-v1",
            )
            .dimension("2x2 diagonal; f64 operator/f32 preconditioner")
            .seed(SEED)
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    let passed = result.status == QaStatus::Passed;
    println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
    if passed {
        println!("SUCCESS");
    } else {
        std::process::exit(1);
    }
}
