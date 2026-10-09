//! Independent CPU evidence for the mixed-precision GMRES-IR promoted residual gate.

use half::f16;
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::mixed_precision::{
    try_gmres_ir_ilu, try_gmres_ir_lu, try_mixed_precision_block_solve,
};
use sonium_solvers::preconditioners::{DiagonalPreconditioner, IluConfig};
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::{AccuracyStatus, BlockFgmresConfig, FgmresConfig, SolverStatus};
use std::time::Instant;

const CASE_ID: &str = "solvers.mixed-precision-promoted-residual.v1";
const SEED: &str = "solvers-mixed-precision-promoted-residual:cpu:v1";

fn spd_with_spectrum(n: usize, kappa: f64) -> CsrMatrix<f64> {
    let q = |i: usize, k: usize| {
        if k == 0 {
            1.0 / (n as f64).sqrt()
        } else {
            (2.0 / n as f64).sqrt()
                * (std::f64::consts::PI * k as f64 * (2 * i + 1) as f64 / (2 * n) as f64).cos()
        }
    };
    let lambda = |k: usize| kappa.powf(k as f64 / (n - 1) as f64);
    let mut dense = DenseMatrix::from_elem(n, n, 0.0_f64);
    for i in 0..n {
        for j in 0..n {
            dense[[i, j]] = (0..n).map(|k| lambda(k) * q(i, k) * q(j, k)).sum();
        }
    }
    CsrMatrix::from_dense(&dense, 1e-300)
}

fn midpoint_tolerance(residual: f64, estimate: f64) -> Result<f64, String> {
    if !residual.is_finite() || !estimate.is_finite() || estimate >= residual {
        return Err(format!(
            "fixture did not expose projected-residual optimism: residual={residual:e}, estimate={estimate:e}"
        ));
    }
    Ok((residual + estimate) * 0.5)
}

fn block_gate_reference() -> Result<(f64, f64, usize), String> {
    let matrix = spd_with_spectrum(128, 1.0e5);
    let x_exact: Vec<f64> = (0..128).map(|i| (i as f64).sin()).collect();
    let rhs_values = matrix.matvec(&x_exact);
    let mut rhs = DenseMatrix::from_elem(128, 1, 0.0_f64);
    for (row, value) in rhs_values.into_iter().enumerate() {
        rhs[[row, 0]] = value;
    }
    let low_matrix = matrix.cast::<f16>();
    let config = BlockFgmresConfig {
        max_iterations: 80,
        restart: 30,
        tolerance: 1.0e-12,
        // The wrapper must override this opt-out for mixed precision.
        true_residual_check: false,
        exact_residual_check_interval: 0,
    };
    let result = try_mixed_precision_block_solve::<f64, f16, _>(
        &matrix,
        &rhs,
        Box::new(DiagonalPreconditioner::from_csr(&low_matrix)),
        &config,
    )
    .map_err(|error| format!("block promoted gate: {error}"))?;
    if result.converged
        || result.residuals[0] > config.tolerance
        || result.true_residuals[0] <= config.tolerance
        || result.op_counts.exact_residual_checks == 0
    {
        return Err(format!(
            "block accepted projected convergence: converged={}, projected={:e}, true={:e}, tolerance={:e}, exact_checks={}",
            result.converged,
            result.residuals[0],
            result.true_residuals[0],
            config.tolerance,
            result.op_counts.exact_residual_checks,
        ));
    }
    Ok((
        result.true_residuals[0],
        result.residuals[0],
        result.op_counts.exact_residual_checks,
    ))
}

fn run() -> Result<(f64, f64, f64, f64, usize, usize, f64, f64, usize), String> {
    let n = 128;
    let x_exact: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();

    let matrix_lu = spd_with_spectrum(n, 1e5);
    let rhs_lu = matrix_lu.matvec(&x_exact);
    let baseline_config = FgmresConfig {
        max_iterations: 80,
        restart: 30,
        tolerance: 1e-30,
        ..FgmresConfig::default()
    };
    let baseline_lu = try_gmres_ir_lu::<f64, f16>(&matrix_lu, &rhs_lu, &baseline_config)
        .map_err(|error| format!("LU baseline: {error}"))?;
    let lu_tolerance = midpoint_tolerance(baseline_lu.residual, baseline_lu.estimated_residual)?;
    let lu_config = FgmresConfig {
        tolerance: lu_tolerance,
        ..baseline_config.clone()
    };
    let gated_lu = try_gmres_ir_lu::<f64, f16>(&matrix_lu, &rhs_lu, &lu_config)
        .map_err(|error| format!("LU promoted gate: {error}"))?;

    let matrix_ilu = spd_with_spectrum(n, 1e3);
    let rhs_ilu = matrix_ilu.matvec(&x_exact);
    let ilu_baseline_config = FgmresConfig {
        max_iterations: 1,
        restart: 30,
        tolerance: 1e-30,
        ..FgmresConfig::default()
    };
    let baseline_ilu = try_gmres_ir_ilu::<f64, f16>(
        &matrix_ilu,
        &rhs_ilu,
        &IluConfig::default(),
        &ilu_baseline_config,
    )
    .map_err(|error| format!("ILU baseline: {error}"))?;
    let ilu_tolerance = midpoint_tolerance(baseline_ilu.residual, baseline_ilu.estimated_residual)?;
    let ilu_config = FgmresConfig {
        tolerance: ilu_tolerance,
        ..ilu_baseline_config.clone()
    };
    let gated_ilu =
        try_gmres_ir_ilu::<f64, f16>(&matrix_ilu, &rhs_ilu, &IluConfig::default(), &ilu_config)
            .map_err(|error| format!("ILU promoted gate: {error}"))?;

    for (name, result, tolerance) in [
        ("LU", &gated_lu, lu_tolerance),
        ("ILU", &gated_ilu, ilu_tolerance),
    ] {
        if result.converged
            || result.status != SolverStatus::MaxIterationsReached
            || result.estimated_residual >= tolerance
            || result.residual <= tolerance
            || result.attainable_accuracy != AccuracyStatus::NotAttained
            || result.iterations == 0
        {
            return Err(format!(
                "{name} accepted projected convergence: status={:?}, converged={}, residual={:e}, estimate={:e}, tolerance={tolerance:e}, attainable={:?}, iterations={}",
                result.status,
                result.converged,
                result.residual,
                result.estimated_residual,
                result.attainable_accuracy,
                result.iterations,
            ));
        }
    }

    let (block_true_residual, block_projected_estimate, block_exact_checks) =
        block_gate_reference()?;

    Ok((
        gated_lu.residual,
        gated_lu.estimated_residual,
        gated_ilu.residual,
        gated_ilu.estimated_residual,
        gated_lu.iterations,
        gated_ilu.iterations,
        block_true_residual,
        block_projected_estimate,
        block_exact_checks,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            lu_residual,
            lu_estimate,
            ilu_residual,
            ilu_estimate,
            lu_iterations,
            ilu_iterations,
            block_true_residual,
            block_projected_estimate,
            block_exact_checks,
        )) => {
            let result = QaResult::passed(CASE_ID)
                .reference("independent DCT-spectrum f64 reference with f16 GMRES-IR LU and ILU")
                .input_checksum("analytic/deterministic-fixture:mixed-precision-promoted-residual-v1")
                .reference_checksum("analytic/deterministic-fixture:mixed-precision-promoted-residual-v1")
                .dimension("128x128 SPD; f64 outer/f16 inner; LU and ILU")
                .tolerance("projected estimate below midpoint but true residual above it; no convergence claimed")
                .seed(SEED)
                .elapsed_ms(started.elapsed().as_millis() as u64)
                .metric("lu_true_residual", lu_residual)
                .metric("lu_projected_estimate", lu_estimate)
                .metric("ilu_true_residual", ilu_residual)
            .metric("ilu_projected_estimate", ilu_estimate)
            .metric("lu_iterations", lu_iterations as f64)
            .metric("ilu_iterations", ilu_iterations as f64)
            .metric("block_true_residual", block_true_residual)
            .metric("block_projected_estimate", block_projected_estimate)
            .metric("block_exact_residual_checks", block_exact_checks as f64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("FAILED: {error}");
            let result = QaResult::failed(CASE_ID, error)
                .reference("independent DCT-spectrum f64 reference with f16 GMRES-IR LU and ILU")
                .input_checksum("analytic/deterministic-fixture:mixed-precision-promoted-residual-v1")
                .reference_checksum("analytic/deterministic-fixture:mixed-precision-promoted-residual-v1")
                .dimension("128x128 SPD; f64 outer/f16 inner; LU and ILU")
                .tolerance("projected estimate below midpoint but true residual above it; no convergence claimed")
                .seed(SEED)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
