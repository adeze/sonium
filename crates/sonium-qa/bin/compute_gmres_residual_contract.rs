//! CPU QA for backend-GMRES original-operator residual certification.

use num_complex::Complex32;
use sonium_compute::{
    Backend, BackendCsr, ChebyshevBackend, CpuBackend, GmresBackendConfig, PolynomialJacobiBackend,
    gmres_backend_preconditioned,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.gmres-residual-contract.v1";
const TOLERANCE: f32 = 1.0e-6;
const MAX_RESIDUAL_ERROR: f64 = 2.0e-5;

struct Metrics {
    strict_cases: usize,
    nonconverged_cases: usize,
    maximum_residual_error: f64,
    maximum_reported_residual: f64,
    independent_residuals_finite: bool,
}

fn complex32_helmholtz_tridiag(n: usize, shift: f32) -> (Vec<i32>, Vec<i32>, Vec<Complex32>) {
    let mut row_ptr = vec![0_i32; n + 1];
    let mut col_idx = Vec::with_capacity(n * 3);
    let mut values = Vec::with_capacity(n * 3);
    for row in 0..n {
        if row > 0 {
            col_idx.push((row - 1) as i32);
            values.push(Complex32::new(-1.0, 0.0));
        }
        col_idx.push(row as i32);
        values.push(Complex32::new(2.0 + shift, 0.0));
        if row + 1 < n {
            col_idx.push((row + 1) as i32);
            values.push(Complex32::new(-1.0, 0.0));
        }
        row_ptr[row + 1] = values.len() as i32;
    }
    (row_ptr, col_idx, values)
}

fn relative_residual(
    row_ptr: &[i32],
    col_idx: &[i32],
    values: &[Complex32],
    x: &[Complex32],
    b: &[Complex32],
) -> f64 {
    let mut residual_sq = 0.0_f64;
    let mut rhs_sq = 0.0_f64;
    for row in 0..b.len() {
        let mut ax = Complex32::new(0.0, 0.0);
        for entry in row_ptr[row] as usize..row_ptr[row + 1] as usize {
            ax += values[entry] * x[col_idx[entry] as usize];
        }
        let residual = b[row] - ax;
        residual_sq += f64::from(residual.norm_sqr());
        rhs_sq += f64::from(b[row].norm_sqr());
    }
    residual_sq.sqrt() / rhs_sq.sqrt()
}

fn run() -> Result<Metrics, String> {
    let backend = CpuBackend::new();
    let n = 64;
    let (row_ptr, col_idx, values) = complex32_helmholtz_tridiag(n, 0.01);
    let matrix = BackendCsr::<CpuBackend, Complex32>::try_from_host(
        &backend, n, n, &row_ptr, &col_idx, &values,
    )
    .map_err(|error| format!("fixture upload failed: {error}"))?;
    let rhs_host: Vec<_> = (0..n)
        .map(|i| Complex32::new(((i as f32) * 0.1).sin(), 0.0))
        .collect();
    let mut rhs = backend
        .raw_alloc::<Complex32>(n)
        .map_err(|error| format!("RHS allocation failed: {error}"))?;
    backend.copy_h2d(&rhs_host, &mut rhs);

    let poly = PolynomialJacobiBackend::<CpuBackend, Complex32>::from_csr_host(
        &backend,
        &matrix,
        &row_ptr,
        &col_idx,
        &values,
        2,
        Complex32::new(1.0, 0.0),
    );
    let cheb = ChebyshevBackend::<CpuBackend, Complex32>::from_csr_host(
        &backend, &matrix, &row_ptr, &col_idx, &values, 5, 30.0,
    );
    let config = GmresBackendConfig {
        max_iterations: 200,
        tolerance: TOLERANCE,
    };
    let mut outputs = [
        backend
            .raw_alloc::<Complex32>(n)
            .map_err(|error| format!("PolyJacobi output allocation failed: {error}"))?,
        backend
            .raw_alloc::<Complex32>(n)
            .map_err(|error| format!("Chebyshev output allocation failed: {error}"))?,
    ];
    let statuses = [
        gmres_backend_preconditioned(
            &backend,
            &matrix,
            &poly,
            &rhs,
            None,
            &mut outputs[0],
            &config,
        )
        .map_err(|error| format!("PolyJacobi GMRES failed: {error}"))?,
        gmres_backend_preconditioned(
            &backend,
            &matrix,
            &cheb,
            &rhs,
            None,
            &mut outputs[1],
            &config,
        )
        .map_err(|error| format!("Chebyshev GMRES failed: {error}"))?,
    ];

    let mut maximum_residual_error = 0.0_f64;
    let mut maximum_reported_residual = 0.0_f64;
    let mut nonconverged_cases = 0;
    let mut independent_residuals_finite = true;
    for (status, output) in statuses.iter().zip(outputs.iter()) {
        let mut solution = vec![Complex32::new(0.0, 0.0); n];
        backend.copy_d2h(output, &mut solution);
        let independent = relative_residual(&row_ptr, &col_idx, &values, &solution, &rhs_host);
        independent_residuals_finite &= independent.is_finite();
        maximum_residual_error =
            maximum_residual_error.max((status.final_residual as f64 - independent).abs());
        maximum_reported_residual = maximum_reported_residual.max(status.final_residual as f64);
        if !status.converged {
            nonconverged_cases += 1;
        }
    }
    if nonconverged_cases != 2 {
        return Err(format!(
            "expected both strict Complex32 cases to fail certified convergence, got {statuses:?}"
        ));
    }
    if !independent_residuals_finite || maximum_residual_error > MAX_RESIDUAL_ERROR {
        return Err(format!(
            "residual certification mismatch: error={maximum_residual_error}, finite={independent_residuals_finite}"
        ));
    }
    Ok(Metrics {
        strict_cases: statuses.len(),
        nonconverged_cases,
        maximum_residual_error,
        maximum_reported_residual,
        independent_residuals_finite,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("strict_cases", metrics.strict_cases as f64)
                .metric("nonconverged_cases", metrics.nonconverged_cases as f64)
                .metric("maximum_residual_error", metrics.maximum_residual_error)
                .metric(
                    "maximum_reported_residual",
                    metrics.maximum_reported_residual,
                )
                .metric(
                    "independent_residuals_finite",
                    metrics.independent_residuals_finite as u8 as f64,
                )
                .minimum("strict_cases", 2.0)
                .minimum("nonconverged_cases", 2.0)
                .maximum("maximum_residual_error", MAX_RESIDUAL_ERROR)
                .minimum("independent_residuals_finite", 1.0)
                .reference("independent Complex32 CSR residual recomputation")
                .reference_checksum(
                    "analytic/deterministic-fixture:compute-gmres-residual-contract-v1",
                )
                .dimension(
                    "64x64 Complex32 tridiagonal; strict projected-vs-original residual gate",
                )
                .seed("compute-gmres-residual-contract:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("GMRES residual QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("GMRES residual QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
