//! Independent CPU evidence for fail-closed mixed-precision adjoint capability propagation.

use sonium_qa::QaResult;
use sonium_solvers::mixed_precision::MixedPrecisionPreconditioner;
use sonium_solvers::{
    AdjointCapability, AdjointSolve, DifferentiableLinearSolve, GmresConfig, LinearOperator,
    Preconditioner, SolverError,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const CASE_ID: &str = "solvers.mixed-precision-adjoint-capability.v1";
const CHECKSUM: &str = "analytic/deterministic-fixture:mixed-precision-adjoint-capability-v1";
const TOLERANCE: f64 = 1.0e-12;

struct CountingIdentity {
    applications: Arc<AtomicUsize>,
}

impl CountingIdentity {
    fn new() -> (Self, Arc<AtomicUsize>) {
        let applications = Arc::new(AtomicUsize::new(0));
        (
            Self {
                applications: Arc::clone(&applications),
            },
            applications,
        )
    }
}

impl LinearOperator<f64> for CountingIdentity {
    fn num_rows(&self) -> usize {
        1
    }

    fn num_cols(&self) -> usize {
        1
    }

    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        self.applications.fetch_add(1, Ordering::Relaxed);
        y.copy_from_slice(x);
    }

    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        self.applications.fetch_add(1, Ordering::Relaxed);
        y.copy_from_slice(x);
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

struct ForwardOnlyLow {
    applications: Arc<AtomicUsize>,
}

impl ForwardOnlyLow {
    fn new() -> (Self, Arc<AtomicUsize>) {
        let applications = Arc::new(AtomicUsize::new(0));
        (
            Self {
                applications: Arc::clone(&applications),
            },
            applications,
        )
    }
}

impl Preconditioner<f32> for ForwardOnlyLow {
    fn apply(&self, r: &[f32]) -> Vec<f32> {
        self.applications.fetch_add(1, Ordering::Relaxed);
        r.to_vec()
    }
}

fn config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 4,
        restart: 1,
        tolerance: TOLERANCE,
        ..GmresConfig::default()
    }
}

fn run() -> Result<[f64; 5], String> {
    let (operator, operator_applications) = CountingIdentity::new();
    let (low, low_applications) = ForwardOnlyLow::new();
    let mixed = MixedPrecisionPreconditioner::<f64, f32>::new(Box::new(low));

    if mixed.transpose_capability() != AdjointCapability::Unsupported
        || mixed.hermitian_capability() != AdjointCapability::Unsupported
    {
        return Err("mixed-precision wrapper advertised an unsupported adjoint as exact".into());
    }

    let handle = DifferentiableLinearSolve::new(&operator, &mixed, config())
        .map_err(|error| format!("mixed-precision differentiable handle setup failed: {error}"))?;

    let mut transpose_rejected = 0.0;
    let mut hermitian_rejected = 0.0;
    for (kind, rejected) in [
        (AdjointSolve::Transpose, &mut transpose_rejected),
        (AdjointSolve::Hermitian, &mut hermitian_rejected),
    ] {
        let error = handle
            .solve_adjoint(&[1.0], kind)
            .expect_err("unsupported mixed adjoint must be rejected");
        if !matches!(error, SolverError::UnsupportedOperation { .. }) {
            return Err(format!(
                "mixed-precision adjoint returned the wrong error: {error}"
            ));
        }
        *rejected = 1.0;
    }

    Ok([
        1.0,
        transpose_rejected,
        hermitian_rejected,
        operator_applications.load(Ordering::Relaxed) as f64,
        low_applications.load(Ordering::Relaxed) as f64,
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(
            [
                capability_contract,
                transpose_rejected,
                hermitian_rejected,
                operator_applications,
                low_applications,
            ],
        ) => {
            let result = QaResult::passed(CASE_ID)
                .reference("fail-closed mixed-precision adjoint capability propagation")
                .input_checksum(CHECKSUM)
                .reference_checksum(CHECKSUM)
                .formulation("forward-only f32 preconditioner wrapped for an f64 differentiable solve")
                .dimension("1x1 counting identity operator; f64 outer/f32 inner preconditioner")
                .tolerance("unsupported transpose/Hermitian actions reject before operator or preconditioner iteration")
                .seed("solvers-mixed-precision-adjoint-capability:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64)
                .maximum("operator_applications", TOLERANCE)
                .maximum("low_precision_applications", TOLERANCE)
                .metric("capability_contract", capability_contract)
                .metric("transpose_rejected", transpose_rejected)
                .metric("hermitian_rejected", hermitian_rejected)
                .metric("operator_applications", operator_applications)
                .metric("low_precision_applications", low_applications);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("FAILED: {error}");
            let result = QaResult::failed(CASE_ID, error)
                .input_checksum(CHECKSUM)
                .reference_checksum(CHECKSUM)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
