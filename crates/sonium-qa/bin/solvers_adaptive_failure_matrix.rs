//! Independent checked failure-exit evidence for adaptive Krylov solvers.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::try_gmres_recycled_preconditioned;
use sonium_solvers::traits::Preconditioner;
use sonium_solvers::{
    AccuracyStatus, BlockFgmresConfig, ComplexField, CsrMatrix, DeflationSubspace, GcroDrConfig,
    GcroDrSubspace, GmresConfig, GmresDrConfig, IdentityPreconditioner, RecycledSubspace,
    SolverError, SolverResultContract, SolverStatus, try_block_fgmres_preconditioned, try_gcro_dr,
    try_gmres_deflated, try_gmres_dr_preconditioned, try_gmres_recycled,
};

const CASE_ID: &str = "solvers.adaptive-failure-matrix.v1";
const CHECKSUM: &str = "analytic/deterministic-fixture:adaptive-failure-matrix-v1";

fn zero_operator() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![0.0, 0.0])
}

fn gmres_config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 1,
        restart: 2,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    }
}

fn check_non_success<T: ComplexField>(
    name: &str,
    result: Result<sonium_solvers::GmresSolution<T>, SolverError>,
) -> Result<f64, String> {
    match result {
        Ok(solution) => {
            if solution.status == SolverStatus::Converged
                || solution.attainable_accuracy == AccuracyStatus::Attained
            {
                return Err(format!("{name}: zero operator reported success"));
            }
            solution
                .validate_contract()
                .map_err(|error| format!("{name}: invalid non-success contract: {error:?}"))?;
            Ok(1.0)
        }
        Err(_) => Ok(1.0),
    }
}

struct NonFinitePreconditioner;

impl Preconditioner<f64> for NonFinitePreconditioner {
    fn apply_into(&self, input: &[f64], output: &mut [f64]) {
        output.copy_from_slice(input);
        output[0] = f64::NAN;
    }

    fn apply_transpose_into(&self, input: &[f64], output: &mut [f64]) {
        self.apply_into(input, output);
    }
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let operator = zero_operator();
    let rhs = [1.0, -2.0];
    let identity = IdentityPreconditioner;
    let config = gmres_config();

    let gmres_dr = catch_unwind(AssertUnwindSafe(|| {
        try_gmres_dr_preconditioned(
            &operator,
            &identity,
            &rhs,
            None,
            &GmresDrConfig {
                gmres: config.clone(),
                num_deflation_vectors: 1,
            },
        )
    }))
    .map_err(|_| "GMRES-DR panicked".to_string())?;
    let gmres_dr = check_non_success("gmres_dr", gmres_dr)?;

    let mut gcro_subspace = GcroDrSubspace::new(1);
    let gcro = catch_unwind(AssertUnwindSafe(|| {
        try_gcro_dr(
            &operator,
            &rhs,
            None,
            &mut gcro_subspace,
            &GcroDrConfig::from_gmres(&config),
        )
    }))
    .map_err(|_| "GCRO-DR panicked".to_string())?;
    let gcro = check_non_success("gcro_dr", gcro)?;

    let mut recycled_subspace = RecycledSubspace::new(1);
    let recycled = catch_unwind(AssertUnwindSafe(|| {
        try_gmres_recycled(&operator, &rhs, None, &mut recycled_subspace, &config)
    }))
    .map_err(|_| "recycled GMRES panicked".to_string())?;
    let recycled = check_non_success("recycled", recycled)?;

    let deflation = DeflationSubspace::try_new(Vec::new(), &operator)
        .map_err(|error| format!("empty deflation fixture rejected: {error:?}"))?;
    let deflated = catch_unwind(AssertUnwindSafe(|| {
        try_gmres_deflated(&operator, &deflation, &rhs, None, &config)
    }))
    .map_err(|_| "deflated GMRES panicked".to_string())?;
    let deflated = check_non_success("deflated", deflated)?;

    let block_rhs = DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 0.0, 0.0, -2.0]);
    let block = catch_unwind(AssertUnwindSafe(|| {
        try_block_fgmres_preconditioned(
            &operator,
            &block_rhs,
            &identity,
            None,
            &BlockFgmresConfig {
                max_iterations: 1,
                restart: 1,
                tolerance: 1.0e-12,
                true_residual_check: true,
                exact_residual_check_interval: 0,
            },
        )
    }))
    .map_err(|_| "block FGMRES panicked".to_string())?;
    let block = match block {
        Ok(solution) => {
            if solution.converged {
                return Err("block FGMRES reported success for zero operator".into());
            }
            solution
                .validate_contract()
                .map_err(|error| format!("block FGMRES contract failed: {error:?}"))?;
            1.0
        }
        Err(_) => 1.0,
    };

    // Seed a populated deprecated recycled state, then force a checked
    // callback failure. The caller's serialized state must remain byte-identical.
    let identity_operator = CsrMatrix::identity(2);
    let mut seeded = RecycledSubspace::new(1);
    let seed = try_gmres_recycled(
        &identity_operator,
        &[1.0, 2.0],
        None,
        &mut seeded,
        &GmresConfig {
            max_iterations: 4,
            restart: 2,
            tolerance: 1.0e-10,
            ..GmresConfig::default()
        },
    )
    .map_err(|error| format!("recycled seed solve failed: {error:?}"))?;
    if !seed.converged {
        return Err(format!("recycled seed did not converge: {:?}", seed.status));
    }
    let before = seeded
        .to_checkpoint_bytes()
        .map_err(|error| format!("serialize seed state failed: {error:?}"))?;
    let failure = try_gmres_recycled_preconditioned(
        &identity_operator,
        &NonFinitePreconditioner,
        &[1.0, 2.0],
        None,
        &mut seeded,
        &GmresConfig {
            max_iterations: 4,
            restart: 2,
            tolerance: 1.0e-10,
            ..GmresConfig::default()
        },
    )
    .expect_err("non-finite preconditioner output must be typed");
    if !matches!(failure, SolverError::OperationFailed { .. }) {
        return Err(format!("unexpected recycled failure: {failure:?}"));
    }
    let after = seeded
        .to_checkpoint_bytes()
        .map_err(|error| format!("serialize failure state failed: {error:?}"))?;
    let state_unchanged = f64::from((before == after) as u8);
    if state_unchanged < 1.0 {
        return Err("recycled state changed after typed callback failure".into());
    }

    Ok(vec![
        ("gmres_dr_failure", gmres_dr),
        ("gcro_dr_failure", gcro),
        ("recycled_failure", recycled),
        ("deflated_failure", deflated),
        ("block_fgmres_failure", block),
        ("state_unchanged", state_unchanged),
        ("panic_free", 1.0),
    ])
}

fn main() {
    let started = Instant::now();
    let previous_panic_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => {
            let mut result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .input_checksum(CHECKSUM)
                .reference("independent zero-operator adaptive Krylov failure and recycled-state fixture")
                .reference_checksum(CHECKSUM)
                .dimension("2x2 CSR; GMRES-DR, GCRO-DR, recycled/deflated GMRES, block FGMRES")
                .formulation("checked adaptive continuation entry points")
                .tolerance("five failure exits remain typed or contract-valid non-converged; state unchanged = 1")
                .seed("solvers-adaptive-failure-matrix:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.minimum(name, 1.0).metric(name, value);
            }
            result
        }
        Ok(Err(error)) => QaResult::failed(CASE_ID, error),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("adaptive failure matrix panicked: {payload:?}"),
        ),
    };
    let passed = matches!(result.status, sonium_qa::QaStatus::Passed);
    println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
    if passed {
        println!("SUCCESS");
    } else {
        eprintln!(
            "FAILED: {}",
            result.message.as_deref().unwrap_or("QA failure")
        );
        std::panic::set_hook(previous_panic_hook);
        std::process::exit(1);
    }
    std::panic::set_hook(previous_panic_hook);
}
