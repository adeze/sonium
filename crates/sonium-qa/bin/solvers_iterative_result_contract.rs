//! Release-backed public result-contract checks for every checked iterative family.

use sonium_qa::QaResult;
use sonium_solvers::LinearOperator;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::{
    AccuracyStatus, BiCgstabConfig, BlockGmresConfig, CaCgConfig, CgConfig, CgsConfig, CsrBuilder,
    FgmresConfig, GmresConfig, GmresSolution, IdentityPreconditioner, IdrsConfig, IdrsStabConfig,
    KrylovOpCounts, MinresConfig, SolverError, SolverResultContract, SolverStatus, TfqmrConfig,
    try_bicgstab, try_bicgstab_pipelined, try_block_gmres, try_ca_cg, try_ca_pcg, try_cg, try_cgs,
    try_fgmres, try_gmres, try_gmres_pipelined, try_idrs, try_idrs_stab, try_minres, try_pcg,
    try_tfqmr,
};
use std::time::Instant;

use sonium_solvers::{
    BlockFgmresConfig, DeflationSubspace, GcroDrConfig, GcroDrSubspace, GmresDrConfig,
    RecycledSubspace, try_block_fgmres_preconditioned, try_gcro_dr, try_gmres_deflated,
    try_gmres_dr_preconditioned, try_gmres_recycled,
};

const CASE_ID: &str = "solvers.iterative-result-contract.v1";

struct Identity;

impl LinearOperator<f64> for Identity {
    fn num_rows(&self) -> usize {
        2
    }

    fn num_cols(&self) -> usize {
        2
    }

    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }

    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }
}

fn record_family_contract<T: SolverResultContract>(
    name: &'static str,
    result: Result<T, SolverError>,
) -> Result<(&'static str, f64), String> {
    let solution = result.map_err(|error| format!("{name} failed: {error}"))?;
    solution
        .validate_contract()
        .map_err(|error| format!("{name} result contract failed: {error}"))?;
    if solution.solver_status() == SolverStatus::Converged
        && solution.accuracy_status() != AccuracyStatus::Attained
    {
        return Err(format!(
            "{name} reported contradictory converged metadata: status={:?}, accuracy={:?}",
            solution.solver_status(),
            solution.accuracy_status()
        ));
    }
    Ok((name, 1.0))
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let identity = Identity;
    let scalar = try_gmres(
        &identity,
        &[1.0, -2.0],
        &GmresConfig {
            max_iterations: 4,
            restart: 2,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
            ..Default::default()
        },
    )
    .map_err(|error| format!("scalar GMRES failed: {error}"))?;
    scalar
        .validate_contract()
        .map_err(|error| format!("scalar result contract failed: {error}"))?;
    let negative_diagnostic_rejected = {
        let candidate = GmresSolution {
            x: scalar.x.clone(),
            iterations: scalar.iterations,
            restarts: scalar.restarts,
            residual: -1.0,
            estimated_residual: scalar.estimated_residual,
            backward_error: scalar.backward_error,
            attainable_accuracy: scalar.attainable_accuracy,
            converged: scalar.converged,
            status: scalar.status,
            op_counts: KrylovOpCounts::default(),
        };
        candidate.validate_contract().is_err()
    };

    let mut builder = CsrBuilder::new(2, 2);
    builder.add_row_entries([(0, 1.0)].into_iter());
    builder.add_row_entries([(1, 1.0)].into_iter());
    let csr = builder.finish();
    let rhs = DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 2.0, -2.0, 0.5]);
    let block = try_block_gmres(
        &csr,
        &rhs,
        &BlockGmresConfig {
            max_iterations: 4,
            restart: 2,
            tolerance: 1.0e-12,
            ..Default::default()
        },
    )
    .map_err(|error| format!("block GMRES failed: {error}"))?;
    block
        .validate_contract()
        .map_err(|error| format!("block result contract failed: {error}"))?;

    let adaptive_rhs = [1.0_f64, -2.0];
    let adaptive_config = GmresConfig {
        max_iterations: 8,
        restart: 4,
        tolerance: 1.0e-10,
        ..GmresConfig::default()
    };
    let adaptive_dr = try_gmres_dr_preconditioned(
        &identity,
        &IdentityPreconditioner,
        &adaptive_rhs,
        None,
        &GmresDrConfig {
            gmres: adaptive_config.clone(),
            num_deflation_vectors: 1,
        },
    );
    let mut gcro_subspace = GcroDrSubspace::new(1);
    let adaptive_gcro = try_gcro_dr(
        &identity,
        &adaptive_rhs,
        None,
        &mut gcro_subspace,
        &GcroDrConfig::from_gmres(&adaptive_config),
    );
    let mut recycled_subspace = RecycledSubspace::new(1);
    let adaptive_recycled = try_gmres_recycled(
        &identity,
        &adaptive_rhs,
        None,
        &mut recycled_subspace,
        &adaptive_config,
    );
    let deflation = DeflationSubspace::try_new(vec![vec![1.0, 0.0]], &identity)
        .map_err(|error| format!("deflation setup failed: {error}"))?;
    let adaptive_deflated =
        try_gmres_deflated(&identity, &deflation, &adaptive_rhs, None, &adaptive_config);
    let adaptive_block_rhs = DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 2.0, -2.0, 0.5]);
    let adaptive_block = try_block_fgmres_preconditioned(
        &identity,
        &adaptive_block_rhs,
        &IdentityPreconditioner,
        None,
        &BlockFgmresConfig {
            max_iterations: 4,
            restart: 2,
            tolerance: 1.0e-10,
            true_residual_check: true,
            exact_residual_check_interval: 1,
        },
    );
    let adaptive_metrics = [
        record_family_contract("gmres_dr_contract", adaptive_dr)?,
        record_family_contract("gcro_dr_contract", adaptive_gcro)?,
        record_family_contract("gmres_recycled_contract", adaptive_recycled)?,
        record_family_contract("gmres_deflated_contract", adaptive_deflated)?,
        record_family_contract("block_fgmres_contract", adaptive_block)?,
    ];

    let family_metrics = [
        record_family_contract(
            "gmres_family_contract",
            try_gmres(&identity, &[1.0, -2.0], &GmresConfig::default()),
        )?,
        record_family_contract(
            "pipelined_gmres_family_contract",
            try_gmres_pipelined(
                &identity,
                &IdentityPreconditioner,
                &[1.0, -2.0],
                None,
                &GmresConfig::default(),
            ),
        )?,
        record_family_contract(
            "fgmres_family_contract",
            try_fgmres(&identity, &[1.0, -2.0], &FgmresConfig::default()),
        )?,
        record_family_contract(
            "bicgstab_family_contract",
            try_bicgstab(&identity, &[1.0, -2.0], &BiCgstabConfig::default()),
        )?,
        record_family_contract(
            "pipelined_bicgstab_family_contract",
            try_bicgstab_pipelined(&identity, &[1.0, -2.0], &BiCgstabConfig::default()),
        )?,
        record_family_contract(
            "cg_family_contract",
            try_cg(&identity, &[1.0, -2.0], &CgConfig::default()),
        )?,
        record_family_contract(
            "pcg_family_contract",
            try_pcg(
                &identity,
                &IdentityPreconditioner,
                &[1.0, -2.0],
                &CgConfig::default(),
            ),
        )?,
        record_family_contract(
            "cgs_family_contract",
            try_cgs(&identity, &[1.0, -2.0], &CgsConfig::default()),
        )?,
        record_family_contract(
            "idrs_family_contract",
            try_idrs(&identity, &[1.0, -2.0], &IdrsConfig::default()),
        )?,
        record_family_contract(
            "idrs_stab_family_contract",
            try_idrs_stab(&identity, &[1.0, -2.0], &IdrsStabConfig::default()),
        )?,
        record_family_contract(
            "minres_family_contract",
            try_minres(&identity, &[1.0, -2.0], &MinresConfig::default()),
        )?,
        record_family_contract(
            "tfqmr_family_contract",
            try_tfqmr(&identity, &[1.0, -2.0], &TfqmrConfig::default()),
        )?,
        record_family_contract(
            "ca_cg_family_contract",
            try_ca_cg(&identity, &[1.0, -2.0], &CaCgConfig::default()),
        )?,
        record_family_contract(
            "ca_pcg_family_contract",
            try_ca_pcg(
                &identity,
                &[1.0, -2.0],
                &IdentityPreconditioner,
                &CaCgConfig::default(),
            ),
        )?,
    ];

    let mut metrics = vec![
        (
            "scalar_status_converged",
            (scalar.solver_status() == sonium_solvers::SolverStatus::Converged) as u8 as f64,
        ),
        (
            "scalar_accuracy_attained",
            (scalar.accuracy_status() == sonium_solvers::AccuracyStatus::Attained) as u8 as f64,
        ),
        (
            "negative_diagnostic_rejected",
            negative_diagnostic_rejected as u8 as f64,
        ),
        (
            "block_status_converged",
            (block.solver_status() == sonium_solvers::SolverStatus::Converged) as u8 as f64,
        ),
        (
            "block_accuracy_attained",
            (block.accuracy_status() == sonium_solvers::AccuracyStatus::Attained) as u8 as f64,
        ),
    ];
    metrics.extend(family_metrics);
    metrics.extend(adaptive_metrics);
    Ok(metrics)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("checked scalar/block and adaptive/recycled/deflated wrapper result status and accuracy contract")
                .reference_checksum("analytic/deterministic-fixture:iterative-result-contract-v1")
                .input_checksum("analytic/deterministic-fixture:iterative-result-contract-v1")
                .tolerance("all status/accuracy and family-contract metrics equal one")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value);
            }
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            eprintln!("ITERATIVE RESULT CONTRACT FAILED");
            std::process::exit(1);
        }
    }
}
