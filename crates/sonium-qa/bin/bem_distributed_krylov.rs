//! Deterministic rank-local distributed BEM GMRES contract.

use num_complex::Complex64;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::{
    DistributedRowPlan, InProcessVectorExchange, plan_distributed_rows, try_gmres_distributed,
    try_gmres_distributed_preconditioned,
};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::GmresConfig;
use sonium_solvers::traits::{LinearOperator, Preconditioner};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.distributed-krylov.v1";
const MAX_RESIDUAL: f64 = 1.0e-12;
const MAX_SOLUTION_ERROR: f64 = 1.0e-10;

struct DiagonalOperator {
    diagonal: Vec<Complex64>,
}

struct ScalePreconditioner;

impl Preconditioner<Complex64> for ScalePreconditioner {
    fn apply_into(&self, r: &[Complex64], y: &mut [Complex64]) {
        for (output, value) in y.iter_mut().zip(r) {
            *output = *value * Complex64::new(0.5, 0.0);
        }
    }
}

impl LinearOperator<Complex64> for DiagonalOperator {
    fn num_rows(&self) -> usize {
        self.diagonal.len()
    }

    fn num_cols(&self) -> usize {
        self.diagonal.len()
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        for ((output, diagonal), value) in y.iter_mut().zip(&self.diagonal).zip(x) {
            *output = *diagonal * *value;
        }
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.apply_into(x, y);
    }
}

fn fixture() -> DistributedRowPlan {
    let mut mesh = generate_icosphere_mesh(1.0, 0);
    for (index, element) in mesh.elements.iter_mut().enumerate() {
        element.dof_addresses = vec![index];
    }
    plan_distributed_rows(&mesh.elements, 2)
        .map_err(|error| error.to_string())
        .unwrap()
}

fn run() -> Result<(f64, f64, usize, usize), String> {
    let plan = fixture();
    let operator = DiagonalOperator {
        diagonal: vec![Complex64::new(2.0, 0.0); plan.num_dofs],
    };
    let exchange = InProcessVectorExchange::new(vec![Complex64::new(0.0, 0.0); plan.num_dofs]);
    let partition = plan
        .local(0)
        .ok_or_else(|| "rank zero partition missing".to_string())?;
    let rhs = vec![Complex64::new(2.0, 0.0); partition.dof_indices.len()];
    let config = GmresConfig {
        max_iterations: 8,
        restart: 4,
        tolerance: MAX_RESIDUAL,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
        min_restart: None,
        max_restart: None,
    };
    let solution = try_gmres_distributed(&plan, 0, &operator, &exchange, &rhs, &config)
        .map_err(|error| error.to_string())?;
    let preconditioned = try_gmres_distributed_preconditioned(
        &plan,
        0,
        &operator,
        &ScalePreconditioner,
        &exchange,
        &rhs,
        &config,
    )
    .map_err(|error| error.to_string())?;
    let block_input =
        DenseMatrix::from_elem(partition.dof_indices.len(), 2, Complex64::new(1.0, 0.0));
    let mut block_output = DenseMatrix::zeros(partition.dof_indices.len(), 2);
    let block_view = sonium_bem::DistributedLinearOperator::try_new(&plan, 0, &operator, &exchange)
        .map_err(|error| error.to_string())?;
    block_view
        .try_apply_block(&block_input, &mut block_output)
        .map_err(|error| error.to_string())?;
    if block_output
        .as_slice()
        .iter()
        .any(|value| (*value - Complex64::new(2.0, 0.0)).norm() > MAX_SOLUTION_ERROR)
    {
        return Err("distributed block action mismatch".to_string());
    }
    let global_residual = block_view
        .try_global_relative_residual(&solution.x, &rhs)
        .map_err(|error| error.to_string())?;

    let solution_error = solution
        .x
        .iter()
        .map(|value| (*value - Complex64::new(1.0, 0.0)).norm())
        .fold(0.0, f64::max);
    let snapshot = exchange.snapshot();
    let snapshot_error = partition
        .dof_indices
        .iter()
        .map(|&dof| (snapshot[dof] - Complex64::new(1.0, 0.0)).norm())
        .fold(0.0, f64::max);
    if solution.residual > MAX_RESIDUAL
        || preconditioned.residual > MAX_RESIDUAL
        || global_residual > MAX_RESIDUAL
        || solution_error > MAX_SOLUTION_ERROR
    {
        return Err(format!(
            "distributed GMRES residual/error exceeded gates: {:.3e}/{:.3e}/{:.3e}/{:.3e}",
            solution.residual, preconditioned.residual, global_residual, solution_error
        ));
    }
    if snapshot_error > MAX_SOLUTION_ERROR {
        return Err(format!("exchange snapshot error {:.3e}", snapshot_error));
    }
    let wrong_exchange = InProcessVectorExchange::new(Vec::<Complex64>::new());
    let rejected_exchange =
        sonium_bem::DistributedLinearOperator::try_new(&plan, 0, &operator, &wrong_exchange)
            .map_err(|error| error.to_string())?
            .try_apply_local(&vec![Complex64::new(1.0, 0.0); partition.dof_indices.len()])
            .is_err();
    let rejected_rhs = try_gmres_distributed(
        &plan,
        0,
        &operator,
        &InProcessVectorExchange::new(vec![Complex64::new(0.0, 0.0); plan.num_dofs]),
        &[],
        &config,
    )
    .is_err();
    if !rejected_exchange || !rejected_rhs {
        return Err("malformed distributed packets were accepted".to_string());
    }
    Ok((
        solution
            .residual
            .max(preconditioned.residual)
            .max(global_residual),
        solution_error,
        solution.iterations.max(preconditioned.iterations),
        2,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok((residual, solution_error, iterations, rejections))) => QaResult::passed(CASE_ID)
            .metric("relative_residual", residual)
            .metric("maximum_solution_error", solution_error)
            .metric("iterations", iterations as f64)
            .metric("typed_rejections", rejections as f64)
            .maximum("relative_residual", MAX_RESIDUAL)
            .maximum("maximum_solution_error", MAX_SOLUTION_ERROR)
            .minimum("typed_rejections", 2.0),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error),
        Err(_) => QaResult::failed(CASE_ID, "panic during distributed Krylov validation"),
    }
    .input_checksum("analytic/deterministic-fixture:bem-distributed-krylov-v1")
    .reference_checksum("analytic/deterministic-fixture:bem-distributed-krylov-v1")
    .dimension("2-rank rank-local Complex64 GMRES")
    .reference("loopback all-gather and independent diagonal solution")
    .seed("bem-distributed-krylov:cpu:v1")
    .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
    let passed = matches!(result.status, QaStatus::Passed);
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !passed {
        std::process::exit(1);
    }
}
