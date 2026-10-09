//! Checked transport and rank-local Krylov contracts for distributed BEM.

use num_complex::Complex64;
use sonium_bem::core::distributed::{DistributedRowPlan, RankPartition};
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::{
    DistributedLinearOperator, InProcessVectorExchange, plan_distributed_rows,
    try_block_gmres_distributed, try_block_gmres_distributed_preconditioned, try_gmres_distributed,
    try_gmres_distributed_preconditioned,
};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::{BlockGmresConfig, GmresConfig};
use sonium_solvers::traits::{AdjointCapability, LinearOperator, Preconditioner};

struct DiagonalOperator {
    diagonal: Vec<Complex64>,
}

struct CoupledOperator;

impl LinearOperator<Complex64> for CoupledOperator {
    fn num_rows(&self) -> usize {
        2
    }

    fn num_cols(&self) -> usize {
        2
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        y[0] = x[0] + x[1];
        y[1] = x[0] + x[1];
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.apply_into(x, y);
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

fn two_row_plan() -> DistributedRowPlan {
    DistributedRowPlan {
        partitions: vec![
            RankPartition {
                rank: 0,
                element_indices: vec![0],
                dof_indices: vec![0],
            },
            RankPartition {
                rank: 1,
                element_indices: vec![1],
                dof_indices: vec![1],
            },
        ],
        num_active_elements: 2,
        num_dofs: 2,
    }
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
        for ((output, diagonal), value) in y.iter_mut().zip(&self.diagonal).zip(x) {
            *output = *diagonal * *value;
        }
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

fn plan() -> sonium_bem::DistributedRowPlan {
    let mut mesh = generate_icosphere_mesh(1.0, 0);
    for (index, element) in mesh.elements.iter_mut().enumerate() {
        element.dof_addresses = vec![index];
    }
    plan_distributed_rows(&mesh.elements, 2).expect("deterministic ownership plan")
}

fn config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 8,
        restart: 4,
        tolerance: 1.0e-12,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
        min_restart: None,
        max_restart: None,
    }
}

#[test]
fn rank_local_action_gathers_global_state_and_selects_owned_rows() {
    let plan = plan();
    let operator = DiagonalOperator {
        diagonal: vec![Complex64::new(2.0, 0.0); plan.num_dofs],
    };
    let exchange = InProcessVectorExchange::new(vec![Complex64::new(0.0, 0.0); plan.num_dofs]);
    let view = DistributedLinearOperator::try_new(&plan, 0, &operator, &exchange)
        .expect("rank-local operator");
    let partition = plan.local(0).expect("rank zero partition");
    let input = vec![Complex64::new(1.0, 0.0); partition.dof_indices.len()];
    let output = view.try_apply_local(&input).expect("rank-local action");
    assert_eq!(output, vec![Complex64::new(2.0, 0.0); input.len()]);
    let snapshot = exchange.snapshot();
    for &dof in &partition.dof_indices {
        assert_eq!(snapshot[dof], Complex64::new(1.0, 0.0));
    }
}

#[test]
fn distributed_gmres_solves_rank_owned_unknowns_with_loopback_exchange() {
    let plan = plan();
    let operator = DiagonalOperator {
        diagonal: vec![Complex64::new(2.0, 0.0); plan.num_dofs],
    };
    let exchange = InProcessVectorExchange::new(vec![Complex64::new(0.0, 0.0); plan.num_dofs]);
    let local_len = plan
        .local(0)
        .expect("rank zero partition")
        .dof_indices
        .len();
    let rhs = vec![Complex64::new(2.0, 0.0); local_len];
    let solution = try_gmres_distributed(&plan, 0, &operator, &exchange, &rhs, &config())
        .expect("distributed GMRES");
    assert!(solution.converged);
    assert!(solution.residual < 1.0e-12);
    assert!(
        solution
            .x
            .iter()
            .all(|value| (*value - Complex64::new(1.0, 0.0)).norm() < 1.0e-10)
    );
}

#[test]
fn distributed_preconditioned_gmres_preserves_local_preconditioner_contract() {
    let plan = plan();
    let operator = DiagonalOperator {
        diagonal: vec![Complex64::new(2.0, 0.0); plan.num_dofs],
    };
    let exchange = InProcessVectorExchange::new(vec![Complex64::new(0.0, 0.0); plan.num_dofs]);
    let local_len = plan
        .local(0)
        .expect("rank zero partition")
        .dof_indices
        .len();
    let rhs = vec![Complex64::new(2.0, 0.0); local_len];
    let solution = try_gmres_distributed_preconditioned(
        &plan,
        0,
        &operator,
        &ScalePreconditioner,
        &exchange,
        &rhs,
        &config(),
    )
    .expect("distributed preconditioned GMRES");
    assert!(solution.converged);
    assert!(solution.residual < 1.0e-12);
    assert!(
        solution
            .x
            .iter()
            .all(|value| (*value - Complex64::new(1.0, 0.0)).norm() < 1.0e-10)
    );
}

#[test]
fn distributed_block_gmres_solves_multiple_owned_rhs_columns() {
    let plan = plan();
    let operator = DiagonalOperator {
        diagonal: vec![Complex64::new(2.0, 0.0); plan.num_dofs],
    };
    let exchange = InProcessVectorExchange::new(vec![Complex64::new(0.0, 0.0); plan.num_dofs]);
    let local_len = plan
        .local(0)
        .expect("rank zero partition")
        .dof_indices
        .len();
    let rhs = DenseMatrix::from_fn(local_len, 2, |row, column| {
        Complex64::new(2.0 * (column + 1) as f64 + row as f64 * 0.0, 0.0)
    });
    let config = BlockGmresConfig {
        max_iterations: 8,
        restart: 4,
        tolerance: 1.0e-12,
        true_residual_check: true,
        exact_residual_check_interval: 1,
    };

    let solution = try_block_gmres_distributed(&plan, 0, &operator, &exchange, &rhs, &config)
        .expect("distributed block GMRES");
    assert!(solution.converged);
    assert_eq!(solution.x.ncols(), 2);
    assert!(
        solution
            .true_residuals
            .iter()
            .all(|residual| *residual < 1.0e-12)
    );
    for row in 0..local_len {
        assert!((solution.x[[row, 0]] - Complex64::new(1.0, 0.0)).norm() < 1.0e-10);
        assert!((solution.x[[row, 1]] - Complex64::new(2.0, 0.0)).norm() < 1.0e-10);
    }
}

#[test]
fn distributed_preconditioned_block_gmres_admits_and_solves_block_rhs() {
    let plan = plan();
    let operator = DiagonalOperator {
        diagonal: vec![Complex64::new(2.0, 0.0); plan.num_dofs],
    };
    let exchange = InProcessVectorExchange::new(vec![Complex64::new(0.0, 0.0); plan.num_dofs]);
    let preconditioner = ScalePreconditioner;
    let config = BlockGmresConfig::default();
    let wrong_rhs = DenseMatrix::zeros(
        plan.local(0)
            .expect("rank zero partition")
            .dof_indices
            .len()
            + 1,
        2,
    );
    assert!(
        try_block_gmres_distributed_preconditioned(
            &plan,
            0,
            &operator,
            &preconditioner,
            &exchange,
            &wrong_rhs,
            &config,
        )
        .is_err()
    );

    let local_len = plan
        .local(0)
        .expect("rank zero partition")
        .dof_indices
        .len();
    let rhs = DenseMatrix::from_fn(local_len, 2, |_, column| {
        Complex64::new(2.0 * (column + 1) as f64, 0.0)
    });
    let solution = try_block_gmres_distributed_preconditioned(
        &plan,
        0,
        &operator,
        &preconditioner,
        &exchange,
        &rhs,
        &config,
    )
    .expect("distributed preconditioned block GMRES");
    assert!(solution.converged);
    assert!(
        solution
            .true_residuals
            .iter()
            .all(|residual| *residual < 1.0e-12)
    );
}

#[test]
fn distributed_admission_rejects_wrong_exchange_and_rhs_shapes() {
    let plan = plan();
    let operator = DiagonalOperator {
        diagonal: vec![Complex64::new(1.0, 0.0); plan.num_dofs],
    };
    let wrong_exchange = InProcessVectorExchange::new(Vec::<Complex64>::new());
    assert!(
        DistributedLinearOperator::try_new(&plan, 0, &operator, &wrong_exchange)
            .expect("operator construction")
            .try_apply_local(&[Complex64::new(1.0, 0.0)])
            .is_err()
    );
    assert!(
        try_gmres_distributed(
            &plan,
            0,
            &operator,
            &InProcessVectorExchange::new(vec![Complex64::new(0.0, 0.0); plan.num_dofs]),
            &[],
            &config(),
        )
        .is_err()
    );
    let local_len = plan
        .local(0)
        .expect("rank zero partition")
        .dof_indices
        .len();
    assert!(
        try_gmres_distributed(
            &plan,
            0,
            &operator,
            &InProcessVectorExchange::new(Vec::<Complex64>::new()),
            &vec![Complex64::new(1.0, 0.0); local_len],
            &config(),
        )
        .is_err()
    );
    let non_finite = vec![Complex64::new(f64::NAN, 0.0); local_len];
    assert!(
        try_gmres_distributed(
            &plan,
            0,
            &operator,
            &InProcessVectorExchange::new(vec![Complex64::new(0.0, 0.0); plan.num_dofs]),
            &non_finite,
            &config(),
        )
        .is_err()
    );
}

#[test]
fn distributed_admission_rejects_operator_dimension_and_unknown_rank() {
    let plan = plan();
    let wrong_operator = DiagonalOperator {
        diagonal: vec![Complex64::new(1.0, 0.0); plan.num_dofs - 1],
    };
    let exchange = InProcessVectorExchange::new(vec![Complex64::new(0.0, 0.0); plan.num_dofs]);
    assert!(DistributedLinearOperator::try_new(&plan, 0, &wrong_operator, &exchange).is_err());
    let operator = DiagonalOperator {
        diagonal: vec![Complex64::new(1.0, 0.0); plan.num_dofs],
    };
    assert!(DistributedLinearOperator::try_new(&plan, 2, &operator, &exchange).is_err());
}

#[test]
fn distributed_transpose_and_hermitian_actions_use_the_same_collective() {
    let plan = plan();
    let operator = DiagonalOperator {
        diagonal: vec![Complex64::new(1.5, 0.0); plan.num_dofs],
    };
    let exchange = InProcessVectorExchange::new(vec![Complex64::new(0.0, 0.0); plan.num_dofs]);
    let view = DistributedLinearOperator::try_new(&plan, 1, &operator, &exchange)
        .expect("rank-local operator");
    let input = vec![Complex64::new(2.0, 0.0); view.local_dimension()];
    let transpose = view
        .try_apply_transpose_local(&input)
        .expect("transpose action");
    let hermitian = view
        .try_apply_hermitian_local(&input)
        .expect("Hermitian action");
    assert_eq!(transpose, vec![Complex64::new(3.0, 0.0); input.len()]);
    assert_eq!(hermitian, transpose);
}

#[test]
fn distributed_block_actions_gather_and_select_owned_rows() {
    let plan = plan();
    let diagonal = (0..plan.num_dofs)
        .map(|index| Complex64::new((index + 1) as f64, 0.0))
        .collect::<Vec<_>>();
    let operator = DiagonalOperator { diagonal };
    let global = (0..plan.num_dofs)
        .map(|index| Complex64::new((index + 1) as f64, 0.25 * index as f64))
        .collect::<Vec<_>>();
    let exchange = InProcessVectorExchange::new(global.clone());
    let rank = 1;
    let partition = plan.local(rank).expect("rank partition");
    let block = DenseMatrix::from_fn(partition.dof_indices.len(), 2, |row, column| {
        let value = global[partition.dof_indices[row]];
        if column == 0 {
            value
        } else {
            value * Complex64::new(-0.5, 0.75)
        }
    });
    let view = DistributedLinearOperator::try_new(&plan, rank, &operator, &exchange)
        .expect("distributed operator");

    let mut output = DenseMatrix::zeros(partition.dof_indices.len(), 2);
    view.try_apply_block(&block, &mut output)
        .expect("distributed forward block action");
    let expected = DenseMatrix::from_fn(partition.dof_indices.len(), 2, |row, column| {
        let dof = partition.dof_indices[row];
        let value = if column == 0 {
            global[dof]
        } else {
            global[dof] * Complex64::new(-0.5, 0.75)
        };
        operator.diagonal[dof] * value
    });
    assert_eq!(output.as_slice(), expected.as_slice());

    let mut transpose = DenseMatrix::zeros(partition.dof_indices.len(), 2);
    view.try_apply_transpose_block(&block, &mut transpose)
        .expect("distributed transpose block action");
    assert_eq!(transpose.as_slice(), expected.as_slice());

    let mut hermitian = DenseMatrix::zeros(partition.dof_indices.len(), 2);
    view.try_apply_hermitian_block(&block, &mut hermitian)
        .expect("distributed Hermitian block action");
    assert_eq!(hermitian.as_slice(), expected.as_slice());

    let wrong_rows = DenseMatrix::zeros(partition.dof_indices.len() + 1, 2);
    assert!(view.try_apply_block(&wrong_rows, &mut output).is_err());
    let non_finite = DenseMatrix::from_elem(
        partition.dof_indices.len(),
        1,
        Complex64::new(f64::NAN, 0.0),
    );
    let mut non_finite_output = DenseMatrix::zeros(partition.dof_indices.len(), 1);
    assert!(
        view.try_apply_block(&non_finite, &mut non_finite_output)
            .is_err()
    );
}

#[test]
fn distributed_global_residual_certificate_catches_remote_row_error() {
    let plan = two_row_plan();
    let exchange = InProcessVectorExchange::new(vec![Complex64::new(0.0, 0.0); 2]);
    let view = DistributedLinearOperator::try_new(&plan, 0, &CoupledOperator, &exchange)
        .expect("distributed coupled operator");
    // Rank zero's local equation is satisfied by x₀ = 1 while the remote row
    // remains wrong because rank one has not contributed its owned value.
    let residual = view
        .try_global_relative_residual(&[Complex64::new(1.0, 0.0)], &[Complex64::new(1.0, 0.0)])
        .expect("global residual certificate");
    assert!((residual - 1.0).abs() < 1.0e-12, "residual={residual}");
}

#[test]
fn distributed_gmres_rejects_local_convergence_with_remote_residual() {
    let plan = two_row_plan();
    let exchange = InProcessVectorExchange::new(vec![Complex64::new(0.0, 0.0); 2]);
    let error = try_gmres_distributed(
        &plan,
        0,
        &CoupledOperator,
        &exchange,
        &[Complex64::new(1.0, 0.0)],
        &config(),
    )
    .expect_err("remote residual must not be silently accepted");
    assert!(error.to_string().contains("global residual"));
}

#[test]
fn distributed_block_gmres_rejects_local_convergence_with_remote_residual() {
    let plan = two_row_plan();
    let exchange = InProcessVectorExchange::new(vec![Complex64::new(0.0, 0.0); 2]);
    let rhs = DenseMatrix::from_elem(1, 1, Complex64::new(1.0, 0.0));
    let config = BlockGmresConfig {
        max_iterations: 8,
        restart: 4,
        tolerance: 1.0e-12,
        true_residual_check: true,
        exact_residual_check_interval: 1,
    };
    let error = try_block_gmres_distributed(&plan, 0, &CoupledOperator, &exchange, &rhs, &config)
        .expect_err("remote block residual must not be silently accepted");
    assert!(error.to_string().contains("global residual"));
}

#[test]
fn distributed_global_residual_certificate_updates_block_true_residuals() {
    let plan = plan();
    let operator = DiagonalOperator {
        diagonal: vec![Complex64::new(2.0, 0.0); plan.num_dofs],
    };
    let exchange = InProcessVectorExchange::new(vec![Complex64::new(0.0, 0.0); plan.num_dofs]);
    let view = DistributedLinearOperator::try_new(&plan, 0, &operator, &exchange)
        .expect("distributed diagonal operator");
    let local_len = view.local_dimension();
    let solution = DenseMatrix::from_elem(local_len, 2, Complex64::new(1.0, 0.0));
    let rhs = DenseMatrix::from_elem(local_len, 2, Complex64::new(2.0, 0.0));
    let residuals = view
        .try_global_relative_residual_block(&solution, &rhs)
        .expect("global block residual certificate");
    assert_eq!(residuals.len(), 2);
    assert!(residuals.iter().all(|residual| *residual < 1.0e-12));
}
