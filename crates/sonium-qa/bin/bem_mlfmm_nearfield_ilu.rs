//! Deterministic MLFMM near-field ILU solve and full-operator residual gate.

use num_complex::Complex64;
use sonium_bem::core::assembly::{MlfmmNearFieldError, MlfmmSystem, NearFieldBlock};
use sonium_bem::core::solver::{
    GmresConfig, LinearOperator, MlfmmOperator, try_gmres_solve_mlfmm_batched_adjoint_with_ilu,
    try_gmres_solve_mlfmm_batched_with_ilu,
};
use sonium_bem::core::{Cluster, ClusterLevel};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "bem.mlfmm-nearfield-ilu.v1";

fn c(re: f64) -> Complex64 {
    Complex64::new(re, 0.0)
}

fn one_cluster_system(num_dofs: usize, dofs: Vec<usize>, coefficient: Complex64) -> MlfmmSystem {
    let mut level = ClusterLevel::new(1);
    level.clusters.push(Cluster::new([0.0, 0.0, 0.0]));
    let mut system = MlfmmSystem::new(num_dofs, 1, vec![level], vec![1]);
    system.leaf_dof_indices = vec![dofs];
    system.near_matrix.push(NearFieldBlock {
        source_cluster: 0,
        field_cluster: 0,
        coefficients: DenseMatrix::from_row_major_vec(1, 1, vec![coefficient]),
    });
    system
}

fn checked_layout_rejection_count() -> usize {
    let mut rejected = 0;

    let mut invalid_cluster = one_cluster_system(1, vec![0], c(1.0));
    invalid_cluster.near_matrix[0].source_cluster = 1;
    if matches!(
        invalid_cluster.try_extract_near_field_matrix(),
        Err(MlfmmNearFieldError::InvalidCluster { .. })
    ) {
        rejected += 1;
    }

    let invalid_shape = one_cluster_system(2, vec![0, 1], c(1.0));
    if matches!(
        invalid_shape.try_extract_near_field_matrix(),
        Err(MlfmmNearFieldError::BlockShape { .. })
    ) {
        rejected += 1;
    }

    let invalid_dof = one_cluster_system(1, vec![1], c(1.0));
    if matches!(
        invalid_dof.try_extract_near_field_matrix(),
        Err(MlfmmNearFieldError::GlobalDof { .. })
    ) {
        rejected += 1;
    }

    let nonfinite = one_cluster_system(1, vec![0], Complex64::new(f64::NAN, 0.0));
    if matches!(
        nonfinite.try_extract_near_field_matrix(),
        Err(MlfmmNearFieldError::NonFiniteCoefficient { .. })
    ) {
        rejected += 1;
    }

    rejected
}

fn main() {
    let started = Instant::now();
    let mut level = ClusterLevel::new(2);
    level.clusters.push(Cluster::new([0.0, 0.0, 0.0]));
    level.clusters.push(Cluster::new([1.0, 0.0, 0.0]));

    let mut system = MlfmmSystem::new(2, 1, vec![level], vec![1]);
    system.leaf_dof_indices = vec![vec![0], vec![1]];
    for (source_cluster, field_cluster, value) in [
        (0, 0, c(2.0)),
        (0, 1, c(0.3)),
        (1, 0, c(-0.4)),
        (1, 1, c(1.5)),
    ] {
        system.near_matrix.push(NearFieldBlock {
            source_cluster,
            field_cluster,
            coefficients: DenseMatrix::from_row_major_vec(1, 1, vec![value]),
        });
    }

    let operator = MlfmmOperator::new(system.clone());
    let rhs = vec![c(1.0), c(-0.5)];
    let config = GmresConfig {
        max_iterations: 20,
        restart: 4,
        tolerance: 1.0e-12,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
        min_restart: None,
        max_restart: None,
    };
    let solution = try_gmres_solve_mlfmm_batched_with_ilu(&system, &rhs, &config)
        .expect("checked MLFMM near-field ILU solve");
    let applied = operator.apply(&solution.x);
    let residual = applied
        .iter()
        .zip(rhs.iter())
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let adjoint_rhs = vec![c(0.75), c(-1.25)];
    let adjoint_solution =
        try_gmres_solve_mlfmm_batched_adjoint_with_ilu(&system, &adjoint_rhs, &config)
            .expect("checked MLFMM adjoint near-field ILU solve");
    let adjoint_residual = operator
        .apply_hermitian(&adjoint_solution.x)
        .iter()
        .zip(adjoint_rhs.iter())
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let probe = vec![c(-0.2), c(0.9)];
    let lhs = operator
        .apply(&probe)
        .iter()
        .zip(adjoint_rhs.iter())
        .map(|(a, b)| a.conj() * b)
        .sum::<Complex64>();
    let rhs_inner = probe
        .iter()
        .zip(operator.apply_hermitian(&adjoint_rhs).iter())
        .map(|(a, b)| a.conj() * b)
        .sum::<Complex64>();
    let inner_product_error = (lhs - rhs_inner).norm();
    let near = system.extract_near_field_matrix();
    let layout_rejections = checked_layout_rejection_count();
    let elapsed_ms = started.elapsed().as_secs_f64() * 1.0e3;

    if !solution.converged
        || !adjoint_solution.converged
        || !residual.is_finite()
        || residual > 1.0e-11
        || !adjoint_residual.is_finite()
        || adjoint_residual > 1.0e-11
        || !inner_product_error.is_finite()
        || inner_product_error > 1.0e-12
        || layout_rejections != 4
    {
        eprintln!(
            "MLFMM near-field ILU failed: converged={} solver_residual={:.6e} full_residual={:.6e}",
            solution.converged, solution.residual, residual
        );
        eprintln!("FAILED");
        std::process::exit(1);
    }

    let result = QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("converged", 1.0)
        .metric("solver_residual", solution.residual)
        .metric("full_operator_residual", residual)
        .metric("adjoint_residual", adjoint_residual)
        .metric("inner_product_error", inner_product_error)
        .metric("near_field_rows", near.nrows() as f64)
        .metric("near_field_cols", near.ncols() as f64)
        .maximum("full_operator_residual", 1.0e-11)
        .maximum("adjoint_residual", 1.0e-11)
        .maximum("inner_product_error", 1.0e-12)
        .formulation("MLFMM matrix-free action with exact leaf near-field ILU")
        .reference("independent 2x2 dense near-field action")
        .tolerance("forward/adjoint full-operator residual <= 1e-11; inner-product error <= 1e-12")
        .seed("bem-mlfmm-nearfield-ilu:cpu:v1")
        .elapsed_ms(elapsed_ms as u64);
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("valid QA result")
    );
    println!("SUCCESS");
}
