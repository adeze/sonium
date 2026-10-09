//! Independent forward/transpose/Hermitian action accuracy for H-matrix.

use num_complex::Complex64;
use sonium_bem::core::assembly::build_tbem_system_stabilized;
use sonium_bem::core::assembly::hmatrix::{HMatrix, HMatrixAction, build_hmatrix};
use sonium_bem::core::types::{
    BoundaryCondition, Cluster, Element, ElementProperty, ElementType, PhysicsParams,
};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "bem.hmatrix-operator-accuracy.v1";
const MAX_ERROR: f64 = 1.0e-8;

fn fixture() -> (Vec<Element>, Vec<[f64; 3]>, Vec<Cluster>) {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.5, 1.0, 0.0],
        [3.0, 0.0, 1.0],
    ];
    let element = |connectivity: Vec<usize>, center: [f64; 3], dof| Element {
        connectivity,
        element_type: ElementType::Tri3,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0; 3]; 3],
        center,
        area: 0.5,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses: vec![dof],
    };
    let elements = vec![
        element(vec![0, 1, 2], [0.5, 0.33, 0.0], 0),
        element(vec![1, 3, 2], [1.17, 0.33, 0.0], 1),
        element(vec![3, 4, 2], [1.83, 0.33, 0.0], 2),
        element(vec![3, 5, 4], [2.17, 0.33, 0.33], 3),
    ];

    let mut left = Cluster::new([0.83, 0.33, 0.0]);
    left.element_indices = vec![0, 1];
    left.num_elements = 2;
    left.radius = 1.0;
    left.near_clusters = vec![0];
    left.far_clusters = vec![1];
    let mut right = Cluster::new([2.0, 0.33, 0.17]);
    right.element_indices = vec![2, 3];
    right.num_elements = 2;
    right.radius = 1.0;
    right.near_clusters = vec![1];
    right.far_clusters = vec![0];
    (elements, nodes, vec![left, right])
}

fn probes() -> Vec<Vec<Complex64>> {
    vec![
        vec![
            Complex64::new(0.2, -0.1),
            Complex64::new(-0.4, 0.3),
            Complex64::new(0.7, 0.2),
            Complex64::new(0.1, -0.5),
        ],
        vec![
            Complex64::new(-0.3, 0.6),
            Complex64::new(0.8, 0.1),
            Complex64::new(0.5, -0.2),
            Complex64::new(-0.6, 0.4),
        ],
    ]
}

fn reference_outputs(
    dense: &sonium_solvers::dense::DenseMatrix<Complex64>,
    probes: &[Vec<Complex64>],
    action: HMatrixAction,
) -> Vec<Vec<Complex64>> {
    probes
        .iter()
        .map(|probe| {
            let mut output = vec![Complex64::new(0.0, 0.0); dense.nrows()];
            match action {
                HMatrixAction::Forward => dense.matvec_into(probe, &mut output),
                HMatrixAction::Transpose => dense.matvec_transpose_into(probe, &mut output),
                HMatrixAction::Hermitian => dense.matvec_hermitian_into(probe, &mut output),
            }
            output
        })
        .collect()
}

fn evaluate() -> Result<(f64, f64, f64, HMatrix), String> {
    let (elements, nodes, clusters) = fixture();
    let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);
    let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
    let dense = build_tbem_system_stabilized(&elements, &nodes, &physics, sign, 1.0).matrix;
    let hmatrix = build_hmatrix(&elements, &nodes, &clusters, &physics, 1.0e-10, 16);
    let probes = probes();
    let forward = hmatrix
        .relative_action_error(
            HMatrixAction::Forward,
            &probes,
            &reference_outputs(&dense, &probes, HMatrixAction::Forward),
        )
        .map_err(|error| error.to_string())?;
    let transpose = hmatrix
        .relative_action_error(
            HMatrixAction::Transpose,
            &probes,
            &reference_outputs(&dense, &probes, HMatrixAction::Transpose),
        )
        .map_err(|error| error.to_string())?;
    let hermitian = hmatrix
        .relative_action_error(
            HMatrixAction::Hermitian,
            &probes,
            &reference_outputs(&dense, &probes, HMatrixAction::Hermitian),
        )
        .map_err(|error| error.to_string())?;

    // Exercise the checked multi-RHS entry points as well as the scalar
    // accuracy gate. The columns intentionally reuse the independent probes
    // above so the block result can be compared without another assembly.
    let block_input =
        DenseMatrix::from_fn(dense.nrows(), probes.len(), |row, rhs| probes[rhs][row]);
    let block_forward = hmatrix
        .matvec_block_checked(&block_input)
        .map_err(|error| error.to_string())?;
    let block_transpose = hmatrix
        .matvec_transpose_block_checked(&block_input)
        .map_err(|error| error.to_string())?;
    let block_hermitian = hmatrix
        .matvec_hermitian_block_checked(&block_input)
        .map_err(|error| error.to_string())?;
    let mut block_error = 0.0_f64;
    for (rhs, probe) in probes.iter().enumerate() {
        let expected_forward =
            reference_outputs(&dense, std::slice::from_ref(probe), HMatrixAction::Forward)[0]
                .clone();
        let expected_transpose = reference_outputs(
            &dense,
            std::slice::from_ref(probe),
            HMatrixAction::Transpose,
        )[0]
        .clone();
        let expected_hermitian = reference_outputs(
            &dense,
            std::slice::from_ref(probe),
            HMatrixAction::Hermitian,
        )[0]
        .clone();
        for row in 0..dense.nrows() {
            block_error =
                block_error.max((block_forward[[row, rhs]] - expected_forward[row]).norm());
            block_error =
                block_error.max((block_transpose[[row, rhs]] - expected_transpose[row]).norm());
            block_error =
                block_error.max((block_hermitian[[row, rhs]] - expected_hermitian[row]).norm());
        }
    }
    if !block_error.is_finite() || block_error > 1.0e-12 {
        return Err(format!(
            "checked H-matrix block action mismatch: {block_error:e}"
        ));
    }

    Ok((forward, transpose, hermitian, hmatrix))
}

fn main() {
    let started = Instant::now();
    let result = std::panic::catch_unwind(evaluate);
    let passed = matches!(
        &result,
        Ok(Ok((forward, transpose, hermitian, _)))
            if *forward <= MAX_ERROR && *transpose <= MAX_ERROR && *hermitian <= MAX_ERROR
    );
    let qa = match result {
        Ok(Ok((forward, transpose, hermitian, hmatrix)))
            if forward <= MAX_ERROR && transpose <= MAX_ERROR && hermitian <= MAX_ERROR =>
        {
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent dense TBEM action")
                .reference_checksum("analytic/deterministic-fixture:bem-hmatrix-operator-accuracy-v1")
                .input_checksum("synthetic/deterministic-fixture:bem-hmatrix-operator-accuracy-v1")
                .dimension("4-DOF H-matrix; two probes; three actions")
                .formulation("forward, algebraic-transpose, and Hermitian action")
                .tolerance("each maximum relative action error <= 1e-8")
                .metric("forward_max_relative_error", forward)
                .metric("transpose_max_relative_error", transpose)
                .metric("hermitian_max_relative_error", hermitian)
                .metric("compression_ratio", hmatrix.compression_ratio)
                .elapsed_ms(started.elapsed().as_millis() as u64)
        }
        Ok(Ok((forward, transpose, hermitian, _))) => QaResult::failed(
            CASE_ID,
            format!(
                "independent action error exceeded budget: forward={forward:e}, transpose={transpose:e}, hermitian={hermitian:e}"
            ),
        )
        .reference("independent dense TBEM action")
        .dimension("4-DOF H-matrix; two probes; three actions")
        .formulation("forward, algebraic-transpose, and Hermitian action")
        .tolerance("each maximum relative action error <= 1e-8")
        .elapsed_ms(started.elapsed().as_millis() as u64),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .reference("independent dense TBEM action")
            .dimension("4-DOF H-matrix; two probes; three actions")
            .formulation("forward, algebraic-transpose, and Hermitian action")
            .tolerance("each maximum relative action error <= 1e-8")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(payload) => QaResult::failed(CASE_ID, format!("panic: {payload:?}"))
            .reference("independent dense TBEM action")
            .dimension("4-DOF H-matrix; two probes; three actions")
            .formulation("forward, algebraic-transpose, and Hermitian action")
            .tolerance("each maximum relative action error <= 1e-8")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    println!("{}", qa.to_json_line().expect("serialize QA result"));
    if !passed {
        std::process::exit(1);
    }
}
