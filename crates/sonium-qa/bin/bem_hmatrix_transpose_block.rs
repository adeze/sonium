//! Checked H-matrix algebraic-transpose block-action evidence.

use num_complex::Complex64;
use sonium_bem::core::assembly::hmatrix::build_hmatrix;
use sonium_bem::core::types::{
    BoundaryCondition, Cluster, Element, ElementProperty, ElementType, PhysicsParams,
};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::traits::LinearOperator;
use std::time::Instant;

const CASE_ID: &str = "bem.hmatrix-transpose-block.v1";

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

fn main() {
    let started = Instant::now();
    let result = std::panic::catch_unwind(|| {
        let (elements, nodes, clusters) = fixture();
        let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);
        let operator = build_hmatrix(&elements, &nodes, &clusters, &physics, 1.0e-8, 4);
        let input = DenseMatrix::from_row_major_vec(
            4,
            2,
            vec![
                Complex64::new(0.2, -0.1),
                Complex64::new(-0.4, 0.3),
                Complex64::new(0.7, 0.2),
                Complex64::new(0.1, -0.5),
                Complex64::new(-0.3, 0.6),
                Complex64::new(0.8, 0.1),
                Complex64::new(0.5, -0.2),
                Complex64::new(-0.6, 0.4),
            ],
        );
        let block = operator.matvec_transpose_block(&input);
        let mut maximum_error: f64 = 0.0;
        for rhs in 0..input.ncols() {
            let column = (0..input.nrows())
                .map(|row| input[[row, rhs]])
                .collect::<Vec<_>>();
            let scalar = operator.apply_transpose(&column);
            for row in 0..input.nrows() {
                maximum_error = maximum_error.max((block[[row, rhs]] - scalar[row]).norm());
            }
        }
        assert!(
            maximum_error <= 1.0e-10,
            "block/scalar transpose error {maximum_error:e}"
        );
        maximum_error
    });
    let passed = result.is_ok();
    let qa = match result {
        Ok(maximum_error) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .reference("scalar H-matrix algebraic-transpose action")
            .reference_checksum("synthetic/deterministic-fixture:bem-hmatrix-transpose-block-v1")
            .input_checksum("synthetic/deterministic-fixture:bem-hmatrix-transpose-block-v1")
            .dimension("4-DOF H-matrix; two transpose RHS columns")
            .formulation("field-cluster partition batched dense/low-rank transpose")
            .tolerance("block/scalar transpose error <= 1e-10")
            .metric("maximum_block_scalar_error", maximum_error)
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(payload) => QaResult::failed(CASE_ID, format!("panic: {payload:?}")),
    };
    println!("{}", qa.to_json_line().expect("serialize QA result"));
    if !passed {
        std::process::exit(1);
    }
}
