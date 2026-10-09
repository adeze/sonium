use num_complex::Complex64;
use sonium_bem::core::assembly::mlfmm::{
    LevelMatrices, MlfmmRowActionError, MlfmmSystem, NearFieldBlock as MlfmmNearFieldBlock,
};
use sonium_bem::core::assembly::slfmm::{
    NearFieldBlock as SlfmmNearFieldBlock, SlfmmRowActionError, SlfmmSystem,
};
use sonium_solvers::dense::DenseMatrix;

fn non_finite_block() -> DenseMatrix<Complex64> {
    DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(f64::INFINITY, 0.0)])
}

fn slfmm_with_non_finite_near_field() -> SlfmmSystem {
    SlfmmSystem {
        near_matrix: vec![SlfmmNearFieldBlock {
            source_cluster: 0,
            field_cluster: 0,
            coefficients: non_finite_block(),
        }],
        t_matrices: vec![DenseMatrix::zeros(0, 1)],
        t_vector: Vec::new(),
        d_matrices: Vec::new(),
        s_matrices: vec![DenseMatrix::zeros(1, 0)],
        rhs: Vec::new(),
        num_dofs: 1,
        num_sphere_points: 0,
        num_expansion_terms: 0,
        num_clusters: 1,
        cluster_dof_indices: vec![vec![0]],
    }
}

fn mlfmm_with_non_finite_near_field() -> MlfmmSystem {
    let mut cluster = sonium_bem::core::Cluster::new([0.0, 0.0, 0.0]);
    cluster.element_indices = vec![0];
    cluster.num_elements = 1;
    let mut level = sonium_bem::core::ClusterLevel::new(1);
    level.clusters.push(cluster);
    let mut system = MlfmmSystem::new(1, 1, vec![level], vec![1]);
    system.leaf_dof_indices = vec![vec![0]];
    system.t_matrices = vec![LevelMatrices {
        level: 0,
        num_clusters: 1,
        matrices: vec![DenseMatrix::from_row_major_vec(
            1,
            1,
            vec![Complex64::new(1.0, 0.0)],
        )],
    }];
    system.d_matrices = vec![Vec::new()];
    system.s_matrices = vec![LevelMatrices {
        level: 0,
        num_clusters: 1,
        matrices: vec![DenseMatrix::from_row_major_vec(
            1,
            1,
            vec![Complex64::new(1.0, 0.0)],
        )],
    }];
    system.near_matrix.push(MlfmmNearFieldBlock {
        source_cluster: 0,
        field_cluster: 0,
        coefficients: non_finite_block(),
    });
    system
}

#[test]
fn slfmm_checked_row_actions_reject_non_finite_scalar_and_block_outputs() {
    let system = slfmm_with_non_finite_near_field();
    let input = [Complex64::new(1.0, 0.0)];
    assert!(matches!(
        system.apply_rows_checked(&input, &[0]),
        Err(SlfmmRowActionError::NonFiniteOutput {
            operation: "forward",
            ..
        })
    ));
    assert!(matches!(
        system.apply_transpose_rows_checked(&input, &[0]),
        Err(SlfmmRowActionError::NonFiniteOutput {
            operation: "transpose",
            ..
        })
    ));

    let mut block_input = DenseMatrix::zeros(1, 2);
    block_input[[0, 0]] = Complex64::new(1.0, 0.0);
    block_input[[0, 1]] = Complex64::new(-2.0, 0.5);
    assert!(matches!(
        system.apply_rows_block_checked(&block_input, &[0]),
        Err(SlfmmRowActionError::NonFiniteOutput {
            operation: "forward block",
            ..
        })
    ));
    assert!(matches!(
        system.apply_transpose_rows_block_checked(&block_input, &[0]),
        Err(SlfmmRowActionError::NonFiniteOutput {
            operation: "transpose block",
            ..
        })
    ));
}

#[test]
fn mlfmm_checked_row_actions_reject_non_finite_scalar_and_block_outputs() {
    let system = mlfmm_with_non_finite_near_field();
    let input = [Complex64::new(1.0, 0.0)];
    assert!(matches!(
        system.apply_rows_checked(&input, &[0]),
        Err(MlfmmRowActionError::NonFiniteOutput {
            operation: "forward",
            ..
        })
    ));
    assert!(matches!(
        system.apply_transpose_rows_checked(&input, &[0]),
        Err(MlfmmRowActionError::NonFiniteOutput {
            operation: "transpose",
            ..
        })
    ));

    let mut block_input = DenseMatrix::zeros(1, 2);
    block_input[[0, 0]] = Complex64::new(1.0, 0.0);
    block_input[[0, 1]] = Complex64::new(-2.0, 0.5);
    assert!(matches!(
        system.apply_rows_block_checked(&block_input, &[0]),
        Err(MlfmmRowActionError::NonFiniteOutput {
            operation: "forward block",
            ..
        })
    ));
    assert!(matches!(
        system.apply_transpose_rows_block_checked(&block_input, &[0]),
        Err(MlfmmRowActionError::NonFiniteOutput {
            operation: "transpose block",
            ..
        })
    ));
}
