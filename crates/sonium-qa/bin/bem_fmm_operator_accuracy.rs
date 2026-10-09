//! Independent forward/transpose/Hermitian action accuracy for SLFMM/MLFMM.
//!
//! This is a tiny deterministic CPU sentinel. It compares both matrix-free
//! backends with an independently assembled dense stabilized TBEM matrix and
//! does not launch a solver or simulator.

use num_complex::Complex64;
use sonium_bem::core::assembly::mlfmm::{
    LevelMatrices, MlfmmRowActionError, MlfmmSystem, NearFieldBlock as MlfmmNearFieldBlock,
};
use sonium_bem::core::assembly::slfmm::{
    NearFieldBlock as SlfmmNearFieldBlock, SlfmmRowActionError, SlfmmSystem,
};
use sonium_bem::core::assembly::{
    FmmAction, build_cluster_tree, build_mlfmm_system_stabilized_checked,
    build_slfmm_system_stabilized_checked, build_tbem_system_stabilized,
};
use sonium_bem::core::types::{
    BoundaryCondition, Cluster, ClusterLevel, Element, ElementProperty, ElementType, PhysicsParams,
};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "bem.fmm-operator-accuracy.v1";
const MAX_ERROR: f64 = 1.0e-6;

fn fixture() -> (Vec<Element>, Vec<[f64; 3]>, Vec<Cluster>) {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [4.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [4.0, 1.0, 0.0],
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
        element(vec![0, 1, 2], [1.0 / 3.0, 1.0 / 3.0, 0.0], 0),
        element(vec![3, 4, 5], [13.0 / 3.0, 1.0 / 3.0, 0.0], 1),
    ];
    let mut left = Cluster::new(elements[0].center);
    left.element_indices = vec![0];
    left.num_elements = 1;
    left.near_clusters = vec![0];
    left.far_clusters = vec![1];
    let mut right = Cluster::new(elements[1].center);
    right.element_indices = vec![1];
    right.num_elements = 1;
    right.near_clusters = vec![1];
    right.far_clusters = vec![0];
    (elements, nodes, vec![left, right])
}

fn probes() -> Vec<Vec<Complex64>> {
    vec![
        vec![Complex64::new(0.7, -0.2), Complex64::new(-0.3, 0.4)],
        vec![Complex64::new(-0.1, 0.6), Complex64::new(0.9, 0.1)],
    ]
}

fn reference_outputs(
    dense: &DenseMatrix<Complex64>,
    probes: &[Vec<Complex64>],
    action: FmmAction,
) -> Vec<Vec<Complex64>> {
    probes
        .iter()
        .map(|probe| {
            let mut output = vec![Complex64::new(0.0, 0.0); dense.nrows()];
            match action {
                FmmAction::Forward => dense.matvec_into(probe, &mut output),
                FmmAction::Transpose => dense.matvec_transpose_into(probe, &mut output),
                FmmAction::Hermitian => dense.matvec_hermitian_into(probe, &mut output),
            }
            output
        })
        .collect()
}

fn non_finite_output_rejections() -> Result<[f64; 4], String> {
    let coefficients =
        DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(f64::INFINITY, 0.0)]);
    let slfmm = SlfmmSystem {
        near_matrix: vec![SlfmmNearFieldBlock {
            source_cluster: 0,
            field_cluster: 0,
            coefficients: coefficients.clone(),
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
    };
    let mut cluster = Cluster::new([0.0, 0.0, 0.0]);
    cluster.element_indices = vec![0];
    cluster.num_elements = 1;
    let mut level = ClusterLevel::new(1);
    level.clusters.push(cluster);
    let mut mlfmm = MlfmmSystem::new(1, 1, vec![level], vec![1]);
    mlfmm.leaf_dof_indices = vec![vec![0]];
    mlfmm.t_matrices = vec![LevelMatrices {
        level: 0,
        num_clusters: 1,
        matrices: vec![DenseMatrix::from_row_major_vec(
            1,
            1,
            vec![Complex64::new(1.0, 0.0)],
        )],
    }];
    mlfmm.s_matrices = vec![LevelMatrices {
        level: 0,
        num_clusters: 1,
        matrices: vec![DenseMatrix::from_row_major_vec(
            1,
            1,
            vec![Complex64::new(1.0, 0.0)],
        )],
    }];
    mlfmm.d_matrices = vec![Vec::new()];
    mlfmm.near_matrix.push(MlfmmNearFieldBlock {
        source_cluster: 0,
        field_cluster: 0,
        coefficients,
    });
    let input = [Complex64::new(1.0, 0.0)];
    let mut block_input = DenseMatrix::zeros(1, 1);
    block_input[[0, 0]] = input[0];

    let slfmm_rejections = [
        matches!(
            slfmm.apply_rows_checked(&input, &[0]),
            Err(SlfmmRowActionError::NonFiniteOutput { .. })
        ),
        matches!(
            slfmm.apply_transpose_rows_checked(&input, &[0]),
            Err(SlfmmRowActionError::NonFiniteOutput { .. })
        ),
        matches!(
            slfmm.apply_rows_block_checked(&block_input, &[0]),
            Err(SlfmmRowActionError::NonFiniteOutput { .. })
        ),
        matches!(
            slfmm.apply_transpose_rows_block_checked(&block_input, &[0]),
            Err(SlfmmRowActionError::NonFiniteOutput { .. })
        ),
    ]
    .into_iter()
    .filter(|rejected| *rejected)
    .count() as f64;
    let mlfmm_rejections = [
        matches!(
            mlfmm.apply_rows_checked(&input, &[0]),
            Err(MlfmmRowActionError::NonFiniteOutput { .. })
        ),
        matches!(
            mlfmm.apply_transpose_rows_checked(&input, &[0]),
            Err(MlfmmRowActionError::NonFiniteOutput { .. })
        ),
        matches!(
            mlfmm.apply_rows_block_checked(&block_input, &[0]),
            Err(MlfmmRowActionError::NonFiniteOutput { .. })
        ),
        matches!(
            mlfmm.apply_transpose_rows_block_checked(&block_input, &[0]),
            Err(MlfmmRowActionError::NonFiniteOutput { .. })
        ),
    ]
    .into_iter()
    .filter(|rejected| *rejected)
    .count() as f64;
    Ok([
        slfmm_rejections,
        mlfmm_rejections,
        f64::from((slfmm_rejections == 4.0) as u8),
        f64::from((mlfmm_rejections == 4.0) as u8),
    ])
}

fn evaluate() -> Result<[f64; 10], String> {
    let (elements, nodes, clusters) = fixture();
    let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
    let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
    let dense = build_tbem_system_stabilized(&elements, &nodes, &physics, sign, 1.0).matrix;
    let probes = probes();
    let cluster_levels = build_cluster_tree(&elements, 1, &physics);
    let slfmm = build_slfmm_system_stabilized_checked(
        &elements, &nodes, &clusters, &physics, 16, 32, 20, sign, 1.0,
    )
    .map_err(|error| error.to_string())?;
    let mlfmm = build_mlfmm_system_stabilized_checked(
        &elements,
        &nodes,
        cluster_levels,
        &physics,
        sign,
        1.0,
    )
    .map_err(|error| error.to_string())?;

    let mut errors = [0.0; 6];
    for (offset, action) in [
        FmmAction::Forward,
        FmmAction::Transpose,
        FmmAction::Hermitian,
    ]
    .into_iter()
    .enumerate()
    {
        let references = reference_outputs(&dense, &probes, action);
        errors[offset] = slfmm
            .relative_action_error(action, &probes, &references)
            .map_err(|error| error.to_string())?;
        errors[offset + 3] = mlfmm
            .relative_action_error(action, &probes, &references)
            .map_err(|error| error.to_string())?;
    }

    let block_input =
        DenseMatrix::from_fn(dense.nrows(), probes.len(), |row, rhs| probes[rhs][row]);
    let slfmm_hermitian_references: Vec<_> = probes
        .iter()
        .map(|probe| {
            slfmm
                .apply_hermitian_checked(probe)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    let mlfmm_hermitian_references: Vec<_> = probes
        .iter()
        .map(|probe| {
            mlfmm
                .apply_hermitian_checked(probe)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    let slfmm_block = slfmm
        .apply_hermitian_block_checked(&block_input)
        .map_err(|error| error.to_string())?;
    let mlfmm_block = mlfmm
        .apply_hermitian_block_checked(&block_input)
        .map_err(|error| error.to_string())?;
    let rows = [1, 0, 1];
    let slfmm_rows = slfmm
        .apply_hermitian_rows_block_checked(&block_input, &rows)
        .map_err(|error| error.to_string())?;
    let mlfmm_rows = mlfmm
        .apply_hermitian_rows_block_checked(&block_input, &rows)
        .map_err(|error| error.to_string())?;
    let mut block_error = 0.0_f64;
    for rhs in 0..probes.len() {
        let slfmm_reference = &slfmm_hermitian_references[rhs];
        let mlfmm_reference = &mlfmm_hermitian_references[rhs];
        for row in 0..dense.nrows() {
            block_error = block_error.max((slfmm_block[[row, rhs]] - slfmm_reference[row]).norm());
            block_error = block_error.max((mlfmm_block[[row, rhs]] - mlfmm_reference[row]).norm());
        }
        for (selected, &row) in rows.iter().enumerate() {
            block_error =
                block_error.max((slfmm_rows[[selected, rhs]] - slfmm_reference[row]).norm());
            block_error =
                block_error.max((mlfmm_rows[[selected, rhs]] - mlfmm_reference[row]).norm());
        }
    }
    if !block_error.is_finite() || block_error > 1.0e-12 {
        return Err(format!(
            "checked FMM Hermitian block mismatch: {block_error:e}"
        ));
    }

    let output_rejections = non_finite_output_rejections()?;
    Ok([
        errors[0],
        errors[1],
        errors[2],
        errors[3],
        errors[4],
        errors[5],
        output_rejections[0],
        output_rejections[1],
        output_rejections[2],
        output_rejections[3],
    ])
}

fn main() {
    let started = Instant::now();
    let result = std::panic::catch_unwind(evaluate);
    let passed = matches!(
        &result,
        Ok(Ok(metrics))
            if metrics[..6].iter().all(|error| *error <= MAX_ERROR)
                && metrics[8] >= 1.0
                && metrics[9] >= 1.0
    );
    let qa = match result {
        Ok(Ok(metrics))
            if metrics[..6].iter().all(|error| *error <= MAX_ERROR)
                && metrics[8] >= 1.0
                && metrics[9] >= 1.0 =>
        {
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent dense stabilized TBEM action")
                .reference_checksum("analytic/deterministic-fixture:bem-fmm-operator-accuracy-v1")
                .input_checksum("synthetic/deterministic-fixture:bem-fmm-operator-accuracy-v1")
                .dimension("2-DOF SLFMM and MLFMM; two probes; three actions")
                .formulation("forward, algebraic-transpose, and Hermitian action")
            .tolerance(
                "each maximum relative action error <= 1e-6; four non-finite row-action paths rejected per backend",
            )
            .metric("slfmm_forward_max_relative_error", metrics[0])
            .metric("slfmm_transpose_max_relative_error", metrics[1])
            .metric("slfmm_hermitian_max_relative_error", metrics[2])
            .metric("mlfmm_forward_max_relative_error", metrics[3])
            .metric("mlfmm_transpose_max_relative_error", metrics[4])
            .metric("mlfmm_hermitian_max_relative_error", metrics[5])
            .metric("slfmm_nonfinite_output_rejections", metrics[6])
            .metric("mlfmm_nonfinite_output_rejections", metrics[7])
            .metric("slfmm_nonfinite_output_gate", metrics[8])
            .metric("mlfmm_nonfinite_output_gate", metrics[9])
                .elapsed_ms(started.elapsed().as_millis() as u64)
        }
        Ok(Ok(metrics)) => QaResult::failed(
            CASE_ID,
            format!(
            "independent action error exceeded budget: SLFMM=[{:.3e}, {:.3e}, {:.3e}], MLFMM=[{:.3e}, {:.3e}, {:.3e}], rejection_metrics=[{:.1}, {:.1}, {:.1}, {:.1}]",
            metrics[0],
            metrics[1],
            metrics[2],
            metrics[3],
            metrics[4],
            metrics[5],
            metrics[6],
            metrics[7],
            metrics[8],
            metrics[9],
            ),
        )
        .reference("independent dense stabilized TBEM action")
        .dimension("2-DOF SLFMM and MLFMM; two probes; three actions")
        .formulation("forward, algebraic-transpose, and Hermitian action")
        .tolerance("each maximum relative action error <= 1e-6")
        .elapsed_ms(started.elapsed().as_millis() as u64),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .reference("independent dense stabilized TBEM action")
            .dimension("2-DOF SLFMM and MLFMM; two probes; three actions")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(payload) => QaResult::failed(CASE_ID, format!("panic: {payload:?}"))
            .reference("independent dense stabilized TBEM action")
            .dimension("2-DOF SLFMM and MLFMM; two probes; three actions")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    println!("{}", qa.to_json_line().expect("serialize QA result"));
    if !passed {
        std::process::exit(1);
    }
}
