//! Checked distributed row ownership for continuous higher-order traces.

use num_complex::Complex64;
use sonium_bem::core::assembly::build_transmission_tbem_system_with_continuous_quadratic;
use sonium_bem::core::distributed::{
    DistributedRowPlan, RankPartition, plan_distributed_rows_with_dof_count,
};
use sonium_bem::core::mesh::generate_cubed_sphere_quad_mesh;
use sonium_bem::core::types::PhysicsParams;
use sonium_solvers::dense::DenseMatrix;

#[test]
fn continuous_quadratic_quad4_rows_cover_both_trace_blocks_once() {
    let mut mesh = generate_cubed_sphere_quad_mesh(0.2, 1);
    let dof_map = mesh
        .continuous_quadratic_dof_map()
        .expect("closed Quad4 mesh has a valid continuous quadratic map");
    for (element_index, element) in mesh.elements.iter_mut().enumerate() {
        if let Some(dofs) = dof_map.element_dofs(element_index) {
            element.dof_addresses = dofs.to_vec();
        }
    }

    let plan = plan_distributed_rows_with_dof_count(&mesh.elements, 3, dof_map.num_dofs())
        .expect("continuous quadratic global DOFs are a valid row layout");
    assert_eq!(plan.num_dofs, dof_map.num_dofs());

    let mut rows = plan
        .transmission_trace_rows(dof_map.num_dofs())
        .expect("pressure and normal-velocity rows cover the interface")
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    rows.sort_unstable();
    assert_eq!(rows, (0..2 * dof_map.num_dofs()).collect::<Vec<_>>());
}

#[test]
fn continuous_quadratic_compressed_row_actions_match_global_actions() {
    let mut mesh = generate_cubed_sphere_quad_mesh(0.2, 1);
    let dof_map = mesh
        .continuous_quadratic_dof_map()
        .expect("closed Quad4 mesh has a valid continuous quadratic map");
    for (element_index, element) in mesh.elements.iter_mut().enumerate() {
        element.dof_addresses = dof_map
            .element_dofs(element_index)
            .expect("active Quad4 has mapped DOFs")
            .to_vec();
    }
    let system = build_transmission_tbem_system_with_continuous_quadratic(
        &mesh.elements,
        &mesh.nodes,
        PhysicsParams::new(420.0, 343.0, 1.21, false),
        PhysicsParams::new(420.0, 1_480.0, 1_000.0, true),
    )
    .expect("continuous quadratic transmission system");
    let compressed = system
        .compress(4, 1.0e-8, 8)
        .expect("compressed continuous quadratic transmission system");
    let input = (0..system.total_dofs())
        .map(|index| Complex64::new(0.1 + index as f64 * 0.011, -0.03 + index as f64 * 0.004))
        .collect::<Vec<_>>();
    let global = compressed
        .apply_checked(&input)
        .expect("global forward action");
    let global_transpose = compressed
        .apply_transpose_checked(&input)
        .expect("global transpose action");
    let plan = plan_distributed_rows_with_dof_count(&mesh.elements, 3, dof_map.num_dofs())
        .expect("continuous quadratic row plan");
    let rows_by_rank = plan
        .transmission_trace_rows(system.interface_dofs())
        .expect("two-trace row expansion");
    for rows in rows_by_rank {
        let local = compressed
            .apply_rows_checked(&input, &rows)
            .expect("local forward row action");
        let local_transpose = compressed
            .apply_transpose_rows_checked(&input, &rows)
            .expect("local transpose row action");
        for (slot, &row) in rows.iter().enumerate() {
            let forward_error = (local[slot] - global[row]).norm();
            let transpose_error = (local_transpose[slot] - global_transpose[row]).norm();
            let forward_tolerance = 1.0e-10 * (1.0 + global[row].norm());
            let transpose_tolerance = 1.0e-10 * (1.0 + global_transpose[row].norm());
            assert!(
                forward_error <= forward_tolerance,
                "forward row {row} slot {slot}: {forward_error:.3e} > {forward_tolerance:.3e}"
            );
            assert!(
                transpose_error <= transpose_tolerance,
                "transpose row {row} slot {slot}: {transpose_error:.3e} > {transpose_tolerance:.3e}"
            );
        }
    }
}

#[test]
fn continuous_quadratic_compressed_block_row_actions_match_global_actions() {
    let mut mesh = generate_cubed_sphere_quad_mesh(0.2, 1);
    let dof_map = mesh
        .continuous_quadratic_dof_map()
        .expect("closed Quad4 mesh valid continuous quadratic map");
    for (element_index, element) in mesh.elements.iter_mut().enumerate() {
        element.dof_addresses = dof_map
            .element_dofs(element_index)
            .expect("active Quad4 mapped DOFs")
            .to_vec();
    }
    let system = build_transmission_tbem_system_with_continuous_quadratic(
        &mesh.elements,
        &mesh.nodes,
        PhysicsParams::new(420.0, 343.0, 1.21, false),
        PhysicsParams::new(420.0, 1_480.0, 1_000.0, true),
    )
    .expect("continuous quadratic transmission system");
    let compressed = system
        .compress(4, 1.0e-8, 8)
        .expect("compressed continuous quadratic transmission system");
    let mut input = DenseMatrix::zeros(system.total_dofs(), 3);
    for column in 0..input.ncols() {
        for row in 0..input.nrows() {
            input[[row, column]] = Complex64::new(
                0.1 + 0.011 * row as f64 + 0.03 * column as f64,
                -0.03 + 0.004 * row as f64 - 0.02 * column as f64,
            );
        }
    }
    let plan = plan_distributed_rows_with_dof_count(&mesh.elements, 3, dof_map.num_dofs())
        .expect("continuous quadratic row plan");
    let rows = plan
        .transmission_trace_rows(system.interface_dofs())
        .expect("two-trace rows")
        .into_iter()
        .flatten()
        .take(9)
        .collect::<Vec<_>>();
    let forward = compressed
        .apply_rows_block_checked(&input, &rows)
        .expect("forward block rows");
    let transpose = compressed
        .apply_transpose_rows_block_checked(&input, &rows)
        .expect("transpose block rows");
    for column in 0..input.ncols() {
        let vector = (0..input.nrows())
            .map(|row| input[[row, column]])
            .collect::<Vec<_>>();
        let expected_forward = compressed
            .apply_rows_checked(&vector, &rows)
            .expect("forward rows");
        let expected_transpose = compressed
            .apply_transpose_rows_checked(&vector, &rows)
            .expect("transpose rows");
        for (local_row, _) in rows.iter().enumerate() {
            let forward_error = (forward[[local_row, column]] - expected_forward[local_row]).norm();
            let transpose_error =
                (transpose[[local_row, column]] - expected_transpose[local_row]).norm();
            assert!(
                forward_error <= 1.0e-10,
                "forward row {local_row}, rhs {column}: {forward_error:.3e}"
            );
            assert!(
                transpose_error <= 1.0e-10,
                "transpose row {local_row}, rhs {column}: {transpose_error:.3e}"
            );
        }
    }
}

#[test]
fn distributed_rows_reject_non_dense_or_repeated_global_layouts() {
    let mut mesh = generate_cubed_sphere_quad_mesh(0.2, 1);
    let dof_map = mesh
        .continuous_quadratic_dof_map()
        .expect("closed Quad4 mesh has a valid continuous quadratic map");
    for (element_index, element) in mesh.elements.iter_mut().enumerate() {
        element.dof_addresses = dof_map
            .element_dofs(element_index)
            .expect("active Quad4 has mapped DOFs")
            .to_vec();
    }

    mesh.elements[0].dof_addresses[0] = dof_map.num_dofs();
    assert!(plan_distributed_rows_with_dof_count(&mesh.elements, 2, dof_map.num_dofs()).is_err());

    let mut repeated = mesh;
    let map = repeated
        .continuous_quadratic_dof_map()
        .expect("topology remains valid");
    repeated.elements[0].dof_addresses = vec![map.element_dofs(0).unwrap()[0]; 9];
    assert!(plan_distributed_rows_with_dof_count(&repeated.elements, 2, map.num_dofs()).is_err());
}

#[test]
fn transmission_trace_rows_reject_duplicate_rank_partitions() {
    let plan = DistributedRowPlan {
        partitions: vec![
            RankPartition {
                rank: 0,
                element_indices: vec![],
                dof_indices: vec![0],
            },
            RankPartition {
                rank: 0,
                element_indices: vec![],
                dof_indices: vec![1],
            },
        ],
        num_active_elements: 0,
        num_dofs: 2,
    };
    assert!(plan.transmission_trace_rows(2).is_err());
}
