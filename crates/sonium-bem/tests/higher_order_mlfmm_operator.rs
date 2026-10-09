use num_complex::Complex64;
use sonium_bem::core::assembly::{
    build_cluster_tree, build_directional_mlfmm_system_stabilized_with_basis_checked,
    build_mlfmm_system_stabilized_with_basis_checked,
    build_tbem_system_continuous_quadratic_stabilized, build_tbem_system_quadratic_stabilized,
};
use sonium_bem::core::mesh::generate_cubed_sphere_quad_mesh;
use sonium_bem::core::types::{BasisContinuity, BasisOrder, BoundaryCondition, PhysicsParams};
use sonium_solvers::dense::DenseMatrix;

fn setup(
    continuity: BasisContinuity,
) -> (
    Vec<sonium_bem::core::types::Element>,
    Vec<[f64; 3]>,
    PhysicsParams,
) {
    let mesh = generate_cubed_sphere_quad_mesh(0.1, 1);
    let physics = PhysicsParams::new(120.0, 343.0, 1.21, false);
    let continuous_map = (continuity == BasisContinuity::Continuous)
        .then(|| mesh.continuous_quadratic_dof_map().unwrap());
    let mut elements = mesh.elements.clone();
    let bc = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
    let mut next_discontinuous = 0;
    for (index, element) in elements.iter_mut().enumerate() {
        element.boundary_condition = bc.clone();
        element.dof_addresses = if let Some(map) = &continuous_map {
            map.element_dofs(index).unwrap().to_vec()
        } else {
            let addresses = (next_discontinuous..next_discontinuous + 9).collect();
            next_discontinuous += 9;
            addresses
        };
    }
    (elements, mesh.nodes, physics)
}

fn relative_error(a: &[Complex64], b: &[Complex64]) -> f64 {
    let numerator = a
        .iter()
        .zip(b)
        .map(|(lhs, rhs)| (*lhs - *rhs).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = b
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(1e-12);
    numerator / denominator
}

#[test]
fn higher_order_mlfmm_matches_tbem_for_continuous_and_discontinuous_layouts() {
    for continuity in [BasisContinuity::Discontinuous, BasisContinuity::Continuous] {
        let (elements, nodes, physics) = setup(continuity);
        let clusters = build_cluster_tree(&elements, 32, &physics);
        let dense = if continuity == BasisContinuity::Continuous {
            build_tbem_system_continuous_quadratic_stabilized(
                &elements, &nodes, &physics, -1.0, 1.0,
            )
        } else {
            build_tbem_system_quadratic_stabilized(&elements, &nodes, &physics, -1.0, 1.0)
        };
        let full = build_mlfmm_system_stabilized_with_basis_checked(
            &elements,
            &nodes,
            clusters.clone(),
            &physics,
            BasisOrder::Quadratic,
            -1.0,
            1.0,
        )
        .unwrap();
        let directional = build_directional_mlfmm_system_stabilized_with_basis_checked(
            &elements,
            &nodes,
            clusters,
            &physics,
            BasisOrder::Quadratic,
            -1.0,
            1.0,
            Default::default(),
        )
        .unwrap();
        let n = dense.matrix.nrows();
        let forward_probe = (0..n)
            .map(|index| Complex64::new((index as f64 * 0.37).sin(), (index as f64 * 0.19).cos()))
            .collect::<Vec<_>>();
        let transpose_probe = (0..n)
            .map(|index| Complex64::new((index as f64 * 0.23).cos(), (index as f64 * 0.41).sin()))
            .collect::<Vec<_>>();
        let dense_forward = dense.matrix.matvec(&forward_probe);
        let dense_transpose = dense.matrix.transposed().matvec(&transpose_probe);
        let near_matrix = full.extract_near_field_matrix();
        let near_error = near_matrix
            .as_slice()
            .iter()
            .zip(dense.matrix.as_slice())
            .map(|(lhs, rhs)| (*lhs - *rhs).norm())
            .fold(0.0, f64::max);
        assert!(
            near_error < 1e-12,
            "{continuity:?} near-field max error {near_error:e}"
        );
        let block_probe = DenseMatrix::from_fn(n, 2, |row, rhs| {
            if rhs == 0 {
                forward_probe[row]
            } else {
                transpose_probe[row]
            }
        });
        for (name, system) in [("full", &full), ("directional", &directional)] {
            let forward_error = relative_error(&system.matvec(&forward_probe), &dense_forward);
            let transpose_error =
                relative_error(&system.matvec_transpose(&transpose_probe), &dense_transpose);
            assert!(
                forward_error < 1e-12,
                "{continuity:?} {name} forward error {forward_error:e}"
            );
            assert!(
                transpose_error < 1e-12,
                "{continuity:?} {name} transpose error {transpose_error:e}"
            );
            let block_forward = system.matvec_block(&block_probe);
            let block_transpose = system.matvec_transpose_block(&block_probe);
            let block_forward_error = relative_error(
                &(0..n)
                    .map(|row| block_forward[[row, 0]])
                    .collect::<Vec<_>>(),
                &dense_forward,
            );
            let block_transpose_error = relative_error(
                &(0..n)
                    .map(|row| block_transpose[[row, 1]])
                    .collect::<Vec<_>>(),
                &dense_transpose,
            );
            assert!(
                block_forward_error < 1e-12,
                "{continuity:?} {name} block forward error {block_forward_error:e}"
            );
            assert!(
                block_transpose_error < 1e-12,
                "{continuity:?} {name} block transpose error {block_transpose_error:e}"
            );
        }
    }
}
