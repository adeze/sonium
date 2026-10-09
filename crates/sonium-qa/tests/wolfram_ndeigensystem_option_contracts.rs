//! Hermetic contracts for option-only examples in Wolfram `NDEigensystem`.

use sonium_fem::{
    mesh::{generators::rectangular_mesh_triangles, interpolate_scalar_at_point},
    physics::solid::eigenmodes::{
        EigenproblemBuildError, IntervalLaplacianConfig, ScalarDirichletTerm,
        ScalarEigenproblemConfig, ScalarRobinTerm, build_interval_laplacian_problem,
        build_scalar_eigenproblem, track_real_modes_across_meshes, try_build_scalar_eigenproblem,
    },
};
use sonium_solvers::{LobpcgConfig, dense::DenseMatrix};

fn solve_first_interval_mode(elements: usize) -> f64 {
    build_interval_laplacian_problem(&IntervalLaplacianConfig {
        elements,
        ..Default::default()
    })
    .solve(&LobpcgConfig {
        num_eigenvalues: 1,
        max_iterations: 300,
        tolerance: 1.0e-10,
        block_padding: 2,
    })
    .unwrap()
    .eigenvalues[0]
}

#[test]
fn wolfram_normalization_contract_is_unit_mass_norm() {
    let problem = build_interval_laplacian_problem(&IntervalLaplacianConfig {
        elements: 24,
        ..Default::default()
    });
    let result = problem
        .solve(&LobpcgConfig {
            num_eigenvalues: 3,
            max_iterations: 400,
            tolerance: 1.0e-10,
            block_padding: 3,
        })
        .unwrap();
    assert_eq!(result.eigenvalues.len(), 3);
    assert!(result.eigenvalues.iter().all(|value| value.is_finite()));
    for mode in 0..3 {
        let vector = result.eigenvectors.column_to_vec(mode);
        let mass_vector: Vec<f64> = (0..problem.mass_like.num_rows)
            .map(|row| {
                problem
                    .mass_like
                    .row_entries(row)
                    .map(|(column, value)| value * vector[column])
                    .sum()
            })
            .collect();
        let mass_norm: f64 = vector.iter().zip(&mass_vector).map(|(x, bx)| x * bx).sum();
        assert!((mass_norm - 1.0).abs() < 1.0e-8, "mode {mode}: {mass_norm}");
    }
}

#[test]
fn wolfram_outside_region_contract_does_not_extrapolate() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 4, 4);
    let values: Vec<_> = mesh.nodes.iter().map(|point| point.x + point.y).collect();
    assert_eq!(
        interpolate_scalar_at_point(&mesh, &values, [1.25, 0.5, 0.0]),
        None
    );
    assert!(
        (interpolate_scalar_at_point(&mesh, &values, [0.25, 0.5, 0.0]).unwrap() - 0.75).abs()
            < 1.0e-12
    );
}

#[test]
fn wolfram_mesh_granularity_contract_converges_quadratically() {
    let exact = std::f64::consts::PI.powi(2);
    let errors: Vec<_> = [8, 16, 32]
        .map(|elements| (solve_first_interval_mode(elements) - exact).abs())
        .into_iter()
        .collect();
    assert!(errors[1] < 0.30 * errors[0], "errors={errors:?}");
    assert!(errors[2] < 0.30 * errors[1], "errors={errors:?}");
}

#[test]
fn wolfram_eigenvector_sign_contract_compares_up_to_sign() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 6, 6);
    let reference = DenseMatrix::from_fn(mesh.num_nodes(), 1, |node, _| {
        let point = &mesh.nodes[node];
        point.x + 2.0 * point.y
    });
    let negated = DenseMatrix::from_fn(mesh.num_nodes(), 1, |node, _| -reference[[node, 0]]);
    let assignment =
        track_real_modes_across_meshes(&mesh, &reference, &mesh, &negated, 17).unwrap();
    assert_eq!(assignment.current_indices, vec![0]);
    assert_eq!(assignment.phase_signs, vec![-1.0]);
    assert!(assignment.overlaps[0] > 1.0 - 1.0e-12);
}

#[test]
fn wolfram_inhomogeneous_eigenproblem_boundary_is_rejected() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 2, 2);
    let error = try_build_scalar_eigenproblem(
        &mesh,
        |_, _, _| 1.0,
        |_, _, _| 0.0,
        |_, _, _| 1.0,
        &ScalarEigenproblemConfig {
            dirichlet_boundary_values: vec![ScalarDirichletTerm {
                marker: 1,
                value: 1.0,
            }],
            ..Default::default()
        },
    )
    .unwrap_err();
    assert_eq!(
        error,
        EigenproblemBuildError::InhomogeneousDirichlet {
            marker: 1,
            value: 1.0
        }
    );
}

#[test]
fn wolfram_generalized_robin_term_is_assembled_on_mass_side() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 4, 4);
    let baseline = build_scalar_eigenproblem(
        &mesh,
        |_, _, _| 1.0,
        |_, _, _| 0.0,
        |_, _, _| 1.0,
        &ScalarEigenproblemConfig::default(),
    );
    let coefficient = 2.5;
    let generalized_robin = build_scalar_eigenproblem(
        &mesh,
        |_, _, _| 1.0,
        |_, _, _| 0.0,
        |_, _, _| 1.0,
        &ScalarEigenproblemConfig {
            robin_mass_terms: vec![ScalarRobinTerm {
                marker: 1,
                coefficient,
            }],
            ..Default::default()
        },
    );
    let baseline_sum: f64 = baseline.mass_like.values.iter().sum();
    let robin_sum: f64 = generalized_robin.mass_like.values.iter().sum();
    // Marker 1 is one unit-length side; a consistent boundary mass matrix
    // integrates the constant function to its length.
    assert!((robin_sum - baseline_sum - coefficient).abs() < 1.0e-12);
    assert_eq!(
        baseline.stiffness_like.values,
        generalized_robin.stiffness_like.values
    );
}
