//! Manufactured vector convergence evidence for the supported vector H1 path.
//!
//! This is intentionally independent of the RT0/H(div) and H(curl) invariant
//! packets: it solves a vector Laplacian problem through the public vector H1
//! space and generic assembly APIs, then checks L2 and H1-seminorm rates.

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_checked, assemble_linear_checked};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{LinearForm, LocalVector, VectorLaplacianForm};
use sonium_fem::mesh::Mesh;
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, VectorH1Space};

#[derive(Clone, Copy)]
struct ManufacturedVectorSource;

impl LinearForm<f64> for ManufacturedVectorSource {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, _] = test.physical_coords;
        let (value, _) = manufactured_vector([x, y, 0.0]);
        let source = [
            2.0 * std::f64::consts::PI.powi(2) * value[0],
            5.0 * std::f64::consts::PI.powi(2) * value[1],
        ];
        let scale = weight * test.det_j;
        for (local, entry) in local_vector.values.iter_mut().enumerate() {
            let basis_dot_source = (0..test.spatial_dim)
                .map(|component| test.vector_value(local, component) * source[component])
                .sum::<f64>();
            *entry += scale * basis_dot_source;
        }
    }
}

fn manufactured_vector([x, y, _]: [f64; 3]) -> ([f64; 2], [[f64; 2]; 2]) {
    let pi = std::f64::consts::PI;
    let sx = (pi * x).sin();
    let cx = (pi * x).cos();
    let s2x = (2.0 * pi * x).sin();
    let c2x = (2.0 * pi * x).cos();
    let sy = (pi * y).sin();
    let cy = (pi * y).cos();
    (
        [sx * sy, s2x * sy],
        [
            [pi * cx * sy, pi * sx * cy],
            [2.0 * pi * c2x * sy, pi * s2x * cy],
        ],
    )
}

fn aggregate_dense(matrix: &TripletMatrix<f64>) -> Vec<Vec<f64>> {
    assert_eq!(matrix.num_rows, matrix.num_cols);
    let mut dense = vec![vec![0.0; matrix.num_cols]; matrix.num_rows];
    for ((&row, &column), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
        dense[row][column] += value;
    }
    dense
}

fn all_boundary_dofs(mesh: &Mesh, components: usize) -> Vec<usize> {
    let mut is_boundary = vec![false; mesh.num_nodes() * components];
    for boundary in &mesh.boundaries {
        for &node in &boundary.nodes {
            for component in 0..components {
                is_boundary[node * components + component] = true;
            }
        }
    }
    is_boundary
        .into_iter()
        .enumerate()
        .filter_map(|(dof, boundary)| boundary.then_some(dof))
        .collect()
}

fn apply_zero_dirichlet(matrix: &mut [Vec<f64>], rhs: &mut [f64], dofs: &[usize]) {
    for &dof in dofs {
        for value in &mut matrix[dof] {
            *value = 0.0;
        }
        for row in matrix.iter_mut() {
            row[dof] = 0.0;
        }
        matrix[dof][dof] = 1.0;
        rhs[dof] = 0.0;
    }
}

fn solve_dense(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();
    assert_eq!(matrix.len(), n);
    for row in &matrix {
        assert_eq!(row.len(), n);
    }
    for column in 0..n {
        let (pivot_row, pivot) = (column..n)
            .map(|row| (row, matrix[row][column].abs()))
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .expect("non-empty elimination column");
        assert!(
            pivot > 1.0e-12,
            "singular vector Laplacian at column {column}"
        );
        matrix.swap(column, pivot_row);
        rhs.swap(column, pivot_row);

        let diagonal = matrix[column][column];
        for entry in &mut matrix[column][column..] {
            *entry /= diagonal;
        }
        rhs[column] /= diagonal;
        let pivot_values = matrix[column].clone();
        for row in column + 1..n {
            let factor = matrix[row][column];
            for (entry, &pivot_value) in matrix[row][column..]
                .iter_mut()
                .zip(&pivot_values[column..])
            {
                *entry -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    for column in (0..n).rev() {
        for row in 0..column {
            rhs[row] -= matrix[row][column] * rhs[column];
        }
    }
    rhs
}

#[derive(Debug)]
struct VectorError {
    l2: f64,
    h1_seminorm: f64,
}

fn solve_and_measure(refinement: usize) -> VectorError {
    let mesh = sonium_fem::mesh::unit_square_triangles(refinement);
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let stiffness = assemble_bilinear_checked(
        &VectorLaplacianForm { coefficient: 1.0 },
        &space,
        &space,
        &mesh,
        5,
    )
    .expect("vector H1 stiffness assembly should pass checked admission");
    let mut matrix = aggregate_dense(&stiffness);
    let mut rhs = assemble_linear_checked(&ManufacturedVectorSource, &space, &mesh, 5)
        .expect("vector H1 source assembly should pass checked admission");
    let boundary = all_boundary_dofs(&mesh, 2);
    apply_zero_dirichlet(&mut matrix, &mut rhs, &boundary);
    let solution = solve_dense(matrix, rhs);

    let mut l2_squared = 0.0;
    let mut h1_squared = 0.0;
    for element_idx in 0..mesh.num_elements() {
        let quadrature = QuadratureRule::new(mesh.elements[element_idx].element_type, 5);
        let local_dofs = space.local_to_global(&mesh, element_idx);
        for point in quadrature.iter() {
            let tabulation =
                space.tabulate(&mesh, element_idx, point.xi(), point.eta(), point.zeta());
            let mut actual = [0.0; 2];
            let mut actual_gradient = [[0.0; 2]; 2];
            for (local, &global) in local_dofs.iter().enumerate() {
                for component in 0..2 {
                    actual[component] +=
                        solution[global] * tabulation.vector_value(local, component);
                }
                let component = local % 2;
                for direction in 0..2 {
                    actual_gradient[component][direction] +=
                        solution[global] * tabulation.grad(local, direction);
                }
            }
            let (expected, expected_gradient) = manufactured_vector(tabulation.physical_coords);
            let weight = point.weight * tabulation.det_j;
            l2_squared += weight
                * actual
                    .iter()
                    .zip(expected)
                    .map(|(&actual, expected)| (actual - expected).powi(2))
                    .sum::<f64>();
            h1_squared += weight
                * actual_gradient
                    .iter()
                    .zip(expected_gradient)
                    .flat_map(|(actual, expected)| actual.into_iter().zip(expected))
                    .map(|(actual, expected)| (actual - expected).powi(2))
                    .sum::<f64>();
        }
    }

    VectorError {
        l2: l2_squared.sqrt(),
        h1_seminorm: h1_squared.sqrt(),
    }
}

fn refinement_rate(coarse: f64, fine: f64) -> f64 {
    assert!(coarse > fine && fine > 0.0);
    (coarse / fine).log2()
}

#[test]
fn vector_laplacian_manufactured_solution_converges_on_three_refinements() {
    let coarse = solve_and_measure(2);
    let medium = solve_and_measure(4);
    let fine = solve_and_measure(8);
    let l2_rate_1 = refinement_rate(coarse.l2, medium.l2);
    let l2_rate_2 = refinement_rate(medium.l2, fine.l2);
    let h1_rate_1 = refinement_rate(coarse.h1_seminorm, medium.h1_seminorm);
    let h1_rate_2 = refinement_rate(medium.h1_seminorm, fine.h1_seminorm);

    assert!(
        l2_rate_1 > 1.5 && l2_rate_2 > 1.5,
        "vector P1 L2 rates were {l2_rate_1:.3}, {l2_rate_2:.3}; errors {coarse:?}, {medium:?}, {fine:?}"
    );
    assert!(
        h1_rate_1 > 0.75 && h1_rate_2 > 0.75,
        "vector P1 H1 rates were {h1_rate_1:.3}, {h1_rate_2:.3}; errors {coarse:?}, {medium:?}, {fine:?}"
    );
}
