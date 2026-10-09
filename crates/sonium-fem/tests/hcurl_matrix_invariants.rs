//! Production-assembly invariants for the CPU H(curl) matrix family.
//!
//! This is deliberately narrower than an external Maxwell validation case:
//! it proves that the checked assembler preserves the algebraic contracts of
//! the supported simplex Nédélec spaces.  The mass matrix must be symmetric
//! positive definite, the curl-curl matrix must be symmetric positive
//! semidefinite, and the order-2 exact sequence must retain curl(grad) = 0
//! after local-to-global assembly.

use std::collections::BTreeMap;

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_checked};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{CurlCurlForm, EdgeMassForm};
use sonium_fem::mesh::{Mesh, unit_cube_tetrahedra, unit_square_triangles};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space, HcurlSpace};

#[derive(Clone, Copy, Debug)]
enum SimplexFamily {
    Triangle,
    Tetrahedron,
}

impl SimplexFamily {
    fn mesh(self) -> Mesh {
        match self {
            Self::Triangle => unit_square_triangles(2),
            Self::Tetrahedron => unit_cube_tetrahedra(1),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Triangle => "triangles",
            Self::Tetrahedron => "tetrahedra",
        }
    }
}

fn aggregate_triplets(matrix: &TripletMatrix<f64>) -> BTreeMap<(usize, usize), f64> {
    let mut aggregate = BTreeMap::new();
    for ((&row, &column), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
        *aggregate.entry((row, column)).or_insert(0.0) += value;
    }
    aggregate
}

fn quadratic_form(matrix: &BTreeMap<(usize, usize), f64>, vector: &[f64]) -> f64 {
    matrix
        .iter()
        .map(|(&(row, column), &value)| value * vector[row] * vector[column])
        .sum()
}

fn assert_symmetric_positive_matrices(family: SimplexFamily, order: usize) {
    let mesh = family.mesh();
    let space = HcurlSpace { order };
    space.validate(&mesh).unwrap_or_else(|error| {
        panic!("{} order-{order} validation failed: {error}", family.name())
    });

    let mass = assemble_bilinear_checked(&EdgeMassForm, &space, &space, &mesh, 7)
        .unwrap_or_else(|error| panic!("{} edge-mass assembly failed: {error}", family.name()));
    let curl_curl = assemble_bilinear_checked(&CurlCurlForm, &space, &space, &mesh, 7)
        .unwrap_or_else(|error| panic!("{} curl-curl assembly failed: {error}", family.name()));
    let mass = aggregate_triplets(&mass);
    let curl_curl = aggregate_triplets(&curl_curl);

    let max_mass_symmetry = mass
        .iter()
        .map(|(&(row, column), &value)| {
            (value - mass.get(&(column, row)).copied().unwrap_or(0.0)).abs()
        })
        .fold(0.0, f64::max);
    let max_curl_symmetry = curl_curl
        .iter()
        .map(|(&(row, column), &value)| {
            (value - curl_curl.get(&(column, row)).copied().unwrap_or(0.0)).abs()
        })
        .fold(0.0, f64::max);
    assert!(
        max_mass_symmetry < 2.0e-12,
        "{} order-{order} edge mass symmetry residual {max_mass_symmetry:.3e}",
        family.name()
    );
    assert!(
        max_curl_symmetry < 2.0e-12,
        "{} order-{order} curl-curl symmetry residual {max_curl_symmetry:.3e}",
        family.name()
    );

    let n = space.num_global_dofs(&mesh);
    let probe: Vec<f64> = (0..n).map(|dof| 0.25 + 0.071 * (dof + 1) as f64).collect();
    let mass_energy = quadratic_form(&mass, &probe);
    let curl_energy = quadratic_form(&curl_curl, &probe);
    assert!(
        mass_energy > 1.0e-10,
        "{} order-{order} edge mass is not positive: {mass_energy:.3e}",
        family.name()
    );
    assert!(
        curl_energy >= -2.0e-11,
        "{} order-{order} curl-curl energy is negative: {curl_energy:.3e}",
        family.name()
    );
}

fn manufactured_scalar([x, y, z]: [f64; 3]) -> f64 {
    0.3 + 0.7 * x - 0.4 * y + 0.2 * z + 0.8 * x * x - 0.6 * x * y + 0.5 * x * z + 0.9 * y * y
        - 0.3 * y * z
        + 0.4 * z * z
}

fn solve_dense(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();
    for column in 0..n {
        let (pivot_row, pivot) = (column..n)
            .map(|row| (row, matrix[row][column].abs()))
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .expect("non-empty elimination column");
        assert!(
            pivot > 1.0e-11,
            "singular H(curl) projection at column {column}"
        );
        matrix.swap(column, pivot_row);
        rhs.swap(column, pivot_row);

        for row in column + 1..n {
            let factor = matrix[row][column] / matrix[column][column];
            matrix[row][column] = 0.0;
            for entry in column + 1..n {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
            rhs[row] -= factor * rhs[column];
        }
    }

    for row in (0..n).rev() {
        let tail = ((row + 1)..n)
            .map(|column| matrix[row][column] * rhs[column])
            .sum::<f64>();
        rhs[row] = (rhs[row] - tail) / matrix[row][row];
    }
    rhs
}

fn fit_gradient_on_element(
    mesh: &Mesh,
    element: usize,
    h1: &H1Space,
    hcurl: &HcurlSpace,
    scalar_coefficients: &[f64],
) -> Vec<f64> {
    let element_type = mesh.elements[element].element_type;
    let dimension = element_type.dimension();
    let scalar_dofs = h1.local_to_global(mesh, element);
    let n_hcurl = hcurl.num_local_dofs(element_type);
    let mut normal = vec![vec![0.0; n_hcurl]; n_hcurl];
    let mut projected = vec![0.0; n_hcurl];

    for point in QuadratureRule::new(element_type, 7).iter() {
        let h1_tab = h1.tabulate(mesh, element, point.xi(), point.eta(), point.zeta());
        let hcurl_tab = hcurl.tabulate(mesh, element, point.xi(), point.eta(), point.zeta());
        let mut gradient = vec![0.0; dimension];
        for (local, &global) in scalar_dofs.iter().enumerate() {
            for (direction, component) in gradient.iter_mut().enumerate() {
                *component += scalar_coefficients[global] * h1_tab.grad(local, direction);
            }
        }

        for (direction, &exact_component) in gradient.iter().enumerate() {
            // This keeps the normal-equation accumulation explicit and avoids
            // relying on any private H(curl) DOF functional.
            let row: Vec<f64> = (0..n_hcurl)
                .map(|local| hcurl_tab.vector_value(local, direction))
                .collect();
            for (lhs, &row_value) in normal.iter_mut().zip(&row) {
                for (rhs_value, &column_value) in lhs.iter_mut().zip(&row) {
                    *rhs_value += row_value * column_value;
                }
            }
            for (entry, &row_value) in projected.iter_mut().zip(&row) {
                *entry += row_value * exact_component;
            }
        }
    }

    solve_dense(normal, projected)
}

#[test]
fn production_hcurl_matrices_are_symmetric_and_positive_on_supported_simplices() {
    for family in [SimplexFamily::Triangle, SimplexFamily::Tetrahedron] {
        for order in [1, 2] {
            assert_symmetric_positive_matrices(family, order);
        }
    }
}

#[test]
fn production_order2_curl_curl_preserves_manufactured_gradient_kernel() {
    for family in [SimplexFamily::Triangle, SimplexFamily::Tetrahedron] {
        let mesh = family.mesh();
        let h1 = H1Space::new(PolynomialDegree::P2);
        let hcurl = HcurlSpace::order2();
        h1.validate(&mesh).expect("P2 H1 validation must pass");
        hcurl
            .validate(&mesh)
            .expect("order-2 H(curl) validation must pass on simplices");

        let coordinates = h1
            .dof_coordinates(&mesh)
            .expect("P2 H1 coordinates must be available");
        let scalar_coefficients: Vec<f64> = coordinates
            .iter()
            .map(|&point| manufactured_scalar(point))
            .collect();
        let mut global_coefficients: Vec<Option<f64>> = vec![None; hcurl.num_global_dofs(&mesh)];

        for element in 0..mesh.num_elements() {
            let local_coefficients =
                fit_gradient_on_element(&mesh, element, &h1, &hcurl, &scalar_coefficients);
            for (&global, &coefficient) in hcurl
                .local_to_global(&mesh, element)
                .iter()
                .zip(&local_coefficients)
            {
                if let Some(previous) = global_coefficients[global] {
                    assert!((previous - coefficient).abs() < 2.0e-8);
                } else {
                    global_coefficients[global] = Some(coefficient);
                }
            }
        }

        let coefficients: Vec<f64> = global_coefficients
            .into_iter()
            .map(|coefficient| coefficient.expect("every global H(curl) DOF must be supported"))
            .collect();
        let curl_curl = assemble_bilinear_checked(&CurlCurlForm, &hcurl, &hcurl, &mesh, 7)
            .expect("curl-curl assembly must pass checked admission");
        let curl_curl = aggregate_triplets(&curl_curl);
        let energy = quadratic_form(&curl_curl, &coefficients);
        assert!(
            energy.abs() < 2.0e-8,
            "{} manufactured gradient curl-curl energy {energy:.3e}",
            family.name()
        );
    }
}
