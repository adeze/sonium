//! W1-05 manufactured convergence and conservation evidence.
//!
//! This file deliberately follows the production function-space and generic
//! assembly APIs.  The supported high-order matrix is:
//!
//! - H1 P2/Q2 on triangles, quadrilaterals, tetrahedra, and hexahedra;
//! - H1 P3 on tetrahedra;
//! - Nédélec order 2 on triangles and tetrahedra.
//!
//! The current production boundary is explicit: H1 P3 is implemented only on
//! tetrahedra, Nédélec order 2 has no quadrilateral/hexahedral basis, H(div)
//! exposes RT0 only, and high-order curved/isoparametric geometry is not
//! implemented.  Those families are not silently treated as passing rows.

use std::collections::BTreeMap;

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_checked};
use sonium_fem::basis::{PolynomialDegree, try_evaluate_shape};
use sonium_fem::forms::{LaplacianForm, MassForm};
use sonium_fem::mesh::{ElementType, Mesh};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space, HcurlSpace, HdivSpace, SpaceValidationError};

#[derive(Clone, Copy, Debug)]
enum ScalarFamily {
    Triangle2d,
    Quadrilateral2d,
    Tetrahedron3d,
    Hexahedron3d,
}

impl ScalarFamily {
    fn mesh(self, refinement: usize) -> Mesh {
        match self {
            Self::Triangle2d => sonium_fem::mesh::unit_square_triangles(refinement),
            Self::Quadrilateral2d => sonium_fem::mesh::unit_square_quads(refinement),
            Self::Tetrahedron3d => sonium_fem::mesh::unit_cube_tetrahedra(refinement),
            Self::Hexahedron3d => sonium_fem::mesh::unit_cube_hexahedra(refinement),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Triangle2d => "P2-triangle-2d",
            Self::Quadrilateral2d => "Q2-quadrilateral-2d",
            Self::Tetrahedron3d => "P2-tetrahedron-3d",
            Self::Hexahedron3d => "Q2-hexahedron-3d",
        }
    }
}

/// A cubic manufactured field is outside the P2/Q2 space, so its error gives
/// a nontrivial h-refinement rate while retaining an analytic value/gradient.
fn manufactured_scalar([x, y, z]: [f64; 3]) -> (f64, [f64; 3]) {
    let value =
        0.2 + 0.7 * x - 0.3 * y + 0.5 * z + 0.4 * x * x - 0.6 * x * y + 0.3 * x * z + 0.8 * y * y
            - 0.2 * y * z
            + 0.5 * z * z
            + 0.17 * x * x * x
            + 0.13 * x * x * y
            - 0.11 * x * y * y
            + 0.07 * y * y * y
            + 0.09 * x * x * z
            - 0.08 * x * y * z
            + 0.06 * y * y * z
            + 0.04 * x * z * z
            - 0.03 * y * z * z
            + 0.05 * z * z * z;
    let gradient = [
        0.7 + 0.8 * x - 0.6 * y + 0.3 * z + 0.51 * x * x + 0.26 * x * y - 0.11 * y * y
            + 0.18 * x * z
            - 0.08 * y * z
            + 0.04 * z * z,
        -0.3 - 0.6 * x + 1.6 * y - 0.2 * z + 0.13 * x * x - 0.22 * x * y + 0.21 * y * y
            - 0.08 * x * z
            + 0.12 * y * z
            - 0.03 * z * z,
        0.5 + 0.3 * x - 0.2 * y + z + 0.09 * x * x - 0.08 * x * y + 0.06 * y * y + 0.08 * x * z
            - 0.06 * y * z
            + 0.15 * z * z,
    ];
    (value, gradient)
}

#[derive(Clone, Copy, Debug, Default)]
struct InterpolationError {
    l2: f64,
    h1_seminorm: f64,
}

fn h1_interpolation_error(family: ScalarFamily, refinement: usize) -> InterpolationError {
    let mesh = family.mesh(refinement);
    let space = H1Space::new(PolynomialDegree::P2);
    space
        .validate(&mesh)
        .unwrap_or_else(|error| panic!("{} space validation failed: {error}", family.name()));
    let coordinates = space
        .dof_coordinates(&mesh)
        .unwrap_or_else(|| panic!("{} must expose P2 DOF coordinates", family.name()));
    let coefficients: Vec<f64> = coordinates
        .iter()
        .map(|&point| manufactured_scalar(point).0)
        .collect();

    let mut l2_squared = 0.0;
    let mut h1_squared = 0.0;
    for element_idx in 0..mesh.num_elements() {
        let element_type = mesh.elements[element_idx].element_type;
        let quadrature = QuadratureRule::new(element_type, 7);
        let local_dofs = space.local_to_global(&mesh, element_idx);
        for point in quadrature.iter() {
            let tabulation =
                space.tabulate(&mesh, element_idx, point.xi(), point.eta(), point.zeta());
            let mut value = 0.0;
            let mut gradient = [0.0; 3];
            for (local, &global) in local_dofs.iter().enumerate() {
                value += coefficients[global] * tabulation.value(local);
                for (direction, component) in
                    gradient.iter_mut().enumerate().take(tabulation.spatial_dim)
                {
                    *component += coefficients[global] * tabulation.grad(local, direction);
                }
            }
            let (exact_value, exact_gradient) = manufactured_scalar(tabulation.physical_coords);
            let weight = point.weight * tabulation.det_j;
            l2_squared += weight * (value - exact_value).powi(2);
            h1_squared += weight
                * gradient
                    .iter()
                    .zip(exact_gradient)
                    .take(tabulation.spatial_dim)
                    .map(|(&actual, exact)| (actual - exact).powi(2))
                    .sum::<f64>();
        }
    }

    InterpolationError {
        l2: l2_squared.sqrt(),
        h1_seminorm: h1_squared.sqrt(),
    }
}

fn refinement_rate(coarse: f64, fine: f64) -> f64 {
    assert!(coarse.is_finite() && fine.is_finite() && coarse > fine && fine > 0.0);
    (coarse / fine).log2()
}

#[test]
fn p2_manufactured_interpolation_converges_in_supported_2d_and_3d_families() {
    for family in [
        ScalarFamily::Triangle2d,
        ScalarFamily::Quadrilateral2d,
        ScalarFamily::Tetrahedron3d,
        ScalarFamily::Hexahedron3d,
    ] {
        let coarse = h1_interpolation_error(family, 1);
        let medium = h1_interpolation_error(family, 2);
        let fine = h1_interpolation_error(family, 4);
        let l2_rate_1 = refinement_rate(coarse.l2, medium.l2);
        let l2_rate_2 = refinement_rate(medium.l2, fine.l2);
        let h1_rate_1 = refinement_rate(coarse.h1_seminorm, medium.h1_seminorm);
        let h1_rate_2 = refinement_rate(medium.h1_seminorm, fine.h1_seminorm);

        // P2 interpolation of a smooth cubic has the expected O(h^3) L2 and
        // O(h^2) H1-seminorm rates on affine cells.  Keep a margin for the
        // tetrahedral quadrature rule's finite exactness degree.
        assert!(
            l2_rate_1 > 2.4 && l2_rate_2 > 2.4,
            "{} L2 rates were {l2_rate_1:.3}, {l2_rate_2:.3}; errors {coarse:?}, {medium:?}, {fine:?}",
            family.name()
        );
        assert!(
            h1_rate_1 > 1.5 && h1_rate_2 > 1.5,
            "{} H1 rates were {h1_rate_1:.3}, {h1_rate_2:.3}; errors {coarse:?}, {medium:?}, {fine:?}",
            family.name()
        );
    }
}

fn aggregate_triplets(matrix: &TripletMatrix<f64>) -> BTreeMap<(usize, usize), f64> {
    let mut aggregate = BTreeMap::new();
    for ((&row, &column), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
        *aggregate.entry((row, column)).or_insert(0.0) += value;
    }
    aggregate
}

fn assert_p2_conservation(family: ScalarFamily) {
    let mesh = family.mesh(2);
    let space = H1Space::new(PolynomialDegree::P2);
    let stiffness = assemble_bilinear_checked(&LaplacianForm, &space, &space, &mesh, 7)
        .unwrap_or_else(|error| panic!("{} stiffness assembly failed: {error}", family.name()));
    let mass = assemble_bilinear_checked(&MassForm, &space, &space, &mesh, 7)
        .unwrap_or_else(|error| panic!("{} mass assembly failed: {error}", family.name()));
    let stiffness = aggregate_triplets(&stiffness);
    let mass = aggregate_triplets(&mass);
    let n = space.num_global_dofs(&mesh);

    let mut row_sums = vec![0.0; n];
    let mut mass_total = 0.0;
    let mut max_symmetry_error: f64 = 0.0;
    for (&(row, column), &value) in &stiffness {
        row_sums[row] += value;
        let transpose = stiffness.get(&(column, row)).copied().unwrap_or(0.0);
        max_symmetry_error = max_symmetry_error.max((value - transpose).abs());
    }
    for &value in mass.values() {
        mass_total += value;
    }

    let max_row_sum = row_sums.iter().copied().map(f64::abs).fold(0.0, f64::max);
    assert!(
        max_row_sum < 2.0e-11,
        "{} constant-field stiffness residual {max_row_sum:.3e}",
        family.name()
    );
    assert!(
        max_symmetry_error < 2.0e-12,
        "{} stiffness symmetry residual {max_symmetry_error:.3e}",
        family.name()
    );
    assert!(
        (mass_total - 1.0).abs() < 2.0e-11,
        "{} mass of the unit domain was {mass_total:.16e}",
        family.name()
    );
}

#[test]
fn p2_assembled_operators_conserve_constants_in_supported_2d_and_3d_families() {
    for family in [
        ScalarFamily::Triangle2d,
        ScalarFamily::Quadrilateral2d,
        ScalarFamily::Tetrahedron3d,
        ScalarFamily::Hexahedron3d,
    ] {
        assert_p2_conservation(family);
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
            "singular exact-sequence fit at column {column}"
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

fn fit_hcurl_gradient(
    mesh: &Mesh,
    element_idx: usize,
    h1: &H1Space,
    hcurl: &HcurlSpace,
    scalar_coefficients: &[f64],
) -> Vec<f64> {
    let element_type = mesh.elements[element_idx].element_type;
    let dimension = element_type.dimension();
    let local_h1 = h1.local_to_global(mesh, element_idx);
    let n_hcurl = hcurl.num_local_dofs(element_type);
    let mut normal = vec![vec![0.0; n_hcurl]; n_hcurl];
    let mut projected = vec![0.0; n_hcurl];

    // The overdetermined point fit avoids depending on a private DOF
    // functional while exercising the public tabulation/Piola path.
    let quadrature = QuadratureRule::new(element_type, 7);
    for point in quadrature.iter() {
        let h1_tabulation = h1.tabulate(mesh, element_idx, point.xi(), point.eta(), point.zeta());
        let hcurl_tabulation =
            hcurl.tabulate(mesh, element_idx, point.xi(), point.eta(), point.zeta());
        let mut exact_gradient = vec![0.0; dimension];
        for (local, &global) in local_h1.iter().enumerate() {
            for (direction, component) in exact_gradient.iter_mut().enumerate() {
                *component += scalar_coefficients[global] * h1_tabulation.grad(local, direction);
            }
        }
        for (direction, &exact_component) in exact_gradient.iter().enumerate().take(dimension) {
            let row: Vec<f64> = (0..n_hcurl)
                .map(|local| hcurl_tabulation.vector_value(local, direction))
                .collect();
            for (lhs, &row_value) in normal.iter_mut().zip(&row) {
                for (rhs, &column_value) in lhs.iter_mut().zip(&row) {
                    *rhs += row_value * column_value;
                }
            }
            for (entry, &row_value) in projected.iter_mut().zip(&row) {
                *entry += row_value * exact_component;
            }
        }
    }
    solve_dense(normal, projected)
}

fn assert_exact_sequence_on_simplex(family: ScalarFamily) {
    let mesh = family.mesh(2);
    let h1 = H1Space::new(PolynomialDegree::P2);
    let hcurl = HcurlSpace::order2();
    h1.validate(&mesh).expect("P2 scalar space must validate");
    hcurl
        .validate(&mesh)
        .expect("order-2 H(curl) space must validate on simplex cells");
    let coordinates = h1
        .dof_coordinates(&mesh)
        .expect("P2 scalar DOF coordinates are required for the manufactured field");
    let scalar_coefficients: Vec<f64> = coordinates
        .iter()
        .map(|&point| manufactured_scalar(point).0)
        .collect();

    for element_idx in 0..mesh.num_elements() {
        let coefficients =
            fit_hcurl_gradient(&mesh, element_idx, &h1, &hcurl, &scalar_coefficients);
        let element_type = mesh.elements[element_idx].element_type;
        let quadrature = QuadratureRule::new(element_type, 7);
        for point in quadrature.iter() {
            let h1_tabulation =
                h1.tabulate(&mesh, element_idx, point.xi(), point.eta(), point.zeta());
            let hcurl_tabulation =
                hcurl.tabulate(&mesh, element_idx, point.xi(), point.eta(), point.zeta());
            let local_h1 = h1.local_to_global(&mesh, element_idx);
            let mut expected = vec![0.0; element_type.dimension()];
            for (local, &global) in local_h1.iter().enumerate() {
                for (direction, component) in expected.iter_mut().enumerate() {
                    *component +=
                        scalar_coefficients[global] * h1_tabulation.grad(local, direction);
                }
            }
            let mut reproduced = vec![0.0; element_type.dimension()];
            for (local, &coefficient) in coefficients.iter().enumerate() {
                for (direction, component) in reproduced.iter_mut().enumerate() {
                    *component += coefficient * hcurl_tabulation.vector_value(local, direction);
                }
            }
            let reconstruction_error = reproduced
                .iter()
                .zip(expected)
                .map(|(&actual, exact)| (actual - exact).abs())
                .fold(0.0, f64::max);
            assert!(
                reconstruction_error < 2.0e-9,
                "{} gradient reconstruction error {reconstruction_error:.3e}",
                family.name()
            );

            let curl_error = if element_type.dimension() == 2 {
                coefficients
                    .iter()
                    .enumerate()
                    .map(|(local, &coefficient)| coefficient * hcurl_tabulation.derivatives[local])
                    .sum::<f64>()
                    .abs()
            } else {
                (0..3)
                    .map(|direction| {
                        coefficients
                            .iter()
                            .enumerate()
                            .map(|(local, &coefficient)| {
                                coefficient * hcurl_tabulation.curl(local, direction)
                            })
                            .sum::<f64>()
                            .abs()
                    })
                    .fold(0.0, f64::max)
            };
            assert!(
                curl_error < 2.0e-8,
                "{} exact-sequence curl residual {curl_error:.3e}",
                family.name()
            );
        }
    }
}

#[test]
fn order2_nedelec_exact_sequence_is_curl_free_in_2d_and_3d() {
    assert_exact_sequence_on_simplex(ScalarFamily::Triangle2d);
    assert_exact_sequence_on_simplex(ScalarFamily::Tetrahedron3d);
}

#[test]
fn w1_05_p3_tensor_and_simplex_families_are_admitted() {
    for element_type in [
        ElementType::Triangle,
        ElementType::Quadrilateral,
        ElementType::Tetrahedron,
        ElementType::Hexahedron,
    ] {
        let shape = try_evaluate_shape(element_type, PolynomialDegree::P3, 0.2, 0.2, 0.2)
            .expect("P3 shape evaluation");
        assert!((shape.values.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        for gradient_sum in (0..shape.grad_dim).map(|axis| {
            shape
                .gradients
                .chunks(shape.grad_dim)
                .map(|gradient| gradient[axis])
                .sum::<f64>()
        }) {
            assert!(gradient_sum.abs() < 1e-12);
        }
    }

    // H(div) has no polynomial-degree constructor: its production space is
    // RT0 only.  Record its RT0 dimensions rather than labeling it a
    // high-order pass.  The basis-level tests cover its flux conservation.
    let hdiv = HdivSpace::new();
    assert_eq!(hdiv.num_local_dofs(ElementType::Triangle), 3);
    assert_eq!(hdiv.num_local_dofs(ElementType::Quadrilateral), 4);
    assert_eq!(hdiv.num_local_dofs(ElementType::Tetrahedron), 4);
    assert_eq!(hdiv.num_local_dofs(ElementType::Hexahedron), 6);

    // Lowest-order tensor-product Nédélec bases are now admitted on
    // quadrilateral/hexahedral cells. Higher-order Nédélec remains a
    // simplex-only contract until its tensor-product interior moments land.
    for order in [1, 2] {
        for (element_type, mesh) in [
            (
                ElementType::Quadrilateral,
                sonium_fem::mesh::unit_square_quads(1),
            ),
            (
                ElementType::Hexahedron,
                sonium_fem::mesh::unit_cube_hexahedra(1),
            ),
        ] {
            let validation = HcurlSpace { order }.validate(&mesh);
            if order == 1 {
                assert_eq!(validation, Ok(()));
            } else {
                assert_eq!(
                    validation,
                    Err(SpaceValidationError::UnsupportedElement {
                        element: 0,
                        element_type,
                        order,
                    })
                );
            }
        }
    }

    assert_eq!(
        HcurlSpace { order: 3 }.validate(&ScalarFamily::Triangle2d.mesh(1)),
        Err(SpaceValidationError::UnsupportedOrder {
            requested: 3,
            supported: "1 and 2",
        })
    );
}
