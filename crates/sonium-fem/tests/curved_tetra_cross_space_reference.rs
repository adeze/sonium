//! Independent reference checks for curved tetrahedral physical blocks.
//!
//! The pointwise reference evaluator follows the covariant/contravariant
//! Piola identities documented by MFEM and FEniCS Basix:
//!
//! - H(curl): `curl(u) = J curl(u_hat) / det(J)`
//! - H(div): `div(v) = div(v_hat) / det(J)`
//!
//! It deliberately does not call [`HcurlSpace::tabulate`] or
//! [`HdivSpace::tabulate`], so these tests can catch an implementation that
//! is internally self-consistent but maps a curved physical block wrongly.

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_two_spaces_checked};
use sonium_fem::basis::{Jacobian, PolynomialDegree, evaluate_shape};
use sonium_fem::basis::{nedelec, raviart_thomas};
use sonium_fem::forms::{CurlCurlForm, DivergenceDivergenceForm};
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, HcurlSpace, HdivOrder, HdivSpace};
use std::collections::BTreeMap;

fn curved_tetrahedron() -> Mesh {
    let mut mesh = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    let mut mesh = mesh.to_p2();

    // The first P2 edge node is the midpoint of the (0, 1) edge. This makes
    // the isoparametric Jacobian vary while keeping it nonsingular.
    let curved_node = mesh.elements[0].nodes[4];
    mesh.nodes[curved_node].y += 0.12;
    mesh
}

fn curved_triangle() -> Mesh {
    let mut mesh = Mesh::new(2);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
    let mut mesh = mesh.to_p2();
    let curved_node = mesh.elements[0].nodes[3];
    mesh.nodes[curved_node].y += 0.12;
    mesh
}

fn entries(matrix: &TripletMatrix<f64>) -> BTreeMap<(usize, usize), f64> {
    let mut result = BTreeMap::new();
    for index in 0..matrix.nnz() {
        *result
            .entry((matrix.rows[index], matrix.cols[index]))
            .or_insert(0.0) += matrix.values[index];
    }
    result
}

fn curved_jacobian(mesh: &Mesh, point: &sonium_fem::quadrature::QuadraturePoint) -> Jacobian {
    let element = &mesh.elements[0];
    let shape = evaluate_shape(
        ElementType::Tetrahedron,
        PolynomialDegree::P2,
        point.xi(),
        point.eta(),
        point.zeta(),
    );
    let coordinates: Vec<[f64; 3]> = element
        .nodes
        .iter()
        .map(|&node| {
            let point = mesh.nodes[node];
            [point.x, point.y, point.z]
        })
        .collect();
    Jacobian::from_3d(&shape.gradients, shape.grad_dim, &coordinates)
}

fn curved_triangle_jacobian(
    mesh: &Mesh,
    point: &sonium_fem::quadrature::QuadraturePoint,
) -> Jacobian {
    let element = &mesh.elements[0];
    let shape = evaluate_shape(
        ElementType::Triangle,
        PolynomialDegree::P2,
        point.xi(),
        point.eta(),
        point.zeta(),
    );
    let coordinates: Vec<[f64; 2]> = element
        .nodes
        .iter()
        .map(|&node| {
            let point = mesh.nodes[node];
            [point.x, point.y]
        })
        .collect();
    Jacobian::from_2d(&shape.gradients, shape.grad_dim, &coordinates)
}

fn map_curl(jacobian: &Jacobian, reference: [f64; 3]) -> [f64; 3] {
    let determinant = jacobian.det;
    [
        (jacobian.matrix[0][0] * reference[0]
            + jacobian.matrix[0][1] * reference[1]
            + jacobian.matrix[0][2] * reference[2])
            / determinant,
        (jacobian.matrix[1][0] * reference[0]
            + jacobian.matrix[1][1] * reference[1]
            + jacobian.matrix[1][2] * reference[2])
            / determinant,
        (jacobian.matrix[2][0] * reference[0]
            + jacobian.matrix[2][1] * reference[1]
            + jacobian.matrix[2][2] * reference[2])
            / determinant,
    ]
}

fn reference_curls(mesh: &Mesh, order: usize, quad_order: usize) -> Vec<Vec<[f64; 3]>> {
    let rule = QuadratureRule::new(ElementType::Tetrahedron, quad_order);
    let topology = sonium_fem::mesh::MeshTopology::build(mesh);
    let signs = topology.element_edge_signs(0);
    rule.iter()
        .map(|point| {
            let curls: Vec<[f64; 3]> = match order {
                1 => nedelec::nedelec1_tetrahedron(point.xi(), point.eta(), point.zeta())
                    .1
                    .to_vec(),
                2 => nedelec::nedelec2_tetrahedron(point.xi(), point.eta(), point.zeta())
                    .1
                    .to_vec(),
                _ => unreachable!(),
            };
            let jacobian = curved_jacobian(mesh, point);
            let determinant = jacobian.det;
            assert!(
                determinant > 0.0,
                "curved tetrahedron Jacobian must stay positive"
            );
            curls
                .iter()
                .enumerate()
                .map(|(index, &curl)| {
                    let sign = if order == 1 { signs[index] as f64 } else { 1.0 };
                    let mapped = map_curl(&jacobian, curl);
                    [sign * mapped[0], sign * mapped[1], sign * mapped[2]]
                })
                .collect()
        })
        .collect()
}

fn reference_curl_curl_block(
    mesh: &Mesh,
    test_order: usize,
    trial_order: usize,
    quad_order: usize,
) -> Vec<Vec<f64>> {
    let rule = QuadratureRule::new(ElementType::Tetrahedron, quad_order);
    let test_curls = reference_curls(mesh, test_order, quad_order);
    let trial_curls = reference_curls(mesh, trial_order, quad_order);
    let test_count = if test_order == 1 { 6 } else { 20 };
    let trial_count = if trial_order == 1 { 6 } else { 20 };
    let mut block = vec![vec![0.0; trial_count]; test_count];

    for (point_index, point) in rule.iter().enumerate() {
        let jacobian = curved_jacobian(mesh, point);
        let weight = point.weight * jacobian.det.abs();
        for (row, test_curl) in test_curls[point_index].iter().enumerate() {
            for (column, trial_curl) in trial_curls[point_index].iter().enumerate() {
                let dot = test_curl
                    .iter()
                    .zip(trial_curl)
                    .map(|(left, right)| left * right)
                    .sum::<f64>();
                block[row][column] += weight * dot;
            }
        }
    }
    block
}

fn reference_curl_curl_block_2d(
    mesh: &Mesh,
    test_order: usize,
    trial_order: usize,
    quad_order: usize,
) -> Vec<Vec<f64>> {
    let rule = QuadratureRule::new(ElementType::Triangle, quad_order);
    let topology = sonium_fem::mesh::MeshTopology::build(mesh);
    let signs = topology.element_edge_signs(0);
    let test_count = if test_order == 1 { 3 } else { 8 };
    let trial_count = if trial_order == 1 { 3 } else { 8 };
    let mut block = vec![vec![0.0; trial_count]; test_count];

    for point in rule.iter() {
        let test_curls: Vec<f64> = match test_order {
            1 => nedelec::nedelec1_triangle(point.xi(), point.eta())
                .1
                .to_vec(),
            2 => nedelec::nedelec2_triangle(point.xi(), point.eta())
                .1
                .to_vec(),
            _ => unreachable!(),
        };
        let trial_curls: Vec<f64> = match trial_order {
            1 => nedelec::nedelec1_triangle(point.xi(), point.eta())
                .1
                .to_vec(),
            2 => nedelec::nedelec2_triangle(point.xi(), point.eta())
                .1
                .to_vec(),
            _ => unreachable!(),
        };
        let jacobian = curved_triangle_jacobian(mesh, point);
        let weight = point.weight * jacobian.det.abs();
        for (row, &test_curl) in test_curls.iter().enumerate() {
            for (column, &trial_curl) in trial_curls.iter().enumerate() {
                let test_sign = if test_order == 1 {
                    signs[row] as f64
                } else {
                    1.0
                };
                let trial_sign = if trial_order == 1 {
                    signs[column] as f64
                } else {
                    1.0
                };
                block[row][column] += weight * test_sign * trial_sign * test_curl * trial_curl
                    / (jacobian.det * jacobian.det);
            }
        }
    }
    block
}

fn reference_divergences(mesh: &Mesh, order: HdivOrder, quad_order: usize) -> Vec<Vec<f64>> {
    let rule = QuadratureRule::new(ElementType::Tetrahedron, quad_order);
    let topology = sonium_fem::mesh::MeshTopology::build(mesh);
    let signs = topology.element_face_signs(0);
    rule.iter()
        .map(|point| {
            let divergences: Vec<f64> = match order {
                HdivOrder::Rt0 => {
                    raviart_thomas::rt0_tetrahedron(point.xi(), point.eta(), point.zeta())
                        .1
                        .to_vec()
                }
                HdivOrder::Rt1 => {
                    raviart_thomas::rt1_tetrahedron(point.xi(), point.eta(), point.zeta())
                        .1
                        .to_vec()
                }
            };
            let determinant = curved_jacobian(mesh, point).det;
            assert!(
                determinant > 0.0,
                "curved tetrahedron Jacobian must stay positive"
            );
            divergences
                .iter()
                .enumerate()
                .map(|(index, &divergence)| {
                    let sign = match order {
                        HdivOrder::Rt0 => signs[index] as f64,
                        HdivOrder::Rt1 if index < 12 => signs[index / 3] as f64,
                        HdivOrder::Rt1 => 1.0,
                    };
                    sign * divergence / determinant
                })
                .collect()
        })
        .collect()
}

fn reference_divergence_block(
    mesh: &Mesh,
    test_order: HdivOrder,
    trial_order: HdivOrder,
    quad_order: usize,
) -> Vec<Vec<f64>> {
    let rule = QuadratureRule::new(ElementType::Tetrahedron, quad_order);
    let test_divergences = reference_divergences(mesh, test_order, quad_order);
    let trial_divergences = reference_divergences(mesh, trial_order, quad_order);
    let test_count = match test_order {
        HdivOrder::Rt0 => 4,
        HdivOrder::Rt1 => 15,
    };
    let trial_count = match trial_order {
        HdivOrder::Rt0 => 4,
        HdivOrder::Rt1 => 15,
    };
    let mut block = vec![vec![0.0; trial_count]; test_count];

    for (point_index, point) in rule.iter().enumerate() {
        let jacobian = curved_jacobian(mesh, point);
        let weight = point.weight * jacobian.det.abs();
        for (row, test_divergence) in test_divergences[point_index].iter().enumerate() {
            for (column, trial_divergence) in trial_divergences[point_index].iter().enumerate() {
                block[row][column] += weight * test_divergence * trial_divergence;
            }
        }
    }
    block
}

fn reference_divergence_block_2d(
    mesh: &Mesh,
    test_order: HdivOrder,
    trial_order: HdivOrder,
    quad_order: usize,
) -> Vec<Vec<f64>> {
    let rule = QuadratureRule::new(ElementType::Triangle, quad_order);
    let topology = sonium_fem::mesh::MeshTopology::build(mesh);
    let signs = topology.element_edge_signs(0);
    let test_count = match test_order {
        HdivOrder::Rt0 => 3,
        HdivOrder::Rt1 => 8,
    };
    let trial_count = match trial_order {
        HdivOrder::Rt0 => 3,
        HdivOrder::Rt1 => 8,
    };
    let mut block = vec![vec![0.0; trial_count]; test_count];

    for point in rule.iter() {
        let test_divergences: Vec<f64> = match test_order {
            HdivOrder::Rt0 => raviart_thomas::rt0_triangle(point.xi(), point.eta())
                .1
                .to_vec(),
            HdivOrder::Rt1 => raviart_thomas::rt1_triangle(point.xi(), point.eta())
                .1
                .to_vec(),
        };
        let trial_divergences: Vec<f64> = match trial_order {
            HdivOrder::Rt0 => raviart_thomas::rt0_triangle(point.xi(), point.eta())
                .1
                .to_vec(),
            HdivOrder::Rt1 => raviart_thomas::rt1_triangle(point.xi(), point.eta())
                .1
                .to_vec(),
        };
        let jacobian = curved_triangle_jacobian(mesh, point);
        let weight = point.weight * jacobian.det.abs();
        for (row, &test_divergence) in test_divergences.iter().enumerate() {
            for (column, &trial_divergence) in trial_divergences.iter().enumerate() {
                let test_sign = match test_order {
                    HdivOrder::Rt0 => signs[row] as f64,
                    HdivOrder::Rt1 if row < 2 * signs.len() => {
                        if row % 2 == 0 {
                            signs[row / 2] as f64
                        } else {
                            1.0
                        }
                    }
                    HdivOrder::Rt1 => 1.0,
                };
                let trial_sign = match trial_order {
                    HdivOrder::Rt0 => signs[column] as f64,
                    HdivOrder::Rt1 if column < 2 * signs.len() => {
                        if column % 2 == 0 {
                            signs[column / 2] as f64
                        } else {
                            1.0
                        }
                    }
                    HdivOrder::Rt1 => 1.0,
                };
                block[row][column] +=
                    weight * test_sign * trial_sign * test_divergence * trial_divergence
                        / (jacobian.det * jacobian.det);
            }
        }
    }
    block
}

fn assert_block_matches_reference(
    actual: &TripletMatrix<f64>,
    test_dofs: &[usize],
    trial_dofs: &[usize],
    reference: &[Vec<f64>],
) {
    let actual = entries(actual);
    let mut max_error: f64 = 0.0;
    for (row, &test_dof) in test_dofs.iter().enumerate() {
        for (column, &trial_dof) in trial_dofs.iter().enumerate() {
            let actual_value = actual.get(&(test_dof, trial_dof)).copied().unwrap_or(0.0);
            max_error = max_error.max((actual_value - reference[row][column]).abs());
        }
    }
    assert!(
        max_error <= 3.0e-10,
        "independent curved reference error {max_error}"
    );
}

#[test]
fn curved_tetrahedral_hcurl_curl_curl_matches_independent_reference() {
    let mesh = curved_tetrahedron();
    let order1 = HcurlSpace::new();
    let order2 = HcurlSpace::order2();
    let forward = assemble_bilinear_two_spaces_checked(&CurlCurlForm, &order1, &order2, &mesh, 10)
        .expect("curved tetrahedral H(curl) curl-curl assembly must succeed");
    let reverse = assemble_bilinear_two_spaces_checked(&CurlCurlForm, &order2, &order1, &mesh, 10)
        .expect("swapped curved tetrahedral H(curl) curl-curl assembly must succeed");
    assert_eq!(
        (forward.num_rows, forward.num_cols),
        (order1.num_global_dofs(&mesh), order2.num_global_dofs(&mesh))
    );
    assert_eq!(
        (reverse.num_rows, reverse.num_cols),
        (order2.num_global_dofs(&mesh), order1.num_global_dofs(&mesh))
    );
    let test_dofs = order1.local_to_global(&mesh, 0);
    let trial_dofs = order2.local_to_global(&mesh, 0);
    assert_block_matches_reference(
        &forward,
        &test_dofs,
        &trial_dofs,
        &reference_curl_curl_block(&mesh, 1, 2, 18),
    );
    assert_block_matches_reference(
        &reverse,
        &trial_dofs,
        &test_dofs,
        &reference_curl_curl_block(&mesh, 2, 1, 18),
    );
}

#[test]
fn curved_triangular_physical_blocks_match_independent_reference() {
    let mesh = curved_triangle();
    let order1 = HcurlSpace::new();
    let order2 = HcurlSpace::order2();
    let hcurl = assemble_bilinear_two_spaces_checked(&CurlCurlForm, &order1, &order2, &mesh, 10)
        .expect("curved triangular H(curl) assembly must succeed");
    assert_block_matches_reference(
        &hcurl,
        &order1.local_to_global(&mesh, 0),
        &order2.local_to_global(&mesh, 0),
        &reference_curl_curl_block_2d(&mesh, 1, 2, 18),
    );

    let rt0 = HdivSpace::new();
    let rt1 = HdivSpace::with_order(HdivOrder::Rt1);
    let hdiv =
        assemble_bilinear_two_spaces_checked(&DivergenceDivergenceForm, &rt0, &rt1, &mesh, 10)
            .expect("curved triangular H(div) assembly must succeed");
    assert_block_matches_reference(
        &hdiv,
        &rt0.local_to_global(&mesh, 0),
        &rt1.local_to_global(&mesh, 0),
        &reference_divergence_block_2d(&mesh, HdivOrder::Rt0, HdivOrder::Rt1, 18),
    );
}

#[test]
fn curved_tetrahedral_hdiv_divergence_matches_independent_reference() {
    let mesh = curved_tetrahedron();
    let rt0 = HdivSpace::new();
    let rt1 = HdivSpace::with_order(HdivOrder::Rt1);
    let forward =
        assemble_bilinear_two_spaces_checked(&DivergenceDivergenceForm, &rt0, &rt1, &mesh, 10)
            .expect("curved tetrahedral H(div) divergence assembly must succeed");
    let reverse =
        assemble_bilinear_two_spaces_checked(&DivergenceDivergenceForm, &rt1, &rt0, &mesh, 10)
            .expect("swapped curved tetrahedral H(div) divergence assembly must succeed");
    assert_eq!(
        (forward.num_rows, forward.num_cols),
        (rt0.num_global_dofs(&mesh), rt1.num_global_dofs(&mesh))
    );
    assert_eq!(
        (reverse.num_rows, reverse.num_cols),
        (rt1.num_global_dofs(&mesh), rt0.num_global_dofs(&mesh))
    );
    let test_dofs = rt0.local_to_global(&mesh, 0);
    let trial_dofs = rt1.local_to_global(&mesh, 0);
    assert_block_matches_reference(
        &forward,
        &test_dofs,
        &trial_dofs,
        &reference_divergence_block(&mesh, HdivOrder::Rt0, HdivOrder::Rt1, 18),
    );
    assert_block_matches_reference(
        &reverse,
        &trial_dofs,
        &test_dofs,
        &reference_divergence_block(&mesh, HdivOrder::Rt1, HdivOrder::Rt0, 18),
    );
}

#[test]
fn curved_tetrahedral_cross_space_assembly_rejects_malformed_topology() {
    let mut mesh = curved_tetrahedron();
    mesh.elements[0].nodes.pop();
    assert!(
        assemble_bilinear_two_spaces_checked(
            &CurlCurlForm,
            &HcurlSpace::new(),
            &HcurlSpace::order2(),
            &mesh,
            8,
        )
        .is_err()
    );
    assert!(
        assemble_bilinear_two_spaces_checked(
            &DivergenceDivergenceForm,
            &HdivSpace::new(),
            &HdivSpace::with_order(HdivOrder::Rt1),
            &mesh,
            8,
        )
        .is_err()
    );
}
