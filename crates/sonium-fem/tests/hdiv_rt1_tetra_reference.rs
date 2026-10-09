//! Independent reference checks for affine tetrahedral RT1.
//!
//! The moments below are evaluated with exact low-order simplex rules rather
//! than by calling any production integration or assembly helper.  The test
//! therefore checks the reference basis duality, divergence, and the
//! topology-owned 3-D numbering contract independently.

use sonium_fem::basis::raviart_thomas::rt1_tetrahedron;
use sonium_fem::mesh::{MeshTopology, unit_cube_tetrahedra};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};

const TRIANGLE_RULE: [([f64; 2], f64); 4] = [
    ([1.0 / 3.0, 1.0 / 3.0], -27.0 / 96.0),
    ([0.6, 0.2], 25.0 / 96.0),
    ([0.2, 0.6], 25.0 / 96.0),
    ([0.2, 0.2], 25.0 / 96.0),
];

const TETRA_RULE: [([f64; 3], f64); 5] = [
    ([0.25, 0.25, 0.25], -2.0 / 15.0),
    ([0.5, 1.0 / 6.0, 1.0 / 6.0], 3.0 / 40.0),
    ([1.0 / 6.0, 0.5, 1.0 / 6.0], 3.0 / 40.0),
    ([1.0 / 6.0, 1.0 / 6.0, 0.5], 3.0 / 40.0),
    ([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], 3.0 / 40.0),
];

fn face_geometry(face: usize, a: f64, b: f64) -> ([f64; 3], [f64; 3], f64) {
    match face {
        0 => ([a, b, 0.0], [0.0, 0.0, -1.0], 1.0),
        1 => ([a, 0.0, b], [0.0, -1.0, 0.0], 1.0),
        2 => ([0.0, a, b], [-1.0, 0.0, 0.0], 1.0),
        3 => (
            [a, b, 1.0 - a - b],
            [1.0 / 3.0_f64.sqrt(); 3],
            3.0_f64.sqrt(),
        ),
        _ => panic!("reference tetrahedron has four faces"),
    }
}

fn face_monomial(face: usize, mode: usize, point: [f64; 3]) -> f64 {
    match mode {
        0 => 1.0,
        1 => match face {
            0 | 1 => point[0],
            2 | 3 => point[1],
            _ => unreachable!(),
        },
        2 => match face {
            0 => point[1],
            1..=3 => point[2],
            _ => unreachable!(),
        },
        _ => panic!("reference tetrahedron has three face modes"),
    }
}

pub fn rt1_tetra_reference_basis_has_identity_face_and_cell_moments() {
    for face in 0..4 {
        for mode in 0..3 {
            for basis_index in 0..15 {
                let moment = TRIANGLE_RULE
                    .iter()
                    .map(|&([a, b], weight)| {
                        let (point, normal, surface_jacobian) = face_geometry(face, a, b);
                        let (basis, _) = rt1_tetrahedron(point[0], point[1], point[2]);
                        let flux = basis[basis_index]
                            .iter()
                            .zip(normal)
                            .map(|(value, normal_component)| value * normal_component)
                            .sum::<f64>();
                        weight * surface_jacobian * flux * face_monomial(face, mode, point)
                    })
                    .sum::<f64>();
                let expected = if basis_index == 3 * face + mode {
                    1.0
                } else {
                    0.0
                };
                assert!(
                    (moment - expected).abs() < 2.0e-11,
                    "face {face} mode {mode} basis {basis_index}: {moment} != {expected}"
                );
            }
        }
    }

    for component in 0..3 {
        for basis_index in 0..15 {
            let moment = TETRA_RULE
                .iter()
                .map(|&([xi, eta, zeta], weight)| {
                    let (basis, _) = rt1_tetrahedron(xi, eta, zeta);
                    weight * basis[basis_index][component]
                })
                .sum::<f64>();
            let expected = if basis_index == 12 + component {
                1.0
            } else {
                0.0
            };
            assert!(
                (moment - expected).abs() < 2.0e-11,
                "cell component {component} basis {basis_index}: {moment} != {expected}"
            );
        }
    }
}

pub fn rt1_tetra_reference_divergence_matches_finite_difference() {
    let point = [0.19, 0.23, 0.17];
    let epsilon = 1.0e-6;
    let (basis, divergences) = rt1_tetrahedron(point[0], point[1], point[2]);

    for basis_index in 0..basis.len() {
        let (x_plus, _) = rt1_tetrahedron(point[0] + epsilon, point[1], point[2]);
        let (x_minus, _) = rt1_tetrahedron(point[0] - epsilon, point[1], point[2]);
        let (y_plus, _) = rt1_tetrahedron(point[0], point[1] + epsilon, point[2]);
        let (y_minus, _) = rt1_tetrahedron(point[0], point[1] - epsilon, point[2]);
        let (z_plus, _) = rt1_tetrahedron(point[0], point[1], point[2] + epsilon);
        let (z_minus, _) = rt1_tetrahedron(point[0], point[1], point[2] - epsilon);
        let numerical = (x_plus[basis_index][0] - x_minus[basis_index][0]) / (2.0 * epsilon)
            + (y_plus[basis_index][1] - y_minus[basis_index][1]) / (2.0 * epsilon)
            + (z_plus[basis_index][2] - z_minus[basis_index][2]) / (2.0 * epsilon);
        assert!(
            (numerical - divergences[basis_index]).abs() < 2.0e-8,
            "basis {basis_index}: finite-difference divergence {numerical} != {}",
            divergences[basis_index]
        );
    }
}

pub fn rt1_tetra_space_has_face_cell_numbering_and_piola_tabulation() {
    let mesh = unit_cube_tetrahedra(1);
    let topology = MeshTopology::build(&mesh);
    let space = HdivSpace::with_order(HdivOrder::Rt1);

    assert!(space.validate(&mesh).is_ok());
    assert_eq!(
        space.num_local_dofs(sonium_fem::mesh::ElementType::Tetrahedron),
        15
    );
    assert_eq!(
        space.rt1_global_dof_layout(&mesh),
        Some((3 * topology.num_faces(), 3 * mesh.elements.len()))
    );
    assert_eq!(
        space.num_global_dofs(&mesh),
        3 * topology.num_faces() + 3 * mesh.elements.len()
    );

    for element in 0..mesh.elements.len() {
        let dofs = space.local_to_global(&mesh, element);
        assert_eq!(dofs.len(), 15);
        for local_face in 0..4 {
            let global_face = topology.element_face_indices(element)[local_face];
            for mode in 0..3 {
                assert_eq!(dofs[3 * local_face + mode], 3 * global_face + mode);
            }
        }
        let cell_offset = 3 * topology.num_faces() + 3 * element;
        assert_eq!(
            &dofs[12..15],
            &[cell_offset, cell_offset + 1, cell_offset + 2]
        );

        let tabulation = space.tabulate(&mesh, element, 0.19, 0.23, 0.17);
        assert_eq!(tabulation.n_dofs, 15);
        assert!(tabulation.values.iter().all(|value| value.is_finite()));
        assert!(tabulation.derivatives.iter().all(|value| value.is_finite()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_basis_moments() {
        rt1_tetra_reference_basis_has_identity_face_and_cell_moments();
    }

    #[test]
    fn reference_divergence() {
        rt1_tetra_reference_divergence_matches_finite_difference();
    }

    #[test]
    fn space_numbering_and_piola() {
        rt1_tetra_space_has_face_cell_numbering_and_piola_tabulation();
    }
}
