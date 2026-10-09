//! Independent reference checks for affine hexahedral RT1.

use sonium_fem::basis::raviart_thomas::{rt1_hexahedron, rt1_hexahedron_oriented};
use sonium_fem::mesh::{ElementType, MeshTopology, unit_cube_hexahedra};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};

fn gauss_3() -> [(f64, f64); 3] {
    [
        (-(3.0_f64 / 5.0).sqrt(), 5.0 / 9.0),
        (0.0, 8.0 / 9.0),
        ((3.0_f64 / 5.0).sqrt(), 5.0 / 9.0),
    ]
}

fn face_point(face: usize, a: f64, b: f64) -> ([f64; 3], [f64; 3]) {
    match face {
        0 => ([a, b, -1.0], [0.0, 0.0, -1.0]),
        1 => ([a, b, 1.0], [0.0, 0.0, 1.0]),
        2 => ([a, -1.0, b], [0.0, -1.0, 0.0]),
        3 => ([a, 1.0, b], [0.0, 1.0, 0.0]),
        4 => ([-1.0, a, b], [-1.0, 0.0, 0.0]),
        5 => ([1.0, a, b], [1.0, 0.0, 0.0]),
        _ => unreachable!(),
    }
}

fn face_test(mode: usize, a: f64, b: f64) -> f64 {
    match mode {
        0 => 1.0,
        1 => a,
        2 => b,
        3 => a * b,
        _ => unreachable!(),
    }
}

fn integrate_face(face: usize, basis: usize, mode: usize) -> f64 {
    gauss_3()
        .iter()
        .map(|&(a, wa)| {
            gauss_3()
                .iter()
                .map(|&(b, wb)| {
                    let (point, normal) = face_point(face, a, b);
                    let (values, _) = rt1_hexahedron(point[0], point[1], point[2]);
                    let flux = values[basis]
                        .iter()
                        .zip(normal)
                        .map(|(value, component)| value * component)
                        .sum::<f64>();
                    wa * wb * flux * face_test(mode, a, b)
                })
                .sum::<f64>()
        })
        .sum()
}

fn integrate_cell(basis: usize, component: usize, mode: usize) -> f64 {
    gauss_3()
        .iter()
        .map(|&(xi, wx)| {
            gauss_3()
                .iter()
                .map(|&(eta, wy)| {
                    gauss_3()
                        .iter()
                        .map(|&(zeta, wz)| {
                            let (values, _) = rt1_hexahedron(xi, eta, zeta);
                            let test = match component {
                                0 => face_test(mode, eta, zeta),
                                1 => face_test(mode, xi, zeta),
                                2 => face_test(mode, xi, eta),
                                _ => unreachable!(),
                            };
                            wx * wy * wz * values[basis][component] * test
                        })
                        .sum::<f64>()
                })
                .sum::<f64>()
        })
        .sum()
}

pub fn rt1_hexa_reference_basis_has_identity_face_and_cell_moments() {
    for face in 0..6 {
        for mode in 0..4 {
            for basis in 0..36 {
                let moment = integrate_face(face, basis, mode);
                let expected = if basis == 4 * face + mode { 1.0 } else { 0.0 };
                assert!(
                    (moment - expected).abs() < 2.0e-12,
                    "face {face} mode {mode} basis {basis}: {moment} != {expected}"
                );
            }
        }
    }

    for component in 0..3 {
        for mode in 0..4 {
            for basis in 0..36 {
                let moment = integrate_cell(basis, component, mode);
                let expected = if basis == 24 + 4 * component + mode {
                    1.0
                } else {
                    0.0
                };
                assert!(
                    (moment - expected).abs() < 2.0e-12,
                    "cell component {component} mode {mode} basis {basis}: {moment} != {expected}"
                );
            }
        }
    }
}

pub fn rt1_hexa_reference_divergence_matches_finite_difference() {
    let point = [0.19, -0.23, 0.17];
    let epsilon = 1.0e-6;
    let (_, divergences) = rt1_hexahedron(point[0], point[1], point[2]);
    for basis in 0..36 {
        let (x_plus, _) = rt1_hexahedron(point[0] + epsilon, point[1], point[2]);
        let (x_minus, _) = rt1_hexahedron(point[0] - epsilon, point[1], point[2]);
        let (y_plus, _) = rt1_hexahedron(point[0], point[1] + epsilon, point[2]);
        let (y_minus, _) = rt1_hexahedron(point[0], point[1] - epsilon, point[2]);
        let (z_plus, _) = rt1_hexahedron(point[0], point[1], point[2] + epsilon);
        let (z_minus, _) = rt1_hexahedron(point[0], point[1], point[2] - epsilon);
        let numerical = (x_plus[basis][0] - x_minus[basis][0]) / (2.0 * epsilon)
            + (y_plus[basis][1] - y_minus[basis][1]) / (2.0 * epsilon)
            + (z_plus[basis][2] - z_minus[basis][2]) / (2.0 * epsilon);
        assert!((numerical - divergences[basis]).abs() < 2.0e-8);
    }
}

pub fn rt1_hexa_space_has_numbering_and_oriented_reference_contract() {
    let mesh = unit_cube_hexahedra(1);
    let topology = MeshTopology::build(&mesh);
    let space = HdivSpace::with_order(HdivOrder::Rt1);

    assert!(space.validate(&mesh).is_ok());
    assert_eq!(space.num_local_dofs(ElementType::Hexahedron), 36);
    assert_eq!(space.rt1_global_dof_layout(&mesh), Some((24, 12)));
    assert_eq!(space.num_global_dofs(&mesh), 36);
    let dofs = space.local_to_global(&mesh, 0);
    assert_eq!(dofs.len(), 36);
    let mut sorted = dofs.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..36).collect::<Vec<_>>());

    let tabulation = space.tabulate(&mesh, 0, 0.19, -0.23, 0.17);
    assert_eq!(tabulation.n_dofs, 36);
    assert!(tabulation.values.iter().all(|value| value.is_finite()));
    assert!(tabulation.derivatives.iter().all(|value| value.is_finite()));
    assert_eq!(topology.element_face_indices(0).len(), 6);
    let (&marker, boundary_faces) = topology
        .boundary_faces
        .iter()
        .find(|(_, faces)| !faces.is_empty())
        .expect("unit cube has boundary markers");
    assert_eq!(
        space.boundary_dofs(&mesh, marker).len(),
        4 * boundary_faces.len()
    );

    let (oriented, _) = rt1_hexahedron_oriented(0.19, -0.23, 0.17, [[0, 1, 1, 0]; 6]);
    assert!(oriented.iter().flatten().all(|value| value.is_finite()));
}

#[test]
fn reference_basis_has_identity_moments() {
    rt1_hexa_reference_basis_has_identity_face_and_cell_moments();
}

#[test]
fn reference_divergence_matches_finite_difference() {
    rt1_hexa_reference_divergence_matches_finite_difference();
}

#[test]
fn space_has_numbering_and_orientation_contract() {
    rt1_hexa_space_has_numbering_and_oriented_reference_contract();
}
