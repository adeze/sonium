//! Independent polynomial-completeness gate for tetrahedral H1/P3.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::unit_cube_tetrahedra;
use sonium_fem::spaces::{FunctionSpace, H1Space};

#[test]
fn h1_p3_tetrahedron_reproduces_cubic_fields() {
    let mesh = unit_cube_tetrahedra(2);
    let space = H1Space::new(PolynomialDegree::P3);
    space.validate(&mesh).expect("P3 tetrahedral H1 admission");
    let coordinates = space
        .dof_coordinates(&mesh)
        .expect("P3 tetrahedral DOF coordinates");

    // An affine tetrahedral map preserves polynomial degree, so a complete
    // cubic nodal basis must reproduce this field at every physical point.
    let cubic = |[x, y, z]: [f64; 3]| {
        x.powi(3) - 2.0 * x.powi(2) * y + 0.75 * x * y * z + 1.5 * y.powi(2) - 0.5 * z.powi(3)
            + 0.25 * x
            - 0.8 * y
            + 0.6 * z
            + 1.0
    };
    let cubic_gradient = |[x, y, z]: [f64; 3]| {
        [
            3.0 * x.powi(2) - 4.0 * x * y + 0.75 * y * z + 0.25,
            -2.0 * x.powi(2) + 0.75 * x * z + 3.0 * y - 0.8,
            0.75 * x * y - 1.5 * z.powi(2) + 0.6,
        ]
    };
    let quadratic = |[x, y, z]: [f64; 3]| x * x + y * y + z * z;
    let quadratic_gradient = |[x, y, z]: [f64; 3]| [2.0 * x, 2.0 * y, 2.0 * z];
    let nodal_points = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0 / 3.0, 0.0, 0.0),
        (2.0 / 3.0, 0.0, 0.0),
        (0.0, 1.0 / 3.0, 0.0),
        (0.0, 2.0 / 3.0, 0.0),
        (0.0, 0.0, 1.0 / 3.0),
        (0.0, 0.0, 2.0 / 3.0),
        (2.0 / 3.0, 1.0 / 3.0, 0.0),
        (1.0 / 3.0, 2.0 / 3.0, 0.0),
        (2.0 / 3.0, 0.0, 1.0 / 3.0),
        (1.0 / 3.0, 0.0, 2.0 / 3.0),
        (0.0, 2.0 / 3.0, 1.0 / 3.0),
        (0.0, 1.0 / 3.0, 2.0 / 3.0),
        (1.0 / 3.0, 1.0 / 3.0, 0.0),
        (1.0 / 3.0, 0.0, 1.0 / 3.0),
        (0.0, 1.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    ];

    for element in 0..mesh.num_elements() {
        let local_to_global = space.local_to_global(&mesh, element);
        if element == 0 {
            for (node, &(xi, eta, zeta)) in nodal_points.iter().enumerate() {
                let tabulation = space.tabulate(&mesh, element, xi, eta, zeta);
                let interpolated = local_to_global
                    .iter()
                    .enumerate()
                    .map(|(local, &global)| tabulation.value(local) * cubic(coordinates[global]))
                    .sum::<f64>();
                let exact = cubic(tabulation.physical_coords);
                assert!(
                    (interpolated - exact).abs() < 5.0e-12,
                    "element {element} nodal point {node} interpolated {interpolated} vs {exact}"
                );
            }
        }
        for &(xi, eta, zeta) in &[(0.11, 0.17, 0.23), (0.32, 0.19, 0.21), (0.07, 0.41, 0.13)] {
            let tabulation = space.tabulate(&mesh, element, xi, eta, zeta);
            let interpolated = local_to_global
                .iter()
                .enumerate()
                .map(|(local, &global)| tabulation.value(local) * cubic(coordinates[global]))
                .sum::<f64>();
            let exact = cubic(tabulation.physical_coords);
            assert!(
                (interpolated - exact).abs() < 5.0e-12,
                "element {element} at ({xi}, {eta}, {zeta}) interpolated {interpolated} vs {exact}"
            );
            let mut interpolated_gradient = [0.0; 3];
            for (local, &global) in local_to_global.iter().enumerate() {
                let coefficient = cubic(coordinates[global]);
                for (direction, value) in interpolated_gradient.iter_mut().enumerate() {
                    *value += coefficient * tabulation.grad(local, direction);
                }
            }
            let exact_gradient = cubic_gradient(tabulation.physical_coords);
            for direction in 0..3 {
                assert!(
                    (interpolated_gradient[direction] - exact_gradient[direction]).abs() < 5.0e-10,
                    "element {element} at ({xi}, {eta}, {zeta}) gradient {direction}: interpolated={} vs {}",
                    interpolated_gradient[direction],
                    exact_gradient[direction]
                );
            }
            let mut quadratic_gradient_interpolated = [0.0; 3];
            for (local, &global) in local_to_global.iter().enumerate() {
                let coefficient = quadratic(coordinates[global]);
                for (direction, value) in quadratic_gradient_interpolated.iter_mut().enumerate() {
                    *value += coefficient * tabulation.grad(local, direction);
                }
            }
            let quadratic_gradient_exact = quadratic_gradient(tabulation.physical_coords);
            for direction in 0..3 {
                assert!(
                    (quadratic_gradient_interpolated[direction]
                        - quadratic_gradient_exact[direction])
                        .abs()
                        < 5.0e-10,
                    "element {element} quadratic gradient {direction}: interpolated={} vs {}",
                    quadratic_gradient_interpolated[direction],
                    quadratic_gradient_exact[direction]
                );
            }
        }
    }
}
