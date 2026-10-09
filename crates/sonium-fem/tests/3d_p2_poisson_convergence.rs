//! W1-08 3D physical convergence evidence for the scalar P2 path.
//!
//! The existing W1 packets verify three-dimensional P2 interpolation and
//! assembled-operator conservation, but they do not solve a three-dimensional
//! physical problem.  This test closes that gap through the public production
//! path: checked stiffness/source assembly, generic Q2/P2 boundary elimination,
//! the FEM direct solver, and an independent manufactured Poisson reference.

use num_complex::Complex64;
use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_checked, assemble_linear_checked};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::{FemSystem, try_apply_homogeneous_dirichlet};
use sonium_fem::forms::{LaplacianForm, SourceForm};
use sonium_fem::mesh::Mesh;
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::solver::{SolverConfig, SolverType, solve_csr_with_guess};
use sonium_fem::spaces::{FunctionSpace, H1Space};

#[derive(Clone, Copy, Debug)]
enum ScalarFamily {
    Tetrahedron3d,
    Hexahedron3d,
    P3Tetrahedron3d,
}

impl ScalarFamily {
    fn mesh(self, refinement: usize) -> Mesh {
        match self {
            Self::Tetrahedron3d => sonium_fem::mesh::unit_cube_tetrahedra(refinement),
            Self::Hexahedron3d => sonium_fem::mesh::unit_cube_hexahedra(refinement),
            Self::P3Tetrahedron3d => sonium_fem::mesh::unit_cube_tetrahedra(refinement),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Tetrahedron3d => "P2-tetrahedron-3d",
            Self::Hexahedron3d => "Q2-hexahedron-3d",
            Self::P3Tetrahedron3d => "P3-tetrahedron-3d",
        }
    }
}

impl ScalarFamily {
    fn degree(self) -> PolynomialDegree {
        match self {
            Self::Tetrahedron3d | Self::Hexahedron3d => PolynomialDegree::P2,
            Self::P3Tetrahedron3d => PolynomialDegree::P3,
        }
    }

    fn manufactured(self, [x, y, z]: [f64; 3]) -> (f64, [f64; 3], f64) {
        match self {
            Self::P3Tetrahedron3d => (
                x.powi(4) + y.powi(4) + z.powi(4),
                [4.0 * x.powi(3), 4.0 * y.powi(3), 4.0 * z.powi(3)],
                -12.0 * (x.powi(2) + y.powi(2) + z.powi(2)),
            ),
            Self::Tetrahedron3d | Self::Hexahedron3d => manufactured([x, y, z]),
        }
    }
}

/// Smooth homogeneous-Dirichlet manufactured field and its Poisson source.
///
/// The three quadratic boundary factors make the field vanish on every face
/// of the unit cube.  The affine factor keeps the field outside both the total
/// degree-2 tetrahedral space and the tensor-product Q2 space, so refinement
/// measures the actual physical discretisation error rather than roundoff.
fn manufactured([x, y, z]: [f64; 3]) -> (f64, [f64; 3], f64) {
    let a = x * (1.0 - x);
    let b = y * (1.0 - y);
    let d = z * (1.0 - z);
    let da = 1.0 - 2.0 * x;
    let db = 1.0 - 2.0 * y;
    let dd = 1.0 - 2.0 * z;
    let alpha = 0.2;
    let beta = 0.15;
    let gamma = 0.1;
    let c = 1.0 + alpha * x + beta * y + gamma * z;

    let value = a * b * d * c;
    let gradient = [
        da * b * d * c + alpha * a * b * d,
        a * db * d * c + beta * a * b * d,
        a * b * dd * c + gamma * a * b * d,
    ];
    let laplacian = b * d * (-2.0 * c + 2.0 * alpha * da)
        + a * d * (-2.0 * c + 2.0 * beta * db)
        + a * b * (-2.0 * c + 2.0 * gamma * dd);
    (value, gradient, -laplacian)
}

#[derive(Clone, Copy, Debug, Default)]
struct Errors {
    l2: f64,
    h1_seminorm: f64,
    residual: f64,
    constrained_dofs: usize,
}

fn solve_and_measure(family: ScalarFamily, refinement: usize) -> Errors {
    let mesh = family.mesh(refinement);
    assert!(
        !mesh.boundaries.is_empty(),
        "{} has no boundary",
        family.name()
    );
    assert!(
        mesh.boundaries.iter().all(|boundary| boundary.marker == 0),
        "{} unit-cube boundaries must use marker 0",
        family.name()
    );

    let space = H1Space::new(family.degree());
    let quadrature_order = if matches!(family, ScalarFamily::P3Tetrahedron3d) {
        8
    } else {
        7
    };
    let stiffness =
        assemble_bilinear_checked(&LaplacianForm, &space, &space, &mesh, quadrature_order)
            .unwrap_or_else(|error| panic!("{} stiffness assembly failed: {error}", family.name()));
    let source = SourceForm::new(|x, y, z| family.manufactured([x, y, z]).2);
    let rhs = assemble_linear_checked(&source, &space, &mesh, quadrature_order)
        .unwrap_or_else(|error| panic!("{} source assembly failed: {error}", family.name()));

    let mut system = FemSystem::new(stiffness, rhs);
    if matches!(family, ScalarFamily::P3Tetrahedron3d) {
        sonium_fem::boundary::try_apply_dirichlet_values(
            &mut system,
            &mesh,
            &space,
            0,
            &|x, y, z| family.manufactured([x, y, z]).0,
            None,
        )
        .unwrap_or_else(|error| panic!("{} Dirichlet elimination failed: {error}", family.name()));
    } else {
        try_apply_homogeneous_dirichlet(&mut system, &mesh, &space, 0).unwrap_or_else(|error| {
            panic!("{} Dirichlet elimination failed: {error}", family.name())
        });
    }
    let constrained_dofs = system.dirichlet_dofs.len();
    assert!(
        constrained_dofs > 0,
        "{} has no constrained DOFs",
        family.name()
    );

    // The generic real-valued forms adapt to the shared FEM complex CSR
    // solver without changing the assembled operator or manufactured source.
    let mut complex_matrix = TripletMatrix::new(system.matrix.num_rows, system.matrix.num_cols);
    for ((&row, &column), &value) in system
        .matrix
        .rows
        .iter()
        .zip(&system.matrix.cols)
        .zip(&system.matrix.values)
    {
        complex_matrix.push(row, column, Complex64::new(value, 0.0));
    }
    let matrix = complex_matrix.to_csr();
    let rhs: Vec<Complex64> = system
        .rhs
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect();
    let config = SolverConfig {
        solver_type: SolverType::Direct,
        ..SolverConfig::default()
    };
    let solution = solve_csr_with_guess(&matrix, &rhs, None, &config)
        .unwrap_or_else(|error| panic!("{} direct solve failed: {error}", family.name()));
    assert!(
        solution.converged,
        "{} solve did not converge",
        family.name()
    );
    assert!(
        solution.residual < 1.0e-10,
        "{} residual was {:.3e}",
        family.name(),
        solution.residual
    );
    if matches!(family, ScalarFamily::P3Tetrahedron3d) && refinement == 1 {
        let coordinates = space
            .dof_coordinates(&mesh)
            .expect("P3 coordinates for boundary check");
        for dof in space.boundary_dofs(&mesh, 0) {
            let expected = family.manufactured(coordinates[dof]).0;
            assert!(
                (solution.values[dof].re - expected).abs() < 1.0e-10,
                "P3 boundary DOF {dof}: got {}, expected {expected}",
                solution.values[dof].re
            );
        }
    }

    let mut l2_squared = 0.0;
    let mut h1_squared = 0.0;
    for element_idx in 0..mesh.num_elements() {
        let element_type = mesh.elements[element_idx].element_type;
        let quadrature = QuadratureRule::new(element_type, quadrature_order);
        let local_dofs = space.local_to_global(&mesh, element_idx);
        for point in quadrature.iter() {
            let tabulation =
                space.tabulate(&mesh, element_idx, point.xi(), point.eta(), point.zeta());
            let mut value = 0.0;
            let mut gradient = [0.0; 3];
            for (local, &global) in local_dofs.iter().enumerate() {
                let coefficient = solution.values[global];
                assert!(coefficient.im.abs() < 1.0e-11);
                value += coefficient.re * tabulation.value(local);
                for (direction, component) in gradient.iter_mut().enumerate() {
                    *component += coefficient.re * tabulation.grad(local, direction);
                }
            }
            let (exact_value, exact_gradient, _) = family.manufactured(tabulation.physical_coords);
            let weight = point.weight * tabulation.det_j;
            l2_squared += weight * (value - exact_value).powi(2);
            h1_squared += weight
                * gradient
                    .iter()
                    .zip(exact_gradient)
                    .map(|(&actual, expected)| (actual - expected).powi(2))
                    .sum::<f64>();
        }
    }

    Errors {
        l2: l2_squared.sqrt(),
        h1_seminorm: h1_squared.sqrt(),
        residual: solution.residual,
        constrained_dofs,
    }
}

fn refinement_rate(coarse: f64, fine: f64) -> f64 {
    assert!(coarse.is_finite() && fine.is_finite() && coarse > fine && fine > 0.0);
    (coarse / fine).log2()
}

pub fn p2_poisson_manufactured_solution_converges_in_3d_tetrahedra_and_hexahedra() {
    for family in [ScalarFamily::Tetrahedron3d, ScalarFamily::Hexahedron3d] {
        let coarse = solve_and_measure(family, 1);
        let medium = solve_and_measure(family, 2);
        let fine = solve_and_measure(family, 4);
        let l2_rate_1 = refinement_rate(coarse.l2, medium.l2);
        let l2_rate_2 = refinement_rate(medium.l2, fine.l2);
        let h1_rate_1 = refinement_rate(coarse.h1_seminorm, medium.h1_seminorm);
        let h1_rate_2 = refinement_rate(medium.h1_seminorm, fine.h1_seminorm);

        // The coarsest Kuhn tetrahedral grid is mildly pre-asymptotic; the
        // second refinement must recover the expected P2 L2 rate, while both
        // levels must show strict convergence with the expected H1 trend.
        assert!(
            l2_rate_1 > 1.7 && l2_rate_2 > 2.5,
            "{} L2 rates were {l2_rate_1:.3}, {l2_rate_2:.3}; errors {coarse:?}, {medium:?}, {fine:?}",
            family.name()
        );
        assert!(
            h1_rate_1 > 0.8 && h1_rate_2 > 1.4,
            "{} H1 rates were {h1_rate_1:.3}, {h1_rate_2:.3}; errors {coarse:?}, {medium:?}, {fine:?}",
            family.name()
        );
        assert!(
            coarse.residual < 1.0e-10 && medium.residual < 1.0e-10 && fine.residual < 1.0e-10,
            "{} residuals were {:.3e}, {:.3e}, {:.3e}",
            family.name(),
            coarse.residual,
            medium.residual,
            fine.residual
        );
        assert!(
            coarse.constrained_dofs > 0
                && medium.constrained_dofs > coarse.constrained_dofs
                && fine.constrained_dofs > medium.constrained_dofs,
            "{} boundary DOF counts did not grow: {}, {}, {}",
            family.name(),
            coarse.constrained_dofs,
            medium.constrained_dofs,
            fine.constrained_dofs
        );
    }
}

#[test]
fn p2_poisson_manufactured_solution_converges_in_3d_tetrahedra_and_hexahedra_test() {
    p2_poisson_manufactured_solution_converges_in_3d_tetrahedra_and_hexahedra();
}

#[test]
fn p3_poisson_manufactured_solution_converges_in_3d_tetrahedra() {
    let family = ScalarFamily::P3Tetrahedron3d;
    let coarse = solve_and_measure(family, 1);
    let medium = solve_and_measure(family, 2);
    let fine = solve_and_measure(family, 4);

    let l2_rate_1 = refinement_rate(coarse.l2, medium.l2);
    let l2_rate_2 = refinement_rate(medium.l2, fine.l2);
    let h1_rate_1 = refinement_rate(coarse.h1_seminorm, medium.h1_seminorm);
    let h1_rate_2 = refinement_rate(medium.h1_seminorm, fine.h1_seminorm);

    // The first level may be mildly pre-asymptotic, but P3 tetrahedra should
    // recover cubic approximation rates on the second refinement.
    assert!(
        l2_rate_1 > 2.3 && l2_rate_2 > 3.0,
        "{} L2 rates {l2_rate_1:.3}, {l2_rate_2:.3}; errors {coarse:?}, {medium:?}, {fine:?}",
        family.name()
    );
    assert!(
        h1_rate_1 > 1.7 && h1_rate_2 > 2.3,
        "{} H1 rates {h1_rate_1:.3}, {h1_rate_2:.3}; errors {coarse:?}, {medium:?}, {fine:?}",
        family.name()
    );
    assert!(
        coarse.residual < 1.0e-10 && medium.residual < 1.0e-10 && fine.residual < 1.0e-10,
        "{} residuals {:.3e}, {:.3e}, {:.3e}",
        family.name(),
        coarse.residual,
        medium.residual,
        fine.residual
    );
}
