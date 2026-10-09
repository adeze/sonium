//! W1-07 production-path scalar P2 Poisson convergence evidence.
//!
//! This is deliberately independent of the vector-Laplacian gate: it uses the
//! scalar H1 space, the generic scalar forms and checked assembly, the generic
//! H1 boundary-DOF path, and the FEM CSR direct solver.  Both simplex and
//! tensor-product cells are covered on the same manufactured problem.

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
    Triangle2d,
    Quadrilateral2d,
}

impl ScalarFamily {
    fn mesh(self, refinement: usize) -> Mesh {
        match self {
            Self::Triangle2d => sonium_fem::mesh::unit_square_triangles(refinement),
            Self::Quadrilateral2d => sonium_fem::mesh::unit_square_quads(refinement),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Triangle2d => "P2-triangle-2d",
            Self::Quadrilateral2d => "Q2-quadrilateral-2d",
        }
    }
}

/// Smooth homogeneous-Dirichlet manufactured field and its Poisson source.
///
/// The extra linear factor makes the field outside the Q2 tensor-product
/// space, so both element families have a nontrivial refinement error.
fn manufactured([x, y, _]: [f64; 3]) -> (f64, [f64; 2], f64) {
    let a = x * (1.0 - x);
    let b = y * (1.0 - y);
    let ax = 1.0 - 2.0 * x;
    let by = 1.0 - 2.0 * y;
    let c = 1.0 + 0.2 * x + 0.15 * y;

    let value = a * b * c;
    let gradient = [ax * b * c + 0.2 * a * b, a * by * c + 0.15 * a * b];
    let laplacian = -2.0 * b * c + 0.4 * ax * b - 2.0 * a * c + 0.3 * a * by;
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
    let space = H1Space::new(PolynomialDegree::P2);
    let stiffness = assemble_bilinear_checked(&LaplacianForm, &space, &space, &mesh, 7)
        .unwrap_or_else(|error| panic!("{} stiffness assembly failed: {error}", family.name()));
    let source = SourceForm::new(|x, y, z| manufactured([x, y, z]).2);
    let rhs = assemble_linear_checked(&source, &space, &mesh, 7)
        .unwrap_or_else(|error| panic!("{} source assembly failed: {error}", family.name()));

    let mut system = FemSystem::new(stiffness, rhs);
    // The triangle generator labels its four sides 1..=4 (two Dirichlet and
    // two Neumann by default), while the quad generator leaves all detected
    // faces at marker 0.  The manufactured problem prescribes zero on every
    // side, so admit the complete marker set explicitly for this gate.
    for marker in 0..=4 {
        try_apply_homogeneous_dirichlet(&mut system, &mesh, &space, marker).unwrap_or_else(
            |error| panic!("{} Dirichlet elimination failed: {error}", family.name()),
        );
    }
    let constrained_dofs = system.dirichlet_dofs.len();

    // The generic forms are real-valued, while the FEM solver surface uses
    // Complex64 for its shared CSR backend.  This is only an adapter: the
    // matrix and RHS still come from the production scalar FEM path.
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
    for &dof in &system.dirichlet_dofs {
        assert!(
            solution.values[dof].norm() < 1.0e-12,
            "{} constrained DOF {dof} was {:.3e}",
            family.name(),
            solution.values[dof].norm()
        );
    }

    let space_solution = solution.values;
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
            let mut gradient = [0.0; 2];
            for (local, &global) in local_dofs.iter().enumerate() {
                let coefficient = space_solution[global].re;
                value += coefficient * tabulation.value(local);
                for (direction, component) in gradient.iter_mut().enumerate() {
                    *component += coefficient * tabulation.grad(local, direction);
                }
            }
            let (exact_value, exact_gradient, _) = manufactured(tabulation.physical_coords);
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

#[test]
fn p2_poisson_manufactured_solution_converges_on_triangles_and_quadrilaterals() {
    for family in [ScalarFamily::Triangle2d, ScalarFamily::Quadrilateral2d] {
        let coarse = solve_and_measure(family, 1);
        let medium = solve_and_measure(family, 2);
        let fine = solve_and_measure(family, 4);
        let l2_rate_1 = refinement_rate(coarse.l2, medium.l2);
        let l2_rate_2 = refinement_rate(medium.l2, fine.l2);
        let h1_rate_1 = refinement_rate(coarse.h1_seminorm, medium.h1_seminorm);
        let h1_rate_2 = refinement_rate(medium.h1_seminorm, fine.h1_seminorm);

        assert!(
            coarse.constrained_dofs > 0,
            "{} has no boundary DOFs",
            family.name()
        );
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
        assert!(
            coarse.residual < 1.0e-10 && medium.residual < 1.0e-10 && fine.residual < 1.0e-10,
            "{} solve residuals were {:.3e}, {:.3e}, {:.3e}",
            family.name(),
            coarse.residual,
            medium.residual,
            fine.residual
        );
    }
}
