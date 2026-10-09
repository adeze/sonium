//! W1-13 production-path higher-order Helmholtz pollution evidence.
//!
//! This packet extends the P1 plane-wave reference with quadratic Lagrange
//! elements.  The exact oblique plane wave is used as an independent
//! reference for `-Δu - k²u = 0`, while the solve, boundary elimination,
//! quadrature, and field evaluation all go through the production FEM APIs.
//! The assertions deliberately stay bounded to affine P2 triangles: they
//! establish higher-order convergence and reduced same-mesh pollution without
//! claiming the full curved/open-boundary Helmholtz matrix.

use num_complex::Complex64;
use sonium_fem::assembly::HelmholtzProblem;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::{DirichletBC, apply_dirichlet};
use sonium_fem::mesh::{ElementType, Mesh, unit_square_triangles};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::solver::{Solution, SolverConfig, SolverType, solve};
use sonium_fem::spaces::{FunctionSpace, H1Space};

const WAVENUMBER: f64 = 8.0;
const PROPAGATION_ANGLE: f64 = 0.37;
const PHASE_PROBE: [f64; 2] = [0.37, 0.43];

fn plane_wave(x: f64, y: f64, z: f64) -> Complex64 {
    assert!(
        z.abs() < 1.0e-12,
        "the reference problem is two-dimensional"
    );
    let phase = WAVENUMBER * (PROPAGATION_ANGLE.cos() * x + PROPAGATION_ANGLE.sin() * y);
    Complex64::new(phase.cos(), phase.sin())
}

fn solve_plane_wave(mesh: &Mesh, degree: PolynomialDegree) -> Solution {
    let mut problem =
        HelmholtzProblem::assemble(mesh, degree, Complex64::new(WAVENUMBER, 0.0), |_, _, _| {
            Complex64::new(0.0, 0.0)
        });

    // rectangular_mesh_triangles labels the four boundary sides 1..=4.  A
    // plane wave satisfies the equation in the interior, so exact Dirichlet
    // data on all sides is the complete boundary-value problem.
    let boundary_conditions: Vec<DirichletBC> = (1..=4)
        .map(|tag| DirichletBC::new(tag, plane_wave))
        .collect();
    apply_dirichlet(&mut problem, mesh, &boundary_conditions);

    let config = SolverConfig {
        solver_type: SolverType::Direct,
        ..SolverConfig::default()
    };
    solve(&problem, &config).expect("plane-wave Helmholtz system should solve")
}

#[derive(Clone, Copy, Debug)]
struct PlaneWaveError {
    relative_l2: f64,
    probe_phase: f64,
    residual: f64,
}

fn measure_plane_wave_error(
    mesh: &Mesh,
    degree: PolynomialDegree,
    solution: &Solution,
) -> PlaneWaveError {
    let space = H1Space::new(degree);
    let quadrature = QuadratureRule::new(ElementType::Triangle, 10);
    let mut error_squared = 0.0;
    let mut reference_squared = 0.0;

    for element in 0..mesh.num_elements() {
        let local_dofs = space.local_to_global(mesh, element);
        for point in quadrature.iter() {
            let tabulation = space.tabulate(mesh, element, point.xi(), point.eta(), point.zeta());
            let numerical = local_dofs
                .iter()
                .enumerate()
                .map(|(local, &global)| solution.values[global] * tabulation.value(local))
                .sum::<Complex64>();
            let exact = plane_wave(
                tabulation.physical_coords[0],
                tabulation.physical_coords[1],
                tabulation.physical_coords[2],
            );
            let weight = point.weight * tabulation.det_j;
            error_squared += weight * (numerical - exact).norm_sqr();
            reference_squared += weight * exact.norm_sqr();
        }
    }

    let probe = mesh
        .nodes
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            let left_distance = (left.x - PHASE_PROBE[0]).hypot(left.y - PHASE_PROBE[1]);
            let right_distance = (right.x - PHASE_PROBE[0]).hypot(right.y - PHASE_PROBE[1]);
            left_distance.total_cmp(&right_distance)
        })
        .expect("unit-square mesh should have a phase-probe-near node");
    let exact_probe = plane_wave(probe.1.x, probe.1.y, probe.1.z);
    let numerical_probe = solution.values[probe.0];
    let phase_ratio = numerical_probe / exact_probe;

    PlaneWaveError {
        relative_l2: (error_squared / reference_squared).sqrt(),
        probe_phase: phase_ratio.arg().abs(),
        residual: solution.residual,
    }
}

fn solve_and_measure(subdivisions: usize, degree: PolynomialDegree) -> PlaneWaveError {
    let mesh = unit_square_triangles(subdivisions);
    let solution = solve_plane_wave(&mesh, degree);
    assert!(
        solution.converged,
        "direct plane-wave solve did not converge"
    );
    measure_plane_wave_error(&mesh, degree, &solution)
}

fn refinement_rate(coarse: f64, fine: f64) -> f64 {
    assert!(coarse.is_finite() && fine.is_finite());
    assert!(coarse > fine && fine > 0.0, "errors must strictly decrease");
    (coarse / fine).log2()
}

#[test]
fn p2_helmholtz_oblique_plane_wave_has_higher_order_convergence() {
    let coarse = solve_and_measure(4, PolynomialDegree::P2);
    let medium = solve_and_measure(8, PolynomialDegree::P2);
    let fine = solve_and_measure(16, PolynomialDegree::P2);

    let l2_rate_1 = refinement_rate(coarse.relative_l2, medium.relative_l2);
    let l2_rate_2 = refinement_rate(medium.relative_l2, fine.relative_l2);
    let phase_rate = refinement_rate(coarse.probe_phase, fine.probe_phase) / 2.0;
    assert!(
        l2_rate_1 > 2.0 && l2_rate_2 > 2.4,
        "P2 plane-wave L2 rates were {l2_rate_1:.3}, {l2_rate_2:.3}; errors {coarse:?}, {medium:?}, {fine:?}"
    );
    assert!(
        phase_rate > 1.8,
        "P2 probe phase error did not converge at a higher-order rate: {phase_rate:.3}; errors {coarse:?}, {medium:?}, {fine:?}"
    );
    assert!(
        fine.relative_l2 < 5.0e-3 && fine.probe_phase < 5.0e-3,
        "resolved P2 mesh still has excessive pollution: {fine:?}"
    );
    assert!(
        coarse.residual < 1.0e-10 && medium.residual < 1.0e-10 && fine.residual < 1.0e-10,
        "direct solve residuals were {:.3e}, {:.3e}, {:.3e}",
        coarse.residual,
        medium.residual,
        fine.residual
    );
}

#[test]
fn p2_reduces_same_mesh_helmholtz_pollution_against_p1() {
    let p1 = solve_and_measure(8, PolynomialDegree::P1);
    let p2 = solve_and_measure(8, PolynomialDegree::P2);

    assert!(
        p2.relative_l2 < 0.25 * p1.relative_l2,
        "P2 L2 pollution was not materially lower than P1: P1 {p1:?}, P2 {p2:?}"
    );
    assert!(
        p2.probe_phase < 0.25 * p1.probe_phase,
        "P2 phase pollution was not materially lower than P1: P1 {p1:?}, P2 {p2:?}"
    );
    assert!(
        p1.residual < 1.0e-10 && p2.residual < 1.0e-10,
        "same-mesh direct solve residuals were {:.3e} and {:.3e}",
        p1.residual,
        p2.residual
    );
}
