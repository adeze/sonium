//! Production residual/Jacobian contracts for the nonlinear consumers.
//!
//! The element-level hyperelastic contracts (residual-at-identity,
//! tangent symmetry, Taylor remainder) live next to the form; the uniform-
//! flow DG residual contracts live next to the operator. What was missing
//! is the production level: the assembled global residual/Jacobian pair
//! that `fem_newton_solve` actually iterates on, and the differentiability
//! of the flow residual that matrix-free Newton-Krylov relies on.
//!
//! - `hyperelastic_tension_newton_converges_with_sane_work_counters`
//!   drives the production `HyperelasticForm` + `fem_newton_solve_with_bcs`
//!   path on a uniaxial-tension problem and pins the Newton work counters.
//! - `compressible_dg_spatial_operator_is_frechet_differentiable` checks
//!   first-order remainder decay of the production DG spatial operator, the
//!   property a finite-difference/Jacobian-free Newton method assumes.
//! - `rusanov_flux_is_consistent` pins the numerical-flux consistency
//!   identity the DG residual is built on.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{Mesh, unit_square_triangles};
use sonium_fem::physics::fluid::compressible_flow::{
    CompressibleBoundaryCondition, CompressibleDgConfig, CompressibleDgOperator,
    CompressibleDgState, ConservativeState, IdealGas, ViscosityModel, rusanov_flux,
};
use sonium_fem::physics::solid::nonlinear_solid::{HyperelasticForm, NeoHookean};
use sonium_fem::solver::{FemNewtonConfig, fem_newton_solve_with_bcs};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::traits::SolverResultContract;

fn unit_square_space() -> VectorH1Space {
    VectorH1Space::new(PolynomialDegree::P1, 2)
}

/// Global DOFs (interleaved `[ux, uy]` per scalar node) on the left/right
/// faces of the unit square, with prescribed values for a tension test that
/// clamps `x = 0` and stretches `x = 1` by `stretch` in `x`.
fn tension_dirichlet(mesh: &Mesh, stretch: f64) -> Vec<(usize, f64)> {
    let mut dofs = Vec::new();
    for (node, point) in mesh.nodes.iter().enumerate() {
        if point.x.abs() < 1e-12 {
            dofs.push((2 * node, 0.0));
            dofs.push((2 * node + 1, 0.0));
        } else if (point.x - 1.0).abs() < 1e-12 {
            dofs.push((2 * node, stretch));
        }
    }
    dofs
}

#[test]
fn hyperelastic_tension_newton_converges_with_sane_work_counters() {
    let mesh = unit_square_triangles(4);
    let space = unit_square_space();
    let material = NeoHookean::from_young(1000.0, 0.3, 2);
    let form = HyperelasticForm::new(material);
    let stretch = 0.1;

    let dirichlet = tension_dirichlet(&mesh, stretch);
    assert!(
        dirichlet.len() >= 2 * 5 + 5,
        "both faces must contribute constrained DOFs"
    );
    let n_dofs = space.num_global_dofs(&mesh);
    let mut initial = vec![0.0; n_dofs];
    for &(dof, value) in &dirichlet {
        initial[dof] = value;
    }

    let config = FemNewtonConfig::default();
    let sol = fem_newton_solve_with_bcs(&form, &space, &mesh, &dirichlet, &initial, &config)
        .expect("tension problem must converge");
    assert!(sol.converged, "status = {:?}", sol.status);
    assert!(sol.residual < 1e-8, "residual = {:e}", sol.residual);
    sol.validate_contract()
        .expect("production Newton result must satisfy the result contract");

    // The production path delegates to `newton_raphson` (one black-box
    // Jacobian solve per Newton iteration, no inner Krylov iterations).
    assert_eq!(
        sol.jacobian_evaluations, sol.iterations,
        "iterations = {}",
        sol.iterations
    );
    assert!(
        sol.residual_evaluations >= sol.iterations + 1,
        "residual_evaluations = {}, iterations = {}",
        sol.residual_evaluations,
        sol.iterations
    );
    assert_eq!(sol.inner_iterations, 0);

    // Prescribed stretch attained, clamped face unmoved.
    for (node, point) in mesh.nodes.iter().enumerate() {
        if (point.x - 1.0).abs() < 1e-12 {
            assert!(
                (sol.x[2 * node] - stretch).abs() < 1e-9,
                "right-face ux = {}",
                sol.x[2 * node]
            );
        }
        if point.x.abs() < 1e-12 {
            assert!(sol.x[2 * node].abs() < 1e-12);
            assert!(sol.x[2 * node + 1].abs() < 1e-12);
        }
    }
}

fn uniform_flow() -> ConservativeState {
    ConservativeState::from_primitive(1.2, [30.0, 5.0, 0.0], 101_325.0, IdealGas::AIR)
}

fn far_field_operator(mesh: &Mesh, reference: ConservativeState) -> CompressibleDgOperator<'_> {
    let mut config = CompressibleDgConfig::default();
    config.viscosity = ViscosityModel::Constant(1.8e-5);
    for face in &mesh.boundaries {
        config.boundaries.insert(
            face.marker,
            CompressibleBoundaryCondition::FarField(reference),
        );
    }
    CompressibleDgOperator::new(mesh, config).expect("operator construction must succeed")
}

fn residual_norm(operator: &CompressibleDgOperator<'_>, state: &CompressibleDgState) -> f64 {
    let residual = operator
        .spatial_operator(state)
        .expect("residual evaluation must succeed");
    let sum: f64 = residual
        .iter()
        .flat_map(|value| value.as_array())
        .map(|v| v * v)
        .sum();
    sum.sqrt()
}

#[test]
fn compressible_dg_spatial_operator_is_frechet_differentiable() {
    let mesh = unit_square_triangles(2);
    let reference = uniform_flow();
    let operator = far_field_operator(&mesh, reference);
    let base = CompressibleDgState::from_function(&mesh, |_| reference)
        .expect("uniform state construction must succeed");

    // Smooth direction field, one 5-vector per DG entry, ordered exactly
    // like `CompressibleDgState::from_function` lays `values` out.
    let mut direction = Vec::with_capacity(base.values.len());
    for element in &mesh.elements {
        for &node in element.vertices() {
            let point = &mesh.nodes[node];
            // Phase-shifted so the wave does not vanish on the coarse
            // mesh nodes (an unshifted sin(2π(x+y)) is ~0 at every node
            // of unit_square_triangles(2), i.e. a zero direction).
            let wave = (2.0 * std::f64::consts::PI * point.x + 0.7).sin()
                * (2.0 * std::f64::consts::PI * point.y + 1.1).sin();
            direction.push([
                0.02 * reference.density * wave,
                0.02 * reference.momentum[0] * wave,
                0.02 * reference.momentum[1] * wave,
                0.0,
                0.02 * reference.total_energy * wave,
            ]);
        }
    }
    assert_eq!(direction.len(), base.values.len());

    // Remainder norms at amplitude 1, 1/2, 1/4: first-order decay means
    // each halving cuts the norm by ~2x (observed order ~1).
    let mut norms = Vec::new();
    for &amplitude in &[1.0, 0.5, 0.25] {
        let mut perturbed = base.clone();
        for (entry, step) in perturbed.values.iter_mut().zip(&direction) {
            entry.density += amplitude * step[0];
            entry.momentum[0] += amplitude * step[1];
            entry.momentum[1] += amplitude * step[2];
            entry.total_energy += amplitude * step[4];
        }
        norms.push(residual_norm(&operator, &perturbed));
    }
    assert!(
        norms[0] > 0.0,
        "the direction must produce a nonzero residual response"
    );
    for (k, window) in norms.windows(2).enumerate() {
        let order = (window[0] / window[1]).log2();
        assert!(
            (0.7..=1.3).contains(&order),
            "halving {k}: norms = {norms:?}, observed order = {order}"
        );
    }
}

#[test]
fn rusanov_flux_is_consistent() {
    let gas = IdealGas::AIR;
    let state = uniform_flow();
    for normal in [[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]] {
        let unit = {
            let n: f64 = (normal[0] * normal[0] + normal[1] * normal[1]).sqrt();
            [normal[0] / n, normal[1] / n, 0.0]
        };
        let numerical = rusanov_flux(state, state, gas, unit);
        let exact = state.euler_flux(gas, unit);
        for (i, (&num, &ex)) in numerical.iter().zip(exact.iter()).enumerate() {
            assert!(
                (num - ex).abs() <= 1e-9 * ex.abs().max(1.0),
                "component {i}: rusanov = {num}, euler = {ex}"
            );
        }
    }
}
