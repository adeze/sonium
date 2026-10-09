//! Minimum-drag shape optimisation in 2D Stokes flow (Pironneau benchmark)
//!
//! Finds the body shape that minimises drag in Stokes flow inside a cylinder.
//! Pironneau (1973) proved the optimal shape is an ogival (rugby-ball) body
//! with pointed cusps and aspect ratio ~3:1 for given area.
//!
//! Setup:
//! ```text
//!        ___________
//!      /             \
//!     /    ~~>        \
//!    |   ~~> (•) ~~>   |   (•) = obstacle, ~~> = Stokes flow
//!     \    ~~>        /
//!      \_____________/
//! ```
//!
//! Usage:
//!   cargo run -p sonium-fem --features cli --example stokes_shape_opt
//!   cargo run -p sonium-fem --features cli --example stokes_shape_opt -- --help
//!   cargo run -p sonium-fem --features cli --example stokes_shape_opt -- --n-modes 6

use clap::Parser;
use std::f64::consts::PI;

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::{Mesh, annular_mesh_triangles};
use sonium_fem::physics::fluid::stokes::navier_stokes::{
    NavierStokesConfig, NavierStokesResult, compute_drag_lift, compute_vorticity_2d, solve_picard,
};
use sonium_fem::spaces::{H1Space, VectorH1Space};

use sonium_solvers::{
    AugmentedLagrangianConfig, Bound, Constraint, ConstraintKind, InnerSolver,
    augmented_lagrangian, dense::DenseMatrix,
};

// ── CLI ─────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "stokes-shape-opt")]
#[command(about = "Minimum-drag body shape in 2D Stokes flow (Pironneau benchmark)")]
struct Cli {
    /// Initial obstacle radius (m)
    #[arg(long, default_value_t = 1.0)]
    r_inner: f64,
    /// Outer cylinder radius (m) — far-field boundary
    #[arg(long, default_value_t = 10.0)]
    r_outer: f64,
    /// Radial mesh divisions
    #[arg(long, default_value_t = 20)]
    n_radial: usize,
    /// Angular mesh divisions
    #[arg(long, default_value_t = 64)]
    n_angular: usize,
    /// Dynamic viscosity (Pa s)
    #[arg(long, default_value_t = 1.0)]
    mu: f64,
    /// Free-stream velocity (m/s)
    #[arg(long, default_value_t = 1.0)]
    u_inf: f64,
    /// Number of Fourier modes (even cosines: cos(0), cos(2t), cos(4t), ...)
    #[arg(long, default_value_t = 4)]
    n_modes: usize,
    /// FD step size for gradient
    #[arg(long, default_value_t = 1e-4)]
    fd_eps: f64,
    /// Output directory
    #[arg(long, default_value = "viz_output")]
    output_dir: String,
}

// ── Shape parametrisation ───────────────────────────────────────────────

/// Radial distance r(θ) = a₀ + a₁ cos(2θ) + a₂ cos(4θ) + ...
/// Even cosines → fore-aft and top-bottom symmetry.
fn shape_radius(theta: f64, params: &[f64]) -> f64 {
    let mut r = params[0];
    for (k, &ak) in params.iter().enumerate().skip(1) {
        r += ak * (2.0 * k as f64 * theta).cos();
    }
    r
}

/// Cross-sectional area from Fourier coefficients (analytical).
/// A = π a₀² + (π/2) Σ_{k≥1} aₖ²
fn shape_area(params: &[f64]) -> f64 {
    let mut a = PI * params[0] * params[0];
    for &ak in &params[1..] {
        a += 0.5 * PI * ak * ak;
    }
    a
}

/// ∂A/∂aₖ: 2π a₀ for k=0, π aₖ for k≥1.
fn shape_area_gradient(params: &[f64]) -> Vec<f64> {
    let mut g = vec![2.0 * PI * params[0]];
    for &ak in &params[1..] {
        g.push(PI * ak);
    }
    g
}

/// Check that r(θ) > 0 everywhere (no self-intersection).
fn shape_is_valid(params: &[f64]) -> bool {
    let n_check = 256;
    (0..n_check).all(|i| {
        let theta = 2.0 * PI * i as f64 / n_check as f64;
        shape_radius(theta, params) > 0.0
    })
}

// ── Mesh deformation ────────────────────────────────────────────────────

/// Store the reference (undeformed) node coordinates and the structured-grid indices.
struct RefMesh {
    mesh: Mesh,
    /// (r_ref, theta) for each node — the reference polar coordinates.
    polar: Vec<(f64, f64)>,
    r_inner: f64,
    r_outer: f64,
    n_radial: usize,
    n_angular: usize,
}

impl RefMesh {
    fn new(r_inner: f64, r_outer: f64, n_radial: usize, n_angular: usize) -> Self {
        let mesh = annular_mesh_triangles(0.0, 0.0, r_inner, r_outer, n_radial, n_angular);
        let dr = (r_outer - r_inner) / n_radial as f64;
        let mut polar = Vec::with_capacity(mesh.num_nodes());
        for r in 0..=n_radial {
            let r_ref = r_inner + r as f64 * dr;
            for a in 0..n_angular {
                let theta = 2.0 * PI * a as f64 / n_angular as f64;
                polar.push((r_ref, theta));
            }
        }
        Self {
            mesh,
            polar,
            r_inner,
            r_outer,
            n_radial,
            n_angular,
        }
    }

    /// Create a deformed mesh for the given shape parameters.
    fn deform(&self, params: &[f64]) -> Mesh {
        let mut mesh = self.mesh.clone();
        for (idx, &(r_ref, theta)) in self.polar.iter().enumerate() {
            let t = (r_ref - self.r_inner) / (self.r_outer - self.r_inner);
            let r_shape = shape_radius(theta, params);
            let r_new = (1.0 - t) * r_shape + t * self.r_outer;
            mesh.nodes[idx].x = r_new * theta.cos();
            mesh.nodes[idx].y = r_new * theta.sin();
        }
        mesh
    }
}

// ── Stokes solve ────────────────────────────────────────────────────────

fn stokes_solve(
    mesh: &Mesh,
    mu: f64,
    u_inf: f64,
    r_outer: f64,
    n_angular: usize,
    n_radial: usize,
) -> NavierStokesResult {
    let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);

    // Inner boundary (no-slip): layer 0 = nodes [0, n_angular)
    // Outer boundary (far-field): last layer = nodes [n_radial*n_angular, (n_radial+1)*n_angular)
    let outer_offset = n_radial * n_angular;
    let mut bc_dofs: Vec<usize> = (0..n_angular)
        .chain(outer_offset..outer_offset + n_angular)
        .flat_map(|node| [node * 2, node * 2 + 1])
        .collect();
    bc_dofs.sort();
    bc_dofs.dedup();

    let r_thresh = r_outer * 0.9;
    let bc_fn = move |x: f64, y: f64, _z: f64| -> Vec<f64> {
        let r = (x * x + y * y).sqrt();
        if r > r_thresh {
            vec![u_inf, 0.0]
        } else {
            vec![0.0, 0.0]
        }
    };

    // rho=1 keeps PSPG stabilisation healthy for P1-P1.
    // 1 Picard iteration with zero initial velocity gives pure Stokes
    // (the convection term ρ(u·∇)u vanishes when u_prev = 0).
    let config = NavierStokesConfig {
        mu,
        rho: 1.0,
        max_picard_iterations: 1,
        picard_tolerance: 1e-10,
        ..Default::default()
    };

    solve_picard(mesh, &v_space, &p_space, &config, &bc_fn, &bc_dofs, |_| {
        vec![0.0, 0.0]
    })
}

fn compute_drag(mesh: &Mesh, result: &NavierStokesResult, mu: f64) -> f64 {
    let (drag, _lift) = compute_drag_lift(mesh, &result.velocity, &result.pressure, mu, 1);
    drag
}

// ── Objective and gradient ──────────────────────────────────────────────

fn evaluate_drag(params: &[f64], ref_mesh: &RefMesh, mu: f64, u_inf: f64) -> f64 {
    if !shape_is_valid(params) {
        return INVALID_PENALTY; // penalty for invalid shape
    }
    let mesh = ref_mesh.deform(params);
    let result = stokes_solve(
        &mesh,
        mu,
        u_inf,
        ref_mesh.r_outer,
        ref_mesh.n_angular,
        ref_mesh.n_radial,
    );
    compute_drag(&mesh, &result, mu)
}

const INVALID_PENALTY: f64 = 1e10;

fn fd_gradient(params: &[f64], eps: f64, ref_mesh: &RefMesh, mu: f64, u_inf: f64) -> Vec<f64> {
    let n = params.len();
    let f0 = evaluate_drag(params, ref_mesh, mu, u_inf);
    let mut grad = vec![0.0; n];
    for i in 0..n {
        let mut pp = params.to_vec();
        let mut pm = params.to_vec();
        pp[i] += eps;
        pm[i] -= eps;
        let fp = evaluate_drag(&pp, ref_mesh, mu, u_inf);
        let fm = evaluate_drag(&pm, ref_mesh, mu, u_inf);
        // Fall back to one-sided difference if a perturbation hits the invalid-shape penalty
        grad[i] = if fp >= INVALID_PENALTY && fm >= INVALID_PENALTY {
            0.0
        } else if fp >= INVALID_PENALTY {
            (f0 - fm) / eps
        } else if fm >= INVALID_PENALTY {
            (fp - f0) / eps
        } else {
            (fp - fm) / (2.0 * eps)
        };
    }
    grad
}

// ── VTU output ──────────────────────────────────────────────────────────

fn write_vtu(mesh: &Mesh, result: &NavierStokesResult, path: &str) {
    let n = mesh.num_nodes();
    let ux: Vec<f64> = (0..n).map(|i| result.velocity[2 * i]).collect();
    let uy: Vec<f64> = (0..n).map(|i| result.velocity[2 * i + 1]).collect();
    let uz = vec![0.0; n];
    let speed: Vec<f64> = (0..n)
        .map(|i| (ux[i] * ux[i] + uy[i] * uy[i]).sqrt())
        .collect();
    let vorticity = compute_vorticity_2d(mesh, &result.velocity);

    let mut writer = VtuWriter::new(mesh);
    writer.add_point_vector("velocity", &ux, &uy, &uz);
    writer.add_point_scalar("pressure", &result.pressure);
    writer.add_point_scalar("speed", &speed);
    writer.add_cell_scalar("vorticity", &vorticity);
    writer.write(path).expect("failed to write VTU");
}

fn write_pyvista_script(path: &str, output_dir: &str) {
    let script = format!(
        r#""""
Visualise Stokes shape optimisation: initial circle vs Pironneau-optimal body.

Requires: pip install pyvista
"""
import pyvista as pv

pv.set_plot_theme("document")

initial = pv.read("{output_dir}/stokes_initial.vtu")
optimal = pv.read("{output_dir}/stokes_optimal.vtu")

pl = pv.Plotter(shape=(2, 2))

pl.subplot(0, 0)
pl.add_text("Initial (circle) - speed", font_size=10)
pl.add_mesh(initial, scalars="speed", cmap="coolwarm", show_edges=False)

pl.subplot(0, 1)
pl.add_text("Optimal (Pironneau) - speed", font_size=10)
pl.add_mesh(optimal, scalars="speed", cmap="coolwarm", show_edges=False)

pl.subplot(1, 0)
pl.add_text("Initial - pressure", font_size=10)
pl.add_mesh(initial, scalars="pressure", cmap="RdBu", show_edges=False)

pl.subplot(1, 1)
pl.add_text("Optimal - pressure", font_size=10)
pl.add_mesh(optimal, scalars="pressure", cmap="RdBu", show_edges=False)

pl.link_views()
pl.show()
"#
    );
    std::fs::write(path, script).expect("failed to write pyvista script");
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();

    println!("=== Stokes Shape Optimisation (Pironneau Benchmark) ===");
    println!();
    println!(
        "Domain:     annular, r_inner = {:.2}, r_outer = {:.2}",
        cli.r_inner, cli.r_outer
    );
    println!(
        "Mesh:       {} radial x {} angular",
        cli.n_radial, cli.n_angular
    );
    println!("Viscosity:  mu = {:.2e}", cli.mu);
    println!("Far-field:  U = {:.2} m/s", cli.u_inf);
    println!("Modes:      {} even Fourier cosines", cli.n_modes);
    println!();

    // Reference mesh (circular obstacle)
    let ref_mesh = RefMesh::new(cli.r_inner, cli.r_outer, cli.n_radial, cli.n_angular);
    let n_nodes = ref_mesh.mesh.num_nodes();
    let n_elems = ref_mesh.mesh.num_elements();
    println!("Mesh: {n_nodes} nodes, {n_elems} elements");

    // Initial params: circle
    let mut params0 = vec![0.0; cli.n_modes];
    params0[0] = cli.r_inner;
    let a_target = shape_area(&params0);
    println!(
        "Initial area: A = {a_target:.6} (= pi r^2 = {:.6})",
        PI * cli.r_inner * cli.r_inner
    );

    // Initial solve
    let drag0 = evaluate_drag(&params0, &ref_mesh, cli.mu, cli.u_inf);
    println!("Initial drag (circle): {drag0:.6}");

    // Gradient check
    let grad0 = fd_gradient(&params0, cli.fd_eps, &ref_mesh, cli.mu, cli.u_inf);
    println!(
        "Initial gradient: {:?}",
        grad0.iter().map(|g| format!("{g:.4e}")).collect::<Vec<_>>()
    );
    println!();

    // ── Optimise ────────────────────────────────────────────────────
    println!("--- Running optimisation ---");

    let r_inner = cli.r_inner;
    let n_modes = cli.n_modes;
    let mu = cli.mu;
    let u_inf = cli.u_inf;
    let fd_eps = cli.fd_eps;

    let objective = |x: &[f64]| -> f64 { evaluate_drag(x, &ref_mesh, mu, u_inf) };
    let gradient = |x: &[f64]| -> Vec<f64> { fd_gradient(x, fd_eps, &ref_mesh, mu, u_inf) };

    // Equality constraint: A(params) = A_target
    let area_constraint = Constraint {
        kind: ConstraintKind::Equality,
        dim: 1,
        fun: Box::new(move |x: &[f64]| vec![shape_area(x) - a_target]),
        jac: Some(Box::new(move |x: &[f64]| {
            let g = shape_area_gradient(x);
            DenseMatrix::from_row_major_vec(1, g.len(), g)
        })),
    };

    // Bounds: a₀ ∈ [0.1r, 3r], aₖ ∈ [-0.5r, 0.5r]
    let mut bounds = vec![Bound::both(0.1 * r_inner, 3.0 * r_inner)];
    for _ in 1..n_modes {
        bounds.push(Bound::both(-0.5 * r_inner, 0.5 * r_inner));
    }

    let config = AugmentedLagrangianConfig {
        max_outer_iterations: 20,
        max_inner_iterations: 30,
        grad_tolerance: 1e-5,
        constraint_tolerance: 1e-5,
        inner_solver: InnerSolver::Lbfgsb,
        bounds,
        ..Default::default()
    };

    let x0 = params0.clone();
    let result = augmented_lagrangian(&objective, &gradient, &x0, &[area_constraint], &config)
        .expect("optimisation failed");

    let opt_params: Vec<f64> = result.x.clone();

    // ── Results ─────────────────────────────────────────────────────
    println!();
    println!("=== Result ===");
    println!("Converged:  {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("Fun evals:  {}", result.fun_evals);
    println!();

    let opt_drag = result.fun;
    let opt_area = shape_area(&opt_params);
    println!("Optimal drag:   {opt_drag:.6}");
    println!("Optimal area:   {opt_area:.6}  (target: {a_target:.6})");
    println!("Drag reduction: {:.1}%", (1.0 - opt_drag / drag0) * 100.0);
    println!();

    // Shape dimensions
    let half_length = shape_radius(0.0, &opt_params); // along flow (θ=0)
    let half_width = shape_radius(PI / 2.0, &opt_params); // perpendicular (θ=π/2)
    let aspect = half_length / half_width;
    println!("Optimal shape:");
    println!(
        "  Fourier coeffs: {:?}",
        opt_params
            .iter()
            .map(|c| format!("{c:.4}"))
            .collect::<Vec<_>>()
    );
    println!("  half-length (flow dir)  = {half_length:.4}");
    println!("  half-width  (perp)      = {half_width:.4}");
    println!("  aspect ratio            = {aspect:.2}");
    println!();
    println!("Pironneau (1973) reference:");
    println!("  Optimal 2D Stokes body is ogival with pointed cusps");
    println!("  Expected: elongated along flow, aspect ratio > 2");

    // ── VTU output ──────────────────────────────────────────────────
    std::fs::create_dir_all(&cli.output_dir).expect("failed to create output dir");

    let path_init = format!("{}/stokes_initial.vtu", cli.output_dir);
    let path_opt = format!("{}/stokes_optimal.vtu", cli.output_dir);

    println!();
    print!("Writing {path_init} ...");
    let init_mesh = ref_mesh.deform(&params0);
    let init_result = stokes_solve(
        &init_mesh,
        cli.mu,
        cli.u_inf,
        ref_mesh.r_outer,
        ref_mesh.n_angular,
        ref_mesh.n_radial,
    );
    write_vtu(&init_mesh, &init_result, &path_init);
    println!(" done");

    print!("Writing {path_opt} ...");
    let opt_mesh = ref_mesh.deform(&opt_params);
    let opt_result = stokes_solve(
        &opt_mesh,
        cli.mu,
        cli.u_inf,
        ref_mesh.r_outer,
        ref_mesh.n_angular,
        ref_mesh.n_radial,
    );
    write_vtu(&opt_mesh, &opt_result, &path_opt);
    println!(" done");

    let script_path = format!("{}/plot_stokes.py", cli.output_dir);
    write_pyvista_script(&script_path, &cli.output_dir);
    println!("Written {script_path}");
    println!();
    println!("Visualise with:  python {script_path}");
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_ref_mesh() -> RefMesh {
        RefMesh::new(1.0, 8.0, 10, 32)
    }

    #[test]
    fn test_area_formula() {
        // Circle: A = π r²
        let params = vec![1.0, 0.0, 0.0];
        let a = shape_area(&params);
        assert!((a - PI).abs() < 1e-12, "circle area: {a} vs {PI}");

        // With perturbation: A = π a₀² + π/2 a₁²
        let params = vec![1.0, 0.5, 0.0];
        let a = shape_area(&params);
        let expected = PI + 0.5 * PI * 0.25;
        assert!((a - expected).abs() < 1e-12, "area: {a} vs {expected}");

        // Verify gradient
        let g = shape_area_gradient(&params);
        let eps = 1e-7;
        for i in 0..params.len() {
            let mut pp = params.clone();
            let mut pm = params.clone();
            pp[i] += eps;
            pm[i] -= eps;
            let fd = (shape_area(&pp) - shape_area(&pm)) / (2.0 * eps);
            let rel = (g[i] - fd).abs() / g[i].abs().max(1e-15);
            assert!(
                rel < 1e-5,
                "dA/da[{i}]: analytical={}, fd={fd}, rel={rel:.2e}",
                g[i]
            );
        }
    }

    #[test]
    fn test_fd_gradient_convergence() {
        let rm = small_ref_mesh();
        let params = vec![1.0, 0.0, 0.0]; // circle

        let g1 = fd_gradient(&params, 1e-3, &rm, 1.0, 1.0);
        let g2 = fd_gradient(&params, 5e-4, &rm, 1.0, 1.0);

        // Gradients at two step sizes should be close (FD converges)
        for i in 0..params.len() {
            if g1[i].abs() < 1e-10 && g2[i].abs() < 1e-10 {
                continue; // both near zero, skip
            }
            let rel = (g1[i] - g2[i]).abs() / g1[i].abs().max(1e-15);
            assert!(
                rel < 0.5,
                "gradient[{i}] not converging: eps=1e-3 -> {:.4e}, eps=5e-4 -> {:.4e}, rel={rel:.2e}",
                g1[i],
                g2[i]
            );
        }
    }

    #[test]
    fn test_optimisation_reduces_drag() {
        let rm = RefMesh::new(1.0, 8.0, 8, 24);
        let mu = 1.0;
        let u_inf = 1.0;

        let params0 = vec![1.0, 0.0, 0.0];
        let drag0 = evaluate_drag(&params0, &rm, mu, u_inf);
        let a_target = shape_area(&params0);

        // Run a few optimisation iterations
        let objective = |x: &[f64]| evaluate_drag(x, &rm, mu, u_inf);
        let gradient = |x: &[f64]| fd_gradient(x, 1e-3, &rm, mu, u_inf);

        let area_constraint = Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(move |x: &[f64]| vec![shape_area(x) - a_target]),
            jac: Some(Box::new(move |x: &[f64]| {
                let g = shape_area_gradient(x);
                DenseMatrix::from_row_major_vec(1, g.len(), g)
            })),
        };

        let bounds = vec![
            Bound::both(0.1, 3.0),
            Bound::both(-0.5, 0.5),
            Bound::both(-0.5, 0.5),
        ];

        let config = AugmentedLagrangianConfig {
            max_outer_iterations: 10,
            max_inner_iterations: 50,
            grad_tolerance: 1e-4,
            constraint_tolerance: 1e-4,
            inner_solver: InnerSolver::Lbfgsb,
            bounds,
            ..Default::default()
        };

        let x0 = params0;
        let result = augmented_lagrangian(&objective, &gradient, &x0, &[area_constraint], &config)
            .expect("optimisation failed");

        let opt_drag = result.fun;
        assert!(
            opt_drag < drag0,
            "optimised drag ({opt_drag:.6}) should be less than circle ({drag0:.6})"
        );

        let opt_area = shape_area(&result.x);
        assert!(
            (opt_area - a_target).abs() < 0.1 * a_target,
            "area constraint violated: {opt_area:.4} vs target {a_target:.4}"
        );
    }
}
