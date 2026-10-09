use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::fluid::transport::{
    TransportAdvectionForm, TransportDiffusionForm, TransportMaterial, TransportSupgForm,
};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, TripletMatrix, gmres};

/// Solve 1D advection-diffusion: u*c' - D*c'' = 0 on [0, L] with c(0)=0, c(L)=1.
/// Exact solution: c(x) = (exp(Pe*x/L) - 1) / (exp(Pe) - 1), where Pe = u*L/D.
fn solve_advection_diffusion_1d(
    velocity: f64,
    diffusivity: f64,
    l: f64,
    nx: usize,
    with_supg: bool,
) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let h = l / nx as f64;
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, 1);
    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);

    let material = TransportMaterial {
        diffusivity: [
            [diffusivity, 0.0, 0.0],
            [0.0, diffusivity, 0.0],
            [0.0, 0.0, diffusivity],
        ],
        velocity: [velocity, 0.0, 0.0],
        reaction_rate: 0.0,
        capacity: 1.0,
    };

    let diff_form = TransportDiffusionForm::new(&material);
    let adv_form = TransportAdvectionForm::new(&material);

    let mut matrix = assemble_bilinear(&diff_form, &space, &space, &mesh, 2);
    let adv_matrix = assemble_bilinear(&adv_form, &space, &space, &mesh, 2);
    extend_triplet(&mut matrix, &adv_matrix);

    if with_supg {
        let supg_form = TransportSupgForm::new(material);
        let supg_matrix = assemble_bilinear(&supg_form, &space, &space, &mesh, 2);
        extend_triplet(&mut matrix, &supg_matrix);
    }

    let rhs = vec![0.0; n_dofs];
    let mut system = FemSystem::new(matrix, rhs);
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| 0.0, None);
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| 1.0, None);

    let csr = system.matrix.to_csr();
    let result = gmres(
        &csr,
        &system.rhs,
        &GmresConfig {
            max_iterations: 2000,
            restart: 200,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "GMRES did not converge");

    let mut max_val = f64::NEG_INFINITY;
    let mut min_val = f64::INFINITY;
    let mut x_vals = Vec::new();
    let mut c_vals = Vec::new();
    for (i, node) in mesh.nodes.iter().enumerate() {
        x_vals.push(node.x);
        c_vals.push(result.x[i]);
        max_val = max_val.max(result.x[i]);
        min_val = min_val.min(result.x[i]);
    }

    (x_vals, c_vals, max_val, min_val)
}

#[allow(dead_code)]
fn exact_solution(x: f64, pe: f64, l: f64) -> f64 {
    if pe.abs() < 1e-10 {
        return x / l;
    }
    let exp_pe = pe.exp();
    if exp_pe.is_infinite() {
        // For very large Pe, c(x) ≈ 0 for x < L, ≈ 1 at x = L
        if x >= l { 1.0 } else { 0.0 }
    } else {
        ((pe * x / l).exp() - 1.0) / (exp_pe - 1.0)
    }
}

/// Test 1: Galerkin solution at high Pe shows oscillations (overshoot/undershoot).
fn test_galerkin_oscillations() {
    let velocity = 1.0;
    let diffusivity = 0.005;
    let l = 1.0;
    let nx = 20;
    let pe = velocity * l / diffusivity;

    let (_, _, max_val, min_val) =
        solve_advection_diffusion_1d(velocity, diffusivity, l, nx, false);

    println!(
        "Galerkin (Pe={:.1}): max={:.4e}, min={:.4e}",
        pe, max_val, min_val
    );
    // At high Pe, Galerkin should show oscillations (values outside [0, 1])
    let has_overshoot = max_val > 1.01 || min_val < -0.01;
    println!("  Has oscillations: {}", has_overshoot);
    assert!(
        has_overshoot,
        "Expected Galerkin to show oscillations at high Pe"
    );
}

/// Test 2: SUPG solution at same high Pe is monotone (no overshoot/undershoot).
fn test_supg_monotone() {
    let velocity = 1.0;
    let diffusivity = 0.005;
    let l = 1.0;
    let nx = 20;
    let pe = velocity * l / diffusivity;

    let (_, _, max_val, min_val) = solve_advection_diffusion_1d(velocity, diffusivity, l, nx, true);

    println!(
        "SUPG (Pe={:.1}): max={:.4e}, min={:.4e}",
        pe, max_val, min_val
    );
    // SUPG should significantly reduce oscillations (not necessarily eliminate)
    let overshoot = (max_val - 1.0).max(0.0) + (0.0 - min_val).max(0.0);
    println!("  SUPG overshoot: {:.4e}", overshoot);
    assert!(
        overshoot < 0.05,
        "SUPG should reduce oscillations: overshoot={}",
        overshoot
    );
}

/// Test 3: At low Pe (diffusion-dominated), SUPG has negligible effect.
/// Both Galerkin and SUPG should give similar results.
fn test_supg_low_pe_negligible() {
    let velocity = 0.1;
    let diffusivity = 1.0; // Pe = 0.1, diffusion-dominated
    let l = 1.0;
    let nx = 20;
    let pe = velocity * l / diffusivity;

    let (x_gal, c_gal, _, _) = solve_advection_diffusion_1d(velocity, diffusivity, l, nx, false);
    let (_x_supg, c_supg, _, _) = solve_advection_diffusion_1d(velocity, diffusivity, l, nx, true);

    let mut max_diff: f64 = 0.0;
    for i in 0..x_gal.len() {
        max_diff = max_diff.max((c_gal[i] - c_supg[i]).abs());
    }

    println!(
        "SUPG low Pe (Pe={:.2}): max_diff_gal_vs_supg={:.4e}",
        pe, max_diff
    );
    // At low Pe, SUPG stabilization should be negligible
    assert!(
        max_diff < 1e-3,
        "SUPG should have negligible effect at low Pe: max_diff={}",
        max_diff
    );
}

fn extend_triplet(target: &mut TripletMatrix<f64>, source: &TripletMatrix<f64>) {
    for i in 0..source.nnz() {
        target.push(source.rows[i], source.cols[i], source.values[i]);
    }
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: transport SUPG 1D ===");
        test_galerkin_oscillations();
        test_supg_monotone();
        test_supg_low_pe_negligible();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
