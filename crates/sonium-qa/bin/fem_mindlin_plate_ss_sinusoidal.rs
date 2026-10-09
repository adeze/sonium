//! QA: FEM Mindlin plate — simply supported square plate under sinusoidal load.
//!
//! The P1 triangular Mindlin element uses a DSG/MITC-style projected shear field
//! in `MindlinPlateForm::new`, avoiding the severe shear locking of the fully
//! compatible `theta - grad(w)` shear strain. This QA validates the stabilized
//! element against the Navier sinusoidal Mindlin plate solution and checks that
//! the stabilized form is less locked than the compatible form.

use sonium_fem::assembly::{assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_homogeneous_dirichlet_dofs};
use sonium_fem::mesh::rectangular_mesh_triangles;
use sonium_fem::physics::solid::mindlin_plate::{
    MindlinPlateForm, MindlinPlateLoadForm, MindlinPlateShearStabilization, MindlinPlateSpace,
};
use sonium_fem::spaces::FunctionSpace;
use sonium_solvers::{GmresConfig, gmres};

const Q0: f64 = 1e6;
const E_MOD: f64 = 1e6;
const NU: f64 = 0.3;
const KAPPA: f64 = 5.0 / 6.0;

fn exact_mindlin_center_deflection(a: f64, b: f64, h: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let lambda = (pi / a).powi(2) + (pi / b).powi(2);
    let d_bending = E_MOD * h.powi(3) / (12.0 * (1.0 - NU * NU));
    let g = E_MOD / (2.0 * (1.0 + NU));
    let kappa_gh = KAPPA * g * h;

    Q0 / (d_bending * lambda * lambda) + Q0 / (kappa_gh * lambda)
}

fn solve_plate(
    a: f64,
    b: f64,
    nx: usize,
    ny: usize,
    h: f64,
    stabilization: MindlinPlateShearStabilization,
) -> (f64, Vec<f64>, usize) {
    let q0 = Q0;

    let mesh = rectangular_mesh_triangles(0.0, a, 0.0, b, nx, ny);
    let space = MindlinPlateSpace::new(PolynomialDegree::P1);

    let form = MindlinPlateForm::with_shear_stabilization(E_MOD, NU, h, KAPPA, stabilization);
    let matrix = assemble_bilinear(&form, &space, &space, &mesh, 3);

    let load_form = MindlinPlateLoadForm {
        load_fn: |x: f64, y: f64| {
            q0 * (std::f64::consts::PI * x / a).sin() * (std::f64::consts::PI * y / b).sin()
        },
    };
    let rhs = assemble_linear(&load_form, &space, &mesh, 3);

    let mut system = FemSystem::new(matrix, rhs);

    // Hard simply supported: w=0 on all edges, θx=0 on bottom/top, θy=0 on left/right
    let mut constrained = Vec::new();
    let tol = 1e-10;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let is_left = node.x.abs() < tol;
        let is_right = (node.x - a).abs() < tol;
        let is_bottom = node.y.abs() < tol;
        let is_top = (node.y - b).abs() < tol;

        if is_left || is_right || is_bottom || is_top {
            constrained.push(3 * i); // w=0
        }
        if is_bottom || is_top {
            constrained.push(3 * i + 1); // θx=0
        }
        if is_left || is_right {
            constrained.push(3 * i + 2); // θy=0
        }
    }

    apply_homogeneous_dirichlet_dofs(&mut system, &constrained);

    let csr = system.matrix.to_csr();
    let result = gmres(
        &csr,
        &system.rhs,
        &GmresConfig {
            max_iterations: 5000,
            restart: 200,
            tolerance: 2e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );

    assert!(
        result.converged,
        "Solver should converge: residual={:.2e}",
        result.residual
    );

    // Extract center deflection
    let center_x = a / 2.0;
    let center_y = b / 2.0;
    let mut center_w = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        if (node.x - center_x).abs() < 1e-8 && (node.y - center_y).abs() < 1e-8 {
            center_w = result.x[3 * i].abs();
            break;
        }
    }

    (center_w, result.x, space.num_global_dofs(&mesh))
}

fn solve_stabilized(a: f64, b: f64, nx: usize, ny: usize, h: f64) -> (f64, Vec<f64>, usize) {
    solve_plate(a, b, nx, ny, h, MindlinPlateShearStabilization::Dsg)
}

fn test_mindlin_plate_matches_navier_solution() {
    let a = 1.0;
    let b = 1.0;

    for &(h, nx, tolerance) in &[(0.1, 40, 0.02), (0.2, 40, 0.02), (0.5, 20, 0.02)] {
        let exact = exact_mindlin_center_deflection(a, b, h);
        let (w, _, dofs) = solve_stabilized(a, b, nx, nx, h);
        let rel_err = (w - exact).abs() / exact;
        println!(
            "Mindlin plate Navier check: h={:.3}, nx={}, w_fem={:.6e}, w_exact={:.6e}, rel_err={:.3}% ({} dofs)",
            h,
            nx,
            w,
            exact,
            100.0 * rel_err,
            dofs
        );
        assert!(
            rel_err < tolerance,
            "DSG/MITC Mindlin plate should match Navier solution: h={h}, rel_err={rel_err:.3e}"
        );
    }
}

fn test_mindlin_plate_refinement_consistency() {
    let a = 1.0;
    let b = 1.0;
    let h = 0.2;

    let (w_20, _, dofs_20) = solve_stabilized(a, b, 20, 20, h);
    let (w_40, _, dofs_40) = solve_stabilized(a, b, 40, 40, h);

    println!(
        "Mindlin plate refinement: h={}, nx=20: w={:.6e} ({} dofs), nx=40: w={:.6e} ({} dofs)",
        h, w_20, dofs_20, w_40, dofs_40
    );

    // Both should be positive and finite
    assert!(
        w_20 > 0.0 && w_20.is_finite(),
        "Deflection should be positive and finite"
    );
    assert!(
        w_40 > 0.0 && w_40.is_finite(),
        "Deflection should be positive and finite"
    );

    let ratio = w_40 / w_20;
    println!("  Ratio w_40/w_20 = {:.4}", ratio);
    assert!(
        ratio > 0.95 && ratio < 1.05,
        "Mesh refinement should give consistent results: ratio={:.4}",
        ratio
    );
}

fn test_mindlin_plate_thickness_trend() {
    let a = 1.0;
    let b = 1.0;
    let nx = 20;

    let (w_thin, _, _) = solve_stabilized(a, b, nx, nx, 0.1);
    let (w_medium, _, _) = solve_stabilized(a, b, nx, nx, 0.2);
    let (w_thick, _, _) = solve_stabilized(a, b, nx, nx, 0.5);

    println!(
        "Mindlin plate thickness trend: h=0.1: w={:.6e}, h=0.2: w={:.6e}, h=0.5: w={:.6e}",
        w_thin, w_medium, w_thick
    );

    assert!(
        w_thin > w_medium && w_medium > w_thick,
        "Thicker plate should have smaller deflection"
    );
}

fn test_dsg_reduces_shear_locking() {
    let a = 1.0;
    let b = 1.0;
    let h = 0.1;
    let nx = 40;
    let exact = exact_mindlin_center_deflection(a, b, h);

    let (w_dsg, _, _) = solve_plate(a, b, nx, nx, h, MindlinPlateShearStabilization::Dsg);
    let (w_consistent, _, _) =
        solve_plate(a, b, nx, nx, h, MindlinPlateShearStabilization::Consistent);
    let err_dsg = (w_dsg - exact).abs() / exact;
    let err_consistent = (w_consistent - exact).abs() / exact;

    println!(
        "Mindlin DSG locking check: h={:.3}, DSG={:.6e} ({:.2}% err), compatible={:.6e} ({:.2}% err)",
        h,
        w_dsg,
        100.0 * err_dsg,
        w_consistent,
        100.0 * err_consistent
    );

    assert!(w_dsg > w_consistent, "DSG projection should be less stiff");
    assert!(
        err_dsg < err_consistent,
        "DSG projection should reduce locking error"
    );
    assert!(
        err_dsg < 0.02,
        "DSG thin-plate error {err_dsg:.3e} exceeds the 2% analytical target"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_mindlin_plate_matches_navier_solution();
        test_mindlin_plate_refinement_consistency();
        test_mindlin_plate_thickness_trend();
        test_dsg_reduces_shear_locking();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
