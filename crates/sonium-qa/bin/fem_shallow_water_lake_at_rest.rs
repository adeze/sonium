use sonium_fem::assembly::{assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::fluid::shallow_water::{
    BedSlopeSourceForm, ShallowWaterGravityWaveForm, ShallowWaterMaterial,
};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, gmres};

/// Test 1: Linear free-surface exact solution on flat bed.
/// For η(x,y) = x + y (linear), ∇²η = 0, so with f=0 and matching Dirichlet BCs,
/// P1 FEM should reproduce the exact solution at nodes.
fn test_shallow_water_linear_exact() {
    let l = 1.0;
    let h = 0.1;
    let nx = 20;
    let ny = 2;
    let gravity = 9.81;
    let mean_depth = 2.0;

    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);
    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);

    let material = ShallowWaterMaterial {
        gravity,
        mean_depth,
        manning_n: 0.0,
    };
    let form = ShallowWaterGravityWaveForm { material };

    let matrix = assemble_bilinear(&form, &space, &space, &mesh, 2);
    let rhs = vec![0.0; n_dofs];

    let mut system = FemSystem::new(matrix, rhs);
    // Dirichlet on all four boundaries matching η(x,y) = x + y
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|x, y, _| x + y, None);
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|x, y, _| x + y, None);
    apply_dirichlet_values(&mut system, &mesh, &space, 3, &|x, y, _| x + y, None);
    apply_dirichlet_values(&mut system, &mesh, &space, 4, &|x, y, _| x + y, None);

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

    let mut max_err: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let exact = node.x + node.y;
        let err = (result.x[i] - exact).abs();
        max_err = max_err.max(err);
    }

    println!("Shallow water linear exact: max_err={:.4e}", max_err);
    assert!(
        max_err < 1e-10,
        "Linear exact error too large: {:.4e}",
        max_err
    );
}

/// Test 2: Lake-at-rest consistency (flat free surface on sloped bed).
/// For a flat free surface η + z_b = H (constant), with η = H - z_b,
/// the gravity-wave stiffness and bed-slope source should balance.
/// We test the residual: A*η_exact - b_source should be small.
fn test_shallow_water_lake_at_rest() {
    let l = 1.0;
    let h = 0.1;
    let nx = 40;
    let ny = 2;
    let gravity = 9.81;
    let mean_depth = 2.0;
    let bed_slope = [0.05, 0.0, 0.0]; // z_b = 0.05 * x
    let total_depth = 3.0; // H = total_depth = constant

    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);
    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);

    let material = ShallowWaterMaterial {
        gravity,
        mean_depth,
        manning_n: 0.0,
    };
    let gravity_form = ShallowWaterGravityWaveForm { material };
    let slope_form = BedSlopeSourceForm::new(material, bed_slope);

    let matrix = assemble_bilinear(&gravity_form, &space, &space, &mesh, 2);
    let rhs = assemble_linear(&slope_form, &space, &mesh, 2);

    // Build exact free-surface elevation: η = H - z_b = H - slope_x * x
    let mut eta_exact = vec![0.0; n_dofs];
    for (i, node) in mesh.nodes.iter().enumerate() {
        eta_exact[i] = total_depth - bed_slope[0] * node.x;
    }

    // Compute residual: A*η_exact - b
    let csr = matrix.to_csr();
    let a_eta = csr.matvec(&eta_exact);
    let mut max_resid: f64 = 0.0;
    for i in 0..n_dofs {
        let resid = (a_eta[i] - rhs[i]).abs();
        max_resid = max_resid.max(resid);
    }

    println!("Shallow water lake-at-rest: max_resid={:.4e}", max_resid);
    // The weak bed-slope form uses the same gradient operator and quadrature
    // as the gravity-wave stiffness, so a P1 lake at rest must balance to
    // roundoff rather than merely being "small relative to scale".
    let scale = gravity * mean_depth * bed_slope[0] * l;
    let rel_resid = max_resid / scale;
    println!("Lake-at-rest relative residual: {:.4e}", rel_resid);
    assert!(
        rel_resid < 1.0e-12,
        "Lake-at-rest residual too large: {:.4e}",
        rel_resid
    );
}

/// Test 3: Dam break dry-bed front speed.
/// For a dam break from height H onto a dry bed, the front propagates at 2*sqrt(g*H).
fn test_shallow_water_dam_break_speed() {
    let gravity = 9.81;
    let h = 4.0;
    let expected_front = 2.0 * f64::sqrt(gravity * h);
    let computed_front =
        sonium_fem::physics::fluid::shallow_water::dam_break_dry_bed_front_speed(gravity, h);
    println!(
        "Dam break front speed: computed={:.4e}, expected={:.4e}",
        computed_front, expected_front
    );
    assert!((computed_front - expected_front).abs() < 1e-10);
}

/// Test 4: HLL wave speed bounds for a Riemann problem.
/// For left state (h_L=4, u_L=0) and right state (h_R=1, u_R=0),
/// the HLL speeds should bracket the physical solution.
fn test_shallow_water_hll_speeds() {
    let gravity = 9.81;
    let h_l = 4.0;
    let u_l = 0.0;
    let h_r = 1.0;
    let u_r = 0.0;

    let (s_l, s_r) = sonium_fem::physics::fluid::shallow_water::shallow_water_hll_wave_speeds(
        gravity, h_l, u_l, h_r, u_r,
    );
    println!("HLL wave speeds: s_L={:.4e}, s_R={:.4e}", s_l, s_r);
    assert!(
        s_l < 0.0,
        "s_L should be negative for subcritical dam break"
    );
    assert!(
        s_r > 0.0,
        "s_R should be positive for subcritical dam break"
    );

    // Also check that the speeds exceed the characteristic speeds
    let c_l = (gravity * h_l).sqrt();
    let c_r = (gravity * h_r).sqrt();
    assert!(
        s_l <= u_l - c_l,
        "s_L should be left of or at left characteristic"
    );
    assert!(
        s_r >= u_r + c_r,
        "s_R should be right of or at right characteristic"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: shallow water lake-at-rest ===");
        test_shallow_water_linear_exact();
        test_shallow_water_lake_at_rest();
        test_shallow_water_dam_break_speed();
        test_shallow_water_hll_speeds();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
