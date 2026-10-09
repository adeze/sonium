use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::fluid::shallow_water::{
    ShallowWaterGravityWaveForm, ShallowWaterMaterial, dam_break_dry_bed_front_speed,
    lake_at_rest_free_surface, shallow_water_hll_wave_speeds,
};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, gmres};

fn test_shallow_water_linear_profile() {
    let l = 1.0;
    let h = 0.1;
    let nx = 40;
    let ny = 2;
    let gravity = 9.81;
    let mean_depth = 1.0;

    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);
    let space = H1Space::new(PolynomialDegree::P1);

    let material = ShallowWaterMaterial {
        gravity,
        mean_depth,
        manning_n: 0.0,
    };
    let form = ShallowWaterGravityWaveForm { material };

    let matrix = assemble_bilinear(&form, &space, &space, &mesh, 2);
    let rhs = vec![0.0; space.num_global_dofs(&mesh)];

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

    let mut max_err: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let exact = node.x / l;
        let err = (result.x[i] - exact).abs();
        max_err = max_err.max(err);
    }

    println!("Shallow water linear profile: max_err={:.4e}", max_err);
    assert!(
        max_err < 1e-10,
        "Shallow water error too large: {:.4e}",
        max_err
    );
}

fn test_shallow_water_wave_speed() {
    let gravity = 9.81;
    let mean_depth = 4.0;
    let material = ShallowWaterMaterial {
        gravity,
        mean_depth,
        manning_n: 0.0,
    };
    let c = material.wave_speed();
    let c_exact = (gravity * mean_depth).sqrt();
    println!(
        "Shallow water wave speed: c={:.4e}, exact={:.4e}",
        c, c_exact
    );
    assert!(
        (c - c_exact).abs() < 1e-10,
        "Wave speed mismatch: {:.4e}",
        (c - c_exact).abs()
    );
}

fn test_lake_at_rest_and_dam_break_helpers() {
    let gravity = 9.81;
    let eta_left = lake_at_rest_free_surface(2.0, 10.0);
    let eta_right = lake_at_rest_free_surface(3.0, 9.0);
    let front_speed = dam_break_dry_bed_front_speed(gravity, 1.0);
    let (s_l, s_r) = shallow_water_hll_wave_speeds(gravity, 1.0, 0.0, 0.25, 0.0);
    let material = ShallowWaterMaterial {
        gravity,
        mean_depth: 1.0,
        manning_n: 0.03,
    };

    println!(
        "Shallow water helpers: eta={:.4e}/{:.4e}, front={:.4e}, HLL=({:.4e},{:.4e}), Fr={:.4e}",
        eta_left,
        eta_right,
        front_speed,
        s_l,
        s_r,
        material.froude_number(1.0)
    );
    assert!((eta_left - eta_right).abs() < 1.0e-12);
    assert!((front_speed - 2.0 * gravity.sqrt()).abs() < 1.0e-12);
    assert!(s_l < 0.0 && s_r > 0.0);
    assert!(material.froude_number(1.0) < 1.0);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: shallow water standing wave ===");
        test_shallow_water_linear_profile();
        test_shallow_water_wave_speed();
        test_lake_at_rest_and_dam_break_helpers();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
