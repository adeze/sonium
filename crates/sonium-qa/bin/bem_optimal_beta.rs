//! Optimal Beta Convergence Test
//!
//! Tests convergence at ka=1.0 with the production Mie-validated β scale.

fn test_optimal_beta() {
    use num_complex::Complex64;

    use sonium_bem::core::assembly::tbem::build_tbem_system_scaled;
    use sonium_bem::core::incident::IncidentField;
    use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
    use sonium_bem::core::solver::{GmresConfig, gmres_solve_tbem_with_ilu};
    use sonium_bem::core::types::{BoundaryCondition, PhysicsParams};
    use sonium_qa::rigid_sphere_surface_relative_l2_error;
    use std::f64::consts::PI;

    println!("=== Mie-validated beta convergence test at ka = 1.0 ===\n");

    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka_target = 1.0;
    let frequency = ka_target * speed_of_sound / (2.0 * PI * radius);
    let k = 2.0 * PI * frequency / speed_of_sound;
    let scale = PhysicsParams::optimal_beta_scale(ka_target);

    let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);
    let beta = physics.burton_miller_beta_scaled(scale);

    println!(
        "ka = {:.2}, β = {:.4}i (scale = {})",
        ka_target, beta.im, scale
    );

    println!(
        "{:>8} | {:>10} | {:>24}",
        "Elems", "GMRES iters", "weighted complex error%"
    );
    println!("{}", "-".repeat(50));

    let subdivs: &[usize] = if std::env::var("MESH_HEAVY").as_deref() == Ok("1") {
        &[1, 2, 3, 4]
    } else {
        &[1, 2, 3]
    };
    let mut errors = Vec::with_capacity(subdivs.len());
    for &subdivisions in subdivs {
        let mesh = generate_icosphere_mesh(radius, subdivisions);
        let n = mesh.elements.len();

        let mut elements = mesh.elements.clone();
        for (i, elem) in elements.iter_mut().enumerate() {
            elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            elem.dof_addresses = vec![i];
        }

        let system = build_tbem_system_scaled(&elements, &mesh.nodes, &physics, scale);

        let incident = IncidentField::plane_wave_z();
        let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
        let normals: Vec<[f64; 3]> = elements.iter().map(|e| e.normal).collect();
        let rhs = incident.compute_rhs_with_beta(&centers, &normals, &physics, beta);

        let gmres_cfg = GmresConfig {
            max_iterations: 500,
            restart: 50,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        };
        let gmres_result = gmres_solve_tbem_with_ilu(&system.matrix, &rhs, &gmres_cfg);
        assert!(gmres_result.converged, "GMRES+ILU failed");
        let error_pct = 100.0
            * rigid_sphere_surface_relative_l2_error(&gmres_result.x, &elements, k, radius, 50);

        println!(
            "{:>8} | {:>10} | {:>23.3}",
            n, gmres_result.iterations, error_pct
        );
        assert!(error_pct.is_finite());
        errors.push(error_pct);
    }

    assert!(
        errors.windows(2).all(|pair| pair[1] < pair[0]),
        "Mie surface error must decrease at every refinement: {errors:?}"
    );
    assert!(
        errors.last().is_some_and(|error| *error < 0.5),
        "finest Mie surface error must be below 0.5%: {errors:?}"
    );

    println!("\n=== Summary ===");
    println!("At ka = 1.0, the production β scale converges monotonically below 0.5%.");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_optimal_beta();
    });
    match result {
        Ok(()) => {
            println!("SUCCESS");
        }
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
