//! Bounded Beta Test
//!
//! Test if using a bounded Burton-Miller β improves solution accuracy.
//! The standard β = i/k can cause ill-conditioning at low frequencies because
//! E[1] errors get amplified. Using β = i/(k + k_ref) where k_ref ~ 1/element_size
//! reduces the β magnitude and thus the E[1] impact.

fn test_bounded_beta() {
    use num_complex::Complex64;

    use sonium_bem::core::assembly::tbem::build_tbem_system_with_beta;
    use sonium_bem::core::incident::IncidentField;
    use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
    use sonium_bem::core::types::{BoundaryCondition, PhysicsParams};
    use sonium_qa::rigid_sphere_surface_relative_l2_error;
    use std::f64::consts::PI;

    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;

    println!("=== Bounded Beta Test ===\n");

    for ka_target in [0.2, 0.5, 1.0] {
        let frequency = ka_target * speed_of_sound / (2.0 * PI * radius);
        let k = 2.0 * PI * frequency / speed_of_sound;

        println!("=== ka = {:.2} ===\n", ka_target);

        let mesh = generate_icosphere_mesh(radius, 2);
        let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);
        let avg_elem_size = mesh.average_element_area().sqrt();

        // Prepare elements
        let mut elements = mesh.elements.clone();
        for (i, elem) in elements.iter_mut().enumerate() {
            elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            elem.dof_addresses = vec![i];
        }

        let n = elements.len();

        // Build RHS
        let incident = IncidentField::plane_wave_z();
        let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
        let normals: Vec<[f64; 3]> = elements.iter().map(|e| e.normal).collect();

        println!(
            "Testing different β values (element size = {:.4})",
            avg_elem_size
        );
        println!("Standard β = i/k = i*{:.4}", 1.0 / k);
        println!("Accuracy uses the area-weighted complex Mie trace at every element.\n");

        // Test different beta scalings
        let beta_factors = [0.1, 0.2, 0.5, 1.0, 2.0];

        for &factor in &beta_factors {
            // Scale β relative to standard value
            let eta = factor / k;
            let beta = Complex64::new(0.0, eta);

            // Build RHS with this beta
            let rhs = incident.compute_rhs_with_beta(&centers, &normals, &physics, beta);

            // Build system
            let system = build_tbem_system_with_beta(&elements, &mesh.nodes, &physics, beta);

            // Solve
            let p = solve_system(&system.matrix, &rhs);
            let avg_p: f64 = p.iter().map(|x| x.norm()).sum::<f64>() / n as f64;
            let error =
                rigid_sphere_surface_relative_l2_error(&p, &elements, k, radius, 50) * 100.0;

            // Check row sum
            let mut avg_row_sum = Complex64::new(0.0, 0.0);
            for i in 0..n {
                let mut row_sum = Complex64::new(0.0, 0.0);
                for j in 0..n {
                    row_sum += system.matrix[[i, j]];
                }
                avg_row_sum += row_sum;
            }
            avg_row_sum /= n as f64;

            println!(
                "  β = {:.2}×(i/k): η = {:.4}, |p| = {:.4}, error = {:.1}%, row_sum = {:.4}+{:.4}i",
                factor, eta, avg_p, error, avg_row_sum.re, avg_row_sum.im
            );
        }

        // Also test optimal bounded beta
        let k_ref = 1.0 / avg_elem_size;
        let eta_bounded = 1.0 / (k + k_ref);
        let beta_bounded = Complex64::new(0.0, eta_bounded);

        let rhs_bounded =
            incident.compute_rhs_with_beta(&centers, &normals, &physics, beta_bounded);

        let system_bounded =
            build_tbem_system_with_beta(&elements, &mesh.nodes, &physics, beta_bounded);
        let p_bounded = solve_system(&system_bounded.matrix, &rhs_bounded);
        let avg_p_bounded: f64 = p_bounded.iter().map(|x| x.norm()).sum::<f64>() / n as f64;
        let error_bounded =
            rigid_sphere_surface_relative_l2_error(&p_bounded, &elements, k, radius, 50) * 100.0;

        let mut avg_row_sum_bounded = Complex64::new(0.0, 0.0);
        for i in 0..n {
            let mut row_sum = Complex64::new(0.0, 0.0);
            for j in 0..n {
                row_sum += system_bounded.matrix[[i, j]];
            }
            avg_row_sum_bounded += row_sum;
        }
        avg_row_sum_bounded /= n as f64;

        println!(
            "\n  Bounded β = i/(k+1/h): η = {:.4}, |p| = {:.4}, error = {:.1}%, row_sum = {:.4}+{:.4}i",
            eta_bounded,
            avg_p_bounded,
            error_bounded,
            avg_row_sum_bounded.re,
            avg_row_sum_bounded.im
        );
        println!();

        assert!(
            error_bounded < 1.0,
            "Area-weighted complex surface error {error_bounded:.2}% exceeds 1%"
        );
    }
}

fn solve_system(
    a: &sonium_solvers::dense::DenseMatrix<num_complex::Complex64>,
    b: &[num_complex::Complex64],
) -> Vec<num_complex::Complex64> {
    use sonium_bem::core::solver::{GmresConfig, gmres_solve_tbem_with_ilu};
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
    let gmres_result = gmres_solve_tbem_with_ilu(a, b, &gmres_cfg);
    assert!(gmres_result.converged, "GMRES+ILU failed");
    gmres_result.x
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_bounded_beta();
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
