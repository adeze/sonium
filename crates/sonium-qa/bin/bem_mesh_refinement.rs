//! Mesh-refinement check for the explicit `beta_scale = 2` TBEM path.
//!
//! Earlier versions averaged three broad angular bins.  A coarse icosphere
//! can leave a bin empty, producing NaN; the final `error > tolerance` check
//! then evaluated false and printed `SUCCESS`.  This version compares the
//! aligned complex field on every element with positive triangle weights.

use num_complex::Complex64;
use sonium_bem::analytical::sphere_scattering_3d;
use sonium_bem::core::assembly::tbem::build_tbem_system_scaled;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::solver::{GmresConfig, gmres_solve_tbem_with_ilu};
use sonium_bem::core::types::{BoundaryCondition, Element, PhysicsParams};
use sonium_qa::validation::convergence_rate_with_sizes;
use sonium_qa::weighted_relative_l2_error;
use std::f64::consts::PI;

fn mie_surface_value(k: f64, radius: f64, element: &Element) -> Complex64 {
    let [x, y, z] = element.center;
    let center_radius = x.hypot(y).hypot(z);
    assert!(center_radius.is_finite() && center_radius > 0.0);
    let theta = (z / center_radius).clamp(-1.0, 1.0).acos();
    sphere_scattering_3d(k, radius, 50, vec![radius], vec![theta]).pressure[0]
}

fn run_refinement(ka: f64) {
    const RADIUS: f64 = 0.1;
    const SPEED_OF_SOUND: f64 = 343.0;
    const DENSITY: f64 = 1.21;

    let frequency = ka * SPEED_OF_SOUND / (2.0 * PI * RADIUS);
    let k = 2.0 * PI * frequency / SPEED_OF_SOUND;
    let physics = PhysicsParams::new(frequency, SPEED_OF_SOUND, DENSITY, false);
    let beta = physics.burton_miller_beta_scaled(2.0);
    let mut errors = Vec::new();
    let mut sizes = Vec::new();

    println!("\nka={ka:.1}, beta_scale=2");
    println!(
        "{:>6} | {:>12} | {:>12} | {:>8}",
        "Elems", "weighted L2", "h", "iters"
    );
    println!("{}", "-".repeat(49));

    // Whole-surface integration is defined even on the 80-element mesh that
    // made the old angular-bin check NaN.  Keep subdivision 4 opt-in because
    // this explicit dense assembly is intentionally the slow reference path.
    let subdivisions: &[usize] = if std::env::var("MESH_HEAVY").as_deref() == Ok("1") {
        &[1, 2, 3, 4]
    } else {
        &[1, 2, 3]
    };
    for &subdivisions in subdivisions {
        let mesh = generate_icosphere_mesh(RADIUS, subdivisions);
        let mut elements = mesh.elements.clone();
        for (i, element) in elements.iter_mut().enumerate() {
            element.boundary_condition =
                BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            element.dof_addresses = vec![i];
        }

        let system = build_tbem_system_scaled(&elements, &mesh.nodes, &physics, 2.0);
        let centers: Vec<[f64; 3]> = elements.iter().map(|element| element.center).collect();
        let normals: Vec<[f64; 3]> = elements.iter().map(|element| element.normal).collect();
        let incident = IncidentField::plane_wave_z();
        // The direct exterior equation uses the negative incident trace on
        // the right-hand side.  Comparing magnitudes used to hide this sign;
        // use the canonical API so the phase convention stays aligned with
        // `build_tbem_system_scaled`.
        let rhs = incident.compute_rhs_with_beta(&centers, &normals, &physics, beta);

        let solve = gmres_solve_tbem_with_ilu(
            &system.matrix,
            &rhs,
            &GmresConfig {
                max_iterations: 1000,
                restart: 50,
                tolerance: 1e-10,
                print_interval: 0,
                convergence_check_interval: 1,
                min_restart: None,
                max_restart: None,
                exact_residual_check_interval: 0,
            },
        );
        assert!(
            solve.converged,
            "GMRES+ILU failed at subdivision {subdivisions}"
        );
        assert!(solve.residual.is_finite(), "GMRES residual is non-finite");

        let reference: Vec<Complex64> = elements
            .iter()
            .map(|element| mie_surface_value(k, RADIUS, element))
            .collect();
        let weights: Vec<f64> = elements.iter().map(|element| element.area).collect();
        let error = weighted_relative_l2_error(&solve.x, &reference, &weights);
        assert!(error.is_finite(), "surface error is non-finite");

        let h = (weights.iter().sum::<f64>() / elements.len() as f64).sqrt();
        assert!(h.is_finite() && h > 0.0);
        println!(
            "{:>6} | {:>12.5e} | {:>12.5e} | {:>8}",
            elements.len(),
            error,
            h,
            solve.iterations
        );
        errors.push(error);
        sizes.push(h);
    }

    let order = convergence_rate_with_sizes(&errors, &sizes).unwrap_or_else(|| {
        panic!("ka={ka}: weighted complex error did not decrease monotonically")
    });
    let final_error = *errors.last().expect("three refinement levels");
    println!(
        "observed mean order={order:.3}, finest error={:.3}%",
        100.0 * final_error
    );

    assert!(
        order >= 0.5,
        "ka={ka}: convergence order {order:.3} is below 0.5"
    );
    assert!(
        final_error <= 0.01,
        "ka={ka}: finest weighted L2 error {:.3}% exceeds 1%",
        100.0 * final_error
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== TBEM mesh refinement with explicit beta scaling ===");
        for ka in [0.5, 1.0] {
            run_refinement(ka);
        }
    });

    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
