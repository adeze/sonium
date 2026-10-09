//! High-resolution Mie-series convergence study
//!
//! Demonstrates that the constant-element BEM converges to the analytical
//! complex Mie field as the mesh is refined.
//!
//! Run with:
//! ```sh
//! cargo run -p sonium-bem --release --example mie_convergence
//! ```

use num_complex::Complex64;
use sonium_bem::analytical::sphere_scattering_3d;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::types::PhysicsParams;
use sonium_bem::core::{BemProblem, BemSolver, SolverMethod};
use sonium_qa::relative_error;
use std::f64::consts::PI;
use std::time::Instant;

fn freq_for_ka(ka: f64, radius: f64, c: f64) -> f64 {
    ka * c / (2.0 * PI * radius)
}

fn test_mie_convergence() {
    let radius = 1.0;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka = 1.0;
    let frequency = freq_for_ka(ka, radius, speed_of_sound);
    let k = 2.0 * PI * frequency / speed_of_sound;

    let eval_r = 2.0;
    let theta_points: Vec<f64> = (0..=36).map(|i| i as f64 * PI / 36.0).collect();
    let analytical = sphere_scattering_3d(k, radius, 30, vec![eval_r], theta_points.clone());

    println!("=== Mie-Series Convergence Study ===");
    println!("  ka = {ka:.2}, radius = {radius}, eval_r = {eval_r}");
    println!(
        "  Analytical reference: 30-term Mie series at {} angles\n",
        theta_points.len()
    );

    println!(
        "{:>10}  {:>6}  {:>12}  {:>10}",
        "Mesh", "DOFs", "L2 err (%)", "Time (s)"
    );
    println!("{}", "-".repeat(45));

    // The default release gate stops at 1,280 elements; subdivision 4 already
    // has 5,120 dense unknowns and belongs in an explicitly requested run.
    // Set MIE_MAX_SUBDIV=4 or 5 to include the heavier levels.
    let max_subdiv = std::env::var("MIE_MAX_SUBDIV")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    assert!(
        max_subdiv >= 3,
        "MIE_MAX_SUBDIV must be at least 3 for a convergence study"
    );

    let mut errors = Vec::with_capacity(max_subdiv);

    for subdiv in 1..=max_subdiv {
        let mesh = generate_icosphere_mesh(radius, subdiv);
        let _n_elem = mesh.elements.len();
        let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);
        let incident = sonium_bem::core::incident::IncidentField::plane_wave_z();

        let problem = BemProblem {
            mesh,
            physics,
            incident_field: incident,
            bc_type: sonium_bem::core::bem_solver::BoundaryConditionType::Rigid,
            use_burton_miller: true,
        };

        let solver = BemSolver::new().with_solver_method(SolverMethod::BiCgStab);

        let t0 = Instant::now();
        let solution = solver.solve(&problem).expect("BEM solve failed");
        let elapsed = t0.elapsed().as_secs_f64();

        let bem_pressures: Vec<Complex64> = theta_points
            .iter()
            .map(|&theta| {
                let pt = [eval_r * theta.sin(), 0.0, eval_r * theta.cos()];
                solution.evaluate_pressure(&pt)
            })
            .collect();

        let error = relative_error(&bem_pressures, &analytical.pressure);
        assert!(
            error.is_finite(),
            "non-finite Mie error at subdivision {subdiv}"
        );
        errors.push(error);

        println!(
            "  ico-{subdiv}  {:>6}  {:>11.2}%  {:>9.2}s",
            solution.num_dofs(),
            error * 100.0,
            elapsed,
        );
    }

    for (index, pair) in errors.windows(2).enumerate() {
        assert!(
            pair[1] < pair[0],
            "Mie error did not decrease from subdivision {} to {}: {errors:?}",
            index + 1,
            index + 2
        );
        let order = (pair[0] / pair[1]).ln() / 2.0_f64.ln();
        assert!(order > 0.5, "Mie convergence order {order:.3} is below 0.5");
    }
    assert!(
        errors.last().is_some_and(|error| *error < 0.005),
        "Finest complex Mie error must be below 0.5%: {errors:?}"
    );

    println!("\nNote: constant-element BEM on flat polyhedral surfaces has O(h) geometry");
    println!("approximation error. The far-field L2 error converges slowly because the flat");
    println!("element normals limit the accuracy of the double-layer kernel K'.");
    println!("Higher-order (quadratic) geometry elements would dramatically improve convergence.");
    println!("Run with MIE_MAX_SUBDIV=4 or 5 to include the heavy dense levels.");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_mie_convergence();
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
