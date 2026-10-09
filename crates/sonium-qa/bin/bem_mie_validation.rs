//! BEM Solver Benchmark Validation
//!
//! Validates the BEM solver against Mie-series analytical solutions for
//! rigid sphere scattering across several ka regimes and mesh resolutions.
//!
//! These tests are intentionally slow (they run actual BEM solves) and
//! compare numerical results to exact analytical solutions.

use num_complex::Complex64;
use sonium_bem::analytical::sphere_scattering_3d;
use sonium_bem::core::{BemProblem, BemSolver, SolverMethod};
use sonium_qa::{relative_error, weighted_relative_l2_error};
use sonium_xem::io::binary_output_dir;
use std::f64::consts::PI;
use std::time::Instant;

const FAR_FIELD_ERROR_TOLERANCE: f64 = 0.01;
const SURFACE_ERROR_TOLERANCE: f64 = 0.005;
const SWEEP_ERROR_TOLERANCE: f64 = 0.01;

fn out_dir() -> std::path::PathBuf {
    binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute frequency such that k·radius = ka_target.
fn freq_for_ka(ka: f64, radius: f64, speed_of_sound: f64) -> f64 {
    ka * speed_of_sound / (2.0 * PI * radius)
}

/// Convert (r, theta) in the x-z plane (phi=0) to Cartesian coordinates.
/// theta is the polar angle from the z-axis (incident wave direction).
fn spherical_to_cartesian(r: f64, theta: f64) -> [f64; 3] {
    [r * theta.sin(), 0.0, r * theta.cos()]
}

/// Build the standard evaluation theta grid (0..=pi, 5 points).
fn standard_theta_points() -> Vec<f64> {
    vec![0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI]
}

/// Collect BEM pressure values at (r, theta) pairs evaluated at fixed r.
fn collect_bem_pressures(
    solution: &sonium_bem::core::BemSolution,
    r: f64,
    theta_points: &[f64],
) -> Vec<Complex64> {
    theta_points
        .iter()
        .map(|&theta| {
            let pt = spherical_to_cartesian(r, theta);
            solution.evaluate_pressure(&pt)
        })
        .collect()
}

/// Compute relative L2 error between two complex pressure vectors.
fn relative_l2_error(analytical: &[Complex64], bem: &[Complex64]) -> f64 {
    relative_error(bem, analytical)
}

// ---------------------------------------------------------------------------
// Test 1: Low frequency (ka = 0.5)
// ---------------------------------------------------------------------------

/// Validates BEM against the Mie series at ka = 0.5 (Rayleigh-to-Mie transition).
///
/// The default icosphere is required to remain below one percent relative L2.
fn test_rigid_sphere_scattering_ka_low() {
    let radius = 1.0;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka = 0.5;
    let frequency = freq_for_ka(ka, radius, speed_of_sound);
    let k = 2.0 * PI * frequency / speed_of_sound;

    println!("\n=== test_rigid_sphere_scattering_ka_low ===");
    println!("  ka = {ka:.4}, freq = {frequency:.2} Hz, k = {k:.6}");

    let problem = BemProblem::rigid_sphere_scattering(radius, frequency, speed_of_sound, density);

    let solver = BemSolver::new().with_solver_method(SolverMethod::BiCgStab);

    let t_start = Instant::now();
    let solution = solver.solve(&problem).expect("BEM solve failed");
    let _ = solution.write_vtu(&problem.mesh, out_dir().join("ka_low.vtu"));
    let solve_ms = t_start.elapsed().as_millis();

    println!(
        "  DOFs: {}, solve time: {} ms",
        solution.num_dofs(),
        solve_ms
    );

    // Evaluation radius and angles
    let eval_r = 2.0;
    let theta_points = standard_theta_points();

    let bem_pressures = collect_bem_pressures(&solution, eval_r, &theta_points);

    // Analytical reference (30 terms is more than sufficient for ka = 0.5)
    let analytical = sphere_scattering_3d(k, radius, 30, vec![eval_r], theta_points.clone());

    let l2_err = relative_l2_error(&analytical.pressure, &bem_pressures);

    println!("  Theta(deg)   BEM |p|       Analytical |p|");
    for (i, &theta) in theta_points.iter().enumerate() {
        println!(
            "    {:6.1}     {:.6}     {:.6}",
            theta.to_degrees(),
            bem_pressures[i].norm(),
            analytical.pressure[i].norm()
        );
    }
    println!(
        "  Relative L2 error: {:.4} ({:.2}%)",
        l2_err,
        l2_err * 100.0
    );

    assert!(l2_err.is_finite(), "L2 error must be finite, got {l2_err}");
    assert!(
        l2_err < FAR_FIELD_ERROR_TOLERANCE,
        "Relative L2 error {:.2}% exceeds 1% threshold for ka=0.5",
        l2_err * 100.0
    );
}

// ---------------------------------------------------------------------------
// Test 2: Mie regime (ka = 1.0) with icosphere subdivision 3
// ---------------------------------------------------------------------------

/// Validates BEM against the Mie series at ka = 1.0 (Mie regime).
///
/// Uses icosphere subdivision level 3 (~1280 elements) for better resolution.
///
/// The 1,280-element icosphere resolves this case below one percent; a large
/// tolerance here would hide operator, phase, or sampling regressions.
fn test_rigid_sphere_scattering_ka_1() {
    let radius = 1.0;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka = 1.0;
    let frequency = freq_for_ka(ka, radius, speed_of_sound);
    let k = 2.0 * PI * frequency / speed_of_sound;

    println!("\n=== test_rigid_sphere_scattering_ka_1 ===");
    println!("  ka = {ka:.4}, freq = {frequency:.2} Hz, k = {k:.6}");

    // Use icosphere at subdivision 3 (642 elements) for better resolution at ka=1
    // rigid_sphere_scattering auto-selects subdivision=3 for ka in [1,5)
    let problem = BemProblem::rigid_sphere_scattering(radius, frequency, speed_of_sound, density);

    let solver = BemSolver::new().with_solver_method(SolverMethod::BiCgStab);

    let t_start = Instant::now();
    let solution = solver.solve(&problem).expect("BEM solve failed");
    let _ = solution.write_vtu(&problem.mesh, out_dir().join("ka_1.vtu"));
    let solve_ms = t_start.elapsed().as_millis();

    println!(
        "  DOFs: {}, solve time: {} ms",
        solution.num_dofs(),
        solve_ms
    );

    let eval_r = 2.0;
    let theta_points = standard_theta_points();

    let bem_pressures = collect_bem_pressures(&solution, eval_r, &theta_points);

    let analytical = sphere_scattering_3d(k, radius, 30, vec![eval_r], theta_points.clone());

    let l2_err = relative_l2_error(&analytical.pressure, &bem_pressures);

    println!("  Theta(deg)   BEM |p|       Analytical |p|");
    for (i, &theta) in theta_points.iter().enumerate() {
        println!(
            "    {:6.1}     {:.6}     {:.6}",
            theta.to_degrees(),
            bem_pressures[i].norm(),
            analytical.pressure[i].norm()
        );
    }
    println!(
        "  Relative L2 error: {:.4} ({:.2}%)",
        l2_err,
        l2_err * 100.0
    );

    assert!(l2_err.is_finite(), "L2 error must be finite, got {l2_err}");
    assert!(
        l2_err < FAR_FIELD_ERROR_TOLERANCE,
        "Relative L2 error {:.2}% exceeds 1% threshold for ka=1.0",
        l2_err * 100.0
    );
}

// ---------------------------------------------------------------------------
// Test 3: h-convergence study
// ---------------------------------------------------------------------------

/// Verifies that BEM error decreases monotonically as the quasi-uniform sphere mesh is refined.
///
/// Four mesh levels are tested at ka = 1.0 over a 37-angle complex field at r=2.
fn test_rigid_sphere_h_convergence() {
    let radius = 1.0;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka = 1.0;
    let frequency = freq_for_ka(ka, radius, speed_of_sound);
    let k = 2.0 * PI * frequency / speed_of_sound;

    let eval_r = 2.0;
    let theta_points: Vec<f64> = (0..=36).map(|index| index as f64 * PI / 36.0).collect();
    let analytical = sphere_scattering_3d(k, radius, 30, vec![eval_r], theta_points.clone());

    // Fibonacci-hull meshes: nominal (n_theta, n_phi) pairs in increasing resolution
    let mesh_levels: &[(usize, usize)] = &[(8, 16), (12, 24), (16, 32), (20, 40)];

    println!("\n=== test_rigid_sphere_h_convergence ===");
    println!(
        "  ka = {ka:.4}, eval radius={eval_r}, {} angular samples",
        theta_points.len()
    );
    println!();
    println!(
        "  {:>14}  {:>5}  {:>12}  {:>10}",
        "Mesh (nT x nP)", "DOFs", "Complex L2%", "Time(ms)"
    );
    println!("  {}", "-".repeat(50));

    let mut errors = Vec::with_capacity(mesh_levels.len());

    for &(n_theta, n_phi) in mesh_levels {
        let problem = BemProblem::rigid_sphere_scattering_custom(
            radius,
            frequency,
            speed_of_sound,
            density,
            n_theta,
            n_phi,
        );

        let solver = BemSolver::new().with_solver_method(SolverMethod::BiCgStab);

        let t_start = Instant::now();
        let solution = solver.solve(&problem).expect("BEM solve failed");
        let _ = solution.write_vtu(
            &problem.mesh,
            out_dir().join(format!("h_conv_{}.vtu", n_theta)),
        );
        let solve_ms = t_start.elapsed().as_millis();

        let bem_pressure = collect_bem_pressures(&solution, eval_r, &theta_points);
        let rel_err = relative_l2_error(&analytical.pressure, &bem_pressure);

        println!(
            "  {:>5} x {:>5}      {:>5}  {:>11.3}%  {:>8} ms",
            n_theta,
            n_phi,
            solution.num_dofs(),
            rel_err * 100.0,
            solve_ms
        );

        errors.push(rel_err);
    }

    // Assert monotonic error decrease across all 4 mesh levels
    for i in 1..errors.len() {
        assert!(
            errors[i] < errors[i - 1],
            "Error should decrease with mesh refinement: level {} error {:.4} >= level {} error {:.4}",
            i,
            errors[i],
            i - 1,
            errors[i - 1]
        );
    }
    assert!(
        errors.iter().all(|error| error.is_finite()),
        "h-convergence errors must be finite"
    );
    assert!(
        errors.last().is_some_and(|error| *error <= 0.01),
        "finest complex h-convergence error must be at most 1%"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Surface pressure validation
// ---------------------------------------------------------------------------

/// Compares BEM surface pressure against analytical Mie-series surface pressure
/// (r = radius) for ka = 1.0.
///
/// Use the production 1,280-element icosphere for this accuracy gate.  UV
/// spheres are intentionally covered by the preceding h-convergence test;
/// mixing their coarse, anisotropic topology into this check previously led
/// to a 90% exception instead of certifying surface accuracy.  Triangle-area
/// weights still make the norm invariant to sampling density.
fn test_rigid_sphere_surface_pressure() {
    let radius = 1.0;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka = 1.0;
    let frequency = freq_for_ka(ka, radius, speed_of_sound);
    let k = 2.0 * PI * frequency / speed_of_sound;

    println!("\n=== test_rigid_sphere_surface_pressure ===");
    println!("  ka = {ka:.4}, freq = {frequency:.2} Hz");

    let problem = BemProblem::rigid_sphere_scattering(radius, frequency, speed_of_sound, density);

    let solver = BemSolver::new().with_solver_method(SolverMethod::BiCgStab);

    let t_start = Instant::now();
    let solution = solver.solve(&problem).expect("BEM solve failed");
    let _ = solution.write_vtu(&problem.mesh, out_dir().join("surface_pressure.vtu"));
    let solve_ms = t_start.elapsed().as_millis();

    println!(
        "  DOFs: {}, solve time: {} ms",
        solution.num_dofs(),
        solve_ms
    );

    // For each BEM element, compute the polar angle theta of its centre, then
    // evaluate the analytical solution at that (radius, theta) point.
    let n_elem = solution.elements.len();
    let mut analytical_pressures = Vec::with_capacity(n_elem);
    let mut bem_surface_pressures = Vec::with_capacity(n_elem);
    let mut surface_weights = Vec::with_capacity(n_elem);

    for (i, elem) in solution.elements.iter().enumerate() {
        let cx = elem.center[0];
        let cy = elem.center[1];
        let cz = elem.center[2];
        let r = (cx * cx + cy * cy + cz * cz).sqrt();

        // Polar angle from z-axis
        let theta = if r > 1e-12 { (cz / r).acos() } else { 0.0 };

        // Compare the numerical boundary density with the exact trace on the
        // physical sphere in the same angular direction.
        let analytical_at_elem = sphere_scattering_3d(k, radius, 30, vec![radius], vec![theta]);
        analytical_pressures.push(analytical_at_elem.pressure[0]);

        // BEM surface pressure for this element
        bem_surface_pressures.push(solution.surface_pressure[i]);
        surface_weights.push(elem.area);
    }

    let l2_err = weighted_relative_l2_error(
        &bem_surface_pressures,
        &analytical_pressures,
        &surface_weights,
    );

    println!(
        "  Surface pressure relative L2 error: {:.4} ({:.2}%)",
        l2_err,
        l2_err * 100.0
    );
    println!(
        "  Max BEM surface |p|:        {:.6}",
        solution.max_surface_pressure()
    );
    println!(
        "  Mean BEM surface |p|:       {:.6}",
        solution.mean_surface_pressure()
    );

    assert!(
        l2_err.is_finite(),
        "Surface pressure L2 error must be finite"
    );
    assert!(
        l2_err < SURFACE_ERROR_TOLERANCE,
        "Surface pressure L2 error {:.2}% exceeds 5% threshold",
        l2_err * 100.0
    );
}

// ---------------------------------------------------------------------------
// Test 5: Frequency sweep
// ---------------------------------------------------------------------------

/// Sweeps ka = 0.5, 1.0, 2.0, 3.0 and enforces the same strict accuracy gate
/// at every frequency.
///
/// Prints a summary table showing elements, DOFs, L2 error, and solve time
/// for each ka value.
fn test_rigid_sphere_ka_sweep() {
    let radius = 1.0;
    let speed_of_sound = 343.0;
    let density = 1.21;

    let ka_values = [0.5f64, 1.0, 2.0, 3.0];
    let eval_r = 2.0;
    let theta_points = standard_theta_points();

    println!("\n=== test_rigid_sphere_ka_sweep ===");
    println!(
        "  {:>6}  {:>8}  {:>5}  {:>12}  {:>10}",
        "ka", "freq(Hz)", "DOFs", "L2 err(%)", "time(ms)"
    );
    println!("  {}", "-".repeat(50));

    let mut all_errors = Vec::with_capacity(ka_values.len());

    for &ka in &ka_values {
        let frequency = freq_for_ka(ka, radius, speed_of_sound);
        let k = 2.0 * PI * frequency / speed_of_sound;

        let problem =
            BemProblem::rigid_sphere_scattering(radius, frequency, speed_of_sound, density);

        let solver = BemSolver::new().with_solver_method(SolverMethod::BiCgStab);

        let t_start = Instant::now();
        let solution = solver.solve(&problem).expect("BEM solve failed");
        let _ = solution.write_vtu(
            &problem.mesh,
            out_dir().join(format!("ka_sweep_ka{:.2}.vtu", ka)),
        );
        let solve_ms = t_start.elapsed().as_millis();

        let bem_pressures = collect_bem_pressures(&solution, eval_r, &theta_points);

        // Number of Mie terms: ka + 15 provides sufficient convergence up to ka=3
        let num_terms = (ka + 15.0) as usize;
        let analytical =
            sphere_scattering_3d(k, radius, num_terms, vec![eval_r], theta_points.clone());

        let l2_err = relative_l2_error(&analytical.pressure, &bem_pressures);

        println!(
            "  {:>6.2}  {:>8.2}  {:>5}  {:>11.2}%  {:>8} ms",
            ka,
            frequency,
            solution.num_dofs(),
            l2_err * 100.0,
            solve_ms
        );

        all_errors.push(l2_err);
    }

    // Every frequency is an accuracy check, not merely a finite-number smoke test.
    for (i, &err) in all_errors.iter().enumerate() {
        let ka = ka_values[i];
        assert!(
            err.is_finite(),
            "L2 error for ka={ka:.1} must be finite, got {err}"
        );
        assert!(err >= 0.0);
        assert!(
            err <= SWEEP_ERROR_TOLERANCE,
            "L2 error for ka={ka:.1} is {:.2}% and exceeds 1%",
            100.0 * err
        );
    }
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_rigid_sphere_scattering_ka_low();
        test_rigid_sphere_scattering_ka_1();
        test_rigid_sphere_h_convergence();
        test_rigid_sphere_surface_pressure();
        test_rigid_sphere_ka_sweep();
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
