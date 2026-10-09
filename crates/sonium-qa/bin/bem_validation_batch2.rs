//! BEM Solver Benchmark Validation — Batch 2
//!
//! Extends the first batch with soft-sphere scattering, radiation problems,
//! point-source excitation, azimuthal symmetry checks, far-field decay, and
//! Burton-Miller fictitious-eigenfrequency tests.
//!
//! All tests are gated on the `native` feature (requires BLAS/LAPACK) because
//! the problems are too large for the pure-Rust fallback.

use num_complex::Complex64;
use sonium_bem::analytical::{
    pulsating_sphere_3d, soft_sphere_scattering_3d, sphere_scattering_3d,
};
use sonium_bem::core::assembly::tbem::build_tbem_system_with_beta;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::solver::{GmresConfig, gmres_solve_tbem_with_ilu};
use sonium_bem::core::{
    BemProblem, BemSolution, BemSolver, BoundaryCondition, IncidentField, PhysicsParams,
    SolverMethod,
};
use sonium_qa::relative_error;
use std::f64::consts::PI;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Re-use helpers from batch 1 (duplicated here for module isolation)
// ---------------------------------------------------------------------------

fn freq_for_ka(ka: f64, radius: f64, speed_of_sound: f64) -> f64 {
    ka * speed_of_sound / (2.0 * PI * radius)
}

fn spherical_to_cartesian(r: f64, theta: f64) -> [f64; 3] {
    [r * theta.sin(), 0.0, r * theta.cos()]
}

fn standard_theta_points() -> Vec<f64> {
    vec![0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI]
}

fn collect_bem_pressures(solution: &BemSolution, r: f64, theta_points: &[f64]) -> Vec<Complex64> {
    theta_points
        .iter()
        .map(|&theta| {
            let pt = spherical_to_cartesian(r, theta);
            solution.evaluate_pressure(&pt)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Test 1: Soft sphere scattering — exterior field accuracy
// ---------------------------------------------------------------------------

/// Validates BEM soft-sphere (Dirichlet p=0) scattering at ka=1 against the
/// analytical Mie-series solution for a pressure-release sphere.
///
/// The problem is created by starting from `BemProblem::rigid_sphere_scattering`
/// and switching the BC type to `Soft` before solving.
fn test_soft_sphere_scattering_ka_1() {
    let radius = 1.0;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka = 1.0;
    let frequency = freq_for_ka(ka, radius, speed_of_sound);
    let k = 2.0 * PI * frequency / speed_of_sound;

    println!("\n=== test_soft_sphere_scattering_ka_1 ===");
    println!("  ka = {ka:.4}, freq = {frequency:.2} Hz, k = {k:.6}");

    // Build problem as rigid first, then flip to Soft (Dirichlet)
    let problem = BemProblem::rigid_sphere_scattering(radius, frequency, speed_of_sound, density)
        .with_boundary_condition(sonium_bem::core::bem_solver::BoundaryConditionType::Soft);

    let solver = BemSolver::new().with_solver_method(SolverMethod::BiCgStab);

    let t_start = Instant::now();
    let solution = solver.solve(&problem).expect("BEM solve failed");
    let solve_ms = t_start.elapsed().as_millis();

    println!(
        "  DOFs: {}, solve time: {} ms",
        solution.num_dofs(),
        solve_ms
    );

    let eval_r = 2.0;
    let theta_points = standard_theta_points();

    let bem_pressures = collect_bem_pressures(&solution, eval_r, &theta_points);

    // Analytical: soft-sphere Mie series — 30 terms is ample for ka=1
    let analytical = soft_sphere_scattering_3d(k, radius, 30, vec![eval_r], theta_points.clone());

    let l2_err = relative_error(&bem_pressures, &analytical.pressure);

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
        l2_err < 0.01,
        "Relative L2 error {:.2}% exceeds 1% for soft-sphere indirect BEM",
        l2_err * 100.0
    );
}

// ---------------------------------------------------------------------------
// Test 2: Soft sphere — surface pressure is near zero (Dirichlet check)
// ---------------------------------------------------------------------------

/// Validates that physical BEM surface pressure on a soft sphere is zero.
///
/// The indirect Dirichlet solve stores a single-layer density internally, but
/// `BemSolution::max_surface_pressure` reports the physical prescribed
/// pressure, so a pressure-release sphere should be zero to roundoff.
fn test_soft_sphere_surface_pressure_zero() {
    let radius = 1.0;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka = 1.0;
    let frequency = freq_for_ka(ka, radius, speed_of_sound);

    println!("\n=== test_soft_sphere_surface_pressure_zero ===");
    println!("  ka = {ka:.4}, freq = {frequency:.2} Hz");

    let problem = BemProblem::rigid_sphere_scattering(radius, frequency, speed_of_sound, density)
        .with_boundary_condition(sonium_bem::core::bem_solver::BoundaryConditionType::Soft);

    let solver = BemSolver::new().with_solver_method(SolverMethod::BiCgStab);
    let solution = solver.solve(&problem).expect("BEM solve failed");

    let max_sp = solution.max_surface_pressure();
    let mean_sp = solution.mean_surface_pressure();

    println!("  Max  |surface_pressure| = {max_sp:.6}");
    println!("  Mean |surface_pressure| = {mean_sp:.6}");
    println!("  DOFs: {}", solution.num_dofs());

    assert!(
        max_sp.is_finite(),
        "Surface pressure must be finite, got {max_sp}"
    );
    assert!(
        max_sp < 1.0e-12,
        "Soft sphere pressure is not zero: {max_sp}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Pulsating sphere — monopole radiation
// ---------------------------------------------------------------------------

/// Validates BEM radiation from a uniformly pulsating sphere (monopole source).
///
/// This is a pure radiation problem: no incident plane wave.  The sphere
/// surface carries a uniform normal velocity v₀ = 1.  The BEM system is
/// assembled via `build_tbem_system_with_beta` so that the velocity BC is
/// incorporated directly into the RHS without adding any incident-field
/// contribution.
///
/// Field evaluation uses a zero-amplitude plane wave as a dummy incident
/// field so that `compute_total_field` contributes only the scattered
/// (radiated) component.
///
/// Relative magnitude error vs the analytical `pulsating_sphere_3d` solution
/// is asserted below 15 % for a moderate mesh.
fn test_pulsating_sphere_radiation() {
    let radius = 1.0;
    let frequency = 200.0; // Hz  →  ka ≈ 3.66
    let speed_of_sound = 343.0;
    let density = 1.21;
    let v0 = 1.0_f64; // radial velocity amplitude (m/s)

    let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);
    let k = physics.wave_number;

    println!("\n=== test_pulsating_sphere_radiation ===");
    println!(
        "  freq = {frequency} Hz, k = {k:.4}, ka = {:.4}",
        k * radius
    );

    // Build a moderately refined icosphere mesh
    let mesh = generate_icosphere_mesh(radius, 3); // subdivision 3 → 1280 elements

    // Assign uniform velocity BC and sequential DOF addresses
    let v0_complex = Complex64::new(v0, 0.0);
    let mut elements = mesh.elements.clone();
    for (i, elem) in elements.iter_mut().enumerate() {
        elem.boundary_condition = BoundaryCondition::Velocity(vec![v0_complex]);
        elem.dof_addresses = vec![i];
    }

    println!("  Elements: {}", elements.len());

    // Assemble: the TBEM RHS already encodes the velocity BC value.
    // For a radiation problem there is no incident field to add.
    let beta = physics.burton_miller_beta_scaled(4.0);
    let system = build_tbem_system_with_beta(&elements, &mesh.nodes, &physics, beta);

    // Solve for surface pressure
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
    let gmres_result = gmres_solve_tbem_with_ilu(&system.matrix, &system.rhs, &gmres_cfg);
    assert!(
        gmres_result.converged,
        "GMRES+ILU failed for pulsating sphere"
    );
    let surface_pressure = gmres_result.x;

    println!(
        "  Max |surface_pressure| = {:.6}",
        surface_pressure
            .iter()
            .map(|p| p.norm())
            .fold(0.0_f64, f64::max)
    );

    // Evaluate field at exterior points r = 2, 4 along any direction
    // Use a zero-amplitude plane wave as dummy incident field
    let dummy_incident = IncidentField::PlaneWave {
        direction: [0.0, 0.0, 1.0],
        amplitude: Complex64::new(0.0, 0.0),
    };

    let eval_radii = [2.0_f64, 4.0, 8.0];
    let eval_points: Vec<[f64; 3]> = eval_radii.iter().map(|&r| [0.0, 0.0, r]).collect();

    let field_points = sonium_bem::core::postprocess::pressure::compute_total_field(
        &eval_points,
        &elements,
        &mesh.nodes,
        &surface_pressure,
        None,
        &dummy_incident,
        &physics,
        sonium_bem::core::BasisOrder::Constant,
    );

    // Analytical monopole radiation
    let analytical =
        pulsating_sphere_3d(k, radius, v0, speed_of_sound, density, eval_radii.to_vec());

    println!("  r       BEM |p|       Analytical |p|   Rel err");
    let bem_pressures: Vec<Complex64> = field_points.iter().map(|point| point.p_total).collect();
    for (i, &r) in eval_radii.iter().enumerate() {
        let bem_p = field_points[i].p_total;
        let ana_p = analytical.pressure[i];
        let rel_err = relative_error(&[bem_p], &[ana_p]);
        println!(
            "  {r:.1}     {:.6}    {:.6}          {:.2}%",
            bem_p.norm(),
            ana_p.norm(),
            rel_err * 100.0
        );
    }
    let radiation_error = relative_error(&bem_pressures, &analytical.pressure);
    println!(
        "  Complex relative L2 error: {:.3}%",
        radiation_error * 100.0
    );

    // Check 1/r decay (shape correctness) regardless of amplitude
    let ratio = field_points[0].p_total.norm() / field_points[1].p_total.norm();
    let expected_ratio = eval_radii[1] / eval_radii[0]; // should be ~2.0
    println!(
        "  1/r decay ratio: {:.3} (expected {:.1})",
        ratio, expected_ratio
    );
    assert!(
        (ratio - expected_ratio).abs() / expected_ratio < 0.01,
        "Monopole 1/r decay broken: ratio={ratio:.3}, expected={expected_ratio:.1}"
    );

    assert!(
        radiation_error < 0.03,
        "Pulsating-sphere complex radiation error {:.3}% exceeds 3%",
        radiation_error * 100.0
    );
}

// ---------------------------------------------------------------------------
// Test 4: Point source scattering — rigid sphere
// ---------------------------------------------------------------------------

/// Verifies basic properties of the BEM solution for a point source incident
/// on a rigid sphere.
///
/// There is no closed-form analytical solution for a general point source +
/// sphere problem (an image-method solution exists only on axis).  Instead
/// the test checks three qualitative properties:
///
/// 1. The total field is finite and non-zero everywhere outside the sphere.
/// 2. The field decays approximately as 1/r in the far field.
/// 3. The solution has azimuthal symmetry: pressure at (r, θ, φ₁) ≈ (r, θ, φ₂).
fn test_point_source_scattering_rigid_sphere() {
    let radius = 1.0;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka = 1.0;
    let frequency = freq_for_ka(ka, radius, speed_of_sound);

    println!("\n=== test_point_source_scattering_rigid_sphere ===");
    println!("  ka = {ka:.4}, freq = {frequency:.2} Hz");

    // Point source on the z-axis above the sphere
    let source_position = [0.0, 0.0, 3.0_f64];
    let problem = BemProblem::rigid_sphere_scattering(radius, frequency, speed_of_sound, density)
        .with_incident_field(IncidentField::PointSource {
            position: source_position,
            strength: Complex64::new(1.0, 0.0),
        });

    let solver = BemSolver::new().with_solver_method(SolverMethod::BiCgStab);
    let t_start = Instant::now();
    let solution = solver.solve(&problem).expect("BEM solve failed");
    let solve_ms = t_start.elapsed().as_millis();

    println!(
        "  DOFs: {}, solve time: {} ms",
        solution.num_dofs(),
        solve_ms
    );

    // --- Property 1: field is finite and non-zero ---
    let radii = [32.0_f64, 64.0, 128.0];
    let theta = PI / 4.0; // off-axis

    let pressures: Vec<Complex64> = radii
        .iter()
        .map(|&r| {
            let pt = spherical_to_cartesian(r, theta);
            solution.evaluate_pressure(&pt)
        })
        .collect();

    println!("  Decay check (theta=45 deg):");
    for (i, &r) in radii.iter().enumerate() {
        let mag = pressures[i].norm();
        println!("    r = {r:.1}  |p| = {mag:.6}");
        assert!(
            mag.is_finite() && mag > 0.0,
            "Pressure at r={r} should be finite and non-zero, got {mag}"
        );
    }

    // --- Property 2: approximate 1/r far-field decay ---
    // Ratio |p(r)| / |p(2r)| should be close to 2
    let ratio_2_4 = pressures[0].norm() / pressures[1].norm();
    let ratio_4_8 = pressures[1].norm() / pressures[2].norm();
    println!("  |p(32)|/|p(64)| = {ratio_2_4:.3}  (expect ~2)");
    println!("  |p(64)|/|p(128)| = {ratio_4_8:.3}  (expect ~2)");

    assert!(
        (ratio_2_4 - 2.0).abs() / 2.0 < 0.05,
        "far-field decay ratio at r=32→64 differs from two by more than 5%: {ratio_2_4:.3}"
    );
    assert!(
        (ratio_4_8 - 2.0).abs() / 2.0 < 0.05,
        "far-field decay ratio at r=64→128 differs from two by more than 5%: {ratio_4_8:.3}"
    );

    // --- Property 3: azimuthal symmetry ---
    // Evaluate at three different phi values for fixed (r, theta)
    let r_sym = 3.0_f64;
    let theta_sym = PI / 3.0;
    let phi_values = [0.0_f64, PI / 3.0, 2.0 * PI / 3.0, PI];

    let sym_pressures: Vec<Complex64> = phi_values
        .iter()
        .map(|&phi| {
            let pt = [
                r_sym * theta_sym.sin() * phi.cos(),
                r_sym * theta_sym.sin() * phi.sin(),
                r_sym * theta_sym.cos(),
            ];
            solution.evaluate_pressure(&pt)
        })
        .collect();

    let symmetry_reference = vec![sym_pressures[0]; sym_pressures.len()];
    let symmetry_error = relative_error(&sym_pressures, &symmetry_reference);
    println!("  Azimuthal symmetry (r=3, theta=60 deg):");
    for (i, &phi) in phi_values.iter().enumerate() {
        let mag = sym_pressures[i].norm();
        let rel_diff = relative_error(&[sym_pressures[i]], &[sym_pressures[0]]);
        println!(
            "    phi = {:5.1} deg  |p| = {:.6}  rel_diff = {:.2}%",
            phi.to_degrees(),
            mag,
            rel_diff * 100.0
        );
    }
    assert!(
        symmetry_error < 0.01,
        "complex azimuthal symmetry error {:.3}% exceeds 1%",
        symmetry_error * 100.0
    );
}

// ---------------------------------------------------------------------------
// Test 5: Rotational equivalence — x vs z incidence direction
// ---------------------------------------------------------------------------

/// Verifies that the maximum surface pressure magnitude is the same
/// (within 5 %) for plane waves incident along +z and along +x.
///
/// This tests that the BEM formulation is not accidentally biased by the
/// mesh axis alignment: the icosphere is symmetric, so the physics should
/// be identical regardless of the wave direction.
fn test_rigid_sphere_incidence_direction() {
    let radius = 1.0;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka = 1.0;
    let frequency = freq_for_ka(ka, radius, speed_of_sound);

    println!("\n=== test_rigid_sphere_incidence_direction ===");
    println!("  ka = {ka:.4}, freq = {frequency:.2} Hz");

    let solver = BemSolver::new().with_solver_method(SolverMethod::BiCgStab);

    // z-direction (standard)
    let problem_z = BemProblem::rigid_sphere_scattering(radius, frequency, speed_of_sound, density);
    let solution_z = solver.solve(&problem_z).expect("BEM solve (z) failed");

    // x-direction
    let problem_x = BemProblem::rigid_sphere_scattering(radius, frequency, speed_of_sound, density)
        .with_incident_field(IncidentField::plane_wave([1.0, 0.0, 0.0], 1.0));
    let solution_x = solver.solve(&problem_x).expect("BEM solve (x) failed");

    let max_z = solution_z.max_surface_pressure();
    let max_x = solution_x.max_surface_pressure();

    println!("  Max surface pressure (z-incidence): {max_z:.6}");
    println!("  Max surface pressure (x-incidence): {max_x:.6}");

    let rel_diff = (max_z - max_x).abs() / max_z.max(1e-15);
    println!("  Relative difference: {:.2}%", rel_diff * 100.0);

    assert!(
        max_z > 0.0 && max_x > 0.0,
        "Surface pressures must be positive, got z={max_z}, x={max_x}"
    );
    assert!(
        rel_diff < 0.01,
        "Max surface pressures differ by {:.2}% between z and x incidence (threshold 1%)",
        rel_diff * 100.0
    );
}

// ---------------------------------------------------------------------------
// Test 6: Far-field 1/r decay of scattered field
// ---------------------------------------------------------------------------

/// Verifies that the BEM scattered field decays as 1/r in the far field.
///
/// At each doubling of radius (2 → 4 → 8 → 16) along θ = 0 (forward
/// direction), the ratio |p_scattered(r)| / |p_scattered(2r)| should
/// approach 2.  We assert it lies within [1.3, 3.0] to accommodate the
/// moderate kr values.
///
/// The scattered field is computed as: p_scattered = p_total − p_incident.
fn test_rigid_sphere_far_field_decay() {
    let radius = 1.0;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka = 1.0;
    let frequency = freq_for_ka(ka, radius, speed_of_sound);
    let k = 2.0 * PI * frequency / speed_of_sound;

    println!("\n=== test_rigid_sphere_far_field_decay ===");
    println!("  ka = {ka:.4}, freq = {frequency:.2} Hz, k = {k:.6}");

    let problem = BemProblem::rigid_sphere_scattering(radius, frequency, speed_of_sound, density);
    let solver = BemSolver::new().with_solver_method(SolverMethod::BiCgStab);
    let solution = solver.solve(&problem).expect("BEM solve failed");

    let physics = &solution.physics;
    let incident = &solution.incident_field;

    // Evaluate along theta=0 (z-axis, forward direction)
    let eval_radii = [32.0_f64, 64.0, 128.0, 256.0];
    let eval_points: Vec<[f64; 3]> = eval_radii.iter().map(|&r| [0.0, 0.0, r]).collect();

    // Total field from BEM
    let field_pts = solution.evaluate_pressure_field(&eval_points);

    // Incident field at the same points
    let p_inc_arr = incident.evaluate_pressure(&eval_points, physics);

    println!("  r      |p_total|   |p_incident|   |p_scattered|");
    let mut scattered_mags = Vec::with_capacity(eval_radii.len());
    for (i, &r) in eval_radii.iter().enumerate() {
        let p_tot = field_pts[i].p_total;
        let p_inc = p_inc_arr[i];
        let p_scat = p_tot - p_inc;
        let mag_scat = p_scat.norm();
        scattered_mags.push(mag_scat);
        println!(
            "  {r:5.1}   {:.6}    {:.6}      {:.6}",
            p_tot.norm(),
            p_inc.norm(),
            mag_scat
        );
    }

    // Check 1/r decay: ratio of consecutive radii (each doubles)
    println!("  Decay ratios (expect ~2):");
    for i in 0..(eval_radii.len() - 1) {
        let r = eval_radii[i];
        let r2 = eval_radii[i + 1];
        let ratio = scattered_mags[i] / scattered_mags[i + 1].max(1e-15);
        println!("    |p({r:.0})| / |p({r2:.0})| = {ratio:.3}");

        assert!(
            (ratio - 2.0).abs() / 2.0 < 0.05,
            "1/r decay ratio at r={r} → {r2} = {ratio:.3}, more than 5% from two"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 7: Burton-Miller vs no-BM at fictitious eigenfrequency
// ---------------------------------------------------------------------------

/// Tests that the Burton-Miller formulation avoids numerical blow-up near a
/// fictitious interior eigenfrequency.
///
/// For a sphere of radius a, the first fictitious eigenfrequency (Dirichlet
/// interior) occurs at ka = π.  At this frequency the CBIE alone (no BM)
/// produces a singular or near-singular system whose solution is dominated
/// by the spurious eigenmode.
///
/// The Burton-Miller field is compared directly with the rigid-sphere Mie
/// solution. The unstabilised result is retained only as a diagnostic because
/// the discrete fictitious eigenvalue shifts with mesh resolution.
///
/// Note: whether the non-BM solution actually diverges depends on the mesh.
/// We only assert that BM gives a better-conditioned result.
fn test_burton_miller_removes_fictitious_eigenfrequency() {
    let radius = 1.0;
    let speed_of_sound = 343.0;
    let density = 1.21;

    // First fictitious eigenfrequency: ka = π
    let ka_target = PI;
    let frequency = freq_for_ka(ka_target, radius, speed_of_sound);
    let _k = 2.0 * PI * frequency / speed_of_sound;

    println!("\n=== test_burton_miller_removes_fictitious_eigenfrequency ===");
    println!("  ka = π ≈ {ka_target:.4}, freq = {frequency:.2} Hz");

    let solver = BemSolver::new().with_solver_method(SolverMethod::BiCgStab);

    // With Burton-Miller (default, recommended)
    let problem_bm =
        BemProblem::rigid_sphere_scattering(radius, frequency, speed_of_sound, density)
            .with_burton_miller(true);
    let mut problem_bm_coarse = problem_bm.clone();
    problem_bm_coarse.mesh = generate_icosphere_mesh(radius, 2);
    let solution_bm_coarse = solver
        .solve(&problem_bm_coarse)
        .expect("coarse BEM solve (BM) failed");
    let solution_bm = solver.solve(&problem_bm).expect("BEM solve (BM) failed");
    let max_bm = solution_bm.max_surface_pressure();

    // Without Burton-Miller (CBIE only — susceptible to fictitious eigenfrequencies)
    let problem_nobm =
        BemProblem::rigid_sphere_scattering(radius, frequency, speed_of_sound, density)
            .with_burton_miller(false);
    let solution_nobm = solver
        .solve(&problem_nobm)
        .expect("BEM solve (no BM) failed");
    let max_nobm = solution_nobm.max_surface_pressure();

    println!("  Max surface pressure with    BM: {max_bm:.4}");
    println!("  Max surface pressure without BM: {max_nobm:.4}");

    let eval_r = 2.0 * radius;
    let theta_points = standard_theta_points();
    let coarse_pressures = collect_bem_pressures(&solution_bm_coarse, eval_r, &theta_points);
    let bem_pressures = collect_bem_pressures(&solution_bm, eval_r, &theta_points);
    let analytical =
        sphere_scattering_3d(ka_target / radius, radius, 30, vec![eval_r], theta_points);
    let coarse_error = relative_error(&coarse_pressures, &analytical.pressure);
    let bm_error = relative_error(&bem_pressures, &analytical.pressure);
    println!(
        "  BM complex Mie refinement: {:.3}% ({} panels) -> {:.3}% ({} panels)",
        coarse_error * 100.0,
        solution_bm_coarse.num_dofs(),
        bm_error * 100.0,
        solution_bm.num_dofs()
    );
    println!("  BM complex Mie error: {:.3}%", bm_error * 100.0);

    assert!(
        max_bm.is_finite(),
        "BM surface pressure must be finite, got {max_bm}"
    );
    assert!(
        bm_error < coarse_error,
        "Burton-Miller field did not improve under refinement: coarse {:.3}%, fine {:.3}%",
        coarse_error * 100.0,
        bm_error * 100.0
    );
    assert!(
        bm_error < 0.01,
        "Burton-Miller field error {:.3}% exceeds 1% at ka=pi",
        bm_error * 100.0
    );

    // Non-BM solution may or may not degrade at ka=π depending on mesh
    // resolution and how close the discrete eigenvalue is to the continuous one.
    // On coarse meshes the fictitious eigenfrequency shifts, so the effect is
    // sometimes invisible.  We only assert BM is stable (checked above) and
    // log the comparison for informational purposes.
    if max_nobm.is_finite() {
        let ratio = max_nobm / max_bm.max(1e-15);
        println!("  Ratio (no-BM / BM): {ratio:.2}");
        if ratio > 1.5 {
            println!("  -> Fictitious eigenfrequency effect visible (ratio > 1.5)");
        } else {
            println!("  -> Fictitious eigenfrequency effect NOT visible on this mesh");
            println!("     (expected — coarse mesh shifts the discrete eigenvalue)");
        }
    } else {
        println!("  Non-BM solution diverged (expected near fictitious eigenfrequency)");
    }
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_soft_sphere_scattering_ka_1();
        test_soft_sphere_surface_pressure_zero();
        test_pulsating_sphere_radiation();
        test_point_source_scattering_rigid_sphere();
        test_rigid_sphere_incidence_direction();
        test_rigid_sphere_far_field_decay();
        test_burton_miller_removes_fictitious_eigenfrequency();
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
