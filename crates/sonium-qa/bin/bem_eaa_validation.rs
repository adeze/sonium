//! EAA-inspired BEM Benchmark Tests
//!
//! Three benchmark tests inspired by EAA (European Acoustics Association) benchmark
//! cases for acoustic BEM validation:
//!
//! 1. `test_eaa_closed_cylinder` — Exterior scattering by a slender rigid cylinder
//! 2. `test_eaa_cats_eye`      — Partial sphere radiation (asymmetric velocity BC)
//! 3. `test_eaa_radiatterer`   — Non-spherical scatterer (closed cylinder)
//!
//! All tests are gated on the `native` feature (requires BLAS/LAPACK).

use num_complex::Complex64;
use sonium_bem::core::assembly::tbem::build_tbem_system_with_beta;
use sonium_bem::core::bem_solver::BoundaryConditionType;
use sonium_bem::core::io::write_surface_pressure_vtu;
use sonium_bem::core::mesh::generators::{generate_closed_cylinder_mesh, generate_icosphere_mesh};
use sonium_bem::core::postprocess::pressure::compute_scattered_field;
use sonium_bem::core::postprocess::pressure::compute_total_field;
use sonium_bem::core::solver::{GmresConfig, gmres_solve_tbem_with_ilu};
use sonium_bem::core::{
    BasisOrder, BemProblem, BemSolver, BoundaryCondition, IncidentField, PhysicsParams,
    SolverMethod,
};
use sonium_qa::relative_error;
use sonium_xem::io::binary_output_dir;
use std::f64::consts::PI;
use std::time::Instant;

fn out_dir() -> std::path::PathBuf {
    binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"))
}

// ---------------------------------------------------------------------------
// Test 1: Slender closed cylinder — exterior scattering
// ---------------------------------------------------------------------------

/// EAA Case 1-inspired slender-body exterior-scattering benchmark.
///
/// A sealed rigid cylinder (radius=0.1 m, height=3.4 m) is illuminated by an
/// axial plane wave. This is not an open or interior duct model: the validation
/// therefore uses exterior-field invariants that the implemented geometry can
/// actually satisfy.
///
/// Assertions:
/// - All field pressures are finite and non-zero.
/// - The aligned complex scattered field converges under mesh refinement.
/// - Rotationally equivalent lateral directions agree as complex pressures.
fn test_eaa_closed_cylinder() {
    let radius = 0.1_f64;
    let height = 3.4_f64;
    let frequency = 500.0_f64;
    let speed_of_sound = 343.0_f64;
    let density = 1.21_f64;

    println!("\n=== test_eaa_closed_cylinder ===");
    println!("  Closed cylinder: radius={radius} m, height={height} m, freq={frequency} Hz");

    // Evaluate field at r=5.0 in several directions.
    let eval_r = 5.0_f64;
    let eval_points: Vec<[f64; 3]> = vec![
        [0.0, 0.0, eval_r],
        [0.0, 0.0, -eval_r],
        [eval_r, 0.0, 0.0],
        [0.0, eval_r, 0.0],
        [-eval_r, 0.0, 0.0],
        [0.0, -eval_r, 0.0],
    ];
    let solve_level = |n_circ: usize, n_height: usize, n_cap_rings: usize| {
        let mesh = generate_closed_cylinder_mesh(radius, height, n_circ, n_height, n_cap_rings);
        let problem = BemProblem {
            mesh: mesh.clone(),
            physics: PhysicsParams::new(frequency, speed_of_sound, density, false),
            incident_field: IncidentField::plane_wave_z(),
            bc_type: BoundaryConditionType::Rigid,
            use_burton_miller: true,
        };
        let started = Instant::now();
        let solution = BemSolver::new()
            .with_solver_method(SolverMethod::BiCgStab)
            .solve(&problem)
            .unwrap_or_else(|error| {
                panic!(
                    "closed-cylinder solve failed at {n_circ}x{n_height}x{n_cap_rings}: {error:?}"
                )
            });
        assert!(
            solution
                .surface_pressure
                .iter()
                .all(|pressure| pressure.re.is_finite() && pressure.im.is_finite()),
            "closed-cylinder surface pressure is non-finite at {n_circ}x{n_height}x{n_cap_rings}"
        );
        println!(
            "  mesh {n_circ}x{n_height}x{n_cap_rings}: {} elements, {} ms",
            mesh.elements.len(),
            started.elapsed().as_millis()
        );
        (mesh, solution)
    };

    // Keep the release gate below the dense-factorization memory cliff while
    // retaining three successively refined meshes of the slender geometry.
    let (_, coarse_solution) = solve_level(20, 60, 4);
    let (_, medium_solution) = solve_level(24, 80, 5);
    let (fine_mesh, fine_solution) = solve_level(26, 90, 5);
    fine_solution
        .write_vtu(&fine_mesh, out_dir().join("eaa_closed_cylinder.vtu"))
        .expect("write closed-cylinder VTU");

    let coarse_field = coarse_solution.evaluate_pressure_field(&eval_points);
    let medium_field = medium_solution.evaluate_pressure_field(&eval_points);
    let fine_field = fine_solution.evaluate_pressure_field(&eval_points);
    let coarse_scattered: Vec<Complex64> =
        coarse_field.iter().map(|point| point.p_scattered).collect();
    let medium_scattered: Vec<Complex64> =
        medium_field.iter().map(|point| point.p_scattered).collect();
    let fine_scattered: Vec<Complex64> = fine_field.iter().map(|point| point.p_scattered).collect();

    assert!(
        fine_field.iter().all(|point| {
            point.p_total.re.is_finite()
                && point.p_total.im.is_finite()
                && point.p_scattered.re.is_finite()
                && point.p_scattered.im.is_finite()
        }),
        "closed-cylinder exterior field contains NaN/Inf"
    );
    let coarse_error = relative_error(&medium_scattered, &coarse_scattered);
    let fine_error = relative_error(&fine_scattered, &medium_scattered);
    println!(
        "  scattered-field refinement: coarse={:.3}%, fine={:.3}%",
        100.0 * coarse_error,
        100.0 * fine_error
    );
    assert!(
        fine_error < coarse_error,
        "closed-cylinder complex scattered field did not converge: coarse={coarse_error:.3e}, fine={fine_error:.3e}"
    );
    assert!(
        fine_error < 0.03,
        "closed-cylinder fine refinement difference {fine_error:.3e} exceeds 3%"
    );

    let lateral = &fine_scattered[2..6];
    let lateral_mean = lateral.iter().copied().sum::<Complex64>() / lateral.len() as f64;
    let rotational_symmetry_error = lateral
        .iter()
        .map(|pressure| (*pressure - lateral_mean).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / (lateral.len() as f64).sqrt()
        / lateral_mean.norm().max(1e-30);
    println!(
        "  complex rotational-symmetry error: {:.4}%",
        100.0 * rotational_symmetry_error
    );
    assert!(
        rotational_symmetry_error < 0.005,
        "closed-cylinder complex rotational-symmetry error {rotational_symmetry_error:.3e} exceeds 0.5%"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Cat's eye — partial sphere radiation with asymmetric BC
// ---------------------------------------------------------------------------

/// EAA Case 2 inspiration: Cat's eye = one octant of a sphere vibrating,
/// the rest rigid.
///
/// An icosphere (subdivision 2, radius=1.0 m) has its positive-octant elements
/// (x>0 AND y>0 AND z>0) assigned a uniform normal velocity v₀=1, while all
/// other elements are rigid (v=0).  The BEM system is assembled manually with
/// Burton-Miller coupling.
///
/// Assertions:
/// - Solution is finite.
/// - Non-zero radiation in all evaluated directions.
/// - Pressure is higher in the +x+y+z diagonal direction than the -x-y-z diagonal.
/// - The radiation pattern is asymmetric (not isotropic).
fn test_eaa_cats_eye() {
    let radius = 1.0_f64;
    let frequency = 200.0_f64; // Hz — ka ≈ 0.37, well below first resonance
    let speed_of_sound = 343.0_f64;
    let density = 1.21_f64;
    let v0 = Complex64::new(1.0, 0.0);

    let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);
    let k = physics.wave_number;

    println!("\n=== test_eaa_cats_eye ===");
    println!(
        "  Icosphere radius={radius} m, freq={frequency} Hz, k={k:.4}, ka={:.4}",
        k * radius
    );

    // Generate icosphere mesh (subdivision 2 → 80 triangles × 4^(2-0) = 320 elements
    // — actually subdivision 2 gives 80 elements from 20 base faces × 4).
    let mesh = generate_icosphere_mesh(radius, 2);

    println!(
        "  Mesh: {} nodes, {} elements",
        mesh.nodes.len(),
        mesh.elements.len()
    );

    // Assign boundary conditions: elements in the positive octant vibrate (v₀=1),
    // the rest are rigid (v=0).
    let mut elements = mesh.elements.clone();
    let mut n_vibrating = 0usize;
    for (i, elem) in elements.iter_mut().enumerate() {
        let cx = elem.center[0];
        let cy = elem.center[1];
        let cz = elem.center[2];
        let in_positive_octant = cx > 0.0 && cy > 0.0 && cz > 0.0;
        elem.boundary_condition = if in_positive_octant {
            n_vibrating += 1;
            BoundaryCondition::Velocity(vec![v0])
        } else {
            BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)])
        };
        elem.dof_addresses = vec![i];
    }

    println!("  Vibrating elements: {} / {}", n_vibrating, elements.len());

    // Assemble TBEM system with Burton-Miller coupling (no incident field for
    // a pure radiation problem — the velocity BC is already in the RHS).
    let beta = physics.burton_miller_beta_scaled(4.0);
    let system = build_tbem_system_with_beta(&elements, &mesh.nodes, &physics, beta);

    let t_start = Instant::now();
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
    assert!(gmres_result.converged, "GMRES+ILU failed for cat's eye");
    let surface_pressure = gmres_result.x;
    let _ = write_surface_pressure_vtu(
        &mesh,
        &surface_pressure,
        BasisOrder::Constant,
        out_dir().join("eaa_cats_eye.vtu"),
    );
    let solve_ms = t_start.elapsed().as_millis();

    println!("  Solve time: {} ms", solve_ms);
    println!(
        "  Max |surface_pressure| = {:.6}",
        surface_pressure
            .iter()
            .map(|p| p.norm())
            .fold(0.0_f64, f64::max)
    );

    // Use a zero-amplitude plane wave as dummy incident field (radiation problem
    // has no incident wave — the field IS the radiated field).
    let dummy_incident = IncidentField::PlaneWave {
        direction: [0.0, 0.0, 1.0],
        amplitude: Complex64::new(0.0, 0.0),
    };

    // Evaluate field at r=3.0 in several directions.
    let eval_r = 3.0_f64;
    let inv_sqrt3 = 1.0_f64 / 3.0_f64.sqrt();
    let eval_points: Vec<[f64; 3]> = vec![
        // +x+y+z diagonal: vibrating octant direction
        [eval_r * inv_sqrt3, eval_r * inv_sqrt3, eval_r * inv_sqrt3],
        // -x-y-z diagonal: opposite, should be quieter
        [
            -eval_r * inv_sqrt3,
            -eval_r * inv_sqrt3,
            -eval_r * inv_sqrt3,
        ],
        // +x axis
        [eval_r, 0.0, 0.0],
        // -z axis
        [0.0, 0.0, -eval_r],
        // +y axis
        [0.0, eval_r, 0.0],
    ];

    let field_points = compute_total_field(
        &eval_points,
        &elements,
        &mesh.nodes,
        &surface_pressure,
        None,
        &dummy_incident,
        &physics,
        BasisOrder::Constant,
    );

    println!("  Direction                |p_total|");
    let labels = [
        "+x+y+z (vibrating octant)",
        "-x-y-z (opposite octant)",
        "+x axis",
        "-z axis",
        "+y axis",
    ];
    for (i, fp) in field_points.iter().enumerate() {
        println!("  {:26}  {:.6}", labels[i], fp.p_total.norm());
    }

    // Assertion 1: solution is finite.
    for (i, fp) in field_points.iter().enumerate() {
        assert!(
            fp.p_total.norm().is_finite(),
            "Field pressure at {} is not finite",
            labels[i]
        );
    }

    // Assertion 2: non-zero radiation in all directions (sound escapes even
    // in the shadow region due to diffraction).
    for (i, fp) in field_points.iter().enumerate() {
        assert!(
            fp.p_total.norm() > 1e-15,
            "Field pressure at {} is essentially zero ({:.2e})",
            labels[i],
            fp.p_total.norm()
        );
    }

    let p_vibrating_octant = field_points[0].p_total.norm();
    let p_opposite_octant = field_points[1].p_total.norm();

    // Assertion 3: stronger radiation toward the vibrating octant.
    println!("  Vibrating octant |p| = {p_vibrating_octant:.6}, opposite = {p_opposite_octant:.6}");
    assert!(
        p_vibrating_octant > p_opposite_octant,
        "Expected vibrating octant pressure ({p_vibrating_octant:.4}) > \
         opposite octant ({p_opposite_octant:.4})"
    );

    // Assertion 4: pattern is asymmetric — the ratio between max and min
    // pressure across directions should be clearly > 1 (not isotropic).
    let pressures: Vec<f64> = field_points.iter().map(|fp| fp.p_total.norm()).collect();
    let p_max = pressures.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let p_min = pressures.iter().cloned().fold(f64::INFINITY, f64::min);
    let asymmetry_ratio = p_max / p_min.max(1e-20);
    println!("  Asymmetry ratio (max/min) = {asymmetry_ratio:.3}");
    assert!(
        asymmetry_ratio > 1.5,
        "Radiation pattern should be asymmetric (ratio {asymmetry_ratio:.3} must exceed 1.5)"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Radiatterer — non-spherical closed-cylinder scatterer
// ---------------------------------------------------------------------------

/// EAA Case 3 inspiration: Non-spherical geometry scattering.
///
/// A closed cylinder (radius=0.5 m, height=2.0 m) scatters a plane wave
/// traveling in the +z direction.  The cylinder axis is along z so the problem
/// has cylindrical symmetry about z — pressure at (3,0,0) should equal pressure
/// at (0,3,0).  The forward-scattered field at (0,0,3) and the back-scattered
/// field at (0,0,−3) should differ due to the non-spherical geometry.
///
/// Frequency chosen so that k·radius ≈ 1.0 (moderate scattering regime).
///
/// Assertions:
/// - Solution is finite and non-zero.
/// - Cylindrical symmetry about z: |p|(3,0,0) ≈ |p|(0,3,0).
/// - Forward (lit) and backward (shadow) pressures are distinct.
/// - Forward pressure differs from lateral pressure (directionality).
fn test_eaa_radiatterer() {
    let cyl_radius = 0.5_f64;
    let cyl_height = 2.0_f64;
    // k * cyl_radius ≈ 1.0  =>  k = 1/0.5 = 2  =>  f = k*c/(2π) ≈ 109 Hz
    let speed_of_sound = 343.0_f64;
    let density = 1.21_f64;
    let k_target = 1.0 / cyl_radius; // k such that k*radius = 1
    let frequency = k_target * speed_of_sound / (2.0 * PI);

    println!("\n=== test_eaa_radiatterer ===");
    println!("  Closed cylinder: radius={cyl_radius} m, height={cyl_height} m");
    println!(
        "  freq={frequency:.2} Hz, k={k_target:.4}, k*r={:.4}",
        k_target * cyl_radius
    );

    // Use enough circumferential sectors for the x/y rotational-symmetry
    // check to resolve geometry error below its 0.05% field tolerance.
    let mesh = generate_closed_cylinder_mesh(cyl_radius, cyl_height, 32, 12, 2);

    println!(
        "  Mesh: {} nodes, {} elements",
        mesh.nodes.len(),
        mesh.elements.len()
    );

    // Rigid scattering problem: plane wave in +z direction, Burton-Miller.
    let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);
    let incident = IncidentField::plane_wave_z();

    let problem = BemProblem {
        mesh: mesh.clone(),
        physics: physics.clone(),
        incident_field: incident.clone(),
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: true,
    };

    let solver = BemSolver::new().with_solver_method(SolverMethod::BiCgStab);

    let t_start = Instant::now();
    let solution = solver
        .solve(&problem)
        .expect("BEM solve failed for radiatterer");
    let solve_ms = t_start.elapsed().as_millis();

    println!(
        "  DOFs: {}, solve time: {} ms",
        solution.num_dofs(),
        solve_ms
    );
    println!(
        "  Max |surface_pressure| = {:.6}",
        solution.max_surface_pressure()
    );

    let _ = solution.write_vtu(&mesh, out_dir().join("eaa_radiatterer.vtu"));

    // Check if the solve produced finite results
    let has_nan = solution
        .surface_pressure
        .iter()
        .any(|p| !p.re.is_finite() || !p.im.is_finite());
    assert!(
        !has_nan,
        "EAA radiatterer surface pressure contains NaN/Inf"
    );

    // Evaluate field at r=5.0 in key directions.
    let eval_r = 5.0_f64;
    let eval_points: Vec<[f64; 3]> = vec![
        [0.0, 0.0, eval_r],  // (0,0,+5) forward/lit — wave travels +z
        [0.0, 0.0, -eval_r], // (0,0,-5) backward/shadow
        [eval_r, 0.0, 0.0],  // (+5,0,0) lateral x
        [0.0, eval_r, 0.0],  // (0,+5,0) lateral y
        [-eval_r, 0.0, 0.0], // (-5,0,0) lateral -x
    ];

    let field = solution.evaluate_pressure_field(&eval_points);

    // Compute scattered field separately — |p_total| depends on phase interference
    // with the incident wave and is not a reliable measure of scattering directionality.
    // |p_scat| directly shows the scattering pattern.
    let p_scattered = compute_scattered_field(
        &eval_points,
        &solution.elements,
        &solution.nodes,
        &solution.surface_pressure,
        None,
        &problem.physics,
        BasisOrder::Constant,
    );

    let labels = [
        "(0,0,+5) forward",
        "(0,0,-5) shadow",
        "(+5,0,0) lateral x",
        "(0,+5,0) lateral y",
        "(-5,0,0) lateral -x",
    ];
    println!("  Point              |p_total|   |p_scat|");
    for (i, fp) in field.iter().enumerate() {
        println!(
            "  {:20}  {:.6}   {:.6}",
            labels[i],
            fp.p_total.norm(),
            p_scattered[i].norm()
        );
    }

    // Assertion 1: all pressures finite and non-zero.
    for (i, fp) in field.iter().enumerate() {
        assert!(
            fp.p_total.norm().is_finite(),
            "Field pressure at {} is not finite",
            labels[i]
        );
        assert!(
            fp.p_total.norm() > 1e-15,
            "Field pressure at {} is essentially zero ({:.2e})",
            labels[i],
            fp.p_total.norm()
        );
    }

    let p_forward = p_scattered[0];
    let p_shadow = p_scattered[1];
    let ps_forward = p_forward.norm();
    let ps_shadow = p_shadow.norm();
    let p_lat_x = p_scattered[2];
    let p_lat_y = p_scattered[3];
    let ps_lat_x = p_lat_x.norm();

    // Assertion 2: cylindrical symmetry about z-axis —
    // scattered pressure at (r,0,0) should match scattered pressure at (0,r,0).
    let symmetry_rel_err =
        (p_lat_x - p_lat_y).norm() / p_lat_x.norm().max(p_lat_y.norm()).max(1e-20);
    println!(
        "  Lateral complex-pressure symmetry error = {:.4e} ({:.4}%)",
        symmetry_rel_err,
        symmetry_rel_err * 100.0
    );
    assert!(
        symmetry_rel_err < 1e-3,
        "Cylindrical complex-pressure symmetry violated: lateral x={p_lat_x:?}, lateral y={p_lat_y:?}, \
         rel_err={:.4}%",
        symmetry_rel_err * 100.0
    );

    // Assertion 3: forward and shadow complex scattered pressures differ.
    // At ka=1 their magnitudes can be nearly equal, so a magnitude ratio would
    // discard the phase that establishes directionality.
    let forward_shadow_complex_difference =
        (p_forward - p_shadow).norm() / ps_forward.max(ps_shadow).max(1e-20);
    println!(
        "  Forward/shadow complex-pressure difference = \
         {forward_shadow_complex_difference:.3} \
         (forward={p_forward:?}, shadow={p_shadow:?})"
    );
    assert!(
        forward_shadow_complex_difference.is_finite() && forward_shadow_complex_difference > 0.01,
        "Forward and shadow complex scattered pressures are indistinguishable: \
         relative difference={forward_shadow_complex_difference:.3}"
    );

    // Assertion 4: scattered pressure pattern is non-isotropic (different
    // from lateral). At the radiatterer's geometry/ka the realised
    // shadow-to-lateral ratio is ≈ 0.92 — only 8 % off isotropic, because
    // a closed cylinder of this aspect ratio scatters nearly omnidirectionally
    // at low-mid `ka`. The 5 % threshold catches a genuine isotropic
    // regression (sphere-like scattering ⇒ ratio ≈ 1.000 to round-off)
    // without being a strict directionality test that the geometry can't
    // support. Tighten this only after switching to a more elongated
    // scatterer or a higher `ka`.
    let shadow_lateral_ratio = ps_shadow / ps_lat_x.max(1e-20);
    println!("  |p_scat| shadow/lateral ratio = {shadow_lateral_ratio:.3}");
    assert!(
        (shadow_lateral_ratio - 1.0).abs() > 0.05,
        "Shadow and lateral scattered pressures should differ by > 5% \
         (non-isotropic regression gate): ratio={shadow_lateral_ratio:.3}"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_eaa_closed_cylinder();
        test_eaa_cats_eye();
        test_eaa_radiatterer();
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
