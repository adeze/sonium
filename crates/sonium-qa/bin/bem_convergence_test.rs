//! Surface-field convergence against the rigid-sphere Mie solution.
//!
//! The old check compared an unweighted mean over every BEM element with the
//! mean of only three Mie directions.  Those are different quantities, so a
//! 5--10% plateau could still report success.  This check aligns one complex
//! Mie value with every collocation element and integrates the relative L2
//! error with the triangle areas.

use num_complex::Complex64;
use sonium_bem::analytical::sphere_scattering_3d;
use sonium_bem::core::bem_solver::{BemProblem, BemSolver, BoundaryConditionType, SolverMethod};
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::types::{Element, PhysicsParams};
use sonium_qa::validation::convergence_rate_with_sizes;
use sonium_qa::weighted_relative_l2_error;
use std::f64::consts::PI;

struct ConvergenceResult {
    errors: Vec<f64>,
    element_sizes: Vec<f64>,
}

fn mie_at_element_center(k: f64, radius: f64, element: &Element) -> Complex64 {
    let [x, y, z] = element.center;
    let r = x.hypot(y).hypot(z);
    assert!(r.is_finite() && r > 0.0, "invalid element center radius");
    let theta = (z / r).clamp(-1.0, 1.0).acos();
    sphere_scattering_3d(k, radius, 50, vec![radius], vec![theta]).pressure[0]
}

fn run_convergence_study(
    ka: f64,
    radius: f64,
    speed_of_sound: f64,
    density: f64,
) -> ConvergenceResult {
    let frequency = ka * speed_of_sound / (2.0 * PI * radius);
    let k = 2.0 * PI * frequency / speed_of_sound;
    let mut errors = Vec::new();
    let mut element_sizes = Vec::new();

    println!("ka = {ka:.2}");
    println!("{:>6} | {:>12} | {:>12}", "Elems", "weighted L2", "h");
    println!("{}", "-".repeat(38));

    for subdivisions in [1, 2, 3] {
        let mesh = generate_icosphere_mesh(radius, subdivisions);
        let n = mesh.elements.len();
        let problem = BemProblem {
            mesh,
            physics: PhysicsParams::new(frequency, speed_of_sound, density, false),
            incident_field: IncidentField::plane_wave_z(),
            bc_type: BoundaryConditionType::Rigid,
            use_burton_miller: true,
        };

        let solution = BemSolver::new()
            .with_solver_method(SolverMethod::BiCgStab)
            .with_tolerance(1e-8)
            .solve(&problem)
            .unwrap_or_else(|error| panic!("BEM solve failed (n={n}): {error}"));

        assert_eq!(solution.surface_pressure.len(), solution.elements.len());
        let reference: Vec<Complex64> = solution
            .elements
            .iter()
            .map(|element| mie_at_element_center(k, radius, element))
            .collect();
        let weights: Vec<f64> = solution
            .elements
            .iter()
            .map(|element| element.area)
            .collect();
        let error = weighted_relative_l2_error(&solution.surface_pressure, &reference, &weights);
        assert!(error.is_finite(), "non-finite weighted surface error");

        let surface_area: f64 = weights.iter().sum();
        let h = (surface_area / n as f64).sqrt();
        assert!(h.is_finite() && h > 0.0, "invalid characteristic mesh size");

        println!("{n:>6} | {error:>12.5e} | {h:>12.5e}");
        errors.push(error);
        element_sizes.push(h);
    }

    ConvergenceResult {
        errors,
        element_sizes,
    }
}

fn validate_convergence(ka: f64, result: &ConvergenceResult) {
    let rate = convergence_rate_with_sizes(&result.errors, &result.element_sizes)
        .unwrap_or_else(|| panic!("ka={ka}: error is non-finite or not monotonically decreasing"));
    let final_error = *result.errors.last().expect("at least one refinement");

    println!("observed mean order: {rate:.3}");
    println!("finest weighted L2 error: {:.3}%", 100.0 * final_error);

    // Piecewise-constant collocation approximates the boundary density with
    // first-order accuracy.  The 1% endpoint is deliberately strict enough
    // to catch a frequency-dependent formulation-sign flip: the canonical
    // exterior sign reaches this bound on the same meshes, while the flipped
    // sign plateaus near 4.4% at low ka.
    assert!(
        rate >= 0.5,
        "ka={ka}: observed convergence order {rate:.3} is below 0.5"
    );
    assert!(
        final_error <= 0.01,
        "ka={ka}: finest surface L2 error {:.3}% exceeds 1%",
        100.0 * final_error
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        let radius = 0.1;
        let speed_of_sound = 343.0;
        let density = 1.21;

        println!("=== Area-weighted complex BEM surface convergence ===\n");
        for ka in [0.5, 1.0] {
            let convergence = run_convergence_study(ka, radius, speed_of_sound, density);
            validate_convergence(ka, &convergence);
            println!();
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
