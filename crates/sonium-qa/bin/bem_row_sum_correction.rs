//! Finite-frequency row-sum regression.
//!
//! A constant density is a null mode of the hypersingular *Laplace* operator
//! on a closed surface.  It is not a null mode of the finite-frequency
//! Helmholtz Burton--Miller operator.  The historical "correction" forced the
//! full matrix row sums to zero and made otherwise valid systems singular.
//!
//! This QA target protects the replacement contract: the legacy API is a
//! diagnostic-only no-op and cannot change either the matrix or its solution.

use num_complex::Complex64;
use sonium_bem::core::assembly::tbem::{
    apply_row_sum_correction, build_tbem_system, build_tbem_system_corrected,
};
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::types::{BoundaryCondition, PhysicsParams};
use sonium_qa::relative_error;
use std::f64::consts::PI;

// This binary pins the legacy no-op contract, so it is the one sanctioned
// user of the deprecated API.
#[allow(deprecated)]
fn test_row_sum_diagnostic_is_non_mutating() {
    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka = 0.2;
    let frequency = ka * speed_of_sound / (2.0 * PI * radius);
    let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);

    println!("=== Finite-frequency row-sum regression (ka={ka:.2}) ===");

    for subdivisions in [0, 1] {
        let mesh = generate_icosphere_mesh(radius, subdivisions);
        let mut elements = mesh.elements.clone();
        for (i, element) in elements.iter_mut().enumerate() {
            element.boundary_condition =
                BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            element.dof_addresses = vec![i];
        }

        let unmodified = build_tbem_system(&elements, &mesh.nodes, &physics);

        let mut through_legacy_api = build_tbem_system(&elements, &mesh.nodes, &physics);
        let diagnostic = apply_row_sum_correction(&mut through_legacy_api);
        assert!(diagnostic.is_finite(), "row-sum diagnostic is non-finite");
        assert_eq!(
            through_legacy_api.matrix.as_slice(),
            unmodified.matrix.as_slice(),
            "finite-frequency row-sum API changed the system matrix"
        );

        let (through_builder, builder_diagnostic) =
            build_tbem_system_corrected(&elements, &mesh.nodes, &physics);
        assert!(
            builder_diagnostic.is_finite(),
            "builder row-sum diagnostic is non-finite"
        );
        assert_eq!(
            through_builder.matrix.as_slice(),
            unmodified.matrix.as_slice(),
            "diagnostic builder changed the finite-frequency system matrix"
        );

        let incident = IncidentField::plane_wave_z();
        let centers: Vec<[f64; 3]> = elements.iter().map(|element| element.center).collect();
        let normals: Vec<[f64; 3]> = elements.iter().map(|element| element.normal).collect();
        let incident_pressure = incident.evaluate_pressure(&centers, &physics);
        let incident_normal_derivative =
            incident.evaluate_normal_derivative(&centers, &normals, &physics);
        let beta = physics.burton_miller_beta();
        let rhs: Vec<Complex64> = incident_pressure
            .iter()
            .zip(&incident_normal_derivative)
            .map(|(pressure, derivative)| *pressure + beta * *derivative)
            .collect();

        let reference_solution = solve_system(&unmodified.matrix, &rhs);
        let diagnostic_solution = solve_system(&through_builder.matrix, &rhs);
        let solution_delta = relative_error(&diagnostic_solution, &reference_solution);
        assert!(
            solution_delta.is_finite() && solution_delta <= 1e-13,
            "diagnostic path changed the solution: relative delta={solution_delta:.3e}"
        );

        println!(
            "subdiv={subdivisions}: elements={}, diagnostic={diagnostic:.3e}, solution delta={solution_delta:.3e}",
            elements.len()
        );
    }
}

fn solve_system(
    matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    rhs: &[Complex64],
) -> Vec<Complex64> {
    use sonium_bem::core::solver::{GmresConfig, gmres_solve_tbem_with_ilu};

    let result = gmres_solve_tbem_with_ilu(
        matrix,
        rhs,
        &GmresConfig {
            max_iterations: 500,
            restart: 50,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "GMRES+ILU failed to converge");
    assert!(
        result.residual.is_finite(),
        "GMRES+ILU reported a non-finite residual"
    );
    result.x
}

fn main() {
    match std::panic::catch_unwind(test_row_sum_diagnostic_is_non_mutating) {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
