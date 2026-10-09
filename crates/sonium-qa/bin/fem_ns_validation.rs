//! Navier-Stokes helper and manufactured-solution validation.
//!
//! Dedicated binaries cover Poiseuille flow, lid-driven cavities, 3-D channel
//! flow, obstacles, and DFG cylinder coefficients. This target validates the
//! shared post-processing/time-stepping helpers and requires mesh convergence
//! for a steady Taylor-Green manufactured solution.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::physics::fluid::stokes::navier_stokes::{
    NavierStokesConfig, compute_vorticity_2d, solve_picard_with_velocity_force,
};
use sonium_fem::spaces::{H1Space, VectorH1Space};

fn test_vorticity_linear_fields_are_exact() {
    let mesh = unit_square_triangles(8);
    let mut velocity = vec![0.0; 2 * mesh.num_nodes()];

    // Rigid rotation u=(-y,x) has omega=dv/dx-du/dy=2.
    for (node, point) in mesh.nodes.iter().enumerate() {
        velocity[2 * node] = -point.y;
        velocity[2 * node + 1] = point.x;
    }
    for (element, value) in compute_vorticity_2d(&mesh, &velocity).iter().enumerate() {
        assert!(
            (value - 2.0).abs() < 1.0e-12,
            "rigid-rotation vorticity on element {element} is {value}, expected 2"
        );
    }

    // Simple shear u=(y,0) has omega=-1.
    for (node, point) in mesh.nodes.iter().enumerate() {
        velocity[2 * node] = point.y;
        velocity[2 * node + 1] = 0.0;
    }
    for (element, value) in compute_vorticity_2d(&mesh, &velocity).iter().enumerate() {
        assert!(
            (value + 1.0).abs() < 1.0e-12,
            "shear vorticity on element {element} is {value}, expected -1"
        );
    }
}

fn test_stabilisation_tau_scaling() {
    use sonium_fem::physics::fluid::stokes::stabilisation::stabilisation_tau;

    let tau_convection = stabilisation_tau(10.0, 0.1, 1.0e-8, None);
    let expected_convection = 0.1 / (2.0 * 10.0);
    assert!(
        (tau_convection - expected_convection).abs() / expected_convection < 1.0e-10,
        "convection-dominated tau={tau_convection}, expected {expected_convection}"
    );

    let tau_diffusion = stabilisation_tau(0.0, 0.1, 1.0, None);
    let expected_diffusion = 0.1_f64.powi(2) / 6.0;
    assert!(
        (tau_diffusion - expected_diffusion).abs() / expected_diffusion < 1.0e-12,
        "diffusion-dominated tau={tau_diffusion}, expected {expected_diffusion}"
    );

    let tau_transient = stabilisation_tau(1.0, 0.1, 0.01, Some(0.001));
    let tau_steady = stabilisation_tau(1.0, 0.1, 0.01, None);
    assert!(tau_transient < tau_steady);
}

fn test_bdf_helpers() {
    use sonium_fem::physics::fluid::stokes::time_stepping::{
        BdfCoefficients, BdfState, extrapolate_velocity,
    };

    for coefficients in [BdfCoefficients::bdf1(), BdfCoefficients::bdf2()] {
        assert!(
            (coefficients.a0 + coefficients.a1 + coefficients.a2).abs() < 1.0e-14,
            "BDF derivative must annihilate a constant field"
        );
    }

    let state = BdfState {
        velocity: vec![3.0, 4.0],
        velocity_prev: vec![3.0, 4.0],
        pressure: vec![],
        time: 0.0,
        step: 1,
    };
    assert_eq!(extrapolate_velocity(&state, 2), vec![3.0, 4.0]);
}

fn test_taylor_green_manufactured_solution_converges() {
    let rho = 1.0;
    let mu = 0.1;
    let pi = std::f64::consts::PI;
    let pi_squared = pi * pi;
    let exact_u = |x: f64, y: f64| (pi * x).sin() * (pi * y).cos();
    let exact_v = |x: f64, y: f64| -(pi * x).cos() * (pi * y).sin();
    let mut errors = Vec::new();

    for subdivisions in [4, 8] {
        let mesh = unit_square_triangles(subdivisions);
        let velocity_space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let tolerance = 1.0e-12;
        let boundary_dofs = locate_dofs_geometrical(&mesh, &velocity_space, |x, y, _| {
            x.abs() < tolerance
                || (x - 1.0).abs() < tolerance
                || y.abs() < tolerance
                || (y - 1.0).abs() < tolerance
        });
        let config = NavierStokesConfig {
            mu,
            rho,
            max_picard_iterations: 30,
            picard_tolerance: 1.0e-9,
            ..Default::default()
        };
        let boundary = |x: f64, y: f64, _z: f64| vec![exact_u(x, y), exact_v(x, y)];
        let force = move |position: [f64; 3], _velocity: [f64; 3]| {
            let x = position[0];
            let y = position[1];
            vec![
                2.0 * mu * pi_squared * (pi * x).sin() * (pi * y).cos(),
                -2.0 * mu * pi_squared * (pi * x).cos() * (pi * y).sin(),
            ]
        };
        let result = solve_picard_with_velocity_force(
            &mesh,
            &velocity_space,
            &pressure_space,
            &config,
            &boundary,
            &boundary_dofs,
            force,
        );
        assert!(
            result.converged,
            "Taylor-Green solve on {subdivisions}x{subdivisions} mesh did not converge: iterations={}, residual={:.3e}",
            result.iterations, result.residual
        );

        let mut squared_error = 0.0;
        let mut squared_norm = 0.0;
        for (node, point) in mesh.nodes.iter().enumerate() {
            let expected = [exact_u(point.x, point.y), exact_v(point.x, point.y)];
            for component in 0..2 {
                squared_error +=
                    (result.velocity[2 * node + component] - expected[component]).powi(2);
                squared_norm += expected[component].powi(2);
            }
        }
        let relative_error = (squared_error / squared_norm).sqrt();
        println!("Taylor-Green n={subdivisions}: relative velocity error={relative_error:.4e}");
        assert!(relative_error.is_finite());
        errors.push(relative_error);
    }

    let observed_order = (errors[0] / errors[1]).ln() / 2.0_f64.ln();
    assert!(
        observed_order >= 1.5,
        "Taylor-Green observed order {observed_order:.3} is below 1.5; errors={errors:?}"
    );
    assert!(
        errors[1] <= 0.02,
        "Taylor-Green finest velocity error {:.2}% exceeds 2%",
        100.0 * errors[1]
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_vorticity_linear_fields_are_exact();
        test_stabilisation_tau_scaling();
        test_bdf_helpers();
        test_taylor_green_manufactured_solution_converges();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
