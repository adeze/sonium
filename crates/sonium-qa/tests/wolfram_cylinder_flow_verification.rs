//! Sonium reproduction of Wolfram's stationary 2D cylinder-flow verification
//! (`FluidDynamics-FEM-Stationary-2D-FluidDynamics-0001`).
//!
//! Source: <https://reference.wolfram.com/language/PDEModels/tutorial/FluidDynamics/FluidDynamicsVerificationTests.html?view=all>

use serde::Deserialize;
use sonium_fem::{
    basis::PolynomialDegree,
    boundary::locate_dofs_geometrical,
    mesh::{
        generators::channel_with_cylinder_triangles, interpolation::interpolate_scalar_at_point,
    },
    physics::fluid::stokes::navier_stokes::{
        NavierStokesConfig, compute_drag_lift_variational, solve_picard,
    },
    spaces::{H1Space, VectorH1Space},
};
use sonium_qa::subprocess::WolframCommandExt;
use std::{env, path::PathBuf, process::Command};

const CHANNEL_LENGTH_M: f64 = 2.2;
const CHANNEL_HEIGHT_M: f64 = 0.41;
const CYLINDER_CENTER_M: [f64; 2] = [0.2, 0.2];
const CYLINDER_RADIUS_M: f64 = 0.05;
const MAX_INLET_VELOCITY_M_S: f64 = 0.3;
const MEAN_INLET_VELOCITY_M_S: f64 = 0.2;
const DYNAMIC_VISCOSITY_PA_S: f64 = 0.001;
const MASS_DENSITY_KG_M3: f64 = 1.0;

#[derive(Clone, Copy, Debug)]
struct CylinderMetrics {
    pressure_difference_pa: f64,
    drag_coefficient: f64,
    lift_coefficient: f64,
}

fn sonium_metrics() -> CylinderMetrics {
    const NX: usize = 99;
    const NY: usize = 18;
    let [cylinder_x, cylinder_y] = CYLINDER_CENTER_M;
    let mesh = channel_with_cylinder_triangles(
        0.0,
        CHANNEL_LENGTH_M,
        0.0,
        CHANNEL_HEIGHT_M,
        NX,
        NY,
        cylinder_x,
        cylinder_y,
        CYLINDER_RADIUS_M,
    );
    let velocity_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let tolerance = CHANNEL_LENGTH_M / NX as f64 * 0.6;

    let mut constrained_dofs =
        locate_dofs_geometrical(&mesh, &velocity_space, |x, _, _| x < tolerance);
    constrained_dofs.extend(locate_dofs_geometrical(
        &mesh,
        &velocity_space,
        |_, y, _| y < tolerance || CHANNEL_HEIGHT_M - y < tolerance,
    ));
    constrained_dofs.extend(locate_dofs_geometrical(
        &mesh,
        &velocity_space,
        |x, y, _| {
            let distance = ((x - cylinder_x).powi(2) + (y - cylinder_y).powi(2)).sqrt();
            (distance - CYLINDER_RADIUS_M).abs() < 1.0e-10
        },
    ));
    constrained_dofs.sort_unstable();
    constrained_dofs.dedup();

    let boundary_values = move |x: f64, y: f64, _: f64| {
        if x < tolerance {
            vec![
                4.0 * MAX_INLET_VELOCITY_M_S * y * (CHANNEL_HEIGHT_M - y)
                    / CHANNEL_HEIGHT_M.powi(2),
                0.0,
            ]
        } else {
            vec![0.0, 0.0]
        }
    };
    let result = solve_picard(
        &mesh,
        &velocity_space,
        &pressure_space,
        &NavierStokesConfig {
            mu: DYNAMIC_VISCOSITY_PA_S,
            rho: MASS_DENSITY_KG_M3,
            max_picard_iterations: 100,
            picard_tolerance: 1.0e-6,
            picard_relaxation: 1.0,
            use_direct_solver: false,
            linear_solver_max_iterations: 5_000,
            linear_solver_tolerance: 1.0e-9,
            ..Default::default()
        },
        &boundary_values,
        &constrained_dofs,
        |_| vec![0.0, 0.0],
    );
    assert!(
        result.converged,
        "DFG cylinder solve failed after {} Picard iterations; residual={:.3e}",
        result.iterations, result.residual
    );

    let (drag, lift) = compute_drag_lift_variational(
        &mesh,
        &result.velocity,
        &result.pressure,
        DYNAMIC_VISCOSITY_PA_S,
        MASS_DENSITY_KG_M3,
        4,
        |_| vec![0.0, 0.0],
    );
    let scale =
        2.0 / (MASS_DENSITY_KG_M3 * MEAN_INLET_VELOCITY_M_S.powi(2) * (2.0 * CYLINDER_RADIUS_M));
    let front = interpolate_scalar_at_point(
        &mesh,
        &result.pressure,
        [cylinder_x - CYLINDER_RADIUS_M, cylinder_y, 0.0],
    )
    .expect("front pressure probe should lie on the cylinder boundary");
    let back = interpolate_scalar_at_point(
        &mesh,
        &result.pressure,
        [cylinder_x + CYLINDER_RADIUS_M, cylinder_y, 0.0],
    )
    .expect("back pressure probe should lie on the cylinder boundary");

    CylinderMetrics {
        pressure_difference_pa: front - back,
        drag_coefficient: scale * drag,
        lift_coefficient: scale * lift,
    }
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    channel_dimensions_m: [f64; 2],
    cylinder_center_m: [f64; 2],
    cylinder_radius_m: f64,
    pressure_difference_pa: f64,
    drag_coefficient: f64,
    lift_coefficient: f64,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/cylinder_flow_verification.wls");
    let mut command = Command::new(&executable);
    command.arg("-file").arg(script);
    #[cfg(target_os = "macos")]
    if env::var_os("WolframKernel").is_none() {
        let kernel = PathBuf::from(
            "/Applications/Wolfram Engine.app/Contents/Resources/Wolfram Player.app/Contents/MacOS/WolframKernel",
        );
        if kernel.is_file() {
            command.env("WolframKernel", kernel);
        }
    }
    let output = match command.wolfram_output() {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Wolfram Engine not available; using published cylinder references only");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram cylinder reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram cylinder output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

fn published_reference() -> CylinderMetrics {
    CylinderMetrics {
        pressure_difference_pa: 0.117_520_166_97,
        drag_coefficient: 5.579_535_233_84,
        lift_coefficient: 0.010_618_948_146,
    }
}

fn assert_metrics(actual: CylinderMetrics, reference: CylinderMetrics) {
    let pressure_error = (actual.pressure_difference_pa - reference.pressure_difference_pa).abs()
        / reference.pressure_difference_pa;
    let drag_error =
        (actual.drag_coefficient - reference.drag_coefficient).abs() / reference.drag_coefficient;
    assert!(
        pressure_error < 0.044,
        "pressure-difference error {:.3}% exceeds 4.4%: {actual:?}",
        100.0 * pressure_error
    );
    assert!(
        drag_error < 0.07,
        "drag error {:.3}% exceeds 7%: {actual:?}",
        100.0 * drag_error
    );
    assert!(
        (actual.lift_coefficient - reference.lift_coefficient).abs() < 0.01,
        "lift coefficient differs by more than 0.01: {actual:?}"
    );
}

#[test]
fn sonium_reproduces_wolfram_stationary_cylinder_flow() {
    let sonium = sonium_metrics();
    assert_metrics(sonium, published_reference());
    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(
            reference.channel_dimensions_m,
            [CHANNEL_LENGTH_M, CHANNEL_HEIGHT_M]
        );
        assert_eq!(reference.cylinder_center_m, CYLINDER_CENTER_M);
        assert_eq!(reference.cylinder_radius_m, CYLINDER_RADIUS_M);
        assert_metrics(
            sonium,
            CylinderMetrics {
                pressure_difference_pa: reference.pressure_difference_pa,
                drag_coefficient: reference.drag_coefficient,
                lift_coefficient: reference.lift_coefficient,
            },
        );
    }
}
