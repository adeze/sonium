//! Low-level Sonium reproduction of Wolfram's stationary compressible
//! Neo-Hookean unit-cube example from the Hyperelasticity tutorial.
//!
//! Wolfram fully clamps `x=0` and prescribes only `u_x=0.004 m` at `x=1`.
//! The test assembles those constraints directly because Sonium's higher-level
//! homogeneous-extension study uses a different minimal rigid anchor.

use serde::Deserialize;
use sonium_fem::{
    basis::PolynomialDegree,
    mesh::box_mesh_tetrahedra,
    physics::solid::nonlinear_solid::{HyperelasticForm, NeoHookean},
    solver::{FemNewtonConfig, fem_newton_solve_with_bcs},
    spaces::{FunctionSpace, VectorH1Space},
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::LinearSolverType;
use std::{env, path::PathBuf, process::Command};

const SHEAR_MODULUS_PA: f64 = 5_762.0;
const BULK_MODULUS_PA: f64 = 2_000_000.0;
const LAME_LAMBDA_PA: f64 = BULK_MODULUS_PA - 2.0 * SHEAR_MODULUS_PA / 3.0;
const MASS_DENSITY_KG_M3: f64 = 980.0;
const EXTENSION_M: f64 = 0.004;
const CELLS_PER_AXIS: usize = 4;
const SAMPLE_POINTS_M: [[f64; 3]; 3] = [[0.25, 0.5, 0.5], [0.5, 0.5, 0.5], [0.75, 0.5, 0.5]];

#[derive(Debug)]
struct SoniumResult {
    sample_displacement_m: [[f64; 3]; 3],
    iterations: usize,
    residual: f64,
}

#[derive(Debug, Deserialize)]
struct WolframReference {
    schema_version: u32,
    case_id: String,
    dimensions_m: [f64; 3],
    shear_modulus_pa: f64,
    bulk_modulus_pa: f64,
    mass_density_kg_m3: f64,
    prescribed_extension_m: f64,
    sample_points_m: [[f64; 3]; 3],
    sample_displacement_m: [[f64; 3]; 3],
}

fn sonium_result(load_steps: usize) -> SoniumResult {
    assert!(load_steps > 0);
    let mesh = box_mesh_tetrahedra(
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        CELLS_PER_AXIS,
        CELLS_PER_AXIS,
        CELLS_PER_AXIS,
    );
    let space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let material = NeoHookean::new(SHEAR_MODULUS_PA, LAME_LAMBDA_PA, 3);
    let form = HyperelasticForm::new(material);
    let mut solution = vec![0.0; space.num_global_dofs(&mesh)];
    let mut iterations = 0;
    let mut residual = f64::INFINITY;

    for step in 1..=load_steps {
        let extension = EXTENSION_M * step as f64 / load_steps as f64;
        let mut boundary_values = Vec::new();
        for (node_index, node) in mesh.nodes.iter().enumerate() {
            if node.x.abs() < 1.0e-12 {
                for component in 0..3 {
                    boundary_values.push((node_index * 3 + component, 0.0));
                }
            } else if (node.x - 1.0).abs() < 1.0e-12 {
                boundary_values.push((node_index * 3, extension));
            }
        }
        let mut config = FemNewtonConfig::default();
        // Near-incompressibility makes ILU(0)'s pivot gate overly strict on
        // this small displacement-controlled system. Jacobi-preconditioned
        // GMRES avoids treating that preconditioner limitation as physics.
        config.inner_solver.solver = LinearSolverType::GmresJacobi;
        config.inner_solver.tolerance = 1.0e-9;
        let result =
            fem_newton_solve_with_bcs(&form, &space, &mesh, &boundary_values, &solution, &config)
                .expect("solve Wolfram's fully clamped Neo-Hookean cube");
        assert!(
            result.converged,
            "Newton solve did not converge at load step {step}"
        );
        iterations += result.iterations;
        residual = result.residual;
        solution = result.x;
    }

    let sample_displacement_m = SAMPLE_POINTS_M.map(|point| {
        let (node_index, distance_squared) = mesh
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| {
                let distance_squared = (node.x - point[0]).powi(2)
                    + (node.y - point[1]).powi(2)
                    + (node.z - point[2]).powi(2);
                (index, distance_squared)
            })
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .expect("non-empty cube mesh");
        assert!(
            distance_squared < 1.0e-24,
            "sample point is not a mesh node"
        );
        [
            solution[node_index * 3],
            solution[node_index * 3 + 1],
            solution[node_index * 3 + 2],
        ]
    });
    SoniumResult {
        sample_displacement_m,
        iterations,
        residual,
    }
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/hyperelastic_cube.wls");
    let output = match Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Wolfram Engine not available; running Sonium-only invariants");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram hyperelastic-cube reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram hyperelastic-cube output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

#[test]
fn sonium_reproduces_wolfram_fully_clamped_hyperelastic_cube() {
    let sonium = sonium_result(1);
    let continued = sonium_result(2);
    assert!(sonium.iterations > 0);
    assert!(
        sonium.residual.is_finite(),
        "one-step residual is not finite: {}",
        sonium.residual
    );
    assert!(
        continued.residual.is_finite(),
        "two-step residual is not finite: {}",
        continued.residual
    );
    let axial = sonium.sample_displacement_m.map(|value| value[0]);
    assert!(axial[0] > 0.0 && axial[0] < axial[1]);
    assert!(axial[1] < axial[2] && axial[2] < EXTENSION_M);
    for (point, (displacement, continued_displacement)) in SAMPLE_POINTS_M.iter().zip(
        sonium
            .sample_displacement_m
            .iter()
            .zip(continued.sample_displacement_m.iter()),
    ) {
        assert!(
            displacement[1].abs() < 1.0e-3 && displacement[2].abs() < 1.0e-3,
            "tetrahedral centerline drift is too large: {displacement:?}"
        );
        assert!(
            (displacement[1] - displacement[2]).abs() < 1.0e-10,
            "y/z symmetry is broken: {displacement:?}"
        );
        for component in 0..3 {
            assert!(
                (displacement[component] - continued_displacement[component]).abs() < 5.0e-8,
                "one-step and two-step load paths disagree at {point:?}: one-step={displacement:?}, two-step={continued_displacement:?}"
            );
        }
    }

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(
            reference.case_id,
            "Hyperelasticity-NeoHookean-Stationary-Cube"
        );
        assert_eq!(reference.dimensions_m, [1.0, 1.0, 1.0]);
        assert_eq!(reference.shear_modulus_pa, SHEAR_MODULUS_PA);
        assert_eq!(reference.bulk_modulus_pa, BULK_MODULUS_PA);
        assert_eq!(reference.mass_density_kg_m3, MASS_DENSITY_KG_M3);
        assert_eq!(reference.prescribed_extension_m, EXTENSION_M);
        assert_eq!(reference.sample_points_m, SAMPLE_POINTS_M);
        for (point, (actual, expected)) in SAMPLE_POINTS_M.iter().zip(
            sonium
                .sample_displacement_m
                .iter()
                .zip(&reference.sample_displacement_m),
        ) {
            let axial_error = (actual[0] - expected[0]).abs();
            assert!(
                axial_error < 5.0e-4,
                "axial mismatch at {point:?}: Sonium={actual:?}, Wolfram={expected:?}, error={axial_error:.6e} m"
            );
            assert!(
                (actual[1] - expected[1]).abs() < 1.0e-3
                    && (actual[2] - expected[2]).abs() < 1.0e-3,
                "transverse mesh-drift mismatch at {point:?}: Sonium={actual:?}, Wolfram={expected:?}"
            );
        }
    }
}
