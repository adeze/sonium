//! Sonium counterpart of Wolfram's published spherical-capacitor example.
//!
//! Source:
//! <https://reference.wolfram.com/language/PDEModels/tutorial/Electromagnetics/ModelCollection/SphericalCapacitor.html?view=all>
//!
//! The test checks potential and exercises Sonium's public electrostatic
//! solved-field API for electric field, energy, capacitance, and charge.

use serde::Deserialize;
use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_dirichlet_values},
    forms::WeightedLaplacianForm,
    mesh::generators::spherical_shell_mesh_tetrahedra,
    physics::electromagnetics::electrostatic::ElectrostaticSolvedField,
    spaces::{FunctionSpace, H1Space},
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{GmresConfig, gmres};
use std::{env, path::PathBuf, process::Command};

const INNER_RADIUS_M: f64 = 0.01;
const OUTER_RADIUS_M: f64 = 0.05;
const INNER_POTENTIAL_V: f64 = 100.0;
const VACUUM_PERMITTIVITY_F_M: f64 = 8.854_187_812_8e-12;
const SAMPLE_RADII_M: [f64; 9] = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05];

fn exact_potential(radius: f64) -> f64 {
    INNER_POTENTIAL_V * INNER_RADIUS_M / (OUTER_RADIUS_M - INNER_RADIUS_M)
        * (OUTER_RADIUS_M / radius - 1.0)
}

fn sonium_potential() -> Vec<f64> {
    // The generator places the same icosphere vertices on each radial layer,
    // so vertex zero supplies an unambiguous sample at every requested radius.
    let mesh = spherical_shell_mesh_tetrahedra(
        0.0,
        0.0,
        0.0,
        INNER_RADIUS_M,
        OUTER_RADIUS_M,
        2,
        SAMPLE_RADII_M.len() - 1,
    );
    let space = H1Space::new(PolynomialDegree::P1);
    let stiffness = assemble_bilinear(
        &WeightedLaplacianForm { coefficient: 1.0 },
        &space,
        &space,
        &mesh,
        1,
    );
    let mut system = FemSystem::new(stiffness, vec![0.0; space.num_global_dofs(&mesh)]);

    apply_dirichlet_values(
        &mut system,
        &mesh,
        &space,
        1,
        &|_, _, _| INNER_POTENTIAL_V,
        None,
    );
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| 0.0, None);

    let result = gmres(
        &system.matrix.to_csr(),
        &system.rhs,
        &GmresConfig {
            max_iterations: 3_000,
            restart: 50,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(
        result.converged,
        "spherical-capacitor Laplace solve did not converge"
    );

    let nodes_per_layer = mesh.nodes.len() / SAMPLE_RADII_M.len();
    (0..SAMPLE_RADII_M.len())
        .map(|layer| result.x[layer * nodes_per_layer])
        .collect()
}

struct CapacitorMetrics {
    radial_field_v_m: Vec<f64>,
    capacitance_f: f64,
    inner_charge_c: f64,
    energy_j: f64,
}

fn exact_radial_field(radius: f64) -> f64 {
    INNER_POTENTIAL_V * INNER_RADIUS_M * OUTER_RADIUS_M
        / ((OUTER_RADIUS_M - INNER_RADIUS_M) * radius * radius)
}

fn exact_capacitance() -> f64 {
    4.0 * std::f64::consts::PI * VACUUM_PERMITTIVITY_F_M * INNER_RADIUS_M * OUTER_RADIUS_M
        / (OUTER_RADIUS_M - INNER_RADIUS_M)
}

fn sonium_capacitor_metrics() -> CapacitorMetrics {
    const RADIAL_LAYERS: usize = 16;
    let mesh = spherical_shell_mesh_tetrahedra(
        0.0,
        0.0,
        0.0,
        INNER_RADIUS_M,
        OUTER_RADIUS_M,
        2,
        RADIAL_LAYERS,
    );
    let space = H1Space::new(PolynomialDegree::P1);
    let stiffness = assemble_bilinear(
        &WeightedLaplacianForm { coefficient: 1.0 },
        &space,
        &space,
        &mesh,
        1,
    );
    let energy_matrix = stiffness.to_csr();
    let mut system = FemSystem::new(stiffness, vec![0.0; space.num_global_dofs(&mesh)]);
    apply_dirichlet_values(
        &mut system,
        &mesh,
        &space,
        1,
        &|_, _, _| INNER_POTENTIAL_V,
        None,
    );
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| 0.0, None);
    let result = gmres(
        &system.matrix.to_csr(),
        &system.rhs,
        &GmresConfig {
            max_iterations: 1_500,
            restart: 50,
            tolerance: 1.0e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "spherical-capacitor metric solve failed");

    let radial_step = (OUTER_RADIUS_M - INNER_RADIUS_M) / RADIAL_LAYERS as f64;
    let radial_field_v_m = (1..SAMPLE_RADII_M.len() - 1)
        .map(|sample| {
            let radius = SAMPLE_RADII_M[sample];
            let solved =
                ElectrostaticSolvedField::new(&mesh, &space, &result.x, VACUUM_PERMITTIVITY_F_M)
                    .expect("valid spherical-capacitor solved field");
            let fields: Vec<_> = mesh
                .elements
                .iter()
                .enumerate()
                .filter_map(|(element_index, element)| {
                    let centroid = element.nodes.iter().fold([0.0; 3], |mut sum, &node| {
                        sum[0] += mesh.nodes[node].x / element.nodes.len() as f64;
                        sum[1] += mesh.nodes[node].y / element.nodes.len() as f64;
                        sum[2] += mesh.nodes[node].z / element.nodes.len() as f64;
                        sum
                    });
                    let centroid_radius = centroid[0].hypot(centroid[1]).hypot(centroid[2]);
                    ((centroid_radius - radius).abs() < 0.65 * radial_step).then(|| {
                        let field = solved
                            .electric_field_at_reference(element_index, 0.25, 0.25, 0.25)
                            .expect("valid tetrahedral field sample");
                        (field[0] * centroid[0] + field[1] * centroid[1] + field[2] * centroid[2])
                            / centroid_radius
                    })
                })
                .collect();
            assert!(!fields.is_empty(), "no field samples near radius {radius}");
            fields.iter().sum::<f64>() / fields.len() as f64
        })
        .collect();

    let solved = ElectrostaticSolvedField::new(&mesh, &space, &result.x, VACUUM_PERMITTIVITY_F_M)
        .expect("valid spherical-capacitor solved field");
    let observables = solved
        .capacitor_observables(&energy_matrix, INNER_POTENTIAL_V)
        .expect("valid spherical-capacitor observables");

    CapacitorMetrics {
        radial_field_v_m,
        capacitance_f: observables.capacitance_f,
        inner_charge_c: observables.charge_c,
        energy_j: observables.energy_j,
    }
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    inner_radius_m: f64,
    outer_radius_m: f64,
    inner_potential_v: f64,
    sample_radii_m: Vec<f64>,
    potential_v: Vec<f64>,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/spherical_capacitor.wls");
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
            eprintln!("Wolfram Engine not available; using analytical capacitor reference only");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram spherical-capacitor example failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram capacitor output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

fn assert_potential_close(name: &str, actual: &[f64], reference: &[f64]) {
    assert_eq!(actual.len(), reference.len());
    let max_error = actual
        .iter()
        .zip(reference)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_error < 0.75,
        "{name} maximum potential error {max_error:.6} V exceeds 0.75 V"
    );
}

#[test]
fn sonium_reproduces_wolfram_spherical_capacitor_potential() {
    let sonium = sonium_potential();
    let exact = SAMPLE_RADII_M.map(exact_potential);
    assert_potential_close("Sonium/analytical", &sonium, &exact);

    if let Some(reference) = wolfram_reference() {
        // v2 extended the payload with field/capacitance/charge/energy
        // references; this test consumes the v1-compatible potential subset.
        assert_eq!(reference.schema_version, 2);
        assert_eq!(reference.inner_radius_m, INNER_RADIUS_M);
        assert_eq!(reference.outer_radius_m, OUTER_RADIUS_M);
        assert_eq!(reference.inner_potential_v, INNER_POTENTIAL_V);
        assert_eq!(reference.sample_radii_m, SAMPLE_RADII_M);
        assert_potential_close("Wolfram/analytical", &reference.potential_v, &exact);
        assert_potential_close("Sonium/Wolfram", &sonium, &reference.potential_v);
    }
}

#[test]
fn sonium_spherical_capacitor_recovers_field_charge_capacitance_and_energy() {
    let metrics = sonium_capacitor_metrics();
    let reference_field: Vec<_> = SAMPLE_RADII_M[1..SAMPLE_RADII_M.len() - 1]
        .iter()
        .map(|&radius| exact_radial_field(radius))
        .collect();
    let field_relative_l2 = metrics
        .radial_field_v_m
        .iter()
        .zip(&reference_field)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f64>()
        .sqrt()
        / reference_field
            .iter()
            .map(|value| value.powi(2))
            .sum::<f64>()
            .sqrt();
    let capacitance_reference = exact_capacitance();
    let capacitance_relative_error =
        (metrics.capacitance_f - capacitance_reference).abs() / capacitance_reference;
    let charge_reference = capacitance_reference * INNER_POTENTIAL_V;
    let energy_reference = 0.5 * capacitance_reference * INNER_POTENTIAL_V.powi(2);

    eprintln!(
        "Spherical capacitor: field_l2={field_relative_l2:.3e}, C={:.8e} F, C_err={capacitance_relative_error:.3e}",
        metrics.capacitance_f
    );
    assert!(field_relative_l2 < 0.03);
    assert!(capacitance_relative_error < 0.01);
    assert!((metrics.inner_charge_c - charge_reference).abs() / charge_reference < 0.01);
    assert!((metrics.energy_j - energy_reference).abs() / energy_reference < 0.01);
}
