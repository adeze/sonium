//! Sonium reproduction of Wolfram verification case
//! `HeatTransfer-FEM-Stationary-2DAxisym-Single-HeatTransfer-0001`.
//!
//! A hollow cylinder (`1 <= r <= 2`, `0 <= z <= 0.5`) has a 100 W/m² inward
//! heat flux on its inner wall, a fixed 10 K outer wall, and insulated ends.
//! The exact field is `T(r)=10(1-log(r/2))`.

use serde::Deserialize;
use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_dirichlet_values},
    mesh::{Mesh, rectangular_mesh_triangles},
    physics::solid::axisymmetric::AxisymmetricScalarGradientForm,
    spaces::H1Space,
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{GmresConfig, gmres};
use std::{env, f64::consts::PI, path::PathBuf, process::Command};

const INNER_RADIUS_M: f64 = 1.0;
const OUTER_RADIUS_M: f64 = 2.0;
const HEIGHT_M: f64 = 0.5;
const CONDUCTIVITY_W_M_K: f64 = 10.0;
const INNER_HEAT_FLUX_W_M2: f64 = 100.0;
const OUTER_TEMPERATURE_K: f64 = 10.0;
const RADIAL_CELLS: usize = 96;
const AXIAL_CELLS: usize = 10;
const SAMPLE_RADII_M: [f64; 5] = [1.0, 1.25, 1.5, 1.75, 2.0];

fn exact_temperature(radius: f64) -> f64 {
    10.0 * (1.0 - (radius / OUTER_RADIUS_M).ln())
}

fn assemble_inner_flux(mesh: &Mesh, rhs: &mut [f64]) -> f64 {
    let mut total_power = 0.0;
    for boundary in mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == 1)
    {
        assert_eq!(boundary.nodes.len(), 2);
        let first = mesh.nodes[boundary.nodes[0]];
        let second = mesh.nodes[boundary.nodes[1]];
        let length = ((second.x - first.x).powi(2) + (second.y - first.y).powi(2)).sqrt();
        // Both nodes lie on the constant-radius inner wall, so two-point P1
        // boundary integration is exact for q * 2*pi*r * N_i.
        let segment_power = INNER_HEAT_FLUX_W_M2 * 2.0 * PI * INNER_RADIUS_M * length;
        rhs[boundary.nodes[0]] += 0.5 * segment_power;
        rhs[boundary.nodes[1]] += 0.5 * segment_power;
        total_power += segment_power;
    }
    total_power
}

struct SoniumResult {
    temperature_k: Vec<f64>,
    input_power_w: f64,
    outer_heat_outflow_w: f64,
}

fn sonium_result() -> SoniumResult {
    let mesh = rectangular_mesh_triangles(
        INNER_RADIUS_M,
        OUTER_RADIUS_M,
        0.0,
        HEIGHT_M,
        RADIAL_CELLS,
        AXIAL_CELLS,
    );
    let space = H1Space::new(PolynomialDegree::P1);
    let stiffness = assemble_bilinear(
        &AxisymmetricScalarGradientForm::new(CONDUCTIVITY_W_M_K),
        &space,
        &space,
        &mesh,
        2,
    );
    let original = stiffness.to_csr();
    let mut rhs = vec![0.0; mesh.nodes.len()];
    let input_power = assemble_inner_flux(&mesh, &mut rhs);
    let unconstrained_rhs = rhs.clone();
    let mut system = FemSystem::new(stiffness, rhs);
    apply_dirichlet_values(
        &mut system,
        &mesh,
        &space,
        2,
        &|_, _, _| OUTER_TEMPERATURE_K,
        None,
    );

    let result = gmres(
        &system.matrix.to_csr(),
        &system.rhs,
        &GmresConfig {
            max_iterations: 2_000,
            restart: 75,
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
        "axisymmetric thermal solve did not converge"
    );

    let residual = original
        .matvec(&result.x)
        .into_iter()
        .zip(unconstrained_rhs)
        .map(|(internal, external)| internal - external)
        .collect::<Vec<_>>();
    let outer_heat_outflow = -mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| (node.x - OUTER_RADIUS_M).abs() < 1.0e-12)
        .map(|(node, _)| residual[node])
        .sum::<f64>();

    let temperature_k = SAMPLE_RADII_M
        .iter()
        .map(|&radius| {
            let radial = ((radius - INNER_RADIUS_M) / (OUTER_RADIUS_M - INNER_RADIUS_M)
                * RADIAL_CELLS as f64)
                .round() as usize;
            // z=0 lies in the first structured node row.
            result.x[radial]
        })
        .collect();
    SoniumResult {
        temperature_k,
        input_power_w: input_power,
        outer_heat_outflow_w: outer_heat_outflow,
    }
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    case_id: String,
    radii_m: [f64; 2],
    height_m: f64,
    conductivity_w_m_k: f64,
    inner_heat_flux_w_m2: f64,
    outer_temperature_k: f64,
    sample_radii_m: Vec<f64>,
    temperature_k: Vec<f64>,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("wolfram/axisymmetric_hollow_cylinder_heat.wls");
    let output = match Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Wolfram Engine not available; using the published logarithmic reference");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram hollow-cylinder reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram hollow-cylinder output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

fn assert_temperature_close(actual: &[f64], reference: &[f64]) {
    assert_eq!(actual.len(), reference.len());
    let squared_error = actual
        .iter()
        .zip(reference)
        .map(|(&got, &want)| (got - want).powi(2))
        .sum::<f64>();
    assert!(
        squared_error.sqrt() < 1.0e-3,
        "temperature L2 sample error is {:.6e} K",
        squared_error.sqrt()
    );
}

#[test]
fn sonium_reproduces_wolfram_axisymmetric_hollow_cylinder() {
    let sonium = sonium_result();
    let exact = SAMPLE_RADII_M.map(exact_temperature);
    assert_temperature_close(&sonium.temperature_k, &exact);
    let exact_power = INNER_HEAT_FLUX_W_M2 * 2.0 * PI * INNER_RADIUS_M * HEIGHT_M;
    assert!((sonium.input_power_w - exact_power).abs() < 1.0e-10);
    assert!(
        (sonium.outer_heat_outflow_w - sonium.input_power_w).abs() / sonium.input_power_w < 1.0e-7,
        "outer outflow {} W does not balance inner input {} W",
        sonium.outer_heat_outflow_w,
        sonium.input_power_w
    );

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(
            reference.case_id,
            "HeatTransfer-FEM-Stationary-2DAxisym-Single-HeatTransfer-0001"
        );
        assert_eq!(reference.radii_m, [INNER_RADIUS_M, OUTER_RADIUS_M]);
        assert_eq!(reference.height_m, HEIGHT_M);
        assert_eq!(reference.conductivity_w_m_k, CONDUCTIVITY_W_M_K);
        assert_eq!(reference.inner_heat_flux_w_m2, INNER_HEAT_FLUX_W_M2);
        assert_eq!(reference.outer_temperature_k, OUTER_TEMPERATURE_K);
        assert_eq!(reference.sample_radii_m, SAMPLE_RADII_M);
        assert_temperature_close(&reference.temperature_k, &exact);
        assert_temperature_close(&sonium.temperature_k, &reference.temperature_k);
    }
}
