//! Sonium reproduction of Wolfram/NAFEMS verification case
//! `HeatTransfer-FEM-Stationary-2DAxisym-Single-HeatTransfer-0002-A`.
//!
//! The meridional domain is `0.02 <= r <= 0.1 m`, `0 <= z <= 0.14 m`.
//! A `5e5 W/m^2` heat flux enters only through the inner-wall segment
//! `0.04 <= z <= 0.1 m`; the outer wall and both ends are fixed at 273.15 K.

use serde::Deserialize;
use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_dirichlet_values},
    mesh::{Mesh, interpolate_scalar_at_point, rectangular_mesh_triangles},
    physics::solid::axisymmetric::AxisymmetricScalarGradientForm,
    spaces::H1Space,
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{GmresConfig, gmres};
use std::{env, f64::consts::PI, path::PathBuf, process::Command};

const INNER_RADIUS_M: f64 = 0.02;
const OUTER_RADIUS_M: f64 = 0.1;
const HEIGHT_M: f64 = 0.14;
const FLUX_Z_MIN_M: f64 = 0.04;
const FLUX_Z_MAX_M: f64 = 0.1;
const CONDUCTIVITY_W_M_K: f64 = 52.0;
const HEAT_FLUX_W_M2: f64 = 5.0e5;
const FIXED_TEMPERATURE_K: f64 = 273.15;
const PUBLISHED_TEMPERATURE_K: f64 = 332.97;
const SAMPLE_POINT_M: [f64; 2] = [0.04, 0.04];
const RADIAL_CELLS: usize = 120;
const AXIAL_CELLS: usize = 210;

fn assemble_segmented_inner_flux(mesh: &Mesh, rhs: &mut [f64]) -> f64 {
    let mut input_power = 0.0;
    for face in &mesh.boundaries {
        if face.marker != 1 || face.nodes.len() != 2 {
            continue;
        }
        let first = mesh.nodes[face.nodes[0]];
        let second = mesh.nodes[face.nodes[1]];
        let midpoint_z = 0.5 * (first.y + second.y);
        if !(FLUX_Z_MIN_M - 1.0e-12..=FLUX_Z_MAX_M + 1.0e-12).contains(&midpoint_z) {
            continue;
        }
        let length = (second.y - first.y).abs();
        let segment_power = HEAT_FLUX_W_M2 * 2.0 * PI * INNER_RADIUS_M * length;
        rhs[face.nodes[0]] += 0.5 * segment_power;
        rhs[face.nodes[1]] += 0.5 * segment_power;
        input_power += segment_power;
    }
    input_power
}

struct SoniumSolution {
    sample_temperature_k: f64,
    heat_balance_relative_error: f64,
}

fn sonium_solution() -> SoniumSolution {
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
    let unconstrained = stiffness.to_csr();
    let mut rhs = vec![0.0; mesh.num_nodes()];
    let input_power = assemble_segmented_inner_flux(&mesh, &mut rhs);
    let unconstrained_rhs = rhs.clone();
    let mut system = FemSystem::new(stiffness, rhs);
    for marker in [2, 3, 4] {
        apply_dirichlet_values(
            &mut system,
            &mesh,
            &space,
            marker,
            &|_, _, _| FIXED_TEMPERATURE_K,
            None,
        );
    }

    let result = gmres(
        &system.matrix.to_csr(),
        &system.rhs,
        &GmresConfig {
            max_iterations: 10_000,
            restart: 120,
            tolerance: 1.0e-11,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(
        result.converged,
        "segmented-flux heat solve did not converge"
    );

    let sample_temperature_k = interpolate_scalar_at_point(
        &mesh,
        &result.x,
        [SAMPLE_POINT_M[0], SAMPLE_POINT_M[1], 0.0],
    )
    .expect("published Wolfram probe must lie inside the annulus section");

    let residual = unconstrained
        .matvec(&result.x)
        .into_iter()
        .zip(unconstrained_rhs)
        .map(|(internal, external)| internal - external)
        .collect::<Vec<_>>();
    let fixed_boundary_outflow = -(0..mesh.num_nodes())
        .filter(|&node| {
            let point = mesh.nodes[node];
            (point.x - OUTER_RADIUS_M).abs() < 1.0e-12
                || point.y.abs() < 1.0e-12
                || (point.y - HEIGHT_M).abs() < 1.0e-12
        })
        .map(|node| residual[node])
        .sum::<f64>();
    let heat_balance_relative_error =
        (fixed_boundary_outflow - input_power).abs() / input_power.abs();

    SoniumSolution {
        sample_temperature_k,
        heat_balance_relative_error,
    }
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    sample_point_m: [f64; 2],
    temperature_k: f64,
    published_temperature_k: f64,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("wolfram/axisymmetric_segmented_flux_heat.wls");
    let output = Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
        .unwrap_or_else(|error| panic!("failed to launch {executable:?}: {error}"));
    assert!(
        output.status.success(),
        "Wolfram segmented-flux heat solve failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram heat output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

#[test]
fn sonium_reproduces_wolfram_axisymmetric_segmented_flux_case() {
    let sonium = sonium_solution();
    let relative_error =
        ((sonium.sample_temperature_k - PUBLISHED_TEMPERATURE_K) / PUBLISHED_TEMPERATURE_K).abs();
    eprintln!(
        "Sonium T(0.04,0.04)={:.8} K relative_error={relative_error:.6e} heat_balance={:.3e}",
        sonium.sample_temperature_k, sonium.heat_balance_relative_error
    );
    assert!(
        relative_error < 1.0e-5,
        "Sonium relative error {relative_error:.6e} exceeds Wolfram's 1e-5 gate"
    );
    assert!(sonium.heat_balance_relative_error < 1.0e-7);

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.sample_point_m, SAMPLE_POINT_M);
        assert_eq!(reference.published_temperature_k, PUBLISHED_TEMPERATURE_K);
        assert!(
            ((reference.temperature_k - PUBLISHED_TEMPERATURE_K) / PUBLISHED_TEMPERATURE_K).abs()
                < 1.0e-5
        );
        // Both codes are independently verified within 1e-5 of the published
        // value above, so by the triangle bound their mutual spread can
        // legitimately reach 2e-5 when they bracket it (independent
        // discretization errors on different meshes). Gate the live
        // cross-check there plus margin instead of re-imposing 1e-5, which
        // would demand same-sidedness by luck rather than correctness.
        assert!(
            ((sonium.sample_temperature_k - reference.temperature_k) / reference.temperature_k)
                .abs()
                < 2.5e-5
        );
    }
}
