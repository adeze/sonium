//! Sonium reproduction of Wolfram verification case
//! `SolidMechanics-FEM-Stationary-2D-PlaneStress-0003`.
//!
//! The official case is a 1 m by 0.2 m rectangular cantilever, fixed on the
//! left and loaded by a uniform -1 MPa traction on its top edge. Wolfram checks
//! the vertical displacement along `y=0` against the Euler-Bernoulli field.

use serde::Deserialize;
use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_homogeneous_dirichlet},
    mesh::{Mesh, rectangular_mesh_triangles},
    physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption},
    spaces::{FunctionSpace, VectorH1Space},
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{GmresConfig, IluPreconditioner, gmres_preconditioned};
use std::{env, path::PathBuf, process::Command};

const LENGTH_M: f64 = 1.0;
const HEIGHT_M: f64 = 0.2;
const THICKNESS_M: f64 = 0.1;
const YOUNG_MODULUS_PA: f64 = 200.0e9;
const POISSON_RATIO: f64 = 0.3;
const MASS_DENSITY_KG_M3: f64 = 7_800.0;
const TOP_TRACTION_PA: f64 = -1.0e6;
const CELLS_X: usize = 80;
const CELLS_Y: usize = 16;
const SAMPLE_X_M: [f64; 11] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

fn exact_vertical_displacement(x: f64) -> f64 {
    let second_moment_per_depth = HEIGHT_M.powi(3) / 12.0;
    TOP_TRACTION_PA * x.powi(2) * (x.powi(2) + 6.0 * LENGTH_M.powi(2) - 4.0 * LENGTH_M * x)
        / (24.0 * second_moment_per_depth * YOUNG_MODULUS_PA)
}

fn assemble_top_traction(mesh: &Mesh, rhs: &mut [f64]) -> f64 {
    let mut total = 0.0;
    for boundary in mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == 4)
    {
        assert_eq!(boundary.nodes.len(), 2);
        let first = mesh.nodes[boundary.nodes[0]];
        let second = mesh.nodes[boundary.nodes[1]];
        let length = ((second.x - first.x).powi(2) + (second.y - first.y).powi(2)).sqrt();
        let force = TOP_TRACTION_PA * length;
        rhs[2 * boundary.nodes[0] + 1] += 0.5 * force;
        rhs[2 * boundary.nodes[1] + 1] += 0.5 * force;
        total += force;
    }
    total
}

struct SoniumResult {
    displacement_m: Vec<f64>,
    applied_force_n_per_m: f64,
    vertical_reaction_n_per_m: f64,
}

fn sonium_result() -> SoniumResult {
    let mesh = rectangular_mesh_triangles(0.0, LENGTH_M, 0.0, HEIGHT_M, CELLS_X, CELLS_Y);
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let stiffness = assemble_bilinear(
        &ElasticityForm::new_2d(
            IsotropicMaterial::new(YOUNG_MODULUS_PA, POISSON_RATIO),
            PlaneAssumption::PlaneStress,
        ),
        &space,
        &space,
        &mesh,
        2,
    );
    let original = stiffness.to_csr();
    let mut rhs = vec![0.0; space.num_global_dofs(&mesh)];
    let applied_force = assemble_top_traction(&mesh, &mut rhs);
    let unconstrained_rhs = rhs.clone();
    let mut system = FemSystem::new(stiffness, rhs);
    apply_homogeneous_dirichlet(&mut system, &mesh, &space, 1);

    let constrained = system.matrix.to_csr();
    let result = gmres_preconditioned(
        &constrained,
        &IluPreconditioner::from_csr(&constrained),
        &system.rhs,
        &GmresConfig {
            max_iterations: 5_000,
            restart: 150,
            tolerance: 1.0e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "loaded cantilever solve did not converge");

    let internal = original.matvec(&result.x);
    let vertical_reaction = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.x.abs() < 1.0e-12)
        .map(|(node, _)| {
            let dof = 2 * node + 1;
            internal[dof] - unconstrained_rhs[dof]
        })
        .sum::<f64>();
    let displacement_m = SAMPLE_X_M
        .iter()
        .map(|&x| {
            let ix = (x / LENGTH_M * CELLS_X as f64).round() as usize;
            result.x[2 * ix + 1]
        })
        .collect();
    SoniumResult {
        displacement_m,
        applied_force_n_per_m: applied_force,
        vertical_reaction_n_per_m: vertical_reaction,
    }
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    case_id: String,
    dimensions_m: [f64; 3],
    young_modulus_pa: f64,
    poisson_ratio: f64,
    mass_density_kg_m3: f64,
    top_traction_pa: f64,
    sample_x_m: Vec<f64>,
    vertical_displacement_m: Vec<f64>,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("wolfram/loaded_cantilever_plane_stress.wls");
    let output = match Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Wolfram Engine not available; using the published beam reference");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram loaded-cantilever reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram loaded-cantilever output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

fn assert_displacement_close(actual: &[f64], reference: &[f64]) {
    assert_eq!(actual.len(), reference.len());
    for (sample, (&got, &want)) in actual.iter().zip(reference).enumerate() {
        assert!(
            (got - want).abs() < 1.0e-4,
            "sample {sample}: displacement {got:.9e} m differs from {want:.9e} m by {:.3e} m",
            (got - want).abs()
        );
    }
}

#[test]
fn sonium_reproduces_wolfram_uniformly_loaded_cantilever() {
    let sonium = sonium_result();
    let exact = SAMPLE_X_M.map(exact_vertical_displacement);
    assert_displacement_close(&sonium.displacement_m, &exact);
    assert!((sonium.applied_force_n_per_m - TOP_TRACTION_PA * LENGTH_M).abs() < 1.0e-8);
    assert!(
        (sonium.vertical_reaction_n_per_m + sonium.applied_force_n_per_m).abs()
            / sonium.applied_force_n_per_m.abs()
            < 1.0e-8,
        "vertical reaction {} N/m does not balance top load {} N/m",
        sonium.vertical_reaction_n_per_m,
        sonium.applied_force_n_per_m
    );

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(
            reference.case_id,
            "SolidMechanics-FEM-Stationary-2D-PlaneStress-0003"
        );
        assert_eq!(reference.dimensions_m, [LENGTH_M, HEIGHT_M, THICKNESS_M]);
        assert_eq!(reference.young_modulus_pa, YOUNG_MODULUS_PA);
        assert_eq!(reference.poisson_ratio, POISSON_RATIO);
        assert_eq!(reference.mass_density_kg_m3, MASS_DENSITY_KG_M3);
        assert_eq!(reference.top_traction_pa, TOP_TRACTION_PA);
        assert_eq!(reference.sample_x_m, SAMPLE_X_M);
        assert_displacement_close(&reference.vertical_displacement_m, &exact);
        assert_displacement_close(&sonium.displacement_m, &reference.vertical_displacement_m);
    }
}
