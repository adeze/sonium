//! Sonium reproduction of Wolfram verification case
//! `SolidMechanics-FEM-Stationary-2D-PlaneStress-0004`.

use serde::Deserialize;
use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_homogeneous_dirichlet},
    mesh::rectangular_mesh_triangles,
    physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption},
    spaces::{FunctionSpace, VectorH1Space},
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{GmresConfig, IluPreconditioner, gmres_preconditioned};
use std::{env, path::PathBuf, process::Command};

const LENGTH_M: f64 = 1.0;
const HEIGHT_M: f64 = 0.2;
const YOUNG_MODULUS_PA: f64 = 200.0e9;
const POISSON_RATIO: f64 = 0.3;
const MASS_DENSITY_KG_M3: f64 = 7_800.0;
const GRAVITY_M_S2: f64 = -9.8;

fn exact_vertical_displacement(x: f64) -> f64 {
    let distributed_load = MASS_DENSITY_KG_M3 * GRAVITY_M_S2 * HEIGHT_M;
    let second_moment = HEIGHT_M.powi(3) / 12.0;
    distributed_load * x.powi(2) * (x.powi(2) + 6.0 * LENGTH_M.powi(2) - 4.0 * LENGTH_M * x)
        / (24.0 * second_moment * YOUNG_MODULUS_PA)
}

fn sonium_tip_displacement() -> f64 {
    let mesh = rectangular_mesh_triangles(0.0, LENGTH_M, 0.0, HEIGHT_M, 40, 8);
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let stiffness = assemble_bilinear(
        &ElasticityForm::new_2d(
            IsotropicMaterial::new(YOUNG_MODULUS_PA, POISSON_RATIO),
            PlaneAssumption::PlaneStress,
        ),
        &space,
        &space,
        &mesh,
        1,
    );
    let mut rhs = vec![0.0; space.num_global_dofs(&mesh)];

    // Consistent P1 body load. Thickness multiplies both load and planar
    // stiffness, so it cancels exactly as in Wolfram's reference formula.
    for element in &mesh.elements {
        assert_eq!(element.nodes.len(), 3);
        let p0 = mesh.nodes[element.nodes[0]];
        let p1 = mesh.nodes[element.nodes[1]];
        let p2 = mesh.nodes[element.nodes[2]];
        let twice_area = ((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)).abs();
        let nodal_load = MASS_DENSITY_KG_M3 * GRAVITY_M_S2 * (0.5 * twice_area) / 3.0;
        for &node in &element.nodes {
            rhs[2 * node + 1] += nodal_load;
        }
    }

    let mut system = FemSystem::new(stiffness, rhs);
    apply_homogeneous_dirichlet(&mut system, &mesh, &space, 1);
    let csr = system.matrix.to_csr();
    let result = gmres_preconditioned(
        &csr,
        &IluPreconditioner::from_csr(&csr),
        &system.rhs,
        &GmresConfig {
            max_iterations: 5_000,
            restart: 100,
            tolerance: 1.0e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(
        result.converged,
        "gravity-cantilever solve did not converge"
    );
    let tip_center = mesh
        .nodes
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (a.x - LENGTH_M).powi(2) + (a.y - HEIGHT_M / 2.0).powi(2);
            let db = (b.x - LENGTH_M).powi(2) + (b.y - HEIGHT_M / 2.0).powi(2);
            da.total_cmp(&db)
        })
        .expect("cantilever mesh should have nodes")
        .0;
    result.x[2 * tip_center + 1]
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    dimensions_m: [f64; 2],
    young_modulus_pa: f64,
    poisson_ratio: f64,
    mass_density_kg_m3: f64,
    gravity_m_s2: f64,
    tip_vertical_displacement_m: f64,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/gravity_cantilever.wls");
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
            eprintln!("Wolfram Engine not available; using analytical cantilever reference only");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram cantilever reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram cantilever output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

fn assert_displacement_close(actual: f64, reference: f64) {
    let relative_error = (actual - reference).abs() / reference.abs();
    assert!(
        relative_error < 0.04,
        "tip displacement {actual:.9e} m differs from {reference:.9e} m by {:.3}%",
        100.0 * relative_error
    );
}

#[test]
fn sonium_reproduces_wolfram_gravity_loaded_cantilever() {
    let sonium = sonium_tip_displacement();
    let exact = exact_vertical_displacement(LENGTH_M);
    assert_displacement_close(sonium, exact);
    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.dimensions_m, [LENGTH_M, HEIGHT_M]);
        assert_eq!(reference.young_modulus_pa, YOUNG_MODULUS_PA);
        assert_eq!(reference.poisson_ratio, POISSON_RATIO);
        assert_eq!(reference.mass_density_kg_m3, MASS_DENSITY_KG_M3);
        assert_eq!(reference.gravity_m_s2, GRAVITY_M_S2);
        assert_displacement_close(reference.tip_vertical_displacement_m, exact);
        assert_displacement_close(sonium, reference.tip_vertical_displacement_m);
    }
}
