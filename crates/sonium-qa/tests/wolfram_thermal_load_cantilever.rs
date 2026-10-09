//! Sequential thermoelastic reproduction of Wolfram's `Thermal Load` model.
//!
//! The five-metre plane-stress cantilever is heated for three hours by a
//! constant top flux while its left and bottom thermal boundaries remain at
//! zero temperature.  The final temperature field is then applied as a
//! thermal-strain load to the clamped elastic body.
//!
//! Sonium's public `assemble_thermal_strain_load` helper currently accepts
//! only one temperature and its load form always uses the 3D Lame coefficient.
//! This test therefore uses element-average temperatures and a test-local
//! plane-stress thermal load form.  The approximation is guarded by a mesh-
//! convergence check; the reusable API gap is tracked in the Wolfram roadmap.

use serde::Deserialize;
use sonium_fem::{
    assembly::{TripletMatrix, assemble_bilinear, assemble_linear_per_element},
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_homogeneous_dirichlet},
    forms::{LinearForm, LocalVector},
    mesh::{Mesh, rectangular_mesh_triangles},
    physics::{
        solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption},
        thermal::heat_conduction::{ConductionForm, HeatCapacityForm, ThermalMaterial},
    },
    spaces::{BasisTabulation, H1Space, VectorH1Space},
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{CsrMatrix, SparseLuFactorization, traits::Preconditioner};
use std::{env, path::PathBuf, process::Command};

const LENGTH_M: f64 = 5.0;
const WIDTH_M: f64 = 0.5;
const DENSITY_KG_M3: f64 = 7_500.0;
const SPECIFIC_HEAT_J_KG_K: f64 = 470.0;
const CONDUCTIVITY_W_M_K: f64 = 44.0;
const YOUNG_MODULUS_PA: f64 = 210.0e9;
const POISSON_RATIO: f64 = 0.3;
const THERMAL_EXPANSION_1_K: f64 = 12.0e-6;
const TOP_HEAT_FLUX_W_M2: f64 = 10_000.0;
const FINAL_TIME_S: f64 = 10_800.0;
const PUBLISHED_MAX_ABS_U_M: f64 = 0.004_900_9;
const PUBLISHED_MAX_ABS_V_M: f64 = 0.024_385_2;

#[derive(Clone, Copy)]
struct PlaneStressThermalLoadForm {
    temperature: f64,
}

impl LinearForm<f64> for PlaneStressThermalLoadForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        assert_eq!(test.spatial_dim, 2);
        let thermal_stress =
            YOUNG_MODULUS_PA * THERMAL_EXPANSION_1_K * self.temperature / (1.0 - POISSON_RATIO);
        let weighted_stress = weight * test.det_j * thermal_stress;
        let node_count = test.n_dofs / 2;
        for node in 0..node_count {
            for component in 0..2 {
                let dof = 2 * node + component;
                local_vector.add(dof, test.grad(dof, component) * weighted_stress);
            }
        }
    }
}

#[derive(Debug)]
struct SoniumResult {
    max_abs_u_m: f64,
    max_abs_v_m: f64,
    stored_heat_j_per_m: f64,
    input_heat_j_per_m: f64,
    relative_free_thermal_residual: f64,
}

fn add_scaled(
    first: &CsrMatrix<f64>,
    first_scale: f64,
    second: &CsrMatrix<f64>,
    second_scale: f64,
) -> CsrMatrix<f64> {
    let mut triplets = TripletMatrix::new(first.num_rows, first.num_cols);
    for row in 0..first.num_rows {
        for entry in first.row_ptrs[row]..first.row_ptrs[row + 1] {
            triplets.push(
                row,
                first.col_indices[entry],
                first_scale * first.values[entry],
            );
        }
        for entry in second.row_ptrs[row]..second.row_ptrs[row + 1] {
            triplets.push(
                row,
                second.col_indices[entry],
                second_scale * second.values[entry],
            );
        }
    }
    triplets.to_csr()
}

fn impose_zero_rows(matrix: &mut CsrMatrix<f64>, dofs: &[usize]) {
    for &dof in dofs {
        for entry in matrix.row_ptrs[dof]..matrix.row_ptrs[dof + 1] {
            matrix.values[entry] = if matrix.col_indices[entry] == dof {
                1.0
            } else {
                0.0
            };
        }
    }
}

fn top_flux_load(mesh: &Mesh) -> Vec<f64> {
    let mut load = vec![0.0; mesh.num_nodes()];
    for boundary in mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == 4)
    {
        assert_eq!(boundary.nodes.len(), 2);
        let first = mesh.nodes[boundary.nodes[0]];
        let second = mesh.nodes[boundary.nodes[1]];
        let length = ((second.x - first.x).powi(2) + (second.y - first.y).powi(2)).sqrt();
        let nodal_load = 0.5 * TOP_HEAT_FLUX_W_M2 * length;
        load[boundary.nodes[0]] += nodal_load;
        load[boundary.nodes[1]] += nodal_load;
    }
    load
}

fn solve_case(cells_x: usize, cells_y: usize, time_step_s: f64) -> SoniumResult {
    let mesh = rectangular_mesh_triangles(0.0, LENGTH_M, 0.0, WIDTH_M, cells_x, cells_y);
    let thermal_space = H1Space::new(PolynomialDegree::P1);
    let thermal_material =
        ThermalMaterial::new(CONDUCTIVITY_W_M_K, DENSITY_KG_M3, SPECIFIC_HEAT_J_KG_K);
    let mass = assemble_bilinear(
        &HeatCapacityForm::new(&thermal_material),
        &thermal_space,
        &thermal_space,
        &mesh,
        2,
    )
    .to_csr();
    let conduction = assemble_bilinear(
        &ConductionForm::from_conductivity(CONDUCTIVITY_W_M_K),
        &thermal_space,
        &thermal_space,
        &mesh,
        2,
    )
    .to_csr();
    let flux = top_flux_load(&mesh);
    let thermal_dirichlet: Vec<usize> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            (node.x.abs() < 1.0e-12 || node.y.abs() < 1.0e-12).then_some(index)
        })
        .collect();
    let mut effective = add_scaled(&mass, 1.0, &conduction, time_step_s);
    impose_zero_rows(&mut effective, &thermal_dirichlet);
    let factorization = SparseLuFactorization::factorize(&effective)
        .expect("thermal effective matrix must factorize");
    let mut temperature = vec![0.0; mesh.num_nodes()];
    let step_count = (FINAL_TIME_S / time_step_s).round() as usize;
    assert!((step_count as f64 * time_step_s - FINAL_TIME_S).abs() < 1.0e-10);
    let mut last_rhs = vec![0.0; mesh.num_nodes()];
    for _ in 0..step_count {
        let mut rhs = mass.matvec(&temperature);
        for (value, source) in rhs.iter_mut().zip(&flux) {
            *value += time_step_s * source;
        }
        for &dof in &thermal_dirichlet {
            rhs[dof] = 0.0;
        }
        temperature = factorization.apply(&rhs);
        last_rhs = rhs;
    }

    let free_residual = effective
        .matvec(&temperature)
        .iter()
        .zip(&last_rhs)
        .enumerate()
        .filter(|(index, _)| !thermal_dirichlet.contains(index))
        .map(|(_, (lhs, rhs))| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt();
    let rhs_norm = last_rhs
        .iter()
        .map(|value| value.powi(2))
        .sum::<f64>()
        .sqrt();

    let forms: Vec<_> = mesh
        .elements
        .iter()
        .map(|element| PlaneStressThermalLoadForm {
            temperature: element
                .nodes
                .iter()
                .map(|&node| temperature[node])
                .sum::<f64>()
                / element.nodes.len() as f64,
        })
        .collect();
    let displacement_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let stiffness = assemble_bilinear(
        &ElasticityForm::new_2d(
            IsotropicMaterial::new(YOUNG_MODULUS_PA, POISSON_RATIO),
            PlaneAssumption::PlaneStress,
        ),
        &displacement_space,
        &displacement_space,
        &mesh,
        2,
    );
    let mechanical_load = assemble_linear_per_element(&forms, &displacement_space, &mesh, 2);
    let mut system = FemSystem::new(stiffness, mechanical_load);
    apply_homogeneous_dirichlet(&mut system, &mesh, &displacement_space, 1);
    let constrained_stiffness = system.matrix.to_csr();
    let displacement = SparseLuFactorization::factorize(&constrained_stiffness)
        .expect("plane-stress stiffness must factorize")
        .apply(&system.rhs);

    let max_abs_u_m = displacement
        .iter()
        .step_by(2)
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let max_abs_v_m = displacement
        .iter()
        .skip(1)
        .step_by(2)
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let stored_heat_j_per_m = mass.matvec(&temperature).iter().sum();

    SoniumResult {
        max_abs_u_m,
        max_abs_v_m,
        stored_heat_j_per_m,
        input_heat_j_per_m: TOP_HEAT_FLUX_W_M2 * LENGTH_M * FINAL_TIME_S,
        relative_free_thermal_residual: free_residual / rhs_norm.max(1.0),
    }
}

#[derive(Debug, Deserialize)]
struct WolframReference {
    schema_version: u32,
    case_id: String,
    max_abs_u_m: f64,
    max_abs_v_m: f64,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/thermal_load_cantilever.wls");
    let output = match Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram Thermal Load reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram Thermal Load output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

#[test]
fn sonium_reproduces_wolfram_thermal_load_cantilever() {
    let coarse = solve_case(75, 8, 60.0);
    let fine = solve_case(100, 10, 60.0);
    eprintln!("Wolfram Thermal Load: coarse={coarse:?}, fine={fine:?}");

    assert!(fine.relative_free_thermal_residual < 1.0e-11);
    assert!(fine.stored_heat_j_per_m > 0.0);
    assert!(fine.stored_heat_j_per_m < fine.input_heat_j_per_m);
    assert!((fine.max_abs_u_m - coarse.max_abs_u_m).abs() / fine.max_abs_u_m < 0.04);
    assert!((fine.max_abs_v_m - coarse.max_abs_v_m).abs() / fine.max_abs_v_m < 0.04);
    assert!((fine.max_abs_u_m - PUBLISHED_MAX_ABS_U_M).abs() / PUBLISHED_MAX_ABS_U_M < 0.08);
    assert!((fine.max_abs_v_m - PUBLISHED_MAX_ABS_V_M).abs() / PUBLISHED_MAX_ABS_V_M < 0.08);

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.case_id, "Wolfram-ThermalLoad");
        assert!((fine.max_abs_u_m - reference.max_abs_u_m).abs() / reference.max_abs_u_m < 0.08);
        assert!((fine.max_abs_v_m - reference.max_abs_v_m).abs() / reference.max_abs_v_m < 0.08);
    }
}
