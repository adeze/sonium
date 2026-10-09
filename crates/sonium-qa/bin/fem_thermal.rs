//! Integration tests for FEM thermal conduction
//!
//! Tests 1D and 2D thermal conduction against analytical solutions.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::mesh::rectangular_mesh_triangles;
use sonium_fem::physics::thermal::heat_conduction::{ConductionForm, ThermalMaterial};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::QaResult;
use sonium_solvers::{GmresConfig, gmres};

const CASE_ID: &str = "fem.thermal-dirichlet.v1";
const MAXIMUM_ERROR: f64 = 1.0e-8;

struct ThermalMetrics {
    midpoint_error: f64,
    maximum_nodal_error: f64,
    dofs: usize,
}

fn thermal_conduction_1d() -> Result<ThermalMetrics, String> {
    let l = 1.0;
    let w = 0.1;
    let conductivity = 50.0;

    let nx = 10;
    let ny = 2;
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, w, nx, ny);

    let degree = PolynomialDegree::P1;
    let space = H1Space::new(degree);
    let n_dofs = space.num_global_dofs(&mesh);

    let material = ThermalMaterial::new(conductivity, 450.0, 7800.0);
    let form = ConductionForm::new(&material);

    let stiffness = sonium_fem::assembly::assemble_bilinear(&form, &space, &space, &mesh, 1);

    let rhs = vec![0.0; n_dofs];
    let mut system = FemSystem::new(stiffness, rhs);

    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| 100.0, None);
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| 200.0, None);

    let csr = system.matrix.to_csr();
    let rhs_array = system.rhs;

    let config = GmresConfig {
        max_iterations: 1000,
        restart: 50,
        tolerance: 1e-12,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let result = gmres(&csr, &rhs_array, &config);
    if !result.converged {
        return Err("GMRES did not converge for the thermal Dirichlet solve".into());
    }

    let mut midpoint_error = None;
    let mut maximum_nodal_error: f64 = 0.0;
    for i in 0..mesh.num_nodes() {
        let expected = 100.0 + 100.0 * mesh.nodes[i].x / l;
        maximum_nodal_error = maximum_nodal_error.max((result.x[i] - expected).abs());
        if (mesh.nodes[i].x - 0.5).abs() < 1e-10 {
            midpoint_error = Some((result.x[i] - 150.0).abs());
        }
    }
    let midpoint_error = midpoint_error.ok_or("midpoint x=0.5 not found in thermal mesh")?;
    if !midpoint_error.is_finite() || !maximum_nodal_error.is_finite() {
        return Err("thermal Dirichlet solution contained a non-finite error metric".into());
    }
    if midpoint_error > MAXIMUM_ERROR || maximum_nodal_error > MAXIMUM_ERROR {
        return Err(format!(
            "thermal Dirichlet error exceeded {MAXIMUM_ERROR:.1e}: midpoint {midpoint_error:.3e}, maximum {maximum_nodal_error:.3e}"
        ));
    }
    Ok(ThermalMetrics {
        midpoint_error,
        maximum_nodal_error,
        dofs: n_dofs,
    })
}

fn thermal_isotropic_material() -> Result<(), String> {
    let l = 0.5;
    let w = 0.05;
    let conductivity = 200.0;

    let nx = 5;
    let ny = 2;
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, w, nx, ny);

    let degree = PolynomialDegree::P1;
    let space = H1Space::new(degree);

    let material = ThermalMaterial::new(conductivity, 500.0, 8000.0);
    let form = ConductionForm::new(&material);

    let stiffness = sonium_fem::assembly::assemble_bilinear(&form, &space, &space, &mesh, 1);

    let rhs = vec![0.0; space.num_global_dofs(&mesh)];
    let system = FemSystem::new(stiffness, rhs);

    let csr = system.matrix.to_csr();
    if csr.num_rows == 0 {
        return Err("isotropic thermal assembly returned an empty matrix".into());
    }
    Ok(())
}

fn main() {
    match thermal_conduction_1d().and_then(|metrics| {
        thermal_isotropic_material()?;
        Ok(metrics)
    }) {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("midpoint_error", metrics.midpoint_error)
                .metric("maximum_nodal_error", metrics.maximum_nodal_error)
                .metric("degrees_of_freedom", metrics.dofs as f64)
                .maximum("midpoint_error", MAXIMUM_ERROR)
                .maximum("maximum_nodal_error", MAXIMUM_ERROR)
                .seed("unit-bar:10x2:p1:dirichlet-100K-200K");
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("FEM THERMAL DIRICHLET FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_thermal_dirichlet_path_matches_linear_reference() {
        let metrics = thermal_conduction_1d().expect("thermal Dirichlet solve");
        assert!(metrics.midpoint_error <= MAXIMUM_ERROR);
        assert!(metrics.maximum_nodal_error <= MAXIMUM_ERROR);
        assert_eq!(metrics.dofs, 33);
    }

    #[test]
    fn isotropic_thermal_assembly_is_nonempty() {
        thermal_isotropic_material().expect("isotropic thermal assembly");
    }
}
