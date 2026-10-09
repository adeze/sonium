//! Release-facing cubic completeness and orientation gate for tetrahedral H1/P3.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::unit_cube_tetrahedra;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "fem.p3-tetra-reproduction.v1";
const TOLERANCE: f64 = 5.0e-12;

fn cubic([x, y, z]: [f64; 3]) -> f64 {
    x.powi(3) - 2.0 * x.powi(2) * y + 0.75 * x * y * z + 1.5 * y.powi(2) - 0.5 * z.powi(3)
        + 0.25 * x
        - 0.8 * y
        + 0.6 * z
        + 1.0
}

fn run() -> Result<(f64, usize, u64), String> {
    let started = Instant::now();
    let mesh = unit_cube_tetrahedra(1);
    let space = H1Space::new(PolynomialDegree::P3);
    space
        .validate(&mesh)
        .map_err(|error| format!("P3 tetrahedral space rejected its mesh: {error}"))?;
    let coordinates = space
        .dof_coordinates(&mesh)
        .ok_or_else(|| "P3 tetrahedral DOF coordinates unavailable".to_string())?;
    let samples = [(0.11, 0.17, 0.23), (0.32, 0.19, 0.21), (0.07, 0.41, 0.13)];
    let mut max_residual: f64 = 0.0;
    let mut checks = 0;

    for element in 0..mesh.num_elements() {
        let local_to_global = space.local_to_global(&mesh, element);
        for &(xi, eta, zeta) in &samples {
            let tabulation = space.tabulate(&mesh, element, xi, eta, zeta);
            let interpolated = local_to_global
                .iter()
                .enumerate()
                .map(|(local, &global)| tabulation.value(local) * cubic(coordinates[global]))
                .sum::<f64>();
            let residual = (interpolated - cubic(tabulation.physical_coords)).abs();
            max_residual = max_residual.max(residual);
            checks += 1;
        }
    }

    if max_residual > TOLERANCE {
        return Err(format!(
            "P3 cubic reproduction residual {max_residual:.3e} exceeds {TOLERANCE:.3e}"
        ));
    }
    Ok((max_residual, checks, started.elapsed().as_millis() as u64))
}

fn main() {
    match run() {
        Ok((max_residual, checks, elapsed_ms)) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("max_cubic_reproduction_residual", max_residual)
                .metric("sample_checks", checks as f64)
                .maximum("max_cubic_reproduction_residual", TOLERANCE)
                .reference("analytic cubic polynomial on affine tetrahedral elements")
                .reference_checksum("analytic/deterministic-fixture:fem-p3-tetra-reproduction-v1")
                .input_checksum("mesh:unit-cube-tetrahedra:1")
                .dimension("3D tetrahedral H1/P3")
                .formulation(
                    "global nodal coordinates, orientation-aware edge DOFs, physical tabulation",
                )
                .tolerance("cubic interpolation residual <= 5e-12")
                .seed("reference-cubic:unit-cube-tetrahedra-v1")
                .timing_ms("p3_cubic_reproduction", elapsed_ms)
                .elapsed_ms(elapsed_ms);
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
            eprintln!("FEM P3 TETRAHEDRAL REPRODUCTION FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
