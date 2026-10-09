//! Mesh-convergence gate for the production cantilever beam path.

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{Beam2DStudy, run_beam2d_static};

const CASE_ID: &str = "structural.beam-mesh-convergence.v1";
const REFERENCE_TOLERANCE: f64 = 1.0e-8;

#[derive(Debug, Clone, Copy)]
struct Row {
    elements: usize,
    dofs: usize,
    computed_tip_m: f64,
    reference_tip_m: f64,
    relative_error: f64,
}

fn run() -> Result<Vec<Row>, String> {
    let base = Beam2DStudy::introductory();
    let reference = base.vertical_tip_force_n * base.geometry.length_m.powi(3)
        / (3.0 * base.material.youngs_modulus_pa * base.geometry.second_moment_m4());
    if !reference.is_finite() || reference.abs() <= f64::EPSILON {
        return Err("independent cantilever reference is not finite/non-zero".into());
    }

    let mut rows = Vec::new();
    for elements in [1usize, 2, 4, 8, 16] {
        let mut study = base.clone();
        study.geometry.elements = elements;
        let result = run_beam2d_static(&study).map_err(|error| error.to_string())?;
        if !result.converged {
            return Err(format!(
                "beam solve did not converge at {elements} elements"
            ));
        }
        let relative_error = (result.tip_deflection_m - reference).abs() / reference.abs();
        let values = [result.tip_deflection_m, reference, relative_error];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "non-finite convergence value at {elements} elements"
            ));
        }
        if relative_error > REFERENCE_TOLERANCE {
            return Err(format!(
                "mesh-convergence reference error at {elements} elements is {relative_error:.3e}"
            ));
        }
        rows.push(Row {
            elements,
            dofs: result.dofs,
            computed_tip_m: result.tip_deflection_m,
            reference_tip_m: reference,
            relative_error,
        });
    }

    if rows.windows(2).any(|pair| pair[1].dofs <= pair[0].dofs) {
        return Err("mesh-convergence DOF sequence is not strictly increasing".into());
    }
    Ok(rows)
}

fn main() {
    match run() {
        Ok(rows) => {
            let mut record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("mesh_levels", rows.len() as f64)
                .metric(
                    "finest_relative_error",
                    rows.last().expect("non-empty mesh sequence").relative_error,
                )
                .maximum("finest_relative_error", REFERENCE_TOLERANCE)
                .seed("beam2d-cantilever-convergence:elements=1,2,4,8,16:cpu");
            for row in rows {
                let prefix = format!("elements_{}", row.elements);
                record = record
                    .metric(format!("{prefix}_dofs"), row.dofs as f64)
                    .metric(format!("{prefix}_tip_deflection_m"), row.computed_tip_m)
                    .metric(format!("{prefix}_reference_tip_m"), row.reference_tip_m)
                    .metric(format!("{prefix}_relative_error"), row.relative_error)
                    .maximum(format!("{prefix}_relative_error"), REFERENCE_TOLERANCE);
            }
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
            eprintln!("STRUCTURAL BEAM CONVERGENCE FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
