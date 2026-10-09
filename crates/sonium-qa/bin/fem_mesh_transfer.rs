//! Production mesh state/adjoint-transfer QA gate.

use num_complex::Complex64;
use sonium_fem::mesh::{
    create_interpolation_data_checked, interpolate_complex_field_adjoint_checked,
    interpolate_complex_field_checked, interpolate_field_adjoint_checked,
    interpolate_field_checked, transfer_design_field_checked, unit_square_triangles,
};
use sonium_qa::QaResult;

const TOLERANCE: f64 = 1e-11;
const ADJOINT_CASE_ID: &str = "fem.mesh-transfer-adjoint.v1";
const DESIGN_CASE_ID: &str = "fem.mesh-transfer-design.v1";
const SEED: &str = "unit-square:uniform-refinement-1x-to-2x:mass-weighted-adjoint";

fn run() -> Result<(f64, f64, f64), String> {
    let source = unit_square_triangles(1);
    let target = unit_square_triangles(2);
    let data = create_interpolation_data_checked(&target, &source)
        .map_err(|error| format!("mesh interpolation metadata rejected: {error}"))?;
    let source_weights: Vec<f64> = source
        .nodes
        .iter()
        .enumerate()
        .map(|(index, _)| 0.7 + 0.2 * index as f64)
        .collect();
    let target_weights: Vec<f64> = target
        .nodes
        .iter()
        .enumerate()
        .map(|(index, _)| 0.8 + 0.05 * index as f64)
        .collect();

    let source_values: Vec<f64> = source
        .nodes
        .iter()
        .map(|node| 0.75 + 0.5 * node.x - 1.25 * node.y)
        .collect();
    let target_dual: Vec<f64> = target
        .nodes
        .iter()
        .map(|node| -0.25 + 1.5 * node.x + 0.25 * node.y)
        .collect();
    let target_values = interpolate_field_checked(&data, &source, &target, &source_values)
        .map_err(|error| format!("scalar primal transfer failed: {error}"))?;
    let scalar_adjoint = interpolate_field_adjoint_checked(
        &data,
        &source,
        &target,
        &target_dual,
        &source_weights,
        &target_weights,
    )
    .map_err(|error| format!("scalar adjoint transfer failed: {error}"))?;
    let scalar_target_pairing: f64 = target_values
        .iter()
        .zip(&target_dual)
        .zip(&target_weights)
        .map(|((value, dual), weight)| value * dual * weight)
        .sum();
    let scalar_source_pairing: f64 = source_values
        .iter()
        .zip(&scalar_adjoint)
        .zip(&source_weights)
        .map(|((value, dual), weight)| value * dual * weight)
        .sum();
    let scalar_error = (scalar_target_pairing - scalar_source_pairing).abs();

    let source_complex: Vec<Complex64> = source
        .nodes
        .iter()
        .map(|node| Complex64::new(0.75 + node.x, -0.5 + node.y))
        .collect();
    let target_complex_dual: Vec<Complex64> = target
        .nodes
        .iter()
        .map(|node| Complex64::new(-0.25 + node.x, 0.75 - node.y))
        .collect();
    let target_complex =
        interpolate_complex_field_checked(&data, &source, &target, &source_complex)
            .map_err(|error| format!("complex primal transfer failed: {error}"))?;
    let complex_adjoint = interpolate_complex_field_adjoint_checked(
        &data,
        &source,
        &target,
        &target_complex_dual,
        &source_weights,
        &target_weights,
    )
    .map_err(|error| format!("complex adjoint transfer failed: {error}"))?;
    let complex_target_pairing: Complex64 = target_complex
        .iter()
        .zip(&target_complex_dual)
        .zip(&target_weights)
        .map(|((value, dual), weight)| value.conj() * dual * *weight)
        .sum();
    let complex_source_pairing: Complex64 = source_complex
        .iter()
        .zip(&complex_adjoint)
        .zip(&source_weights)
        .map(|((value, dual), weight)| value.conj() * dual * *weight)
        .sum();
    let complex_error = (complex_target_pairing - complex_source_pairing).norm();

    let source_design = [0.25, 0.75];
    let target_design = transfer_design_field_checked(&source, &target, &source_design)
        .map_err(|error| format!("design-field transfer failed: {error}"))?;
    let source_design_integral: f64 = source
        .elements
        .iter()
        .enumerate()
        .map(|(element, _)| source.element_measure(element) * source_design[element])
        .sum();
    let target_design_integral: f64 = target
        .elements
        .iter()
        .enumerate()
        .map(|(element, _)| target.element_measure(element) * target_design[element])
        .sum();
    let design_error = (source_design_integral - target_design_integral).abs();

    if scalar_error > TOLERANCE || complex_error > TOLERANCE || design_error > TOLERANCE {
        return Err(format!(
            "mesh transfer tolerance exceeded: scalar={scalar_error:.3e}, complex={complex_error:.3e}, design={design_error:.3e}"
        ));
    }
    Ok((scalar_error, complex_error, design_error))
}

fn main() {
    match run() {
        Ok((scalar_error, complex_error, design_error)) => {
            let record = QaResult::passed(ADJOINT_CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("scalar_pairing_error", scalar_error)
                .metric("complex_hermitian_pairing_error", complex_error)
                .maximum("scalar_pairing_error", TOLERANCE)
                .maximum("complex_hermitian_pairing_error", TOLERANCE)
                .seed(SEED)
                .to_json_line()
                .expect("mesh transfer QA result must serialize");
            println!("QA_RESULT: {record}");
            let design_record = QaResult::passed(DESIGN_CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("design_field_integral_error", design_error)
                .maximum("design_field_integral_error", TOLERANCE)
                .seed(SEED)
                .to_json_line()
                .expect("design transfer QA result must serialize");
            println!("QA_RESULT: {design_record}");
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(ADJOINT_CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
                .expect("mesh transfer QA failure result must serialize");
            println!("QA_RESULT: {record}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
