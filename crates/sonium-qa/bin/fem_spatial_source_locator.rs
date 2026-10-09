//! CPU-only FEM spatial source-element locator contract.

use sonium_fem::mesh::{
    InterpolationError, create_interpolation_data_checked, interpolate_field_checked,
    interpolate_scalar_at_point_checked, transfer_design_field_checked, unit_square_triangles,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.spatial-source-locator.v1";

fn run() -> Result<(usize, usize, f64, usize, usize), String> {
    let source = unit_square_triangles(16);
    let target = unit_square_triangles(32);
    let source_values: Vec<f64> = source
        .nodes
        .iter()
        .map(|node| 1.25 - 0.75 * node.x + 2.0 * node.y)
        .collect();

    let data = create_interpolation_data_checked(&target, &source)
        .map_err(|error| format!("source locator metadata failed: {error}"))?;
    let transferred = interpolate_field_checked(&data, &source, &target, &source_values)
        .map_err(|error| format!("source locator scalar transfer failed: {error}"))?;
    let max_error = target
        .nodes
        .iter()
        .zip(&transferred)
        .map(|(node, value)| (value - (1.25 - 0.75 * node.x + 2.0 * node.y)).abs())
        .fold(0.0, f64::max);

    let design_values: Vec<f64> = (0..source.num_elements())
        .map(|element| 1.0 + element as f64)
        .collect();
    let design_transfer = transfer_design_field_checked(&source, &target, &design_values)
        .map_err(|error| format!("source locator design transfer failed: {error}"))?;
    if design_transfer.len() != target.num_elements()
        || design_transfer.iter().any(|value| !value.is_finite())
    {
        return Err("source locator produced an invalid design transfer".into());
    }

    let point = [0.37, 0.61, 0.0];
    let point_value = interpolate_scalar_at_point_checked(&source, &source_values, point)
        .map_err(|error| format!("interior point lookup failed: {error}"))?
        .ok_or_else(|| "interior point was not located".to_string())?;
    let expected = 1.25 - 0.75 * point[0] + 2.0 * point[1];
    if (point_value - expected).abs() > 1.0e-12 {
        return Err(format!(
            "interior point error {} exceeds tolerance",
            (point_value - expected).abs()
        ));
    }

    let mut rejection_checks = 0;
    if interpolate_scalar_at_point_checked(&source, &source_values, [1.1, 0.5, 0.0])
        .map_err(|error| format!("outside point lookup failed: {error}"))?
        .is_none()
    {
        rejection_checks += 1;
    }
    if matches!(
        interpolate_scalar_at_point_checked(&source, &source_values, [f64::NAN, 0.5, 0.0]),
        Err(InterpolationError::NonFinitePoint)
    ) {
        rejection_checks += 1;
    }

    Ok((
        source.num_elements(),
        target.num_nodes(),
        max_error,
        data.target_to_source.iter().flatten().count(),
        rejection_checks,
    ))
}

fn main() {
    let result = match run() {
        Ok((source_elements, target_nodes, max_error, mapped_nodes, rejection_checks)) => {
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("source_elements", source_elements as f64)
                .minimum("source_elements", 1.0)
                .metric("target_nodes", target_nodes as f64)
                .minimum("target_nodes", 1.0)
                .metric("mapped_target_nodes", mapped_nodes as f64)
                .metric("max_scalar_transfer_error", max_error)
                .maximum("max_scalar_transfer_error", 2.0e-12)
                .metric("rejection_checks", rejection_checks as f64)
                .minimum("rejection_checks", 2.0)
                .dimension("2-D affine triangular source/target refinement grids")
                .formulation("deterministic BVH source ownership for scalar and design transfer")
                .input_checksum("unit-square-triangles-16-to-32:spatial-locator-v1")
                .reference_checksum("analytic-linear-field-and-fail-closed-lookup:v1")
                .seed("fem-spatial-source-locator:cpu:v1")
        }
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
