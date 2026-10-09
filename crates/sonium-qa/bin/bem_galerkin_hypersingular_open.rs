//! CPU evidence for open-surface Galerkin hypersingular boundary terms.

use std::time::Instant;

use num_complex::Complex64;
use sonium_bem::core::assembly::common::get_element_coords;
use sonium_bem::core::integration::{
    galerkin_single_layer_self_quad4_with_basis, galerkin_single_layer_self_triangle_with_basis,
};
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_bem::{BasisOrder, PhysicsParams, assemble_galerkin_hypersingular_with_global_dofs};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "bem.galerkin-hypersingular-open-boundary.v1";

fn panel(
    element_type: ElementType,
    connectivity: Vec<usize>,
    dof_addresses: Vec<usize>,
) -> Element {
    let node_count = element_type.num_nodes();
    Element {
        connectivity,
        element_type,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; node_count],
        center: [0.5, 0.5, 0.0],
        area: 1.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses,
    }
}

fn direct_error(
    assembled: &sonium_solvers::dense::DenseMatrix<Complex64>,
    direct: &sonium_bem::core::integration::GalerkinIntegrationResult,
) -> f64 {
    let count = assembled.nrows();
    (0..count)
        .flat_map(|row| (0..count).map(move |column| (row, column)))
        .map(|(row, column)| {
            let direct_value = (direct.basis_results[row][column].d2g_dnxdny_integral
                + direct.basis_results[column][row].d2g_dnxdny_integral)
                * 0.5;
            (assembled[[row, column]] - direct_value).norm()
        })
        .fold(0.0_f64, f64::max)
}

fn run() -> Result<(f64, f64, f64, f64), String> {
    let physics =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?;
    let tri_nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let tri_element = panel(ElementType::Tri3, vec![0, 1, 2], (0..3).collect());
    let tri_linear = assemble_galerkin_hypersingular_with_global_dofs(
        std::slice::from_ref(&tri_element),
        &tri_nodes,
        &physics,
        BasisOrder::Linear,
        4,
        3,
    )
    .map_err(|error| error.to_string())?;
    let tri_coords = get_element_coords(&tri_element, &tri_nodes);
    let tri_linear_direct =
        galerkin_single_layer_self_triangle_with_basis(&tri_coords, &physics, 3, 3, 4);
    let tri_linear_error = direct_error(&tri_linear.hypersingular, &tri_linear_direct);

    let tri_quadratic_element = panel(ElementType::Tri3, vec![0, 1, 2], (0..6).collect());
    let tri_quadratic = assemble_galerkin_hypersingular_with_global_dofs(
        std::slice::from_ref(&tri_quadratic_element),
        &tri_nodes,
        &physics,
        BasisOrder::Quadratic,
        4,
        6,
    )
    .map_err(|error| error.to_string())?;
    let tri_quadratic_direct =
        galerkin_single_layer_self_triangle_with_basis(&tri_coords, &physics, 6, 6, 4);
    let tri_quadratic_error = direct_error(&tri_quadratic.hypersingular, &tri_quadratic_direct);

    let quad_nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let quad_element = panel(ElementType::Quad4, vec![0, 1, 2, 3], (0..4).collect());
    let quad_linear = assemble_galerkin_hypersingular_with_global_dofs(
        std::slice::from_ref(&quad_element),
        &quad_nodes,
        &physics,
        BasisOrder::Linear,
        4,
        4,
    )
    .map_err(|error| error.to_string())?;
    let quad_coords = get_element_coords(&quad_element, &quad_nodes);
    let quad_linear_direct =
        galerkin_single_layer_self_quad4_with_basis(&quad_coords, &physics, 4, 4);
    let quad_linear_error = direct_error(&quad_linear.hypersingular, &quad_linear_direct);

    let quad_quadratic_element = panel(ElementType::Quad4, vec![0, 1, 2, 3], (0..9).collect());
    let quad_quadratic = assemble_galerkin_hypersingular_with_global_dofs(
        std::slice::from_ref(&quad_quadratic_element),
        &quad_nodes,
        &physics,
        BasisOrder::Quadratic,
        4,
        9,
    )
    .map_err(|error| error.to_string())?;
    let quad_quadratic_direct =
        galerkin_single_layer_self_quad4_with_basis(&quad_coords, &physics, 9, 4);
    let quad_quadratic_error = direct_error(&quad_quadratic.hypersingular, &quad_quadratic_direct);

    for (label, matrix) in [
        ("tri-linear", &tri_linear.hypersingular),
        ("tri-quadratic", &tri_quadratic.hypersingular),
        ("quad-linear", &quad_linear.hypersingular),
        ("quad-quadratic", &quad_quadratic.hypersingular),
    ] {
        if matrix
            .as_slice()
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(format!("{label} open hypersingular matrix is non-finite"));
        }
    }
    Ok((
        tri_linear_error,
        tri_quadratic_error,
        quad_linear_error,
        quad_quadratic_error,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((tri_linear, tri_quadratic, quad_linear, quad_quadratic)) => QaResult::passed(CASE_ID)
            .input_checksum("bem:single-open-panel:Tri3-Quad4:P1-P2")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-galerkin-hypersingular-open-boundary-v1",
            )
            .dimension("3-D CPU; open Tri3/Quad4 P1/P2 finite-part boundary terms")
            .metric("tri_linear_direct_error", tri_linear)
            .metric("tri_quadratic_direct_error", tri_quadratic)
            .metric("quad_linear_direct_error", quad_linear)
            .metric("quad_quadratic_direct_error", quad_quadratic)
            .maximum("tri_linear_direct_error", 1.0e-10)
            .maximum("tri_quadratic_direct_error", 1.0e-10)
            .maximum("quad_linear_direct_error", 1.0e-10)
            .maximum("quad_quadratic_direct_error", 1.0e-10)
            .seed("bem-galerkin-hypersingular-open-boundary:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if !matches!(result.status, QaStatus::Passed) {
        std::process::exit(1);
    }
}
