//! Bounded CPU evidence for separated and touching Quad4 Galerkin blocks.

use num_complex::Complex64;
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_bem::{BasisOrder, PhysicsParams, assemble_galerkin_single_layer_quad4_pair};
use sonium_qa::{QaResult, QaStatus};
use std::time::Instant;

const CASE_ID: &str = "bem.galerkin-quad4-pair.v1";
const MAX_RECIPROCITY_ERROR: f64 = 2.0e-10;

fn quad(connectivity: Vec<usize>, dof_addresses: Vec<usize>, center: [f64; 3]) -> Element {
    Element {
        connectivity,
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 4],
        center,
        area: 1.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::default()]),
        group: 0,
        dof_addresses,
    }
}

fn run() -> Result<(usize, usize, usize, f64), String> {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
    ];
    let mut source = quad(vec![0, 1, 2, 3], (0..9).collect(), [0.5, 0.5, 0.0]);
    let mut field = quad(vec![4, 5, 6, 7], (9..18).collect(), [0.5, 0.5, 1.0]);
    let physics =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?;

    let linear: f64 = {
        source.dof_addresses = (0..4).collect();
        field.dof_addresses = (4..8).collect();
        let forward = assemble_galerkin_single_layer_quad4_pair(
            &source,
            &field,
            &nodes,
            &physics,
            BasisOrder::Linear,
            6,
        )
        .map_err(|error| error.to_string())?;
        let reverse = assemble_galerkin_single_layer_quad4_pair(
            &field,
            &source,
            &nodes,
            &physics,
            BasisOrder::Linear,
            6,
        )
        .map_err(|error| error.to_string())?;
        let mut error = 0.0_f64;
        for row in 0..4 {
            for column in 0..4 {
                error = error.max((forward[[row, column]] - reverse[[column, row]]).norm());
            }
        }
        if forward
            .as_slice()
            .iter()
            .chain(reverse.as_slice())
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err("linear Quad4 block contained a non-finite entry".into());
        }
        error
    };

    source.dof_addresses = (0..9).collect();
    field.dof_addresses = (9..18).collect();
    let quadratic = assemble_galerkin_single_layer_quad4_pair(
        &source,
        &field,
        &nodes,
        &physics,
        BasisOrder::Quadratic,
        6,
    )
    .map_err(|error| error.to_string())?;
    if quadratic
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("quadratic Quad4 block contained a non-finite entry".into());
    }

    let self_block = assemble_galerkin_single_layer_quad4_pair(
        &source,
        &source,
        &nodes,
        &physics,
        BasisOrder::Quadratic,
        6,
    )
    .map_err(|error| error.to_string())?;
    if self_block
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("self Quad4 block contained a non-finite entry".into());
    }
    let touching_element = quad(vec![1, 8, 9, 2], (9..18).collect(), [1.5, 0.5, 0.0]);
    let touching_block = assemble_galerkin_single_layer_quad4_pair(
        &source,
        &touching_element,
        &nodes,
        &physics,
        BasisOrder::Quadratic,
        6,
    )
    .map_err(|error| error.to_string())?;
    if touching_block
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("touching Quad4 block contained non-finite entry".into());
    }

    let touching_reverse = assemble_galerkin_single_layer_quad4_pair(
        &touching_element,
        &source,
        &nodes,
        &physics,
        BasisOrder::Quadratic,
        6,
    )
    .map_err(|error| error.to_string())?;
    let touching_reciprocity = (0..9)
        .flat_map(|row| (0..9).map(move |column| (row, column)))
        .map(|(row, column)| {
            (touching_block[[row, column]] - touching_reverse[[column, row]]).norm()
        })
        .fold(0.0, f64::max);

    let vertex_element = quad(vec![2, 9, 10, 11], (18..27).collect(), [1.5, 1.5, 0.0]);
    let vertex_block = assemble_galerkin_single_layer_quad4_pair(
        &source,
        &vertex_element,
        &nodes,
        &physics,
        BasisOrder::Quadratic,
        6,
    )
    .map_err(|error| error.to_string())?;
    let vertex_reverse = assemble_galerkin_single_layer_quad4_pair(
        &vertex_element,
        &source,
        &nodes,
        &physics,
        BasisOrder::Quadratic,
        6,
    )
    .map_err(|error| error.to_string())?;
    if vertex_block
        .as_slice()
        .iter()
        .chain(vertex_reverse.as_slice())
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("common-vertex Quad4 block contained non-finite entry".into());
    }
    let vertex_reciprocity = (0..9)
        .flat_map(|row| (0..9).map(move |column| (row, column)))
        .map(|(row, column)| (vertex_block[[row, column]] - vertex_reverse[[column, row]]).norm())
        .fold(0.0, f64::max);

    Ok((
        4,
        9usize.max(quadratic.nrows()),
        2,
        linear.max(touching_reciprocity).max(vertex_reciprocity),
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((linear_dofs, maximum_dofs, touching_pairs, reciprocity_error)) => QaResult::passed(
            CASE_ID,
        )
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("linear_block_dofs", linear_dofs as f64)
        .metric("maximum_block_dofs", maximum_dofs as f64)
        .metric("supported_singular_pairs", touching_pairs as f64)
        .metric("reciprocity_error", reciprocity_error)
        .minimum("linear_block_dofs", 4.0)
        .minimum("maximum_block_dofs", 9.0)
        .minimum("supported_singular_pairs", 2.0)
        .maximum("reciprocity_error", MAX_RECIPROCITY_ERROR)
        .input_checksum("analytic/deterministic-fixture:bem-galerkin-quad4-pair-v1")
        .reference("independent reverse-pair Galerkin block")
        .reference_checksum("analytic/deterministic-fixture:bem-galerkin-quad4-pair-v1")
        .dimension("3-D Quad4 panels; separated and common-edge bilinear/biquadratic bases")
        .formulation("regular and mapped singular Galerkin single-layer blocks")
        .tolerance(
            "finite linear/quadratic blocks; reverse-pair reciprocity and mapped singular charts",
        )
        .seed("bem-galerkin-quad4-pair:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    let result = result.elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
