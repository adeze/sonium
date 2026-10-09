//! Bounded CPU evidence for the separated curved-Quad4 Galerkin Müller block.

use std::time::Instant;

use num_complex::Complex64;
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_bem::{
    BasisOrder, GalerkinQuadratureAdmissionError, GalerkinQuadraturePolicy,
    GalerkinTransmissionCompressionError, PhysicsParams,
    assemble_galerkin_muller_transmission_quad4_pair,
    assemble_galerkin_muller_transmission_with_global_dofs,
    assemble_galerkin_muller_transmission_with_global_dofs_checked,
    compress_galerkin_muller_transmission_with_global_dofs_checked,
};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "bem.galerkin-quad4-transmission-pair.v1";

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

fn media() -> Result<(PhysicsParams, PhysicsParams), String> {
    Ok((
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?,
        PhysicsParams::try_new(420.0, 1_480.0, 1_000.0, true).map_err(|error| error.to_string())?,
    ))
}

fn nodes(curved: bool) -> Vec<[f64; 3]> {
    let bend = if curved { 0.3 } else { 0.0 };
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, bend],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.2],
        [1.0, 0.0, 1.2],
        [1.0, 1.0, 1.2 + bend],
        [0.0, 1.0, 1.2],
    ]
}

fn closed_cube() -> (Vec<Element>, Vec<[f64; 3]>) {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let connectivity = [
        vec![0, 1, 2, 3],
        vec![4, 7, 6, 5],
        vec![0, 4, 5, 1],
        vec![1, 5, 6, 2],
        vec![2, 6, 7, 3],
        vec![3, 7, 4, 0],
    ];
    let elements = connectivity
        .into_iter()
        .map(|connectivity| {
            let dof_addresses = connectivity.clone();
            quad(connectivity, dof_addresses, [0.5, 0.5, 0.5])
        })
        .collect();
    (elements, nodes)
}

fn run() -> Result<(usize, usize, usize, f64, bool, bool), String> {
    let (exterior, interior) = media()?;
    let curved_nodes = nodes(true);
    let mut curved = vec![
        quad(vec![0, 1, 2, 3], (0..4).collect(), [0.5, 0.5, 0.0]),
        quad(vec![4, 5, 6, 7], (0..4).collect(), [0.5, 0.5, 1.2]),
    ];
    let curved_linear = assemble_galerkin_muller_transmission_quad4_pair(
        &curved[0],
        &curved[1],
        &curved_nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        6,
    )
    .map_err(|error| error.to_string())?;
    if curved_linear
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("curved linear transmission block was non-finite".into());
    }

    let mut quadratic = curved.clone();
    for element in &mut quadratic {
        element.dof_addresses = (0..9).collect();
    }
    let quadratic_block = assemble_galerkin_muller_transmission_quad4_pair(
        &quadratic[0],
        &quadratic[1],
        &curved_nodes,
        &exterior,
        &interior,
        BasisOrder::Quadratic,
        6,
    )
    .map_err(|error| error.to_string())?;
    if quadratic_block
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("curved quadratic transmission block was non-finite".into());
    }

    let flat_nodes = nodes(false);
    let flat = assemble_galerkin_muller_transmission_quad4_pair(
        &curved[0],
        &curved[1],
        &flat_nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        6,
    )
    .map_err(|error| error.to_string())?;
    let geometry_delta = curved_linear
        .as_slice()
        .iter()
        .zip(flat.as_slice())
        .map(|(curved, flat)| (*curved - *flat).norm())
        .fold(0.0, f64::max);
    if geometry_delta <= 1.0e-6 {
        return Err(format!(
            "curved geometry delta {geometry_delta:e} too small"
        ));
    }

    curved[1].connectivity = vec![1, 4, 5, 2];
    let touching = assemble_galerkin_muller_transmission_quad4_pair(
        &curved[0],
        &curved[1],
        &curved_nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        6,
    )
    .map_err(|error| error.to_string())?;
    if touching.shape() != (8, 8)
        || touching
            .as_slice()
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("touching curved pair was not finite 8x8".into());
    }

    let self_block = assemble_galerkin_muller_transmission_quad4_pair(
        &curved[0],
        &curved[0],
        &curved_nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        6,
    )
    .map_err(|error| error.to_string())?;
    if self_block.shape() != (8, 8)
        || self_block
            .as_slice()
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        || !self_block
            .as_slice()
            .iter()
            .any(|value| value.norm() > 1.0e-12)
    {
        return Err("self curved pair was not finite and nonzero 8x8".into());
    }

    let (cube_elements, cube_nodes) = closed_cube();
    let global = assemble_galerkin_muller_transmission_with_global_dofs(
        &cube_elements,
        &cube_nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        2,
        8,
    )
    .map_err(|error| error.to_string())?;
    if global
        .matrix()
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("closed Quad4 global Müller matrix was non-finite".into());
    }

    let policy = GalerkinQuadraturePolicy {
        coarse_order: 1,
        fine_order: 2,
        absolute_tolerance: 1.0e6,
        relative_tolerance: 1.0e6,
    };
    let checked_global = assemble_galerkin_muller_transmission_with_global_dofs_checked(
        &cube_elements,
        &cube_nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        8,
        policy,
    )
    .map_err(|error| error.to_string())?;
    if checked_global
        .matrix()
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("checked Quad4 global Müller matrix was non-finite".into());
    }
    let strict = GalerkinQuadraturePolicy {
        absolute_tolerance: 0.0,
        relative_tolerance: 0.0,
        ..policy
    };
    let strict_rejected = matches!(
        assemble_galerkin_muller_transmission_with_global_dofs_checked(
            &cube_elements,
            &cube_nodes,
            &exterior,
            &interior,
            BasisOrder::Linear,
            8,
            strict,
        ),
        Err(GalerkinQuadratureAdmissionError::NotConverged(_))
    );
    if !strict_rejected {
        return Err("strict Quad4 global quadrature policy was not rejected".into());
    }
    let compressed = compress_galerkin_muller_transmission_with_global_dofs_checked(
        &cube_elements,
        &cube_nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        8,
        policy,
        2,
        1.0e-6,
        4,
    )
    .map_err(|error| error.to_string())?;
    let trial = vec![Complex64::new(0.1, -0.02); compressed.total_dofs()];
    let action = sonium_solvers::traits::LinearOperator::apply(&compressed, &trial);
    if action
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("checked Quad4 compressed action was non-finite".into());
    }
    let strict_compression_rejected = matches!(
        compress_galerkin_muller_transmission_with_global_dofs_checked(
            &cube_elements,
            &cube_nodes,
            &exterior,
            &interior,
            BasisOrder::Linear,
            8,
            strict,
            2,
            1.0e-6,
            4,
        ),
        Err(GalerkinTransmissionCompressionError::Quadrature(
            GalerkinQuadratureAdmissionError::NotConverged(_)
        ))
    );
    if !strict_compression_rejected {
        return Err("strict Quad4 compression policy was not rejected".into());
    }

    Ok((
        curved_linear.nrows(),
        quadratic_block.nrows(),
        global.matrix().nrows(),
        geometry_delta,
        strict_rejected,
        strict_compression_rejected,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((
            linear_dimension,
            quadratic_dimension,
            global_dimension,
            geometry_delta,
            strict_rejected,
            strict_compression_rejected,
        )) => QaResult::passed(CASE_ID)
            .metric("linear_matrix_dimension", linear_dimension as f64)
            .metric("quadratic_matrix_dimension", quadratic_dimension as f64)
            .metric("global_matrix_dimension", global_dimension as f64)
            .metric("curved_geometry_delta", geometry_delta)
            .metric(
                "strict_quadrature_rejected",
                if strict_rejected { 1.0 } else { 0.0 },
            )
            .metric(
                "strict_compression_rejected",
                if strict_compression_rejected {
                    1.0
                } else {
                    0.0
                },
            )
            .minimum("curved_geometry_delta", 1.0e-6)
            .input_checksum("analytic:separated-curved-quad4-muller-v1")
            .reference_checksum("analytic:separated-curved-quad4-muller-weak-form-v1")
            .dimension("3-D separated isoparametric Quad4 two-trace Galerkin block")
            .formulation(
                "bilinear/biquadratic physical-map V-K-Kprime-W composition; touching rejection",
            )
            .seed("bem-galerkin-quad4-transmission-pair:cpu:v1"),
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
