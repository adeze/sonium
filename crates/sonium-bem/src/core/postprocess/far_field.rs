//! Far-field radiation from constant-panel Cauchy data.
//!
//! For the shared `e^{-iωt}` convention, this module returns the amplitude
//! `F(d)` defined by
//!
//! ```text
//! p_s(r d) = exp(i k r) F(d) / (4π r) + O(r⁻²),
//! ```
//!
//! where `d` is the unit observation direction.  The implementation is a
//! bounded constant-panel production slice: it uses the same flat-panel
//! geometry and Gaussian rules as the existing field evaluator, but avoids
//! evaluating at an arbitrarily large radius.  Linear panels and near-field
//! evaluation remain outside this API.

use num_complex::Complex64;
use thiserror::Error;

use crate::core::integration::gauss::{quad_quadrature, triangle_quadrature};
use crate::core::types::{Element, ElementType, PhysicsParams};

/// Failure while evaluating a constant-panel far-field amplitude.
#[derive(Debug, Error, PartialEq)]
pub enum FarFieldError {
    /// No observation directions were supplied.
    #[error("far-field evaluation requires at least one observation direction")]
    EmptyDirections,
    /// A direction was non-finite or had zero length.
    #[error("observation direction {index} must be finite and non-zero")]
    InvalidDirection {
        /// Index of the invalid observation direction.
        index: usize,
    },
    /// The mesh has no physical boundary panels.
    #[error("far-field evaluation requires at least one boundary element")]
    EmptyBoundary,
    /// A boundary panel has malformed connectivity.
    #[error("boundary element {element} has {actual} connectivity entries; expected {expected}")]
    InvalidConnectivity {
        /// Index of the malformed boundary element.
        element: usize,
        /// Supplied connectivity length.
        actual: usize,
        /// Required connectivity length.
        expected: usize,
    },
    /// A panel refers to a node outside the supplied coordinate array.
    #[error("boundary element {element} references node {node} outside the mesh")]
    InvalidNodeIndex {
        /// Index of the boundary element with invalid connectivity.
        element: usize,
        /// Out-of-range node index.
        node: usize,
    },
    /// A panel is geometrically degenerate.
    #[error("boundary element {element} is geometrically degenerate")]
    DegenerateElement {
        /// Index of the geometrically degenerate boundary element.
        element: usize,
    },
    /// The constant-panel pressure vector does not match the boundary.
    #[error("surface pressure has length {actual}; expected {expected} boundary panels")]
    SurfacePressureCount {
        /// Number of boundary panels.
        expected: usize,
        /// Supplied pressure sample count.
        actual: usize,
    },
    /// The optional normal-velocity vector does not match the boundary.
    #[error("surface velocity has length {actual}; expected {expected} boundary panels")]
    SurfaceVelocityCount {
        /// Number of boundary panels.
        expected: usize,
        /// Supplied velocity sample count.
        actual: usize,
    },
    /// A scalar or complex sample was not finite.
    #[error("far-field input contains a non-finite {label}")]
    NonFiniteInput {
        /// Name of the non-finite input quantity.
        label: &'static str,
    },
    /// The current implementation is limited to constant panels.
    #[error("far-field evaluation currently supports constant panels only")]
    UnsupportedBasis,
}

/// Compute constant-panel far-field amplitudes for unit observation directions.
///
/// The returned value is independent of receiver radius and uses the outgoing
/// Green-function normalization described at the top of this module.  A
/// supplied velocity is the physical normal particle velocity; `None` is the
/// rigid-boundary case.  The panel pressure is the scattered pressure used by
/// the existing constant-panel field representation.
///
/// Directions are normalized internally, so callers may provide any finite
/// non-zero vectors.  The checked path rejects malformed geometry, samples,
/// and physical parameters before quadrature.
pub fn compute_constant_panel_far_field_checked(
    directions: &[[f64; 3]],
    elements: &[Element],
    nodes: &[[f64; 3]],
    surface_pressure: &[Complex64],
    surface_velocity: Option<&[Complex64]>,
    physics: &PhysicsParams,
) -> Result<Vec<Complex64>, FarFieldError> {
    if directions.is_empty() {
        return Err(FarFieldError::EmptyDirections);
    }
    physics
        .validate()
        .map_err(|_| FarFieldError::NonFiniteInput { label: "physics" })?;

    for (index, direction) in directions.iter().enumerate() {
        let norm = direction
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm <= f64::MIN_POSITIVE {
            return Err(FarFieldError::InvalidDirection { index });
        }
    }
    if nodes.iter().flatten().any(|value| !value.is_finite()) {
        return Err(FarFieldError::NonFiniteInput { label: "node" });
    }

    let boundary_elements: Vec<(usize, &Element)> = elements
        .iter()
        .enumerate()
        .filter(|(_, element)| !element.property.is_evaluation())
        .collect();
    if boundary_elements.is_empty() {
        return Err(FarFieldError::EmptyBoundary);
    }
    if surface_pressure.len() != boundary_elements.len() {
        return Err(FarFieldError::SurfacePressureCount {
            expected: boundary_elements.len(),
            actual: surface_pressure.len(),
        });
    }
    if let Some(velocity) = surface_velocity {
        if velocity.len() != boundary_elements.len() {
            return Err(FarFieldError::SurfaceVelocityCount {
                expected: boundary_elements.len(),
                actual: velocity.len(),
            });
        }
        if velocity
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(FarFieldError::NonFiniteInput {
                label: "surface velocity",
            });
        }
    }
    if surface_pressure
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(FarFieldError::NonFiniteInput {
            label: "surface pressure",
        });
    }

    for &(element_index, element) in &boundary_elements {
        validate_element(element_index, element, nodes)?;
    }

    directions
        .iter()
        .map(|direction| {
            let norm = direction
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            let unit_direction = [
                direction[0] / norm,
                direction[1] / norm,
                direction[2] / norm,
            ];
            boundary_elements.iter().enumerate().try_fold(
                Complex64::new(0.0, 0.0),
                |amplitude, (panel_index, &(element_index, element))| {
                    panel_amplitude(
                        element_index,
                        element,
                        nodes,
                        unit_direction,
                        surface_pressure[panel_index],
                        surface_velocity.map(|velocity| velocity[panel_index]),
                        physics,
                    )
                    .map(|panel| amplitude + panel)
                },
            )
        })
        .collect()
}

fn validate_element(
    element_index: usize,
    element: &Element,
    nodes: &[[f64; 3]],
) -> Result<(), FarFieldError> {
    let expected = element.element_type.num_nodes();
    if element.connectivity.len() != expected {
        return Err(FarFieldError::InvalidConnectivity {
            element: element_index,
            actual: element.connectivity.len(),
            expected,
        });
    }
    if let Some(&node) = element
        .connectivity
        .iter()
        .find(|&&node| node >= nodes.len())
    {
        return Err(FarFieldError::InvalidNodeIndex {
            element: element_index,
            node,
        });
    }
    if element
        .connectivity
        .iter()
        .enumerate()
        .any(|(local, &node)| element.connectivity[..local].contains(&node))
    {
        return Err(FarFieldError::InvalidConnectivity {
            element: element_index,
            actual: element.connectivity.len(),
            expected,
        });
    }
    Ok(())
}

fn panel_amplitude(
    element_index: usize,
    element: &Element,
    nodes: &[[f64; 3]],
    direction: [f64; 3],
    pressure: Complex64,
    velocity: Option<Complex64>,
    physics: &PhysicsParams,
) -> Result<Complex64, FarFieldError> {
    let coordinates: Vec<[f64; 3]> = element
        .connectivity
        .iter()
        .map(|&node| nodes[node])
        .collect();
    let velocity = velocity.unwrap_or_else(|| Complex64::new(0.0, 0.0));
    let mut result = Complex64::new(0.0, 0.0);

    match element.element_type {
        ElementType::Tri3 => {
            let shape_ds = [-1.0, 1.0, 0.0];
            let shape_dt = [-1.0, 0.0, 1.0];
            for (xi, eta, weight) in triangle_quadrature(3) {
                let shape = [1.0 - xi - eta, xi, eta];
                result += quadrature_contribution(
                    &coordinates,
                    &shape,
                    &shape_ds,
                    &shape_dt,
                    weight,
                    direction,
                    pressure,
                    velocity,
                    physics,
                    element_index,
                )?;
            }
        }
        ElementType::Quad4 => {
            for (xi, eta, weight) in quad_quadrature(2) {
                let shape = [
                    0.25 * (1.0 - xi) * (1.0 - eta),
                    0.25 * (1.0 + xi) * (1.0 - eta),
                    0.25 * (1.0 + xi) * (1.0 + eta),
                    0.25 * (1.0 - xi) * (1.0 + eta),
                ];
                let shape_ds = [
                    -0.25 * (1.0 - eta),
                    0.25 * (1.0 - eta),
                    0.25 * (1.0 + eta),
                    -0.25 * (1.0 + eta),
                ];
                let shape_dt = [
                    -0.25 * (1.0 - xi),
                    -0.25 * (1.0 + xi),
                    0.25 * (1.0 + xi),
                    0.25 * (1.0 - xi),
                ];
                result += quadrature_contribution(
                    &coordinates,
                    &shape,
                    &shape_ds,
                    &shape_dt,
                    weight,
                    direction,
                    pressure,
                    velocity,
                    physics,
                    element_index,
                )?;
            }
        }
    }
    Ok(result)
}

fn quadrature_contribution(
    coordinates: &[[f64; 3]],
    shape: &[f64],
    shape_ds: &[f64],
    shape_dt: &[f64],
    weight: f64,
    direction: [f64; 3],
    pressure: Complex64,
    velocity: Complex64,
    physics: &PhysicsParams,
    element_index: usize,
) -> Result<Complex64, FarFieldError> {
    let point = weighted_point(coordinates, shape);
    let tangent_s = weighted_point(coordinates, shape_ds);
    let tangent_t = weighted_point(coordinates, shape_dt);
    let cross = cross_product(tangent_s, tangent_t);
    let jacobian = norm(cross);
    if !jacobian.is_finite() || jacobian <= 1.0e-15 {
        return Err(FarFieldError::DegenerateElement {
            element: element_index,
        });
    }
    let normal = [
        cross[0] / jacobian,
        cross[1] / jacobian,
        cross[2] / jacobian,
    ];
    let phase = -physics.harmonic_factor * physics.wave_number * dot(direction, point);
    let double_layer = Complex64::new(
        0.0,
        -physics.harmonic_factor * physics.wave_number * dot(direction, normal),
    ) * pressure;
    let single_layer = Complex64::new(0.0, -physics.pressure_factor) * velocity;
    Ok(sonium_fastmath::exp_i(phase) * (double_layer + single_layer) * (jacobian * weight))
}

fn weighted_point(coordinates: &[[f64; 3]], weights: &[f64]) -> [f64; 3] {
    let mut point = [0.0; 3];
    for (coordinate, &weight) in coordinates.iter().zip(weights) {
        for axis in 0..3 {
            point[axis] += coordinate[axis] * weight;
        }
    }
    point
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross_product(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}
