//! Checked quadratic scalar natural-boundary integration.
//!
//! The legacy checked natural-boundary entry points intentionally implement
//! the P1 node/edge/face contract.  This module supplies the next explicit
//! contract: P2 Lagrange traces on quadratic 2-D edges and triangular 3-D
//! faces, including the isoparametric geometry Jacobian.

use super::generic::{BoundaryConditionError, FemSystem};
use crate::mesh::Mesh;
use crate::quadrature::{gauss_legendre_1d, gauss_triangle};
use num_complex::Complex64;
use std::ops::{AddAssign, Mul};

#[derive(Debug, Clone)]
struct TracePoint {
    boundary: usize,
    nodes: Vec<usize>,
    shape: Vec<f64>,
    x: f64,
    y: f64,
    z: f64,
    weight: f64,
}

/// Apply a real P2 Neumann flux on marked quadratic edges/faces.
///
/// This API expects global P2 scalar DOF numbering to use the mesh node IDs.
/// In 2-D each marked trace has three nodes `[v0, v1, edge_midpoint]`; in
/// 3-D each marked triangular trace has six nodes `[v0, v1, v2, e01, e12,
/// e20]`. All validation and callback evaluation completes before the target
/// RHS is modified.
pub fn try_apply_neumann_real_p2(
    system: &mut FemSystem<f64>,
    mesh: &Mesh,
    marker: i32,
    flux_fn: &dyn Fn(f64, f64, f64) -> f64,
) -> Result<(), BoundaryConditionError> {
    try_apply_neumann_p2(system, mesh, marker, flux_fn, f64::is_finite)
}

/// Apply a complex P2 Neumann flux on marked quadratic edges/faces.
pub fn try_apply_neumann_complex_p2(
    system: &mut FemSystem<Complex64>,
    mesh: &Mesh,
    marker: i32,
    flux_fn: &dyn Fn(f64, f64, f64) -> Complex64,
) -> Result<(), BoundaryConditionError> {
    try_apply_neumann_p2(system, mesh, marker, flux_fn, |value| {
        value.re.is_finite() && value.im.is_finite()
    })
}

/// Apply a real P3 Neumann flux on explicitly supplied cubic edges/faces.
///
/// The trace node order is `[v0, v1, e01_1, e01_2]` for a 2-D edge and
/// `[v0, v1, v2, e01_1, e01_2, e12_1, e12_2, e20_1, e20_2, face]` for a
/// 3-D triangular face.
pub fn try_apply_neumann_real_p3(
    system: &mut FemSystem<f64>,
    mesh: &Mesh,
    marker: i32,
    flux_fn: &dyn Fn(f64, f64, f64) -> f64,
) -> Result<(), BoundaryConditionError> {
    try_apply_neumann_p3(system, mesh, marker, flux_fn, f64::is_finite)
}

/// Apply a complex P3 Neumann flux on explicitly supplied cubic edges/faces.
pub fn try_apply_neumann_complex_p3(
    system: &mut FemSystem<Complex64>,
    mesh: &Mesh,
    marker: i32,
    flux_fn: &dyn Fn(f64, f64, f64) -> Complex64,
) -> Result<(), BoundaryConditionError> {
    try_apply_neumann_p3(system, mesh, marker, flux_fn, |value| {
        value.re.is_finite() && value.im.is_finite()
    })
}

/// Apply a complex P2 Robin term on marked quadratic edges/faces.
///
/// The contribution is `alpha * ∫ phi_i phi_j` to the matrix and
/// `∫ g phi_i` to the RHS. The operation is transactional on geometry,
/// callback, and DOF validation failure.
pub fn try_apply_robin_complex_p2(
    system: &mut FemSystem<Complex64>,
    mesh: &Mesh,
    marker: i32,
    alpha_fn: &dyn Fn(f64, f64, f64) -> Complex64,
    g_fn: &dyn Fn(f64, f64, f64) -> Complex64,
) -> Result<(), BoundaryConditionError> {
    system.validate()?;
    let samples = p2_trace_quadrature(mesh, marker, system.rhs.len())?;
    let mut rhs_delta = vec![Complex64::new(0.0, 0.0); system.rhs.len()];
    let mut matrix_terms = Vec::new();
    for sample in samples {
        let alpha = alpha_fn(sample.x, sample.y, sample.z);
        let g = g_fn(sample.x, sample.y, sample.z);
        if ![alpha.re, alpha.im, g.re, g.im]
            .into_iter()
            .all(f64::is_finite)
        {
            return Err(BoundaryConditionError::NonFiniteBoundaryValue {
                boundary: sample.boundary,
            });
        }
        for (local_i, &row) in sample.nodes.iter().enumerate() {
            rhs_delta[row] += g * (sample.shape[local_i] * sample.weight);
            for (local_j, &column) in sample.nodes.iter().enumerate() {
                matrix_terms.push((
                    row,
                    column,
                    alpha * (sample.shape[local_i] * sample.shape[local_j] * sample.weight),
                ));
            }
        }
    }
    for (target, delta) in system.rhs.iter_mut().zip(rhs_delta) {
        *target += delta;
    }
    for (row, column, value) in matrix_terms {
        system.matrix.push(row, column, value);
    }
    Ok(())
}

/// Apply a complex P3 Robin term on explicitly supplied cubic edges/faces.
pub fn try_apply_robin_complex_p3(
    system: &mut FemSystem<Complex64>,
    mesh: &Mesh,
    marker: i32,
    alpha_fn: &dyn Fn(f64, f64, f64) -> Complex64,
    g_fn: &dyn Fn(f64, f64, f64) -> Complex64,
) -> Result<(), BoundaryConditionError> {
    system.validate()?;
    let samples = p3_trace_quadrature(mesh, marker, system.rhs.len())?;
    let mut rhs_delta = vec![Complex64::new(0.0, 0.0); system.rhs.len()];
    let mut matrix_terms = Vec::new();
    for sample in samples {
        let alpha = alpha_fn(sample.x, sample.y, sample.z);
        let g = g_fn(sample.x, sample.y, sample.z);
        if ![alpha.re, alpha.im, g.re, g.im]
            .into_iter()
            .all(f64::is_finite)
        {
            return Err(BoundaryConditionError::NonFiniteBoundaryValue {
                boundary: sample.boundary,
            });
        }
        for (local_i, &row) in sample.nodes.iter().enumerate() {
            rhs_delta[row] += g * (sample.shape[local_i] * sample.weight);
            for (local_j, &column) in sample.nodes.iter().enumerate() {
                matrix_terms.push((
                    row,
                    column,
                    alpha * (sample.shape[local_i] * sample.shape[local_j] * sample.weight),
                ));
            }
        }
    }
    for (target, delta) in system.rhs.iter_mut().zip(rhs_delta) {
        *target += delta;
    }
    for (row, column, value) in matrix_terms {
        system.matrix.push(row, column, value);
    }
    Ok(())
}

fn try_apply_neumann_p2<T>(
    system: &mut FemSystem<T>,
    mesh: &Mesh,
    marker: i32,
    flux_fn: &dyn Fn(f64, f64, f64) -> T,
    is_finite: impl Fn(T) -> bool,
) -> Result<(), BoundaryConditionError>
where
    T: Copy + Default + AddAssign + Mul<f64, Output = T> + Send + Sync,
{
    system.validate()?;
    let samples = p2_trace_quadrature(mesh, marker, system.rhs.len())?;
    let mut rhs_delta = vec![T::default(); system.rhs.len()];
    for sample in samples {
        let flux = flux_fn(sample.x, sample.y, sample.z);
        if !is_finite(flux) {
            return Err(BoundaryConditionError::NonFiniteBoundaryValue {
                boundary: sample.boundary,
            });
        }
        for (local_index, &node) in sample.nodes.iter().enumerate() {
            rhs_delta[node] += flux * (sample.shape[local_index] * sample.weight);
        }
    }
    for (target, delta) in system.rhs.iter_mut().zip(rhs_delta) {
        *target += delta;
    }
    Ok(())
}

fn try_apply_neumann_p3<T>(
    system: &mut FemSystem<T>,
    mesh: &Mesh,
    marker: i32,
    flux_fn: &dyn Fn(f64, f64, f64) -> T,
    is_finite: impl Fn(T) -> bool,
) -> Result<(), BoundaryConditionError>
where
    T: Copy + Default + AddAssign + Mul<f64, Output = T> + Send + Sync,
{
    system.validate()?;
    let samples = p3_trace_quadrature(mesh, marker, system.rhs.len())?;
    let mut rhs_delta = vec![T::default(); system.rhs.len()];
    for sample in samples {
        let flux = flux_fn(sample.x, sample.y, sample.z);
        if !is_finite(flux) {
            return Err(BoundaryConditionError::NonFiniteBoundaryValue {
                boundary: sample.boundary,
            });
        }
        for (local_index, &node) in sample.nodes.iter().enumerate() {
            rhs_delta[node] += flux * (sample.shape[local_index] * sample.weight);
        }
    }
    for (target, delta) in system.rhs.iter_mut().zip(rhs_delta) {
        *target += delta;
    }
    Ok(())
}

fn p2_trace_quadrature(
    mesh: &Mesh,
    marker: i32,
    num_dofs: usize,
) -> Result<Vec<TracePoint>, BoundaryConditionError> {
    let mut samples = Vec::new();
    for (boundary_index, boundary) in mesh.boundaries.iter().enumerate() {
        if boundary.marker != marker {
            continue;
        }
        let expected_nodes = match (mesh.dimension, boundary.nodes.len()) {
            (2, 3) | (3, 6) => boundary.nodes.len(),
            _ => {
                return Err(BoundaryConditionError::UnsupportedBoundary {
                    boundary: boundary_index,
                    nodes: boundary.nodes.len(),
                });
            }
        };
        for &node in &boundary.nodes {
            if node >= mesh.nodes.len() {
                return Err(BoundaryConditionError::MeshNodeOutOfBounds {
                    boundary: boundary_index,
                    node,
                    num_nodes: mesh.nodes.len(),
                });
            }
            if node >= num_dofs {
                return Err(BoundaryConditionError::BoundaryNodeOutOfBounds {
                    boundary: boundary_index,
                    node,
                    num_dofs,
                });
            }
            let point = mesh.nodes[node];
            if ![point.x, point.y, point.z].into_iter().all(f64::is_finite) {
                return Err(BoundaryConditionError::InvalidBoundaryGeometry {
                    boundary: boundary_index,
                });
            }
        }

        if mesh.dimension == 2 {
            let [p0, p1, pm] = [
                mesh.nodes[boundary.nodes[0]],
                mesh.nodes[boundary.nodes[1]],
                mesh.nodes[boundary.nodes[2]],
            ];
            for quadrature_point in gauss_legendre_1d(4) {
                let t = 0.5 * (quadrature_point.xi() + 1.0);
                let shape = [
                    (1.0 - t) * (1.0 - 2.0 * t),
                    t * (2.0 * t - 1.0),
                    4.0 * t * (1.0 - t),
                ];
                let derivative = [-3.0 + 4.0 * t, -1.0 + 4.0 * t, 4.0 - 8.0 * t];
                let x = shape[0] * p0.x + shape[1] * p1.x + shape[2] * pm.x;
                let y = shape[0] * p0.y + shape[1] * p1.y + shape[2] * pm.y;
                let z = shape[0] * p0.z + shape[1] * p1.z + shape[2] * pm.z;
                let tangent = [
                    derivative[0] * p0.x + derivative[1] * p1.x + derivative[2] * pm.x,
                    derivative[0] * p0.y + derivative[1] * p1.y + derivative[2] * pm.y,
                    derivative[0] * p0.z + derivative[1] * p1.z + derivative[2] * pm.z,
                ];
                let jacobian = tangent
                    .into_iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                if !jacobian.is_finite() || jacobian <= 0.0 {
                    return Err(BoundaryConditionError::InvalidBoundaryGeometry {
                        boundary: boundary_index,
                    });
                }
                samples.push(TracePoint {
                    boundary: boundary_index,
                    nodes: boundary.nodes.clone(),
                    shape: shape.to_vec(),
                    x,
                    y,
                    z,
                    weight: 0.5 * quadrature_point.weight * jacobian,
                });
            }
        } else {
            let points: Vec<_> = boundary
                .nodes
                .iter()
                .map(|&node| mesh.nodes[node])
                .collect();
            for quadrature_point in gauss_triangle(4) {
                let xi = quadrature_point.xi();
                let eta = quadrature_point.eta();
                let l0 = 1.0 - xi - eta;
                let l1 = xi;
                let l2 = eta;
                let shape = [
                    l0 * (2.0 * l0 - 1.0),
                    l1 * (2.0 * l1 - 1.0),
                    l2 * (2.0 * l2 - 1.0),
                    4.0 * l0 * l1,
                    4.0 * l1 * l2,
                    4.0 * l2 * l0,
                ];
                let dxi = [
                    -(4.0 * l0 - 1.0),
                    4.0 * l1 - 1.0,
                    0.0,
                    4.0 * (l0 - l1),
                    4.0 * l2,
                    -4.0 * l2,
                ];
                let deta = [
                    -(4.0 * l0 - 1.0),
                    0.0,
                    4.0 * l2 - 1.0,
                    -4.0 * l1,
                    4.0 * l1,
                    4.0 * (l0 - l2),
                ];
                let position = |component: fn(&crate::mesh::Point) -> f64| {
                    shape
                        .iter()
                        .zip(points.iter())
                        .map(|(&value, point)| value * component(point))
                        .sum::<f64>()
                };
                let tangent_x = |component: fn(&crate::mesh::Point) -> f64| {
                    dxi.iter()
                        .zip(points.iter())
                        .map(|(&value, point)| value * component(point))
                        .sum::<f64>()
                };
                let tangent_y = |component: fn(&crate::mesh::Point) -> f64| {
                    deta.iter()
                        .zip(points.iter())
                        .map(|(&value, point)| value * component(point))
                        .sum::<f64>()
                };
                let p_x = position(|point| point.x);
                let p_y = position(|point| point.y);
                let p_z = position(|point| point.z);
                let dxi_vec = [
                    tangent_x(|point| point.x),
                    tangent_x(|point| point.y),
                    tangent_x(|point| point.z),
                ];
                let deta_vec = [
                    tangent_y(|point| point.x),
                    tangent_y(|point| point.y),
                    tangent_y(|point| point.z),
                ];
                let cross = [
                    dxi_vec[1] * deta_vec[2] - dxi_vec[2] * deta_vec[1],
                    dxi_vec[2] * deta_vec[0] - dxi_vec[0] * deta_vec[2],
                    dxi_vec[0] * deta_vec[1] - dxi_vec[1] * deta_vec[0],
                ];
                let jacobian = cross
                    .into_iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                if !jacobian.is_finite() || jacobian <= 0.0 {
                    return Err(BoundaryConditionError::InvalidBoundaryGeometry {
                        boundary: boundary_index,
                    });
                }
                samples.push(TracePoint {
                    boundary: boundary_index,
                    nodes: boundary.nodes.clone(),
                    shape: shape.to_vec(),
                    x: p_x,
                    y: p_y,
                    z: p_z,
                    weight: quadrature_point.weight * jacobian,
                });
            }
        }
        debug_assert_eq!(expected_nodes, boundary.nodes.len());
    }
    Ok(samples)
}

fn p3_trace_quadrature(
    mesh: &Mesh,
    marker: i32,
    num_dofs: usize,
) -> Result<Vec<TracePoint>, BoundaryConditionError> {
    let mut samples = Vec::new();
    for (boundary_index, boundary) in mesh.boundaries.iter().enumerate() {
        if boundary.marker != marker {
            continue;
        }
        let expected_nodes = match (mesh.dimension, boundary.nodes.len()) {
            (2, 4) | (3, 10) => boundary.nodes.len(),
            _ => {
                return Err(BoundaryConditionError::UnsupportedBoundary {
                    boundary: boundary_index,
                    nodes: boundary.nodes.len(),
                });
            }
        };
        for &node in &boundary.nodes {
            if node >= mesh.nodes.len() {
                return Err(BoundaryConditionError::MeshNodeOutOfBounds {
                    boundary: boundary_index,
                    node,
                    num_nodes: mesh.nodes.len(),
                });
            }
            if node >= num_dofs {
                return Err(BoundaryConditionError::BoundaryNodeOutOfBounds {
                    boundary: boundary_index,
                    node,
                    num_dofs,
                });
            }
            let point = mesh.nodes[node];
            if ![point.x, point.y, point.z].into_iter().all(f64::is_finite) {
                return Err(BoundaryConditionError::InvalidBoundaryGeometry {
                    boundary: boundary_index,
                });
            }
        }

        if mesh.dimension == 2 {
            let [p0, p1, p2, p3] = [
                mesh.nodes[boundary.nodes[0]],
                mesh.nodes[boundary.nodes[1]],
                mesh.nodes[boundary.nodes[2]],
                mesh.nodes[boundary.nodes[3]],
            ];
            let reference_nodes = [0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0];
            for quadrature_point in gauss_legendre_1d(6) {
                let t = 0.5 * (quadrature_point.xi() + 1.0);
                let (shape, derivative) = p3_edge_shape_values(t, reference_nodes);
                let points = [p0, p1, p2, p3];
                let x = shape
                    .iter()
                    .zip(points)
                    .map(|(&value, point)| value * point.x)
                    .sum::<f64>();
                let y = shape
                    .iter()
                    .zip(points)
                    .map(|(&value, point)| value * point.y)
                    .sum::<f64>();
                let z = shape
                    .iter()
                    .zip(points)
                    .map(|(&value, point)| value * point.z)
                    .sum::<f64>();
                let tangent = [
                    derivative
                        .iter()
                        .zip(points)
                        .map(|(&value, point)| value * point.x)
                        .sum::<f64>(),
                    derivative
                        .iter()
                        .zip(points)
                        .map(|(&value, point)| value * point.y)
                        .sum::<f64>(),
                    derivative
                        .iter()
                        .zip(points)
                        .map(|(&value, point)| value * point.z)
                        .sum::<f64>(),
                ];
                let jacobian = tangent
                    .into_iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                if !jacobian.is_finite() || jacobian <= 0.0 {
                    return Err(BoundaryConditionError::InvalidBoundaryGeometry {
                        boundary: boundary_index,
                    });
                }
                samples.push(TracePoint {
                    boundary: boundary_index,
                    nodes: boundary.nodes.clone(),
                    shape: shape.to_vec(),
                    x,
                    y,
                    z,
                    weight: 0.5 * quadrature_point.weight * jacobian,
                });
            }
        } else {
            let points: Vec<_> = boundary
                .nodes
                .iter()
                .map(|&node| mesh.nodes[node])
                .collect();
            for quadrature_point in gauss_triangle(4) {
                let xi = quadrature_point.xi();
                let eta = quadrature_point.eta();
                let l0 = 1.0 - xi - eta;
                let l1 = xi;
                let l2 = eta;
                let (shape, gradients) = p3_triangle_shape_values(l0, l1, l2);
                let position = |component: fn(&crate::mesh::Point) -> f64| {
                    shape
                        .iter()
                        .zip(points.iter())
                        .map(|(&value, point)| value * component(point))
                        .sum::<f64>()
                };
                let derivative = |component: usize| {
                    [
                        gradients
                            .iter()
                            .zip(points.iter())
                            .map(|(gradient, point)| {
                                gradient[0] * component_value(point, component)
                            })
                            .sum::<f64>(),
                        gradients
                            .iter()
                            .zip(points.iter())
                            .map(|(gradient, point)| {
                                gradient[1] * component_value(point, component)
                            })
                            .sum::<f64>(),
                    ]
                };
                let dxi = [derivative(0)[0], derivative(1)[0], derivative(2)[0]];
                let deta = [derivative(0)[1], derivative(1)[1], derivative(2)[1]];
                let cross = [
                    dxi[1] * deta[2] - dxi[2] * deta[1],
                    dxi[2] * deta[0] - dxi[0] * deta[2],
                    dxi[0] * deta[1] - dxi[1] * deta[0],
                ];
                let jacobian = cross
                    .into_iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                if !jacobian.is_finite() || jacobian <= 0.0 {
                    return Err(BoundaryConditionError::InvalidBoundaryGeometry {
                        boundary: boundary_index,
                    });
                }
                samples.push(TracePoint {
                    boundary: boundary_index,
                    nodes: boundary.nodes.clone(),
                    shape: shape.to_vec(),
                    x: position(|point| point.x),
                    y: position(|point| point.y),
                    z: position(|point| point.z),
                    weight: quadrature_point.weight * jacobian,
                });
            }
        }
        debug_assert_eq!(expected_nodes, boundary.nodes.len());
    }
    Ok(samples)
}

fn component_value(point: &crate::mesh::Point, component: usize) -> f64 {
    match component {
        0 => point.x,
        1 => point.y,
        2 => point.z,
        _ => unreachable!("P3 trace geometry has only three coordinates"),
    }
}

fn p3_edge_shape_values(t: f64, nodes: [f64; 4]) -> ([f64; 4], [f64; 4]) {
    let mut shape = [0.0; 4];
    let mut derivative = [0.0; 4];
    for i in 0..4 {
        let mut value = 1.0;
        for j in 0..4 {
            if i != j {
                value *= (t - nodes[j]) / (nodes[i] - nodes[j]);
            }
        }
        shape[i] = value;
        for k in 0..4 {
            if i == k {
                continue;
            }
            let mut term = 1.0 / (nodes[i] - nodes[k]);
            for j in 0..4 {
                if j != i && j != k {
                    term *= (t - nodes[j]) / (nodes[i] - nodes[j]);
                }
            }
            derivative[i] += term;
        }
    }
    (shape, derivative)
}

fn p3_triangle_shape_values(l0: f64, l1: f64, l2: f64) -> ([f64; 10], [[f64; 2]; 10]) {
    const MODES: [[usize; 3]; 10] = [
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
        [2, 1, 0],
        [1, 2, 0],
        [0, 2, 1],
        [0, 1, 2],
        [1, 0, 2],
        [2, 0, 1],
        [1, 1, 1],
    ];
    let barycentric = [l0, l1, l2];
    let mut shape = [0.0; 10];
    let mut gradients = [[0.0; 2]; 10];
    for (index, mode) in MODES.into_iter().enumerate() {
        let factors: [(f64, f64); 3] = [
            falling_factor(barycentric[0], mode[0]),
            falling_factor(barycentric[1], mode[1]),
            falling_factor(barycentric[2], mode[2]),
        ];
        shape[index] = factors[0].0 * factors[1].0 * factors[2].0;
        gradients[index][0] = -factors[0].1 * factors[1].0 * factors[2].0
            + factors[0].0 * factors[1].1 * factors[2].0;
        gradients[index][1] = -factors[0].1 * factors[1].0 * factors[2].0
            + factors[0].0 * factors[1].0 * factors[2].1;
    }
    (shape, gradients)
}

fn falling_factor(value: f64, count: usize) -> (f64, f64) {
    let denominator = (1..=count).product::<usize>().max(1) as f64;
    let mut factor = 1.0;
    for offset in 0..count {
        factor *= 3.0 * value - offset as f64;
    }
    let mut derivative = 0.0;
    for selected in 0..count {
        let mut term = 3.0;
        for offset in 0..count {
            if offset != selected {
                term *= 3.0 * value - offset as f64;
            }
        }
        derivative += term;
    }
    (factor / denominator, derivative / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::TripletMatrix;
    use crate::mesh::{BoundaryType, Point, unit_cube_tetrahedra, unit_square_triangles};

    fn p3_edge_mesh() -> Mesh {
        let mut mesh = Mesh::new(2);
        for point in [
            Point::new_2d(0.0, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(1.0 / 3.0, 0.0),
            Point::new_2d(2.0 / 3.0, 0.0),
        ] {
            mesh.add_node(point);
        }
        mesh.add_boundary(vec![0, 1, 2, 3], BoundaryType::Neumann, 0, 0, 0);
        mesh
    }

    fn p3_face_mesh() -> Mesh {
        let mut mesh = Mesh::new(3);
        for point in [
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(1.0 / 3.0, 0.0, 0.0),
            Point::new_3d(2.0 / 3.0, 0.0, 0.0),
            Point::new_3d(2.0 / 3.0, 1.0 / 3.0, 0.0),
            Point::new_3d(1.0 / 3.0, 2.0 / 3.0, 0.0),
            Point::new_3d(0.0, 2.0 / 3.0, 0.0),
            Point::new_3d(0.0, 1.0 / 3.0, 0.0),
            Point::new_3d(1.0 / 3.0, 1.0 / 3.0, 0.0),
        ] {
            mesh.add_node(point);
        }
        mesh.add_boundary(
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            BoundaryType::Neumann,
            0,
            0,
            0,
        );
        mesh
    }

    #[test]
    fn p2_edge_neumann_preserves_constant_resultant_and_midpoint_dof() {
        let mut mesh = unit_square_triangles(1).to_p2();
        for boundary in &mut mesh.boundaries {
            boundary.marker = 0;
        }
        let mut system = FemSystem::new(
            TripletMatrix::new(mesh.num_nodes(), mesh.num_nodes()),
            vec![0.0; mesh.num_nodes()],
        );
        try_apply_neumann_real_p2(&mut system, &mesh, 0, &|_, _, _| 1.0).unwrap();
        assert!((system.rhs.iter().sum::<f64>() - 4.0).abs() < 1.0e-12);
        assert!(system.rhs[mesh.num_nodes() - 1].is_finite());
        assert!(system.rhs.iter().filter(|value| **value != 0.0).count() > 4);
    }

    #[test]
    fn p2_edge_robin_has_expected_integrated_matrix_and_rhs() {
        let mut mesh = unit_square_triangles(1).to_p2();
        for boundary in &mut mesh.boundaries {
            boundary.marker = 0;
        }
        let mut system = FemSystem::new(
            TripletMatrix::new(mesh.num_nodes(), mesh.num_nodes()),
            vec![Complex64::new(0.0, 0.0); mesh.num_nodes()],
        );
        try_apply_robin_complex_p2(
            &mut system,
            &mesh,
            0,
            &|_, _, _| Complex64::new(1.0, 0.0),
            &|_, _, _| Complex64::new(2.0, 0.0),
        )
        .unwrap();
        assert!((system.rhs.iter().map(|value| value.re).sum::<f64>() - 8.0).abs() < 1.0e-12);
        assert!(
            (system
                .matrix
                .values
                .iter()
                .map(|value| value.re)
                .sum::<f64>()
                - 4.0)
                .abs()
                < 1.0e-12
        );
        let csr = system.matrix.to_csr();
        assert!((csr.get(0, 1) - Complex64::new(-1.0 / 30.0, 0.0)).norm() < 1.0e-12);
    }

    #[test]
    fn p2_triangular_face_neumann_preserves_unit_cube_surface_resultant() {
        let mut mesh = unit_cube_tetrahedra(1).to_p2();
        for boundary in &mut mesh.boundaries {
            boundary.marker = 0;
        }
        let mut system = FemSystem::new(
            TripletMatrix::new(mesh.num_nodes(), mesh.num_nodes()),
            vec![0.0; mesh.num_nodes()],
        );
        try_apply_neumann_real_p2(&mut system, &mesh, 0, &|_, _, _| 1.0).unwrap();
        assert!((system.rhs.iter().sum::<f64>() - 6.0).abs() < 1.0e-10);
    }

    #[test]
    fn p2_natural_boundary_is_transactional_on_callback_and_topology_errors() {
        let mut mesh = unit_square_triangles(1).to_p2();
        for boundary in &mut mesh.boundaries {
            boundary.marker = 0;
        }
        let mut system = FemSystem::new(
            TripletMatrix::new(mesh.num_nodes(), mesh.num_nodes()),
            vec![0.0; mesh.num_nodes()],
        );
        let original_rhs = system.rhs.clone();
        let original_nnz = system.matrix.nnz();
        let error = try_apply_neumann_real_p2(&mut system, &mesh, 0, &|_, _, _| f64::NAN);
        assert!(matches!(
            error,
            Err(BoundaryConditionError::NonFiniteBoundaryValue { .. })
        ));
        assert_eq!(system.rhs, original_rhs);
        assert_eq!(system.matrix.nnz(), original_nnz);

        let mut p1 = unit_square_triangles(1);
        p1.detect_boundaries();
        assert!(matches!(
            try_apply_neumann_real_p2(&mut system, &p1, 0, &|_, _, _| 1.0),
            Err(BoundaryConditionError::UnsupportedBoundary { .. })
        ));
        assert_eq!(system.rhs, original_rhs);
        assert_eq!(system.matrix.nnz(), original_nnz);
    }

    #[test]
    fn p2_curved_edge_uses_finite_isoparametric_jacobian() {
        let mut mesh = unit_square_triangles(1).to_p2();
        for boundary in &mut mesh.boundaries {
            boundary.marker = 0;
        }
        let midpoint = mesh
            .boundaries
            .iter()
            .find(|boundary| boundary.nodes.len() == 3)
            .expect("P2 boundary edge")
            .nodes[2];
        mesh.nodes[midpoint].z = 0.2;
        let mut system = FemSystem::new(
            TripletMatrix::new(mesh.num_nodes(), mesh.num_nodes()),
            vec![0.0; mesh.num_nodes()],
        );
        try_apply_neumann_real_p2(&mut system, &mesh, 0, &|x, y, z| x + y + z).unwrap();
        assert!(system.rhs.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn p3_edge_and_face_traces_preserve_constant_resultants() {
        let edge = p3_edge_mesh();
        let mut edge_system = FemSystem::new(
            TripletMatrix::new(edge.num_nodes(), edge.num_nodes()),
            vec![0.0; edge.num_nodes()],
        );
        try_apply_neumann_real_p3(&mut edge_system, &edge, 0, &|_, _, _| 1.0).unwrap();
        assert!((edge_system.rhs.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
        assert!(
            (p3_edge_shape_values(0.37, [0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0])
                .0
                .into_iter()
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1.0e-14
        );

        let face = p3_face_mesh();
        let mut face_system = FemSystem::new(
            TripletMatrix::new(face.num_nodes(), face.num_nodes()),
            vec![0.0; face.num_nodes()],
        );
        try_apply_neumann_real_p3(&mut face_system, &face, 0, &|_, _, _| 1.0).unwrap();
        assert!((face_system.rhs.iter().sum::<f64>() - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn p3_trace_rejection_is_transactional() {
        let edge = p3_edge_mesh();
        let mut system = FemSystem::new(
            TripletMatrix::new(edge.num_nodes(), edge.num_nodes()),
            vec![0.0; edge.num_nodes()],
        );
        let original_rhs = system.rhs.clone();
        assert!(matches!(
            try_apply_neumann_real_p3(&mut system, &edge, 0, &|_, _, _| f64::NAN),
            Err(BoundaryConditionError::NonFiniteBoundaryValue { .. })
        ));
        assert_eq!(system.rhs, original_rhs);
    }
}
