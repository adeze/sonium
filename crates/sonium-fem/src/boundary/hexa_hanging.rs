//! Affine hexahedral H1 hanging-face constraints.
//!
//! This module is intentionally separate from the triangular-face contract:
//! quadrilateral traces use tensor-product Lagrange interpolation and have a
//! different ownership topology.  Isoparametric geometry is rejected until a
//! physical-face-map implementation is available.

use super::hanging::{H1HangingQuadFaceConstraint, HangingNodeError};
use super::periodic::{MpcSystem, MpcTerm};
use crate::basis::PolynomialDegree;
use crate::mesh::{ElementType, Mesh};
use crate::spaces::{FunctionSpace, H1Space, VectorH1Space};
use num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
struct FaceRecord {
    face: [usize; 4],
    area: f64,
    trace: Vec<(usize, [f64; 2])>,
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    face_index: usize,
    reference: [f64; 2],
    area: f64,
}

/// Build Q1/Q2/Q3 H1 constraints for non-conforming affine hexahedral faces.
pub fn h1_hanging_quad_face_constraints_3d(
    mesh: &Mesh,
    space: &H1Space,
    tolerance: f64,
) -> Result<Vec<H1HangingQuadFaceConstraint>, HangingNodeError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(HangingNodeError::InvalidTolerance(tolerance));
    }
    if mesh.dimension != 3 {
        return Err(HangingNodeError::UnsupportedDimension(mesh.dimension));
    }
    for (element_index, element) in mesh.elements.iter().enumerate() {
        if element.element_type != ElementType::Hexahedron {
            return Err(HangingNodeError::UnsupportedH1Element {
                element_index,
                element_type: element.element_type,
            });
        }
    }

    // Check the topology shape before generic connectivity validation.  A
    // curved/isoparametric hexahedron may carry valid node indices but still
    // be outside this affine face-trace contract; report that contract error
    // instead of leaking a lower-level connectivity diagnostic.
    for (element_index, element) in mesh.elements.iter().enumerate() {
        if element.nodes.len() != element.element_type.num_vertices() {
            return Err(HangingNodeError::UnsupportedGeometry {
                element_index,
                actual: element.nodes.len(),
                expected: element.element_type.num_vertices(),
            });
        }
    }

    mesh.validate_connectivity()
        .map_err(|errors| HangingNodeError::InvalidMesh { errors })?;
    space
        .validate(mesh)
        .map_err(HangingNodeError::InvalidH1Space)?;

    let coordinates = space
        .dof_coordinates(mesh)
        .ok_or(HangingNodeError::MissingDofCoordinates)?;
    if coordinates.len() != space.num_global_dofs(mesh) {
        return Err(HangingNodeError::MissingDofCoordinates);
    }
    for (dof, point) in coordinates.iter().enumerate() {
        if !point.iter().all(|value| value.is_finite()) {
            return Err(HangingNodeError::NonFiniteDofCoordinate { dof });
        }
    }

    let expected_trace = match space.degree {
        PolynomialDegree::P1 => 4,
        PolynomialDegree::P2 => 9,
        PolynomialDegree::P3 => 16,
    };
    let mut owners = BTreeMap::<[usize; 4], (usize, [usize; 4])>::new();
    for (element_index, element) in mesh.elements.iter().enumerate() {
        let vertices = element.vertices();
        for local_face in element.element_type.local_faces() {
            let face = [
                vertices[local_face[0]],
                vertices[local_face[1]],
                vertices[local_face[2]],
                vertices[local_face[3]],
            ];
            let mut key = face;
            key.sort_unstable();
            owners.entry(key).or_insert((element_index, face));
        }
    }

    let mut faces = Vec::with_capacity(owners.len());
    for (_, (owner_element, face)) in owners {
        let p0 = mesh.nodes[face[0]];
        let p1 = mesh.nodes[face[1]];
        let p3 = mesh.nodes[face[3]];
        let u = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
        let v = [p3.x - p0.x, p3.y - p0.y, p3.z - p0.z];
        let area = norm(cross(u, v));
        if !area.is_finite() || area <= tolerance {
            return Err(HangingNodeError::DegenerateQuadrilateralFace { face });
        }

        let mut trace = space
            .local_to_global(mesh, owner_element)
            .into_iter()
            .filter_map(|dof| {
                let (reference, distance) = face_reference(coordinates[dof], mesh, face, area)?;
                let margin = tolerance / area.max(tolerance);
                (distance <= tolerance
                    && reference[0] >= -margin
                    && reference[0] <= 1.0 + margin
                    && reference[1] >= -margin
                    && reference[1] <= 1.0 + margin)
                    .then_some((
                        dof,
                        [reference[0].clamp(0.0, 1.0), reference[1].clamp(0.0, 1.0)],
                    ))
            })
            .collect::<Vec<_>>();
        trace.sort_by(|left, right| {
            left.1[1]
                .total_cmp(&right.1[1])
                .then_with(|| left.1[0].total_cmp(&right.1[0]))
                .then_with(|| left.0.cmp(&right.0))
        });
        trace.dedup_by(|left, right| left.0 == right.0);
        if trace.len() != expected_trace {
            return Err(HangingNodeError::InvalidQuadrilateralFaceTraceDofs {
                element_index: owner_element,
                face,
                actual: trace.len(),
                expected: expected_trace,
            });
        }
        faces.push(FaceRecord { face, area, trace });
    }

    // Only the largest face in a coplanar overlap owns the interface.  The
    // mesh also contains the four fine-side faces, and treating those as
    // independent coarse owners makes edge/vertex DOFs appear to have two
    // equally valid constraints.  A smaller face whose center is contained
    // by a larger affine face is therefore a subordinate trace and is not a
    // candidate owner.
    let candidate_faces = faces
        .iter()
        .enumerate()
        .filter_map(|(face_index, face)| {
            let center = face_center(mesh, face.face);
            let covered_by_larger_face = faces.iter().enumerate().any(|(other_index, other)| {
                if face_index == other_index || other.area <= face.area + tolerance {
                    return false;
                }
                let Some((reference, distance)) =
                    face_reference(center, mesh, other.face, other.area)
                else {
                    return false;
                };
                distance <= tolerance
                    && reference.iter().all(|value| {
                        *value >= -tolerance / other.area.max(tolerance)
                            && *value <= 1.0 + tolerance / other.area.max(tolerance)
                    })
            });
            let covers_smaller_face = faces.iter().enumerate().any(|(other_index, other)| {
                if face_index == other_index || other.area + tolerance >= face.area {
                    return false;
                }
                let other_center = face_center(mesh, other.face);
                let Some((reference, distance)) =
                    face_reference(other_center, mesh, face.face, face.area)
                else {
                    return false;
                };
                distance <= tolerance
                    && reference.iter().all(|value| {
                        *value >= -tolerance / face.area.max(tolerance)
                            && *value <= 1.0 + tolerance / face.area.max(tolerance)
                    })
            });
            (!covered_by_larger_face && covers_smaller_face).then_some(face_index)
        })
        .collect::<Vec<_>>();

    let mut candidates = BTreeMap::<usize, Vec<Candidate>>::new();
    for face_index in candidate_faces {
        let face = &faces[face_index];
        for (dof, point) in coordinates.iter().enumerate() {
            if face.trace.iter().any(|(trace_dof, _)| {
                *trace_dof == dof
                    || coordinates[*trace_dof]
                        .iter()
                        .zip(point)
                        .all(|(master, candidate)| (master - candidate).abs() <= tolerance)
            }) {
                continue;
            }
            let Some((reference, distance)) = face_reference(*point, mesh, face.face, face.area)
            else {
                continue;
            };
            let margin = tolerance / face.area.max(tolerance);
            if distance <= tolerance
                && reference[0] >= -margin
                && reference[0] <= 1.0 + margin
                && reference[1] >= -margin
                && reference[1] <= 1.0 + margin
            {
                candidates.entry(dof).or_default().push(Candidate {
                    face_index,
                    reference: [reference[0].clamp(0.0, 1.0), reference[1].clamp(0.0, 1.0)],
                    area: face.area,
                });
            }
        }
    }

    let mut constraints = Vec::with_capacity(candidates.len());
    for (slave, mut node_candidates) in candidates {
        node_candidates.sort_by(|left, right| {
            right
                .area
                .total_cmp(&left.area)
                .then_with(|| left.face_index.cmp(&right.face_index))
        });
        let best = node_candidates[0];
        if node_candidates.iter().skip(1).any(|candidate| {
            (candidate.area - best.area).abs() <= tolerance
                && candidate.face_index != best.face_index
        }) {
            return Err(HangingNodeError::AmbiguousConstraint { slave });
        }
        let coarse = &faces[best.face_index];
        constraints.push(H1HangingQuadFaceConstraint {
            slave,
            face: coarse.face,
            reference: best.reference,
            masters: lagrange_terms(&coarse.trace, best.reference, space.degree)?,
        });
    }
    Ok(constraints)
}

/// Build an [`MpcSystem`] for affine hexahedral Q1/Q2/Q3 face constraints.
pub fn h1_hanging_quad_face_mpc_system_3d(
    mesh: &Mesh,
    space: &H1Space,
    tolerance: f64,
) -> Result<MpcSystem, HangingNodeError> {
    let constraints = h1_hanging_quad_face_constraints_3d(mesh, space, tolerance)?;
    let mpcs = constraints
        .iter()
        .map(H1HangingQuadFaceConstraint::to_mpc)
        .collect::<Vec<_>>();
    MpcSystem::new(space.num_global_dofs(mesh), &mpcs).map_err(HangingNodeError::Mpc)
}

/// Build a checked vector-H1 MPC system for affine hexahedral hanging faces.
///
/// Vector DOFs use the interleaved numbering owned by [`VectorH1Space`]; the
/// scalar quadrilateral-face trace is lifted component-wise by the shared
/// boundary ownership helper.
pub fn vector_h1_hanging_quad_face_mpc_system_3d(
    mesh: &Mesh,
    space: &VectorH1Space,
    tolerance: f64,
) -> Result<MpcSystem, HangingNodeError> {
    let scalar_space = H1Space::new(space.degree());
    let scalar_constraints = h1_hanging_quad_face_constraints_3d(mesh, &scalar_space, tolerance)?;
    let scalar_mpcs = scalar_constraints
        .iter()
        .map(H1HangingQuadFaceConstraint::to_mpc)
        .collect::<Vec<_>>();
    super::hanging::lift_h1_mpc_constraints(
        &scalar_mpcs,
        scalar_space.num_global_dofs(mesh),
        space.num_global_dofs(mesh),
        space.num_components(),
    )
}

fn face_reference(
    point: [f64; 3],
    mesh: &Mesh,
    face: [usize; 4],
    area: f64,
) -> Option<([f64; 2], f64)> {
    let p0 = mesh.nodes[face[0]];
    let p1 = mesh.nodes[face[1]];
    let p3 = mesh.nodes[face[3]];
    let u = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
    let v = [p3.x - p0.x, p3.y - p0.y, p3.z - p0.z];
    let d = [point[0] - p0.x, point[1] - p0.y, point[2] - p0.z];
    let uu = dot(u, u);
    let uv = dot(u, v);
    let vv = dot(v, v);
    let determinant = uu * vv - uv * uv;
    if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
        return None;
    }
    let du = dot(d, u);
    let dv = dot(d, v);
    let reference = [
        (du * vv - dv * uv) / determinant,
        (dv * uu - du * uv) / determinant,
    ];
    let distance = dot(d, cross(u, v)).abs() / area;
    reference
        .iter()
        .all(|value| value.is_finite())
        .then_some((reference, distance))
}

fn face_center(mesh: &Mesh, face: [usize; 4]) -> [f64; 3] {
    let points = face.map(|node| mesh.nodes[node]);
    [
        points.iter().map(|point| point.x).sum::<f64>() / 4.0,
        points.iter().map(|point| point.y).sum::<f64>() / 4.0,
        points.iter().map(|point| point.z).sum::<f64>() / 4.0,
    ]
}

fn lagrange_terms(
    trace: &[(usize, [f64; 2])],
    reference: [f64; 2],
    degree: PolynomialDegree,
) -> Result<Vec<MpcTerm>, HangingNodeError> {
    let nodes: &[f64] = match degree {
        PolynomialDegree::P1 => &[0.0, 1.0],
        PolynomialDegree::P2 => &[0.0, 0.5, 1.0],
        PolynomialDegree::P3 => &[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
    };
    let mut occupied = BTreeSet::new();
    let mut masters = Vec::with_capacity(trace.len());
    for (dof, point) in trace {
        let i = nearest(nodes, point[0]);
        let j = nearest(nodes, point[1]);
        if (nodes[i] - point[0]).abs() > 1.0e-8 || (nodes[j] - point[1]).abs() > 1.0e-8 {
            return Err(HangingNodeError::InvalidFaceInterpolation);
        }
        if !occupied.insert((i, j)) {
            return Err(HangingNodeError::InvalidFaceInterpolation);
        }
        masters.push(MpcTerm {
            dof: *dof,
            coefficient: Complex64::new(
                basis(nodes, i, reference[0]) * basis(nodes, j, reference[1]),
                0.0,
            ),
        });
    }
    Ok(masters)
}

fn nearest(nodes: &[f64], value: f64) -> usize {
    nodes
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| (value - **left).abs().total_cmp(&(value - **right).abs()))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn basis(nodes: &[f64], index: usize, value: f64) -> f64 {
    nodes
        .iter()
        .enumerate()
        .filter(|(other, _)| *other != index)
        .fold(1.0, |product, (_, node)| {
            product * (value - *node) / (nodes[index] - *node)
        })
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm(value: [f64; 3]) -> f64 {
    dot(value, value).sqrt()
}
