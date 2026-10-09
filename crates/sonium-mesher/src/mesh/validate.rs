//! Mesh validation and repair.
//!
//! Operates on a `Mesh` after construction and reports — or fixes — common
//! failure modes that break downstream FEM/BEM assembly:
//!
//! - **Inverted elements**: negative signed Jacobian at the reference vertex
//!   for P1 cells, or on a fixed reference lattice for P2/Q2 cells, detected
//!   by [`check_orientation`].
//! - **Duplicate nodes**: two `Point`s within Euclidean tolerance, found by
//!   [`find_duplicate_nodes`] and merged by [`merge_duplicate_nodes`].
//! - **Degenerate elements**: zero (or near-zero) measure, located by
//!   [`find_degenerate_elements`] and dropped by [`remove_degenerate_elements`].
//! - **Manifoldness**: every interior face borders exactly two elements;
//!   reported by [`check_manifold`].
//!
//! For batch use, [`repair`] orchestrates the above in the order
//! merge-dedup → drop-degenerate → optional flip-inverted, with a single
//! configurable [`RepairOptions`].

use super::types::{Element, ElementType, Mesh, MeshConnectivityError, PhysicalGroupError};
use std::collections::HashMap;
use std::fmt;

/// Typed failure from checked mesh validation and repair operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MeshValidationError {
    /// The public mesh vectors failed structural validation.
    InvalidMesh { errors: Vec<MeshConnectivityError> },
    /// Physical-group metadata is malformed or incompatible with the mesh dimension.
    InvalidPhysicalGroups(PhysicalGroupError),
    /// A duplicate-node tolerance must be finite and strictly positive.
    InvalidTolerance(f64),
    /// A degenerate-element measure floor must be finite and non-negative.
    InvalidMeasure(f64),
}

impl fmt::Display for MeshValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMesh { errors } => {
                write!(
                    f,
                    "mesh connectivity validation failed with {} error(s)",
                    errors.len()
                )
            }
            Self::InvalidPhysicalGroups(error) => {
                write!(f, "mesh physical-group validation failed: {error}")
            }
            Self::InvalidTolerance(value) => {
                write!(
                    f,
                    "duplicate-node tolerance must be finite and positive, got {value}"
                )
            }
            Self::InvalidMeasure(value) => {
                write!(
                    f,
                    "degenerate-element measure floor must be finite and non-negative, got {value}"
                )
            }
        }
    }
}

impl std::error::Error for MeshValidationError {}

/// Validate the complete public mesh admission surface used by checked
/// transformations: connectivity plus marker/physical-group metadata.
pub fn validate_mesh_admission(mesh: &Mesh) -> Result<(), MeshValidationError> {
    mesh.validate_connectivity()
        .map_err(|errors| MeshValidationError::InvalidMesh { errors })?;
    mesh.validate_physical_groups()
        .map_err(MeshValidationError::InvalidPhysicalGroups)
}

/// Indices of elements whose reference-vertex signed Jacobian is negative.
///
/// For tetrahedra this is `det((v1-v0, v2-v0, v3-v0)) < 0`; for triangles it
/// is the 2D cross product `< 0`. Quads/hexes report a negative Jacobian on
/// **any** of their reference corners. The returned list is in element-index
/// order with no duplicates.
pub fn check_orientation(mesh: &Mesh) -> Vec<usize> {
    match try_check_orientation(mesh) {
        Ok(result) => result,
        Err(_) if has_valid_element_connectivity(mesh) => check_orientation_unchecked(mesh),
        Err(_) => Vec::new(),
    }
}

/// Check element orientation after validating all public connectivity.
pub fn try_check_orientation(mesh: &Mesh) -> Result<Vec<usize>, MeshValidationError> {
    validate_mesh_admission(mesh)?;
    Ok(check_orientation_unchecked(mesh))
}

fn check_orientation_unchecked(mesh: &Mesh) -> Vec<usize> {
    let mut out = Vec::new();
    for (idx, elem) in mesh.elements.iter().enumerate() {
        if !is_well_oriented(mesh, elem) {
            out.push(idx);
        }
    }
    out
}

/// Boundary metadata may legitimately be stale while a caller is repairing a
/// directly-mutated element ordering. Keep report-only geometry helpers safe
/// in that intermediate state while still rejecting invalid element indices.
fn has_valid_element_connectivity(mesh: &Mesh) -> bool {
    (mesh.dimension == 2 || mesh.dimension == 3)
        && mesh
            .nodes
            .iter()
            .all(|point| point.x.is_finite() && point.y.is_finite() && point.z.is_finite())
        && mesh.elements.iter().all(|element| {
            element.element_type.dimension() == mesh.dimension
                && (element.nodes.len() == element.element_type.num_nodes_p1()
                    || element.nodes.len() == element.element_type.num_nodes_p2())
                && element.nodes.iter().all(|&node| node < mesh.nodes.len())
        })
}

fn is_well_oriented(mesh: &Mesh, elem: &Element) -> bool {
    if elem.try_polynomial_order() == Ok(2) {
        return high_order_jacobian_is_positive(mesh, elem);
    }

    let v = elem.vertices();
    let p: Vec<_> = v.iter().map(|&i| mesh.nodes[i]).collect();
    match elem.element_type {
        ElementType::Triangle => {
            let ux = p[1].x - p[0].x;
            let uy = p[1].y - p[0].y;
            let vx = p[2].x - p[0].x;
            let vy = p[2].y - p[0].y;
            ux * vy - uy * vx > 0.0
        }
        ElementType::Quadrilateral => {
            // All four corners must have positive cross of (next - this, prev - this).
            for i in 0..4 {
                let prev = (i + 3) % 4;
                let next = (i + 1) % 4;
                let ux = p[next].x - p[i].x;
                let uy = p[next].y - p[i].y;
                let vx = p[prev].x - p[i].x;
                let vy = p[prev].y - p[i].y;
                if ux * vy - uy * vx <= 0.0 {
                    return false;
                }
            }
            true
        }
        ElementType::Tetrahedron => signed_tet_volume(&p[0], &p[1], &p[2], &p[3]) > 0.0,
        ElementType::Hexahedron => {
            // The hex must have positive Jacobian at all eight corners; check
            // the trihedral parallelepiped spanned by the three edges at each.
            const EDGES: [[usize; 3]; 8] = [
                [1, 3, 4],
                [2, 0, 5],
                [3, 1, 6],
                [0, 2, 7],
                // At the top corners the vertical edge points in -z, so
                // reverse the two in-face edges to retain a right-handed
                // local trihedron. The previous order rejected every
                // otherwise valid standard Hex8/Hex27 cell.
                [7, 5, 0],
                [4, 6, 1],
                [5, 7, 2],
                [6, 4, 3],
            ];
            for (i, neigh) in EDGES.iter().enumerate() {
                let det = signed_tet_volume(&p[i], &p[neigh[0]], &p[neigh[1]], &p[neigh[2]]);
                if det <= 0.0 {
                    return false;
                }
            }
            true
        }
    }
}

/// Reference points used for curved Q2 tensor-product cells. The endpoints,
/// centre, and off-centre points catch folds introduced by edge, face, or
/// interior control nodes while keeping checked transformations bounded.
const Q2_TENSOR_JACOBIAN_SAMPLES: [f64; 7] = [-1.0, -0.75, -0.25, 0.0, 0.25, 0.75, 1.0];

const HEX27_REFERENCE_NODES: [[usize; 3]; 27] = [
    [0, 0, 0],
    [2, 0, 0],
    [2, 2, 0],
    [0, 2, 0],
    [0, 0, 2],
    [2, 0, 2],
    [2, 2, 2],
    [0, 2, 2],
    [1, 0, 0],
    [2, 1, 0],
    [1, 2, 0],
    [0, 1, 0],
    [1, 0, 2],
    [2, 1, 2],
    [1, 2, 2],
    [0, 1, 2],
    [0, 0, 1],
    [2, 0, 1],
    [2, 2, 1],
    [0, 2, 1],
    [1, 1, 0],
    [1, 1, 2],
    [1, 0, 1],
    [1, 2, 1],
    [0, 1, 1],
    [2, 1, 1],
    [1, 1, 1],
];

fn high_order_jacobian_is_positive(mesh: &Mesh, elem: &Element) -> bool {
    match elem.element_type {
        ElementType::Triangle => (0..=4).all(|i| {
            (0..=4 - i)
                .all(|j| p2_triangle_jacobian(mesh, elem, i as f64 / 4.0, j as f64 / 4.0) > 0.0)
        }),
        ElementType::Quadrilateral => Q2_TENSOR_JACOBIAN_SAMPLES.iter().all(|&xi| {
            Q2_TENSOR_JACOBIAN_SAMPLES
                .iter()
                .all(|&eta| q2_quadrilateral_jacobian(mesh, elem, xi, eta) > 0.0)
        }),
        ElementType::Tetrahedron => (0..=4).all(|i| {
            (0..=4 - i).all(|j| {
                (0..=4 - i - j).all(|k| {
                    p2_tetrahedron_jacobian(
                        mesh,
                        elem,
                        i as f64 / 4.0,
                        j as f64 / 4.0,
                        k as f64 / 4.0,
                    ) > 0.0
                })
            })
        }),
        ElementType::Hexahedron => Q2_TENSOR_JACOBIAN_SAMPLES.iter().all(|&xi| {
            Q2_TENSOR_JACOBIAN_SAMPLES.iter().all(|&eta| {
                Q2_TENSOR_JACOBIAN_SAMPLES
                    .iter()
                    .all(|&zeta| q2_hexahedron_jacobian(mesh, elem, xi, eta, zeta) > 0.0)
            })
        }),
    }
}

fn p2_triangle_jacobian(mesh: &Mesh, elem: &Element, xi: f64, eta: f64) -> f64 {
    let lambda = [1.0 - xi - eta, xi, eta];
    let gradient = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
    let derivative = [
        [
            (4.0 * lambda[0] - 1.0) * gradient[0][0],
            (4.0 * lambda[0] - 1.0) * gradient[0][1],
        ],
        [
            (4.0 * lambda[1] - 1.0) * gradient[1][0],
            (4.0 * lambda[1] - 1.0) * gradient[1][1],
        ],
        [
            (4.0 * lambda[2] - 1.0) * gradient[2][0],
            (4.0 * lambda[2] - 1.0) * gradient[2][1],
        ],
        [4.0 * (lambda[0] - lambda[1]), -4.0 * lambda[1]],
        [4.0 * lambda[2], 4.0 * lambda[1]],
        [-4.0 * lambda[2], 4.0 * (lambda[0] - lambda[2])],
    ];
    determinant_2d(mesh, elem, &derivative)
}

fn p2_tetrahedron_jacobian(mesh: &Mesh, elem: &Element, xi: f64, eta: f64, zeta: f64) -> f64 {
    let lambda = [1.0 - xi - eta - zeta, xi, eta, zeta];
    let gradient = [
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let mut derivative = [[0.0; 3]; 10];
    for node in 0..4 {
        for axis in 0..3 {
            derivative[node][axis] = (4.0 * lambda[node] - 1.0) * gradient[node][axis];
        }
    }
    for (node, (left, right)) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        .into_iter()
        .enumerate()
    {
        for axis in 0..3 {
            derivative[node + 4][axis] =
                4.0 * (lambda[left] * gradient[right][axis] + lambda[right] * gradient[left][axis]);
        }
    }
    determinant_3d(mesh, elem, &derivative)
}

fn q2_lagrange_and_derivative(coordinate: f64) -> ([f64; 3], [f64; 3]) {
    (
        [
            0.5 * coordinate * (coordinate - 1.0),
            1.0 - coordinate * coordinate,
            0.5 * coordinate * (coordinate + 1.0),
        ],
        [coordinate - 0.5, -2.0 * coordinate, coordinate + 0.5],
    )
}

fn q2_quadrilateral_jacobian(mesh: &Mesh, elem: &Element, xi: f64, eta: f64) -> f64 {
    let (lx, dlx) = q2_lagrange_and_derivative(xi);
    let (ly, dly) = q2_lagrange_and_derivative(eta);
    let derivative = [
        [dlx[0] * ly[0], lx[0] * dly[0]],
        [dlx[2] * ly[0], lx[2] * dly[0]],
        [dlx[2] * ly[2], lx[2] * dly[2]],
        [dlx[0] * ly[2], lx[0] * dly[2]],
        [dlx[1] * ly[0], lx[1] * dly[0]],
        [dlx[2] * ly[1], lx[2] * dly[1]],
        [dlx[1] * ly[2], lx[1] * dly[2]],
        [dlx[0] * ly[1], lx[0] * dly[1]],
        [dlx[1] * ly[1], lx[1] * dly[1]],
    ];
    determinant_2d(mesh, elem, &derivative)
}

fn q2_hexahedron_jacobian(mesh: &Mesh, elem: &Element, xi: f64, eta: f64, zeta: f64) -> f64 {
    let (lx, dlx) = q2_lagrange_and_derivative(xi);
    let (ly, dly) = q2_lagrange_and_derivative(eta);
    let (lz, dlz) = q2_lagrange_and_derivative(zeta);
    let mut derivative = [[0.0; 3]; 27];
    for (node, &[i, j, k]) in HEX27_REFERENCE_NODES.iter().enumerate() {
        derivative[node] = [
            dlx[i] * ly[j] * lz[k],
            lx[i] * dly[j] * lz[k],
            lx[i] * ly[j] * dlz[k],
        ];
    }
    determinant_3d(mesh, elem, &derivative)
}

fn determinant_2d(mesh: &Mesh, elem: &Element, derivative: &[[f64; 2]]) -> f64 {
    let mut dx_dxi = 0.0;
    let mut dx_deta = 0.0;
    let mut dy_dxi = 0.0;
    let mut dy_deta = 0.0;
    for (&node, &[d_xi, d_eta]) in elem.nodes.iter().zip(derivative.iter()) {
        let point = mesh.nodes[node];
        dx_dxi += d_xi * point.x;
        dx_deta += d_eta * point.x;
        dy_dxi += d_xi * point.y;
        dy_deta += d_eta * point.y;
    }
    dx_dxi * dy_deta - dx_deta * dy_dxi
}

fn determinant_3d(mesh: &Mesh, elem: &Element, derivative: &[[f64; 3]]) -> f64 {
    let mut jacobian = [[0.0; 3]; 3];
    for (&node, derivative) in elem.nodes.iter().zip(derivative.iter()) {
        let point = mesh.nodes[node];
        for (physical, coordinate) in [point.x, point.y, point.z].into_iter().enumerate() {
            for reference in 0..3 {
                jacobian[physical][reference] += coordinate * derivative[reference];
            }
        }
    }
    jacobian[0][0] * (jacobian[1][1] * jacobian[2][2] - jacobian[1][2] * jacobian[2][1])
        - jacobian[0][1] * (jacobian[1][0] * jacobian[2][2] - jacobian[1][2] * jacobian[2][0])
        + jacobian[0][2] * (jacobian[1][0] * jacobian[2][1] - jacobian[1][1] * jacobian[2][0])
}

fn signed_tet_volume(
    a: &super::types::Point,
    b: &super::types::Point,
    c: &super::types::Point,
    d: &super::types::Point,
) -> f64 {
    let bx = b.x - a.x;
    let by = b.y - a.y;
    let bz = b.z - a.z;
    let cx = c.x - a.x;
    let cy = c.y - a.y;
    let cz = c.z - a.z;
    let dx = d.x - a.x;
    let dy = d.y - a.y;
    let dz = d.z - a.z;
    (bx * (cy * dz - cz * dy) - by * (cx * dz - cz * dx) + bz * (cx * dy - cy * dx)) / 6.0
}

/// Pairs `(keep, drop)` of node indices judged equal under Euclidean
/// tolerance `tol`. Results are deterministic: for each cluster of nearly
/// coincident nodes, the lowest index is the keeper and the others are
/// flagged for dropping. Uses a uniform spatial-hash bucket of size `tol`
/// for `O(N)` average behaviour, falling back to `O(N²)` only if many nodes
/// collapse onto the same bucket.
///
/// Swallows validation failures as an empty result; prefer
/// [`try_find_duplicate_nodes`] for externally supplied meshes or tolerances.
pub fn find_duplicate_nodes(mesh: &Mesh, tol: f64) -> Vec<(usize, usize)> {
    try_find_duplicate_nodes(mesh, tol).unwrap_or_default()
}

/// Find duplicate nodes after validating mesh connectivity and tolerance.
pub fn try_find_duplicate_nodes(
    mesh: &Mesh,
    tol: f64,
) -> Result<Vec<(usize, usize)>, MeshValidationError> {
    validate_mesh_admission(mesh)?;
    if !tol.is_finite() || tol <= 0.0 {
        return Err(MeshValidationError::InvalidTolerance(tol));
    }
    // The bucket hash floors `coord / tol` into `i64`. Coordinates whose
    // scaled magnitude exceeds `i64::MAX` would saturate and collapse
    // distant points into one bucket, silently merging the whole mesh.
    let inv = 1.0 / tol;
    let overflow = mesh
        .nodes
        .iter()
        .flat_map(|p| [p.x, p.y, p.z])
        .any(|coord| !coord.is_finite() || coord.abs() * inv >= i64::MAX as f64);
    if overflow {
        return Err(MeshValidationError::InvalidTolerance(tol));
    }
    Ok(find_duplicate_nodes_unchecked(mesh, tol))
}

/// Spatial-hash bucket coordinate with explicit saturation.
fn bucket_coord(value: f64, inv: f64) -> i64 {
    // `as i64` saturates silently on overflow; clamp first so the bucketing
    // stays total even if an unchecked caller bypasses the magnitude gate in
    // `try_find_duplicate_nodes`.
    (value * inv)
        .floor()
        .clamp(i64::MIN as f64, i64::MAX as f64) as i64
}

fn find_duplicate_nodes_unchecked(mesh: &Mesh, tol: f64) -> Vec<(usize, usize)> {
    let inv = 1.0 / tol;
    let mut buckets: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let mut replacement = vec![usize::MAX; mesh.nodes.len()];

    for (idx, p) in mesh.nodes.iter().enumerate() {
        let key = (
            bucket_coord(p.x, inv),
            bucket_coord(p.y, inv),
            bucket_coord(p.z, inv),
        );
        let mut matched = None;
        // Search the 3³ neighbouring buckets so a point near a bucket
        // boundary still finds its true cluster.
        'outer: for di in -1i64..=1 {
            for dj in -1i64..=1 {
                for dk in -1i64..=1 {
                    if let Some(bucket) = buckets.get(&(key.0 + di, key.1 + dj, key.2 + dk)) {
                        for &other in bucket {
                            if mesh.nodes[other].distance(p) <= tol {
                                matched = Some(other);
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
        if let Some(keep) = matched {
            replacement[idx] = keep;
        } else {
            buckets.entry(key).or_default().push(idx);
        }
    }

    replacement
        .into_iter()
        .enumerate()
        .filter_map(|(drop, keep)| {
            if keep != usize::MAX {
                Some((keep, drop))
            } else {
                None
            }
        })
        .collect()
}

/// Merge nodes flagged by [`find_duplicate_nodes`]: rewrites element node
/// indices and boundary face node indices to the keeper, then compacts the
/// node array, dropping the orphaned slots. Returns the number of nodes
/// removed.
///
/// Best-effort on invalid input (a no-op when validation fails); prefer
/// [`try_merge_duplicate_nodes`] when the caller must distinguish "nothing
/// to merge" from "invalid mesh".
pub fn merge_duplicate_nodes(mesh: &mut Mesh, tol: f64) -> usize {
    let duplicates = find_duplicate_nodes(mesh, tol);
    if duplicates.is_empty() {
        return 0;
    }

    let mut remap: Vec<usize> = (0..mesh.nodes.len()).collect();
    for &(keep, drop) in &duplicates {
        remap[drop] = keep;
    }
    // Resolve transitive renames: keep may itself have been remapped if a
    // cluster has three or more members merged sequentially.
    for i in 0..remap.len() {
        let mut r = remap[i];
        while remap[r] != r {
            r = remap[r];
        }
        remap[i] = r;
    }

    // Compact: keep only nodes that survive (i.e., remap[i] == i).
    let mut compact_idx = vec![usize::MAX; mesh.nodes.len()];
    let mut new_nodes = Vec::with_capacity(mesh.nodes.len() - duplicates.len());
    for (i, p) in mesh.nodes.iter().enumerate() {
        if remap[i] == i {
            compact_idx[i] = new_nodes.len();
            new_nodes.push(*p);
        }
    }
    let removed = mesh.nodes.len() - new_nodes.len();
    mesh.nodes = new_nodes;

    let translate = |idx: usize| compact_idx[remap[idx]];
    for elem in &mut mesh.elements {
        for n in &mut elem.nodes {
            *n = translate(*n);
        }
    }
    for b in &mut mesh.boundaries {
        for n in &mut b.nodes {
            *n = translate(*n);
        }
    }
    mesh.invalidate_caches();
    removed
}

/// Merge duplicate nodes transactionally after validating the input and
/// post-repair connectivity.
pub fn try_merge_duplicate_nodes(mesh: &mut Mesh, tol: f64) -> Result<usize, MeshValidationError> {
    validate_mesh_admission(mesh)?;
    if !tol.is_finite() || tol <= 0.0 {
        return Err(MeshValidationError::InvalidTolerance(tol));
    }
    let mut candidate = mesh.clone();
    let removed = merge_duplicate_nodes(&mut candidate, tol);
    validate_mesh_admission(&candidate)?;
    *mesh = candidate;
    Ok(removed)
}

/// Indices of elements whose measure (area/volume) is less than or equal
/// to `min_measure`. Use a small absolute tolerance — `1e-14` for unit-scale
/// meshes — rather than zero, to catch numerically collapsed elements that
/// will cause divide-by-zero in the FEM assembly.
pub fn find_degenerate_elements(mesh: &Mesh, min_measure: f64) -> Vec<usize> {
    try_find_degenerate_elements(mesh, min_measure).unwrap_or_default()
}

/// Find degenerate elements after validating connectivity and the measure
/// threshold.
pub fn try_find_degenerate_elements(
    mesh: &Mesh,
    min_measure: f64,
) -> Result<Vec<usize>, MeshValidationError> {
    validate_mesh_admission(mesh)?;
    if !min_measure.is_finite() || min_measure < 0.0 {
        return Err(MeshValidationError::InvalidMeasure(min_measure));
    }
    Ok(find_degenerate_elements_unchecked(mesh, min_measure))
}

fn find_degenerate_elements_unchecked(mesh: &Mesh, min_measure: f64) -> Vec<usize> {
    (0..mesh.num_elements())
        .filter(|&i| mesh.element_measure(i) <= min_measure)
        .collect()
}

/// Drop elements located by [`find_degenerate_elements`]. Boundary faces
/// owned by dropped elements are also removed; surviving boundaries have
/// their `element_idx` field rebased onto the compacted element array.
/// Returns the number of elements removed.
pub fn remove_degenerate_elements(mesh: &mut Mesh, min_measure: f64) -> usize {
    let degenerate = find_degenerate_elements(mesh, min_measure);
    if degenerate.is_empty() {
        return 0;
    }
    let drop: std::collections::HashSet<usize> = degenerate.iter().copied().collect();
    let mut old_to_new = vec![usize::MAX; mesh.elements.len()];
    let mut kept = Vec::with_capacity(mesh.elements.len() - drop.len());
    for (i, e) in mesh.elements.drain(..).enumerate() {
        if !drop.contains(&i) {
            old_to_new[i] = kept.len();
            kept.push(e);
        }
    }
    let removed = mesh.elements.len(); // before reassign — wait: drained.
    mesh.elements = kept;

    mesh.boundaries.retain_mut(|b| {
        let new_idx = old_to_new[b.element_idx];
        if new_idx == usize::MAX {
            false
        } else {
            b.element_idx = new_idx;
            true
        }
    });

    mesh.invalidate_caches();

    // We computed `removed` after the drain — but `mesh.elements.len()` after
    // the drain was the count we already had. Recompute against the original.
    let _ = removed;
    drop.len()
}

/// Remove degenerate elements transactionally after validating the input and
/// resulting connectivity.
pub fn try_remove_degenerate_elements(
    mesh: &mut Mesh,
    min_measure: f64,
) -> Result<usize, MeshValidationError> {
    validate_mesh_admission(mesh)?;
    if !min_measure.is_finite() || min_measure < 0.0 {
        return Err(MeshValidationError::InvalidMeasure(min_measure));
    }
    let mut candidate = mesh.clone();
    let removed = remove_degenerate_elements(&mut candidate, min_measure);
    validate_mesh_admission(&candidate)?;
    *mesh = candidate;
    Ok(removed)
}

/// Outcome of [`check_manifold`].
#[derive(Debug, Clone, Default)]
pub struct ManifoldReport {
    /// Faces that border zero or more than two elements, keyed by sorted
    /// node tuple, with the list of incident element indices.
    pub non_manifold_faces: Vec<(Vec<usize>, Vec<usize>)>,
    /// Face counts in the well-formed case (one entry per interior face
    /// counted exactly twice; one entry per boundary face counted exactly
    /// once).
    pub total_faces: usize,
    pub interior_faces: usize,
    pub boundary_faces: usize,
}

impl ManifoldReport {
    pub fn is_manifold(&self) -> bool {
        self.non_manifold_faces.is_empty()
    }
}

/// Report each face's owning element count. A 2-manifold mesh has every
/// face borrowed by exactly 1 (boundary) or 2 (interior) elements; any other
/// count is recorded in [`ManifoldReport::non_manifold_faces`].
pub fn check_manifold(mesh: &Mesh) -> ManifoldReport {
    match try_check_manifold(mesh) {
        Ok(result) => result,
        Err(_) if has_valid_element_connectivity(mesh) => check_manifold_unchecked(mesh),
        Err(_) => ManifoldReport::default(),
    }
}

/// Check manifoldness after validating all public connectivity.
pub fn try_check_manifold(mesh: &Mesh) -> Result<ManifoldReport, MeshValidationError> {
    validate_mesh_admission(mesh)?;
    Ok(check_manifold_unchecked(mesh))
}

fn check_manifold_unchecked(mesh: &Mesh) -> ManifoldReport {
    let mut owners: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    for (elem_idx, elem) in mesh.elements.iter().enumerate() {
        let entities = if elem.element_type.dimension() == 3 {
            elem.faces()
        } else {
            elem.edges().into_iter().map(|(a, b)| vec![a, b]).collect()
        };
        for mut key in entities {
            key.sort_unstable();
            owners.entry(key).or_default().push(elem_idx);
        }
    }

    let mut report = ManifoldReport::default();
    for (key, list) in owners {
        report.total_faces += 1;
        match list.len() {
            1 => report.boundary_faces += 1,
            2 => report.interior_faces += 1,
            _ => report.non_manifold_faces.push((key, list)),
        }
    }
    report
}

/// What to do during [`repair`].
#[derive(Debug, Clone, Copy)]
pub struct RepairOptions {
    /// Node-merge tolerance. `None` skips duplicate-node merging.
    pub merge_tolerance: Option<f64>,
    /// Element-measure floor for degenerate removal. `None` skips removal.
    pub min_measure: Option<f64>,
    /// If true, flip vertex order of inverted triangles / tetrahedra. (Quads
    /// and hexes are reported but not flipped — fixing them requires a real
    /// geometric repair, not a vertex swap.)
    pub flip_inverted: bool,
}

impl Default for RepairOptions {
    fn default() -> Self {
        Self {
            merge_tolerance: Some(1e-12),
            min_measure: Some(0.0),
            flip_inverted: true,
        }
    }
}

/// Outcome counters from [`repair`].
#[derive(Debug, Clone, Default)]
pub struct RepairReport {
    pub merged_nodes: usize,
    pub removed_elements: usize,
    pub flipped_elements: usize,
    pub remaining_inverted: usize,
    pub non_manifold_faces: usize,
}

/// Orchestrate the repair passes in dependency order: merge duplicates →
/// drop degenerates → flip inverted simplices → re-check manifoldness.
pub fn repair(mesh: &mut Mesh, opts: RepairOptions) -> RepairReport {
    let mut report = RepairReport::default();

    if let Some(tol) = opts.merge_tolerance {
        report.merged_nodes = merge_duplicate_nodes(mesh, tol);
    }
    if let Some(floor) = opts.min_measure {
        report.removed_elements = remove_degenerate_elements(mesh, floor);
    }
    if opts.flip_inverted {
        for idx in check_orientation(mesh).into_iter() {
            let elem = &mut mesh.elements[idx];
            match elem.element_type {
                ElementType::Triangle => {
                    elem.nodes.swap(1, 2);
                    report.flipped_elements += 1;
                }
                ElementType::Tetrahedron => {
                    elem.nodes.swap(2, 3);
                    report.flipped_elements += 1;
                }
                ElementType::Quadrilateral | ElementType::Hexahedron => {
                    // Cannot trivially flip — leave as-is and count below.
                }
            }
        }
        report.remaining_inverted = check_orientation(mesh).len();
    }
    // The flip pass mutates element connectivity directly.  Keep all derived
    // connectivity views coherent even when no node merge or element removal
    // happened earlier in this repair call.
    if report.flipped_elements > 0 {
        mesh.invalidate_caches();
    }
    report.non_manifold_faces = check_manifold(mesh).non_manifold_faces.len();
    report
}

/// Apply the configured repair passes transactionally with typed failures.
/// Invalid public connectivity or repair thresholds leave the caller's mesh
/// unchanged.
pub fn try_repair(
    mesh: &mut Mesh,
    opts: RepairOptions,
) -> Result<RepairReport, MeshValidationError> {
    validate_mesh_admission(mesh)?;
    if let Some(tol) = opts
        .merge_tolerance
        .filter(|tol| !tol.is_finite() || *tol <= 0.0)
    {
        return Err(MeshValidationError::InvalidTolerance(tol));
    }
    if let Some(floor) = opts
        .min_measure
        .filter(|floor| !floor.is_finite() || *floor < 0.0)
    {
        return Err(MeshValidationError::InvalidMeasure(floor));
    }
    let mut candidate = mesh.clone();
    let report = repair(&mut candidate, opts);
    validate_mesh_admission(&candidate)?;
    *mesh = candidate;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::generators::{
        unit_cube_hexahedra, unit_cube_tetrahedra, unit_square_quads, unit_square_triangles,
    };
    use crate::mesh::promote_to_p2;
    use crate::mesh::types::Point;

    #[test]
    fn well_oriented_triangle_mesh_passes() {
        let mesh = unit_square_triangles(2);
        assert!(check_orientation(&mesh).is_empty());
    }

    #[test]
    fn duplicate_search_rejects_overflow_scale_without_merging() {
        let mut mesh = unit_square_triangles(1);
        // Astronomical coordinates at a tiny tolerance would saturate the
        // `i64` bucket hash and merge unrelated nodes; report instead.
        for node in &mut mesh.nodes {
            node.x *= 1e19;
            node.y *= 1e19;
        }
        assert!(matches!(
            try_find_duplicate_nodes(&mesh, 1e-9),
            Err(MeshValidationError::InvalidTolerance(_))
        ));
        let sane = unit_square_triangles(1);
        assert!(try_find_duplicate_nodes(&sane, 1e-9).is_ok());
    }

    #[test]
    fn high_order_jacobian_gate_accepts_standard_p2_cells() {
        let meshes = [
            promote_to_p2(&unit_square_triangles(1)),
            promote_to_p2(&unit_square_quads(1)),
            promote_to_p2(&unit_cube_tetrahedra(1)),
            promote_to_p2(&unit_cube_hexahedra(1)),
        ];
        for mesh in meshes {
            assert!(check_orientation(&mesh).is_empty());
        }
    }

    #[test]
    fn high_order_jacobian_gate_rejects_curved_folds_with_valid_vertices() {
        let mut triangle = promote_to_p2(&unit_square_triangles(1));
        triangle.nodes[triangle.elements[0].nodes[3]].y = 1.0;
        assert!(!check_orientation(&triangle).is_empty());

        let mut quadrilateral = promote_to_p2(&unit_square_quads(1));
        quadrilateral.nodes[quadrilateral.elements[0].nodes[4]].y = 2.0;
        assert!(!check_orientation(&quadrilateral).is_empty());

        let mut tetrahedron = promote_to_p2(&unit_cube_tetrahedra(1));
        tetrahedron.nodes[tetrahedron.elements[0].nodes[4]].z = 1.0;
        assert!(!check_orientation(&tetrahedron).is_empty());

        let mut hexahedron = promote_to_p2(&unit_cube_hexahedra(1));
        hexahedron.nodes[hexahedron.elements[0].nodes[20]].z = 2.0;
        assert!(!check_orientation(&hexahedron).is_empty());
    }

    #[test]
    fn flipped_triangle_is_reported_and_repaired() {
        let mut mesh = unit_square_triangles(1);
        mesh.elements[0].nodes.swap(1, 2);
        let bad = check_orientation(&mesh);
        assert_eq!(bad, vec![0]);

        let report = repair(
            &mut mesh,
            RepairOptions {
                merge_tolerance: None,
                min_measure: None,
                flip_inverted: true,
            },
        );
        assert_eq!(report.flipped_elements, 1);
        assert_eq!(report.remaining_inverted, 0);
    }

    #[test]
    fn duplicate_nodes_are_merged() {
        let mut mesh = Mesh::new(3);
        let n0 = mesh.add_node(Point::new_3d(0.0, 0.0, 0.0));
        let n1 = mesh.add_node(Point::new_3d(1.0, 0.0, 0.0));
        let n2 = mesh.add_node(Point::new_3d(0.0, 1.0, 0.0));
        let n3 = mesh.add_node(Point::new_3d(0.0, 0.0, 1.0));
        // Inject a duplicate of n0.
        let dup = mesh.add_node(Point::new_3d(1e-16, 0.0, 0.0));
        mesh.add_element(ElementType::Tetrahedron, vec![n0, n1, n2, n3]);
        mesh.add_element(ElementType::Tetrahedron, vec![dup, n1, n2, n3]);

        let removed = merge_duplicate_nodes(&mut mesh, 1e-12);
        assert_eq!(removed, 1);
        assert_eq!(mesh.nodes.len(), 4);
        // Both tets now reference the same vertex 0.
        assert_eq!(mesh.elements[0].nodes[0], mesh.elements[1].nodes[0]);
    }

    #[test]
    fn degenerate_element_is_removed() {
        let mut mesh = Mesh::new(2);
        let a = mesh.add_node(Point::new_2d(0.0, 0.0));
        let b = mesh.add_node(Point::new_2d(1.0, 0.0));
        let c = mesh.add_node(Point::new_2d(0.0, 1.0));
        let d = mesh.add_node(Point::new_2d(1.0, 0.0)); // colinear/dup → degenerate triangle
        mesh.add_element(ElementType::Triangle, vec![a, b, c]);
        mesh.add_element(ElementType::Triangle, vec![a, b, d]); // collapsed

        let removed = remove_degenerate_elements(&mut mesh, 1e-14);
        assert_eq!(removed, 1);
        assert_eq!(mesh.num_elements(), 1);
    }

    #[test]
    fn manifold_report_on_cube_is_2_manifold() {
        let mesh = unit_cube_tetrahedra(2);
        let report = check_manifold(&mesh);
        assert!(report.is_manifold());
        assert!(report.interior_faces > 0);
        assert!(report.boundary_faces > 0);
    }

    #[test]
    fn checked_validation_rejects_malformed_public_connectivity_without_panicking() {
        let mut mesh = Mesh::new(3);
        for point in [
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(0.0, 0.0, 1.0),
        ] {
            mesh.add_node(point);
        }
        mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, usize::MAX]);

        assert!(matches!(
            try_check_orientation(&mesh),
            Err(MeshValidationError::InvalidMesh { .. })
        ));
        assert!(check_orientation(&mesh).is_empty());
        assert!(try_check_manifold(&mesh).is_err());
        assert!(check_manifold(&mesh).non_manifold_faces.is_empty());
    }

    #[test]
    fn checked_admission_rejects_invalid_physical_group_metadata() {
        let mut mesh = unit_square_triangles(1);
        mesh.physical_groups = serde_json::from_value(serde_json::json!({
            "groups": [{
                "dimension": 3,
                "tag": 1,
                "name": "volume",
            }]
        }))
        .expect("test-only registry wire format is valid");

        assert!(matches!(
            validate_mesh_admission(&mesh),
            Err(MeshValidationError::InvalidPhysicalGroups(
                PhysicalGroupError::DimensionExceedsMesh {
                    dimension: 3,
                    mesh_dimension: 2,
                }
            ))
        ));
        assert!(matches!(
            try_check_orientation(&mesh),
            Err(MeshValidationError::InvalidPhysicalGroups(_))
        ));
    }

    #[test]
    fn checked_repair_is_transactional_for_invalid_options() {
        let mut mesh = unit_square_triangles(1);
        let original_nodes = mesh.nodes.clone();
        let original_elements = mesh.elements.clone();

        assert!(matches!(
            try_repair(
                &mut mesh,
                RepairOptions {
                    merge_tolerance: Some(f64::NAN),
                    min_measure: None,
                    flip_inverted: true,
                }
            ),
            Err(MeshValidationError::InvalidTolerance(value)) if value.is_nan()
        ));
        assert_eq!(mesh.nodes, original_nodes);
        assert_eq!(
            mesh.elements
                .iter()
                .map(|element| (element.element_type, element.nodes.clone(), element.id))
                .collect::<Vec<_>>(),
            original_elements
                .iter()
                .map(|element| (element.element_type, element.nodes.clone(), element.id))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn checked_repair_and_degenerate_removal_preserve_connectivity() {
        let mut mesh = Mesh::new(2);
        let a = mesh.add_node(Point::new_2d(0.0, 0.0));
        let b = mesh.add_node(Point::new_2d(1.0, 0.0));
        let c = mesh.add_node(Point::new_2d(0.0, 1.0));
        let d = mesh.add_node(Point::new_2d(1.0, 0.0));
        mesh.add_element(ElementType::Triangle, vec![a, b, c]);
        mesh.add_element(ElementType::Triangle, vec![a, b, d]);

        assert_eq!(try_find_degenerate_elements(&mesh, 1e-14).unwrap(), vec![1]);
        assert_eq!(try_remove_degenerate_elements(&mut mesh, 1e-14).unwrap(), 1);
        let report = try_repair(
            &mut mesh,
            RepairOptions {
                merge_tolerance: None,
                min_measure: None,
                flip_inverted: true,
            },
        )
        .expect("remaining valid mesh can be repaired");
        assert_eq!(report.remaining_inverted, 0);
        assert!(mesh.validate_connectivity().is_ok());
    }
}
