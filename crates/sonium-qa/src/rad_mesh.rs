//! Quad-dominant shell patches extruded into OpenCourant bricks.
//!
//! This module builds small crash-mesh sections (Taylor bar, square and
//! round crush tubes, clamped plate) from quad-dominant surface patches and
//! turns them into [`OpenCourantCase`] brick meshes by straight extrusion or into a
//! [`RadExtendedCase`] shell deck. It is Phase-4 lane 4C: it constructs
//! `OpenCourantCase` with existing fields only and adds no fields to it.
//!
//! Indexing follows the [`RadMesh`] convention: `nodes` storage is 0-based
//! while `quads`, `tris`, and `bricks` connectivity is 1-based. Patch quads
//! are wound counter-clockwise seen from +z so [`ShellPatch::extrude`]
//! yields positive-volume bricks. Every delivered builder holds a minimum
//! scaled Jacobian above [`MIN_SCALED_JACOBIAN`] and every extrusion is a
//! closed solid (see [`measure_solid`]).
//!
//! # Examples
//!
//! ```rust
//! use sonium_qa::rad_mesh::{measure_patch, rect_patch};
//!
//! let patch = rect_patch(2, 2, 0.1, 0.1);
//! let metrics = measure_patch(&patch);
//! assert_eq!(metrics.quads, 4);
//! assert!((metrics.quad_fraction - 1.0).abs() < 1e-12);
//! ```

// Rust guideline compliant 2026-02-21

use crate::analytical::rad::{
    OpenCourantCase, RadBoundaryCondition, RadConcentratedLoad, RadElasticMaterial, RadError,
    RadExtendedCase, RadFunction, RadMesh, RadNodeSet, RadProbe, RadShellMesh, RadShellProperty,
    RadThRequest, RadWall, shel_channel,
};
use std::collections::HashMap;
use std::fmt;

/// Minimum scaled Jacobian accepted on delivered builders.
///
/// Phase-4 lane 4C quality bar: every quad, triangle, and brick corner of
/// the shipped patch and extrusion builders must score above this. The
/// value 0.2 admits the radially projected boundary quads of
/// [`taylor_patch`] (which score near 0.5) while rejecting degenerate or
/// inverted corners.
pub const MIN_SCALED_JACOBIAN: f64 = 0.2;

/// Taylor bar length in metres (1 inch, matches `fem_opencourant_taylor_bar`).
const TAYLOR_LENGTH_M: f64 = 0.0254;
/// Taylor bar radius in metres (1/8 inch diameter section).
const TAYLOR_RADIUS_M: f64 = 0.003_175;
/// Taylor impact speed in m/s (elastic copper prove-out leg).
const TAYLOR_VELOCITY_M_PER_S: f64 = 227.0;
/// Crush-box outer section width in metres (50 mm square tube).
const CRASH_OUTER_M: f64 = 0.05;
/// Crush-box inner section width in metres (2 mm wall each side).
const CRASH_INNER_M: f64 = 0.046;
/// Crush-box column length in metres (matches the rail crush leg).
const CRASH_LENGTH_M: f64 = 0.2;
/// Clamped plate radius in metres (matches `shell_clamped_plate_case`).
const PLATE_RADIUS_M: f64 = 1.0;
/// Clamped plate thickness in metres.
const PLATE_THICK_M: f64 = 0.02;

/// Quad-dominant surface patch with 1-based connectivity.
///
/// `nodes` holds coordinates in metres (0-based storage); `quads` and
/// `tris` index it 1-based, matching [`RadShellMesh`]. Builders wind every
/// face counter-clockwise seen from +z.
#[derive(Debug, Clone, Default)]
pub struct ShellPatch {
    /// Node coordinates in metres, 0-based storage.
    pub nodes: Vec<[f64; 3]>,
    /// Four-node faces, 1-based node ids.
    pub quads: Vec<[usize; 4]>,
    /// Three-node faces, 1-based node ids.
    pub tris: Vec<[usize; 3]>,
}

impl ShellPatch {
    /// Count patch nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Count quad faces.
    pub fn num_quads(&self) -> usize {
        self.quads.len()
    }

    /// Count triangle faces.
    pub fn num_tris(&self) -> usize {
        self.tris.len()
    }

    /// Check whether the patch holds no faces.
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty() && self.tris.is_empty()
    }

    /// Validate index ranges and coordinate finiteness.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::Parse`] when a face references a missing node
    /// or a coordinate is non-finite.
    pub fn validate(&self) -> Result<(), RadError> {
        for (index, node) in self.nodes.iter().enumerate() {
            if !node.iter().all(|value| value.is_finite()) {
                return Err(RadError::Parse(format!(
                    "patch node {} is non-finite",
                    index + 1
                )));
            }
        }
        for (index, quad) in self.quads.iter().enumerate() {
            for node in quad {
                if *node == 0 || *node > self.nodes.len() {
                    return Err(RadError::Parse(format!(
                        "patch quad {} references missing node {node}",
                        index + 1
                    )));
                }
            }
        }
        for (index, tri) in self.tris.iter().enumerate() {
            for node in tri {
                if *node == 0 || *node > self.nodes.len() {
                    return Err(RadError::Parse(format!(
                        "patch tri {} references missing node {node}",
                        index + 1
                    )));
                }
            }
        }
        Ok(())
    }

    /// Share of quad faces over all faces.
    ///
    /// A pure quad patch scores 1.0; an empty patch scores 1.0 vacuously
    /// (callers gate on face counts separately).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_qa::rad_mesh::disk_patch;
    ///
    /// let patch = disk_patch(6, 24, 1.0);
    /// assert!(patch.quad_fraction() > 0.8);
    /// ```
    pub fn quad_fraction(&self) -> f64 {
        let total = self.quads.len() + self.tris.len();
        if total == 0 {
            return 1.0;
        }
        self.quads.len() as f64 / total as f64
    }

    /// Minimum scaled Jacobian over every patch face.
    ///
    /// Quads use the corner sine against the face normal, triangles the
    /// Knupp mean ratio; an empty patch scores 1.0 vacuously.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_qa::rad_mesh::{MIN_SCALED_JACOBIAN, rect_patch};
    ///
    /// let patch = rect_patch(3, 3, 1.0, 1.0);
    /// assert!(patch.min_scaled_jacobian() > MIN_SCALED_JACOBIAN);
    /// ```
    pub fn min_scaled_jacobian(&self) -> f64 {
        if self.is_empty() {
            return 1.0;
        }
        let mut worst = 1.0_f64;
        for quad in &self.quads {
            worst = worst.min(quad_scaled_jacobian(&self.nodes, quad));
        }
        for tri in &self.tris {
            worst = worst.min(tri_scaled_jacobian(&self.nodes, tri));
        }
        worst
    }

    /// Refine every face into quads.
    ///
    /// Each quad splits into four, each triangle into three; edge midpoints
    /// are shared between neighbors so the result stays conforming (no
    /// hanging nodes, no cracks) and ready for [`ShellPatch::extrude`].
    /// Winding stays counter-clockwise seen from +z.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_qa::rad_mesh::disk_patch;
    ///
    /// let patch = disk_patch(4, 12, 1.0).to_all_quad();
    /// assert!(patch.tris.is_empty());
    /// assert!((patch.quad_fraction() - 1.0).abs() < 1e-12);
    /// ```
    pub fn to_all_quad(&self) -> Self {
        let mut nodes = self.nodes.clone();
        let mut midpoint_of: HashMap<[usize; 2], usize> = HashMap::new();
        // Midpoint per sorted endpoint pair, shared across faces. Takes
        // the vec and map as arguments (rather than capturing) so the
        // subdivision loops below can also push centroids.
        let edge_mid = |endpoints: [usize; 2],
                        nodes: &mut Vec<[f64; 3]>,
                        map: &mut HashMap<[usize; 2], usize>|
         -> usize {
            let mut key = endpoints;
            key.sort_unstable();
            if let Some(found) = map.get(&key) {
                return *found;
            }
            let a = nodes[endpoints[0] - 1];
            let b = nodes[endpoints[1] - 1];
            nodes.push([
                0.5 * (a[0] + b[0]),
                0.5 * (a[1] + b[1]),
                0.5 * (a[2] + b[2]),
            ]);
            let id = nodes.len();
            map.insert(key, id);
            id
        };
        let mut quads = Vec::with_capacity(self.quads.len() * 4 + self.tris.len() * 3);
        for quad in &self.quads {
            let mid01 = edge_mid([quad[0], quad[1]], &mut nodes, &mut midpoint_of);
            let mid12 = edge_mid([quad[1], quad[2]], &mut nodes, &mut midpoint_of);
            let mid23 = edge_mid([quad[2], quad[3]], &mut nodes, &mut midpoint_of);
            let mid30 = edge_mid([quad[3], quad[0]], &mut nodes, &mut midpoint_of);
            let mut centroid = [0.0, 0.0, 0.0];
            for corner in quad {
                let node = nodes[corner - 1];
                centroid[0] += node[0];
                centroid[1] += node[1];
                centroid[2] += node[2];
            }
            nodes.push([centroid[0] / 4.0, centroid[1] / 4.0, centroid[2] / 4.0]);
            let center = nodes.len();
            quads.push([quad[0], mid01, center, mid30]);
            quads.push([quad[1], mid12, center, mid01]);
            quads.push([quad[2], mid23, center, mid12]);
            quads.push([quad[3], mid30, center, mid23]);
        }
        for tri in &self.tris {
            let mid_ab = edge_mid([tri[0], tri[1]], &mut nodes, &mut midpoint_of);
            let mid_bc = edge_mid([tri[1], tri[2]], &mut nodes, &mut midpoint_of);
            let mid_ca = edge_mid([tri[2], tri[0]], &mut nodes, &mut midpoint_of);
            let mut centroid = [0.0, 0.0, 0.0];
            for corner in tri {
                let node = nodes[corner - 1];
                centroid[0] += node[0];
                centroid[1] += node[1];
                centroid[2] += node[2];
            }
            nodes.push([centroid[0] / 3.0, centroid[1] / 3.0, centroid[2] / 3.0]);
            let center = nodes.len();
            quads.push([tri[0], mid_ab, center, mid_ca]);
            quads.push([tri[1], mid_bc, center, mid_ab]);
            quads.push([tri[2], mid_ca, center, mid_bc]);
        }
        Self {
            nodes,
            quads,
            tris: Vec::new(),
        }
    }

    /// Extrude a pure quad patch into bricks along +z.
    ///
    /// Each quad becomes a stack of `layers` bricks spanning
    /// `start_m..end_m`; only the patch x/y coordinates are kept. The
    /// result is a closed solid when the patch is edge-manifold.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::Parse`] when the patch holds triangles (run
    /// [`ShellPatch::to_all_quad`] first), when `layers` is zero, when the
    /// span is degenerate or non-finite, or when a quad winds clockwise
    /// seen from +z.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_qa::rad_mesh::rect_patch;
    ///
    /// let mesh = rect_patch(2, 2, 1.0, 1.0).extrude(0.0, 1.0, 2).unwrap();
    /// assert_eq!(mesh.num_bricks(), 8);
    /// ```
    pub fn extrude(&self, start_m: f64, end_m: f64, layers: usize) -> Result<RadMesh, RadError> {
        self.validate()?;
        if !self.tris.is_empty() {
            return Err(RadError::Parse(
                "extrusion needs a pure quad patch; run to_all_quad first".to_string(),
            ));
        }
        if layers == 0 {
            return Err(RadError::Parse(
                "extrusion needs at least one layer".to_string(),
            ));
        }
        if !start_m.is_finite() || !end_m.is_finite() || end_m <= start_m {
            return Err(RadError::Parse(format!(
                "extrusion needs a positive finite span, got [{start_m}, {end_m}]"
            )));
        }
        for (index, quad) in self.quads.iter().enumerate() {
            let corners = [
                self.nodes[quad[0] - 1],
                self.nodes[quad[1] - 1],
                self.nodes[quad[2] - 1],
                self.nodes[quad[3] - 1],
            ];
            if newell_normal(&corners)[2] <= 0.0 {
                return Err(RadError::Parse(format!(
                    "extrusion needs quads wound counter-clockwise from +z (quad {})",
                    index + 1
                )));
            }
        }
        let per_layer = self.nodes.len();
        let mut nodes = Vec::with_capacity(per_layer * (layers + 1));
        for layer in 0..=layers {
            let z = start_m + (end_m - start_m) * layer as f64 / layers as f64;
            for node in &self.nodes {
                nodes.push([node[0], node[1], z]);
            }
        }
        let mut bricks = Vec::with_capacity(self.quads.len() * layers);
        for layer in 0..layers {
            let base = layer * per_layer;
            let top = (layer + 1) * per_layer;
            for quad in &self.quads {
                bricks.push([
                    base + quad[0],
                    base + quad[1],
                    base + quad[2],
                    base + quad[3],
                    top + quad[0],
                    top + quad[1],
                    top + quad[2],
                    top + quad[3],
                ]);
            }
        }
        Ok(RadMesh { nodes, bricks })
    }
}

/// Scaled Jacobian of one quad face.
///
/// Each corner scores the sine between its edges against the Newell face
/// normal (1.0 for a rectangle, 0.0 for a degenerate or bow-tie corner);
/// the quad scores its worst corner. The score is winding-invariant, so
/// orientation checks (like the extruder's +z gate) read the Newell
/// normal separately.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_mesh::quad_scaled_jacobian;
///
/// let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
/// assert!((quad_scaled_jacobian(&nodes, &[1, 2, 3, 4]) - 1.0).abs() < 1e-12);
/// ```
pub fn quad_scaled_jacobian(nodes: &[[f64; 3]], quad: &[usize; 4]) -> f64 {
    let corners = [
        nodes[quad[0] - 1],
        nodes[quad[1] - 1],
        nodes[quad[2] - 1],
        nodes[quad[3] - 1],
    ];
    let normal = newell_normal(&corners);
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if length <= 0.0 {
        return 0.0;
    }
    let unit = [normal[0] / length, normal[1] / length, normal[2] / length];
    let mut worst = 1.0_f64;
    for corner in 0..4 {
        let current = corners[corner];
        let next = corners[(corner + 1) % 4];
        let prev = corners[(corner + 3) % 4];
        let edge_next = [
            next[0] - current[0],
            next[1] - current[1],
            next[2] - current[2],
        ];
        let edge_prev = [
            prev[0] - current[0],
            prev[1] - current[1],
            prev[2] - current[2],
        ];
        let cross = [
            edge_next[1] * edge_prev[2] - edge_next[2] * edge_prev[1],
            edge_next[2] * edge_prev[0] - edge_next[0] * edge_prev[2],
            edge_next[0] * edge_prev[1] - edge_next[1] * edge_prev[0],
        ];
        let denom = edge_len(&edge_next) * edge_len(&edge_prev);
        if denom <= 0.0 {
            return 0.0;
        }
        let sine = (cross[0] * unit[0] + cross[1] * unit[1] + cross[2] * unit[2]) / denom;
        worst = worst.min(sine);
    }
    worst
}

/// Scaled Jacobian of one triangle face.
///
/// Knupp mean ratio `4 sqrt(3) A / sum e^2` (1.0 for equilateral, 0.0
/// for degenerate), matching the mesher quality convention.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_mesh::tri_scaled_jacobian;
///
/// let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866_025_403_784_438_6, 0.0]];
/// assert!(tri_scaled_jacobian(&nodes, &[1, 2, 3]) > 0.99);
/// ```
pub fn tri_scaled_jacobian(nodes: &[[f64; 3]], tri: &[usize; 3]) -> f64 {
    let corners = [nodes[tri[0] - 1], nodes[tri[1] - 1], nodes[tri[2] - 1]];
    let edge = |a: [f64; 3], b: [f64; 3]| {
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let dz = b[2] - a[2];
        dx * dx + dy * dy + dz * dz
    };
    let denom =
        edge(corners[0], corners[1]) + edge(corners[1], corners[2]) + edge(corners[2], corners[0]);
    if denom <= 0.0 {
        return 0.0;
    }
    let ab = [
        corners[1][0] - corners[0][0],
        corners[1][1] - corners[0][1],
        corners[1][2] - corners[0][2],
    ];
    let ac = [
        corners[2][0] - corners[0][0],
        corners[2][1] - corners[0][1],
        corners[2][2] - corners[0][2],
    ];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let area = 0.5 * edge_len(&cross);
    (4.0 * 3.0_f64.sqrt() * area / denom).clamp(0.0, 1.0)
}

/// Scaled Jacobian of one brick element.
///
/// Each corner scores `det / (|e1| |e2| |e3|)` (1.0 for a cube, 0.0 for
/// a collapsed corner, negative for inverted winding); the brick scores
/// its worst corner.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_mesh::brick_scaled_jacobian;
///
/// let nodes = vec![
///     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
/// ];
/// let brick = [1, 2, 3, 4, 5, 6, 7, 8];
/// assert!((brick_scaled_jacobian(&nodes, &brick) - 1.0).abs() < 1e-12);
/// ```
pub fn brick_scaled_jacobian(nodes: &[[f64; 3]], brick: &[usize; 8]) -> f64 {
    // Corner neighbors in OpenCourant bottom-then-top ordering. Triples are
    // right-handed: the top corners list the downward edge second so the
    // determinant stays positive on valid bricks.
    const NEIGHBORS: [[usize; 3]; 8] = [
        [1, 3, 4],
        [2, 0, 5],
        [3, 1, 6],
        [0, 2, 7],
        [5, 0, 7],
        [6, 1, 4],
        [7, 2, 5],
        [4, 3, 6],
    ];
    let corners: Vec<[f64; 3]> = brick.iter().map(|id| nodes[id - 1]).collect();
    let mut worst = 1.0_f64;
    for (corner, neighbors) in NEIGHBORS.iter().enumerate() {
        let origin = corners[corner];
        let edges: Vec<[f64; 3]> = neighbors
            .iter()
            .map(|neighbor| {
                let tip = corners[*neighbor];
                [tip[0] - origin[0], tip[1] - origin[1], tip[2] - origin[2]]
            })
            .collect();
        let det = edges[0][0] * (edges[1][1] * edges[2][2] - edges[1][2] * edges[2][1])
            - edges[0][1] * (edges[1][0] * edges[2][2] - edges[1][2] * edges[2][0])
            + edges[0][2] * (edges[1][0] * edges[2][1] - edges[1][1] * edges[2][0]);
        let denom = edge_len(&edges[0]) * edge_len(&edges[1]) * edge_len(&edges[2]);
        if denom <= 0.0 {
            return 0.0;
        }
        worst = worst.min(det / denom);
    }
    worst
}

/// Signed volume of one brick in cubic metres.
///
/// Five-tet decomposition sharing the main diagonal; negative flags an
/// inverted brick.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_mesh::brick_signed_volume;
///
/// let nodes = vec![
///     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
/// ];
/// let brick = [1, 2, 3, 4, 5, 6, 7, 8];
/// assert!((brick_signed_volume(&nodes, &brick) - 1.0).abs() < 1e-12);
/// ```
pub fn brick_signed_volume(nodes: &[[f64; 3]], brick: &[usize; 8]) -> f64 {
    let corners: Vec<[f64; 3]> = brick.iter().map(|id| nodes[id - 1]).collect();
    // Same five-tet split the rad harness tests use for brick checks.
    const TETS: [[usize; 4]; 5] = [
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 3, 4, 6],
        [3, 4, 6, 7],
        [1, 4, 5, 6],
    ];
    TETS.iter()
        .map(|tet| {
            tet_volume(
                &corners[tet[0]],
                &corners[tet[1]],
                &corners[tet[2]],
                &corners[tet[3]],
            )
        })
        .sum()
}

fn tet_volume(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3], d: &[f64; 3]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let ad = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    (ab[0] * (ac[1] * ad[2] - ac[2] * ad[1]) - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
        + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0]))
        / 6.0
}

fn newell_normal(corners: &[[f64; 3]]) -> [f64; 3] {
    let mut normal = [0.0, 0.0, 0.0];
    for i in 0..corners.len() {
        let current = corners[i];
        let next = corners[(i + 1) % corners.len()];
        normal[0] += (current[1] - next[1]) * (current[2] + next[2]);
        normal[1] += (current[2] - next[2]) * (current[0] + next[0]);
        normal[2] += (current[0] - next[0]) * (current[1] + next[1]);
    }
    normal
}

fn edge_len(edge: &[f64; 3]) -> f64 {
    (edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2]).sqrt()
}

/// Quality summary of one surface patch.
#[derive(Debug, Clone, Copy)]
pub struct PatchMetrics {
    /// Node count.
    pub nodes: usize,
    /// Quad face count.
    pub quads: usize,
    /// Triangle face count.
    pub tris: usize,
    /// Share of quad faces over all faces.
    pub quad_fraction: f64,
    /// Minimum scaled Jacobian over all faces.
    pub min_scaled_jacobian: f64,
}

impl fmt::Display for PatchMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "patch: {} nodes, {} quads, {} tris, quad fraction {:.3}, min scaled Jacobian {:.3}",
            self.nodes, self.quads, self.tris, self.quad_fraction, self.min_scaled_jacobian
        )
    }
}

/// Quality summary of one brick solid.
#[derive(Debug, Clone, Copy)]
pub struct SolidMetrics {
    /// Node count.
    pub nodes: usize,
    /// Brick count.
    pub bricks: usize,
    /// Minimum scaled Jacobian over all bricks.
    pub min_scaled_jacobian: f64,
    /// Minimum signed brick volume in cubic metres.
    pub min_signed_volume: f64,
    /// Closed-solid check: no open edges, no non-manifold faces.
    pub watertight: bool,
    /// Boundary edges not shared by exactly two boundary faces.
    pub open_edges: usize,
    /// Faces shared by more than two bricks.
    pub nonmanifold_faces: usize,
}

impl fmt::Display for SolidMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "solid: {} nodes, {} bricks, min scaled Jacobian {:.3}, min volume {:.3e} m^3, watertight {} (open edges {}, non-manifold faces {})",
            self.nodes,
            self.bricks,
            self.min_scaled_jacobian,
            self.min_signed_volume,
            self.watertight,
            self.open_edges,
            self.nonmanifold_faces
        )
    }
}

/// Measure one surface patch.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_mesh::{measure_patch, rect_patch};
///
/// let metrics = measure_patch(&rect_patch(2, 3, 1.0, 1.0));
/// assert_eq!((metrics.quads, metrics.tris), (6, 0));
/// ```
pub fn measure_patch(patch: &ShellPatch) -> PatchMetrics {
    PatchMetrics {
        nodes: patch.num_nodes(),
        quads: patch.num_quads(),
        tris: patch.num_tris(),
        quad_fraction: patch.quad_fraction(),
        min_scaled_jacobian: patch.min_scaled_jacobian(),
    }
}

/// Measure one brick solid, including watertightness.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_mesh::{measure_solid, rect_patch};
///
/// let mesh = rect_patch(2, 2, 1.0, 1.0).extrude(0.0, 1.0, 1).unwrap();
/// let metrics = measure_solid(&mesh);
/// assert!(metrics.watertight);
/// ```
pub fn measure_solid(mesh: &RadMesh) -> SolidMetrics {
    let mut min_jacobian = 1.0_f64;
    let mut min_volume = f64::INFINITY;
    for brick in &mesh.bricks {
        min_jacobian = min_jacobian.min(brick_scaled_jacobian(&mesh.nodes, brick));
        min_volume = min_volume.min(brick_signed_volume(&mesh.nodes, brick));
    }
    if mesh.bricks.is_empty() {
        min_volume = 0.0;
    }
    let (watertight, open_edges, nonmanifold_faces) = watertight_detail(mesh);
    SolidMetrics {
        nodes: mesh.num_nodes(),
        bricks: mesh.num_bricks(),
        min_scaled_jacobian: min_jacobian,
        min_signed_volume: min_volume,
        watertight,
        open_edges,
        nonmanifold_faces,
    }
}

/// Check whether a brick mesh is a closed solid.
///
/// Every interior face must be shared by exactly two bricks and every
/// boundary edge by exactly two boundary faces; empty meshes fail.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_mesh::{is_solid_watertight, rect_patch};
///
/// let mesh = rect_patch(1, 1, 1.0, 1.0).extrude(0.0, 1.0, 1).unwrap();
/// assert!(is_solid_watertight(&mesh));
/// ```
pub fn is_solid_watertight(mesh: &RadMesh) -> bool {
    watertight_detail(mesh).0
}

fn watertight_detail(mesh: &RadMesh) -> (bool, usize, usize) {
    if mesh.bricks.is_empty() {
        return (false, 0, 0);
    }
    // Faces in OpenCourant bottom-then-top ordering, canonically sorted.
    const FACES: [[usize; 4]; 6] = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ];
    // Sorted keys count sharing; one unsorted representative per key keeps
    // true edge adjacency (sorted tuples scramble it).
    let mut counts: HashMap<[usize; 4], (usize, [usize; 4])> = HashMap::new();
    for brick in &mesh.bricks {
        for face in &FACES {
            let oriented = [
                brick[face[0]],
                brick[face[1]],
                brick[face[2]],
                brick[face[3]],
            ];
            let mut key = oriented;
            key.sort_unstable();
            counts
                .entry(key)
                .and_modify(|slot| slot.0 += 1)
                .or_insert((1, oriented));
        }
    }
    let nonmanifold_faces = counts.values().filter(|slot| slot.0 > 2).count();
    let mut edges: HashMap<[usize; 2], usize> = HashMap::new();
    for (count, face) in counts.values() {
        if *count != 1 {
            continue;
        }
        for side in 0..4 {
            let mut edge = [face[side], face[(side + 1) % 4]];
            edge.sort_unstable();
            *edges.entry(edge).or_insert(0) += 1;
        }
    }
    let open_edges = edges.values().filter(|count| **count != 2).count();
    let watertight = nonmanifold_faces == 0 && open_edges == 0;
    (watertight, open_edges, nonmanifold_faces)
}

/// Mesh a rectangle with pure quads.
///
/// Covers `[0, width] x [0, height]` at z = 0 with `nx` by `ny` quads,
/// wound counter-clockwise seen from +z.
///
/// # Panics
///
/// Panics for zero counts or non-positive extents.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_mesh::rect_patch;
///
/// let patch = rect_patch(2, 3, 1.0, 2.0);
/// assert_eq!(patch.num_nodes(), 12);
/// assert_eq!(patch.num_quads(), 6);
/// ```
pub fn rect_patch(nx: usize, ny: usize, width_m: f64, height_m: f64) -> ShellPatch {
    assert!(nx > 0 && ny > 0, "rect patch needs positive counts");
    assert!(
        width_m > 0.0 && height_m > 0.0,
        "rect patch needs positive extents"
    );
    let id = |i: usize, j: usize| j * (nx + 1) + i + 1;
    let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1));
    for j in 0..=ny {
        for i in 0..=nx {
            nodes.push([
                width_m * i as f64 / nx as f64,
                height_m * j as f64 / ny as f64,
                0.0,
            ]);
        }
    }
    let mut quads = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            quads.push([id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)]);
        }
    }
    ShellPatch {
        nodes,
        quads,
        tris: Vec::new(),
    }
}

/// Mesh a disk with a center fan plus quad rings.
///
/// Node 1 is the center; ring `i` sits at `radius * i / n_rings` with
/// `n_sectors` nodes. The hub fan uses triangles, every outer band uses
/// quads, so the patch is quad-dominant.
///
/// # Panics
///
/// Panics for fewer than two rings, three sectors, or a non-positive
/// radius.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_mesh::disk_patch;
///
/// let patch = disk_patch(4, 12, 1.0);
/// assert_eq!(patch.num_tris(), 12);
/// assert_eq!(patch.num_quads(), 36);
/// ```
pub fn disk_patch(n_rings: usize, n_sectors: usize, radius_m: f64) -> ShellPatch {
    assert!(n_rings >= 2, "disk patch needs at least two rings");
    assert!(n_sectors >= 3, "disk patch needs at least three sectors");
    assert!(radius_m > 0.0, "disk patch needs a positive radius");
    // Node 1 is the center; ring i sector j follows in row-major order.
    let id = |ring: usize, sector: usize| 1 + (ring - 1) * n_sectors + sector + 1;
    let mut nodes = vec![[0.0, 0.0, 0.0]];
    for ring in 1..=n_rings {
        let radius = radius_m * ring as f64 / n_rings as f64;
        for sector in 0..n_sectors {
            let theta = 2.0 * std::f64::consts::PI * sector as f64 / n_sectors as f64;
            nodes.push([radius * theta.cos(), radius * theta.sin(), 0.0]);
        }
    }
    let mut tris = Vec::with_capacity(n_sectors);
    for sector in 0..n_sectors {
        tris.push([1, id(1, sector), id(1, (sector + 1) % n_sectors)]);
    }
    let mut quads = Vec::with_capacity((n_rings - 1) * n_sectors);
    for ring in 1..n_rings {
        for sector in 0..n_sectors {
            // Inner-outer-outer-inner keeps counter-clockwise winding from
            // +z (inner-inner-outer-outer would run clockwise).
            quads.push([
                id(ring, sector),
                id(ring + 1, sector),
                id(ring + 1, (sector + 1) % n_sectors),
                id(ring, (sector + 1) % n_sectors),
            ]);
        }
    }
    ShellPatch { nodes, quads, tris }
}

/// Tributary area per node of [`disk_patch`].
///
/// Half-station ring bands split over sectors; entries sum to the disk
/// area and index the patch nodes 0-based in patch order.
///
/// # Panics
///
/// Panics for the same invalid inputs as [`disk_patch`].
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_mesh::disk_nodal_areas;
///
/// let areas = disk_nodal_areas(1.0, 4, 12);
/// let total: f64 = areas.iter().sum();
/// assert!((total - std::f64::consts::PI).abs() < 1e-9);
/// ```
pub fn disk_nodal_areas(radius_m: f64, n_rings: usize, n_sectors: usize) -> Vec<f64> {
    assert!(n_rings >= 2, "disk areas need at least two rings");
    assert!(n_sectors >= 3, "disk areas need at least three sectors");
    assert!(radius_m > 0.0, "disk areas need a positive radius");
    let ring_radius = |ring: usize| radius_m * ring as f64 / n_rings as f64;
    // Outer half-station of ring i (i = 0 is the center hub edge).
    let station = |ring: usize| {
        if ring == 0 {
            ring_radius(1) / 2.0
        } else if ring >= n_rings {
            radius_m
        } else {
            (ring_radius(ring) + ring_radius(ring + 1)) / 2.0
        }
    };
    let id = |ring: usize, sector: usize| 1 + (ring - 1) * n_sectors + sector + 1;
    let mut areas = vec![0.0; 1 + n_rings * n_sectors];
    areas[0] = std::f64::consts::PI * station(0).powi(2);
    let mut inner = station(0);
    for ring in 1..=n_rings {
        let outer = station(ring);
        let band = std::f64::consts::PI * (outer.powi(2) - inner.powi(2)) / n_sectors as f64;
        for sector in 0..n_sectors {
            areas[id(ring, sector) - 1] = band;
        }
        inner = outer;
    }
    areas
}

/// Mesh a circular section with projected pure quads.
///
/// Clips a `cells_per_side` Cartesian grid to the disk and projects
/// region-boundary nodes radially onto the circle, so the Taylor bar
/// section keeps clean boundary quads without triangles.
///
/// # Panics
///
/// Panics for fewer than two cells per side or a non-positive radius.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_mesh::taylor_patch;
///
/// let patch = taylor_patch(6, 0.003_175);
/// assert!(patch.tris.is_empty());
/// assert!(!patch.quads.is_empty());
/// ```
pub fn taylor_patch(cells_per_side: usize, radius_m: f64) -> ShellPatch {
    assert!(
        cells_per_side >= 2,
        "taylor patch needs at least two cells per side"
    );
    assert!(radius_m > 0.0, "taylor patch needs a positive radius");
    let n = cells_per_side;
    let step = 2.0 * radius_m / n as f64;
    let corner = |i: usize| -radius_m + step * i as f64;
    let mut kept = vec![false; n * n];
    for i in 0..n {
        for j in 0..n {
            kept[i * n + j] = (0..=1).all(|di| {
                (0..=1).all(|dj| {
                    let (x, y) = (corner(i + di), corner(j + dj));
                    x * x + y * y <= radius_m * radius_m
                })
            });
        }
    }
    let mut grid_id = vec![0usize; (n + 1) * (n + 1)];
    let mut nodes = Vec::new();
    for i in 0..=n {
        for j in 0..=n {
            let mut used = false;
            let mut boundary = false;
            for di in 0..=1 {
                for dj in 0..=1 {
                    let (ci, cj) = (i + 1 - di, j + 1 - dj);
                    if ci == 0 || cj == 0 || ci > n || cj > n {
                        continue;
                    }
                    if kept[(ci - 1) * n + (cj - 1)] {
                        used = true;
                    } else {
                        boundary = true;
                    }
                }
            }
            if i == 0 || j == 0 || i == n || j == n {
                boundary = true;
            }
            if !used {
                continue;
            }
            let (mut x, mut y) = (corner(i), corner(j));
            if boundary {
                let r = (x * x + y * y).sqrt();
                if r > 1e-12 {
                    x *= radius_m / r;
                    y *= radius_m / r;
                }
            }
            grid_id[i * (n + 1) + j] = nodes.len() + 1;
            nodes.push([x, y, 0.0]);
        }
    }
    let mut quads = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if !kept[i * n + j] {
                continue;
            }
            let at = |di: usize, dj: usize| grid_id[(i + di) * (n + 1) + (j + dj)];
            quads.push([at(0, 0), at(1, 0), at(1, 1), at(0, 1)]);
        }
    }
    ShellPatch {
        nodes,
        quads,
        tris: Vec::new(),
    }
}

/// Mesh a square tube wall section with pure quads.
///
/// The wall between the outer and inner squares uses `side_quads` cells
/// along each side and `wall_quads` cells across the thickness; every
/// cell is a rectangle.
///
/// # Panics
///
/// Panics for non-positive counts or widths that do not nest
/// (`outer > inner > 0`).
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_mesh::square_ring_patch;
///
/// let patch = square_ring_patch(0.05, 0.046, 4, 1);
/// assert!(patch.tris.is_empty());
/// assert!(!patch.quads.is_empty());
/// ```
pub fn square_ring_patch(
    outer_m: f64,
    inner_m: f64,
    side_quads: usize,
    wall_quads: usize,
) -> ShellPatch {
    assert!(
        side_quads > 0 && wall_quads > 0,
        "square ring needs positive counts"
    );
    assert!(
        outer_m > inner_m && inner_m > 0.0,
        "square ring needs outer > inner > 0"
    );
    // Grid lines cluster at the wall: outer band, middle span, outer band.
    let half_outer = outer_m / 2.0;
    let half_inner = inner_m / 2.0;
    let mut lines = Vec::with_capacity(2 * wall_quads + side_quads + 1);
    for k in 0..=wall_quads {
        lines.push(-half_outer + (half_outer - half_inner) * k as f64 / wall_quads as f64);
    }
    for k in 1..side_quads {
        lines.push(-half_inner + 2.0 * half_inner * k as f64 / side_quads as f64);
    }
    for k in 0..=wall_quads {
        lines.push(half_inner + (half_outer - half_inner) * k as f64 / wall_quads as f64);
    }
    let n = lines.len() - 1;
    let middle = wall_quads..(wall_quads + side_quads);
    let mut grid_id = vec![0usize; (n + 1) * (n + 1)];
    let mut nodes = Vec::new();
    let node_at = |i: usize, j: usize, grid_id: &mut Vec<usize>, nodes: &mut Vec<[f64; 3]>| {
        if grid_id[i * (n + 1) + j] == 0 {
            grid_id[i * (n + 1) + j] = nodes.len() + 1;
            nodes.push([lines[i], lines[j], 0.0]);
        }
        grid_id[i * (n + 1) + j]
    };
    let mut quads = Vec::new();
    for i in 0..n {
        for j in 0..n {
            // Drop the hollow middle block; keep the full wall ring.
            if middle.contains(&i) && middle.contains(&j) {
                continue;
            }
            let a = node_at(i, j, &mut grid_id, &mut nodes);
            let b = node_at(i + 1, j, &mut grid_id, &mut nodes);
            let c = node_at(i + 1, j + 1, &mut grid_id, &mut nodes);
            let d = node_at(i, j + 1, &mut grid_id, &mut nodes);
            quads.push([a, b, c, d]);
        }
    }
    ShellPatch {
        nodes,
        quads,
        tris: Vec::new(),
    }
}

/// Mesh an annulus wall section with pure quads.
///
/// Polar grid with `n_rings` radial bands and `n_sectors` sectors between
/// the inner and outer radii, wound counter-clockwise seen from +z.
///
/// # Panics
///
/// Panics for a zero ring count, fewer than three sectors, or radii that
/// do not nest (`outer > inner > 0`).
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_mesh::annulus_patch;
///
/// let patch = annulus_patch(0.025, 0.023, 1, 24);
/// assert_eq!(patch.num_quads(), 24);
/// ```
pub fn annulus_patch(outer_m: f64, inner_m: f64, n_rings: usize, n_sectors: usize) -> ShellPatch {
    assert!(n_rings > 0, "annulus needs at least one ring");
    assert!(n_sectors >= 3, "annulus needs at least three sectors");
    assert!(
        outer_m > inner_m && inner_m > 0.0,
        "annulus needs outer > inner > 0"
    );
    let id = |ring: usize, sector: usize| ring * n_sectors + sector + 1;
    let mut nodes = Vec::with_capacity((n_rings + 1) * n_sectors);
    for ring in 0..=n_rings {
        let radius = inner_m + (outer_m - inner_m) * ring as f64 / n_rings as f64;
        for sector in 0..n_sectors {
            let theta = 2.0 * std::f64::consts::PI * sector as f64 / n_sectors as f64;
            nodes.push([radius * theta.cos(), radius * theta.sin(), 0.0]);
        }
    }
    let mut quads = Vec::with_capacity(n_rings * n_sectors);
    for ring in 0..n_rings {
        for sector in 0..n_sectors {
            // Inner-outer-outer-inner keeps counter-clockwise winding from
            // +z (inner-inner-outer-outer would run clockwise).
            quads.push([
                id(ring, sector),
                id(ring + 1, sector),
                id(ring + 1, (sector + 1) % n_sectors),
                id(ring, (sector + 1) % n_sectors),
            ]);
        }
    }
    ShellPatch {
        nodes,
        quads,
        tris: Vec::new(),
    }
}

/// Build the Taylor bar impact case from an extruded section.
///
/// A copper bar floats one approach gap above a rising anvil; by Galilean
/// equivalence the contact force matches the flying-bar test. Probes read
/// the anvil force plateau (`rho c A V`) and the impact-face travel. Uses
/// existing [`OpenCourantCase`] fields only.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_mesh::taylor_bar_case;
///
/// let case = taylor_bar_case();
/// case.mesh.validate().unwrap();
/// assert_eq!(case.probes.len(), 2);
/// ```
pub fn taylor_bar_case() -> OpenCourantCase {
    let section = taylor_patch(6, TAYLOR_RADIUS_M);
    // One microsecond of free approach before contact starts.
    let gap = TAYLOR_VELOCITY_M_PER_S * 1.0e-6;
    let stop = 4.0e-6;
    let mut mesh = section
        .extrude(gap, gap + TAYLOR_LENGTH_M, 8)
        .expect("taylor section is pure quad");
    let per_layer = section.num_nodes();
    let face_center = section
        .nodes
        .iter()
        .enumerate()
        .min_by(|a, b| {
            let ra = a.1[0] * a.1[0] + a.1[1] * a.1[1];
            let rb = b.1[0] * b.1[0] + b.1[1] * b.1[1];
            ra.total_cmp(&rb)
        })
        .map(|(index, _)| index + 1)
        .unwrap_or(1);
    let face_set: Vec<usize> = (1..=per_layer).collect();
    // Secondary group covers the whole bar (mushrooming spreads contact
    // past the initial face); the driver node is appended after so it
    // stays out of the contact group.
    let bar_nodes: Vec<usize> = (1..=mesh.num_nodes()).collect();
    let anvil_point = [0.0, 0.0, 0.0];
    let driver = mesh.add_node(anvil_point);
    let area = std::f64::consts::PI * TAYLOR_RADIUS_M * TAYLOR_RADIUS_M;
    let youngs: f64 = 117.0e9;
    let density: f64 = 8930.0;
    let plateau = density * (youngs / density).sqrt() * area * TAYLOR_VELOCITY_M_PER_S;
    let travel = TAYLOR_VELOCITY_M_PER_S * (stop - 1.0e-6);
    OpenCourantCase {
        name: "mesh_taylor_bar".to_string(),
        title: "sonium_qa rad_mesh taylor bar elastic impact".to_string(),
        mesh,
        material: RadElasticMaterial {
            youngs_modulus_pa: youngs,
            poisson_ratio: 0.35,
            density_kg_per_m3: density,
        },
        node_sets: vec![
            RadNodeSet {
                name: "impact face".to_string(),
                nodes: face_set,
            },
            RadNodeSet {
                name: "bar".to_string(),
                nodes: bar_nodes,
            },
        ],
        bcs: Vec::new(),
        functions: vec![RadFunction {
            name: "constant wall velocity".to_string(),
            points: vec![(0.0, 1.0), (stop, 1.0)],
        }],
        cloads: Vec::new(),
        walls: vec![RadWall {
            name: "rising anvil".to_string(),
            normal: [0.0, 0.0, 1.0],
            point_m: anvil_point,
            friction: 0.0,
            velocity_m_per_s: Some([0.0, 0.0, TAYLOR_VELOCITY_M_PER_S]),
            driver_node: Some(driver),
            secondary_set: Some(1),
            search_distance_m: 0.0,
        }],
        th_nodes: vec![face_center],
        probes: vec![
            RadProbe {
                label: "anvil FNZ plateau".to_string(),
                channel: "FNZ/RWALL/1".to_string(),
                // Wall channels report the force ON the wall, so the
                // rising anvil reads negative.
                expected: -plateau,
                abs_tol: 1.0e3,
                rel_tol: 0.25,
            },
            RadProbe {
                label: "impact face DZ".to_string(),
                channel: format!("DZ/{face_center}"),
                expected: travel,
                abs_tol: 1e-9,
                rel_tol: 0.05,
            },
        ],
        stop_time_s: stop,
        th_period_s: 1.0e-7,
        anim_period_s: 1.0e-6,
    }
}

/// Build the square-tube crush case from an extruded ring.
///
/// A hollow steel box column along z crushes 1% quasi-statically between
/// a fixed anvil and a moving crusher. Probes read the elastic crusher
/// force (`E A delta / L`) and the loaded-face travel. Uses existing
/// [`OpenCourantCase`] fields only.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_mesh::crash_box_case;
///
/// let case = crash_box_case();
/// case.mesh.validate().unwrap();
/// assert_eq!(case.walls.len(), 2);
/// ```
pub fn crash_box_case() -> OpenCourantCase {
    let section = square_ring_patch(CRASH_OUTER_M, CRASH_INNER_M, 4, 1);
    let layers = 10;
    let mut mesh = section
        .extrude(0.0, CRASH_LENGTH_M, layers)
        .expect("crash section is pure quad");
    let youngs = 200.0e9;
    let crush = 0.01 * CRASH_LENGTH_M;
    let speed = 0.1;
    let stop = crush / speed;
    let area = CRASH_OUTER_M * CRASH_OUTER_M - CRASH_INNER_M * CRASH_INNER_M;
    let expected_force = youngs * area * crush / CRASH_LENGTH_M;
    // Split secondary groups (root/tip halves): a node secondary to both
    // walls trips starter warning 312. The driver node is appended after
    // so it stays out of both groups.
    let mut root_nodes = Vec::new();
    let mut tip_nodes = Vec::new();
    for (index, node) in mesh.nodes.iter().enumerate() {
        if node[2] <= CRASH_LENGTH_M / 2.0 {
            root_nodes.push(index + 1);
        } else {
            tip_nodes.push(index + 1);
        }
    }
    // Last pre-driver node: top layer, rides the full crush travel.
    let top_node = mesh.num_nodes();
    let crusher_point = [0.0, 0.0, CRASH_LENGTH_M];
    let driver = mesh.add_node(crusher_point);
    OpenCourantCase {
        name: "mesh_crash_box".to_string(),
        title: "sonium_qa rad_mesh square tube crush".to_string(),
        mesh,
        material: RadElasticMaterial {
            youngs_modulus_pa: youngs,
            poisson_ratio: 0.3,
            density_kg_per_m3: 7800.0,
        },
        node_sets: vec![
            RadNodeSet {
                name: "column root".to_string(),
                nodes: root_nodes,
            },
            RadNodeSet {
                name: "column tip".to_string(),
                nodes: tip_nodes,
            },
        ],
        bcs: Vec::new(),
        functions: vec![RadFunction {
            name: "constant crush velocity".to_string(),
            points: vec![(0.0, 1.0), (stop, 1.0)],
        }],
        cloads: Vec::new(),
        walls: vec![
            RadWall {
                name: "fixed anvil".to_string(),
                normal: [0.0, 0.0, 1.0],
                point_m: [0.0, 0.0, 0.0],
                friction: 0.0,
                velocity_m_per_s: None,
                driver_node: None,
                secondary_set: Some(0),
                search_distance_m: 0.0,
            },
            RadWall {
                name: "crusher".to_string(),
                normal: [0.0, 0.0, -1.0],
                point_m: crusher_point,
                friction: 0.0,
                velocity_m_per_s: Some([0.0, 0.0, -speed]),
                driver_node: Some(driver),
                secondary_set: Some(1),
                search_distance_m: 0.0,
            },
        ],
        th_nodes: vec![top_node],
        probes: vec![
            RadProbe {
                label: "crusher FNZ".to_string(),
                channel: "FNZ/RWALL/2".to_string(),
                // Reaction on the crusher points +z (see rail crush leg).
                expected: expected_force,
                abs_tol: 1.0e4,
                rel_tol: 0.15,
            },
            RadProbe {
                label: "loaded face DZ".to_string(),
                channel: format!("DZ/{top_node}"),
                expected: -crush,
                abs_tol: 1e-9,
                rel_tol: 0.05,
            },
        ],
        stop_time_s: stop,
        th_period_s: stop / 40.0,
        anim_period_s: stop / 4.0,
    }
}

/// Build the clamped plate shell case from a disk patch.
///
/// A polar QEPH disk with a clamped rim carries uniform pressure as
/// ring-wise nodal loads; the center probe replays the Kirchhoff closed
/// form. Exercises the quad-dominant patch (quads plus center fan) as
/// shell elements through the extended writer.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_mesh::shell_plate_case;
///
/// let case = shell_plate_case();
/// case.validate().unwrap();
/// assert!(!case.shells.tris.is_empty());
/// ```
pub fn shell_plate_case() -> RadExtendedCase {
    let (young, poisson, pressure): (f64, f64, f64) = (200.0e9, 0.3, 1000.0);
    let (n_rings, n_sectors) = (6, 24);
    let patch = disk_patch(n_rings, n_sectors, PLATE_RADIUS_M);
    let areas = disk_nodal_areas(PLATE_RADIUS_M, n_rings, n_sectors);
    let ring_nodes = |ring: usize| {
        if ring == 0 {
            vec![1]
        } else {
            (0..n_sectors)
                .map(|sector| 1 + (ring - 1) * n_sectors + sector + 1)
                .collect::<Vec<usize>>()
        }
    };
    let rim = ring_nodes(n_rings);
    // One node set per ring: tributary areas are uniform within a ring.
    let mut node_sets = vec![RadNodeSet {
        name: "rim".to_string(),
        nodes: rim,
    }];
    let mut cloads = Vec::new();
    for ring in 0..=n_rings {
        let nodes = ring_nodes(ring);
        let area = areas[nodes[0] - 1];
        node_sets.push(RadNodeSet {
            name: format!("pressure ring {ring}"),
            nodes: nodes.clone(),
        });
        cloads.push(RadConcentratedLoad {
            name: format!("pressure ring {ring}"),
            set: node_sets.len() - 1,
            direction: 3,
            function: 0,
            force_scale_n: -pressure * area,
        });
    }
    // Kirchhoff center deflection of the clamped circular plate.
    let rigidity = young * PLATE_THICK_M.powi(3) / (12.0 * (1.0 - poisson.powi(2)));
    let center = -pressure * PLATE_RADIUS_M.powi(4) / (64.0 * rigidity);
    // Radial moment at the probe quad centroid radius (Timoshenko
    // plates): the quad N1-N2 edge runs radially after the
    // counter-clockwise fix, so M1 reads M_r, not the hoop moment.
    let probe_radius = PLATE_RADIUS_M * 1.5 / n_rings as f64;
    let radial = pressure / 16.0
        * ((1.0 + poisson) * PLATE_RADIUS_M.powi(2) - (3.0 + poisson) * probe_radius.powi(2));
    let base = OpenCourantCase {
        name: "mesh_shell_plate".to_string(),
        title: "sonium_qa rad_mesh clamped plate pressure".to_string(),
        mesh: RadMesh {
            nodes: patch.nodes.clone(),
            bricks: Vec::new(),
        },
        material: RadElasticMaterial {
            youngs_modulus_pa: young,
            poisson_ratio: poisson,
            density_kg_per_m3: 7800.0,
        },
        node_sets,
        bcs: vec![RadBoundaryCondition {
            name: "clamp rim".to_string(),
            set: 0,
            lock: [true, true, true, true, true, true],
        }],
        functions: vec![RadFunction {
            name: "ramp 0 to 1 then hold".to_string(),
            points: vec![(0.0, 0.0), (0.8, 1.0), (1.0, 1.0)],
        }],
        cloads,
        walls: Vec::new(),
        th_nodes: vec![1],
        probes: vec![
            RadProbe {
                label: "center DZ".to_string(),
                channel: "DZ/1".to_string(),
                expected: center,
                abs_tol: 1e-9,
                rel_tol: 0.05,
            },
            RadProbe {
                label: "probe shell THIC".to_string(),
                channel: shel_channel("THIC", 1),
                expected: PLATE_THICK_M,
                abs_tol: 1e-9,
                rel_tol: 1e-6,
            },
            RadProbe {
                label: "probe shell M1".to_string(),
                channel: shel_channel("M1", 1),
                // M1 carries moment per unit length per thickness squared.
                expected: radial / PLATE_THICK_M.powi(2),
                abs_tol: 1e3,
                rel_tol: 0.15,
            },
        ],
        stop_time_s: 1.0,
        th_period_s: 0.01,
        anim_period_s: 0.1,
    };
    let mut case = RadExtendedCase::wrap_elastic(base);
    case.shells = RadShellMesh {
        quads: patch.quads,
        tris: patch.tris,
    };
    case.shell_prop = Some(RadShellProperty::qeph("plate section", PLATE_THICK_M));
    case.th_shell = RadThRequest::shell_resultants(vec![1]);
    case
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_patch_counts_and_winding() {
        let patch = rect_patch(2, 3, 1.0, 2.0);
        patch.validate().unwrap();
        assert_eq!(patch.num_nodes(), 12);
        assert_eq!(patch.num_quads(), 6);
        assert!(patch.tris.is_empty());
        assert!((patch.quad_fraction() - 1.0).abs() < 1e-15);
        assert!((patch.min_scaled_jacobian() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn disk_patch_is_quad_dominant() {
        let patch = disk_patch(6, 24, 1.0);
        patch.validate().unwrap();
        assert_eq!(patch.num_nodes(), 1 + 6 * 24);
        assert_eq!(patch.num_tris(), 24);
        assert_eq!(patch.num_quads(), 5 * 24);
        assert!((patch.quad_fraction() - 120.0 / 144.0).abs() < 1e-15);
        assert!(patch.min_scaled_jacobian() > MIN_SCALED_JACOBIAN);
    }

    #[test]
    fn taylor_patch_stays_pure_quad() {
        let patch = taylor_patch(6, TAYLOR_RADIUS_M);
        patch.validate().unwrap();
        assert!(patch.tris.is_empty());
        assert!(!patch.quads.is_empty());
        // Boundary nodes land on the circle after radial projection.
        for node in &patch.nodes {
            let r = (node[0] * node[0] + node[1] * node[1]).sqrt();
            assert!(r <= TAYLOR_RADIUS_M + 1e-12, "node escapes: {node:?}");
        }
        assert!(patch.min_scaled_jacobian() > MIN_SCALED_JACOBIAN);
    }

    #[test]
    fn ring_patches_tile_their_walls() {
        let square = square_ring_patch(0.05, 0.046, 4, 1);
        square.validate().unwrap();
        assert!(square.tris.is_empty());
        // 6x6 cells minus the 4x4 hollow middle block.
        assert_eq!(square.num_quads(), 36 - 16);
        assert!((square.min_scaled_jacobian() - 1.0).abs() < 1e-12);
        let annulus = annulus_patch(0.025, 0.023, 2, 24);
        annulus.validate().unwrap();
        assert_eq!(annulus.num_quads(), 48);
        assert!(annulus.min_scaled_jacobian() > 0.99);
    }

    #[test]
    fn delivered_patches_face_positive_z() {
        let patches = [
            rect_patch(3, 2, 0.1, 0.2),
            disk_patch(6, 24, 1.0),
            taylor_patch(6, TAYLOR_RADIUS_M),
            square_ring_patch(CRASH_OUTER_M, CRASH_INNER_M, 4, 1),
            annulus_patch(0.025, 0.023, 1, 24),
            disk_patch(4, 12, 1.0).to_all_quad(),
        ];
        for patch in &patches {
            assert!(!patch.quads.is_empty());
            for quad in &patch.quads {
                let corners = [
                    patch.nodes[quad[0] - 1],
                    patch.nodes[quad[1] - 1],
                    patch.nodes[quad[2] - 1],
                    patch.nodes[quad[3] - 1],
                ];
                assert!(newell_normal(&corners)[2] > 0.0, "quad {quad:?} faces -z");
            }
        }
    }

    #[test]
    fn to_all_quad_clears_tris_and_keeps_quality() {
        let patch = disk_patch(4, 12, 1.0).to_all_quad();
        patch.validate().unwrap();
        assert!(patch.tris.is_empty());
        // 49 nodes plus 96 shared edge midpoints plus 48 centroids; every
        // quad splits into four and every tri into three.
        assert_eq!(patch.num_nodes(), 49 + 96 + 48);
        assert_eq!(patch.num_quads(), 36 * 4 + 12 * 3);
        assert!((patch.quad_fraction() - 1.0).abs() < 1e-15);
        assert!(patch.min_scaled_jacobian() > MIN_SCALED_JACOBIAN);
    }

    #[test]
    fn extrude_rejects_tris_and_bad_spans() {
        let patch = disk_patch(4, 12, 1.0);
        assert!(patch.extrude(0.0, 1.0, 2).is_err());
        let clean = patch.to_all_quad();
        assert!(clean.extrude(0.0, 1.0, 0).is_err());
        assert!(clean.extrude(1.0, 0.0, 1).is_err());
        assert!(clean.extrude(f64::NAN, 1.0, 1).is_err());
        let mut flipped = rect_patch(1, 1, 1.0, 1.0);
        flipped.quads[0] = [1, 4, 3, 2];
        assert!(flipped.extrude(0.0, 1.0, 1).is_err());
    }

    #[test]
    fn extrusions_are_watertight_positive_solids() {
        let patches = [
            rect_patch(3, 2, 0.1, 0.2),
            taylor_patch(6, TAYLOR_RADIUS_M),
            square_ring_patch(CRASH_OUTER_M, CRASH_INNER_M, 4, 1),
            annulus_patch(0.025, 0.023, 1, 24),
            disk_patch(4, 12, 1.0).to_all_quad(),
        ];
        for patch in &patches {
            let mesh = patch.extrude(0.0, 0.1, 3).unwrap();
            mesh.validate().unwrap();
            let metrics = measure_solid(&mesh);
            assert!(metrics.watertight, "open edges: {}", metrics.open_edges);
            assert_eq!(metrics.nonmanifold_faces, 0);
            assert!(metrics.min_signed_volume > 0.0);
            assert!(
                metrics.min_scaled_jacobian > MIN_SCALED_JACOBIAN,
                "jacobian: {}",
                metrics.min_scaled_jacobian
            );
        }
    }

    #[test]
    fn watertight_check_catches_open_solids() {
        let mesh = rect_patch(1, 1, 1.0, 1.0).extrude(0.0, 1.0, 2).unwrap();
        assert!(is_solid_watertight(&mesh));
        // A non-conformal stack (top brick bottom face shares only three
        // nodes with the bottom brick top face) leaves open edges.
        let mut open = rect_patch(1, 1, 1.0, 1.0).extrude(0.0, 1.0, 2).unwrap();
        open.nodes.push(open.nodes[4]);
        open.bricks[1][0] = open.nodes.len();
        let metrics = measure_solid(&open);
        assert!(!metrics.watertight);
        assert!(metrics.open_edges > 0);
        // A triple brick shares every face three times (non-manifold).
        let mut folded = rect_patch(1, 1, 1.0, 1.0).extrude(0.0, 1.0, 1).unwrap();
        folded.bricks.push(folded.bricks[0]);
        folded.bricks.push(folded.bricks[0]);
        let metrics = measure_solid(&folded);
        assert!(!metrics.watertight);
        assert_eq!(metrics.nonmanifold_faces, 6);
        let empty = RadMesh {
            nodes: Vec::new(),
            bricks: Vec::new(),
        };
        assert!(!is_solid_watertight(&empty));
    }

    #[test]
    fn quality_kernels_flag_degenerate_corners() {
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        assert!((quad_scaled_jacobian(&nodes, &[1, 2, 3, 4]) - 1.0).abs() < 1e-12);
        // Planar scores are winding-invariant; bow-tie quads collapse to 0.
        assert!((quad_scaled_jacobian(&nodes, &[4, 3, 2, 1]) - 1.0).abs() < 1e-12);
        let bowtie = vec![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        assert_eq!(quad_scaled_jacobian(&bowtie, &[1, 2, 3, 4]), 0.0);
        let collapsed = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert_eq!(tri_scaled_jacobian(&collapsed, &[1, 2, 3]), 0.0);
    }

    #[test]
    fn patch_validation_catches_bad_connectivity() {
        let mut patch = rect_patch(1, 1, 1.0, 1.0);
        patch.quads.push([1, 2, 3, 99]);
        assert!(patch.validate().is_err());
        let mut patch = rect_patch(1, 1, 1.0, 1.0);
        patch.nodes[0] = [f64::NAN, 0.0, 0.0];
        assert!(patch.validate().is_err());
    }

    #[test]
    fn metrics_display_reports_key_numbers() {
        let patch = disk_patch(4, 12, 1.0);
        let text = format!("{}", measure_patch(&patch));
        assert!(text.contains("36 quads"));
        assert!(text.contains("12 tris"));
        assert!(text.contains("quad fraction"));
        let mesh = rect_patch(1, 1, 1.0, 1.0).extrude(0.0, 1.0, 1).unwrap();
        let text = format!("{}", measure_solid(&mesh));
        assert!(text.contains("watertight true"));
        assert!(text.contains("min scaled Jacobian"));
    }

    #[test]
    fn taylor_case_matches_closed_forms() {
        let case = taylor_bar_case();
        case.mesh.validate().unwrap();
        assert!(!case.mesh.bricks.is_empty());
        let area = std::f64::consts::PI * TAYLOR_RADIUS_M * TAYLOR_RADIUS_M;
        let plateau = 8930.0 * (117.0e9f64 / 8930.0).sqrt() * area * 227.0;
        assert!((case.probes[0].expected + plateau).abs() < 1e-9);
        assert!((case.probes[1].expected - 227.0 * 3.0e-6).abs() < 1e-15);
        let metrics = measure_solid(&case.mesh);
        assert!(metrics.min_scaled_jacobian > MIN_SCALED_JACOBIAN);
    }

    #[test]
    fn crash_case_matches_closed_forms() {
        let case = crash_box_case();
        case.mesh.validate().unwrap();
        assert!(!case.mesh.bricks.is_empty());
        let area = CRASH_OUTER_M * CRASH_OUTER_M - CRASH_INNER_M * CRASH_INNER_M;
        let expected = 200.0e9 * area * 0.01 * CRASH_LENGTH_M / CRASH_LENGTH_M;
        assert!((case.probes[0].expected - expected).abs() < 1e-6);
        assert!((case.probes[1].expected + 0.01 * CRASH_LENGTH_M).abs() < 1e-15);
        // Contact groups split the column and exclude the driver node.
        assert_eq!(
            case.node_sets[0].nodes.len() + case.node_sets[1].nodes.len(),
            case.mesh.num_nodes() - 1
        );
    }

    #[test]
    fn shell_plate_case_matches_closed_forms() {
        let case = shell_plate_case();
        case.validate().unwrap();
        assert!(!case.shells.quads.is_empty());
        assert!(!case.shells.tris.is_empty());
        let rigidity = 200.0e9 * 0.02_f64.powi(3) / (12.0 * (1.0 - 0.3_f64.powi(2)));
        let exact = -1000.0 / (64.0 * rigidity);
        assert!((case.base.probes[0].expected - exact).abs() < 1e-15);
        // Ring loads integrate to the pressure resultant.
        let mut resultant = 0.0;
        for load in &case.base.cloads {
            let set = &case.base.node_sets[load.set];
            resultant += load.force_scale_n * set.nodes.len() as f64;
        }
        assert!(
            (resultant + 1000.0 * std::f64::consts::PI).abs() < 1e-6,
            "pressure resultant is {resultant}"
        );
    }
}
