//! Mesh quality metrics for all supported element types.
//!
//! Each metric is normalised so a regular reference element scores **1.0**
//! and a degenerate element scores **0.0** (lower is worse). This lets the
//! same threshold drive marking decisions across triangle, quad, tet, and hex
//! meshes:
//!
//! - [`mean_ratio`] — Knupp's algebraic mean-ratio (`q = 1` for the regular
//!   simplex, `q → 0` as a corner collapses). Defined for triangles and
//!   tetrahedra. For quads and hexes we report the minimum corner Jacobian
//!   ratio, which behaves the same way.
//! - [`aspect_ratio`] — bounding-edge / inradius normalised to the regular
//!   reference (1.0 for an equilateral / regular element, grows without
//!   bound for a sliver). Inverted to 1/aspect by [`aspect_quality`] so it
//!   composes with [`mean_ratio`].
//! - [`min_dihedral_angle`] / [`max_dihedral_angle`] — in radians, only
//!   meaningful for 3D elements.
//!
//! Histogram helpers ([`edge_length_histogram`], [`element_measure_histogram`])
//! aggregate the per-element scalars into bucketed counts for diagnostics.

use super::types::{Element, ElementType, Mesh, Point};
use serde::Serialize;
use std::fmt;

/// Per-element quality summary returned by [`per_element_quality`].
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ElementQuality {
    pub mean_ratio: f64,
    pub aspect_quality: f64,
    pub min_dihedral: f64,
    pub max_dihedral: f64,
}

impl Default for ElementQuality {
    fn default() -> Self {
        Self {
            mean_ratio: 0.0,
            aspect_quality: 0.0,
            min_dihedral: 0.0,
            max_dihedral: 0.0,
        }
    }
}

/// Compute a per-element quality summary for every element in the mesh,
/// in element-index order.
pub fn per_element_quality(mesh: &Mesh) -> Vec<ElementQuality> {
    mesh.elements
        .iter()
        .map(|elem| ElementQuality {
            mean_ratio: mean_ratio(mesh, elem),
            aspect_quality: aspect_quality(mesh, elem),
            min_dihedral: min_dihedral_angle(mesh, elem),
            max_dihedral: max_dihedral_angle(mesh, elem),
        })
        .collect()
}

/// Knupp's algebraic mean ratio for a single element.
///
/// 1.0 for a regular reference element (equilateral triangle, regular
/// tetrahedron, square, cube). Returns 0.0 for degenerate or unsupported
/// configurations. For quads and hexes this is implemented as the minimum
/// over the element's corners of the local-Jacobian-to-edge-norm ratio,
/// which agrees with the simplex definition on the reference element.
pub fn mean_ratio(mesh: &Mesh, elem: &Element) -> f64 {
    if !has_valid_vertices(mesh, elem) {
        return 0.0;
    }
    // Borrow nodes by index into stack arrays: the hot per-element loop must
    // not allocate a `Vec` per element.
    match elem.element_type {
        ElementType::Triangle => {
            let idx = elem.vertices();
            let v = [mesh.nodes[idx[0]], mesh.nodes[idx[1]], mesh.nodes[idx[2]]];
            triangle_mean_ratio(&v)
        }
        ElementType::Tetrahedron => {
            let idx = elem.vertices();
            let v = [
                mesh.nodes[idx[0]],
                mesh.nodes[idx[1]],
                mesh.nodes[idx[2]],
                mesh.nodes[idx[3]],
            ];
            tet_mean_ratio(&v)
        }
        ElementType::Quadrilateral => {
            let idx = elem.vertices();
            let v = [
                mesh.nodes[idx[0]],
                mesh.nodes[idx[1]],
                mesh.nodes[idx[2]],
                mesh.nodes[idx[3]],
            ];
            quad_corner_ratio(&v)
        }
        ElementType::Hexahedron => {
            let idx = elem.vertices();
            let v = [
                mesh.nodes[idx[0]],
                mesh.nodes[idx[1]],
                mesh.nodes[idx[2]],
                mesh.nodes[idx[3]],
                mesh.nodes[idx[4]],
                mesh.nodes[idx[5]],
                mesh.nodes[idx[6]],
                mesh.nodes[idx[7]],
            ];
            hex_corner_ratio(&v)
        }
    }
}

fn triangle_mean_ratio(v: &[Point]) -> f64 {
    // q = 4·√3·A / (e0² + e1² + e2²), maps equilateral → 1.
    let e0 = edge_sq(&v[0], &v[1]);
    let e1 = edge_sq(&v[1], &v[2]);
    let e2 = edge_sq(&v[2], &v[0]);
    let s = e0 + e1 + e2;
    if s <= 0.0 {
        return 0.0;
    }
    let ax = v[1].x - v[0].x;
    let ay = v[1].y - v[0].y;
    let bx = v[2].x - v[0].x;
    let by = v[2].y - v[0].y;
    let area = 0.5 * (ax * by - ay * bx).abs();
    let q = 4.0 * (3.0_f64).sqrt() * area / s;
    q.clamp(0.0, 1.0)
}

fn tet_mean_ratio(v: &[Point]) -> f64 {
    // Liu-Joe η-quality: q = 12·(3V)^{2/3} / Σ edge². Equal to 1 for the
    // regular tet, vanishes as any vertex collapses onto the opposite face.
    let e: f64 = [
        edge_sq(&v[0], &v[1]),
        edge_sq(&v[0], &v[2]),
        edge_sq(&v[0], &v[3]),
        edge_sq(&v[1], &v[2]),
        edge_sq(&v[1], &v[3]),
        edge_sq(&v[2], &v[3]),
    ]
    .iter()
    .sum();
    if e <= 0.0 {
        return 0.0;
    }
    let vol = signed_tet_volume(&v[0], &v[1], &v[2], &v[3]).abs();
    // (3V)^(2/3) == cbrt((3V)^2); `cbrt` is cheaper and more accurate than
    // the general `powf` path.
    let t = 3.0 * vol;
    let num = 12.0 * (t * t).cbrt();
    (num / e).clamp(0.0, 1.0)
}

fn quad_corner_ratio(v: &[Point]) -> f64 {
    // Minimum corner Jacobian / (||e_prev|| · ||e_next||) — 1 for a square.
    let mut q_min = f64::INFINITY;
    for i in 0..4 {
        let prev = (i + 3) % 4;
        let next = (i + 1) % 4;
        let ux = v[next].x - v[i].x;
        let uy = v[next].y - v[i].y;
        let vx = v[prev].x - v[i].x;
        let vy = v[prev].y - v[i].y;
        let cross = ux * vy - uy * vx;
        let u_len = (ux * ux + uy * uy).sqrt();
        let v_len = (vx * vx + vy * vy).sqrt();
        if u_len * v_len <= 0.0 {
            return 0.0;
        }
        // For a square, cross = u_len·v_len; clamp ratio to [0,1].
        q_min = q_min.min(cross / (u_len * v_len));
    }
    q_min.clamp(0.0, 1.0)
}

fn hex_corner_ratio(v: &[Point]) -> f64 {
    // Minimum corner-Jacobian / (||e1||·||e2||·||e3||) — 1 for a unit cube.
    const NEIGH: [[usize; 3]; 8] = [
        [1, 3, 4],
        [2, 0, 5],
        [3, 1, 6],
        [0, 2, 7],
        [5, 7, 0],
        [6, 4, 1],
        [7, 5, 2],
        [4, 6, 3],
    ];
    let mut q_min = f64::INFINITY;
    for (i, neigh) in NEIGH.iter().enumerate() {
        let det = signed_tet_volume(&v[i], &v[neigh[0]], &v[neigh[1]], &v[neigh[2]]) * 6.0;
        let e1 = (edge_sq(&v[i], &v[neigh[0]])).sqrt();
        let e2 = (edge_sq(&v[i], &v[neigh[1]])).sqrt();
        let e3 = (edge_sq(&v[i], &v[neigh[2]])).sqrt();
        let denom = e1 * e2 * e3;
        if denom <= 0.0 {
            return 0.0;
        }
        q_min = q_min.min(det / denom);
    }
    q_min.clamp(0.0, 1.0)
}

/// Edge-length aspect ratio normalised so 1.0 == regular element. Always in
/// `(0, 1]`. Returns 0.0 on a zero-measure element.
pub fn aspect_quality(mesh: &Mesh, elem: &Element) -> f64 {
    if !has_valid_vertices(mesh, elem) {
        return 0.0;
    }
    let edges = elem.edges();
    if edges.is_empty() {
        return 0.0;
    }
    let mut min_len = f64::INFINITY;
    let mut max_len: f64 = 0.0;
    for (a, b) in edges {
        let len = mesh.nodes[a].distance(&mesh.nodes[b]);
        min_len = min_len.min(len);
        max_len = max_len.max(len);
    }
    if max_len <= 0.0 {
        0.0
    } else {
        (min_len / max_len).clamp(0.0, 1.0)
    }
}

/// Smallest dihedral angle (radians) between any two faces of a 3D element.
/// Returns 0 for 2D elements.
pub fn min_dihedral_angle(mesh: &Mesh, elem: &Element) -> f64 {
    dihedral_extremes(mesh, elem).0
}

/// Largest dihedral angle (radians) between any two faces of a 3D element.
pub fn max_dihedral_angle(mesh: &Mesh, elem: &Element) -> f64 {
    dihedral_extremes(mesh, elem).1
}

fn dihedral_extremes(mesh: &Mesh, elem: &Element) -> (f64, f64) {
    if elem.element_type.dimension() != 3 || !has_valid_vertices(mesh, elem) {
        return (0.0, 0.0);
    }
    let faces = elem.faces();
    let mut min_a = f64::INFINITY;
    let mut max_a: f64 = 0.0;

    for i in 0..faces.len() {
        for j in (i + 1)..faces.len() {
            let shared: Vec<usize> = faces[i]
                .iter()
                .filter(|x| faces[j].contains(x))
                .copied()
                .collect();
            if shared.len() < 2 {
                continue;
            }
            // Treat any two shared vertices as the edge direction; for our
            // element types adjacent faces share exactly one edge.
            let e0 = mesh.nodes[shared[0]];
            let e1 = mesh.nodes[shared[1]];
            let ex = e1.x - e0.x;
            let ey = e1.y - e0.y;
            let ez = e1.z - e0.z;
            let e_len = (ex * ex + ey * ey + ez * ez).sqrt();
            if e_len < 1e-15 {
                continue;
            }
            let edir = [ex / e_len, ey / e_len, ez / e_len];
            let mid = Point::new_3d(
                0.5 * (e0.x + e1.x),
                0.5 * (e0.y + e1.y),
                0.5 * (e0.z + e1.z),
            );

            let in_plane = |face: &[usize]| -> Option<[f64; 3]> {
                let mut cx = 0.0;
                let mut cy = 0.0;
                let mut cz = 0.0;
                let mut n = 0;
                for &idx in face {
                    if !shared.contains(&idx) {
                        let p = mesh.nodes[idx];
                        cx += p.x;
                        cy += p.y;
                        cz += p.z;
                        n += 1;
                    }
                }
                if n == 0 {
                    return None;
                }
                let nf = n as f64;
                let vx = cx / nf - mid.x;
                let vy = cy / nf - mid.y;
                let vz = cz / nf - mid.z;
                let dp = vx * edir[0] + vy * edir[1] + vz * edir[2];
                let wx = vx - dp * edir[0];
                let wy = vy - dp * edir[1];
                let wz = vz - dp * edir[2];
                let wlen = (wx * wx + wy * wy + wz * wz).sqrt();
                if wlen < 1e-15 {
                    None
                } else {
                    Some([wx / wlen, wy / wlen, wz / wlen])
                }
            };

            let wi = match in_plane(&faces[i]) {
                Some(v) => v,
                None => continue,
            };
            let wj = match in_plane(&faces[j]) {
                Some(v) => v,
                None => continue,
            };
            let dot = (wi[0] * wj[0] + wi[1] * wj[1] + wi[2] * wj[2]).clamp(-1.0, 1.0);
            let dihedral = dot.acos();
            min_a = min_a.min(dihedral);
            max_a = max_a.max(dihedral);
        }
    }
    if min_a.is_infinite() {
        (0.0, 0.0)
    } else {
        (min_a, max_a)
    }
}

fn has_valid_vertices(mesh: &Mesh, elem: &Element) -> bool {
    elem.nodes.len() >= elem.element_type.num_vertices()
        && elem.vertices().iter().all(|&node| {
            mesh.nodes.get(node).is_some_and(|point| {
                point.x.is_finite() && point.y.is_finite() && point.z.is_finite()
            })
        })
}

fn edge_sq(a: &Point, b: &Point) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dx * dx + dy * dy + dz * dz
}

fn signed_tet_volume(a: &Point, b: &Point, c: &Point, d: &Point) -> f64 {
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

/// Bucketed counts for a scalar over a mesh. `min` / `max` define the bin
/// range; values outside are clamped to the edge bins.
#[derive(Debug, Clone, Serialize)]
pub struct Histogram {
    pub min: f64,
    pub max: f64,
    pub bins: Vec<usize>,
}

impl Histogram {
    fn new(min: f64, max: f64, n_bins: usize) -> Self {
        Self {
            min,
            max,
            bins: vec![0; n_bins.max(1)],
        }
    }

    fn record(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        let span = self.max - self.min;
        let n = self.bins.len() as f64;
        let bin = if span > 0.0 {
            (((x - self.min) / span) * n).floor()
        } else {
            0.0
        };
        let idx = (bin.clamp(0.0, n - 1.0)) as usize;
        self.bins[idx] += 1;
    }
}

/// Edge length histogram over all element edges of the mesh.
pub fn edge_length_histogram(mesh: &Mesh, n_bins: usize) -> Histogram {
    let mut min_len = f64::INFINITY;
    let mut max_len: f64 = 0.0;
    let mut lens = Vec::with_capacity(mesh.num_elements() * 6);
    for elem in &mesh.elements {
        if !has_valid_vertices(mesh, elem) {
            continue;
        }
        for (a, b) in elem.edges() {
            let len = mesh.nodes[a].distance(&mesh.nodes[b]);
            min_len = min_len.min(len);
            max_len = max_len.max(len);
            lens.push(len);
        }
    }
    if lens.is_empty() {
        min_len = 0.0;
    }
    let mut hist = Histogram::new(min_len, max_len, n_bins);
    for &l in &lens {
        hist.record(l);
    }
    hist
}

/// Histogram of [`Mesh::element_measure`] (area in 2D, volume in 3D).
pub fn element_measure_histogram(mesh: &Mesh, n_bins: usize) -> Histogram {
    let mut measures = Vec::with_capacity(mesh.num_elements());
    let mut lo = f64::INFINITY;
    let mut hi: f64 = 0.0;
    for i in 0..mesh.num_elements() {
        let m = if has_valid_vertices(mesh, &mesh.elements[i]) {
            mesh.element_measure(i)
        } else {
            0.0
        };
        lo = lo.min(m);
        hi = hi.max(m);
        measures.push(m);
    }
    if measures.is_empty() {
        lo = 0.0;
    }
    let mut hist = Histogram::new(lo, hi, n_bins);
    for &m in &measures {
        hist.record(m);
    }
    hist
}

/// Aggregate report returned by [`mesh_quality_report`].
#[derive(Debug, Clone, Serialize)]
pub struct MeshQualityReport {
    /// Number of elements whose vertex references or coordinates are invalid.
    ///
    /// Invalid elements are represented with zero quality so aggregate
    /// thresholds fail closed; this counter keeps that failure actionable in
    /// machine-readable QA artifacts instead of looking like ordinary poor
    /// quality.
    pub invalid_elements: usize,
    pub min_mean_ratio: f64,
    pub mean_mean_ratio: f64,
    pub min_aspect_quality: f64,
    pub min_dihedral: f64,
    pub max_dihedral: f64,
    pub edge_lengths: Histogram,
    pub element_measures: Histogram,
}

/// Hard acceptance criteria for a generated or transformed mesh.
///
/// The defaults are intentionally conservative about malformed elements while
/// leaving numerical quality thresholds to the consuming physics/order. A
/// caller can therefore use the same report for diagnostics and enforce the
/// regime-specific gate before handing a mesh to FEM/BEM assembly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct QualityThresholds {
    pub min_mean_ratio: f64,
    pub min_aspect_quality: f64,
    pub min_dihedral: f64,
    pub max_dihedral: f64,
    pub reject_invalid_elements: bool,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_mean_ratio: 0.0,
            min_aspect_quality: 0.0,
            min_dihedral: 0.0,
            max_dihedral: std::f64::consts::PI,
            reject_invalid_elements: true,
        }
    }
}

/// Why a mesh failed a [`QualityThresholds`] gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QualityFailure {
    InvalidThresholds,
    InvalidElements {
        count: usize,
    },
    BelowThreshold {
        metric: &'static str,
        actual: f64,
        minimum: f64,
    },
    AboveThreshold {
        metric: &'static str,
        actual: f64,
        maximum: f64,
    },
}

impl fmt::Display for QualityFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidThresholds => {
                f.write_str("mesh quality thresholds are not finite or ordered")
            }
            Self::InvalidElements { count } => write!(f, "mesh contains {count} invalid elements"),
            Self::BelowThreshold {
                metric,
                actual,
                minimum,
            } => {
                write!(f, "mesh {metric} {actual:.6} is below minimum {minimum:.6}")
            }
            Self::AboveThreshold {
                metric,
                actual,
                maximum,
            } => {
                write!(f, "mesh {metric} {actual:.6} is above maximum {maximum:.6}")
            }
        }
    }
}

impl std::error::Error for QualityFailure {}

impl MeshQualityReport {
    /// Enforce hard quality limits before a solver consumes this mesh.
    pub fn validate(&self, thresholds: QualityThresholds) -> Result<(), QualityFailure> {
        if !thresholds.min_mean_ratio.is_finite()
            || !thresholds.min_aspect_quality.is_finite()
            || !thresholds.min_dihedral.is_finite()
            || !thresholds.max_dihedral.is_finite()
            || !(0.0..=1.0).contains(&thresholds.min_mean_ratio)
            || !(0.0..=1.0).contains(&thresholds.min_aspect_quality)
            || thresholds.min_dihedral < 0.0
            || thresholds.max_dihedral < thresholds.min_dihedral
            || thresholds.max_dihedral > std::f64::consts::PI
        {
            return Err(QualityFailure::InvalidThresholds);
        }
        if thresholds.reject_invalid_elements && self.invalid_elements > 0 {
            return Err(QualityFailure::InvalidElements {
                count: self.invalid_elements,
            });
        }
        if self.min_mean_ratio < thresholds.min_mean_ratio {
            return Err(QualityFailure::BelowThreshold {
                metric: "mean ratio",
                actual: self.min_mean_ratio,
                minimum: thresholds.min_mean_ratio,
            });
        }
        if self.min_aspect_quality < thresholds.min_aspect_quality {
            return Err(QualityFailure::BelowThreshold {
                metric: "aspect quality",
                actual: self.min_aspect_quality,
                minimum: thresholds.min_aspect_quality,
            });
        }
        if self.min_dihedral < thresholds.min_dihedral {
            return Err(QualityFailure::BelowThreshold {
                metric: "minimum dihedral",
                actual: self.min_dihedral,
                minimum: thresholds.min_dihedral,
            });
        }
        if self.max_dihedral > thresholds.max_dihedral {
            return Err(QualityFailure::AboveThreshold {
                metric: "maximum dihedral",
                actual: self.max_dihedral,
                maximum: thresholds.max_dihedral,
            });
        }
        Ok(())
    }
}

/// Summarise mesh quality across all elements.
pub fn mesh_quality_report(mesh: &Mesh, n_bins: usize) -> MeshQualityReport {
    let qs = per_element_quality(mesh);
    let invalid_elements = mesh
        .elements
        .iter()
        .filter(|element| !has_valid_vertices(mesh, element))
        .count();
    let mut min_mr = f64::INFINITY;
    let mut sum_mr = 0.0;
    let mut min_aq = f64::INFINITY;
    let mut min_d = f64::INFINITY;
    let mut max_d: f64 = 0.0;
    for q in &qs {
        min_mr = min_mr.min(q.mean_ratio);
        sum_mr += q.mean_ratio;
        min_aq = min_aq.min(q.aspect_quality);
        if q.min_dihedral > 0.0 {
            min_d = min_d.min(q.min_dihedral);
        }
        max_d = max_d.max(q.max_dihedral);
    }
    let n = qs.len().max(1) as f64;
    MeshQualityReport {
        invalid_elements,
        min_mean_ratio: if qs.is_empty() { 0.0 } else { min_mr },
        mean_mean_ratio: sum_mr / n,
        min_aspect_quality: if qs.is_empty() { 0.0 } else { min_aq },
        min_dihedral: if min_d.is_infinite() { 0.0 } else { min_d },
        max_dihedral: max_d,
        edge_lengths: edge_length_histogram(mesh, n_bins),
        element_measures: element_measure_histogram(mesh, n_bins),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::generators::{unit_cube_hexahedra, unit_cube_tetrahedra, unit_square_quads};
    use crate::mesh::types::Point;

    fn regular_tet() -> Mesh {
        let mut mesh = Mesh::new(3);
        // Regular tetrahedron with edge length √2 inscribed in the unit cube.
        mesh.add_node(Point::new_3d(0.0, 0.0, 0.0));
        mesh.add_node(Point::new_3d(1.0, 1.0, 0.0));
        mesh.add_node(Point::new_3d(1.0, 0.0, 1.0));
        mesh.add_node(Point::new_3d(0.0, 1.0, 1.0));
        mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
        mesh
    }

    #[test]
    fn regular_tet_scores_one() {
        let mesh = regular_tet();
        let q = mean_ratio(&mesh, &mesh.elements[0]);
        assert!((q - 1.0).abs() < 1e-6, "expected q≈1, got {q}");
    }

    #[test]
    fn non_finite_nodes_fail_closed_in_quality_metrics() {
        let mut mesh = regular_tet();
        mesh.nodes[2].x = f64::NAN;
        let element = &mesh.elements[0];
        assert_eq!(mean_ratio(&mesh, element), 0.0);
        assert_eq!(aspect_quality(&mesh, element), 0.0);
        assert_eq!(min_dihedral_angle(&mesh, element), 0.0);
        assert_eq!(max_dihedral_angle(&mesh, element), 0.0);
        let report = mesh_quality_report(&mesh, 4);
        assert_eq!(report.invalid_elements, 1);
        assert_eq!(report.min_mean_ratio, 0.0);
        assert_eq!(report.min_aspect_quality, 0.0);
    }

    #[test]
    fn quality_gate_rejects_invalid_elements_and_bad_thresholds() {
        let mut mesh = regular_tet();
        mesh.nodes[0].x = f64::INFINITY;
        let report = mesh_quality_report(&mesh, 4);
        assert!(matches!(
            report.validate(QualityThresholds::default()),
            Err(QualityFailure::InvalidElements { count: 1 })
        ));

        let good = mesh_quality_report(&regular_tet(), 4);
        let thresholds = QualityThresholds {
            min_mean_ratio: 1.1,
            ..QualityThresholds::default()
        };
        assert!(matches!(
            good.validate(thresholds),
            Err(QualityFailure::InvalidThresholds)
        ));
        let thresholds = QualityThresholds {
            min_dihedral: 2.0,
            ..QualityThresholds::default()
        };
        assert!(matches!(
            good.validate(thresholds),
            Err(QualityFailure::BelowThreshold {
                metric: "minimum dihedral",
                ..
            })
        ));
    }

    #[test]
    fn unit_square_quads_score_one() {
        let mesh = unit_square_quads(2);
        let qs = per_element_quality(&mesh);
        for q in qs {
            assert!(
                q.mean_ratio > 0.99,
                "quad mean ratio fell to {}",
                q.mean_ratio
            );
        }
    }

    #[test]
    fn cube_hex_min_dihedral_is_right_angle() {
        let mesh = unit_cube_hexahedra(1);
        let q = min_dihedral_angle(&mesh, &mesh.elements[0]);
        assert!(
            (q - std::f64::consts::FRAC_PI_2).abs() < 1e-6,
            "min dihedral should be π/2, got {q}"
        );
    }

    #[test]
    fn report_runs_on_cube_tet_mesh() {
        let mesh = unit_cube_tetrahedra(2);
        let report = mesh_quality_report(&mesh, 8);
        assert!(report.min_mean_ratio > 0.0);
        assert!(report.mean_mean_ratio > 0.0);
        assert_eq!(report.edge_lengths.bins.len(), 8);
    }

    #[test]
    fn malformed_elements_fail_closed_in_quality_reports() {
        let mut mesh = Mesh::new(3);
        mesh.add_node(Point::new_3d(0.0, 0.0, 0.0));
        mesh.add_element(ElementType::Tetrahedron, vec![0, 99]);

        let element = &mesh.elements[0];
        assert_eq!(mean_ratio(&mesh, element), 0.0);
        assert_eq!(aspect_quality(&mesh, element), 0.0);
        assert_eq!(min_dihedral_angle(&mesh, element), 0.0);
        assert_eq!(max_dihedral_angle(&mesh, element), 0.0);
        let report = mesh_quality_report(&mesh, 4);
        assert_eq!(report.min_mean_ratio, 0.0);
        assert_eq!(report.min_aspect_quality, 0.0);
        assert_eq!(report.element_measures.bins.iter().sum::<usize>(), 1);
    }
}
