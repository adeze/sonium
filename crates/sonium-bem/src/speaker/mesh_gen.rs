//! Surface mesh generation for the loudspeaker cabinet.
//!
//! Generates a watertight BEM surface mesh from the parametric model,
//! consisting of box faces, front baffle with holes, driver aperture
//! disks, waveguide surface, and internal separator.

use std::f64::consts::PI;

use sonium_mesher::mesh::triangulate_points_2d;
use sonium_xem::io::vtu::{CellKind, MeshView, VtuWriter};

use super::model::LoudspeakerModel;
use super::spline::CubicSpline;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Generated surface mesh ready for BEM.
pub struct SpeakerMesh {
    /// Node coordinates (N × 3).
    pub nodes: Vec<[f64; 3]>,
    /// Triangle connectivity (each triangle = 3 node indices).
    pub triangles: Vec<[usize; 3]>,
    /// Per-element region tag for boundary condition assignment.
    pub region: Vec<MeshRegion>,
}

/// Region identifier for boundary condition assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshRegion {
    /// Cabinet wall (MDF, rigid with admittance).
    CabinetWall,
    /// Woofer piston (velocity BC).
    WooferAperture,
    /// Tweeter piston (velocity BC).
    TweeterAperture,
    /// Waveguide surface (hard plastic, rigid with admittance).
    Waveguide,
    /// Internal separator plate (rigid).
    Separator,
}

// ---------------------------------------------------------------------------
// Mesh builder
// ---------------------------------------------------------------------------

struct MeshBuilder {
    nodes: Vec<[f64; 3]>,
    triangles: Vec<[usize; 3]>,
    regions: Vec<MeshRegion>,
}

impl MeshBuilder {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            triangles: Vec::new(),
            regions: Vec::new(),
        }
    }

    fn add_node(&mut self, p: [f64; 3]) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(p);
        idx
    }

    /// Add a triangle. Caller must ensure winding gives outward normal.
    fn add_tri(&mut self, a: usize, b: usize, c: usize, region: MeshRegion) {
        self.triangles.push([a, b, c]);
        self.regions.push(region);
    }

    fn finish(self) -> SpeakerMesh {
        SpeakerMesh {
            nodes: self.nodes,
            triangles: self.triangles,
            region: self.regions,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate the full surface mesh from a loudspeaker model.
///
/// The mesh is a watertight closed surface suitable for exterior BEM.
/// All normals point outward from the cabinet body.
pub fn generate_loudspeaker_mesh(model: &LoudspeakerModel) -> SpeakerMesh {
    let h_max = model.simulation.max_element_size();
    let mut b = MeshBuilder::new();

    let w2 = model.cabinet.width / 2.0;
    let h2 = model.cabinet.height / 2.0;
    let d2 = model.cabinet.depth / 2.0;

    // Woofer and tweeter centers on front baffle (z = +d2)
    let woofer_center = [0.0, model.woofer.y_offset];
    let tweeter_center = [0.0, model.tweeter.y_offset];

    // Waveguide rim radius at baffle surface (t=0)
    let h_spline = CubicSpline::from_control_points(&model.waveguide.horizontal_spline);
    let v_spline = CubicSpline::from_control_points(&model.waveguide.vertical_spline);
    let wg_rim_h = h_spline.evaluate(0.0);
    let wg_rim_v = v_spline.evaluate(0.0);

    // Number of points around circles, based on element size
    let n_woofer_ring = ring_count(model.woofer.radius, h_max);
    let n_wg_ring = ring_count(wg_rim_h.max(wg_rim_v), h_max);

    // --- 1. Five box faces (back, top, bottom, left, right) ---
    // Back face: z = -d2, normal = (0,0,-1)
    add_rect_face(
        &mut b,
        [-w2, -h2, -d2],
        [w2, -h2, -d2],
        [w2, h2, -d2],
        [-w2, h2, -d2],
        h_max,
        MeshRegion::CabinetWall,
    );
    // Top face: y = +h2, normal = (0,+1,0)
    add_rect_face(
        &mut b,
        [-w2, h2, -d2],
        [w2, h2, -d2],
        [w2, h2, d2],
        [-w2, h2, d2],
        h_max,
        MeshRegion::CabinetWall,
    );
    // Bottom face: y = -h2, normal = (0,-1,0)
    add_rect_face(
        &mut b,
        [-w2, -h2, d2],
        [w2, -h2, d2],
        [w2, -h2, -d2],
        [-w2, -h2, -d2],
        h_max,
        MeshRegion::CabinetWall,
    );
    // Left face: x = -w2, normal = (-1,0,0)
    add_rect_face(
        &mut b,
        [-w2, -h2, d2],
        [-w2, -h2, -d2],
        [-w2, h2, -d2],
        [-w2, h2, d2],
        h_max,
        MeshRegion::CabinetWall,
    );
    // Right face: x = +w2, normal = (+1,0,0)
    add_rect_face(
        &mut b,
        [w2, -h2, -d2],
        [w2, -h2, d2],
        [w2, h2, d2],
        [w2, h2, -d2],
        h_max,
        MeshRegion::CabinetWall,
    );

    // --- 2. Front baffle with holes (z = +d2) ---
    // Generate boundary rings for the two holes
    let woofer_ring_2d: Vec<(f64, f64)> = circle_points_2d(
        woofer_center[0],
        woofer_center[1],
        model.woofer.radius,
        n_woofer_ring,
    );

    // For the waveguide, the rim is an ellipse
    let wg_ring_2d: Vec<(f64, f64)> = ellipse_points_2d(
        tweeter_center[0],
        tweeter_center[1],
        wg_rim_h,
        wg_rim_v,
        n_wg_ring,
    );

    // Register ring nodes in the builder (lifted to z = d2)
    let woofer_ring_start = b.nodes.len();
    for &(x, y) in &woofer_ring_2d {
        b.add_node([x, y, d2]);
    }
    let wg_ring_start = b.nodes.len();
    for &(x, y) in &wg_ring_2d {
        b.add_node([x, y, d2]);
    }

    // Delaunay triangulation of the baffle
    add_baffle_with_holes(
        &mut b,
        [-w2, -h2],
        [w2, h2],
        &woofer_ring_2d,
        woofer_ring_start,
        woofer_center,
        model.woofer.radius,
        &wg_ring_2d,
        wg_ring_start,
        tweeter_center,
        wg_rim_h,
        wg_rim_v,
        d2,
        h_max,
    );

    // --- 3. Woofer aperture disk ---
    add_disk(
        &mut b,
        &woofer_ring_2d,
        woofer_ring_start,
        woofer_center,
        d2,
        h_max,
        MeshRegion::WooferAperture,
        true, // normal = +z (outward = forward)
    );

    // --- 4. Waveguide surface + tweeter aperture ---
    add_waveguide(
        &mut b,
        &model.waveguide,
        &wg_ring_2d,
        wg_ring_start,
        tweeter_center,
        model.tweeter.radius,
        d2,
        h_max,
    );

    // --- 5. Internal separator ---
    let sep_y = model.separator.y_position;
    add_rect_face(
        &mut b,
        [-w2, sep_y, -d2],
        [w2, sep_y, -d2],
        [w2, sep_y, d2],
        [-w2, sep_y, d2],
        h_max,
        MeshRegion::Separator,
    );

    b.finish()
}

// ---------------------------------------------------------------------------
// Rectangular face
// ---------------------------------------------------------------------------

/// Add a structured triangle mesh for a rectangular face.
///
/// Corners must be given in CCW order when viewed from the outward side.
fn add_rect_face(
    b: &mut MeshBuilder,
    p0: [f64; 3],
    p1: [f64; 3],
    p2: [f64; 3],
    p3: [f64; 3],
    h_max: f64,
    region: MeshRegion,
) {
    // Edge lengths
    let edge01 = dist(p0, p1);
    let edge12 = dist(p1, p2);

    let nx = (edge01 / h_max).ceil().max(1.0) as usize;
    let ny = (edge12 / h_max).ceil().max(1.0) as usize;

    let base = b.nodes.len();

    // Generate grid nodes
    for j in 0..=ny {
        let v = j as f64 / ny as f64;
        for i in 0..=nx {
            let u = i as f64 / nx as f64;
            // Bilinear interpolation
            let x = (1.0 - u) * (1.0 - v) * p0[0]
                + u * (1.0 - v) * p1[0]
                + u * v * p2[0]
                + (1.0 - u) * v * p3[0];
            let y = (1.0 - u) * (1.0 - v) * p0[1]
                + u * (1.0 - v) * p1[1]
                + u * v * p2[1]
                + (1.0 - u) * v * p3[1];
            let z = (1.0 - u) * (1.0 - v) * p0[2]
                + u * (1.0 - v) * p1[2]
                + u * v * p2[2]
                + (1.0 - u) * v * p3[2];
            b.add_node([x, y, z]);
        }
    }

    // Generate triangles (2 per grid cell)
    for j in 0..ny {
        for i in 0..nx {
            let bl = base + j * (nx + 1) + i;
            let br = bl + 1;
            let tl = bl + (nx + 1);
            let tr = tl + 1;
            b.add_tri(bl, br, tr, region);
            b.add_tri(bl, tr, tl, region);
        }
    }
}

// ---------------------------------------------------------------------------
// Front baffle with holes (Delaunay)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn add_baffle_with_holes(
    b: &mut MeshBuilder,
    rect_min: [f64; 2],
    rect_max: [f64; 2],
    woofer_ring: &[(f64, f64)],
    woofer_ring_start: usize,
    woofer_center: [f64; 2],
    woofer_radius: f64,
    wg_ring: &[(f64, f64)],
    wg_ring_start: usize,
    wg_center: [f64; 2],
    wg_rim_h: f64,
    wg_rim_v: f64,
    z: f64,
    h_max: f64,
) {
    // Collect all 2D points: rectangle boundary + interior grid (excluding holes) + ring nodes
    let mut pts_2d: Vec<(f64, f64)> = Vec::new();
    let mut node_map: Vec<Option<usize>> = Vec::new(); // maps Delaunay idx → global node idx

    // Rectangle boundary points
    let nx = ((rect_max[0] - rect_min[0]) / h_max).ceil().max(2.0) as usize;
    let ny = ((rect_max[1] - rect_min[1]) / h_max).ceil().max(2.0) as usize;

    // Bottom edge
    for i in 0..nx {
        let x = rect_min[0] + (rect_max[0] - rect_min[0]) * i as f64 / nx as f64;
        pts_2d.push((x, rect_min[1]));
        node_map.push(None);
    }
    // Right edge
    for j in 0..ny {
        let y = rect_min[1] + (rect_max[1] - rect_min[1]) * j as f64 / ny as f64;
        pts_2d.push((rect_max[0], y));
        node_map.push(None);
    }
    // Top edge
    for i in (1..=nx).rev() {
        let x = rect_min[0] + (rect_max[0] - rect_min[0]) * i as f64 / nx as f64;
        pts_2d.push((x, rect_max[1]));
        node_map.push(None);
    }
    // Left edge
    for j in (1..=ny).rev() {
        let y = rect_min[1] + (rect_max[1] - rect_min[1]) * j as f64 / ny as f64;
        pts_2d.push((rect_min[0], y));
        node_map.push(None);
    }

    // Interior grid points (skip those inside holes)
    let step = h_max * 0.9;
    let mut y = rect_min[1] + step;
    while y < rect_max[1] - step * 0.5 {
        let mut x = rect_min[0] + step;
        while x < rect_max[0] - step * 0.5 {
            if !inside_circle(x, y, woofer_center, woofer_radius + h_max * 0.3)
                && !inside_ellipse(x, y, wg_center, wg_rim_h + h_max * 0.3, wg_rim_v + h_max * 0.3)
            {
                pts_2d.push((x, y));
                node_map.push(None);
            }
            x += step;
        }
        y += step;
    }

    // Woofer ring points (already registered as nodes)
    for (i, &pt) in woofer_ring.iter().enumerate() {
        pts_2d.push(pt);
        node_map.push(Some(woofer_ring_start + i));
    }

    // Waveguide ring points (already registered as nodes)
    for (i, &pt) in wg_ring.iter().enumerate() {
        pts_2d.push(pt);
        node_map.push(Some(wg_ring_start + i));
    }

    // Ensure non-ring nodes have global node indices
    // Pre-resolve all Delaunay points to global node indices
    let mut global_indices = vec![0usize; pts_2d.len()];
    for (i, map_entry) in node_map.iter().enumerate() {
        if let Some(global) = *map_entry {
            global_indices[i] = global;
        } else {
            let (x, y) = pts_2d[i];
            global_indices[i] = b.add_node([x, y, z]);
        }
    }

    // Add triangles, rejecting those inside holes
    for [i0, i1, i2] in triangulate_points_2d(&pts_2d) {
        // Compute centroid
        let cx = (pts_2d[i0].0 + pts_2d[i1].0 + pts_2d[i2].0) / 3.0;
        let cy = (pts_2d[i0].1 + pts_2d[i1].1 + pts_2d[i2].1) / 3.0;

        // Reject triangles inside holes
        if inside_circle(cx, cy, woofer_center, woofer_radius)
            || inside_ellipse(cx, cy, wg_center, wg_rim_h, wg_rim_v)
        {
            continue;
        }

        // Front baffle normal is +z (outward), so CCW from front = CCW in xy plane
        let g0 = global_indices[i0];
        let g1 = global_indices[i1];
        let g2 = global_indices[i2];

        // Check winding: cross product z-component should be positive for +z normal
        let ax = pts_2d[i1].0 - pts_2d[i0].0;
        let ay = pts_2d[i1].1 - pts_2d[i0].1;
        let bx = pts_2d[i2].0 - pts_2d[i0].0;
        let by = pts_2d[i2].1 - pts_2d[i0].1;
        let cross_z = ax * by - ay * bx;

        if cross_z > 0.0 {
            b.add_tri(g0, g1, g2, MeshRegion::CabinetWall);
        } else {
            b.add_tri(g0, g2, g1, MeshRegion::CabinetWall);
        }
    }
}

// ---------------------------------------------------------------------------
// Disk (driver aperture)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn add_disk(
    b: &mut MeshBuilder,
    ring: &[(f64, f64)],
    ring_start: usize,
    center: [f64; 2],
    z: f64,
    h_max: f64,
    region: MeshRegion,
    normal_positive_z: bool,
) {
    let n = ring.len();
    let radius = ring
        .iter()
        .map(|&(x, y)| ((x - center[0]).powi(2) + (y - center[1]).powi(2)).sqrt())
        .fold(0.0, f64::max);

    // For small disks (tweeter), a simple fan triangulation suffices
    if radius / h_max < 3.0 {
        let center_node = b.add_node([center[0], center[1], z]);
        for i in 0..n {
            let j = (i + 1) % n;
            if normal_positive_z {
                b.add_tri(center_node, ring_start + i, ring_start + j, region);
            } else {
                b.add_tri(center_node, ring_start + j, ring_start + i, region);
            }
        }
        return;
    }

    // For larger disks, add concentric rings and triangulate
    let n_rings = (radius / h_max).ceil().max(1.0) as usize;
    let mut prev_ring_start = ring_start;
    let mut prev_ring_count = n;

    for ring_idx in 1..n_rings {
        let t = 1.0 - ring_idx as f64 / n_rings as f64; // goes from ~1 to ~0
        let r = radius * t;
        let n_inner = (n as f64 * t).ceil().max(3.0) as usize;

        let inner_start = b.nodes.len();
        for k in 0..n_inner {
            let angle = 2.0 * PI * k as f64 / n_inner as f64;
            let x = center[0] + r * angle.cos();
            let y = center[1] + r * angle.sin();
            b.add_node([x, y, z]);
        }

        // Stitch outer ring to inner ring
        stitch_rings(
            b,
            prev_ring_start,
            prev_ring_count,
            inner_start,
            n_inner,
            region,
            normal_positive_z,
        );

        prev_ring_start = inner_start;
        prev_ring_count = n_inner;
    }

    // Final center fan
    let center_node = b.add_node([center[0], center[1], z]);
    for i in 0..prev_ring_count {
        let j = (i + 1) % prev_ring_count;
        if normal_positive_z {
            b.add_tri(center_node, prev_ring_start + i, prev_ring_start + j, region);
        } else {
            b.add_tri(center_node, prev_ring_start + j, prev_ring_start + i, region);
        }
    }
}

// ---------------------------------------------------------------------------
// Waveguide surface
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn add_waveguide(
    b: &mut MeshBuilder,
    wg: &super::model::WaveguideConfig,
    rim_ring: &[(f64, f64)],
    rim_ring_start: usize,
    tweeter_center: [f64; 2],
    _tweeter_radius: f64,
    baffle_z: f64,
    h_max: f64,
) {
    let h_spline = CubicSpline::from_control_points(&wg.horizontal_spline);
    let v_spline = CubicSpline::from_control_points(&wg.vertical_spline);

    let n_u = rim_ring.len(); // azimuthal resolution matches rim
    let n_v = (wg.depth / h_max).ceil().max(2.0) as usize;

    // Build rings from baffle (v=0) to tweeter (v=1)
    // v=0 ring is already rim_ring_start..rim_ring_start+n_u
    let mut prev_start = rim_ring_start;
    let mut prev_count = n_u;

    for v_idx in 1..=n_v {
        let t = v_idx as f64 / n_v as f64; // 0→1
        let z = baffle_z - wg.depth * t; // goes from baffle inward (negative z)
        let rh = h_spline.evaluate(t);
        let rv = v_spline.evaluate(t);

        let is_last = v_idx == n_v;
        let this_count = if is_last {
            // Tweeter aperture ring — use same count
            n_u
        } else {
            n_u
        };

        let this_start = b.nodes.len();
        for k in 0..this_count {
            let angle = 2.0 * PI * k as f64 / this_count as f64;
            let x = tweeter_center[0] + rh * angle.cos();
            let y = tweeter_center[1] + rv * angle.sin();
            b.add_node([x, y, z]);
        }

        // Stitch to previous ring.
        // Waveguide normals must point into the acoustic domain (outward from
        // the speaker body). The outer ring is closer to the baffle (+z),
        // inner rings go inward (-z). flip=true gives outward-facing normals.
        stitch_rings(
            b,
            prev_start,
            prev_count,
            this_start,
            this_count,
            MeshRegion::Waveguide,
            true,
        );

        prev_start = this_start;
        prev_count = this_count;
    }

    // Close with tweeter aperture disk at the deepest ring
    let tweeter_z = baffle_z - wg.depth;
    let tweeter_ring: Vec<(f64, f64)> = (0..prev_count)
        .map(|k| {
            let p = b.nodes[prev_start + k];
            (p[0], p[1])
        })
        .collect();

    // Tweeter disk — normal points +z (outward from speaker)
    add_disk(
        b,
        &tweeter_ring,
        prev_start,
        tweeter_center,
        tweeter_z,
        h_max,
        MeshRegion::TweeterAperture,
        true,
    );
}

// ---------------------------------------------------------------------------
// Ring stitching
// ---------------------------------------------------------------------------

/// Stitch two concentric rings with triangles.
///
/// `outer_start..outer_start+outer_count` and `inner_start..inner_start+inner_count`.
fn stitch_rings(
    b: &mut MeshBuilder,
    outer_start: usize,
    outer_count: usize,
    inner_start: usize,
    inner_count: usize,
    region: MeshRegion,
    flip: bool,
) {
    // Walk both rings with a two-pointer approach.
    // Total triangles = outer_count + inner_count (each advance emits one triangle).
    let total = outer_count + inner_count;
    let mut io = 0usize;
    let mut ii = 0usize;

    for _ in 0..total {
        let o0 = outer_start + (io % outer_count);
        let o1 = outer_start + ((io + 1) % outer_count);
        let i0 = inner_start + (ii % inner_count);
        let i1 = inner_start + ((ii + 1) % inner_count);

        // Decide whether to advance the outer or inner pointer.
        // Advance outer when it's behind (or tied) relative to inner.
        let advance_outer = if io >= outer_count {
            false
        } else if ii >= inner_count {
            true
        } else {
            (io as f64 / outer_count as f64) <= (ii as f64 / inner_count as f64)
        };

        if advance_outer {
            if flip {
                b.add_tri(o0, i0, o1, region);
            } else {
                b.add_tri(o0, o1, i0, region);
            }
            io += 1;
        } else {
            if flip {
                b.add_tri(i0, i1, outer_start + (io % outer_count), region);
            } else {
                b.add_tri(i0, outer_start + (io % outer_count), i1, region);
            }
            ii += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// VTU export
// ---------------------------------------------------------------------------

impl MeshView for SpeakerMesh {
    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    fn num_cells(&self) -> usize {
        self.triangles.len()
    }

    fn nodes(&self) -> Box<dyn Iterator<Item = [f64; 3]> + '_> {
        Box::new(self.nodes.iter().copied())
    }

    fn cells(&self) -> Box<dyn Iterator<Item = (CellKind, Vec<u64>)> + '_> {
        Box::new(self.triangles.iter().map(|tri| {
            (
                CellKind::Triangle,
                vec![tri[0] as u64, tri[1] as u64, tri[2] as u64],
            )
        }))
    }
}

/// Export a `SpeakerMesh` (with per-cell region tag) to a `.vtu` for ParaView.
pub fn export_vtu(mesh: &SpeakerMesh, path: &str) {
    let regions: Vec<f64> = mesh.region.iter().map(|r| *r as i32 as f64).collect();
    let mut w = VtuWriter::new(mesh);
    w.add_cell_scalar("region", &regions);
    w.write(path).expect("Cannot write VTU file");
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn ring_count(radius: f64, h_max: f64) -> usize {
    let circumference = 2.0 * PI * radius;
    (circumference / h_max).ceil().max(8.0) as usize
}

fn circle_points_2d(cx: f64, cy: f64, r: f64, n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let angle = 2.0 * PI * i as f64 / n as f64;
            (cx + r * angle.cos(), cy + r * angle.sin())
        })
        .collect()
}

fn ellipse_points_2d(cx: f64, cy: f64, rx: f64, ry: f64, n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let angle = 2.0 * PI * i as f64 / n as f64;
            (cx + rx * angle.cos(), cy + ry * angle.sin())
        })
        .collect()
}

fn inside_circle(x: f64, y: f64, center: [f64; 2], radius: f64) -> bool {
    let dx = x - center[0];
    let dy = y - center[1];
    dx * dx + dy * dy < radius * radius
}

fn inside_ellipse(x: f64, y: f64, center: [f64; 2], rx: f64, ry: f64) -> bool {
    let dx = (x - center[0]) / rx;
    let dy = (y - center[1]) / ry;
    dx * dx + dy * dy < 1.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::model::*;

    fn sample_model() -> LoudspeakerModel {
        LoudspeakerModel {
            cabinet: CabinetDimensions {
                height: 0.40,
                width: 0.22,
                depth: 0.28,
            },
            woofer: DriverConfig {
                radius: 0.065,
                y_offset: -0.08,
                piston_velocity: 0.001,
                transfer_function: None,
            },
            tweeter: DriverConfig {
                radius: 0.013,
                y_offset: 0.10,
                piston_velocity: 0.001,
                transfer_function: None,
            },
            separator: SeparatorConfig {
                y_position: 0.02,
                thickness: 0.018,
            },
            waveguide: WaveguideConfig {
                horizontal_spline: vec![[0.0, 0.05], [0.33, 0.04], [0.66, 0.025], [1.0, 0.013]],
                vertical_spline: vec![[0.0, 0.04], [0.33, 0.03], [0.66, 0.02], [1.0, 0.013]],
                depth: 0.025,
            },
            simulation: SimulationConfig {
                freq_start: 200.0,
                freq_end: 5000.0,   // lower for faster mesh
                freq_points: 20,
                speed_of_sound: 343.0,
                air_density: 1.21,
                max_mesh_frequency: Some(5000.0),
            },
        }
    }

    #[test]
    fn test_generate_mesh_basic() {
        let model = sample_model();
        let mesh = generate_loudspeaker_mesh(&model);

        assert!(!mesh.nodes.is_empty(), "mesh has no nodes");
        assert!(!mesh.triangles.is_empty(), "mesh has no triangles");
        assert_eq!(mesh.triangles.len(), mesh.region.len());

        // All indices valid
        for tri in &mesh.triangles {
            assert!(tri[0] < mesh.nodes.len());
            assert!(tri[1] < mesh.nodes.len());
            assert!(tri[2] < mesh.nodes.len());
        }
    }

    #[test]
    fn test_mesh_has_all_regions() {
        let model = sample_model();
        let mesh = generate_loudspeaker_mesh(&model);

        let has_wall = mesh.region.iter().any(|r| *r == MeshRegion::CabinetWall);
        let has_woofer = mesh.region.iter().any(|r| *r == MeshRegion::WooferAperture);
        let has_tweeter = mesh.region.iter().any(|r| *r == MeshRegion::TweeterAperture);
        let has_waveguide = mesh.region.iter().any(|r| *r == MeshRegion::Waveguide);
        let has_separator = mesh.region.iter().any(|r| *r == MeshRegion::Separator);

        assert!(has_wall, "missing CabinetWall region");
        assert!(has_woofer, "missing WooferAperture region");
        assert!(has_tweeter, "missing TweeterAperture region");
        assert!(has_waveguide, "missing Waveguide region");
        assert!(has_separator, "missing Separator region");
    }

    #[test]
    fn test_no_degenerate_triangles() {
        let model = sample_model();
        let mesh = generate_loudspeaker_mesh(&model);

        for (i, tri) in mesh.triangles.iter().enumerate() {
            assert_ne!(tri[0], tri[1], "degenerate triangle {i}");
            assert_ne!(tri[1], tri[2], "degenerate triangle {i}");
            assert_ne!(tri[0], tri[2], "degenerate triangle {i}");

            // Check area > 0
            let p0 = mesh.nodes[tri[0]];
            let p1 = mesh.nodes[tri[1]];
            let p2 = mesh.nodes[tri[2]];
            let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let cross = [
                v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0],
            ];
            let area = 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            assert!(area > 1e-15, "zero-area triangle {i}");
        }
    }

    #[test]
    fn test_export_vtk() {
        let model = sample_model();
        let mesh = generate_loudspeaker_mesh(&model);
        let path = format!("{}/speaker_test.vtk", std::env::temp_dir().display());
        export_vtu(&mesh, &path);
        assert!(std::path::Path::new(&path).exists());
        let _ = std::fs::remove_file(&path);
    }
}
