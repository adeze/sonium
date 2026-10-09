//! Surface mesh generation for the loudspeaker cabinet.
//!
//! Generates a watertight BEM surface mesh from the parametric model,
//! consisting of box faces, front baffle with holes, driver aperture
//! disks, waveguide surface, and internal separator.

use std::f64::consts::PI;
use std::io::Write as IoWrite;

use sonium_mesher::mesh::triangulate_points_2d;

use crate::model::LoudspeakerModel;
use crate::spline::CubicSpline;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Generated surface mesh ready for BEM.
#[derive(Clone)]
pub struct SpeakerMesh {
    /// Node coordinates (N × 3).
    pub nodes: Vec<[f64; 3]>,
    /// Triangle connectivity (each triangle = 3 node indices).
    pub triangles: Vec<[usize; 3]>,
    /// Per-element region tag for boundary condition assignment.
    pub region: Vec<MeshRegion>,
    /// Per-node normals for curved surfaces. `[0,0,0]` means flat (use element normal).
    pub node_normals: Vec<[f64; 3]>,
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
    /// Additional direct-radiator piston. The index is the stable driver
    /// index in [`LoudspeakerModel::drivers`].
    DriverAperture(usize),
    /// Product bass-reflex port aperture (volume-flow/impedance BC).
    PortAperture,
    /// Waveguide surface (hard plastic, rigid with admittance).
    Waveguide,
}

/// Connectivity and ring resolution pinned for a waveguide optimization stage.
#[derive(Clone)]
pub(crate) struct PinnedWaveguideMesh {
    reference_mesh: SpeakerMesh,
    waveguide_coordinates: Vec<Option<(f64, f64)>>,
    tweeter_only_nodes: Vec<bool>,
    baffle_z: f64,
    tweeter_center: [f64; 2],
}

impl PinnedWaveguideMesh {
    /// Resolve counts from the complete design envelope, then retain the
    /// reference candidate's connectivity for every shape evaluation.
    pub(crate) fn new(
        reference: &LoudspeakerModel,
        envelope: &LoudspeakerModel,
    ) -> Result<Self, String> {
        let envelope_waveguide = envelope
            .waveguide
            .as_ref()
            .ok_or_else(|| "pinned waveguide mesh requires a waveguide".to_string())?;
        let h_spline = CubicSpline::try_from_control_points(&envelope_waveguide.horizontal_spline)
            .map_err(|error| format!("invalid envelope horizontal spline: {error:?}"))?;
        let v_spline = CubicSpline::try_from_control_points(&envelope_waveguide.vertical_spline)
            .map_err(|error| format!("invalid envelope vertical spline: {error:?}"))?;
        let h_max = envelope.simulation.max_element_size();
        let baffle_exclusion_horizontal = h_spline
            .try_evaluate(0.0)
            .map_err(|error| format!("invalid envelope spline query: {error}"))?;
        let baffle_exclusion_vertical = v_spline
            .try_evaluate(0.0)
            .map_err(|error| format!("invalid envelope spline query: {error}"))?;
        let ring_count = ring_count(
            baffle_exclusion_horizontal.max(baffle_exclusion_vertical),
            h_max,
        );
        let axial_rings = (envelope_waveguide.depth / h_max).ceil().max(2.0) as usize;
        let counts = PinnedWaveguideCounts {
            ring_count,
            axial_rings,
            baffle_exclusion_horizontal,
            baffle_exclusion_vertical,
        };
        let mesh = try_generate_loudspeaker_mesh_with_counts(envelope, Some(counts))?;
        let mut waveguide_nodes = vec![false; mesh.nodes.len()];
        let mut tweeter_nodes = vec![false; mesh.nodes.len()];
        for (triangle, region) in mesh.triangles.iter().zip(&mesh.region) {
            let flags = match region {
                MeshRegion::Waveguide => Some(&mut waveguide_nodes),
                MeshRegion::TweeterAperture => Some(&mut tweeter_nodes),
                MeshRegion::PortAperture => None,
                _ => None,
            };
            if let Some(flags) = flags {
                for &node in triangle {
                    flags[node] = true;
                }
            }
        }
        let reference_waveguide = envelope
            .waveguide
            .as_ref()
            .ok_or_else(|| "pinned waveguide mesh requires a reference waveguide".to_string())?;
        let reference_h =
            CubicSpline::try_from_control_points(&reference_waveguide.horizontal_spline)
                .map_err(|error| format!("invalid reference horizontal spline: {error:?}"))?;
        let reference_v =
            CubicSpline::try_from_control_points(&reference_waveguide.vertical_spline)
                .map_err(|error| format!("invalid reference vertical spline: {error:?}"))?;
        let baffle_z = reference.cabinet.depth * 0.5;
        let tweeter_center = [reference.tweeter.x_offset, reference.tweeter.y_offset];
        let waveguide_coordinates = mesh
            .nodes
            .iter()
            .zip(&waveguide_nodes)
            .map(|(node, &is_waveguide)| {
                if !is_waveguide {
                    return None;
                }
                let t = ((baffle_z - node[2]) / reference_waveguide.depth).clamp(0.0, 1.0);
                // Option-returning tabulation closure: a corrupt entry skips
                // the node instead of panicking the worker thread.
                let rh = reference_h.try_evaluate(t).ok()?;
                let rv = reference_v.try_evaluate(t).ok()?;
                let theta =
                    ((node[1] - tweeter_center[1]) / rv).atan2((node[0] - tweeter_center[0]) / rh);
                Some((t, theta))
            })
            .collect();
        let tweeter_only_nodes = tweeter_nodes
            .iter()
            .zip(waveguide_nodes)
            .map(|(&tweeter, waveguide)| tweeter && !waveguide)
            .collect();
        Ok(Self {
            reference_mesh: mesh,
            waveguide_coordinates,
            tweeter_only_nodes,
            baffle_z,
            tweeter_center,
        })
    }

    pub(crate) fn generate(&self, model: &LoudspeakerModel) -> Result<SpeakerMesh, String> {
        let waveguide = model
            .waveguide
            .as_ref()
            .ok_or_else(|| "pinned waveguide mesh requires a candidate waveguide".to_string())?;
        let h_spline = CubicSpline::try_from_control_points(&waveguide.horizontal_spline)
            .map_err(|error| format!("invalid candidate horizontal spline: {error}"))?;
        let v_spline = CubicSpline::try_from_control_points(&waveguide.vertical_spline)
            .map_err(|error| format!("invalid candidate vertical spline: {error}"))?;
        let mut mesh = self.reference_mesh.clone();
        for (index, coordinates) in self.waveguide_coordinates.iter().enumerate() {
            let Some((t, theta)) = coordinates else {
                continue;
            };
            let cos = theta.cos();
            let sin = theta.sin();
            // `t` is clamped to [0, 1] when the coordinates are tabulated, so
            // these queries are finite by construction; `?` turns a corrupt
            // table entry into `Err` instead of panicking the worker thread.
            let rh = h_spline
                .try_evaluate(*t)
                .map_err(|error| format!("invalid candidate spline query t={t}: {error}"))?;
            let rv = v_spline
                .try_evaluate(*t)
                .map_err(|error| format!("invalid candidate spline query t={t}: {error}"))?;
            let rh_prime = h_spline
                .try_derivative(*t)
                .map_err(|error| format!("invalid candidate spline query t={t}: {error}"))?;
            let rv_prime = v_spline
                .try_derivative(*t)
                .map_err(|error| format!("invalid candidate spline query t={t}: {error}"))?;
            mesh.nodes[index] = [
                self.tweeter_center[0] + rh * cos,
                self.tweeter_center[1] + rv * sin,
                self.baffle_z - waveguide.depth * t,
            ];
            let nx = waveguide.depth * rv * cos;
            let ny = waveguide.depth * rh * sin;
            let nz = rh_prime * rv * cos * cos + rv_prime * rh * sin * sin;
            let length = (nx * nx + ny * ny + nz * nz).sqrt();
            if !length.is_finite() || length <= 1.0e-15 {
                return Err("candidate waveguide produces a degenerate mesh normal".to_string());
            }
            mesh.node_normals[index] = [nx / length, ny / length, nz / length];
        }
        let tweeter_z = self.baffle_z - waveguide.depth;
        for (index, &tweeter_only) in self.tweeter_only_nodes.iter().enumerate() {
            if tweeter_only {
                mesh.nodes[index][2] = tweeter_z;
            }
        }
        Ok(mesh)
    }
}

#[derive(Clone, Copy)]
struct PinnedWaveguideCounts {
    ring_count: usize,
    axial_rings: usize,
    baffle_exclusion_horizontal: f64,
    baffle_exclusion_vertical: f64,
}

// ---------------------------------------------------------------------------
// Mesh builder
// ---------------------------------------------------------------------------

struct MeshBuilder {
    nodes: Vec<[f64; 3]>,
    normals: Vec<[f64; 3]>,
    triangles: Vec<[usize; 3]>,
    regions: Vec<MeshRegion>,
}

struct PortGeometry {
    ring: Vec<(f64, f64)>,
    ring_start: usize,
    center: [f64; 2],
    radius: f64,
}

impl MeshBuilder {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            normals: Vec::new(),
            triangles: Vec::new(),
            regions: Vec::new(),
        }
    }

    fn add_node(&mut self, p: [f64; 3]) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(p);
        self.normals.push([0.0, 0.0, 0.0]);
        idx
    }

    fn set_normal(&mut self, idx: usize, n: [f64; 3]) {
        self.normals[idx] = n;
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
            node_normals: self.normals,
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
///
/// # Panics
///
/// Panics when the waveguide splines are invalid; prefer
/// [`try_generate_loudspeaker_mesh`], which reports them as `Err`.
pub fn generate_loudspeaker_mesh(model: &LoudspeakerModel) -> SpeakerMesh {
    try_generate_loudspeaker_mesh(model).expect("invalid waveguide spline control points")
}

/// Fallible version of [`generate_loudspeaker_mesh`]: an invalid waveguide
/// spline returns `Err` instead of panicking the caller (or an optimiser
/// worker thread).
pub fn try_generate_loudspeaker_mesh(model: &LoudspeakerModel) -> Result<SpeakerMesh, String> {
    try_generate_loudspeaker_mesh_with_counts(model, None)
}

fn try_generate_loudspeaker_mesh_with_counts(
    model: &LoudspeakerModel,
    pinned_waveguide_counts: Option<PinnedWaveguideCounts>,
) -> Result<SpeakerMesh, String> {
    let h_max = model.simulation.max_element_size();
    let mut b = MeshBuilder::new();

    let w2 = model.cabinet.width / 2.0;
    let h2 = model.cabinet.height / 2.0;
    let d2 = model.cabinet.depth / 2.0;

    // Woofer and tweeter centers on front baffle (z = +d2)
    let woofer_center = [model.woofer.x_offset, model.woofer.y_offset];
    let tweeter_center = [model.tweeter.x_offset, model.tweeter.y_offset];

    let n_woofer_ring = ring_count(model.woofer.radius, h_max);

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
    let woofer_ring_2d: Vec<(f64, f64)> = circle_points_2d(
        woofer_center[0],
        woofer_center[1],
        model.woofer.radius,
        n_woofer_ring,
    );

    // Register woofer ring nodes
    let woofer_ring_start = b.nodes.len();
    for &(x, y) in &woofer_ring_2d {
        b.add_node([x, y, d2]);
    }
    let port_geometry = add_product_port_ring(&mut b, model, h_max, d2);

    if let Some(ref wg) = model.waveguide {
        // --- Waveguide path: tweeter hole is the waveguide rim ellipse ---
        let h_spline = CubicSpline::try_from_control_points(&wg.horizontal_spline)
            .map_err(|error| format!("invalid waveguide horizontal spline: {error}"))?;
        let v_spline = CubicSpline::try_from_control_points(&wg.vertical_spline)
            .map_err(|error| format!("invalid waveguide vertical spline: {error}"))?;
        let wg_rim_h = h_spline
            .try_evaluate(0.0)
            .map_err(|error| format!("invalid waveguide spline query: {error}"))?;
        let wg_rim_v = v_spline
            .try_evaluate(0.0)
            .map_err(|error| format!("invalid waveguide spline query: {error}"))?;
        let n_wg_ring = pinned_waveguide_counts
            .map(|counts| counts.ring_count)
            .unwrap_or_else(|| ring_count(wg_rim_h.max(wg_rim_v), h_max));

        let wg_ring_2d: Vec<(f64, f64)> = ellipse_points_2d(
            tweeter_center[0],
            tweeter_center[1],
            pinned_waveguide_counts
                .map(|counts| counts.baffle_exclusion_horizontal)
                .unwrap_or(wg_rim_h),
            pinned_waveguide_counts
                .map(|counts| counts.baffle_exclusion_vertical)
                .unwrap_or(wg_rim_v),
            n_wg_ring,
        );
        let wg_ring_start = b.nodes.len();
        for &(x, y) in &wg_ring_2d {
            b.add_node([x, y, d2]);
        }

        let mut extra_rings = Vec::new();
        let mut extra_ring_starts = Vec::new();
        for driver in &model.additional_drivers {
            let ring = circle_points_2d(
                driver.x_offset,
                driver.y_offset,
                driver.radius,
                ring_count(driver.radius, h_max),
            );
            let start = b.nodes.len();
            for &(x, y) in &ring {
                b.add_node([x, y, d2]);
            }
            extra_ring_starts.push(start);
            extra_rings.push(ring);
        }
        let mut extra_holes: Vec<_> = extra_rings
            .iter()
            .enumerate()
            .map(|(index, ring)| ExtraBaffleHole {
                ring,
                ring_start: extra_ring_starts[index],
                center: [
                    model.additional_drivers[index].x_offset,
                    model.additional_drivers[index].y_offset,
                ],
                radius: model.additional_drivers[index].radius,
            })
            .collect();
        if let Some(port) = &port_geometry {
            extra_holes.push(ExtraBaffleHole {
                ring: &port.ring,
                ring_start: port.ring_start,
                center: port.center,
                radius: port.radius,
            });
        }

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
            &extra_holes,
        );

        // Woofer aperture (cone or flat disk)
        if let Some(ref cone) = model.woofer.cone {
            add_cone_shell(
                &mut b,
                &woofer_ring_2d,
                woofer_ring_start,
                woofer_center,
                d2,
                cone.cone_depth,
                cone.dustcap_radius,
                model.woofer.radius,
                h_max,
                MeshRegion::WooferAperture,
            );
        } else {
            add_disk(
                &mut b,
                &woofer_ring_2d,
                woofer_ring_start,
                woofer_center,
                d2,
                h_max,
                MeshRegion::WooferAperture,
                true,
            );
        }

        // Waveguide surface + tweeter aperture
        add_waveguide(
            &mut b,
            wg,
            &wg_ring_2d,
            wg_ring_start,
            tweeter_center,
            model.tweeter.radius,
            d2,
            h_max,
            pinned_waveguide_counts.map(|counts| counts.axial_rings),
        )?;
        add_additional_apertures(
            &mut b,
            &model.additional_drivers,
            &extra_rings,
            &extra_ring_starts,
            d2,
            h_max,
        );
        add_port_aperture(&mut b, port_geometry.as_ref(), d2, h_max);
    } else {
        // --- No-waveguide path: both holes are circles ---
        let n_tweeter_ring = ring_count(model.tweeter.radius, h_max);
        let tweeter_ring_2d: Vec<(f64, f64)> = circle_points_2d(
            tweeter_center[0],
            tweeter_center[1],
            model.tweeter.radius,
            n_tweeter_ring,
        );
        let tweeter_ring_start = b.nodes.len();
        for &(x, y) in &tweeter_ring_2d {
            b.add_node([x, y, d2]);
        }

        // Baffle with two circular holes
        let mut extra_rings = Vec::new();
        let mut extra_ring_starts = Vec::new();
        for driver in &model.additional_drivers {
            let ring = circle_points_2d(
                driver.x_offset,
                driver.y_offset,
                driver.radius,
                ring_count(driver.radius, h_max),
            );
            let start = b.nodes.len();
            for &(x, y) in &ring {
                b.add_node([x, y, d2]);
            }
            extra_ring_starts.push(start);
            extra_rings.push(ring);
        }
        let mut extra_holes: Vec<_> = extra_rings
            .iter()
            .enumerate()
            .map(|(index, ring)| ExtraBaffleHole {
                ring,
                ring_start: extra_ring_starts[index],
                center: [
                    model.additional_drivers[index].x_offset,
                    model.additional_drivers[index].y_offset,
                ],
                radius: model.additional_drivers[index].radius,
            })
            .collect();
        if let Some(port) = &port_geometry {
            extra_holes.push(ExtraBaffleHole {
                ring: &port.ring,
                ring_start: port.ring_start,
                center: port.center,
                radius: port.radius,
            });
        }

        add_baffle_with_holes(
            &mut b,
            [-w2, -h2],
            [w2, h2],
            &woofer_ring_2d,
            woofer_ring_start,
            woofer_center,
            model.woofer.radius,
            &tweeter_ring_2d,
            tweeter_ring_start,
            tweeter_center,
            model.tweeter.radius,
            model.tweeter.radius,
            d2,
            h_max,
            &extra_holes,
        );

        // Woofer aperture (cone or flat disk)
        if let Some(ref cone) = model.woofer.cone {
            add_cone_shell(
                &mut b,
                &woofer_ring_2d,
                woofer_ring_start,
                woofer_center,
                d2,
                cone.cone_depth,
                cone.dustcap_radius,
                model.woofer.radius,
                h_max,
                MeshRegion::WooferAperture,
            );
        } else {
            add_disk(
                &mut b,
                &woofer_ring_2d,
                woofer_ring_start,
                woofer_center,
                d2,
                h_max,
                MeshRegion::WooferAperture,
                true,
            );
        }

        // Tweeter aperture (cone or flat disk)
        if let Some(ref cone) = model.tweeter.cone {
            add_cone_shell(
                &mut b,
                &tweeter_ring_2d,
                tweeter_ring_start,
                tweeter_center,
                d2,
                cone.cone_depth,
                cone.dustcap_radius,
                model.tweeter.radius,
                h_max,
                MeshRegion::TweeterAperture,
            );
        } else {
            add_disk(
                &mut b,
                &tweeter_ring_2d,
                tweeter_ring_start,
                tweeter_center,
                d2,
                h_max,
                MeshRegion::TweeterAperture,
                true,
            );
        }
        add_additional_apertures(
            &mut b,
            &model.additional_drivers,
            &extra_rings,
            &extra_ring_starts,
            d2,
            h_max,
        );
        add_port_aperture(&mut b, port_geometry.as_ref(), d2, h_max);
    }

    // The internal separator plate is acoustically isolated by the sealed
    // cabinet and intentionally omitted from the exterior BEM mesh: it has
    // no effect on the radiated field and including it produces spurious
    // Green's-function couplings on a non-physical interior surface.

    Ok(b.finish())
}

fn add_product_port_ring(
    b: &mut MeshBuilder,
    model: &LoudspeakerModel,
    h_max: f64,
    z: f64,
) -> Option<PortGeometry> {
    let port = model.product_design.as_ref()?.cabinet.port.as_ref()?;
    let center = port.position_m;
    let radius = port.diameter_m / 2.0;
    let ring = circle_points_2d(center[0], center[1], radius, ring_count(radius, h_max));
    let ring_start = b.nodes.len();
    for &(x, y) in &ring {
        b.add_node([x, y, z]);
    }
    Some(PortGeometry {
        ring,
        ring_start,
        center,
        radius,
    })
}

fn add_port_aperture(b: &mut MeshBuilder, geometry: Option<&PortGeometry>, z: f64, h_max: f64) {
    if let Some(port) = geometry {
        add_disk(
            b,
            &port.ring,
            port.ring_start,
            port.center,
            z,
            h_max,
            MeshRegion::PortAperture,
            true,
        );
    }
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

struct ExtraBaffleHole<'a> {
    ring: &'a [(f64, f64)],
    ring_start: usize,
    center: [f64; 2],
    radius: f64,
}

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
    extra_holes: &[ExtraBaffleHole<'_>],
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
                && !inside_ellipse(
                    x,
                    y,
                    wg_center,
                    wg_rim_h + h_max * 0.3,
                    wg_rim_v + h_max * 0.3,
                )
                && !extra_holes
                    .iter()
                    .any(|hole| inside_circle(x, y, hole.center, hole.radius + h_max * 0.3))
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

    for hole in extra_holes {
        for (i, &pt) in hole.ring.iter().enumerate() {
            pts_2d.push(pt);
            node_map.push(Some(hole.ring_start + i));
        }
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
            || extra_holes
                .iter()
                .any(|hole| inside_circle(cx, cy, hole.center, hole.radius))
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

fn add_additional_apertures(
    b: &mut MeshBuilder,
    drivers: &[crate::model::DriverConfig],
    rings: &[Vec<(f64, f64)>],
    ring_starts: &[usize],
    z: f64,
    h_max: f64,
) {
    for (offset, ((driver, ring), &ring_start)) in
        drivers.iter().zip(rings).zip(ring_starts).enumerate()
    {
        let region = MeshRegion::DriverAperture(offset + 2);
        if let Some(cone) = driver.cone {
            add_cone_shell(
                b,
                ring,
                ring_start,
                [driver.x_offset, driver.y_offset],
                z,
                cone.cone_depth,
                cone.dustcap_radius,
                driver.radius,
                h_max,
                region,
            );
        } else {
            add_disk(
                b,
                ring,
                ring_start,
                [driver.x_offset, driver.y_offset],
                z,
                h_max,
                region,
                true,
            );
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

        // Stitch outer ring to inner ring. `stitch_rings` emits +z normals
        // with `flip = false` (CCW rings viewed from +z), so the desired
        // normal direction is the negation of the stitch flip flag.
        stitch_rings(
            b,
            prev_ring_start,
            prev_ring_count,
            inner_start,
            n_inner,
            region,
            !normal_positive_z,
        );

        prev_ring_start = inner_start;
        prev_ring_count = n_inner;
    }

    // Final center fan
    let center_node = b.add_node([center[0], center[1], z]);
    for i in 0..prev_ring_count {
        let j = (i + 1) % prev_ring_count;
        if normal_positive_z {
            b.add_tri(
                center_node,
                prev_ring_start + i,
                prev_ring_start + j,
                region,
            );
        } else {
            b.add_tri(
                center_node,
                prev_ring_start + j,
                prev_ring_start + i,
                region,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Conical shell (driver cone)
// ---------------------------------------------------------------------------

/// Add a truncated conical shell from outer ring (at baffle) to dustcap disk.
///
/// The outer ring at `baffle_z` connects to the baffle hole. The cone recesses
/// behind the baffle to `baffle_z - cone_depth` at `dustcap_radius`.
/// Node normals are set to the analytical cone normal for curved BEM integration.
#[allow(clippy::too_many_arguments)]
fn add_cone_shell(
    b: &mut MeshBuilder,
    outer_ring: &[(f64, f64)],
    outer_ring_start: usize,
    center: [f64; 2],
    baffle_z: f64,
    cone_depth: f64,
    dustcap_radius: f64,
    outer_radius: f64,
    h_max: f64,
    region: MeshRegion,
) {
    let dr = outer_radius - dustcap_radius;
    let slope = cone_depth / dr;
    let slope_length = (dr * dr + cone_depth * cone_depth).sqrt();
    let n_rings = (slope_length / h_max).ceil().max(2.0) as usize;
    let n_outer = outer_ring.len();

    // Set normals on outer ring nodes (they are shared with the baffle, but
    // the cone normal takes priority for BEM curved-surface integration).
    let inv_norm = 1.0 / (1.0 + slope * slope).sqrt();
    for (k, &(x, y)) in outer_ring.iter().enumerate() {
        let dx = x - center[0];
        let dy = y - center[1];
        let r = (dx * dx + dy * dy).sqrt();
        if r > 1e-15 {
            let cos_theta = dx / r;
            let sin_theta = dy / r;
            let n = [
                -slope * cos_theta * inv_norm,
                -slope * sin_theta * inv_norm,
                inv_norm,
            ];
            b.set_normal(outer_ring_start + k, n);
        }
    }

    let mut prev_start = outer_ring_start;
    let mut prev_count = n_outer;
    let dustcap_z = baffle_z - cone_depth;

    // Cone slant rings, including the final rim at t=1 (z=dustcap_z,
    // r=dustcap_radius). Each new ring carries the cone-slant analytical
    // normal so the BEM curved-surface integration sees a smooth cone.
    for ring_idx in 1..=n_rings {
        let t = ring_idx as f64 / n_rings as f64;
        let r = outer_radius * (1.0 - t) + dustcap_radius * t;
        let z = baffle_z - cone_depth * t;
        let n_pts = ring_count(r.max(h_max * 0.25), h_max).max(8);

        let this_start = b.nodes.len();
        for k in 0..n_pts {
            let angle = 2.0 * PI * k as f64 / n_pts as f64;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let x = center[0] + r * cos_a;
            let y = center[1] + r * sin_a;
            let idx = b.add_node([x, y, z]);
            b.set_normal(
                idx,
                [
                    -slope * cos_a * inv_norm,
                    -slope * sin_a * inv_norm,
                    inv_norm,
                ],
            );
        }

        stitch_rings(b, prev_start, prev_count, this_start, n_pts, region, false);

        prev_start = this_start;
        prev_count = n_pts;
    }

    // The flat dustcap disk shares the rim geometry with the cone slant's
    // innermost ring, but its outward normal is +z, not the cone-slant
    // normal. Reusing the same rim nodes would mix slanted and flat normals
    // on dustcap triangles via the BEM curved-surface interpolation
    // (`has_curved` activates whenever any node has a non-zero normal).
    // Allocate a separate rim ring at the same (x, y, dustcap_z) with zero
    // (flat) normals so the dustcap is integrated as a true planar surface.
    if dustcap_radius > 1e-12 {
        let dustcap_rim_start = b.nodes.len();
        let dustcap_ring: Vec<(f64, f64)> = (0..prev_count)
            .map(|k| {
                let p = b.nodes[prev_start + k];
                b.add_node([p[0], p[1], dustcap_z]);
                (p[0], p[1])
            })
            .collect();

        add_disk(
            b,
            &dustcap_ring,
            dustcap_rim_start,
            center,
            dustcap_z,
            h_max,
            region,
            true, // outward normal = +z
        );
    }
    // dustcap_radius == 0 ⇒ the cone tapers to a point, no flat dustcap.
}

// ---------------------------------------------------------------------------
// Waveguide surface
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn add_waveguide(
    b: &mut MeshBuilder,
    wg: &crate::model::WaveguideConfig,
    rim_ring: &[(f64, f64)],
    rim_ring_start: usize,
    tweeter_center: [f64; 2],
    _tweeter_radius: f64,
    baffle_z: f64,
    h_max: f64,
    pinned_axial_rings: Option<usize>,
) -> Result<(), String> {
    let h_spline = CubicSpline::try_from_control_points(&wg.horizontal_spline)
        .map_err(|error| format!("invalid waveguide horizontal spline: {error}"))?;
    let v_spline = CubicSpline::try_from_control_points(&wg.vertical_spline)
        .map_err(|error| format!("invalid waveguide vertical spline: {error}"))?;

    let n_u = rim_ring.len(); // azimuthal resolution matches rim
    let n_v = pinned_axial_rings.unwrap_or_else(|| (wg.depth / h_max).ceil().max(2.0) as usize);

    // Build rings from baffle (v=0) to tweeter (v=1)
    // v=0 ring is already rim_ring_start..rim_ring_start+n_u
    let mut prev_start = rim_ring_start;
    let mut prev_count = n_u;

    for v_idx in 1..=n_v {
        let t = v_idx as f64 / n_v as f64; // 0→1
        let z = baffle_z - wg.depth * t; // goes from baffle inward (negative z)
        // `t` is a finite ratio by construction; `?` still converts a corrupt
        // query into `Err` instead of panicking the meshing thread.
        let rh = h_spline
            .try_evaluate(t)
            .map_err(|error| format!("invalid waveguide spline query t={t}: {error}"))?;
        let rv = v_spline
            .try_evaluate(t)
            .map_err(|error| format!("invalid waveguide spline query t={t}: {error}"))?;
        let rh_prime = h_spline
            .try_derivative(t)
            .map_err(|error| format!("invalid waveguide spline query t={t}: {error}"))?;
        let rv_prime = v_spline
            .try_derivative(t)
            .map_err(|error| format!("invalid waveguide spline query t={t}: {error}"))?;

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
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let x = tweeter_center[0] + rh * cos_a;
            let y = tweeter_center[1] + rv * sin_a;
            let idx = b.add_node([x, y, z]);

            // Analytical normal for the parametric waveguide surface
            // S(t,θ) = (cx + rh·cosθ, cy + rv·sinθ, baffle_z - depth·t)
            // ∂S/∂t × ∂S/∂θ gives the outward-pointing normal.
            let nx = wg.depth * rv * cos_a;
            let ny = wg.depth * rh * sin_a;
            let nz = rh_prime * rv * cos_a * cos_a + rv_prime * rh * sin_a * sin_a;
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            if len > 1e-15 {
                b.set_normal(idx, [nx / len, ny / len, nz / len]);
            }
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

    // Set normals on the rim ring nodes (v=0, already existing)
    let t0 = 0.0;
    let rh0 = h_spline
        .try_evaluate(t0)
        .map_err(|error| format!("invalid waveguide spline query t={t0}: {error}"))?;
    let rv0 = v_spline
        .try_evaluate(t0)
        .map_err(|error| format!("invalid waveguide spline query t={t0}: {error}"))?;
    let rh_prime0 = h_spline
        .try_derivative(t0)
        .map_err(|error| format!("invalid waveguide spline query t={t0}: {error}"))?;
    let rv_prime0 = v_spline
        .try_derivative(t0)
        .map_err(|error| format!("invalid waveguide spline query t={t0}: {error}"))?;
    for k in 0..n_u {
        let angle = 2.0 * PI * k as f64 / n_u as f64;
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let nx = wg.depth * rv0 * cos_a;
        let ny = wg.depth * rh0 * sin_a;
        let nz = rh_prime0 * rv0 * cos_a * cos_a + rv_prime0 * rh0 * sin_a * sin_a;
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len > 1e-15 {
            b.set_normal(rim_ring_start + k, [nx / len, ny / len, nz / len]);
        }
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
    Ok(())
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
// VTU export (ParaView)
// ---------------------------------------------------------------------------

impl sonium_xem::io::vtu::MeshView for SpeakerMesh {
    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    fn num_cells(&self) -> usize {
        self.triangles.len()
    }

    fn nodes(&self) -> Box<dyn Iterator<Item = [f64; 3]> + '_> {
        Box::new(self.nodes.iter().copied())
    }

    fn cells(&self) -> Box<dyn Iterator<Item = (sonium_xem::io::vtu::CellKind, Vec<u64>)> + '_> {
        Box::new(self.triangles.iter().map(|tri| {
            (
                sonium_xem::io::vtu::CellKind::Triangle,
                vec![tri[0] as u64, tri[1] as u64, tri[2] as u64],
            )
        }))
    }
}

/// Export a `SpeakerMesh` (with per-cell region tag) to a `.vtu` for ParaView.
pub fn export_vtu(mesh: &SpeakerMesh, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
    let regions: Vec<f64> = mesh
        .region
        .iter()
        .map(|region| match region {
            MeshRegion::CabinetWall => 0.0,
            MeshRegion::WooferAperture => 1.0,
            MeshRegion::TweeterAperture => 2.0,
            MeshRegion::Waveguide => 3.0,
            MeshRegion::DriverAperture(index) => 4.0 + *index as f64,
            MeshRegion::PortAperture => 4.0,
        })
        .collect();
    let mut w = sonium_xem::io::vtu::VtuWriter::new(mesh);
    w.add_cell_scalar("region", &regions);
    w.write(path)
}

// ---------------------------------------------------------------------------
// Gmsh .geo export
// ---------------------------------------------------------------------------

/// Export a `SpeakerMesh` to a Gmsh `.geo` file for visualisation.
pub fn export_geo(mesh: &SpeakerMesh, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
    use std::collections::HashMap;

    let mut f = std::fs::File::create(path)?;

    writeln!(f, "// Speaker mesh – generated by sonium-sim-speaker")?;
    writeln!(f)?;

    // Points (1-indexed in Gmsh)
    for (i, p) in mesh.nodes.iter().enumerate() {
        writeln!(
            f,
            "Point({}) = {{{:.8e}, {:.8e}, {:.8e}}};",
            i + 1,
            p[0],
            p[1],
            p[2]
        )?;
    }
    writeln!(f)?;

    // Deduplicate edges and emit Lines
    let mut edge_map: HashMap<(usize, usize), usize> = HashMap::new();
    let mut line_id = 0usize;
    // For each triangle, store (line_id, sign) triples.
    let mut tri_lines: Vec<[i64; 3]> = Vec::with_capacity(mesh.triangles.len());

    for tri in &mesh.triangles {
        let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
        let mut signed = [0i64; 3];
        for (k, &(a, b)) in edges.iter().enumerate() {
            let canon = (a.min(b), a.max(b));
            let id = if let Some(&id) = edge_map.get(&canon) {
                id
            } else {
                line_id += 1;
                // Emit the line with canonical ordering (smaller index first)
                writeln!(
                    f,
                    "Line({}) = {{{}, {}}};",
                    line_id,
                    canon.0 + 1,
                    canon.1 + 1
                )?;
                edge_map.insert(canon, line_id);
                line_id
            };
            // Positive if the edge direction matches canonical, negative otherwise
            signed[k] = if a <= b { id as i64 } else { -(id as i64) };
        }
        tri_lines.push(signed);
    }
    writeln!(f)?;

    // Line Loops and Plane Surfaces (built-in kernel: separate tag namespaces)
    for (i, signed) in tri_lines.iter().enumerate() {
        let id = i + 1;
        writeln!(
            f,
            "Line Loop({id}) = {{{}, {}, {}}};",
            signed[0], signed[1], signed[2]
        )?;
        writeln!(f, "Plane Surface({id}) = {{{id}}};")?;
    }
    writeln!(f)?;

    // Physical Surfaces grouped by region
    let region_names = [
        (MeshRegion::CabinetWall, "CabinetWall"),
        (MeshRegion::WooferAperture, "WooferAperture"),
        (MeshRegion::TweeterAperture, "TweeterAperture"),
        (MeshRegion::Waveguide, "Waveguide"),
        (MeshRegion::PortAperture, "PortAperture"),
    ];
    for (region, name) in &region_names {
        let ids: Vec<String> = mesh
            .region
            .iter()
            .enumerate()
            .filter(|(_, r)| *r == region)
            .map(|(i, _)| (i + 1).to_string())
            .collect();
        if !ids.is_empty() {
            writeln!(f, "Physical Surface(\"{name}\") = {{{}}};", ids.join(", "))?;
        }
    }

    // Make the exported geometry immediately visible when opened in Gmsh.
    writeln!(f, "Mesh 2;")?;
    Ok(())
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
    use crate::model::*;
    use sonium_sim_speaker_modeling::{
        CabinetDesign, CrossoverBranch, CrossoverNetwork, ProductDesign, ProductDriver,
    };

    fn sample_model() -> LoudspeakerModel {
        LoudspeakerModel {
            cabinet: CabinetDimensions {
                height: 0.40,
                width: 0.22,
                depth: 0.28,
            },
            woofer: DriverConfig {
                radius: 0.065,
                x_offset: 0.0,
                y_offset: -0.08,
                piston_velocity: 0.001,
                transfer_function: None,
                driver_model: None,
                cone: None,
            },
            tweeter: DriverConfig {
                radius: 0.013,
                x_offset: 0.0,
                y_offset: 0.10,
                piston_velocity: 0.001,
                transfer_function: None,
                driver_model: None,
                cone: None,
            },
            additional_drivers: Vec::new(),
            product_design: None,
            separator: SeparatorConfig {
                y_position: 0.02,
                thickness: 0.018,
            },
            waveguide: Some(WaveguideConfig {
                horizontal_spline: vec![[0.0, 0.05], [0.33, 0.04], [0.66, 0.025], [1.0, 0.013]],
                vertical_spline: vec![[0.0, 0.04], [0.33, 0.03], [0.66, 0.02], [1.0, 0.013]],
                depth: 0.025,
            }),
            materials: SpeakerMaterialAssignment::default(),
            simulation: SimulationConfig {
                freq_start: 200.0,
                freq_end: 5000.0, // lower for faster mesh
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

        let has_wall = mesh.region.contains(&MeshRegion::CabinetWall);
        let has_woofer = mesh.region.contains(&MeshRegion::WooferAperture);
        let has_tweeter = mesh.region.contains(&MeshRegion::TweeterAperture);
        let has_waveguide = mesh.region.contains(&MeshRegion::Waveguide);

        assert!(has_wall, "missing CabinetWall region");
        assert!(has_woofer, "missing WooferAperture region");
        assert!(has_tweeter, "missing TweeterAperture region");
        assert!(has_waveguide, "missing Waveguide region");
    }

    #[test]
    fn product_port_is_a_baffle_hole_with_its_own_region() {
        let mut model = sample_model();
        model.waveguide = None;
        let mut cabinet = CabinetDesign::default();
        cabinet.port.as_mut().unwrap().position_m = [0.075, 0.0];
        model.product_design = Some(ProductDesign {
            cabinet,
            drivers: vec![
                ProductDriver {
                    id: "woofer".into(),
                    radius_m: model.woofer.radius,
                    position_m: [model.woofer.x_offset, model.woofer.y_offset, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    force_factor_tm: 5.0,
                    coil_resistance_ohm: 5.0,
                    coil_reactance_ohm: 1.0,
                    mechanical_resistance_n_s_per_m: 0.5,
                    mechanical_reactance_n_s_per_m: 2.0,
                    crossover_branch: 0,
                },
                ProductDriver {
                    id: "tweeter".into(),
                    radius_m: model.tweeter.radius,
                    position_m: [model.tweeter.x_offset, model.tweeter.y_offset, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    force_factor_tm: 2.0,
                    coil_resistance_ohm: 4.0,
                    coil_reactance_ohm: 0.5,
                    mechanical_resistance_n_s_per_m: 0.3,
                    mechanical_reactance_n_s_per_m: 1.0,
                    crossover_branch: 1,
                },
            ],
            crossover: CrossoverNetwork {
                branches: vec![CrossoverBranch::default(), CrossoverBranch::default()],
            },
            waveguide: None,
            structural_acoustic: None,
        });
        model.validate().unwrap();
        let mesh = generate_loudspeaker_mesh(&model);
        let port = model.product_design.as_ref().unwrap().cabinet.port.unwrap();
        let center = port.position_m;
        let radius = port.diameter_m / 2.0;
        let port_count = mesh
            .region
            .iter()
            .filter(|&&region| region == MeshRegion::PortAperture)
            .count();
        assert!(port_count > 0);
        for (tri, &region) in mesh.triangles.iter().zip(&mesh.region) {
            let centroid = tri.iter().fold([0.0; 2], |mut sum, &node| {
                sum[0] += mesh.nodes[node][0];
                sum[1] += mesh.nodes[node][1];
                sum
            });
            let centroid = [centroid[0] / 3.0, centroid[1] / 3.0];
            let distance =
                ((centroid[0] - center[0]).powi(2) + (centroid[1] - center[1]).powi(2)).sqrt();
            if region == MeshRegion::PortAperture {
                assert!(distance <= radius + 1.0e-9);
            } else if region == MeshRegion::CabinetWall
                && (tri.iter().map(|&node| mesh.nodes[node][2]).sum::<f64>() / 3.0
                    - model.cabinet.depth / 2.0)
                    .abs()
                    < 1.0e-9
            {
                assert!(
                    distance >= radius - 1.0e-9,
                    "cabinet triangle centroid {:?} lies inside port {:?} r={radius}",
                    centroid,
                    center
                );
            }
        }
    }

    #[test]
    fn additional_driver_aperture_is_meshed_at_its_placement() {
        let mut model = sample_model();
        model.waveguide = None;
        model.additional_drivers.push(DriverConfig {
            radius: 0.008,
            x_offset: 0.08,
            y_offset: 0.10,
            piston_velocity: 0.002,
            transfer_function: None,
            driver_model: None,
            cone: None,
        });
        model.validate().unwrap();

        let mesh = generate_loudspeaker_mesh(&model);
        let mut nodes = Vec::new();
        for (triangle, region) in mesh.triangles.iter().zip(&mesh.region) {
            if *region == MeshRegion::DriverAperture(2) {
                nodes.extend(triangle.iter().copied());
            }
        }
        assert!(
            !nodes.is_empty(),
            "additional driver aperture was not meshed"
        );
        let center = nodes.iter().fold([0.0; 2], |mut sum, &node| {
            sum[0] += mesh.nodes[node][0];
            sum[1] += mesh.nodes[node][1];
            sum
        });
        let count = nodes.len() as f64;
        assert!((center[0] / count - 0.08).abs() < 0.01);
        assert!((center[1] / count - 0.10).abs() < 0.01);
    }

    #[test]
    fn test_internal_separator_absent_from_mesh() {
        // The internal separator is acoustically isolated by the sealed cabinet
        // and must not be part of the exterior BEM mesh — including it would
        // contaminate the exterior solution with spurious mid-face couplings.
        let model = sample_model();
        let mesh = generate_loudspeaker_mesh(&model);

        let sep_y = model.separator.y_position;
        let half_thickness = model.separator.thickness / 2.0;

        for tri in &mesh.triangles {
            let p0 = mesh.nodes[tri[0]];
            let p1 = mesh.nodes[tri[1]];
            let p2 = mesh.nodes[tri[2]];
            let cy = (p0[1] + p1[1] + p2[1]) / 3.0;

            // No triangle's centroid should sit on the separator plane,
            // strictly inside the cabinet interior in x and z.
            let on_sep_plane = (cy - sep_y).abs() < half_thickness.max(1e-9);
            let cx = (p0[0] + p1[0] + p2[0]) / 3.0;
            let cz = (p0[2] + p1[2] + p2[2]) / 3.0;
            let interior_xz = cx.abs() < model.cabinet.width / 2.0 - 1e-9
                && cz.abs() < model.cabinet.depth / 2.0 - 1e-9;
            assert!(
                !(on_sep_plane && interior_xz),
                "found triangle on the internal separator plane"
            );
        }
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
            let area =
                0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            assert!(area > 1e-15, "zero-area triangle {i}");
        }
    }

    #[test]
    fn test_cone_shell_element_normals_align_with_node_normals() {
        let mut model = sample_model();
        model.woofer.cone = Some(ConeConfig {
            cone_depth: 0.008,
            dustcap_radius: 0.015,
        });

        let mesh = generate_loudspeaker_mesh(&model);
        let mut checked = 0usize;
        let mut woofer_triangles = 0usize;
        let mut woofer_triangles_with_node_normals = 0usize;
        let nonzero_mesh_normals = mesh
            .node_normals
            .iter()
            .filter(|&&normal| norm(normal) > 1e-12)
            .count();

        for (i, (tri, region)) in mesh.triangles.iter().zip(mesh.region.iter()).enumerate() {
            if *region != MeshRegion::WooferAperture {
                continue;
            }
            woofer_triangles += 1;

            let mut normal_sum = [0.0; 3];
            let mut normal_count = 0usize;
            for &node_idx in tri {
                let n = mesh.node_normals[node_idx];
                if norm(n) > 1e-12 {
                    normal_sum[0] += n[0];
                    normal_sum[1] += n[1];
                    normal_sum[2] += n[2];
                    normal_count += 1;
                }
            }
            if normal_count > 0 {
                woofer_triangles_with_node_normals += 1;
            }
            if normal_count < 2 {
                continue;
            }

            let element_normal = unit_triangle_normal(&mesh.nodes, *tri);
            let node_normal = normalize(normal_sum);

            assert!(
                element_normal[2] > 0.0,
                "cone element {i} points away from the acoustic side: {element_normal:?}"
            );
            assert!(
                dot(element_normal, node_normal) > 0.5,
                "cone element {i} normal {element_normal:?} opposes node normal {node_normal:?}"
            );
            checked += 1;
        }

        assert!(
            checked > 0,
            "no cone shell triangles were checked; woofer_triangles={woofer_triangles}, \
             woofer_triangles_with_node_normals={woofer_triangles_with_node_normals}, \
             nonzero_mesh_normals={nonzero_mesh_normals}"
        );
    }

    #[test]
    fn test_cone_dustcap_rim_normals_are_separate() {
        // The cone slant rim and the flat dustcap rim are geometrically
        // coincident but topologically distinct so the BEM does not mix
        // cone-slant normals into flat-dustcap triangles via the curved-
        // surface interpolation path. Verify that for every dustcap
        // triangle (those whose centroid lies on the dustcap plane and
        // inside dustcap_radius) all three node normals are zero, and that
        // every cone-slant triangle has all three node normals non-zero.
        let mut model = sample_model();
        let cone = ConeConfig {
            cone_depth: 0.008,
            dustcap_radius: 0.015,
        };
        model.woofer.cone = Some(cone);

        let mesh = generate_loudspeaker_mesh(&model);
        let woofer_y = model.woofer.y_offset;
        let baffle_z = model.cabinet.depth / 2.0;
        let dustcap_z = baffle_z - cone.cone_depth;

        let mut dustcap_triangles = 0usize;
        let mut slant_triangles = 0usize;

        for (tri, region) in mesh.triangles.iter().zip(mesh.region.iter()) {
            if *region != MeshRegion::WooferAperture {
                continue;
            }
            let p0 = mesh.nodes[tri[0]];
            let p1 = mesh.nodes[tri[1]];
            let p2 = mesh.nodes[tri[2]];
            let cz = (p0[2] + p1[2] + p2[2]) / 3.0;
            let cx = (p0[0] + p1[0] + p2[0]) / 3.0;
            let cy_centroid = (p0[1] + p1[1] + p2[1]) / 3.0 - woofer_y;
            let r_centroid = (cx * cx + cy_centroid * cy_centroid).sqrt();

            let on_dustcap =
                (cz - dustcap_z).abs() < 1e-9 && r_centroid < cone.dustcap_radius - 1e-9;

            let nonzero: usize = tri
                .iter()
                .filter(|&&n| norm(mesh.node_normals[n]) > 1e-12)
                .count();

            if on_dustcap {
                dustcap_triangles += 1;
                assert_eq!(
                    nonzero, 0,
                    "dustcap triangle has {nonzero} non-zero node normals; cone-slant \
                     normals must not bleed onto the flat dustcap"
                );
            } else {
                slant_triangles += 1;
                assert_eq!(
                    nonzero, 3,
                    "cone slant triangle should have all 3 nodes with non-zero \
                     (analytical) normals, got {nonzero}"
                );
            }
        }

        assert!(dustcap_triangles > 0, "no dustcap triangles found");
        assert!(slant_triangles > 0, "no cone slant triangles found");
    }

    #[test]
    fn test_export_geo() {
        let model = sample_model();
        let mesh = generate_loudspeaker_mesh(&model);
        let path = format!("{}/speaker_test.geo", std::env::temp_dir().display());
        export_geo(&mesh, &path).unwrap();
        assert!(std::path::Path::new(&path).exists());
        let geo = std::fs::read_to_string(&path).unwrap();
        assert!(geo.ends_with("Mesh 2;\n"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_invalid_waveguide_spline_returns_err_not_panic() {
        let mut model = sample_model();
        let waveguide = model.waveguide.as_mut().expect("sample has waveguide");
        waveguide.horizontal_spline = vec![[0.0, 0.05]];
        let error = try_generate_loudspeaker_mesh(&model)
            .err()
            .expect("invalid spline must fail, not panic");
        assert!(
            error.contains("horizontal spline"),
            "must name the offending spline, got: {error}"
        );

        let error = PinnedWaveguideMesh::new(&model, &model)
            .err()
            .expect("pinned topology must fail, not panic");
        assert!(
            error.contains("horizontal spline"),
            "pinned topology must propagate the error, got: {error}"
        );
    }

    fn unit_triangle_normal(nodes: &[[f64; 3]], tri: [usize; 3]) -> [f64; 3] {
        let p0 = nodes[tri[0]];
        let p1 = nodes[tri[1]];
        let p2 = nodes[tri[2]];
        let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        normalize([
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ])
    }

    fn normalize(v: [f64; 3]) -> [f64; 3] {
        let len = norm(v);
        assert!(len > 1e-15, "cannot normalize zero vector");
        [v[0] / len, v[1] / len, v[2] / len]
    }

    fn norm(v: [f64; 3]) -> f64 {
        dot(v, v).sqrt()
    }

    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
}
