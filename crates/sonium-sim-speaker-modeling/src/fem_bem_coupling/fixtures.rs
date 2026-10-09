//! Geometry fixtures for the interior-FEM/exterior-BEM coupling.
//!
//! Two structured cylindrical fixtures are provided:
//!
//! - [`cylindrical_box_with_circular_aperture`]: sealed cylindrical cabinet
//!   with a circular aperture centred on the front face (radiation /
//!   baffled-piston validation).
//! - [`cylindrical_box_with_external_port`]: cylindrical cabinet with a
//!   coaxial cylindrical port tube protruding from the front face
//!   (Helmholtz-resonator validation).
//!
//! # Mesh construction
//!
//! The tetrahedral volume meshes are built by a deterministic structured
//! cylindrical grid builder (rings × radial bands × axial levels, Kuhn
//! decomposition of the hex cells and a matching 3-tet prism decomposition
//! at the axis), rather than by `tetrahedralize_surface_with_facets`. The
//! structured builder remains deliberate: it gives exact control over the
//! aperture circle (it coincides with a radial ring), so boundary markers
//! classify cleanly and the FEM boundary faces align 1:1 with the BEM aperture
//! elements via
//! [`crate::fem_bem_coupling::mesh_bridge::build_exterior_bem_mesh`].
//! `sonium-mesher` now recovers the formerly problematic annular and
//! re-entrant constrained facets; that generic path is appropriate for future
//! non-axisymmetric product geometry, not a reason to destabilize this
//! analytical coupling fixture.
//!
//! Boundary markers:
//!
//! - [`WALL_MARKER`] — rigid cabinet walls,
//! - [`APERTURE_MARKER`] — the aperture disc (FEM↔BEM interface),
//! - [`DRIVER_MARKER`] — the back face (z = 0), usable as a
//!   prescribed-velocity piston drive for the interior solve.

use sonium_fem::mesh::{ElementType as FemElementType, Mesh as FemMesh, Point};
use std::collections::HashMap;

/// Boundary marker for rigid cabinet walls.
pub const WALL_MARKER: i32 = 101;
/// Boundary marker for the aperture disc (FEM↔BEM interface).
pub const APERTURE_MARKER: i32 = 102;
/// Boundary marker for the driver piston face (back face, z = 0).
pub const DRIVER_MARKER: i32 = 103;

/// Errors raised while building a coupling fixture.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FixtureError {
    #[error("{field} must be finite and greater than zero")]
    InvalidDimension { field: &'static str },
    #[error("aperture radius {aperture} must be smaller than the box radius {radius}")]
    ApertureTooLarge { aperture: f64, radius: f64 },
    #[error("tetrahedralized mesh is missing boundary faces for marker {marker}")]
    MissingBoundaryMarker { marker: i32 },
}

fn require_positive(field: &'static str, value: f64) -> Result<(), FixtureError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(FixtureError::InvalidDimension { field })
    }
}

/// Discretization controls for the cylindrical fixtures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CylinderFixtureDiscretization {
    /// Nodes around each ring (aperture circle, cylinder perimeter).
    pub n_circumference: usize,
    /// Axial cell count along the box depth.
    pub n_box_depth: usize,
    /// Radial band count on the box end-cap annulus (from the aperture
    /// circle to the box radius).
    pub n_cap_radial: usize,
    /// Radial band count inside the aperture circle.
    pub n_aperture_radial: usize,
    /// Axial cell count along the port tube (external-port fixture only).
    pub n_port_depth: usize,
}

impl Default for CylinderFixtureDiscretization {
    fn default() -> Self {
        Self {
            n_circumference: 24,
            n_box_depth: 6,
            n_cap_radial: 4,
            n_aperture_radial: 3,
            n_port_depth: 4,
        }
    }
}

/// A tagged tetrahedral FEM mesh plus the geometry data needed by the
/// coupled solve and its validations.
#[derive(Debug, Clone)]
pub struct CouplingFixture {
    /// Interior tet mesh with tagged boundary faces.
    pub fem_mesh: FemMesh,
    /// Marker of rigid walls.
    pub wall_marker: i32,
    /// Marker of the aperture disc.
    pub aperture_marker: i32,
    /// Marker of the driver piston face, if requested.
    pub driver_marker: Option<i32>,
    /// Box (cavity) radius.
    pub box_radius_m: f64,
    /// Box depth.
    pub box_depth_m: f64,
    /// Aperture radius (geometric).
    pub aperture_radius_m: f64,
    /// Port tube length for the external-port fixture.
    pub port_length_m: Option<f64>,
}

impl CouplingFixture {
    /// Cavity volume seen by the lumped Helmholtz reference: the box volume
    /// (the external port tube is outside the box).
    pub fn cavity_volume_m3(&self) -> f64 {
        std::f64::consts::PI * self.box_radius_m * self.box_radius_m * self.box_depth_m
    }

    /// Geometric aperture area.
    pub fn aperture_area_m2(&self) -> f64 {
        std::f64::consts::PI * self.aperture_radius_m * self.aperture_radius_m
    }

    /// Number of FEM boundary faces carrying the aperture marker.
    pub fn num_aperture_faces(&self) -> usize {
        self.fem_mesh
            .boundaries
            .iter()
            .filter(|b| b.marker == self.aperture_marker)
            .count()
    }
}

/// Deterministic structured cylindrical tet-mesh builder.
///
/// Nodes live on `(level, ring, sector)` integer coordinates: `ring = 0` is
/// the axis node, rings `1..=n_r[level]` are concentric circles of `n_c`
/// nodes. Cells between consecutive levels use the min ring count of the
/// two levels, which naturally handles the box→port junction (port levels
/// carry only the aperture rings).
struct CylinderGrid {
    points: Vec<Point>,
    tets: Vec<[usize; 4]>,
    ids: HashMap<(usize, usize, usize), usize>,
    n_c: usize,
}

impl CylinderGrid {
    fn new(n_c: usize) -> Self {
        Self {
            points: Vec::new(),
            tets: Vec::new(),
            ids: HashMap::new(),
            n_c,
        }
    }

    fn node(&mut self, level: usize, ring: usize, sector: usize, p: Point) -> usize {
        *self
            .ids
            .entry((level, ring, sector % self.n_c))
            .or_insert_with(|| {
                self.points.push(p);
                self.points.len() - 1
            })
    }

    fn ring_node(&mut self, level: usize, ring: usize, sector: usize, r: f64, z: f64) -> usize {
        let phi = 2.0 * std::f64::consts::PI * (sector % self.n_c) as f64 / self.n_c as f64;
        self.node(
            level,
            ring,
            sector,
            Point::new_3d(r * phi.cos(), r * phi.sin(), z),
        )
    }

    fn add_tet(&mut self, a: usize, b: usize, c: usize, d: usize) {
        let (pa, pb, pc, pd) = (
            &self.points[a],
            &self.points[b],
            &self.points[c],
            &self.points[d],
        );
        let v1 = [pb.x - pa.x, pb.y - pa.y, pb.z - pa.z];
        let v2 = [pc.x - pa.x, pc.y - pa.y, pc.z - pa.z];
        let v3 = [pd.x - pa.x, pd.y - pa.y, pd.z - pa.z];
        let det = v1[0] * (v2[1] * v3[2] - v2[2] * v3[1]) - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
            + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]);
        if det > 0.0 {
            self.tets.push([a, b, c, d]);
        } else {
            // Flip orientation by swapping the last two vertices.
            self.tets.push([a, b, d, c]);
        }
    }

    /// Build the grid. `radii[level]` lists the ring radii present at that
    /// axial level (strictly increasing, all > 0); `z_levels` the level
    /// heights (strictly increasing).
    fn build(&mut self, radii: &[Vec<f64>], z_levels: &[f64]) {
        let n_levels = z_levels.len();
        for l in 0..n_levels {
            let z = z_levels[l];
            self.node(l, 0, 0, Point::new_3d(0.0, 0.0, z));
            for (jr, &r) in radii[l].iter().enumerate() {
                for j in 0..self.n_c {
                    self.ring_node(l, jr + 1, j, r, z);
                }
            }
        }
        for l in 0..n_levels - 1 {
            let n_r = radii[l].len().min(radii[l + 1].len());
            // Axis prisms: a = axis_l, b = (1,j,l), c = (1,j1,l),
            // A = axis_{l+1}, B = (1,j,l+1), C = (1,j1,l+1).
            // Decomposition (a,b,c,A), (b,c,A,C), (b,A,B,C) matches the Kuhn
            // face diagonals of the first hex band and of the neighbour
            // prism (side diagonal ring_l ↔ axis_{l+1}).
            let axis_l = self.ids[&(l, 0, 0)];
            let axis_u = self.ids[&(l + 1, 0, 0)];
            for j in 0..self.n_c {
                let j1 = (j + 1) % self.n_c;
                let b = self.ids[&(l, 1, j)];
                let c = self.ids[&(l, 1, j1)];
                let b_up = self.ids[&(l + 1, 1, j)];
                let c_up = self.ids[&(l + 1, 1, j1)];
                self.add_tet(axis_l, b, c, axis_u);
                self.add_tet(b, c, axis_u, c_up);
                self.add_tet(b, axis_u, b_up, c_up);
            }
            // Hex bands between rings jr and jr+1, Kuhn 6-tet split
            // (translation-invariant, consistent across shared faces).
            for jr in 1..n_r {
                for j in 0..self.n_c {
                    let j1 = (j + 1) % self.n_c;
                    let v000 = self.ids[&(l, jr, j)];
                    let v100 = self.ids[&(l, jr, j1)];
                    let v010 = self.ids[&(l, jr + 1, j)];
                    let v110 = self.ids[&(l, jr + 1, j1)];
                    let v001 = self.ids[&(l + 1, jr, j)];
                    let v101 = self.ids[&(l + 1, jr, j1)];
                    let v011 = self.ids[&(l + 1, jr + 1, j)];
                    let v111 = self.ids[&(l + 1, jr + 1, j1)];
                    self.add_tet(v000, v100, v110, v111);
                    self.add_tet(v000, v110, v010, v111);
                    self.add_tet(v000, v010, v011, v111);
                    self.add_tet(v000, v011, v001, v111);
                    self.add_tet(v000, v001, v101, v111);
                    self.add_tet(v000, v101, v100, v111);
                }
            }
        }
    }

    fn into_mesh(self, aperture_radius: f64) -> FemMesh {
        let mut mesh = FemMesh::new(3);
        for p in &self.points {
            mesh.add_node(*p);
        }
        for tet in &self.tets {
            mesh.add_element(FemElementType::Tetrahedron, tet.to_vec());
        }
        mesh.detect_boundaries();

        let z_top = self
            .points
            .iter()
            .map(|p| p.z)
            .fold(f64::NEG_INFINITY, f64::max);
        let z_tol = z_top.abs().max(1.0) * 1e-9;
        for face in &mut mesh.boundaries {
            let (mut cz, mut cr, mut count) = (0.0, 0.0, 0.0);
            for &n in &face.nodes {
                let p = &mesh.nodes[n];
                cz += p.z;
                cr += (p.x * p.x + p.y * p.y).sqrt();
                count += 1.0;
            }
            cz /= count;
            cr /= count;
            face.marker = if cz < z_tol {
                DRIVER_MARKER
            } else if cz > z_top - z_tol && cr < aperture_radius {
                APERTURE_MARKER
            } else {
                WALL_MARKER
            };
        }
        mesh
    }
}

/// Ring radii for the cap grid: `n_ap` bands inside the aperture circle
/// `[0, aperture]`, then `n_out` bands to the box radius.
fn cap_radii(aperture: f64, radius: f64, n_ap: usize, n_out: usize) -> Vec<f64> {
    let mut r: Vec<f64> = (1..=n_ap)
        .map(|j| aperture * j as f64 / n_ap as f64)
        .collect();
    r.extend((1..=n_out).map(|j| aperture + (radius - aperture) * j as f64 / n_out as f64));
    r
}

/// Build a sealed cylindrical cabinet (radius `box_radius_m`, depth
/// `box_depth_m`, axis = z, back face at z = 0) with a circular aperture of
/// radius `aperture_radius_m` centred on the front face (z = depth).
///
/// Boundary markers: [`APERTURE_MARKER`] on the aperture disc,
/// [`DRIVER_MARKER`] on the back face (piston drive), [`WALL_MARKER`]
/// everywhere else.
pub fn cylindrical_box_with_circular_aperture(
    box_radius_m: f64,
    box_depth_m: f64,
    aperture_radius_m: f64,
    discretization: CylinderFixtureDiscretization,
) -> Result<CouplingFixture, FixtureError> {
    require_positive("box_radius_m", box_radius_m)?;
    require_positive("box_depth_m", box_depth_m)?;
    require_positive("aperture_radius_m", aperture_radius_m)?;
    if aperture_radius_m >= box_radius_m {
        return Err(FixtureError::ApertureTooLarge {
            aperture: aperture_radius_m,
            radius: box_radius_m,
        });
    }
    let d = discretization;
    let cap = cap_radii(
        aperture_radius_m,
        box_radius_m,
        d.n_aperture_radial,
        d.n_cap_radial,
    );
    let n_levels = d.n_box_depth + 1;
    let radii: Vec<Vec<f64>> = (0..n_levels).map(|_| cap.clone()).collect();
    let z_levels: Vec<f64> = (0..n_levels)
        .map(|l| box_depth_m * l as f64 / d.n_box_depth as f64)
        .collect();

    let mut grid = CylinderGrid::new(d.n_circumference);
    grid.build(&radii, &z_levels);
    let fem_mesh = grid.into_mesh(aperture_radius_m);

    let fixture = CouplingFixture {
        fem_mesh,
        wall_marker: WALL_MARKER,
        aperture_marker: APERTURE_MARKER,
        driver_marker: Some(DRIVER_MARKER),
        box_radius_m,
        box_depth_m,
        aperture_radius_m,
        port_length_m: None,
    };
    check_markers(&fixture)?;
    Ok(fixture)
}

/// Build a cylindrical cabinet with a coaxial external port tube: box of
/// radius `box_radius_m` and depth `box_depth_m` (z in `[0, box_depth_m]`),
/// port tube of radius `port_radius_m` and length `port_length_m` protruding
/// from the front face (z in `[box_depth_m, box_depth_m + port_length_m]`).
/// The aperture is the port mouth disc at the tube end.
///
/// Boundary markers: [`APERTURE_MARKER`] on the port mouth,
/// [`DRIVER_MARKER`] on the back face, [`WALL_MARKER`] on box walls, front
/// annulus, and tube wall.
pub fn cylindrical_box_with_external_port(
    box_radius_m: f64,
    box_depth_m: f64,
    port_radius_m: f64,
    port_length_m: f64,
    discretization: CylinderFixtureDiscretization,
) -> Result<CouplingFixture, FixtureError> {
    require_positive("box_radius_m", box_radius_m)?;
    require_positive("box_depth_m", box_depth_m)?;
    require_positive("port_radius_m", port_radius_m)?;
    require_positive("port_length_m", port_length_m)?;
    if port_radius_m >= box_radius_m {
        return Err(FixtureError::ApertureTooLarge {
            aperture: port_radius_m,
            radius: box_radius_m,
        });
    }
    let d = discretization;
    let cap = cap_radii(
        port_radius_m,
        box_radius_m,
        d.n_aperture_radial,
        d.n_cap_radial,
    );
    let port: Vec<f64> = (1..=d.n_aperture_radial)
        .map(|j| port_radius_m * j as f64 / d.n_aperture_radial as f64)
        .collect();
    let mut radii: Vec<Vec<f64>> = (0..=d.n_box_depth).map(|_| cap.clone()).collect();
    let mut z_levels: Vec<f64> = (0..=d.n_box_depth)
        .map(|l| box_depth_m * l as f64 / d.n_box_depth as f64)
        .collect();
    for l in 1..=d.n_port_depth {
        radii.push(port.clone());
        z_levels.push(box_depth_m + port_length_m * l as f64 / d.n_port_depth as f64);
    }

    let mut grid = CylinderGrid::new(d.n_circumference);
    grid.build(&radii, &z_levels);
    let fem_mesh = grid.into_mesh(port_radius_m);

    let fixture = CouplingFixture {
        fem_mesh,
        wall_marker: WALL_MARKER,
        aperture_marker: APERTURE_MARKER,
        driver_marker: Some(DRIVER_MARKER),
        box_radius_m,
        box_depth_m,
        aperture_radius_m: port_radius_m,
        port_length_m: Some(port_length_m),
    };
    check_markers(&fixture)?;
    Ok(fixture)
}

fn check_markers(fixture: &CouplingFixture) -> Result<(), FixtureError> {
    for tag in [fixture.aperture_marker, fixture.wall_marker] {
        if fixture.fem_mesh.boundaries.iter().all(|b| b.marker != tag) {
            return Err(FixtureError::MissingBoundaryMarker { marker: tag });
        }
    }
    if let Some(driver) = fixture.driver_marker
        && fixture
            .fem_mesh
            .boundaries
            .iter()
            .all(|b| b.marker != driver)
    {
        return Err(FixtureError::MissingBoundaryMarker { marker: driver });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fem_bem_coupling::mesh_bridge::build_exterior_bem_mesh;
    use sonium_fem::mesh::boundary_mesh::create_boundary_submesh;

    fn coarse() -> CylinderFixtureDiscretization {
        CylinderFixtureDiscretization {
            n_circumference: 16,
            n_box_depth: 4,
            n_cap_radial: 3,
            n_aperture_radial: 2,
            n_port_depth: 3,
        }
    }

    #[test]
    fn aperture_fixture_tags_and_mesh_quality() {
        let fixture = cylindrical_box_with_circular_aperture(0.15, 0.30, 0.05, coarse()).unwrap();
        // Aperture: fan (16) + one band (2×16) = 48 faces.
        assert_eq!(fixture.num_aperture_faces(), 48);
        let driver_faces = fixture
            .fem_mesh
            .boundaries
            .iter()
            .filter(|b| b.marker == DRIVER_MARKER)
            .count();
        assert!(driver_faces > 0);
        let wall_faces = fixture
            .fem_mesh
            .boundaries
            .iter()
            .filter(|b| b.marker == WALL_MARKER)
            .count();
        assert!(wall_faces > 0);
        assert!(fixture.fem_mesh.num_elements() > 0);

        // All tets positively oriented and non-degenerate.
        for tet in &fixture.fem_mesh.elements {
            let [a, b, c, d] = [tet.nodes[0], tet.nodes[1], tet.nodes[2], tet.nodes[3]];
            let (pa, pb, pc, pd) = (
                &fixture.fem_mesh.nodes[a],
                &fixture.fem_mesh.nodes[b],
                &fixture.fem_mesh.nodes[c],
                &fixture.fem_mesh.nodes[d],
            );
            let v1 = [pb.x - pa.x, pb.y - pa.y, pb.z - pa.z];
            let v2 = [pc.x - pa.x, pc.y - pa.y, pc.z - pa.z];
            let v3 = [pd.x - pa.x, pd.y - pa.y, pd.z - pa.z];
            let det = v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
                - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
                + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]);
            assert!(det > 0.0, "tet {:?} has non-positive volume", tet.nodes);
        }
    }

    #[test]
    fn aperture_submesh_mapping_and_area() {
        let radius = 0.15;
        let aperture = 0.05;
        let fixture =
            cylindrical_box_with_circular_aperture(radius, 0.30, aperture, coarse()).unwrap();
        let sub = create_boundary_submesh(&fixture.fem_mesh, APERTURE_MARKER);
        assert!(sub.num_elements() > 0);

        // Every aperture node sits at z = depth inside the aperture radius.
        for &parent in &sub.node_to_parent {
            let p = &fixture.fem_mesh.nodes[parent];
            assert!((p.z - 0.30).abs() < 1e-9, "aperture node z = {}", p.z);
            let r = (p.x * p.x + p.y * p.y).sqrt();
            assert!(r <= aperture + 1e-9, "aperture node r = {r}");
        }
        // Roundtrip mapping.
        for (sub_idx, &parent) in sub.node_to_parent.iter().enumerate() {
            assert_eq!(sub.submesh_node(parent), Some(sub_idx));
        }

        // Faceted aperture area ≈ π a² (polygon approximation).
        let bem = build_exterior_bem_mesh(
            &fixture.fem_mesh,
            APERTURE_MARKER,
            &[WALL_MARKER, DRIVER_MARKER],
        )
        .unwrap();
        let area = bem.aperture_area();
        let exact = std::f64::consts::PI * aperture * aperture;
        let rel = (area - exact).abs() / exact;
        assert!(rel < 0.05, "aperture area {area} vs {exact} (rel {rel})");
    }

    #[test]
    fn bem_converter_normals_point_outward() {
        let fixture = cylindrical_box_with_circular_aperture(0.15, 0.30, 0.05, coarse()).unwrap();
        let bem = build_exterior_bem_mesh(
            &fixture.fem_mesh,
            APERTURE_MARKER,
            &[WALL_MARKER, DRIVER_MARKER],
        )
        .unwrap();
        // Interior reference point on the axis.
        let reference = [0.0, 0.0, 0.15];
        for elem in &bem.mesh.elements {
            let d = [
                elem.center[0] - reference[0],
                elem.center[1] - reference[1],
                elem.center[2] - reference[2],
            ];
            let dot = elem.normal[0] * d[0] + elem.normal[1] * d[1] + elem.normal[2] * d[2];
            assert!(dot > 0.0, "inward normal at {:?} (dot {dot})", elem.center);
        }

        // Total surface area: side 2πRL + caps 2πR².
        let r = 0.15;
        let l = 0.30;
        let expected = 2.0 * std::f64::consts::PI * r * l + 2.0 * std::f64::consts::PI * r * r;
        let total: f64 = bem.mesh.elements.iter().map(|e| e.area).sum();
        let rel = (total - expected).abs() / expected;
        assert!(rel < 0.05, "surface area {total} vs {expected} (rel {rel})");
    }

    #[test]
    fn port_fixture_tags_and_alignment() {
        let fixture = cylindrical_box_with_external_port(0.15, 0.25, 0.03, 0.10, coarse()).unwrap();
        assert!(fixture.num_aperture_faces() > 0);
        let sub = create_boundary_submesh(&fixture.fem_mesh, APERTURE_MARKER);
        let z_mouth = 0.25 + 0.10;
        for &parent in &sub.node_to_parent {
            let p = &fixture.fem_mesh.nodes[parent];
            assert!((p.z - z_mouth).abs() < 1e-9, "aperture node z = {}", p.z);
            let r = (p.x * p.x + p.y * p.y).sqrt();
            assert!(r <= 0.03 + 1e-9, "aperture node r = {r}");
        }
        let bem = build_exterior_bem_mesh(
            &fixture.fem_mesh,
            APERTURE_MARKER,
            &[WALL_MARKER, DRIVER_MARKER],
        )
        .unwrap();
        assert_eq!(bem.aperture_elements.len(), fixture.num_aperture_faces());
        // Outward normals for the re-entrant junction too: reference point
        // on the box axis.
        let reference = [0.0, 0.0, 0.125];
        for elem in &bem.mesh.elements {
            let d = [
                elem.center[0] - reference[0],
                elem.center[1] - reference[1],
                elem.center[2] - reference[2],
            ];
            let dot = elem.normal[0] * d[0] + elem.normal[1] * d[1] + elem.normal[2] * d[2];
            assert!(dot > 0.0, "inward normal at {:?} (dot {dot})", elem.center);
        }
    }
}
