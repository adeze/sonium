//! Spline-port meridian geometry and fixed-topology mesh morphing (PORT-OPTIM Step 2).
//!
//! [`SplinePortGeometry`] pairs the straight reference tube and its end-effect
//! chambers with a [`SplinePortProfile`]. Optimisation candidates are produced
//! by morphing one baseline mesh (plan decision D6): duct nodes are radially
//! scaled by `r(x)/R0` — wall nodes land on `r(x)`, interior nodes scale
//! proportionally — and a few Laplacian smoothing passes relax the duct
//! interior while every boundary node stays exact. Morphing clones the
//! baseline connectivity, so the [`MARKER_BOX_CAP`]/[`MARKER_EXTERIOR_CAP`]/
//! [`MARKER_WALL`] classification and the deterministic jitter that keeps P1–P1
//! problems stable (the no-P1–P1 rule) are preserved by construction.
//!
//! [`SplinePortGeometry::build_validation_mesh`] regenerates a mesh from
//! scratch for the same profile, with independent node lines conforming to
//! `r(x)`. It exists only for validation cross-checks at independent
//! resolutions, never for the optimisation loop, where remeshing would make
//! the objective discontinuous.
//!
//! Validated envelope: pure geometry. No flow solve, no noise claim. Morphed
//! meshes feed the Tier 2 harness (Step 5), which re-audits element positivity
//! before any solve.

use serde::{Deserialize, Serialize};
use sonium_fem::mesh::{BoundaryType, ElementType, Mesh, Point, triangulate};

use super::error::PortFlowError;
use super::geometry::{
    MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL, SlotPortGeometry, build_slot_port_mesh,
    spaced_lines,
};
use super::profile::SplinePortProfile;

/// Default number of Laplacian smoothing passes applied to the duct interior
/// after radial scaling. Boundary nodes stay exact on every pass.
pub const DEFAULT_MORPH_SMOOTHING_PASSES: usize = 3;

/// Chamber–duct–chamber meridian geometry carrying a spline port profile.
///
/// The reference tube (`duct_length_m`, `bore_radius_m`) and the chambers
/// (`chamber_length_m`, `chamber_radius_m`) fix the optimisation envelope;
/// [`SplinePortProfile`] supplies the duct wall `r(x)`. The 2D meridian mesh
/// spans `x ∈ [−Lc, Ld + Lc]`, `|y| ≤ Rc`, with the duct occupying
/// `x ∈ [0, Ld]`, `|y| ≤ r(x)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplinePortGeometry {
    /// Duct (profile) length `Ld` in m; must match the profile length.
    pub duct_length_m: f64,
    /// Straight-bore radius `R0` in m; must match the profile bore.
    pub bore_radius_m: f64,
    /// Axial length of each end-effect chamber in m.
    pub chamber_length_m: f64,
    /// Chamber half-height (meridian radius) in m; must exceed `R0`.
    pub chamber_radius_m: f64,
    /// Target cell size in m.
    pub mesh_size_m: f64,
    /// Duct wall profile `r(x)` on `[0, Ld]`.
    pub profile: SplinePortProfile,
}

impl SplinePortGeometry {
    fn validate(&self) -> Result<(), PortFlowError> {
        for (name, value) in [
            ("duct_length_m", self.duct_length_m),
            ("bore_radius_m", self.bore_radius_m),
            ("chamber_length_m", self.chamber_length_m),
            ("chamber_radius_m", self.chamber_radius_m),
            ("mesh_size_m", self.mesh_size_m),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(PortFlowError::InvalidConfig(format!(
                    "{name} must be positive and finite, got {value}"
                )));
            }
        }
        if self.chamber_radius_m <= self.bore_radius_m {
            return Err(PortFlowError::InvalidConfig(format!(
                "chamber_radius_m ({}) must exceed bore_radius_m ({}) to resolve end effects",
                self.chamber_radius_m, self.bore_radius_m
            )));
        }
        let length_tol = 1.0e-9 * self.duct_length_m.max(1.0e-12);
        if (self.profile.length_m() - self.duct_length_m).abs() > length_tol {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline profile length {} does not match duct_length_m {}",
                self.profile.length_m(),
                self.duct_length_m
            )));
        }
        let bore_tol = 1.0e-9 * self.bore_radius_m.max(1.0e-12);
        if (self.profile.bore_radius_m() - self.bore_radius_m).abs() > bore_tol {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline profile bore {} does not match bore_radius_m {}",
                self.profile.bore_radius_m(),
                self.bore_radius_m
            )));
        }
        // Re-audit the profile (cheap dense-grid check) so a deserialized or
        // hand-built geometry cannot smuggle an illegal wall into a solve.
        self.profile.validate()?;
        Ok(())
    }

    /// The straight-tube slot geometry this spline geometry morphs from.
    ///
    /// Duct height `2·R0` and chamber height `2·Rc` reproduce the reference
    /// bore exactly; [`Self::build_baseline_mesh`] meshes it with the same
    /// deterministic builder the slot-port path uses.
    pub fn baseline_slot_geometry(&self) -> SlotPortGeometry {
        SlotPortGeometry {
            duct_length_m: self.duct_length_m,
            duct_height_m: 2.0 * self.bore_radius_m,
            chamber_length_m: self.chamber_length_m,
            chamber_height_m: 2.0 * self.chamber_radius_m,
            mesh_size_m: self.mesh_size_m,
        }
    }

    /// Build the straight-tube baseline mesh every candidate morphs from.
    ///
    /// The profile is validated but otherwise ignored: the baseline is always
    /// the reference bore, built once and shared by all candidates.
    pub fn build_baseline_mesh(&self) -> Result<Mesh, PortFlowError> {
        self.validate()?;
        build_slot_port_mesh(&self.baseline_slot_geometry())
    }

    /// Whether the profile is exactly the straight bore (`u = 0` everywhere).
    ///
    /// Morphing such a profile is the identity map, so
    /// [`Self::morph_baseline_mesh`] returns the baseline clone untouched
    /// (bit-identical, no smoothing drift).
    fn is_straight(&self) -> bool {
        self.profile
            .radii()
            .iter()
            .all(|radius| *radius == self.bore_radius_m)
    }

    /// Reject baselines that cannot belong to this reference geometry.
    ///
    /// Checks the dimension, the presence of all three boundary markers, and
    /// the domain extents (`x ∈ [−Lc, Ld + Lc]`, `|y| ≤ Rc`, exact on the
    /// unjittered caps and chamber walls). This catches dimension mismatches
    /// (profile length/bore against geometry, baseline against geometry)
    /// before any node moves.
    fn check_baseline_compatible(baseline: &Mesh, geo: &SplinePortGeometry) -> Result<(), PortFlowError> {
        if baseline.dimension != 2 {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline morph needs a 2D meridian baseline, got dimension {}",
                baseline.dimension
            )));
        }
        for marker in [MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL] {
            let count = baseline
                .boundaries
                .iter()
                .filter(|face| face.marker == marker)
                .count();
            if count == 0 {
                return Err(PortFlowError::Mesh(format!(
                    "spline morph baseline has no marker-{marker} boundary edges"
                )));
            }
        }
        if baseline.nodes.is_empty() {
            return Err(PortFlowError::Mesh(
                "spline morph baseline has no nodes".to_string(),
            ));
        }
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        for point in &baseline.nodes {
            x_min = x_min.min(point.x);
            x_max = x_max.max(point.x);
            y_min = y_min.min(point.y);
            y_max = y_max.max(point.y);
        }
        let tol = 1.0e-9;
        let x_lo = -geo.chamber_length_m;
        let x_hi = geo.duct_length_m + geo.chamber_length_m;
        if (x_min - x_lo).abs() > tol
            || (x_max - x_hi).abs() > tol
            || (y_min + geo.chamber_radius_m).abs() > tol
            || (y_max - geo.chamber_radius_m).abs() > tol
        {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline morph baseline extents x ∈ [{x_min}, {x_max}], y ∈ [{y_min}, {y_max}] \
                 do not match reference x ∈ [{x_lo}, {x_hi}], |y| ≤ {}",
                geo.chamber_radius_m
            )));
        }
        Ok(())
    }

    /// Morph the shared straight-tube baseline onto this geometry's profile.
    ///
    /// Every node is radially scaled by `r(x)/R0` (the profile clamps chamber
    /// stations to exactly `R0`, so chamber nodes are bit-identical and the
    /// duct ends stay exact), then `smoothing_passes` Laplacian passes relax
    /// interior duct nodes while all boundary nodes stay exact. Smoothing
    /// stops early when a pass would invert an element, so even
    /// extreme-but-legal walls keep a valid mesh. Connectivity
    /// and markers are cloned untouched, preserving the no-P1–P1 jitter.
    ///
    /// Returns [`PortFlowError::Mesh`] when the morphed mesh contains an
    /// inverted or degenerate triangle.
    pub fn morph_baseline_mesh(
        &self,
        baseline: &Mesh,
        smoothing_passes: usize,
    ) -> Result<Mesh, PortFlowError> {
        self.validate()?;
        Self::check_baseline_compatible(baseline, self)?;
        if self.is_straight() {
            return Ok(baseline.clone());
        }
        let mut mesh = baseline.clone();
        let r0 = self.bore_radius_m;
        for point in &mut mesh.nodes {
            point.y *= self.profile.local_radius_m(point.x) / r0;
        }
        laplacian_smooth_duct_interior(&mut mesh, self.duct_length_m, smoothing_passes);
        let min_area = min_triangle_area_m2(&mesh);
        if min_area <= 0.0 || !min_area.is_finite() {
            return Err(PortFlowError::Mesh(format!(
                "spline morph produced inverted or degenerate elements \
                 (min signed area {min_area:.3e} m²)"
            )));
        }
        Ok(mesh)
    }

    /// Meridian half-height at axial coordinate `x_m`.
    ///
    /// Duct stations delegate to the profile; chamber stations return the
    /// chamber radius. Coordinates outside the modeled domain are rejected
    /// rather than silently clamped.
    pub fn local_radius_m(&self, x_m: f64) -> Result<f64, PortFlowError> {
        self.validate()?;
        if !x_m.is_finite() {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline-port axial coordinate must be finite, got {x_m}"
            )));
        }
        let x_min = -self.chamber_length_m;
        let x_max = self.duct_length_m + self.chamber_length_m;
        if x_m < x_min || x_m > x_max {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline-port axial coordinate {x_m} lies outside [{x_min}, {x_max}]"
            )));
        }
        if x_m >= 0.0 && x_m <= self.duct_length_m {
            Ok(self.profile.local_radius_m(x_m))
        } else {
            Ok(self.chamber_radius_m)
        }
    }

    /// Regenerate a mesh from scratch for this geometry's profile.
    ///
    /// Validation meshes only (never the optimisation loop): node lines
    /// conform to `r(x)` — duct columns span `±r(x)`, chamber columns split
    /// at the local wall — the profile knot stations become node lines so
    /// control features are resolved, and interior nodes carry the same
    /// deterministic hash jitter as [`build_slot_port_mesh`] (no-P1–P1 rule).
    /// Boundary edges are classified into [`MARKER_BOX_CAP`],
    /// [`MARKER_EXTERIOR_CAP`], and [`MARKER_WALL`] by position.
    pub fn build_validation_mesh(&self) -> Result<Mesh, PortFlowError> {
        self.validate()?;
        let lc = self.chamber_length_m;
        let ld = self.duct_length_m;
        let half_hc = self.chamber_radius_m;
        let cell = self.mesh_size_m;
        let wall = |x: f64| self.profile.local_radius_m(x);

        // Node lines with exact endpoints at x = 0, x = Ld (duct step faces)
        // and the caps, plus the profile knot stations so control features
        // are resolved even on coarse validation meshes.
        let mut x_lines = spaced_lines(-lc, 0.0, cell);
        x_lines.extend(spaced_lines(0.0, ld, cell).into_iter().skip(1));
        for &knot in self.profile.knots() {
            if knot > 0.0 && knot < ld {
                x_lines.push(knot);
            }
        }
        x_lines.extend(spaced_lines(ld, ld + lc, cell).into_iter().skip(1));
        x_lines.sort_by(f64::total_cmp);
        let min_gap = cell * 0.25;
        let protected =
            |x: f64| [(-lc), 0.0, ld, ld + lc].iter().any(|t| (x - t).abs() <= 1.0e-12);
        let mut deduped: Vec<f64> = Vec::with_capacity(x_lines.len());
        for x in x_lines {
            match deduped.last() {
                Some(&last) if x - last <= min_gap && !protected(x) => {}
                _ => deduped.push(x),
            }
        }
        let x_lines = deduped;

        let nx = x_lines.len();
        let mut points: Vec<(f64, f64)> = Vec::new();
        for (xi, &x) in x_lines.iter().enumerate() {
            let r_wall = wall(x);
            // Columns strictly inside the duct span only the local duct band;
            // the solid blocks above/below the duct carry no nodes at all.
            let duct_interior = x > 0.0 && x < ld;
            let duct_band = spaced_lines(-r_wall, r_wall, cell);
            let mut chamber_column = spaced_lines(-half_hc, -r_wall, cell);
            chamber_column.extend(duct_band.iter().skip(1).copied());
            chamber_column.extend(spaced_lines(r_wall, half_hc, cell).into_iter().skip(1));
            let ys: &[f64] = if duct_interior {
                &duct_band
            } else {
                &chamber_column
            };
            let ny = ys.len();
            let dx = match xi {
                0 => x_lines[1] - x,
                i if i + 1 == nx => x - x_lines[i - 1],
                i => (x - x_lines[i - 1]).min(x_lines[i + 1] - x),
            };
            for (yi, &y) in ys.iter().enumerate() {
                let dy = match yi {
                    0 => ys[1] - y,
                    i if i + 1 == ny => y - ys[i - 1],
                    i => (y - ys[i - 1]).min(ys[i + 1] - y),
                };
                let on_cap = xi == 0 || xi + 1 == nx;
                let on_column_end = yi == 0 || yi + 1 == ny;
                let on_step_face = (x == 0.0 || x == ld) && y.abs() >= r_wall;
                let boundary_aligned = on_cap || on_column_end || on_step_face;
                let (mut px, mut py) = (x, y);
                if !boundary_aligned {
                    // Deterministic hash jitter (±20% of a cell), same scheme
                    // as build_slot_port_mesh so validation meshes are
                    // reproducible but unstructured.
                    let h1 = (xi as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add((yi as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
                    let h2 = (yi as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add((xi as u64).wrapping_mul(0x1656_67B1_9E37_79F9));
                    px += ((h1 >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.4 * dx;
                    py += ((h2 >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.4 * dy;
                }
                points.push((px, py));
            }
        }

        let triangles = triangulate::triangulate_points_2d(&points);
        let mut mesh = Mesh::new(2);
        for &(x, y) in &points {
            mesh.add_node(Point::new_2d(x, y));
        }
        for tri in triangles {
            let cx = (points[tri[0]].0 + points[tri[1]].0 + points[tri[2]].0) / 3.0;
            let cy = (points[tri[0]].1 + points[tri[1]].1 + points[tri[2]].1) / 3.0;
            // Drop Delaunay triangles bridging the solid blocks above/below
            // the (possibly dipped) duct wall.
            let in_solid = cx > 0.0 && cx < ld && cy.abs() > wall(cx);
            if !in_solid {
                mesh.add_element(ElementType::Triangle, tri.to_vec());
            }
        }
        if mesh.elements.is_empty() {
            return Err(PortFlowError::Mesh(
                "triangulation produced no fluid elements".to_string(),
            ));
        }
        mesh.detect_boundaries();

        let tol = cell * 1.0e-6;
        mesh.set_boundary_condition(BoundaryType::Neumann, MARKER_BOX_CAP, |pts| {
            pts.iter().all(|p| (p.x + lc).abs() < tol)
        });
        mesh.set_boundary_condition(BoundaryType::Neumann, MARKER_EXTERIOR_CAP, |pts| {
            pts.iter().all(|p| (p.x - (ld + lc)).abs() < tol)
        });
        // Everything that is not a cap is a wall (same complement rule as the
        // slot builder: corner-adjacent wall edges share one node with a cap).
        mesh.set_boundary_condition(BoundaryType::Dirichlet, MARKER_WALL, |pts| {
            !pts.iter().all(|p| (p.x + lc).abs() < tol)
                && !pts.iter().all(|p| (p.x - (ld + lc)).abs() < tol)
        });

        let count = |marker: i32| {
            mesh.boundaries
                .iter()
                .filter(|b| b.marker == marker)
                .count()
        };
        if count(MARKER_BOX_CAP) == 0 || count(MARKER_EXTERIOR_CAP) == 0 || count(MARKER_WALL) == 0 {
            return Err(PortFlowError::Mesh(format!(
                "boundary classification failed: box_cap={}, exterior_cap={}, wall={}",
                count(MARKER_BOX_CAP),
                count(MARKER_EXTERIOR_CAP),
                count(MARKER_WALL),
            )));
        }
        Ok(mesh)
    }
}

/// Minimum signed triangle area in m² over a 2D mesh.
///
/// Negative means an inverted element; zero a degenerate one. Non-triangle
/// elements and dangling indices are ignored. Returns `+∞` when the mesh has
/// no triangles.
pub fn min_triangle_area_m2(mesh: &Mesh) -> f64 {
    let mut min = f64::INFINITY;
    for element in &mesh.elements {
        if !matches!(element.element_type, ElementType::Triangle) || element.nodes.len() < 3 {
            continue;
        }
        let (Some(a), Some(b), Some(c)) = (
            mesh.nodes.get(element.nodes[0]),
            mesh.nodes.get(element.nodes[1]),
            mesh.nodes.get(element.nodes[2]),
        ) else {
            continue;
        };
        let area = 0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y));
        min = min.min(area);
    }
    min
}

/// Relax interior duct nodes toward their neighbour average, synchronously.
///
/// Only interior (non-boundary) nodes with `x ∈ [0, Ld]` move; caps, chamber
/// walls, duct walls, and step faces stay bit-exact, and chamber interiors
/// are untouched so the deformation stays local to the duct. Out-of-range
/// element indices are skipped rather than panicking on a malformed mesh.
/// Smoothing stops early (reverting the offending pass) when a pass would
/// invert or degenerate an element.
fn laplacian_smooth_duct_interior(mesh: &mut Mesh, duct_length_m: f64, passes: usize) {
    if passes == 0 || mesh.nodes.is_empty() {
        return;
    }
    let count = mesh.nodes.len();
    let mut is_boundary = vec![false; count];
    for face in &mesh.boundaries {
        for &node in &face.nodes {
            if let Some(slot) = is_boundary.get_mut(node) {
                *slot = true;
            }
        }
    }
    let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); count];
    for element in &mesh.elements {
        let valid: Vec<usize> = element
            .nodes
            .iter()
            .copied()
            .filter(|&node| node < count)
            .collect();
        for (position, &a) in valid.iter().enumerate() {
            for &b in &valid[position + 1..] {
                neighbours[a].push(b);
                neighbours[b].push(a);
            }
        }
    }
    for list in &mut neighbours {
        list.sort_unstable();
        list.dedup();
    }
    let free: Vec<bool> = mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| {
            !is_boundary[index] && point.x >= 0.0 && point.x <= duct_length_m
        })
        .collect();
    for _ in 0..passes {
        let previous = mesh.nodes.clone();
        for (index, point) in mesh.nodes.iter_mut().enumerate() {
            if !free[index] || neighbours[index].is_empty() {
                continue;
            }
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            for &other in &neighbours[index] {
                sum_x += previous[other].x;
                sum_y += previous[other].y;
            }
            let degree = neighbours[index].len() as f64;
            point.x = sum_x / degree;
            point.y = sum_y / degree;
        }
        // Smoothing is a quality improvement, never a validity risk: a pass
        // that would invert or degenerate an element (possible next to very
        // steep walls) is reverted and smoothing stops early.
        let min_area = min_triangle_area_m2(mesh);
        if min_area <= 0.0 || !min_area.is_finite() {
            mesh.nodes = previous;
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::profile::{KnotSpacing, ProfileOptions, ShapeLimits};
    use super::*;

    const L: f64 = 0.1;
    const R0: f64 = 0.025;
    const RMIN: f64 = 0.015;
    const LC: f64 = 0.05;
    const RC: f64 = 0.04;
    const H: f64 = 0.005;

    fn reference_geometry(profile: SplinePortProfile) -> SplinePortGeometry {
        SplinePortGeometry {
            duct_length_m: L,
            bore_radius_m: R0,
            chamber_length_m: LC,
            chamber_radius_m: RC,
            mesh_size_m: H,
            profile,
        }
    }

    fn straight_geometry() -> SplinePortGeometry {
        let profile =
            SplinePortProfile::straight(5, L, R0, RMIN, ProfileOptions::default()).unwrap();
        reference_geometry(profile)
    }

    fn dipped_profile() -> SplinePortProfile {
        SplinePortProfile::from_normalized(5, L, R0, RMIN, &[0.25, 0.75], ProfileOptions::default())
            .unwrap()
    }

    fn baseline_mesh() -> Mesh {
        straight_geometry().build_baseline_mesh().unwrap()
    }

    fn marker_count(mesh: &Mesh, marker: i32) -> usize {
        mesh.boundaries
            .iter()
            .filter(|face| face.marker == marker)
            .count()
    }

    fn total_triangle_area_m2(mesh: &Mesh) -> f64 {
        let mut total = 0.0;
        for element in &mesh.elements {
            if !matches!(element.element_type, ElementType::Triangle) {
                continue;
            }
            let (a, b, c) = (
                mesh.nodes[element.nodes[0]],
                mesh.nodes[element.nodes[1]],
                mesh.nodes[element.nodes[2]],
            );
            total += 0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y));
        }
        total
    }

    #[test]
    fn baseline_mesh_matches_straight_slot_reference() {
        let geo = straight_geometry();
        let baseline = geo.build_baseline_mesh().unwrap();
        let slot = build_slot_port_mesh(&geo.baseline_slot_geometry()).unwrap();
        // Same deterministic builder and inputs: identical meshes.
        assert_eq!(baseline.nodes.len(), slot.nodes.len());
        assert_eq!(baseline.elements.len(), slot.elements.len());
        assert_eq!(baseline.boundaries.len(), slot.boundaries.len());
        for (a, b) in baseline.nodes.iter().zip(slot.nodes.iter()) {
            assert!(a.x == b.x && a.y == b.y && a.z == b.z);
        }
        assert!(marker_count(&baseline, MARKER_BOX_CAP) >= 1);
        assert!(marker_count(&baseline, MARKER_EXTERIOR_CAP) >= 1);
        assert!(marker_count(&baseline, MARKER_WALL) >= 8);
        assert!(min_triangle_area_m2(&baseline) > 0.0);
    }

    #[test]
    fn straight_profile_morph_returns_baseline_exactly() {
        let geo = straight_geometry();
        let baseline = geo.build_baseline_mesh().unwrap();
        let morphed = geo
            .morph_baseline_mesh(&baseline, DEFAULT_MORPH_SMOOTHING_PASSES)
            .unwrap();
        assert_eq!(morphed.nodes.len(), baseline.nodes.len());
        for (a, b) in morphed.nodes.iter().zip(baseline.nodes.iter()) {
            assert!(
                a.x == b.x && a.y == b.y && a.z == b.z,
                "u = 0 morph must be the identity map"
            );
        }
        let markers = |mesh: &Mesh| {
            mesh.boundaries
                .iter()
                .map(|face| face.marker)
                .collect::<Vec<_>>()
        };
        assert_eq!(markers(&morphed), markers(&baseline));
    }

    #[test]
    fn morph_without_smoothing_applies_pure_radial_scaling() {
        let profile = dipped_profile();
        let geo = reference_geometry(profile.clone());
        let baseline = baseline_mesh();
        let morphed = geo.morph_baseline_mesh(&baseline, 0).unwrap();
        for (moved, base) in morphed.nodes.iter().zip(baseline.nodes.iter()) {
            // Same operation order as the morph (`y *= r / R0`) for bit-exactness.
            let expected = base.y * (profile.local_radius_m(base.x) / R0);
            assert!(
                moved.x == base.x,
                "morph must never move nodes axially"
            );
            assert!(
                moved.y == expected,
                "node at x = {} scales by r(x)/R0",
                base.x
            );
        }
        assert!(min_triangle_area_m2(&morphed) > 0.0);
    }

    #[test]
    fn smoothed_morph_keeps_boundary_exact_and_chambers_identical() {
        let profile = dipped_profile();
        let geo = reference_geometry(profile);
        let baseline = baseline_mesh();
        let pure = geo.morph_baseline_mesh(&baseline, 0).unwrap();
        let smoothed = geo
            .morph_baseline_mesh(&baseline, DEFAULT_MORPH_SMOOTHING_PASSES)
            .unwrap();
        let mut is_boundary = vec![false; baseline.nodes.len()];
        for face in &baseline.boundaries {
            for &node in &face.nodes {
                is_boundary[node] = true;
            }
        }
        for (index, ((smooth, unsmoothed), base)) in smoothed
            .nodes
            .iter()
            .zip(pure.nodes.iter())
            .zip(baseline.nodes.iter())
            .enumerate()
        {
            if is_boundary[index] {
                assert!(
                    smooth.x == unsmoothed.x && smooth.y == unsmoothed.y,
                    "smoothing moved boundary node {index}"
                );
            }
            if base.x < 0.0 || base.x > L {
                assert!(
                    smooth.x == base.x && smooth.y == base.y,
                    "chamber node {index} must stay bit-identical"
                );
            }
        }
        assert!(min_triangle_area_m2(&smoothed) > 0.0);
    }

    #[test]
    fn morphed_wall_nodes_sit_on_profile_with_exact_ends() {
        let profile = dipped_profile();
        let geo = reference_geometry(profile.clone());
        let baseline = baseline_mesh();
        let morphed = geo
            .morph_baseline_mesh(&baseline, DEFAULT_MORPH_SMOOTHING_PASSES)
            .unwrap();
        // Duct wall edges lie on ±r(x) to floating-point roundoff.
        for face in morphed
            .boundaries
            .iter()
            .filter(|face| face.marker == MARKER_WALL)
        {
            for &node in &face.nodes {
                let point = morphed.nodes[node];
                if point.x > 0.0 && point.x < L {
                    let wall = profile.local_radius_m(point.x);
                    assert!(
                        (point.y.abs() - wall).abs() < 1.0e-12,
                        "wall node at x = {} leaves r(x) = {wall}",
                        point.x
                    );
                }
            }
        }
        // Step planes and caps are scaled by exactly 1 and never smoothed.
        let mut is_boundary = vec![false; baseline.nodes.len()];
        for face in &baseline.boundaries {
            for &node in &face.nodes {
                is_boundary[node] = true;
            }
        }
        for (index, (moved, base)) in morphed
            .nodes
            .iter()
            .zip(baseline.nodes.iter())
            .enumerate()
        {
            if is_boundary[index] && (base.x == 0.0 || base.x == L) {
                assert!(
                    moved.x == base.x && moved.y == base.y,
                    "step-plane boundary node {index} must stay exact"
                );
            }
        }
        // Exact ends: step corners sit at exactly ±R0.
        for end in [0.0, L] {
            for sign in [-1.0, 1.0] {
                let found = morphed.nodes.iter().any(|point| {
                    point.x == end && point.y == sign * R0
                });
                assert!(found, "missing exact step corner ({end}, {})", sign * R0);
            }
        }
    }

    #[test]
    fn morphed_mesh_preserves_markers_and_shrinks_inward() {
        let geo = reference_geometry(dipped_profile());
        let baseline = baseline_mesh();
        let morphed = geo
            .morph_baseline_mesh(&baseline, DEFAULT_MORPH_SMOOTHING_PASSES)
            .unwrap();
        for marker in [MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL] {
            assert_eq!(
                marker_count(&morphed, marker),
                marker_count(&baseline, marker),
                "morph must preserve marker-{marker} classification"
            );
        }
        assert!(morphed.boundaries.iter().all(|face| face.marker != 0));
        assert!(min_triangle_area_m2(&morphed) > 0.0);
        // Inward-only wall motion removes fluid area from the duct.
        assert!(
            total_triangle_area_m2(&morphed) < total_triangle_area_m2(&baseline),
            "dipped morph must shrink the fluid area"
        );
    }

    #[test]
    fn small_control_counts_morph_under_default_limits() {
        let baseline = baseline_mesh();
        for n in [3, 4] {
            let corners = 1usize << (n - 3);
            for bits in 0..corners {
                let u: Vec<f64> = (0..n - 3)
                    .map(|i| f64::from((bits >> i) & 1 == 1))
                    .collect();
                let profile = SplinePortProfile::from_normalized(
                    n,
                    L,
                    R0,
                    RMIN,
                    &u,
                    ProfileOptions::default(),
                )
                .unwrap_or_else(|error| panic!("N = {n} corner {u:?} must construct: {error}"));
                let geo = reference_geometry(profile);
                let morphed = geo
                    .morph_baseline_mesh(&baseline, DEFAULT_MORPH_SMOOTHING_PASSES)
                    .unwrap();
                assert!(
                    min_triangle_area_m2(&morphed) > 0.0,
                    "N = {n} corner {u:?} produced an inverted element"
                );
                for marker in [MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL] {
                    assert_eq!(marker_count(&morphed, marker), marker_count(&baseline, marker));
                }
            }
        }
    }

    #[test]
    fn default_shape_limits_reject_full_depth_n5_corners() {
        // A full-span dip over one N = 5 interval peaks at 1.5·span/h = 0.6,
        // above the default slope limit 0.5: the Step 6 optimiser must treat
        // these corners as ordinary infeasible candidates.
        for u in [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]] {
            assert!(
                SplinePortProfile::from_normalized(5, L, R0, RMIN, &u, ProfileOptions::default())
                    .is_err(),
                "corner {u:?} should exceed the default slope limit"
            );
        }
        assert!(
            SplinePortProfile::from_normalized(5, L, R0, RMIN, &[0.0, 0.0], ProfileOptions::default())
                .is_ok()
        );
    }

    #[test]
    fn full_admissible_box_corners_morph_without_inversions() {
        // The whole u ∈ [0, 1]^(N-3) box corners, N = 5: relaxed slope limit
        // admits the full-depth dips (see the rejection test above) while the
        // default curvature budget still holds.
        let options = ProfileOptions {
            limits: ShapeLimits {
                slope_max: 1.0,
                ..ShapeLimits::default()
            },
            ..ProfileOptions::default()
        };
        let baseline = baseline_mesh();
        for bits in 0..4usize {
            let u = [f64::from(bits & 1 == 1), f64::from(bits & 2 == 2)];
            let profile =
                SplinePortProfile::from_normalized(5, L, R0, RMIN, &u, options)
                    .unwrap_or_else(|error| panic!("corner {u:?} must construct: {error}"));
            let geo = reference_geometry(profile);
            let morphed = geo
                .morph_baseline_mesh(&baseline, DEFAULT_MORPH_SMOOTHING_PASSES)
                .unwrap();
            assert!(
                min_triangle_area_m2(&morphed) > 0.0,
                "corner {u:?} produced an inverted element"
            );
            for marker in [MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL] {
                assert_eq!(marker_count(&morphed, marker), marker_count(&baseline, marker));
            }
        }
    }

    #[test]
    fn extreme_eight_point_corners_morph_cleanly() {
        // 32 corners × {uniform, Chebyshev} under limits that admit the
        // steepest legal walls (Chebyshev end intervals concentrate the
        // full-span dip): morphing robustness, not the optimiser envelope.
        let relaxed = ShapeLimits {
            slope_max: 5.0,
            curvature_budget_m3: 1.0e5,
        };
        let baseline = baseline_mesh();
        for spacing in [KnotSpacing::Uniform, KnotSpacing::Chebyshev] {
            let options = ProfileOptions {
                spacing,
                limits: relaxed,
                ..ProfileOptions::default()
            };
            for bits in 0..32usize {
                let u: Vec<f64> = (0..5)
                    .map(|i| f64::from((bits >> i) & 1 == 1))
                    .collect();
                let profile = SplinePortProfile::from_normalized(8, L, R0, RMIN, &u, options)
                    .unwrap_or_else(|error| {
                        panic!("{spacing:?} corner {u:?} must construct: {error}")
                    });
                let geo = reference_geometry(profile);
                // Pure radial scaling never inverts (orientation-preserving
                // map); heavy smoothing stays valid only through the
                // revert-and-stop guard (unguarded passes invert here).
                for passes in [0, DEFAULT_MORPH_SMOOTHING_PASSES, 50] {
                    let morphed = geo.morph_baseline_mesh(&baseline, passes).unwrap();
                    assert!(
                        min_triangle_area_m2(&morphed) > 0.0,
                        "{spacing:?} corner {u:?} with {passes} passes produced an inverted element"
                    );
                }
            }
        }
    }

    #[test]
    fn validation_mesh_from_scratch_is_classified_and_conforming() {
        // Deepest N = 5 corner (full-throat dip, relaxed slope limit): the
        // from-scratch builder must handle it as well as gentle profiles.
        let deep_options = ProfileOptions {
            limits: ShapeLimits {
                slope_max: 1.0,
                ..ShapeLimits::default()
            },
            ..ProfileOptions::default()
        };
        let deep_corner =
            SplinePortProfile::from_normalized(5, L, R0, RMIN, &[1.0, 1.0], deep_options).unwrap();
        for profile in [
            dipped_profile(),
            straight_geometry().profile.clone(),
            deep_corner,
        ] {
            let geo = reference_geometry(profile.clone());
            let mesh = geo.build_validation_mesh().unwrap();
            assert!(mesh.boundaries.iter().all(|face| face.marker != 0));
            assert!(marker_count(&mesh, MARKER_BOX_CAP) >= 1);
            assert!(marker_count(&mesh, MARKER_EXTERIOR_CAP) >= 1);
            assert!(marker_count(&mesh, MARKER_WALL) >= 8);
            assert!(
                min_triangle_area_m2(&mesh) > 0.0,
                "validation mesh has an inverted element"
            );
            // Duct wall nodes sit exactly on the generated ±r(x) lines.
            for face in mesh
                .boundaries
                .iter()
                .filter(|face| face.marker == MARKER_WALL)
            {
                for &node in &face.nodes {
                    let point = mesh.nodes[node];
                    if point.x > 0.0 && point.x < L {
                        let wall = profile.local_radius_m(point.x);
                        assert!(
                            (point.y.abs() - wall).abs() < 1.0e-12,
                            "validation wall node at x = {} leaves r(x) = {wall}",
                            point.x
                        );
                    }
                }
            }
            // Caps sit exactly on their planes.
            for face in mesh
                .boundaries
                .iter()
                .filter(|face| face.marker == MARKER_BOX_CAP)
            {
                assert!(
                    face.nodes
                        .iter()
                        .all(|&node| (mesh.nodes[node].x + LC).abs() < 1.0e-12)
                );
            }
            for face in mesh
                .boundaries
                .iter()
                .filter(|face| face.marker == MARKER_EXTERIOR_CAP)
            {
                assert!(
                    face.nodes
                        .iter()
                        .all(|&node| (mesh.nodes[node].x - (L + LC)).abs() < 1.0e-12)
                );
            }
        }
    }

    #[test]
    fn local_radius_delegates_to_profile_and_chambers() {
        let profile = dipped_profile();
        let geo = reference_geometry(profile.clone());
        for station in [0.0, 0.017, 0.05, 0.083, L] {
            assert!(
                (geo.local_radius_m(station).unwrap() - profile.local_radius_m(station)).abs()
                    < 1.0e-15
            );
        }
        assert!((geo.local_radius_m(-LC / 2.0).unwrap() - RC).abs() < 1.0e-15);
        assert!((geo.local_radius_m(L + LC / 2.0).unwrap() - RC).abs() < 1.0e-15);
        for bad in [-LC - 1.0e-6, L + LC + 1.0e-6, f64::NAN, f64::INFINITY] {
            assert!(
                geo.local_radius_m(bad).is_err(),
                "station {bad} should be rejected"
            );
        }
    }

    #[test]
    fn geometry_validation_rejects_mismatched_configs() {
        let profile = dipped_profile();
        let good = reference_geometry(profile.clone());
        assert!(good.build_baseline_mesh().is_ok());

        let long_profile =
            SplinePortProfile::from_normalized(5, 2.0 * L, R0, RMIN, &[0.2, 0.3], ProfileOptions::default())
                .unwrap();
        assert!(
            SplinePortGeometry {
                profile: long_profile,
                ..good.clone()
            }
            .build_baseline_mesh()
            .is_err()
        );

        // Gentle dip: the wider bore doubles the radius span, so the same
        // normalised depth would exceed the default slope limit.
        let wide_profile =
            SplinePortProfile::from_normalized(5, L, 2.0 * R0, RMIN, &[0.05, 0.05], ProfileOptions::default())
                .unwrap();
        assert!(
            SplinePortGeometry {
                profile: wide_profile,
                ..good.clone()
            }
            .build_baseline_mesh()
            .is_err()
        );

        for bad in [
            SplinePortGeometry {
                chamber_radius_m: R0,
                ..good.clone()
            },
            SplinePortGeometry {
                mesh_size_m: 0.0,
                ..good.clone()
            },
            SplinePortGeometry {
                duct_length_m: f64::NAN,
                ..good.clone()
            },
        ] {
            assert!(bad.build_baseline_mesh().is_err());
        }
    }

    #[test]
    fn morph_rejects_foreign_baselines() {
        let geo = reference_geometry(dipped_profile());
        // Baseline from a different chamber size: extents mismatch.
        let mut foreign_dims = geo.baseline_slot_geometry();
        foreign_dims.chamber_height_m = 2.0 * (RC + 0.01);
        let foreign = build_slot_port_mesh(&foreign_dims).unwrap();
        assert!(geo.morph_baseline_mesh(&foreign, 0).is_err());
        // Wrong dimension entirely.
        assert!(geo.morph_baseline_mesh(&Mesh::new(3), 0).is_err());
    }

    #[test]
    fn spline_geometry_survives_serde_round_trip() {
        let geo = reference_geometry(dipped_profile());
        let json = serde_json::to_string(&geo).unwrap();
        let back: SplinePortGeometry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, geo);
    }
}
