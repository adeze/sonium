//! 2D slot-port (chamber–duct–chamber) geometry and mesh generation.
//!
//! Layout: box chamber on the left (`x ∈ [−Lc, 0]`), duct in the middle
//! (`x ∈ [0, Ld]`, `y ∈ [−h/2, h/2]`), exterior chamber on the right
//! (`x ∈ [Ld, Ld+Lc]`). The chambers resolve the duct end effects
//! (contraction/expansion at the terminations), so the chamber height must
//! exceed the duct height. [`MARKER_BOX_CAP`] classifies the `x = −Lc` face,
//! [`MARKER_EXTERIOR_CAP`] the `x = Ld+Lc` face, and [`MARKER_WALL`] every
//! other boundary edge (chamber walls, duct walls, and the step faces at the
//! duct ends).

use serde::{Deserialize, Serialize};
use sonium_fem::mesh::{BoundaryType, ElementType, Mesh, Point, triangulate};

use super::error::PortFlowError;

/// No-slip wall marker: chamber top/bottom walls, duct walls, duct step faces.
pub const MARKER_WALL: i32 = 1;
/// Box-side cap at `x = −chamber_length_m` (termination towards the enclosure).
pub const MARKER_BOX_CAP: i32 = 2;
/// Exterior-side cap at `x = duct_length_m + chamber_length_m`.
pub const MARKER_EXTERIOR_CAP: i32 = 3;

/// Geometry of a 2D slot port with end-effect chambers at both terminations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SlotPortGeometry {
    pub duct_length_m: f64,
    pub duct_height_m: f64,
    pub chamber_length_m: f64,
    pub chamber_height_m: f64,
    pub mesh_size_m: f64,
}

impl SlotPortGeometry {
    fn validate(self) -> Result<(), PortFlowError> {
        for (name, value) in [
            ("duct_length_m", self.duct_length_m),
            ("duct_height_m", self.duct_height_m),
            ("chamber_length_m", self.chamber_length_m),
            ("chamber_height_m", self.chamber_height_m),
            ("mesh_size_m", self.mesh_size_m),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(PortFlowError::InvalidConfig(format!(
                    "{name} must be positive and finite, got {value}"
                )));
            }
        }
        if self.chamber_height_m <= self.duct_height_m {
            return Err(PortFlowError::InvalidConfig(format!(
                "chamber_height_m ({}) must exceed duct_height_m ({}) to resolve end effects",
                self.chamber_height_m, self.duct_height_m
            )));
        }
        Ok(())
    }
}

/// Uniformly spaced node lines over `[a, b]` with cell size at most `cell`;
/// endpoints land exactly on `a` and `b` so boundary-conforming lines stay exact.
fn spaced_lines(a: f64, b: f64, cell: f64) -> Vec<f64> {
    let n = ((b - a) / cell).ceil().max(1.0) as usize;
    (0..=n)
        .map(|i| {
            if i == n {
                b
            } else {
                a + (b - a) * i as f64 / n as f64
            }
        })
        .collect()
}

/// Build an unstructured triangular mesh of the chamber–duct–chamber domain.
///
/// Node lines conform to the duct walls (`y = ±h/2`) and the chamber extents,
/// interior nodes are deterministically hash-jittered (same scheme as the
/// `marked_channel_mesh` helper in sonium-fem's Navier–Stokes tests), and the
/// point set is Delaunay-triangulated. Triangles whose centroid falls in the
/// solid blocks above/below the duct are dropped; the jittered connectivity
/// keeps downstream P1–P1 saddle problems free of the checkerboard null mode
/// that perfectly structured grids exhibit. Boundary edges are classified by
/// position into [`MARKER_BOX_CAP`], [`MARKER_EXTERIOR_CAP`], and
/// [`MARKER_WALL`].
pub fn build_slot_port_mesh(geo: &SlotPortGeometry) -> Result<Mesh, PortFlowError> {
    geo.validate()?;
    let lc = geo.chamber_length_m;
    let ld = geo.duct_length_m;
    let half_hc = geo.chamber_height_m / 2.0;
    let half_hd = geo.duct_height_m / 2.0;
    let cell = geo.mesh_size_m;

    // Node lines with exact endpoints at x = 0, x = Ld (duct step faces),
    // x = −Lc / x = Ld+Lc (caps), and y = ±h/2 (duct walls).
    let mut x_lines = spaced_lines(-lc, 0.0, cell);
    x_lines.extend(spaced_lines(0.0, ld, cell).into_iter().skip(1));
    x_lines.extend(spaced_lines(ld, ld + lc, cell).into_iter().skip(1));

    let duct_band = spaced_lines(-half_hd, half_hd, cell);
    let mut chamber_column = spaced_lines(-half_hc, -half_hd, cell);
    chamber_column.extend(duct_band.iter().skip(1).copied());
    chamber_column.extend(spaced_lines(half_hd, half_hc, cell).into_iter().skip(1));

    let nx = x_lines.len();
    let mut points: Vec<(f64, f64)> = Vec::new();
    for (xi, &x) in x_lines.iter().enumerate() {
        // Columns strictly inside the duct span only the duct band; the solid
        // blocks above/below the duct carry no nodes at all.
        let duct_interior = x > 0.0 && x < ld;
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
            let on_step_face = (x == 0.0 || x == ld) && y.abs() >= half_hd;
            let boundary_aligned = on_cap || on_column_end || on_step_face;
            let (mut px, mut py) = (x, y);
            if !boundary_aligned {
                // Deterministic hash jitter (±20% of a cell) so the mesh is
                // reproducible but unstructured. Same scheme as the
                // marked_channel_mesh helper in sonium-fem's N-S tests.
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
        // Drop Delaunay triangles bridging the solid blocks above/below the
        // duct (the convex hull fills those concave voids).
        let in_solid = cx > 0.0 && cx < ld && cy.abs() > half_hd;
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
    // Everything that is not a cap is a wall. Corner-adjacent wall edges share
    // one node with the cap line, so "not all points on a cap" is the correct
    // complement (an edge never straddles two cap planes).
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_geometry() -> SlotPortGeometry {
        SlotPortGeometry {
            duct_length_m: 0.1,
            duct_height_m: 0.02,
            chamber_length_m: 0.05,
            chamber_height_m: 0.08,
            mesh_size_m: 0.005,
        }
    }

    #[test]
    fn slot_port_mesh_has_three_classified_boundaries() {
        let geo = test_geometry();
        let mesh = build_slot_port_mesh(&geo).unwrap();
        assert_eq!(mesh.dimension, 2);
        let count = |m: i32| mesh.boundaries.iter().filter(|b| b.marker == m).count();
        assert!(count(MARKER_BOX_CAP) >= 1);
        assert!(count(MARKER_EXTERIOR_CAP) >= 1);
        assert!(count(MARKER_WALL) >= 8);
        // Duct end effects need chamber height > duct height (sanity of layout):
        let max_y = mesh
            .nodes
            .iter()
            .map(|p| p.y)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((max_y - 0.04).abs() < 1e-12);
    }

    #[test]
    fn slot_port_mesh_is_unstructured_with_conforming_walls() {
        let geo = test_geometry();
        let mesh = build_slot_port_mesh(&geo).unwrap();
        let ld = geo.duct_length_m;
        let lc = geo.chamber_length_m;
        let half_hc = geo.chamber_height_m / 2.0;
        let half_hd = geo.duct_height_m / 2.0;
        let tol = 1.0e-12;

        // Every boundary edge is classified (no leftover marker 0).
        assert!(mesh.boundaries.iter().all(|b| b.marker != 0));

        // Caps sit exactly on their planes.
        for b in mesh
            .boundaries
            .iter()
            .filter(|b| b.marker == MARKER_BOX_CAP)
        {
            assert!(b.nodes.iter().all(|&i| (mesh.nodes[i].x + lc).abs() < tol));
        }
        for b in mesh
            .boundaries
            .iter()
            .filter(|b| b.marker == MARKER_EXTERIOR_CAP)
        {
            assert!(
                b.nodes
                    .iter()
                    .all(|&i| (mesh.nodes[i].x - (ld + lc)).abs() < tol)
            );
        }

        // Wall edges conform exactly to the chamber walls, duct walls, or the
        // step faces at the duct ends: the triangulation respects the no-slip
        // geometry despite the interior jitter.
        let on_wall_line = |p: &Point| {
            let on_chamber_wall = (p.y.abs() - half_hc).abs() < tol;
            let on_duct_wall = (p.y.abs() - half_hd).abs() < tol && p.x > -tol && p.x < ld + tol;
            let on_step_face =
                (p.x.abs() < tol || (p.x - ld).abs() < tol) && p.y.abs() > half_hd - tol;
            on_chamber_wall || on_duct_wall || on_step_face
        };
        for b in mesh.boundaries.iter().filter(|b| b.marker == MARKER_WALL) {
            assert!(
                b.nodes.iter().all(|&i| on_wall_line(&mesh.nodes[i])),
                "wall edge {:?} leaves the wall lines",
                b.nodes
            );
        }

        // Connectivity is genuinely unstructured: at least one interior node
        // sits off every node line (the deterministic jitter fired).
        let x_lines: Vec<f64> = mesh
            .boundaries
            .iter()
            .filter(|b| b.marker == MARKER_WALL)
            .flat_map(|b| b.nodes.iter().map(|&i| mesh.nodes[i].x))
            .collect();
        let jittered = mesh.nodes.iter().any(|p| {
            p.x > -lc + tol
                && p.x < ld + lc - tol
                && p.y.abs() < half_hc - tol
                && !x_lines.iter().any(|&x| (x - p.x).abs() < tol)
        });
        assert!(jittered, "no interior node was jittered off the grid");

        // No orphan nodes: every node is referenced by at least one element.
        let mut referenced = vec![false; mesh.nodes.len()];
        for element in &mesh.elements {
            for &i in &element.nodes {
                referenced[i] = true;
            }
        }
        assert!(referenced.iter().all(|&r| r));
    }

    #[test]
    fn slot_port_geometry_rejects_degenerate_configs() {
        let mut geo = test_geometry();
        geo.chamber_height_m = geo.duct_height_m;
        assert!(matches!(
            build_slot_port_mesh(&geo),
            Err(PortFlowError::InvalidConfig(_))
        ));
        let geo = SlotPortGeometry {
            mesh_size_m: -0.005,
            ..test_geometry()
        };
        assert!(matches!(
            build_slot_port_mesh(&geo),
            Err(PortFlowError::InvalidConfig(_))
        ));
    }
}
