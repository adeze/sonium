//! Simple 2D C-core magnetic circuit mesh generator.
//!
//! The core [`Mesh`](super::Mesh) type does not carry element material IDs, so
//! this generator returns a sidecar `element_regions` vector aligned with
//! `mesh.elements`.

use super::{ElementType, Mesh, Point};

/// Element region tag for generated C-core magnetic circuits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CCoreRegion {
    /// Surrounding air.
    Air = 0,
    /// Soft magnetic C-core material.
    Core = 1,
    /// Coil window/current-carrying region.
    Coil = 2,
    /// Air gap between pole tips.
    Gap = 3,
}

/// Rectilinear C-core mesh configuration.
#[derive(Debug, Clone, Copy)]
pub struct CCoreMeshConfig {
    /// Outer width of the generated domain.
    pub width: f64,
    /// Outer height of the generated domain.
    pub height: f64,
    /// Thickness of the magnetic core legs/yokes.
    pub core_thickness: f64,
    /// Gap opening on the right side.
    pub gap_height: f64,
    /// Number of cells along x.
    pub nx: usize,
    /// Number of cells along y.
    pub ny: usize,
}

impl Default for CCoreMeshConfig {
    fn default() -> Self {
        Self {
            width: 1.0,
            height: 1.0,
            core_thickness: 0.2,
            gap_height: 0.25,
            nx: 24,
            ny: 24,
        }
    }
}

/// C-core mesh plus element region tags.
#[derive(Debug, Clone)]
pub struct CCoreMesh {
    pub mesh: Mesh,
    /// Region tag aligned with `mesh.elements`.
    pub element_regions: Vec<CCoreRegion>,
}

/// Generate a simple triangular C-core magnetic circuit mesh.
///
/// Boundary markers are inherited from the rectangular domain:
/// `1 = left`, `2 = right`, `3 = bottom`, `4 = top`.
pub fn c_core_mesh_triangles(config: CCoreMeshConfig) -> CCoreMesh {
    assert!(config.width > 0.0);
    assert!(config.height > 0.0);
    assert!(config.core_thickness > 0.0);
    assert!(config.gap_height > 0.0);
    assert!(config.nx > 0 && config.ny > 0);

    let mut mesh = Mesh::new(2);
    let dx = config.width / config.nx as f64;
    let dy = config.height / config.ny as f64;

    for j in 0..=config.ny {
        for i in 0..=config.nx {
            mesh.add_node(Point::new_2d(i as f64 * dx, j as f64 * dy));
        }
    }

    let node = |i: usize, j: usize| -> usize { j * (config.nx + 1) + i };
    let mut regions = Vec::with_capacity(config.nx * config.ny * 2);

    for j in 0..config.ny {
        for i in 0..config.nx {
            let n00 = node(i, j);
            let n10 = node(i + 1, j);
            let n01 = node(i, j + 1);
            let n11 = node(i + 1, j + 1);
            let cx = (i as f64 + 0.5) * dx;
            let cy = (j as f64 + 0.5) * dy;
            let region = classify_c_core_point(cx, cy, config);

            mesh.add_element(ElementType::Triangle, vec![n00, n10, n11]);
            regions.push(region);
            mesh.add_element(ElementType::Triangle, vec![n00, n11, n01]);
            regions.push(region);
        }
    }

    mesh.detect_boundaries();
    for boundary in &mut mesh.boundaries {
        let p0 = mesh.nodes[boundary.nodes[0]];
        let p1 = mesh.nodes[boundary.nodes[1]];
        let mx = 0.5 * (p0.x + p1.x);
        let my = 0.5 * (p0.y + p1.y);
        boundary.marker = if mx <= 1e-12 {
            1
        } else if (mx - config.width).abs() <= 1e-12 {
            2
        } else if my <= 1e-12 {
            3
        } else {
            4
        };
    }

    CCoreMesh {
        mesh,
        element_regions: regions,
    }
}

fn classify_c_core_point(x: f64, y: f64, config: CCoreMeshConfig) -> CCoreRegion {
    let t = config.core_thickness;
    let h = config.height;
    let w = config.width;
    let gap_min = 0.5 * (h - config.gap_height);
    let gap_max = 0.5 * (h + config.gap_height);

    let in_left_leg = x <= t;
    let in_top_yoke = y >= h - t;
    let in_bottom_yoke = y <= t;
    let in_right_pole = x >= w - t && (y <= gap_min || y >= gap_max);
    let in_core = in_left_leg || in_top_yoke || in_bottom_yoke || in_right_pole;
    if in_core {
        return CCoreRegion::Core;
    }

    let in_gap = x >= w - t && y > gap_min && y < gap_max;
    if in_gap {
        return CCoreRegion::Gap;
    }

    let in_coil_window = x > t && x < w - t && y > t && y < h - t;
    if in_coil_window {
        CCoreRegion::Coil
    } else {
        CCoreRegion::Air
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_core_mesh_has_regions() {
        let generated = c_core_mesh_triangles(CCoreMeshConfig {
            nx: 8,
            ny: 8,
            ..CCoreMeshConfig::default()
        });

        assert_eq!(
            generated.element_regions.len(),
            generated.mesh.elements.len()
        );
        assert!(generated.element_regions.contains(&CCoreRegion::Core));
        assert!(generated.element_regions.contains(&CCoreRegion::Coil));
        assert!(generated.element_regions.contains(&CCoreRegion::Gap));
    }
}
