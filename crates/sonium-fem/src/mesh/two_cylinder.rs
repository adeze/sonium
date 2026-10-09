//! Simple two-cylinder planar mesh generator for electrostatic QA cases.

use crate::mesh::{BoundaryType, ElementType, Mesh, Point};

/// Generate a rectangular triangular mesh with two circular holes.
///
/// Boundary markers: `1 = outer box`, `2 = first cylinder`, `3 = second cylinder`.
/// Elements whose centroids fall inside either cylinder are removed. This is a
/// structured-grid generator intended for analytical QA smoke tests rather than
/// production geometry.
#[allow(clippy::too_many_arguments)]
pub fn two_cylinder_mesh_triangles(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    c1: (f64, f64, f64),
    c2: (f64, f64, f64),
    nx: usize,
    ny: usize,
) -> Mesh {
    assert!(x_max > x_min && y_max > y_min);
    assert!(c1.2 > 0.0 && c2.2 > 0.0);
    assert!(nx > 0 && ny > 0);

    let mut mesh = Mesh::new(2);
    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;
    for j in 0..=ny {
        for i in 0..=nx {
            mesh.add_node(Point::new_2d(x_min + i as f64 * dx, y_min + j as f64 * dy));
        }
    }
    let node = |i: usize, j: usize| -> usize { j * (nx + 1) + i };
    let inside = |x: f64, y: f64, c: (f64, f64, f64)| {
        let dx = x - c.0;
        let dy = y - c.1;
        dx * dx + dy * dy < c.2 * c.2
    };

    for j in 0..ny {
        for i in 0..nx {
            let tris = [
                [node(i, j), node(i + 1, j), node(i + 1, j + 1)],
                [node(i, j), node(i + 1, j + 1), node(i, j + 1)],
            ];
            for tri in tris {
                let x = (mesh.nodes[tri[0]].x + mesh.nodes[tri[1]].x + mesh.nodes[tri[2]].x) / 3.0;
                let y = (mesh.nodes[tri[0]].y + mesh.nodes[tri[1]].y + mesh.nodes[tri[2]].y) / 3.0;
                if !inside(x, y, c1) && !inside(x, y, c2) {
                    mesh.add_element(ElementType::Triangle, tri.to_vec());
                }
            }
        }
    }

    mesh.detect_boundaries();
    for boundary in &mut mesh.boundaries {
        let (mut x, mut y) = (0.0, 0.0);
        for &n in &boundary.nodes {
            x += mesh.nodes[n].x;
            y += mesh.nodes[n].y;
        }
        x /= boundary.nodes.len() as f64;
        y /= boundary.nodes.len() as f64;
        let d1 = ((x - c1.0).powi(2) + (y - c1.1).powi(2)).sqrt();
        let d2 = ((x - c2.0).powi(2) + (y - c2.1).powi(2)).sqrt();
        boundary.boundary_type = BoundaryType::Dirichlet;
        boundary.marker = if (d1 - c1.2).abs() < 1.5 * dx.max(dy) {
            2
        } else if (d2 - c2.2).abs() < 1.5 * dx.max(dy) {
            3
        } else {
            1
        };
    }
    mesh
}
