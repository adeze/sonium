//! Simple surface mesh generators for shell QA problems.

use crate::mesh::{BoundaryType, ElementType, Mesh, Point};
use std::f64::consts::PI;

/// Triangular disk surface mesh in the `z = 0` plane.
///
/// Boundary marker `1` marks the outer circular rim.
pub fn circular_shell_mesh_triangles(radius: f64, n_radial: usize, n_theta: usize) -> Mesh {
    assert!(radius > 0.0);
    assert!(n_radial > 0);
    assert!(n_theta >= 3);

    let mut mesh = Mesh::new(3);
    let center = mesh.add_node(Point::new_3d(0.0, 0.0, 0.0));
    let mut rings: Vec<Vec<usize>> = Vec::new();
    for ir in 1..=n_radial {
        let r = radius * ir as f64 / n_radial as f64;
        let mut ring = Vec::with_capacity(n_theta);
        for it in 0..n_theta {
            let t = 2.0 * PI * it as f64 / n_theta as f64;
            ring.push(mesh.add_node(Point::new_3d(r * t.cos(), r * t.sin(), 0.0)));
        }
        rings.push(ring);
    }

    for it in 0..n_theta {
        let next = (it + 1) % n_theta;
        mesh.add_element(
            ElementType::Triangle,
            vec![center, rings[0][it], rings[0][next]],
        );
    }
    for ir in 1..n_radial {
        let inner = &rings[ir - 1];
        let outer = &rings[ir];
        for it in 0..n_theta {
            let next = (it + 1) % n_theta;
            mesh.add_element(
                ElementType::Triangle,
                vec![inner[it], outer[it], outer[next]],
            );
            mesh.add_element(
                ElementType::Triangle,
                vec![inner[it], outer[next], inner[next]],
            );
        }
    }

    let outer = &rings[n_radial - 1];
    for it in 0..n_theta {
        mesh.add_boundary(
            vec![outer[it], outer[(it + 1) % n_theta]],
            BoundaryType::Dirichlet,
            1,
            0,
            it,
        );
    }
    mesh
}

/// Open cylindrical shell surface mesh.
///
/// Boundary markers: `1 = z_min`, `2 = z_max`.
pub fn cylindrical_shell_mesh_triangles(
    radius: f64,
    length: f64,
    n_theta: usize,
    n_z: usize,
) -> Mesh {
    assert!(radius > 0.0);
    assert!(length > 0.0);
    assert!(n_theta >= 3);
    assert!(n_z > 0);

    let mut mesh = Mesh::new(3);
    let node = |iz: usize, it: usize| -> usize { iz * n_theta + (it % n_theta) };
    for iz in 0..=n_z {
        let z = length * iz as f64 / n_z as f64;
        for it in 0..n_theta {
            let t = 2.0 * PI * it as f64 / n_theta as f64;
            mesh.add_node(Point::new_3d(radius * t.cos(), radius * t.sin(), z));
        }
    }
    for iz in 0..n_z {
        for it in 0..n_theta {
            let next = (it + 1) % n_theta;
            mesh.add_element(
                ElementType::Triangle,
                vec![node(iz, it), node(iz + 1, it), node(iz + 1, next)],
            );
            mesh.add_element(
                ElementType::Triangle,
                vec![node(iz, it), node(iz + 1, next), node(iz, next)],
            );
        }
    }
    for it in 0..n_theta {
        mesh.add_boundary(
            vec![node(0, it), node(0, it + 1)],
            BoundaryType::Dirichlet,
            1,
            0,
            it,
        );
        mesh.add_boundary(
            vec![node(n_z, it), node(n_z, it + 1)],
            BoundaryType::Dirichlet,
            2,
            0,
            it,
        );
    }
    mesh
}

/// Latitude-longitude spherical shell surface mesh.
pub fn spherical_shell_surface_mesh_triangles(radius: f64, n_theta: usize, n_phi: usize) -> Mesh {
    assert!(radius > 0.0);
    assert!(n_theta >= 3);
    assert!(n_phi >= 6);

    let mut mesh = Mesh::new(3);
    let north = mesh.add_node(Point::new_3d(0.0, 0.0, radius));
    let south = mesh.add_node(Point::new_3d(0.0, 0.0, -radius));
    let mut rings: Vec<Vec<usize>> = Vec::new();
    for i in 1..n_theta {
        let theta = PI * i as f64 / n_theta as f64;
        let z = radius * theta.cos();
        let r = radius * theta.sin();
        let mut ring = Vec::with_capacity(n_phi);
        for j in 0..n_phi {
            let phi = 2.0 * PI * j as f64 / n_phi as f64;
            ring.push(mesh.add_node(Point::new_3d(r * phi.cos(), r * phi.sin(), z)));
        }
        rings.push(ring);
    }

    for j in 0..n_phi {
        let next = (j + 1) % n_phi;
        mesh.add_element(
            ElementType::Triangle,
            vec![north, rings[0][next], rings[0][j]],
        );
    }
    for i in 0..rings.len() - 1 {
        for j in 0..n_phi {
            let next = (j + 1) % n_phi;
            mesh.add_element(
                ElementType::Triangle,
                vec![rings[i][j], rings[i][next], rings[i + 1][next]],
            );
            mesh.add_element(
                ElementType::Triangle,
                vec![rings[i][j], rings[i + 1][next], rings[i + 1][j]],
            );
        }
    }
    let last = rings.len() - 1;
    for j in 0..n_phi {
        let next = (j + 1) % n_phi;
        mesh.add_element(
            ElementType::Triangle,
            vec![rings[last][j], rings[last][next], south],
        );
    }
    mesh
}
