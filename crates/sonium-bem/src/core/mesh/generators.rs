//! Mesh generators for analytical test geometries
//!
//! Provides functions to generate surface meshes for standard geometries
//! used in BEM validation: spheres, cylinders, etc.

use num_complex::Complex64;
use sonium_mesher::convexhull::{ConvexHull3D, Vertex};
use sonium_mesher::mesh::triangulate_points_2d;
use std::collections::BTreeMap;
use std::f64::consts::PI;

// Array2 use removed (Mesh::nodes migrated to Vec<[f64; 3]>)
use crate::core::mesh::element::compute_element_area;
use crate::core::types::{BoundaryCondition, Element, ElementProperty, ElementType, Mesh};

/// Generate a quasi-uniform triangulated spherical surface mesh.
///
/// `n_theta` and `n_phi` retain their historical role as angular resolution
/// controls. Their product determines the nominal vertex count; vertices are
/// distributed with equal-area Fibonacci sampling and triangulated by their
/// convex hull so element aspect ratios remain bounded at the poles.
///
/// # Arguments
/// * `radius` - Sphere radius
/// * `n_theta` - Number of divisions in polar direction (latitude)
/// * `n_phi` - Number of divisions in azimuthal direction (longitude)
///
/// # Returns
/// A `Mesh` with triangular elements covering the sphere surface
///
/// # Example
/// ```ignore
/// let mesh = generate_sphere_mesh(1.0, 16, 32);
/// ```
pub fn generate_sphere_mesh(radius: f64, n_theta: usize, n_phi: usize) -> Mesh {
    assert!(
        radius.is_finite() && radius > 0.0,
        "radius must be positive"
    );
    assert!(n_theta >= 3, "n_theta must be at least 3");
    assert!(n_phi >= 3, "n_phi must be at least 3");

    // Preserve the historical nominal vertex count while avoiding fixed-size
    // longitude rings. Those rings become arbitrarily skinny near the poles
    // as n_theta grows, which stalls BEM boundary-trace convergence. A
    // Fibonacci distribution is equal-area; its convex hull supplies a closed,
    // shape-regular triangular surface with outward geometry reconstructed by
    // `create_mesh_from_data`.
    let vertex_count = (n_theta - 1) * n_phi + 2;
    let golden_angle = PI * (3.0 - 5.0_f64.sqrt());
    let points: Vec<Vertex> = (0..vertex_count)
        .map(|index| {
            let z = 1.0 - 2.0 * (index as f64 + 0.5) / vertex_count as f64;
            let radial = (1.0 - z * z).max(0.0).sqrt();
            let azimuth = golden_angle * index as f64;
            Vertex::new(
                radius * radial * azimuth.cos(),
                radius * radial * azimuth.sin(),
                radius * z,
            )
        })
        .collect();
    let hull = ConvexHull3D::build(&points).expect("sphere points must form a 3D convex hull");
    let nodes = hull
        .vertices()
        .iter()
        .map(|vertex| [vertex.x, vertex.y, vertex.z])
        .collect();
    let elements = hull
        .faces()
        .iter()
        .map(|face| face.indices().to_vec())
        .collect();

    create_mesh_from_data(nodes, elements, radius)
}

/// Generate a conforming cubed-sphere surface made entirely of Quad4 panels.
///
/// Each cube-face grid node is projected radially onto the requested sphere.
/// Integer cube coordinates are used as topology keys, so vertices and edges
/// shared by adjacent faces have one global mesh node rather than coincident
/// duplicates. `face_resolution` is the number of Quad4 panels along each
/// cube edge; the resulting mesh has `6 * face_resolution²` panels.
pub fn generate_cubed_sphere_quad_mesh(radius: f64, face_resolution: usize) -> Mesh {
    assert!(
        radius.is_finite() && radius > 0.0,
        "radius must be positive"
    );
    assert!(face_resolution >= 1, "face_resolution must be at least 1");

    let scale = i32::try_from(face_resolution).expect("face resolution must fit i32");
    let coordinate = |index: usize| -scale + 2 * index as i32;
    let mut node_by_cube_coordinate = BTreeMap::new();
    let mut nodes = Vec::new();
    let mut connectivity = Vec::with_capacity(6 * face_resolution * face_resolution);

    for face in 0..6 {
        let key = |u: i32, v: i32| match face {
            0 => (scale, u, v),
            1 => (-scale, u, v),
            2 => (u, scale, v),
            3 => (u, -scale, v),
            4 => (u, v, scale),
            5 => (u, v, -scale),
            _ => unreachable!(),
        };
        let reverse = matches!(face, 1 | 2 | 5);
        for i in 0..face_resolution {
            for j in 0..face_resolution {
                let u0 = coordinate(i);
                let u1 = coordinate(i + 1);
                let v0 = coordinate(j);
                let v1 = coordinate(j + 1);
                let corners = [key(u0, v0), key(u1, v0), key(u1, v1), key(u0, v1)];
                let indices = corners.map(|cube_coordinate| {
                    *node_by_cube_coordinate
                        .entry(cube_coordinate)
                        .or_insert_with(|| {
                            let [x, y, z] = [
                                cube_coordinate.0 as f64,
                                cube_coordinate.1 as f64,
                                cube_coordinate.2 as f64,
                            ];
                            let length = (x * x + y * y + z * z).sqrt();
                            let index = nodes.len();
                            nodes.push([
                                radius * x / length,
                                radius * y / length,
                                radius * z / length,
                            ]);
                            index
                        })
                });
                connectivity.push(if reverse {
                    vec![indices[0], indices[3], indices[2], indices[1]]
                } else {
                    indices.to_vec()
                });
            }
        }
    }

    create_mesh_from_data(nodes, connectivity, radius)
}

#[allow(dead_code)]
fn generate_latitude_sphere_mesh_legacy(radius: f64, n_theta: usize, n_phi: usize) -> Mesh {
    let mut nodes = Vec::new();
    let mut elements = Vec::new();

    // Generate nodes for latitude bands only (no pole vertices).
    // Row i (0-indexed) corresponds to theta = PI * (i+1) / n_theta.
    // Row 0 is the first ring near the north pole.
    // Row (n_theta-2) is the last ring near the south pole.
    for i in 1..n_theta {
        let theta = PI * i as f64 / n_theta as f64;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for j in 0..n_phi {
            let phi = 2.0 * PI * j as f64 / n_phi as f64;
            let x = radius * sin_theta * phi.cos();
            let y = radius * sin_theta * phi.sin();
            let z = radius * cos_theta;
            nodes.push([x, y, z]);
        }
    }

    // Middle bands (quad -> 2 triangles) between adjacent latitude rings
    // n_theta-1 latitude rings total -> n_theta-2 bands between them
    for i in 0..(n_theta - 2) {
        let row_start = i * n_phi;
        let next_row_start = (i + 1) * n_phi;

        for j in 0..n_phi {
            let j_next = (j + 1) % n_phi;

            let n0 = row_start + j;
            let n1 = row_start + j_next;
            let n2 = next_row_start + j;
            let n3 = next_row_start + j_next;

            elements.push(vec![n0, n2, n1]);
            elements.push(vec![n1, n2, n3]);
        }
    }

    // --- North polar cap ---
    // Boundary ring = row 0 (indices 0..n_phi), near the north pole.
    // Project these points to a tangent plane at the north pole using
    // gnomonic projection: (x, y) / z maps the sphere onto the tangent plane.
    // Near the pole z > 0, so this is well-behaved.
    let north_ring_start = 0;
    let north_boundary_2d: Vec<(f64, f64)> = (0..n_phi)
        .map(|j| {
            let idx = north_ring_start + j;
            let z = nodes[idx][2];
            (nodes[idx][0] / z, nodes[idx][1] / z)
        })
        .collect();

    // Triangulate the north cap disk. The "radius" in the projected plane is
    // sin(theta1)/cos(theta1) = tan(theta1).
    let theta1 = PI / n_theta as f64;
    let projected_radius_north = theta1.tan();
    let (north_interior_2d, north_triangles) =
        triangulate_disk_cap_2d(&north_boundary_2d, (0.0, 0.0), projected_radius_north, 0);

    // Add interior cap nodes projected back onto the sphere.
    let north_interior_start = nodes.len();
    for &(px, py) in &north_interior_2d {
        // Inverse gnomonic: given (px, py) on tangent plane at north pole,
        // the 3D point on the sphere of radius r is:
        // rho = sqrt(px^2 + py^2), c = atan(rho)
        // x = r * sin(c) * px / rho, y = r * sin(c) * py / rho, z = r * cos(c)
        let rho = (px * px + py * py).sqrt();
        if rho < 1e-15 {
            // Center point = north pole
            nodes.push([0.0, 0.0, radius]);
        } else {
            let c = rho.atan();
            let x = radius * c.sin() * px / rho;
            let y = radius * c.sin() * py / rho;
            let z = radius * c.cos();
            nodes.push([x, y, z]);
        }
    }

    for tri in &north_triangles {
        let global: Vec<usize> = tri
            .iter()
            .map(|&idx| {
                if idx < n_phi {
                    north_ring_start + idx
                } else {
                    north_interior_start + (idx - n_phi)
                }
            })
            .collect();
        // Delaunay gives CCW winding viewed from +z. For north cap, the outward
        // normal should point roughly toward +z (away from origin for the north cap
        // region). CCW from +z gives a +z-pointing normal, which is outward.
        elements.push(vec![global[0], global[1], global[2]]);
    }

    // --- South polar cap ---
    // Boundary ring = last row (row n_theta-2), near the south pole.
    let south_ring_start = (n_theta - 2) * n_phi;
    let south_boundary_2d: Vec<(f64, f64)> = (0..n_phi)
        .map(|j| {
            let idx = south_ring_start + j;
            let z = nodes[idx][2];
            // Project onto tangent plane at south pole: (x, y) / (-z)
            // (z < 0 near south pole, so -z > 0)
            (nodes[idx][0] / (-z), nodes[idx][1] / (-z))
        })
        .collect();

    let theta_last = PI * (n_theta - 1) as f64 / n_theta as f64;
    let projected_radius_south = (PI - theta_last).tan(); // = tan(theta1) same as north
    let (south_interior_2d, south_triangles) =
        triangulate_disk_cap_2d(&south_boundary_2d, (0.0, 0.0), projected_radius_south, 0);

    let south_interior_start = nodes.len();
    for &(px, py) in &south_interior_2d {
        // Inverse gnomonic from south pole tangent plane
        let rho = (px * px + py * py).sqrt();
        if rho < 1e-15 {
            nodes.push([0.0, 0.0, -radius]);
        } else {
            let c = rho.atan();
            let x = radius * c.sin() * px / rho;
            let y = radius * c.sin() * py / rho;
            let z = -radius * c.cos();
            nodes.push([x, y, z]);
        }
    }

    for tri in &south_triangles {
        let global: Vec<usize> = tri
            .iter()
            .map(|&idx| {
                if idx < n_phi {
                    south_ring_start + idx
                } else {
                    south_interior_start + (idx - n_phi)
                }
            })
            .collect();
        // For south cap: the outward normal should point toward -z.
        // Delaunay CCW viewed from +z gives +z normal. We need -z, so reverse winding.
        elements.push(vec![global[0], global[2], global[1]]);
    }

    create_mesh_from_data(nodes, elements, radius)
}

/// Generate an icosphere mesh (subdivided icosahedron)
///
/// Geodesic alternative to the Fibonacci-hull sphere generator.
///
/// # Arguments
/// * `radius` - Sphere radius
/// * `subdivisions` - Number of subdivision iterations (0=icosahedron, 1=42 verts, 2=162 verts)
///
/// # Returns
/// A `Mesh` with triangular elements
pub fn generate_icosphere_mesh(radius: f64, subdivisions: usize) -> Mesh {
    // Golden ratio
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let _scale = radius / (1.0 + phi * phi).sqrt();

    // Initial icosahedron vertices
    let mut vertices: Vec<[f64; 3]> = vec![
        [-1.0, phi, 0.0],
        [1.0, phi, 0.0],
        [-1.0, -phi, 0.0],
        [1.0, -phi, 0.0],
        [0.0, -1.0, phi],
        [0.0, 1.0, phi],
        [0.0, -1.0, -phi],
        [0.0, 1.0, -phi],
        [phi, 0.0, -1.0],
        [phi, 0.0, 1.0],
        [-phi, 0.0, -1.0],
        [-phi, 0.0, 1.0],
    ];

    // Scale to unit sphere
    for v in &mut vertices {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }

    // Initial icosahedron faces
    let mut faces: Vec<[usize; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    // Subdivide
    for _ in 0..subdivisions {
        let mut new_faces = Vec::new();
        let mut edge_midpoints: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();

        for face in &faces {
            let v0 = face[0];
            let v1 = face[1];
            let v2 = face[2];

            // Get or create midpoints
            let m01 = get_midpoint(&mut vertices, &mut edge_midpoints, v0, v1);
            let m12 = get_midpoint(&mut vertices, &mut edge_midpoints, v1, v2);
            let m20 = get_midpoint(&mut vertices, &mut edge_midpoints, v2, v0);

            // Create 4 new faces
            new_faces.push([v0, m01, m20]);
            new_faces.push([v1, m12, m01]);
            new_faces.push([v2, m20, m12]);
            new_faces.push([m01, m12, m20]);
        }

        faces = new_faces;
    }

    // Scale to desired radius
    let nodes: Vec<[f64; 3]> = vertices
        .iter()
        .map(|v| [v[0] * radius, v[1] * radius, v[2] * radius])
        .collect();

    let elements: Vec<Vec<usize>> = faces.iter().map(|f| vec![f[0], f[1], f[2]]).collect();

    create_mesh_from_data(nodes, elements, radius)
}

/// Helper for icosphere: get or create edge midpoint
fn get_midpoint(
    vertices: &mut Vec<[f64; 3]>,
    cache: &mut std::collections::HashMap<(usize, usize), usize>,
    v0: usize,
    v1: usize,
) -> usize {
    let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };

    if let Some(&idx) = cache.get(&key) {
        return idx;
    }

    // Create new vertex at midpoint, projected to unit sphere
    let mid = [
        (vertices[v0][0] + vertices[v1][0]) / 2.0,
        (vertices[v0][1] + vertices[v1][1]) / 2.0,
        (vertices[v0][2] + vertices[v1][2]) / 2.0,
    ];

    let len = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();
    let normalized = [mid[0] / len, mid[1] / len, mid[2] / len];

    let idx = vertices.len();
    vertices.push(normalized);
    cache.insert(key, idx);

    idx
}

/// Generate a cylindrical surface mesh
///
/// Creates a mesh for a finite cylinder (without end caps).
///
/// # Arguments
/// * `radius` - Cylinder radius
/// * `height` - Cylinder height (centered at z=0)
/// * `n_circumference` - Number of divisions around circumference
/// * `n_height` - Number of divisions along height
///
/// # Returns
/// A `Mesh` with quadrilateral elements
pub fn generate_cylinder_mesh(
    radius: f64,
    height: f64,
    n_circumference: usize,
    n_height: usize,
) -> Mesh {
    let mut nodes = Vec::new();
    let mut elements = Vec::new();

    let z_min = -height / 2.0;
    let dz = height / n_height as f64;

    // Generate nodes
    for i in 0..=n_height {
        let z = z_min + i as f64 * dz;

        for j in 0..n_circumference {
            let phi = 2.0 * PI * j as f64 / n_circumference as f64;
            let x = radius * phi.cos();
            let y = radius * phi.sin();
            nodes.push([x, y, z]);
        }
    }

    // Generate quad elements
    for i in 0..n_height {
        let row_start = i * n_circumference;
        let next_row_start = (i + 1) * n_circumference;

        for j in 0..n_circumference {
            let j_next = (j + 1) % n_circumference;

            let n0 = row_start + j;
            let n1 = row_start + j_next;
            let n2 = next_row_start + j_next;
            let n3 = next_row_start + j;

            elements.push(vec![n0, n1, n2, n3]);
        }
    }

    create_mesh_from_data(nodes, elements, radius)
}

/// Generate a cylinder mesh with end caps (closed cylinder)
///
/// Uses Delaunay triangulation for the end caps to avoid degenerate
/// fan triangles around a center pole vertex.
pub fn generate_closed_cylinder_mesh(
    radius: f64,
    height: f64,
    n_circumference: usize,
    n_height: usize,
    n_cap_rings: usize,
) -> Mesh {
    let mut nodes = Vec::new();
    let mut elements = Vec::new();

    let z_min = -height / 2.0;
    let z_max = height / 2.0;
    let dz = height / n_height as f64;

    // Lateral surface nodes: rows 0..=n_height, each with n_circumference nodes
    // Bottom ring is row 0 (indices 0..n_circumference)
    // Top ring is row n_height (indices n_height*n_circumference..(n_height+1)*n_circumference)
    for i in 0..=n_height {
        let z = z_min + i as f64 * dz;
        for j in 0..n_circumference {
            let phi = 2.0 * PI * j as f64 / n_circumference as f64;
            nodes.push([radius * phi.cos(), radius * phi.sin(), z]);
        }
    }

    // Lateral surface elements (quads)
    for i in 0..n_height {
        let row_start = i * n_circumference;
        let next_row_start = (i + 1) * n_circumference;
        for j in 0..n_circumference {
            let j_next = (j + 1) % n_circumference;
            elements.push(vec![
                row_start + j,
                row_start + j_next,
                next_row_start + j_next,
                next_row_start + j,
            ]);
        }
    }

    // --- Bottom cap (z = z_min) ---
    // Boundary ring for bottom cap = lateral row 0 (indices 0..n_circumference)
    let bottom_boundary_start = 0;
    let bottom_boundary_2d: Vec<(f64, f64)> = (0..n_circumference)
        .map(|j| {
            (
                nodes[bottom_boundary_start + j][0],
                nodes[bottom_boundary_start + j][1],
            )
        })
        .collect();

    let (bottom_interior_2d, bottom_triangles) =
        triangulate_disk_cap_2d(&bottom_boundary_2d, (0.0, 0.0), radius, n_cap_rings);

    // Add interior cap nodes as 3D points at z_min
    let bottom_interior_start = nodes.len();
    for &(x, y) in &bottom_interior_2d {
        nodes.push([x, y, z_min]);
    }

    // Map Delaunay indices to global node indices.
    // In the Delaunay, boundary indices 0..n_circumference map to lateral row 0.
    // Interior indices n_circumference.. map to bottom_interior_start + (idx - n_circumference).
    for tri in &bottom_triangles {
        let global: Vec<usize> = tri
            .iter()
            .map(|&idx| {
                if idx < n_circumference {
                    bottom_boundary_start + idx
                } else {
                    bottom_interior_start + (idx - n_circumference)
                }
            })
            .collect();
        // The 2-D triangulator does not guarantee a stable winding. Orient the
        // connectivity explicitly because `create_mesh_from_data` may flip the
        // stored outward normal, but cannot repair connectivity metadata.
        elements.push(orient_flat_cap_triangle(
            &nodes,
            [global[0], global[1], global[2]],
            -1.0,
        ));
    }

    // --- Top cap (z = z_max) ---
    let top_boundary_start = n_height * n_circumference;
    let top_boundary_2d: Vec<(f64, f64)> = (0..n_circumference)
        .map(|j| {
            (
                nodes[top_boundary_start + j][0],
                nodes[top_boundary_start + j][1],
            )
        })
        .collect();

    let (top_interior_2d, top_triangles) =
        triangulate_disk_cap_2d(&top_boundary_2d, (0.0, 0.0), radius, n_cap_rings);

    let top_interior_start = nodes.len();
    for &(x, y) in &top_interior_2d {
        nodes.push([x, y, z_max]);
    }

    for tri in &top_triangles {
        let global: Vec<usize> = tri
            .iter()
            .map(|&idx| {
                if idx < n_circumference {
                    top_boundary_start + idx
                } else {
                    top_interior_start + (idx - n_circumference)
                }
            })
            .collect();
        elements.push(orient_flat_cap_triangle(
            &nodes,
            [global[0], global[1], global[2]],
            1.0,
        ));
    }

    create_mesh_from_data(nodes, elements, radius)
}

/// Orient a triangle on a horizontal cap so its connectivity normal points in
/// the requested z direction.
///
/// The triangulator's output winding is intentionally treated as unspecified;
/// relying on it makes the generated connectivity disagree with the outward
/// normal stored by [`create_mesh_from_data`].
fn orient_flat_cap_triangle(
    nodes: &[[f64; 3]],
    mut triangle: [usize; 3],
    desired_normal_z: f64,
) -> Vec<usize> {
    let p0 = nodes[triangle[0]];
    let p1 = nodes[triangle[1]];
    let p2 = nodes[triangle[2]];
    let signed_area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
    if signed_area * desired_normal_z < 0.0 {
        triangle.swap(1, 2);
    }
    triangle.to_vec()
}

/// Triangulate a flat circular disk using Delaunay triangulation.
/// Returns (interior_nodes, triangles) where interior_nodes are the non-boundary
/// points added, and triangles reference indices into a combined array of
/// [boundary_nodes..., interior_nodes...].
///
/// # Arguments
/// * `boundary_nodes_2d` - 2D coordinates of the boundary ring (already exist from lateral mesh)
/// * `center` - Center of the disk in 2D
/// * `radius` - Disk radius
/// * `n_rings` - Number of interior concentric rings (0 = boundary only)
fn triangulate_disk_cap_2d(
    boundary_nodes_2d: &[(f64, f64)],
    center: (f64, f64),
    radius: f64,
    n_rings: usize,
) -> (Vec<(f64, f64)>, Vec<[usize; 3]>) {
    let n_boundary = boundary_nodes_2d.len();
    let n_circ = n_boundary;

    // Collect all points: boundary ring first, then interior rings, then center
    let mut all_points: Vec<(f64, f64)> = boundary_nodes_2d.to_vec();
    let mut interior_nodes: Vec<(f64, f64)> = Vec::new();

    // Interior rings at radii radius * i / (n_rings + 1) for i=1..n_rings
    for ring_idx in 1..=n_rings {
        let r = radius * ring_idx as f64 / (n_rings + 1) as f64;
        // Stagger by half-step for better element quality
        let offset_angle = PI / n_circ as f64 * ring_idx as f64;
        for j in 0..n_circ {
            let phi = 2.0 * PI * j as f64 / n_circ as f64 + offset_angle;
            let pt = (center.0 + r * phi.cos(), center.1 + r * phi.sin());
            interior_nodes.push(pt);
            all_points.push(pt);
        }
    }

    // Always add center point
    interior_nodes.push(center);
    all_points.push(center);

    let triangles = triangulate_points_2d(&all_points);

    (interior_nodes, triangles)
}

/// Create a Mesh struct from raw node and element data
fn create_mesh_from_data(
    nodes: Vec<[f64; 3]>,
    connectivity: Vec<Vec<usize>>,
    _char_length: f64,
) -> Mesh {
    let n_nodes = nodes.len();
    let n_elements = connectivity.len();

    // Create node array
    let mut node_array: Vec<[f64; 3]> = vec![[0.0; 3]; n_nodes];
    for (i, node) in nodes.iter().enumerate() {
        node_array[i][0] = node[0];
        node_array[i][1] = node[1];
        node_array[i][2] = node[2];
    }

    // Create elements
    let mut elements = Vec::with_capacity(n_elements);
    for (idx, conn) in connectivity.iter().enumerate() {
        let element_type = if conn.len() == 3 {
            ElementType::Tri3
        } else {
            ElementType::Quad4
        };

        let mut elem = Element {
            connectivity: conn.clone(),
            element_type,
            property: ElementProperty::Surface,
            normal: [0.0; 3],
            node_normals: vec![[0.0; 3]; element_type.num_nodes()],
            center: [0.0; 3],
            area: 0.0,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![idx],
        };

        // Compute geometry
        compute_element_geometry(&mut elem, &node_array);
        elements.push(elem);
    }

    // Compute node normals (average of adjacent element normals)
    let mut node_counts = vec![0usize; n_nodes];

    for elem in &elements {
        for &node_idx in &elem.connectivity {
            node_counts[node_idx] += 1;
        }
    }

    // Update element node normals using actual surface normals at each node
    // For sphere meshes: node normal = node_position / |node_position| (outward radial)
    for elem in &mut elements {
        for (local_idx, &node_idx) in elem.connectivity.iter().enumerate() {
            let mut len = 0.0;
            for coord in &node_array[node_idx] {
                len += coord * coord;
            }
            len = len.sqrt();
            if len > 1e-15 {
                for (j, coord) in node_array[node_idx].iter().enumerate() {
                    elem.node_normals[local_idx][j] = coord / len;
                }
            } else {
                for (j, normal) in elem.normal.iter().enumerate() {
                    elem.node_normals[local_idx][j] = *normal;
                }
            }
        }
    }

    Mesh {
        nodes: node_array,
        elements,
        external_node_numbers: (0..n_nodes as i32).collect(),
        external_element_numbers: (0..n_elements as i32).collect(),
        num_boundary_nodes: n_nodes,
        num_evaluation_nodes: 0,
        num_boundary_elements: n_elements,
        num_evaluation_elements: 0,
        symmetry_planes: [0, 0, 0],
        symmetry_coordinates: [0.0, 0.0, 0.0],
        num_reflections: 0,
        node_to_elements: Vec::new(),
    }
}

/// Compute element center, area, and normal
fn compute_element_geometry(elem: &mut Element, nodes: &[[f64; 3]]) {
    let n = elem.connectivity.len();

    // Compute center
    elem.center = [0.0; 3];
    for &node_idx in &elem.connectivity {
        for (center_j, node_j) in elem.center.iter_mut().zip(nodes[node_idx].iter()) {
            *center_j += node_j;
        }
    }
    for center_j in elem.center.iter_mut() {
        *center_j /= n as f64;
    }

    // Compute normal and area
    if n == 3 {
        // Triangle
        let n0 = elem.connectivity[0];
        let n1 = elem.connectivity[1];
        let n2 = elem.connectivity[2];

        let v1 = [
            nodes[n1][0] - nodes[n0][0],
            nodes[n1][1] - nodes[n0][1],
            nodes[n1][2] - nodes[n0][2],
        ];
        let v2 = [
            nodes[n2][0] - nodes[n0][0],
            nodes[n2][1] - nodes[n0][1],
            nodes[n2][2] - nodes[n0][2],
        ];

        // Cross product
        let cross = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ];

        let len = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        elem.area = len / 2.0;

        if len > 1e-15 {
            elem.normal = [cross[0] / len, cross[1] / len, cross[2] / len];
        }
    } else {
        // Quad - use diagonal cross product
        let n0 = elem.connectivity[0];
        let n1 = elem.connectivity[1];
        let n2 = elem.connectivity[2];
        let n3 = elem.connectivity[3];

        let d1 = [
            nodes[n2][0] - nodes[n0][0],
            nodes[n2][1] - nodes[n0][1],
            nodes[n2][2] - nodes[n0][2],
        ];
        let d2 = [
            nodes[n3][0] - nodes[n1][0],
            nodes[n3][1] - nodes[n1][1],
            nodes[n3][2] - nodes[n1][2],
        ];

        let cross = [
            d1[1] * d2[2] - d1[2] * d2[1],
            d1[2] * d2[0] - d1[0] * d2[2],
            d1[0] * d2[1] - d1[1] * d2[0],
        ];

        let len = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        // Keep generated Quad4 metadata aligned with the shared geometry
        // helper used by imported NumCalc meshes and high-level admission.
        // The diagonal cross remains the normal convention; its magnitude is
        // not the canonical warped-quad area convention.
        elem.area = compute_element_area(nodes, &elem.connectivity, elem.element_type);

        if len > 1e-15 {
            elem.normal = [cross[0] / len, cross[1] / len, cross[2] / len];
        }
    }

    // Ensure normal points outward (for convex shapes centered at origin)
    // Check: normal should point in same direction as center (n · c > 0)
    let n_dot_c = elem.normal[0] * elem.center[0]
        + elem.normal[1] * elem.center[1]
        + elem.normal[2] * elem.center[2];

    if n_dot_c < 0.0 {
        // Flip normal to point outward
        elem.normal[0] = -elem.normal[0];
        elem.normal[1] = -elem.normal[1];
        elem.normal[2] = -elem.normal[2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubed_sphere_quad_mesh_is_conforming_radial_and_outward() {
        let resolution = 3;
        let mesh = generate_cubed_sphere_quad_mesh(2.0, resolution);
        assert_eq!(mesh.elements.len(), 6 * resolution * resolution);
        assert_eq!(
            mesh.nodes.len(),
            6 * (resolution + 1) * (resolution + 1) - 12 * (resolution + 1) + 8
        );
        assert!(
            mesh.elements
                .iter()
                .all(|element| element.element_type == ElementType::Quad4)
        );
        for node in &mesh.nodes {
            let norm = (node[0].powi(2) + node[1].powi(2) + node[2].powi(2)).sqrt();
            assert!((norm - 2.0).abs() < 1.0e-12);
        }
        for element in &mesh.elements {
            assert!(element.area > 0.0);
            let alignment = element
                .normal
                .iter()
                .zip(element.center)
                .map(|(&normal, center)| normal * center)
                .sum::<f64>();
            assert!(alignment > 0.0, "cubed-sphere panel normal points inward");
        }

        let dof_map = mesh
            .continuous_quadratic_dof_map()
            .expect("conforming cubed sphere must admit continuous Q2 DOFs");
        let local_slots = 9 * mesh.elements.len();
        assert!(dof_map.num_dofs() < local_slots);
    }

    #[test]
    fn test_sphere_mesh_generation() {
        let mesh = generate_sphere_mesh(1.0, 8, 16);

        assert!(mesh.nodes.len() > 0);
        assert!(!mesh.elements.is_empty());

        // Check all nodes are on the sphere
        for i in 0..mesh.nodes.len() {
            let r =
                (mesh.nodes[i][0].powi(2) + mesh.nodes[i][1].powi(2) + mesh.nodes[i][2].powi(2))
                    .sqrt();
            assert!((r - 1.0).abs() < 1e-10, "Node {} not on sphere: r={}", i, r);
        }

        // Check elements are valid triangles
        for elem in &mesh.elements {
            assert_eq!(elem.connectivity.len(), 3);
            assert!(elem.area > 0.0);
        }
    }

    #[test]
    fn generated_warped_quad_uses_shared_area_convention() {
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
        ];
        let mut element = Element {
            connectivity: vec![0, 1, 2, 3],
            element_type: ElementType::Quad4,
            property: ElementProperty::Surface,
            normal: [0.0; 3],
            node_normals: vec![[0.0; 3]; 4],
            center: [0.0; 3],
            area: 0.0,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0],
        };

        compute_element_geometry(&mut element, &nodes);
        assert_eq!(
            element.area,
            compute_element_area(&nodes, &element.connectivity, ElementType::Quad4)
        );
    }

    #[test]
    fn test_icosphere_mesh_generation() {
        let mesh = generate_icosphere_mesh(1.0, 2);

        // Subdivision 2 should give 162 vertices
        assert_eq!(mesh.nodes.len(), 162);

        // Check all nodes are on unit sphere
        for i in 0..mesh.nodes.len() {
            let r =
                (mesh.nodes[i][0].powi(2) + mesh.nodes[i][1].powi(2) + mesh.nodes[i][2].powi(2))
                    .sqrt();
            assert!((r - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cylinder_mesh_generation() {
        let mesh = generate_cylinder_mesh(1.0, 2.0, 16, 8);

        assert!(mesh.nodes.len() > 0);
        assert!(!mesh.elements.is_empty());

        // Check all nodes are on cylinder surface
        for i in 0..mesh.nodes.len() {
            let r = (mesh.nodes[i][0].powi(2) + mesh.nodes[i][1].powi(2)).sqrt();
            assert!(
                (r - 1.0).abs() < 1e-10,
                "Node {} not on cylinder: r={}",
                i,
                r
            );
        }

        // Check elements are valid quads
        for elem in &mesh.elements {
            assert_eq!(elem.connectivity.len(), 4);
            assert!(elem.area > 0.0);
        }
    }

    #[test]
    fn test_sphere_normals_point_outward() {
        let mesh = generate_icosphere_mesh(1.0, 1);

        for elem in &mesh.elements {
            // Normal should point away from origin (outward)
            let dot = elem.normal[0] * elem.center[0]
                + elem.normal[1] * elem.center[1]
                + elem.normal[2] * elem.center[2];
            assert!(dot > 0.0, "Normal should point outward");
        }
    }

    #[test]
    fn test_sphere_surface_area() {
        // Higher resolution for accurate area
        let mesh = generate_icosphere_mesh(1.0, 4);

        let total_area: f64 = mesh.elements.iter().map(|e| e.area).sum();
        let expected_area = 4.0 * PI; // Surface area of unit sphere

        let error = (total_area - expected_area).abs() / expected_area;
        assert!(error < 0.01, "Surface area error: {:.2}%", error * 100.0);
    }

    #[test]
    fn test_closed_cylinder_cap_quality() {
        let mesh = generate_closed_cylinder_mesh(1.0, 2.0, 12, 8, 2);
        for (i, elem) in mesh.elements.iter().enumerate() {
            assert!(
                elem.area > 1e-10,
                "Degenerate element {} with area {}",
                i,
                elem.area
            );
        }
    }

    #[test]
    fn closed_cylinder_connectivity_normals_match_for_eaa_mesh() {
        let mesh = generate_closed_cylinder_mesh(0.1, 3.4, 20, 60, 4);
        for (index, element) in mesh.elements.iter().enumerate() {
            let expected = crate::core::mesh::element::compute_element_normal(
                &mesh.nodes,
                &element.connectivity,
                element.element_type,
            );
            let alignment = element
                .normal
                .iter()
                .zip(expected)
                .map(|(stored, connectivity)| stored * connectivity)
                .sum::<f64>();
            assert!(
                alignment > 1.0 - 1.0e-8,
                "element {index} normal disagrees with connectivity: {alignment}"
            );
        }
    }

    #[test]
    fn test_sphere_pole_quality() {
        let mesh = generate_sphere_mesh(1.0, 16, 32);
        for (i, elem) in mesh.elements.iter().enumerate() {
            assert!(
                elem.area > 1e-10,
                "Degenerate element {} with area {}",
                i,
                elem.area
            );
        }
    }

    #[test]
    fn test_sphere_mesh_is_quasi_uniform_near_the_poles() {
        let mesh = generate_sphere_mesh(1.0, 16, 32);
        let mut worst_edge_ratio = 1.0_f64;

        for element in &mesh.elements {
            let points: Vec<_> = element
                .connectivity
                .iter()
                .map(|&index| mesh.nodes[index])
                .collect();
            let mut edges = [0.0; 3];
            for edge in 0..3 {
                let a = points[edge];
                let b = points[(edge + 1) % 3];
                edges[edge] =
                    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            }
            let shortest = edges.iter().copied().fold(f64::INFINITY, f64::min);
            let longest = edges.iter().copied().fold(0.0_f64, f64::max);
            worst_edge_ratio = worst_edge_ratio.max(longest / shortest);
        }

        assert!(
            worst_edge_ratio < 3.0,
            "sphere triangles must remain shape-regular near the poles; worst edge ratio={worst_edge_ratio:.3}"
        );
    }
}
