//! Pressure-traction boundary condition for the incompressible N-S solver.
//!
//! Adds the natural boundary term ∫_Γ (−p n)·v dΓ for a constant imposed
//! pressure `p` on each marked boundary face, letting a duct be driven by a
//! pressure difference instead of a prescribed velocity profile.

/// Constant pressure imposed on one boundary marker.
#[derive(Debug, Clone, Copy)]
pub struct PressureTraction {
    pub marker: i32,
    pub pressure_pa: f64,
}

/// Adds ∫_Γ (−p·n)·φ dΓ into `rhs` for P1 velocity DOFs laid out as
/// `dof = node * dim + comp`. Exact for constant p on P1 faces.
pub fn assemble_pressure_tractions(
    mesh: &crate::mesh::Mesh,
    dim: usize,
    tractions: &[PressureTraction],
    rhs: &mut [f64],
) {
    for tr in tractions {
        for face in &mesh.boundaries {
            if face.marker != tr.marker {
                continue;
            }
            let owner = &mesh.elements[face.element_idx];
            let centroid = element_centroid(mesh, owner);
            match face.nodes.len() {
                2 => {
                    let a = &mesh.nodes[face.nodes[0]];
                    let b = &mesh.nodes[face.nodes[1]];
                    let (n, len) = outward_normal_2d(a, b, &centroid);
                    for &nd in &face.nodes {
                        // `dim.min(2)`: a 2D edge only carries x/y components,
                        // even if the caller passes a 3-component layout.
                        for c in 0..dim.min(2) {
                            rhs[nd * dim + c] += -tr.pressure_pa * n[c] * (len / 2.0);
                        }
                    }
                }
                3 => {
                    let (n, area) = outward_normal_tri(mesh, &face.nodes, &centroid);
                    for &nd in &face.nodes {
                        for c in 0..3 {
                            rhs[nd * dim + c] += -tr.pressure_pa * n[c] * (area / 3.0);
                        }
                    }
                }
                _ => {} // quad faces: split or skip; slot/round ports use tri/edge meshes
            }
        }
    }
}

/// Space-aware variant of [`assemble_pressure_tractions`].
///
/// P1 velocity spaces delegate to [`assemble_pressure_tractions`] and keep
/// its exact arithmetic. Higher-order (P2) traces use consistent boundary
/// quadrature with the space's edge-midpoint DOFs (laid out as
/// `scalar_dof * components + comp`, with the midpoint at
/// `num_nodes + edge_idx`): 3-point Gauss–Legendre on 2D edges, the degree-2
/// triangle rule on 3D triangular faces — the same weights as
/// [`crate::boundary::apply_traction_vector`]. For constant `-p·n` this gives
/// the exact P2 consistent loads `L/6, L/6, 4L/6` on an edge and
/// `0, 0, 0, A/3, A/3, A/3` on a triangle.
pub fn assemble_pressure_tractions_for_space<VSpace: crate::spaces::FunctionSpace>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    tractions: &[PressureTraction],
    rhs: &mut [f64],
) {
    let components = v_space.num_components();
    if v_space.polynomial_degree() < 2 {
        assemble_pressure_tractions(mesh, components, tractions, rhs);
        return;
    }

    let topology = crate::mesh::topology::MeshTopology::build(mesh);
    let n_nodes = mesh.num_nodes();
    let edge_midpoint_dof = |a: usize, b: usize| -> usize {
        let key = if a < b { (a, b) } else { (b, a) };
        n_nodes + topology.edge_map[&key]
    };
    let quad_1d = crate::quadrature::gauss_legendre_1d(3);
    let quad_tri = crate::quadrature::gauss_triangle(3);

    for tr in tractions {
        for face in &mesh.boundaries {
            if face.marker != tr.marker {
                continue;
            }
            let owner = &mesh.elements[face.element_idx];
            let centroid = element_centroid(mesh, owner);
            match face.nodes.len() {
                2 => {
                    let a = &mesh.nodes[face.nodes[0]];
                    let b = &mesh.nodes[face.nodes[1]];
                    let (n, len) = outward_normal_2d(a, b, &centroid);
                    let scalar_dofs = [
                        face.nodes[0],
                        face.nodes[1],
                        edge_midpoint_dof(face.nodes[0], face.nodes[1]),
                    ];
                    for qp in &quad_1d {
                        let t = 0.5 * (qp.coords[0] + 1.0);
                        let w = 0.5 * qp.weight * len;
                        // P2 edge trace shape functions at edge parameter t.
                        let shapes = [
                            (1.0 - t) * (1.0 - 2.0 * t),
                            t * (2.0 * t - 1.0),
                            4.0 * t * (1.0 - t),
                        ];
                        for (s, &scalar_dof) in scalar_dofs.iter().enumerate() {
                            // `components.min(2)`: a 2D edge only carries x/y
                            // components, even for a 3-component layout.
                            for c in 0..components.min(2) {
                                rhs[scalar_dof * components + c] +=
                                    -tr.pressure_pa * n[c] * shapes[s] * w;
                            }
                        }
                    }
                }
                3 => {
                    let (n, area) = outward_normal_tri(mesh, &face.nodes, &centroid);
                    let nodes = &face.nodes;
                    let scalar_dofs = [
                        nodes[0],
                        nodes[1],
                        nodes[2],
                        edge_midpoint_dof(nodes[0], nodes[1]),
                        edge_midpoint_dof(nodes[1], nodes[2]),
                        edge_midpoint_dof(nodes[2], nodes[0]),
                    ];
                    for qp in &quad_tri {
                        let l1 = qp.coords[0];
                        let l2 = qp.coords[1];
                        let l0 = 1.0 - l1 - l2;
                        // P2 triangle trace shape functions in area coordinates.
                        let shapes = [
                            l0 * (2.0 * l0 - 1.0),
                            l1 * (2.0 * l1 - 1.0),
                            l2 * (2.0 * l2 - 1.0),
                            4.0 * l0 * l1,
                            4.0 * l1 * l2,
                            4.0 * l2 * l0,
                        ];
                        // Reference-triangle weights sum to 1/2; the surface
                        // Jacobian is |e1 × e2| = 2·A.
                        let w = qp.weight * 2.0 * area;
                        for (s, &scalar_dof) in scalar_dofs.iter().enumerate() {
                            for c in 0..components.min(3) {
                                rhs[scalar_dof * components + c] +=
                                    -tr.pressure_pa * n[c] * shapes[s] * w;
                            }
                        }
                    }
                }
                _ => {} // quad faces: split or skip; slot/round ports use tri/edge meshes
            }
        }
    }
}

/// Centroid of an element, as a 3D point (z = 0 for 2D meshes).
fn element_centroid(mesh: &crate::mesh::Mesh, elem: &crate::mesh::Element) -> [f64; 3] {
    let mut c = [0.0; 3];
    for &nd in &elem.nodes {
        let p = &mesh.nodes[nd];
        c[0] += p.x;
        c[1] += p.y;
        c[2] += p.z;
    }
    let inv = 1.0 / elem.nodes.len() as f64;
    [c[0] * inv, c[1] * inv, c[2] * inv]
}

/// Unit outward normal and length of a 2D boundary edge `a`–`b`.
///
/// The edge tangent is rotated 90° and normalised; the sign is chosen so the
/// normal points away from the owning element's centroid.
fn outward_normal_2d(
    a: &crate::mesh::Point,
    b: &crate::mesh::Point,
    centroid: &[f64; 3],
) -> ([f64; 2], f64) {
    let tx = b.x - a.x;
    let ty = b.y - a.y;
    let len = (tx * tx + ty * ty).sqrt();
    // Rotate the tangent 90° counter-clockwise.
    let mut n = [ty / len, -tx / len];
    let mid = [0.5 * (a.x + b.x), 0.5 * (a.y + b.y)];
    let dot = (mid[0] - centroid[0]) * n[0] + (mid[1] - centroid[1]) * n[1];
    if dot < 0.0 {
        n = [-n[0], -n[1]];
    }
    (n, len)
}

/// Unit outward normal and area of a triangular boundary face in 3D.
///
/// The normal is the normalised cross product of the first two edges; the
/// sign is chosen so it points away from the owning element's centroid.
fn outward_normal_tri(
    mesh: &crate::mesh::Mesh,
    nodes: &[usize],
    centroid: &[f64; 3],
) -> ([f64; 3], f64) {
    let p0 = &mesh.nodes[nodes[0]];
    let p1 = &mesh.nodes[nodes[1]];
    let p2 = &mesh.nodes[nodes[2]];
    let e1 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
    let e2 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
    let cross = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let norm = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    // |e1 × e2| = 2·A for a triangle.
    let area = 0.5 * norm;
    let mut n = [cross[0] / norm, cross[1] / norm, cross[2] / norm];
    let mid = [
        (p0.x + p1.x + p2.x) / 3.0,
        (p0.y + p1.y + p2.y) / 3.0,
        (p0.z + p1.z + p2.z) / 3.0,
    ];
    let dot = (mid[0] - centroid[0]) * n[0]
        + (mid[1] - centroid[1]) * n[1]
        + (mid[2] - centroid[2]) * n[2];
    if dot < 0.0 {
        n = [-n[0], -n[1], -n[2]];
    }
    (n, area)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::mesh::{BoundaryFace, BoundaryType, Element, ElementType, Mesh, Point};
    use crate::spaces::{FunctionSpace, VectorH1Space};

    /// Unit square [0,1]² split into two triangles along the 0–3 diagonal,
    /// with all four perimeter edges as boundary faces. Only the right edge
    /// (x = 1, nodes 1–3) carries `marker`; the rest use marker 99. Each
    /// face's `element_idx` points at the element adjacent to that edge so
    /// the outward-normal centroid test works.
    fn unit_square_with_marked_edge(marker: i32) -> Mesh {
        let mut mesh = Mesh::new(2);
        mesh.nodes = vec![
            Point::new_2d(0.0, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(0.0, 1.0),
            Point::new_2d(1.0, 1.0),
        ];
        mesh.elements = vec![
            Element::new(ElementType::Triangle, vec![0, 1, 3], 0),
            Element::new(ElementType::Triangle, vec![0, 3, 2], 1),
        ];
        let face = |nodes: Vec<usize>, face_marker: i32, element_idx: usize| BoundaryFace {
            nodes,
            boundary_type: BoundaryType::Neumann,
            marker: face_marker,
            element_idx,
            local_idx: 0,
        };
        mesh.boundaries = vec![
            face(vec![0, 1], 99, 0),     // bottom
            face(vec![1, 3], marker, 0), // right (x = 1)
            face(vec![3, 2], 99, 1),     // top
            face(vec![2, 0], 99, 1),     // left
        ];
        mesh
    }

    #[test]
    fn uniform_pressure_on_edge_gives_p_times_length() {
        let mesh = unit_square_with_marked_edge(7);
        let mut rhs = vec![0.0; 4 * 2];
        assemble_pressure_tractions(
            &mesh,
            2,
            &[PressureTraction {
                marker: 7,
                pressure_pa: 100.0,
            }],
            &mut rhs,
        );
        // Right-edge nodes are 1 and 3 (x=1). Outward normal is (+1, 0), so
        // traction −p·n pushes in −x: each node gets −p·L/2 = −50 in x, 0 in y.
        assert!((rhs[1 * 2] + 50.0).abs() < 1e-12);
        assert!((rhs[3 * 2] + 50.0).abs() < 1e-12);
        assert!(rhs[1 * 2 + 1].abs() < 1e-12 && rhs[3 * 2 + 1].abs() < 1e-12);
        // All other DOFs untouched; total x-force = −p·L = −100.
        let total: f64 = rhs.iter().step_by(2).sum();
        assert!((total + 100.0).abs() < 1e-12);
    }

    #[test]
    fn p2_edge_traction_work_equals_negative_pressure_times_flux() {
        let mut mesh = unit_square_with_marked_edge(7);
        mesh.boundaries.retain(|face| face.marker == 7);
        mesh.boundaries[0].local_idx = 1;
        let space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let mut rhs = vec![0.0; space.num_global_dofs(&mesh)];
        let pressure = 17.0;
        assemble_pressure_tractions_for_space(
            &mesh,
            &space,
            &[PressureTraction {
                marker: 7,
                pressure_pa: pressure,
            }],
            &mut rhs,
        );
        let velocity: Vec<f64> = (0..space.num_global_dofs(&mesh))
            .map(|dof| {
                let point = space.dof_coordinate(&mesh, dof).unwrap();
                if dof % 2 == 0 { 2.0 + point[1] } else { 0.0 }
            })
            .collect();
        let work = rhs.iter().zip(&velocity).map(|(f, u)| f * u).sum::<f64>();
        let independent_flux = 2.5; // integral_0^1 (2+y) dy
        assert!((work + pressure * independent_flux).abs() < 1.0e-12);
    }

    #[test]
    fn p2_triangle_traction_work_equals_negative_pressure_times_flux() {
        let mut mesh = Mesh::new(3);
        mesh.nodes = vec![
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(0.0, 0.0, 1.0),
        ];
        mesh.elements = vec![Element::new(ElementType::Tetrahedron, vec![0, 1, 2, 3], 0)];
        mesh.boundaries = vec![BoundaryFace {
            nodes: vec![1, 2, 3],
            boundary_type: BoundaryType::Neumann,
            marker: 8,
            element_idx: 0,
            local_idx: 3,
        }];
        let space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let mut rhs = vec![0.0; space.num_global_dofs(&mesh)];
        let pressure = 23.0;
        assemble_pressure_tractions_for_space(
            &mesh,
            &space,
            &[PressureTraction {
                marker: 8,
                pressure_pa: pressure,
            }],
            &mut rhs,
        );
        let velocity: Vec<f64> = (0..space.num_global_dofs(&mesh))
            .map(|dof| [1.0, 2.0, 3.0][dof % 3])
            .collect();
        let work = rhs.iter().zip(&velocity).map(|(f, u)| f * u).sum::<f64>();
        let independent_flux = 3.0;
        assert!((work + pressure * independent_flux).abs() < 1.0e-12);
    }
}
