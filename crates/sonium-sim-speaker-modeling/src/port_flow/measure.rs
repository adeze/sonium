//! Boundary flux and average-pressure measurement over marked faces.
//!
//! Used by the port-flow coupling (volume velocity through the duct caps) and
//! by the THD/IMD analysis (mean pressure at the terminations).

use sonium_fem::mesh::topology::MeshTopology;
use sonium_fem::mesh::{BoundaryFace, Mesh};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};

/// ∫_Γ u·n dΓ over the P1 faces carrying `marker`.
///
/// `velocity` is interleaved as `dof = node * dim + comp`. Exact for
/// piecewise-linear u on P1 faces: every face node carries an equal share of
/// the face measure.
pub fn integrate_face_flux(mesh: &Mesh, marker: i32, velocity: &[f64], dim: usize) -> f64 {
    let mut flux = 0.0;
    for face in &mesh.boundaries {
        if face.marker != marker {
            continue;
        }
        let (n, measure) = outward_normal_and_measure(mesh, face);
        let share = measure / face.nodes.len() as f64;
        for &nd in &face.nodes {
            // A 2D edge only carries x/y components; its normal's z is zero,
            // so `dim.min(3)` is safe for both 2D and 3D layouts.
            for c in 0..dim.min(3) {
                flux += velocity[nd * dim + c] * n[c] * share;
            }
        }
    }
    flux
}

/// ∫_Γ u·n dΓ over the faces carrying `marker`, exact for the velocity
/// space's polynomial degree.
///
/// `velocity` is interleaved as `dof = scalar_dof * components + comp`, where
/// scalar DOFs are the mesh vertices followed by the P2 edge midpoints
/// (`mesh.num_nodes() + MeshTopology::edge_map[&(min(a, b), max(a, b))]`).
///
/// - P1: delegates to [`integrate_face_flux`] (nodal equal-share, exact for
///   piecewise-linear traces).
/// - P2: Simpson on 2D edges — `L·(u_a + 4·u_m + u_b)/6`; on 3D triangular
///   faces the P2 trace integrates with zero vertex weights and `A/3` on each
///   edge midpoint.
pub fn integrate_face_flux_for_space(
    mesh: &Mesh,
    v_space: &VectorH1Space,
    marker: i32,
    velocity: &[f64],
) -> f64 {
    let dim = v_space.num_components();
    if v_space.polynomial_degree() == 1 {
        return integrate_face_flux(mesh, marker, velocity, dim);
    }
    assert_eq!(
        v_space.polynomial_degree(),
        2,
        "integrate_face_flux_for_space supports P1/P2 velocity spaces"
    );
    let topology = MeshTopology::build(mesh);
    let n_nodes = mesh.num_nodes();
    let edge_midpoint_dof = |a: usize, b: usize| -> usize {
        let key = (a.min(b), a.max(b));
        n_nodes + topology.edge_map[&key]
    };
    let mut flux = 0.0;
    for face in &mesh.boundaries {
        if face.marker != marker {
            continue;
        }
        let (n, measure) = outward_normal_and_measure(mesh, face);
        match face.nodes.len() {
            2 => {
                // Simpson: L·(u_a + 4·u_m + u_b)/6, exact for quadratic traces.
                let (a, b) = (face.nodes[0], face.nodes[1]);
                let m = edge_midpoint_dof(a, b);
                for c in 0..dim.min(3) {
                    let u = (velocity[a * dim + c]
                        + 4.0 * velocity[m * dim + c]
                        + velocity[b * dim + c])
                        / 6.0;
                    flux += u * n[c] * measure;
                }
            }
            3 => {
                // P2 tri face: vertex weights vanish; each of the three edge
                // midpoints carries A/3.
                for k in 0..3 {
                    let m = edge_midpoint_dof(face.nodes[k], face.nodes[(k + 1) % 3]);
                    for c in 0..dim.min(3) {
                        flux += velocity[m * dim + c] * n[c] * (measure / 3.0);
                    }
                }
            }
            _ => {}
        }
    }
    flux
}

/// Area-weighted mean pressure over the P1 faces carrying `marker`.
///
/// Each face contributes its nodal average weighted by its measure. Returns
/// 0.0 when no face carries the marker.
pub fn average_face_pressure(mesh: &Mesh, marker: i32, pressure: &[f64]) -> f64 {
    let mut weighted = 0.0;
    let mut measure_sum = 0.0;
    for face in &mesh.boundaries {
        if face.marker != marker {
            continue;
        }
        let (_, measure) = outward_normal_and_measure(mesh, face);
        let p_face: f64 =
            face.nodes.iter().map(|&nd| pressure[nd]).sum::<f64>() / face.nodes.len() as f64;
        weighted += p_face * measure;
        measure_sum += measure;
    }
    if measure_sum > 0.0 {
        weighted / measure_sum
    } else {
        0.0
    }
}

/// Unit outward normal and measure (length in 2D, area in 3D) of a boundary
/// face, signed to point away from the owning element's centroid.
///
/// Mirrors `outward_normal_2d` / `outward_normal_tri` in sonium-fem
/// `physics/stokes/traction.rs`; duplicated here because the crate-boundary
/// discipline keeps the modeling crate from adding sonium-fem APIs.
pub(crate) fn outward_normal_and_measure(mesh: &Mesh, face: &BoundaryFace) -> ([f64; 3], f64) {
    let owner = &mesh.elements[face.element_idx];
    let mut centroid = [0.0; 3];
    for &nd in &owner.nodes {
        let p = &mesh.nodes[nd];
        centroid[0] += p.x;
        centroid[1] += p.y;
        centroid[2] += p.z;
    }
    let inv = 1.0 / owner.nodes.len() as f64;
    let centroid = [centroid[0] * inv, centroid[1] * inv, centroid[2] * inv];
    let at = |i: usize| &mesh.nodes[face.nodes[i]];
    match face.nodes.len() {
        2 => {
            let (a, b) = (at(0), at(1));
            let (tx, ty) = (b.x - a.x, b.y - a.y);
            let len = (tx * tx + ty * ty).sqrt();
            // Rotate the tangent 90° counter-clockwise, then flip to outward.
            let mut n = [ty / len, -tx / len, 0.0];
            let mid = [0.5 * (a.x + b.x), 0.5 * (a.y + b.y)];
            if (mid[0] - centroid[0]) * n[0] + (mid[1] - centroid[1]) * n[1] < 0.0 {
                n = [-n[0], -n[1], 0.0];
            }
            (n, len)
        }
        3 => {
            let (p0, p1, p2) = (at(0), at(1), at(2));
            let e1 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
            let e2 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
            let cross = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let norm = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            let mut n = [cross[0] / norm, cross[1] / norm, cross[2] / norm];
            let mid = [
                (p0.x + p1.x + p2.x) / 3.0,
                (p0.y + p1.y + p2.y) / 3.0,
                (p0.z + p1.z + p2.z) / 3.0,
            ];
            let dot: f64 = (0..3).map(|i| (mid[i] - centroid[i]) * n[i]).sum();
            if dot < 0.0 {
                n = [-n[0], -n[1], -n[2]];
            }
            // |e1 × e2| = 2·A for a triangle.
            (n, 0.5 * norm)
        }
        _ => ([0.0; 3], 0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::super::geometry::{
        MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, SlotPortGeometry, build_slot_port_mesh,
    };
    use super::*;
    use sonium_fem::basis::PolynomialDegree;
    use sonium_fem::mesh::unit_cube_tetrahedra;

    fn slot_mesh() -> Mesh {
        let geo = SlotPortGeometry {
            duct_length_m: 0.1,
            duct_height_m: 0.02,
            chamber_length_m: 0.05,
            chamber_height_m: 0.08,
            mesh_size_m: 0.01,
        };
        build_slot_port_mesh(&geo).unwrap()
    }

    #[test]
    fn uniform_flow_gives_exact_flux_and_pressure() {
        let mesh = slot_mesh();
        let n = mesh.num_nodes();
        // Uniform u = (2.0, 0): flux through box cap (x = −0.05, outward normal −x)
        // = −2.0 * cap_height; cap height = chamber_height = 0.08.
        let velocity: Vec<f64> = (0..n).flat_map(|_| [2.0, 0.0]).collect();
        let flux = integrate_face_flux(&mesh, MARKER_BOX_CAP, &velocity, 2);
        assert!((flux + 0.16).abs() < 1e-12, "flux {flux}");
        let pressure: Vec<f64> = vec![333.0; n];
        assert!(
            (average_face_pressure(&mesh, MARKER_EXTERIOR_CAP, &pressure) - 333.0).abs() < 1e-12
        );
    }

    #[test]
    fn uniform_flow_gives_exact_flux_on_p2_space() {
        let mesh = slot_mesh();
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let n_dofs = v_space.num_global_dofs(&mesh);
        // Same uniform u = (2.0, 0), now including edge-midpoint DOFs:
        // Simpson/midpoint rules are exact for a constant trace.
        let velocity: Vec<f64> = (0..n_dofs / 2).flat_map(|_| [2.0, 0.0]).collect();
        let flux = integrate_face_flux_for_space(&mesh, &v_space, MARKER_BOX_CAP, &velocity);
        assert!((flux + 0.16).abs() < 1e-12, "flux {flux}");
        // P1 delegation path agrees with the legacy function.
        let p1_space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let n1 = mesh.num_nodes();
        let velocity_p1: Vec<f64> = (0..n1).flat_map(|_| [2.0, 0.0]).collect();
        let flux_p1 = integrate_face_flux_for_space(&mesh, &p1_space, MARKER_BOX_CAP, &velocity_p1);
        assert!((flux_p1 + 0.16).abs() < 1e-12, "flux_p1 {flux_p1}");
    }

    #[test]
    fn p2_flux_uses_simpson_midpoint_weight() {
        let mesh = slot_mesh();
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let topology = MeshTopology::build(&mesh);
        let mut velocity = vec![0.0; v_space.num_global_dofs(&mesh)];

        // Zero at the cap vertices and u_x = 2 at every cap-edge midpoint.
        // Simpson gives -L * (0 + 4*2 + 0) / 6 on the box cap, whose total
        // length is 0.08 m and whose outward normal is -x.
        for face in mesh
            .boundaries
            .iter()
            .filter(|face| face.marker == MARKER_BOX_CAP)
        {
            let a = face.nodes[0];
            let b = face.nodes[1];
            let edge = (a.min(b), a.max(b));
            let midpoint = mesh.num_nodes() + topology.edge_map[&edge];
            velocity[midpoint * 2] = 2.0;
        }

        let flux = integrate_face_flux_for_space(&mesh, &v_space, MARKER_BOX_CAP, &velocity);
        let expected = -0.08 * 8.0 / 6.0;
        assert!(
            (flux - expected).abs() < 1e-12,
            "flux {flux}, expected {expected}"
        );
    }

    #[test]
    fn p2_triangular_face_flux_uses_midpoint_weights_and_outward_sign() {
        const MARKER: i32 = 77;
        let mut mesh = unit_cube_tetrahedra(1);
        for face in &mut mesh.boundaries {
            face.marker = 0;
        }
        let face_index = mesh
            .boundaries
            .iter()
            .position(|face| face.nodes.iter().all(|&node| mesh.nodes[node].x == 0.0))
            .expect("unit cube must have a triangular face on x = 0");
        mesh.boundaries[face_index].marker = MARKER;

        let face_nodes = mesh.boundaries[face_index].nodes.clone();
        assert_eq!(face_nodes.len(), 3);
        let topology = MeshTopology::build(&mesh);
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let mut velocity = vec![0.0; v_space.num_global_dofs(&mesh)];

        // The selected triangle has area 1/2 and outward normal -x. Its three
        // edge-midpoint x velocities are 1, 2, and 3 while all vertex values
        // remain zero. The exact P2 face rule is A/3 per midpoint, hence
        // flux = -(1/2)/3 * (1 + 2 + 3) = -1.
        for (edge_index, midpoint_velocity) in [1.0, 2.0, 3.0].into_iter().enumerate() {
            let a = face_nodes[edge_index];
            let b = face_nodes[(edge_index + 1) % 3];
            let edge = (a.min(b), a.max(b));
            let midpoint = mesh.num_nodes() + topology.edge_map[&edge];
            velocity[midpoint * 3] = midpoint_velocity;
        }

        let flux = integrate_face_flux_for_space(&mesh, &v_space, MARKER, &velocity);
        assert!((flux + 1.0).abs() < 1e-12, "flux {flux}, expected -1");
    }
}
