//! Gap function computation for node-to-surface contact
//!
//! Computes the closest-point projection of a node onto a triangle in 3D,
//! the signed gap distance, and the outward surface normal.

/// Result of projecting a point onto a triangle.
#[derive(Debug, Clone, Copy)]
pub struct ProjectionResult {
    /// Closest point on the triangle (in physical coordinates).
    pub closest: [f64; 3],
    /// Barycentric coordinates of the projection on the triangle (xi, eta).
    /// The third coordinate is `1 - xi - eta`.
    pub xi: f64,
    pub eta: f64,
    /// Reference/material barycentric coordinates used for tangential slip.
    pub reference_xi: f64,
    pub reference_eta: f64,
    /// Signed gap: positive = separated, negative = penetrating.
    pub gap: f64,
    /// Unit outward normal at the projection point.
    pub normal: [f64; 3],
    /// Distance from the query point to the closest point.
    pub distance: f64,
}

/// A detected contact pair with all geometric data needed for force computation.
#[derive(Debug, Clone)]
pub struct ContactPair {
    /// Parent mesh node index of the slave (contacting) node.
    pub slave_node: usize,
    /// Triangle index in the ContactSurface.
    pub master_tri: usize,
    /// Parent mesh node indices of the master triangle vertices.
    pub master_nodes: [usize; 3],
    /// Barycentric coordinates of the projection point on the master triangle.
    pub xi: f64,
    pub eta: f64,
    pub reference_xi: f64,
    pub reference_eta: f64,
    /// Signed gap (positive = separated, negative = penetrating).
    pub gap: f64,
    /// Unit outward normal at the contact point.
    pub normal: [f64; 3],
    /// Slave tributary area in m^2 (nodal area of the slave node).
    ///
    /// Tractions (Pa) integrate to nodal forces (N) against this area;
    /// the master scatter reuses the same weight so force is conserved.
    pub area: f64,
    /// Reference tangential relative position for the fixed material pair.
    ///
    /// The current tangential slip is measured against this value so a
    /// tangential penalty does not create friction merely because the two
    /// surfaces were initially separated.
    pub reference_relative_tangent: [f64; 3],
}

/// Project point `p` onto triangle defined by vertices `v0`, `v1`, `v2`.
///
/// Returns the closest point on the triangle (or its edges/vertices),
/// the barycentric coordinates, and the triangle normal.
///
/// The normal is computed as the cross product `(v1-v0) x (v2-v0)`, normalized.
/// The gap sign convention: positive when `p` is on the normal side (separated),
/// negative when penetrating.
pub fn project_point_to_triangle(
    p: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> ProjectionResult {
    // Edge vectors
    let e0 = sub(v1, v0);
    let e1 = sub(v2, v0);
    let v = sub(p, v0);

    // Triangle normal (not yet normalized)
    let n_raw = cross(e0, e1);
    let n_len = norm(n_raw);

    // Degenerate triangle check
    if n_len < 1e-30 {
        return ProjectionResult {
            closest: v0,
            xi: 0.0,
            eta: 0.0,
            reference_xi: 0.0,
            reference_eta: 0.0,
            gap: 0.0,
            normal: [0.0, 0.0, 1.0],
            distance: norm(sub(p, v0)),
        };
    }

    let normal = scale(n_raw, 1.0 / n_len);

    // Parametric projection onto the triangle plane
    let d00 = dot(e0, e0);
    let d01 = dot(e0, e1);
    let d11 = dot(e1, e1);
    let d20 = dot(v, e0);
    let d21 = dot(v, e1);

    let denom = d00 * d11 - d01 * d01;
    let mut xi = (d11 * d20 - d01 * d21) / denom;
    let mut eta = (d00 * d21 - d01 * d20) / denom;

    // Clamp to triangle: project onto nearest edge/vertex if outside
    if xi < 0.0 || eta < 0.0 || xi + eta > 1.0 {
        // Find the closest point among edges and vertices
        let (clamped_xi, clamped_eta) = clamp_to_triangle(xi, eta, v0, v1, v2, p);
        xi = clamped_xi;
        eta = clamped_eta;
    }

    let zeta = 1.0 - xi - eta;
    let closest = [
        zeta * v0[0] + xi * v1[0] + eta * v2[0],
        zeta * v0[1] + xi * v1[1] + eta * v2[1],
        zeta * v0[2] + xi * v1[2] + eta * v2[2],
    ];

    let diff = sub(p, closest);
    let distance = norm(diff);
    let gap = dot(diff, normal);

    ProjectionResult {
        closest,
        xi,
        eta,
        reference_xi: xi,
        reference_eta: eta,
        gap,
        normal,
        distance,
    }
}

/// Clamp barycentric coordinates to the triangle, finding the closest point
/// on the boundary (edges or vertices) when the projection falls outside.
fn clamp_to_triangle(
    _xi: f64,
    _eta: f64,
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
    p: [f64; 3],
) -> (f64, f64) {
    // Test all three edges and three vertices, pick closest
    let mut best_dist = f64::MAX;
    let mut best_xi = 0.0;
    let mut best_eta = 0.0;

    // Edge v0-v1: eta=0, xi in [0,1]
    {
        let t = clamp_edge_param(v0, v1, p);
        let xi = t;
        let eta = 0.0;
        let zeta = 1.0 - xi;
        let cp = [
            zeta * v0[0] + xi * v1[0],
            zeta * v0[1] + xi * v1[1],
            zeta * v0[2] + xi * v1[2],
        ];
        let d = dist_sq(p, cp);
        if d < best_dist {
            best_dist = d;
            best_xi = xi;
            best_eta = eta;
        }
    }

    // Edge v0-v2: xi=0, eta in [0,1]
    {
        let t = clamp_edge_param(v0, v2, p);
        let xi = 0.0;
        let eta = t;
        let zeta = 1.0 - eta;
        let cp = [
            zeta * v0[0] + eta * v2[0],
            zeta * v0[1] + eta * v2[1],
            zeta * v0[2] + eta * v2[2],
        ];
        let d = dist_sq(p, cp);
        if d < best_dist {
            best_dist = d;
            best_xi = xi;
            best_eta = eta;
        }
    }

    // Edge v1-v2: xi + eta = 1, parameterize as v1 + t*(v2-v1), t in [0,1]
    {
        let t = clamp_edge_param(v1, v2, p);
        let xi = 1.0 - t;
        let eta = t;
        let cp = [
            xi * v1[0] + eta * v2[0],
            xi * v1[1] + eta * v2[1],
            xi * v1[2] + eta * v2[2],
        ];
        let d = dist_sq(p, cp);
        if d < best_dist {
            // last edge — no need to update best_dist
            best_xi = xi;
            best_eta = eta;
        }
    }

    (best_xi, best_eta)
}

/// Parameter t in [0,1] for closest point on segment a->b to point p.
fn clamp_edge_param(a: [f64; 3], b: [f64; 3], p: [f64; 3]) -> f64 {
    let ab = sub(b, a);
    let ap = sub(p, a);
    let denom = dot(ab, ab);
    if denom < 1e-30 {
        return 0.0;
    }
    (dot(ap, ab) / denom).clamp(0.0, 1.0)
}

// --- 3D vector helpers ---

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn dist_sq(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_interior() {
        // Triangle in z=0 plane
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];

        // Point directly above centroid
        let p = [1.0 / 3.0, 1.0 / 3.0, 1.0];
        let r = project_point_to_triangle(p, v0, v1, v2);

        assert!((r.gap - 1.0).abs() < 1e-12, "gap={}", r.gap);
        assert!((r.normal[2] - 1.0).abs() < 1e-12);
        assert!(r.xi > 0.0 && r.eta > 0.0 && r.xi + r.eta < 1.0);
    }

    #[test]
    fn test_project_below_triangle() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];

        // Point below the triangle
        let p = [0.25, 0.25, -0.5];
        let r = project_point_to_triangle(p, v0, v1, v2);

        assert!(
            r.gap < 0.0,
            "gap should be negative (penetrating): {}",
            r.gap
        );
        assert!((r.gap - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn test_project_onto_edge() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];

        // Point outside the triangle, closest to edge v0-v1
        let p = [0.5, -1.0, 0.0];
        let r = project_point_to_triangle(p, v0, v1, v2);

        // Closest point should be on edge v0-v1 at y=0
        assert!((r.closest[1]).abs() < 1e-12, "closest y={}", r.closest[1]);
        assert!(r.closest[0] > 0.0 && r.closest[0] < 1.0);
    }

    #[test]
    fn test_project_onto_vertex() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];

        // Point far from vertex v0
        let p = [-1.0, -1.0, 0.0];
        let r = project_point_to_triangle(p, v0, v1, v2);

        // Closest point should be v0
        assert!((r.closest[0]).abs() < 1e-12);
        assert!((r.closest[1]).abs() < 1e-12);
        assert!((r.closest[2]).abs() < 1e-12);
    }

    #[test]
    fn test_barycentric_sum() {
        let v0 = [1.0, 0.0, 0.0];
        let v1 = [0.0, 2.0, 0.0];
        let v2 = [0.0, 0.0, 3.0];

        let p = [0.3, 0.5, 0.8];
        let r = project_point_to_triangle(p, v0, v1, v2);

        let zeta = 1.0 - r.xi - r.eta;
        assert!(
            r.xi >= -1e-12 && r.eta >= -1e-12 && zeta >= -1e-12,
            "barycentric: xi={}, eta={}, zeta={}",
            r.xi,
            r.eta,
            zeta
        );
        assert!(
            (r.xi + r.eta + zeta - 1.0).abs() < 1e-12,
            "sum={}",
            r.xi + r.eta + zeta
        );
    }
}
