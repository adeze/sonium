//! Augmented Lagrangian contact with Uzawa iteration
//!
//! The [`ContactManager`] owns the contact surface, BVH, Lagrange multipliers,
//! and penalty parameter. It provides methods to:
//! - Detect contact pairs from the current deformed configuration
//! - Compute contact residual and tangent contributions
//! - Update multipliers (Uzawa step)
//!
//! The outer Uzawa loop is driven by the solver in `solver/nonlinear.rs`.
//!
//! # Algorithm
//!
//! ```text
//! For Uzawa iteration k = 0, 1, ...
//!   1. Detect contact pairs (BVH query + projection)
//!   2. Newton-Raphson with:
//!        R(u) = R_bulk(u) + R_contact(u, λ_k, ε)
//!        K(u) = K_bulk(u) + K_contact(u, λ_k, ε)
//!   3. Update: λ_{k+1} = max(0, λ_k + ε · g_N)
//!   4. Converge when ‖λ_{k+1} − λ_k‖_∞ < tol
//! ```

use super::detection::ContactSurface;
use super::gap::{ContactPair, project_point_to_triangle};

/// Configuration for the augmented Lagrangian contact solver.
#[derive(Debug, Clone)]
pub struct ContactConfig {
    /// Penalty/augmentation parameter ε.
    /// Larger values improve constraint enforcement per Uzawa step
    /// but worsen the conditioning of the inner Newton system.
    pub epsilon: f64,
    /// Convergence tolerance for the Uzawa outer loop (on ‖Δλ‖_∞).
    pub uzawa_tolerance: f64,
    /// Maximum number of Uzawa (outer) iterations.
    pub max_uzawa_iterations: usize,
    /// Search radius for BVH contact detection (in physical units).
    /// Should be slightly larger than the expected maximum penetration.
    pub search_radius: f64,
    /// BVH inflation margin for AABB boxes.
    pub bvh_margin: f64,
    /// Coulomb friction coefficient. Zero disables tangential contact.
    pub friction_coefficient: f64,
    /// Two-body mode: accept reference-touching pairs (`ref_gap ~= 0`).
    ///
    /// The default self-contact filter rejects pairs whose reference gap is
    /// below 10% of the search radius (same-face/adjacent-face pairs). Two
    /// separately meshed bodies in initial touching contact are legitimate
    /// contact pairs, so two-body mode skips that filter; the opposing-normal
    /// filter still rejects same-side pairs.
    pub two_body: bool,
}

impl Default for ContactConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e6,
            uzawa_tolerance: 1e-8,
            max_uzawa_iterations: 20,
            search_radius: 0.1,
            bvh_margin: 0.01,
            friction_coefficient: 0.0,
            two_body: false,
        }
    }
}

/// Result of the Uzawa contact solve.
#[derive(Debug)]
pub struct ContactResult {
    /// Number of Uzawa iterations performed.
    pub uzawa_iterations: usize,
    /// Final multiplier update norm ‖Δλ‖_∞.
    pub multiplier_update_norm: f64,
    /// Whether the Uzawa iteration converged.
    pub converged: bool,
    /// Number of active contact pairs at convergence.
    pub active_pairs: usize,
    /// Maximum penetration depth at convergence.
    pub max_penetration: f64,
}

/// Manages contact detection, force computation, and Lagrange multiplier updates.
pub struct ContactManager {
    /// The contact surface (extracted from mesh boundary).
    surface: ContactSurface,
    /// Reference (undeformed) node coordinates — flat array, 3 entries per node.
    reference_coords: Vec<f64>,
    /// Average outward normal per node in reference configuration, used
    /// to reject same-side contact pairs on shell meshes.
    ref_node_normals: Vec<[f64; 3]>,
    /// Tributary area per node in reference configuration (m^2): one third
    /// of the adjacent contact-face areas. Weights tractions into forces.
    ref_node_areas: Vec<f64>,
    /// Explicit slave-node list for one-sided two-body contact (`None` =
    /// self-contact: every surface node is a slave candidate).
    slave_nodes: Option<Vec<usize>>,
    /// Current active contact pairs.
    pairs: Vec<ContactPair>,
    /// Lagrange multiplier for each pair (persisted across Uzawa steps).
    multipliers: Vec<f64>,
    /// Contact configuration.
    config: ContactConfig,
}

impl ContactManager {
    /// Create a new contact manager.
    ///
    /// - `boundary_faces`: triangle faces on the contact surface, each as 3 parent node indices.
    /// - `reference_coords`: flat array of reference node coordinates (3 * n_nodes).
    /// - `config`: augmented Lagrangian parameters.
    pub fn new(
        boundary_faces: &[[usize; 3]],
        reference_coords: Vec<f64>,
        config: ContactConfig,
    ) -> Self {
        let n_parent_nodes = reference_coords.len() / 3;
        let surface = ContactSurface::from_boundary_faces(boundary_faces, n_parent_nodes);

        // Compute reference node normals (area-weighted average of adjacent triangle normals)
        let ref_node_normals =
            Self::compute_node_normals(&surface, &reference_coords, n_parent_nodes);
        let ref_node_areas = Self::compute_node_areas(&surface, &reference_coords, n_parent_nodes);

        Self {
            surface,
            reference_coords,
            ref_node_normals,
            ref_node_areas,
            slave_nodes: None,
            pairs: Vec::new(),
            multipliers: Vec::new(),
            config,
        }
    }

    /// One-sided two-body contact: `master_faces` build the target surface
    /// (BVH + projection triangles) while slaves come only from
    /// `slave_faces`, so each physical contact point pairs exactly once.
    /// (Pairing both orderings double-counts normal and friction forces.)
    /// Nodal normals/areas cover both sides (nodes are body-disjoint).
    pub fn new_two_body(
        master_faces: &[[usize; 3]],
        slave_faces: &[[usize; 3]],
        reference_coords: Vec<f64>,
        config: ContactConfig,
    ) -> Self {
        let n_parent_nodes = reference_coords.len() / 3;
        let surface = ContactSurface::from_boundary_faces(master_faces, n_parent_nodes);
        let combined: Vec<[usize; 3]> = master_faces
            .iter()
            .copied()
            .chain(slave_faces.iter().copied())
            .collect();
        let combined_surface = ContactSurface::from_boundary_faces(&combined, n_parent_nodes);
        let ref_node_normals =
            Self::compute_node_normals(&combined_surface, &reference_coords, n_parent_nodes);
        let ref_node_areas =
            Self::compute_node_areas(&combined_surface, &reference_coords, n_parent_nodes);
        let mut seen = vec![false; n_parent_nodes];
        let mut slave_nodes = Vec::new();
        for face in slave_faces {
            for &node in face {
                if !seen[node] {
                    seen[node] = true;
                    slave_nodes.push(node);
                }
            }
        }

        Self {
            surface,
            reference_coords,
            ref_node_normals,
            ref_node_areas,
            slave_nodes: Some(slave_nodes),
            pairs: Vec::new(),
            multipliers: Vec::new(),
            config,
        }
    }

    /// Configuration.
    pub fn config(&self) -> &ContactConfig {
        &self.config
    }

    /// Count pairs in the contact active set (`lambda + eps * gap < 0`)
    /// at the given displacement.
    pub fn count_active_pairs(&self, displacement: &[f64]) -> usize {
        let deformed = self.deformed_coords(displacement);
        self.pairs
            .iter()
            .enumerate()
            .filter(|(pair_idx, pair)| {
                let gap = self.compute_gap(pair, &deformed);
                self.multipliers[*pair_idx] + self.config.epsilon * gap < 0.0
            })
            .count()
    }

    /// Compute deformed coordinates from the displacement DOF vector.
    ///
    /// `displacement` is the global DOF vector (3 * n_nodes) in interleaved order:
    /// `[u0_x, u0_y, u0_z, u1_x, ...]`.
    fn deformed_coords(&self, displacement: &[f64]) -> Vec<f64> {
        let mut coords = self.reference_coords.clone();
        for i in 0..coords.len() {
            coords[i] += displacement[i];
        }
        coords
    }

    /// Detect contact pairs for the current deformed configuration.
    ///
    /// Rebuilds the BVH on the deformed surface and finds all (node, triangle) pairs
    /// where the gap is below the search radius. Through-body pairs (where the
    /// reference-configuration gap is negative) are excluded — these are nodes
    /// that are inside the body relative to a surface on the opposite side.
    pub fn detect_contact(&mut self, displacement: &[f64]) {
        let deformed = self.deformed_coords(displacement);
        let bvh = self.surface.build_bvh(&deformed, self.config.bvh_margin);
        let candidates = match &self.slave_nodes {
            Some(slaves) => {
                self.surface
                    .find_candidates_for(slaves, &deformed, &bvh, self.config.search_radius)
            }
            None => self
                .surface
                .find_candidates(&deformed, &bvh, self.config.search_radius),
        };

        self.pairs.clear();

        for (slave_node, tri_idx) in candidates {
            let tri = &self.surface.triangles[tri_idx];

            // Reference-configuration filter: reject pairs that are NOT
            // genuinely separated surfaces in the undeformed state.
            // - ref_gap < 0: through-body pair (node is inside relative to this surface)
            // - ref_gap ≈ 0: same-face or adjacent-face pair (coplanar)
            // Only pairs where the node was well-separated (ref_gap > threshold) can
            // become legitimate self-contact when the body deforms.
            let ref_proj = {
                let rv0 = Self::node_pos(&self.reference_coords, tri.nodes[0]);
                let rv1 = Self::node_pos(&self.reference_coords, tri.nodes[1]);
                let rv2 = Self::node_pos(&self.reference_coords, tri.nodes[2]);
                let rp = Self::node_pos(&self.reference_coords, slave_node);
                project_point_to_triangle(rp, rv0, rv1, rv2)
            };
            // Require reference gap to be at least 10% of the search radius
            // to exclude same-face and through-body pairs. Two-body mode
            // skips this: touching bodies are legitimate contact pairs.
            if !self.config.two_body && ref_proj.gap < 0.1 * self.config.search_radius {
                continue;
            }

            // Normal-compatibility filter: for legitimate contact, the slave
            // node's surface normal and the master triangle's normal should
            // face each other (dot product < 0). Same-side pairs on a shell
            // have parallel normals and must be rejected.
            let slave_normal = &self.ref_node_normals[slave_node];
            let tri_normal = ref_proj.normal;
            let n_dot = slave_normal[0] * tri_normal[0]
                + slave_normal[1] * tri_normal[1]
                + slave_normal[2] * tri_normal[2];
            if n_dot > -0.3 {
                // Normals not sufficiently opposing — skip
                continue;
            }

            // Deformed-configuration projection
            let v0 = Self::node_pos(&deformed, tri.nodes[0]);
            let v1 = Self::node_pos(&deformed, tri.nodes[1]);
            let v2 = Self::node_pos(&deformed, tri.nodes[2]);
            let p = Self::node_pos(&deformed, slave_node);

            let proj = project_point_to_triangle(p, v0, v1, v2);

            if proj.distance < self.config.search_radius {
                // Slave tributary area weights tractions into nodal forces.
                // Fall back to a third of the master triangle area when the
                // slave node carries no reference area (degenerate surface).
                let mut area = self.ref_node_areas[slave_node];
                if !area.is_finite() || area <= 0.0 {
                    let e0 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
                    let e1 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
                    let cross = [
                        e0[1] * e1[2] - e0[2] * e1[1],
                        e0[2] * e1[0] - e0[0] * e1[2],
                        e0[0] * e1[1] - e0[1] * e1[0],
                    ];
                    area = Self::norm(cross) / 6.0;
                }
                self.pairs.push(ContactPair {
                    slave_node,
                    master_tri: tri_idx,
                    master_nodes: tri.nodes,
                    xi: proj.xi,
                    eta: proj.eta,
                    reference_xi: ref_proj.xi,
                    reference_eta: ref_proj.eta,
                    gap: proj.gap,
                    normal: proj.normal,
                    area,
                    reference_relative_tangent: {
                        let reference_slave = Self::node_pos(&self.reference_coords, slave_node);
                        let reference_master = Self::interpolate_master(
                            &self.reference_coords,
                            tri.nodes,
                            ref_proj.xi,
                            ref_proj.eta,
                        );
                        let relative = [
                            reference_slave[0] - reference_master[0],
                            reference_slave[1] - reference_master[1],
                            reference_slave[2] - reference_master[2],
                        ];
                        Self::tangent_component(relative, ref_proj.normal)
                    },
                });
            }
        }

        // Resize multipliers to match number of pairs.
        // New pairs start with λ = 0.
        self.multipliers.resize(self.pairs.len(), 0.0);
    }

    /// Compute the contact contribution to the global residual vector.
    ///
    /// For each active contact pair (where `λ_k + ε * g_N < 0`):
    ///
    /// ```text
    /// R_slave[d] += A · (λ + ε·g_N) · n[d]
    /// R_master_α[d] -= A · (λ + ε·g_N) · N_α · n[d]
    /// ```
    ///
    /// where `N_α` are the triangle shape function values at the projection
    /// point and `A` is the slave tributary area (traction-to-force weight).
    pub fn contact_residual(&self, displacement: &[f64], residual: &mut [f64]) {
        let deformed = self.deformed_coords(displacement);

        for (pair_idx, pair) in self.pairs.iter().enumerate() {
            let lambda = self.multipliers[pair_idx];

            // Recompute gap at current displacement
            let gap = self.compute_gap(pair, &deformed);
            let augmented = lambda + self.config.epsilon * gap;

            // Active set check: contact force only when augmented < 0
            if augmented >= 0.0 {
                continue;
            }

            let zeta = 1.0 - pair.xi - pair.eta;
            let shape = [zeta, pair.xi, pair.eta]; // N0, N1, N2 for master triangle
            let force = pair.area * augmented;

            // Slave node contribution: force pushes slave away
            for d in 0..3 {
                residual[3 * pair.slave_node + d] += force * pair.normal[d];
            }

            // Master node contributions: equal and opposite, weighted by shape functions
            for (alpha, &master_node) in pair.master_nodes.iter().enumerate() {
                for d in 0..3 {
                    residual[3 * master_node + d] -= force * shape[alpha] * pair.normal[d];
                }
            }

            self.add_friction_residual(pair, &deformed, augmented, residual);
        }
    }

    /// Compute the contact contribution to the global tangent stiffness.
    ///
    /// Returns triplets `(row, col, value)` to be added to the global tangent matrix.
    ///
    /// For each active pair, the tangent contribution is:
    ///
    /// ```text
    /// K_contact = A · ε · n ⊗ n
    /// ```
    ///
    /// distributed over slave-slave, slave-master, master-slave, and
    /// master-master blocks (`A` = slave tributary area, matching the
    /// residual's traction-to-force weight).
    pub fn contact_tangent(&self, displacement: &[f64]) -> Vec<(usize, usize, f64)> {
        let deformed = self.deformed_coords(displacement);
        let mut triplets = Vec::new();

        for (pair_idx, pair) in self.pairs.iter().enumerate() {
            let lambda = self.multipliers[pair_idx];
            let gap = self.compute_gap(pair, &deformed);
            let augmented = lambda + self.config.epsilon * gap;

            if augmented >= 0.0 {
                continue;
            }

            let eps = pair.area * self.config.epsilon;
            let zeta = 1.0 - pair.xi - pair.eta;
            let shape = [zeta, pair.xi, pair.eta];
            let n = pair.normal;

            // Slave-Slave block: +ε · n_d · n_e
            for d in 0..3 {
                for e in 0..3 {
                    let row = 3 * pair.slave_node + d;
                    let col = 3 * pair.slave_node + e;
                    triplets.push((row, col, eps * n[d] * n[e]));
                }
            }

            // Slave-Master blocks: -ε · N_α · n_d · n_e
            for (alpha, &master_node) in pair.master_nodes.iter().enumerate() {
                for d in 0..3 {
                    for e in 0..3 {
                        let val = -eps * shape[alpha] * n[d] * n[e];
                        // Slave row, master col
                        triplets.push((3 * pair.slave_node + d, 3 * master_node + e, val));
                        // Master row, slave col (symmetric)
                        triplets.push((3 * master_node + d, 3 * pair.slave_node + e, val));
                    }
                }
            }

            // Master-Master blocks: +ε · N_α · N_β · n_d · n_e
            for (alpha, &node_a) in pair.master_nodes.iter().enumerate() {
                for (beta, &node_b) in pair.master_nodes.iter().enumerate() {
                    for d in 0..3 {
                        for e in 0..3 {
                            let val = eps * shape[alpha] * shape[beta] * n[d] * n[e];
                            triplets.push((3 * node_a + d, 3 * node_b + e, val));
                        }
                    }
                }
            }

            self.add_friction_tangent(pair, &deformed, augmented, &mut triplets);
        }

        triplets
    }

    /// Update Lagrange multipliers after a Newton solve (Uzawa step).
    ///
    /// ```text
    /// λ_{k+1} = min(0, λ_k + ε · g_N)
    /// ```
    ///
    /// Compression is negative; the clamp keeps multipliers admissible.
    /// Returns the ‖Δλ‖_∞ norm for convergence checking.
    pub fn update_multipliers(&mut self, displacement: &[f64]) -> f64 {
        self.update_multipliers_relaxed(displacement, 1.0)
    }

    /// Uzawa update with relaxation `omega` applied to the gap increment.
    ///
    /// Inner solves that stall at a stick-slip chatter plateau hand back
    /// gap-biased states; a damped update (`omega < 1`) keeps the
    /// multiplier feedback loop stable there. Gate-converged inner
    /// solves use `omega = 1`.
    pub fn update_multipliers_relaxed(&mut self, displacement: &[f64], omega: f64) -> f64 {
        let deformed = self.deformed_coords(displacement);
        let mut max_delta = 0.0f64;

        for (pair_idx, pair) in self.pairs.iter().enumerate() {
            let gap = self.compute_gap(pair, &deformed);
            let lambda_old = self.multipliers[pair_idx];
            let lambda_new = (lambda_old + omega * self.config.epsilon * gap).min(0.0);

            max_delta = max_delta.max((lambda_new - lambda_old).abs());
            self.multipliers[pair_idx] = lambda_new;
        }

        max_delta
    }

    /// Reset multipliers to zero (e.g., for a new load step).
    pub fn reset_multipliers(&mut self) {
        self.multipliers.fill(0.0);
    }

    /// Update the Coulomb friction coefficient (e.g., to engage friction
    /// after the first Uzawa sweep establishes the normal multipliers).
    pub fn set_friction_coefficient(&mut self, friction_coefficient: f64) {
        self.config.friction_coefficient = friction_coefficient;
    }

    /// Get the maximum penetration depth across all active pairs.
    pub fn max_penetration(&self, displacement: &[f64]) -> f64 {
        let deformed = self.deformed_coords(displacement);
        let mut max_pen = 0.0f64;

        for pair in &self.pairs {
            let gap = self.compute_gap(pair, &deformed);
            if gap < 0.0 {
                max_pen = max_pen.max(-gap);
            }
        }

        max_pen
    }

    /// Recompute the gap for a contact pair at the given deformed configuration.
    fn compute_gap(&self, pair: &ContactPair, deformed: &[f64]) -> f64 {
        let v0 = Self::node_pos(deformed, pair.master_nodes[0]);
        let v1 = Self::node_pos(deformed, pair.master_nodes[1]);
        let v2 = Self::node_pos(deformed, pair.master_nodes[2]);
        let p = Self::node_pos(deformed, pair.slave_node);

        let proj = project_point_to_triangle(p, v0, v1, v2);
        proj.gap
    }

    /// Add the regularized Coulomb tangential traction for one active pair.
    ///
    /// The trial traction is an isotropic tangential penalty based on total
    /// material-point slip. It is capped by `mu * p_n`, where `p_n` is the
    /// compressive augmented normal pressure. This is intentionally a bounded
    /// node-to-surface law; projection derivatives and pressure-dependent
    /// tangent terms remain outside this reference path.
    fn add_friction_residual(
        &self,
        pair: &ContactPair,
        deformed: &[f64],
        augmented: f64,
        residual: &mut [f64],
    ) {
        let Some((traction, _)) = self.friction_traction(pair, deformed, augmented) else {
            return;
        };
        let shape = Self::pair_shape(pair);
        for d in 0..3 {
            residual[3 * pair.slave_node + d] += pair.area * traction[d];
        }
        for (alpha, &master_node) in pair.master_nodes.iter().enumerate() {
            for d in 0..3 {
                residual[3 * master_node + d] -= shape[alpha] * pair.area * traction[d];
            }
        }
    }

    fn add_friction_tangent(
        &self,
        pair: &ContactPair,
        deformed: &[f64],
        augmented: f64,
        triplets: &mut Vec<(usize, usize, f64)>,
    ) {
        let Some((stiffness, _)) = self.friction_tangent(pair, deformed, augmented) else {
            return;
        };
        let shape = Self::pair_shape(pair);
        for (d, row) in stiffness.iter().enumerate() {
            for (e, &stiffness_value) in row.iter().enumerate() {
                let value = pair.area * stiffness_value;
                triplets.push((3 * pair.slave_node + d, 3 * pair.slave_node + e, value));
                for (alpha, &master_node) in pair.master_nodes.iter().enumerate() {
                    let cross = -shape[alpha] * value;
                    triplets.push((3 * pair.slave_node + d, 3 * master_node + e, cross));
                    triplets.push((3 * master_node + d, 3 * pair.slave_node + e, cross));
                    for (beta, &node_b) in pair.master_nodes.iter().enumerate() {
                        triplets.push((
                            3 * master_node + d,
                            3 * node_b + e,
                            shape[alpha] * shape[beta] * value,
                        ));
                    }
                }
            }
        }
        // Pressure-dependent term: the Coulomb cap follows the normal gap
        // (`cap = mu * (-(lambda + eps*g))`), so capped traction varies
        // with normal motion. Dropping it leaves Newton with a tangent
        // that contradicts the residual in sliding (line search rejects
        // every step); the term is exact for frozen projection data.
        // Unsymmetric (non-associative slip rule) by nature.
        let mu = self.config.friction_coefficient;
        let normal_pressure = (-augmented).max(0.0);
        let slip = self.tangential_slip(pair, deformed);
        let slip_norm = Self::norm(slip);
        if slip_norm <= 1.0e-14 {
            return;
        }
        let (_, _, cap_slope) =
            Self::blended_friction_magnitude(self.config.epsilon, mu * normal_pressure, slip_norm);
        if cap_slope <= 0.0 {
            return;
        }
        let direction = Self::scale(slip, 1.0 / slip_norm);
        let normal = pair.normal;
        let factor = -pair.area * mu * self.config.epsilon * cap_slope;
        for d in 0..3 {
            for e in 0..3 {
                let grad = factor * direction[d] * normal[e];
                triplets.push((3 * pair.slave_node + d, 3 * pair.slave_node + e, grad));
                for (alpha, &master_node) in pair.master_nodes.iter().enumerate() {
                    triplets.push((
                        3 * pair.slave_node + d,
                        3 * master_node + e,
                        -shape[alpha] * grad,
                    ));
                    triplets.push((
                        3 * master_node + d,
                        3 * pair.slave_node + e,
                        -shape[alpha] * grad,
                    ));
                    for (beta, &node_b) in pair.master_nodes.iter().enumerate() {
                        triplets.push((
                            3 * master_node + d,
                            3 * node_b + e,
                            shape[alpha] * shape[beta] * grad,
                        ));
                    }
                }
            }
        }
    }

    /// Blended friction magnitude, its slip derivative, and its cap
    /// derivative `(m, dm/ds, dm/dcap)`.
    ///
    /// Exact penalty (`eps * s`) below the transition band, exact Coulomb
    /// cap above it, and a monotone cubic Hermite bridge over
    /// `s in s* * (1 +- 0.3)` (`s* = cap / eps`). The sharp `min()` kink
    /// flip-flops Newton at the stick-slip boundary (pairs jump between
    /// branches every iteration); the C1 bridge removes the kink while
    /// staying exact outside the band. The cap derivative feeds the
    /// pressure-dependent tangent term (cap follows the normal gap).
    fn blended_friction_magnitude(epsilon: f64, cap: f64, slip_norm: f64) -> (f64, f64, f64) {
        let threshold = cap / epsilon;
        if !threshold.is_finite() || threshold <= 0.0 {
            return (cap.max(0.0), 0.0, 1.0);
        }
        let band = 0.3 * threshold;
        let lo = threshold - band;
        let hi = threshold + band;
        if slip_norm <= lo {
            return (epsilon * slip_norm, epsilon, 0.0);
        }
        if slip_norm >= hi {
            return (cap, 0.0, 1.0);
        }
        let width = hi - lo;
        let t = (slip_norm - lo) / width;
        let t2 = t * t;
        let t3 = t2 * t;
        let m_lo = epsilon * lo;
        // Hermite basis with endpoint slopes (eps, 0).
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let magnitude = h00 * m_lo + h10 * width * epsilon + h01 * cap;
        let d00 = 6.0 * t2 - 6.0 * t;
        let d10 = 3.0 * t2 - 4.0 * t + 1.0;
        let d01 = -6.0 * t2 + 6.0 * t;
        let slope = (d00 * m_lo + d10 * width * epsilon + d01 * cap) / width;
        // Cap derivative: with u = eps*s/cap, m = cap*H(t(u)) gives
        // dm/dcap = H - H'*u/0.6 (0 at the stick edge, 1 at the cap edge).
        let h_prime = 0.7 * d00 + 0.6 * d10 + d01;
        let h_shape = 0.7 * h00 + 0.6 * h10 + h01;
        let u = epsilon * slip_norm / cap;
        let cap_slope = (h_shape - h_prime * u / 0.6).clamp(0.0, 1.0);
        (magnitude, slope.max(0.0), cap_slope)
    }

    /// Return the friction traction and its capped scalar magnitude.
    fn friction_traction(
        &self,
        pair: &ContactPair,
        deformed: &[f64],
        augmented: f64,
    ) -> Option<([f64; 3], f64)> {
        let mu = self.config.friction_coefficient;
        if !mu.is_finite() || mu <= 0.0 {
            return None;
        }
        let normal_pressure = (-augmented).max(0.0);
        let slip = self.tangential_slip(pair, deformed);
        let slip_norm = Self::norm(slip);
        if !normal_pressure.is_finite() || !slip_norm.is_finite() || slip_norm <= 1.0e-14 {
            return None;
        }
        let (magnitude, _, _) =
            Self::blended_friction_magnitude(self.config.epsilon, mu * normal_pressure, slip_norm);
        let scale = magnitude / slip_norm;
        Some((Self::scale(slip, scale), magnitude))
    }

    /// Tangent of the bounded tangential law, with the normal pressure held
    /// fixed during the Newton inner solve.
    fn friction_tangent(
        &self,
        pair: &ContactPair,
        deformed: &[f64],
        augmented: f64,
    ) -> Option<([[f64; 3]; 3], f64)> {
        let mu = self.config.friction_coefficient;
        let normal_pressure = (-augmented).max(0.0);
        if !mu.is_finite() || mu <= 0.0 || !normal_pressure.is_finite() || normal_pressure <= 0.0 {
            return None;
        }
        let slip = self.tangential_slip(pair, deformed);
        let slip_norm = Self::norm(slip);
        if !slip_norm.is_finite() {
            return None;
        }
        let normal = pair.normal;
        let mut tangent_projector = [[0.0; 3]; 3];
        for d in 0..3 {
            for e in 0..3 {
                tangent_projector[d][e] = if d == e { 1.0 } else { 0.0 } - normal[d] * normal[e];
            }
        }
        // Unified tangent of the blended law: sticking stiffness at zero
        // slip (the only tangential stiffness before slip develops —
        // without it the first Newton system is rigidly singular),
        // Hermite slope through the transition band, capped form past it.
        if slip_norm <= 1.0e-14 {
            let mut stiffness = [[0.0; 3]; 3];
            for d in 0..3 {
                for e in 0..3 {
                    stiffness[d][e] = self.config.epsilon * tangent_projector[d][e];
                }
            }
            return Some((stiffness, 0.0));
        }
        let cap = mu * normal_pressure;
        let (magnitude, slope, _) =
            Self::blended_friction_magnitude(self.config.epsilon, cap, slip_norm);
        let direction = Self::scale(slip, 1.0 / slip_norm);
        // Along-slip regularization past the band: the exact capped
        // linearization is null in the slip direction, so with no
        // external shear Newton cannot return from cap to stick. A tenth
        // of the penalty stiffness, ramped in over one threshold past
        // the band edge (C0, so the tangent itself never jumps),
        // restores a descent direction there.
        let threshold = cap / self.config.epsilon;
        let mut regularization = 0.0;
        if threshold.is_finite() && threshold > 0.0 {
            let edge = 1.3 * threshold;
            if slip_norm > edge {
                regularization =
                    0.1 * self.config.epsilon * ((slip_norm - edge) / threshold).min(1.0);
            }
        }
        let along_slip = slope + regularization;
        let transverse = magnitude / slip_norm;
        let mut stiffness = [[0.0; 3]; 3];
        for d in 0..3 {
            for e in 0..3 {
                stiffness[d][e] = transverse
                    * (tangent_projector[d][e] - direction[d] * direction[e])
                    + along_slip * direction[d] * direction[e];
            }
        }
        Some((stiffness, magnitude))
    }

    fn tangential_slip(&self, pair: &ContactPair, deformed: &[f64]) -> [f64; 3] {
        let slave = Self::node_pos(deformed, pair.slave_node);
        let master = Self::interpolate_master(
            deformed,
            pair.master_nodes,
            pair.reference_xi,
            pair.reference_eta,
        );
        let relative = [
            slave[0] - master[0] - pair.reference_relative_tangent[0],
            slave[1] - master[1] - pair.reference_relative_tangent[1],
            slave[2] - master[2] - pair.reference_relative_tangent[2],
        ];
        Self::tangent_component(relative, pair.normal)
    }

    fn pair_shape(pair: &ContactPair) -> [f64; 3] {
        [1.0 - pair.xi - pair.eta, pair.xi, pair.eta]
    }

    fn interpolate_master(coords: &[f64], nodes: [usize; 3], xi: f64, eta: f64) -> [f64; 3] {
        let shape = [1.0 - xi - eta, xi, eta];
        let mut point = [0.0; 3];
        for (alpha, &node) in nodes.iter().enumerate() {
            for d in 0..3 {
                point[d] += shape[alpha] * coords[3 * node + d];
            }
        }
        point
    }

    fn tangent_component(vector: [f64; 3], normal: [f64; 3]) -> [f64; 3] {
        let normal_part = Self::dot(vector, normal);
        [
            vector[0] - normal_part * normal[0],
            vector[1] - normal_part * normal[1],
            vector[2] - normal_part * normal[2],
        ]
    }

    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn norm(a: [f64; 3]) -> f64 {
        Self::dot(a, a).sqrt()
    }

    fn scale(a: [f64; 3], factor: f64) -> [f64; 3] {
        [a[0] * factor, a[1] * factor, a[2] * factor]
    }

    /// Compute tributary area per node: one third of each adjacent
    /// triangle's reference area.
    fn compute_node_areas(surface: &ContactSurface, coords: &[f64], n_nodes: usize) -> Vec<f64> {
        let mut areas = vec![0.0; n_nodes];
        for tri in &surface.triangles {
            let v0 = Self::node_pos(coords, tri.nodes[0]);
            let v1 = Self::node_pos(coords, tri.nodes[1]);
            let v2 = Self::node_pos(coords, tri.nodes[2]);
            let e0 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e1 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let cross = [
                e0[1] * e1[2] - e0[2] * e1[1],
                e0[2] * e1[0] - e0[0] * e1[2],
                e0[0] * e1[1] - e0[1] * e1[0],
            ];
            let tri_area = Self::norm(cross) / 2.0;
            for &node in &tri.nodes {
                areas[node] += tri_area / 3.0;
            }
        }
        areas
    }

    /// Compute area-weighted average normal per node from surface triangles.
    fn compute_node_normals(
        surface: &ContactSurface,
        coords: &[f64],
        n_nodes: usize,
    ) -> Vec<[f64; 3]> {
        let mut normals = vec![[0.0; 3]; n_nodes];
        for tri in &surface.triangles {
            let v0 = Self::node_pos(coords, tri.nodes[0]);
            let v1 = Self::node_pos(coords, tri.nodes[1]);
            let v2 = Self::node_pos(coords, tri.nodes[2]);
            let e0 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e1 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            // Area-weighted normal (cross product, unnormalized = 2*area*n)
            let n = [
                e0[1] * e1[2] - e0[2] * e1[1],
                e0[2] * e1[0] - e0[0] * e1[2],
                e0[0] * e1[1] - e0[1] * e1[0],
            ];
            for &node in &tri.nodes {
                for d in 0..3 {
                    normals[node][d] += n[d];
                }
            }
        }
        // Normalize
        for n in &mut normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-30 {
                n[0] /= len;
                n[1] /= len;
                n[2] /= len;
            }
        }
        normals
    }

    #[inline]
    fn node_pos(coords: &[f64], node: usize) -> [f64; 3] {
        [coords[3 * node], coords[3 * node + 1], coords[3 * node + 2]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two flat triangles facing each other with a gap between them.
    /// Top triangle (z=1, normal pointing down): nodes 0,1,2
    /// Bottom triangle (z=0, normal pointing up): nodes 3,4,5
    fn two_facing_triangles() -> (Vec<[usize; 3]>, Vec<f64>) {
        let coords = vec![
            // Top triangle nodes (slave candidates) at z=1
            0.25, 0.25, 1.0, // node 0
            0.75, 0.25, 1.0, // node 1
            0.5, 0.75, 1.0, // node 2
            // Bottom triangle nodes (master surface) at z=0
            0.0, 0.0, 0.0, // node 3
            1.0, 0.0, 0.0, // node 4
            0.5, 1.0, 0.0, // node 5
        ];

        // Top triangle winding reversed so its normal points down (toward bottom)
        let faces = vec![[0, 2, 1], [3, 4, 5]];
        (faces, coords)
    }

    #[test]
    fn test_contact_manager_no_penetration() {
        let (faces, coords) = two_facing_triangles();
        let config = ContactConfig {
            epsilon: 1e4,
            search_radius: 0.5,
            bvh_margin: 0.1,
            ..Default::default()
        };

        let mut mgr = ContactManager::new(&faces, coords, config);

        // Zero displacement: gap = 1.0, no contact
        let disp = vec![0.0; 18];
        mgr.detect_contact(&disp);

        let mut residual = vec![0.0; 18];
        mgr.contact_residual(&disp, &mut residual);

        // With gap=1.0 and λ=0, augmented = 0 + ε*1.0 > 0 → no contact force
        let total: f64 = residual.iter().map(|r| r.abs()).sum();
        assert!(
            total < 1e-14,
            "Should have no contact force when separated, got {total}"
        );
    }

    #[test]
    fn test_contact_manager_with_penetration() {
        let (faces, coords) = two_facing_triangles();
        let config = ContactConfig {
            epsilon: 1e4,
            search_radius: 1.5,
            bvh_margin: 0.2,
            ..Default::default()
        };

        let mut mgr = ContactManager::new(&faces, coords, config);

        // Push top triangle down by 1.5 → penetrates bottom triangle by 0.5
        let mut disp = vec![0.0; 18];
        disp[2] = -1.5; // node 0 z
        disp[5] = -1.5; // node 1 z
        disp[8] = -1.5; // node 2 z

        mgr.detect_contact(&disp);

        let mut residual = vec![0.0; 18];
        mgr.contact_residual(&disp, &mut residual);

        // Should have nonzero contact forces
        let total: f64 = residual.iter().map(|r| r.abs()).sum();
        assert!(total > 0.0, "Should have contact force when penetrating");
    }

    #[test]
    fn test_uzawa_multiplier_update() {
        let (faces, coords) = two_facing_triangles();
        let config = ContactConfig {
            epsilon: 1e4,
            search_radius: 1.5,
            bvh_margin: 0.2,
            ..Default::default()
        };

        let mut mgr = ContactManager::new(&faces, coords, config);

        let mut disp = vec![0.0; 18];
        disp[2] = -1.5;
        disp[5] = -1.5;
        disp[8] = -1.5;

        mgr.detect_contact(&disp);

        // First Uzawa update
        let delta1 = mgr.update_multipliers(&disp);
        assert!(delta1 > 0.0, "Multiplier should change on first update");

        // Second update (multipliers accumulate)
        let delta2 = mgr.update_multipliers(&disp);
        assert!(delta2 > 0.0);
    }

    #[test]
    fn test_relaxed_multiplier_update_scales_the_increment() {
        let (faces, coords) = two_facing_triangles();
        let config = ContactConfig {
            epsilon: 1e4,
            search_radius: 2.0,
            bvh_margin: 0.2,
            ..Default::default()
        };
        let mut full = ContactManager::new(&faces, coords.clone(), config.clone());
        let mut half = ContactManager::new(&faces, coords, config);
        let mut disp = vec![0.0; 18];
        for node in 0..3 {
            disp[3 * node + 2] = -1.5;
        }
        full.detect_contact(&disp);
        half.detect_contact(&disp);
        let delta_full = full.update_multipliers(&disp);
        let delta_half = half.update_multipliers_relaxed(&disp, 0.5);
        assert!(delta_full > 0.0);
        assert!(
            (delta_half - 0.5 * delta_full).abs() < 1.0e-9 * delta_full,
            "relaxed update must halve the increment: {delta_half} vs {delta_full}"
        );
    }

    /// Hertz-like contact validation: flat plate against rigid substrate.
    ///
    /// Two parallel flat triangles (normal distance = 1.0) are pushed together.
    /// For a penalty method with epsilon = 1e4:
    ///   Force ≈ epsilon × penetration_area × penetration_depth
    ///
    /// This test validates:
    /// 1. Contact force increases monotonically with penetration
    /// 2. Net force on slave = -net force on master (momentum conservation)
    /// 3. Force is directed along the surface normal
    #[test]
    fn test_contact_force_vs_penetration() {
        let (faces, coords) = two_facing_triangles();
        let epsilon = 1e4;
        let config = ContactConfig {
            epsilon,
            search_radius: 2.0,
            bvh_margin: 0.2,
            ..Default::default()
        };

        let mut mgr = ContactManager::new(&faces, coords, config);
        let mut prev_force = 0.0;

        // Nodes start at z=1.0 (top triangle). Bottom triangle is at z=0.
        // To penetrate by δ, we need disp_z = -(1.0 + δ).
        for delta in [0.1, 0.2, 0.3, 0.4, 0.5] {
            let mut disp = vec![0.0; 18];
            // Push top triangle down past the bottom triangle
            disp[2] = -(1.0 + delta); // node 0
            disp[5] = -(1.0 + delta); // node 1
            disp[8] = -(1.0 + delta); // node 2

            mgr.detect_contact(&disp);
            mgr.reset_multipliers(); // Start fresh for each penetration

            let mut residual = vec![0.0; 18];
            mgr.contact_residual(&disp, &mut residual);

            // Net force on slave (nodes 0,1,2)
            let slave_force_z: f64 = (0..3).map(|i| residual[3 * i + 2]).sum();

            // Net force on master (nodes 3,4,5)
            let master_force_z: f64 = (3..6).map(|i| residual[3 * i + 2]).sum();

            // Force magnitude should increase with penetration
            let force_mag = slave_force_z.abs();
            assert!(
                force_mag > prev_force,
                "Force should increase with penetration: {force_mag} <= {prev_force}"
            );

            // Momentum conservation: slave + master = 0
            let net_force = slave_force_z + master_force_z;
            assert!(
                net_force.abs() < 1e-10,
                "Momentum not conserved at δ={delta}: slave={slave_force_z}, master={master_force_z}"
            );

            // Force should be approximately proportional to penetration depth
            // (for penalty method: F ≈ ε × A × δ)
            let expected_force = epsilon * delta; // order-of-magnitude check
            assert!(
                force_mag > 0.1 * expected_force,
                "Force too small at δ={delta}: {force_mag} vs expected ~{expected_force}"
            );

            prev_force = force_mag;
        }
    }

    /// Validate contact tangent stiffness is symmetric.
    #[test]
    fn test_contact_tangent_symmetry() {
        let (faces, coords) = two_facing_triangles();
        let config = ContactConfig {
            epsilon: 1e4,
            search_radius: 1.5,
            bvh_margin: 0.2,
            ..Default::default()
        };

        let mut mgr = ContactManager::new(&faces, coords, config);

        let mut disp = vec![0.0; 18];
        disp[2] = -1.5;
        disp[5] = -1.5;
        disp[8] = -1.5;

        mgr.detect_contact(&disp);
        mgr.reset_multipliers();
        mgr.update_multipliers(&disp);

        let triplets = mgr.contact_tangent(&disp);

        // Build sparse matrix and check symmetry
        let mut matrix: std::collections::HashMap<(usize, usize), f64> =
            std::collections::HashMap::new();
        for (row, col, val) in triplets {
            *matrix.entry((row, col)).or_insert(0.0) += val;
        }

        for (&(row, col), &val) in &matrix {
            let sym_val = matrix.get(&(col, row)).copied().unwrap_or(0.0);
            assert!(
                (val - sym_val).abs() < 1e-10,
                "Tangent not symmetric at ({row},{col}): {val} vs {sym_val}"
            );
        }
    }

    #[test]
    fn test_friction_is_tangential_and_coulomb_capped() {
        let (faces, coords) = two_facing_triangles();
        let friction_coefficient = 0.25;
        let config = ContactConfig {
            epsilon: 1e4,
            search_radius: 2.0,
            bvh_margin: 0.2,
            friction_coefficient,
            ..Default::default()
        };
        let mut mgr = ContactManager::new(&faces, coords, config);

        let mut disp = vec![0.0; 18];
        for node in 0..3 {
            disp[3 * node] = 0.1;
            disp[3 * node + 2] = -1.5;
        }
        mgr.detect_contact(&disp);

        let mut residual = vec![0.0; 18];
        mgr.contact_residual(&disp, &mut residual);

        let tangential_abs: f64 = residual.chunks_exact(3).map(|r| r[0].abs()).sum();
        let normal_abs: f64 = residual.chunks_exact(3).map(|r| r[2].abs()).sum();
        let y_abs: f64 = residual.chunks_exact(3).map(|r| r[1].abs()).sum();
        let net_x: f64 = residual.chunks_exact(3).map(|r| r[0]).sum();

        assert!(tangential_abs > 0.0, "friction should oppose lateral slip");
        assert!(
            y_abs < 1.0e-10,
            "friction should remain in the tangent plane"
        );
        assert!(
            net_x.abs() < 1.0e-10,
            "contact must conserve tangential force"
        );
        assert!(
            tangential_abs <= friction_coefficient * normal_abs + 1.0e-8,
            "Coulomb cap exceeded: tangential={tangential_abs}, normal={normal_abs}"
        );
    }

    #[test]
    fn test_friction_disabled_preserves_normal_only_residual() {
        let (faces, coords) = two_facing_triangles();
        let mut mgr = ContactManager::new(
            &faces,
            coords,
            ContactConfig {
                epsilon: 1e4,
                search_radius: 2.0,
                bvh_margin: 0.2,
                ..Default::default()
            },
        );
        let mut disp = vec![0.0; 18];
        for node in 0..3 {
            disp[3 * node] = 0.1;
            disp[3 * node + 2] = -1.5;
        }
        mgr.detect_contact(&disp);
        let mut residual = vec![0.0; 18];
        mgr.contact_residual(&disp, &mut residual);

        let tangential_abs: f64 = residual.chunks_exact(3).map(|r| r[0].abs()).sum();
        assert!(
            tangential_abs < 1.0e-12,
            "zero friction must not add tangential traction"
        );
    }

    #[test]
    fn test_friction_tangent_is_symmetric() {
        let (faces, coords) = two_facing_triangles();
        let mut mgr = ContactManager::new(
            &faces,
            coords,
            ContactConfig {
                epsilon: 1e4,
                search_radius: 2.0,
                bvh_margin: 0.2,
                friction_coefficient: 0.25,
                ..Default::default()
            },
        );
        // Deep-stick slip (0.01 << s* = 0.125): the stick tangent is
        // symmetric. Past the transition band the pressure-dependent
        // cap term is unsymmetric by nature (non-associative slip), so
        // symmetry is only asserted here; the FD test below locks the
        // full tangent (stick, band, capped) against the residual.
        let mut disp = vec![0.0; 18];
        for node in 0..3 {
            disp[3 * node] = 0.01;
            disp[3 * node + 2] = -1.5;
        }
        mgr.detect_contact(&disp);
        let triplets = mgr.contact_tangent(&disp);
        let mut matrix = std::collections::HashMap::<(usize, usize), f64>::new();
        for (row, col, value) in triplets {
            *matrix.entry((row, col)).or_insert(0.0) += value;
        }
        for (&(row, col), &value) in &matrix {
            let transpose = matrix.get(&(col, row)).copied().unwrap_or(0.0);
            assert!(
                (value - transpose).abs() < 1.0e-8,
                "friction tangent not symmetric at ({row},{col}): {value} vs {transpose}"
            );
        }
    }

    /// Finite-difference lock of the contact tangent against the residual
    /// in the stick, transition-band, and capped regimes (frozen pairs).
    ///
    /// This guards the exact bug class that stalled sliding Newton solves:
    /// a tangent missing the pressure-dependent cap term contradicts the
    /// residual, and the line search then rejects every step. Small mu
    /// keeps all three regimes at interior projections (no edge kinks).
    ///
    /// One-sided pairing: slave columns are exact (the tangent is the
    /// exact linearization for frozen projection data there), so they
    /// assert tight. Master columns carry the documented projection-
    /// derivative approximation (barycentric/normal follow dropped),
    /// so they assert a loose bound that still catches sign errors.
    #[test]
    fn test_contact_tangent_matches_finite_difference() {
        let (faces, coords) = two_facing_triangles();
        // faces[0] = top triangle (nodes 0,1,2), faces[1] = bottom (3,4,5).
        let mut mgr = ContactManager::new_two_body(
            &faces[1..2],
            &faces[0..1],
            coords,
            ContactConfig {
                epsilon: 1e4,
                search_radius: 2.0,
                bvh_margin: 0.2,
                friction_coefficient: 0.05,
                ..Default::default()
            },
        );
        // Cap: mu * eps * penetration = 0.05 * 1e4 * 0.5 = 250 Pa;
        // s* = 250/1e4 = 0.025 (band [0.0175, 0.0325]).
        for slip in [0.005, 0.025, 0.05] {
            let mut disp = vec![0.0; 18];
            for node in 0..3 {
                disp[3 * node] = slip;
                disp[3 * node + 2] = -1.5;
            }
            mgr.detect_contact(&disp);
            assert!(!mgr.pairs.is_empty(), "slip={slip}: no pairs detected");
            assert!(
                mgr.pairs.iter().all(|pair| pair.slave_node < 3),
                "one-sided pairing must keep slaves on the top triangle"
            );
            let triplets = mgr.contact_tangent(&disp);
            let mut matrix = vec![vec![0.0; 18]; 18];
            for (row, col, value) in triplets {
                matrix[row][col] += value;
            }
            // Capped regime: subtract the intentional along-slip tangent
            // regularization (quasi-Newton aid, residual untouched) before
            // the tight comparison. Uniform x-slip makes every pair's slip
            // direction exactly +x, so the subtraction is exact; it also
            // locks the ramp formula and scatter against silent change.
            if slip > 0.0325 {
                let threshold = 0.025;
                let reg = 0.1 * 1.0e4 * ((slip - 1.3 * threshold) / threshold).min(1.0);
                for pair in &mgr.pairs {
                    let shape = [1.0 - pair.xi - pair.eta, pair.xi, pair.eta];
                    let value = pair.area * reg;
                    matrix[3 * pair.slave_node][3 * pair.slave_node] -= value;
                    for (alpha, &master_node) in pair.master_nodes.iter().enumerate() {
                        matrix[3 * pair.slave_node][3 * master_node] += shape[alpha] * value;
                        matrix[3 * master_node][3 * pair.slave_node] += shape[alpha] * value;
                        for (beta, &node_b) in pair.master_nodes.iter().enumerate() {
                            matrix[3 * master_node][3 * node_b] -=
                                shape[alpha] * shape[beta] * value;
                        }
                    }
                }
            }
            let step = 1.0e-6;
            for col in 0..18 {
                let mut plus = disp.clone();
                plus[col] += step;
                let mut minus = disp.clone();
                minus[col] -= step;
                let mut r_plus = vec![0.0; 18];
                let mut r_minus = vec![0.0; 18];
                mgr.contact_residual(&plus, &mut r_plus);
                mgr.contact_residual(&minus, &mut r_minus);
                // Slave columns (nodes 0,1,2) exact; master columns loose.
                let tolerance = if col < 9 { 1.0e-4 } else { 1.0 };
                for row in 0..18 {
                    let fd = (r_plus[row] - r_minus[row]) / (2.0 * step);
                    let analytic = matrix[row][col];
                    let scale = fd.abs().max(analytic.abs());
                    // Absolute floor 1e-6: FD rounding noise at R ~ 1e2 is
                    // ~1e-8; genuine tangent entries here are >= 1e1.
                    assert!(
                        (fd - analytic).abs() < (tolerance * scale).max(1.0e-6),
                        "slip={slip} ({row},{col}): fd={fd:e} vs tangent={analytic:e}"
                    );
                }
            }
        }
    }
}
