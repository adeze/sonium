//! Penalty contact for explicit transient dynamics.
//!
//! Complements the quasi-static augmented-Lagrangian path with the force-only
//! operators an explicit integrator needs: node-to-surface penalty contact,
//! self-contact through the topological filter, regularized Coulomb friction
//! from nodal velocities, shell thickness offsets, and spotweld/tiebreak
//! failure hooked to the cohesive zone laws. Detection reuses the shared
//! BVH and closest-point projection; no tangent is assembled because the
//! central-difference scheme never needs one.
//!
//! Phase 2 additions: the penalty pass also records the frozen
//! [`PenaltyPair`](super::candidates::PenaltyPair) handoff for the
//! tiebreak lane, [`TiedInterface`] ties dissimilar meshes
//! node-to-surface (TYPE2-ward), [`rigid_wall_contact`] evaluates the
//! analytical rigid plane the Taylor gate needs, and
//! [`symmetric_penalty_pass`] is the TYPE7-ward symmetric hook. Theory
//! references below cite the Altair Radioss 2022 Theory Manual.

// Rust guideline compliant 2026-02-21

use super::candidates::PenaltyPair;
use super::detection::ContactSurface;
use super::gap::{project_point_to_triangle, ProjectionResult};
use crate::physics::solid::fracture::cohesive::{CohesiveState, MixedModeCohesiveLaw};

/// Suggested nodal penalty stiffness in N/m.
///
/// Scales with the bulk stiffness `E * Le` of the adjacent elements times
/// a factor (50 standard): stiff enough to limit penetration to a few
/// percent of the element size, soft enough to preserve the CFL step.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::contact::explicit::suggested_penalty;
///
/// let k = suggested_penalty(200.0e9, 0.01, 50.0);
/// assert!((k - 100.0e9).abs() < 1.0);
/// ```
pub fn suggested_penalty(youngs: f64, element_length: f64, factor: f64) -> f64 {
    factor * youngs * element_length
}

/// Maximum approach of an elastic sphere impact in m.
///
/// Energy balance of Hertzian contact: `delta = (15 m V^2 /
/// (16 E* sqrt(R)))^(2/5)` for mass `m`, normal velocity `V`, radius `R`,
/// and effective modulus `E*`. The quasi-static Hertz force then peaks at
/// `(4/3) E* sqrt(R) delta^(3/2)`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::contact::explicit::hertz_impact_max_approach;
///
/// let d = hertz_impact_max_approach(1.0, 1.0, 0.05, 100.0e9);
/// assert!(d > 0.0 && d < 1.0e-3);
/// ```
pub fn hertz_impact_max_approach(mass: f64, velocity: f64, radius: f64, estar: f64) -> f64 {
    (15.0 * mass * velocity * velocity / (16.0 * estar * radius.sqrt())).powf(0.4)
}

/// Contact duration of an elastic sphere impact in s.
///
/// Rayleigh's result `Tc = 2.94 delta_max / V` from the Hertzian energy
/// balance, with the maximum approach from
/// [`hertz_impact_max_approach`].
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::contact::explicit::hertz_impact_contact_time;
///
/// let t = hertz_impact_contact_time(1.0, 1.0, 0.05, 100.0e9);
/// assert!(t > 0.0 && t < 1.0e-3);
/// ```
pub fn hertz_impact_contact_time(mass: f64, velocity: f64, radius: f64, estar: f64) -> f64 {
    let approach = hertz_impact_max_approach(mass, velocity, radius, estar);
    // Rayleigh contact-time constant for Hertzian impact.
    2.94 * approach / velocity.max(1e-30)
}

/// Peak Hertz force of an elastic sphere impact in N.
///
/// Evaluates `(4/3) E* sqrt(R) delta^(3/2)` at the maximum approach from
/// [`hertz_impact_max_approach`].
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::contact::explicit::hertz_impact_peak_force;
///
/// let f = hertz_impact_peak_force(1.0, 1.0, 0.05, 100.0e9);
/// assert!(f > 0.0);
/// ```
pub fn hertz_impact_peak_force(mass: f64, velocity: f64, radius: f64, estar: f64) -> f64 {
    let approach = hertz_impact_max_approach(mass, velocity, radius, estar);
    4.0 / 3.0 * estar * radius.sqrt() * approach.powf(1.5)
}

/// Configuration of the explicit penalty contact operator.
#[derive(Debug, Clone, Copy)]
pub struct ExplicitContactConfig {
    /// Nodal penalty stiffness in N/m.
    pub penalty: f64,
    /// Coulomb friction coefficient, dimensionless.
    pub friction_coefficient: f64,
    /// Friction regularization velocity in m/s.
    pub friction_velocity: f64,
    /// BVH search radius in m.
    pub search_radius: f64,
    /// Shell thickness offset in m (half-thickness sum).
    pub thickness_offset: f64,
}

impl Default for ExplicitContactConfig {
    fn default() -> Self {
        Self {
            penalty: 1.0e9,
            friction_coefficient: 0.0,
            friction_velocity: 0.01,
            search_radius: 0.01,
            thickness_offset: 0.0,
        }
    }
}

/// Penalty node-to-surface contact for explicit integration.
///
/// Holds the master surface plus the slave node list and scatters penalty
/// plus friction forces into the caller force vector. Self-contact uses
/// one surface for both roles with the topological adjacency filter;
/// two-body contact pairs an explicit slave list against the master.
/// Shell contact passes the summed half-thicknesses as `thickness_offset`
/// so mid-surface meshes touch at the physical outer fibers.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::contact::explicit::{ExplicitContact, ExplicitContactConfig};
///
/// let master = vec![[0, 1, 2], [0, 2, 3]];
/// let contact = ExplicitContact::new_two_body(&master, &[4], 5, ExplicitContactConfig::default());
/// assert_eq!(contact.num_slave_nodes(), 1);
/// ```
#[derive(Debug)]
pub struct ExplicitContact {
    /// Master triangles in parent node numbering.
    master: ContactSurface,
    /// Slave node ids in parent numbering.
    slaves: Vec<usize>,
    /// Contact parameters.
    config: ExplicitContactConfig,
    /// Spotweld and tiebreak connectors.
    spotwelds: SpotweldSet,
}

impl ExplicitContact {
    /// Creates a two-body operator from master faces and slave nodes.
    ///
    /// Takes master boundary triangles, slave node ids, the parent node
    /// count, and the contact parameters.
    pub fn new_two_body(
        master_faces: &[[usize; 3]],
        slave_nodes: &[usize],
        num_parent_nodes: usize,
        config: ExplicitContactConfig,
    ) -> Self {
        Self {
            master: ContactSurface::from_boundary_faces(master_faces, num_parent_nodes),
            slaves: slave_nodes.to_vec(),
            config,
            spotwelds: SpotweldSet::new(),
        }
    }

    /// Creates a self-contact operator over one surface.
    pub fn new_self(
        faces: &[[usize; 3]],
        num_parent_nodes: usize,
        config: ExplicitContactConfig,
    ) -> Self {
        let master = ContactSurface::from_boundary_faces(faces, num_parent_nodes);
        let slaves = master.node_indices.clone();
        Self {
            master,
            slaves,
            config,
            spotwelds: SpotweldSet::new(),
        }
    }

    /// Counts the slave nodes.
    pub fn num_slave_nodes(&self) -> usize {
        self.slaves.len()
    }

    /// Borrows the contact parameters.
    pub fn config(&self) -> &ExplicitContactConfig {
        &self.config
    }

    /// Registers a spotweld connector, returning its id.
    pub fn add_spotweld(&mut self, weld: Spotweld) -> usize {
        self.spotwelds.push(weld)
    }

    /// Counts failed spotwelds.
    pub fn num_failed_welds(&self) -> usize {
        self.spotwelds.num_failed()
    }

    /// Scatters contact forces into the flat `3N` force vector.
    ///
    /// Takes deformed coordinates and half-step velocities (both flat
    /// `3N`) and adds penalty normal plus regularized Coulomb friction
    /// forces for every penetrating pair, then the spotweld forces.
    /// Force is conserved: master triangles receive the negated slave
    /// force through the projection weights.
    ///
    /// # Panics
    ///
    /// Panics when the buffers are shorter than `3N`.
    pub fn apply(&mut self, deformed_coords: &[f64], velocities: &[f64], forces: &mut [f64]) {
        self.penalty_pass(deformed_coords, velocities, forces, None);
        self.spotwelds.apply(deformed_coords, forces);
    }

    /// Scatters contact forces and records penalty pairs.
    ///
    /// Same scatter as [`apply`](Self::apply), additionally returning one
    /// frozen [`PenaltyPair`] per penetrating pair (slave side, gap
    /// negative in penetration) for the lane 2D tiebreak filter. Pairs
    /// arrive in slave order. The recorded force is the scattered slave
    /// force verbatim (push-out along +normal in compression), so the
    /// frozen compression-positive `normal_force()` reports it directly
    /// (ruling P2-R1 fixed an inverted sign in an earlier revision).
    ///
    /// # Panics
    ///
    /// Panics when the buffers are shorter than `3N`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::contact::explicit::{ExplicitContact, ExplicitContactConfig};
    ///
    /// let config = ExplicitContactConfig { search_radius: 0.5, ..Default::default() };
    /// let mut contact =
    ///     ExplicitContact::new_two_body(&[[0, 1, 2], [0, 2, 3]], &[4], 5, config);
    /// let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, -0.01];
    /// let velocities = vec![0.0; 15];
    /// let mut forces = vec![0.0; 15];
    /// let pairs = contact.apply_with_pairs(&coords, &velocities, &mut forces);
    /// assert_eq!(pairs.len(), 1);
    /// assert_eq!(pairs[0].slave, 4);
    /// assert!(pairs[0].gap < 0.0);
    /// ```
    pub fn apply_with_pairs(
        &mut self,
        deformed_coords: &[f64],
        velocities: &[f64],
        forces: &mut [f64],
    ) -> Vec<PenaltyPair> {
        let mut pairs = Vec::new();
        self.penalty_pass(deformed_coords, velocities, forces, Some(&mut pairs));
        self.spotwelds.apply(deformed_coords, forces);
        pairs
    }

    /// Penalty normal plus friction scatter, optionally recording pairs.
    ///
    /// Shared by [`apply`](Self::apply) and
    /// [`apply_with_pairs`](Self::apply_with_pairs) so force values never
    /// diverge between the scatter-only and handoff paths.
    fn penalty_pass(
        &self,
        deformed_coords: &[f64],
        velocities: &[f64],
        forces: &mut [f64],
        mut pairs: Option<&mut Vec<PenaltyPair>>,
    ) {
        let num_nodes = self.master.parent_to_surface.len();
        assert!(
            deformed_coords.len() >= 3 * num_nodes
                && velocities.len() >= 3 * num_nodes
                && forces.len() >= 3 * num_nodes,
            "explicit contact needs flat 3N coordinate, velocity, and force buffers"
        );
        let margin = self.config.search_radius + self.config.thickness_offset;
        let bvh = self.master.build_bvh(deformed_coords, margin);
        let candidates =
            self.master
                .find_candidates_for(&self.slaves, deformed_coords, &bvh, margin);
        // Closest-pair selection: shared edges project onto several
        // triangles, so each slave keeps only its deepest projection.
        let node_at = |n: usize| {
            [
                deformed_coords[3 * n],
                deformed_coords[3 * n + 1],
                deformed_coords[3 * n + 2],
            ]
        };
        let mut best: Vec<Option<(usize, ProjectionResult)>> = vec![None; num_nodes];
        for (slave, tri_idx) in candidates {
            let tri = &self.master.triangles[tri_idx];
            let projection = project_point_to_triangle(
                node_at(slave),
                node_at(tri.nodes[0]),
                node_at(tri.nodes[1]),
                node_at(tri.nodes[2]),
            );
            let deeper = match &best[slave] {
                None => true,
                Some((_, prev)) => projection.gap < prev.gap,
            };
            if deeper {
                best[slave] = Some((tri_idx, projection));
            }
        }
        for (slave, entry) in best.iter().enumerate() {
            let Some((tri_idx, projection)) = entry else {
                continue;
            };
            let tri_idx = *tri_idx;
            let projection = *projection;
            let tri = &self.master.triangles[tri_idx];
            let penetration = self.config.thickness_offset - projection.gap;
            if penetration <= 0.0 {
                continue;
            }
            let normal_force = self.config.penalty * penetration;
            let normal = projection.normal;
            // Relative tangential velocity at the contact point.
            let zeta = 1.0 - projection.xi - projection.eta;
            let weights = [zeta, projection.xi, projection.eta];
            let mut master_v = [0.0; 3];
            for (w, node) in weights.iter().zip(tri.nodes.iter()) {
                master_v[0] += w * velocities[3 * node];
                master_v[1] += w * velocities[3 * node + 1];
                master_v[2] += w * velocities[3 * node + 2];
            }
            let slave_v = [
                velocities[3 * slave],
                velocities[3 * slave + 1],
                velocities[3 * slave + 2],
            ];
            let mut rel = [
                slave_v[0] - master_v[0],
                slave_v[1] - master_v[1],
                slave_v[2] - master_v[2],
            ];
            let vn = rel[0] * normal[0] + rel[1] * normal[1] + rel[2] * normal[2];
            rel[0] -= vn * normal[0];
            rel[1] -= vn * normal[1];
            rel[2] -= vn * normal[2];
            let speed = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
            // Regularized Coulomb: mu Fn tanh(|vt| / v0) along -t.
            let friction = if speed > 0.0 && self.config.friction_coefficient > 0.0 {
                let v0 = self.config.friction_velocity.max(1e-12);
                self.config.friction_coefficient * normal_force * (speed / v0).tanh() / speed
            } else {
                0.0
            };
            let slave_force = [
                normal_force * normal[0] - friction * rel[0],
                normal_force * normal[1] - friction * rel[1],
                normal_force * normal[2] - friction * rel[2],
            ];
            forces[3 * slave] += slave_force[0];
            forces[3 * slave + 1] += slave_force[1];
            forces[3 * slave + 2] += slave_force[2];
            for (w, node) in weights.iter().zip(tri.nodes.iter()) {
                forces[3 * node] -= w * slave_force[0];
                forces[3 * node + 1] -= w * slave_force[1];
                forces[3 * node + 2] -= w * slave_force[2];
            }
            if let Some(records) = pairs.as_deref_mut() {
                records.push(PenaltyPair {
                    slave,
                    gap: -penetration,
                    normal,
                    force: slave_force,
                });
            }
        }
    }
}

/// Spotweld connector with cohesive failure.
///
/// Ties a slave node to a fixed barycentric point of a master triangle:
/// the relative displacement from the initial offset drives a mixed-mode
/// cohesive law, and the traction times `area` scatters as equal and
/// opposite forces. Failure is irreversible once the cohesive damage
/// reaches one. Models spotwelds, rivets, and tiebreak interfaces.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::contact::explicit::Spotweld;
/// use sonium_fem::physics::solid::fracture::cohesive::MixedModeCohesiveLaw;
///
/// let weld = Spotweld::new(4, [0, 1, 2], [0.5, 0.25, 0.25], MixedModeCohesiveLaw::epoxy(), 1.0e-6);
/// assert!(!weld.failed);
/// ```
#[derive(Debug, Clone)]
pub struct Spotweld {
    /// Slave node id in parent numbering.
    pub slave: usize,
    /// Master triangle nodes in parent numbering.
    pub master: [usize; 3],
    /// Fixed barycentric weights over the master triangle.
    pub weights: [f64; 3],
    /// Mixed-mode cohesive failure law.
    pub law: MixedModeCohesiveLaw,
    /// Tributary area in m^2.
    pub area: f64,
    /// Initial slave-to-master offset in m, set on first evaluation.
    pub initial_offset: Option<[f64; 3]>,
    /// Cohesive history state.
    pub state: CohesiveState,
    /// Irreversible failure flag.
    pub failed: bool,
}

impl Spotweld {
    /// Creates a weld from connectivity, law, and tributary area.
    pub fn new(
        slave: usize,
        master: [usize; 3],
        weights: [f64; 3],
        law: MixedModeCohesiveLaw,
        area: f64,
    ) -> Self {
        Self {
            slave,
            master,
            weights,
            law,
            area,
            initial_offset: None,
            state: CohesiveState::new(),
            failed: false,
        }
    }
}

/// Owned set of spotweld connectors.
#[derive(Debug, Clone, Default)]
pub struct SpotweldSet {
    /// Connectors by registration order.
    welds: Vec<Spotweld>,
}

impl SpotweldSet {
    /// Creates an empty weld set.
    pub fn new() -> Self {
        Self { welds: Vec::new() }
    }

    /// Registers a weld, returning its id.
    pub fn push(&mut self, weld: Spotweld) -> usize {
        self.welds.push(weld);
        self.welds.len() - 1
    }

    /// Counts failed welds.
    pub fn num_failed(&self) -> usize {
        self.welds.iter().filter(|w| w.failed).count()
    }

    /// Scatters weld forces into the flat `3N` force vector.
    ///
    /// # Panics
    ///
    /// Panics when a buffer is shorter than the referenced nodes.
    pub fn apply(&mut self, deformed_coords: &[f64], forces: &mut [f64]) {
        for weld in &mut self.welds {
            if weld.failed {
                continue;
            }
            let node_at = |n: usize| {
                [
                    deformed_coords[3 * n],
                    deformed_coords[3 * n + 1],
                    deformed_coords[3 * n + 2],
                ]
            };
            let slave_p = node_at(weld.slave);
            let mut master_p = [0.0; 3];
            for (w, node) in weld.weights.iter().zip(weld.master.iter()) {
                let p = node_at(*node);
                master_p[0] += w * p[0];
                master_p[1] += w * p[1];
                master_p[2] += w * p[2];
            }
            let offset = [
                slave_p[0] - master_p[0],
                slave_p[1] - master_p[1],
                slave_p[2] - master_p[2],
            ];
            let initial = *weld.initial_offset.get_or_insert(offset);
            // Opening along the initial connector axis plus shear parts.
            let axis_len =
                (initial[0] * initial[0] + initial[1] * initial[1] + initial[2] * initial[2])
                    .sqrt();
            let axis = if axis_len > 1e-15 {
                [
                    initial[0] / axis_len,
                    initial[1] / axis_len,
                    initial[2] / axis_len,
                ]
            } else {
                [0.0, 0.0, 1.0]
            };
            let relative = [
                offset[0] - initial[0],
                offset[1] - initial[1],
                offset[2] - initial[2],
            ];
            let normal_open = relative[0] * axis[0] + relative[1] * axis[1] + relative[2] * axis[2];
            let shear = [
                relative[0] - normal_open * axis[0],
                relative[1] - normal_open * axis[1],
                relative[2] - normal_open * axis[2],
            ];
            // Shear basis completion for the cohesive opening vector.
            let helper = if axis[2].abs() < 0.9 {
                [0.0, 0.0, 1.0]
            } else {
                [1.0, 0.0, 0.0]
            };
            let mut t1 = [
                axis[1] * helper[2] - axis[2] * helper[1],
                axis[2] * helper[0] - axis[0] * helper[2],
                axis[0] * helper[1] - axis[1] * helper[0],
            ];
            let t1_len = (t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2])
                .sqrt()
                .max(1e-30);
            t1[0] /= t1_len;
            t1[1] /= t1_len;
            t1[2] /= t1_len;
            let t2 = [
                axis[1] * t1[2] - axis[2] * t1[1],
                axis[2] * t1[0] - axis[0] * t1[2],
                axis[0] * t1[1] - axis[1] * t1[0],
            ];
            let opening = [
                normal_open,
                shear[0] * t1[0] + shear[1] * t1[1] + shear[2] * t1[2],
                shear[0] * t2[0] + shear[1] * t2[1] + shear[2] * t2[2],
            ];
            let traction = weld.law.traction_3d(opening, &mut weld.state);
            if weld.state.damage >= 1.0 {
                weld.failed = true;
                continue;
            }
            // Traction back to world axes, times area, onto slave.
            let world = [
                (traction[0] * axis[0] + traction[1] * t1[0] + traction[2] * t2[0]) * weld.area,
                (traction[0] * axis[1] + traction[1] * t1[1] + traction[2] * t2[1]) * weld.area,
                (traction[0] * axis[2] + traction[1] * t1[2] + traction[2] * t2[2]) * weld.area,
            ];
            forces[3 * weld.slave] -= world[0];
            forces[3 * weld.slave + 1] -= world[1];
            forces[3 * weld.slave + 2] -= world[2];
            for (w, node) in weld.weights.iter().zip(weld.master.iter()) {
                forces[3 * node] += w * world[0];
                forces[3 * node + 1] += w * world[1];
                forces[3 * node + 2] += w * world[2];
            }
        }
    }
}

/// Hertz effective modulus of two elastic bodies.
///
/// Combines Young's moduli and Poisson ratios as `1/E* = (1-nu1^2)/E1 +
/// (1-nu2^2)/E2`. A rigid counterface is the `E2 -> infinity` limit, for
/// which the caller passes `f64::INFINITY`.
///
/// # Panics
///
/// Panics when either modulus is not positive.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::contact::explicit::hertz_effective_modulus;
///
/// let estar = hertz_effective_modulus(200.0e9, 0.3, f64::INFINITY, 0.0);
/// assert!((estar - 200.0e9 / 0.91).abs() / estar < 1e-15);
/// ```
pub fn hertz_effective_modulus(e1: f64, nu1: f64, e2: f64, nu2: f64) -> f64 {
    assert!(
        e1 > 0.0 && e2 > 0.0,
        "Hertz moduli must be positive (rigid is infinite)"
    );
    1.0 / ((1.0 - nu1 * nu1) / e1 + (1.0 - nu2 * nu2) / e2)
}

/// Penalized Hertz normal contact response.
///
/// Superposes the penalty compliance `F/k_p` in series with the Hertz
/// approach `delta = a^2/R`, where `a^3 = 3FR/(4E*)`. The tangent Hertz
/// stiffness `k_H = dF/ddelta = 2E*a` composes with the penalty into
/// `1/k_eff = 1/k_H + 1/k_p`.
#[derive(Debug, Clone, Copy)]
pub struct PenalizedHertz {
    /// Hertz contact radius `a` in m (penalty-independent at fixed force).
    pub contact_radius: f64,
    /// Elastic Hertz approach `a^2/R` in m.
    pub hertz_approach: f64,
    /// Total approach including `F/k_p` in m.
    pub total_approach: f64,
    /// Tangent Hertz stiffness `2E*a` in N/m.
    pub hertz_stiffness: f64,
    /// Series stiffness of Hertz plus penalty springs in N/m.
    pub effective_stiffness: f64,
}

/// Penalized Hertz response of a sphere at fixed normal force.
///
/// Evaluates the closed form behind [`PenalizedHertz`]: Hertz geometry from
/// force, radius, and effective modulus, plus the series penalty spring.
/// Backs the `solids_penalty_hertz` Wolfram golden.
///
/// # Panics
///
/// Panics when the force is negative or radius, modulus, or penalty is not
/// positive.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::contact::explicit::{hertz_effective_modulus, penalized_hertz};
///
/// let estar = hertz_effective_modulus(200.0e9, 0.3, f64::INFINITY, 0.0);
/// let soft = penalized_hertz(1000.0, 0.01, estar, 1.0e8);
/// let stiff = penalized_hertz(1000.0, 0.01, estar, 1.0e12);
/// assert_eq!(soft.contact_radius, stiff.contact_radius);
/// assert!(soft.total_approach > stiff.total_approach);
/// assert!(stiff.effective_stiffness > soft.effective_stiffness);
/// ```
pub fn penalized_hertz(force: f64, radius: f64, estar: f64, penalty: f64) -> PenalizedHertz {
    assert!(
        force >= 0.0 && radius > 0.0 && estar > 0.0 && penalty > 0.0,
        "penalized Hertz needs non-negative force and positive radius, modulus, penalty"
    );
    let contact_radius = (3.0 * force * radius / (4.0 * estar)).powf(1.0 / 3.0);
    let hertz_approach = contact_radius * contact_radius / radius;
    let hertz_stiffness = 2.0 * estar * contact_radius;
    PenalizedHertz {
        contact_radius,
        hertz_approach,
        total_approach: hertz_approach + force / penalty,
        hertz_stiffness,
        effective_stiffness: 1.0 / (1.0 / hertz_stiffness + 1.0 / penalty),
    }
}

/// One tied slave node with its master interpolation.
///
/// Constrains `u_slave = sum(weights * u_master) + offset` per masked
/// component: the slave follows the master triangle's barycentric point it
/// was tied to, with no sliding. Theory Manual "Tied Interface (TYPE2)"
/// (pp. 213-215): the secondary node stays at the same position on its
/// main segment; the main surface should be the coarser mesh.
#[derive(Debug, Clone, Copy)]
pub struct TiedPair {
    /// Slave node id in parent numbering.
    pub slave: usize,
    /// Master triangle nodes in parent numbering.
    pub master: [usize; 3],
    /// Barycentric weights over the master triangle (sum to one).
    pub weights: [f64; 3],
    /// Reference slave-minus-master offset in m per component.
    pub offset: [f64; 3],
    /// Tied displacement components (untied ones stay free).
    pub mask: [bool; 3],
}

/// Penalty-kinematic tied node-to-surface interface.
///
/// Ties secondary nodes to a main triangulated surface for mesh-transition
/// and dissimilar-material stacks. Two enforcement paths share the pairs:
/// [`apply_penalty`](Self::apply_penalty) scatters penalty springs for
/// explicit dynamics (no matrix needed), while [`eliminate`](Self::eliminate)
/// folds the exact kinematic constraint into a dense stiffness for
/// static/implicit solves, with [`recover`](Self::recover) restoring slave
/// displacements afterwards.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::contact::explicit::TiedInterface;
///
/// let faces = vec![[0, 1, 2]];
/// let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.25, 0.25, 0.0];
/// let tie = TiedInterface::tie_nodes_to_surface(&[3], &faces, &coords, 1.0e12);
/// assert_eq!(tie.num_pairs(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct TiedInterface {
    pairs: Vec<TiedPair>,
    penalty: f64,
}

impl TiedInterface {
    /// Ties slave nodes to their nearest master triangles.
    ///
    /// Projects each slave onto every master face at the reference
    /// coordinates, keeps the closest projection's barycentric weights and
    /// slave-minus-master offset, and ties all three components (adjust
    /// [`TiedPair::mask`] afterwards for partially prescribed slaves). The
    /// setup-time search is quadratic; pre-filter huge slave lists with
    /// [`contact_candidates`](super::broad_phase::contact_candidates).
    ///
    /// # Panics
    ///
    /// Panics when no master face is given or the coordinate buffer is
    /// shorter than the referenced nodes.
    pub fn tie_nodes_to_surface(
        slave_nodes: &[usize],
        master_faces: &[[usize; 3]],
        ref_coords: &[f64],
        penalty: f64,
    ) -> Self {
        assert!(
            !master_faces.is_empty(),
            "tied interface needs at least one master face"
        );
        let node_at = |n: usize| {
            assert!(
                3 * n + 2 < ref_coords.len(),
                "tied interface reference coordinates too short for node {n}"
            );
            [
                ref_coords[3 * n],
                ref_coords[3 * n + 1],
                ref_coords[3 * n + 2],
            ]
        };
        let mut pairs = Vec::with_capacity(slave_nodes.len());
        for &slave in slave_nodes {
            let slave_p = node_at(slave);
            let mut best: Option<(usize, ProjectionResult)> = None;
            for (face_idx, face) in master_faces.iter().enumerate() {
                let projection = project_point_to_triangle(
                    slave_p,
                    node_at(face[0]),
                    node_at(face[1]),
                    node_at(face[2]),
                );
                let closer = match &best {
                    None => true,
                    Some((_, prev)) => projection.distance < prev.distance,
                };
                if closer {
                    best = Some((face_idx, projection));
                }
            }
            let (face_idx, projection) = best.expect("master faces are non-empty");
            let master = master_faces[face_idx];
            let weights = [
                1.0 - projection.xi - projection.eta,
                projection.xi,
                projection.eta,
            ];
            let mut master_p = [0.0; 3];
            for (weight, node) in weights.iter().zip(master.iter()) {
                let p = node_at(*node);
                master_p[0] += weight * p[0];
                master_p[1] += weight * p[1];
                master_p[2] += weight * p[2];
            }
            pairs.push(TiedPair {
                slave,
                master,
                weights,
                offset: [
                    slave_p[0] - master_p[0],
                    slave_p[1] - master_p[1],
                    slave_p[2] - master_p[2],
                ],
                mask: [true, true, true],
            });
        }
        Self { pairs, penalty }
    }

    /// Counts the tied pairs.
    pub fn num_pairs(&self) -> usize {
        self.pairs.len()
    }

    /// Borrows the tied pairs.
    pub fn pairs(&self) -> &[TiedPair] {
        &self.pairs
    }

    /// Borrows the tied pairs mutably (e.g. to adjust masks).
    pub fn pairs_mut(&mut self) -> &mut [TiedPair] {
        &mut self.pairs
    }

    /// Penalty stiffness in N/m used by
    /// [`apply_penalty`](Self::apply_penalty).
    pub fn penalty(&self) -> f64 {
        self.penalty
    }

    /// Scatters tied penalty forces into the flat `3N` force vector.
    ///
    /// Each masked component pulls the slave toward its master
    /// interpolation with stiffness [`penalty`](Self::penalty); masters
    /// receive the opposite through the barycentric weights, so momentum is
    /// conserved. Use in explicit cycles; use [`eliminate`](Self::eliminate)
    /// for exact static tying.
    ///
    /// # Panics
    ///
    /// Panics when a buffer is shorter than the referenced nodes.
    pub fn apply_penalty(&self, deformed_coords: &[f64], forces: &mut [f64]) {
        let node_at = |coords: &[f64], n: usize| {
            assert!(
                3 * n + 2 < coords.len(),
                "tied penalty buffers too short for node {n}"
            );
            [coords[3 * n], coords[3 * n + 1], coords[3 * n + 2]]
        };
        for pair in &self.pairs {
            let slave_p = node_at(deformed_coords, pair.slave);
            let mut master_p = [0.0; 3];
            for (weight, node) in pair.weights.iter().zip(pair.master.iter()) {
                let p = node_at(deformed_coords, *node);
                master_p[0] += weight * p[0];
                master_p[1] += weight * p[1];
                master_p[2] += weight * p[2];
            }
            for component in 0..3 {
                if !pair.mask[component] {
                    continue;
                }
                let relative = slave_p[component] - master_p[component] - pair.offset[component];
                let force = -self.penalty * relative;
                forces[3 * pair.slave + component] += force;
                for (weight, node) in pair.weights.iter().zip(pair.master.iter()) {
                    forces[3 * node + component] -= weight * force;
                }
            }
        }
    }

    /// Folds exact kinematic ties into a dense system.
    ///
    /// Eliminates each masked slave degree of freedom `s` via `u_s = sum(w
    /// u_m) + g` from the row-major `(3N, 3N)` stiffness and `3N` load,
    /// leaving identity rows for later [`recover`](Self::recover). Slave
    /// degrees of freedom must be unique across pairs and no master may
    /// itself be an eliminated slave (no chaining). Dirichlet conditions on
    /// masters must be re-applied afterwards, because folding touches
    /// master rows; Dirichlet-fixed slave components must be untied via
    /// [`TiedPair::mask`] instead.
    ///
    /// # Panics
    ///
    /// Panics for wrongly sized systems, chained or repeated slave degrees
    /// of freedom, or node ids beyond `num_nodes`.
    pub fn eliminate(&self, stiffness: &mut [f64], rhs: &mut [f64], num_nodes: usize) {
        let n = 3 * num_nodes;
        assert!(
            stiffness.len() == n * n && rhs.len() == n,
            "tied elimination needs a dense 3N system"
        );
        let mut eliminated = vec![false; n];
        for pair in &self.pairs {
            assert!(
                pair.slave < num_nodes && pair.master.iter().all(|m| *m < num_nodes),
                "tied pair references nodes beyond the system"
            );
            for component in 0..3 {
                if !pair.mask[component] {
                    continue;
                }
                let slave_dof = 3 * pair.slave + component;
                let master_dofs = [
                    3 * pair.master[0] + component,
                    3 * pair.master[1] + component,
                    3 * pair.master[2] + component,
                ];
                assert!(
                    !eliminated[slave_dof],
                    "tied slave degree of freedom {slave_dof} eliminated twice"
                );
                assert!(
                    master_dofs.iter().all(|m| !eliminated[*m]),
                    "tied masters must not be eliminated slaves (no chaining)"
                );
                assert!(
                    !master_dofs.contains(&slave_dof),
                    "tied slave must not be its own master"
                );
                eliminated[slave_dof] = true;
                let weights = pair.weights;
                let offset = pair.offset[component];
                let stiff_ss = stiffness[slave_dof * n + slave_dof];
                let load_s = rhs[slave_dof];
                // Fold u_s = w'u_r + g: K' = K + k_rs w' + w k_sr + k_ss w
                // w', f' = f + w(f_s - k_ss g) - g k_rs.
                let weight_of = |dof: usize| {
                    master_dofs
                        .iter()
                        .zip(weights.iter())
                        .find(|(m, _)| **m == dof)
                        .map(|(_, w)| *w)
                        .unwrap_or(0.0)
                };
                let load_coef = load_s - stiff_ss * offset;
                for row in 0..n {
                    if row == slave_dof {
                        continue;
                    }
                    let stiff_rs = stiffness[row * n + slave_dof];
                    let weight_row = weight_of(row);
                    rhs[row] += weight_row * load_coef - offset * stiff_rs;
                    for (master, weight) in master_dofs.iter().zip(weights.iter()) {
                        stiffness[row * n + *master] += stiff_rs * weight
                            + weight_row * (stiffness[slave_dof * n + *master] + stiff_ss * weight);
                    }
                    if weight_row != 0.0 {
                        for col in 0..n {
                            if col == slave_dof || master_dofs.contains(&col) {
                                continue;
                            }
                            stiffness[row * n + col] += weight_row * stiffness[slave_dof * n + col];
                        }
                    }
                }
                for k in 0..n {
                    stiffness[slave_dof * n + k] = 0.0;
                    stiffness[k * n + slave_dof] = 0.0;
                }
                stiffness[slave_dof * n + slave_dof] = 1.0;
                rhs[slave_dof] = 0.0;
            }
        }
    }

    /// Restores slave displacements after [`eliminate`](Self::eliminate).
    ///
    /// Fills each masked slave component from its master interpolation plus
    /// the reference offset. Call after solving the reduced system.
    ///
    /// # Panics
    ///
    /// Panics when the displacement buffer is shorter than the referenced
    /// nodes.
    pub fn recover(&self, displacements: &mut [f64]) {
        for pair in &self.pairs {
            for component in 0..3 {
                if !pair.mask[component] {
                    continue;
                }
                assert!(
                    3 * pair.slave + 2 < displacements.len()
                        && pair.master.iter().all(|m| 3 * m + 2 < displacements.len()),
                    "tied recovery buffer too short for pair nodes"
                );
                let mut value = pair.offset[component];
                for (weight, node) in pair.weights.iter().zip(pair.master.iter()) {
                    value += weight * displacements[3 * node + component];
                }
                displacements[3 * pair.slave + component] = value;
            }
        }
    }
}

/// Analytical rigid plane for explicit contact.
///
/// The infinite-plane wall of Theory Manual "Rigid Wall" (pp. 221-224):
/// slave nodes penetrating the plane rebound along its unit normal with
/// optional regularized Coulomb friction. The wall has infinite mass, so no
/// reaction is scattered back (the manual's moving-wall momentum note,
/// p. 223). Fixed walls use zero [`velocity`](Self::velocity).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::contact::explicit::RigidWall;
///
/// let wall = RigidWall::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0e9);
/// assert!((wall.gap([0.0, 0.0, -0.002]) + 0.002).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RigidWall {
    /// A point on the plane in m.
    pub point: [f64; 3],
    /// Unit plane normal in wall-to-body direction.
    pub normal: [f64; 3],
    /// Wall velocity in m/s (zero for a fixed wall).
    pub velocity: [f64; 3],
    /// Nodal penalty stiffness in N/m.
    pub penalty: f64,
    /// Coulomb friction coefficient, dimensionless.
    pub friction_coefficient: f64,
    /// Friction regularization velocity in m/s.
    pub friction_velocity: f64,
}

impl RigidWall {
    /// Builds a fixed frictionless wall from a point, normal, and penalty.
    ///
    /// The normal is normalized; [`velocity`](Self::velocity) starts at zero
    /// and friction off. Set the remaining fields for moving or frictional
    /// walls.
    ///
    /// # Panics
    ///
    /// Panics when the normal is zero or not finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::contact::explicit::RigidWall;
    ///
    /// let wall = RigidWall::new([0.0, 0.0, 0.0], [0.0, 0.0, 2.0], 1.0e9);
    /// assert!((wall.normal[2] - 1.0).abs() < 1e-15);
    /// ```
    pub fn new(point: [f64; 3], normal: [f64; 3], penalty: f64) -> Self {
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        assert!(
            length.is_finite() && length > 0.0,
            "rigid wall normal must be nonzero and finite"
        );
        Self {
            point,
            normal: [normal[0] / length, normal[1] / length, normal[2] / length],
            velocity: [0.0; 3],
            penalty,
            friction_coefficient: 0.0,
            friction_velocity: 0.01,
        }
    }

    /// Signed gap of a position to the plane in m.
    ///
    /// Positive on the body side (separated), negative past the plane
    /// (penetrating).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::contact::explicit::RigidWall;
    ///
    /// let wall = RigidWall::new([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 1.0e9);
    /// assert!((wall.gap([0.5, 0.0, 0.0]) - 0.5).abs() < 1e-15);
    /// ```
    pub fn gap(&self, position: [f64; 3]) -> f64 {
        (position[0] - self.point[0]) * self.normal[0]
            + (position[1] - self.point[1]) * self.normal[1]
            + (position[2] - self.point[2]) * self.normal[2]
    }
}

/// Penalty contact of slave nodes against a rigid plane.
///
/// Evaluates [`RigidWall::gap`] per slave, scatters penalty normal plus
/// regularized Coulomb friction forces into the force vector, and returns
/// one frozen [`PenaltyPair`] per penetrating node for the tiebreak lane.
/// The Taylor gate driver calls this for the anvil plane.
///
/// # Panics
///
/// Panics when the penalty or friction coefficient is negative, the normal
/// is not unit length, or a buffer is shorter than the referenced nodes.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::contact::explicit::{RigidWall, rigid_wall_contact};
///
/// let wall = RigidWall::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0e9);
/// let coords = vec![0.0, 0.0, -0.01];
/// let velocities = vec![0.0; 3];
/// let mut forces = vec![0.0; 3];
/// let pairs = rigid_wall_contact(&wall, &[0], &coords, &velocities, &mut forces);
/// assert_eq!(pairs.len(), 1);
/// assert!((forces[2] - 1.0e7).abs() < 1.0);
/// ```
pub fn rigid_wall_contact(
    wall: &RigidWall,
    slave_nodes: &[usize],
    coords: &[f64],
    velocities: &[f64],
    forces: &mut [f64],
) -> Vec<PenaltyPair> {
    assert!(
        wall.penalty >= 0.0 && wall.friction_coefficient >= 0.0,
        "rigid wall needs non-negative penalty and friction"
    );
    let normal_length = (wall.normal[0] * wall.normal[0]
        + wall.normal[1] * wall.normal[1]
        + wall.normal[2] * wall.normal[2])
        .sqrt();
    assert!(
        (normal_length - 1.0).abs() < 1.0e-9,
        "rigid wall normal must be unit length"
    );
    if let Some(&max_slave) = slave_nodes.iter().max() {
        assert!(
            coords.len() >= 3 * (max_slave + 1)
                && velocities.len() >= 3 * (max_slave + 1)
                && forces.len() >= 3 * (max_slave + 1),
            "rigid wall buffers too short for the slave nodes"
        );
    }
    let mut pairs = Vec::new();
    for &slave in slave_nodes {
        let position = [
            coords[3 * slave],
            coords[3 * slave + 1],
            coords[3 * slave + 2],
        ];
        let gap = wall.gap(position);
        if gap >= 0.0 {
            continue;
        }
        let normal_force = wall.penalty * (-gap);
        let normal = wall.normal;
        let mut rel = [
            velocities[3 * slave] - wall.velocity[0],
            velocities[3 * slave + 1] - wall.velocity[1],
            velocities[3 * slave + 2] - wall.velocity[2],
        ];
        let normal_speed = rel[0] * normal[0] + rel[1] * normal[1] + rel[2] * normal[2];
        rel[0] -= normal_speed * normal[0];
        rel[1] -= normal_speed * normal[1];
        rel[2] -= normal_speed * normal[2];
        let speed = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
        // Regularized Coulomb: mu Fn tanh(|vt| / v0) along -t.
        let friction = if speed > 0.0 && wall.friction_coefficient > 0.0 {
            let scale = wall.friction_velocity.max(1e-12);
            wall.friction_coefficient * normal_force * (speed / scale).tanh() / speed
        } else {
            0.0
        };
        let slave_force = [
            normal_force * normal[0] - friction * rel[0],
            normal_force * normal[1] - friction * rel[1],
            normal_force * normal[2] - friction * rel[2],
        ];
        forces[3 * slave] += slave_force[0];
        forces[3 * slave + 1] += slave_force[1];
        forces[3 * slave + 2] += slave_force[2];
        pairs.push(PenaltyPair {
            slave,
            gap,
            normal,
            force: slave_force,
        });
    }
    pairs
}

/// Symmetric two-pass penalty contact (TYPE7-ward hook).
///
/// Runs two one-sided [`ExplicitContact`] passes (A slaves on B master, B
/// slaves on A master), sums their forces, and concatenates their
/// [`PenaltyPair`] records. Each physical contact then contributes from
/// both discretizations, so callers typically halve each pass's penalty to
/// preserve the total interface stiffness.
///
/// TYPE7 mapping (Theory Manual "Interface Overview", pp. 234-235 and
/// "Contacts Modeling", pp. 239-248): node-to-surface both sides is this
/// symmetric pass; candidate search is
/// [`contact_candidates`](super::broad_phase::contact_candidates) (bucket
/// plus sweep, the manual's box-sort/fast-sort family); penalty processing
/// with regularized Coulomb friction is [`ExplicitContact`]; projection is
/// barycentric on triangles via
/// [`project_point_to_triangle`](super::gap::project_point_to_triangle).
/// Still future: edge-to-edge contact (TYPE11), single-surface
/// auto-contact, double-sided shell orientation handling, penetration
/// stiffening of the penalty, topological candidate search, and
/// history-state stick/slip friction.
///
/// # Panics
///
/// Panics when the buffers are shorter than either pass's `3N`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::contact::explicit::{
///     ExplicitContact, ExplicitContactConfig, symmetric_penalty_pass,
/// };
///
/// let config = ExplicitContactConfig { search_radius: 0.5, ..Default::default() };
/// let mut lower = ExplicitContact::new_two_body(&[[0, 1, 2], [0, 2, 3]], &[4, 5, 6, 7], 8, config);
/// let mut upper = ExplicitContact::new_two_body(&[[4, 6, 5], [4, 7, 6]], &[0, 1, 2, 3], 8, config);
/// let coords = vec![
///     0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
///     0.0, 0.0, -0.01, 1.0, 0.0, -0.01, 1.0, 1.0, -0.01, 0.0, 1.0, -0.01,
/// ];
/// let velocities = vec![0.0; 24];
/// let mut forces = vec![0.0; 24];
/// let pairs = symmetric_penalty_pass(&mut lower, &mut upper, &coords, &velocities, &mut forces);
/// assert_eq!(pairs.len(), 8);
/// let total: f64 = forces.iter().sum();
/// assert!(total.abs() < 1e-3);
/// ```
pub fn symmetric_penalty_pass(
    first: &mut ExplicitContact,
    second: &mut ExplicitContact,
    deformed_coords: &[f64],
    velocities: &[f64],
    forces: &mut [f64],
) -> Vec<PenaltyPair> {
    let mut pairs = first.apply_with_pairs(deformed_coords, velocities, forces);
    pairs.extend(second.apply_with_pairs(deformed_coords, velocities, forces));
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Master plate (two triangles, z = 0, outward +z) plus slave node.
    fn plate_setup(slave: [f64; 3]) -> (Vec<f64>, ExplicitContact) {
        let coords = vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            1.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, //
            slave[0], slave[1], slave[2],
        ];
        let config = ExplicitContactConfig {
            penalty: 1.0e9,
            search_radius: 0.5,
            ..Default::default()
        };
        let contact = ExplicitContact::new_two_body(&[[0, 1, 2], [0, 2, 3]], &[4], 5, config);
        (coords, contact)
    }

    #[test]
    fn penalty_pushes_penetrating_node_out() {
        let (coords, mut contact) = plate_setup([0.5, 0.5, -0.01]);
        let velocities = vec![0.0; 15];
        let mut forces = vec![0.0; 15];
        contact.apply(&coords, &velocities, &mut forces);
        // Slave pushed along +z with penalty times penetration.
        assert!((forces[3 * 4 + 2] - 1.0e9 * 0.01).abs() < 1.0);
        assert!(forces[3 * 4].abs() < 1e-6 && forces[3 * 4 + 1].abs() < 1e-6);
        // Master receives the negated force: momentum conserved.
        let total: f64 = forces.iter().sum();
        assert!(total.abs() < 1e-3, "force imbalance {total}");
    }

    #[test]
    fn separated_node_draws_no_force() {
        let (coords, mut contact) = plate_setup([0.5, 0.5, 0.1]);
        let velocities = vec![0.0; 15];
        let mut forces = vec![0.0; 15];
        contact.apply(&coords, &velocities, &mut forces);
        assert!(forces.iter().all(|f| *f == 0.0));
    }

    #[test]
    fn thickness_offset_shifts_activation() {
        let (coords, mut contact) = plate_setup([0.5, 0.5, 0.005]);
        contact.config.thickness_offset = 0.01;
        let velocities = vec![0.0; 15];
        let mut forces = vec![0.0; 15];
        contact.apply(&coords, &velocities, &mut forces);
        // Gap 0.005 sits inside the 0.01 offset: penetration 0.005.
        assert!((forces[3 * 4 + 2] - 1.0e9 * 0.005).abs() < 1.0);
    }

    #[test]
    fn friction_opposes_sliding_within_coulomb_cap() {
        let (coords, mut contact) = plate_setup([0.5, 0.5, -0.01]);
        contact.config.friction_coefficient = 0.3;
        contact.config.friction_velocity = 1e-4;
        let mut velocities = vec![0.0; 15];
        velocities[3 * 4] = 2.0;
        let mut forces = vec![0.0; 15];
        contact.apply(&coords, &velocities, &mut forces);
        let normal = forces[3 * 4 + 2];
        let tangent = forces[3 * 4];
        assert!(tangent < 0.0, "friction must oppose sliding");
        assert!(
            tangent.abs() <= 0.3 * normal + 1.0,
            "friction {tangent} exceeds mu Fn"
        );
    }

    #[test]
    fn self_contact_builds_and_runs() {
        // Closed tetrahedron surface: self-contact must filter neighbors.
        let faces = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let config = ExplicitContactConfig {
            penalty: 1.0e9,
            search_radius: 0.1,
            ..Default::default()
        };
        let mut contact = ExplicitContact::new_self(&faces, 4, config);
        assert_eq!(contact.num_slave_nodes(), 4);
        let coords = vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ];
        let velocities = vec![0.0; 12];
        let mut forces = vec![0.0; 12];
        contact.apply(&coords, &velocities, &mut forces);
        // Separated tetrahedron draws no self-contact force.
        assert!(forces.iter().all(|f| *f == 0.0));
    }

    #[test]
    fn hertz_impact_satisfies_energy_balance() {
        let (m, v, r, estar) = (2.0, 3.0, 0.05, 110.0e9);
        let approach = hertz_impact_max_approach(m, v, r, estar);
        let kinetic = 0.5 * m * v * v;
        let stored = 8.0 / 15.0 * estar * r.sqrt() * approach.powf(2.5);
        assert!((kinetic - stored).abs() / kinetic < 1e-12);
        let peak = hertz_impact_peak_force(m, v, r, estar);
        assert!((peak - 4.0 / 3.0 * estar * r.sqrt() * approach.powf(1.5)).abs() < 1e-6);
        let time = hertz_impact_contact_time(m, v, r, estar);
        assert!((time - 2.94 * approach / v).abs() / time < 1e-12);
    }

    #[test]
    fn spotweld_fails_irreversibly() {
        let mut welds = SpotweldSet::new();
        welds.push(Spotweld::new(
            1,
            [0, 0, 0],
            [1.0, 0.0, 0.0],
            MixedModeCohesiveLaw::epoxy(),
            1.0e-6,
        ));
        // Anchor at origin, slave pulled along +z past failure.
        let mut coords = vec![0.0; 6];
        coords[5] = 0.001;
        let mut forces = vec![0.0; 6];
        welds.apply(&coords, &mut forces);
        // First evaluation sets the offset; no force yet.
        assert!(forces.iter().all(|f| *f == 0.0));
        // Epoxy failure opening is ~3e-5 m; pull far past it.
        coords[5] = 0.01;
        welds.apply(&coords, &mut forces);
        assert_eq!(welds.num_failed(), 1);
        // Failed welds carry no force afterwards.
        let mut healed = vec![0.0; 6];
        healed[5] = 0.0015;
        let mut forces = vec![0.0; 6];
        welds.apply(&healed, &mut forces);
        assert!(forces.iter().all(|f| *f == 0.0));
    }

    #[test]
    fn penalty_pairs_match_scatter() {
        let (coords, mut contact) = plate_setup([0.5, 0.5, -0.01]);
        let velocities = vec![0.0; 15];
        let mut forces = vec![0.0; 15];
        let pairs = contact.apply_with_pairs(&coords, &velocities, &mut forces);
        assert_eq!(pairs.len(), 1);
        let pair = pairs[0];
        assert_eq!(pair.slave, 4);
        assert!((pair.gap + 0.01).abs() < 1e-15);
        // Recorded slave force equals the scattered slave force.
        assert!((pair.force[0] - forces[12]).abs() < 1e-9);
        assert!((pair.force[1] - forces[13]).abs() < 1e-9);
        assert!((pair.force[2] - forces[14]).abs() < 1e-9);
        // The push-out force acts along +normal, matching the frozen
        // compression-positive normal_force() (ruling P2-R1).
        assert!((pair.normal_force() - 1.0e9 * 0.01).abs() < 1.0);
        assert_eq!(pair.normal, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn separated_pass_collects_no_pairs() {
        let (coords, mut contact) = plate_setup([0.5, 0.5, 0.1]);
        let velocities = vec![0.0; 15];
        let mut forces = vec![0.0; 15];
        let pairs = contact.apply_with_pairs(&coords, &velocities, &mut forces);
        assert!(pairs.is_empty());
    }

    #[test]
    fn penalized_hertz_series_spring_identities() {
        let (force, radius, estar, penalty) = (1000.0, 0.01, 150.0e9, 1.0e10);
        let response = penalized_hertz(force, radius, estar, penalty);
        // Hertz geometry: a^3 = 3 F R / (4 E*).
        let cube = response.contact_radius.powi(3);
        assert!((cube - 3.0 * force * radius / (4.0 * estar)).abs() / cube < 1e-14);
        // Series approach: Hertz plus penalty compliance.
        assert!(
            (response.total_approach - response.hertz_approach - force / penalty).abs() < 1e-18
        );
        assert!((response.hertz_approach - response.contact_radius.powi(2) / radius).abs() < 1e-18);
        // Series stiffness from the tangent Hertz stiffness 2 E* a.
        assert!((response.hertz_stiffness - 2.0 * estar * response.contact_radius).abs() < 1e-3);
        let series = 1.0 / (1.0 / response.hertz_stiffness + 1.0 / penalty);
        assert!((response.effective_stiffness - series).abs() / series < 1e-14);
    }

    #[test]
    fn hertz_effective_modulus_rigid_limit() {
        let full = hertz_effective_modulus(200.0e9, 0.3, 200.0e9, 0.3);
        assert!((full - 200.0e9 / (2.0 * 0.91)).abs() / full < 1e-15);
        let rigid = hertz_effective_modulus(200.0e9, 0.3, f64::INFINITY, 0.0);
        assert!((rigid - 200.0e9 / 0.91).abs() / rigid < 1e-15);
    }

    /// Single master triangle plus a coincident slave node.
    fn tied_setup() -> (Vec<f64>, TiedInterface) {
        let coords = vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.25, 0.25, 0.0,
        ];
        let tie = TiedInterface::tie_nodes_to_surface(&[3], &[[0, 1, 2]], &coords, 1.0e6);
        (coords, tie)
    }

    #[test]
    fn tied_weights_reproduce_linear_field() {
        let (_, tie) = tied_setup();
        assert_eq!(tie.num_pairs(), 1);
        let pair = tie.pairs()[0];
        assert!((pair.weights[0] - 0.5).abs() < 1e-15);
        assert!((pair.weights[1] - 0.25).abs() < 1e-15);
        assert!((pair.weights[2] - 0.25).abs() < 1e-15);
        assert!(pair.offset.iter().all(|o| o.abs() < 1e-15));
        // Barycentric interpolation is exact on linear fields.
        let field = |p: [f64; 3]| 1.0 + 2.0 * p[0] + 3.0 * p[1] + 4.0 * p[2];
        let nodes = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let interpolated: f64 = pair
            .weights
            .iter()
            .zip(nodes.iter())
            .map(|(w, p)| w * field(*p))
            .sum();
        assert!((interpolated - field([0.25, 0.25, 0.0])).abs() < 1e-15);
    }

    #[test]
    fn tied_penalty_transmits_and_conserves() {
        let (mut coords, tie) = tied_setup();
        // Lift the slave 1 cm off the master plane.
        coords[3 * 3 + 2] = 0.01;
        let mut forces = vec![0.0; 12];
        tie.apply_penalty(&coords, &mut forces);
        // Slave pulled back with penalty times separation.
        assert!((forces[3 * 3 + 2] + 1.0e6 * 0.01).abs() < 1e-6);
        assert!(forces[3 * 3].abs() < 1e-9 && forces[3 * 3 + 1].abs() < 1e-9);
        // Master scatter through the weights conserves momentum.
        assert!((forces[2] - 0.5 * 1.0e6 * 0.01).abs() < 1e-6);
        assert!((forces[5] - 0.25 * 1.0e6 * 0.01).abs() < 1e-6);
        assert!((forces[8] - 0.25 * 1.0e6 * 0.01).abs() < 1e-6);
        let total: f64 = forces.iter().sum();
        assert!(total.abs() < 1e-6, "force imbalance {total}");
    }

    #[test]
    fn tied_elimination_recovers_exact_tie() {
        use crate::physics::solid::nonlinear_solid::dense_solve;

        let (_, tie) = tied_setup();
        // Identity stiffness: each DOF is a unit ground spring.
        let n = 12;
        let mut stiffness = vec![0.0; n * n];
        for i in 0..n {
            stiffness[i * n + i] = 1.0;
        }
        // Unit load on the slave z DOF only.
        let mut rhs = vec![0.0; n];
        rhs[3 * 3 + 2] = 1.0;
        tie.eliminate(&mut stiffness, &mut rhs, 4);
        dense_solve(&mut stiffness, &mut rhs, n);
        let mut displacements = rhs;
        tie.recover(&mut displacements);
        // Closed form: u_s = f w'w / (1 + w'w), u_m = f w / (1 + w'w).
        let weights = tie.pairs()[0].weights;
        let wtw: f64 = weights.iter().map(|w| w * w).sum();
        let expected_slave = wtw / (1.0 + wtw);
        assert!((displacements[3 * 3 + 2] - expected_slave).abs() < 1e-12);
        for (node, weight) in weights.iter().enumerate() {
            assert!((displacements[3 * node + 2] - weight / (1.0 + wtw)).abs() < 1e-12);
        }
        // Unloaded components stay zero; the tie holds exactly.
        assert!(displacements[3 * 3].abs() < 1e-14);
        let tied: f64 = weights
            .iter()
            .enumerate()
            .map(|(k, w)| w * displacements[3 * k + 2])
            .sum();
        assert!((displacements[3 * 3 + 2] - tied).abs() < 1e-14);
    }

    #[test]
    fn tied_mask_leaves_components_free() {
        use crate::physics::solid::nonlinear_solid::dense_solve;

        let (_, mut tie) = tied_setup();
        tie.pairs_mut()[0].mask = [true, false, true];
        let n = 12;
        let mut stiffness = vec![0.0; n * n];
        for i in 0..n {
            stiffness[i * n + i] = 1.0;
        }
        let mut rhs = vec![0.0; n];
        rhs[3 * 3 + 1] = 2.0;
        tie.eliminate(&mut stiffness, &mut rhs, 4);
        dense_solve(&mut stiffness, &mut rhs, n);
        // Untied y DOF answers its own load; tied DOFs stay put.
        assert!((rhs[3 * 3 + 1] - 2.0).abs() < 1e-14);
        assert!(rhs[3 * 3].abs() < 1e-14 && rhs[3 * 3 + 2].abs() < 1e-14);
    }

    #[test]
    fn rigid_wall_gap_force_identities() {
        let wall = RigidWall::new([0.0, 0.0, 0.5], [0.0, 0.0, 1.0], 2.0e9);
        // Gap measures body-side distance from the plane.
        assert!((wall.gap([1.0, -2.0, 0.7]) - 0.2).abs() < 1e-15);
        assert!((wall.gap([0.0, 0.0, 0.45]) + 0.05).abs() < 1e-15);
        let coords = vec![0.0, 0.0, 0.45, 0.0, 0.0, 0.7];
        let velocities = vec![0.0; 6];
        let mut forces = vec![0.0; 6];
        let pairs = rigid_wall_contact(&wall, &[0, 1], &coords, &velocities, &mut forces);
        // Only the penetrating node responds, with penalty times depth.
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].slave, 0);
        assert!((pairs[0].gap + 0.05).abs() < 1e-15);
        assert!((forces[2] - 2.0e9 * 0.05).abs() < 1.0);
        assert!(forces[3].abs() + forces[4].abs() + forces[5].abs() < 1e-12);
        // Compression-positive normal_force() matches the push-out (P2-R1).
        assert!((pairs[0].normal_force() - 2.0e9 * 0.05).abs() < 1.0);
    }

    #[test]
    fn rigid_wall_friction_opposes_sliding() {
        let mut wall = RigidWall::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0e9);
        wall.friction_coefficient = 0.4;
        wall.friction_velocity = 1e-4;
        let coords = vec![0.0, 0.0, -0.01];
        let velocities = vec![3.0, 0.0, -1.0];
        let mut forces = vec![0.0; 3];
        let pairs = rigid_wall_contact(&wall, &[0], &coords, &velocities, &mut forces);
        assert_eq!(pairs.len(), 1);
        assert!(forces[0] < 0.0, "friction must oppose sliding");
        assert!(
            forces[0].abs() <= 0.4 * forces[2] + 1.0,
            "friction exceeds the Coulomb cap"
        );
        // Normal approach velocity does not leak into friction.
        assert!(forces[1].abs() < 1e-6);
    }

    #[test]
    fn symmetric_pass_pairs_both_sides_and_conserves() {
        let config = ExplicitContactConfig {
            search_radius: 0.5,
            ..Default::default()
        };
        let mut lower =
            ExplicitContact::new_two_body(&[[0, 1, 2], [0, 2, 3]], &[4, 5, 6, 7], 8, config);
        let mut upper =
            ExplicitContact::new_two_body(&[[4, 6, 5], [4, 7, 6]], &[0, 1, 2, 3], 8, config);
        let coords = vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            1.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, -0.01, //
            1.0, 0.0, -0.01, //
            1.0, 1.0, -0.01, //
            0.0, 1.0, -0.01,
        ];
        let velocities = vec![0.0; 24];
        let mut forces = vec![0.0; 24];
        let pairs =
            symmetric_penalty_pass(&mut lower, &mut upper, &coords, &velocities, &mut forces);
        // Four slaves per pass, one pair each.
        assert_eq!(pairs.len(), 8);
        // Plates push apart and momentum is conserved.
        assert!(forces[2] < 0.0 && forces[3 * 4 + 2] > 0.0);
        let total: f64 = forces.iter().sum();
        assert!(total.abs() < 1e-3, "force imbalance {total}");
    }
}
