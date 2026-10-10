//! Penalty contact for explicit transient dynamics.
//!
//! Complements the quasi-static augmented-Lagrangian path with the force-only
//! operators an explicit integrator needs: node-to-surface penalty contact,
//! self-contact through the topological filter, regularized Coulomb friction
//! from nodal velocities, shell thickness offsets, and spotweld/tiebreak
//! failure hooked to the cohesive zone laws. Detection reuses the shared
//! BVH and closest-point projection; no tangent is assembled because the
//! central-difference scheme never needs one.

// Rust guideline compliant 2026-02-21

use super::detection::ContactSurface;
use super::gap::{ProjectionResult, project_point_to_triangle};
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
        }
        self.spotwelds.apply(deformed_coords, forces);
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
                (initial[0] * initial[0] + initial[1] * initial[1] + initial[2] * initial[2]).sqrt();
            let axis = if axis_len > 1e-15 {
                [initial[0] / axis_len, initial[1] / axis_len, initial[2] / axis_len]
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
            let helper = if axis[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
            let mut t1 = [
                axis[1] * helper[2] - axis[2] * helper[1],
                axis[2] * helper[0] - axis[0] * helper[2],
                axis[0] * helper[1] - axis[1] * helper[0],
            ];
            let t1_len = (t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]).sqrt().max(1e-30);
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
}
