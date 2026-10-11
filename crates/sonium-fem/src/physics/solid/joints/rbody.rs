//! Finite-rotation rigid bodies for explicit dynamics.
//!
//! A [`FiniteRigidBody`] carries one master node plus rigidly attached slave
//! nodes through large rotations: the body owns a 3x3 rotation matrix and an
//! angular velocity, projects slave positions and velocities from the live
//! master state each explicit cycle, aggregates mass properties from nodal
//! masses, and recovers resultant loads and momenta. It complements the
//! small-rotation [`RigidBody`](super::rigid::RigidBody) links, which stay
//! linear and untouched; use this module whenever rotations exceed the
//! small-angle regime or slave motion must follow an
//! [`ExplicitState`](sonium_compute::explicit_state::ExplicitState) cycle.
//!
//! Naming: the small-rotation type already owns the `RigidBody` name in the
//! coordinator-owned glob re-export, so the finite-rotation path is spelled
//! `FiniteRigidBody`.
//!
//! # Examples
//!
//! ```rust
//! use sonium_compute::explicit_state::ExplicitState;
//! use sonium_fem::physics::solid::joints::FiniteRigidBody;
//!
//! let reference = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
//! let body = FiniteRigidBody::new(0, vec![1], reference)
//!     .with_angular_velocity([0.0, 0.0, 0.3]);
//! let mut state = ExplicitState::new(2);
//! state.masses = vec![1.0, 1.0];
//! body.apply_to_state(&mut state);
//! assert!((state.positions[3] - 1.0).abs() < 1.0e-15);
//! ```

// Rust guideline compliant 2026-02-21

use crate::boundary::periodic::MultiPointConstraint;
use sonium_compute::explicit_state::ExplicitState;

/// Rotation matrix from an axis and angle.
///
/// Rodrigues form for a right-handed rotation about `axis` by `angle`
/// radians. The axis is normalized internally.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::joints::{rotation_determinant, rotation_matrix};
///
/// let rotation = rotation_matrix([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
/// assert!((rotation_determinant(&rotation) - 1.0).abs() < 1.0e-15);
/// // x axis maps onto y axis.
/// assert!((rotation[0][1] + 1.0).abs() < 1.0e-15);
/// assert!((rotation[1][0] - 1.0).abs() < 1.0e-15);
/// ```
///
/// # Panics
///
/// Panics for a zero-length or non-finite axis, or a non-finite angle.
#[must_use]
pub fn rotation_matrix(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
    assert!(angle.is_finite(), "rotation angle must be finite");
    let norm = dot(axis, axis).sqrt();
    assert!(
        norm.is_finite() && norm > 0.0,
        "rotation axis must be finite and nonzero"
    );
    let (x, y, z) = (axis[0] / norm, axis[1] / norm, axis[2] / norm);
    let (sine, cosine) = angle.sin_cos();
    let versine = 1.0 - cosine;
    [
        [
            cosine + x * x * versine,
            x * y * versine - z * sine,
            x * z * versine + y * sine,
        ],
        [
            y * x * versine + z * sine,
            cosine + y * y * versine,
            y * z * versine - x * sine,
        ],
        [
            z * x * versine - y * sine,
            z * y * versine + x * sine,
            cosine + z * z * versine,
        ],
    ]
}

/// Determinant of a 3x3 row-major matrix.
///
/// Rigid rotations keep this at exactly one; drift away from one
/// signals accumulated roundoff in long explicit runs.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::joints::{rotation_determinant, rotation_matrix};
///
/// let rotation = rotation_matrix([1.0, 1.0, 1.0], 0.7);
/// assert!((rotation_determinant(&rotation) - 1.0).abs() < 1.0e-15);
/// ```
#[must_use]
pub fn rotation_determinant(rotation: &[[f64; 3]; 3]) -> f64 {
    let (a, b, c) = (rotation[0], rotation[1], rotation[2]);
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

/// Rotation rate for a spatial angular velocity.
///
/// Returns `R_dot = skew(omega) * R`, the time derivative of a rotation
/// matrix whose body spins with spatial angular velocity `omega`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::joints::{
///     angular_velocity_from_rate, rotation_matrix, rotation_rate,
/// };
///
/// let rotation = rotation_matrix([0.0, 0.0, 1.0], 0.3);
/// let omega = [0.1, -0.2, 0.3];
/// let rate = rotation_rate(&rotation, omega);
/// let recovered = angular_velocity_from_rate(&rotation, &rate);
/// for i in 0..3 {
///     assert!((recovered[i] - omega[i]).abs() < 1.0e-14);
/// }
/// ```
///
/// # Panics
///
/// Panics for non-finite rotation or angular-velocity entries.
#[must_use]
pub fn rotation_rate(rotation: &[[f64; 3]; 3], omega: [f64; 3]) -> [[f64; 3]; 3] {
    assert!(is_finite_matrix(rotation), "rotation must be finite");
    assert!(is_finite_vector(omega), "angular velocity must be finite");
    mat_mul(&skew(omega), rotation)
}

/// Angular velocity recovered from a rotation rate.
///
/// Forms `Omega = R_dot * R^T` and returns its axial vector, so the
/// result is the spatial angular velocity that generated `rate`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::joints::{
///     angular_velocity_from_rate, rotation_matrix, rotation_rate,
/// };
///
/// let rotation = rotation_matrix([0.0, 1.0, 0.0], 1.1);
/// let rate = rotation_rate(&rotation, [0.0, 2.0, 0.0]);
/// let omega = angular_velocity_from_rate(&rotation, &rate);
/// assert!((omega[1] - 2.0).abs() < 1.0e-14);
/// ```
///
/// # Panics
///
/// Panics for non-finite rotation or rotation-rate entries.
#[must_use]
pub fn angular_velocity_from_rate(rotation: &[[f64; 3]; 3], rate: &[[f64; 3]; 3]) -> [f64; 3] {
    assert!(is_finite_matrix(rotation), "rotation must be finite");
    assert!(is_finite_matrix(rate), "rotation rate must be finite");
    let omega_mat = mat_mul(rate, &mat_transpose(rotation));
    [
        0.5 * (omega_mat[2][1] - omega_mat[1][2]),
        0.5 * (omega_mat[0][2] - omega_mat[2][0]),
        0.5 * (omega_mat[1][0] - omega_mat[0][1]),
    ]
}

/// Aggregated rigid-body mass properties.
///
/// Inertia is taken about the center of mass in the reference
/// configuration; rotate it by the body rotation for current-frame use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidInertia {
    /// Total mass in kg over master plus slaves.
    pub mass: f64,
    /// Center of mass in meters (reference configuration).
    pub center_of_mass: [f64; 3],
    /// Symmetric 3x3 inertia about the center of mass in kg m^2.
    pub inertia: [[f64; 3]; 3],
}

/// Finite-rotation rigid body with master-slave projection.
///
/// The body stores reference positions indexed by node id (mirroring the
/// small-rotation [`RigidBody`](super::rigid::RigidBody) layout), a rotation
/// matrix, and a spatial angular velocity. Each explicit cycle,
/// [`apply_to_state`](Self::apply_to_state) rebuilds slave positions as
/// `x_s = x_m + R (X_s - X_m)` and slave velocities as
/// `v_s = v_m + omega x R (X_s - X_m)` from the live master state, so the
/// master keeps its normal integrator (or imposed-driver) update while the
/// slaves follow rigidly.
#[derive(Debug, Clone)]
pub struct FiniteRigidBody {
    master: usize,
    slaves: Vec<usize>,
    reference: Vec<[f64; 3]>,
    rotation: [[f64; 3]; 3],
    angular_velocity: [f64; 3],
}

impl FiniteRigidBody {
    /// Create a body with identity rotation and zero spin.
    ///
    /// Reference positions are indexed by node id and shared with the
    /// caller mesh; only master and slave entries are read.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::joints::FiniteRigidBody;
    ///
    /// let body = FiniteRigidBody::new(0, vec![1, 2], vec![[0.0; 3]; 3]);
    /// assert_eq!(body.master(), 0);
    /// assert_eq!(body.slaves(), &[1, 2]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when the master or a slave id is outside `reference`, when
    /// a slave repeats the master, or when a reference entry is non-finite.
    #[must_use]
    pub fn new(master: usize, slaves: Vec<usize>, reference: Vec<[f64; 3]>) -> Self {
        assert!(
            master < reference.len(),
            "master node {master} outside reference positions"
        );
        for &slave in &slaves {
            assert!(
                slave < reference.len(),
                "slave node {slave} outside reference positions"
            );
            assert!(
                slave != master,
                "slave node {slave} must differ from the master"
            );
        }
        for position in &reference {
            assert!(
                is_finite_vector(*position),
                "reference positions must be finite"
            );
        }
        Self {
            master,
            slaves,
            reference,
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            angular_velocity: [0.0; 3],
        }
    }

    /// Master node carrying the independent motion.
    #[must_use]
    pub fn master(&self) -> usize {
        self.master
    }

    /// Slave nodes rigidly attached to the master.
    #[must_use]
    pub fn slaves(&self) -> &[usize] {
        &self.slaves
    }

    /// Reference positions indexed by node id.
    #[must_use]
    pub fn reference(&self) -> &[[f64; 3]] {
        &self.reference
    }

    /// Current rotation matrix taking reference offsets to current offsets.
    #[must_use]
    pub fn rotation(&self) -> &[[f64; 3]; 3] {
        &self.rotation
    }

    /// Spatial angular velocity in rad/s.
    #[must_use]
    pub fn angular_velocity(&self) -> [f64; 3] {
        self.angular_velocity
    }

    /// Replace the rotation matrix, checking orthogonality loosely.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::joints::{FiniteRigidBody, rotation_matrix};
    ///
    /// let body = FiniteRigidBody::new(0, vec![1], vec![[0.0; 3]; 2])
    ///     .with_rotation(rotation_matrix([0.0, 0.0, 1.0], 0.3));
    /// assert!((body.determinant() - 1.0).abs() < 1.0e-15);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics for a non-finite matrix, or when `R R^T` deviates from the
    /// identity by more than `1.0e-9` in any entry.
    #[must_use]
    pub fn with_rotation(mut self, rotation: [[f64; 3]; 3]) -> Self {
        self.set_rotation(rotation);
        self
    }

    /// Replace the spatial angular velocity in rad/s.
    ///
    /// # Panics
    ///
    /// Panics for a non-finite angular velocity.
    #[must_use]
    pub fn with_angular_velocity(mut self, angular_velocity: [f64; 3]) -> Self {
        self.set_angular_velocity(angular_velocity);
        self
    }

    /// Set the rotation matrix, checking orthogonality loosely.
    ///
    /// # Panics
    ///
    /// Panics for a non-finite matrix, or when `R R^T` deviates from the
    /// identity by more than `1.0e-9` in any entry.
    pub fn set_rotation(&mut self, rotation: [[f64; 3]; 3]) {
        assert!(is_finite_matrix(&rotation), "rotation must be finite");
        let check = mat_mul(&rotation, &mat_transpose(&rotation));
        for (i, row) in check.iter().enumerate() {
            for (j, &entry) in row.iter().enumerate() {
                let expected = f64::from(i == j);
                assert!(
                    (entry - expected).abs() <= 1.0e-9,
                    "rotation must stay orthogonal"
                );
            }
        }
        self.rotation = rotation;
    }

    /// Set the rotation from an axis and angle.
    ///
    /// # Panics
    ///
    /// Panics for a zero-length or non-finite axis, or a non-finite angle.
    pub fn set_rotation_from_axis_angle(&mut self, axis: [f64; 3], angle: f64) {
        self.rotation = rotation_matrix(axis, angle);
    }

    /// Set the spatial angular velocity in rad/s.
    ///
    /// # Panics
    ///
    /// Panics for a non-finite angular velocity.
    pub fn set_angular_velocity(&mut self, angular_velocity: [f64; 3]) {
        assert!(
            is_finite_vector(angular_velocity),
            "angular velocity must be finite"
        );
        self.angular_velocity = angular_velocity;
    }

    /// Determinant of the current rotation matrix.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        rotation_determinant(&self.rotation)
    }

    /// Advance the rotation by the stored spin over `dt`.
    ///
    /// Left-multiplies the incremental Rodrigues rotation for the spatial
    /// angular velocity, `R <- dR(omega dt) R`, which is the exact flow
    /// for constant spin over the step.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::joints::FiniteRigidBody;
    ///
    /// let mut body = FiniteRigidBody::new(0, vec![1], vec![[0.0; 3]; 2])
    ///     .with_angular_velocity([0.0, 0.0, 1.0]);
    /// body.advance_rotation(0.1);
    /// assert!((body.determinant() - 1.0).abs() < 1.0e-14);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics for a non-finite time step.
    pub fn advance_rotation(&mut self, dt: f64) {
        assert!(dt.is_finite(), "time step must be finite");
        let rate = dot(self.angular_velocity, self.angular_velocity).sqrt();
        if rate == 0.0 || dt == 0.0 {
            return;
        }
        let axis = [
            self.angular_velocity[0] / rate,
            self.angular_velocity[1] / rate,
            self.angular_velocity[2] / rate,
        ];
        let increment = rotation_matrix(axis, rate * dt);
        self.rotation = mat_mul(&increment, &self.rotation);
    }

    /// Re-orthonormalize the rotation against roundoff drift.
    ///
    /// Gram-Schmidt on the rows plus a cross-product rebuild of the third
    /// row keeps `det(R) = 1` over long explicit runs. Call periodically,
    /// not necessarily every cycle.
    pub fn renormalize_rotation(&mut self) {
        let mut rows = self.rotation;
        for row in &mut rows[0..2] {
            let norm = dot(*row, *row).sqrt();
            if norm > 0.0 {
                row[0] /= norm;
                row[1] /= norm;
                row[2] /= norm;
            }
        }
        // Remove the row-0 component of row 1, then renormalize.
        let along = dot(rows[1], rows[0]);
        rows[1][0] -= along * rows[0][0];
        rows[1][1] -= along * rows[0][1];
        rows[1][2] -= along * rows[0][2];
        let norm = dot(rows[1], rows[1]).sqrt();
        if norm > 0.0 {
            rows[1][0] /= norm;
            rows[1][1] /= norm;
            rows[1][2] /= norm;
        }
        rows[2] = cross(rows[0], rows[1]);
        self.rotation = rows;
    }

    /// Reference offset from the master to `node`.
    ///
    /// # Panics
    ///
    /// Panics when `node` is outside the reference positions.
    #[must_use]
    pub fn relative_reference(&self, node: usize) -> [f64; 3] {
        assert!(
            node < self.reference.len(),
            "node {node} outside reference positions"
        );
        [
            self.reference[node][0] - self.reference[self.master][0],
            self.reference[node][1] - self.reference[self.master][1],
            self.reference[node][2] - self.reference[self.master][2],
        ]
    }

    /// Current offset from the master to `node`.
    ///
    /// Returns `R (X_node - X_master)` with the stored rotation.
    ///
    /// # Panics
    ///
    /// Panics when `node` is outside the reference positions.
    #[must_use]
    pub fn current_offset(&self, node: usize) -> [f64; 3] {
        mat_vec(&self.rotation, self.relative_reference(node))
    }

    /// Slave position for a given master position.
    #[must_use]
    pub fn slave_position(&self, slave: usize, master_position: [f64; 3]) -> [f64; 3] {
        let offset = self.current_offset(slave);
        [
            master_position[0] + offset[0],
            master_position[1] + offset[1],
            master_position[2] + offset[2],
        ]
    }

    /// Slave velocity for a given master velocity.
    #[must_use]
    pub fn slave_velocity(&self, slave: usize, master_velocity: [f64; 3]) -> [f64; 3] {
        let spin = cross(self.angular_velocity, self.current_offset(slave));
        [
            master_velocity[0] + spin[0],
            master_velocity[1] + spin[1],
            master_velocity[2] + spin[2],
        ]
    }

    /// Project slaves from the live master state.
    ///
    /// Reads the master position and velocity from `state`, then overwrites
    /// every slave position and velocity with the rigid motion. Call once
    /// per explicit cycle after the master update; the master entries are
    /// left untouched.
    ///
    /// # Panics
    ///
    /// Panics when a body node id exceeds the state, or when the master
    /// position or velocity read back non-finite.
    pub fn apply_to_state(&self, state: &mut ExplicitState) {
        let nodes = state.node_count();
        assert!(self.master < nodes, "master node outside explicit state");
        let master_position = [
            state.positions[3 * self.master],
            state.positions[3 * self.master + 1],
            state.positions[3 * self.master + 2],
        ];
        let master_velocity = [
            state.velocities[3 * self.master],
            state.velocities[3 * self.master + 1],
            state.velocities[3 * self.master + 2],
        ];
        assert!(
            is_finite_vector(master_position) && is_finite_vector(master_velocity),
            "master state must be finite"
        );
        for &slave in &self.slaves {
            assert!(slave < nodes, "slave node {slave} outside explicit state");
            let position = self.slave_position(slave, master_position);
            let velocity = self.slave_velocity(slave, master_velocity);
            state.positions[3 * slave..3 * slave + 3].copy_from_slice(&position);
            state.velocities[3 * slave..3 * slave + 3].copy_from_slice(&velocity);
        }
    }

    /// Aggregate mass, center of mass, and COM inertia.
    ///
    /// Sums over the master plus slaves with reference-configuration
    /// geometry and the caller-supplied nodal masses.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::joints::FiniteRigidBody;
    ///
    /// let body = FiniteRigidBody::new(
    ///     0,
    ///     vec![1],
    ///     vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    /// );
    /// let inertia = body.aggregate(&[1.0, 1.0]);
    /// assert!((inertia.mass - 2.0).abs() < 1.0e-15);
    /// assert!((inertia.center_of_mass[0] - 1.0).abs() < 1.0e-15);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when `masses` is shorter than the node ids, or holds
    /// negative or non-finite entries on body nodes.
    #[must_use]
    pub fn aggregate(&self, masses: &[f64]) -> RigidInertia {
        let mut total = 0.0;
        let mut center = [0.0; 3];
        let mut nodes = Vec::with_capacity(self.slaves.len() + 1);
        nodes.push(self.master);
        nodes.extend_from_slice(&self.slaves);
        for &node in &nodes {
            assert!(
                node < masses.len(),
                "node {node} outside mass vector of length {}",
                masses.len()
            );
            let mass = masses[node];
            assert!(
                mass.is_finite() && mass >= 0.0,
                "nodal masses must be finite and nonnegative"
            );
            total += mass;
            center[0] += mass * self.reference[node][0];
            center[1] += mass * self.reference[node][1];
            center[2] += mass * self.reference[node][2];
        }
        if total > 0.0 {
            center[0] /= total;
            center[1] /= total;
            center[2] /= total;
        }
        let mut inertia = [[0.0; 3]; 3];
        for &node in &nodes {
            let mass = masses[node];
            let arm = [
                self.reference[node][0] - center[0],
                self.reference[node][1] - center[1],
                self.reference[node][2] - center[2],
            ];
            let squared = dot(arm, arm);
            for i in 0..3 {
                for j in 0..3 {
                    inertia[i][j] += mass * (squared * f64::from(i == j) - arm[i] * arm[j]);
                }
            }
        }
        RigidInertia {
            mass: total,
            center_of_mass: center,
            inertia,
        }
    }

    /// Resultant force and moment about the master.
    ///
    /// Sums the live nodal forces over the master plus slaves with arms
    /// taken from current positions, giving the net external load the
    /// master integrator must carry (`M a_m = F`, `I alpha = M`).
    ///
    /// # Panics
    ///
    /// Panics when a body node id exceeds the state.
    #[must_use]
    pub fn resultant_load(&self, state: &ExplicitState) -> ([f64; 3], [f64; 3]) {
        let nodes = state.node_count();
        assert!(self.master < nodes, "master node outside explicit state");
        let master_position = [
            state.positions[3 * self.master],
            state.positions[3 * self.master + 1],
            state.positions[3 * self.master + 2],
        ];
        let mut force = [0.0; 3];
        let mut moment = [0.0; 3];
        let mut accumulate = |node: usize| {
            assert!(node < nodes, "node {node} outside explicit state");
            let nodal = [
                state.forces[3 * node],
                state.forces[3 * node + 1],
                state.forces[3 * node + 2],
            ];
            let arm = [
                state.positions[3 * node] - master_position[0],
                state.positions[3 * node + 1] - master_position[1],
                state.positions[3 * node + 2] - master_position[2],
            ];
            let couple = cross(arm, nodal);
            force[0] += nodal[0];
            force[1] += nodal[1];
            force[2] += nodal[2];
            moment[0] += couple[0];
            moment[1] += couple[1];
            moment[2] += couple[2];
        };
        accumulate(self.master);
        for &slave in &self.slaves {
            accumulate(slave);
        }
        (force, moment)
    }

    /// Linear and angular momenta about the master.
    ///
    /// Sums `m v` and `r x m v` over the master plus slaves with arms from
    /// current positions, for momentum-consistency checks of the projected
    /// rigid velocity field.
    ///
    /// # Panics
    ///
    /// Panics when a body node id exceeds the state.
    #[must_use]
    pub fn momenta(&self, state: &ExplicitState) -> ([f64; 3], [f64; 3]) {
        let nodes = state.node_count();
        assert!(self.master < nodes, "master node outside explicit state");
        let master_position = [
            state.positions[3 * self.master],
            state.positions[3 * self.master + 1],
            state.positions[3 * self.master + 2],
        ];
        let mut linear = [0.0; 3];
        let mut angular = [0.0; 3];
        let mut accumulate = |node: usize| {
            assert!(node < nodes, "node {node} outside explicit state");
            let mass = state.masses[node];
            let velocity = [
                state.velocities[3 * node],
                state.velocities[3 * node + 1],
                state.velocities[3 * node + 2],
            ];
            let arm = [
                state.positions[3 * node] - master_position[0],
                state.positions[3 * node + 1] - master_position[1],
                state.positions[3 * node + 2] - master_position[2],
            ];
            let spin = cross(
                arm,
                [mass * velocity[0], mass * velocity[1], mass * velocity[2]],
            );
            linear[0] += mass * velocity[0];
            linear[1] += mass * velocity[1];
            linear[2] += mass * velocity[2];
            angular[0] += spin[0];
            angular[1] += spin[1];
            angular[2] += spin[2];
        };
        accumulate(self.master);
        for &slave in &self.slaves {
            accumulate(slave);
        }
        (linear, angular)
    }

    /// Small-rotation constraints rebuilt at the current orientation.
    ///
    /// Reuses [`rigid_link_constraints`](super::rigid::rigid_link_constraints)
    /// with the rotated offsets for implicit coupling or diagnostics; the
    /// explicit path uses [`apply_to_state`](Self::apply_to_state) instead.
    #[must_use]
    pub fn linearized_constraints(&self) -> Vec<MultiPointConstraint> {
        let mut out = Vec::with_capacity(6 * self.slaves.len());
        for &slave in &self.slaves {
            out.extend(super::rigid::rigid_link_constraints(
                self.master,
                slave,
                self.current_offset(slave),
                6,
            ));
        }
        out
    }
}

/// Dot product of two 3-vectors.
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product `a x b`.
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Skew-symmetric matrix of a 3-vector.
fn skew(omega: [f64; 3]) -> [[f64; 3]; 3] {
    [
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0],
    ]
}

/// Row-major 3x3 matrix product.
fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// Row-major 3x3 transpose.
fn mat_transpose(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

/// Row-major matrix-vector product.
fn mat_vec(a: &[[f64; 3]; 3], x: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
        a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
        a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2],
    ]
}

/// Finiteness check for a 3-vector.
fn is_finite_vector(v: [f64; 3]) -> bool {
    v[0].is_finite() && v[1].is_finite() && v[2].is_finite()
}

/// Finiteness check for a 3x3 matrix.
fn is_finite_matrix(m: &[[f64; 3]; 3]) -> bool {
    m.iter().all(|row| is_finite_vector(*row))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four-node tetrahedron reference used across projection tests.
    fn tetra_reference() -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }

    fn tetra_body() -> FiniteRigidBody {
        FiniteRigidBody::new(0, vec![1, 2, 3], tetra_reference())
    }

    fn pair_distance(positions: &[f64], a: usize, b: usize) -> f64 {
        let dx = positions[3 * a] - positions[3 * b];
        let dy = positions[3 * a + 1] - positions[3 * b + 1];
        let dz = positions[3 * a + 2] - positions[3 * b + 2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    #[test]
    fn rotation_matrix_is_orthonormal() {
        let rotation = rotation_matrix([1.0, 2.0, 3.0], 0.7);
        let check = mat_mul(&rotation, &mat_transpose(&rotation));
        for (i, row) in check.iter().enumerate() {
            for (j, &entry) in row.iter().enumerate() {
                let expected = f64::from(i == j);
                assert!(
                    (entry - expected).abs() < 1.0e-15,
                    "orthonormality failed at ({i}, {j})"
                );
            }
        }
        assert!((rotation_determinant(&rotation) - 1.0).abs() < 1.0e-15);
    }

    #[test]
    fn advance_rotation_matches_axis_angle() {
        let mut body = tetra_body().with_angular_velocity([0.0, 0.0, 0.3]);
        for _ in 0..100 {
            body.advance_rotation(0.01);
        }
        let expected = rotation_matrix([0.0, 0.0, 1.0], 0.3);
        for (got_row, want_row) in body.rotation.iter().zip(expected.iter()) {
            for (&got, &want) in got_row.iter().zip(want_row.iter()) {
                assert!(
                    (got - want).abs() < 1.0e-12,
                    "rotation integration drift: {got} vs {want}"
                );
            }
        }
        assert!((body.determinant() - 1.0).abs() < 1.0e-14);
    }

    #[test]
    fn omega_recovery_round_trip() {
        let rotation = rotation_matrix([1.0, -1.0, 0.5], 1.2);
        let omega = [0.4, -1.7, 2.3];
        let rate = rotation_rate(&rotation, omega);
        let recovered = angular_velocity_from_rate(&rotation, &rate);
        for (component, (&got, &want)) in recovered.iter().zip(omega.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1.0e-13,
                "omega recovery failed at component {component}"
            );
        }
    }

    #[test]
    fn slave_projection_preserves_distances() {
        let mut body = tetra_body();
        body.set_rotation_from_axis_angle([1.0, 1.0, 1.0], 0.7);
        body.set_angular_velocity([0.2, -0.1, 0.4]);
        let mut state = ExplicitState::new(4);
        state.masses = vec![1.0; 4];
        state.positions[0..3].copy_from_slice(&[2.0, -1.0, 0.5]);
        state.velocities[0..3].copy_from_slice(&[0.5, 0.25, -0.75]);
        body.apply_to_state(&mut state);
        let reference: Vec<f64> = tetra_reference().iter().flatten().copied().collect();
        for a in 0..4 {
            for b in (a + 1)..4 {
                let before = pair_distance(&reference, a, b);
                let after = pair_distance(&state.positions, a, b);
                assert!(
                    (after - before).abs() < 1.0e-14,
                    "pair ({a}, {b}) stretched: {after} vs {before}"
                );
            }
        }
    }

    #[test]
    fn aggregate_matches_closed_form() {
        let body = FiniteRigidBody::new(0, vec![1], vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let inertia = body.aggregate(&[1.0, 1.0]);
        assert!((inertia.mass - 2.0).abs() < 1.0e-15);
        assert!((inertia.center_of_mass[0] - 1.0).abs() < 1.0e-15);
        // Two unit masses at x = +-1 about the center: Iyy = Izz = 2.
        assert!(inertia.inertia[0][0].abs() < 1.0e-15);
        assert!((inertia.inertia[1][1] - 2.0).abs() < 1.0e-15);
        assert!((inertia.inertia[2][2] - 2.0).abs() < 1.0e-15);
        assert!(inertia.inertia[0][1].abs() < 1.0e-15);
    }

    #[test]
    fn momenta_match_rigid_field() {
        let mut body = tetra_body().with_angular_velocity([0.0, 0.0, 2.0]);
        body.set_rotation_from_axis_angle([0.0, 0.0, 1.0], 0.5);
        let mut state = ExplicitState::new(4);
        state.masses = vec![1.0, 2.0, 3.0, 4.0];
        state.positions[0..3].copy_from_slice(&[1.0, 2.0, 3.0]);
        state.velocities[0..3].copy_from_slice(&[0.5, -0.5, 1.0]);
        body.apply_to_state(&mut state);
        let (linear, angular) = body.momenta(&state);
        let inertia = body.aggregate(&state.masses);
        // Linear momentum equals total mass times COM velocity.
        let com_offset = mat_vec(
            &rotation_matrix([0.0, 0.0, 1.0], 0.5),
            [
                inertia.center_of_mass[0] - body.reference[body.master][0],
                inertia.center_of_mass[1] - body.reference[body.master][1],
                inertia.center_of_mass[2] - body.reference[body.master][2],
            ],
        );
        let com_spin = cross(body.angular_velocity, com_offset);
        let com_velocity = [0.5 + com_spin[0], -0.5 + com_spin[1], 1.0 + com_spin[2]];
        for (component, (&got, &velocity)) in linear.iter().zip(com_velocity.iter()).enumerate() {
            assert!(
                (got - inertia.mass * velocity).abs() < 1.0e-12,
                "linear momentum mismatch at {component}"
            );
        }
        // Angular momentum about the COM equals I_com * omega in the
        // current frame: shift from the master, then compare.
        let rotation = rotation_matrix([0.0, 0.0, 1.0], 0.5);
        let rotated_inertia = mat_mul(
            &mat_mul(&rotation, &inertia.inertia),
            &mat_transpose(&rotation),
        );
        let expected = mat_vec(&rotated_inertia, body.angular_velocity);
        let shift = cross(com_offset, linear);
        for (component, ((&got, &offset), &want)) in angular
            .iter()
            .zip(shift.iter())
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                (got - offset - want).abs() < 1.0e-11,
                "angular momentum mismatch at {component}"
            );
        }
    }

    #[test]
    fn resultant_matches_hand_sum() {
        let body = tetra_body();
        let mut state = ExplicitState::new(4);
        state.masses = vec![1.0; 4];
        for (node, position) in tetra_reference().iter().enumerate() {
            state.positions[3 * node..3 * node + 3].copy_from_slice(position);
        }
        // Unit x-force on node 1 at arm (1, 0, 0): no moment.
        // Unit y-force on node 2 at arm (0, 1, 0): no moment.
        // Unit z-force on node 3 at arm (0, 0, 1): no moment.
        state.forces = vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ];
        let (force, moment) = body.resultant_load(&state);
        assert!((force[0] - 1.0).abs() < 1.0e-15);
        assert!((force[1] - 1.0).abs() < 1.0e-15);
        assert!((force[2] - 1.0).abs() < 1.0e-15);
        assert!(moment[0].abs() < 1.0e-15);
        assert!(moment[1].abs() < 1.0e-15);
        assert!(moment[2].abs() < 1.0e-15);
        // Off-axis load: z-force on node 1 gives moment -y.
        state.forces[5] = 2.0;
        let (_, moment) = body.resultant_load(&state);
        assert!(moment[0].abs() < 1.0e-15);
        assert!((moment[1] + 2.0).abs() < 1.0e-15);
        assert!(moment[2].abs() < 1.0e-15);
    }

    #[test]
    fn linearized_constraints_use_rotated_offsets() {
        let mut body = FiniteRigidBody::new(0, vec![1], vec![[0.0; 3], [1.0, 0.0, 0.0]]);
        body.set_rotation_from_axis_angle([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
        // Offset (1, 0, 0) rotates onto (0, 1, 0).
        let offset = body.current_offset(1);
        assert!(offset[0].abs() < 1.0e-15);
        assert!((offset[1] - 1.0).abs() < 1.0e-15);
        let constraints = body.linearized_constraints();
        assert_eq!(constraints.len(), 6);
        // Slave ux picks up -ry * theta_z = -1 * theta_z.
        let arm = constraints[0]
            .masters
            .iter()
            .find(|term| term.dof == 5)
            .unwrap();
        assert!((arm.coefficient.re + 1.0).abs() < 1.0e-15);
    }

    #[test]
    fn renormalize_repairs_drift() {
        let mut body = tetra_body().with_angular_velocity([0.7, -0.3, 0.5]);
        for _ in 0..2000 {
            body.advance_rotation(0.01);
        }
        body.renormalize_rotation();
        let check = mat_mul(body.rotation(), &mat_transpose(body.rotation()));
        for (i, row) in check.iter().enumerate() {
            for (j, &entry) in row.iter().enumerate() {
                let expected = f64::from(i == j);
                assert!(
                    (entry - expected).abs() < 1.0e-15,
                    "renormalization failed at ({i}, {j})"
                );
            }
        }
        assert!((body.determinant() - 1.0).abs() < 1.0e-15);
    }
}
