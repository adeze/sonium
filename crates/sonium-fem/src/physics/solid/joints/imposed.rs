//! Imposed displacement, velocity, and acceleration histories.
//!
//! An [`ImposedMotion`] drives a node set along one direction with a scalar
//! [`TimeHistory`] (piecewise-linear table, closed-form profile, or a
//! caller-supplied hook), applied per explicit cycle over the frozen
//! [`ExplicitState`](sonium_compute::explicit_state::ExplicitState). It covers
//! IMPOSED rigid drivers: displacement histories pin the parallel position and
//! velocity, velocity histories pin the parallel velocity, and acceleration
//! histories scatter `m a` into the force scratch. Components transverse to
//! the drive direction are always preserved from the live state.
//!
//! # Examples
//!
//! ```rust
//! use sonium_compute::explicit_state::ExplicitState;
//! use sonium_fem::physics::solid::joints::{ImposedKind, ImposedMotion, TimeHistory};
//!
//! let motion = ImposedMotion::new(
//!     vec![0],
//!     vec![[0.0, 0.0, 0.0]],
//!     [1.0, 0.0, 0.0],
//!     ImposedKind::Velocity,
//!     TimeHistory::Constant { value: 2.0 },
//! );
//! let mut state = ExplicitState::new(1);
//! motion.apply(&mut state, 0.0);
//! assert!((state.velocities[0] - 2.0).abs() < 1.0e-15);
//! ```

// Rust guideline compliant 2026-02-21

use sonium_compute::explicit_state::ExplicitState;
use std::fmt;
use std::sync::Arc;

/// Imposed quantity along the drive direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImposedKind {
    /// Pin parallel position and velocity from the history.
    Displacement,
    /// Pin parallel velocity from the history.
    Velocity,
    /// Scatter `m a` into forces from the history.
    Acceleration,
}

/// One piecewise-linear knot: value at a time in seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeSample {
    /// Knot time in seconds.
    pub time: f64,
    /// History value at the knot.
    pub value: f64,
}

/// Scalar time history driving an imposed motion.
///
/// Tables interpolate linearly between knots and clamp past the ends;
/// closed forms evaluate analytically with analytic derivatives; custom
/// hooks defer to caller code with finite-difference derivatives.
#[derive(Clone)]
pub enum TimeHistory {
    /// Piecewise-linear knots sorted by ascending time.
    Table {
        /// Knots in ascending time order with unique times.
        samples: Vec<TimeSample>,
    },
    /// Constant value for all times.
    Constant {
        /// History value.
        value: f64,
    },
    /// Linear ramp `rate * t`.
    Ramp {
        /// Slope per second.
        rate: f64,
    },
    /// Sine profile `amplitude * sin(omega t + phase)`.
    Sine {
        /// Peak amplitude.
        amplitude: f64,
        /// Angular frequency in rad/s.
        angular_frequency: f64,
        /// Phase offset in radians.
        phase: f64,
    },
    /// Caller-supplied profile evaluated at the query time.
    Custom(Arc<dyn Fn(f64) -> f64 + Send + Sync>),
}

impl fmt::Debug for TimeHistory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Table { samples } => f.debug_struct("Table").field("samples", samples).finish(),
            Self::Constant { value } => f.debug_struct("Constant").field("value", value).finish(),
            Self::Ramp { rate } => f.debug_struct("Ramp").field("rate", rate).finish(),
            Self::Sine {
                amplitude,
                angular_frequency,
                phase,
            } => f
                .debug_struct("Sine")
                .field("amplitude", amplitude)
                .field("angular_frequency", angular_frequency)
                .field("phase", phase)
                .finish(),
            Self::Custom(_) => f.debug_tuple("Custom").field(&"..").finish(),
        }
    }
}

impl TimeHistory {
    /// Build a clamped piecewise-linear table from knots.
    ///
    /// Knots are sorted by time; duplicates collapse to the last value so
    /// evaluation stays total on any input order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::joints::{TimeHistory, TimeSample};
    ///
    /// let history = TimeHistory::table(vec![
    ///     TimeSample { time: 1.0, value: 2.0 },
    ///     TimeSample { time: 0.0, value: 0.0 },
    /// ]);
    /// assert!((history.evaluate(0.5) - 1.0).abs() < 1.0e-15);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics for an empty sample list or a non-finite knot.
    #[must_use]
    pub fn table(mut samples: Vec<TimeSample>) -> Self {
        assert!(!samples.is_empty(), "time table needs at least one knot");
        for sample in &samples {
            assert!(
                sample.time.is_finite() && sample.value.is_finite(),
                "time table knots must be finite"
            );
        }
        samples.sort_by(|a, b| a.time.total_cmp(&b.time));
        let mut unique: Vec<TimeSample> = Vec::with_capacity(samples.len());
        for sample in samples {
            match unique.last_mut() {
                Some(last) if last.time == sample.time => {
                    last.value = sample.value;
                }
                _ => unique.push(sample),
            }
        }
        Self::Table { samples: unique }
    }

    /// Evaluate the history value at time `t`.
    ///
    /// # Panics
    ///
    /// Panics for a non-finite query time, or for a directly constructed
    /// `Table` with no knots (the [`table`](Self::table) constructor
    /// rejects that upfront).
    #[must_use]
    pub fn evaluate(&self, t: f64) -> f64 {
        assert!(t.is_finite(), "history time must be finite");
        match self {
            Self::Table { samples } => {
                assert!(!samples.is_empty(), "time table needs at least one knot");
                if t <= samples[0].time {
                    return samples[0].value;
                }
                if t >= samples[samples.len() - 1].time {
                    return samples[samples.len() - 1].value;
                }
                let upper = samples.partition_point(|sample| sample.time <= t);
                let (left, right) = (samples[upper - 1], samples[upper]);
                let span = right.time - left.time;
                // Zero-width guard for hand-built tables; the constructor
                // collapses duplicates so this only triggers on direct
                // variant construction with repeated times.
                if span <= 0.0 {
                    return left.value;
                }
                let fraction = (t - left.time) / span;
                left.value + fraction * (right.value - left.value)
            }
            Self::Constant { value } => *value,
            Self::Ramp { rate } => rate * t,
            Self::Sine {
                amplitude,
                angular_frequency,
                phase,
            } => amplitude * (angular_frequency * t + phase).sin(),
            Self::Custom(hook) => hook(t),
        }
    }

    /// Evaluate the first time derivative at `t`.
    ///
    /// Tables return the enclosing segment slope and zero past the clamped
    /// ends; custom hooks use a central difference with step
    /// `sqrt(eps) * max(1, |t|)`.
    ///
    /// # Panics
    ///
    /// Panics for a non-finite query time, or an empty direct-built table.
    #[must_use]
    pub fn evaluate_rate(&self, t: f64) -> f64 {
        assert!(t.is_finite(), "history time must be finite");
        match self {
            Self::Table { samples } => {
                assert!(!samples.is_empty(), "time table needs at least one knot");
                if t <= samples[0].time || t >= samples[samples.len() - 1].time {
                    return 0.0;
                }
                let upper = samples.partition_point(|sample| sample.time <= t);
                let (left, right) = (samples[upper - 1], samples[upper]);
                let span = right.time - left.time;
                if span <= 0.0 {
                    return 0.0;
                }
                (right.value - left.value) / span
            }
            Self::Constant { .. } => 0.0,
            Self::Ramp { rate } => *rate,
            Self::Sine {
                amplitude,
                angular_frequency,
                phase,
            } => amplitude * angular_frequency * (angular_frequency * t + phase).cos(),
            Self::Custom(hook) => {
                let step = f64::EPSILON.sqrt() * (1.0 + t.abs());
                (hook(t + step) - hook(t - step)) / (2.0 * step)
            }
        }
    }

    /// Evaluate the second time derivative at `t`.
    ///
    /// Tables, constants, and ramps return zero; custom hooks use a second
    /// central difference with the same step as
    /// [`evaluate_rate`](Self::evaluate_rate).
    ///
    /// # Panics
    ///
    /// Panics for a non-finite query time, or an empty direct-built table.
    #[must_use]
    pub fn evaluate_acceleration(&self, t: f64) -> f64 {
        assert!(t.is_finite(), "history time must be finite");
        match self {
            Self::Table { samples } => {
                assert!(!samples.is_empty(), "time table needs at least one knot");
                0.0
            }
            Self::Constant { .. } | Self::Ramp { .. } => 0.0,
            Self::Sine {
                amplitude,
                angular_frequency,
                phase,
            } => {
                -amplitude
                    * angular_frequency
                    * angular_frequency
                    * (angular_frequency * t + phase).sin()
            }
            Self::Custom(hook) => {
                let step = f64::EPSILON.sqrt() * (1.0 + t.abs());
                (hook(t + step) - 2.0 * hook(t) + hook(t - step)) / (step * step)
            }
        }
    }
}

/// Time-history drive applied to a node set.
///
/// Nodes carry their own reference positions (parallel to `nodes`, captured
/// at the history datum), so displacement histories pin absolute positions
/// without needing mesh-wide lookup. The drive direction is normalized at
/// construction; transverse components always survive from live state.
#[derive(Debug, Clone)]
pub struct ImposedMotion {
    nodes: Vec<usize>,
    reference: Vec<[f64; 3]>,
    direction: [f64; 3],
    kind: ImposedKind,
    history: TimeHistory,
}

impl ImposedMotion {
    /// Create a drive over nodes with per-node references.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::joints::{ImposedKind, ImposedMotion, TimeHistory};
    ///
    /// let motion = ImposedMotion::new(
    ///     vec![0, 1],
    ///     vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ///     [0.0, 1.0, 0.0],
    ///     ImposedKind::Displacement,
    ///     TimeHistory::Constant { value: 0.5 },
    /// );
    /// assert_eq!(motion.nodes(), &[0, 1]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics for an empty node set, a node/reference length mismatch, a
    /// zero-length or non-finite direction, or non-finite references.
    #[must_use]
    pub fn new(
        nodes: Vec<usize>,
        reference: Vec<[f64; 3]>,
        direction: [f64; 3],
        kind: ImposedKind,
        history: TimeHistory,
    ) -> Self {
        assert!(!nodes.is_empty(), "imposed node set must be nonempty");
        assert_eq!(
            nodes.len(),
            reference.len(),
            "nodes and references must match"
        );
        let norm = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        assert!(
            norm.is_finite() && norm > 0.0,
            "drive direction must be finite and nonzero"
        );
        for position in &reference {
            assert!(
                position[0].is_finite() && position[1].is_finite() && position[2].is_finite(),
                "reference positions must be finite"
            );
        }
        Self {
            nodes,
            reference,
            direction: [
                direction[0] / norm,
                direction[1] / norm,
                direction[2] / norm,
            ],
            kind,
            history,
        }
    }

    /// Driven node ids.
    #[must_use]
    pub fn nodes(&self) -> &[usize] {
        &self.nodes
    }

    /// Imposed quantity along the drive direction.
    #[must_use]
    pub fn kind(&self) -> ImposedKind {
        self.kind
    }

    /// Scalar time history behind the drive.
    #[must_use]
    pub fn history(&self) -> &TimeHistory {
        &self.history
    }

    /// Unit drive direction.
    #[must_use]
    pub fn direction(&self) -> [f64; 3] {
        self.direction
    }

    /// Prescribed displacement at `t` for displacement drives.
    ///
    /// Returns `None` unless the kind is [`ImposedKind::Displacement`].
    #[must_use]
    pub fn prescribed_displacement(&self, t: f64) -> Option<f64> {
        match self.kind {
            ImposedKind::Displacement => Some(self.history.evaluate(t)),
            ImposedKind::Velocity | ImposedKind::Acceleration => None,
        }
    }

    /// Prescribed velocity at `t` for displacement or velocity drives.
    ///
    /// Displacement drives differentiate the history; velocity drives read
    /// it directly; acceleration drives return `None`.
    #[must_use]
    pub fn prescribed_velocity(&self, t: f64) -> Option<f64> {
        match self.kind {
            ImposedKind::Displacement => Some(self.history.evaluate_rate(t)),
            ImposedKind::Velocity => Some(self.history.evaluate(t)),
            ImposedKind::Acceleration => None,
        }
    }

    /// Prescribed acceleration at `t` for displacement or acceleration drives.
    ///
    /// Displacement drives take the second derivative; acceleration drives
    /// read the history directly; velocity drives return `None`.
    #[must_use]
    pub fn prescribed_acceleration(&self, t: f64) -> Option<f64> {
        match self.kind {
            ImposedKind::Displacement => Some(self.history.evaluate_acceleration(t)),
            ImposedKind::Velocity => None,
            ImposedKind::Acceleration => Some(self.history.evaluate(t)),
        }
    }

    /// Apply the drive for the stored kind at time `t`.
    ///
    /// Routes to [`apply_displacement`](Self::apply_displacement),
    /// [`apply_velocity`](Self::apply_velocity), or
    /// [`apply_acceleration`](Self::apply_acceleration). Call once per
    /// explicit cycle: displacement and velocity after the kinematic update,
    /// acceleration right after the force reset so `m a` joins assembly.
    ///
    /// # Panics
    ///
    /// Panics when a node id exceeds the state, or for a non-finite time.
    pub fn apply(&self, state: &mut ExplicitState, t: f64) {
        match self.kind {
            ImposedKind::Displacement => self.apply_displacement(state, t),
            ImposedKind::Velocity => self.apply_velocity(state, t),
            ImposedKind::Acceleration => self.apply_acceleration(state, t),
        }
    }

    /// Pin parallel position and velocity from the history.
    ///
    /// Sets `x = X + transverse(x - X) + h(t) dir` and
    /// `v = transverse(v) + h'(t) dir` per node, preserving transverse
    /// motion from the live state. The history is read as a displacement
    /// regardless of the stored kind.
    ///
    /// # Panics
    ///
    /// Panics when a node id exceeds the state, or for a non-finite time.
    pub fn apply_displacement(&self, state: &mut ExplicitState, t: f64) {
        let displacement = self.history.evaluate(t);
        let velocity = self.history.evaluate_rate(t);
        for (index, &node) in self.nodes.iter().enumerate() {
            assert!(
                node < state.node_count(),
                "node {node} outside explicit state"
            );
            let reference = self.reference[index];
            let position = [
                state.positions[3 * node],
                state.positions[3 * node + 1],
                state.positions[3 * node + 2],
            ];
            let drift = [
                position[0] - reference[0],
                position[1] - reference[1],
                position[2] - reference[2],
            ];
            let parallel = drift[0] * self.direction[0]
                + drift[1] * self.direction[1]
                + drift[2] * self.direction[2];
            let pinned = [
                reference[0] + drift[0] - parallel * self.direction[0]
                    + displacement * self.direction[0],
                reference[1] + drift[1] - parallel * self.direction[1]
                    + displacement * self.direction[1],
                reference[2] + drift[2] - parallel * self.direction[2]
                    + displacement * self.direction[2],
            ];
            state.positions[3 * node..3 * node + 3].copy_from_slice(&pinned);
            let current = [
                state.velocities[3 * node],
                state.velocities[3 * node + 1],
                state.velocities[3 * node + 2],
            ];
            let current_parallel = current[0] * self.direction[0]
                + current[1] * self.direction[1]
                + current[2] * self.direction[2];
            let pinned_velocity = [
                current[0] - current_parallel * self.direction[0] + velocity * self.direction[0],
                current[1] - current_parallel * self.direction[1] + velocity * self.direction[1],
                current[2] - current_parallel * self.direction[2] + velocity * self.direction[2],
            ];
            state.velocities[3 * node..3 * node + 3].copy_from_slice(&pinned_velocity);
        }
    }

    /// Pin parallel velocity from the history.
    ///
    /// Sets `v = transverse(v) + h(t) dir` per node; positions are left for
    /// the integrator. The history is read as a velocity regardless of the
    /// stored kind.
    ///
    /// # Panics
    ///
    /// Panics when a node id exceeds the state, or for a non-finite time.
    pub fn apply_velocity(&self, state: &mut ExplicitState, t: f64) {
        let velocity = self.history.evaluate(t);
        for &node in &self.nodes {
            assert!(
                node < state.node_count(),
                "node {node} outside explicit state"
            );
            let current = [
                state.velocities[3 * node],
                state.velocities[3 * node + 1],
                state.velocities[3 * node + 2],
            ];
            let parallel = current[0] * self.direction[0]
                + current[1] * self.direction[1]
                + current[2] * self.direction[2];
            let pinned = [
                current[0] - parallel * self.direction[0] + velocity * self.direction[0],
                current[1] - parallel * self.direction[1] + velocity * self.direction[1],
                current[2] - parallel * self.direction[2] + velocity * self.direction[2],
            ];
            state.velocities[3 * node..3 * node + 3].copy_from_slice(&pinned);
        }
    }

    /// Scatter `m a` into the force scratch from the history.
    ///
    /// Adds `mass * h(t) dir` per node on top of assembled forces, so the
    /// integrator accelerates the parallel component at `h(t)`. The history
    /// is read as an acceleration regardless of the stored kind.
    ///
    /// # Panics
    ///
    /// Panics when a node id exceeds the state, or for a non-finite time.
    pub fn apply_acceleration(&self, state: &mut ExplicitState, t: f64) {
        let acceleration = self.history.evaluate(t);
        for &node in &self.nodes {
            assert!(
                node < state.node_count(),
                "node {node} outside explicit state"
            );
            let push = state.masses[node] * acceleration;
            state.forces[3 * node] += push * self.direction[0];
            state.forces[3 * node + 1] += push * self.direction[1];
            state.forces[3 * node + 2] += push * self.direction[2];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(time: f64, value: f64) -> TimeSample {
        TimeSample { time, value }
    }

    #[test]
    fn table_interpolates_and_clamps() {
        let history = TimeHistory::table(vec![sample(0.0, 0.0), sample(2.0, 4.0)]);
        assert!((history.evaluate(1.0) - 2.0).abs() < 1.0e-15);
        assert!((history.evaluate(0.5) - 1.0).abs() < 1.0e-15);
        // Clamp past the ends.
        assert!((history.evaluate(-1.0) - 0.0).abs() < 1.0e-15);
        assert!((history.evaluate(5.0) - 4.0).abs() < 1.0e-15);
        // Exact knots.
        assert!((history.evaluate(0.0) - 0.0).abs() < 1.0e-15);
        assert!((history.evaluate(2.0) - 4.0).abs() < 1.0e-15);
    }

    #[test]
    fn table_sorts_knots() {
        let history = TimeHistory::table(vec![sample(2.0, 4.0), sample(0.0, 0.0)]);
        assert!((history.evaluate(1.0) - 2.0).abs() < 1.0e-15);
    }

    #[test]
    fn table_rate_is_segment_slope() {
        let history =
            TimeHistory::table(vec![sample(0.0, 1.0), sample(1.0, 3.0), sample(3.0, 4.0)]);
        assert!((history.evaluate_rate(0.5) - 2.0).abs() < 1.0e-15);
        assert!((history.evaluate_rate(2.0) - 0.5).abs() < 1.0e-15);
        // Clamped ends carry zero rate.
        assert!((history.evaluate_rate(-2.0)).abs() < 1.0e-15);
        assert!((history.evaluate_rate(9.0)).abs() < 1.0e-15);
        assert!((history.evaluate_acceleration(1.5)).abs() < 1.0e-15);
    }

    #[test]
    fn sine_matches_closed_form() {
        let history = TimeHistory::Sine {
            amplitude: 2.0,
            angular_frequency: 3.0,
            phase: 0.5,
        };
        for t in [0.0_f64, 0.25, 1.0, 2.5] {
            let expected = 2.0 * (3.0 * t + 0.5).sin();
            let expected_rate = 6.0 * (3.0 * t + 0.5).cos();
            let expected_accel = -18.0 * (3.0 * t + 0.5).sin();
            assert!(
                (history.evaluate(t) - expected).abs() < 1.0e-15,
                "sine value at t = {t}"
            );
            assert!(
                (history.evaluate_rate(t) - expected_rate).abs() < 1.0e-14,
                "sine rate at t = {t}"
            );
            assert!(
                (history.evaluate_acceleration(t) - expected_accel).abs() < 1.0e-13,
                "sine acceleration at t = {t}"
            );
        }
    }

    #[test]
    fn custom_hook_tracks_profile() {
        let history = TimeHistory::Custom(Arc::new(|t| t * t));
        assert!((history.evaluate(3.0) - 9.0).abs() < 1.0e-15);
        // Central differences recover 2t and 2 within truncation.
        assert!((history.evaluate_rate(3.0) - 6.0).abs() < 1.0e-7);
        assert!((history.evaluate_acceleration(3.0) - 2.0).abs() < 1.0e-4);
    }

    #[test]
    fn displacement_preserves_transverse() {
        let motion = ImposedMotion::new(
            vec![0],
            vec![[0.0, 0.0, 0.0]],
            [0.0, 1.0, 0.0],
            ImposedKind::Displacement,
            TimeHistory::table(vec![sample(0.0, 0.0), sample(1.0, 2.0)]),
        );
        assert_eq!(motion.prescribed_displacement(0.5), Some(1.0));
        assert_eq!(motion.prescribed_velocity(0.5), Some(2.0));
        assert_eq!(motion.prescribed_acceleration(0.5), Some(0.0));
        let mut state = ExplicitState::new(1);
        // Transverse drift plus transverse velocity must survive.
        state.positions = vec![0.3, 0.9, -0.2];
        state.velocities = vec![1.5, 7.0, -2.5];
        motion.apply(&mut state, 0.5);
        assert!((state.positions[0] - 0.3).abs() < 1.0e-15);
        assert!((state.positions[1] - 1.0).abs() < 1.0e-15);
        assert!((state.positions[2] + 0.2).abs() < 1.0e-15);
        assert!((state.velocities[0] - 1.5).abs() < 1.0e-15);
        assert!((state.velocities[1] - 2.0).abs() < 1.0e-15);
        assert!((state.velocities[2] + 2.5).abs() < 1.0e-15);
    }

    #[test]
    fn velocity_tracks_profile() {
        let motion = ImposedMotion::new(
            vec![0, 1],
            vec![[0.0; 3], [1.0, 0.0, 0.0]],
            [1.0, 0.0, 0.0],
            ImposedKind::Velocity,
            TimeHistory::Sine {
                amplitude: 1.0,
                angular_frequency: 2.0,
                phase: 0.0,
            },
        );
        assert_eq!(motion.prescribed_displacement(0.5), None);
        assert!(motion.prescribed_velocity(0.5).is_some());
        assert_eq!(motion.prescribed_acceleration(0.5), None);
        let mut state = ExplicitState::new(2);
        state.velocities = vec![9.0, 1.0, 2.0, 8.0, 3.0, 4.0];
        let expected = (2.0 * 0.5f64).sin();
        motion.apply(&mut state, 0.5);
        // Parallel components pinned, transverse untouched, positions free.
        assert!((state.velocities[0] - expected).abs() < 1.0e-15);
        assert!((state.velocities[3] - expected).abs() < 1.0e-15);
        assert!((state.velocities[1] - 1.0).abs() < 1.0e-15);
        assert!((state.velocities[5] - 4.0).abs() < 1.0e-15);
        assert!(state.positions.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn acceleration_adds_mass_scaled_force() {
        let motion = ImposedMotion::new(
            vec![0, 1],
            vec![[0.0; 3], [0.0; 3]],
            [0.0, 0.0, 1.0],
            ImposedKind::Acceleration,
            TimeHistory::Constant { value: 3.0 },
        );
        assert_eq!(motion.prescribed_acceleration(1.0), Some(3.0));
        let mut state = ExplicitState::new(2);
        state.masses = vec![2.0, 5.0];
        state.forces = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        motion.apply(&mut state, 1.0);
        assert!((state.forces[0] - 1.0).abs() < 1.0e-15);
        assert!((state.forces[2] - 7.0).abs() < 1.0e-15);
        assert!((state.forces[5] - 17.0).abs() < 1.0e-15);
    }
}
