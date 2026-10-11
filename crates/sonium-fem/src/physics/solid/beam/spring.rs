//! One-dimensional crash springs and rivets for explicit dynamics.
//!
//! Two-node [`Spring1D`] elements (RESSORT, Reference Guide `/SPRING`
//! element with `/PROP/TYPE4`, `/PROP/TYPE8`, `/PROP/TYPE25`, or
//! `/PROP/TYPE32`) evaluate axial force from elongation and elongation
//! rate along the current element axis, plus an optional nonlinear
//! force-deflection curve on the frozen [`Funct1D`] type, viscous
//! damping, pretension, and tensile/compressive rupture. [`Rivet`]
//! models a spotweld-style connector between coincident nodes as a stiff
//! 6-DOF link with a force-moment interaction rupture criterion; the
//! legacy `/PROP/TYPE5 (RIVET)` card is obsolete since v13.0
//! (Reference Guide p. 2652), so rivets follow the spring-element
//! formulation below.
//!
//! Deformation sign follows the Theory Manual "Deformation Sign
//! Convention" (p. 187, Eqs 649-650): elongation is positive when the
//! displacement of node 2 minus node 1 along the axis is positive, and
//! the axial force is positive in tension. Force evaluation follows
//! "Translational Forces" (p. 187, Eq 651), the linear and nonlinear
//! spring laws follow "Linear Spring" (p. 179) and "Nonlinear Elastic
//! Spring" (p. 180), damping follows "Nonlinear Dashpot" (p. 183), and
//! the stable step follows "Time Step" for springs (p. 185).
//!
//! Unlike the quasi-static [`AxialSpring`](super::super::joints::zero_d::AxialSpring)
//! and [`Spring3D`](super::super::joints::zero_d::Spring3D), which assemble
//! stiffness matrices, this module evaluates explicit internal forces,
//! tangents, and power directly from nodal positions and velocities.

// Rust guideline compliant 2026-02-21

use super::super::plasticity::funct::Funct1D;

/// Rupture limits of a 1-D spring in tension and compression.
///
/// A spring breaks irreversibly once the axial force or the elongation
/// leaves the `[compression, tension]` box. Limits use the p. 187 sign
/// convention: tension positive, compression negative.
#[derive(Debug, Clone, Copy)]
pub struct SpringRupture {
    /// Tensile force limit in N (positive).
    pub max_tension: f64,
    /// Compressive force limit in N (positive magnitude).
    pub max_compression: f64,
    /// Tensile elongation limit in m (positive).
    pub max_elongation: f64,
    /// Compressive shortening limit in m (positive magnitude).
    pub max_shortening: f64,
}

impl SpringRupture {
    /// Builds rupture limits from positive magnitudes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::beam::SpringRupture;
    ///
    /// let rupture = SpringRupture::new(5000.0, 8000.0, 0.02, 0.01);
    /// assert!(rupture.tripped(6000.0, 0.0));
    /// assert!(!rupture.tripped(1000.0, 0.0));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when any limit is not positive finite.
    #[must_use]
    pub fn new(
        max_tension: f64,
        max_compression: f64,
        max_elongation: f64,
        max_shortening: f64,
    ) -> Self {
        for (name, value) in [
            ("max_tension", max_tension),
            ("max_compression", max_compression),
            ("max_elongation", max_elongation),
            ("max_shortening", max_shortening),
        ] {
            assert!(
                value.is_finite() && value > 0.0,
                "{name} must be positive finite"
            );
        }
        Self {
            max_tension,
            max_compression,
            max_elongation,
            max_shortening,
        }
    }

    /// Reports whether force (N) or elongation (m) trips rupture.
    ///
    /// Force is positive in tension, elongation positive in extension.
    #[must_use]
    pub fn tripped(&self, force: f64, elongation: f64) -> bool {
        force > self.max_tension
            || force < -self.max_compression
            || elongation > self.max_elongation
            || elongation < -self.max_shortening
    }
}

/// Two-node 1-D crash spring (RESSORT).
///
/// Axial force combines a linear stiffness, an optional nonlinear
/// [`Funct1D`] force-deflection curve, viscous damping, and a constant
/// pretension force:
/// `F = k d + f_curve(d) + c v + F_pre`, positive in tension. The
/// curve holds its endpoint values outside its range (clamped
/// [`Funct1D`] evaluation). Pretension follows `/PROP/TYPE32 (SPR_PRE)`
/// (Reference Guide pp. 1563-1567): a positive `pretension` adds a
/// tensile force present at zero elongation.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::beam::{Spring1D, Spring1DState};
///
/// let spring = Spring1D::linear(1000.0, 10.0, 0.5);
/// let state = Spring1DState::new();
/// let out = spring.nodal_forces(
///     &state,
///     [0.0, 0.0, 0.0],
///     [1.1, 0.0, 0.0],
///     [0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0],
///     1.0,
/// );
/// assert!((out.axial_force - 100.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone)]
pub struct Spring1D {
    /// Linear stiffness in N/m.
    pub stiffness: f64,
    /// Viscous damping in N s/m.
    pub damping: f64,
    /// Lumped mass in kg, split evenly to both nodes.
    pub mass: f64,
    /// Optional nonlinear force-deflection curve, N over m.
    pub curve: Option<Funct1D>,
    /// Constant pretension force in N, positive in tension.
    pub pretension: f64,
    /// Optional rupture limits (`None` never breaks).
    pub rupture: Option<SpringRupture>,
}

impl Spring1D {
    /// Builds a linear spring from stiffness, damping, and mass.
    ///
    /// # Panics
    ///
    /// Panics for negative stiffness, damping, or mass.
    #[must_use]
    pub fn linear(stiffness: f64, damping: f64, mass: f64) -> Self {
        assert!(
            stiffness.is_finite() && stiffness >= 0.0,
            "stiffness must be finite non-negative"
        );
        assert!(
            damping.is_finite() && damping >= 0.0,
            "damping must be finite non-negative"
        );
        assert!(
            mass.is_finite() && mass >= 0.0,
            "mass must be finite non-negative"
        );
        Self {
            stiffness,
            damping,
            mass,
            curve: None,
            pretension: 0.0,
            rupture: None,
        }
    }

    /// Attaches a nonlinear force-deflection curve in N over m.
    #[must_use]
    pub fn with_curve(mut self, curve: Funct1D) -> Self {
        self.curve = Some(curve);
        self
    }

    /// Attaches a constant pretension force in N, positive in tension.
    #[must_use]
    pub fn with_pretension(mut self, pretension: f64) -> Self {
        self.pretension = pretension;
        self
    }

    /// Attaches rupture limits.
    #[must_use]
    pub fn with_rupture(mut self, rupture: SpringRupture) -> Self {
        self.rupture = Some(rupture);
        self
    }

    /// Axial force in N at elongation (m) and rate (m/s).
    ///
    /// Returns zero once `state` is broken. Positive is tension.
    #[must_use]
    pub fn axial_force(&self, state: &Spring1DState, elongation: f64, rate: f64) -> f64 {
        if state.broken {
            return 0.0;
        }
        let nonlinear = self.curve.as_ref().map_or(0.0, |c| c.eval(elongation));
        self.stiffness * elongation + nonlinear + self.damping * rate + self.pretension
    }

    /// Axial tangent stiffness in N/m at an elongation.
    ///
    /// Adds the curve segment slope to the linear stiffness; zero when
    /// broken. Damping does not enter the static tangent.
    #[must_use]
    pub fn axial_tangent(&self, state: &Spring1DState, elongation: f64) -> f64 {
        if state.broken {
            return 0.0;
        }
        self.stiffness + self.curve.as_ref().map_or(0.0, |c| c.slope(elongation))
    }

    /// Nodal forces, power, and energy rate for current kinematics.
    ///
    /// Takes global positions and velocities of nodes 1 and 2 plus the
    /// stress-free length `rest_length`. Elongation is the current
    /// length minus rest length; the axis runs from node 1 to node 2.
    /// Node 1 carries `-F axis`, node 2 `+F axis` with `F` positive in
    /// tension, matching the p. 187 sign convention.
    ///
    /// # Panics
    ///
    /// Panics when the nodes coincide (undefined axis) or the rest
    /// length is negative.
    #[must_use]
    pub fn nodal_forces(
        &self,
        state: &Spring1DState,
        pos1: [f64; 3],
        pos2: [f64; 3],
        vel1: [f64; 3],
        vel2: [f64; 3],
        rest_length: f64,
    ) -> SpringForceOut {
        assert!(rest_length >= 0.0, "rest length must be non-negative");
        let delta = [pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]];
        let length = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        assert!(length > 0.0, "spring nodes must not coincide");
        let axis = [delta[0] / length, delta[1] / length, delta[2] / length];
        let elongation = length - rest_length;
        let rate = (vel2[0] - vel1[0]) * axis[0]
            + (vel2[1] - vel1[1]) * axis[1]
            + (vel2[2] - vel1[2]) * axis[2];
        let force = self.axial_force(state, elongation, rate);
        SpringForceOut {
            force1: [-force * axis[0], -force * axis[1], -force * axis[2]],
            force2: [force * axis[0], force * axis[1], force * axis[2]],
            axial_force: force,
            elongation,
            rate,
            power: force * rate,
            tangent: self.axial_tangent(state, elongation),
        }
    }

    /// Advances history: accumulates energy and checks rupture.
    ///
    /// Adds `0.5 (F_old + F) d(d)` trapezoidal work to the stored
    /// energy... callers pass the previous axial force; on the first
    /// step pass the current force for an explicit Euler increment.
    /// Rupture trips irreversibly from the current force/elongation.
    pub fn commit(
        &self,
        state: &mut Spring1DState,
        force_old: f64,
        force: f64,
        elongation_old: f64,
        elongation: f64,
    ) {
        if state.broken {
            return;
        }
        state.energy += 0.5 * (force_old + force) * (elongation - elongation_old);
        state.peak_force = state.peak_force.max(force.abs());
        if let Some(rupture) = &self.rupture
            && rupture.tripped(force, elongation)
        {
            state.broken = true;
        }
    }
}

/// History state of one [`Spring1D`] element.
#[derive(Debug, Clone)]
pub struct Spring1DState {
    /// True once rupture tripped; forces stay zero afterwards.
    pub broken: bool,
    /// Accumulated internal energy in J.
    pub energy: f64,
    /// Peak absolute axial force seen in N.
    pub peak_force: f64,
}

impl Spring1DState {
    /// Creates a virgin unbroken state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            broken: false,
            energy: 0.0,
            peak_force: 0.0,
        }
    }
}

impl Default for Spring1DState {
    fn default() -> Self {
        Self::new()
    }
}

/// Nodal force evaluation of one [`Spring1D`].
#[derive(Debug, Clone, Copy)]
pub struct SpringForceOut {
    /// Force on node 1 in N (global).
    pub force1: [f64; 3],
    /// Force on node 2 in N (global).
    pub force2: [f64; 3],
    /// Axial force in N, positive in tension.
    pub axial_force: f64,
    /// Elongation in m, positive in extension.
    pub elongation: f64,
    /// Elongation rate in m/s.
    pub rate: f64,
    /// Mechanical power `F v` in W.
    pub power: f64,
    /// Axial tangent stiffness in N/m.
    pub tangent: f64,
}

/// Critical explicit step of a 2-node spring from nodal masses.
///
/// Returns `2 / sqrt(k (1/m1 + 1/m2))`, the central-difference limit
/// `2 / omega_max` of the two-mass oscillator (Theory Manual "Time
/// Step" for springs, p. 185). Zero stiffness returns infinity; a zero
/// mass on either node returns zero (no stable explicit step).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::beam::spring_critical_step;
///
/// let dt = spring_critical_step(0.5, 0.5, 1000.0);
/// assert!((dt - 2.0 / (1000.0f64 * (2.0 + 2.0)).sqrt()).abs() < 1e-15);
/// ```
#[must_use]
pub fn spring_critical_step(mass1: f64, mass2: f64, stiffness: f64) -> f64 {
    assert!(
        mass1.is_finite() && mass1 >= 0.0,
        "nodal masses must be finite non-negative"
    );
    assert!(
        mass2.is_finite() && mass2 >= 0.0,
        "nodal masses must be finite non-negative"
    );
    assert!(
        stiffness.is_finite() && stiffness >= 0.0,
        "stiffness must be finite non-negative"
    );
    if stiffness == 0.0 {
        return f64::INFINITY;
    }
    if mass1 == 0.0 || mass2 == 0.0 {
        return 0.0;
    }
    2.0 / (stiffness * (1.0 / mass1 + 1.0 / mass2)).sqrt()
}

/// Spotweld-style rivet connector between coincident nodes.
///
/// Models the weld as a stiff diagonal 6-DOF link: translational
/// penalty stiffness plus rotational penalty, with optional viscous
/// damping, evaluated in the global frame. Forces follow the p. 187
/// sign convention on relative displacement `u2 - u1`. Rupture uses a
/// force-moment interaction: `(Fa/Fa_max)^2 + (Fs/Fs_max)^2 +
/// (M/M_max)^2 >= 1` breaks the nugget irreversibly, where `Fa` is the
/// global-z force, `Fs` the in-plane shear resultant, and `M` the
/// moment resultant. Drivers with skewed welds pre-rotate the relative
/// kinematics into the weld frame.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::beam::{Rivet, RivetState};
///
/// let rivet = Rivet::new(1.0e9, 1.0e6, 5000.0, 2000.0, 50.0);
/// let mut state = RivetState::new();
/// let out = rivet.forces(&state, [1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0; 6]);
/// assert!((out.force[0] - 1000.0).abs() < 1e-6);
/// rivet.commit(&mut state, &out);
/// assert!(!state.broken);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Rivet {
    /// Translational penalty stiffness in N/m.
    pub penalty: f64,
    /// Rotational penalty stiffness in N m/rad.
    pub penalty_rotation: f64,
    /// Translational viscous damping in N s/m.
    pub damping: f64,
    /// Axial (z) force capacity in N.
    pub max_axial: f64,
    /// In-plane shear capacity in N.
    pub max_shear: f64,
    /// Moment capacity in N m.
    pub max_moment: f64,
}

impl Rivet {
    /// Builds a rivet from stiffnesses and capacities.
    ///
    /// # Panics
    ///
    /// Panics for non-positive stiffnesses or capacities.
    #[must_use]
    pub fn new(
        penalty: f64,
        penalty_rotation: f64,
        max_axial: f64,
        max_shear: f64,
        max_moment: f64,
    ) -> Self {
        assert!(
            penalty.is_finite() && penalty > 0.0,
            "penalty must be positive finite"
        );
        assert!(
            penalty_rotation.is_finite() && penalty_rotation > 0.0,
            "rotational penalty must be positive finite"
        );
        for (name, value) in [
            ("max_axial", max_axial),
            ("max_shear", max_shear),
            ("max_moment", max_moment),
        ] {
            assert!(
                value.is_finite() && value > 0.0,
                "{name} must be positive finite"
            );
        }
        Self {
            penalty,
            penalty_rotation,
            damping: 0.0,
            max_axial,
            max_shear,
            max_moment,
        }
    }

    /// Attaches translational viscous damping in N s/m.
    #[must_use]
    pub fn with_damping(mut self, damping: f64) -> Self {
        assert!(
            damping.is_finite() && damping >= 0.0,
            "damping must be finite non-negative"
        );
        self.damping = damping;
        self
    }

    /// Rivet forces for relative displacement and velocity.
    ///
    /// Takes `rel = u2 - u1` (m, m, m, rad, rad, rad) and its rate;
    /// returns the force on node 2 (node 1 carries the opposite).
    /// Broken rivets return zero forces with utilization frozen above
    /// one... utilization still evaluates from the elastic predictor so
    /// post-processing sees the overload.
    #[must_use]
    pub fn forces(&self, state: &RivetState, rel: [f64; 6], rate: [f64; 6]) -> RivetOut {
        let mut force = [0.0; 6];
        for i in 0..3 {
            force[i] = self.penalty * rel[i] + self.damping * rate[i];
        }
        for i in 3..6 {
            force[i] = self.penalty_rotation * rel[i];
        }
        let shear = (force[0] * force[0] + force[1] * force[1]).sqrt();
        let moment = (force[3] * force[3] + force[4] * force[4] + force[5] * force[5]).sqrt();
        let utilization = (force[2] / self.max_axial).powi(2)
            + (shear / self.max_shear).powi(2)
            + (moment / self.max_moment).powi(2);
        if state.broken {
            force = [0.0; 6];
        }
        RivetOut {
            force,
            utilization,
            power: force[0] * rate[0]
                + force[1] * rate[1]
                + force[2] * rate[2]
                + force[3] * rate[3]
                + force[4] * rate[4]
                + force[5] * rate[5],
        }
    }

    /// Advances history: breaks the nugget past unit utilization.
    pub fn commit(&self, state: &mut RivetState, out: &RivetOut) {
        if out.utilization >= 1.0 {
            state.broken = true;
        }
        state.utilization = out.utilization;
    }
}

/// History state of one [`Rivet`].
#[derive(Debug, Clone, Copy)]
pub struct RivetState {
    /// True once the interaction criterion tripped.
    pub broken: bool,
    /// Last interaction utilization (rupture at one).
    pub utilization: f64,
}

impl RivetState {
    /// Creates a virgin unbroken state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            broken: false,
            utilization: 0.0,
        }
    }
}

impl Default for RivetState {
    fn default() -> Self {
        Self::new()
    }
}

/// Force evaluation of one [`Rivet`].
#[derive(Debug, Clone, Copy)]
pub struct RivetOut {
    /// Force-moment on node 2 in N and N m.
    pub force: [f64; 6],
    /// Interaction utilization (rupture at one).
    pub utilization: f64,
    /// Mechanical power in W.
    pub power: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_spring_matches_closed_form() {
        let spring = Spring1D::linear(2000.0, 5.0, 1.0);
        let state = Spring1DState::new();
        let out = spring.nodal_forces(
            &state,
            [0.0, 0.0, 0.0],
            [1.05, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            1.0,
        );
        // F = k d + c v = 2000 * 0.05 + 5 * 2 = 110 N tension.
        assert!((out.axial_force - 110.0).abs() < 1e-9);
        assert!((out.elongation - 0.05).abs() < 1e-12);
        assert!((out.rate - 2.0).abs() < 1e-12);
        assert!((out.force2[0] - 110.0).abs() < 1e-9);
        assert!((out.force1[0] + 110.0).abs() < 1e-9);
        assert!((out.power - 220.0).abs() < 1e-9);
        assert!((out.tangent - 2000.0).abs() < 1e-12);
    }

    #[test]
    fn spring_axis_follows_diagonal() {
        let spring = Spring1D::linear(1000.0, 0.0, 0.0);
        let state = Spring1DState::new();
        let out = spring.nodal_forces(
            &state,
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            4.0,
        );
        // Length 5, elongation 1, axis (3/5, 4/5, 0), F = 1000 N.
        assert!((out.axial_force - 1000.0).abs() < 1e-9);
        assert!((out.force2[0] - 600.0).abs() < 1e-9);
        assert!((out.force2[1] - 800.0).abs() < 1e-9);
        assert!(out.force2[2].abs() < 1e-12);
    }

    #[test]
    fn nonlinear_curve_adds_to_linear() {
        let curve = Funct1D::new(vec![0.0, 0.1], vec![0.0, 50.0]).unwrap();
        let spring = Spring1D::linear(1000.0, 0.0, 0.0).with_curve(curve);
        let state = Spring1DState::new();
        // F = 1000 * 0.05 + 25 = 75 N; tangent 1000 + 500.
        assert!((spring.axial_force(&state, 0.05, 0.0) - 75.0).abs() < 1e-9);
        assert!((spring.axial_tangent(&state, 0.05) - 1500.0).abs() < 1e-9);
        // Clamped ends hold the last ordinate.
        assert!((spring.axial_force(&state, 1.0, 0.0) - 1050.0).abs() < 1e-9);
    }

    #[test]
    fn pretension_shifts_zero_crossing() {
        let spring = Spring1D::linear(1000.0, 0.0, 0.0).with_pretension(250.0);
        let state = Spring1DState::new();
        assert!((spring.axial_force(&state, 0.0, 0.0) - 250.0).abs() < 1e-12);
        assert!(spring.axial_force(&state, -0.25, 0.0).abs() < 1e-12);
    }

    #[test]
    fn rupture_trips_and_unloads() {
        let rupture = SpringRupture::new(100.0, 100.0, 0.2, 0.2);
        let spring = Spring1D::linear(1000.0, 0.0, 0.0).with_rupture(rupture);
        let mut state = Spring1DState::new();
        spring.commit(&mut state, 0.0, 90.0, 0.0, 0.09);
        assert!(!state.broken);
        assert!(state.energy > 0.0);
        spring.commit(&mut state, 90.0, 150.0, 0.09, 0.15);
        assert!(state.broken);
        assert_eq!(spring.axial_force(&state, 0.15, 0.0), 0.0);
        assert_eq!(spring.axial_tangent(&state, 0.15), 0.0);
    }

    #[test]
    fn compression_rupture_is_symmetric() {
        let rupture = SpringRupture::new(100.0, 60.0, 0.2, 0.05);
        assert!(rupture.tripped(-70.0, 0.0));
        assert!(rupture.tripped(0.0, -0.06));
        assert!(!rupture.tripped(-50.0, -0.04));
    }

    #[test]
    fn critical_step_matches_oscillator() {
        let dt = spring_critical_step(2.0, 2.0, 1000.0);
        assert!((dt - 2.0 / (1000.0f64).sqrt()).abs() < 1e-15);
        assert_eq!(spring_critical_step(1.0, 1.0, 0.0), f64::INFINITY);
        assert_eq!(spring_critical_step(0.0, 1.0, 1000.0), 0.0);
    }

    #[test]
    fn rivet_shear_and_moment_combine() {
        let rivet = Rivet::new(1.0e6, 1.0e3, 5000.0, 2000.0, 50.0);
        let state = RivetState::new();
        // Pure shear 1000 N -> utilization (1000/2000)^2 = 0.25.
        let out = rivet.forces(&state, [1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0; 6]);
        assert!((out.force[0] - 1000.0).abs() < 1e-6);
        assert!((out.utilization - 0.25).abs() < 1e-12);
        // Combined shear + moment: 0.25 + (30/50)^2 = 0.61.
        let out = rivet.forces(&state, [1.0e-3, 0.0, 0.0, 0.03, 0.0, 0.0], [0.0; 6]);
        assert!((out.utilization - 0.61).abs() < 1e-12);
    }

    #[test]
    fn rivet_breaks_past_unit_utilization() {
        let rivet = Rivet::new(1.0e6, 1.0e3, 5000.0, 2000.0, 50.0);
        let mut state = RivetState::new();
        let out = rivet.forces(&state, [0.0, 0.0, 6.0e-3, 0.0, 0.0, 0.0], [0.0; 6]);
        // Axial 6000 N > 5000 N capacity -> utilization 1.44.
        assert!(out.utilization > 1.0);
        rivet.commit(&mut state, &out);
        assert!(state.broken);
        let after = rivet.forces(&state, [0.0, 0.0, 6.0e-3, 0.0, 0.0, 0.0], [0.0; 6]);
        assert!(after.force.iter().all(|f| *f == 0.0));
        assert!(after.utilization > 1.0);
    }

    #[test]
    fn spring_energy_matches_trapezoid() {
        let spring = Spring1D::linear(1000.0, 0.0, 0.0);
        let mut state = Spring1DState::new();
        // Ramp 0 -> 0.1 m in one commit: W = 0.5 * 100 * 0.1 = 5 J.
        spring.commit(&mut state, 0.0, 100.0, 0.0, 0.1);
        assert!((state.energy - 5.0).abs() < 1e-12);
        assert!((state.peak_force - 100.0).abs() < 1e-12);
    }
}
