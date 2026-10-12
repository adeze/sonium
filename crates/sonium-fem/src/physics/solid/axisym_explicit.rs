//! Explicit axisymmetric QUAD ring elements with C1 materials.
//!
//! Four-node bilinear QUAD in the meridian `(r, z)` plane, revolved about
//! the `z` axis into a ring element (Theory Manual "Element Library",
//! p. 80: 4-node quadrilaterals for axisymmetrical analysis;
//! `/ANALY N2D3D = 1`, Reference Guide pp. 100-101; `/QUAD`, Reference
//! Guide pp. 183-184). The explicit path uses centroid one-point
//! integration (Theory Manual "Reduced Integration Method", p. 85: the
//! default internal-force method) with Flanagan-Belytschko viscous
//! hourglass control ("Flanagan-Belytschko Formulation", pp. 86-89) and
//! Von Neumann-Richtmyer bulk viscosity ("Shock Waves", pp. 91-92).
//! Stress comes from any frozen C1 [`StressUpdateModel`]; rate-dependent
//! materials evaluate at the estimated step rate. The stable step is the
//! Courant bound `L / c` (Theory Manual p. 35).
//!
//! Node order is counterclockwise from `(xi, eta) = (-1, -1)`; nodal
//! vectors interleave `[v_r, v_z]` per node, matching
//! [`axisymmetric`](super::axisymmetric). Axisymmetric strain
//! `[e_rr, e_zz, e_th, g_rz]` maps onto C1 Voigt as `[rr, zz, th, rz,
//! 0, 0]` through [`c1_of_axisym_strain`](super::axisymmetric::c1_of_axisym_strain):
//! the hoop direction plays the role of C1 `zz`, and the two
//! out-of-meridian shears vanish by symmetry. Ring integrals carry the
//! `2 pi r` weight; the centroid radius is clamped by `axis_epsilon`
//! wherever it divides, so elements with an edge on the axis stay
//! finite when the axis nodes carry no radial motion.
//!
//! # Solid-shell verdict (lane 3E, DEFER)
//!
//! Do not build an axisymmetric solid-shell on the current seeds. OpenCourant
//! solid-shells are a 3D-only family (HA8/HSEPH/PA6/SHELL16: solid
//! elements plus a penalty-enforced constant through-thickness normal
//! stress; Theory Manual pp. 164-165) with no axisymmetric member to be
//! faithful to; the in-repo seeds are implicit-only (`shell::revolution`
//! assembles stiffness, no explicit force path or C1 materials;
//! `nonlinear_solid::{eas, fbar}` are implicit locking cures; the crash
//! thick-shell hook is the `NoThickShell` zero stub). A solid-shell ring
//! would need new through-thickness assumed-strain plus explicit
//! stabilization theory, which is out of lane scope. The QUAD ring here
//! covers bulk axisymmetric response instead.
//!
//! # Examples
//!
//! ```rust
//! use sonium_fem::physics::solid::axisym_explicit::{
//!     AxisymExplicitParams, QuadRingGeom, QuadRingState, quad_ring_forces,
//! };
//! use sonium_fem::physics::solid::plasticity::{HardeningLaw, J2Material, PlasticState};
//!
//! let geom = QuadRingGeom::new([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]).unwrap();
//! let material = J2Material::new(200.0e9, 0.3, 1.0e12, HardeningLaw::Perfect);
//! let mut state = QuadRingState::new(PlasticState::new_3d());
//! let params = AxisymExplicitParams::new(7800.0, 4500.0);
//! // Axial stretch at 1/s: hourglass-free, expansion, no viscosity.
//! let velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0];
//! let out = quad_ring_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params).unwrap();
//! assert!(!out.yielding);
//! assert_eq!(out.bulk_q, 0.0);
//! ```

// Rust guideline compliant 2026-02-21

use super::axisymmetric::{axisym_of_c1_stress, c1_of_axisym_strain};
use super::plasticity::update::StressUpdateModel;
use sonium_compute::explicit_hourglass::{
    BULK_VISCOSITY_LINEAR_DEFAULT, BULK_VISCOSITY_QUADRATIC_DEFAULT, bulk_viscosity_q,
};

/// Flanagan-Belytschko hourglass base mode of the QUAD.
///
/// The bilinear `xi * eta` pattern, `+1`/`-1` at the four nodes in lane
/// node order. It is orthogonal to the constant field by construction
/// (`sum h = 0`); [`QuadRingGeom`] subtracts the linear-field content
/// with the centroid gradients to form the Gamma vector (Theory Manual
/// Eqs 326-331, pp. 86-89). On any parallelogram the correction
/// vanishes and Gamma equals this mode bit-for-bit.
pub const QUAD_HOURGLASS_BASE: [f64; 4] = [1.0, -1.0, 1.0, -1.0];

/// Default clamp radius for divisions by `r`.
///
/// Matches the `1.0e-12` axis epsilon of the implicit axisymmetric
/// forms: far below any modeled radius, far above denormal range, so
/// hoop terms on axis-adjacent elements stay finite without shifting
/// physical results.
pub const AXIS_EPSILON_DEFAULT: f64 = 1.0e-12;

/// Natural coordinates `(xi, eta)` of the four QUAD nodes.
const NATURAL_COORDS: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

/// Error of the axisymmetric explicit path.
#[derive(Debug, Clone)]
pub struct AxisymExplicitError {
    /// Human-readable failure description.
    pub message: String,
}

impl AxisymExplicitError {
    /// Builds an error from a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for AxisymExplicitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "axisymmetric explicit error: {}", self.message)
    }
}

impl std::error::Error for AxisymExplicitError {}

/// Explicit-step parameters of the QUAD ring path.
#[derive(Debug, Clone, Copy)]
pub struct AxisymExplicitParams {
    /// Mass density in kg/m^3.
    pub density: f64,
    /// Sound speed in m/s.
    pub sound_speed: f64,
    /// Viscous hourglass scale, dimensionless.
    ///
    /// Multiplies `rho c L^2` (N s/m) into the Flanagan-Belytschko
    /// viscous coefficient. The default 0.05 sits in the literature
    /// 0.01-0.10 band: large enough to damp spurious modes within a
    /// few steps, small enough to leave physical waves alone.
    pub hg_scale: f64,
    /// Quadratic shock-viscosity coefficient (default 1.10).
    pub bulk_c0: f64,
    /// Linear shock-viscosity coefficient (default 0.05).
    pub bulk_c1: f64,
}

impl AxisymExplicitParams {
    /// Builds parameters from density and sound speed with defaults.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::axisym_explicit::AxisymExplicitParams;
    ///
    /// let params = AxisymExplicitParams::new(7800.0, 4500.0);
    /// assert!((params.hg_scale - 0.05).abs() < 1e-15);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics for negative density, sound speed, or viscosity
    /// coefficients.
    #[must_use]
    pub fn new(density: f64, sound_speed: f64) -> Self {
        assert!(
            density.is_finite() && density >= 0.0,
            "density must be finite non-negative"
        );
        assert!(
            sound_speed.is_finite() && sound_speed >= 0.0,
            "sound speed must be finite non-negative"
        );
        Self {
            density,
            sound_speed,
            hg_scale: 0.05,
            bulk_c0: BULK_VISCOSITY_QUADRATIC_DEFAULT,
            bulk_c1: BULK_VISCOSITY_LINEAR_DEFAULT,
        }
    }

    /// Overrides the viscous hourglass scale (dimensionless).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::axisym_explicit::AxisymExplicitParams;
    ///
    /// let params = AxisymExplicitParams::new(7800.0, 4500.0).with_hg_scale(0.0);
    /// assert_eq!(params.hg_scale, 0.0);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics for a negative or non-finite scale.
    #[must_use]
    pub fn with_hg_scale(mut self, hg_scale: f64) -> Self {
        assert!(
            hg_scale.is_finite() && hg_scale >= 0.0,
            "hourglass scale must be finite non-negative"
        );
        self.hg_scale = hg_scale;
        self
    }
}

/// Centroid-integrated QUAD ring geometry and hourglass vector.
#[derive(Debug, Clone)]
pub struct QuadRingGeom {
    /// Node `(r, z)` coordinates in lane node order, in m.
    pub coords: [[f64; 2]; 4],
    /// Meridional area in m^2 (`4 det J` at the centroid).
    pub area: f64,
    /// Centroid radius in m (nodal mean, unclamped).
    pub radius_bar: f64,
    /// Ring volume in m^3 (`2 pi r_bar A`).
    pub volume: f64,
    /// Centroid physical gradients `[dN/dr, dN/dz]` per node, in 1/m.
    pub grad: [[f64; 2]; 4],
    /// Flanagan-Belytschko Gamma vector, dimensionless.
    pub gamma: [f64; 4],
    /// Characteristic length `A / d_max` in m.
    pub char_length: f64,
    /// Clamp radius for divisions by `r`, in m.
    pub axis_epsilon: f64,
}

impl QuadRingGeom {
    /// Builds ring geometry from four meridian node coordinates.
    ///
    /// Nodes run counterclockwise from `(xi, eta) = (-1, -1)`. The
    /// volume `2 pi r_bar A` is exact for parallelograms (constant
    /// Jacobian, centroid radius equal to the area-weighted mean) and
    /// the one-point approximation otherwise. The characteristic
    /// length divides the area by the longest diagonal: the longest
    /// straight wave path across a convex QUAD, hence a conservative
    /// Courant length.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::axisym_explicit::QuadRingGeom;
    ///
    /// let geom = QuadRingGeom::new([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]).unwrap();
    /// assert!((geom.area - 1.0).abs() < 1e-14);
    /// assert!((geom.radius_bar - 1.5).abs() < 1e-14);
    /// assert!((geom.volume - 3.0 * std::f64::consts::PI).abs() < 1e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`AxisymExplicitError`] for non-finite coordinates or a
    /// degenerate (non-positive centroid Jacobian) element.
    pub fn new(coords: [[f64; 2]; 4]) -> Result<Self, AxisymExplicitError> {
        if !coords.iter().flatten().all(|v| v.is_finite()) {
            return Err(AxisymExplicitError::new(
                "quad ring coordinates must be finite",
            ));
        }
        // Centroid Jacobian J[j, m] = sum_I x[j, I] dN_I/d(xi_m):
        // dN/dxi = xi_I / 4, dN/deta = eta_I / 4 at the centroid.
        let mut jac = [[0.0; 2]; 2];
        for (node, nat) in NATURAL_COORDS.iter().enumerate() {
            for j in 0..2 {
                jac[j][0] += coords[node][j] * nat[0] / 4.0;
                jac[j][1] += coords[node][j] * nat[1] / 4.0;
            }
        }
        let det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
        if !det.is_finite() || det <= 0.0 {
            return Err(AxisymExplicitError::new(
                "quad ring needs a positive centroid Jacobian",
            ));
        }
        let inv = [
            [jac[1][1] / det, -jac[0][1] / det],
            [-jac[1][0] / det, jac[0][0] / det],
        ];
        let mut grad = [[0.0; 2]; 4];
        for (node, nat) in NATURAL_COORDS.iter().enumerate() {
            let dxi = nat[0] / 4.0;
            let deta = nat[1] / 4.0;
            grad[node][0] = inv[0][0] * dxi + inv[1][0] * deta;
            grad[node][1] = inv[0][1] * dxi + inv[1][1] * deta;
        }
        // Gamma_I = h_I - (h . r) b_r,I - (h . z) b_z,I.
        let mut hr = 0.0;
        let mut hz = 0.0;
        for node in 0..4 {
            hr += QUAD_HOURGLASS_BASE[node] * coords[node][0];
            hz += QUAD_HOURGLASS_BASE[node] * coords[node][1];
        }
        let mut gamma = [0.0; 4];
        for node in 0..4 {
            gamma[node] = QUAD_HOURGLASS_BASE[node] - hr * grad[node][0] - hz * grad[node][1];
        }
        let area = 4.0 * det;
        let radius_bar = (coords[0][0] + coords[1][0] + coords[2][0] + coords[3][0]) / 4.0;
        let diag13 = diagonal(coords[0], coords[2]);
        let diag24 = diagonal(coords[1], coords[3]);
        Ok(Self {
            coords,
            area,
            radius_bar,
            volume: 2.0 * std::f64::consts::PI * radius_bar * area,
            grad,
            gamma,
            char_length: area / diag13.max(diag24),
            axis_epsilon: AXIS_EPSILON_DEFAULT,
        })
    }

    /// Overrides the clamp radius for divisions by `r`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::axisym_explicit::QuadRingGeom;
    ///
    /// let geom =
    ///     QuadRingGeom::new([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]])
    ///         .unwrap()
    ///         .with_axis_epsilon(1.0e-9);
    /// assert_eq!(geom.axis_epsilon, 1.0e-9);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics for a non-positive or non-finite epsilon.
    #[must_use]
    pub fn with_axis_epsilon(mut self, axis_epsilon: f64) -> Self {
        assert!(
            axis_epsilon.is_finite() && axis_epsilon > 0.0,
            "axis epsilon must be finite positive"
        );
        self.axis_epsilon = axis_epsilon;
        self
    }

    /// Clamped centroid radius in m for hoop divisions.
    #[must_use]
    pub fn clamped_radius(&self) -> f64 {
        self.radius_bar.abs().max(self.axis_epsilon)
    }

    /// Lumped nodal masses in kg, one quarter of the total each.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::axisym_explicit::QuadRingGeom;
    ///
    /// let geom = QuadRingGeom::new([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]).unwrap();
    /// let masses = geom.lumped_masses(7800.0);
    /// assert!((masses.iter().sum::<f64>() - 7800.0 * geom.volume).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn lumped_masses(&self, density: f64) -> [f64; 4] {
        let nodal = density * self.volume / 4.0;
        [nodal; 4]
    }

    /// Critical explicit step `L / c` in s.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::axisym_explicit::QuadRingGeom;
    ///
    /// let geom = QuadRingGeom::new([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]).unwrap();
    /// assert!((geom.critical_step(4500.0) - geom.char_length / 4500.0).abs() < 1e-18);
    /// ```
    #[must_use]
    pub fn critical_step(&self, sound_speed: f64) -> f64 {
        self.char_length / sound_speed
    }

    /// Viscous hourglass coefficient in N s/m at the parameters.
    #[must_use]
    pub fn hg_coefficient(&self, params: &AxisymExplicitParams) -> f64 {
        params.hg_scale * params.density * params.sound_speed * self.char_length.powi(2)
    }
}

/// History state of one QUAD ring element (single centroid point).
#[derive(Debug, Clone)]
pub struct QuadRingState<S> {
    /// Material state at the centroid point.
    pub point: S,
    /// Accumulated hourglass energy in J.
    pub hg_energy: f64,
}

impl<S> QuadRingState<S> {
    /// Builds ring state from a centroid material-point state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::axisym_explicit::QuadRingState;
    /// use sonium_fem::physics::solid::plasticity::PlasticState;
    ///
    /// let state = QuadRingState::new(PlasticState::new_3d());
    /// assert_eq!(state.hg_energy, 0.0);
    /// ```
    #[must_use]
    pub fn new(point: S) -> Self {
        Self {
            point,
            hg_energy: 0.0,
        }
    }
}

/// Internal-force evaluation of one QUAD ring element.
#[derive(Debug, Clone)]
pub struct QuadRingOut {
    /// Nodal internal forces in N, interleaved `[f_r, f_z]` per node.
    pub forces: [f64; 8],
    /// Viscous hourglass power in W (non-negative).
    pub hg_power: f64,
    /// Shock viscosity pressure in Pa.
    pub bulk_q: f64,
    /// Equivalent plastic strain increment of the step.
    pub dep: f64,
    /// True when the centroid point yielded.
    pub yielding: bool,
}

/// Explicit internal forces of one QUAD ring element.
///
/// Advances the centroid C1 material by the uniform-strain increment
/// from the nodal velocities over `dt`, commits it, and returns `W B^T
/// (sigma + q I)` plus viscous hourglass forces, with `W = 2 pi r_bar
/// A`. Rate-dependent materials evaluate at the estimated step rate.
/// Hourglass energy accumulates into the state with explicit Euler
/// (`P dt`).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::axisym_explicit::{
///     AxisymExplicitParams, QuadRingGeom, QuadRingState, quad_ring_forces,
/// };
/// use sonium_fem::physics::solid::plasticity::{HardeningLaw, J2Material, PlasticState};
///
/// let geom = QuadRingGeom::new([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]).unwrap();
/// let material = J2Material::new(200.0e9, 0.3, 1.0e12, HardeningLaw::Perfect);
/// let mut state = QuadRingState::new(PlasticState::new_3d());
/// let params = AxisymExplicitParams::new(7800.0, 4500.0);
/// let velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0];
/// let out = quad_ring_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params).unwrap();
/// // Uniaxial strain at 1e-3: sigma_zz = (lambda + 2 mu) e.
/// let lame = 200.0e9 * 0.3 / (1.3 * 0.4);
/// let mu = 200.0e9 / 2.6;
/// assert!((state.point.stress[1] - (lame + 2.0 * mu) * 1.0e-3).abs() < 1.0);
/// ```
///
/// # Errors
///
/// Returns [`AxisymExplicitError`] for non-positive `dt` or non-finite input.
pub fn quad_ring_forces<M>(
    geom: &QuadRingGeom,
    state: &mut QuadRingState<M::State>,
    material: &M,
    velocities: &[f64; 8],
    dt: f64,
    params: &AxisymExplicitParams,
) -> Result<QuadRingOut, AxisymExplicitError>
where
    M: StressUpdateModel<StrainIncrement = [f64; 6]>,
{
    if dt.is_nan() || dt <= 0.0 {
        return Err(AxisymExplicitError::new(
            "explicit step dt must be positive",
        ));
    }
    if !velocities.iter().all(|v| v.is_finite()) {
        return Err(AxisymExplicitError::new(
            "quad ring velocities must be finite",
        ));
    }
    let radius = geom.clamped_radius();
    let mut axisym = [0.0; 4];
    let mut mean_dr = 0.0;
    for (node, grad) in geom.grad.iter().enumerate() {
        let dr = velocities[node * 2] * dt;
        let dz = velocities[node * 2 + 1] * dt;
        axisym[0] += grad[0] * dr;
        axisym[1] += grad[1] * dz;
        axisym[3] += grad[1] * dr + grad[0] * dz;
        mean_dr += dr;
    }
    // Centroid shape values are all 1/4: hoop strain is the mean
    // radial displacement over the clamped radius.
    axisym[2] = mean_dr / (4.0 * radius);
    let d_eps = c1_of_axisym_strain(&axisym);
    let vol_rate = (axisym[0] + axisym[1] + axisym[2]) / dt;
    let bulk_q = bulk_viscosity_q(
        params.density,
        params.sound_speed,
        geom.char_length,
        params.bulk_c0,
        params.bulk_c1,
        vol_rate,
    );
    let update = if material.is_rate_dependent() {
        material.update_with_rate(&state.point, &d_eps, dt)
    } else {
        material.update(&state.point, &d_eps)
    };
    material.commit(&mut state.point, &update);
    let hoop = axisym_of_c1_stress(&update.stress);
    let srr = hoop[0] + bulk_q;
    let szz = hoop[1] + bulk_q;
    let sth = hoop[2] + bulk_q;
    let srz = hoop[3];
    let mut forces = [0.0; 8];
    for (node, grad) in geom.grad.iter().enumerate() {
        forces[node * 2] = (srr * grad[0] + srz * grad[1] + sth / (4.0 * radius)) * geom.volume;
        forces[node * 2 + 1] = (srz * grad[0] + szz * grad[1]) * geom.volume;
    }
    // Viscous hourglass forces oppose the modal rates; with the
    // internal-force sign convention they align with the velocities
    // and report non-negative dissipated power.
    let coeff = geom.hg_coefficient(params);
    let mut modal_r = 0.0;
    let mut modal_z = 0.0;
    for node in 0..4 {
        modal_r += geom.gamma[node] * velocities[node * 2];
        modal_z += geom.gamma[node] * velocities[node * 2 + 1];
    }
    let mut hg_power = 0.0;
    for node in 0..4 {
        let fr = coeff * geom.gamma[node] * modal_r;
        let fz = coeff * geom.gamma[node] * modal_z;
        forces[node * 2] += fr;
        forces[node * 2 + 1] += fz;
        hg_power += fr * velocities[node * 2] + fz * velocities[node * 2 + 1];
    }
    state.hg_energy += hg_power * dt;
    Ok(QuadRingOut {
        forces,
        hg_power,
        bulk_q,
        dep: update.delta_equiv_plastic_strain,
        yielding: update.yielding,
    })
}

/// Euclidean distance of two meridian points.
fn diagonal(p0: [f64; 2], p1: [f64; 2]) -> f64 {
    let dr = p1[0] - p0[0];
    let dz = p1[1] - p0[1];
    (dr * dr + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::solid::plasticity::update::StressUpdate;
    use crate::physics::solid::plasticity::{HardeningLaw, J2Material, PlasticState};

    fn elastic_j2() -> J2Material {
        J2Material::new(200.0e9, 0.3, 1.0e12, HardeningLaw::Perfect)
    }

    fn unit_patch() -> QuadRingGeom {
        QuadRingGeom::new([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]).unwrap()
    }

    fn steel_params() -> AxisymExplicitParams {
        AxisymExplicitParams::new(7800.0, 4500.0)
    }

    #[test]
    fn patch_geometry_matches_closed_form() {
        let geom = unit_patch();
        assert!((geom.area - 1.0).abs() < 1e-14);
        assert!((geom.radius_bar - 1.5).abs() < 1e-14);
        assert!((geom.volume - 3.0 * std::f64::consts::PI).abs() < 1e-12);
        // Centroid gradients of the unit square are +/- 1/2.
        let expected_r = [-0.5, 0.5, 0.5, -0.5];
        let expected_z = [-0.5, -0.5, 0.5, 0.5];
        for node in 0..4 {
            assert!((geom.grad[node][0] - expected_r[node]).abs() < 1e-15);
            assert!((geom.grad[node][1] - expected_z[node]).abs() < 1e-15);
            assert_eq!(geom.gamma[node], QUAD_HOURGLASS_BASE[node]);
        }
        assert!((geom.char_length - 1.0 / 2.0_f64.sqrt()).abs() < 1e-15);
        let masses = geom.lumped_masses(7800.0);
        for mass in masses {
            assert!((mass - 7800.0 * geom.volume / 4.0).abs() < 1e-12);
        }
    }

    #[test]
    fn uniaxial_strain_gives_closed_form_stress() {
        let geom = unit_patch();
        let material = elastic_j2();
        let mut state = QuadRingState::new(PlasticState::new_3d());
        let params = steel_params();
        // Axial stretch at 1/s over dt = 1e-3: hourglass-free field.
        let velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let out =
            quad_ring_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params).unwrap();
        assert!(!out.yielding);
        assert_eq!(out.dep, 0.0);
        // Linear fields carry no hourglass content: power is ~zero.
        assert!(out.hg_power.abs() < 1e-9);
        // Expansion carries no shock viscosity.
        assert_eq!(out.bulk_q, 0.0);
        let lame = 200.0e9 * 0.3 / (1.3 * 0.4);
        let mu = 200.0e9 / 2.6;
        // Committed state stores internal order; normals 0..3 match C1.
        assert!((state.point.stress[0] - lame * 1.0e-3).abs() < 1.0);
        assert!((state.point.stress[1] - (lame + 2.0 * mu) * 1.0e-3).abs() < 1.0);
        assert!((state.point.stress[2] - lame * 1.0e-3).abs() < 1.0);
        // Axial forces self-equilibrate; radial forces balance the
        // hoop resultant sth * W / r_bar.
        let mut fr = 0.0;
        let mut fz = 0.0;
        for node in 0..4 {
            fr += out.forces[node * 2];
            fz += out.forces[node * 2 + 1];
        }
        let scale: f64 = out.forces.iter().map(|f| f.abs()).sum::<f64>();
        assert!(fz.abs() / scale < 1e-12);
        let expected_fr = lame * 1.0e-3 * geom.volume / geom.clamped_radius();
        assert!((fr - expected_fr).abs() / expected_fr < 1e-12);
    }

    #[test]
    fn shear_patch_gives_closed_form_traction() {
        let geom = unit_patch();
        let material = elastic_j2();
        let mut state = QuadRingState::new(PlasticState::new_3d());
        let params = steel_params();
        // Axial velocity linear in r: uniform engineering shear rate.
        let velocities = [0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0];
        let out =
            quad_ring_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params).unwrap();
        assert!(!out.yielding);
        // C1 shear rz (index 3) lands on internal index 5.
        let mu = 200.0e9 / 2.6;
        assert!((state.point.stress[5] - mu * 1.0e-3).abs() < 1.0);
        assert!(out.forces.iter().all(|f| f.is_finite()));
    }

    #[test]
    fn hourglass_mode_dissipates() {
        let geom = unit_patch();
        let material = elastic_j2();
        let mut state = QuadRingState::new(PlasticState::new_3d());
        let params = steel_params();
        // Pure hourglass velocity pattern in r.
        let mut velocities = [0.0; 8];
        for node in 0..4 {
            velocities[node * 2] = QUAD_HOURGLASS_BASE[node];
        }
        let out =
            quad_ring_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params).unwrap();
        // Uniform strain of a pure hourglass pattern vanishes: only
        // the viscous forces resist, with positive power.
        assert_eq!(out.dep, 0.0);
        assert!(out.hg_power > 0.0);
        assert!(state.hg_energy > 0.0);
        // Modal content (gamma . v) = 4 on the square: P = c * 16.
        let expected = geom.hg_coefficient(&params) * 16.0;
        assert!((out.hg_power - expected).abs() / expected < 1e-12);
    }

    #[test]
    fn rigid_translation_gives_zero_forces() {
        let geom = unit_patch();
        let material = elastic_j2();
        let mut state = QuadRingState::new(PlasticState::new_3d());
        let params = steel_params();
        // Pure axial translation: uniform radial motion would strain
        // the hoop (circumference change), which is physical.
        let velocities = [0.0, -3.0, 0.0, -3.0, 0.0, -3.0, 0.0, -3.0];
        let out =
            quad_ring_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params).unwrap();
        assert!(out.forces.iter().all(|f| f.abs() < 1e-6));
        assert!(out.hg_power.abs() < 1e-9);
        assert_eq!(state.hg_energy, 0.0);
    }

    #[test]
    fn on_axis_element_stays_finite() {
        // Bar cross-section element with an edge on the axis: axis
        // nodes carry no radial velocity (symmetry condition).
        let geom = QuadRingGeom::new([[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]).unwrap();
        let material = elastic_j2();
        let mut state = QuadRingState::new(PlasticState::new_3d());
        let params = steel_params();
        let velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let out =
            quad_ring_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params).unwrap();
        assert!(!out.yielding);
        assert!(out.forces.iter().all(|f| f.is_finite()));
        assert!((geom.volume - 2.0 * std::f64::consts::PI * 0.25 * 0.5).abs() < 1e-14);
    }

    #[test]
    fn plastic_step_yields_and_accumulates() {
        let geom = unit_patch();
        let material = J2Material::mild_steel();
        let mut state = QuadRingState::new(PlasticState::new_3d());
        let params = steel_params();
        let velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        // One percent axial strain overwhelms the 250 MPa yield.
        let out =
            quad_ring_forces(&geom, &mut state, &material, &velocities, 1.0e-2, &params).unwrap();
        assert!(out.yielding);
        assert!(out.dep > 0.0);
        assert!(state.point.equiv_plastic_strain > 0.0);
        let first = state.point.equiv_plastic_strain;
        let again =
            quad_ring_forces(&geom, &mut state, &material, &velocities, 1.0e-2, &params).unwrap();
        assert!(again.yielding);
        assert!(state.point.equiv_plastic_strain > first);
    }

    /// Rate-sensitive probe material: the quasi-static update stays
    /// silent while the rate update reports the step rate.
    #[derive(Debug, Clone, Copy)]
    struct RateProbe;

    /// Trivial state for the rate probe.
    #[derive(Debug, Clone, Copy)]
    struct RateProbeState;

    impl StressUpdateModel for RateProbe {
        type State = RateProbeState;
        type StrainIncrement = [f64; 6];

        fn update(&self, _state: &Self::State, _d_eps: &Self::StrainIncrement) -> StressUpdate {
            StressUpdate {
                stress: [0.0; 6],
                tangent: [0.0; 36],
                delta_equiv_plastic_strain: 0.0,
                yielding: false,
            }
        }

        fn commit(&self, _state: &mut Self::State, _update: &StressUpdate) {}

        fn update_with_rate(
            &self,
            _state: &Self::State,
            _d_eps: &Self::StrainIncrement,
            dt: f64,
        ) -> StressUpdate {
            StressUpdate {
                stress: [1.0 / dt, 0.0, 0.0, 0.0, 0.0, 0.0],
                tangent: [0.0; 36],
                delta_equiv_plastic_strain: 0.0,
                yielding: false,
            }
        }

        fn is_rate_dependent(&self) -> bool {
            true
        }
    }

    #[test]
    fn rate_dependent_material_uses_step_rate() {
        let geom = unit_patch();
        let mut state = QuadRingState::new(RateProbeState);
        let params = steel_params();
        let velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let out =
            quad_ring_forces(&geom, &mut state, &RateProbe, &velocities, 2.0e-3, &params).unwrap();
        // Radial nodal force carries srr = 1/dt through B^T; the
        // linear driver has no radial modal content, so node 2 sees
        // no hourglass force.
        let srr = 1.0 / 2.0e-3;
        let expected = srr * geom.grad[2][0] * geom.volume;
        assert!((out.forces[4] - expected).abs() / expected.abs() < 1e-12);
    }

    #[test]
    fn bulk_viscosity_pressurizes_compression_only() {
        let geom = unit_patch();
        let material = elastic_j2();
        let params = steel_params();
        // Uniform inward radial velocity: volumetric compression.
        let squeeze = [-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0];
        let mut state = QuadRingState::new(PlasticState::new_3d());
        let out =
            quad_ring_forces(&geom, &mut state, &material, &squeeze, 1.0e-3, &params).unwrap();
        assert!(out.bulk_q > 0.0);
        // Uniform outward velocity: expansion carries no viscosity.
        let expand = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let mut state = QuadRingState::new(PlasticState::new_3d());
        let out = quad_ring_forces(&geom, &mut state, &material, &expand, 1.0e-3, &params).unwrap();
        assert_eq!(out.bulk_q, 0.0);
    }

    #[test]
    fn degenerate_and_bad_inputs_error() {
        assert!(QuadRingGeom::new([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]).is_err());
        assert!(QuadRingGeom::new([[1.0, 0.0], [2.0, 0.0], [2.0, f64::NAN], [1.0, 1.0]]).is_err());
        // Clockwise winding inverts the centroid Jacobian.
        assert!(QuadRingGeom::new([[1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 0.0]]).is_err());
        let geom = unit_patch();
        let material = elastic_j2();
        let mut state = QuadRingState::new(PlasticState::new_3d());
        let params = steel_params();
        let velocities = [0.0; 8];
        assert!(quad_ring_forces(&geom, &mut state, &material, &velocities, 0.0, &params).is_err());
        assert!(
            quad_ring_forces(&geom, &mut state, &material, &velocities, -1.0e-3, &params).is_err()
        );
        let bad = [0.0, 0.0, 0.0, f64::INFINITY, 0.0, 0.0, 0.0, 0.0];
        assert!(quad_ring_forces(&geom, &mut state, &material, &bad, 1.0e-3, &params).is_err());
    }
}
