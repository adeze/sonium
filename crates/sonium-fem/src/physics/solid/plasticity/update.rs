//! Shared stress-update contract for plasticity models.
//!
//! This module defines contract C1 of the FEM elasticity program: a single
//! [`StressUpdateModel`] trait behind which every plasticity formulation
//! (small-strain J2, finite-strain J2, and later fiber/layer models) exposes
//! a pure stress update plus an explicit state commit. Separating the trial
//! update from the commit lets global Newton iterations probe candidate
//! states without mutating history variables.
//!
//! # Conventions
//!
//! - SI units throughout: stress in Pa, tangent in Pa.
//! - Voigt ordering is `[xx, yy, zz, xy, yz, zx]`. Shear strains are
//!   engineering shears (`gamma_xy = 2 eps_xy`). Shear stresses are the
//!   plain tensor components.
//! - `update` is pure: it never mutates the state. `commit` applies a
//!   previously computed [`StressUpdate`] to the state.
//!
//! The legacy small-strain functions in [`super::return_mapping`] keep their
//! own internal Voigt ordering (`[xx, yy, zz, yz, xz, xy]`) and their own
//! dynamically sized update type. The [`StressUpdateModel`] implementation
//! for [`J2Material`](super::J2Material) converts between the two orderings,
//! so existing callers are unaffected. Conversion helpers are public so that
//! fiber and layer lanes can reuse them.

// Rust guideline compliant 2026-02-21

use super::material::{J2Material, PlasticState};
use super::return_mapping::return_mapping_3d;

/// Maps a C1 Voigt index to the legacy internal Voigt index.
///
/// C1 order is `[xx, yy, zz, xy, yz, zx]` while the legacy return mapping
/// uses `[xx, yy, zz, yz, xz, xy]`.
const INTERNAL_OF_C1: [usize; 6] = [0, 1, 2, 5, 3, 4];

/// Maps a legacy internal Voigt index to the C1 Voigt index.
const C1_OF_INTERNAL: [usize; 6] = [0, 1, 2, 4, 5, 3];

/// Converts a C1-ordered update into the frozen contract shape.
///
/// Contract C1 (FEM elasticity program, Phase A): every stress update model
/// returns Cauchy stress in Voigt order `[xx, yy, zz, xy, yz, zx]` in Pa
/// together with the row-major 6x6 consistent tangent. For finite-strain
/// models the stress is the Cauchy stress; the tangent equals the elastic
/// stiffness whenever the step is elastic.
#[derive(Debug, Clone, Copy)]
pub struct StressUpdate {
    /// Cauchy stress in Voigt order `[xx, yy, zz, xy, yz, zx]` (Pa).
    pub stress: [f64; 6],
    /// Consistent tangent, row-major 6x6 (Pa).
    pub tangent: [f64; 36],
    /// Equivalent plastic strain increment for this step.
    pub delta_equiv_plastic_strain: f64,
    /// True when the trial state violated the yield surface.
    pub yielding: bool,
}

/// Equivalent total-strain increment of a C1 Voigt increment.
///
/// Returns `sqrt(2/3 dev : dev)` with engineering shear handling, the
/// scalar numerator crash models divide by the step time to estimate the
/// strain rate. Pure helper: no state, no side effects.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::update::voigt_equivalent_increment_c1;
///
/// let uni = [0.002, -0.001, -0.001, 0.0, 0.0, 0.0];
/// let eq = voigt_equivalent_increment_c1(&uni);
/// assert!((eq - 0.002).abs() < 1e-12);
/// ```
pub fn voigt_equivalent_increment_c1(d_eps: &[f64; 6]) -> f64 {
    let mean = (d_eps[0] + d_eps[1] + d_eps[2]) / 3.0;
    let d0 = d_eps[0] - mean;
    let d1 = d_eps[1] - mean;
    let d2 = d_eps[2] - mean;
    // Tensor shear is half the engineering shear, and the contraction
    // carries 2x the squared tensor shears: 2 (g/2)^2 = g^2 / 2.
    let shear = 0.5 * (d_eps[3] * d_eps[3] + d_eps[4] * d_eps[4] + d_eps[5] * d_eps[5]);
    ((2.0 / 3.0) * (d0 * d0 + d1 * d1 + d2 * d2 + shear)).sqrt()
}

/// Updates stress for a strain increment and commits history.
///
/// Models implement a pure trial `update` plus an explicit `commit` so that
/// global equilibrium iterations can evaluate candidate states without
/// touching history variables. All quantities use SI units.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::{
///     J2Material, PlasticState, StressUpdateModel,
/// };
///
/// let material = J2Material::mild_steel();
/// let mut state = PlasticState::new_3d();
/// let d_eps = [1.0e-6, -0.3e-6, -0.3e-6, 0.0, 0.0, 0.0];
/// let update = material.update(&state, &d_eps);
/// material.commit(&mut state, &update);
/// ```
pub trait StressUpdateModel {
    /// History state at one material point.
    type State: Clone;
    /// Strain increment driving one update step.
    type StrainIncrement;
    /// Computes the trial stress update without mutating state.
    ///
    /// The returned stress uses C1 Voigt order `[xx, yy, zz, xy, yz, zx]`.
    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate;
    /// Applies a previously computed update to the history state.
    fn commit(&self, state: &mut Self::State, update: &StressUpdate);

    /// Trial update with the step time for rate-dependent models.
    ///
    /// Provided method (Phase B additive C1 extension): the default ignores
    /// `dt` and delegates to [`update`](Self::update), so every existing
    /// implementation (J2, finite-strain J2, fiber adapters, shell layers)
    /// keeps compiling and behaving unchanged. Rate-dependent crash models
    /// override this to evaluate the flow stress at the estimated strain
    /// rate `voigt_equivalent_increment_c1(d_eps) / dt`; `update` keeps the
    /// quasi-static response at the reference strain rate. Pure like
    /// `update`: must not mutate `state`. Callers pass `dt > 0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::plasticity::{
    ///     J2Material, PlasticState, StressUpdateModel,
    /// };
    ///
    /// let material = J2Material::mild_steel();
    /// let state = PlasticState::new_3d();
    /// let d_eps = [1.0e-6, -0.3e-6, -0.3e-6, 0.0, 0.0, 0.0];
    /// let slow = material.update(&state, &d_eps);
    /// let fast = material.update_with_rate(&state, &d_eps, 1.0e-6);
    /// assert_eq!(slow.stress, fast.stress);
    /// ```
    fn update_with_rate(
        &self,
        state: &Self::State,
        d_eps: &Self::StrainIncrement,
        dt: f64,
    ) -> StressUpdate {
        let _ = dt;
        self.update(state, d_eps)
    }

    /// Reports whether the model responds to strain rate.
    ///
    /// Provided method (Phase B additive C1 extension): defaults to `false`.
    /// Rate-dependent crash models return `true` so drivers know to call
    /// [`update_with_rate`](Self::update_with_rate) instead of `update`.
    fn is_rate_dependent(&self) -> bool {
        false
    }
}

/// Reorders a C1 Voigt 6-vector into legacy internal order.
///
/// Maps `[xx, yy, zz, xy, yz, zx]` to `[xx, yy, zz, yz, xz, xy]`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::update::c1_to_internal_6;
///
/// let c1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// assert_eq!(c1_to_internal_6(&c1), [1.0, 2.0, 3.0, 5.0, 6.0, 4.0]);
/// ```
pub fn c1_to_internal_6(c1: &[f64; 6]) -> [f64; 6] {
    std::array::from_fn(|i| c1[C1_OF_INTERNAL[i]])
}

/// Reorders a legacy internal Voigt 6-vector into C1 order.
///
/// Maps `[xx, yy, zz, yz, xz, xy]` to `[xx, yy, zz, xy, yz, zx]`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::update::internal_to_c1_6;
///
/// let internal = [1.0, 2.0, 3.0, 5.0, 6.0, 4.0];
/// assert_eq!(internal_to_c1_6(&internal), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
pub fn internal_to_c1_6(internal: &[f64; 6]) -> [f64; 6] {
    std::array::from_fn(|a| internal[INTERNAL_OF_C1[a]])
}

/// Reorders a C1 row-major 6x6 matrix into legacy internal order.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::update::c1_to_internal_36;
///
/// let mut c1 = [0.0; 36];
/// c1[3 * 6 + 3] = 7.0;
/// let internal = c1_to_internal_36(&c1);
/// assert_eq!(internal[5 * 6 + 5], 7.0);
/// ```
pub fn c1_to_internal_36(c1: &[f64; 36]) -> [f64; 36] {
    let mut out = [0.0; 36];
    for i in 0..6 {
        for j in 0..6 {
            out[i * 6 + j] = c1[C1_OF_INTERNAL[i] * 6 + C1_OF_INTERNAL[j]];
        }
    }
    out
}

/// Reorders a legacy internal row-major 6x6 matrix into C1 order.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::update::internal_to_c1_36;
///
/// let mut internal = [0.0; 36];
/// internal[5 * 6 + 5] = 7.0;
/// let c1 = internal_to_c1_36(&internal);
/// assert_eq!(c1[3 * 6 + 3], 7.0);
/// ```
pub fn internal_to_c1_36(internal: &[f64; 36]) -> [f64; 36] {
    let mut out = [0.0; 36];
    for a in 0..6 {
        for b in 0..6 {
            out[a * 6 + b] = internal[INTERNAL_OF_C1[a] * 6 + INTERNAL_OF_C1[b]];
        }
    }
    out
}

/// Von Mises stress of a C1-ordered tensor (not necessarily deviatoric).
///
/// Subtracts the hydrostatic part internally, so callers may pass a full
/// Cauchy stress in C1 Voigt order.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::update::von_mises_c1;
///
/// let uniaxial = [300.0e6, 0.0, 0.0, 0.0, 0.0, 0.0];
/// assert!((von_mises_c1(&uniaxial) - 300.0e6).abs() < 1.0);
/// ```
pub fn von_mises_c1(stress: &[f64; 6]) -> f64 {
    let p = (stress[0] + stress[1] + stress[2]) / 3.0;
    let s0 = stress[0] - p;
    let s1 = stress[1] - p;
    let s2 = stress[2] - p;
    let j2 = 0.5 * (s0 * s0 + s1 * s1 + s2 * s2)
        + stress[3] * stress[3]
        + stress[4] * stress[4]
        + stress[5] * stress[5];
    (3.0 * j2).sqrt()
}

impl StressUpdateModel for J2Material {
    type State = PlasticState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        assert!(
            state.stress.len() == 6 && state.back_stress.len() == 6,
            "J2Material C1 update needs a 3D PlasticState with 6 components"
        );
        let d_internal = c1_to_internal_6(d_eps);
        let legacy = return_mapping_3d(self, state, &d_internal);
        let stress_in: [f64; 6] = legacy.stress.as_slice().try_into().unwrap_or_else(|_| {
            panic!("return_mapping_3d must return 6 stress components");
        });
        let tangent_in: [f64; 36] = legacy.tangent.as_slice().try_into().unwrap_or_else(|_| {
            panic!("return_mapping_3d must return a 6x6 tangent");
        });
        StressUpdate {
            stress: internal_to_c1_6(&stress_in),
            tangent: internal_to_c1_36(&tangent_in),
            delta_equiv_plastic_strain: legacy.delta_equiv_plastic_strain,
            yielding: legacy.yielding,
        }
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        assert!(
            state.stress.len() == 6 && state.back_stress.len() == 6,
            "J2Material C1 commit needs a 3D PlasticState with 6 components"
        );
        let stress_in = c1_to_internal_6(&update.stress);
        state.stress.copy_from_slice(&stress_in);
        state.equiv_plastic_strain += update.delta_equiv_plastic_strain;
        state.yielding = update.yielding;

        // Back-stress follows the same Prager rule as `update_plastic_state`:
        // Delta alpha = (2/3) C Delta_gamma s_new / q_new.
        let c = self.kinematic_hardening_modulus();
        if update.yielding && update.delta_equiv_plastic_strain > 0.0 && c > 0.0 {
            let q = von_mises_c1(&update.stress);
            if q > 1e-15 {
                let p = (stress_in[0] + stress_in[1] + stress_in[2]) / 3.0;
                let scale = (2.0 / 3.0) * c * update.delta_equiv_plastic_strain / q;
                let dev: [f64; 6] = std::array::from_fn(|i| {
                    if i < 3 {
                        stress_in[i] - p
                    } else {
                        stress_in[i]
                    }
                });
                for (back, &s) in state.back_stress.iter_mut().zip(dev.iter()) {
                    *back += scale * s;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::material::HardeningLaw;
    use super::*;

    #[test]
    fn test_voigt_round_trip() {
        let c1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(internal_to_c1_6(&c1_to_internal_6(&c1)), c1);
        let mut t = [0.0; 36];
        for (i, x) in t.iter_mut().enumerate() {
            *x = i as f64 + 0.5;
        }
        assert_eq!(internal_to_c1_36(&c1_to_internal_36(&t)), t);
    }

    #[test]
    fn test_trait_matches_legacy_path() {
        let material = J2Material::new(
            200e9,
            0.3,
            250e6,
            HardeningLaw::LinearIsotropic { modulus: 1e9 },
        );
        // Same physical strain in both orderings: uniaxial tension.
        let d_c1 = [0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0];
        let d_internal = c1_to_internal_6(&d_c1);

        let mut via_trait = PlasticState::new_3d();
        let update = material.update(&via_trait, &d_c1);
        material.commit(&mut via_trait, &update);

        let mut via_legacy = PlasticState::new_3d();
        let legacy = return_mapping_3d(&material, &via_legacy, &d_internal);
        super::super::return_mapping::update_plastic_state(&mut via_legacy, &legacy, &material);

        assert_eq!(via_trait.yielding, via_legacy.yielding);
        assert!(
            (via_trait.equiv_plastic_strain - via_legacy.equiv_plastic_strain).abs() < 1e-15,
            "plastic strain mismatch"
        );
        for i in 0..6 {
            assert!(
                (via_trait.stress[i] - via_legacy.stress[i]).abs() < 1e-6,
                "stress[{i}] mismatch"
            );
        }
        let legacy_tangent: [f64; 36] = legacy.tangent.as_slice().try_into().unwrap();
        let expected_tangent = internal_to_c1_36(&legacy_tangent);
        for (i, (&got, &exp)) in update
            .tangent
            .iter()
            .zip(expected_tangent.iter())
            .enumerate()
        {
            let scale = exp.abs().max(1.0);
            assert!((got - exp).abs() / scale < 1e-12, "tangent[{i}] mismatch");
        }
    }

    #[test]
    fn test_trait_kinematic_back_stress_parity() {
        let material = J2Material::new(
            200e9,
            0.3,
            250e6,
            HardeningLaw::LinearKinematic { modulus: 5e9 },
        );
        let d_c1 = [0.01, -0.003, -0.003, 0.0, 0.0, 0.0];

        let mut via_trait = PlasticState::new_3d();
        let update = material.update(&via_trait, &d_c1);
        assert!(update.yielding);
        material.commit(&mut via_trait, &update);

        let d_internal = c1_to_internal_6(&d_c1);
        let mut via_legacy = PlasticState::new_3d();
        let legacy = return_mapping_3d(&material, &via_legacy, &d_internal);
        super::super::return_mapping::update_plastic_state(&mut via_legacy, &legacy, &material);

        for i in 0..6 {
            assert!(
                (via_trait.back_stress[i] - via_legacy.back_stress[i]).abs() < 1e-6,
                "back_stress[{i}] mismatch"
            );
        }
    }

    #[test]
    fn test_trait_update_is_pure() {
        let material = J2Material::mild_steel();
        let state = PlasticState::new_3d();
        let before = state.clone();
        let d_c1 = [0.005, -0.0015, -0.0015, 0.001, 0.0, 0.0];
        let _ = material.update(&state, &d_c1);
        assert_eq!(state.stress, before.stress);
        assert_eq!(state.back_stress, before.back_stress);
        assert_eq!(state.equiv_plastic_strain, before.equiv_plastic_strain);
        assert_eq!(state.yielding, before.yielding);
    }

    #[test]
    fn test_trait_shear_ordering() {
        // A pure xy shear must only produce an xy shear stress.
        let material = J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);
        let state = PlasticState::new_3d();
        let update = material.update(&state, &[0.0, 0.0, 0.0, 1.0e-6, 0.0, 0.0]);
        assert!(!update.yielding);
        let expected = material.shear_modulus() * 1.0e-6;
        assert!((update.stress[3] - expected).abs() < 1.0);
        assert!(update.stress[4].abs() < 1.0e-6);
        assert!(update.stress[5].abs() < 1.0e-6);
    }
}
