//! Explicit H8 plasticity with damage and erosion hooks.
//!
//! Runs Johnson-Cook ([`Law2`]) and small-strain J2 plasticity behind the
//! frozen [`hex_explicit_forces`](super::hex_explicit::hex_explicit_forces)
//! material generic: one centroid material point per under-integrated H8,
//! updated and committed through the C1
//! [`StressUpdateModel`](super::super::plasticity::update::StressUpdateModel)
//! contract. After the generic force evaluation,
//! [`hex_plastic_forces`] accumulates Johnson-Cook damage at the same
//! strain rate the flow stress saw and reports the `(ep, damage,
//! porosity)` erosion features that feed
//! [`ErosionSet`](super::super::fracture::ductile::ErosionSet) sweeps.
//! Drivers skip eroded elements with [`hex_plastic_eroded`], which keeps
//! the sparsity pattern and lumped masses stable while zeroing the load.
//!
//! Voigt ordering is C1 (`[xx, yy, zz, xy, yz, zx]`, engineering shears)
//! and all quantities are SI, matching the H8 explicit path.

// Rust guideline compliant 2026-02-21

use super::super::plasticity::J2Material;
use super::super::plasticity::finite_strain::rotate_voigt_c1;
use super::super::plasticity::law_metals::{Law2, Law36};
use super::super::plasticity::update::{StressUpdate, StressUpdateModel};
use super::hex_explicit::{
    HexExplicitError, HexExplicitGeom, HexExplicitParams, HexExplicitState, HexForceOut,
    hex_explicit_forces,
};

/// Plastic material usable on the explicit H8 path.
///
/// Extends the frozen C1 [`StressUpdateModel`] with the two operations an
/// explicit driver needs past the stress update: Johnson-Cook damage
/// accumulation at the committed step rate, and the erosion features for
/// [`ErosionSet`](super::super::fracture::ductile::ErosionSet) sweeps.
/// Damage-free materials keep the default no-op accumulation.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::nonlinear_solid::ExplicitPlasticMaterial;
/// use sonium_fem::physics::solid::plasticity::{J2Material, PlasticState};
///
/// let material = J2Material::mild_steel();
/// let mut state = PlasticState::new_3d();
/// let increment = material.accumulate_explicit_damage(&mut state, &[0.0; 6], 0.0, 1.0);
/// assert_eq!(increment, 0.0);
/// assert_eq!(material.erosion_features(&state), (0.0, 0.0, 0.0));
/// ```
pub trait ExplicitPlasticMaterial: StressUpdateModel<StrainIncrement = [f64; 6]> {
    /// Accumulates damage for one committed explicit step.
    ///
    /// Takes the committed centroid stress (C1 Voigt, Pa), the step
    /// equivalent plastic strain increment, and the step strain rate in
    /// 1/s; returns the damage added this step. The default keeps
    /// damage-free materials untouched and returns zero.
    fn accumulate_explicit_damage(
        &self,
        state: &mut Self::State,
        stress: &[f64; 6],
        dep: f64,
        strain_rate: f64,
    ) -> f64 {
        let _ = (state, stress, dep, strain_rate);
        0.0
    }

    /// Reports `(ep, damage, porosity)` erosion features.
    ///
    /// Takes the committed material-point state; returns the equivalent
    /// plastic strain, the Johnson-Cook damage in `[0, 1]`, and the GTN
    /// porosity (zero for non-porous models), ready for an erosion sweep.
    fn erosion_features(&self, state: &Self::State) -> (f64, f64, f64);

    /// Committed centroid Cauchy stress in C1 Voigt order (Pa).
    ///
    /// Probe-channel output (`SX/SY/SZ/SXY/SYZ/SZX` per element) for
    /// explicit bench drivers; the mechanics never reads it back.
    fn centroid_stress(state: &Self::State) -> [f64; 6];
}

/// Plastic material with tensor-history rotation for corotational H8 updates.
///
/// All tensor-valued history must rotate; scalar hardening and damage remain
/// unchanged. Rotations are row-major orthogonal matrices supplied by the
/// objective force evaluator. The original explicit material trait is retained.
pub trait ObjectivePlasticMaterial: ExplicitPlasticMaterial {
    /// Rotates committed tensor history by the supplied orthogonal matrix.
    fn rotate_state(&self, state: &mut Self::State, rotation: &[f64; 9]);
}

impl ObjectivePlasticMaterial for Law2 {
    fn rotate_state(&self, state: &mut Self::State, rotation: &[f64; 9]) {
        state.stress = rotate_voigt_c1(&state.stress, rotation);
        state.back_stress = rotate_voigt_c1(&state.back_stress, rotation);
    }
}

impl ObjectivePlasticMaterial for Law36 {
    fn rotate_state(&self, state: &mut Self::State, rotation: &[f64; 9]) {
        state.stress = rotate_voigt_c1(&state.stress, rotation);
        state.back_stress = rotate_voigt_c1(&state.back_stress, rotation);
        // C1 plastic strains use engineering shears, whereas the shared tensor
        // rotation accepts stress-like shear components.
        let mut tensor = state.plastic;
        for shear in &mut tensor[3..] {
            *shear *= 0.5;
        }
        tensor = rotate_voigt_c1(&tensor, rotation);
        for shear in &mut tensor[3..] {
            *shear *= 2.0;
        }
        state.plastic = tensor;
    }
}

impl ExplicitPlasticMaterial for J2Material {
    fn erosion_features(&self, state: &Self::State) -> (f64, f64, f64) {
        (state.equiv_plastic_strain, 0.0, 0.0)
    }

    fn centroid_stress(state: &Self::State) -> [f64; 6] {
        // PlasticState stores a variable-length Voigt vector (3 in
        // plane stress, 6 in 3D); missing shear entries read as zero.
        let mut full = [0.0; 6];
        for (slot, value) in state.stress.iter().take(6).enumerate() {
            full[slot] = *value;
        }
        full
    }
}

impl ExplicitPlasticMaterial for Law2 {
    fn accumulate_explicit_damage(
        &self,
        state: &mut Self::State,
        stress: &[f64; 6],
        dep: f64,
        strain_rate: f64,
    ) -> f64 {
        let before = state.damage;
        if dep > 0.0 {
            // Law2::accumulate_damage reads only the stress and the plastic
            // increment; the zero tangent below is never touched.
            let update = StressUpdate {
                stress: *stress,
                tangent: [0.0; 36],
                delta_equiv_plastic_strain: dep,
                yielding: true,
            };
            self.accumulate_damage(state, &update, strain_rate);
        }
        state.damage - before
    }

    fn erosion_features(&self, state: &Self::State) -> (f64, f64, f64) {
        (state.equiv_plastic_strain, state.damage, 0.0)
    }

    fn centroid_stress(state: &Self::State) -> [f64; 6] {
        state.stress
    }
}

impl ExplicitPlasticMaterial for Law36 {
    fn erosion_features(&self, state: &Self::State) -> (f64, f64, f64) {
        // Law36 carries no damage variable; rupture is a binary flag,
        // and porosity stays zero (non-porous model). Damage
        // accumulation keeps the default no-op.
        (
            state.equiv_plastic_strain,
            if state.ruptured { 1.0 } else { 0.0 },
            0.0,
        )
    }

    fn centroid_stress(state: &Self::State) -> [f64; 6] {
        state.stress
    }
}

/// Plastic internal-force evaluation of one H8 element.
#[derive(Debug, Clone)]
pub struct HexPlasticOut {
    /// Generic H8 force evaluation (forces, shock, hardening outputs).
    pub base: HexForceOut,
    /// Johnson-Cook damage accumulated this step (zero without damage).
    pub damage_increment: f64,
    /// `(ep, damage, porosity)` features for the erosion sweep.
    pub features: (f64, f64, f64),
}

/// Explicit plastic internal forces of one H8 element.
///
/// Runs the generic [`hex_explicit_forces`] update for a plastic material,
/// then accumulates Johnson-Cook damage at the reported step rate and
/// returns the erosion features of the committed state. Damage-free
/// materials (plain J2) report a zero increment.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::nonlinear_solid::{
///     HexExplicitGeom, HexExplicitParams, HexExplicitState, HexMesh, hex_plastic_forces,
/// };
/// use sonium_fem::physics::solid::plasticity::{HardeningLaw, J2Material, PlasticState};
///
/// let geom = HexExplicitGeom::from_mesh(&HexMesh::unit_cube(1.0), 0).unwrap();
/// let material = J2Material::new(200.0e9, 0.3, 250.0e6, HardeningLaw::Perfect);
/// let mut state = HexExplicitState::new(PlasticState::new_3d());
/// let params = HexExplicitParams::new(7800.0, 4500.0);
/// let out = hex_plastic_forces(&geom, &mut state, &material, &[0.0; 24], 1.0e-6, &params).unwrap();
/// assert!(!out.base.yielding);
/// assert_eq!(out.damage_increment, 0.0);
/// ```
///
/// # Errors
///
/// Returns [`HexExplicitError`] for non-positive `dt` or non-finite input.
pub fn hex_plastic_forces<M>(
    geom: &HexExplicitGeom,
    state: &mut HexExplicitState<M::State>,
    material: &M,
    velocities: &[f64; 24],
    dt: f64,
    params: &HexExplicitParams,
) -> Result<HexPlasticOut, HexExplicitError>
where
    M: ExplicitPlasticMaterial,
{
    let base = hex_explicit_forces(geom, state, material, velocities, dt, params)?;
    let damage_increment = material.accumulate_explicit_damage(
        &mut state.centroid,
        &base.stress,
        base.dep,
        base.strain_rate,
    );
    let features = material.erosion_features(&state.centroid);
    Ok(HexPlasticOut {
        base,
        damage_increment,
        features,
    })
}

/// Advances an isotropic H8 plastic material with an objective spin update.
///
/// Splits the incremental Jaumann rotation into two exact half rotations
/// around the existing strain/return-mapping update. Final internal forces use
/// the rotated Cauchy stress; bulk viscosity and hourglass forces are retained.
/// Pure rigid rotation rotates tensor history without producing plastic work.
/// This is a spin-based hypoelastic update, not the finite-strain J2 model's
/// separate Green-Naghdi/logarithmic-strain formulation.
///
/// # Errors
/// Rejects nonfinite or nonpositive timestep, velocities, or spin increment;
/// forwards errors from the original plastic force evaluator.
pub fn hex_plastic_forces_objective<M: ObjectivePlasticMaterial>(
    geom: &HexExplicitGeom,
    state: &mut HexExplicitState<M::State>,
    material: &M,
    velocities: &[f64; 24],
    dt: f64,
    params: &HexExplicitParams,
) -> Result<HexPlasticOut, HexExplicitError> {
    if !dt.is_finite() || dt <= 0.0 || velocities.iter().any(|v| !v.is_finite()) {
        return Err(HexExplicitError::new(
            "objective hex update requires finite velocity and positive finite dt",
        ));
    }
    let mut gradient = [[0.0; 3]; 3];
    for (node, grad) in geom.grad_b.iter().enumerate() {
        for (axis, row) in gradient.iter_mut().enumerate() {
            for (entry, derivative) in row.iter_mut().zip(grad) {
                *entry += velocities[3 * node + axis] * derivative;
            }
        }
    }
    let spin = [
        0.5 * (gradient[2][1] - gradient[1][2]),
        0.5 * (gradient[0][2] - gradient[2][0]),
        0.5 * (gradient[1][0] - gradient[0][1]),
    ];
    let magnitude = spin[0].hypot(spin[1]).hypot(spin[2]);
    let angle = 0.5 * magnitude * dt;
    if !angle.is_finite() {
        return Err(HexExplicitError::new(
            "objective hex spin increment must be finite",
        ));
    }
    if magnitude == 0.0 {
        return hex_plastic_forces(geom, state, material, velocities, dt, params);
    }
    let rotation = super::super::joints::rotation_matrix(spin, angle);
    let rotation = std::array::from_fn(|i| rotation[i / 3][i % 3]);
    material.rotate_state(&mut state.centroid, &rotation);
    let mut out = hex_plastic_forces(geom, state, material, velocities, dt, params)?;
    material.rotate_state(&mut state.centroid, &rotation);
    let stress = M::centroid_stress(&state.centroid);
    let delta: [f64; 6] = std::array::from_fn(|i| stress[i] - out.base.stress[i]);
    for (node, g) in geom.grad_b.iter().enumerate() {
        out.base.forces[3 * node] +=
            geom.volume * (delta[0] * g[0] + delta[3] * g[1] + delta[5] * g[2]);
        out.base.forces[3 * node + 1] +=
            geom.volume * (delta[3] * g[0] + delta[1] * g[1] + delta[4] * g[2]);
        out.base.forces[3 * node + 2] +=
            geom.volume * (delta[5] * g[0] + delta[4] * g[1] + delta[2] * g[2]);
    }
    out.base.stress = stress;
    Ok(out)
}

/// Zero plastic response of an eroded H8 element.
///
/// Drivers return this for elements the [`ErosionSet`](super::super::fracture::ductile::ErosionSet)
/// already deleted instead of calling [`hex_plastic_forces`]: no forces,
/// no shock pressure, no hardening outputs, and null features.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::nonlinear_solid::hex_plastic_eroded;
///
/// let out = hex_plastic_eroded();
/// assert!(out.base.forces.iter().all(|f| *f == 0.0));
/// assert_eq!(out.features, (0.0, 0.0, 0.0));
/// ```
#[must_use]
pub fn hex_plastic_eroded() -> HexPlasticOut {
    HexPlasticOut {
        base: HexForceOut {
            forces: [0.0; 24],
            hg_power: 0.0,
            bulk_q: 0.0,
            dep: 0.0,
            yielding: false,
            stress: [0.0; 6],
            strain_rate: 0.0,
        },
        damage_increment: 0.0,
        features: (0.0, 0.0, 0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::super::hex::HexMesh;
    use super::*;
    use crate::physics::solid::fracture::ductile::{
        ErosionCriterion, ErosionSet, JohnsonCookDamage, stress_triaxiality_c1,
    };
    use crate::physics::solid::plasticity::HardeningLaw;
    use crate::physics::solid::plasticity::PlasticState;
    use crate::physics::solid::plasticity::law_metals::{
        Law2Flow, Law2State, Law36State, MaxRateEffect, RateCurve,
    };
    use crate::physics::solid::plasticity::update::von_mises_c1;

    #[test]
    fn objective_rigid_rotation_rotates_stress_without_plastic_work() {
        let mesh = HexMesh::unit_cube(1.0);
        let geom = HexExplicitGeom::from_mesh(&mesh, 0).unwrap();
        let material = closed_form_jc();
        let params = HexExplicitParams::new(7800.0, 4500.0);
        let mut state = HexExplicitState::new(Law2State::new(293.0));
        state.centroid.stress[0] = 100e6;
        state.centroid.back_stress[0] = 15e6;
        state.centroid.equiv_plastic_strain = 0.2;
        let mut legacy = state.clone();
        let mut velocities = [0.0; 24];
        // Uniform z spin through exactly 90 degrees. Its symmetric velocity
        // gradient vanishes; the analytical final stress has only sigma_yy.
        let omega = std::f64::consts::FRAC_PI_2;
        for (node, p) in mesh.coords.iter().enumerate() {
            velocities[3 * node] = -omega * p[1];
            velocities[3 * node + 1] = omega * p[0];
        }
        let old =
            hex_plastic_forces(&geom, &mut legacy, &material, &velocities, 1.0, &params).unwrap();
        assert!((old.base.stress[0] - 100e6).abs() < 1e-5);
        let out =
            hex_plastic_forces_objective(&geom, &mut state, &material, &velocities, 1.0, &params)
                .unwrap();
        for (actual, expected) in out.base.stress.iter().zip([0.0, 100e6, 0.0, 0.0, 0.0, 0.0]) {
            assert!((actual - expected).abs() < 1e-5);
        }
        assert!((state.centroid.back_stress[1] - 15e6).abs() < 1e-6);
        assert!((state.centroid.equiv_plastic_strain - 0.2).abs() < 1e-14);
        assert_eq!(out.base.dep, 0.0);
        assert_eq!(out.damage_increment, 0.0);
        assert!(out.base.hg_power.abs() < 1e-7);
        for (node, grad) in geom.grad_b.iter().enumerate() {
            assert!(out.base.forces[3 * node].abs() < 1e-5);
            assert!((out.base.forces[3 * node + 1] - 100e6 * grad[1]).abs() < 1e-5);
            assert!(out.base.forces[3 * node + 2].abs() < 1e-5);
        }
    }

    #[test]
    fn objective_update_rejects_nonfinite_inputs_without_mutating_history() {
        let geom = HexExplicitGeom::from_mesh(&HexMesh::unit_cube(1.0), 0).unwrap();
        let material = closed_form_jc();
        let params = HexExplicitParams::new(7800.0, 4500.0);
        let mut state = HexExplicitState::new(Law2State::new(293.0));
        state.centroid.stress[0] = 100e6;
        for dt in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            assert!(
                hex_plastic_forces_objective(&geom, &mut state, &material, &[0.0; 24], dt, &params)
                    .is_err()
            );
            assert_eq!(state.centroid.stress[0], 100e6);
        }
    }

    #[test]
    fn objective_law36_rotation_preserves_engineering_plastic_shear() {
        let curve = RateCurve::new(0.0, vec![0.0, 0.5], vec![250e6, 250e6], 1.0).unwrap();
        let material = Law36::new(200e9, 0.3, vec![curve], 0.0, 1e20, 1e20, 2e20, 3e20);
        let mut state = Law36State::new();
        state.plastic[0] = 0.04;
        state.equiv_plastic_strain = 0.04;
        // Rotate the axial plastic tensor 45 degrees around z. Analytically,
        // eps_xx=eps_yy=.02 and engineering gamma_xy=.04.
        let c = std::f64::consts::FRAC_1_SQRT_2;
        let rotation = [c, -c, 0.0, c, c, 0.0, 0.0, 0.0, 1.0];
        material.rotate_state(&mut state, &rotation);
        for (actual, expected) in state.plastic.iter().zip([0.02, 0.02, 0.0, 0.04, 0.0, 0.0]) {
            assert!((actual - expected).abs() < 1e-14);
        }
        assert_eq!(state.equiv_plastic_strain, 0.04);
    }

    /// Isothermal rate-free Johnson-Cook steel for closed-form checks.
    ///
    /// Zero rate sensitivity, no thermal run, and no adiabatic heating keep
    /// the yielded von Mises stress exactly at `a + b ep^n`.
    fn closed_form_jc() -> Law2 {
        Law2::new(
            200.0e9,
            0.3,
            Law2Flow::JohnsonCook {
                a: 400.0e6,
                b: 300.0e6,
                n: 0.5,
                c: 0.0,
            },
            1.0,
            0.0,
            0.0,
            298.0,
            0.0,
            1.0e30,
            MaxRateEffect::Scaled,
            1.0e30,
            0.0,
            1.0e30,
        )
    }

    fn unit_cube() -> HexExplicitGeom {
        HexExplicitGeom::from_mesh(&HexMesh::unit_cube(1.0), 0).unwrap()
    }

    fn steel_params() -> HexExplicitParams {
        HexExplicitParams::new(7800.0, 4500.0)
    }

    /// Uniaxial-strain velocity field at rate `rate` (1/s) on the unit cube.
    fn uniaxial_velocities(rate: f64) -> [f64; 24] {
        let mesh = HexMesh::unit_cube(1.0);
        let mut velocities = [0.0; 24];
        for node in 0..8 {
            velocities[node * 3] = rate * mesh.coords[node][0];
        }
        velocities
    }

    #[test]
    fn jc_yield_matches_flow_stress_closed_form() {
        let geom = unit_cube();
        let material = closed_form_jc();
        let mut state = HexExplicitState::new(Law2State::new(298.0));
        let params = steel_params();
        // One percent uniaxial strain at 1/s: far past yield.
        let out = hex_plastic_forces(
            &geom,
            &mut state,
            &material,
            &uniaxial_velocities(1.0),
            1.0e-2,
            &params,
        )
        .unwrap();
        assert!(out.base.yielding);
        assert!(out.base.dep > 0.0);
        assert_eq!(out.damage_increment, 0.0);
        let ep = state.centroid.equiv_plastic_strain;
        assert!((out.features.0 - ep).abs() < 1e-15);
        // Radial return lands exactly on the isothermal rate-free surface.
        let expected = 400.0e6 + 300.0e6 * ep.powf(0.5);
        let got = von_mises_c1(&out.base.stress);
        assert!(
            (got - expected).abs() / expected < 1e-9,
            "von Mises {got:.6e} vs flow {expected:.6e}"
        );
    }

    #[test]
    fn jc_rate_dispatches_through_the_generic() {
        let geom = unit_cube();
        let mut material = closed_form_jc();
        material.flow = Law2Flow::JohnsonCook {
            a: 400.0e6,
            b: 300.0e6,
            n: 0.5,
            c: 0.02,
        };
        assert!(StressUpdateModel::is_rate_dependent(&material));
        let params = steel_params();
        // Same one-percent strain at 1/s and at 1000/s.
        let mut slow = HexExplicitState::new(Law2State::new(298.0));
        let slow_out = hex_plastic_forces(
            &geom,
            &mut slow,
            &material,
            &uniaxial_velocities(1.0),
            1.0e-2,
            &params,
        )
        .unwrap();
        let mut fast = HexExplicitState::new(Law2State::new(298.0));
        let fast_out = hex_plastic_forces(
            &geom,
            &mut fast,
            &material,
            &uniaxial_velocities(1000.0),
            1.0e-5,
            &params,
        )
        .unwrap();
        assert!(slow_out.base.yielding && fast_out.base.yielding);
        assert!(fast_out.base.strain_rate > 100.0 * slow_out.base.strain_rate);
        // Matching increments at a higher rate must harden the response.
        let slow_q = von_mises_c1(&slow_out.base.stress);
        let fast_q = von_mises_c1(&fast_out.base.stress);
        assert!(
            fast_q > slow_q,
            "fast {fast_q:.6e} must exceed slow {slow_q:.6e}"
        );
    }

    #[test]
    fn jc_damage_matches_direct_accumulation() {
        let geom = unit_cube();
        let mut material = Law2::c4340_steel();
        let damage = JohnsonCookDamage::c4340_steel();
        material.damage = Some(damage);
        let mut state = HexExplicitState::new(Law2State::new(298.0));
        let params = steel_params();
        // Two percent uniaxial strain: yields and damages.
        let out = hex_plastic_forces(
            &geom,
            &mut state,
            &material,
            &uniaxial_velocities(1.0),
            2.0e-2,
            &params,
        )
        .unwrap();
        assert!(out.base.yielding);
        assert!(out.damage_increment > 0.0);
        assert!((out.features.1 - state.centroid.damage).abs() < 1e-15);
        // The wrapper must agree exactly with the direct damage rule at the
        // reported step stress, increment, rate, and committed temperature.
        let triax = stress_triaxiality_c1(&out.base.stress);
        let expected = damage.increment(
            triax,
            out.base.dep,
            out.base.strain_rate,
            material.homologous_temperature(state.centroid.temperature),
        );
        assert!(
            (out.damage_increment - expected).abs() < 1e-15,
            "wrapper {} vs direct {expected}",
            out.damage_increment
        );
    }

    #[test]
    fn j2_flows_without_damage() {
        let geom = unit_cube();
        let material = J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);
        let params = steel_params();
        let mut state = HexExplicitState::new(PlasticState::new_3d());
        // Microstrain stays elastic with null features.
        let elastic = hex_plastic_forces(
            &geom,
            &mut state,
            &material,
            &uniaxial_velocities(1.0),
            1.0e-6,
            &params,
        )
        .unwrap();
        assert!(!elastic.base.yielding);
        assert_eq!(elastic.damage_increment, 0.0);
        assert_eq!(elastic.features, (0.0, 0.0, 0.0));
        // One percent yields; only the plastic-strain feature fills in.
        let plastic = hex_plastic_forces(
            &geom,
            &mut state,
            &material,
            &uniaxial_velocities(1.0),
            1.0e-2,
            &params,
        )
        .unwrap();
        assert!(plastic.base.yielding);
        assert!(plastic.features.0 > 0.0);
        assert_eq!(plastic.damage_increment, 0.0);
        assert_eq!((plastic.features.1, plastic.features.2), (0.0, 0.0));
    }

    #[test]
    fn features_drive_erosion_and_zero_response() {
        let criterion = ErosionCriterion::plastic_strain(0.01);
        let mut set = ErosionSet::new(2);
        let material = J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);
        let virgin = PlasticState::new_3d();
        let mut yielded = PlasticState::new_3d();
        yielded.equiv_plastic_strain = 0.02;
        let features = [
            material.erosion_features(&virgin),
            material.erosion_features(&yielded),
        ];
        assert_eq!(set.sweep(&criterion, &features), vec![1]);
        assert!(set.is_active(0));
        assert!(!set.is_active(1));
        let out = hex_plastic_eroded();
        assert!(out.base.forces.iter().all(|f| *f == 0.0));
        assert_eq!(out.base.hg_power, 0.0);
        assert_eq!(out.base.bulk_q, 0.0);
        assert!(!out.base.yielding);
    }

    #[test]
    fn law36_caps_at_tabulated_yield() {
        let geom = unit_cube();
        let curve = RateCurve::new(0.0, vec![0.0, 0.5], vec![250.0e6, 250.0e6], 1.0).unwrap();
        let material = Law36::new(
            200.0e9,
            0.3,
            vec![curve],
            0.0,
            1.0e20,
            1.0e20,
            2.0e20,
            3.0e20,
        );
        let mut state = HexExplicitState::new(Law36State::new());
        let params = steel_params();
        // One percent uniaxial strain at 1/s: far past yield.
        let out = hex_plastic_forces(
            &geom,
            &mut state,
            &material,
            &uniaxial_velocities(1.0),
            1.0e-2,
            &params,
        )
        .unwrap();
        assert!(out.base.yielding);
        assert!(out.base.dep > 0.0);
        assert_eq!(out.damage_increment, 0.0);
        let ep = state.centroid.equiv_plastic_strain;
        assert!((out.features.0 - ep).abs() < 1e-15);
        assert_eq!((out.features.1, out.features.2), (0.0, 0.0));
        // Radial return lands on the flat tabulated surface.
        let got = von_mises_c1(&out.base.stress);
        assert!(
            (got - 250.0e6).abs() / 250.0e6 < 0.01,
            "von Mises {got:.6e} off the 250 MPa surface"
        );
        // Uniaxial *strain* (laterally constrained), not uniaxial
        // stress: the 1% volumetric expansion carries mean stress
        // K*0.01 on top of the capped deviator (sxx:syy:szz = 2:-1:-1,
        // so sxx = 250 MPa * 2/3), and each of the four face nodes
        // carries a quarter of the axial stress times unit area.
        let bulk = 200.0e9 / (3.0 * (1.0 - 2.0 * 0.3));
        let mark = (bulk * 0.01 + 250.0e6 * 2.0 / 3.0) / 4.0;
        let peak = out
            .base
            .forces
            .iter()
            .map(|f| f.abs())
            .fold(0.0f64, f64::max);
        assert!(
            (peak - mark).abs() / mark < 0.05,
            "peak nodal force {peak:.6e} off the {mark:.6e} N constrained-strain mark"
        );
    }

    #[test]
    fn rejects_bad_input() {
        let geom = unit_cube();
        let material = closed_form_jc();
        let mut state = HexExplicitState::new(Law2State::new(298.0));
        let params = steel_params();
        assert!(
            hex_plastic_forces(&geom, &mut state, &material, &[0.0; 24], 0.0, &params).is_err()
        );
        let mut bad = [0.0; 24];
        bad[5] = f64::INFINITY;
        assert!(hex_plastic_forces(&geom, &mut state, &material, &bad, 1.0e-6, &params).is_err());
    }
}
