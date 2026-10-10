//! Fiber-section integration with J2 plasticity.
//!
//! A beam cross-section is discretized into uniaxial fibers. Each fiber calls
//! a stress-update model through the authoritative contract C1 trait
//! ([`StressUpdateModel`]); section forces follow from fiber resultants.

// Rust guideline compliant 2026-02-21

use crate::physics::solid::plasticity::update::{StressUpdate, StressUpdateModel};
use crate::physics::solid::plasticity::{J2Material, PlasticState};

// === Fiber-layer adapter contract (owned by this file) ===

/// Builds a model strain increment from a uniaxial fiber strain increment.
///
/// The increment must leave the fiber in uniaxial stress: transverse
/// components are chosen so transverse stresses (near-)vanish, making the
/// fiber axial stress the true uniaxial response. Adapters receive the
/// committed state so they can iterate the transverse split when needed.
pub trait FiberCompatibleModel: StressUpdateModel {
    /// Build the increment consumed by [`StressUpdateModel::update`].
    fn uniaxial_increment(&self, state: &Self::State, d_axial: f64) -> Self::StrainIncrement;
}

// === Concrete adapters over the existing small-strain J2 model ===

/// Uniaxial J2 adapter implementing the C1 seam trait.
///
/// Transverse increments use elastic Poisson contraction, exact in the
/// elastic regime and an engineering approximation after yield (transverse
/// constraint overstresses fibers under large plastic flow; use
/// [`UniaxialStressJ2Adapter`] for plastic-hinge work). The axial stress
/// and axial tangent drive the fiber response.
#[derive(Debug, Clone, Copy)]
pub struct UniaxialJ2Adapter {
    /// J2 material with hardening law.
    pub material: J2Material,
}

impl UniaxialJ2Adapter {
    /// Wrap an existing J2 material for fiber use.
    #[must_use]
    pub fn new(material: J2Material) -> Self {
        Self { material }
    }
}

impl StressUpdateModel for UniaxialJ2Adapter {
    type State = PlasticState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        self.material.update(state, d_eps)
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        self.material.commit(state, update)
    }
}

impl FiberCompatibleModel for UniaxialJ2Adapter {
    fn uniaxial_increment(&self, _state: &Self::State, d_axial: f64) -> Self::StrainIncrement {
        let transverse = -self.material.poisson_ratio * d_axial;
        [d_axial, transverse, transverse, 0.0, 0.0, 0.0]
    }
}

/// Uniaxial-stress J2 adapter with iterated transverse split.
///
/// Solves for transverse increments that zero the transverse stresses with
/// Newton iteration on the 2x2 transverse block (elastic Poisson start,
/// exact in one step pre-yield). Fibers therefore cap at the uniaxial flow
/// stress and section moments respect the plastic limit, which the
/// Poisson-transverse adapter overestimates under large plastic flow.
#[derive(Debug, Clone, Copy)]
pub struct UniaxialStressJ2Adapter {
    /// J2 material with hardening law.
    pub material: J2Material,
}

impl UniaxialStressJ2Adapter {
    /// Wrap an existing J2 material for fiber use.
    #[must_use]
    pub fn new(material: J2Material) -> Self {
        Self { material }
    }
}

impl StressUpdateModel for UniaxialStressJ2Adapter {
    type State = PlasticState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        self.material.update(state, d_eps)
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        self.material.commit(state, update)
    }
}

impl FiberCompatibleModel for UniaxialStressJ2Adapter {
    fn uniaxial_increment(&self, state: &Self::State, d_axial: f64) -> Self::StrainIncrement {
        let mut transverse = -self.material.poisson_ratio * d_axial;
        let mut other = transverse;
        // Residual transverse stress below this cannot move section
        // resultants at QA precision (1e-9 of the axial stress scale).
        let tol =
            1.0e-9 * (self.material.youngs_modulus * d_axial.abs() + self.material.yield_stress);
        for _ in 0..20 {
            let increment = [d_axial, transverse, other, 0.0, 0.0, 0.0];
            let update = self.material.update(state, &increment);
            let f0 = update.stress[1];
            let f1 = update.stress[2];
            if f0.abs() <= tol && f1.abs() <= tol {
                return [d_axial, transverse, other, 0.0, 0.0, 0.0];
            }
            // 2x2 transverse Jacobian from the yy/zz tangent block.
            let t = &update.tangent;
            let (a, b, c, d) = (t[7], t[8], t[13], t[14]);
            let det = a * d - b * c;
            let scale = a.abs() * d.abs() + b.abs() * c.abs();
            if det.abs() <= 1.0e-12 * scale.max(1.0e-300) {
                // Degenerate transverse block: keep the best iterate; the
                // strain condensation in the caller still applies.
                return [d_axial, transverse, other, 0.0, 0.0, 0.0];
            }
            transverse -= (d * f0 - b * f1) / det;
            other -= (a * f1 - c * f0) / det;
        }
        debug_assert!(false, "uniaxial-stress Newton did not converge");
        [d_axial, transverse, other, 0.0, 0.0, 0.0]
    }
}

// === Fiber section ===

/// One uniaxial fiber at a section point.
#[derive(Debug, Clone, Copy)]
pub struct Fiber {
    /// Local `y` coordinate in m.
    pub y: f64,
    /// Local `z` coordinate in m.
    pub z: f64,
    /// Tributary area in m^2.
    pub area: f64,
}

impl Fiber {
    /// Place a fiber with tributary area.
    #[must_use]
    pub fn new(y: f64, z: f64, area: f64) -> Self {
        Self { y, z, area }
    }

    /// Rectangular grid of fibers over `[-b/2, b/2] x [-h/2, h/2]`.
    #[must_use]
    pub fn grid(b: f64, h: f64, ny: usize, nz: usize) -> Vec<Fiber> {
        assert!(ny > 0 && nz > 0, "fiber grid needs positive counts");
        let area = b * h / (ny as f64 * nz as f64);
        let mut fibers = Vec::with_capacity(ny * nz);
        for i in 0..ny {
            let y = -b / 2.0 + b * (i as f64 + 0.5) / ny as f64;
            for k in 0..nz {
                let z = -h / 2.0 + h * (k as f64 + 0.5) / nz as f64;
                fibers.push(Fiber::new(y, z, area));
            }
        }
        fibers
    }
}

/// Section resultants and tangent for one fiber integration.
#[derive(Debug, Clone)]
pub struct FiberResult {
    /// Axial force in N.
    pub n: f64,
    /// Bending moment about local `y` in N m.
    pub my: f64,
    /// Bending moment about local `z` in N m.
    pub mz: f64,
    /// Section tangent for `[eps0, kappa_y, kappa_z]`, row-major 3x3.
    pub tangent: [f64; 9],
    /// Per-fiber trial updates in fiber order.
    pub updates: Vec<StressUpdate>,
}

/// Fiber-discretized beam section with C1 material response.
///
/// Axial strain follows `eps = e0 - y kappa_z + z kappa_y`, consistent with
/// the [`BeamElement3D`](super::element3d::BeamElement3D) sign convention.
#[derive(Debug, Clone)]
pub struct FiberSection<M: FiberCompatibleModel> {
    /// Stress-update model shared by all fibers.
    pub model: M,
    /// Fibers across the section.
    pub fibers: Vec<Fiber>,
}

impl<M: FiberCompatibleModel> FiberSection<M> {
    /// Build a section from a model and fiber layout.
    #[must_use]
    pub fn new(model: M, fibers: Vec<Fiber>) -> Self {
        Self { model, fibers }
    }

    /// Fresh committed states for every fiber.
    #[must_use]
    pub fn initial_states(&self, make: impl Fn() -> M::State) -> Vec<M::State> {
        (0..self.fibers.len()).map(|_| make()).collect()
    }

    /// Axial strain of every fiber for section deformations.
    #[must_use]
    pub fn fiber_strains(&self, e0: f64, kappa_y: f64, kappa_z: f64) -> Vec<f64> {
        self.fibers
            .iter()
            .map(|fiber| e0 - fiber.y * kappa_z + fiber.z * kappa_y)
            .collect()
    }

    /// Trial section response for section deformation increments.
    ///
    /// # Panics
    ///
    /// Panics when `states` does not match the fiber count.
    #[must_use]
    pub fn integrate(
        &self,
        states: &[M::State],
        d_e0: f64,
        d_kappa_y: f64,
        d_kappa_z: f64,
    ) -> FiberResult {
        assert_eq!(
            states.len(),
            self.fibers.len(),
            "fiber state count mismatch"
        );
        let mut result = FiberResult {
            n: 0.0,
            my: 0.0,
            mz: 0.0,
            tangent: [0.0; 9],
            updates: Vec::with_capacity(self.fibers.len()),
        };
        for (fiber, state) in self.fibers.iter().zip(states.iter()) {
            let d_axial = d_e0 - fiber.y * d_kappa_z + fiber.z * d_kappa_y;
            let increment = self.model.uniaxial_increment(state, d_axial);
            let update = self.model.update(state, &increment);
            let sigma = update.stress[0];
            let et = uniaxial_tangent(&update.tangent);
            result.n += sigma * fiber.area;
            result.my += sigma * fiber.z * fiber.area;
            result.mz += -sigma * fiber.y * fiber.area;
            // Chain rule rows for [e0, kappa_y, kappa_z].
            let grad = [1.0, fiber.z, -fiber.y];
            for i in 0..3 {
                for j in 0..3 {
                    result.tangent[i * 3 + j] += et * fiber.area * grad[i] * grad[j];
                }
            }
            result.updates.push(update);
        }
        result
    }

    /// Commit trial updates into stored fiber states.
    ///
    /// # Panics
    ///
    /// Panics when state or update counts do not match the fiber count.
    pub fn commit(&self, states: &mut [M::State], result: &FiberResult) {
        assert_eq!(
            states.len(),
            self.fibers.len(),
            "fiber state count mismatch"
        );
        assert_eq!(
            result.updates.len(),
            self.fibers.len(),
            "fiber update count mismatch"
        );
        for (state, update) in states.iter_mut().zip(result.updates.iter()) {
            self.model.commit(state, update);
        }
    }
}

/// Condense a 3-D tangent to the uniaxial fiber tangent.
///
/// Transverse stress increments vanish (`dσyy = dσzz = 0`), which determines
/// the transverse strain increments from the `yy/zz` block and yields
/// `Et = dσxx / dεxx`. In the elastic regime this recovers Young's modulus.
#[must_use]
pub fn uniaxial_tangent(tangent: &[f64; 36]) -> f64 {
    let a11 = tangent[7];
    let a12 = tangent[8];
    let a21 = tangent[13];
    let a22 = tangent[14];
    let det = a11 * a22 - a12 * a21;
    if det.abs() < 1.0e-300 {
        return tangent[0];
    }
    let b1 = -tangent[6];
    let b2 = -tangent[12];
    let trans_y = (b1 * a22 - b2 * a12) / det;
    let trans_z = (a11 * b2 - a21 * b1) / det;
    tangent[0] + tangent[1] * trans_y + tangent[2] * trans_z
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::solid::plasticity::HardeningLaw;

    fn steel_adapter() -> UniaxialJ2Adapter {
        UniaxialJ2Adapter::new(J2Material::new(
            200.0e9,
            0.3,
            250.0e6,
            HardeningLaw::LinearIsotropic { modulus: 1.0e9 },
        ))
    }

    #[test]
    fn elastic_fiber_matches_young_modulus() {
        let adapter = steel_adapter();
        let state = PlasticState::new_3d();
        let increment = adapter.uniaxial_increment(&state, 1.0e-4);
        let update = adapter.update(&state, &increment);
        assert!(!update.yielding);
        assert!((update.stress[0] - 200.0e9 * 1.0e-4).abs() / 20.0e6 < 1.0e-9);
        // Raw tangent carries the constrained modulus; condensation gives E.
        let constrained = 200.0e9 * (1.0 - 0.3) / ((1.0 + 0.3) * (1.0 - 0.6));
        assert!((update.tangent[0] - constrained).abs() / constrained < 1.0e-9);
        assert!((uniaxial_tangent(&update.tangent) - 200.0e9).abs() / 200.0e9 < 1.0e-9);
    }

    #[test]
    fn fiber_section_recovers_ea_ei_elastically() {
        let b = 0.02;
        let h = 0.01;
        let section = FiberSection::new(steel_adapter(), Fiber::grid(b, h, 8, 8));
        let states = section.initial_states(PlasticState::new_3d);
        let e = 200.0e9;
        let axial = section.integrate(&states, 1.0e-6, 0.0, 0.0);
        assert!((axial.n / 1.0e-6 - e * b * h).abs() / (e * b * h) < 0.02);
        let bend_z = section.integrate(&states, 0.0, 0.0, 1.0e-4);
        let eiz = e * h * b.powi(3) / 12.0;
        assert!((bend_z.mz / 1.0e-4 - eiz).abs() / eiz < 0.05);
        // Tangent diagonal carries EA and the EI terms.
        assert!((axial.tangent[0] - e * b * h).abs() / (e * b * h) < 0.02);
    }

    #[test]
    fn uniaxial_stress_fiber_caps_at_flow_stress() {
        let adapter = UniaxialStressJ2Adapter::new(J2Material::new(
            200.0e9,
            0.3,
            250.0e6,
            HardeningLaw::Perfect,
        ));
        let state = PlasticState::new_3d();
        // Ten times the yield strain: axial stress must cap at sigma_y with
        // vanishing transverse stress (true uniaxial response).
        let increment = adapter.uniaxial_increment(&state, 1.25e-2);
        let update = adapter.update(&state, &increment);
        assert!(update.yielding);
        assert!((update.stress[0] - 250.0e6).abs() / 250.0e6 < 1.0e-6);
        assert!(update.stress[1].abs() / 250.0e6 < 1.0e-6);
        assert!(update.stress[2].abs() / 250.0e6 < 1.0e-6);
    }

    #[test]
    fn cyclic_loading_hardens_and_commits() {
        let adapter = steel_adapter();
        let section = FiberSection::new(adapter, Fiber::grid(0.02, 0.01, 4, 4));
        let mut states = section.initial_states(PlasticState::new_3d);
        // Pull past yield (yield strain is 1.25e-3 for this steel).
        let first = section.integrate(&states, 3.0e-3, 0.0, 0.0);
        assert!(first.updates.iter().any(|update| update.yielding));
        section.commit(&mut states, &first);
        assert!(states.iter().all(|state| state.equiv_plastic_strain > 0.0));
        // Partial unload stays elastic inside the hardened surface.
        let unload = section.integrate(&states, -1.5e-3, 0.0, 0.0);
        assert!(unload.updates.iter().all(|update| !update.yielding));
        section.commit(&mut states, &unload);
        // Reload from the unloaded state stays elastic with margin.
        let reload = section.integrate(&states, 1.4e-3, 0.0, 0.0);
        assert!(reload.updates.iter().all(|update| !update.yielding));
        // Pushing past the forward point yields again on the hardened surface.
        let push = section.integrate(&states, 2.0e-3, 0.0, 0.0);
        assert!(push.updates.iter().any(|update| update.yielding));
    }
}
