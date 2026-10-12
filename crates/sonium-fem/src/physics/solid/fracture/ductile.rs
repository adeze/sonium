//! Ductile damage, porous plasticity, and element erosion.
//!
//! Collects the failure side of the crash chain: Johnson-Cook damage
//! accumulation driven by triaxiality, rate, and temperature; a simplified
//! Gurson-Tvergaard-Needleman porous-plasticity model with Needleman void
//! kinetics; and the erosion criterion plus deletion set that remove failed
//! elements through a stable path. Necking reference helpers (Considere
//! uniform strain, Bridgman correction) live here because the necking-bar
//! benchmark consumes them.

// Rust guideline compliant 2026-02-21

use crate::physics::solid::plasticity::update::{StressUpdate, StressUpdateModel, von_mises_c1};
use crate::physics::solid::plasticity::{J2Material, PlasticState};

/// Stress triaxiality `p / q` of a C1 Voigt stress.
///
/// Returns the hydrostatic pressure over the von Mises stress, with the
/// denominator floored at 1e-9 Pa times the pressure scale so hydrostatic
/// states report a bounded value instead of infinity.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::fracture::ductile::stress_triaxiality_c1;
///
/// let uniaxial = [300.0e6, 0.0, 0.0, 0.0, 0.0, 0.0];
/// assert!((stress_triaxiality_c1(&uniaxial) - 1.0 / 3.0).abs() < 1e-12);
/// ```
pub fn stress_triaxiality_c1(stress: &[f64; 6]) -> f64 {
    let p = (stress[0] + stress[1] + stress[2]) / 3.0;
    let q = von_mises_c1(stress);
    p / q.max(1e-9 * p.abs().max(1.0))
}

/// Johnson-Cook ductile damage model.
///
/// Fracture strain `ef = (D1 + D2 exp(D3 triax)) (1 + D4 ln e*)
/// (1 + D5 T*)` with normalized rate `e*` and homologous temperature `T*`.
/// Damage accumulates as `D += dep / ef` and fractures at `D = 1`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::fracture::ductile::JohnsonCookDamage;
///
/// let damage = JohnsonCookDamage::c4340_steel();
/// let ef = damage.fracture_strain(1.0 / 3.0, 1.0, 0.0);
/// assert!(ef > 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct JohnsonCookDamage {
    /// Damage constant D1, dimensionless.
    pub d1: f64,
    /// Damage constant D2, dimensionless.
    pub d2: f64,
    /// Triaxiality sensitivity D3, dimensionless.
    pub d3: f64,
    /// Rate sensitivity D4, dimensionless.
    pub d4: f64,
    /// Temperature sensitivity D5, dimensionless.
    pub d5: f64,
    /// Reference strain rate in 1/s.
    pub reference_strain_rate: f64,
}

impl JohnsonCookDamage {
    /// Creates a Johnson-Cook damage model from five constants.
    pub fn new(d1: f64, d2: f64, d3: f64, d4: f64, d5: f64, reference_strain_rate: f64) -> Self {
        Self {
            d1,
            d2,
            d3,
            d4,
            d5,
            reference_strain_rate,
        }
    }

    /// AISI 4340 steel after Johnson and Cook (1985).
    pub fn c4340_steel() -> Self {
        Self::new(0.05, 3.44, -2.12, 0.002, 0.61, 1.0)
    }

    /// Fracture strain at triaxiality, rate, and homologous temperature.
    ///
    /// Takes stress triaxiality (dimensionless), equivalent plastic strain
    /// rate (1/s), and homologous temperature in `[0, 1]`; returns the
    /// plastic strain to fracture, floored at 1e-9 to keep accumulation
    /// finite under extreme compression.
    pub fn fracture_strain(
        &self,
        triaxiality: f64,
        strain_rate: f64,
        homologous_temperature: f64,
    ) -> f64 {
        let star = (strain_rate.max(1e-12) / self.reference_strain_rate).max(1e-9);
        let triax = self.d1 + self.d2 * (self.d3 * triaxiality).exp();
        let rate = 1.0 + self.d4 * star.ln();
        let temp = 1.0 + self.d5 * homologous_temperature.clamp(0.0, 1.0);
        (triax * rate * temp).max(1e-9)
    }

    /// Damage increment for a plastic strain step.
    ///
    /// Returns `dep / ef(triax, rate, T*)`, the Johnson-Cook accumulation
    /// rule; callers add it to the stored damage and erode at 1.
    pub fn increment(
        &self,
        triaxiality: f64,
        dep: f64,
        strain_rate: f64,
        homologous_temperature: f64,
    ) -> f64 {
        dep / self.fracture_strain(triaxiality, strain_rate, homologous_temperature)
    }
}

/// Void-nucleation parameters after Chu and Needleman.
///
/// Nucleation rate `f_dot = (f_n / (s_n sqrt(2 pi)))
/// exp(-((ep - e_n) / (s_n sqrt(2)))^2) ep_dot`: voids nucleate with a
/// Gaussian distribution around the nucleation strain `e_n`.
#[derive(Debug, Clone, Copy)]
pub struct VoidNucleation {
    /// Volume fraction of nucleating particles.
    pub particle_fraction: f64,
    /// Mean nucleation strain.
    pub mean_strain: f64,
    /// Standard deviation of the nucleation strain.
    pub std_strain: f64,
}

impl VoidNucleation {
    /// Nucleation rate per unit plastic strain at `ep`.
    pub fn rate(&self, ep: f64) -> f64 {
        let s = self.std_strain.max(1e-12);
        let z = (ep - self.mean_strain) / s;
        self.particle_fraction / (s * (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * z * z).exp()
    }
}

/// Simplified Gurson-Tvergaard-Needleman porous plasticity.
///
/// Yield function `Phi = (q/sy)^2 + 2 q1 f* cosh(1.5 q2 p/sy) -
/// (1 + q3 f*^2)` with the Tvergaard-Needleman effective porosity `f*`.
/// The stress update runs the J2 return mapping on the matrix material and
/// then pulls the trial stress back to the GTN surface by proportional
/// scaling, the standard explicit-code approximation: it is exact along
/// the trial direction and avoids a coupled `(p, q)` Newton solve.
/// Porosity evolves in `commit` through void growth plus optional
/// Chu-Needleman nucleation. Erode the point at `failure_porosity`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::fracture::ductile::{GursonMaterial, GursonState};
/// use sonium_fem::physics::solid::plasticity::{J2Material, StressUpdateModel};
///
/// let material = GursonMaterial::mild_steel();
/// let state = GursonState::new();
/// let d_eps = [0.01, -0.003, -0.003, 0.0, 0.0, 0.0];
/// let update = material.update(&state, &d_eps);
/// assert!(update.yielding);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GursonMaterial {
    /// Dense-matrix J2 material (elasticity plus matrix hardening).
    pub matrix: J2Material,
    /// Tvergaard q1 parameter (default 1.5).
    pub q1: f64,
    /// Tvergaard q2 parameter (default 1.0).
    pub q2: f64,
    /// Tvergaard q3 parameter (default q1^2).
    pub q3: f64,
    /// Initial void volume fraction.
    pub initial_porosity: f64,
    /// Porosity at void coalescence onset.
    pub critical_porosity: f64,
    /// Porosity at ductile failure (erosion trigger).
    pub failure_porosity: f64,
    /// Optional strain-controlled nucleation model.
    pub nucleation: Option<VoidNucleation>,
}

impl GursonMaterial {
    /// Creates a GTN material from a matrix and void parameters.
    pub fn new(
        matrix: J2Material,
        q1: f64,
        q2: f64,
        q3: f64,
        initial_porosity: f64,
        critical_porosity: f64,
        failure_porosity: f64,
    ) -> Self {
        Self {
            matrix,
            q1,
            q2,
            q3,
            initial_porosity,
            critical_porosity,
            failure_porosity,
            nucleation: None,
        }
    }

    /// Mild steel matrix with textbook void parameters.
    pub fn mild_steel() -> Self {
        Self::new(J2Material::mild_steel(), 1.5, 1.0, 2.25, 0.001, 0.05, 0.20)
    }

    /// Effective porosity after Tvergaard and Needleman.
    ///
    /// Returns `f` below coalescence and accelerates linearly toward
    /// `1/q1` (total loss of stress capacity) between `critical` and
    /// `failure` porosity.
    pub fn effective_porosity(&self, porosity: f64) -> f64 {
        if porosity <= self.critical_porosity {
            return porosity.max(0.0);
        }
        let ultimate = 1.0 / self.q1.max(1e-12);
        let span = (self.failure_porosity - self.critical_porosity).max(1e-12);
        let weight = ((porosity - self.critical_porosity) / span).min(1.0);
        self.critical_porosity + (ultimate - self.critical_porosity) * weight
    }

    /// GTN yield function value (negative inside, zero on the surface).
    ///
    /// Takes pressure `p` and von Mises stress `q` in Pa, matrix flow
    /// stress in Pa, and the current porosity.
    pub fn yield_function(&self, p: f64, q: f64, matrix_yield: f64, porosity: f64) -> f64 {
        let sy = matrix_yield.max(1e-9);
        let f_star = self.effective_porosity(porosity);
        let q_term = (q / sy) * (q / sy);
        let cosh = (1.5 * self.q2 * p / sy).cosh();
        q_term + 2.0 * self.q1 * f_star * cosh - (1.0 + self.q3 * f_star * f_star)
    }
}

/// History state of one GTN material point.
#[derive(Debug, Clone)]
pub struct GursonState {
    /// Matrix J2 state in legacy internal Voigt order.
    pub plastic: PlasticState,
    /// Void volume fraction.
    pub porosity: f64,
}

impl GursonState {
    /// Creates a virgin state with the given initial porosity.
    pub fn with_porosity(porosity: f64) -> Self {
        Self {
            plastic: PlasticState::new_3d(),
            porosity,
        }
    }

    /// Creates a virgin state with zero porosity.
    ///
    /// Prefer [`with_porosity`](Self::with_porosity) with the material
    /// `initial_porosity` for production runs.
    pub fn new() -> Self {
        Self::with_porosity(0.0)
    }
}

impl Default for GursonState {
    fn default() -> Self {
        Self::new()
    }
}

impl StressUpdateModel for GursonMaterial {
    type State = GursonState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        // Matrix J2 return first, then proportional pull-back to GTN.
        let matrix_update = self.matrix.update(&state.plastic, d_eps);
        let p = (matrix_update.stress[0] + matrix_update.stress[1] + matrix_update.stress[2]) / 3.0;
        let q = von_mises_c1(&matrix_update.stress);
        let sy = self
            .matrix
            .current_yield_stress(state.plastic.equiv_plastic_strain)
            .max(1e-9);
        let porosity = state.porosity.max(self.initial_porosity).max(0.0);
        if self.yield_function(p, q, sy, porosity) <= 0.0 {
            return matrix_update;
        }
        // Proportional scaling: solve Phi(s p, s q) = 0 for s in (0, 1]
        // by bisection; Phi is monotone along the ray for s > 0.
        let mut lo = 0.0;
        let mut hi = 1.0;
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if self.yield_function(mid * p, mid * q, sy, porosity) > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let s = 0.5 * (lo + hi);
        let scaled: [f64; 6] = std::array::from_fn(|i| matrix_update.stress[i] * s);
        let tangent: [f64; 36] = std::array::from_fn(|i| matrix_update.tangent[i] * s);
        StressUpdate {
            stress: scaled,
            tangent,
            delta_equiv_plastic_strain: matrix_update.delta_equiv_plastic_strain,
            yielding: true,
        }
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        self.matrix.commit(&mut state.plastic, update);
        if !update.yielding || update.delta_equiv_plastic_strain <= 0.0 {
            return;
        }
        // Void growth: (1 - f) tr(dep) with the GTN volumetric flow.
        let p = (update.stress[0] + update.stress[1] + update.stress[2]) / 3.0;
        let sy = self
            .matrix
            .current_yield_stress(state.plastic.equiv_plastic_strain)
            .max(1e-9);
        let f_star = self.effective_porosity(state.porosity);
        let volumetric = update.delta_equiv_plastic_strain
            * 2.0
            * self.q1
            * f_star
            * (1.5 * self.q2 / sy)
            * (1.5 * self.q2 * p / sy).sinh();
        // Signed growth lets hydrostatic compression close voids again;
        // the clamp below keeps the fraction inside [0, 1].
        state.porosity += (1.0 - state.porosity) * volumetric;
        // Nucleation adds fresh voids with the Gaussian kinetics.
        if let Some(nucleation) = self.nucleation {
            state.porosity += nucleation.rate(state.plastic.equiv_plastic_strain)
                * update.delta_equiv_plastic_strain;
        }
        state.porosity = state.porosity.clamp(0.0, 1.0);
    }
}

/// Element erosion criterion for ductile failure.
///
/// An element erodes when any enabled limit trips: accumulated plastic
/// strain, Johnson-Cook damage, GTN porosity, or tabulated/Lode failure
/// damage. Limits set to infinity (the default for damage, porosity, and
/// tabular damage) disable that leg.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::fracture::ductile::ErosionCriterion;
///
/// let criterion = ErosionCriterion::plastic_strain(0.5);
/// assert!(!criterion.eroded(0.4, 0.0, 0.0));
/// assert!(criterion.eroded(0.6, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ErosionCriterion {
    /// Erosion threshold on equivalent plastic strain.
    pub max_equiv_plastic_strain: f64,
    /// Erosion threshold on Johnson-Cook damage in `[0, 1]`.
    pub max_damage: f64,
    /// Erosion threshold on GTN porosity.
    pub max_porosity: f64,
    /// Erosion threshold on tabulated (`/FAIL/TAB1`) or Wierzbicki damage.
    pub max_tabular_damage: f64,
}

impl ErosionCriterion {
    /// Creates a criterion that erodes on plastic strain only.
    pub fn plastic_strain(limit: f64) -> Self {
        Self {
            max_equiv_plastic_strain: limit,
            max_damage: f64::INFINITY,
            max_porosity: f64::INFINITY,
            max_tabular_damage: f64::INFINITY,
        }
    }

    /// Creates a criterion that erodes on damage or porosity only.
    pub fn damage_driven(max_damage: f64, max_porosity: f64) -> Self {
        Self {
            max_equiv_plastic_strain: f64::INFINITY,
            max_damage,
            max_porosity,
            max_tabular_damage: f64::INFINITY,
        }
    }

    /// Creates a criterion that erodes on tabulated failure damage only.
    ///
    /// Trips when the `/FAIL/TAB1` accumulated damage or the Wierzbicki
    /// damage reaches `limit` (the card `Dcrit` or 1.0 respectively).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::fracture::ductile::ErosionCriterion;
    ///
    /// let criterion = ErosionCriterion::tabular(1.1);
    /// assert!(!criterion.eroded_with_tabular(0.0, 0.0, 0.0, 1.0));
    /// assert!(criterion.eroded_with_tabular(0.0, 0.0, 0.0, 1.2));
    /// ```
    pub fn tabular(limit: f64) -> Self {
        Self {
            max_equiv_plastic_strain: f64::INFINITY,
            max_damage: f64::INFINITY,
            max_porosity: f64::INFINITY,
            max_tabular_damage: limit,
        }
    }

    /// Tests one element against every enabled limit.
    ///
    /// Takes equivalent plastic strain, Johnson-Cook damage, and porosity;
    /// returns true when any enabled limit is met or exceeded.
    pub fn eroded(&self, equiv_plastic_strain: f64, damage: f64, porosity: f64) -> bool {
        equiv_plastic_strain >= self.max_equiv_plastic_strain
            || damage >= self.max_damage
            || porosity >= self.max_porosity
    }

    /// Tests one element including tabulated failure damage.
    ///
    /// Extends [`eroded`](Self::eroded) with the `/FAIL/TAB1` or
    /// Wierzbicki damage leg; existing three-tuple callers are unaffected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::fracture::ductile::ErosionCriterion;
    ///
    /// let criterion = ErosionCriterion::tabular(1.1);
    /// assert!(!criterion.eroded_with_tabular(10.0, 10.0, 10.0, 0.0));
    /// assert!(criterion.eroded_with_tabular(0.0, 0.0, 0.0, 1.1));
    /// ```
    pub fn eroded_with_tabular(
        &self,
        equiv_plastic_strain: f64,
        damage: f64,
        porosity: f64,
        tabular_damage: f64,
    ) -> bool {
        self.eroded(equiv_plastic_strain, damage, porosity)
            || tabular_damage >= self.max_tabular_damage
    }
}

/// Stable element deletion set for eroded elements.
///
/// Erosion never renumbers the mesh: deleted elements stay in place with
/// zero stiffness and zero stress while their nodes keep their global ids.
/// Force assembly skips deleted elements through [`is_active`](Self::is_active),
/// which keeps the sparsity pattern and the lumped mass map stable across
/// deletion events. Deletion is irreversible.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::fracture::ductile::ErosionSet;
///
/// let mut set = ErosionSet::new(4);
/// set.erode(1);
/// assert!(!set.is_active(1));
/// assert_eq!(set.active_count(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct ErosionSet {
    /// Per-element deletion flags, true once eroded.
    deleted: Vec<bool>,
}

impl ErosionSet {
    /// Creates a deletion set for `num_elements` live elements.
    pub fn new(num_elements: usize) -> Self {
        Self {
            deleted: vec![false; num_elements],
        }
    }

    /// Counts the elements.
    pub fn len(&self) -> usize {
        self.deleted.len()
    }

    /// Reports whether the set holds no elements.
    pub fn is_empty(&self) -> bool {
        self.deleted.is_empty()
    }

    /// Erodes one element; irreversible and idempotent.
    ///
    /// # Panics
    ///
    /// Panics when `element` is out of range.
    pub fn erode(&mut self, element: usize) {
        self.deleted[element] = true;
    }

    /// Erodes every element whose features trip the criterion.
    ///
    /// Takes the criterion plus one `(ep, damage, porosity)` tuple per
    /// element; returns the ids newly eroded by this sweep.
    ///
    /// # Panics
    ///
    /// Panics when `features` is shorter than the element count.
    pub fn sweep(
        &mut self,
        criterion: &ErosionCriterion,
        features: &[(f64, f64, f64)],
    ) -> Vec<usize> {
        assert!(
            features.len() >= self.deleted.len(),
            "erosion sweep needs one feature tuple per element"
        );
        let mut fresh = Vec::new();
        for (id, done) in self.deleted.iter_mut().enumerate() {
            if *done {
                continue;
            }
            let (ep, damage, porosity) = features[id];
            if criterion.eroded(ep, damage, porosity) {
                *done = true;
                fresh.push(id);
            }
        }
        fresh
    }

    /// Erodes elements tripping the tabulated leg too.
    ///
    /// Takes the criterion plus one `(ep, damage, porosity, tabular)`
    /// tuple per element; returns the ids newly eroded by this sweep.
    /// Existing three-tuple sweeps are unaffected.
    ///
    /// # Panics
    ///
    /// Panics when `features` is shorter than the element count.
    pub fn sweep_with_tabular(
        &mut self,
        criterion: &ErosionCriterion,
        features: &[(f64, f64, f64, f64)],
    ) -> Vec<usize> {
        assert!(
            features.len() >= self.deleted.len(),
            "erosion sweep needs one feature tuple per element"
        );
        let mut fresh = Vec::new();
        for (id, done) in self.deleted.iter_mut().enumerate() {
            if *done {
                continue;
            }
            let (ep, damage, porosity, tabular) = features[id];
            if criterion.eroded_with_tabular(ep, damage, porosity, tabular) {
                *done = true;
                fresh.push(id);
            }
        }
        fresh
    }

    /// Reports whether an element still carries load.
    ///
    /// # Panics
    ///
    /// Panics when `element` is out of range.
    pub fn is_active(&self, element: usize) -> bool {
        !self.deleted[element]
    }

    /// Counts the elements still carrying load.
    pub fn active_count(&self) -> usize {
        self.deleted.iter().filter(|done| !**done).count()
    }
}

/// Uniform necking strain of a Hollomon power-law bar.
///
/// Considere's criterion gives diffuse necking at `d sigma / d eps = sigma`,
/// which for `sigma = K eps^n` is exactly `eps = n`. The full necking bar
/// has no closed form; this value plus the Wolfram incremental reference
/// and the OpenCourant deck are the validation legs (see the necking QA).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::fracture::ductile::hollomon_necking_strain;
///
/// assert!((hollomon_necking_strain(0.2) - 0.2).abs() < 1e-15);
/// ```
pub fn hollomon_necking_strain(hardening_exponent: f64) -> f64 {
    hardening_exponent.max(0.0)
}

/// Bridgman correction factor for a necked round bar.
///
/// Returns the ratio of the axial true stress to the average axial stress
/// over the neck, `1 / ((1 + 2/x) ln(1 + x/2))` with `x = a / R` (neck
/// radius over neck curvature radius). Tends to 1 for a shallow neck.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::fracture::ductile::bridgman_correction;
///
/// assert!((bridgman_correction(1e-9) - 1.0).abs() < 1e-6);
/// assert!(bridgman_correction(1.0) < 1.0);
/// ```
pub fn bridgman_correction(a_over_r: f64) -> f64 {
    let x = a_over_r.max(0.0);
    if x < 1e-9 {
        return 1.0;
    }
    1.0 / ((1.0 + 2.0 / x) * (1.0 + x / 2.0).ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jc_fracture_strain_matches_closed_form() {
        let damage = JohnsonCookDamage::c4340_steel();
        let triax = 1.0f64 / 3.0;
        let expected = (0.05 + 3.44 * (-2.12 * triax).exp()) * (1.0 + 0.002 * (1.0f64).ln());
        let got = damage.fracture_strain(triax, 1.0, 0.0);
        assert!((got - expected).abs() < 1e-12, "got {got}, want {expected}");
    }

    #[test]
    fn jc_triaxiality_orders_ductility() {
        let damage = JohnsonCookDamage::c4340_steel();
        let shear = damage.fracture_strain(0.0, 1.0, 0.0);
        let tension = damage.fracture_strain(1.0 / 3.0, 1.0, 0.0);
        let biaxial = damage.fracture_strain(2.0 / 3.0, 1.0, 0.0);
        assert!(shear > tension && tension > biaxial);
    }

    #[test]
    fn gurson_reduces_strength_with_porosity() {
        let material = GursonMaterial::mild_steel();
        let sy = 250.0e6;
        // Dense matrix at yield sits on the surface.
        assert!(material.yield_function(0.0, sy, sy, 0.0).abs() < 1e-9);
        // Porosity pushes the same state outside the surface.
        assert!(material.yield_function(0.0, sy, sy, 0.05) > 0.0);
        // Hydrostatic tension alone yields a porous solid.
        assert!(material.yield_function(200.0e6, 0.0, sy, 0.10) > 0.0);
        // ... but moderate pressure stays elastic at the same porosity.
        assert!(material.yield_function(50.0e6, 0.0, sy, 0.10) < 0.0);
    }

    #[test]
    fn gurson_effective_porosity_accelerates_after_coalescence() {
        let material = GursonMaterial::mild_steel();
        assert!((material.effective_porosity(0.01) - 0.01).abs() < 1e-15);
        let mid = material.effective_porosity(0.125);
        assert!(mid > 0.125, "coalescence must accelerate, got {mid}");
        let failed = material.effective_porosity(0.20);
        assert!((failed - 1.0 / 1.5).abs() < 1e-12);
    }

    #[test]
    fn gurson_update_stays_on_surface() {
        let material = GursonMaterial::mild_steel();
        let state = GursonState::with_porosity(0.05);
        let d_eps = [0.01, -0.003, -0.003, 0.0, 0.0, 0.0];
        let update = material.update(&state, &d_eps);
        assert!(update.yielding);
        let p = (update.stress[0] + update.stress[1] + update.stress[2]) / 3.0;
        let q = von_mises_c1(&update.stress);
        let phi = material.yield_function(p, q, 250.0e6, 0.05);
        assert!(phi.abs() < 1e-9, "surface residual {phi}");
    }

    #[test]
    fn erosion_set_sweep_is_irreversible() {
        let criterion = ErosionCriterion::damage_driven(1.0, 0.2);
        let mut set = ErosionSet::new(3);
        let fresh = set.sweep(
            &criterion,
            &[(0.0, 0.5, 0.0), (0.0, 1.2, 0.0), (0.0, 0.0, 0.3)],
        );
        assert_eq!(fresh, vec![1, 2]);
        assert_eq!(set.active_count(), 1);
        // Healing the features must not resurrect elements.
        let fresh = set.sweep(&criterion, &[(0.0, 0.0, 0.0); 3]);
        assert!(fresh.is_empty());
        assert!(!set.is_active(1));
    }

    #[test]
    fn tabular_leg_trips_without_touching_legacy_legs() {
        let legacy = ErosionCriterion::damage_driven(1.0, 0.2);
        assert!(!legacy.eroded_with_tabular(0.0, 0.5, 0.0, 100.0));
        let tab = ErosionCriterion::tabular(1.1);
        assert!(!tab.eroded_with_tabular(0.0, 0.0, 0.0, 1.0));
        assert!(tab.eroded_with_tabular(0.0, 0.0, 0.0, 1.1));
        let mut set = ErosionSet::new(2);
        let fresh = set.sweep_with_tabular(&tab, &[(0.0, 0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.2)]);
        assert_eq!(fresh, vec![1]);
        assert_eq!(set.active_count(), 1);
    }

    #[test]
    fn bridgman_matches_closed_form() {
        let x = 1.0f64;
        let expected = 1.0 / ((1.0 + 2.0 / x) * (1.0 + x / 2.0).ln());
        assert!((bridgman_correction(x) - expected).abs() < 1e-15);
        // Deep notch correction is well below one.
        assert!(bridgman_correction(2.0) < 0.8);
    }
}
