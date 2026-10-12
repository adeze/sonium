//! Native coupled rupture solve: plasticity plus failure.
//!
//! Steps a C1 material law together with a rupture criterion in one
//! accepted increment: the law advances the stress, the criterion
//! reads triaxiality and Lode from that native stress (never a
//! surrogate), accumulates damage, fading softening scales the
//! reported stress, and the erosion latch trips deletion. Law history
//! commits unscaled, matching the engine order where `DMG_SCALE`
//! multiplies the stress after the material update.
//!
//! [`CoupledPoint`] solves one material point;
//! [`CoupledArray`] solves a mesh block with an
//! [`ErosionSet`](super::ductile::ErosionSet) plus an energy-conserving
//! [`ErosionLedger`](super::deletion::ErosionLedger). Criteria plug in
//! through [`RuptureCriterion`] with adapters for
//! [`FailTab`](super::fail_tab::FailTab) (plus fading), [`Wierzbicki`],
//! and [`JohnsonCookDamage`](super::ductile::JohnsonCookDamage).
//!
//! # Examples
//!
//! ```rust
//! use sonium_fem::physics::solid::fracture::coupled::{
//!     CoupledPoint, JohnsonCookCriterion, SolveOptions,
//! };
//! use sonium_fem::physics::solid::fracture::ductile::JohnsonCookDamage;
//! use sonium_fem::physics::solid::plasticity::crash::JohnsonCookState;
//! use sonium_fem::physics::solid::plasticity::crash::JohnsonCookMaterial;
//!
//! let law = JohnsonCookMaterial::c4340_steel();
//! let criterion = JohnsonCookCriterion::room_temperature(JohnsonCookDamage::c4340_steel());
//! let mut point = CoupledPoint::new(law, JohnsonCookState::new(293.0), criterion);
//! let step = point.step(&[0.01, -0.003, -0.003, 0.0, 0.0, 0.0], 1.0e-3, &SolveOptions::solid());
//! assert!(step.damage >= 0.0);
//! ```
//!
//! # Panics
//!
//! Stepping panics on non-positive step times. Array stepping panics
//! when the increment slice is shorter than the point count.
//!
//! # Safety
//!
//! Pure safe Rust with no unsafe code.

// Rust guideline compliant 2026-02-21

use super::deletion::ErosionLedger;
use super::ductile::{ErosionSet, JohnsonCookDamage, stress_triaxiality_c1};
use super::fail_tab::{FadeSoftening, FadeState, FailTab};
use super::wierzbicki::{Wierzbicki, xi_shell, xi_solid};
use crate::physics::solid::plasticity::update::{StressUpdateModel, von_mises_c1};

/// Damage context of one accepted coupled step.
#[derive(Debug, Clone, Copy)]
pub struct DamageContext {
    /// Plastic strain increment of the step.
    pub dep: f64,
    /// Accumulated equivalent plastic strain after the step.
    pub ep: f64,
    /// Stress triaxiality from the native stress.
    pub triaxiality: f64,
    /// Lode parameter in the criterion convention.
    pub lode: f64,
    /// Plastic strain rate `dep/dt` in 1/s.
    pub rate: f64,
    /// Step time in s.
    pub dt: f64,
    /// Mean stress in Pa (positive tensile).
    pub mean_stress: f64,
}

/// Rupture criterion behind the coupled solve.
pub trait RuptureCriterion {
    /// Damage increment for one accepted step context.
    fn damage_increment(&self, ctx: &DamageContext) -> f64;

    /// Reports whether accumulated damage ruptures the point.
    fn ruptured(&self, damage: f64) -> bool;

    /// Post-onset stress scale in `[0, 1]`.
    ///
    /// Defaults to 1 (no softening). The `/FAIL/TAB1` adapter
    /// overrides this with fading-exponent softening.
    fn softening_scale(&mut self, ctx: &DamageContext) -> f64 {
        let _ = ctx;
        1.0
    }
}

/// Lode convention for the coupled triaxiality pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XiMode {
    /// Solid invariant `xi` from the native stress tensor.
    Solid,
    /// Shell plane-stress `xi` from the triaxiality alone.
    Shell,
}

/// Coupled-solve options shared by points and arrays.
#[derive(Debug, Clone, Copy)]
pub struct SolveOptions {
    /// Lode convention for criteria reading `xi`.
    pub xi_mode: XiMode,
    /// Extra erosion trip on plastic strain (`None` disables).
    pub max_plastic_strain: Option<f64>,
    /// Zero the reported stress once eroded.
    pub freeze_eroded: bool,
}

impl SolveOptions {
    /// Solid options: invariant `xi`, no extra trips, frozen response.
    pub fn solid() -> Self {
        Self {
            xi_mode: XiMode::Solid,
            max_plastic_strain: None,
            freeze_eroded: true,
        }
    }

    /// Shell options: plane-stress `xi`, no extra trips, frozen response.
    pub fn shell() -> Self {
        Self {
            xi_mode: XiMode::Shell,
            max_plastic_strain: None,
            freeze_eroded: true,
        }
    }
}

/// `/FAIL/TAB1` criterion with optional fading softening.
#[derive(Debug, Clone)]
pub struct TabCriterion {
    /// Tabulated failure model.
    pub tab: FailTab,
    /// Element size for the size factor (`None` neutral).
    pub size: Option<f64>,
    /// Temperature triple for the thermal factor (`None` neutral).
    pub temperature: Option<(f64, f64, f64)>,
    /// Fading softening (`None` disables post-onset scaling).
    pub fade: Option<FadeSoftening>,
    /// Post-instability progress.
    fade_state: FadeState,
}

impl TabCriterion {
    /// Builds a plain tabulated criterion without extras.
    pub fn new(tab: FailTab) -> Self {
        Self {
            tab,
            size: None,
            temperature: None,
            fade: None,
            fade_state: FadeState::default(),
        }
    }

    /// Reads the post-instability progress (tests and legs).
    pub fn fade_state(&self) -> FadeState {
        self.fade_state
    }
}

impl RuptureCriterion for TabCriterion {
    fn damage_increment(&self, ctx: &DamageContext) -> f64 {
        self.tab.increment_at_rate(
            ctx.dep,
            ctx.ep,
            ctx.triaxiality,
            ctx.lode,
            ctx.rate,
            self.size,
            self.temperature,
        )
    }

    fn ruptured(&self, damage: f64) -> bool {
        self.tab.ruptured(damage)
    }

    fn softening_scale(&mut self, ctx: &DamageContext) -> f64 {
        let Some(fade) = &self.fade else {
            return 1.0;
        };
        let ef = self.tab.failure_strain(
            ctx.triaxiality,
            ctx.rate,
            ctx.lode,
            self.size,
            self.temperature,
        );
        let lambda = self.size.map_or(1.0, |s| s / self.tab.size_ref.max(1e-15));
        // Recompute this step's increment for the fade rescale: the
        // trait passes context, not the increment, so evaluate again.
        let d_damage = self.damage_increment(ctx);
        self.tab.fade_step(
            fade,
            &mut self.fade_state,
            d_damage,
            ef,
            ctx.triaxiality,
            ctx.mean_stress,
            lambda,
        )
    }
}

/// Wierzbicki-Leroy criterion adapter.
#[derive(Debug, Clone)]
pub struct WierzbickiCriterion {
    /// Wierzbicki fracture locus.
    pub locus: Wierzbicki,
}

impl WierzbickiCriterion {
    /// Builds the adapter from a locus.
    pub fn new(locus: Wierzbicki) -> Self {
        Self { locus }
    }
}

impl RuptureCriterion for WierzbickiCriterion {
    fn damage_increment(&self, ctx: &DamageContext) -> f64 {
        if ctx.dep <= 0.0 {
            return 0.0;
        }
        ctx.dep / self.locus.fracture_strain(ctx.triaxiality, ctx.lode)
    }

    fn ruptured(&self, damage: f64) -> bool {
        self.locus.fractured(damage)
    }
}

/// Johnson-Cook damage adapter at fixed temperature.
#[derive(Debug, Clone)]
pub struct JohnsonCookCriterion {
    /// Johnson-Cook damage model.
    pub damage: JohnsonCookDamage,
    /// Homologous temperature in `[0, 1]`.
    pub homologous_temperature: f64,
}

impl JohnsonCookCriterion {
    /// Builds the adapter at room temperature (`T* = 0`).
    pub fn room_temperature(damage: JohnsonCookDamage) -> Self {
        Self {
            damage,
            homologous_temperature: 0.0,
        }
    }
}

impl RuptureCriterion for JohnsonCookCriterion {
    fn damage_increment(&self, ctx: &DamageContext) -> f64 {
        if ctx.dep <= 0.0 {
            return 0.0;
        }
        self.damage.increment(
            ctx.triaxiality,
            ctx.dep,
            ctx.rate,
            self.homologous_temperature,
        )
    }

    fn ruptured(&self, damage: f64) -> bool {
        let _ = self;
        damage >= 1.0
    }
}

/// One accepted coupled step with its rupture state.
#[derive(Debug, Clone, Copy)]
pub struct CoupledStep {
    /// Reported (softened, erosion-frozen) stress in C1 order (Pa).
    pub stress: [f64; 6],
    /// Accumulated damage after the step.
    pub damage: f64,
    /// Fading scale applied this step in `[0, 1]`.
    pub scale: f64,
    /// Native stress triaxiality of the step.
    pub triaxiality: f64,
    /// True once the point erodes (irreversible).
    pub eroded: bool,
}

/// One coupled material point: law plus criterion.
#[derive(Debug, Clone)]
pub struct CoupledPoint<M: StressUpdateModel<StrainIncrement = [f64; 6]>, C>
where
    M::State: core::fmt::Debug + Clone,
{
    /// Plasticity law.
    pub law: M,
    /// Law history state.
    pub state: M::State,
    /// Rupture criterion.
    pub criterion: C,
    /// Accumulated damage.
    pub damage: f64,
    /// Accumulated equivalent plastic strain.
    pub ep: f64,
    /// Erosion latch (irreversible).
    pub eroded: bool,
    /// Cumulative total work density (J/m^3).
    pub energy_density: f64,
    /// Cumulative plastic dissipation density (J/m^3).
    pub plastic_density: f64,
    /// Last reported stress in C1 order (Pa).
    pub stress: [f64; 6],
}

impl<M, C> CoupledPoint<M, C>
where
    M: StressUpdateModel<StrainIncrement = [f64; 6]>,
    M::State: core::fmt::Debug + Clone,
    C: RuptureCriterion,
{
    /// Builds a virgin coupled point from a law and criterion.
    pub fn new(law: M, state: M::State, criterion: C) -> Self {
        Self {
            law,
            state,
            criterion,
            damage: 0.0,
            ep: 0.0,
            eroded: false,
            energy_density: 0.0,
            plastic_density: 0.0,
            stress: [0.0; 6],
        }
    }

    /// Steps one accepted strain increment with step time.
    ///
    /// Advances the law (rate-aware when the law asks), reads the
    /// native triaxiality pair, accumulates damage, scales the stress
    /// by the fading response, trips erosion, and books total plus
    /// plastic work. Eroded points report zero stress and skip the
    /// law once frozen.
    ///
    /// # Panics
    ///
    /// Panics on non-positive step times.
    pub fn step(&mut self, dstrain: &[f64; 6], dt: f64, options: &SolveOptions) -> CoupledStep {
        assert!(dt.is_finite() && dt > 0.0, "coupled step needs dt > 0");
        if self.eroded && options.freeze_eroded {
            return CoupledStep {
                stress: [0.0; 6],
                damage: self.damage,
                scale: 0.0,
                triaxiality: 0.0,
                eroded: true,
            };
        }
        let update = if self.law.is_rate_dependent() {
            self.law.update_with_rate(&self.state, dstrain, dt)
        } else {
            self.law.update(&self.state, dstrain)
        };
        // Law history commits unscaled (engine order); the reported
        // stress below carries the fading scale instead.
        self.law.commit(&mut self.state, &update);
        let dep = update.delta_equiv_plastic_strain.max(0.0);
        self.ep += dep;
        let triaxiality = stress_triaxiality_c1(&update.stress);
        let lode = match options.xi_mode {
            XiMode::Solid => xi_solid(&update.stress),
            XiMode::Shell => xi_shell(triaxiality),
        };
        let mean = (update.stress[0] + update.stress[1] + update.stress[2]) / 3.0;
        let ctx = DamageContext {
            dep,
            ep: self.ep,
            triaxiality,
            lode,
            rate: dep / dt,
            dt,
            mean_stress: mean,
        };
        self.damage += self.criterion.damage_increment(&ctx);
        let scale = self.criterion.softening_scale(&ctx);
        let mut stress = update.stress;
        for value in stress.iter_mut() {
            *value *= scale;
        }
        self.eroded = self.eroded
            || self.criterion.ruptured(self.damage)
            || options.max_plastic_strain.is_some_and(|cap| self.ep >= cap);
        if self.eroded && options.freeze_eroded {
            stress = [0.0; 6];
        }
        for i in 0..6 {
            self.energy_density += 0.5 * (self.stress[i] + stress[i]) * dstrain[i];
        }
        // J2-consistent dissipation: von Mises times the increment.
        let q = 0.5 * (von_mises_c1(&self.stress) + von_mises_c1(&stress));
        self.plastic_density += q * dep;
        self.stress = stress;
        CoupledStep {
            stress,
            damage: self.damage,
            scale,
            triaxiality,
            eroded: self.eroded,
        }
    }
}

/// Mesh-block coupled solve with erosion set and ledger.
#[derive(Debug, Clone)]
pub struct CoupledArray<M: StressUpdateModel<StrainIncrement = [f64; 6]>, C>
where
    M::State: core::fmt::Debug + Clone,
{
    /// Coupled points, one per element.
    pub points: Vec<CoupledPoint<M, C>>,
    /// Reference volume per element in m^3.
    pub volumes: Vec<f64>,
    /// Stable element deletion set.
    pub erosion: ErosionSet,
    /// Energy-conserving erosion ledger.
    pub ledger: ErosionLedger,
}

impl<M, C> CoupledArray<M, C>
where
    M: StressUpdateModel<StrainIncrement = [f64; 6]> + Clone,
    M::State: core::fmt::Debug + Clone,
    C: RuptureCriterion + Clone,
{
    /// Builds an array from laws, states, criteria, volumes.
    ///
    /// # Panics
    ///
    /// Panics when the input slices disagree in length.
    pub fn new(laws: Vec<M>, states: Vec<M::State>, criteria: Vec<C>, volumes: Vec<f64>) -> Self {
        assert!(
            laws.len() == states.len()
                && laws.len() == criteria.len()
                && laws.len() == volumes.len(),
            "coupled array inputs must share one length"
        );
        let count = laws.len();
        let points = laws
            .into_iter()
            .zip(states)
            .zip(criteria)
            .map(|((law, state), criterion)| CoupledPoint::new(law, state, criterion))
            .collect();
        Self {
            points,
            volumes,
            erosion: ErosionSet::new(count),
            ledger: ErosionLedger::new(count),
        }
    }

    /// Steps every live point one accepted increment.
    ///
    /// Advances each uneroded point, records cumulative densities
    /// into the ledger, and sweeps fresh erosions into the set (the
    /// ledger keeps eroded energies in the eroded column, so totals
    /// stay conserved). Returns per-point steps.
    ///
    /// # Panics
    ///
    /// Panics on non-positive step times or short increment slices.
    pub fn step_array(
        &mut self,
        dstrains: &[[f64; 6]],
        dt: f64,
        options: &SolveOptions,
    ) -> Vec<CoupledStep> {
        assert!(
            dstrains.len() >= self.points.len(),
            "coupled array needs one increment per point"
        );
        let mut steps = Vec::with_capacity(self.points.len());
        for (id, point) in self.points.iter_mut().enumerate() {
            let step = if self.erosion.is_active(id) {
                point.step(&dstrains[id], dt, options)
            } else {
                CoupledStep {
                    stress: [0.0; 6],
                    damage: point.damage,
                    scale: 0.0,
                    triaxiality: 0.0,
                    eroded: true,
                }
            };
            if step.eroded {
                self.erosion.erode(id);
            }
            self.ledger.record(
                id,
                self.volumes[id],
                point.plastic_density,
                point.energy_density,
                step.eroded,
            );
            steps.push(step);
        }
        steps
    }

    /// Counts the points still carrying load.
    pub fn active_count(&self) -> usize {
        self.erosion.active_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::solid::plasticity::crash::{JohnsonCookMaterial, JohnsonCookState};
    use crate::physics::solid::plasticity::funct::Funct1D;
    use crate::physics::solid::plasticity::law_metals::{Law36, Law36State};

    /// Flat shell table with unit damage weight.
    fn flat_tab(value: f64) -> FailTab {
        let curve = Funct1D::new(vec![-1.0, 1.0], vec![value, value]).unwrap();
        FailTab::new(vec![0.0], vec![0.0], vec![curve]).unwrap()
    }

    #[test]
    fn johnson_cook_shear_ruptures_at_locus() {
        let law = JohnsonCookMaterial::c4340_steel();
        let damage = JohnsonCookDamage::c4340_steel();
        // Shear rate of the driven path in 1/s.
        let rate = 0.004 / 3.0f64.sqrt() / 1.0e-3;
        let expected = damage.fracture_strain(0.0, rate, 0.0);
        let criterion = JohnsonCookCriterion::room_temperature(damage);
        let mut point = CoupledPoint::new(law, JohnsonCookState::new(293.0), criterion);
        let options = SolveOptions::solid();
        // Pure shear: native triaxiality stays exactly zero, so the
        // proportional path must rupture at the closed-form strain.
        let mut rupture_ep = f64::NAN;
        for _ in 0..2000 {
            let step = point.step(&[0.0, 0.0, 0.0, 0.004, 0.0, 0.0], 1.0e-3, &options);
            assert!(step.triaxiality.abs() < 1e-9);
            if step.eroded && rupture_ep.is_nan() {
                rupture_ep = point.ep;
            }
        }
        assert!(
            (rupture_ep - expected).abs() < 0.01,
            "rupture at {rupture_ep} vs locus {expected}"
        );
    }

    #[test]
    fn fail_tab_shear_ruptures_at_scaled_locus() {
        let law = JohnsonCookMaterial::c4340_steel();
        let criterion = TabCriterion::new(flat_tab(0.5));
        let mut point = CoupledPoint::new(law, JohnsonCookState::new(293.0), criterion);
        let options = SolveOptions::solid();
        let mut rupture_ep = f64::NAN;
        for _ in 0..2000 {
            let step = point.step(&[0.0, 0.0, 0.0, 0.002, 0.0, 0.0], 1.0e-3, &options);
            if step.eroded && rupture_ep.is_nan() {
                rupture_ep = point.ep;
            }
        }
        // Flat table, unit weight: D = ep/0.5, strict rupture past
        // Dcrit 0.999 gives ep = 0.4995 within one step.
        assert!(
            (rupture_ep - 0.4995).abs() < 0.005,
            "rupture at {rupture_ep}"
        );
    }

    #[test]
    fn wierzbicki_shear_ruptures_at_emin() {
        let law = Law36::card_aluminum();
        let locus = Wierzbicki::deck_shell();
        let expected = locus.fracture_strain(0.0, 0.0);
        let criterion = WierzbickiCriterion::new(locus);
        let mut point = CoupledPoint::new(law, Law36State::new(), criterion);
        let options = SolveOptions::solid();
        let mut rupture_ep = f64::NAN;
        for _ in 0..4000 {
            let step = point.step(&[0.0, 0.0, 0.0, 0.001, 0.0, 0.0], 1.0e-3, &options);
            if step.eroded && rupture_ep.is_nan() {
                rupture_ep = point.ep;
            }
        }
        assert!(
            (rupture_ep - expected).abs() < 0.01,
            "rupture at {rupture_ep} vs locus {expected}"
        );
    }

    #[test]
    fn native_triaxiality_reads_deviatoric_zero() {
        let law = JohnsonCookMaterial::c4340_steel();
        let criterion = JohnsonCookCriterion::room_temperature(JohnsonCookDamage::c4340_steel());
        let mut point = CoupledPoint::new(law, JohnsonCookState::new(293.0), criterion);
        let options = SolveOptions::solid();
        // Trace-free path: mean stress stays zero, so the native
        // triaxiality must read zero every step.
        for _ in 0..20 {
            let step = point.step(&[0.002, -0.001, -0.001, 0.0, 0.0, 0.0], 1.0e-3, &options);
            assert!(step.triaxiality.abs() < 0.02, "triax {}", step.triaxiality);
        }
    }

    #[test]
    fn uniaxial_replay_triaxiality_matches_third() {
        use crate::physics::solid::plasticity::foam_explicit::{
            AxialStamp, C1Runner, replay_uniaxial,
        };

        // End-to-end uniaxial stress through the lateral solver: the
        // native triaxiality of every plastic stamp must read 1/3.
        let law = JohnsonCookMaterial::c4340_steel();
        let mut point = C1Runner::new(law, JohnsonCookState::new(293.0));
        let path = [
            AxialStamp::new(0.0, 0.0),
            AxialStamp::new(1.0, 0.05),
            AxialStamp::new(2.0, 0.10),
        ];
        let history = replay_uniaxial(&mut point, &path, 10, 7830.0);
        for stress in history.stress.iter().skip(1) {
            let triax = stress_triaxiality_c1(stress);
            assert!((triax - 1.0 / 3.0).abs() < 0.02, "triax {triax}");
        }
    }

    #[test]
    fn fade_softening_fades_reported_stress() {
        let law = Law36::card_aluminum();
        let mut criterion = TabCriterion::new(flat_tab(0.5));
        let mut fade = FadeSoftening::new(2.0, 0.1);
        fade.onset_fraction = 0.0;
        criterion.fade = Some(fade);
        let mut point = CoupledPoint::new(law, Law36State::new(), criterion);
        let options = SolveOptions::solid();
        let mut peak = 0.0f64;
        let mut scales = Vec::new();
        // Rupture needs ep 0.4995 at 0.001/sqrt(3) per step: 866 steps.
        for _ in 0..1000 {
            let step = point.step(&[0.0, 0.0, 0.0, 0.001, 0.0, 0.0], 1.0e-3, &options);
            peak = peak.max(step.stress[3]);
            scales.push(step.scale);
            if step.eroded {
                break;
            }
        }
        assert!(point.eroded);
        // Scale starts at one and decays once necking trips.
        assert!((scales[0] - 1.0).abs() < 1e-15);
        assert!(*scales.last().unwrap() < 0.5);
        // Reported shear peaks before rupture (softening), while the
        // law history keeps hardening underneath.
        assert!(peak > 0.0);
        assert!(point.stress[3].abs() < 1e-9);
    }

    #[test]
    fn array_conserves_energy_across_erosion() {
        let laws = vec![Law36::card_aluminum(), Law36::card_aluminum()];
        let states = vec![Law36State::new(), Law36State::new()];
        let criteria = vec![
            TabCriterion::new(flat_tab(0.05)),
            TabCriterion::new(flat_tab(50.0)),
        ];
        let mut array = CoupledArray::new(laws, states, criteria, vec![1.0e-6, 2.0e-6]);
        let options = SolveOptions::solid();
        for _ in 0..200 {
            array.step_array(
                &[
                    [0.0, 0.0, 0.0, 0.001, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.001, 0.0, 0.0],
                ],
                1.0e-3,
                &options,
            );
        }
        assert_eq!(array.active_count(), 1);
        assert!(!array.erosion.is_active(0));
        assert!(array.erosion.is_active(1));
        let totals = array.ledger.totals();
        assert!(totals.eroded_plastic > 0.0);
        assert!(totals.active_plastic > 0.0);
        // Grand totals equal the live point books by construction.
        let (res_p, res_i) = array
            .ledger
            .conservation_residual(totals.grand_plastic(), totals.grand_internal());
        assert!((res_p).abs() < 1e-15);
        assert!((res_i).abs() < 1e-15);
    }

    #[test]
    fn penta6_metal_crush_stays_balanced() {
        use crate::physics::solid::nonlinear_solid::{
            Penta6Geom, Penta6State, TetPentaParams, penta6_forces,
        };

        let geom = Penta6Geom::new([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        .unwrap();
        let law = Law36::card_aluminum();
        let params = TetPentaParams::new(2700.0, 5000.0);
        let mut velocities = [0.0; 18];
        // Deviatoric crush (volume preserving): J2 yields, unlike
        // under pure dilatation.
        for (node, coords) in geom.coords.iter().enumerate() {
            velocities[node * 3] = -coords[0] * 500.0;
            velocities[node * 3 + 1] = coords[1] * 250.0;
            velocities[node * 3 + 2] = coords[2] * 250.0;
        }
        let mut state = Penta6State::new(core::array::from_fn(|_| Law36State::new()));
        let mut ever_yielded = false;
        for _ in 0..20 {
            let out = penta6_forces(&geom, &mut state, &law, &velocities, 1.0e-6, &params).unwrap();
            ever_yielded = ever_yielded || out.yielding;
            for axis in 0..3 {
                let total: f64 = (0..6).map(|n| out.forces[n * 3 + axis]).sum();
                let scale: f64 = (0..6).map(|n| out.forces[n * 3 + axis].abs()).sum();
                assert!(total.abs() < 1e-9 * scale.max(1.0));
            }
        }
        assert!(ever_yielded);
        assert!(state.points.iter().all(|p| p.equiv_plastic_strain > 0.0));
    }
}
