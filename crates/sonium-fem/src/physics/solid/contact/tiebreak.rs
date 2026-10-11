//! Bonded/failed tracking and rupture criteria for tiebreak interfaces.
//!
//! Lane 2D consumes the frozen [`PenaltyPair`](super::candidates::PenaltyPair)
//! handoff (P2C2): gap sign, master-to-slave normal, and slave-side force
//! keep the producer (2B) meanings, and tests construct pairs literally.
//! This module owns the bonded/failed state ([`TiebreakState`]), two
//! rupture criteria ([`TiebreakCriterion`]), and deterministic rupture
//! sequencing over `&[PenaltyPair]`.
//!
//! ## Criteria and TYPE-equivalent semantics
//!
//! - `Force`: quadratic tension/shear interaction
//!   `(Ft/Nr)^2 + (Fs/Sr)^2 >= 1`, the `/INTER/TYPE2` spotweld rupture
//!   form (tension-only normal; compression never ruptures). The engine
//!   compares with strict `>`; this implementation trips at `>= 1` so the
//!   boundary ruptures exactly and deterministically.
//! - `Energy`: linear-elastic available release `G = tn^2/2Kn + ts^2/2Ks`
//!   against the Benzeggagh-Kenane mixed-mode toughness of the reused
//!   [`MixedModeCohesiveLaw`](crate::physics::solid::fracture::cohesive::MixedModeCohesiveLaw),
//!   i.e. a cohesive (energy) tiebreak. Only the law stiffnesses, fracture
//!   energies, and BK exponent feed the criterion; strengths do not enter.
//!
//! ## Determinism
//!
//! Both criteria are pure functions of one pair. [`TiebreakState::rupture`]
//! scans the input slice in order and reports newly ruptured pairs as input
//! indices in ascending order, so identical input order yields identical
//! rupture order. Tie-breaking rule: position wins — simultaneous trips
//! order by input position, never by utilization magnitude or slave id.
//! Failed slaves are keyed by slave node id and drop out permanently.
//!
//! ## Validation status
//!
//! The deterministic criteria are unit-proven here (boundary exactness,
//! irreversibility, sequencing). Rupture sequencing is engine-golden-only
//! per the W-tier backlog: history-coupled with no closed form, so no
//! `.wls` items and no criterion benches ship in Phase 2. Phase 3+
//! validation targets the miniqa `RUPTURE/FAIL_TAB/ELEM_SAMP` and
//! `RUPTURE/Wierzbcki/User_law/shell` cases plus rail-crush-class decks
//! (crash-fold family: `CRASH/crashtriangles`,
//! `CRASH/crashsection/box1_42`).
//!
//! ## Relation to the other weld paths
//!
//! - [`super::explicit::Spotweld`] evolves cohesive damage from kinematic
//!   openings for registered welds (progressive softening plus traction).
//! - [`zero_d::Spotweld`](crate::physics::solid::joints::zero_d::Spotweld)
//!   is the quasi-static stiff-bushing nugget with a post-processing
//!   strength check.
//! - This module filters force-carrying penalty pairs: it reports rupture
//!   indices and drops failed slaves from [`TiebreakState::bonded_pairs`];
//!   the 2B producer owns zeroing and releasing the forces.
//!
//! ## Future work
//!
//! Progressive pre-rupture softening inside the pair filter, moment and
//! rotational rupture components (pairs carry force only), rate, thermal,
//! and fatigue effects, and `/INTER/TYPE2` kinematic release coupling stay
//! future.

// Rust guideline compliant 2026-02-21

use super::candidates::PenaltyPair;
use crate::physics::solid::fracture::cohesive::MixedModeCohesiveLaw;
use std::collections::BTreeSet;

/// Floor for strictly positive denominators.
///
/// Keeps utilization finite when a caller passes a zero strength,
/// stiffness, area, or toughness: a zero rupture threshold then trips on
/// any nonzero load instead of producing NaN. Small enough to never shift
/// a physical threshold.
const POSITIVE_FLOOR: f64 = 1e-30;

/// Surrogate shear/normal mixity for exactly zero tension.
///
/// [`MixedModeCohesiveLaw::mixed_fracture_energy`] tends to the shear
/// toughness as `beta -> infinity`; `1e15` saturates the Benzeggagh-Kenane
/// blend to `Gs` exactly in `f64` arithmetic while keeping `beta^2 = 1e30`
/// far from overflow.
const PURE_SHEAR_MIXITY: f64 = 1e15;

/// Rupture criterion for one bonded pair.
///
/// Both variants map a [`PenaltyPair`](super::candidates::PenaltyPair) to a
/// dimensionless utilization through [`TiebreakCriterion::utilization`];
/// the pair ruptures when utilization reaches one. Evaluation is a pure
/// function of the pair: no history, no allocation.
#[derive(Debug, Clone, Copy)]
pub enum TiebreakCriterion {
    /// Quadratic tension/shear interaction (`/INTER/TYPE2` spotweld form).
    ///
    /// Utilization is `(Ft/Nr)^2 + (Fs/Sr)^2` with tension-only normal
    /// force `Ft` and shear magnitude `Fs` from the frozen pair split.
    Force {
        /// Tensile rupture force in N.
        normal_rupture: f64,
        /// Shear rupture force in N.
        shear_rupture: f64,
    },
    /// Cohesive energy release against BK mixed-mode toughness.
    ///
    /// Utilization is the available release `G = tn^2/2Kn + ts^2/2Ks`
    /// from the pair tractions over the tributary area, divided by the
    /// mixed-mode toughness at the elastic shear/normal mixity. Reuses
    /// the law stiffnesses, fracture energies, and BK exponent.
    Energy {
        /// Mixed-mode cohesive law supplying stiffnesses and toughness.
        law: MixedModeCohesiveLaw,
        /// Tributary area in m^2 converting pair force to traction.
        area: f64,
    },
}

impl TiebreakCriterion {
    /// Rupture utilization of one pair, rupturing at one.
    ///
    /// Returns the dimensionless rupture metric: force interaction ratio
    /// for [`TiebreakCriterion::Force`], available-over-critical energy
    /// release for [`TiebreakCriterion::Energy`]. Compression carries no
    /// normal load in either variant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::contact::candidates::PenaltyPair;
    /// use sonium_fem::physics::solid::contact::tiebreak::TiebreakCriterion;
    ///
    /// let criterion = TiebreakCriterion::Force {
    ///     normal_rupture: 1000.0,
    ///     shear_rupture: 1000.0,
    /// };
    /// // Tension pulls the slave back toward the master (-normal);
    /// // contact push-out (+normal) never ruptures.
    /// let pair = PenaltyPair {
    ///     slave: 0,
    ///     gap: 1.0e-6,
    ///     normal: [0.0, 0.0, 1.0],
    ///     force: [0.0, 0.0, -1000.0],
    /// };
    /// assert_eq!(criterion.utilization(&pair), 1.0);
    /// assert!(criterion.tripped(&pair));
    /// ```
    #[must_use]
    pub fn utilization(&self, pair: &PenaltyPair) -> f64 {
        match *self {
            Self::Force {
                normal_rupture,
                shear_rupture,
            } => {
                let (tensile, shear) = tensile_shear_forces(pair);
                let normal = tensile / normal_rupture.max(POSITIVE_FLOOR);
                let shear = shear / shear_rupture.max(POSITIVE_FLOOR);
                normal * normal + shear * shear
            }
            Self::Energy { law, area } => {
                let (tensile_force, shear_force) = tensile_shear_forces(pair);
                let area = area.max(POSITIVE_FLOOR);
                let normal_traction = tensile_force / area;
                let shear_traction = shear_force / area;
                let normal_stiffness = law.normal.penalty_stiffness.max(POSITIVE_FLOOR);
                let shear_stiffness = law.shear.penalty_stiffness.max(POSITIVE_FLOOR);
                let normal_opening = normal_traction / normal_stiffness;
                let shear_opening = shear_traction / shear_stiffness;
                // Elastic mixity from the inferred openings; exact-zero
                // tension takes the pure-shear limit without dividing.
                let mixity = if normal_opening > 0.0 {
                    shear_opening / normal_opening
                } else if shear_opening > 0.0 {
                    PURE_SHEAR_MIXITY
                } else {
                    0.0
                };
                let available =
                    0.5 * (normal_traction * normal_opening + shear_traction * shear_opening);
                let critical = law.mixed_fracture_energy(mixity).max(POSITIVE_FLOOR);
                available / critical
            }
        }
    }

    /// Reports whether one pair trips the criterion.
    ///
    /// Returns true exactly when [`TiebreakCriterion::utilization`] reaches
    /// one, so the boundary ruptures rather than surviving by rounding.
    #[must_use]
    pub fn tripped(&self, pair: &PenaltyPair) -> bool {
        self.utilization(pair) >= 1.0
    }
}

/// Tensile normal force and shear magnitude of one pair.
///
/// Splits the frozen slave-side force through the pair normal: tension is
/// the compressive [`PenaltyPair::normal_force`] negated and clamped at
/// zero (compression never drives rupture); shear is the magnitude of
/// [`PenaltyPair::tangential_force`]. Both in newtons.
fn tensile_shear_forces(pair: &PenaltyPair) -> (f64, f64) {
    let tensile = (-pair.normal_force()).max(0.0);
    let shear = pair.tangential_force();
    let shear_magnitude = (shear[0] * shear[0] + shear[1] * shear[1] + shear[2] * shear[2]).sqrt();
    (tensile, shear_magnitude)
}

/// Bonded/failed tracking plus rupture sequencing for tiebreak pairs.
///
/// Holds one [`TiebreakCriterion`] and the set of ruptured slave node ids.
/// [`TiebreakState::rupture`] evaluates a penalty-pair slice in order and
/// records newly ruptured slaves permanently; [`TiebreakState::bonded_pairs`]
/// drops failed slaves from a slice for the downstream force scatter.
#[derive(Debug, Clone)]
pub struct TiebreakState {
    criterion: TiebreakCriterion,
    failed: BTreeSet<usize>,
}

impl TiebreakState {
    /// Creates bonded tracking for one rupture criterion.
    ///
    /// Starts with no failed slaves; every pair evaluates as bonded until
    /// [`TiebreakState::rupture`] trips it.
    pub fn new(criterion: TiebreakCriterion) -> Self {
        Self {
            criterion,
            failed: BTreeSet::new(),
        }
    }

    /// Borrows the rupture criterion by value.
    pub fn criterion(&self) -> TiebreakCriterion {
        self.criterion
    }

    /// Reports whether one slave node already ruptured.
    pub fn is_failed(&self, slave: usize) -> bool {
        self.failed.contains(&slave)
    }

    /// Counts the ruptured slave nodes.
    pub fn num_failed(&self) -> usize {
        self.failed.len()
    }

    /// Lists ruptured slave ids in ascending order.
    ///
    /// Deterministic snapshot for diagnostics and golden comparisons; the
    /// ordering is by slave id, independent of rupture order.
    pub fn failed_slaves(&self) -> Vec<usize> {
        self.failed.iter().copied().collect()
    }

    /// Sequences rupture over one penalty-pair slice.
    ///
    /// Scans `pairs` in input order, skips already-failed slaves, trips the
    /// criterion on the rest, and records newly ruptured slaves permanently.
    /// Returns the input indices of the pairs that ruptured on this call in
    /// ascending order, so identical input order yields identical rupture
    /// order. Simultaneous trips order by input position, never by
    /// utilization magnitude or slave id.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::contact::candidates::PenaltyPair;
    /// use sonium_fem::physics::solid::contact::tiebreak::{TiebreakCriterion, TiebreakState};
    ///
    /// let mut state = TiebreakState::new(TiebreakCriterion::Force {
    ///     normal_rupture: 1000.0,
    ///     shear_rupture: 1000.0,
    /// });
    /// let normal = [0.0, 0.0, 1.0];
    /// let pairs = vec![
    ///     PenaltyPair { slave: 0, gap: 0.0, normal, force: [0.0, 0.0, -100.0] },
    ///     PenaltyPair { slave: 1, gap: 0.0, normal, force: [0.0, 0.0, -1500.0] },
    /// ];
    /// assert_eq!(state.rupture(&pairs), vec![1]);
    /// assert!(state.is_failed(1));
    /// assert!(!state.is_failed(0));
    /// ```
    pub fn rupture(&mut self, pairs: &[PenaltyPair]) -> Vec<usize> {
        let mut ruptured = Vec::new();
        for (index, pair) in pairs.iter().enumerate() {
            if self.failed.contains(&pair.slave) {
                continue;
            }
            if self.criterion.tripped(pair) {
                self.failed.insert(pair.slave);
                ruptured.push(index);
            }
        }
        ruptured
    }

    /// Drops failed slaves from one penalty-pair slice.
    ///
    /// Returns the bonded pairs in input order; failed slaves stay out
    /// permanently once [`TiebreakState::rupture`] records them.
    pub fn bonded_pairs(&self, pairs: &[PenaltyPair]) -> Vec<PenaltyPair> {
        pairs
            .iter()
            .copied()
            .filter(|pair| !self.failed.contains(&pair.slave))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::solid::fracture::cohesive::BilinearCohesiveLaw;

    /// Builds a pair with tensile normal load plus x-shear load.
    ///
    /// Tension pulls the slave back toward the master (`-z`), so the
    /// frozen [`PenaltyPair::normal_force`] (compression-positive)
    /// reports `-tensile`; shear rides along `+x`. A negative
    /// `tensile` is contact push-out (compression, immune).
    fn loaded_pair(slave: usize, tensile: f64, shear_x: f64) -> PenaltyPair {
        PenaltyPair {
            slave,
            gap: 1.0e-6,
            normal: [0.0, 0.0, 1.0],
            force: [shear_x, 0.0, -tensile],
        }
    }

    fn force_criterion() -> TiebreakCriterion {
        TiebreakCriterion::Force {
            normal_rupture: 1000.0,
            shear_rupture: 2000.0,
        }
    }

    #[test]
    fn force_trips_exactly_at_boundary() {
        let criterion = force_criterion();
        let at = loaded_pair(0, 1000.0, 0.0);
        assert_eq!(criterion.utilization(&at), 1.0);
        assert!(criterion.tripped(&at));
        let below = loaded_pair(0, 1000.0 * (1.0 - 1.0e-12), 0.0);
        assert!(criterion.utilization(&below) < 1.0);
        assert!(!criterion.tripped(&below));
        let shear_at = loaded_pair(1, 0.0, 2000.0);
        assert_eq!(criterion.utilization(&shear_at), 1.0);
        assert!(criterion.tripped(&shear_at));
    }

    #[test]
    fn force_uses_quadratic_interaction() {
        let criterion = force_criterion();
        // 0.75^2 + 0.75^2 = 1.125: combined rupture with no single
        // component (0.5625) tripping on its own. Exact in binary.
        let combined = loaded_pair(0, 750.0, 1500.0);
        assert_eq!(criterion.utilization(&combined), 1.125);
        assert!(criterion.tripped(&combined));
        let tension_only = loaded_pair(0, 750.0, 0.0);
        assert_eq!(criterion.utilization(&tension_only), 0.5625);
        assert!(!criterion.tripped(&tension_only));
        let shear_only = loaded_pair(0, 0.0, 1500.0);
        assert_eq!(criterion.utilization(&shear_only), 0.5625);
        assert!(!criterion.tripped(&shear_only));
    }

    #[test]
    fn force_ignores_compression() {
        let criterion = force_criterion();
        // Negative tensile argument is compression on the slave.
        let crushing = loaded_pair(0, -1.0e9, 0.0);
        assert_eq!(criterion.utilization(&crushing), 0.0);
        assert!(!criterion.tripped(&crushing));
    }

    #[test]
    fn energy_trips_exactly_at_boundary() {
        let stiffness = 1.0e12;
        let traction = 30.0e6;
        // Same operations as the criterion, so available equals critical
        // bit-for-bit at the boundary.
        let toughness = 0.5 * (traction * (traction / stiffness));
        let law = MixedModeCohesiveLaw::new(
            BilinearCohesiveLaw::new(stiffness, 30.0e6, toughness),
            BilinearCohesiveLaw::new(stiffness, 30.0e6, toughness),
            1.5,
        );
        let criterion = TiebreakCriterion::Energy { law, area: 1.0 };
        let at = loaded_pair(0, traction, 0.0);
        assert_eq!(criterion.utilization(&at), 1.0);
        assert!(criterion.tripped(&at));
        let below = loaded_pair(0, traction * (1.0 - 1.0e-9), 0.0);
        assert!(criterion.utilization(&below) < 1.0);
        assert!(!criterion.tripped(&below));
        let above = loaded_pair(0, traction * (1.0 + 1.0e-9), 0.0);
        assert!(criterion.utilization(&above) > 1.0);
        assert!(criterion.tripped(&above));
    }

    #[test]
    fn energy_selects_toughness_by_mode() {
        let stiffness = 1.0e12;
        let law = MixedModeCohesiveLaw::new(
            BilinearCohesiveLaw::new(stiffness, 30.0e6, 1000.0),
            BilinearCohesiveLaw::new(stiffness, 30.0e6, 100.0),
            1.5,
        );
        let criterion = TiebreakCriterion::Energy { law, area: 1.0 };
        // Available release 150 J/m^2: above shear toughness, below normal.
        let traction = (2.0 * stiffness * 150.0).sqrt();
        let shear = loaded_pair(0, 0.0, traction);
        assert!(criterion.tripped(&shear));
        let normal = loaded_pair(1, traction, 0.0);
        assert!(!criterion.tripped(&normal));
    }

    #[test]
    fn energy_ignores_compression() {
        let law = MixedModeCohesiveLaw::epoxy();
        let criterion = TiebreakCriterion::Energy { law, area: 1.0e-6 };
        let crushing = loaded_pair(0, -1.0e9, 0.0);
        assert_eq!(criterion.utilization(&crushing), 0.0);
        assert!(!criterion.tripped(&crushing));
    }

    #[test]
    fn rupture_is_irreversible() {
        let mut state = TiebreakState::new(force_criterion());
        assert_eq!(state.rupture(&[loaded_pair(5, 1500.0, 0.0)]), vec![0]);
        assert!(state.is_failed(5));
        assert_eq!(state.num_failed(), 1);
        // Sub-threshold reloads never heal the weld.
        assert!(state.rupture(&[loaded_pair(5, 10.0, 0.0)]).is_empty());
        assert!(state.rupture(&[]).is_empty());
        assert!(state.is_failed(5));
        assert_eq!(state.num_failed(), 1);
        assert_eq!(state.failed_slaves(), vec![5]);
    }

    #[test]
    fn rupture_sequences_a_scripted_series() {
        let mut state = TiebreakState::new(force_criterion());
        // Step 1: simultaneous trips report in input order — position wins
        // over slave id (3 < 7 < 9) and over load magnitude.
        let step1 = [
            loaded_pair(9, 5000.0, 0.0),
            loaded_pair(3, 1200.0, 0.0),
            loaded_pair(7, 1001.0, 0.0),
            loaded_pair(1, 100.0, 0.0),
        ];
        assert_eq!(state.rupture(&step1), vec![0, 1, 2]);
        assert_eq!(state.num_failed(), 3);
        // Step 2: failed slaves are skipped even under overload; only the
        // newly tripped slave reports, at its own input index.
        let step2 = [
            loaded_pair(1, 100.0, 0.0),
            loaded_pair(9, 9000.0, 0.0),
            loaded_pair(2, 2500.0, 0.0),
        ];
        assert_eq!(state.rupture(&step2), vec![2]);
        // Step 3: repeats never re-report; the survivor stays bonded.
        assert!(state.rupture(&step1).is_empty());
        assert!(state.rupture(&step2).is_empty());
        assert!(!state.is_failed(1));
        assert!(state.is_failed(2));
        assert_eq!(state.failed_slaves(), vec![2, 3, 7, 9]);
        // Survivors keep input order and data through the filter.
        let bonded = state.bonded_pairs(&step2);
        assert_eq!(bonded.len(), 1);
        assert_eq!(bonded[0].slave, 1);
        assert_eq!(bonded[0].force, [0.0, 0.0, -100.0]);
    }

    #[test]
    fn rupture_handles_empty_and_singleton() {
        let mut state = TiebreakState::new(force_criterion());
        assert!(state.rupture(&[]).is_empty());
        assert_eq!(state.num_failed(), 0);
        assert!(state.bonded_pairs(&[]).is_empty());
        assert!(state.rupture(&[loaded_pair(0, 10.0, 0.0)]).is_empty());
        assert_eq!(state.rupture(&[loaded_pair(0, 5000.0, 0.0)]), vec![0]);
        assert!(
            state
                .bonded_pairs(&[loaded_pair(0, 5000.0, 0.0)])
                .is_empty()
        );
    }
}
