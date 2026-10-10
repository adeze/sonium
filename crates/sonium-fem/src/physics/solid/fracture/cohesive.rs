//! Cohesive zone models for adhesives and delamination.
//!
//! Provides traction-separation laws with damage history: a bilinear law
//! with closed-form onset and failure openings, a Needleman exponential
//! law, and a mixed-mode law with the Benzeggagh-Kenane propagation
//! criterion. The explicit contact path consumes these laws for spotweld
//! and tiebreak failure; the patch-test driver below validates each law
//! against its analytic peak traction and dissipated energy.

// Rust guideline compliant 2026-02-21

/// History state of one cohesive integration point.
#[derive(Debug, Clone, Copy)]
pub struct CohesiveState {
    /// Maximum effective opening reached, in m.
    pub max_opening: f64,
    /// Stiffness damage in `[0, 1]`.
    pub damage: f64,
    /// Dissipated energy density in J/m^2.
    pub dissipated: f64,
}

impl CohesiveState {
    /// Creates a virgin cohesive state.
    pub fn new() -> Self {
        Self {
            max_opening: 0.0,
            damage: 0.0,
            dissipated: 0.0,
        }
    }
}

impl Default for CohesiveState {
    fn default() -> Self {
        Self::new()
    }
}

/// Bilinear traction-separation law.
///
/// Rises with `penalty_stiffness` to `strength` at the onset opening,
/// then softens linearly to zero at the failure opening fixed by
/// `fracture_energy = strength * failure_opening / 2`. Unloading and
/// reloading follow the damaged secant. Compression (negative normal
/// opening) carries the full penalty stiffness with no damage.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::fracture::cohesive::{BilinearCohesiveLaw, CohesiveState};
///
/// let law = BilinearCohesiveLaw::new(1.0e12, 30.0e6, 500.0);
/// let mut state = CohesiveState::new();
/// let peak = law.traction(law.onset_opening(), &mut state);
/// assert!((peak - 30.0e6).abs() / 30.0e6 < 1e-12);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BilinearCohesiveLaw {
    /// Penalty stiffness in Pa/m.
    pub penalty_stiffness: f64,
    /// Interface strength in Pa.
    pub strength: f64,
    /// Fracture energy in J/m^2.
    pub fracture_energy: f64,
}

impl BilinearCohesiveLaw {
    /// Creates a bilinear law from stiffness, strength, and energy.
    pub fn new(penalty_stiffness: f64, strength: f64, fracture_energy: f64) -> Self {
        Self {
            penalty_stiffness,
            strength,
            fracture_energy,
        }
    }

    /// Epoxy adhesive with 30 MPa strength and 500 J/m^2 toughness.
    pub fn epoxy() -> Self {
        Self::new(1.0e12, 30.0e6, 500.0)
    }

    /// Opening at damage onset in m (`strength / stiffness`).
    pub fn onset_opening(&self) -> f64 {
        self.strength / self.penalty_stiffness.max(1e-30)
    }

    /// Opening at complete failure in m (`2 G / strength`).
    pub fn failure_opening(&self) -> f64 {
        2.0 * self.fracture_energy / self.strength.max(1e-30)
    }

    /// Traction in Pa at an opening in m, updating history.
    ///
    /// Takes the normal opening (positive in tension) and advances the
    /// damage state; returns the traction. Pure in compression.
    pub fn traction(&self, opening: f64, state: &mut CohesiveState) -> f64 {
        if opening <= 0.0 {
            return self.penalty_stiffness * opening;
        }
        let d0 = self.onset_opening();
        let df = self.failure_opening().max(d0 * (1.0 + 1e-12));
        state.max_opening = state.max_opening.max(opening);
        let dmax = state.max_opening;
        if dmax <= d0 {
            state.damage = 0.0;
            return self.penalty_stiffness * opening;
        }
        // Bilinear damage: secant through the softening branch.
        let damage = ((df * (dmax - d0)) / (dmax * (df - d0))).clamp(0.0, 1.0);
        state.damage = damage;
        let traction = (1.0 - damage) * self.penalty_stiffness * opening;
        state.dissipated = self.fracture_energy * damage_progress(dmax, d0, df);
        traction.max(0.0)
    }
}

/// Fraction of fracture energy dissipated at opening `dmax`.
///
/// Integrates the bilinear curve analytically: quadratic rise to onset,
/// then the softening-branch area. Returns a value in `[0, 1]`.
fn damage_progress(dmax: f64, d0: f64, df: f64) -> f64 {
    if dmax <= d0 {
        return 0.0;
    }
    if dmax >= df {
        return 1.0;
    }
    // Dissipated = area under the loading curve minus the elastically
    // stored triangle, normalized by the total triangle 1/2 sigma df.
    // With branch ratio T/sigma = (df - dmax)/(df - d0) this reduces to
    // dmax/df - branch d0/df: zero at onset, one at failure.
    let branch = (df - dmax) / (df - d0);
    (dmax - branch * d0) / df
}

/// Needleman exponential traction-separation law.
///
/// Traction `T = e sigma_c (d/d_c) exp(-d/d_c)` with characteristic
/// opening `d_c = G / (e sigma_c)`: peaks at `sigma_c`, dissipates
/// exactly `G`, and never fully severs (traction decays asymptotically).
/// Unloading follows the secant to the origin.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::fracture::cohesive::{ExponentialCohesiveLaw, CohesiveState};
///
/// let law = ExponentialCohesiveLaw::new(30.0e6, 500.0);
/// let mut state = CohesiveState::new();
/// let peak = law.traction(law.characteristic_opening(), &mut state);
/// assert!((peak - 30.0e6).abs() / 30.0e6 < 1e-12);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ExponentialCohesiveLaw {
    /// Interface strength in Pa.
    pub strength: f64,
    /// Fracture energy in J/m^2.
    pub fracture_energy: f64,
}

impl ExponentialCohesiveLaw {
    /// Creates an exponential law from strength and energy.
    pub fn new(strength: f64, fracture_energy: f64) -> Self {
        Self {
            strength,
            fracture_energy,
        }
    }

    /// Characteristic opening `G / (e sigma_c)` in m.
    pub fn characteristic_opening(&self) -> f64 {
        self.fracture_energy / (std::f64::consts::E * self.strength.max(1e-30))
    }

    /// Traction in Pa at an opening in m, updating history.
    ///
    /// Monotonic loading follows the exponential envelope; unloading and
    /// reloading follow the secant fixed by the historic maximum.
    pub fn traction(&self, opening: f64, state: &mut CohesiveState) -> f64 {
        if opening <= 0.0 {
            // Contact stiffness matches the initial envelope slope.
            let k0 = std::f64::consts::E * self.strength
                / self.characteristic_opening().max(1e-30);
            return k0 * opening;
        }
        let dc = self.characteristic_opening().max(1e-30);
        state.max_opening = state.max_opening.max(opening);
        let envelope = |d: f64| {
            std::f64::consts::E * self.strength * (d / dc) * (-d / dc).exp()
        };
        let traction = if opening >= state.max_opening {
            envelope(opening)
        } else {
            envelope(state.max_opening) * opening / state.max_opening.max(1e-30)
        };
        let x = state.max_opening / dc;
        state.dissipated = self.fracture_energy * (1.0 - (1.0 + x) * (-x).exp());
        state.damage = (state.dissipated / self.fracture_energy.max(1e-30)).clamp(0.0, 1.0);
        traction.max(0.0)
    }
}

/// Mixed-mode bilinear law with Benzeggagh-Kenane propagation.
///
/// Combines a normal and a shear bilinear law through the effective
/// opening `dm = sqrt(dn^2 + ds^2 + dt^2)` (normal part active in
/// tension only) with mode mixity `beta = shear / normal`. The mixed
/// fracture energy follows Benzeggagh-Kenane: `G = G_n + (G_s - G_n)
/// (beta^2 / (1 + beta^2))^eta`. Traction components share one damage
/// variable driven by the effective opening.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::fracture::cohesive::{BilinearCohesiveLaw, MixedModeCohesiveLaw, CohesiveState};
///
/// let law = MixedModeCohesiveLaw::epoxy();
/// let mut state = CohesiveState::new();
/// let t = law.traction_3d([1.0e-6, 0.0, 0.0], &mut state);
/// assert!(t[0] > 0.0 && t[1] == 0.0 && t[2] == 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MixedModeCohesiveLaw {
    /// Normal-direction bilinear law.
    pub normal: BilinearCohesiveLaw,
    /// Shear-direction bilinear law (shared by both shear axes).
    pub shear: BilinearCohesiveLaw,
    /// Benzeggagh-Kenane exponent, dimensionless.
    pub bk_eta: f64,
}

impl MixedModeCohesiveLaw {
    /// Creates a mixed-mode law from normal, shear, and BK exponent.
    pub fn new(
        normal: BilinearCohesiveLaw,
        shear: BilinearCohesiveLaw,
        bk_eta: f64,
    ) -> Self {
        Self {
            normal,
            shear,
            bk_eta,
        }
    }

    /// Epoxy adhesive with matched shear properties and BK exponent 1.5.
    pub fn epoxy() -> Self {
        Self::new(
            BilinearCohesiveLaw::epoxy(),
            BilinearCohesiveLaw::new(1.0e12, 30.0e6, 800.0),
            1.5,
        )
    }

    /// Mixed-mode fracture energy in J/m^2 at a shear/normal ratio.
    ///
    /// Takes `beta = shear_opening / normal_opening`; pure normal returns
    /// `G_n`, pure shear returns `G_s`, mixed states interpolate by BK.
    pub fn mixed_fracture_energy(&self, beta: f64) -> f64 {
        let gn = self.normal.fracture_energy;
        let gs = self.shear.fracture_energy;
        let mix = (beta * beta / (1.0 + beta * beta)).powf(self.bk_eta);
        gn + (gs - gn) * mix
    }

    /// Traction vector in Pa at an opening vector in m.
    ///
    /// Takes `[normal, shear_s, shear_t]` openings and advances the shared
    /// damage history; returns the traction components. Compressive normal
    /// openings carry penalty contact with no damage.
    pub fn traction_3d(&self, opening: [f64; 3], state: &mut CohesiveState) -> [f64; 3] {
        let dn = opening[0];
        let shear = (opening[1] * opening[1] + opening[2] * opening[2]).sqrt();
        if dn <= 0.0 && shear == 0.0 {
            return [self.normal.penalty_stiffness * dn, 0.0, 0.0];
        }
        let tension = dn.max(0.0);
        let dm = (tension * tension + shear * shear).sqrt();
        if dm == 0.0 {
            return [self.normal.penalty_stiffness * dn, 0.0, 0.0];
        }
        let beta = shear / tension.max(1e-30);
        let mixed_g = self.mixed_fracture_energy(beta);
        // Equivalent single-mode law at this mixity: onset from the
        // stiffness-weighted strength, failure from the mixed energy.
        let kn = self.normal.penalty_stiffness;
        let ks = self.shear.penalty_stiffness;
        let keq = (kn * tension * tension + ks * shear * shear) / (dm * dm);
        let t0 = ((self.normal.strength * tension).powi(2)
            + (self.shear.strength * shear).powi(2))
        .sqrt()
            / dm;
        let d0 = t0 / keq.max(1e-30);
        let df = (2.0 * mixed_g / t0.max(1e-30)).max(d0 * (1.0 + 1e-12));
        state.max_opening = state.max_opening.max(dm);
        let dmax = state.max_opening;
        let damage = if dmax <= d0 {
            0.0
        } else {
            ((df * (dmax - d0)) / (dmax * (df - d0))).clamp(0.0, 1.0)
        };
        state.damage = damage;
        state.dissipated = mixed_g * damage_progress(dmax, d0, df);
        let secant = 1.0 - damage;
        [
            if dn <= 0.0 {
                self.normal.penalty_stiffness * dn
            } else {
                secant * kn * dn
            },
            secant * ks * opening[1],
            secant * ks * opening[2],
        ]
    }
}

/// Monotonic patch-test driver for a mode-I law.
///
/// Opens one cohesive point from zero to `max_opening` in `steps` equal
/// increments through `traction_fn`, returning the `(opening, traction)`
/// curve. The caller integrates the curve against the fracture energy.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::fracture::cohesive::{BilinearCohesiveLaw, CohesiveState, cohesive_patch_response};
///
/// let law = BilinearCohesiveLaw::epoxy();
/// let mut state = CohesiveState::new();
/// let curve = cohesive_patch_response(200, law.failure_opening(), |d| law.traction(d, &mut state));
/// assert_eq!(curve.len(), 201);
/// ```
pub fn cohesive_patch_response(
    steps: usize,
    max_opening: f64,
    mut traction_fn: impl FnMut(f64) -> f64,
) -> Vec<(f64, f64)> {
    let steps = steps.max(1);
    (0..=steps)
        .map(|i| {
            let opening = max_opening * i as f64 / steps as f64;
            (opening, traction_fn(opening))
        })
        .collect()
}

/// Trapezoidal area under a traction-separation curve.
///
/// Returns the dissipated energy density in J/m^2 for comparison against
/// the fracture energy of the law under test.
pub fn patch_dissipated_energy(curve: &[(f64, f64)]) -> f64 {
    curve
        .windows(2)
        .map(|w| 0.5 * (w[0].1 + w[1].1) * (w[1].0 - w[0].0))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_patch_dissipates_fracture_energy() {
        let law = BilinearCohesiveLaw::epoxy();
        let mut state = CohesiveState::new();
        let curve =
            cohesive_patch_response(2000, law.failure_opening(), |d| law.traction(d, &mut state));
        let dissipated = patch_dissipated_energy(&curve);
        // Bilinear curve is piecewise linear; trapezoid is near-exact.
        assert!(
            (dissipated - law.fracture_energy).abs() / law.fracture_energy < 1e-6,
            "dissipated {dissipated}, want {}",
            law.fracture_energy
        );
        assert!((state.damage - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bilinear_peak_matches_strength() {
        let law = BilinearCohesiveLaw::epoxy();
        let mut state = CohesiveState::new();
        let curve =
            cohesive_patch_response(2000, law.failure_opening(), |d| law.traction(d, &mut state));
        let peak = curve.iter().map(|(_, t)| *t).fold(0.0, f64::max);
        assert!((peak - law.strength).abs() / law.strength < 1e-3);
    }

    #[test]
    fn bilinear_unloads_on_damaged_secant() {
        let law = BilinearCohesiveLaw::epoxy();
        let mut state = CohesiveState::new();
        let d0 = law.onset_opening();
        let df = law.failure_opening();
        let half = 0.5 * (d0 + df);
        let loading = law.traction(half, &mut state);
        assert!(state.damage > 0.0 && state.damage < 1.0);
        // Reloading to the historic maximum returns the same traction.
        let reloading = law.traction(half, &mut state);
        assert!((reloading - loading).abs() < 1e-9);
        // Partial unloading follows the secant through the origin.
        let partial = law.traction(0.5 * half, &mut state);
        assert!((partial - 0.5 * loading).abs() / loading < 1e-9);
    }

    #[test]
    fn bilinear_compression_is_undamaged_penalty() {
        let law = BilinearCohesiveLaw::epoxy();
        let mut state = CohesiveState::new();
        let t = law.traction(-1.0e-6, &mut state);
        assert!((t + law.penalty_stiffness * 1.0e-6).abs() < 1e-6);
        assert_eq!(state.damage, 0.0);
    }

    #[test]
    fn exponential_patch_matches_peak_and_energy() {
        let law = ExponentialCohesiveLaw::new(30.0e6, 500.0);
        let mut state = CohesiveState::new();
        let dc = law.characteristic_opening();
        let peak = law.traction(dc, &mut state);
        assert!((peak - 30.0e6).abs() / 30.0e6 < 1e-12);
        // Integrate to 20 dc where the tail holds under 1e-7 of G.
        let mut tail = CohesiveState::new();
        let curve = cohesive_patch_response(20000, 20.0 * dc, |d| law.traction(d, &mut tail));
        let dissipated = patch_dissipated_energy(&curve);
        assert!(
            (dissipated - 500.0).abs() / 500.0 < 1e-4,
            "dissipated {dissipated}"
        );
    }

    #[test]
    fn mixed_mode_recovers_pure_modes() {
        let law = MixedModeCohesiveLaw::epoxy();
        assert!((law.mixed_fracture_energy(0.0) - law.normal.fracture_energy).abs() < 1e-9);
        let shear = law.mixed_fracture_energy(1.0e9);
        assert!((shear - law.shear.fracture_energy).abs() / law.shear.fracture_energy < 1e-6);
        // Pure-normal response matches the normal law peak.
        let mut state = CohesiveState::new();
        let t = law.traction_3d([law.normal.onset_opening(), 0.0, 0.0], &mut state);
        assert!((t[0] - law.normal.strength).abs() / law.normal.strength < 1e-9);
        assert_eq!(t[1], 0.0);
        assert_eq!(t[2], 0.0);
    }

    #[test]
    fn mixed_mode_dissipates_bk_energy() {
        let law = MixedModeCohesiveLaw::epoxy();
        // 45-degree mixed opening: beta = 1.
        let beta = 1.0;
        let mixed_g = law.mixed_fracture_energy(beta);
        assert!(mixed_g > law.normal.fracture_energy && mixed_g < law.shear.fracture_energy);
        let mut state = CohesiveState::new();
        let failure = 2.0 * mixed_g / 30.0e6 * 2.0_f64.sqrt();
        let steps = 2000;
        let mut dissipated = 0.0;
        let mut prev = (0.0, [0.0; 3]);
        for i in 1..=steps {
            let dm = failure * i as f64 / steps as f64;
            let d = dm / 2.0_f64.sqrt();
            let t = law.traction_3d([d, d, 0.0], &mut state);
            dissipated += 0.5
                * ((prev.1[0] + t[0]) * (d - prev.0) + (prev.1[1] + t[1]) * (d - prev.0));
            prev = (d, t);
        }
        assert!(
            (dissipated - mixed_g).abs() / mixed_g < 0.02,
            "dissipated {dissipated}, want {mixed_g}"
        );
    }
}
