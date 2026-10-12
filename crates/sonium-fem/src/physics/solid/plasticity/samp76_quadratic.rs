//! Quadratic LAW76 with nonassociated isochoric flow (`IFORM=0`, `IQUAD=1`,
//! `Nu_p=0.5`) and proportional, nondecreasing hardening.
//!
//! Tension and compression curves use equivalent plastic strain. The engine
//! shear curve uses `sqrt(3)/2` times that strain; here the supplied common
//! multiplier explicitly uses equivalent strain for all three strengths.
//! Shear strength denotes shear stress, rather than von Mises stress.
//! No rate, damage, failure, or automatic convexity correction is applied.
//! Callers supply engineering C1 increments in the corotational frame.
//!
//! # Examples
//!
//! ```rust
//! use sonium_fem::physics::solid::plasticity::{
//!     funct::Funct1D, samp76_quadratic::{Samp76Quadratic, Samp76QuadraticState},
//! };
//! let hardening = Funct1D::new(vec![0.0, 0.5], vec![1.0, 11.0]).unwrap();
//! let law = Samp76Quadratic::new(350e6, 0.0, 500e3, 900e3, 420e3, hardening).unwrap();
//! let mut state = Samp76QuadraticState::default();
//! assert!(law.step(&mut state, &[-0.004, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap().yielding);
//! ```
// Rust guideline compliant 2026-02-21

use super::crash::elastic_tangent_c1;
use super::funct::Funct1D;
use super::law_foams::{deviator_c1, pressure_c1};
use super::update::{StressUpdate, StressUpdateModel, von_mises_c1};

/// Quadratic SAMP card with a common hardening multiplier.
#[derive(Debug, Clone)]
pub struct Samp76Quadratic {
    bulk: f64,
    shear: f64,
    a0: f64,
    a1: f64,
    a2: f64,
    hardening: Funct1D,
}

/// Committed C1 Cauchy stress and equivalent plastic strain.
#[derive(Debug, Clone, Default)]
pub struct Samp76QuadraticState {
    /// Stress `[xx, yy, zz, xy, yz, zx]`.
    pub stress: [f64; 6],
    /// Equivalent plastic strain (tension/compression curve coordinate).
    pub plastic_strain: f64,
}

impl Samp76Quadratic {
    /// Creates a card with initial uniaxial strengths and shear stress strength.
    /// The hardening multiplier must be positive and nondecreasing over the
    /// reached strain range. This contract is checked during the return solve.
    ///
    /// # Errors
    /// Rejects nonfinite parameters, invalid elasticity, nonpositive strengths,
    /// or nonconvex initial curves (`3 shear² < tension compression`).
    pub fn new(
        young: f64,
        poisson: f64,
        tension: f64,
        compression: f64,
        shear_strength: f64,
        hardening: Funct1D,
    ) -> Result<Self, String> {
        if [young, poisson, tension, compression, shear_strength]
            .iter()
            .any(|v| !v.is_finite())
            || young <= 0.0
            || poisson <= -1.0
            || poisson >= 0.5
            || tension <= 0.0
            || compression <= 0.0
            || shear_strength <= 0.0
            || 3.0 * shear_strength * shear_strength < tension * compression
        {
            return Err("invalid quadratic LAW76 parameters".into());
        }
        let product = tension * compression;
        let law = Self {
            bulk: young / (3.0 * (1.0 - 2.0 * poisson)),
            shear: young / (2.0 * (1.0 + poisson)),
            a0: 3.0 * shear_strength * shear_strength,
            a1: 9.0 * shear_strength * shear_strength * (compression - tension) / product,
            a2: 9.0 * (product - 3.0 * shear_strength * shear_strength) / product,
            hardening,
        };
        if [law.bulk, law.shear, law.a0, law.a1, law.a2]
            .iter()
            .any(|v| !v.is_finite())
        {
            return Err("overflowed quadratic LAW76 parameters".into());
        }
        law.yield_squared(0.0, 0.0)?;
        Ok(law)
    }

    /// Squared von Mises yield radius at compressive-positive pressure.
    ///
    /// # Errors
    /// Rejects invalid hardening, nonfinite inputs, and pressure beyond the
    /// yield surface's hydrostatic closure. Isochoric flow cannot return such
    /// a pressure; silently clipping its radius to zero would be incorrect.
    pub fn yield_squared(&self, pressure: f64, plastic_strain: f64) -> Result<f64, String> {
        let h = self.hardening.eval(plastic_strain);
        let squared = self.a0 * h * h + self.a1 * h * pressure + self.a2 * pressure * pressure;
        if !pressure.is_finite()
            || !plastic_strain.is_finite()
            || plastic_strain < 0.0
            || !h.is_finite()
            || h <= 0.0
            || self.hardening.slope(plastic_strain) < 0.0
            || !squared.is_finite()
            || squared < 0.0
        {
            return Err("LAW76 hardening or pressure outside isochoric return domain".into());
        }
        Ok(squared)
    }

    fn solve(
        &self,
        state: &Samp76QuadraticState,
        increment: &[f64; 6],
    ) -> Result<([f64; 6], bool, f64), String> {
        if increment
            .iter()
            .chain(state.stress.iter())
            .any(|v| !v.is_finite())
        {
            return Err("nonfinite LAW76 state or increment".into());
        }
        let trace = increment[..3].iter().sum::<f64>();
        let trial = std::array::from_fn(|i| {
            state.stress[i]
                + if i < 3 {
                    2.0 * self.shear * (increment[i] - trace / 3.0) + self.bulk * trace
                } else {
                    self.shear * increment[i]
                }
        });
        let p = pressure_c1(&trial);
        let q = von_mises_c1(&trial);
        let radius = self.yield_squared(p, state.plastic_strain)?;
        if q * q <= radius {
            return Ok((trial, false, 0.0));
        }
        let mut low = 0.0;
        let mut high = q / (3.0 * self.shear);
        for _ in 0..64 {
            let dep = 0.5 * (low + high);
            let q_new = q - 3.0 * self.shear * dep;
            let radius = self.yield_squared(p, state.plastic_strain + dep)?;
            if q_new * q_new > radius {
                low = dep;
            } else {
                high = dep;
            }
        }
        let dep = 0.5 * (low + high);
        let factor = (q - 3.0 * self.shear * dep) / q;
        let dev = deviator_c1(&trial);
        let stress = std::array::from_fn(|i| factor * dev[i] - if i < 3 { p } else { 0.0 });
        Ok((stress, true, dep))
    }

    /// Commits an increment after checking its material return domain.
    ///
    /// # Errors
    /// Rejects invalid state, strain, hardening, or hydrostatic pressure without
    /// changing the committed state.
    pub fn step(
        &self,
        state: &mut Samp76QuadraticState,
        increment: &[f64; 6],
    ) -> Result<StressUpdate, String> {
        self.solve(state, increment)?;
        let out = self.update(state, increment);
        if out
            .stress
            .iter()
            .chain(out.tangent.iter())
            .any(|v| !v.is_finite())
        {
            return Err("invalid LAW76 stress or tangent".into());
        }
        self.commit(state, &out);
        Ok(out)
    }
}

impl StressUpdateModel for Samp76Quadratic {
    type State = Samp76QuadraticState;
    type StrainIncrement = [f64; 6];
    /// Invalid unchecked inputs produce NaN. Use `step` for checked commits.
    fn update(&self, state: &Self::State, increment: &Self::StrainIncrement) -> StressUpdate {
        let (stress, yielding, dep) =
            self.solve(state, increment)
                .unwrap_or(([f64::NAN; 6], false, f64::NAN));
        let mut tangent = elastic_tangent_c1(self.bulk, self.shear);
        if yielding {
            for col in 0..6 {
                let h = 1e-7 * increment[col].abs().max(1e-3);
                let mut plus = *increment;
                let mut minus = *increment;
                plus[col] += h;
                minus[col] -= h;
                let a = self
                    .solve(state, &plus)
                    .map(|v| v.0)
                    .unwrap_or([f64::NAN; 6]);
                let b = self
                    .solve(state, &minus)
                    .map(|v| v.0)
                    .unwrap_or([f64::NAN; 6]);
                for row in 0..6 {
                    tangent[row * 6 + col] = (a[row] - b[row]) / (2.0 * h);
                }
            }
        }
        StressUpdate {
            stress,
            tangent,
            yielding,
            delta_equiv_plastic_strain: dep,
        }
    }
    fn commit(&self, state: &mut Self::State, out: &StressUpdate) {
        state.stress = out.stress;
        state.plastic_strain += out.delta_equiv_plastic_strain;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn law() -> Samp76Quadratic {
        Samp76Quadratic::new(
            350e6,
            0.0,
            500000.0,
            900000.0,
            420000.0,
            Funct1D::new(vec![0.0, 0.5], vec![1.0, 11.0]).unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn quadratic_surface_calibrates_all_three_initial_strengths() {
        for (p, q) in [
            (-500000.0 / 3.0, 500000.0),
            (900000.0 / 3.0, 900000.0),
            (0.0, 3.0_f64.sqrt() * 420000.0),
        ] {
            assert!((law().yield_squared(p, 0.0).unwrap() - q * q).abs() / (q * q) < 1e-14);
        }
    }
    #[test]
    fn return_satisfies_updated_hardening_surface_and_elastic_unloading() {
        let law = law();
        let mut state = Samp76QuadraticState::default();
        assert!(
            law.step(&mut state, &[-0.004, 0.0, 0.0, 0.0, 0.0, 0.0])
                .unwrap()
                .yielding
        );
        let q = von_mises_c1(&state.stress);
        assert!(
            (q * q
                - law
                    .yield_squared(pressure_c1(&state.stress), state.plastic_strain)
                    .unwrap())
            .abs()
                / (q * q)
                < 1e-13
        );
        assert!(state.plastic_strain > 0.0005);
        let old = state.stress[0];
        assert!(
            !law.step(&mut state, &[1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])
                .unwrap()
                .yielding
        );
        assert!((state.stress[0] - old - 3500.0).abs() < 1e-8);
    }
    #[test]
    fn isochoric_flow_rejects_unreturnable_pressure_without_committing() {
        let mut state = Samp76QuadraticState::default();
        assert!(
            law()
                .step(&mut state, &[0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
                .is_err()
        );
        assert_eq!(state.stress, [0.0; 6]);
        assert_eq!(state.plastic_strain, 0.0);
    }
}
