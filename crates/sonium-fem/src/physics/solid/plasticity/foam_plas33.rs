//! Plain `/MAT/FOAM_PLAS` LAW33 principal-cap material.
//!
//! Implements the Ka=0 card without tabulated or rate-dependent multipliers.
//! The constitutive equations follow engine `mat033/sigeps33.F`, CASE(1):
//! gamma = J - 1 + gamma0, air = max(0, -P0 gamma/(1+gamma-phi)),
//! and yield = abs(A+B(1+C gamma)). Each skeleton principal stress is
//! capped independently, then the trapped-air pressure is subtracted.
//! Normal elasticity has modulus E and engineering shear has modulus E/2.
//! Callers own rotations and the geometrical volume ratio J; strain
//! increments are measured in the current corotational material frame.
//!
//! # Examples
//!
//! ```rust
//! use sonium_fem::physics::solid::plasticity::foam_plas33::{
//!     FoamPlas33, FoamPlasIncrement, FoamPlasState,
//! };
//! let law = FoamPlas33::new(350e6, -177000.0, 677000.0, -3.34,
//!                          1e5, 0.1, 0.0).unwrap();
//! let mut state = FoamPlasState::default();
//! let update = law.step(&mut state, &FoamPlasIncrement {
//!     strain: [0.95_f64.ln(), 0.0, 0.0, 0.0, 0.0, 0.0], volume_ratio: 0.95,
//! }).unwrap();
//! assert!(update.yielding && state.stress[0] < 0.0);
//! ```
// Rust guideline compliant 2026-02-21

use super::law_foams::jacobi_eigen_c1;
use super::update::{StressUpdate, StressUpdateModel, voigt_equivalent_increment_c1};

/// Plain LAW33 card in consistent stress units.
#[derive(Debug, Clone, Copy)]
pub struct FoamPlas33 {
    young: f64,
    a: f64,
    b: f64,
    c: f64,
    pressure0: f64,
    porosity: f64,
    gamma0: f64,
}

/// Material-frame strain increment and independently computed volume ratio.
#[derive(Debug, Clone, Copy)]
pub struct FoamPlasIncrement {
    /// Engineering C1 strain increment `[xx, yy, zz, xy, yz, zx]`.
    pub strain: [f64; 6],
    /// Current volume/reference-volume, supplied by the element kinematics.
    pub volume_ratio: f64,
}

/// Committed LAW33 stress and plastic history.
#[derive(Debug, Clone)]
pub struct FoamPlasState {
    /// C1 Cauchy stress in the current material frame.
    pub stress: [f64; 6],
    /// Last geometrical volume ratio accepted through `FoamPlas33::step`.
    pub volume_ratio: f64,
    /// Accumulated equivalent strain rejected by principal caps.
    pub equivalent_plastic_strain: f64,
}

impl Default for FoamPlasState {
    fn default() -> Self {
        Self {
            stress: [0.0; 6],
            volume_ratio: 1.0,
            equivalent_plastic_strain: 0.0,
        }
    }
}

impl FoamPlas33 {
    /// Creates a plain LAW33 card from E, A, B, C, P0, phi, gamma0.
    ///
    /// # Errors
    /// Rejects nonfinite inputs, nonpositive E, negative P0, porosity outside
    /// `[0,1)`, or an initial air denominator that is not positive.
    pub fn new(
        young: f64,
        a: f64,
        b: f64,
        c: f64,
        pressure0: f64,
        porosity: f64,
        gamma0: f64,
    ) -> Result<Self, String> {
        if [young, a, b, c, pressure0, porosity, gamma0]
            .iter()
            .any(|v| !v.is_finite())
            || young <= 0.0
            || pressure0 < 0.0
            || !(0.0..1.0).contains(&porosity)
            || 1.0 + gamma0 <= porosity
        {
            return Err("invalid plain LAW33 card".into());
        }
        Ok(Self {
            young,
            a,
            b,
            c,
            pressure0,
            porosity,
            gamma0,
        })
    }

    /// Trapped-air pressure and principal cap at a volume ratio.
    ///
    /// # Errors
    /// Rejects nonfinite or nonpositive volumes and closed air denominators.
    pub fn pressure_and_cap(&self, volume_ratio: f64) -> Result<(f64, f64), String> {
        let gamma = volume_ratio - 1.0 + self.gamma0;
        if !volume_ratio.is_finite() || volume_ratio <= 0.0 || 1.0 + gamma <= self.porosity {
            return Err("LAW33 volume reaches the closed-pore singularity".into());
        }
        Ok((
            (-self.pressure0 * gamma / (1.0 + gamma - self.porosity)).max(0.0),
            (self.a + self.b * (1.0 + self.c * gamma)).abs(),
        ))
    }

    fn solve(&self, state: &FoamPlasState, increment: &FoamPlasIncrement) -> ([f64; 6], bool, f64) {
        let Ok((air, cap)) = self.pressure_and_cap(increment.volume_ratio) else {
            return ([f64::NAN; 6], false, f64::NAN);
        };
        let trial = std::array::from_fn(|i| {
            state.stress[i]
                + self.young * increment.strain[i] * if i < 3 { 1.0 } else { 0.5 }
                + if i < 3 { air } else { 0.0 }
        });
        let (values, vectors) = jacobi_eigen_c1(&trial, 12);
        let capped = values.map(|v| v.clamp(-cap, cap));
        let yielding = values.iter().zip(&capped).any(|(v, c)| v != c);
        let tensor: [f64; 9] = std::array::from_fn(|i| {
            let row = i / 3;
            let col = i % 3;
            (0..3)
                .map(|k| vectors[row][k] * capped[k] * vectors[col][k])
                .sum()
        });
        let skeleton = [
            tensor[0], tensor[4], tensor[8], tensor[1], tensor[5], tensor[2],
        ];
        let stress = std::array::from_fn(|i| skeleton[i] - if i < 3 { air } else { 0.0 });
        let rejected = std::array::from_fn(|i| {
            (trial[i] - skeleton[i]) / self.young * if i < 3 { 1.0 } else { 2.0 }
        });
        (stress, yielding, voigt_equivalent_increment_c1(&rejected))
    }

    /// Advances and commits a valid material-frame increment.
    ///
    /// # Errors
    /// Rejects nonfinite strain/stress or invalid volume before committing.
    pub fn step(
        &self,
        state: &mut FoamPlasState,
        increment: &FoamPlasIncrement,
    ) -> Result<StressUpdate, String> {
        self.pressure_and_cap(increment.volume_ratio)?;
        if increment.strain.iter().any(|v| !v.is_finite())
            || state.stress.iter().any(|v| !v.is_finite())
        {
            return Err("nonfinite LAW33 state or increment".into());
        }
        let update = self.update(state, increment);
        if update.stress.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite LAW33 trial stress".into());
        }
        self.commit(state, &update);
        state.volume_ratio = increment.volume_ratio;
        Ok(update)
    }
}

impl StressUpdateModel for FoamPlas33 {
    type State = FoamPlasState;
    type StrainIncrement = FoamPlasIncrement;

    /// Returns stress and the strain tangent at fixed volume ratio.
    ///
    /// Kinematic volume changes are a separate input: this tangent is the
    /// partial derivative with respect to the six strain components at fixed
    /// J. Direct invalid inputs propagate NaN; use `step` for checked commits.
    fn update(&self, state: &Self::State, increment: &Self::StrainIncrement) -> StressUpdate {
        let (stress, yielding, dep) = self.solve(state, increment);
        let mut tangent = [0.0; 36];
        if yielding {
            // Centered differentiation avoids an incorrect elastic tangent
            // through rotated or partly active principal caps.
            for column in 0..6 {
                let h = 1e-7 * increment.strain[column].abs().max(1e-3);
                let mut plus = *increment;
                let mut minus = *increment;
                plus.strain[column] += h;
                minus.strain[column] -= h;
                let a = self.solve(state, &plus).0;
                let b = self.solve(state, &minus).0;
                for row in 0..6 {
                    tangent[row * 6 + column] = (a[row] - b[row]) / (2.0 * h);
                }
            }
        } else {
            for i in 0..6 {
                tangent[i * 6 + i] = self.young * if i < 3 { 1.0 } else { 0.5 };
            }
        }
        StressUpdate {
            stress,
            tangent,
            delta_equiv_plastic_strain: dep,
            yielding,
        }
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        state.stress = update.stress;
        state.equivalent_plastic_strain += update.delta_equiv_plastic_strain;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn law() -> FoamPlas33 {
        FoamPlas33::new(350e6, -177000.0, 677000.0, -3.34, 1e5, 0.1, 0.0).unwrap()
    }

    #[test]
    fn plain_card_elastic_normal_and_engineering_shear_slopes() {
        let out = law().update(
            &FoamPlasState::default(),
            &FoamPlasIncrement {
                strain: [1e-4, 0.0, 0.0, 2e-4, 0.0, 0.0],
                volume_ratio: 1.0001,
            },
        );
        assert!(!out.yielding);
        assert!((out.stress[0] - 35000.0).abs() < 1e-8);
        assert!((out.stress[3] - 35000.0).abs() < 1e-8);
    }

    #[test]
    fn compress_hold_unload_preserves_cap_and_history() {
        let law = law();
        let mut state = FoamPlasState::default();
        let inc = FoamPlasIncrement {
            strain: [0.95_f64.ln(), 0.0, 0.0, 0.0, 0.0, 0.0],
            volume_ratio: 0.95,
        };
        assert!(law.step(&mut state, &inc).unwrap().yielding);
        let (air, cap) = law.pressure_and_cap(0.95).unwrap();
        assert!((state.stress[0] + cap + air).abs() < 1e-8);
        let held = state.stress;
        law.step(
            &mut state,
            &FoamPlasIncrement {
                strain: [0.0; 6],
                volume_ratio: 0.95,
            },
        )
        .unwrap();
        assert_eq!(state.stress, held);
        let out = law
            .step(
                &mut state,
                &FoamPlasIncrement {
                    strain: [1e-4, 0.0, 0.0, 0.0, 0.0, 0.0],
                    volume_ratio: 0.950095,
                },
            )
            .unwrap();
        assert!(!out.yielding);
        assert!((out.stress[0] - held[0] - 35000.0).abs() < 1e-8);
        assert!(state.equivalent_plastic_strain > 0.01);
    }

    #[test]
    fn rotated_principal_cap_is_frame_indifferent() {
        let law = law();
        let state = FoamPlasState::default();
        let base = law.update(
            &state,
            &FoamPlasIncrement {
                strain: [-0.02, 0.001, 0.0, 0.0, 0.0, 0.0],
                volume_ratio: 0.98,
            },
        );
        let theta = 0.37_f64;
        let c = theta.cos();
        let s = theta.sin();
        let rotated = law.update(
            &state,
            &FoamPlasIncrement {
                strain: [
                    -0.02 * c * c + 0.001 * s * s,
                    -0.02 * s * s + 0.001 * c * c,
                    0.0,
                    2.0 * (-0.021) * s * c,
                    0.0,
                    0.0,
                ],
                volume_ratio: 0.98,
            },
        );
        let expected = [
            base.stress[0] * c * c + base.stress[1] * s * s,
            base.stress[0] * s * s + base.stress[1] * c * c,
            base.stress[2],
            (base.stress[0] - base.stress[1]) * s * c,
            0.0,
            0.0,
        ];
        for (actual, expected) in rotated.stress.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-7);
        }
    }

    #[test]
    fn invalid_volume_does_not_commit() {
        let mut state = FoamPlasState::default();
        assert!(
            law()
                .step(
                    &mut state,
                    &FoamPlasIncrement {
                        strain: [0.0; 6],
                        volume_ratio: 0.05
                    }
                )
                .is_err()
        );
        assert_eq!(state.stress, [0.0; 6]);
        assert_eq!(state.volume_ratio, 1.0);
    }
}
