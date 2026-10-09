//! Azimuthal Fourier modes shared by axisymmetric acoustics.
//!
//! Axisymmetric formulations expand the field in spinning modes
//! `exp(imφ)` with integer azimuthal order `m`. This module owns the order
//! type plus the on-axis regularity rules every consumer (modal duct
//! analysis, axisymmetric LEE/LPF assembly, ring-source analogies) must
//! honor:
//!
//! ```text
//! order m      p, u_z      u_r, u_φ
//! m = 0        finite      vanish ∝ r
//! |m| = 1      vanish ∝ r  finite
//! |m| ≥ 2      vanish ∝ r^|m|  vanish ∝ r^(|m|-1)
//! ```
//!
//! Positive `m` spins toward increasing `φ` under the `exp(+iωt)`
//! convention; the sign carries the rotor-stator spinning direction for
//! Tyler–Sofrin interaction modes.

use num_complex::Complex64;

/// Integer azimuthal order of one spinning Fourier mode.
///
/// All integers are valid orders, so construction is infallible; regularity
/// questions are answered by the axis helpers below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AzimuthalMode {
    /// Signed spinning order; nonzero values carry a rotation direction.
    order: i32,
}

impl AzimuthalMode {
    /// Builds the mode of signed spinning order `order`.
    pub fn new(order: i32) -> Self {
        Self { order }
    }

    /// Builds the axisymmetric (`m = 0`) mode.
    pub fn axisymmetric() -> Self {
        Self { order: 0 }
    }

    /// Returns the signed spinning order.
    pub fn order(self) -> i32 {
        self.order
    }

    /// Returns the unsigned order used by Bessel-function evaluations.
    pub fn abs_order(self) -> u32 {
        self.order.unsigned_abs()
    }

    /// Reports whether this is the axisymmetric mode.
    pub fn is_axisymmetric(self) -> bool {
        self.order == 0
    }

    /// Reports whether pressure must vanish on the axis.
    ///
    /// Only the `m = 0` mode carries finite on-axis pressure; every
    /// spinning mode has a pressure node at `r = 0`.
    pub fn pressure_vanishes_on_axis(self) -> bool {
        self.order != 0
    }

    /// Radial power of pressure and axial velocity near the axis.
    ///
    /// Scalar fields scale as `r^|m|`: finite for `m = 0`, linear for
    /// `|m| = 1`, and higher-order afterwards.
    pub fn scalar_axis_power(self) -> u32 {
        self.abs_order()
    }

    /// Radial powers of `(u_r, u_φ, u_z)` near the axis.
    ///
    /// The transverse pair scales as `r^||m|-1|` (finite for the first
    /// spinning mode, vanishing otherwise) while axial velocity follows the
    /// scalar power `r^|m|`.
    pub fn velocity_axis_powers(self) -> [u32; 3] {
        let transverse = self.abs_order().abs_diff(1);
        [transverse, transverse, self.abs_order()]
    }

    /// Spinning direction: `+1`, `0` (axisymmetric), or `-1`.
    pub fn spinning_direction(self) -> i8 {
        self.order.signum() as i8
    }

    /// Evaluates the modal phase `exp(imφ)` at one azimuth.
    pub fn modal_phase(self, azimuth_rad: f64) -> Complex64 {
        Complex64::from_polar(1.0, f64::from(self.order) * azimuth_rad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_powers_follow_regularity_table() {
        assert_eq!(AzimuthalMode::new(0).scalar_axis_power(), 0);
        assert_eq!(AzimuthalMode::new(0).velocity_axis_powers(), [1, 1, 0]);
        assert!(!AzimuthalMode::new(0).pressure_vanishes_on_axis());
        for (order, scalar, transverse) in [(1, 1, 0), (-1, 1, 0), (2, 2, 1), (-3, 3, 2)] {
            let mode = AzimuthalMode::new(order);
            assert_eq!(mode.scalar_axis_power(), scalar, "m = {order}");
            assert_eq!(
                mode.velocity_axis_powers(),
                [transverse, transverse, scalar],
                "m = {order}"
            );
            assert!(mode.pressure_vanishes_on_axis());
        }
    }

    #[test]
    fn spinning_sign_and_phase_are_consistent() {
        assert_eq!(AzimuthalMode::new(2).spinning_direction(), 1);
        assert_eq!(AzimuthalMode::new(-2).spinning_direction(), -1);
        assert_eq!(AzimuthalMode::axisymmetric().spinning_direction(), 0);
        let period = AzimuthalMode::new(3).modal_phase(0.0);
        let wrapped = AzimuthalMode::new(3).modal_phase(std::f64::consts::TAU);
        assert!((period - wrapped).norm() < 1e-15);
        let opposite = AzimuthalMode::new(-3).modal_phase(0.7);
        assert!((opposite - AzimuthalMode::new(3).modal_phase(0.7).conj()).norm() < 1e-15);
    }
}
