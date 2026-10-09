//! Fixed-point Payne–Hanek range reduction for finite `f64` inputs.
//!
//! The normal Cody–Waite path is deliberately small and fast, but it cannot
//! retain enough quadrant bits once the input exponent grows. This module
//! keeps a multi-precision slow path independent of platform libm: π is
//! constructed once at 2048 fixed-point bits with Machin's formula, then the
//! input mantissa is reduced against fixed-point `2/π` and `π/2` integers.

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, ToPrimitive, Zero};
use std::sync::OnceLock;

const FIXED_BITS: usize = 2048;

struct Constants {
    pi_over_two: BigUint,
    two_over_pi: BigUint,
}

static CONSTANTS: OnceLock<Constants> = OnceLock::new();

/// Reduced remainder and quadrant for `x = q·π/2 + remainder`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Reduction {
    pub remainder: f64,
    pub quadrant: u8,
}

fn constants() -> &'static Constants {
    CONSTANTS.get_or_init(|| {
        // Machin: π = 16 atan(1/5) - 4 atan(1/239). Computing the series
        // directly in fixed point avoids shipping a platform-dependent π
        // literal and gives the reduction more bits than any f64 input can
        // consume.
        let atan_1_5 = atan_inverse_fixed(5);
        let atan_1_239 = atan_inverse_fixed(239);
        let pi: BigInt = BigInt::from_biguint(Sign::Plus, atan_1_5) * 16
            - BigInt::from_biguint(Sign::Plus, atan_1_239) * 4;
        let pi = pi
            .to_biguint()
            .expect("Machin fixed-point pi must be positive");
        let pi_over_two = &pi >> 1;
        let numerator = BigUint::one() << (2 * FIXED_BITS + 1);
        let two_over_pi = numerator / &pi;
        Constants {
            pi_over_two,
            two_over_pi,
        }
    })
}

fn atan_inverse_fixed(inverse: u32) -> BigUint {
    let scale = BigUint::one() << FIXED_BITS;
    let inverse_big = BigUint::from(inverse);
    let inverse_squared = &inverse_big * &inverse_big;
    let mut inverse_power = inverse_big;
    let mut sum = BigInt::zero();
    let mut index = 0u32;

    loop {
        let denominator = &inverse_power * BigUint::from(2 * index + 1);
        let term = &scale / denominator;
        if term.is_zero() {
            break;
        }
        let signed = BigInt::from_biguint(Sign::Plus, term);
        if index.is_multiple_of(2) {
            sum += signed;
        } else {
            sum -= signed;
        }
        inverse_power *= &inverse_squared;
        index += 1;
    }

    sum.to_biguint()
        .expect("atan(1/inverse) fixed-point value must be positive")
}

fn round_shift_even(value: &BigUint, shift: usize) -> BigUint {
    if shift == 0 {
        return value.clone();
    }
    let mut quotient = value >> shift;
    let remainder_mask = (BigUint::one() << shift) - BigUint::one();
    let remainder = value & remainder_mask;
    let half = BigUint::one() << (shift - 1);
    if remainder > half || (remainder == half && quotient.bit(0)) {
        quotient += BigUint::one();
    }
    quotient
}

/// Convert a fixed-point non-negative integer to f64 without overflowing on
/// the 2048-bit scale. Only the leading 53 bits are needed for the reduced
/// remainder, whose magnitude is bounded by π/4.
fn fixed_to_f64(value: &BigUint) -> f64 {
    if value.is_zero() {
        return 0.0;
    }
    let bits = value.bits() as i32;
    let keep_shift = (bits - 53).max(0) as usize;
    let leading = (value >> keep_shift)
        .to_f64()
        .expect("leading fixed-point bits fit in f64");
    leading * 2.0_f64.powi(keep_shift as i32 - FIXED_BITS as i32)
}

fn finite_mantissa_and_exponent(value: f64) -> (u64, i32) {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent == 0 {
        (fraction, -1074)
    } else {
        ((1_u64 << 52) | fraction, exponent - 1023 - 52)
    }
}

/// Reduce any non-zero finite f64 through fixed-point `2/π` and `π/2`.
pub(crate) fn reduce(value: f64) -> Reduction {
    debug_assert!(value.is_finite() && value != 0.0);
    let negative = value.is_sign_negative();
    let magnitude = value.abs();
    let (mantissa, exponent) = finite_mantissa_and_exponent(magnitude);
    let constants = constants();

    // z / 2^FIXED_BITS approximates |x|·2/π. Round to nearest-even to obtain
    // the exact quadrant integer for the fixed-point precision available.
    let product = BigUint::from(mantissa) * &constants.two_over_pi;
    let shift = FIXED_BITS as i32 - exponent;
    let quotient = if shift > 0 {
        round_shift_even(&product, shift as usize)
    } else {
        product << (-shift as usize)
    };

    // Both |x| and q·π/2 are represented at the same fixed-point scale.
    let x_scaled = BigUint::from(mantissa) << (exponent + FIXED_BITS as i32) as usize;
    let q_pi_over_two = &quotient * &constants.pi_over_two;
    let (remainder_magnitude, positive_remainder) = if x_scaled >= q_pi_over_two {
        (x_scaled - q_pi_over_two, true)
    } else {
        (q_pi_over_two - x_scaled, false)
    };
    let remainder = fixed_to_f64(&remainder_magnitude)
        * if positive_remainder { 1.0 } else { -1.0 }
        * if negative { -1.0 } else { 1.0 };

    let quotient_mod = (quotient & BigUint::from(3_u8))
        .to_u8()
        .expect("quadrant fits in u8");
    let quadrant = if negative {
        (4 - quotient_mod) & 3
    } else {
        quotient_mod
    };
    Reduction {
        remainder,
        quadrant,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point_reduction_stays_in_principal_interval() {
        for &value in &[1.0e3 + 0.25, -1.0e6, 1.0e12, -1.0e20, f64::MAX, -f64::MAX] {
            let reduction = reduce(value);
            assert!(
                reduction.remainder.abs() <= std::f64::consts::FRAC_PI_4 + 1.0e-12,
                "x={value:e}, reduction={reduction:?}"
            );
            assert!(reduction.quadrant < 4);
        }
    }

    #[test]
    fn fixed_point_reduction_preserves_sign_and_known_quadrants() {
        for &value in &[1.0e-300, 0.3, 1.0e3 + 0.25, 1.0e12, 1.0e20, f64::MAX] {
            let positive = reduce(value);
            let negative = reduce(-value);
            assert_eq!(
                negative.quadrant,
                (4 - positive.quadrant) & 3,
                "quadrant symmetry failed for {value:e}"
            );
            assert_eq!(
                negative.remainder.to_bits(),
                (-positive.remainder).to_bits(),
                "remainder sign symmetry failed for {value:e}"
            );
        }

        let pi = std::f64::consts::PI;
        for (multiple, quadrant) in [(1.0, 1), (2.0, 2), (3.0, 3), (4.0, 0)] {
            let reduction = reduce(multiple * pi / 2.0);
            assert_eq!(reduction.quadrant, quadrant);
            assert!(
                reduction.remainder.abs() < 2.0e-15,
                "remainder for {multiple}π/2 was {}",
                reduction.remainder
            );
        }
    }
}
