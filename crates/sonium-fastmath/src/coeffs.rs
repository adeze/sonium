//! Minimax polynomial coefficients for `sin(r)` and `cos(r)` on
//! `r ∈ [−π/4, π/4]`, plus Cody–Waite range-reduction constants for `π/2`.
//!
//! Coefficients are lifted verbatim from [Sleef 3.5][sleef] (Boost License
//! 1.0, compatible with this crate's licence). They target the `u10`
//! accuracy class: `|err(sin)| ≤ 1 ULP`, `|err(cos)| ≤ 1 ULP` for the
//! polynomial evaluation alone over `[−π/4, π/4]`. Combined with two-term
//! Cody–Waite reduction, end-to-end error stays within the published fast-path
//! envelope over its documented domain.
//!
//! [sleef]: https://sleef.org/

// Sleef-published literals carry one more digit than f64 can represent
// exactly so that re-deriving them from a different float type round-trips
// cleanly. Clippy flags this as `excessive_precision`; the LSD rounds to
// the nearest f64 either way, and we want the source-of-truth bit visible.
#![allow(clippy::excessive_precision)]

// ---------------------------------------------------------------------------
// Cody–Waite π/2 split — two-term, sufficient for |k| < 2²⁰
// ---------------------------------------------------------------------------

/// `PI/2` high part. Bit pattern `0x3FF921FB54400000`. The low 20 bits of
/// the mantissa are zero, so for any integer `|k| < 2²⁰` the product
/// `k * PI_OVER_2_HI` is exact in f64.
pub const PI_OVER_2_HI: f64 = 1.5707963267341256;

/// `PI/2` low part. `π/2 - PI_OVER_2_HI` rounded to nearest f64. Together
/// the two reconstruct `π/2` to ~95 bits of precision.
pub const PI_OVER_2_LO: f64 = 6.077100506506192e-11;

/// `2/π`. Used to compute the quadrant index `k = round(x · 2/π)` before
/// range reduction.
pub const TWO_OVER_PI: f64 = std::f64::consts::FRAC_2_PI;

// ---------------------------------------------------------------------------
// sin(r) polynomial — odd, factored as r * P(r²)
// ---------------------------------------------------------------------------
//
// sin(r) ≈ r + r·s·u   where s = r²  and
//   u = ((((((((S0·s + S1)·s + S2)·s + S3)·s + S4)·s + S5)·s + S6)·s + S7)·s + S8)
//
// Note that the constant term S8 corresponds to the -r³/6 Taylor term;
// the Horner chain evaluates the polynomial from the highest-degree
// coefficient down to S8, which becomes the `+ r * r² * (... coefficient)`
// contribution at the end.

pub const SIN_S0: f64 = -7.972_559_550_090_378_7e-18;
pub const SIN_S1: f64 = 2.810_099_727_108_632e-15;
pub const SIN_S2: f64 = -7.647_122_191_181_588e-13;
pub const SIN_S3: f64 = 1.605_904_306_056_645e-10;
pub const SIN_S4: f64 = -2.505_210_837_635_021e-8;
pub const SIN_S5: f64 = 2.755_731_922_391_987_5e-6;
pub const SIN_S6: f64 = -0.000_198_412_698_412_696_16;
pub const SIN_S7: f64 = 0.008_333_333_333_333_33;
pub const SIN_S8: f64 = -0.166_666_666_666_666_66;

// ---------------------------------------------------------------------------
// cos(r) polynomial — even, factored as 1 + s · Q(s)
// ---------------------------------------------------------------------------
//
// cos(r) ≈ 1 + s · v   where s = r²  and
//   v = ((((((C0·s + C1)·s + C2)·s + C3)·s + C4)·s + C5)·s + C6)
//
// C6 = -1/2 corresponds to the -r²/2 Taylor term; the Horner chain works
// from the highest-degree coefficient down to C6.

pub const COS_C0: f64 = -1.136_153_502_390_974_3e-11;
pub const COS_C1: f64 = 2.087_574_712_070_400_5e-9;
pub const COS_C2: f64 = -2.755_731_440_288_475_7e-7;
pub const COS_C3: f64 = 2.480_158_728_900_018_7e-5;
pub const COS_C4: f64 = -0.001_388_888_888_887_140_2;
pub const COS_C5: f64 = 0.041_666_666_666_666_555;
pub const COS_C6: f64 = -0.5;
