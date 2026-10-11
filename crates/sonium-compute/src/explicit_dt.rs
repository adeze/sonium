//! Critical time step and mass scaling for explicit dynamics.
//!
//! Phase 1 lane 1B kernels: per-element Courant limits with a global
//! min-reduce (serial and `rayon` parallel), uniform and selective mass
//! scaling, and the closed-form central-difference amplification factor.
//!
//! All routines are `f64`, SI units, and slice-based; this module never
//! imports sparse-matrix, FEM, or solver types (P1C2).

// Rust guideline compliant 2026-02-21

use rayon::prelude::*;

/// Stability limit of the undamped central-difference operator.
///
/// The recurrence `u_{n+1} - 2u_n + u_{n-1} = -Ω²u_n` with `Ω = ωΔt` is
/// stable iff `Ω ≤ 2`; at `Ω = 2` the roots coalesce at `A = -1`.
const CD_STABILITY_LIMIT: f64 = 2.0;

/// Sound speed of a 1-D bar.
///
/// Computes `c = √(E/ρ)`.
///
/// # Examples
///
/// ```rust
/// let c = sonium_compute::explicit_dt::sound_speed_bar(200.0e9, 7800.0);
/// assert!((c - 5063.7).abs() < 0.1);
/// ```
///
/// # Panics
///
/// Panics if either argument is not positive and finite.
///
#[must_use]
pub fn sound_speed_bar(youngs_modulus: f64, density: f64) -> f64 {
    assert!(
        youngs_modulus.is_finite() && youngs_modulus > 0.0,
        "Young's modulus must be positive and finite, got {youngs_modulus}"
    );
    assert!(
        density.is_finite() && density > 0.0,
        "density must be positive and finite, got {density}"
    );
    (youngs_modulus / density).sqrt()
}

/// Critical step of one element, `Δt = L_e/c`.
///
/// Asserts its inputs so the min-reduce below never propagates a
/// non-physical quotient silently.
///
/// # Panics
///
/// Panics if the length or sound speed is not positive and finite.
///
fn element_dt(char_length: f64, sound_speed: f64) -> f64 {
    assert!(
        char_length.is_finite() && char_length > 0.0,
        "characteristic length must be positive and finite, got {char_length}"
    );
    assert!(
        sound_speed.is_finite() && sound_speed > 0.0,
        "sound speed must be positive and finite, got {sound_speed}"
    );
    char_length / sound_speed
}

/// Global critical step as the elementwise minimum.
///
/// Returns `min_e (L_e/c_e)` over the mesh. The stable explicit step is
/// a safety factor (typically 0.9) times this value; the factor itself
/// is the caller's policy, not part of this kernel.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::explicit_dt::critical_dt;
///
/// let dt = critical_dt(&[0.01, 0.1, 1.0], &[5000.0, 5000.0, 5000.0]);
/// assert!((dt - 2.0e-6).abs() < 1.0e-18);
/// ```
///
/// # Panics
///
/// Panics if the slices are empty or of unequal length, or if any
/// length or sound speed is not positive and finite.
///
#[must_use]
pub fn critical_dt(char_lengths: &[f64], sound_speeds: &[f64]) -> f64 {
    assert!(
        !char_lengths.is_empty(),
        "critical_dt needs at least one element"
    );
    assert_eq!(
        char_lengths.len(),
        sound_speeds.len(),
        "length/sound-speed slice mismatch: {} vs {}",
        char_lengths.len(),
        sound_speeds.len()
    );
    char_lengths
        .iter()
        .zip(sound_speeds.iter())
        .map(|(&le, &c)| element_dt(le, c))
        .fold(f64::INFINITY, f64::min)
}

/// Global critical step with a parallel min-reduce.
///
/// Same result as [`critical_dt`]; the reduction runs over a `rayon`
/// thread pool. Prefer this once the element count exceeds L2-resident
/// sizes (see the `explicit_dt` bench); below that the serial form wins
/// on dispatch cost.
///
/// # Panics
///
/// Panics under the same conditions as [`critical_dt`].
///
#[must_use]
pub fn critical_dt_parallel(char_lengths: &[f64], sound_speeds: &[f64]) -> f64 {
    assert!(
        !char_lengths.is_empty(),
        "critical_dt_parallel needs at least one element"
    );
    assert_eq!(
        char_lengths.len(),
        sound_speeds.len(),
        "length/sound-speed slice mismatch: {} vs {}",
        char_lengths.len(),
        sound_speeds.len()
    );
    char_lengths
        .par_iter()
        .zip(sound_speeds.par_iter())
        .map(|(&le, &c)| element_dt(le, c))
        .reduce(|| f64::INFINITY, f64::min)
}

/// Modal critical step `Δt = 2/ω_max`.
///
/// Modal form of the same central-difference stability limit behind
/// [`critical_dt`]: a mesh whose highest eigenfrequency is `ω_max` is
/// stable iff `Δt ≤ 2/ω_max`.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::explicit_dt::critical_dt_modal;
///
/// assert!((critical_dt_modal(1000.0) - 0.002).abs() < 1.0e-18);
/// ```
///
/// # Panics
///
/// Panics if `omega_max` is not positive and finite.
///
#[must_use]
pub fn critical_dt_modal(omega_max: f64) -> f64 {
    assert!(
        omega_max.is_finite() && omega_max > 0.0,
        "omega_max must be positive and finite, got {omega_max}"
    );
    CD_STABILITY_LIMIT / omega_max
}

/// Courant number `C = cΔt/L_e` of one element.
///
/// The element is stable iff `C ≤ 1`. The value `1.0` marks the
/// stability boundary exactly.
///
/// # Panics
///
/// Panics if the length or sound speed is not positive and finite, or
/// if `dt` is negative or NaN.
///
#[must_use]
pub fn courant_number(char_length: f64, sound_speed: f64, dt: f64) -> f64 {
    assert!(
        char_length.is_finite() && char_length > 0.0,
        "characteristic length must be positive and finite, got {char_length}"
    );
    assert!(
        sound_speed.is_finite() && sound_speed > 0.0,
        "sound speed must be positive and finite, got {sound_speed}"
    );
    assert!(
        dt >= 0.0 && dt.is_finite(),
        "dt must be non-negative and finite, got {dt}"
    );
    sound_speed * dt / char_length
}

/// Central-difference amplification magnitude and phase.
///
/// For the undamped single-degree-of-freedom oscillator the ansatz
/// `u_n = A^n` in `u_{n+1} - 2u_n + u_{n-1} = -Ω²u_n` gives the two
/// roots `A = 1 - Ω²/2 ± iΩ√(1-Ω²/4)` with `Ω = ωΔt`. This returns
/// `(|A|, arg A)` of the principal root (non-negative imaginary part).
/// Below the stability limit `|A| = 1` (no algorithmic damping) and
/// `arg A` is the numerical phase advance per step, from which the
/// period ratio follows as `T_num/T = Ω/arg A`. Above the limit both
/// roots are real and negative; the dominant (larger-magnitude) root
/// sets `|A| > 1` with phase `π`.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::explicit_dt::cd_amplification;
///
/// let (abs, arg) = cd_amplification(1.0);
/// assert!((abs - 1.0).abs() < 1.0e-15);
/// assert!((arg - std::f64::consts::FRAC_PI_3).abs() < 1.0e-15);
/// ```
///
/// # Panics
///
/// Panics if `omega_dt` is negative, infinite, or NaN.
///
#[must_use]
pub fn cd_amplification(omega_dt: f64) -> (f64, f64) {
    assert!(
        omega_dt.is_finite() && omega_dt >= 0.0,
        "omega_dt must be non-negative and finite, got {omega_dt}"
    );
    if omega_dt <= CD_STABILITY_LIMIT {
        let real = 1.0 - 0.5 * omega_dt * omega_dt;
        let imag = omega_dt * (1.0 - 0.25 * omega_dt * omega_dt).sqrt();
        (real.hypot(imag), imag.atan2(real))
    } else {
        // Both roots real and negative; the dominant root is
        // r = 1 - Ω²/2 - Ω√(Ω²/4-1) with |r| = Ω²/2 - 1 + Ω√(Ω²/4-1).
        let radical = omega_dt * (0.25 * omega_dt * omega_dt - 1.0).sqrt();
        let magnitude = 0.5 * omega_dt * omega_dt - 1.0 + radical;
        (magnitude, std::f64::consts::PI)
    }
}

/// Uniform mass scaling of lumped masses in place.
///
/// Multiplies every entry by `step_scale²`, so each eigenfrequency
/// shifts `ω → ω/step_scale` and the Courant step grows by exactly
/// `step_scale`. The added mass per node is `(s²-1)m`. Masses stay
/// non-negative; zero entries (not yet assembled) are preserved.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::explicit_dt::mass_scale_uniform;
///
/// let mut masses = vec![1.0, 2.0];
/// mass_scale_uniform(&mut masses, 2.0);
/// assert_eq!(masses, vec![4.0, 8.0]);
/// ```
///
/// # Panics
///
/// Panics if `step_scale` is below 1.0 or not finite (scaling down
/// would shrink the stable step), or if any mass is negative or not
/// finite.
///
pub fn mass_scale_uniform(masses: &mut [f64], step_scale: f64) {
    assert!(
        step_scale.is_finite() && step_scale >= 1.0,
        "step_scale must be finite and >= 1.0, got {step_scale}"
    );
    let factor = step_scale * step_scale;
    for mass in masses.iter_mut() {
        assert!(
            mass.is_finite() && *mass >= 0.0,
            "masses must be finite and non-negative, got {mass}"
        );
        *mass *= factor;
    }
}

/// Added mass for one element to reach a target step.
///
/// Selective-mass-scaling hook (AMS style): returns the `Δm ≥ 0` that
/// lifts this element's critical step `L_e/c_e` to `target_dt`, i.e.
/// `Δm = m((target_dt·c_e/L_e)² - 1)`, or `0.0` when the element
/// already satisfies the target. The caller scatters `Δm` to the
/// element's nodes; this kernel only computes the scalar.
///
/// # Panics
///
/// Panics if any argument is not positive and finite, except that a
/// zero element mass is allowed and yields zero added mass.
///
#[must_use]
pub fn selective_added_mass(
    char_length: f64,
    sound_speed: f64,
    element_mass: f64,
    target_dt: f64,
) -> f64 {
    assert!(
        element_mass.is_finite() && element_mass >= 0.0,
        "element_mass must be finite and non-negative, got {element_mass}"
    );
    assert!(
        target_dt.is_finite() && target_dt > 0.0,
        "target_dt must be positive and finite, got {target_dt}"
    );
    let element_step = element_dt(char_length, sound_speed);
    if target_dt <= element_step {
        0.0
    } else {
        let ratio = target_dt / element_step;
        element_mass * (ratio * ratio - 1.0)
    }
}

/// Selective mass scaling over element slices in place.
///
/// Applies [`selective_added_mass`] per element and returns the total
/// added mass. Only sub-target elements gain mass, so the stable step
/// rises toward `target_dt` with minimal inertia distortion. An empty
/// mesh is a valid no-op returning `0.0`.
///
/// # Panics
///
/// Panics if the slices differ in length or any value violates the
/// [`selective_added_mass`] contract.
///
pub fn mass_scale_selective(
    element_masses: &mut [f64],
    char_lengths: &[f64],
    sound_speeds: &[f64],
    target_dt: f64,
) -> f64 {
    assert_eq!(
        element_masses.len(),
        char_lengths.len(),
        "mass/length slice mismatch: {} vs {}",
        element_masses.len(),
        char_lengths.len()
    );
    assert_eq!(
        element_masses.len(),
        sound_speeds.len(),
        "mass/sound-speed slice mismatch: {} vs {}",
        element_masses.len(),
        sound_speeds.len()
    );
    assert!(
        target_dt.is_finite() && target_dt > 0.0,
        "target_dt must be positive and finite, got {target_dt}"
    );
    let mut added_total = 0.0;
    for ((mass, &le), &c) in element_masses
        .iter_mut()
        .zip(char_lengths.iter())
        .zip(sound_speeds.iter())
    {
        let added = selective_added_mass(le, c, *mass, target_dt);
        *mass += added;
        added_total += added;
    }
    added_total
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Steel sound speed (`Sqrt[200*10^9/7800]`, 50-digit reference
    /// `5063.6968354183330813...`; this literal is the nearest `f64`).
    const STEEL_C: f64 = 5063.696835418333;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "actual {actual:.17e} vs expected {expected:.17e}"
        );
    }

    #[test]
    fn sound_speed_matches_steel_reference() {
        let c = sound_speed_bar(200.0e9, 7800.0);
        assert_close(c, STEEL_C, 1.0e-9);
        assert_close(c * c, 200.0e9 / 7800.0, 1.0e-6);
    }

    #[test]
    fn critical_dt_takes_element_minimum() {
        let dt = critical_dt(&[1.0, 0.01, 0.1], &[STEEL_C; 3]);
        assert_close(dt, 0.01 / STEEL_C, 1.0e-21);
    }

    #[test]
    fn critical_dt_parallel_matches_serial() {
        let lengths: Vec<f64> = (1..=4096).map(|i| f64::from(i) * 1.0e-4).collect();
        let speeds: Vec<f64> = (0..4096)
            .map(|i| 4000.0 + f64::from(i % 97) * 13.0)
            .collect();
        assert_eq!(
            critical_dt(&lengths, &speeds),
            critical_dt_parallel(&lengths, &speeds)
        );
    }

    #[test]
    fn modal_dt_is_two_over_omega() {
        assert_close(critical_dt_modal(2500.0), 8.0e-4, 1.0e-18);
    }

    #[test]
    fn courant_number_is_one_at_critical_step() {
        let dt = critical_dt(&[0.05], &[STEEL_C]);
        assert_close(courant_number(0.05, STEEL_C, dt), 1.0, 1.0e-15);
        assert!(courant_number(0.05, STEEL_C, 0.9 * dt) < 1.0);
    }

    #[test]
    fn cd_amplification_unit_gains_below_limit() {
        for &omega in &[0.0, 0.1, 1.0, 1.99, 2.0] {
            let (abs, arg) = cd_amplification(omega);
            assert_close(abs, 1.0, 2.0e-16);
            assert!((0.0..=PI).contains(&arg), "omega {omega}: arg {arg}");
        }
        let (_, arg) = cd_amplification(1.0);
        assert_close(arg, PI / 3.0, 1.0e-15);
        let (_, arg) = cd_amplification(2.0);
        assert_close(arg, PI, 1.0e-15);
    }

    #[test]
    fn cd_amplification_grows_above_limit() {
        let (abs, arg) = cd_amplification(2.01);
        assert!(abs > 1.0, "unstable gain must exceed 1, got {abs}");
        assert_close(abs, 1.221_301_093_164_733, 1.0e-14);
        assert_close(arg, PI, 0.0);
    }

    #[test]
    fn uniform_scaling_squares_masses() {
        let mut masses = vec![1.0, 2.0, 3.0];
        mass_scale_uniform(&mut masses, 2.0);
        assert_eq!(masses, vec![4.0, 8.0, 12.0]);
        mass_scale_uniform(&mut masses, 1.0);
        assert_eq!(masses, vec![4.0, 8.0, 12.0]);
    }

    #[test]
    fn selective_hook_lifts_only_sub_target_elements() {
        // Element step 1 ms; doubling the step needs Δm = 3m.
        assert_close(selective_added_mass(5.0, 5000.0, 4.0, 0.002), 12.0, 1.0e-15);
        assert_eq!(selective_added_mass(5.0, 5000.0, 4.0, 0.001), 0.0);
        assert_eq!(selective_added_mass(5.0, 5000.0, 4.0, 0.0005), 0.0);
    }

    #[test]
    fn selective_scaling_reports_total_added_mass() {
        let mut masses = vec![4.0, 4.0, 4.0];
        let added = mass_scale_selective(&mut masses, &[5.0, 50.0, 500.0], &[5000.0; 3], 0.002);
        // Steps are 1/10/100 ms; only the first element gains 3m.
        assert_close(added, 12.0, 1.0e-15);
        assert_eq!(masses, vec![16.0, 4.0, 4.0]);
        assert_eq!(mass_scale_selective(&mut [], &[], &[], 0.002), 0.0);
    }

    #[test]
    #[should_panic(expected = "at least one element")]
    fn critical_dt_rejects_empty_mesh() {
        let _ = critical_dt(&[], &[]);
    }

    #[test]
    #[should_panic(expected = "slice mismatch")]
    fn critical_dt_rejects_mismatched_slices() {
        let _ = critical_dt(&[0.1, 0.2], &[5000.0]);
    }

    #[test]
    #[should_panic(expected = "positive and finite")]
    fn critical_dt_rejects_nonpositive_length() {
        let _ = critical_dt(&[0.1, -0.2], &[5000.0, 5000.0]);
    }

    #[test]
    #[should_panic(expected = ">= 1.0")]
    fn uniform_scaling_rejects_downscaling() {
        mass_scale_uniform(&mut [1.0], 0.5);
    }

    #[test]
    #[should_panic(expected = "non-negative and finite")]
    fn cd_amplification_rejects_negative_omega() {
        let _ = cd_amplification(-0.1);
    }
}
