//! Propeller noise source model
//!
//! Models the acoustic signature of a propeller as tonal components at
//! the blade-passing frequency (BPF) and its harmonics. The source
//! amplitudes are computed using Gutin's formula (thickness + loading noise).
//!
//! For use with the waveguide port infrastructure, the propeller source
//! is converted to modal excitation coefficients at the duct inlet.
//!
//! # References
//!
//! - Gutin (1948), "On the sound field of a rotating propeller"
//! - Lowson (1965), "The sound field for singularities in motion"
//! - Hanson (1980), "Helicoidal surface theory for harmonic noise of propellers"

use std::f64::consts::PI;

/// Propeller noise source specification.
#[derive(Debug, Clone)]
pub struct PropellerSource {
    /// Number of blades
    pub n_blades: usize,
    /// Rotational speed [rev/min]
    pub rpm: f64,
    /// Total thrust [N]
    pub thrust: f64,
    /// Total torque [N·m]
    pub torque: f64,
    /// Tip radius [m]
    pub tip_radius: f64,
    /// Hub radius [m]
    pub hub_radius: f64,
    /// Speed of sound [m/s]
    pub speed_of_sound: f64,
    /// Air density [kg/m³]
    pub density: f64,
}

impl PropellerSource {
    /// Create a typical small drone propeller source.
    ///
    /// Default values approximate a 5-inch (0.127m) drone propeller
    /// at hover conditions.
    pub fn drone_5inch(rpm: f64) -> Self {
        Self {
            n_blades: 2,
            rpm,
            thrust: 3.0,  // ~300g hover thrust
            torque: 0.05, // estimated
            tip_radius: 0.0635,
            hub_radius: 0.008,
            speed_of_sound: 343.0,
            density: 1.204,
        }
    }

    /// Blade-passing frequency [Hz].
    pub fn blade_passing_frequency(&self) -> f64 {
        self.n_blades as f64 * self.rpm.abs() / 60.0
    }

    /// Angular velocity [rad/s].
    pub fn omega(&self) -> f64 {
        2.0 * PI * self.rpm / 60.0
    }

    /// Tip Mach number.
    pub fn tip_mach(&self) -> f64 {
        self.omega().abs() * self.tip_radius / self.speed_of_sound
    }

    /// Frequency of the n-th harmonic of the BPF [Hz].
    ///
    /// `harmonic = 1` is the fundamental BPF, `harmonic = 2` is the first overtone, etc.
    pub fn harmonic_frequency(&self, harmonic: usize) -> f64 {
        harmonic as f64 * self.blade_passing_frequency()
    }

    /// Peak acoustic pressure phasor [Pa] for each BPF harmonic at 1 m,
    /// using a compact rotating-loading source model.
    ///
    /// `p_n(r) = ω_n · F_load / (4π·c·r) · |J_{nB}(nB·M_t)|`
    ///
    /// `F_load = hypot(T, Q/R)` combines the axial thrust and equivalent
    /// tangential force from torque. The returned value is a reference
    /// magnitude; downstream callers apply observer-direction directivity.
    ///
    /// At low tip Mach (M_t ≪ 1), `J_{nB}(nB·M_t)` follows the small-argument
    /// asymptote `(nB·M_t/2)^{nB} / (nB)!` — the `(nB)!` factorial decay is
    /// what the previous empirical `M_t^n` model omitted, leading it to
    /// over-predict harmonic content by many orders of magnitude in the
    /// PC-fan / shrouded-axial regime (M_t ~ 0.01).
    pub fn harmonic_amplitudes(&self, n_harmonics: usize) -> Vec<f64> {
        self.harmonic_amplitudes_at_distance(n_harmonics, 1.0)
    }

    /// Peak harmonic pressure phasors [Pa] at `distance_m`.
    ///
    /// The loading-dipole factor is `ω F/(c r)`: it has pressure units and
    /// is independent of density when thrust and torque are prescribed as
    /// forces. The previous implementation used `ω²/(ρ c³)`, which had
    /// units of inverse seconds rather than pascals.
    pub fn harmonic_amplitudes_at_distance(&self, n_harmonics: usize, distance_m: f64) -> Vec<f64> {
        let b = self.n_blades as f64;
        let omega = self.omega().abs();
        let m_tip = self.tip_mach();
        let radius = self.tip_radius.abs().max(1.0e-12);
        let r_ref = distance_m.abs().max(1.0e-12);
        let loading_force = self.thrust.abs().hypot(self.torque.abs() / radius);
        let base = loading_force / (4.0 * PI * self.speed_of_sound * r_ref);

        (1..=n_harmonics)
            .map(|n| {
                let n_f = n as f64;
                let nb = n_f * b;
                let nb_int = (nb as usize).max(1);

                let loading = base * nb * omega;
                // Bessel J_{nB}(nB·M_t·sinθ) at θ = π/2.
                let bessel = bessel_jn_small_argument(nb_int, nb * m_tip);
                loading * bessel.abs()
            })
            .collect()
    }

    /// Radiated sound power [W] associated with each peak-pressure phasor.
    ///
    /// The reference pressure is interpreted with unit-mean directivity, so
    /// `W = 4πr² p_peak² / (2ρc)` and is independent of the chosen reference
    /// radius.
    pub fn harmonic_sound_powers(&self, n_harmonics: usize) -> Vec<f64> {
        let impedance = (self.density * self.speed_of_sound).max(1.0e-30);
        self.harmonic_amplitudes(n_harmonics)
            .into_iter()
            .map(|peak_pressure| 4.0 * PI * peak_pressure * peak_pressure / (2.0 * impedance))
            .collect()
    }

    /// Peak pressure phasor [Pa] of an equivalent plane wave at a duct
    /// inlet with area `inlet_area_m2`.
    ///
    /// Half of the free-field source power is coupled into the forward duct:
    /// `p_peak = sqrt(2ρc · (W/2) / A)`. This preserves acoustic power instead
    /// of applying a 1 m far-field pressure directly at the rotor plane.
    pub fn inlet_pressure_amplitude_for_area(&self, harmonic: usize, inlet_area_m2: f64) -> f64 {
        if harmonic == 0 || inlet_area_m2 <= 0.0 {
            return 0.0;
        }
        let power_w = self.harmonic_sound_powers(harmonic)[harmonic - 1];
        (self.density * self.speed_of_sound * power_w / inlet_area_m2)
            .max(0.0)
            .sqrt()
    }

    /// Convert the n-th harmonic to a uniform pressure excitation amplitude [Pa]
    /// at the duct inlet cross-section.
    ///
    /// This is the simplest source model: the propeller radiates a plane wave
    /// into the duct with amplitude proportional to the Gutin harmonic level.
    /// For higher-order modes, use `to_modal_excitation` with a `WaveguidePort`.
    pub fn inlet_pressure_amplitude(&self, harmonic: usize) -> f64 {
        let annulus_area = PI * (self.tip_radius.powi(2) - self.hub_radius.powi(2)).max(0.0);
        self.inlet_pressure_amplitude_for_area(harmonic, annulus_area)
    }

    /// Azimuthal order of the n-th rotor-alone harmonic (`m = nB`).
    ///
    /// Isolated-rotor harmonics spin with the blade count times the
    /// harmonic index; rotor-stator stages instead radiate the
    /// Tyler-Sofrin orders from [`interaction_mode_order`].
    pub fn rotor_alone_order(&self, harmonic: usize) -> i32 {
        (harmonic.saturating_mul(self.n_blades)) as i32
    }

    /// Polar directivity of one harmonic, normalized to the disc plane.
    ///
    /// Returns `J_{|m|}(nB·M_t·sinθ) / J_{|m|}(nB·M_t)` with `θ` measured
    /// from the rotation axis: unity in the disc plane (`θ = π/2`),
    /// matching [`PropellerSource::harmonic_amplitudes`], with the
    /// on-axis null of spinning modes.
    pub fn harmonic_directivity(&self, harmonic: usize, polar_angle_rad: f64) -> f64 {
        let helix = harmonic as f64 * self.n_blades as f64 * self.tip_mach();
        spinning_mode_directivity(self.rotor_alone_order(harmonic), helix, polar_angle_rad)
    }

    /// Peak pressure phasor [Pa] of one harmonic at distance and polar angle.
    ///
    /// Scales the disc-plane [`PropellerSource::harmonic_amplitudes_at_distance`]
    /// level by [`PropellerSource::harmonic_directivity`]. Returns zero for
    /// the zeroth harmonic.
    ///
    /// The lumped loading force follows the rotating in-plane (torque)
    /// Bessel pattern; a pure-thrust axial dipole additionally carries a
    /// `cosθ` obliquity that stays caller-side, consistent with the
    /// reference-magnitude contract of
    /// [`PropellerSource::harmonic_amplitudes_at_distance`].
    pub fn harmonic_amplitude_at_direction(
        &self,
        harmonic: usize,
        distance_m: f64,
        polar_angle_rad: f64,
    ) -> f64 {
        if harmonic == 0 {
            return 0.0;
        }
        self.harmonic_amplitudes_at_distance(harmonic, distance_m)[harmonic - 1]
            * self.harmonic_directivity(harmonic, polar_angle_rad)
    }
}

/// Tyler-Sofrin interaction order for one stator-lobe count.
///
/// Returns `m = nB − kV` for `B` rotor blades, `V` stator vanes, `n`-th
/// blade-passing harmonic, and `k` stator lobes.
pub fn interaction_mode_order(
    blades: usize,
    vanes: usize,
    harmonic: usize,
    stator_lobes: usize,
) -> i32 {
    (harmonic.saturating_mul(blades)) as i32 - (stator_lobes.saturating_mul(vanes)) as i32
}

/// Polar directivity of one rotor-stator interaction mode.
///
/// Returns `(m, directivity)` with the Tyler-Sofrin order and the
/// spinning-mode directivity at the rotor helix argument `nB·M_t`.
/// Spinning modes (`m ≠ 0`) vanish on the axis; the cut-on/cut-off
/// screening of these orders lives in [`crate::physics::acoustics::aeroacoustics::duct_modes`].
pub fn interaction_mode_directivity(
    blades: usize,
    vanes: usize,
    harmonic: usize,
    stator_lobes: usize,
    tip_mach: f64,
    polar_angle_rad: f64,
) -> (i32, f64) {
    let order = interaction_mode_order(blades, vanes, harmonic, stator_lobes);
    let helix = harmonic as f64 * blades as f64 * tip_mach.abs();
    (
        order,
        spinning_mode_directivity(order, helix, polar_angle_rad),
    )
}

/// Normalized polar directivity of a spinning tonal mode.
///
/// Returns `J_{|m|}(z·sinθ) / J_{|m|}(z)` for azimuthal order `m`, helix
/// argument `z`, and polar angle `θ` from the axis: unity in the disc
/// plane, an on-axis null for spinning modes, and omnidirectional unit
/// response in the compact (`z → 0`, `m = 0`) limit. Near an in-plane
/// null the small-argument shape `sinθ^|m|` keeps the ratio bounded.
pub fn spinning_mode_directivity(
    azimuthal_order: i32,
    helix_argument: f64,
    polar_angle_rad: f64,
) -> f64 {
    use spec_math::Bessel as _;
    if !helix_argument.is_finite() || !polar_angle_rad.is_finite() {
        return 0.0;
    }
    let order = azimuthal_order.unsigned_abs();
    let sine = polar_angle_rad.sin().abs();
    let reference = helix_argument.bessel_jv(f64::from(order));
    if reference.abs() <= 1e-15 {
        return sine.powi(order as i32);
    }
    (helix_argument * sine).bessel_jv(f64::from(order)) / reference
}

/// Bessel `J_n(x)` evaluated via the small-argument power series:
///
/// `J_n(x) = Σ_{k=0..} (−1)^k / (k! · (n+k)!) · (x/2)^{n+2k}`
///
/// Accurate to machine precision when `|x| ≤ n + 1` (the range that
/// covers all rotor-noise applications at M_tip up to the tonal cut-on).
/// Outside that band the series is still convergent but the leading-order
/// term ceases to dominate; for low-Mach rotor noise we never reach that
/// regime so the simple series is fine. Returns 0 for `n == 0` with `x = 0`
/// (which would be `J_0(0) = 1` — but this branch is not hit by the
/// loading-noise path because `nB ≥ 1`).
fn bessel_jn_small_argument(n: usize, x: f64) -> f64 {
    if n == 0 {
        // J_0(x) series: Σ (−1)^k / (k!)² · (x/2)^{2k}
        let half_x_sq = 0.25 * x * x;
        let mut term = 1.0_f64;
        let mut sum = term;
        for k in 1..50 {
            term *= -half_x_sq / ((k as f64) * (k as f64));
            sum += term;
            if term.abs() < 1.0e-18 * sum.abs().max(1.0) {
                break;
            }
        }
        return sum;
    }
    let half_x = 0.5 * x;
    let half_x_sq = half_x * half_x;
    // Leading term: (x/2)^n / n! Compute via running product to avoid
    // overflow in factorial when n is large (n ≤ 64 covers nB up to 8
    // blades × 8 harmonics).
    let mut term = 1.0_f64;
    for k in 1..=n {
        term *= half_x / (k as f64);
    }
    let mut sum = term;
    for k in 1..50 {
        // T_{k+1}/T_k = −(x/2)² / (k · (n+k))
        term *= -half_x_sq / ((k as f64) * ((n + k) as f64));
        sum += term;
        if term.abs() < 1.0e-18 * sum.abs().max(1.0e-300) {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpf_calculation() {
        let prop = PropellerSource::drone_5inch(10000.0);
        // 2 blades × 10000 RPM / 60 = 333.3 Hz
        let bpf = prop.blade_passing_frequency();
        assert!((bpf - 333.333).abs() < 0.01, "BPF = {bpf}");
    }

    #[test]
    fn test_harmonics_decay() {
        let prop = PropellerSource::drone_5inch(8000.0);
        let amps = prop.harmonic_amplitudes(5);

        assert_eq!(amps.len(), 5);
        // Each successive harmonic should be smaller
        for i in 1..amps.len() {
            assert!(
                amps[i] < amps[i - 1],
                "Harmonic {} ({:.4e}) should be < harmonic {} ({:.4e})",
                i + 1,
                amps[i],
                i,
                amps[i - 1]
            );
        }
    }

    #[test]
    fn test_tip_mach() {
        let prop = PropellerSource::drone_5inch(10000.0);
        let m = prop.tip_mach();
        // Ω = 2π × 10000/60 ≈ 1047 rad/s, r = 0.0635m → v_tip ≈ 66.5 m/s → M ≈ 0.194
        assert!((m - 0.194).abs() < 0.01, "Tip Mach = {m}");
    }

    #[test]
    fn test_bessel_j0_known_values() {
        // Reference values: A&S Table 9.1.
        let j0_at_1 = bessel_jn_small_argument(0, 1.0);
        assert!(
            (j0_at_1 - 0.7651976866).abs() < 1.0e-9,
            "J_0(1) = {j0_at_1}"
        );
        let j0_at_25 = bessel_jn_small_argument(0, 2.5);
        assert!(
            (j0_at_25 - (-0.0483837764)).abs() < 1.0e-9,
            "J_0(2.5) = {j0_at_25}"
        );
    }

    #[test]
    fn test_bessel_jn_known_values() {
        // J_2(1) ≈ 0.114903484931900.
        let j2_at_1 = bessel_jn_small_argument(2, 1.0);
        assert!(
            (j2_at_1 - 0.1149034849).abs() < 1.0e-9,
            "J_2(1) = {j2_at_1}"
        );
        // J_5(1) ≈ 2.497577e-4.
        let j5_at_1 = bessel_jn_small_argument(5, 1.0);
        assert!(
            (j5_at_1 - 2.49757730e-4).abs() < 1.0e-12,
            "J_5(1) = {j5_at_1}"
        );
    }

    #[test]
    fn test_bessel_jn_small_argument_matches_leading_order() {
        // For x ≪ n, J_n(x) ≈ (x/2)^n / n! to high precision.
        let x = 0.1_f64;
        let n = 8_usize;
        let leading = (x / 2.0).powi(n as i32) / (1..=n).map(|k| k as f64).product::<f64>();
        let bessel = bessel_jn_small_argument(n, x);
        let rel = (bessel - leading).abs() / leading.max(1.0e-300);
        assert!(
            rel < 1.0e-3,
            "rel error {rel}, leading {leading}, bessel {bessel}"
        );
    }

    #[test]
    fn test_harmonic_amplitudes_decay_faster_than_legacy_at_low_mach() {
        // At low tip Mach the Lowson Bessel form decays as ~(M_t·nB/2)^nB / (nB)!,
        // dramatically faster than the legacy `M_t^n` empirical decay. This
        // is a regression check that the upgrade actually changed the model.
        let mut prop = PropellerSource::drone_5inch(2_000.0); // M_tip ≈ 0.039
        prop.n_blades = 8; // PC-fan-style blade count
        let amps = prop.harmonic_amplitudes(3);
        // Strict monotone decay survives.
        assert!(
            amps[1] < amps[0],
            "h2 {} should be < h1 {}",
            amps[1],
            amps[0]
        );
        assert!(
            amps[2] < amps[1],
            "h3 {} should be < h2 {}",
            amps[2],
            amps[1]
        );
        // Each successive ratio should be much smaller than the legacy
        // empirical model's M_t ≈ 0.039 ratio.
        let ratio_h2 = amps[1] / amps[0].max(1.0e-300);
        let m_tip = prop.tip_mach();
        assert!(
            ratio_h2 < m_tip,
            "Bessel-form harmonic ratio {ratio_h2:.3e} should be < legacy M_t {m_tip:.3e}"
        );
    }

    #[test]
    fn test_loading_pressure_has_physical_force_and_distance_scaling() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let at_one_metre = source.harmonic_amplitudes_at_distance(1, 1.0)[0];
        let at_two_metres = source.harmonic_amplitudes_at_distance(1, 2.0)[0];
        assert!((at_one_metre / at_two_metres - 2.0).abs() < 1.0e-12);

        let mut doubled_loading = source.clone();
        doubled_loading.thrust *= 2.0;
        doubled_loading.torque *= 2.0;
        let doubled = doubled_loading.harmonic_amplitudes(1)[0];
        assert!((doubled / at_one_metre - 2.0).abs() < 1.0e-12);

        let mut different_density = source;
        different_density.density *= 3.0;
        let density_invariant = different_density.harmonic_amplitudes(1)[0];
        assert!((density_invariant / at_one_metre - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_frequency_is_nonnegative_for_reverse_rotation() {
        let source = PropellerSource::drone_5inch(-10_000.0);
        assert!(source.blade_passing_frequency() > 0.0);
        assert!(source.harmonic_frequency(2) > 0.0);
        assert!(source.tip_mach() > 0.0);
    }

    #[test]
    fn test_spinning_directivity_is_unity_in_plane_with_on_axis_null() {
        // Rotor-alone m = nB = 2 at drone tip Mach.
        let prop = PropellerSource::drone_5inch(10_000.0);
        let in_plane = prop.harmonic_directivity(1, PI / 2.0);
        assert!((in_plane - 1.0).abs() < 1e-15, "in-plane {in_plane}");
        let on_axis = prop.harmonic_directivity(1, 0.0);
        assert!(on_axis.abs() < 1e-12, "on-axis {on_axis}");
        // Compact axisymmetric mode is omnidirectional.
        assert!((spinning_mode_directivity(0, 0.0, 0.3) - 1.0).abs() < 1e-15);
        assert!((spinning_mode_directivity(0, 0.0, 2.1) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_directional_amplitude_matches_disc_plane_level() {
        let prop = PropellerSource::drone_5inch(18_000.0);
        let disc = prop.harmonic_amplitudes_at_distance(2, 3.0)[1];
        let directed = prop.harmonic_amplitude_at_direction(2, 3.0, PI / 2.0);
        assert!((directed / disc - 1.0).abs() < 1e-15);
        assert_eq!(prop.harmonic_amplitude_at_direction(0, 3.0, PI / 2.0), 0.0);
        // Amplitude falls toward the axis for spinning harmonics.
        let near_axis = prop.harmonic_amplitude_at_direction(2, 3.0, 0.2);
        assert!(near_axis < directed);
    }

    #[test]
    fn test_interaction_orders_follow_tyler_sofrin() {
        assert_eq!(interaction_mode_order(9, 10, 1, 0), 9);
        assert_eq!(interaction_mode_order(9, 10, 1, 1), -1);
        assert_eq!(interaction_mode_order(9, 10, 1, 2), -11);
        let (order, directivity) = interaction_mode_directivity(9, 10, 1, 1, 0.1, PI / 2.0);
        assert_eq!(order, -1);
        assert!((directivity - 1.0).abs() < 1e-15);
        let (_, on_axis) = interaction_mode_directivity(9, 10, 1, 1, 0.1, 0.0);
        assert!(on_axis.abs() < 1e-12);
    }

    #[test]
    fn test_inlet_pressure_conserves_forward_sound_power() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let area = PI * 0.07_f64.powi(2);
        let peak_pressure = source.inlet_pressure_amplitude_for_area(1, area);
        let duct_power =
            peak_pressure * peak_pressure * area / (2.0 * source.density * source.speed_of_sound);
        let radiated_power = source.harmonic_sound_powers(1)[0];
        assert!((duct_power / radiated_power - 0.5).abs() < 1.0e-12);

        let larger_area_pressure = source.inlet_pressure_amplitude_for_area(1, 4.0 * area);
        assert!((peak_pressure / larger_area_pressure - 2.0).abs() < 1.0e-12);
    }
}
