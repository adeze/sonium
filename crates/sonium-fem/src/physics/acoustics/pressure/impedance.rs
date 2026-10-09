//! Impedance models for porous and absorbing materials
//!
//! Provides empirical models that compute the characteristic impedance Z_c
//! and propagation constant k_c of porous materials, from which the surface
//! impedance of a backed layer and the corresponding Robin BC coefficient
//! are derived.
//!
//! # References
//!
//! - Delany & Bazley (1970), "Acoustical properties of fibrous absorbent
//!   materials", Applied Acoustics 3(2).
//! - Miki (1990), "Acoustical properties of porous materials — Modifications
//!   of Delany-Bazley models", J. Acoust. Soc. Jpn. 11(1).
//! - Johnson, Koplik & Dashen (1987), J. Fluid Mech. 176;
//!   Champoux & Allard (1991), J. Appl. Phys. 70(4) — the JCA model.
//!   All models here use the e^{−jωt} convention (passive: Re(Z_c) > 0,
//!   Im(k_c) > 0).

use num_complex::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Porous material models
// ---------------------------------------------------------------------------

/// Delany-Bazley empirical model for fibrous porous materials.
///
/// Valid range: 0.01 < ρ₀f/σ < 1.0 (flow resistivity 1000–50000 Pa·s/m²).
/// Adapted to the e^{−jωt} convention (positive Im for attenuation).
#[derive(Debug, Clone, Copy)]
pub struct DelanyBazley {
    /// Flow resistivity σ \[Pa·s/m²\]
    pub flow_resistivity: f64,
}

impl DelanyBazley {
    pub fn new(flow_resistivity: f64) -> Self {
        Self { flow_resistivity }
    }
}

/// Miki correction to the Delany-Bazley model.
///
/// Provides better physical behaviour: Re(Z_c) > 0 and Im(k_c) > 0
/// are guaranteed across the full frequency range.
///
/// Reference: Miki (1990).
#[derive(Debug, Clone, Copy)]
pub struct Miki {
    /// Flow resistivity σ \[Pa·s/m²\]
    pub flow_resistivity: f64,
}

impl Miki {
    pub fn new(flow_resistivity: f64) -> Self {
        Self { flow_resistivity }
    }
}

/// Johnson–Champoux–Allard semi-phenomenological model for rigid-frame
/// porous materials.
///
/// Unlike the one-parameter Delany–Bazley/Miki fits (fibrous absorbers
/// only), JCA takes five microstructure parameters — flow resistivity σ,
/// open porosity φ, tortuosity α∞, and the viscous/thermal characteristic
/// lengths Λ, Λ′ — and covers foams and felts as well as fibres.
///
/// Effective density (Johnson 1987) and bulk modulus (Champoux & Allard
/// 1991), written here with `s = −jω` for the crate's e^{−jωt} convention
/// (the textbook form uses `+jω` under e^{+jωt}):
///
/// ```text
/// ρ_eq = (α∞·ρ₀/φ)·(1 + σφ/(s·ρ₀α∞)·√(1 + s·4α∞²ηρ₀/(σ²Λ²φ²)))
/// K_eq = (γP₀/φ)/(γ − (γ−1)/(1 + 8η/(s·Pr·ρ₀Λ′²)·√(1 + s·ρ₀PrΛ′²/(16η))))
/// ```
///
/// with air viscosity η, Prandtl number Pr, ratio of specific heats γ, and
/// ambient pressure `P₀ = ρ₀c₀²/γ`. `Z_c = √(ρ_eq·K_eq)`,
/// `k_c = ω·√(ρ_eq/K_eq)`.
///
/// # References
///
/// - Johnson, Koplik & Dashen (1987), "Theory of dynamic permeability and
///   tortuosity in fluid-saturated porous media", J. Fluid Mech. 176.
/// - Champoux & Allard (1991), "Dynamic tortuosity and bulk modulus in
///   air-saturated porous media", J. Appl. Phys. 70(4).
/// - Allard & Atalla, "Propagation of Sound in Porous Media" (2e), Ch. 5–6;
///   Cox & D'Antonio, "Acoustic Absorbers and Diffusers" (3e), Eqs. 6.19–6.25.
#[derive(Debug, Clone, Copy)]
pub struct JohnsonChampouxAllard {
    /// Static airflow resistivity σ \[Pa·s/m²\]
    pub flow_resistivity: f64,
    /// Open porosity φ ∈ (0, 1\]
    pub porosity: f64,
    /// Tortuosity α∞ ≥ 1 (α∞ = 1 for straight cylindrical pores)
    pub tortuosity: f64,
    /// Viscous characteristic length Λ \[m\]
    pub viscous_length: f64,
    /// Thermal characteristic length Λ′ \[m\] (Λ′ ≥ Λ for most materials)
    pub thermal_length: f64,
    /// Air dynamic viscosity η \[Pa·s\] (1.84e−5 at 20 °C)
    pub dynamic_viscosity: f64,
    /// Air Prandtl number (≈ 0.71)
    pub prandtl: f64,
    /// Ratio of specific heats (≈ 1.4 for air)
    pub specific_heat_ratio: f64,
}

impl JohnsonChampouxAllard {
    /// Build with standard-air thermal constants (η = 1.84e−5 Pa·s,
    /// Pr = 0.71, γ = 1.4).
    pub fn new(
        flow_resistivity: f64,
        porosity: f64,
        tortuosity: f64,
        viscous_length: f64,
        thermal_length: f64,
    ) -> Self {
        Self {
            flow_resistivity,
            porosity,
            tortuosity,
            viscous_length,
            thermal_length,
            dynamic_viscosity: 1.84e-5,
            prandtl: 0.71,
            specific_heat_ratio: 1.4,
        }
    }

    fn check_params(&self) {
        assert!(self.flow_resistivity > 0.0, "σ must be positive");
        assert!(
            self.porosity > 0.0 && self.porosity <= 1.0,
            "φ must lie in (0, 1]"
        );
        assert!(self.tortuosity >= 1.0, "α∞ must be ≥ 1");
        assert!(self.viscous_length > 0.0, "Λ must be positive");
        assert!(self.thermal_length > 0.0, "Λ′ must be positive");
    }

    /// Complex effective density ρ_eq(f) \[kg/m³\].
    pub fn effective_density(&self, f: f64, rho: f64) -> Complex64 {
        assert!(f > 0.0, "frequency must be positive, got {f}");
        self.check_params();
        let omega = 2.0 * PI * f;
        // s = −jω: the e^{−jωt} counterpart of the textbook +jω.
        let s = Complex64::new(0.0, -omega);
        let sigma = self.flow_resistivity;
        let phi = self.porosity;
        let alpha = self.tortuosity;
        let lambda = self.viscous_length;
        let eta = self.dynamic_viscosity;
        let static_term = sigma * phi / (s * rho * alpha);
        let shape = 4.0 * alpha * alpha * eta * rho / (sigma * sigma * lambda * lambda * phi * phi);
        (alpha * rho / phi) * (1.0 + static_term * (1.0 + s * shape).sqrt())
    }

    /// Complex effective bulk modulus K_eq(f) \[Pa\].
    pub fn effective_bulk_modulus(&self, f: f64, rho: f64, c: f64) -> Complex64 {
        assert!(f > 0.0, "frequency must be positive, got {f}");
        self.check_params();
        let omega = 2.0 * PI * f;
        let s = Complex64::new(0.0, -omega);
        let gamma = self.specific_heat_ratio;
        let p0 = rho * c * c / gamma;
        let eta = self.dynamic_viscosity;
        let prandtl = self.prandtl;
        let lambda_p = self.thermal_length;
        let static_term = 8.0 * eta / (s * prandtl * rho * lambda_p * lambda_p);
        let shape = rho * prandtl * lambda_p * lambda_p / (16.0 * eta);
        let denom = gamma - (gamma - 1.0) / (1.0 + static_term * (1.0 + s * shape).sqrt());
        (gamma * p0 / self.porosity) / denom
    }
}

impl PorousMaterialModel for JohnsonChampouxAllard {
    fn characteristic_impedance(&self, f: f64, rho: f64, c: f64) -> Complex64 {
        (self.effective_density(f, rho) * self.effective_bulk_modulus(f, rho, c)).sqrt()
    }

    fn propagation_constant(&self, f: f64, rho: f64, c: f64) -> Complex64 {
        let omega = 2.0 * PI * f;
        omega * (self.effective_density(f, rho) / self.effective_bulk_modulus(f, rho, c)).sqrt()
    }
}

/// Common interface for porous material models.
pub trait PorousMaterialModel {
    /// Characteristic impedance Z_c(f) \[Pa·s/m\].
    fn characteristic_impedance(&self, f: f64, rho: f64, c: f64) -> Complex64;

    /// Complex propagation constant k_c(f) \[1/m\].
    fn propagation_constant(&self, f: f64, rho: f64, c: f64) -> Complex64;
}

impl PorousMaterialModel for DelanyBazley {
    fn characteristic_impedance(&self, f: f64, rho: f64, c: f64) -> Complex64 {
        assert!(f > 0.0, "frequency must be positive, got {f}");
        let x = rho * f / self.flow_resistivity;
        let z0 = rho * c;
        Complex64::new(1.0 + 0.0571 * x.powf(-0.754), 0.0870 * x.powf(-0.732)) * z0
    }

    fn propagation_constant(&self, f: f64, rho: f64, c: f64) -> Complex64 {
        assert!(f > 0.0, "frequency must be positive, got {f}");
        let k0 = 2.0 * PI * f / c;
        let x = rho * f / self.flow_resistivity;
        Complex64::new(1.0 + 0.0978 * x.powf(-0.700), 0.1890 * x.powf(-0.595)) * k0
    }
}

impl PorousMaterialModel for Miki {
    fn characteristic_impedance(&self, f: f64, rho: f64, c: f64) -> Complex64 {
        assert!(f > 0.0, "frequency must be positive, got {f}");
        let x = rho * f / self.flow_resistivity;
        let z0 = rho * c;
        Complex64::new(1.0 + 0.0699 * x.powf(-0.632), 0.1070 * x.powf(-0.632)) * z0
    }

    fn propagation_constant(&self, f: f64, rho: f64, c: f64) -> Complex64 {
        assert!(f > 0.0, "frequency must be positive, got {f}");
        let k0 = 2.0 * PI * f / c;
        let x = rho * f / self.flow_resistivity;
        Complex64::new(1.0 + 0.1600 * x.powf(-0.618), 0.1090 * x.powf(-0.618)) * k0
    }
}

// ---------------------------------------------------------------------------
// Surface impedance of a backed layer
// ---------------------------------------------------------------------------

/// Compute the surface impedance of a porous layer of thickness `d` backed
/// by a rigid wall.
///
/// ```text
/// Z_s = j Z_c cot(k_c d)
/// ```
///
/// Returns the specific surface impedance Z_s \[Pa·s/m\].
pub fn surface_impedance_rigid_backed<M: PorousMaterialModel>(
    model: &M,
    f: f64,
    thickness: f64,
    rho: f64,
    c: f64,
) -> Complex64 {
    let z_c = model.characteristic_impedance(f, rho, c);
    let k_c = model.propagation_constant(f, rho, c);
    let j = Complex64::new(0.0, 1.0);
    j * z_c * complex_cot(k_c * thickness)
}

/// Convert a surface impedance Z_s to the Robin BC coefficient α for the
/// Helmholtz assembler.
///
/// The Robin BC is: `∂p/∂n + α p = 0`, and the assembler system is
/// `A = K − k²M + α B`. The coefficient is:
///
/// ```text
/// α = -i k ρ₀ c₀ / Z_s
/// ```
///
/// For perfect absorption (`Z_s = ρ₀c₀`), `α = -ik` (Sommerfeld ABC under
/// Sonium's `exp(-iωt)` convention).
pub fn impedance_to_robin_coefficient(
    surface_impedance: Complex64,
    k: f64,
    rho: f64,
    c: f64,
) -> Complex64 {
    let minus_i = Complex64::new(0.0, -1.0);
    minus_i * k * rho * c / surface_impedance
}

/// Convenience: compute the Robin BC coefficient directly from a porous
/// material model, layer thickness, and frequency.
pub fn porous_layer_robin_coefficient<M: PorousMaterialModel>(
    model: &M,
    f: f64,
    thickness: f64,
    rho: f64,
    c: f64,
) -> Complex64 {
    let z_s = surface_impedance_rigid_backed(model, f, thickness, rho, c);
    let k = 2.0 * PI * f / c;
    impedance_to_robin_coefficient(z_s, k, rho, c)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Complex cotangent: cot(z) = j(e^{2jz} + 1)/(e^{2jz} − 1).
fn complex_cot(z: Complex64) -> Complex64 {
    let j = Complex64::new(0.0, 1.0);
    let two_jz = 2.0 * j * z;
    // Overflow protection: Re(2jz) = −2 Im(z)
    if two_jz.re > 350.0 {
        return j;
    }
    if two_jz.re < -350.0 {
        return -j;
    }
    let e = two_jz.exp();
    j * (e + 1.0) / (e - 1.0)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const RHO: f64 = 1.204;
    const C: f64 = 343.0;

    #[test]
    fn test_delany_bazley_impedance_positive_real() {
        let db = DelanyBazley::new(10_000.0);
        let z_c = db.characteristic_impedance(1000.0, RHO, C);
        // Real part > ρ₀c₀ (always greater than free-field impedance)
        assert!(z_c.re > RHO * C, "Re(Z_c) = {}", z_c.re);
        // Imaginary part > 0 for e^{-jωt} convention
        assert!(z_c.im > 0.0, "Im(Z_c) = {}", z_c.im);
    }

    #[test]
    fn test_delany_bazley_propagation_attenuation() {
        let db = DelanyBazley::new(10_000.0);
        let k_c = db.propagation_constant(1000.0, RHO, C);
        let k0 = 2.0 * PI * 1000.0 / C;
        // Real part > k₀ (slower phase velocity in porous medium)
        assert!(k_c.re > k0, "Re(k_c)/k₀ = {}", k_c.re / k0);
        // Imaginary part > 0 (attenuation)
        assert!(k_c.im > 0.0, "Im(k_c) = {}", k_c.im);
    }

    #[test]
    fn test_miki_vs_delany_bazley() {
        let db = DelanyBazley::new(20_000.0);
        let mk = Miki::new(20_000.0);
        let z_db = db.characteristic_impedance(500.0, RHO, C);
        let z_mk = mk.characteristic_impedance(500.0, RHO, C);
        // Both should be similar but not identical
        let rel_diff = (z_db - z_mk).norm() / z_db.norm();
        assert!(rel_diff < 0.3, "Models differ by {:.0}%", rel_diff * 100.0);
        // Miki should also have positive real and imaginary parts
        assert!(z_mk.re > 0.0);
        assert!(z_mk.im > 0.0);
    }

    #[test]
    fn test_thick_layer_approaches_characteristic_impedance() {
        let db = DelanyBazley::new(10_000.0);
        let z_c = db.characteristic_impedance(1000.0, RHO, C);
        // Very thick layer: surface impedance → Z_c
        let z_s = surface_impedance_rigid_backed(&db, 1000.0, 1.0, RHO, C);
        let rel_error = (z_s - z_c).norm() / z_c.norm();
        assert!(
            rel_error < 0.05,
            "Z_s = {z_s}, Z_c = {z_c}, error = {:.1}%",
            rel_error * 100.0
        );
    }

    #[test]
    fn test_robin_coefficient_perfect_absorption() {
        let z_s = Complex64::new(RHO * C, 0.0); // Perfect absorption
        let f = 1000.0;
        let k = 2.0 * PI * f / C;
        let alpha = impedance_to_robin_coefficient(z_s, k, RHO, C);
        // Under exp(-iωt), perfect absorption is α = -ik.
        let expected = Complex64::new(0.0, -k);
        assert!(
            (alpha - expected).norm() < 1e-10,
            "α = {alpha}, expected -ik = {expected}"
        );
    }

    #[test]
    fn test_robin_coefficient_rigid_wall() {
        // Z_s → ∞: α → 0 (rigid wall, no absorption)
        let z_s = Complex64::new(1e10, 0.0);
        let alpha = impedance_to_robin_coefficient(z_s, 18.3, RHO, C);
        assert!(alpha.norm() < 1e-4, "|α| = {}", alpha.norm());
    }

    #[test]
    fn test_complex_cot_special_values() {
        // cot(π/4) = 1
        let result = complex_cot(Complex64::new(PI / 4.0, 0.0));
        assert!(
            (result.re - 1.0).abs() < 1e-10 && result.im.abs() < 1e-10,
            "cot(π/4) = {result}"
        );
    }

    #[test]
    #[should_panic(expected = "frequency must be positive")]
    fn test_delany_bazley_panics_at_zero_freq() {
        DelanyBazley::new(10_000.0).characteristic_impedance(0.0, RHO, C);
    }

    #[test]
    #[should_panic(expected = "frequency must be positive")]
    fn test_miki_panics_at_zero_freq() {
        Miki::new(10_000.0).propagation_constant(0.0, RHO, C);
    }

    /// Canonical JCA specimen used across these tests: the comparison point
    /// from the phonometry porous-absorber guide (σ = 20 kPa·s/m², φ = 0.98,
    /// α∞ = 1, Λ = Λ′ = 87 µm), itself reading Allard & Atalla data.
    fn canonical_jca() -> JohnsonChampouxAllard {
        JohnsonChampouxAllard::new(20_000.0, 0.98, 1.0, 8.7e-5, 8.7e-5)
    }

    #[test]
    fn test_jca_matches_independent_oracle() {
        // Independent oracle: the phonometry guide evaluates this exact
        // specimen at f = 200 Hz and prints Z_c/(ρ₀c₀) = (2.321 − 2.075j)
        // under e^{+jωt}, i.e. the complex conjugate (2.321 + 2.075j)
        // under this crate's e^{−jωt} convention. Tolerance 2e−3 absolute
        // covers their 3-decimal print rounding with margin.
        let jca = canonical_jca();
        let z_norm = jca.characteristic_impedance(200.0, RHO, C) / (RHO * C);
        assert!(
            (z_norm.re - 2.321).abs() < 2e-3,
            "Re(Z_c/z0) = {}, expected 2.321",
            z_norm.re
        );
        assert!(
            (z_norm.im - 2.075).abs() < 2e-3,
            "Im(Z_c/z0) = {}, expected +2.075",
            z_norm.im
        );
    }

    #[test]
    fn test_jca_agrees_with_miki_at_oracle_point() {
        // Same specimen, same frequency: the in-crate Miki fit and JCA both
        // model fibrous absorbers, so their characteristic impedances must
        // agree closely (the oracle prints 2.286−1.965j vs 2.321−2.075j,
        // ≈3% apart). This cross-checks the JCA wiring against the
        // long-standing closed-form Miki implementation.
        let jca = canonical_jca();
        let miki = Miki::new(20_000.0);
        let z_jca = jca.characteristic_impedance(200.0, RHO, C);
        let z_miki = miki.characteristic_impedance(200.0, RHO, C);
        let rel = (z_jca - z_miki).norm() / z_miki.norm();
        // Loose by design: published Miki coefficient variants differ by
        // ~10% among themselves (the oracle's Miki prints 2.286−1.965j vs
        // 2.141+1.747j from the in-crate variant). This guards the wiring,
        // the oracle test above guards the absolute level.
        assert!(rel < 0.15, "JCA/Miki relative gap = {rel:.4}");
    }

    #[test]
    fn test_jca_high_frequency_limits() {
        // ρ_eq → α∞ρ₀/φ and K_eq → γP₀/φ as f → ∞ (inertial/adiabatic
        // limits). Corrections decay as 1/√ω (thermal relaxation is the
        // slower one, ~0.9% at 2 MHz), so the probe sits at 100 MHz where
        // both are ~1e−3. This is a formula-asymptote check, not a
        // physical operating point.
        let jca = canonical_jca();
        let rho_eq = jca.effective_density(1.0e8, RHO);
        let k_eq = jca.effective_bulk_modulus(1.0e8, RHO, C);
        let rho_lim = 1.0 * RHO / 0.98;
        let k_lim = 1.4 * (RHO * C * C / 1.4) / 0.98;
        assert!(
            (rho_eq - rho_lim).norm() / rho_lim < 5e-3,
            "ρ_eq = {rho_eq}, limit {rho_lim}"
        );
        assert!(
            (k_eq - k_lim).norm() / k_lim < 5e-3,
            "K_eq = {k_eq}, limit {k_lim}"
        );
    }

    #[test]
    fn test_jca_dc_flow_resistivity_limit() {
        // jωρ_eq → σ at DC (the CJCA guide states jωρ_e → σ; under e^{−jωt}
        // the counterpart is (−jω)·ρ_eq → σ).
        let jca = canonical_jca();
        let omega = 2.0 * PI * 0.01;
        let rho_eq = jca.effective_density(0.01, RHO);
        let s_rho = Complex64::new(0.0, -omega) * rho_eq;
        let rel = (s_rho - Complex64::new(20_000.0, 0.0)).norm() / 20_000.0;
        assert!(rel < 0.01, "(−jω)ρ_eq = {s_rho}, expected σ");
    }

    #[test]
    fn test_jca_passive_across_audio_band() {
        // Passivity under e^{−jωt}: Re(Z_c) > 0 and Im(k_c) > 0 from 20 Hz
        // to 20 kHz for the canonical specimen.
        let jca = canonical_jca();
        for f in [20.0, 50.0, 100.0, 200.0, 500.0, 1_000.0, 5_000.0, 20_000.0] {
            let z_c = jca.characteristic_impedance(f, RHO, C);
            let k_c = jca.propagation_constant(f, RHO, C);
            assert!(z_c.re > 0.0, "f = {f}: Re(Z_c) = {}", z_c.re);
            assert!(k_c.im > 0.0, "f = {f}: Im(k_c) = {}", k_c.im);
        }
    }

    #[test]
    fn test_jca_backed_layer_absorbs() {
        // A 50 mm rigid-backed layer of the canonical specimen at 500 Hz
        // must present a passive surface impedance (Re(Z_s) > 0).
        let jca = canonical_jca();
        let z_s = surface_impedance_rigid_backed(&jca, 500.0, 0.05, RHO, C);
        assert!(z_s.re > 0.0, "Re(Z_s) = {}", z_s.re);
        assert!(z_s.is_finite(), "Z_s = {z_s}");
    }
}
