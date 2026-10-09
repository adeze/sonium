//! Viscothermal acoustic models for microacoustic FEM analysis
//!
//! Models losses due to viscosity and thermal conduction in the acoustic
//! boundary layers near rigid walls. Two complementary approaches:
//!
//! 1. **Equivalent fluid model** (Zwikker-Kosten / LRF): Replaces the
//!    standard Helmholtz equation with a modified form using complex,
//!    frequency-dependent effective density and bulk modulus. Captures
//!    both viscous and thermal losses in narrow waveguides (tubes, slits).
//!
//! 2. **Boundary layer impedance (BLI)**: Adds a frequency-dependent
//!    surface admittance on rigid walls to account for thermal losses in
//!    the boundary layer. Used for larger volumes where the BL is thin
//!    relative to the geometry.
//!
//! # When to use which
//!
//! - **Narrow channels** (radius/height ≲ 10× the viscous BL thickness
//!   `δ_v = √(2η/ρω)`): LRF equivalent fluid — the BL fills the duct, so a
//!   wall correction cannot capture it.
//! - **Large volumes with thin BL** (`δ_v` ≪ geometry): BLI wall admittance
//!   on top of lossless Helmholtz — far cheaper than meshing the BL.
//! - **Neither limit** (BL comparable to features, separation, turbulence):
//!   full linearized Navier–Stokes
//!   ([`crate::physics::acoustics::aeroacoustics::linearized_navier_stokes`]); the
//!   most expensive option, reserved for cases the two approximations
//!   demonstrably miss.
//!
//! # References
//!
//! - Stinson (1991), "The propagation of plane sound waves in narrow and wide
//!   circular tubes, and generalization to uniform tubes of arbitrary cross-
//!   sectional shape", JASA 89(2).
//! - Beltman (1999), "Viscothermal wave propagation including acousto-elastic
//!   interaction", PhD thesis, University of Twente.
//! - Bruneau (2006), "Fundamentals of Acoustics", ISTE.
//! - Kampinga (2010), "Viscothermal acoustics using finite elements", PhD
//!   thesis, University of Twente.

use crate::forms::{BilinearForm, LocalMatrix};
use crate::spaces::BasisTabulation;
use num_complex::Complex64;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Fluid properties
// ---------------------------------------------------------------------------

/// Thermophysical properties of the acoustic medium.
///
/// All properties are at the static (equilibrium) state.
/// Use [`air_20c`](FluidProperties::air_20c) for standard air.
#[derive(Debug, Clone)]
pub struct FluidProperties {
    /// Static density ρ₀ \[kg/m³\]
    pub density: f64,
    /// Speed of sound c₀ \[m/s\]
    pub speed_of_sound: f64,
    /// Dynamic viscosity μ \[Pa·s\]
    pub dynamic_viscosity: f64,
    /// Thermal conductivity κ \[W/(m·K)\]
    pub thermal_conductivity: f64,
    /// Specific heat at constant pressure C_p \[J/(kg·K)\]
    pub specific_heat_cp: f64,
    /// Ratio of specific heats γ = C_p/C_v \[-\]
    pub ratio_of_specific_heats: f64,
    /// Ambient (static) pressure P₀ \[Pa\]
    pub ambient_pressure: f64,
}

impl FluidProperties {
    /// Standard air at 20 °C, 101.325 kPa.
    pub fn air_20c() -> Self {
        Self {
            density: 1.204,
            speed_of_sound: 343.0,
            dynamic_viscosity: 1.825e-5,
            thermal_conductivity: 0.0257,
            specific_heat_cp: 1005.0,
            ratio_of_specific_heats: 1.400,
            ambient_pressure: 101_325.0,
        }
    }

    /// Prandtl number Pr = μ C_p / κ.
    pub fn prandtl_number(&self) -> f64 {
        self.dynamic_viscosity * self.specific_heat_cp / self.thermal_conductivity
    }

    /// Adiabatic bulk modulus K₀ = γ P₀ \[Pa\].
    pub fn bulk_modulus(&self) -> f64 {
        self.ratio_of_specific_heats * self.ambient_pressure
    }

    /// Viscous boundary layer thickness δ_v = √(2μ / (ρ₀ω)) \[m\].
    ///
    /// # Panics
    /// Panics if `omega <= 0`.
    pub fn viscous_bl_thickness(&self, omega: f64) -> f64 {
        assert!(omega > 0.0, "omega must be positive, got {omega}");
        (2.0 * self.dynamic_viscosity / (self.density * omega)).sqrt()
    }

    /// Thermal boundary layer thickness δ_t = √(2κ / (ρ₀ C_p ω)) \[m\].
    ///
    /// # Panics
    /// Panics if `omega <= 0`.
    pub fn thermal_bl_thickness(&self, omega: f64) -> f64 {
        assert!(omega > 0.0, "omega must be positive, got {omega}");
        (2.0 * self.thermal_conductivity / (self.density * self.specific_heat_cp * omega)).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Cross-section geometry
// ---------------------------------------------------------------------------

/// Cross-section geometry for the equivalent fluid (LRF) model.
#[derive(Debug, Clone, Copy)]
pub enum CrossSection {
    /// Circular tube of given radius \[m\].
    Circular { radius: f64 },
    /// Infinite parallel-plate slit of given half-width \[m\].
    Slit { half_width: f64 },
}

// ---------------------------------------------------------------------------
// Equivalent fluid model
// ---------------------------------------------------------------------------

/// Equivalent fluid model for viscothermal losses in narrow waveguides.
///
/// Computes frequency-dependent complex effective density ρ_eff(ω) and
/// bulk modulus K_eff(ω) using the Low Reduced Frequency (LRF) model
/// (Zwikker-Kosten theory generalised by Stinson).
///
/// The modified Helmholtz equation is:
/// ```text
/// ∇·(1/ρ_eff ∇p) + ω²/K_eff · p = 0
/// ```
///
/// For spatially uniform properties this decomposes as:
/// ```text
/// A = (1/ρ_eff) K − (ω²/K_eff) M
/// ```
/// which fits the [`PrebakedAssembler<2>`](crate::assembly::PrebakedAssembler)
/// framework — no re-assembly of the element matrices is needed across a
/// frequency sweep.
#[derive(Debug, Clone)]
pub struct EquivalentFluid {
    pub fluid: FluidProperties,
    pub cross_section: CrossSection,
}

impl EquivalentFluid {
    pub fn new(fluid: FluidProperties, cross_section: CrossSection) -> Self {
        Self {
            fluid,
            cross_section,
        }
    }

    /// Complex effective density ρ_eff(ω) \[kg/m³\].
    ///
    /// Accounts for viscous losses. Low-frequency limit (wide BL):
    /// Poiseuille-like flow with increased inertia. High-frequency limit
    /// (thin BL): ρ_eff → ρ₀ (inviscid bulk).
    pub fn effective_density(&self, omega: f64) -> Complex64 {
        let f_v = self.viscous_profile_function(omega);
        Complex64::new(self.fluid.density, 0.0) / (Complex64::new(1.0, 0.0) - f_v)
    }

    /// Complex effective bulk modulus K_eff(ω) \[Pa\].
    ///
    /// Accounts for thermal losses. Low-frequency limit (wide BL):
    /// K_eff → P₀ (isothermal). High-frequency limit (thin BL):
    /// K_eff → γP₀ (adiabatic).
    pub fn effective_bulk_modulus(&self, omega: f64) -> Complex64 {
        let gamma = self.fluid.ratio_of_specific_heats;
        let k0 = self.fluid.bulk_modulus();
        let f_t = self.thermal_profile_function(omega);
        Complex64::new(k0, 0.0)
            / (Complex64::new(1.0, 0.0) + Complex64::new(gamma - 1.0, 0.0) * f_t)
    }

    /// Complex propagation wavenumber k_eq(ω) = ω √(ρ_eff / K_eff) \[1/m\].
    ///
    /// `Re(k_eq)` is the phase constant; `Im(k_eq) > 0` is the attenuation
    /// constant (nepers/m).
    pub fn complex_wavenumber(&self, omega: f64) -> Complex64 {
        let rho = self.effective_density(omega);
        let k = self.effective_bulk_modulus(omega);
        Complex64::new(omega, 0.0) * (rho / k).sqrt()
    }

    /// Complex characteristic impedance Z_eq(ω) = √(ρ_eff · K_eff) \[Pa·s/m\].
    pub fn complex_impedance(&self, omega: f64) -> Complex64 {
        let rho = self.effective_density(omega);
        let k = self.effective_bulk_modulus(omega);
        (rho * k).sqrt()
    }

    /// Coefficients for [`PrebakedAssembler<2>`](crate::assembly::PrebakedAssembler):
    /// `[stiffness_coeff, mass_coeff]`.
    ///
    /// The system matrix is assembled as:
    /// `A = coeffs[0] * K + coeffs[1] * M`
    /// where K is the Laplacian stiffness and M the scalar mass matrix.
    pub fn prebaked_coefficients(&self, omega: f64) -> [Complex64; 2] {
        let rho_eff = self.effective_density(omega);
        let k_eff = self.effective_bulk_modulus(omega);
        let inv_rho = Complex64::new(1.0, 0.0) / rho_eff;
        let omega_sq_over_k = Complex64::new(omega * omega, 0.0) / k_eff;
        [inv_rho, -omega_sq_over_k]
    }

    // -- internal profile functions --

    /// Viscous velocity profile function F_v(ω) (Stinson 1991, adapted to e^{−jωt}).
    ///
    /// Circular: F_v = 2 J₁(z) / (z J₀(z)),  z = k_s · a,  k_s = √(jωρ₀/μ)
    /// Slit:     F_v = tanh(z) / z,            z = β · h,    β  = √(−jωρ₀/μ)
    ///
    /// The two geometries use conjugate square roots because the circular
    /// formula arises from J₀ (oscillatory representation) while the slit
    /// formula arises from cosh (exponential representation): β = j k_s.
    fn viscous_profile_function(&self, omega: f64) -> Complex64 {
        let s = self.shear_wave_number(omega);
        match self.cross_section {
            CrossSection::Circular { radius } => {
                let z = Complex64::new(s * radius, 0.0) * sqrt_j();
                let ratio = bessel_j1_over_j0(z);
                Complex64::new(2.0, 0.0) * ratio / z
            }
            CrossSection::Slit { half_width } => {
                let z = Complex64::new(s * half_width, 0.0) * sqrt_neg_j();
                complex_tanh(z) / z
            }
        }
    }

    /// Thermal profile function F_t(ω).
    fn thermal_profile_function(&self, omega: f64) -> Complex64 {
        let s_t = self.thermal_wave_number(omega);
        match self.cross_section {
            CrossSection::Circular { radius } => {
                let z = Complex64::new(s_t * radius, 0.0) * sqrt_j();
                let ratio = bessel_j1_over_j0(z);
                Complex64::new(2.0, 0.0) * ratio / z
            }
            CrossSection::Slit { half_width } => {
                let z = Complex64::new(s_t * half_width, 0.0) * sqrt_neg_j();
                complex_tanh(z) / z
            }
        }
    }

    /// Shear wave number s_v = √(ω ρ₀ / μ) \[1/m\].
    fn shear_wave_number(&self, omega: f64) -> f64 {
        (omega * self.fluid.density / self.fluid.dynamic_viscosity).sqrt()
    }

    /// Thermal wave number s_t = s_v √Pr \[1/m\].
    fn thermal_wave_number(&self, omega: f64) -> f64 {
        self.shear_wave_number(omega) * self.fluid.prandtl_number().sqrt()
    }
}

// ---------------------------------------------------------------------------
// Viscothermal Helmholtz bilinear form (one-shot assembly)
// ---------------------------------------------------------------------------

/// Viscothermal Helmholtz bilinear form for one-shot assembly at a fixed ω.
///
/// Assembles:
/// ```text
/// a(p,q) = (1/ρ_eff) ∫ ∇p·∇q dΩ − (ω²/K_eff) ∫ p q dΩ
/// ```
///
/// For frequency sweeps, prefer the [`PrebakedAssembler`](crate::assembly::PrebakedAssembler)
/// approach via [`EquivalentFluid::prebaked_coefficients`].
#[derive(Debug, Clone)]
pub struct ViscothermalHelmholtzForm {
    /// Coefficient for the stiffness (gradient) term: 1/ρ_eff
    stiffness_coeff: Complex64,
    /// Coefficient for the mass term: −ω²/K_eff (sign included)
    mass_coeff: Complex64,
}

impl ViscothermalHelmholtzForm {
    pub fn new(equivalent_fluid: &EquivalentFluid, omega: f64) -> Self {
        let [stiffness_coeff, mass_coeff] = equivalent_fluid.prebaked_coefficients(omega);
        Self {
            stiffness_coeff,
            mass_coeff,
        }
    }
}

impl BilinearForm<Complex64> for ViscothermalHelmholtzForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<Complex64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j;

        for i in 0..n {
            for j in 0..n {
                let mut grad_dot = 0.0;
                for d in 0..dim {
                    grad_dot += test.grad(i, d) * trial.grad(j, d);
                }
                let mass = test.value(i) * trial.value(j);

                let val = self.stiffness_coeff * Complex64::new(grad_dot * w, 0.0)
                    + self.mass_coeff * Complex64::new(mass * w, 0.0);
                local_matrix.add(i, j, val);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Thermal boundary layer impedance (BLI)
// ---------------------------------------------------------------------------

/// Thermal boundary layer wall admittance coefficient for the Robin BC.
///
/// On a flat, rigid, isothermal wall the thermal boundary layer modifies
/// the hard-wall condition ∂p/∂n = 0 to the impedance BC ∂p/∂n + α p = 0
/// where α is the returned coefficient.
///
/// In the assembled system `A = K − k²M + α B` (or the viscothermal
/// variant), pass this value as the boundary coefficient for the wall tag.
///
/// Derivation follows Kampinga (2010) §3.3 (e^{−jωt} convention):
/// ```text
/// α_th = (γ−1) k² (1−j) δ_t / 2
/// ```
pub fn thermal_wall_admittance(fluid: &FluidProperties, omega: f64) -> Complex64 {
    let gamma = fluid.ratio_of_specific_heats;
    let c0 = fluid.speed_of_sound;
    let k = omega / c0;
    let delta_t = fluid.thermal_bl_thickness(omega);
    let factor = (gamma - 1.0) * k * k * delta_t / 2.0;
    // (1−j) * factor = factor − j*factor
    Complex64::new(factor, -factor)
}

/// Build boundary coefficient map with thermal BLI for the given wall tags.
///
/// All tagged walls receive the same thermal wall admittance at the
/// specified angular frequency. The returned map is directly usable as
/// the `boundary_coeffs` argument of the assembler.
pub fn thermal_bli_boundary_coefficients(
    fluid: &FluidProperties,
    omega: f64,
    wall_tags: &[usize],
) -> HashMap<usize, Complex64> {
    let alpha = thermal_wall_admittance(fluid, omega);
    wall_tags.iter().map(|&tag| (tag, alpha)).collect()
}

// ---------------------------------------------------------------------------
// Complex Bessel function helpers
// ---------------------------------------------------------------------------

/// √(j) = e^{jπ/4} = (1+j)/√2.
///
/// Used for the circular tube profile function where the argument is
/// k_s a with k_s = √(jωρ₀/μ) (shear wavenumber, e^{−jωt} convention).
fn sqrt_j() -> Complex64 {
    Complex64::new(
        std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
    )
}

/// √(−j) = e^{−jπ/4} = (1−j)/√2.
///
/// Used for the slit profile function where the argument is
/// βh with β = √(−jωρ₀/μ) (the cosh/tanh representation, β = jk_s).
fn sqrt_neg_j() -> Complex64 {
    Complex64::new(
        std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    )
}

/// Compute J₁(z)/J₀(z) using Miller's backward recurrence.
///
/// Numerically stable for all complex z. For large |z| in the lower
/// half-plane (our use case with z ∝ √(−j)), falls back to the Hankel
/// function asymptotic J₁/J₀ → −j.
fn bessel_j1_over_j0(z: Complex64) -> Complex64 {
    let z_norm = z.norm();

    if z_norm < 1e-15 {
        return z / 2.0; // Small-argument limit: J₁/J₀ ≈ z/2
    }

    // For large |z|: Hankel function asymptotic.
    // In the upper half-plane (Im(z) > 0, our use case with √(j)):
    //   H₀⁽²⁾ dominates → J₁/J₀ → +j.
    // Error is O(1/|z|), and the profile function F = 2J₁/(zJ₀) → 0 anyway.
    if z_norm > 500.0 {
        return Complex64::new(0.0, 1.0);
    }

    // Backward recurrence: r_{n-1} = 1 / (2n/z − r_n), starting from r_N = 0.
    // Number of steps: ~|z|/2 + margin for convergence.
    let n_terms = (z_norm / 2.0) as usize + 40;

    let mut r = Complex64::new(0.0, 0.0);
    for n in (1..=n_terms).rev() {
        let two_n_over_z = Complex64::new(2.0 * n as f64, 0.0) / z;
        r = Complex64::new(1.0, 0.0) / (two_n_over_z - r);
    }
    r
}

/// Compute tanh(z) for complex z, with overflow protection for large |Re(z)|.
fn complex_tanh(z: Complex64) -> Complex64 {
    if z.re > 350.0 {
        return Complex64::new(1.0, 0.0);
    }
    if z.re < -350.0 {
        return Complex64::new(-1.0, 0.0);
    }
    let e2z = (z * 2.0).exp();
    (e2z - Complex64::new(1.0, 0.0)) / (e2z + Complex64::new(1.0, 0.0))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn air() -> FluidProperties {
        FluidProperties::air_20c()
    }

    // -- Fluid properties --------------------------------------------------

    #[test]
    fn test_air_consistency() {
        let a = air();
        // ρ c² should equal γ P₀ (ideal gas relation)
        let rho_c_sq = a.density * a.speed_of_sound * a.speed_of_sound;
        let gamma_p0 = a.ratio_of_specific_heats * a.ambient_pressure;
        assert!(
            (rho_c_sq / gamma_p0 - 1.0).abs() < 0.01,
            "ρc² = {rho_c_sq}, γP₀ = {gamma_p0}"
        );
    }

    #[test]
    fn test_prandtl_number() {
        let pr = air().prandtl_number();
        assert!((pr - 0.71).abs() < 0.02, "Pr = {pr}, expected ~0.71");
    }

    #[test]
    fn test_boundary_layer_thicknesses() {
        let a = air();
        let omega = 2.0 * std::f64::consts::PI * 1000.0;
        let delta_v = a.viscous_bl_thickness(omega);
        let delta_t = a.thermal_bl_thickness(omega);

        // At 1 kHz: δ_v ≈ 69 μm
        assert!(
            (delta_v * 1e6 - 69.0).abs() < 5.0,
            "δ_v = {:.1} μm",
            delta_v * 1e6
        );
        // δ_t > δ_v since Pr < 1
        assert!(delta_t > delta_v, "δ_t should exceed δ_v for Pr < 1");
        // ratio = 1/√Pr
        let ratio = delta_t / delta_v;
        let expected = 1.0 / a.prandtl_number().sqrt();
        assert!(
            (ratio - expected).abs() < 0.01,
            "δ_t/δ_v = {ratio}, expected {expected}"
        );
    }

    // -- Bessel ratio ------------------------------------------------------

    #[test]
    fn test_bessel_ratio_small_z() {
        // J₁(z)/J₀(z) ≈ z/2 for small z
        let z = Complex64::new(0.01, 0.0);
        let ratio = bessel_j1_over_j0(z);
        assert!(
            (ratio - z / 2.0).norm() < 1e-6,
            "ratio = {ratio}, expected {}",
            z / 2.0
        );
    }

    #[test]
    fn test_bessel_ratio_real_argument() {
        // J₁(1)/J₀(1) = 0.4400506 / 0.7651977 ≈ 0.575081
        let z = Complex64::new(1.0, 0.0);
        let ratio = bessel_j1_over_j0(z);
        assert!(
            (ratio.re - 0.575081).abs() < 1e-4,
            "J₁(1)/J₀(1) = {}",
            ratio.re
        );
        assert!(ratio.im.abs() < 1e-14);

        // J₁(5)/J₀(5) = −0.32758 / −0.17760 ≈ 1.8438
        let z5 = Complex64::new(5.0, 0.0);
        let r5 = bessel_j1_over_j0(z5);
        assert!((r5.re - 1.8438).abs() < 1e-3, "J₁(5)/J₀(5) = {}", r5.re);
    }

    #[test]
    fn test_bessel_ratio_large_argument() {
        // For z with large positive Im (our case with √(j)), J₁/J₀ → +j.
        // F_v = 2 r / z → 0.
        let z = Complex64::new(200.0, 200.0);
        let ratio = bessel_j1_over_j0(z);
        let f_v = Complex64::new(2.0, 0.0) * ratio / z;
        assert!(f_v.norm() < 0.1, "F_v should be small for large z: {f_v}");
    }

    // -- Complex tanh ------------------------------------------------------

    #[test]
    fn test_complex_tanh_zero() {
        assert!(complex_tanh(Complex64::new(0.0, 0.0)).norm() < 1e-15);
    }

    #[test]
    fn test_complex_tanh_large_real() {
        let t = complex_tanh(Complex64::new(10.0, 0.0));
        assert!(
            (t - Complex64::new(1.0, 0.0)).norm() < 1e-8,
            "tanh(10) = {t}"
        );
    }

    #[test]
    fn test_complex_tanh_pure_imaginary() {
        // tanh(j π/4) = j tan(π/4) = j
        let z = Complex64::new(0.0, std::f64::consts::FRAC_PI_4);
        let t = complex_tanh(z);
        assert!(
            (t - Complex64::new(0.0, 1.0)).norm() < 1e-10,
            "tanh(jπ/4) = {t}"
        );
    }

    // -- Equivalent fluid: lossless / wide-tube limits ---------------------

    #[test]
    fn test_equivalent_fluid_lossless_limit() {
        let mut fluid = air();
        fluid.dynamic_viscosity = 1e-30;
        fluid.thermal_conductivity = 1e-30;

        let eq = EquivalentFluid::new(fluid.clone(), CrossSection::Circular { radius: 0.01 });
        let omega = 2.0 * std::f64::consts::PI * 1000.0;

        let rho = eq.effective_density(omega);
        let k_eff = eq.effective_bulk_modulus(omega);

        assert!(
            (rho.re - fluid.density).abs() / fluid.density < 1e-6,
            "ρ_eff = {rho}"
        );
        assert!(rho.im.abs() < 1e-6, "ρ_eff should be real: {rho}");
        assert!(
            (k_eff.re - fluid.bulk_modulus()).abs() / fluid.bulk_modulus() < 1e-6,
            "K_eff = {k_eff}"
        );
        assert!(
            k_eff.im.abs() / fluid.bulk_modulus() < 1e-6,
            "K_eff should be real: {k_eff}"
        );
    }

    #[test]
    fn test_equivalent_fluid_wide_tube() {
        let fluid = air();
        let eq = EquivalentFluid::new(fluid.clone(), CrossSection::Circular { radius: 1.0 });
        let omega = 2.0 * std::f64::consts::PI * 1000.0;

        let rho = eq.effective_density(omega);
        let k_eff = eq.effective_bulk_modulus(omega);

        assert!(
            (rho.re / fluid.density - 1.0).abs() < 1e-3,
            "ρ_eff/ρ₀ = {}",
            rho.re / fluid.density
        );
        assert!(
            (k_eff.re / fluid.bulk_modulus() - 1.0).abs() < 1e-3,
            "K_eff/K₀ = {}",
            k_eff.re / fluid.bulk_modulus()
        );
    }

    // -- Equivalent fluid: narrow tube losses ------------------------------

    #[test]
    fn test_narrow_tube_has_attenuation() {
        let fluid = air();
        let eq = EquivalentFluid::new(fluid.clone(), CrossSection::Circular { radius: 0.5e-3 });
        let omega = 2.0 * std::f64::consts::PI * 100.0;

        let k_eq = eq.complex_wavenumber(omega);
        let k_lossless = omega / fluid.speed_of_sound;

        // Imaginary part (attenuation) should be positive
        assert!(k_eq.im > 0.0, "Im(k_eq) = {}, expected > 0", k_eq.im);
        // Non-negligible attenuation for a 0.5mm tube at 100 Hz
        assert!(
            k_eq.im / k_lossless > 0.01,
            "α/k = {}",
            k_eq.im / k_lossless
        );
        // Phase constant should be close to (but not equal to) k_lossless
        assert!(
            (k_eq.re / k_lossless - 1.0).abs() < 0.5,
            "k_re/k₀ = {}",
            k_eq.re / k_lossless
        );
    }

    #[test]
    fn test_kirchhoff_formula_circular_tube() {
        // Kirchhoff's wide-tube attenuation formula:
        //   α ≈ (1/(a c₀)) √(ν ω/2) [1 + (γ−1)/√Pr]
        let fluid = air();
        let a = 5e-3; // 5 mm — wide enough for Kirchhoff limit
        let f = 1000.0;
        let omega = 2.0 * std::f64::consts::PI * f;

        let eq = EquivalentFluid::new(fluid.clone(), CrossSection::Circular { radius: a });
        let k_eq = eq.complex_wavenumber(omega);

        let nu = fluid.dynamic_viscosity / fluid.density;
        let pr = fluid.prandtl_number();
        let gamma = fluid.ratio_of_specific_heats;
        let c0 = fluid.speed_of_sound;

        let alpha_kirchhoff =
            (1.0 / (a * c0)) * (nu * omega / 2.0).sqrt() * (1.0 + (gamma - 1.0) / pr.sqrt());

        let alpha_model = k_eq.im;

        let rel_error = (alpha_model / alpha_kirchhoff - 1.0).abs();
        assert!(
            rel_error < 0.10,
            "Kirchhoff: {alpha_kirchhoff:.4}, model: {alpha_model:.4}, error: {:.1}%",
            rel_error * 100.0
        );
    }

    // -- Slit geometry -----------------------------------------------------

    #[test]
    fn test_slit_positive_properties() {
        let fluid = air();
        let eq = EquivalentFluid::new(fluid, CrossSection::Slit { half_width: 1e-3 });
        let omega = 2.0 * std::f64::consts::PI * 1000.0;

        let rho = eq.effective_density(omega);
        let k_eff = eq.effective_bulk_modulus(omega);

        assert!(rho.re > 0.0, "Re(ρ_eff) = {}", rho.re);
        assert!(k_eff.re > 0.0, "Re(K_eff) = {}", k_eff.re);
    }

    #[test]
    fn test_slit_has_attenuation() {
        let fluid = air();
        let eq = EquivalentFluid::new(fluid.clone(), CrossSection::Slit { half_width: 0.5e-3 });
        let omega = 2.0 * std::f64::consts::PI * 1000.0;
        let k_eq = eq.complex_wavenumber(omega);

        assert!(
            k_eq.im > 0.0,
            "Slit should have attenuation: Im(k_eq) = {}",
            k_eq.im
        );
    }

    // -- Thermal wall admittance -------------------------------------------

    #[test]
    fn test_thermal_wall_admittance_dissipative() {
        let fluid = air();
        let omega = 2.0 * std::f64::consts::PI * 1000.0;
        let alpha = thermal_wall_admittance(&fluid, omega);

        // Re(α) > 0: adds real damping to the system
        assert!(alpha.re > 0.0, "Re(α) = {}", alpha.re);
        // |α| << k: thermal BLI is a small correction on open walls
        let k = omega / fluid.speed_of_sound;
        assert!(alpha.norm() / k < 0.1, "|α|/k = {}", alpha.norm() / k);
    }

    #[test]
    fn test_thermal_bli_boundary_coefficients() {
        let fluid = air();
        let omega = 2.0 * std::f64::consts::PI * 1000.0;
        let tags = [1, 3, 5];

        let coeffs = thermal_bli_boundary_coefficients(&fluid, omega, &tags);

        assert_eq!(coeffs.len(), 3);
        let expected = thermal_wall_admittance(&fluid, omega);
        for &tag in &tags {
            let c = coeffs[&tag];
            assert!((c - expected).norm() < 1e-15);
        }
    }

    // -- Prebaked coefficients consistency ----------------------------------

    #[test]
    fn test_prebaked_coefficients_consistency() {
        let fluid = air();
        let eq = EquivalentFluid::new(fluid.clone(), CrossSection::Circular { radius: 1e-3 });
        let omega = 2.0 * std::f64::consts::PI * 500.0;

        let [c_k, c_m] = eq.prebaked_coefficients(omega);
        let rho_eff = eq.effective_density(omega);
        let k_eff = eq.effective_bulk_modulus(omega);

        let expected_ck = Complex64::new(1.0, 0.0) / rho_eff;
        let expected_cm = -Complex64::new(omega * omega, 0.0) / k_eff;

        assert!(
            (c_k - expected_ck).norm() < 1e-14,
            "stiffness: {c_k} vs {expected_ck}"
        );
        assert!(
            (c_m - expected_cm).norm() < 1e-14,
            "mass: {c_m} vs {expected_cm}"
        );
    }

    // -- Form assembly consistency -----------------------------------------

    #[test]
    fn test_viscothermal_form_matches_prebaked() {
        use crate::basis::PolynomialDegree;
        use crate::mesh::unit_square_triangles;
        use crate::spaces::{FunctionSpace, H1Space};

        let fluid = air();
        let eq = EquivalentFluid::new(fluid, CrossSection::Circular { radius: 1e-3 });
        let omega = 2.0 * std::f64::consts::PI * 1000.0;

        let form = ViscothermalHelmholtzForm::new(&eq, omega);
        let [c_k, c_m] = eq.prebaked_coefficients(omega);

        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        // Assemble with ViscothermalHelmholtzForm
        let mut mat_vt = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut mat_vt);

        // Assemble stiffness + mass separately and combine
        let mut mat_k = LocalMatrix::zeros(n);
        crate::forms::LaplacianForm.integrate_point(&tab, &tab, 1.0, &mut mat_k);
        let mut mat_m = LocalMatrix::zeros(n);
        crate::forms::MassForm.integrate_point(&tab, &tab, 1.0, &mut mat_m);

        for i in 0..n {
            for j in 0..n {
                let expected = c_k * Complex64::new(mat_k.get(i, j), 0.0)
                    + c_m * Complex64::new(mat_m.get(i, j), 0.0);
                let got = mat_vt.get(i, j);
                assert!(
                    (got - expected).norm() < 1e-14,
                    "({i},{j}): got {got}, expected {expected}"
                );
            }
        }
    }

    // -- Regression: omega=0 must panic, not produce NaN --

    #[test]
    #[should_panic(expected = "omega must be positive")]
    fn test_viscous_bl_panics_at_zero_omega() {
        air().viscous_bl_thickness(0.0);
    }

    #[test]
    #[should_panic(expected = "omega must be positive")]
    fn test_thermal_bl_panics_at_zero_omega() {
        air().thermal_bl_thickness(0.0);
    }

    /// Validate viscothermal attenuation against wide-tube analytical limit.
    ///
    /// For a circular duct at low frequencies (wide boundary layer),
    /// the attenuation coefficient approximates:
    ///   α ≈ (1/(a·c₀)) · √(ω·μ/ρ₀) · [1 + (γ−1)/√Pr]
    ///
    /// Reference: Stinson (1991), Eq. (18).
    #[test]
    fn test_duct_attenuation_matches_wide_tube_limit() {
        let fluid = air();
        let a = 1e-3; // 1 mm radius
        let eq = EquivalentFluid::new(fluid.clone(), CrossSection::Circular { radius: a });
        let c0 = fluid.speed_of_sound;
        let pr = fluid.prandtl_number();
        let gamma = fluid.ratio_of_specific_heats;

        // Low frequency: 100 Hz (wide BL)
        let f = 100.0;
        let omega = 2.0 * std::f64::consts::PI * f;
        let k_eq = eq.complex_wavenumber(omega);
        let alpha_num = k_eq.im; // numerical attenuation [nepers/m]

        // Wide-tube analytical approximation
        let alpha_analytical = (1.0 / (a * c0))
            * (omega * fluid.dynamic_viscosity / fluid.density).sqrt()
            * (1.0 + (gamma - 1.0) / pr.sqrt());

        // Allow 20% error in the wide-tube limit (it's an approximation)
        let rel_err = (alpha_num - alpha_analytical).abs() / alpha_analytical;
        assert!(
            rel_err < 0.20,
            "Wide-tube attenuation: numerical={alpha_num:.4e}, analytical={alpha_analytical:.4e}, rel_err={rel_err:.2}"
        );
    }

    /// Validate physical properties of viscothermal attenuation.
    #[test]
    fn test_attenuation_physical_properties() {
        let fluid = air();
        let a = 1e-3;
        let eq = EquivalentFluid::new(fluid.clone(), CrossSection::Circular { radius: a });

        let freqs = [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0];

        for f in freqs {
            let omega = 2.0 * std::f64::consts::PI * f;
            let k_eq = eq.complex_wavenumber(omega);
            let alpha = k_eq.im;

            // Attenuation must be positive
            assert!(alpha > 0.0, "Attenuation must be positive at {f} Hz");

            // Phase velocity must be positive and finite
            let phase_vel = omega / k_eq.re;
            assert!(
                phase_vel > 0.0 && phase_vel.is_finite(),
                "Phase velocity must be positive and finite at {f} Hz"
            );

            // Phase velocity must be <= speed of sound (dispersion slows the wave)
            assert!(
                phase_vel <= fluid.speed_of_sound * 1.01,
                "Phase velocity {phase_vel} exceeds c₀ {} at {f} Hz",
                fluid.speed_of_sound
            );
        }
    }

    /// Validate that the characteristic impedance is physically reasonable.
    #[test]
    fn test_characteristic_impedance_physical() {
        let fluid = air();
        let eq = EquivalentFluid::new(fluid.clone(), CrossSection::Circular { radius: 1e-3 });

        // At high frequency, Z_eq → ρ₀·c₀ (inviscid limit)
        let omega_high = 2.0 * std::f64::consts::PI * 10000.0;
        let z_high = eq.complex_impedance(omega_high);
        let z_inviscid = fluid.density * fluid.speed_of_sound;

        let rel_err = (z_high.re - z_inviscid).abs() / z_inviscid;
        assert!(
            rel_err < 0.05,
            "High-freq impedance should approach inviscid limit: Z={z_high:.2e}, Z₀={z_inviscid:.2e}"
        );
    }
}
