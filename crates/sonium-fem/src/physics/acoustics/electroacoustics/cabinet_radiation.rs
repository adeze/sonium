//! Simplified cabinet panel radiation model
//!
//! Estimates the acoustic radiation from vibrating cabinet walls (box coloration)
//! using analytical rectangular plate mode shapes. Each wall is modeled as a
//! simply-supported rectangular plate driven by the internal cavity pressure.
//!
//! # Model
//!
//! For each panel (wall) and each mode (m, n):
//!
//! 1. **Mode frequency**: `f_mn = (c_plate/2) · √((m/a)² + (n/b)²)`
//!    where `c_plate = √(D/(ρ_s·h))` is the plate bending wave speed,
//!    `D = E·h³/(12(1-ν²))` is flexural rigidity.
//!
//! 2. **Modal force**: uniform pressure loading excites only odd-odd modes:
//!    `F_mn = p_int · (4ab) / (mnπ²)` for odd m,n; zero for even.
//!
//! 3. **Modal displacement**: `X_mn = F_mn / (M_mn · ((ω_mn² - ω²) + jω·η·ω_mn))`
//!    where `M_mn = ρ_s·h·a·b/4` and η is the structural loss factor.
//!
//! 4. **Radiation**: each mode radiates as a baffled piston with effective area
//!    dependent on the mode shape's net volume displacement.
//!
//! # Limitations
//!
//! - Simply-supported boundary conditions (real cabinets are between SS and clamped)
//! - No coupling between panels (each wall radiates independently)
//! - Interior pressure assumed uniform (valid below first cavity mode)
//! - Radiation efficiency approximated (low-frequency piston model)

use std::f64::consts::PI;

use num_complex::Complex64;

/// Material properties for a cabinet panel.
#[derive(Debug, Clone, Copy)]
pub struct PanelMaterial {
    /// Young's modulus (Pa). MDF: ~3.5 GPa, plywood: ~10 GPa.
    pub youngs_modulus: f64,
    /// Poisson's ratio. MDF: ~0.25, plywood: ~0.3.
    pub poissons_ratio: f64,
    /// Material density (kg/m³). MDF: ~750, plywood: ~600.
    pub density: f64,
    /// Panel thickness (m).
    pub thickness: f64,
    /// Structural loss factor η (dimensionless). MDF: ~0.1, plywood: ~0.02.
    pub loss_factor: f64,
}

impl PanelMaterial {
    /// Standard 18mm MDF.
    pub fn mdf_18mm() -> Self {
        Self {
            youngs_modulus: 3.5e9,
            poissons_ratio: 0.25,
            density: 750.0,
            thickness: 0.018,
            loss_factor: 0.10,
        }
    }

    /// Standard 18mm birch plywood.
    pub fn plywood_18mm() -> Self {
        Self {
            youngs_modulus: 10.0e9,
            poissons_ratio: 0.30,
            density: 600.0,
            thickness: 0.018,
            loss_factor: 0.02,
        }
    }

    /// Flexural rigidity D = E·h³ / (12·(1 - ν²)) (N·m).
    pub fn flexural_rigidity(&self) -> f64 {
        let h = self.thickness;
        self.youngs_modulus * h * h * h / (12.0 * (1.0 - self.poissons_ratio.powi(2)))
    }

    /// Surface density ρ_s·h (kg/m²).
    pub fn surface_density(&self) -> f64 {
        self.density * self.thickness
    }

    /// Critical frequency: where bending wavelength equals acoustic wavelength.
    /// `f_c = c₀² / (π·h) · √(12·ρ·(1-ν²) / E)`
    pub fn critical_frequency(&self, speed_of_sound: f64) -> f64 {
        let c0 = speed_of_sound;
        let h = self.thickness;
        let rho = self.density;
        let e = self.youngs_modulus;
        let nu = self.poissons_ratio;
        c0 * c0 / (PI * h) * (12.0 * rho * (1.0 - nu * nu) / e).sqrt()
    }
}

/// A rectangular cabinet panel.
#[derive(Debug, Clone)]
pub struct CabinetPanel {
    /// Panel width (m) — the shorter dimension.
    pub width: f64,
    /// Panel height (m) — the longer dimension.
    pub height: f64,
    /// Material properties.
    pub material: PanelMaterial,
}

/// Mode shape result for a single panel mode.
#[derive(Debug, Clone)]
pub struct PanelMode {
    /// Mode indices (m, n), both ≥ 1.
    pub m: usize,
    pub n: usize,
    /// Natural frequency (Hz).
    pub frequency: f64,
    /// Modal mass (kg).
    pub modal_mass: f64,
    /// Modal force coefficient: F_mn = coeff × p_internal.
    /// Zero for even mode indices.
    pub force_coefficient: f64,
}

/// Modal displacement of one excited panel mode at one frequency.
///
/// `displacement_m` is the complex modal amplitude `X_mn` of the mode-shape
/// expansion `w(x, y) = Σ X_mn · sin(mπx/a)·sin(nπy/b)`. It carries the same
/// amplitude convention as the pressure passed to
/// [`CabinetPanel::modal_displacements`] (RMS pressure → RMS displacement).
/// Modes with zero force coefficient (even m or n under uniform pressure) are
/// omitted.
#[derive(Debug, Clone, Copy)]
pub struct PanelModeDisplacement {
    /// Mode indices (m, n), both ≥ 1 and odd for uniform pressure loading.
    pub m: usize,
    pub n: usize,
    /// Complex modal displacement amplitude (m).
    pub displacement_m: Complex64,
}

impl CabinetPanel {
    /// Compute the natural frequency of mode (m, n) for a simply-supported plate.
    ///
    /// `f_mn = (π/(2·a·b)) · √(D/ρ_s) · (m²·b² + n²·a²) / (a·b)`
    ///
    /// Simplified: `f_mn = π/(2·ρ_s) · D^(1/2) · ((m/a)² + (n/b)²)`
    pub fn mode_frequency(&self, m: usize, n: usize) -> f64 {
        let d = self.material.flexural_rigidity();
        let rho_s = self.material.surface_density();
        let a = self.width;
        let b = self.height;

        let km = m as f64 * PI / a;
        let kn = n as f64 * PI / b;
        let omega_sq = d / rho_s * (km * km + kn * kn).powi(2);
        omega_sq.sqrt() / (2.0 * PI)
    }

    /// List all modes below a maximum frequency, sorted by frequency.
    pub fn modes_below(&self, max_freq: f64, max_order: usize) -> Vec<PanelMode> {
        let a = self.width;
        let b = self.height;
        let rho_s = self.material.surface_density();
        let modal_mass = rho_s * a * b / 4.0; // same for all modes (simply supported)

        let mut modes = Vec::new();
        for m in 1..=max_order {
            for n in 1..=max_order {
                let f = self.mode_frequency(m, n);
                if f > max_freq {
                    continue;
                }

                // Force coefficient for uniform pressure loading:
                // F_mn = (4·a·b)/(m·n·π²) for odd m AND odd n, else 0
                let force_coeff = if m % 2 == 1 && n % 2 == 1 {
                    4.0 * a * b / (m as f64 * n as f64 * PI * PI)
                } else {
                    0.0
                };

                modes.push(PanelMode {
                    m,
                    n,
                    frequency: f,
                    modal_mass,
                    force_coefficient: force_coeff,
                });
            }
        }
        modes.sort_by(|a, b| a.frequency.total_cmp(&b.frequency));
        modes
    }

    /// Compute the panel's modal displacements under internal pressure.
    ///
    /// Returns the complex modal amplitude `X_mn` of every excited mode (odd
    /// m,n only) at `frequency_hz`:
    /// `X_mn = F_mn / (M_mn · (ω_mn² - ω² + j·η·ω·ω_mn))`.
    ///
    /// The returned phasors inherit the amplitude convention of `pressure`
    /// (RMS pressure → RMS displacement, peak pressure → peak displacement).
    ///
    /// Modes with natural frequency up to `max_mode_frequency_hz` (and order up
    /// to `max_mode_order`) are included. Unlike the radiation path, a
    /// displacement metric must retain modes well above the drive frequency:
    /// they carry the quasi-static (stiffness-controlled) deflection shape.
    pub fn modal_displacements(
        &self,
        frequency_hz: f64,
        pressure: Complex64,
        max_mode_frequency_hz: f64,
        max_mode_order: usize,
    ) -> Vec<PanelModeDisplacement> {
        let omega = 2.0 * PI * frequency_hz;
        let eta = self.material.loss_factor;
        let modes = self.modes_below(max_mode_frequency_hz, max_mode_order);

        modes
            .iter()
            .filter(|mode| mode.force_coefficient.abs() >= 1e-30)
            .map(|mode| {
                let omega_mn = 2.0 * PI * mode.frequency;
                let f_mn = mode.force_coefficient * pressure;
                let denom =
                    Complex64::new(omega_mn * omega_mn - omega * omega, eta * omega * omega_mn);
                let x_mn = f_mn / (Complex64::new(mode.modal_mass, 0.0) * denom);
                PanelModeDisplacement {
                    m: mode.m,
                    n: mode.n,
                    displacement_m: x_mn,
                }
            })
            .collect()
    }

    /// Compute the panel's displacement response to internal pressure.
    ///
    /// Returns the RMS spatially-averaged surface displacement (m) for a given
    /// internal pressure amplitude (Pa) — the displacement analogue of
    /// [`CabinetPanel::surface_velocity`], with the same modal radiation
    /// factor `(4ab/(mnπ²))/(ab)` weighting. `max_mode_frequency_hz` selects
    /// the retained modes; pass at least ~10× the panel's fundamental for a
    /// converged quasi-static deflection. For the peak displacement at the
    /// panel antinode (the quantity a displacement limit constrains), sum the
    /// complex [`PanelModeDisplacement`] values with their center sign
    /// `sin(mπ/2)·sin(nπ/2)` instead.
    pub fn surface_displacement(
        &self,
        frequency: f64,
        pressure_amplitude: f64,
        max_mode_frequency_hz: f64,
        max_mode_order: usize,
    ) -> f64 {
        let area = self.width * self.height;
        let x_sq_sum: f64 = self
            .modal_displacements(
                frequency,
                Complex64::new(pressure_amplitude, 0.0),
                max_mode_frequency_hz,
                max_mode_order,
            )
            .iter()
            .map(|mode| {
                // Same weighting as surface_velocity: the odd-odd net-volume
                // factor (4ab/(mnπ²)) normalized by the panel area.
                let force_coeff = 4.0 * area / (mode.m as f64 * mode.n as f64 * PI * PI);
                mode.displacement_m.norm_sqr() * (force_coeff / area).powi(2)
            })
            .sum();
        x_sq_sum.sqrt()
    }

    /// Compute the panel's velocity response to internal pressure at a given frequency.
    ///
    /// Returns the RMS surface velocity (m/s) for a given internal pressure amplitude (Pa).
    /// This sums contributions from all modes below the evaluation frequency × 3.
    pub fn surface_velocity(
        &self,
        frequency: f64,
        pressure_amplitude: f64,
        max_mode_order: usize,
    ) -> f64 {
        let omega = 2.0 * PI * frequency;
        let area = self.width * self.height;
        let v_sq_sum: f64 = self
            .modal_displacements(
                frequency,
                Complex64::new(pressure_amplitude, 0.0),
                frequency * 3.0,
                max_mode_order,
            )
            .iter()
            .map(|mode| {
                let v_mn = mode.displacement_m * Complex64::new(0.0, omega);
                // Each mode's average velocity contribution (RMS over the panel area)
                // For simply-supported modes: <sin²(mπx/a)·sin²(nπy/b)> = 1/4
                // But only the net volume displacement matters for radiation.
                // For odd-odd modes: net volume velocity = (4ab/(mnπ²)) × V_mn_peak
                // We use the peak modal velocity × modal radiation factor
                let force_coeff = 4.0 * area / (mode.m as f64 * mode.n as f64 * PI * PI);
                v_mn.norm_sqr() * (force_coeff / area).powi(2)
            })
            .sum();
        v_sq_sum.sqrt()
    }

    /// Radiated sound power from the panel at a given frequency.
    ///
    /// Uses the radiation efficiency of a baffled piston with the panel's area.
    /// At low frequencies (ka << 1): σ ≈ (ka)²/2 where a is the effective radius.
    ///
    /// Returns power in watts for the given internal pressure amplitude.
    pub fn radiated_power(
        &self,
        frequency: f64,
        pressure_amplitude: f64,
        density: f64,
        speed_of_sound: f64,
        max_mode_order: usize,
    ) -> f64 {
        let v_rms = self.surface_velocity(frequency, pressure_amplitude, max_mode_order);
        let area = self.width * self.height;

        // Radiation efficiency: simplified model
        // σ ≈ min(1, (f/f_c)²) where f_c is the critical frequency
        let f_c = self.material.critical_frequency(speed_of_sound);
        let sigma = (frequency / f_c).powi(2).min(1.0);

        // Radiated power: P = ρ₀·c₀·σ·S·<v²>
        density * speed_of_sound * sigma * area * v_rms * v_rms
    }

    /// SPL contribution from the panel relative to a reference level (dB).
    ///
    /// Returns 10·log10(W_panel) + constant, useful for comparing panel
    /// contributions at different frequencies.
    pub fn radiated_spl_relative(
        &self,
        frequency: f64,
        pressure_amplitude: f64,
        density: f64,
        speed_of_sound: f64,
        max_mode_order: usize,
    ) -> f64 {
        let power = self.radiated_power(
            frequency,
            pressure_amplitude,
            density,
            speed_of_sound,
            max_mode_order,
        );
        if power > 1e-30 {
            10.0 * power.log10()
        } else {
            -200.0
        }
    }
}

/// A complete cabinet with multiple panels.
#[derive(Debug, Clone)]
pub struct Cabinet {
    /// All cabinet panels (typically 5: front baffle is open for the driver).
    pub panels: Vec<CabinetPanel>,
}

impl Cabinet {
    /// Create a rectangular sealed cabinet (5 panels, front baffle excluded).
    ///
    /// `width`, `height`, `depth` are external dimensions (m).
    /// Panel thickness is taken from the material.
    pub fn sealed_box(width: f64, height: f64, depth: f64, material: PanelMaterial) -> Self {
        let t = material.thickness;
        // Internal dimensions
        let w_int = width - 2.0 * t;
        let h_int = height - 2.0 * t;
        let d_int = depth - 2.0 * t;

        Self {
            panels: vec![
                // Top and bottom
                CabinetPanel {
                    width: w_int,
                    height: d_int,
                    material,
                },
                CabinetPanel {
                    width: w_int,
                    height: d_int,
                    material,
                },
                // Left and right sides
                CabinetPanel {
                    width: d_int,
                    height: h_int,
                    material,
                },
                CabinetPanel {
                    width: d_int,
                    height: h_int,
                    material,
                },
                // Back panel
                CabinetPanel {
                    width: w_int,
                    height: h_int,
                    material,
                },
            ],
        }
    }

    /// Total radiated power from all panels at a given frequency and internal pressure.
    pub fn total_radiated_power(
        &self,
        frequency: f64,
        pressure_amplitude: f64,
        density: f64,
        speed_of_sound: f64,
        max_mode_order: usize,
    ) -> f64 {
        self.panels
            .iter()
            .map(|p| {
                p.radiated_power(
                    frequency,
                    pressure_amplitude,
                    density,
                    speed_of_sound,
                    max_mode_order,
                )
            })
            .sum()
    }

    /// SPL contribution from all panels combined (dB re 1 pW).
    pub fn total_radiated_spl(
        &self,
        frequency: f64,
        pressure_amplitude: f64,
        density: f64,
        speed_of_sound: f64,
        max_mode_order: usize,
    ) -> f64 {
        let power = self.total_radiated_power(
            frequency,
            pressure_amplitude,
            density,
            speed_of_sound,
            max_mode_order,
        );
        if power > 1e-30 {
            10.0 * power.log10()
        } else {
            -200.0
        }
    }

    /// Compute panel radiation SPL vs frequency sweep.
    pub fn radiation_vs_frequency(
        &self,
        frequencies: &[f64],
        pressure_amplitude: f64,
        density: f64,
        speed_of_sound: f64,
        max_mode_order: usize,
    ) -> Vec<(f64, f64)> {
        frequencies
            .iter()
            .map(|&f| {
                let spl = self.total_radiated_spl(
                    f,
                    pressure_amplitude,
                    density,
                    speed_of_sound,
                    max_mode_order,
                );
                (f, spl)
            })
            .collect()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const RHO: f64 = 1.204;
    const C: f64 = 343.0;

    fn test_panel() -> CabinetPanel {
        CabinetPanel {
            width: 0.20,  // 20 cm
            height: 0.30, // 30 cm
            material: PanelMaterial::mdf_18mm(),
        }
    }

    #[test]
    fn test_mdf_flexural_rigidity() {
        let mat = PanelMaterial::mdf_18mm();
        let d = mat.flexural_rigidity();
        // D = 3.5e9 × 0.018³ / (12 × (1 - 0.25²)) = 3.5e9 × 5.832e-6 / 11.25 ≈ 1814 N·m
        assert!(d > 1000.0 && d < 3000.0, "D = {d:.1} N·m");
    }

    #[test]
    fn test_panel_fundamental_frequency() {
        let panel = test_panel();
        let f11 = panel.mode_frequency(1, 1);
        // Simply-supported plate (1,1) mode should be in the hundreds of Hz for MDF
        assert!(f11 > 50.0 && f11 < 1000.0, "f_11 = {f11:.1} Hz");
    }

    #[test]
    fn test_mode_frequencies_increase_with_order() {
        let panel = test_panel();
        let f11 = panel.mode_frequency(1, 1);
        let f12 = panel.mode_frequency(1, 2);
        let f21 = panel.mode_frequency(2, 1);
        let f22 = panel.mode_frequency(2, 2);

        assert!(f12 > f11, "f12 > f11");
        assert!(f21 > f11, "f21 > f11");
        assert!(f22 > f12, "f22 > f12");
        assert!(f22 > f21, "f22 > f21");
    }

    #[test]
    fn test_modes_below_frequency() {
        let panel = test_panel();
        let modes = panel.modes_below(2000.0, 10);

        assert!(!modes.is_empty(), "Should have modes below 2 kHz");

        // Should be sorted by frequency
        for i in 1..modes.len() {
            assert!(modes[i].frequency >= modes[i - 1].frequency);
        }

        // Only odd-odd modes should have nonzero force coefficient
        for mode in &modes {
            if mode.m % 2 == 0 || mode.n % 2 == 0 {
                assert!(
                    mode.force_coefficient.abs() < 1e-30,
                    "Even mode ({},{}) should have zero force coeff",
                    mode.m,
                    mode.n
                );
            } else {
                assert!(
                    mode.force_coefficient > 0.0,
                    "Odd mode ({},{}) should have nonzero force coeff",
                    mode.m,
                    mode.n
                );
            }
        }
    }

    #[test]
    fn test_rigid_panel_zero_radiation() {
        // Infinitely stiff panel (E → huge) → mode frequencies → ∞ → zero response
        let mut panel = test_panel();
        panel.material.youngs_modulus = 1e15; // essentially rigid

        let power = panel.radiated_power(500.0, 100.0, RHO, C, 10);
        assert!(
            power < 1e-20,
            "Rigid panel should not radiate: P = {power:.2e} W"
        );
    }

    #[test]
    fn test_thicker_panel_radiates_less() {
        let thin = CabinetPanel {
            width: 0.20,
            height: 0.30,
            material: PanelMaterial {
                thickness: 0.012,
                ..PanelMaterial::mdf_18mm()
            },
        };
        let thick = CabinetPanel {
            width: 0.20,
            height: 0.30,
            material: PanelMaterial {
                thickness: 0.025,
                ..PanelMaterial::mdf_18mm()
            },
        };

        let f_test = 500.0;
        let p_test = 100.0; // 100 Pa internal pressure

        let power_thin = thin.radiated_power(f_test, p_test, RHO, C, 10);
        let power_thick = thick.radiated_power(f_test, p_test, RHO, C, 10);

        assert!(
            power_thin > power_thick,
            "Thinner panel should radiate more: thin={power_thin:.2e}W, thick={power_thick:.2e}W"
        );
    }

    #[test]
    fn test_sealed_box_has_five_panels() {
        let cab = Cabinet::sealed_box(0.22, 0.40, 0.30, PanelMaterial::mdf_18mm());
        assert_eq!(cab.panels.len(), 5);
    }

    #[test]
    fn test_cabinet_radiation_positive() {
        let cab = Cabinet::sealed_box(0.22, 0.40, 0.30, PanelMaterial::mdf_18mm());
        let power = cab.total_radiated_power(500.0, 100.0, RHO, C, 10);
        assert!(power > 0.0, "Cabinet should radiate some power");
    }

    #[test]
    fn test_radiation_vs_frequency_sweep() {
        let cab = Cabinet::sealed_box(0.22, 0.40, 0.30, PanelMaterial::mdf_18mm());
        let freqs: Vec<f64> = (1..=10).map(|i| i as f64 * 100.0).collect();
        let result = cab.radiation_vs_frequency(&freqs, 100.0, RHO, C, 10);

        assert_eq!(result.len(), 10);
        // At some frequency there should be a resonance peak
        let max_spl = result
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_spl = result.iter().map(|(_, s)| *s).fold(f64::INFINITY, f64::min);
        assert!(max_spl > min_spl, "Should see variation across frequency");
    }

    #[test]
    fn test_critical_frequency() {
        let mat = PanelMaterial::mdf_18mm();
        let fc = mat.critical_frequency(C);
        // For 18mm MDF, critical frequency is typically around 2–5 kHz
        assert!(
            fc > 500.0 && fc < 10000.0,
            "Critical frequency = {fc:.0} Hz"
        );
    }

    #[test]
    fn test_higher_loss_factor_reduces_peak() {
        let low_loss = CabinetPanel {
            width: 0.20,
            height: 0.30,
            material: PanelMaterial {
                loss_factor: 0.01,
                ..PanelMaterial::mdf_18mm()
            },
        };
        let high_loss = CabinetPanel {
            width: 0.20,
            height: 0.30,
            material: PanelMaterial {
                loss_factor: 0.50,
                ..PanelMaterial::mdf_18mm()
            },
        };

        // At a panel resonance, high damping should reduce the peak velocity
        let f_res = low_loss.mode_frequency(1, 1);
        let v_low = low_loss.surface_velocity(f_res, 100.0, 10);
        let v_high = high_loss.surface_velocity(f_res, 100.0, 10);

        assert!(
            v_low > v_high,
            "Higher loss factor should reduce resonance peak: v_low={v_low:.4e}, v_high={v_high:.4e}"
        );
    }

    #[test]
    fn test_surface_displacement_consistent_with_surface_velocity() {
        // Every modal velocity is iω × the modal displacement, and both
        // surface metrics use the same per-mode weighting, so the identity
        // v = ω·x holds exactly when both use the same mode set.
        let panel = test_panel();
        let f = 100.0;
        let cutoff = 10.0 * panel.mode_frequency(1, 1);
        let x = panel.surface_displacement(f, 100.0, cutoff, 10);
        let v = panel
            .modal_displacements(f, Complex64::new(100.0, 0.0), cutoff, 10)
            .iter()
            .map(|mode| {
                let area = panel.width * panel.height;
                let force_coeff = 4.0 * area / (mode.m as f64 * mode.n as f64 * PI * PI);
                (2.0 * PI * f * mode.displacement_m.norm() * force_coeff / area).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        let omega = 2.0 * PI * f;
        assert!(
            (v - omega * x).abs() / v.max(1e-30) < 1e-12,
            "v={v:.6e}, ω·x={:.6e}",
            omega * x
        );
    }

    #[test]
    fn test_modal_displacement_center_matches_navier_static_series() {
        // Quasi-static uniform-pressure deflection at the panel center of a
        // simply-supported plate, Navier series:
        //   w_max = Σ_{m,n odd} 16·p·s_mn / (π⁶·D·m·n·((m/a)²+(n/b)²)²)
        // with s_mn = sin(mπ/2)·sin(nπ/2) the mode shape at the center.
        let panel = test_panel();
        let p = 100.0;
        let f_11 = panel.mode_frequency(1, 1);
        let modes = panel.modal_displacements(1.0, Complex64::new(p, 0.0), 25.0 * f_11, 12);
        assert!(!modes.is_empty(), "quasi-static solve must retain modes");

        let center: f64 = modes
            .iter()
            .map(|mode| {
                let sign = ((mode.m as f64) * PI / 2.0).sin() * ((mode.n as f64) * PI / 2.0).sin();
                (mode.displacement_m * sign).re
            })
            .sum();

        let d = panel.material.flexural_rigidity();
        let (a, b) = (panel.width, panel.height);
        let mut navier = 0.0;
        for m in (1..=25).step_by(2) {
            for n in (1..=25).step_by(2) {
                let sign = ((m as f64) * PI / 2.0).sin() * ((n as f64) * PI / 2.0).sin();
                let k2 = (m as f64 / a).powi(2) + (n as f64 / b).powi(2);
                navier += 16.0 * p * sign / (PI.powi(6) * d * m as f64 * n as f64 * k2 * k2);
            }
        }
        assert!(
            (center - navier).abs() / navier.abs() < 0.01,
            "center={center:.6e} m, Navier={navier:.6e} m"
        );
    }

    #[test]
    fn test_modal_displacement_11_matches_quasi_static_formula() {
        // X_11 at ω→0: F_11 / (M_11·ω_11²) computed by hand from the panel
        // mode data.
        let panel = test_panel();
        let p = 250.0;
        let f_11 = panel.mode_frequency(1, 1);
        let mode = panel
            .modes_below(2.0 * f_11, 4)
            .into_iter()
            .find(|mode| mode.m == 1 && mode.n == 1)
            .expect("(1,1) mode present");
        let omega_11 = 2.0 * PI * mode.frequency;
        let expected = mode.force_coefficient * p / (mode.modal_mass * omega_11 * omega_11);

        let modes = panel.modal_displacements(1.0, Complex64::new(p, 0.0), 2.0 * f_11, 4);
        let x_11 = modes
            .iter()
            .find(|mode| mode.m == 1 && mode.n == 1)
            .expect("(1,1) displacement present")
            .displacement_m;
        assert!(
            (x_11.re - expected).abs() / expected < 1e-5,
            "X_11={:.6e}, expected={expected:.6e}",
            x_11.re
        );
        // Quasi-static response far below resonance is nearly in phase; the
        // out-of-phase part is the small damping term η·ω/ω_11.
        assert!(x_11.im.abs() / expected < 1e-3, "X_11 imag={:.3e}", x_11.im);
    }
}
