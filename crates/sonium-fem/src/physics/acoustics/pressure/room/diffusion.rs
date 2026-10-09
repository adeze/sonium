//! Acoustic diffusion equation for energy-based room analysis
//!
//! The acoustic diffusion equation models the spatial distribution of
//! sound energy density in rooms using a diffusion analogy:
//!
//! ```text
//! ∂w/∂t = ∇·(D ∇w) − c α_air w + P/V
//! ```
//!
//! In steady state: `∇·(D ∇w) − c α_air w + P/V = 0`
//!
//! where:
//! - w = acoustic energy density \[J/m³\]
//! - D = λc/3 = diffusion coefficient \[m²/s\]
//! - λ = 4V/S = mean free path
//! - c = speed of sound
//! - α_air = air absorption coefficient \[1/m\]
//! - P = source power \[W\]
//!
//! # Boundary conditions
//!
//! On absorbing walls: `−D ∂w/∂n = h_w w` where
//! `h_w = c α_wall / (4(1 − α_wall))` is the wall exchange coefficient.
//!
//! This is a Robin BC that can be assembled using the existing FEM forms.
//!
//! # Advantages
//!
//! - Very fast compared to wave-based methods
//! - Handles large, complex geometries (buildings, factories)
//! - Naturally handles coupled rooms (diffusion across openings)
//! - Gives SPL distribution and reverberation time in one solve

// ---------------------------------------------------------------------------
// Diffusion parameters
// ---------------------------------------------------------------------------

/// Room parameters for the acoustic diffusion equation.
#[derive(Debug, Clone)]
pub struct DiffusionRoom {
    /// Room volume \[m³\].
    pub volume: f64,
    /// Total surface area \[m²\].
    pub total_surface_area: f64,
    /// Speed of sound \[m/s\].
    pub speed_of_sound: f64,
}

impl DiffusionRoom {
    pub fn new(volume: f64, total_surface_area: f64, speed_of_sound: f64) -> Self {
        Self {
            volume,
            total_surface_area,
            speed_of_sound,
        }
    }

    /// Mean free path: λ = 4V/S \[m\].
    pub fn mean_free_path(&self) -> f64 {
        4.0 * self.volume / self.total_surface_area
    }

    /// Diffusion coefficient: D = λc/3 \[m²/s\].
    pub fn diffusion_coefficient(&self) -> f64 {
        self.mean_free_path() * self.speed_of_sound / 3.0
    }

    /// Wall exchange coefficient for a surface with absorption α.
    ///
    /// h_w = c α / (4(1 − α)) \[m/s\].
    ///
    /// The Robin BC is: −D ∂w/∂n = h_w w.
    /// For the FEM assembler, the Robin coefficient is: α_robin = h_w / D.
    /// # Panics
    /// Panics if `absorption` is not in \[0, 1).
    pub fn wall_exchange_coefficient(&self, absorption: f64) -> f64 {
        assert!(
            (0.0..1.0).contains(&absorption),
            "absorption must be in [0, 1), got {absorption}"
        );
        self.speed_of_sound * absorption / (4.0 * (1.0 - absorption))
    }

    /// FEM Robin BC coefficient: α_robin = h_w / D.
    ///
    /// Use as: system matrix = D × K + Σ_walls (h_w_i / D) × D × B_i
    ///       = D × K + Σ_walls h_w_i × B_i.
    ///
    /// So the boundary coefficient for the prebaked assembler is h_w_i
    /// (the wall exchange coefficient), when the stiffness is already
    /// scaled by D.
    pub fn robin_coefficient(&self, absorption: f64) -> f64 {
        self.wall_exchange_coefficient(absorption)
    }
}

// ---------------------------------------------------------------------------
// Steady-state system setup
// ---------------------------------------------------------------------------

/// Coefficients for the FEM diffusion system.
///
/// The system is: `(D K + Σ h_w B_walls + c α_air M) w = (P/V) M 1`
///
/// where K is the Laplacian stiffness, M is the mass matrix, and B are
/// boundary mass matrices for each wall.
#[derive(Debug, Clone)]
pub struct DiffusionSystemCoefficients {
    /// Stiffness coefficient (= diffusion coefficient D).
    pub stiffness_coeff: f64,
    /// Mass coefficient (= c × air_absorption).
    pub mass_coeff: f64,
    /// Source term coefficient (= P/V, source power per unit volume).
    pub source_coeff: f64,
}

impl DiffusionSystemCoefficients {
    /// Compute from room parameters and source power.
    pub fn new(room: &DiffusionRoom, source_power: f64, air_absorption: f64) -> Self {
        Self {
            stiffness_coeff: room.diffusion_coefficient(),
            mass_coeff: room.speed_of_sound * air_absorption,
            source_coeff: source_power / room.volume,
        }
    }
}

// ---------------------------------------------------------------------------
// SPL from energy density
// ---------------------------------------------------------------------------

/// Convert energy density w to sound pressure level \[dB re 20 μPa\].
///
/// p_rms² = w × ρ₀ c₀² (energy density to mean square pressure).
/// SPL = 10 log₁₀(p_rms² / p_ref²) where p_ref = 20 μPa.
pub fn energy_density_to_spl(w: f64, density: f64, speed_of_sound: f64) -> f64 {
    let p_rms_sq = w * density * speed_of_sound * speed_of_sound;
    let p_ref = 20.0e-6;
    10.0 * (p_rms_sq / (p_ref * p_ref)).max(1e-30).log10()
}

/// Convert SPL to energy density.
pub fn spl_to_energy_density(spl_db: f64, density: f64, speed_of_sound: f64) -> f64 {
    let p_ref = 20.0e-6;
    let p_rms_sq = p_ref * p_ref * 10.0_f64.powf(spl_db / 10.0);
    p_rms_sq / (density * speed_of_sound * speed_of_sound)
}

// ---------------------------------------------------------------------------
// Reverberation time from diffusion
// ---------------------------------------------------------------------------

/// Reverberation time from the diffusion model (analytical, for a single room).
///
/// For a rectangular room with uniform absorption, the eigenvalue approach
/// gives the same result as the Eyring formula:
///
/// RT60 ≈ 0.161 V / (−S ln(1−ᾱ) + 4mV)
///
/// This is a wrapper around [`super::metrics::rt60_eyring`].
pub fn diffusion_rt60(room: &DiffusionRoom, mean_absorption: f64, air_absorption: f64) -> f64 {
    let air_term = 4.0 * air_absorption * room.volume;
    super::metrics::rt60_eyring(
        room.volume,
        room.total_surface_area,
        mean_absorption,
        air_term,
    )
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_free_path() {
        // Cubic room 5×5×5: V = 125 m³, S = 150 m²
        let room = DiffusionRoom::new(125.0, 150.0, 343.0);
        let mfp = room.mean_free_path();
        let expected = 4.0 * 125.0 / 150.0; // 3.33 m
        assert!((mfp - expected).abs() < 1e-10);
    }

    #[test]
    fn test_diffusion_coefficient() {
        let room = DiffusionRoom::new(125.0, 150.0, 343.0);
        let d = room.diffusion_coefficient();
        let expected = room.mean_free_path() * 343.0 / 3.0;
        assert!((d - expected).abs() < 1e-10);
    }

    #[test]
    fn test_wall_exchange_coefficient() {
        let room = DiffusionRoom::new(125.0, 150.0, 343.0);
        // α = 0.1: h = 343 × 0.1 / (4 × 0.9) = 9.53 m/s
        let h = room.wall_exchange_coefficient(0.1);
        let expected = 343.0 * 0.1 / (4.0 * 0.9);
        assert!((h - expected).abs() < 1e-6, "h = {h}, expected {expected}");
    }

    #[test]
    fn test_wall_exchange_fully_absorbing() {
        let room = DiffusionRoom::new(100.0, 100.0, 343.0);
        // α → 1: h → ∞ (perfectly absorbing wall)
        let h = room.wall_exchange_coefficient(0.99);
        assert!(h > 1000.0, "h should be very large for α ≈ 1");
    }

    #[test]
    fn test_energy_density_spl_roundtrip() {
        let spl = 60.0; // dB
        let rho = 1.204;
        let c = 343.0;
        let w = spl_to_energy_density(spl, rho, c);
        let spl_back = energy_density_to_spl(w, rho, c);
        assert!(
            (spl_back - spl).abs() < 1e-6,
            "Round-trip: {spl} → {w} → {spl_back}"
        );
    }

    #[test]
    fn test_diffusion_rt60_typical_room() {
        // Living room: V = 50 m³, S = 80 m², ᾱ = 0.15
        let room = DiffusionRoom::new(50.0, 80.0, 343.0);
        let rt = diffusion_rt60(&room, 0.15, 0.0);
        // Expected: 0.161 × 50 / (−80 × ln(0.85)) ≈ 0.161 × 50 / 12.99 ≈ 0.62 s
        assert!(rt > 0.3 && rt < 1.5, "RT60 = {rt:.2} s");
    }

    // -- Regression: absorption=1 must panic, not produce inf --

    #[test]
    #[should_panic(expected = "absorption must be in [0, 1)")]
    fn test_wall_exchange_panics_at_absorption_one() {
        let room = DiffusionRoom::new(100.0, 100.0, 343.0);
        room.wall_exchange_coefficient(1.0);
    }
}
