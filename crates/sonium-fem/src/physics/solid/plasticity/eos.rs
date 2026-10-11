//! Polynomial and JWL equations of state.
//!
//! Hydrodynamic pressure laws for explicit solids: the linear polynomial
//! EOS (`/EOS/POLYNOMIAL`, Reference Guide pp.1267-1269, Eq. 841-842)
//! and the Jones-Wilkins-Lee explosive EOS (`/MAT/LAW5`, Reference Guide
//! pp.1051-1055, Eq. 651-654). The Theory Manual carries no EOS section,
//! so both laws cite the card comments that define them; every equation
//! below names its manual source. All quantities are SI (Pa, J/m^3,
//! kg/m^3, m/s) unless noted.
//!
//! Both cards are plain parameter structs: construct them with a struct
//! literal or the card-example seeds ([`PolynomialEos::steel`],
//! [`JwlEos::tnt`]). Energy arguments are internal energy per unit
//! reference volume; the card `E0` fields only record the initial value.
//!
//! ```rust
//! use sonium_fem::physics::solid::plasticity::eos::{JwlEos, PolynomialEos};
//!
//! let steel = PolynomialEos::steel();
//! let mu = steel.compression(1.1 * steel.rho0);
//! assert!((steel.pressure(mu, 0.0) - 2.0e10).abs() < 1.0e6);
//! let tnt = JwlEos::tnt();
//! assert!(tnt.pressure_jwl(1.0, tnt.e0, 0.0) > 0.0);
//! ```

// Rust guideline compliant 2026-02-21

/// Polynomial hydrodynamic EOS (`/EOS/POLYNOMIAL`).
///
/// Pressure follows Reference Guide Eq. 841 with Eq. 842 compression;
/// comment 2 drops the quadratic term in expansion (`mu < 0`). The
/// card `E0`/`Psh` fields ride along for deck round-trips: Eq. 841
/// shows the unshifted form, so callers pass the evolved energy
/// directly and apply any shift themselves.
#[derive(Debug, Clone, Copy)]
pub struct PolynomialEos {
    /// Constant pressure coefficient in Pa (card `C0`).
    pub c0: f64,
    /// Linear pressure coefficient in Pa (card `C1`).
    pub c1: f64,
    /// Quadratic pressure coefficient in Pa (card `C2`).
    pub c2: f64,
    /// Cubic pressure coefficient in Pa (card `C3`).
    pub c3: f64,
    /// Constant energy coefficient, dimensionless (card `C4`).
    pub c4: f64,
    /// Linear energy coefficient, dimensionless (card `C5`).
    pub c5: f64,
    /// Reference density in kg/m^3 (card `RHO_0`).
    pub rho0: f64,
    /// Initial internal energy per unit reference volume (card `E0`).
    pub e0: f64,
    /// Card pressure shift in Pa (not part of Eq. 841).
    pub psh: f64,
}

impl PolynomialEos {
    /// Compression from density (Eq. 842: `mu = rho / rho0 - 1`).
    pub fn compression(&self, rho: f64) -> f64 {
        rho / self.rho0 - 1.0
    }

    /// Pressure from compression and energy (Eq. 841).
    ///
    /// `P = C0 + C1 mu + C2 mu^2 + C3 mu^3 + (C4 + C5 mu) E` with
    /// `E` in J/m^3; the `C2` term vanishes for `mu < 0` (comment 2).
    pub fn pressure(&self, mu: f64, energy: f64) -> f64 {
        let quad = if mu < 0.0 { 0.0 } else { self.c2 * mu * mu };
        self.c0 + self.c1 * mu + quad + self.c3 * mu * mu * mu + (self.c4 + self.c5 * mu) * energy
    }

    /// Steel after the card example (p.1268, SI units).
    ///
    /// Transcribes `/EOS/POLYNOMIAL/8` (`C1 = 1.69`, `C2 = 3.1`,
    /// `C4 = C5 = 1.83`, `RHO_I = 7.89 g/cm^3`) from g-cm-microsecond
    /// (Mbar) into SI.
    pub fn steel() -> Self {
        Self {
            c0: 0.0,
            c1: 1.69e11,
            c2: 3.1e11,
            c3: 0.0,
            c4: 1.83,
            c5: 1.83,
            rho0: 7890.0,
            e0: 0.0,
            psh: 0.0,
        }
    }
}

/// Jones-Wilkins-Lee explosive EOS (`/MAT/LAW5`).
///
/// Detonation-products pressure follows Eq. 651; the engine output
/// blends it with the unreacted branch by burn fraction (Eq. 652).
/// Afterburning energy `Q` enters Eq. 651 as `E + Q` (comment 1).
#[derive(Debug, Clone, Copy)]
pub struct JwlEos {
    /// First JWL coefficient in Pa (card `A`).
    pub a: f64,
    /// Second JWL coefficient in Pa (card `B`).
    pub b: f64,
    /// First JWL exponent, dimensionless (card `R1`).
    pub r1: f64,
    /// Second JWL exponent, dimensionless (card `R2`).
    pub r2: f64,
    /// Adiabatic constant `gamma - 1`, dimensionless (card `OMEGA`).
    pub omega: f64,
    /// Initial density in kg/m^3 (card `RHO_I`).
    pub rho0: f64,
    /// Initial pressure of unreacted explosive in Pa (card `P0`).
    pub p0: f64,
    /// Pressure shift in Pa (card `Psh`, Eq. 652).
    pub psh: f64,
    /// Unreacted bulk modulus in Pa (card `Bunreacted`, Eq. 652).
    pub b_unreacted: f64,
    /// Detonation velocity in m/s (card `D`, Eq. 654).
    pub detonation_velocity: f64,
    /// Chapman-Jouguet pressure in Pa (card `P_CJ`, Eq. 654).
    pub pcj: f64,
    /// Initial internal energy per unit initial volume (card `E0`).
    pub e0: f64,
}

impl JwlEos {
    /// Compression from density (Eq. 653: `mu = rho / rho0 - 1`).
    pub fn compression(&self, rho: f64) -> f64 {
        rho / self.rho0 - 1.0
    }

    /// Relative volume from density (`V = V / V0 = rho0 / rho`).
    pub fn relative_volume(&self, rho: f64) -> f64 {
        self.rho0 / rho
    }

    /// JWL detonation-products pressure (Eq. 651).
    ///
    /// Takes relative volume `V`, internal energy `E`, and afterburning
    /// energy `Q` (both J/m^3); returns Pa.
    pub fn pressure_jwl(&self, volume: f64, energy: f64, afterburn: f64) -> f64 {
        let first = self.a * (1.0 - self.omega / (self.r1 * volume)) * (-self.r1 * volume).exp();
        let second = self.b * (1.0 - self.omega / (self.r2 * volume)) * (-self.r2 * volume).exp();
        first + second + self.omega * (energy + afterburn) / volume
    }

    /// Volume branch of the burn fraction (Eq. 654).
    ///
    /// `B_f1 = (rho0 D^2 / P_CJ) (1 - V)`; may leave `[0, 1]`, the
    /// caller (`burn_fraction`) clips it into range.
    pub fn burn_fraction_volume(&self, volume: f64) -> f64 {
        self.rho0 * self.detonation_velocity * self.detonation_velocity / self.pcj * (1.0 - volume)
    }

    /// Time branch of the burn fraction (Eq. 654).
    ///
    /// `(T - T_det) / (1.5 dx)` past lighting time, else 0; takes
    /// current time, lighting time (s), and element size (m).
    pub fn burn_fraction_time(time: f64, lighting_time: f64, dx: f64) -> f64 {
        if time >= lighting_time {
            (time - lighting_time) / (1.5 * dx)
        } else {
            0.0
        }
    }

    /// Burn fraction combining both branches (Eq. 654 + `IBFRAC`).
    ///
    /// Flag 0 (default) reads `min(B_f1, B_f2)` and flag 1 reads
    /// `min(1, B_f1)`; other flags fall back to the default form.
    /// Note Eq. 654 prints `max` while the `IBFRAC` paragraph below it
    /// specifies `min` as the default: the explicit flag paragraph
    /// governs here.
    pub fn burn_fraction(
        &self,
        volume: f64,
        time: f64,
        lighting_time: f64,
        dx: f64,
        ibfrac: u8,
    ) -> f64 {
        let bf1 = self.burn_fraction_volume(volume);
        match ibfrac {
            1 => bf1.min(1.0),
            _ => bf1.min(Self::burn_fraction_time(time, lighting_time, dx)),
        }
    }

    /// Engine output pressure blending burnt and unreacted (Eq. 652).
    ///
    /// `P = B P_jwl + (1 - B)(P0 + B_unreacted mu) - P_sh` with burn
    /// fraction `B`, relative volume, compression, energies (J/m^3).
    pub fn pressure(&self, volume: f64, mu: f64, energy: f64, afterburn: f64, burn: f64) -> f64 {
        let burnt = self.pressure_jwl(volume, energy, afterburn);
        let unreacted = self.p0 + self.b_unreacted * mu;
        burn * burnt + (1.0 - burn) * unreacted - self.psh
    }

    /// TNT after the card example (p.1054, SI units).
    ///
    /// Transcribes `/MAT/JWL/2/123` (`A = 3.7121`, `B = 0.0323`,
    /// `R1 = 4.15`, `R2 = 0.95`, `OMEGA = 0.3`, `D = 0.693`,
    /// `P_CJ = 0.21`, `E0 = 0.07`, `RHO_I = 1.63`) from g-cm-microsecond
    /// into SI.
    pub fn tnt() -> Self {
        Self {
            a: 3.7121e11,
            b: 0.0323e11,
            r1: 4.15,
            r2: 0.95,
            omega: 0.3,
            rho0: 1630.0,
            p0: 0.0,
            psh: 0.0,
            b_unreacted: 0.0,
            detonation_velocity: 6930.0,
            pcj: 0.21e11,
            e0: 0.07e11,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{JwlEos, PolynomialEos};

    #[test]
    fn polynomial_steel_compression_matches_hand_value() {
        let steel = PolynomialEos::steel();
        // mu = 0.1, E = 0: 1.69e11 * 0.1 + 3.1e11 * 0.01 = 2.0e10.
        assert!((steel.pressure(0.1, 0.0) - 2.0e10).abs() < 1.0);
        // Energy leg: (1.83 + 1.83 * 0.1) * 1e9 = 2.013e9.
        assert!((steel.pressure(0.1, 1.0e9) - 2.2013e10).abs() < 1.0e3);
    }

    #[test]
    fn polynomial_expansion_drops_quadratic_term() {
        let steel = PolynomialEos::steel();
        // mu = -0.05: C2 term gone, C1/C3/energy legs stay.
        let got = steel.pressure(-0.05, 1.0e9);
        let want = 1.69e11 * -0.05 + (1.83 + 1.83 * -0.05) * 1.0e9;
        assert!((got - want).abs() < 1.0e3);
    }

    #[test]
    fn compression_and_volume_stay_consistent() {
        let steel = PolynomialEos::steel();
        let mu = steel.compression(1.1 * steel.rho0);
        assert!((mu - 0.1).abs() < 1e-15);
        let tnt = JwlEos::tnt();
        let v = tnt.relative_volume(2.0 * tnt.rho0);
        assert!((v - 0.5).abs() < 1e-15);
        assert!((tnt.compression(2.0 * tnt.rho0) - (1.0 / v - 1.0)).abs() < 1e-15);
    }

    #[test]
    fn jwl_tnt_matches_closed_form() {
        let tnt = JwlEos::tnt();
        // V = 1, E = E0, Q = 0: textbook JWL closed form.
        let want = 3.7121e11 * (1.0 - 0.3 / 4.15) * (-4.15f64).exp()
            + 0.0323e11 * (1.0 - 0.3 / 0.95) * (-0.95f64).exp()
            + 0.3 * 0.07e11;
        assert!((tnt.pressure_jwl(1.0, tnt.e0, 0.0) - want).abs() < 1.0);
        // Unburnt output is the unreacted branch (P0 = 0 here).
        assert!((tnt.pressure(1.0, 0.0, tnt.e0, 0.0, 0.0) - 0.0).abs() < 1.0);
        // Fully burnt output is the JWL pressure.
        assert!((tnt.pressure(1.0, 0.0, tnt.e0, 0.0, 1.0) - want).abs() < 1.0);
    }

    #[test]
    fn burn_fraction_branches_follow_eq654() {
        let tnt = JwlEos::tnt();
        // Undisturbed volume: B_f1 = 0.
        assert!(tnt.burn_fraction_volume(1.0).abs() < 1e-15);
        // Time branch before lighting is 0, after is linear.
        assert!(JwlEos::burn_fraction_time(0.5, 1.0, 0.01) == 0.0);
        assert!((JwlEos::burn_fraction_time(1.03, 1.0, 0.01) - 2.0).abs() < 1e-12);
        // IBFRAC = 1 clips the volume branch at 1.
        assert!((tnt.burn_fraction(0.0, 0.0, 0.0, 0.01, 1) - 1.0).abs() < 1e-12);
    }
}
