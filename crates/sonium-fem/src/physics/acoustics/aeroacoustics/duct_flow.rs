//! Quasi-1D isentropic duct flow model
//!
//! Computes the steady-state mean flow through an axisymmetric duct
//! using isentropic area-velocity relations. Given the inlet Mach number
//! and the duct cross-sectional area profile A(z), computes U(z), ρ(z),
//! c(z) at each axial station.
//!
//! This provides a lightweight mean flow model for the convected Helmholtz
//! equation without requiring a full Navier-Stokes solve. It implements
//! the [`MeanFlowField`] trait so it plugs directly into the LPF/LEE
//! assemblers.
//!
//! # Limitations
//!
//! - Assumes isentropic, inviscid, quasi-1D flow (no recirculation, no BL)
//! - Valid for subsonic ducts only (M < 1 everywhere)
//! - The propeller is modeled as a total pressure / velocity jump at a station
//!
//! # References
//!
//! Anderson, "Modern Compressible Flow", Ch. 3 — quasi-1D isentropic flow.

use super::mean_flow::{MeanFlow, MeanFlowField};

/// Quasi-1D isentropic duct flow field.
///
/// Given an inlet Mach number and a duct area profile `A(z)`, computes
/// the flow properties at any point (r, z) in the meridional plane.
/// The axial velocity varies with z only; radial velocity is zero
/// in the quasi-1D approximation.
pub struct DuctFlowField {
    /// Reference (stagnation) conditions
    pub reference: MeanFlow,
    /// Inlet Mach number
    pub mach_inlet: f64,
    /// Axial stations (sorted ascending)
    z_stations: Vec<f64>,
    /// Mach number at each station
    mach_profile: Vec<f64>,
    /// Axial velocity at each station
    velocity_profile: Vec<f64>,
    /// Density at each station
    density_profile: Vec<f64>,
    /// Speed of sound at each station
    sos_profile: Vec<f64>,
    /// Pressure at each station
    pressure_profile: Vec<f64>,
}

impl DuctFlowField {
    /// Create a duct flow field from an area profile.
    ///
    /// # Arguments
    ///
    /// * `reference` — stagnation/reference conditions (MeanFlow at rest)
    /// * `mach_inlet` — Mach number at the inlet (must be < 1)
    /// * `area_profile` — closure `|z| -> A(z)` giving cross-sectional area
    /// * `z_min`, `z_max` — axial extent
    /// * `n_stations` — number of axial stations for the profile
    pub fn from_area_profile<F: Fn(f64) -> f64>(
        reference: MeanFlow,
        mach_inlet: f64,
        area_profile: F,
        z_min: f64,
        z_max: f64,
        n_stations: usize,
    ) -> Self {
        assert!(
            mach_inlet > 0.0 && mach_inlet < 1.0,
            "inlet Mach must be subsonic (0 < M < 1)"
        );
        assert!(n_stations >= 2);

        let gamma = reference.gamma;
        let rho_0 = reference.density;
        let c_0 = reference.speed_of_sound;
        let p_0 = reference.pressure;

        let dz = (z_max - z_min) / (n_stations - 1) as f64;
        let a_inlet = area_profile(z_min);

        // Isentropic area ratio: A/A* = (1/M) * [(2/(gamma+1)) * (1 + (gamma-1)/2 * M^2)]^((gamma+1)/(2*(gamma-1)))
        let area_ratio_from_mach = |m: f64| -> f64 {
            let gp1 = gamma + 1.0;
            let gm1 = gamma - 1.0;
            let exp = gp1 / (2.0 * gm1);
            (1.0 / m) * ((2.0 / gp1) * (1.0 + 0.5 * gm1 * m * m)).powf(exp)
        };

        let a_star_over_a_inlet = 1.0 / area_ratio_from_mach(mach_inlet);

        // For each station, find M from A(z)/A* using Newton iteration
        let find_mach_subsonic = |area_ratio: f64| -> f64 {
            let mut m = mach_inlet; // initial guess
            for _ in 0..50 {
                let f = area_ratio_from_mach(m) - area_ratio;
                // Derivative: d(A/A*)/dM
                let gp1 = gamma + 1.0;
                let gm1 = gamma - 1.0;
                let exp = gp1 / (2.0 * gm1);
                let base = (2.0 / gp1) * (1.0 + 0.5 * gm1 * m * m);
                let df = -1.0 / (m * m) * base.powf(exp)
                    + (1.0 / m) * exp * base.powf(exp - 1.0) * (2.0 / gp1) * gm1 * m;
                if df.abs() < 1e-30 {
                    break;
                }
                let dm = -f / df;
                m = (m + dm).clamp(0.001, 0.999);
                if dm.abs() < 1e-12 {
                    break;
                }
            }
            m
        };

        // Isentropic relations: T/T0 = (1 + (gamma-1)/2 M^2)^(-1)
        let isentropic_factor = |m: f64| -> f64 { 1.0 + 0.5 * (gamma - 1.0) * m * m };

        let mut z_stations = Vec::with_capacity(n_stations);
        let mut mach_profile = Vec::with_capacity(n_stations);
        let mut velocity_profile = Vec::with_capacity(n_stations);
        let mut density_profile = Vec::with_capacity(n_stations);
        let mut sos_profile = Vec::with_capacity(n_stations);
        let mut pressure_profile = Vec::with_capacity(n_stations);

        for i in 0..n_stations {
            let z = z_min + i as f64 * dz;
            let a_z = area_profile(z);
            let area_ratio = a_z / a_inlet / a_star_over_a_inlet; // A(z) / A*

            let m = if area_ratio < 1.0 + 1e-10 {
                // Near or past throat — clamp to subsonic
                0.999
            } else {
                find_mach_subsonic(area_ratio)
            };

            let factor = isentropic_factor(m);
            let c_local = c_0 / factor.sqrt(); // c = c0 * sqrt(T/T0) = c0 / sqrt(1 + ...)
            let rho_local = rho_0 * factor.powf(-1.0 / (gamma - 1.0));
            let p_local = p_0 * factor.powf(-gamma / (gamma - 1.0));
            let u_local = m * c_local;

            z_stations.push(z);
            mach_profile.push(m);
            velocity_profile.push(u_local);
            density_profile.push(rho_local);
            sos_profile.push(c_local);
            pressure_profile.push(p_local);
        }

        Self {
            reference,
            mach_inlet,
            z_stations,
            mach_profile,
            velocity_profile,
            density_profile,
            sos_profile,
            pressure_profile,
        }
    }

    /// Interpolate a profile value at axial position z.
    fn interpolate(&self, z: f64, profile: &[f64]) -> f64 {
        let zs = &self.z_stations;
        if z <= zs[0] {
            return profile[0];
        }
        if z >= *zs.last().unwrap() {
            return *profile.last().unwrap();
        }
        // Binary search for interval
        let idx = zs.partition_point(|&zi| zi < z).saturating_sub(1);
        let idx = idx.min(zs.len() - 2);
        let t = (z - zs[idx]) / (zs[idx + 1] - zs[idx]);
        profile[idx] * (1.0 - t) + profile[idx + 1] * t
    }

    /// Get the local Mach number at axial position z.
    pub fn mach_at(&self, z: f64) -> f64 {
        self.interpolate(z, &self.mach_profile)
    }

    /// Get the local axial velocity at axial position z.
    pub fn velocity_at(&self, z: f64) -> f64 {
        self.interpolate(z, &self.velocity_profile)
    }
}

impl MeanFlowField for DuctFlowField {
    fn density(&self, x: [f64; 3]) -> f64 {
        let z = x[1]; // y-coordinate is axial in the meridional plane
        self.interpolate(z, &self.density_profile)
    }

    fn velocity(&self, x: [f64; 3]) -> [f64; 3] {
        let z = x[1];
        // Axial velocity only (quasi-1D); in the (r,z) plane, axial = y-direction
        [0.0, self.interpolate(z, &self.velocity_profile), 0.0]
    }

    fn speed_of_sound(&self, x: [f64; 3]) -> f64 {
        let z = x[1];
        self.interpolate(z, &self.sos_profile)
    }

    fn pressure(&self, x: [f64; 3]) -> f64 {
        let z = x[1];
        self.interpolate(z, &self.pressure_profile)
    }

    fn velocity_gradient(&self, _x: [f64; 3]) -> [[f64; 3]; 3] {
        // Quasi-1D: dU/dz could be computed from finite differences on the profile,
        // but for the uniform-Mach LPF assembler this is not needed.
        [[0.0; 3]; 3]
    }

    fn density_gradient(&self, _x: [f64; 3]) -> [f64; 3] {
        [0.0; 3]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_straight_duct_constant_mach() {
        let reference = MeanFlow::air_standard([0.0, 0.0, 0.0]);
        let r = 0.05;
        let area = PI * r * r;

        let flow = DuctFlowField::from_area_profile(
            reference,
            0.1,
            |_z| area, // constant area
            0.0,
            1.0,
            100,
        );

        // In a straight duct, Mach should be constant
        for z in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let m = flow.mach_at(z);
            assert!(
                (m - 0.1).abs() < 1e-6,
                "Mach at z={z} should be 0.1, got {m}"
            );
        }
    }

    #[test]
    fn test_expansion_reduces_mach() {
        let reference = MeanFlow::air_standard([0.0, 0.0, 0.0]);
        let r1 = 0.025;
        let r2 = 0.075;

        let flow = DuctFlowField::from_area_profile(
            reference,
            0.1,
            |z| {
                let r = if (0.3..=0.7).contains(&z) { r2 } else { r1 };
                PI * r * r
            },
            0.0,
            1.0,
            200,
        );

        let m_inlet = flow.mach_at(0.0);
        let m_chamber = flow.mach_at(0.5);

        assert!(
            m_chamber < m_inlet,
            "Expansion should reduce Mach: inlet={m_inlet}, chamber={m_chamber}"
        );
    }

    #[test]
    fn test_mean_flow_field_trait() {
        let reference = MeanFlow::air_standard([0.0, 0.0, 0.0]);
        let flow =
            DuctFlowField::from_area_profile(reference, 0.1, |_| PI * 0.05 * 0.05, 0.0, 1.0, 50);

        // velocity should be in the y (axial) direction
        let vel = flow.velocity([0.025, 0.5, 0.0]);
        assert!(vel[0].abs() < 1e-10, "radial velocity should be zero");
        assert!(vel[1] > 0.0, "axial velocity should be positive");
        assert!(vel[2].abs() < 1e-10, "azimuthal velocity should be zero");

        // density should be close to stagnation (low Mach)
        let rho = flow.density([0.025, 0.5, 0.0]);
        assert!((rho - 1.204).abs() < 0.01, "density ~ 1.204 at low Mach");
    }
}
