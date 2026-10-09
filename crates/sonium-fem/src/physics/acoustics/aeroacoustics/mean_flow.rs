//! Mean flow properties for aeroacoustics simulations
//!
//! The mean flow represents the background steady-state flow field
//! upon which acoustic perturbations propagate. It can be uniform
//! (constant properties) or spatially varying (imported from CFD).

/// Uniform mean flow properties.
///
/// For a homogeneous background flow where all properties are constant
/// throughout the domain. Enables efficient prebaked assembly for
/// frequency sweeps.
#[derive(Debug, Clone, Copy)]
pub struct MeanFlow {
    /// Mean density rho_0 [kg/m^3]
    pub density: f64,
    /// Mean speed of sound c_0 [m/s]
    pub speed_of_sound: f64,
    /// Mean velocity U_0 [m/s]
    pub velocity: [f64; 3],
    /// Mean pressure p_0 [Pa]
    pub pressure: f64,
    /// Mean temperature T_0 [K]
    pub temperature: f64,
    /// Dynamic viscosity mu [Pa*s]
    pub dynamic_viscosity: f64,
    /// Thermal conductivity kappa [W/(m*K)]
    pub thermal_conductivity: f64,
    /// Specific heat at constant pressure c_p [J/(kg*K)]
    pub specific_heat_cp: f64,
    /// Ratio of specific heats gamma = c_p / c_v
    pub gamma: f64,
}

impl MeanFlow {
    /// Create a mean flow for air at standard conditions (20 deg C, 1 atm).
    pub fn air_standard(velocity: [f64; 3]) -> Self {
        Self {
            density: 1.204,
            speed_of_sound: 343.0,
            velocity,
            pressure: 101_325.0,
            temperature: 293.15,
            dynamic_viscosity: 1.825e-5,
            thermal_conductivity: 0.0257,
            specific_heat_cp: 1005.0,
            gamma: 1.4,
        }
    }

    /// Mach number vector M = U_0 / c_0.
    pub fn mach_vector(&self) -> [f64; 3] {
        let c0 = self.speed_of_sound;
        [
            self.velocity[0] / c0,
            self.velocity[1] / c0,
            self.velocity[2] / c0,
        ]
    }

    /// Mach number magnitude |M|.
    pub fn mach_number(&self) -> f64 {
        let m = self.mach_vector();
        (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt()
    }

    /// Wavenumber k = omega / c_0.
    pub fn wavenumber(&self, omega: f64) -> f64 {
        omega / self.speed_of_sound
    }

    /// Kinematic viscosity nu = mu / rho_0.
    pub fn kinematic_viscosity(&self) -> f64 {
        self.dynamic_viscosity / self.density
    }

    /// Prandtl number Pr = mu * c_p / kappa.
    pub fn prandtl_number(&self) -> f64 {
        self.dynamic_viscosity * self.specific_heat_cp / self.thermal_conductivity
    }

    /// Specific heat at constant volume c_v = c_p / gamma.
    pub fn specific_heat_cv(&self) -> f64 {
        self.specific_heat_cp / self.gamma
    }

    /// Bulk modulus K_0 = rho_0 * c_0^2.
    pub fn bulk_modulus(&self) -> f64 {
        self.density * self.speed_of_sound * self.speed_of_sound
    }
}

/// Spatially varying mean flow field.
///
/// Implement this trait for mean flow data imported from CFD solvers
/// (e.g. via CGNS). The uniform [`MeanFlow`] implements this trait
/// with constant values and zero gradients.
pub trait MeanFlowField: Send + Sync {
    /// Mean density rho_0 at position x.
    fn density(&self, x: [f64; 3]) -> f64;
    /// Mean velocity U_0 at position x.
    fn velocity(&self, x: [f64; 3]) -> [f64; 3];
    /// Mean speed of sound c_0 at position x.
    fn speed_of_sound(&self, x: [f64; 3]) -> f64;
    /// Mean pressure p_0 at position x.
    fn pressure(&self, x: [f64; 3]) -> f64;
    /// Velocity gradient tensor dU_i/dx_j (row = component, col = direction).
    fn velocity_gradient(&self, x: [f64; 3]) -> [[f64; 3]; 3];
    /// Density gradient d(rho_0)/dx_j.
    fn density_gradient(&self, x: [f64; 3]) -> [f64; 3];
}

impl MeanFlowField for MeanFlow {
    fn density(&self, _x: [f64; 3]) -> f64 {
        self.density
    }
    fn velocity(&self, _x: [f64; 3]) -> [f64; 3] {
        self.velocity
    }
    fn speed_of_sound(&self, _x: [f64; 3]) -> f64 {
        self.speed_of_sound
    }
    fn pressure(&self, _x: [f64; 3]) -> f64 {
        self.pressure
    }
    fn velocity_gradient(&self, _x: [f64; 3]) -> [[f64; 3]; 3] {
        [[0.0; 3]; 3]
    }
    fn density_gradient(&self, _x: [f64; 3]) -> [f64; 3] {
        [0.0; 3]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_air_standard_properties() {
        let flow = MeanFlow::air_standard([34.3, 0.0, 0.0]);
        assert!((flow.density - 1.204).abs() < 1e-6);
        assert!((flow.speed_of_sound - 343.0).abs() < 1e-6);
        assert!((flow.mach_number() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_mach_vector() {
        let flow = MeanFlow::air_standard([171.5, 171.5, 0.0]);
        let m = flow.mach_vector();
        assert!((m[0] - 0.5).abs() < 1e-10);
        assert!((m[1] - 0.5).abs() < 1e-10);
        assert!((m[2]).abs() < 1e-15);
        let mag = flow.mach_number();
        assert!((mag - (0.5_f64 * 2.0).sqrt() / 2.0 * 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_derived_quantities() {
        let flow = MeanFlow::air_standard([0.0, 0.0, 0.0]);
        let nu = flow.kinematic_viscosity();
        assert!((nu - 1.825e-5 / 1.204).abs() < 1e-10);

        let pr = flow.prandtl_number();
        assert!((pr - 1.825e-5 * 1005.0 / 0.0257).abs() < 1e-6);

        let cv = flow.specific_heat_cv();
        assert!((cv - 1005.0 / 1.4).abs() < 1e-6);
    }

    #[test]
    fn test_mean_flow_field_uniform() {
        let flow = MeanFlow::air_standard([100.0, 0.0, 0.0]);
        let x = [1.0, 2.0, 3.0];
        assert!((flow.density(x) - flow.density).abs() < 1e-15);
        assert_eq!(flow.velocity(x), flow.velocity);
        assert_eq!(flow.velocity_gradient(x), [[0.0; 3]; 3]);
        assert_eq!(flow.density_gradient(x), [0.0; 3]);
    }
}
