//! Ideal-gas material laws for compressible flow.

/// Calorically perfect gas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IdealGas {
    pub gamma: f64,
    pub gas_constant: f64,
}

impl IdealGas {
    pub const AIR: Self = Self {
        gamma: 1.4,
        gas_constant: 287.05,
    };

    pub fn pressure(&self, density: f64, internal_energy_per_mass: f64) -> f64 {
        (self.gamma - 1.0) * density * internal_energy_per_mass
    }

    pub fn speed_of_sound(&self, pressure: f64, density: f64) -> f64 {
        (self.gamma * pressure / density).sqrt()
    }

    pub fn temperature(&self, pressure: f64, density: f64) -> f64 {
        pressure / (density * self.gas_constant)
    }

    pub fn specific_heat_cp(&self) -> f64 {
        self.gamma * self.gas_constant / (self.gamma - 1.0)
    }
}

/// Dynamic-viscosity law used by the compressible Navier--Stokes DG solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViscosityModel {
    Constant(f64),
    Sutherland {
        reference_viscosity: f64,
        reference_temperature: f64,
        sutherland_temperature: f64,
    },
}

impl ViscosityModel {
    pub const AIR: Self = Self::Sutherland {
        reference_viscosity: 1.716e-5,
        reference_temperature: 273.15,
        sutherland_temperature: 110.4,
    };

    pub fn dynamic_viscosity(self, temperature: f64) -> f64 {
        match self {
            Self::Constant(viscosity) => viscosity,
            Self::Sutherland {
                reference_viscosity,
                reference_temperature,
                sutherland_temperature,
            } => {
                let temperature = temperature.max(f64::MIN_POSITIVE);
                reference_viscosity
                    * (temperature / reference_temperature).powf(1.5)
                    * (reference_temperature + sutherland_temperature)
                    / (temperature + sutherland_temperature)
            }
        }
    }

    pub fn thermal_conductivity(self, temperature: f64, gas: IdealGas, prandtl: f64) -> f64 {
        self.dynamic_viscosity(temperature) * gas.specific_heat_cp() / prandtl
    }

    pub fn is_valid(self) -> bool {
        match self {
            Self::Constant(viscosity) => viscosity.is_finite() && viscosity > 0.0,
            Self::Sutherland {
                reference_viscosity,
                reference_temperature,
                sutherland_temperature,
            } => {
                reference_viscosity.is_finite()
                    && reference_viscosity > 0.0
                    && reference_temperature.is_finite()
                    && reference_temperature > 0.0
                    && sutherland_temperature.is_finite()
                    && sutherland_temperature >= 0.0
            }
        }
    }
}

/// Conservative Euler state `[rho, rho*u, rho*v, rho*w, rho*E]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConservativeState {
    pub density: f64,
    pub momentum: [f64; 3],
    pub total_energy: f64,
}

impl ConservativeState {
    pub fn from_primitive(density: f64, velocity: [f64; 3], pressure: f64, gas: IdealGas) -> Self {
        let kinetic = 0.5 * density * velocity.iter().map(|u| u * u).sum::<f64>();
        let internal = pressure / (gas.gamma - 1.0);
        Self {
            density,
            momentum: [
                density * velocity[0],
                density * velocity[1],
                density * velocity[2],
            ],
            total_energy: internal + kinetic,
        }
    }

    pub fn velocity(&self) -> [f64; 3] {
        [
            self.momentum[0] / self.density,
            self.momentum[1] / self.density,
            self.momentum[2] / self.density,
        ]
    }

    pub fn kinetic_energy_per_volume(&self) -> f64 {
        let m2 = self.momentum.iter().map(|m| m * m).sum::<f64>();
        0.5 * m2 / self.density
    }

    pub fn pressure(&self, gas: IdealGas) -> f64 {
        (gas.gamma - 1.0) * (self.total_energy - self.kinetic_energy_per_volume())
    }

    pub fn euler_flux(&self, gas: IdealGas, normal: [f64; 3]) -> [f64; 5] {
        let velocity = self.velocity();
        let un = velocity
            .iter()
            .zip(normal.iter())
            .map(|(u, n)| u * n)
            .sum::<f64>();
        let p = self.pressure(gas);
        [
            self.density * un,
            self.momentum[0] * un + p * normal[0],
            self.momentum[1] * un + p * normal[1],
            self.momentum[2] * un + p * normal[2],
            (self.total_energy + p) * un,
        ]
    }

    pub fn max_wave_speed(&self, gas: IdealGas, normal: [f64; 3]) -> f64 {
        let velocity = self.velocity();
        let un = velocity
            .iter()
            .zip(normal.iter())
            .map(|(u, n)| u * n)
            .sum::<f64>();
        un.abs() + gas.speed_of_sound(self.pressure(gas), self.density)
    }

    pub fn as_array(self) -> [f64; 5] {
        [
            self.density,
            self.momentum[0],
            self.momentum[1],
            self.momentum[2],
            self.total_energy,
        ]
    }

    pub fn from_array(values: [f64; 5]) -> Self {
        Self {
            density: values[0],
            momentum: [values[1], values[2], values[3]],
            total_energy: values[4],
        }
    }

    pub fn is_physical(self, gas: IdealGas, density_floor: f64, pressure_floor: f64) -> bool {
        self.density.is_finite()
            && self.total_energy.is_finite()
            && self.momentum.iter().all(|value| value.is_finite())
            && self.density >= density_floor
            && self.pressure(gas).is_finite()
            && self.pressure(gas) >= pressure_floor
    }
}

pub fn rusanov_flux(
    left: ConservativeState,
    right: ConservativeState,
    gas: IdealGas,
    normal: [f64; 3],
) -> [f64; 5] {
    let fl = left.euler_flux(gas, normal);
    let fr = right.euler_flux(gas, normal);
    let ul = [
        left.density,
        left.momentum[0],
        left.momentum[1],
        left.momentum[2],
        left.total_energy,
    ];
    let ur = [
        right.density,
        right.momentum[0],
        right.momentum[1],
        right.momentum[2],
        right.total_energy,
    ];
    let smax = left
        .max_wave_speed(gas, normal)
        .max(right.max_wave_speed(gas, normal));
    let mut flux = [0.0; 5];
    for i in 0..5 {
        flux[i] = 0.5 * (fl[i] + fr[i]) - 0.5 * smax * (ur[i] - ul[i]);
    }
    flux
}

pub fn local_lax_friedrichs_speed(
    left: ConservativeState,
    right: ConservativeState,
    gas: IdealGas,
    normal: [f64; 3],
) -> f64 {
    left.max_wave_speed(gas, normal)
        .max(right.max_wave_speed(gas, normal))
}

pub fn isentropic_area_ratio(mach: f64, gamma: f64) -> f64 {
    let m = mach.max(1.0e-12);
    let factor = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * m * m);
    (1.0 / m) * factor.powf((gamma + 1.0) / (2.0 * (gamma - 1.0)))
}

pub fn sod_shock_tube_states(gas: IdealGas) -> (ConservativeState, ConservativeState) {
    (
        ConservativeState::from_primitive(1.0, [0.0, 0.0, 0.0], 1.0, gas),
        ConservativeState::from_primitive(0.125, [0.0, 0.0, 0.0], 0.1, gas),
    )
}

#[cfg(test)]
mod transport_tests {
    use super::*;

    #[test]
    fn sutherland_air_matches_reference_and_increases_with_temperature() {
        let law = ViscosityModel::AIR;
        let reference = law.dynamic_viscosity(273.15);
        assert!((reference - 1.716e-5).abs() < 1.0e-12);
        assert!(law.dynamic_viscosity(500.0) > reference);
        assert!(law.thermal_conductivity(300.0, IdealGas::AIR, 0.72) > 0.0);
    }
}
