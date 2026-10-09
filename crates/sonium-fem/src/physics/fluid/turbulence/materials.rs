//! Coefficients for turbulence transport models.

/// Spalart-Allmaras model constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpalartAllmarasConstants {
    pub cb1: f64,
    pub cb2: f64,
    pub sigma: f64,
    pub cw1: f64,
}

impl Default for SpalartAllmarasConstants {
    fn default() -> Self {
        Self {
            cb1: 0.1355,
            cb2: 0.622,
            sigma: 2.0 / 3.0,
            cw1: 3.239_067_816,
        }
    }
}

impl SpalartAllmarasConstants {
    pub fn production(&self, modified_eddy_viscosity: f64, vorticity_magnitude: f64) -> f64 {
        self.cb1 * vorticity_magnitude.max(0.0) * modified_eddy_viscosity.max(0.0)
    }

    pub fn destruction(&self, modified_eddy_viscosity: f64, wall_distance: f64) -> f64 {
        let d = wall_distance.max(1.0e-12);
        self.cw1 * modified_eddy_viscosity.max(0.0).powi(2) / (d * d)
    }

    pub fn turbulent_viscosity(
        &self,
        modified_eddy_viscosity: f64,
        molecular_viscosity: f64,
    ) -> f64 {
        let chi = modified_eddy_viscosity.max(0.0) / molecular_viscosity.max(1.0e-30);
        let cv1_cubed = 7.1_f64.powi(3);
        modified_eddy_viscosity.max(0.0) * chi.powi(3) / (chi.powi(3) + cv1_cubed)
    }
}

/// k-omega SST transport constants used by the two scalar equations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KOmegaSstConstants {
    pub beta_star: f64,
    pub beta: f64,
    pub sigma_k: f64,
    pub sigma_omega: f64,
}

impl Default for KOmegaSstConstants {
    fn default() -> Self {
        Self {
            beta_star: 0.09,
            beta: 0.075,
            sigma_k: 0.85,
            sigma_omega: 0.5,
        }
    }
}

impl KOmegaSstConstants {
    pub fn k_reaction(&self, omega: f64) -> f64 {
        self.beta_star * omega.max(0.0)
    }

    pub fn omega_reaction(&self, omega: f64) -> f64 {
        self.beta * omega.max(0.0)
    }

    pub fn production_limiter(&self, production: f64, k: f64, omega: f64) -> f64 {
        production.min(10.0 * self.beta_star * k.max(0.0) * omega.max(0.0))
    }
}

pub fn eddy_viscosity_k_omega(k: f64, omega: f64, limiter: f64) -> f64 {
    let denominator = omega.max(limiter.max(1.0e-30)).max(1.0e-30);
    k.max(0.0) / denominator
}

pub fn logarithmic_wall_velocity(
    friction_velocity: f64,
    wall_distance: f64,
    kinematic_viscosity: f64,
    kappa: f64,
    b: f64,
) -> f64 {
    let y_plus = friction_velocity * wall_distance / kinematic_viscosity.max(1.0e-30);
    if y_plus <= 1.0 {
        friction_velocity * y_plus
    } else {
        friction_velocity * (y_plus.ln() / kappa + b)
    }
}
