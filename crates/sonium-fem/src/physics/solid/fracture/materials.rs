//! Material data for phase-field fracture.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseFieldFractureMaterial {
    pub critical_energy_release_rate: f64,
    pub length_scale: f64,
    pub residual_stiffness: f64,
}

impl PhaseFieldFractureMaterial {
    pub fn degradation(&self, damage: f64) -> f64 {
        let d = damage.clamp(0.0, 1.0);
        (1.0 - self.residual_stiffness) * (1.0 - d).powi(2) + self.residual_stiffness
    }

    pub fn crack_surface_density(&self, damage: f64, grad_damage_norm2: f64) -> f64 {
        0.5 * self.critical_energy_release_rate
            * (damage * damage / self.length_scale + self.length_scale * grad_damage_norm2)
    }
}
