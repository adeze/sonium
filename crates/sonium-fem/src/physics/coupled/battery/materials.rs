//! Porous-electrode and active-particle material data.

pub const FARADAY_CONSTANT: f64 = 96_485.332_12;
pub const GAS_CONSTANT: f64 = 8.314_462_618;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatteryRegionKind {
    NegativeElectrode,
    Separator,
    PositiveElectrode,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatteryMaterial {
    pub region: BatteryRegionKind,
    pub porosity: f64,
    pub bruggeman_exponent: f64,
    pub electrolyte_diffusivity: f64,
    pub electrolyte_conductivity: f64,
    pub solid_conductivity: f64,
    pub transference_number: f64,
    pub particle_radius: f64,
    pub solid_diffusivity: f64,
    pub maximum_solid_concentration: f64,
    pub specific_surface_area: f64,
    pub exchange_current_density: f64,
    pub charge_transfer_coefficient: f64,
    pub density: f64,
    pub specific_heat: f64,
    pub thermal_conductivity: f64,
}

impl BatteryMaterial {
    pub fn validate(self) -> Result<Self, BatteryError> {
        if !(0.0..1.0).contains(&self.porosity) {
            return Err(BatteryError::InvalidPorosity);
        }
        if !(0.0..1.0).contains(&self.transference_number) {
            return Err(BatteryError::InvalidTransferenceNumber);
        }
        let positive = [
            self.electrolyte_diffusivity,
            self.electrolyte_conductivity,
            self.particle_radius,
            self.solid_diffusivity,
            self.maximum_solid_concentration,
            self.density,
            self.specific_heat,
            self.thermal_conductivity,
        ];
        if positive
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(BatteryError::NonPositiveCoefficient);
        }
        if self.region != BatteryRegionKind::Separator
            && (self.solid_conductivity <= 0.0
                || self.specific_surface_area <= 0.0
                || self.exchange_current_density <= 0.0)
        {
            return Err(BatteryError::NonPositiveElectrodeCoefficient);
        }
        if !(0.0..=1.0).contains(&self.charge_transfer_coefficient)
            || !self.bruggeman_exponent.is_finite()
        {
            return Err(BatteryError::InvalidKinetics);
        }
        Ok(self)
    }

    pub fn effective_electrolyte_diffusivity(&self) -> f64 {
        self.electrolyte_diffusivity * self.porosity.powf(self.bruggeman_exponent)
    }

    pub fn effective_electrolyte_conductivity(&self) -> f64 {
        self.electrolyte_conductivity * self.porosity.powf(self.bruggeman_exponent)
    }

    pub fn butler_volmer_current_density(&self, overpotential: f64, temperature: f64) -> f64 {
        if self.region == BatteryRegionKind::Separator {
            return 0.0;
        }
        let alpha = self.charge_transfer_coefficient;
        let scaled = FARADAY_CONSTANT * overpotential / (GAS_CONSTANT * temperature);
        self.exchange_current_density * ((alpha * scaled).exp() - (-(1.0 - alpha) * scaled).exp())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatteryHeatBreakdown {
    pub ohmic: f64,
    pub reaction: f64,
    pub entropic: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatteryLocalHeatState {
    pub solid_current: [f64; 3],
    pub electrolyte_current: [f64; 3],
    pub solid_conductivity: f64,
    pub electrolyte_conductivity: f64,
    pub interfacial_current_density: f64,
    pub overpotential: f64,
    pub temperature: f64,
    pub entropy_coefficient: f64,
}

impl BatteryHeatBreakdown {
    pub fn total(&self) -> f64 {
        self.ohmic + self.reaction + self.entropic
    }

    pub fn from_local_state(state: BatteryLocalHeatState) -> Self {
        let squared = |vector: [f64; 3]| vector.iter().map(|value| value * value).sum::<f64>();
        Self {
            ohmic: squared(state.solid_current) / state.solid_conductivity
                + squared(state.electrolyte_current) / state.electrolyte_conductivity,
            reaction: state.interfacial_current_density * state.overpotential,
            entropic: state.interfacial_current_density
                * state.temperature
                * state.entropy_coefficient,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatteryError {
    InvalidPorosity,
    InvalidTransferenceNumber,
    NonPositiveCoefficient,
    NonPositiveElectrodeCoefficient,
    InvalidKinetics,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn electrode() -> BatteryMaterial {
        BatteryMaterial {
            region: BatteryRegionKind::NegativeElectrode,
            porosity: 0.3,
            bruggeman_exponent: 1.5,
            electrolyte_diffusivity: 1e-10,
            electrolyte_conductivity: 1.0,
            solid_conductivity: 100.0,
            transference_number: 0.4,
            particle_radius: 5e-6,
            solid_diffusivity: 1e-14,
            maximum_solid_concentration: 30_000.0,
            specific_surface_area: 3e5,
            exchange_current_density: 2.0,
            charge_transfer_coefficient: 0.5,
            density: 2000.0,
            specific_heat: 1000.0,
            thermal_conductivity: 2.0,
        }
    }

    #[test]
    fn symmetric_butler_volmer_is_antisymmetric() {
        let material = electrode();
        let forward = material.butler_volmer_current_density(0.05, 300.0);
        let reverse = material.butler_volmer_current_density(-0.05, 300.0);
        assert!((forward + reverse).abs() < 1e-12);
    }

    #[test]
    fn heat_breakdown_preserves_all_three_sources() {
        let heat = BatteryHeatBreakdown::from_local_state(BatteryLocalHeatState {
            solid_current: [2.0, 0.0, 0.0],
            electrolyte_current: [1.0, 0.0, 0.0],
            solid_conductivity: 2.0,
            electrolyte_conductivity: 1.0,
            interfacial_current_density: 3.0,
            overpotential: 0.1,
            temperature: 300.0,
            entropy_coefficient: 1e-4,
        });
        assert_eq!(heat.ohmic, 3.0);
        assert!((heat.reaction - 0.3).abs() < 1e-15);
        assert!((heat.entropic - 0.09).abs() < 1e-15);
        assert!((heat.total() - 3.39).abs() < 1e-15);
    }
}
