//! Semiconductor carrier transport coefficients.

pub const BOLTZMANN: f64 = 1.380_649e-23;
pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CarrierKind {
    Electron,
    Hole,
}

impl CarrierKind {
    pub fn charge_sign(self) -> f64 {
        match self {
            Self::Electron => -1.0,
            Self::Hole => 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CarrierMaterial {
    pub mobility: f64,
    pub lifetime: f64,
    pub temperature: f64,
    pub kind: CarrierKind,
}

impl CarrierMaterial {
    pub fn diffusivity(&self) -> f64 {
        self.mobility * BOLTZMANN * self.temperature / ELEMENTARY_CHARGE
    }

    pub fn drift_coefficient(&self) -> f64 {
        self.kind.charge_sign() * self.mobility
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DopingProfile {
    pub donors: f64,
    pub acceptors: f64,
}

impl DopingProfile {
    pub fn net_charge_density(&self) -> f64 {
        ELEMENTARY_CHARGE * (self.donors - self.acceptors)
    }

    pub fn majority_carrier_density(&self) -> f64 {
        self.donors.max(self.acceptors)
    }
}

pub fn thermal_voltage(temperature: f64) -> f64 {
    BOLTZMANN * temperature / ELEMENTARY_CHARGE
}

pub fn bernoulli_function(psi: f64) -> f64 {
    if psi.abs() < 1.0e-8 {
        1.0 - 0.5 * psi + psi * psi / 12.0
    } else {
        psi / (psi.exp() - 1.0)
    }
}

pub fn scharfetter_gummel_flux(
    diffusivity: f64,
    left_density: f64,
    right_density: f64,
    potential_drop_over_vt: f64,
    edge_length: f64,
) -> f64 {
    diffusivity / edge_length
        * (left_density * bernoulli_function(potential_drop_over_vt)
            - right_density * bernoulli_function(-potential_drop_over_vt))
}

pub fn pn_junction_built_in_potential(
    acceptor_density: f64,
    donor_density: f64,
    intrinsic_density: f64,
    temperature: f64,
) -> f64 {
    thermal_voltage(temperature)
        * (acceptor_density * donor_density / (intrinsic_density * intrinsic_density)).ln()
}
