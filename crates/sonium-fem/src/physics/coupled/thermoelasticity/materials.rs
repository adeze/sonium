//! Thermoelastic material parameters.

/// Small-strain isotropic thermoelastic material.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermoelasticMaterial {
    pub young_modulus: f64,
    pub poisson_ratio: f64,
    pub thermal_expansion: f64,
    pub reference_temperature: f64,
}

impl ThermoelasticMaterial {
    pub fn thermal_strain(&self, temperature: f64) -> f64 {
        self.thermal_expansion * (temperature - self.reference_temperature)
    }

    pub fn lame_lambda(&self) -> f64 {
        self.young_modulus * self.poisson_ratio
            / ((1.0 + self.poisson_ratio) * (1.0 - 2.0 * self.poisson_ratio))
    }

    pub fn shear_modulus(&self) -> f64 {
        self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    pub fn constrained_thermal_stress_3d(&self, temperature: f64) -> f64 {
        let bulk = self.lame_lambda() + 2.0 * self.shear_modulus() / 3.0;
        3.0 * bulk * self.thermal_expansion * (temperature - self.reference_temperature)
    }

    pub fn constrained_thermal_stress_1d(&self, temperature: f64) -> f64 {
        self.young_modulus * self.thermal_strain(temperature)
    }
}

/// Homogeneous layer for thermoelastic laminate/beam checks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermoelasticLayer {
    pub young_modulus: f64,
    pub thickness: f64,
    pub thermal_expansion: f64,
}

/// Free curvature of a perfectly bonded thermoelastic strip.
///
/// Uses classical Euler-Bernoulli section resultants with unit width and solves
/// `N = 0`, `M = 0` for axial strain and curvature under thermal strain.
pub fn bonded_layer_free_curvature(layers: &[ThermoelasticLayer], delta_temperature: f64) -> f64 {
    assert!(
        !layers.is_empty(),
        "at least one thermoelastic layer is required"
    );
    let mut a = 0.0;
    let mut b = 0.0;
    let mut d = 0.0;
    let mut nt = 0.0;
    let mut mt = 0.0;
    let mut z0 = 0.0;

    for layer in layers {
        let z1 = z0 + layer.thickness;
        let area = z1 - z0;
        let first = 0.5 * (z1 * z1 - z0 * z0);
        let second = (z1 * z1 * z1 - z0 * z0 * z0) / 3.0;
        let e = layer.young_modulus;
        let thermal = layer.thermal_expansion * delta_temperature;
        a += e * area;
        b += e * first;
        d += e * second;
        nt += e * thermal * area;
        mt += e * thermal * first;
        z0 = z1;
    }

    let det = a * d - b * b;
    assert!(
        det.abs() > f64::EPSILON,
        "thermoelastic section is singular"
    );
    (a * mt - b * nt) / det
}
