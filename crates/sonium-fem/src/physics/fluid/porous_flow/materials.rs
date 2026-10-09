//! Material coefficients for Darcy and storage flow models.

/// Darcy porous medium parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DarcyMaterial {
    /// Permeability tensor `k` in m^2.
    pub permeability: [[f64; 3]; 3],
    /// Dynamic viscosity `mu` in Pa.s.
    pub viscosity: f64,
    /// Specific storage `S_s` in 1/Pa.
    pub specific_storage: f64,
    /// Porosity, used by coupled transport/poroelastic models.
    pub porosity: f64,
}

impl DarcyMaterial {
    pub fn isotropic(permeability: f64, viscosity: f64, specific_storage: f64) -> Self {
        Self {
            permeability: [
                [permeability, 0.0, 0.0],
                [0.0, permeability, 0.0],
                [0.0, 0.0, permeability],
            ],
            viscosity,
            specific_storage,
            porosity: 1.0,
        }
    }

    pub fn hydraulic_mobility(&self) -> [[f64; 3]; 3] {
        let mut out = [[0.0; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value = self.permeability[i][j] / self.viscosity;
            }
        }
        out
    }

    pub fn hydraulic_diffusivity_isotropic(&self) -> f64 {
        self.permeability[0][0] / (self.viscosity * self.specific_storage)
    }
}

/// Richards-equation coefficients for unsaturated flow coefficient injection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RichardsMaterial {
    /// Saturated permeability tensor `k_s`.
    pub saturated_permeability: [[f64; 3]; 3],
    /// Dynamic viscosity `mu` in Pa.s.
    pub viscosity: f64,
    /// Residual (irreducible) saturation.
    pub residual_saturation: f64,
    /// Saturation exponent in relative permeability.
    pub saturation_exponent: f64,
    /// Moisture capacity `C(h)` in 1/Pa.
    pub moisture_capacity: f64,
    /// Optional van Genuchten-Mualem retention parameters.
    pub retention: Option<VanGenuchtenMualem>,
}

impl RichardsMaterial {
    pub fn new(
        saturated_permeability: [[f64; 3]; 3],
        viscosity: f64,
        residual_saturation: f64,
        saturation_exponent: f64,
        moisture_capacity: f64,
    ) -> Self {
        Self {
            saturated_permeability,
            viscosity,
            residual_saturation,
            saturation_exponent,
            moisture_capacity,
            retention: None,
        }
    }

    pub fn with_retention(mut self, retention: VanGenuchtenMualem) -> Self {
        self.retention = Some(retention);
        self
    }

    /// Relative permeability factor `kr(s) = [(s-eps)/(1-eps)]^m`, clamped to [0,1].
    ///
    pub fn with_saturation(sat: f64, residual: f64, exponent: f64) -> f64 {
        if sat <= residual {
            return 0.0;
        }
        let s = ((sat - residual) / (1.0 - residual)).clamp(0.0, 1.0);
        s.powf(exponent.max(1.0))
    }

    pub fn hydraulic_mobility(&self, saturation: f64) -> [[f64; 3]; 3] {
        let kr = if let Some(retention) = self.retention {
            retention.relative_permeability(saturation)
        } else {
            Self::with_saturation(
                saturation,
                self.residual_saturation,
                self.saturation_exponent,
            )
        };
        let mut out = [[0.0; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value = self.saturated_permeability[i][j] * kr / self.viscosity;
            }
        }
        out
    }
}

/// van Genuchten-Mualem saturation closure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VanGenuchtenMualem {
    /// Inverse capillary-pressure scale `alpha` in 1/Pa.
    pub alpha: f64,
    /// van Genuchten exponent `n > 1`.
    pub n: f64,
    /// Pore-connectivity exponent, commonly 0.5.
    pub pore_connectivity: f64,
    /// Residual saturation.
    pub residual_saturation: f64,
    /// Saturated saturation.
    pub saturated_saturation: f64,
}

impl VanGenuchtenMualem {
    pub fn m(&self) -> f64 {
        1.0 - 1.0 / self.n
    }

    pub fn effective_saturation(&self, pressure_head: f64) -> f64 {
        if pressure_head >= 0.0 {
            return 1.0;
        }
        let arg = (self.alpha * pressure_head.abs()).powf(self.n);
        (1.0 + arg).powf(-self.m()).clamp(0.0, 1.0)
    }

    pub fn saturation_from_pressure(&self, pressure_head: f64) -> f64 {
        self.residual_saturation
            + self.effective_saturation(pressure_head)
                * (self.saturated_saturation - self.residual_saturation)
    }

    pub fn relative_permeability(&self, saturation: f64) -> f64 {
        let denom = (self.saturated_saturation - self.residual_saturation).max(f64::EPSILON);
        let se = ((saturation - self.residual_saturation) / denom).clamp(0.0, 1.0);
        if se <= 0.0 {
            return 0.0;
        }
        let m = self.m();
        let bracket = 1.0 - (1.0 - se.powf(1.0 / m)).powf(m);
        se.powf(self.pore_connectivity) * bracket * bracket
    }

    pub fn moisture_capacity(&self, pressure_head: f64, porosity: f64) -> f64 {
        if pressure_head >= 0.0 {
            return 0.0;
        }
        let h = pressure_head.abs();
        let n = self.n;
        let m = self.m();
        let alpha_h_n = (self.alpha * h).powf(n);
        let dse_dh =
            m * n * self.alpha.powf(n) * h.powf(n - 1.0) * (1.0 + alpha_h_n).powf(-m - 1.0);
        porosity * (self.saturated_saturation - self.residual_saturation) * dse_dh
    }
}

/// Biot poroelastic coefficients for displacement-pressure coupling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiotMaterial {
    pub biot_alpha: f64,
    pub storage: f64,
    pub drained_bulk_modulus: f64,
}

impl BiotMaterial {
    pub fn undrained_bulk_modulus(&self) -> f64 {
        self.drained_bulk_modulus + self.biot_alpha * self.biot_alpha / self.storage
    }

    pub fn skempton_coefficient(&self) -> f64 {
        self.biot_alpha
            / (self.biot_alpha * self.biot_alpha + self.storage * self.drained_bulk_modulus)
    }

    pub fn fast_p_wave_speed(&self, bulk_density: f64, shear_modulus: f64) -> f64 {
        let modulus = self.undrained_bulk_modulus() + 4.0 * shear_modulus / 3.0;
        (modulus / bulk_density).sqrt()
    }
}

/// Terzaghi one-dimensional consolidation coefficient `c_v = k / (mu * S_s)`.
pub fn terzaghi_consolidation_coefficient(material: &DarcyMaterial) -> f64 {
    material.hydraulic_diffusivity_isotropic()
}

/// First Fourier-term excess pore-pressure ratio for one-way drained
/// Terzaghi consolidation.
pub fn terzaghi_excess_pressure_ratio(depth_fraction: f64, time_factor: f64) -> f64 {
    let z = depth_fraction.clamp(0.0, 1.0);
    (4.0 / std::f64::consts::PI)
        * (std::f64::consts::FRAC_PI_2 * z).sin()
        * (-(std::f64::consts::PI.powi(2) * time_factor) / 4.0).exp()
}

/// Theis confined-aquifer drawdown using the Cooper-Jacob late-time
/// approximation. Valid for small `u = r^2 S / (4 T t)`.
pub fn theis_cooper_jacob_drawdown(
    pumping_rate: f64,
    transmissivity: f64,
    storativity: f64,
    radius: f64,
    time: f64,
) -> f64 {
    let u = radius * radius * storativity / (4.0 * transmissivity * time);
    pumping_rate / (4.0 * std::f64::consts::PI * transmissivity)
        * (-0.577_215_664_901_532_9 - u.ln())
}
