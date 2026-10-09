//! Beam cross-section properties

/// Beam cross-section properties.
#[derive(Debug, Clone, Copy)]
pub struct BeamSection {
    /// Young's modulus (Pa).
    pub youngs_modulus: f64,
    /// Second moment of area / moment of inertia (m^4).
    pub moment_of_inertia: f64,
    /// Cross-sectional area (m^2).
    pub area: f64,
    /// Material density (kg/m^3).
    pub density: f64,
    /// Shear correction factor for Timoshenko beams (kappa). Set to 0 for Euler-Bernoulli.
    pub shear_correction: f64,
    /// Shear modulus G (Pa). Only needed for Timoshenko.
    pub shear_modulus: f64,
}

impl BeamSection {
    /// Create an Euler-Bernoulli beam section (no shear deformation).
    pub fn euler_bernoulli(e: f64, i: f64, a: f64, rho: f64) -> Self {
        Self {
            youngs_modulus: e,
            moment_of_inertia: i,
            area: a,
            density: rho,
            shear_correction: 0.0,
            shear_modulus: 0.0,
        }
    }

    /// Create a Timoshenko beam section.
    pub fn timoshenko(e: f64, i: f64, a: f64, rho: f64, nu: f64, kappa: f64) -> Self {
        let g = e / (2.0 * (1.0 + nu));
        Self {
            youngs_modulus: e,
            moment_of_inertia: i,
            area: a,
            density: rho,
            shear_correction: kappa,
            shear_modulus: g,
        }
    }

    /// Rectangular cross-section.
    pub fn rectangular(e: f64, width: f64, height: f64, rho: f64) -> Self {
        let a = width * height;
        let i = width * height.powi(3) / 12.0;
        Self::euler_bernoulli(e, i, a, rho)
    }

    /// Circular cross-section.
    pub fn circular(e: f64, radius: f64, rho: f64) -> Self {
        let a = std::f64::consts::PI * radius * radius;
        let i = std::f64::consts::PI * radius.powi(4) / 4.0;
        Self::euler_bernoulli(e, i, a, rho)
    }

    /// Bending stiffness EI.
    pub fn ei(&self) -> f64 {
        self.youngs_modulus * self.moment_of_inertia
    }

    /// Axial stiffness EA.
    pub fn ea(&self) -> f64 {
        self.youngs_modulus * self.area
    }

    /// Mass per unit length.
    pub fn mass_per_length(&self) -> f64 {
        self.density * self.area
    }

    /// Is this a Timoshenko beam (includes shear deformation)?
    pub fn is_timoshenko(&self) -> bool {
        self.shear_correction > 0.0
    }
}

/// H-shaped (I-beam) cross-section parametrised by 3 variables.
///
/// ```text
///      b
///   |------|
///   |======|  t (flange)
///      ||
///      || t (web)    h (web height)
///      ||
///   |======|  t (flange)
///   |------|
/// ```
///
/// - `b`: flange width
/// - `h`: web height (between flanges, not total height)
/// - `t`: uniform thickness (flanges and web)
///
/// Total height = h + 2t.
#[derive(Debug, Clone, Copy)]
pub struct HSection {
    /// Flange width (m).
    pub b: f64,
    /// Web height between flanges (m).
    pub h: f64,
    /// Uniform thickness for flanges and web (m).
    pub t: f64,
}

impl HSection {
    pub fn new(b: f64, h: f64, t: f64) -> Self {
        Self { b, h, t }
    }

    /// Cross-sectional area: A = 2bt + ht = t(2b + h).
    pub fn area(&self) -> f64 {
        self.t * (2.0 * self.b + self.h)
    }

    /// Second moment of area about the strong axis:
    /// I = [b·(h+2t)³ − (b−t)·h³] / 12
    pub fn moment_of_inertia(&self) -> f64 {
        let big_h = self.h + 2.0 * self.t; // total height
        (self.b * big_h.powi(3) - (self.b - self.t) * self.h.powi(3)) / 12.0
    }

    /// Partial derivatives of area: [∂A/∂b, ∂A/∂h, ∂A/∂t].
    pub fn area_derivatives(&self) -> [f64; 3] {
        let da_db = 2.0 * self.t;
        let da_dh = self.t;
        let da_dt = 2.0 * self.b + self.h;
        [da_db, da_dh, da_dt]
    }

    /// Partial derivatives of moment of inertia: [∂I/∂b, ∂I/∂h, ∂I/∂t].
    pub fn inertia_derivatives(&self) -> [f64; 3] {
        let big_h = self.h + 2.0 * self.t;
        // ∂I/∂b = [(h+2t)³ − h³] / 12
        let di_db = (big_h.powi(3) - self.h.powi(3)) / 12.0;
        // ∂I/∂h = [3b(h+2t)² − 3(b−t)h²] / 12
        let di_dh =
            (3.0 * self.b * big_h.powi(2) - 3.0 * (self.b - self.t) * self.h.powi(2)) / 12.0;
        // ∂I/∂t = [6b(h+2t)² + h³] / 12
        let di_dt = (6.0 * self.b * big_h.powi(2) + self.h.powi(3)) / 12.0;
        [di_db, di_dh, di_dt]
    }

    /// Convert to a [`BeamSection`] for use with beam elements.
    pub fn to_beam_section(&self, youngs_modulus: f64, density: f64) -> BeamSection {
        BeamSection::euler_bernoulli(
            youngs_modulus,
            self.moment_of_inertia(),
            self.area(),
            density,
        )
    }
}
