//! Manufactured one-dimensional linear-elasticity references.
//!
//! This module defines a static, constant-section axial bar under the
//! equilibrium convention
//!
//! ```text
//! -d(E A du/dx)/dx = b(x),
//! ```
//!
//! where `E` is Young's modulus, `A` is cross-sectional area, `u` is axial
//! displacement, and `b` is the distributed axial load in N/m.  It exposes
//! the exact displacement, strain, stress, axial force, and load so a
//! consuming discretization can check each physical term independently.
//!
//! Companion closed-form references live in the [`beams`], [`shells`], and
//! [`contact`] submodules, declared here so the analytical registry stays
//! untouched.

// Rust guideline compliant 2026-02-21

#[path = "elasticity_beams.rs"]
pub mod beams;
#[path = "elasticity_contact.rs"]
pub mod contact;
#[path = "elasticity_shells.rs"]
pub mod shells;

use std::fmt;

/// Invalid input to a manufactured one-dimensional elasticity reference.
#[derive(Debug, Clone, PartialEq)]
pub enum ElasticityManufacturedError {
    /// A model parameter is non-finite or outside the documented domain.
    InvalidParameter {
        /// Parameter name.
        field: &'static str,
        /// Required domain or invariant.
        requirement: &'static str,
    },
    /// A sampled spatial interval is invalid.
    InvalidInterval {
        /// Start coordinate in metres.
        x_min: f64,
        /// End coordinate in metres.
        x_max: f64,
    },
    /// An evaluated phase or exact term is not representable.
    EvaluationOverflow,
}

impl fmt::Display for ElasticityManufacturedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { field, requirement } => {
                write!(
                    f,
                    "manufactured elasticity parameter {field} must be {requirement}"
                )
            }
            Self::InvalidInterval { x_min, x_max } => write!(
                f,
                "manufactured elasticity interval must be finite and strictly increasing, got [{x_min}, {x_max}]"
            ),
            Self::EvaluationOverflow => write!(
                f,
                "manufactured elasticity evaluation cannot be represented in binary64"
            ),
        }
    }
}

impl std::error::Error for ElasticityManufacturedError {}

/// Exact state and equilibrium terms at one axial-bar evaluation point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticityPoint1d {
    /// Evaluation coordinate in metres.
    pub x_m: f64,
    /// Exact axial displacement `u` in metres.
    pub displacement_m: f64,
    /// Exact axial strain `du/dx`, dimensionless.
    pub axial_strain: f64,
    /// Exact axial stress `E du/dx` in Pa.
    pub axial_stress_pa: f64,
    /// Exact axial force `E A du/dx` in N.
    pub axial_force_n: f64,
    /// Exact distributed load `b = -d(E A du/dx)/dx` in N/m.
    pub distributed_load_n_per_m: f64,
}

impl ElasticityPoint1d {
    fn all_finite(self) -> bool {
        [
            self.x_m,
            self.displacement_m,
            self.axial_strain,
            self.axial_stress_pa,
            self.axial_force_n,
            self.distributed_load_n_per_m,
        ]
        .iter()
        .all(|value| value.is_finite())
    }
}

/// Uniform exact samples and equilibrium terms for an axial-bar reference.
#[derive(Debug, Clone, PartialEq)]
pub struct ElasticitySample1d {
    /// Sample coordinates in metres.
    pub positions_m: Vec<f64>,
    /// Exact axial displacements in metres.
    pub displacements_m: Vec<f64>,
    /// Exact axial strains, dimensionless.
    pub axial_strains: Vec<f64>,
    /// Exact axial stresses in Pa.
    pub axial_stresses_pa: Vec<f64>,
    /// Exact axial forces in N.
    pub axial_forces_n: Vec<f64>,
    /// Exact distributed loads in N/m.
    pub distributed_loads_n_per_m: Vec<f64>,
}

impl ElasticitySample1d {
    /// Verify a finite, consistently-sized exact field set.
    pub fn validate(&self) -> Result<(), ElasticityManufacturedError> {
        let count = self.positions_m.len();
        if count < 2 {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "positions_m",
                requirement: "at least 2 samples",
            });
        }
        if [
            self.displacements_m.len(),
            self.axial_strains.len(),
            self.axial_stresses_pa.len(),
            self.axial_forces_n.len(),
            self.distributed_loads_n_per_m.len(),
        ]
        .iter()
        .any(|&length| length != count)
        {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "sample fields",
                requirement: "the same length as positions_m",
            });
        }
        if self
            .positions_m
            .iter()
            .chain(self.displacements_m.iter())
            .chain(self.axial_strains.iter())
            .chain(self.axial_stresses_pa.iter())
            .chain(self.axial_forces_n.iter())
            .chain(self.distributed_loads_n_per_m.iter())
            .any(|value| !value.is_finite())
        {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        Ok(())
    }
}

/// A smooth manufactured solution for a constant-section static axial bar.
///
/// The exact displacement is `u(x) = U sin(q(x-x₀))`, where `U` is a finite
/// signed displacement amplitude and `q > 0` is the spatial wavenumber.  The
/// reference assumes small-strain, homogeneous, isotropic, one-dimensional
/// linear elasticity with constant positive `E` and `A`; it is not a bending,
/// nonlinear, or dynamic structural model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManufacturedElasticity1d {
    youngs_modulus_pa: f64,
    area_m2: f64,
    spatial_wavenumber_per_m: f64,
    displacement_amplitude_m: f64,
    origin_m: f64,
}

impl ManufacturedElasticity1d {
    /// Construct a checked constant-section axial-bar reference.
    pub fn try_new(
        youngs_modulus_pa: f64,
        area_m2: f64,
        spatial_wavenumber_per_m: f64,
        displacement_amplitude_m: f64,
        origin_m: f64,
    ) -> Result<Self, ElasticityManufacturedError> {
        for (field, value, positive) in [
            ("youngs_modulus_pa", youngs_modulus_pa, true),
            ("area_m2", area_m2, true),
            ("spatial_wavenumber_per_m", spatial_wavenumber_per_m, true),
            ("displacement_amplitude_m", displacement_amplitude_m, false),
            ("origin_m", origin_m, false),
        ] {
            if !value.is_finite() || (positive && value <= 0.0) {
                return Err(ElasticityManufacturedError::InvalidParameter {
                    field,
                    requirement: if positive {
                        "finite and positive"
                    } else {
                        "finite"
                    },
                });
            }
        }
        let stiffness_n = youngs_modulus_pa * area_m2;
        let load_scale = stiffness_n * spatial_wavenumber_per_m.powi(2);
        if !stiffness_n.is_finite() || !load_scale.is_finite() {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        Ok(Self {
            youngs_modulus_pa,
            area_m2,
            spatial_wavenumber_per_m,
            displacement_amplitude_m,
            origin_m,
        })
    }

    /// Young's modulus `E` in Pa.
    pub fn youngs_modulus_pa(self) -> f64 {
        self.youngs_modulus_pa
    }

    /// Cross-sectional area `A` in m².
    pub fn area_m2(self) -> f64 {
        self.area_m2
    }

    /// Spatial sinusoid wavenumber `q` in rad/m.
    pub fn spatial_wavenumber_per_m(self) -> f64 {
        self.spatial_wavenumber_per_m
    }

    /// Evaluate the exact displacement and equilibrium terms at `x`.
    pub fn try_evaluate(self, x_m: f64) -> Result<ElasticityPoint1d, ElasticityManufacturedError> {
        if !x_m.is_finite() {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "x_m",
                requirement: "finite",
            });
        }
        let phase = self.spatial_wavenumber_per_m * (x_m - self.origin_m);
        let stiffness_n = self.youngs_modulus_pa * self.area_m2;
        let load_scale = stiffness_n * self.spatial_wavenumber_per_m.powi(2);
        if !phase.is_finite() || !stiffness_n.is_finite() || !load_scale.is_finite() {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        let displacement_m = self.displacement_amplitude_m * phase.sin();
        let axial_strain =
            self.displacement_amplitude_m * self.spatial_wavenumber_per_m * phase.cos();
        let point = ElasticityPoint1d {
            x_m,
            displacement_m,
            axial_strain,
            axial_stress_pa: self.youngs_modulus_pa * axial_strain,
            axial_force_n: stiffness_n * axial_strain,
            distributed_load_n_per_m: load_scale * displacement_m,
        };
        if !point.all_finite() {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        Ok(point)
    }

    /// Sample the reference and exact equilibrium terms on a uniform grid.
    pub fn try_sample(
        self,
        x_min_m: f64,
        x_max_m: f64,
        num_points: usize,
    ) -> Result<ElasticitySample1d, ElasticityManufacturedError> {
        if num_points < 2 {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "num_points",
                requirement: "at least 2",
            });
        }
        if !x_min_m.is_finite() || !x_max_m.is_finite() || x_max_m <= x_min_m {
            return Err(ElasticityManufacturedError::InvalidInterval {
                x_min: x_min_m,
                x_max: x_max_m,
            });
        }
        let dx = (x_max_m - x_min_m) / (num_points - 1) as f64;
        if !dx.is_finite() {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        let mut sample = ElasticitySample1d {
            positions_m: Vec::with_capacity(num_points),
            displacements_m: Vec::with_capacity(num_points),
            axial_strains: Vec::with_capacity(num_points),
            axial_stresses_pa: Vec::with_capacity(num_points),
            axial_forces_n: Vec::with_capacity(num_points),
            distributed_loads_n_per_m: Vec::with_capacity(num_points),
        };
        for index in 0..num_points {
            let point = self.try_evaluate(x_min_m + index as f64 * dx)?;
            sample.positions_m.push(point.x_m);
            sample.displacements_m.push(point.displacement_m);
            sample.axial_strains.push(point.axial_strain);
            sample.axial_stresses_pa.push(point.axial_stress_pa);
            sample.axial_forces_n.push(point.axial_force_n);
            sample
                .distributed_loads_n_per_m
                .push(point.distributed_load_n_per_m);
        }
        sample.validate()?;
        Ok(sample)
    }
}

/// Exact state and equilibrium terms at one plane-strain point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticityPoint2d {
    /// Evaluation coordinates `(x, y)` in metres.
    pub position_m: [f64; 2],
    /// Exact displacement `(u_x, u_y)` in metres.
    pub displacement_m: [f64; 2],
    /// Exact infinitesimal strain `(xx, yy, xy)`, dimensionless.
    pub strain: [f64; 3],
    /// Exact Cauchy stress `(xx, yy, zz, xy)` in Pa.
    pub stress_pa: [f64; 4],
    /// Exact body force `(b_x, b_y)` in N/m^3.
    pub body_force_n_per_m3: [f64; 2],
}

impl ElasticityPoint2d {
    fn all_finite(self) -> bool {
        self.position_m
            .iter()
            .chain(self.displacement_m.iter())
            .chain(self.strain.iter())
            .chain(self.stress_pa.iter())
            .chain(self.body_force_n_per_m3.iter())
            .all(|value| value.is_finite())
    }
}

/// A smooth manufactured solution for plane-strain linear elasticity.
///
/// The exact displacement is `u_x = U sin(kx X) sin(ky Y)` and
/// `u_y = V sin(kx X) sin(ky Y)` with shifted coordinates
/// `X = x - x0`, `Y = y - y0`. Stresses follow isotropic Hooke's law
/// in plane strain and the body force closes the Navier-Cauchy
/// equilibrium `div sigma + b = 0` exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManufacturedElasticity2d {
    lame_lambda_pa: f64,
    shear_modulus_pa: f64,
    wavenumber_x_per_m: f64,
    wavenumber_y_per_m: f64,
    amplitude_x_m: f64,
    amplitude_y_m: f64,
    origin_m: [f64; 2],
}

impl ManufacturedElasticity2d {
    /// Construct a checked plane-strain reference from `E` and `nu`.
    ///
    /// # Errors
    ///
    /// Returns [`ElasticityManufacturedError`] when a modulus, wavenumber,
    /// or amplitude is non-finite, when `E <= 0`, or when `nu` leaves the
    /// open interval `(-1, 1/2)`.
    pub fn try_new(
        youngs_modulus_pa: f64,
        poisson_ratio: f64,
        wavenumber_x_per_m: f64,
        wavenumber_y_per_m: f64,
        amplitude_x_m: f64,
        amplitude_y_m: f64,
        origin_m: [f64; 2],
    ) -> Result<Self, ElasticityManufacturedError> {
        if !youngs_modulus_pa.is_finite() || youngs_modulus_pa <= 0.0 {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "youngs_modulus_pa",
                requirement: "finite and positive",
            });
        }
        if !poisson_ratio.is_finite() || !(-1.0..0.5).contains(&poisson_ratio) {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "poisson_ratio",
                requirement: "finite and inside (-1, 1/2)",
            });
        }
        for (field, value, positive) in [
            ("wavenumber_x_per_m", wavenumber_x_per_m, true),
            ("wavenumber_y_per_m", wavenumber_y_per_m, true),
            ("amplitude_x_m", amplitude_x_m, false),
            ("amplitude_y_m", amplitude_y_m, false),
            ("origin_m[0]", origin_m[0], false),
            ("origin_m[1]", origin_m[1], false),
        ] {
            if !value.is_finite() || (positive && value <= 0.0) {
                return Err(ElasticityManufacturedError::InvalidParameter {
                    field,
                    requirement: if positive {
                        "finite and positive"
                    } else {
                        "finite"
                    },
                });
            }
        }
        let shear_modulus_pa = youngs_modulus_pa / (2.0 * (1.0 + poisson_ratio));
        let lame_lambda_pa = youngs_modulus_pa * poisson_ratio
            / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
        if !shear_modulus_pa.is_finite() || !lame_lambda_pa.is_finite() {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        Ok(Self {
            lame_lambda_pa,
            shear_modulus_pa,
            wavenumber_x_per_m,
            wavenumber_y_per_m,
            amplitude_x_m,
            amplitude_y_m,
            origin_m,
        })
    }

    /// First Lame parameter in Pa.
    pub fn lame_lambda_pa(self) -> f64 {
        self.lame_lambda_pa
    }

    /// Shear modulus in Pa.
    pub fn shear_modulus_pa(self) -> f64 {
        self.shear_modulus_pa
    }

    /// Evaluate the exact displacement, stress, and body force.
    ///
    /// # Errors
    ///
    /// Returns [`ElasticityManufacturedError`] for non-finite coordinates
    /// or non-representable exact terms.
    pub fn try_evaluate(
        self,
        x_m: f64,
        y_m: f64,
    ) -> Result<ElasticityPoint2d, ElasticityManufacturedError> {
        if !x_m.is_finite() || !y_m.is_finite() {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "position_m",
                requirement: "finite",
            });
        }
        let (kx, ky) = (self.wavenumber_x_per_m, self.wavenumber_y_per_m);
        let (phase_x, phase_y) = (kx * (x_m - self.origin_m[0]), ky * (y_m - self.origin_m[1]));
        if !phase_x.is_finite() || !phase_y.is_finite() {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        let (sx, cx, sy, cy) = (phase_x.sin(), phase_x.cos(), phase_y.sin(), phase_y.cos());
        let (u_amp, v_amp) = (self.amplitude_x_m, self.amplitude_y_m);
        let (mu, lambda) = (self.shear_modulus_pa, self.lame_lambda_pa);

        let displacement_m = [u_amp * sx * sy, v_amp * sx * sy];
        let divergence = u_amp * kx * cx * sy + v_amp * ky * sx * cy;
        let laplacian = kx * kx + ky * ky;
        let grad_div = [
            -u_amp * kx * kx * sx * sy + v_amp * kx * ky * cx * cy,
            u_amp * kx * ky * cx * cy - v_amp * ky * ky * sx * sy,
        ];
        let body_force_n_per_m3 = [
            mu * laplacian * displacement_m[0] - (lambda + mu) * grad_div[0],
            mu * laplacian * displacement_m[1] - (lambda + mu) * grad_div[1],
        ];
        let strain = [
            u_amp * kx * cx * sy,
            v_amp * ky * sx * cy,
            0.5 * (u_amp * ky * sx * cy + v_amp * kx * cx * sy),
        ];
        let stress_pa = [
            lambda * divergence + 2.0 * mu * strain[0],
            lambda * divergence + 2.0 * mu * strain[1],
            lambda * divergence,
            2.0 * mu * strain[2],
        ];
        let point = ElasticityPoint2d {
            position_m: [x_m, y_m],
            displacement_m,
            strain,
            stress_pa,
            body_force_n_per_m3,
        };
        if !point.all_finite() {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        Ok(point)
    }
}

/// Exact state and equilibrium terms at one 3-D elasticity point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticityPoint3d {
    /// Evaluation coordinates `(x, y, z)` in metres.
    pub position_m: [f64; 3],
    /// Exact displacement `(u_x, u_y, u_z)` in metres.
    pub displacement_m: [f64; 3],
    /// Exact infinitesimal strain in Voigt order `xx, yy, zz, xy, yz, zx`.
    pub strain: [f64; 6],
    /// Exact Cauchy stress in Voigt order `xx, yy, zz, xy, yz, zx`, Pa.
    pub stress_pa: [f64; 6],
    /// Exact body force `(b_x, b_y, b_z)` in N/m^3.
    pub body_force_n_per_m3: [f64; 3],
}

impl ElasticityPoint3d {
    fn all_finite(self) -> bool {
        self.position_m
            .iter()
            .chain(self.displacement_m.iter())
            .chain(self.strain.iter())
            .chain(self.stress_pa.iter())
            .chain(self.body_force_n_per_m3.iter())
            .all(|value| value.is_finite())
    }
}

/// A smooth manufactured solution for 3-D linear elasticity.
///
/// The exact displacement is `u_i = A_i sin(kx X) sin(ky Y) sin(kz Z)`
/// per component. Stresses follow isotropic Hooke's law and the body
/// force closes `div sigma + b = 0` exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManufacturedElasticity3d {
    lame_lambda_pa: f64,
    shear_modulus_pa: f64,
    wavenumber_per_m: [f64; 3],
    amplitude_m: [f64; 3],
    origin_m: [f64; 3],
}

impl ManufacturedElasticity3d {
    /// Construct a checked 3-D reference from `E` and `nu`.
    ///
    /// # Errors
    ///
    /// Returns [`ElasticityManufacturedError`] when a modulus, wavenumber,
    /// or amplitude is non-finite, when `E <= 0`, or when `nu` leaves the
    /// open interval `(-1, 1/2)`.
    pub fn try_new(
        youngs_modulus_pa: f64,
        poisson_ratio: f64,
        wavenumber_per_m: [f64; 3],
        amplitude_m: [f64; 3],
        origin_m: [f64; 3],
    ) -> Result<Self, ElasticityManufacturedError> {
        if !youngs_modulus_pa.is_finite() || youngs_modulus_pa <= 0.0 {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "youngs_modulus_pa",
                requirement: "finite and positive",
            });
        }
        if !poisson_ratio.is_finite() || !(-1.0..0.5).contains(&poisson_ratio) {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "poisson_ratio",
                requirement: "finite and inside (-1, 1/2)",
            });
        }
        for (field, value, positive) in [
            ("wavenumber_per_m[0]", wavenumber_per_m[0], true),
            ("wavenumber_per_m[1]", wavenumber_per_m[1], true),
            ("wavenumber_per_m[2]", wavenumber_per_m[2], true),
            ("amplitude_m[0]", amplitude_m[0], false),
            ("amplitude_m[1]", amplitude_m[1], false),
            ("amplitude_m[2]", amplitude_m[2], false),
            ("origin_m[0]", origin_m[0], false),
            ("origin_m[1]", origin_m[1], false),
            ("origin_m[2]", origin_m[2], false),
        ] {
            if !value.is_finite() || (positive && value <= 0.0) {
                return Err(ElasticityManufacturedError::InvalidParameter {
                    field,
                    requirement: if positive {
                        "finite and positive"
                    } else {
                        "finite"
                    },
                });
            }
        }
        let shear_modulus_pa = youngs_modulus_pa / (2.0 * (1.0 + poisson_ratio));
        let lame_lambda_pa = youngs_modulus_pa * poisson_ratio
            / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
        if !shear_modulus_pa.is_finite() || !lame_lambda_pa.is_finite() {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        Ok(Self {
            lame_lambda_pa,
            shear_modulus_pa,
            wavenumber_per_m,
            amplitude_m,
            origin_m,
        })
    }

    /// Evaluate the exact displacement, stress, and body force.
    ///
    /// # Errors
    ///
    /// Returns [`ElasticityManufacturedError`] for non-finite coordinates
    /// or non-representable exact terms.
    pub fn try_evaluate(
        self,
        x_m: f64,
        y_m: f64,
        z_m: f64,
    ) -> Result<ElasticityPoint3d, ElasticityManufacturedError> {
        if ![x_m, y_m, z_m].iter().all(|value| value.is_finite()) {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "position_m",
                requirement: "finite",
            });
        }
        let k = self.wavenumber_per_m;
        let phases = [
            k[0] * (x_m - self.origin_m[0]),
            k[1] * (y_m - self.origin_m[1]),
            k[2] * (z_m - self.origin_m[2]),
        ];
        if !phases.iter().all(|phase| phase.is_finite()) {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        let (sx, cx) = (phases[0].sin(), phases[0].cos());
        let (sy, cy) = (phases[1].sin(), phases[1].cos());
        let (sz, cz) = (phases[2].sin(), phases[2].cos());
        let a = self.amplitude_m;
        let (mu, lambda) = (self.shear_modulus_pa, self.lame_lambda_pa);

        let shape = sx * sy * sz;
        let displacement_m = [a[0] * shape, a[1] * shape, a[2] * shape];
        let divergence =
            a[0] * k[0] * cx * sy * sz + a[1] * k[1] * sx * cy * sz + a[2] * k[2] * sx * sy * cz;
        let laplacian = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
        let grad_div = [
            -a[0] * k[0] * k[0] * shape
                + a[1] * k[0] * k[1] * cx * cy * sz
                + a[2] * k[0] * k[2] * cx * sy * cz,
            a[0] * k[0] * k[1] * cx * cy * sz - a[1] * k[1] * k[1] * shape
                + a[2] * k[1] * k[2] * sx * cy * cz,
            a[0] * k[0] * k[2] * cx * sy * cz + a[1] * k[1] * k[2] * sx * cy * cz
                - a[2] * k[2] * k[2] * shape,
        ];
        let body_force_n_per_m3 = [
            mu * laplacian * displacement_m[0] - (lambda + mu) * grad_div[0],
            mu * laplacian * displacement_m[1] - (lambda + mu) * grad_div[1],
            mu * laplacian * displacement_m[2] - (lambda + mu) * grad_div[2],
        ];
        let strain = [
            a[0] * k[0] * cx * sy * sz,
            a[1] * k[1] * sx * cy * sz,
            a[2] * k[2] * sx * sy * cz,
            0.5 * (a[0] * k[1] * sx * cy * sz + a[1] * k[0] * cx * sy * sz),
            0.5 * (a[1] * k[2] * sx * sy * cz + a[2] * k[1] * sx * cy * sz),
            0.5 * (a[2] * k[0] * cx * sy * sz + a[0] * k[2] * sx * sy * cz),
        ];
        let mut stress_pa = [0.0; 6];
        for i in 0..3 {
            stress_pa[i] = lambda * divergence + 2.0 * mu * strain[i];
        }
        for i in 3..6 {
            stress_pa[i] = 2.0 * mu * strain[i];
        }
        let point = ElasticityPoint3d {
            position_m: [x_m, y_m, z_m],
            displacement_m,
            strain,
            stress_pa,
            body_force_n_per_m3,
        };
        if !point.all_finite() {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        Ok(point)
    }
}

/// Exact state of a 1-D compressible neo-Hookean bar point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperelasticBarPoint {
    /// Evaluation coordinate in metres.
    pub x_m: f64,
    /// Exact axial stretch `lambda > 0`, dimensionless.
    pub stretch: f64,
    /// Exact nominal stress `P` in Pa.
    pub nominal_stress_pa: f64,
    /// Exact tangent modulus `dP/dlambda` in Pa.
    pub tangent_pa: f64,
    /// Exact body force `B = -dP/dX` in N/m^3.
    pub body_force_n_per_m3: f64,
}

impl HyperelasticBarPoint {
    fn all_finite(self) -> bool {
        [
            self.x_m,
            self.stretch,
            self.nominal_stress_pa,
            self.tangent_pa,
            self.body_force_n_per_m3,
        ]
        .iter()
        .all(|value| value.is_finite())
    }
}

/// A manufactured solution for a 1-D compressible neo-Hookean bar.
///
/// The exact stretch is `lambda(X) = 1 + a sin(k(X - X0))` with
/// `|a| < 1`, the nominal stress is
/// `P = mu lambda + (lambda_L ln lambda - mu) / lambda`, and the body
/// force `B = -dP/dX` closes equilibrium exactly. The tangent
/// `dP/dlambda` lets Newton solvers check their linearization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperelasticBarMms {
    shear_modulus_pa: f64,
    lame_lambda_pa: f64,
    amplitude: f64,
    wavenumber_per_m: f64,
    origin_m: f64,
}

impl HyperelasticBarMms {
    /// Construct a checked neo-Hookean bar reference.
    ///
    /// # Errors
    ///
    /// Returns [`ElasticityManufacturedError`] when a parameter is
    /// non-finite, when `mu <= 0` or `lambda_L < 0`, when
    /// `|amplitude| >= 1` (which would admit non-positive stretch), or
    /// when the wavenumber is not positive.
    pub fn try_new(
        shear_modulus_pa: f64,
        lame_lambda_pa: f64,
        amplitude: f64,
        wavenumber_per_m: f64,
        origin_m: f64,
    ) -> Result<Self, ElasticityManufacturedError> {
        if !shear_modulus_pa.is_finite() || shear_modulus_pa <= 0.0 {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "shear_modulus_pa",
                requirement: "finite and positive",
            });
        }
        if !lame_lambda_pa.is_finite() || lame_lambda_pa < 0.0 {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "lame_lambda_pa",
                requirement: "finite and non-negative",
            });
        }
        if !amplitude.is_finite() || amplitude.abs() >= 1.0 {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "amplitude",
                requirement: "finite with absolute value below 1",
            });
        }
        if !wavenumber_per_m.is_finite() || wavenumber_per_m <= 0.0 {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "wavenumber_per_m",
                requirement: "finite and positive",
            });
        }
        if !origin_m.is_finite() {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "origin_m",
                requirement: "finite",
            });
        }
        Ok(Self {
            shear_modulus_pa,
            lame_lambda_pa,
            amplitude,
            wavenumber_per_m,
            origin_m,
        })
    }

    /// Evaluate the exact stretch, stress, tangent, and body force.
    ///
    /// # Errors
    ///
    /// Returns [`ElasticityManufacturedError`] for non-finite coordinates
    /// or non-representable exact terms.
    pub fn try_evaluate(
        self,
        x_m: f64,
    ) -> Result<HyperelasticBarPoint, ElasticityManufacturedError> {
        if !x_m.is_finite() {
            return Err(ElasticityManufacturedError::InvalidParameter {
                field: "x_m",
                requirement: "finite",
            });
        }
        let phase = self.wavenumber_per_m * (x_m - self.origin_m);
        if !phase.is_finite() {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        let stretch = 1.0 + self.amplitude * phase.sin();
        let stretch_gradient = self.amplitude * self.wavenumber_per_m * phase.cos();
        if stretch <= 0.0 || !stretch.is_finite() || !stretch_gradient.is_finite() {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        let mu = self.shear_modulus_pa;
        let lambda_lame = self.lame_lambda_pa;
        let ln_stretch = stretch.ln();
        let nominal_stress_pa = mu * stretch + (lambda_lame * ln_stretch - mu) / stretch;
        let tangent_pa =
            mu + lambda_lame * (1.0 - ln_stretch) / stretch.powi(2) + mu / stretch.powi(2);
        let point = HyperelasticBarPoint {
            x_m,
            stretch,
            nominal_stress_pa,
            tangent_pa,
            body_force_n_per_m3: -tangent_pa * stretch_gradient,
        };
        if !point.all_finite() {
            return Err(ElasticityManufacturedError::EvaluationOverflow);
        }
        Ok(point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    fn reference() -> ManufacturedElasticity1d {
        ManufacturedElasticity1d::try_new(70.0e9, 2.5e-4, 2.3, 1.2e-3, -0.15).unwrap()
    }

    #[test]
    fn exact_terms_close_axial_equilibrium() {
        let reference = reference();
        let x = 0.45;
        let h = 1e-5;
        let point = reference.try_evaluate(x).unwrap();
        let left = reference.try_evaluate(x - h).unwrap().axial_force_n;
        let right = reference.try_evaluate(x + h).unwrap().axial_force_n;
        assert_abs_diff_eq!(
            -(right - left) / (2.0 * h),
            point.distributed_load_n_per_m,
            epsilon = 2e-4
        );
        assert_abs_diff_eq!(
            point.axial_stress_pa * reference.area_m2(),
            point.axial_force_n,
            epsilon = 1e-9
        );
    }

    #[test]
    fn dirichlet_endpoints_and_strain_match_exact_formula() {
        let length = 2.0;
        let reference =
            ManufacturedElasticity1d::try_new(210.0e9, 1.0e-4, PI / length, 2.0e-3, 0.0).unwrap();
        let sample = reference.try_sample(0.0, length, 9).unwrap();
        assert_abs_diff_eq!(sample.displacements_m[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(
            sample.displacements_m.last().copied().unwrap(),
            0.0,
            epsilon = 1e-15
        );
        assert_abs_diff_eq!(
            sample.axial_strains[0],
            2.0e-3 * PI / length,
            epsilon = 1e-15
        );
    }

    #[test]
    fn displacement_and_strain_match_centered_difference() {
        let reference = reference();
        let x = 0.45;
        let h = 1e-5;
        let point = reference.try_evaluate(x).unwrap();
        let left = reference.try_evaluate(x - h).unwrap().displacement_m;
        let right = reference.try_evaluate(x + h).unwrap().displacement_m;
        assert_abs_diff_eq!(
            (right - left) / (2.0 * h),
            point.axial_strain,
            epsilon = 1e-12
        );
    }

    #[test]
    fn checked_admission_rejects_bad_models_and_samples() {
        assert!(matches!(
            ManufacturedElasticity1d::try_new(0.0, 1.0, 1.0, 1.0, 0.0),
            Err(ElasticityManufacturedError::InvalidParameter {
                field: "youngs_modulus_pa",
                ..
            })
        ));
        let reference = reference();
        assert!(matches!(
            reference.try_evaluate(f64::NAN),
            Err(ElasticityManufacturedError::InvalidParameter { field: "x_m", .. })
        ));
        assert!(matches!(
            reference.try_sample(1.0, 0.0, 4),
            Err(ElasticityManufacturedError::InvalidInterval { .. })
        ));
        assert!(matches!(
            reference.try_sample(0.0, 1.0, 1),
            Err(ElasticityManufacturedError::InvalidParameter {
                field: "num_points",
                ..
            })
        ));
    }

    #[test]
    fn plane_strain_reference_closes_equilibrium() {
        let reference =
            ManufacturedElasticity2d::try_new(70.0e9, 0.33, 2.1, 1.7, 1.1e-3, -0.9e-3, [0.1, -0.2])
                .unwrap();
        let (x, y, h) = (0.45, -0.3, 1e-6);
        let point = reference.try_evaluate(x, y).unwrap();
        // div sigma + b = 0 checked by centered differences.
        let dxx = (reference.try_evaluate(x + h, y).unwrap().stress_pa[0]
            - reference.try_evaluate(x - h, y).unwrap().stress_pa[0])
            / (2.0 * h);
        let dxy_dy = (reference.try_evaluate(x, y + h).unwrap().stress_pa[3]
            - reference.try_evaluate(x, y - h).unwrap().stress_pa[3])
            / (2.0 * h);
        let dyx_dx = (reference.try_evaluate(x + h, y).unwrap().stress_pa[3]
            - reference.try_evaluate(x - h, y).unwrap().stress_pa[3])
            / (2.0 * h);
        let dyy = (reference.try_evaluate(x, y + h).unwrap().stress_pa[1]
            - reference.try_evaluate(x, y - h).unwrap().stress_pa[1])
            / (2.0 * h);
        let scale = point.body_force_n_per_m3[0].abs().max(1.0);
        assert_abs_diff_eq!(
            dxx + dxy_dy + point.body_force_n_per_m3[0],
            0.0,
            epsilon = 1e-6 * scale
        );
        let scale = point.body_force_n_per_m3[1].abs().max(1.0);
        assert_abs_diff_eq!(
            dyx_dx + dyy + point.body_force_n_per_m3[1],
            0.0,
            epsilon = 1e-6 * scale
        );
        // Strain matches centered differences of displacement.
        let dux_dx = (reference.try_evaluate(x + h, y).unwrap().displacement_m[0]
            - reference.try_evaluate(x - h, y).unwrap().displacement_m[0])
            / (2.0 * h);
        assert_abs_diff_eq!(dux_dx, point.strain[0], epsilon = 1e-12);
        // Hooke's law in plane strain.
        let mu = reference.shear_modulus_pa();
        let lambda = reference.lame_lambda_pa();
        let trace = point.strain[0] + point.strain[1];
        assert_abs_diff_eq!(
            point.stress_pa[2],
            lambda * trace,
            epsilon = 1e-6 * point.stress_pa[2].abs().max(1.0)
        );
        assert_abs_diff_eq!(
            point.stress_pa[0],
            lambda * trace + 2.0 * mu * point.strain[0],
            epsilon = 1e-6 * point.stress_pa[0].abs().max(1.0)
        );
    }

    #[test]
    fn three_dimensional_reference_closes_equilibrium() {
        let reference = ManufacturedElasticity3d::try_new(
            210.0e9,
            0.3,
            [1.9, 2.3, 1.3],
            [1.0e-3, 0.7e-3, -1.2e-3],
            [0.0, 0.1, -0.1],
        )
        .unwrap();
        let (x, y, z, h) = (0.4, -0.25, 0.15, 1e-6);
        let point = reference.try_evaluate(x, y, z).unwrap();
        // First equilibrium row: dxx(sxx) + dy(sxy) + dz(sxz) + bx = 0.
        let dxx = (reference.try_evaluate(x + h, y, z).unwrap().stress_pa[0]
            - reference.try_evaluate(x - h, y, z).unwrap().stress_pa[0])
            / (2.0 * h);
        let dxy = (reference.try_evaluate(x, y + h, z).unwrap().stress_pa[3]
            - reference.try_evaluate(x, y - h, z).unwrap().stress_pa[3])
            / (2.0 * h);
        let dxz = (reference.try_evaluate(x, y, z + h).unwrap().stress_pa[5]
            - reference.try_evaluate(x, y, z - h).unwrap().stress_pa[5])
            / (2.0 * h);
        let scale = point.body_force_n_per_m3[0].abs().max(1.0);
        assert_abs_diff_eq!(
            dxx + dxy + dxz + point.body_force_n_per_m3[0],
            0.0,
            epsilon = 1e-6 * scale
        );
        // Third equilibrium row: dxz + dyz + dzz + bz = 0.
        let dzx = (reference.try_evaluate(x + h, y, z).unwrap().stress_pa[5]
            - reference.try_evaluate(x - h, y, z).unwrap().stress_pa[5])
            / (2.0 * h);
        let dyz = (reference.try_evaluate(x, y + h, z).unwrap().stress_pa[4]
            - reference.try_evaluate(x, y - h, z).unwrap().stress_pa[4])
            / (2.0 * h);
        let dzz = (reference.try_evaluate(x, y, z + h).unwrap().stress_pa[2]
            - reference.try_evaluate(x, y, z - h).unwrap().stress_pa[2])
            / (2.0 * h);
        let scale = point.body_force_n_per_m3[2].abs().max(1.0);
        assert_abs_diff_eq!(
            dzx + dyz + dzz + point.body_force_n_per_m3[2],
            0.0,
            epsilon = 1e-6 * scale
        );
        // Rejects incompressible-limit Poisson ratios.
        assert!(matches!(
            ManufacturedElasticity3d::try_new(
                1.0,
                0.5,
                [1.0, 1.0, 1.0],
                [0.1, 0.1, 0.1],
                [0.0, 0.0, 0.0]
            ),
            Err(ElasticityManufacturedError::InvalidParameter {
                field: "poisson_ratio",
                ..
            })
        ));
    }

    #[test]
    fn hyperelastic_bar_matches_stress_and_equilibrium() {
        let mu = 80.0e9;
        let lambda = 120.0e9;
        let reference = HyperelasticBarMms::try_new(mu, lambda, 0.2, 2.0, 0.1).unwrap();
        let x = 0.45;
        let h = 1e-7;
        let point = reference.try_evaluate(x).unwrap();
        // Closed form at the evaluation point.
        let stretch = 1.0 + 0.2 * (2.0 * (x - 0.1)).sin();
        assert_abs_diff_eq!(point.stretch, stretch, epsilon = 1e-15);
        assert_abs_diff_eq!(
            point.nominal_stress_pa,
            mu * stretch + (lambda * stretch.ln() - mu) / stretch,
            epsilon = 1e-6
        );
        // Tangent matches centered differences of the stress.
        let forward = reference.try_evaluate(x + h).unwrap();
        let backward = reference.try_evaluate(x - h).unwrap();
        let d_stress = (forward.nominal_stress_pa - backward.nominal_stress_pa) / (2.0 * h);
        assert_abs_diff_eq!(
            d_stress + point.body_force_n_per_m3,
            0.0,
            epsilon = 1e-3 * point.body_force_n_per_m3.abs().max(1.0)
        );
        let d_stretch = (forward.stretch - backward.stretch) / (2.0 * h);
        assert_abs_diff_eq!(
            point.tangent_pa * d_stretch,
            d_stress,
            epsilon = 1e-6 * d_stress.abs().max(1.0)
        );
        // Unit stretch recovers zero stress.
        let rest = HyperelasticBarMms::try_new(mu, lambda, 0.0, 2.0, 0.0)
            .unwrap()
            .try_evaluate(0.3)
            .unwrap();
        assert_abs_diff_eq!(rest.nominal_stress_pa, 0.0, epsilon = 1e-9);
        // Amplitudes admitting non-positive stretch are rejected.
        assert!(matches!(
            HyperelasticBarMms::try_new(mu, lambda, 1.0, 2.0, 0.0),
            Err(ElasticityManufacturedError::InvalidParameter {
                field: "amplitude",
                ..
            })
        ));
    }
}
