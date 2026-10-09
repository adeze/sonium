//! CPU constant-panel 2-D boundary element scattering.
//!
//! This module is deliberately separate from the surface-element kernels in
//! [`super::assembly`].  A 2-D boundary is a line with the outgoing Helmholtz
//! Green function
//!
//! ```text
//! G₂(r) = i H₀⁽¹⁾(k r) / 4
//! ```
//!
//! rather than a three-dimensional surface with an `1/(4πr)` kernel.  The
//! implementation below is a production CPU slice for a rigid circular
//! cylinder.  It uses constant line panels, four-point Gauss integration, and
//! the exterior single-layer normal-derivative equation
//!
//! ```text
//! (-1/2 I + K') σ = -∂ₙ p_inc.
//! ```
//!
//! The returned far-field amplitude is normalized by
//!
//! ```text
//! p_s(r d) = exp(i h (k r + π/4)) F(d) / sqrt(8π k r) + O(r⁻³ᐟ²),
//! ```
//!
//! where `h` is the signed outgoing spatial factor carried by
//! [`PhysicsParams`].  Consequently `scattered_power_per_length` reports the
//! physically meaningful 2-D power flux per unit cylinder length, with the
//! peak/RMS factor taken from the XEM medium when the checked XEM constructor
//! is used.

use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;
use std::f64::consts::PI;
use thiserror::Error;

use crate::core::types::{PhysicsParams, PhysicsParamsError};
use sonium_wave::special::helmholtz::greens_function_2d;

const GAUSS_POINTS: [f64; 4] = [
    -0.861_136_311_594_052_6,
    -0.339_981_043_584_856_3,
    0.339_981_043_584_856_3,
    0.861_136_311_594_052_6,
];
const GAUSS_WEIGHTS: [f64; 4] = [
    0.347_854_845_137_453_8,
    0.652_145_154_862_546_1,
    0.652_145_154_862_546_1,
    0.347_854_845_137_453_8,
];
const MIN_SCALE: f64 = f64::MIN_POSITIVE;

/// A straight constant line panel on a 2-D boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoDBoundaryPanel {
    /// Counter-clockwise panel start point in metres.
    pub start: [f64; 2],
    /// Counter-clockwise panel end point in metres.
    pub end: [f64; 2],
    /// Panel midpoint in metres.
    pub center: [f64; 2],
    /// Outward unit normal at the panel midpoint.
    pub normal: [f64; 2],
    /// Panel length in metres.
    pub length: f64,
}

/// A checked incident plane wave for the 2-D solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoDPlaneWave {
    direction: [f64; 2],
    amplitude: Complex64,
}

impl TwoDPlaneWave {
    /// Construct a plane wave from a finite non-zero direction and a finite
    /// complex pressure amplitude.
    pub fn try_new(direction: [f64; 2], amplitude: Complex64) -> Result<Self, TwoDBemError> {
        let norm = direction[0].hypot(direction[1]);
        if !norm.is_finite() || norm <= 0.0 {
            return Err(TwoDBemError::InvalidIncidentDirection { direction });
        }
        if !complex_is_finite(amplitude) {
            return Err(TwoDBemError::NonFiniteInput {
                quantity: "incident amplitude",
            });
        }
        Ok(Self {
            direction: [direction[0] / norm, direction[1] / norm],
            amplitude,
        })
    }

    /// Unit peak/RMS pressure travelling in the positive x direction.
    pub fn plane_wave_x() -> Self {
        Self {
            direction: [1.0, 0.0],
            amplitude: Complex64::new(1.0, 0.0),
        }
    }

    /// Return the normalized propagation direction.
    pub fn direction(self) -> [f64; 2] {
        self.direction
    }

    /// Return the complex pressure amplitude.
    pub fn amplitude(self) -> Complex64 {
        self.amplitude
    }
}

/// Errors from checked 2-D BEM construction, solve, field, and power paths.
#[derive(Debug, Error, PartialEq)]
pub enum TwoDBemError {
    /// The cylinder radius was not finite and positive.
    #[error("2-D cylinder radius must be finite and positive, got {value}")]
    InvalidRadius {
        /// Supplied cylinder radius in metres.
        value: f64,
    },
    /// The line mesh does not have enough panels for a closed cylinder.
    #[error("2-D cylinder panel count must be at least 8, got {value}")]
    InvalidPanelCount {
        /// Supplied number of line panels.
        value: usize,
    },
    /// A Fourier shape deformation was invalid or broke the star-shaped bound.
    #[error("2-D cylinder deformation invalid: {reason}")]
    InvalidDeformation {
        /// Why the deformation was rejected.
        reason: &'static str,
    },
    /// The underlying BEM physical quantities or convention are invalid.
    #[error("invalid 2-D BEM physics: {0}")]
    InvalidPhysics(#[source] PhysicsParamsError),
    /// exterior medium failed physical validation.
    #[error("invalid exterior 2-D BEM physics: {0}")]
    InvalidExteriorPhysics(#[source] PhysicsParamsError),
    /// interior medium failed physical validation.
    #[error("invalid interior 2-D BEM physics: {0}")]
    InvalidInteriorPhysics(#[source] PhysicsParamsError),
    /// exterior and interior frequencies do not describe one harmonic solve.
    #[error("2-D transmission frequencies differ: exterior {exterior}, interior {interior}")]
    FrequencyMismatch {
        /// Exterior medium frequency.
        exterior: f64,
        /// Interior medium frequency.
        interior: f64,
    },
    /// exterior and interior Green functions use incompatible spatial signs.
    #[error("2-D transmission harmonic factors differ: exterior {exterior}, interior {interior}")]
    HarmonicMismatch {
        /// Exterior harmonic sign.
        exterior: f64,
        /// Interior harmonic sign.
        interior: f64,
    },
    /// The incident direction was invalid.
    #[error("incident direction must be finite and non-zero, got {direction:?}")]
    InvalidIncidentDirection {
        /// Supplied propagation direction.
        direction: [f64; 2],
    },
    /// A complex input or output was not finite.
    #[error("2-D BEM encountered a non-finite {quantity}")]
    NonFiniteInput {
        /// Name of the non-finite input quantity.
        quantity: &'static str,
    },
    /// The dense direct solver rejected the assembled line-BEM matrix.
    #[error("2-D BEM direct solve failed: {0}")]
    DirectSolve(String),
    /// The solved density failed the original matrix residual check.
    #[error("2-D BEM relative residual {residual:.3e} exceeds tolerance {tolerance:.3e}")]
    Residual {
        /// Measured relative residual.
        residual: f64,
        /// Requested residual tolerance.
        tolerance: f64,
    },
    /// A field point was not a finite point in the exterior cylinder domain.
    #[error("2-D field point {index} must be finite and strictly outside the cylinder")]
    InvalidEvaluationPoint {
        /// Index of the invalid evaluation point.
        index: usize,
    },
    /// The angular quadrature used for power was too small.
    #[error("2-D power quadrature requires at least 8 angular samples, got {value}")]
    InvalidAngularSamples {
        /// Supplied angular quadrature sample count.
        value: usize,
    },
    /// The computed field or power was not finite.
    #[error("2-D BEM produced a non-finite {quantity}")]
    NonFiniteResult {
        /// Name of the non-finite result quantity.
        quantity: &'static str,
    },
    /// Caller supplied a transmission-system RHS with the wrong size.
    #[error("2-D transmission RHS length {actual} does not match expected {expected}")]
    InvalidRhsLength {
        /// Required two-medium operator RHS length.
        expected: usize,
        /// Supplied RHS length.
        actual: usize,
    },
    /// Multi-RHS row-major storage would overflow `usize` indexing.
    #[error("2-D transmission batch shape {rows}x{columns} overflows storage")]
    BatchDimensionOverflow {
        /// Number of operator rows in the requested batch.
        rows: usize,
        /// Number of right-hand-side columns in the requested batch.
        columns: usize,
    },
    /// A caller supplied an invalid residual tolerance.
    #[error("2-D residual tolerance must be finite and positive, got {value}")]
    InvalidTolerance {
        /// Supplied residual tolerance.
        value: f64,
    },
}

/// One star-shaped Fourier deformation mode as a fraction of the base radius.
///
/// The deformed boundary is `r(theta) = radius * (1 + sum(cosine * cos(order *
/// theta) + sine * sin(order * theta)))`. Order zero is rejected: a uniform
/// radius shift belongs in `radius`, not in the deformation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoDFourierMode {
    /// Angular mode order, at least one.
    pub order: usize,
    /// Cosine amplitude as a fraction of the base radius.
    pub cosine: f64,
    /// Sine amplitude as a fraction of the base radius.
    pub sine: f64,
}

/// Production CPU constant-panel BEM problem for a rigid circular cylinder.
#[derive(Debug, Clone)]
pub struct TwoDRigidCylinder {
    panels: Vec<TwoDBoundaryPanel>,
    radius: f64,
    physics: PhysicsParams,
    rms_amplitudes: bool,
    deformation: Vec<TwoDFourierMode>,
    bounding_radius: f64,
}

impl TwoDRigidCylinder {
    /// Build a regular counter-clockwise line mesh for a rigid cylinder using
    /// the canonical peak-phasor BEM convention.
    pub fn try_new(
        radius: f64,
        panel_count: usize,
        physics: PhysicsParams,
    ) -> Result<Self, TwoDBemError> {
        Self::try_new_with_amplitude_convention(radius, panel_count, physics, false)
    }

    /// Build a cylinder from typed XEM length and convention-carrying medium.
    ///
    /// The medium's harmonic sign selects outgoing Hankel kind, and its
    /// `rms_amplitudes` bit controls the factor used by the power contract.
    pub fn try_from_xem(
        radius: sonium_xem::LengthM,
        medium: sonium_xem::AcousticMedium,
        panel_count: usize,
    ) -> Result<Self, TwoDBemError> {
        let rms_amplitudes = medium.conventions.rms_amplitudes;
        let physics =
            PhysicsParams::try_new_medium(medium, false).map_err(TwoDBemError::InvalidPhysics)?;
        Self::try_new_with_amplitude_convention(radius.get(), panel_count, physics, rms_amplitudes)
    }

    fn try_new_with_amplitude_convention(
        radius: f64,
        panel_count: usize,
        physics: PhysicsParams,
        rms_amplitudes: bool,
    ) -> Result<Self, TwoDBemError> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(TwoDBemError::InvalidRadius { value: radius });
        }
        if panel_count < 8 {
            return Err(TwoDBemError::InvalidPanelCount { value: panel_count });
        }
        physics.validate().map_err(TwoDBemError::InvalidPhysics)?;

        let mut panels = Vec::with_capacity(panel_count);
        for index in 0..panel_count {
            let start_angle = 2.0 * PI * index as f64 / panel_count as f64;
            let end_angle = 2.0 * PI * (index + 1) as f64 / panel_count as f64;
            let start = [radius * start_angle.cos(), radius * start_angle.sin()];
            let end = [radius * end_angle.cos(), radius * end_angle.sin()];
            let center = [0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1])];
            let dx = end[0] - start[0];
            let dy = end[1] - start[1];
            let length = dx.hypot(dy);
            let normal_length = center[0].hypot(center[1]);
            panels.push(TwoDBoundaryPanel {
                start,
                end,
                center,
                normal: [center[0] / normal_length, center[1] / normal_length],
                length,
            });
        }

        Ok(Self {
            panels,
            radius,
            physics,
            rms_amplitudes,
            deformation: Vec::new(),
            bounding_radius: radius,
        })
    }

    /// Build a rigid star-shaped cylinder with Fourier boundary deformation.
    ///
    /// The total deformation `sum(|cosine| + |sine|)` must stay at or below
    /// 0.3, which keeps the boundary star-shaped with `r(theta) >= 0.7 *
    /// radius` and no self-intersection. An empty mode list reproduces the
    /// circular cylinder up to floating-point rounding.
    pub fn try_new_deformed(
        radius: f64,
        panel_count: usize,
        physics: PhysicsParams,
        modes: &[TwoDFourierMode],
    ) -> Result<Self, TwoDBemError> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(TwoDBemError::InvalidRadius { value: radius });
        }
        if panel_count < 8 {
            return Err(TwoDBemError::InvalidPanelCount { value: panel_count });
        }
        physics.validate().map_err(TwoDBemError::InvalidPhysics)?;
        let mut total = 0.0_f64;
        for mode in modes {
            if mode.order == 0 {
                return Err(TwoDBemError::InvalidDeformation {
                    reason: "Fourier order must be at least one",
                });
            }
            if !mode.cosine.is_finite() || !mode.sine.is_finite() {
                return Err(TwoDBemError::InvalidDeformation {
                    reason: "Fourier amplitudes must be finite",
                });
            }
            total += mode.cosine.abs() + mode.sine.abs();
        }
        // Relative slack mirrors the exterior-point guard: amplitudes that sum
        // to 0.3 in exact arithmetic (for example 0.2 + 0.1) must not trip on
        // floating-point rounding.
        if total > 0.3 * (1.0 + 1.0e-9) {
            return Err(TwoDBemError::InvalidDeformation {
                reason: "total deformation amplitude must stay at or below 0.3",
            });
        }
        let panels = build_star_panels(radius, panel_count, modes);
        Ok(Self {
            panels,
            radius,
            physics,
            rms_amplitudes: false,
            deformation: modes.to_vec(),
            bounding_radius: radius * (1.0 + total),
        })
    }

    /// Return the cylinder radius in metres.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Return the declared Fourier deformation modes (empty for a circle).
    pub fn deformation_modes(&self) -> &[TwoDFourierMode] {
        &self.deformation
    }

    /// Return the bounding radius covering every boundary point.
    pub fn bounding_radius(&self) -> f64 {
        self.bounding_radius
    }

    /// Return the validated acoustic parameters used by this problem.
    pub fn physics(&self) -> &PhysicsParams {
        &self.physics
    }

    /// Return the constant line-panel geometry.
    pub fn panels(&self) -> &[TwoDBoundaryPanel] {
        &self.panels
    }

    /// Solve the rigid-cylinder Neumann problem for one incident plane wave.
    pub fn solve(
        &self,
        incident: TwoDPlaneWave,
        residual_tolerance: f64,
    ) -> Result<TwoDRigidCylinderSolution, TwoDBemError> {
        if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
            return Err(TwoDBemError::InvalidTolerance {
                value: residual_tolerance,
            });
        }
        let count = self.panels.len();
        let mut matrix = DenseMatrix::<Complex64>::zeros(count, count);
        let mut rhs = vec![Complex64::default(); count];
        for (row, field_panel) in self.panels.iter().enumerate() {
            let incident_pressure = incident_pressure(
                incident,
                field_panel.center,
                self.physics.wave_number,
                self.physics.harmonic_factor,
            );
            let normal_component = incident.direction[0] * field_panel.normal[0]
                + incident.direction[1] * field_panel.normal[1];
            let derivative = Complex64::new(
                0.0,
                self.physics.harmonic_factor * self.physics.wave_number * normal_component,
            ) * incident_pressure;
            rhs[row] = -derivative;
            for (column, source_panel) in self.panels.iter().enumerate() {
                let integral = if row == column {
                    // The normal derivative of a single layer over the same
                    // straight panel is zero in the principal-value part;
                    // the exterior jump is represented by -1/2 below.
                    Complex64::default()
                } else {
                    integrate_adjoint_kernel(
                        field_panel.center,
                        field_panel.normal,
                        source_panel,
                        self.physics.wave_number,
                        self.physics.harmonic_factor,
                    )
                };
                matrix[[row, column]] = integral;
            }
            matrix[[row, row]] -= Complex64::new(0.5, 0.0);
        }

        let density = sonium_solvers::direct::lu_solve(&matrix, &rhs)
            .map_err(|error| TwoDBemError::DirectSolve(error.to_string()))?;
        if density.iter().any(|value| !complex_is_finite(*value)) {
            return Err(TwoDBemError::NonFiniteResult {
                quantity: "density",
            });
        }
        let residual = relative_residual(&matrix, &density, &rhs);
        if !residual.is_finite() {
            return Err(TwoDBemError::NonFiniteResult {
                quantity: "residual",
            });
        }
        if residual > residual_tolerance {
            return Err(TwoDBemError::Residual {
                residual,
                tolerance: residual_tolerance,
            });
        }

        Ok(TwoDRigidCylinderSolution {
            panels: self.panels.clone(),
            exterior_radius: self.bounding_radius,
            physics: self.physics.clone(),
            rms_amplitudes: self.rms_amplitudes,
            incident,
            density,
            relative_residual: residual,
        })
    }

    /// Solve several rigid-cylinder scattering problems with one assembled
    /// operator and a shared LU factorization.
    pub fn solve_many(
        &self,
        incidents: &[TwoDPlaneWave],
        residual_tolerance: f64,
    ) -> Result<Vec<TwoDRigidCylinderSolution>, TwoDBemError> {
        if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
            return Err(TwoDBemError::InvalidTolerance {
                value: residual_tolerance,
            });
        }
        if incidents.is_empty() {
            return Ok(Vec::new());
        }

        let count = self.panels.len();
        let mut matrix = DenseMatrix::<Complex64>::zeros(count, count);
        for (row, field_panel) in self.panels.iter().enumerate() {
            for (column, source_panel) in self.panels.iter().enumerate() {
                matrix[[row, column]] = if row == column {
                    Complex64::default()
                } else {
                    integrate_adjoint_kernel(
                        field_panel.center,
                        field_panel.normal,
                        source_panel,
                        self.physics.wave_number,
                        self.physics.harmonic_factor,
                    )
                };
            }
            matrix[[row, row]] -= Complex64::new(0.5, 0.0);
        }
        let factorization = sonium_solvers::direct::lu_factorize(&matrix)
            .map_err(|error| TwoDBemError::DirectSolve(error.to_string()))?;
        let rhs_count = incidents.len();
        let block_len =
            count
                .checked_mul(rhs_count)
                .ok_or(TwoDBemError::BatchDimensionOverflow {
                    rows: count,
                    columns: rhs_count,
                })?;
        let mut rhs_block = vec![Complex64::default(); block_len];
        for (column, &incident) in incidents.iter().enumerate() {
            for (row, field_panel) in self.panels.iter().enumerate() {
                let incident_value = incident_pressure(
                    incident,
                    field_panel.center,
                    self.physics.wave_number,
                    self.physics.harmonic_factor,
                );
                let normal_component = incident.direction[0] * field_panel.normal[0]
                    + incident.direction[1] * field_panel.normal[1];
                let derivative = Complex64::new(
                    0.0,
                    self.physics.harmonic_factor * self.physics.wave_number * normal_component,
                ) * incident_value;
                rhs_block[row * rhs_count + column] = -derivative;
            }
        }
        if rhs_block.iter().any(|value| !complex_is_finite(*value)) {
            return Err(TwoDBemError::NonFiniteResult {
                quantity: "rigid transmission batch RHS",
            });
        }
        let solution_block = factorization
            .solve_many(&rhs_block, rhs_count)
            .map_err(|error| TwoDBemError::DirectSolve(error.to_string()))?;
        let mut solutions = Vec::with_capacity(rhs_count);
        for (column, &incident) in incidents.iter().enumerate() {
            let density = (0..count)
                .map(|row| solution_block[row * rhs_count + column])
                .collect::<Vec<_>>();
            if density.iter().any(|value| !complex_is_finite(*value)) {
                return Err(TwoDBemError::NonFiniteResult {
                    quantity: "rigid density batch",
                });
            }
            let rhs = (0..count)
                .map(|row| rhs_block[row * rhs_count + column])
                .collect::<Vec<_>>();
            let residual = relative_residual(&matrix, &density, &rhs);
            if !residual.is_finite() {
                return Err(TwoDBemError::NonFiniteResult {
                    quantity: "rigid batch residual",
                });
            }
            if residual > residual_tolerance {
                return Err(TwoDBemError::Residual {
                    residual,
                    tolerance: residual_tolerance,
                });
            }
            solutions.push(TwoDRigidCylinderSolution {
                panels: self.panels.clone(),
                exterior_radius: self.bounding_radius,
                physics: self.physics.clone(),
                rms_amplitudes: self.rms_amplitudes,
                incident,
                density,
                relative_residual: residual,
            });
        }
        Ok(solutions)
    }
}

/// A solved 2-D rigid-cylinder scattering state.
#[derive(Debug, Clone)]
pub struct TwoDRigidCylinderSolution {
    panels: Vec<TwoDBoundaryPanel>,
    exterior_radius: f64,
    physics: PhysicsParams,
    rms_amplitudes: bool,
    incident: TwoDPlaneWave,
    density: Vec<Complex64>,
    /// Relative residual certified against the assembled line-BEM matrix.
    pub relative_residual: f64,
}

impl TwoDRigidCylinderSolution {
    /// Return the solved constant-panel single-layer density.
    pub fn density(&self) -> &[Complex64] {
        &self.density
    }

    /// Evaluate the scattered pressure at finite exterior points in pascals.
    pub fn evaluate_scattered_pressure(
        &self,
        points: &[[f64; 2]],
    ) -> Result<Vec<Complex64>, TwoDBemError> {
        let mut values = Vec::with_capacity(points.len());
        for (index, &point) in points.iter().enumerate() {
            self.validate_exterior_point(index, point)?;
            let mut pressure = Complex64::default();
            for (panel, &density) in self.panels.iter().zip(&self.density) {
                pressure += density
                    * integrate_single_layer(
                        point,
                        panel,
                        self.physics.wave_number,
                        self.physics.harmonic_factor,
                    );
            }
            if !complex_is_finite(pressure) {
                return Err(TwoDBemError::NonFiniteResult {
                    quantity: "scattered pressure",
                });
            }
            values.push(pressure);
        }
        Ok(values)
    }

    /// Evaluate total incident plus scattered pressure at exterior points.
    pub fn evaluate_total_pressure(
        &self,
        points: &[[f64; 2]],
    ) -> Result<Vec<Complex64>, TwoDBemError> {
        let scattered = self.evaluate_scattered_pressure(points)?;
        let mut total = Vec::with_capacity(points.len());
        for (&point, &scattered_pressure) in points.iter().zip(&scattered) {
            total.push(
                scattered_pressure
                    + incident_pressure(
                        self.incident,
                        point,
                        self.physics.wave_number,
                        self.physics.harmonic_factor,
                    ),
            );
        }
        if total.iter().any(|value| !complex_is_finite(*value)) {
            return Err(TwoDBemError::NonFiniteResult {
                quantity: "total pressure",
            });
        }
        Ok(total)
    }

    /// Evaluate the dimensionful far-field amplitude in a unit direction.
    ///
    /// For the shared outgoing convention, `F` is defined by the asymptotic
    /// relation documented at module level.  Its squared magnitude is the
    /// quantity used by the per-unit-length power integral.
    pub fn far_field_amplitude(&self, angle: f64) -> Result<Complex64, TwoDBemError> {
        if !angle.is_finite() {
            return Err(TwoDBemError::NonFiniteInput {
                quantity: "far-field angle",
            });
        }
        let direction = [angle.cos(), angle.sin()];
        let mut amplitude = Complex64::default();
        for (panel, &density) in self.panels.iter().zip(&self.density) {
            amplitude += density
                * integrate_far_field_phase(
                    direction,
                    panel,
                    self.physics.wave_number,
                    self.physics.harmonic_factor,
                );
        }
        if !complex_is_finite(amplitude) {
            return Err(TwoDBemError::NonFiniteResult {
                quantity: "far-field amplitude",
            });
        }
        Ok(amplitude)
    }

    /// Integrate scattered acoustic power per unit cylinder length.
    ///
    /// The input phasor is interpreted as RMS when the XEM medium used to
    /// construct the problem declared `rms_amplitudes`; otherwise it is a
    /// peak phasor and the usual factor `1/2` is applied.
    pub fn scattered_power_per_length(&self, angular_samples: usize) -> Result<f64, TwoDBemError> {
        if angular_samples < 8 {
            return Err(TwoDBemError::InvalidAngularSamples {
                value: angular_samples,
            });
        }
        let mut sum = 0.0;
        for index in 0..angular_samples {
            let angle = 2.0 * PI * index as f64 / angular_samples as f64;
            let amplitude = self.far_field_amplitude(angle)?;
            sum += amplitude.norm_sqr();
        }
        let angular_integral = 2.0 * PI * sum / angular_samples as f64;
        let amplitude_factor = if self.rms_amplitudes { 1.0 } else { 0.5 };
        let power = amplitude_factor * angular_integral
            / (8.0
                * PI
                * self.physics.wave_number
                * self.physics.density
                * self.physics.speed_of_sound);
        if !power.is_finite() || power < 0.0 {
            return Err(TwoDBemError::NonFiniteResult {
                quantity: "scattered power per length",
            });
        }
        Ok(power)
    }

    fn validate_exterior_point(&self, index: usize, point: [f64; 2]) -> Result<(), TwoDBemError> {
        let radius = point[0].hypot(point[1]);
        if point.iter().any(|value| !value.is_finite())
            || !radius.is_finite()
            || radius <= self.exterior_radius * (1.0 + 1.0e-10)
        {
            return Err(TwoDBemError::InvalidEvaluationPoint { index });
        }
        Ok(())
    }
}

/// Production CPU constant-panel BEM for a penetrable circular fluid
/// inclusion in two dimensions.
///
/// The unknowns are exterior scattered and interior single-layer
/// densities.  The two equations enforce pressure continuity and
/// density-weighted normal-velocity continuity.  This is intentionally a
/// dense direct CPU reference path; it is the reduced-dimensional analogue
/// used to validate the general transmission operator before accelerated or
/// axisymmetric implementations are attempted.
#[derive(Debug, Clone)]
pub struct TwoDFluidCylinder {
    panels: Vec<TwoDBoundaryPanel>,
    radius: f64,
    exterior: PhysicsParams,
    interior: PhysicsParams,
    rms_amplitudes: bool,
}

impl TwoDFluidCylinder {
    /// Build a peak-phasor penetrable-cylinder problem from validated media.
    pub fn try_new(
        radius: f64,
        panel_count: usize,
        exterior: PhysicsParams,
        interior: PhysicsParams,
    ) -> Result<Self, TwoDBemError> {
        Self::try_new_with_amplitude_convention(radius, panel_count, exterior, interior, false)
    }

    /// Build a penetrable-cylinder problem from typed XEM media.
    pub fn try_from_xem(
        radius: sonium_xem::LengthM,
        exterior: sonium_xem::AcousticMedium,
        interior: sonium_xem::AcousticMedium,
        panel_count: usize,
    ) -> Result<Self, TwoDBemError> {
        let rms_amplitudes = exterior.conventions.rms_amplitudes;
        if interior.conventions.rms_amplitudes != rms_amplitudes {
            return Err(TwoDBemError::HarmonicMismatch {
                exterior: if rms_amplitudes { 1.0 } else { 0.0 },
                interior: if interior.conventions.rms_amplitudes {
                    1.0
                } else {
                    0.0
                },
            });
        }
        let exterior_physics = PhysicsParams::try_new_medium(exterior, false)
            .map_err(TwoDBemError::InvalidExteriorPhysics)?;
        let interior_physics = PhysicsParams::try_new_medium(interior, true)
            .map_err(TwoDBemError::InvalidInteriorPhysics)?;
        Self::try_new_with_amplitude_convention(
            radius.get(),
            panel_count,
            exterior_physics,
            interior_physics,
            rms_amplitudes,
        )
    }

    fn try_new_with_amplitude_convention(
        radius: f64,
        panel_count: usize,
        exterior: PhysicsParams,
        interior: PhysicsParams,
        rms_amplitudes: bool,
    ) -> Result<Self, TwoDBemError> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(TwoDBemError::InvalidRadius { value: radius });
        }
        if panel_count < 8 {
            return Err(TwoDBemError::InvalidPanelCount { value: panel_count });
        }
        exterior
            .validate()
            .map_err(TwoDBemError::InvalidExteriorPhysics)?;
        interior
            .validate()
            .map_err(TwoDBemError::InvalidInteriorPhysics)?;
        let frequency_scale = exterior
            .frequency
            .abs()
            .max(interior.frequency.abs())
            .max(1.0);
        if (exterior.frequency - interior.frequency).abs() > 1.0e-12 * frequency_scale {
            return Err(TwoDBemError::FrequencyMismatch {
                exterior: exterior.frequency,
                interior: interior.frequency,
            });
        }
        if exterior.harmonic_factor != interior.harmonic_factor {
            return Err(TwoDBemError::HarmonicMismatch {
                exterior: exterior.harmonic_factor,
                interior: interior.harmonic_factor,
            });
        }

        let mut panels = Vec::with_capacity(panel_count);
        for index in 0..panel_count {
            let start_angle = 2.0 * PI * index as f64 / panel_count as f64;
            let end_angle = 2.0 * PI * (index + 1) as f64 / panel_count as f64;
            let start = [radius * start_angle.cos(), radius * start_angle.sin()];
            let end = [radius * end_angle.cos(), radius * end_angle.sin()];
            let center = [0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1])];
            let dx = end[0] - start[0];
            let dy = end[1] - start[1];
            let length = dx.hypot(dy);
            let normal_length = center[0].hypot(center[1]);
            panels.push(TwoDBoundaryPanel {
                start,
                end,
                center,
                normal: [center[0] / normal_length, center[1] / normal_length],
                length,
            });
        }

        Ok(Self {
            panels,
            radius,
            exterior,
            interior,
            rms_amplitudes,
        })
    }

    /// Return the cylinder radius in metres.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Return the constant line-panel geometry.
    pub fn panels(&self) -> &[TwoDBoundaryPanel] {
        &self.panels
    }

    /// Return validated exterior medium parameters.
    pub fn exterior_physics(&self) -> &PhysicsParams {
        &self.exterior
    }

    /// Return validated interior medium parameters.
    pub fn interior_physics(&self) -> &PhysicsParams {
        &self.interior
    }

    /// Assemble the checked dense transmission operator and incident right-hand side.
    ///
    /// The matrix is returned so independent callers can certify its action,
    /// transpose/Hermitian contracts, and adjoint residuals without reaching
    /// into the implementation's private quadrature helpers.
    pub fn assemble_transmission_system(
        &self,
        incident: TwoDPlaneWave,
    ) -> Result<(DenseMatrix<Complex64>, Vec<Complex64>), TwoDBemError> {
        let n = self.panels.len();
        let mut matrix = DenseMatrix::<Complex64>::zeros(2 * n, 2 * n);
        let mut rhs = vec![Complex64::default(); 2 * n];

        for (row, field_panel) in self.panels.iter().enumerate() {
            let incident_value = incident_pressure(
                incident,
                field_panel.center,
                self.exterior.wave_number,
                self.exterior.harmonic_factor,
            );
            let incident_derivative = incident_normal_derivative(
                incident,
                field_panel.center,
                field_panel.normal,
                self.exterior.wave_number,
                self.exterior.harmonic_factor,
            );
            rhs[row] = -incident_value;
            rhs[n + row] = -incident_derivative / self.exterior.density;

            for (column, source_panel) in self.panels.iter().enumerate() {
                let exterior_single = integrate_single_layer(
                    field_panel.center,
                    source_panel,
                    self.exterior.wave_number,
                    self.exterior.harmonic_factor,
                );
                let interior_single = integrate_single_layer(
                    field_panel.center,
                    source_panel,
                    self.interior.wave_number,
                    self.interior.harmonic_factor,
                );
                matrix[[row, column]] = exterior_single;
                matrix[[row, n + column]] = -interior_single;

                let exterior_adjoint = if row == column {
                    Complex64::default()
                } else {
                    integrate_adjoint_kernel(
                        field_panel.center,
                        field_panel.normal,
                        source_panel,
                        self.exterior.wave_number,
                        self.exterior.harmonic_factor,
                    )
                };
                let interior_adjoint = if row == column {
                    Complex64::default()
                } else {
                    integrate_adjoint_kernel(
                        field_panel.center,
                        field_panel.normal,
                        source_panel,
                        self.interior.wave_number,
                        self.interior.harmonic_factor,
                    )
                };

                let mut exterior_normal = exterior_adjoint / self.exterior.density;
                let mut interior_normal = -interior_adjoint / self.interior.density;
                if row == column {
                    exterior_normal -= Complex64::new(0.5 / self.exterior.density, 0.0);
                    interior_normal -= Complex64::new(0.5 / self.interior.density, 0.0);
                }
                matrix[[n + row, column]] = exterior_normal;
                matrix[[n + row, n + column]] = interior_normal;
            }
        }

        if matrix
            .as_slice()
            .iter()
            .any(|value| !complex_is_finite(*value))
            || rhs.iter().any(|value| !complex_is_finite(*value))
        {
            return Err(TwoDBemError::NonFiniteResult {
                quantity: "transmission system",
            });
        }
        Ok((matrix, rhs))
    }

    /// Solve the Hermitian-adjoint transmission system `Aᴴ y = rhs`.
    pub fn solve_adjoint(
        &self,
        rhs: &[Complex64],
        residual_tolerance: f64,
    ) -> Result<TwoDTransmissionAdjointSolution, TwoDBemError> {
        if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
            return Err(TwoDBemError::InvalidTolerance {
                value: residual_tolerance,
            });
        }
        let expected = 2 * self.panels.len();
        if rhs.len() != expected {
            return Err(TwoDBemError::InvalidRhsLength {
                expected,
                actual: rhs.len(),
            });
        }
        if rhs.iter().any(|value| !complex_is_finite(*value)) {
            return Err(TwoDBemError::NonFiniteInput {
                quantity: "adjoint RHS",
            });
        }

        let (matrix, _) = self.assemble_transmission_system(TwoDPlaneWave::plane_wave_x())?;
        let mut hermitian = DenseMatrix::<Complex64>::zeros(expected, expected);
        for row in 0..expected {
            for column in 0..expected {
                hermitian[[row, column]] = matrix[[column, row]].conj();
            }
        }
        let coefficients = sonium_solvers::direct::lu_solve(&hermitian, rhs)
            .map_err(|error| TwoDBemError::DirectSolve(error.to_string()))?;
        if coefficients.iter().any(|value| !complex_is_finite(*value)) {
            return Err(TwoDBemError::NonFiniteResult {
                quantity: "adjoint density",
            });
        }
        let residual = relative_residual(&hermitian, &coefficients, rhs);
        if !residual.is_finite() {
            return Err(TwoDBemError::NonFiniteResult {
                quantity: "adjoint residual",
            });
        }
        if residual > residual_tolerance {
            return Err(TwoDBemError::Residual {
                residual,
                tolerance: residual_tolerance,
            });
        }
        Ok(TwoDTransmissionAdjointSolution {
            coefficients,
            relative_residual: residual,
            panels: self.panels.clone(),
            radius: self.radius,
            exterior: self.exterior.clone(),
        })
    }

    /// Solve several Hermitian-adjoint transmission systems with one
    /// assembled/factorized operator.
    ///
    /// The right-hand sides are independent columns of `Aᴴ Y = B`.  Every
    /// column is validated and checked against the exact assembled operator;
    /// malformed input or a failed residual gate returns no partial batch.
    pub fn solve_adjoint_many(
        &self,
        rhs_columns: &[Vec<Complex64>],
        residual_tolerance: f64,
    ) -> Result<Vec<TwoDTransmissionAdjointSolution>, TwoDBemError> {
        if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
            return Err(TwoDBemError::InvalidTolerance {
                value: residual_tolerance,
            });
        }
        if rhs_columns.is_empty() {
            return Ok(Vec::new());
        }
        let expected = 2 * self.panels.len();
        for rhs in rhs_columns {
            if rhs.len() != expected {
                return Err(TwoDBemError::InvalidRhsLength {
                    expected,
                    actual: rhs.len(),
                });
            }
            if rhs.iter().any(|value| !complex_is_finite(*value)) {
                return Err(TwoDBemError::NonFiniteInput {
                    quantity: "adjoint RHS",
                });
            }
        }

        let (matrix, _) = self.assemble_transmission_system(TwoDPlaneWave::plane_wave_x())?;
        let mut hermitian = DenseMatrix::<Complex64>::zeros(expected, expected);
        for row in 0..expected {
            for column in 0..expected {
                hermitian[[row, column]] = matrix[[column, row]].conj();
            }
        }

        let factorization = sonium_solvers::direct::lu_factorize(&hermitian)
            .map_err(|error| TwoDBemError::DirectSolve(error.to_string()))?;
        let nrhs = rhs_columns.len();
        let mut rhs_block = vec![Complex64::default(); expected * nrhs];
        for (column, rhs) in rhs_columns.iter().enumerate() {
            for (row, value) in rhs.iter().enumerate() {
                rhs_block[row * nrhs + column] = *value;
            }
        }
        let solution_block = factorization
            .solve_many(&rhs_block, nrhs)
            .map_err(|error| TwoDBemError::DirectSolve(error.to_string()))?;

        let mut solutions = Vec::with_capacity(nrhs);
        for column in 0..nrhs {
            let coefficients = (0..expected)
                .map(|row| solution_block[row * nrhs + column])
                .collect::<Vec<_>>();
            if coefficients.iter().any(|value| !complex_is_finite(*value)) {
                return Err(TwoDBemError::NonFiniteResult {
                    quantity: "adjoint density",
                });
            }
            let residual = relative_residual(&hermitian, &coefficients, &rhs_columns[column]);
            if !residual.is_finite() {
                return Err(TwoDBemError::NonFiniteResult {
                    quantity: "adjoint residual",
                });
            }
            if residual > residual_tolerance {
                return Err(TwoDBemError::Residual {
                    residual,
                    tolerance: residual_tolerance,
                });
            }
            solutions.push(TwoDTransmissionAdjointSolution {
                coefficients,
                relative_residual: residual,
                panels: self.panels.clone(),
                radius: self.radius,
                exterior: self.exterior.clone(),
            });
        }
        Ok(solutions)
    }

    /// Solve the two-medium transmission problem for one incident plane wave.
    pub fn solve(
        &self,
        incident: TwoDPlaneWave,
        residual_tolerance: f64,
    ) -> Result<TwoDFluidCylinderSolution, TwoDBemError> {
        if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
            return Err(TwoDBemError::InvalidTolerance {
                value: residual_tolerance,
            });
        }

        let n = self.panels.len();
        let mut matrix = DenseMatrix::<Complex64>::zeros(2 * n, 2 * n);
        let mut rhs = vec![Complex64::default(); 2 * n];
        for (row, field_panel) in self.panels.iter().enumerate() {
            let incident_value = incident_pressure(
                incident,
                field_panel.center,
                self.exterior.wave_number,
                self.exterior.harmonic_factor,
            );
            let incident_derivative = incident_normal_derivative(
                incident,
                field_panel.center,
                field_panel.normal,
                self.exterior.wave_number,
                self.exterior.harmonic_factor,
            );
            rhs[row] = -incident_value;
            rhs[n + row] = -incident_derivative / self.exterior.density;

            for (column, source_panel) in self.panels.iter().enumerate() {
                let exterior_single = integrate_single_layer(
                    field_panel.center,
                    source_panel,
                    self.exterior.wave_number,
                    self.exterior.harmonic_factor,
                );
                let interior_single = integrate_single_layer(
                    field_panel.center,
                    source_panel,
                    self.interior.wave_number,
                    self.interior.harmonic_factor,
                );
                matrix[[row, column]] = exterior_single;
                matrix[[row, n + column]] = -interior_single;

                let exterior_adjoint = if row == column {
                    Complex64::default()
                } else {
                    integrate_adjoint_kernel(
                        field_panel.center,
                        field_panel.normal,
                        source_panel,
                        self.exterior.wave_number,
                        self.exterior.harmonic_factor,
                    )
                };
                let interior_adjoint = if row == column {
                    Complex64::default()
                } else {
                    integrate_adjoint_kernel(
                        field_panel.center,
                        field_panel.normal,
                        source_panel,
                        self.interior.wave_number,
                        self.interior.harmonic_factor,
                    )
                };

                let mut exterior_normal = exterior_adjoint / self.exterior.density;
                let mut interior_normal = -interior_adjoint / self.interior.density;
                if row == column {
                    exterior_normal -= Complex64::new(0.5 / self.exterior.density, 0.0);
                    interior_normal -= Complex64::new(0.5 / self.interior.density, 0.0);
                }
                matrix[[n + row, column]] = exterior_normal;
                matrix[[n + row, n + column]] = interior_normal;
            }
        }

        let values = sonium_solvers::direct::lu_solve(&matrix, &rhs)
            .map_err(|error| TwoDBemError::DirectSolve(error.to_string()))?;
        if values.len() != 2 * n || values.iter().any(|value| !complex_is_finite(*value)) {
            return Err(TwoDBemError::NonFiniteResult {
                quantity: "transmission density",
            });
        }
        let relative_residual = relative_residual(&matrix, &values, &rhs);
        if !relative_residual.is_finite() {
            return Err(TwoDBemError::NonFiniteResult {
                quantity: "transmission residual",
            });
        }
        if relative_residual > residual_tolerance {
            return Err(TwoDBemError::Residual {
                residual: relative_residual,
                tolerance: residual_tolerance,
            });
        }

        Ok(TwoDFluidCylinderSolution {
            panels: self.panels.clone(),
            radius: self.radius,
            exterior: self.exterior.clone(),
            interior: self.interior.clone(),
            incident,
            exterior_density: values[..n].to_vec(),
            interior_density: values[n..].to_vec(),
            rms_amplitudes: self.rms_amplitudes,
            relative_residual,
        })
    }

    /// Solve several two-medium transmission problems with one assembled
    /// operator and a shared LU factorization.
    ///
    /// Each incident wave supplies one right-hand-side column. The returned
    /// solutions retain their corresponding incident wave and independently
    /// certify the original assembled operator residual.
    pub fn solve_many(
        &self,
        incidents: &[TwoDPlaneWave],
        residual_tolerance: f64,
    ) -> Result<Vec<TwoDFluidCylinderSolution>, TwoDBemError> {
        if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
            return Err(TwoDBemError::InvalidTolerance {
                value: residual_tolerance,
            });
        }
        if incidents.is_empty() {
            return Ok(Vec::new());
        }

        let (matrix, first_rhs) = self.assemble_transmission_system(incidents[0])?;
        let expected = matrix.nrows();
        let factorization = sonium_solvers::direct::lu_factorize(&matrix)
            .map_err(|error| TwoDBemError::DirectSolve(error.to_string()))?;
        let rhs_count = incidents.len();
        let block_len =
            expected
                .checked_mul(rhs_count)
                .ok_or(TwoDBemError::BatchDimensionOverflow {
                    rows: expected,
                    columns: rhs_count,
                })?;
        let mut rhs_block = vec![Complex64::default(); block_len];
        for (row, &value) in first_rhs.iter().enumerate() {
            rhs_block[row * rhs_count] = value;
        }
        for (column, &incident) in incidents.iter().enumerate().skip(1) {
            let rhs = self.incident_rhs(incident)?;
            for (row, &value) in rhs.iter().enumerate() {
                rhs_block[row * rhs_count + column] = value;
            }
        }

        let solution_block = factorization
            .solve_many(&rhs_block, rhs_count)
            .map_err(|error| TwoDBemError::DirectSolve(error.to_string()))?;
        let n = self.panels.len();
        let mut solutions = Vec::with_capacity(rhs_count);
        for (column, &incident) in incidents.iter().enumerate() {
            let coefficients = (0..expected)
                .map(|row| solution_block[row * rhs_count + column])
                .collect::<Vec<_>>();
            if coefficients.iter().any(|value| !complex_is_finite(*value)) {
                return Err(TwoDBemError::NonFiniteResult {
                    quantity: "transmission density batch",
                });
            }
            let rhs = (0..expected)
                .map(|row| rhs_block[row * rhs_count + column])
                .collect::<Vec<_>>();
            let residual = relative_residual(&matrix, &coefficients, &rhs);
            if !residual.is_finite() {
                return Err(TwoDBemError::NonFiniteResult {
                    quantity: "transmission batch residual",
                });
            }
            if residual > residual_tolerance {
                return Err(TwoDBemError::Residual {
                    residual,
                    tolerance: residual_tolerance,
                });
            }
            solutions.push(TwoDFluidCylinderSolution {
                panels: self.panels.clone(),
                radius: self.radius,
                exterior: self.exterior.clone(),
                interior: self.interior.clone(),
                incident,
                exterior_density: coefficients[..n].to_vec(),
                interior_density: coefficients[n..].to_vec(),
                rms_amplitudes: self.rms_amplitudes,
                relative_residual: residual,
            });
        }
        Ok(solutions)
    }

    fn incident_rhs(&self, incident: TwoDPlaneWave) -> Result<Vec<Complex64>, TwoDBemError> {
        let n = self.panels.len();
        let mut rhs = vec![Complex64::default(); 2 * n];
        for (row, field_panel) in self.panels.iter().enumerate() {
            let incident_value = incident_pressure(
                incident,
                field_panel.center,
                self.exterior.wave_number,
                self.exterior.harmonic_factor,
            );
            let incident_derivative = incident_normal_derivative(
                incident,
                field_panel.center,
                field_panel.normal,
                self.exterior.wave_number,
                self.exterior.harmonic_factor,
            );
            rhs[row] = -incident_value;
            rhs[n + row] = -incident_derivative / self.exterior.density;
        }
        if rhs.iter().any(|value| !complex_is_finite(*value)) {
            return Err(TwoDBemError::NonFiniteResult {
                quantity: "transmission batch RHS",
            });
        }
        Ok(rhs)
    }
}

/// Certified solution of the Hermitian-adjoint 2-D transmission system.
#[derive(Debug, Clone)]
pub struct TwoDTransmissionAdjointSolution {
    coefficients: Vec<Complex64>,
    relative_residual: f64,
    panels: Vec<TwoDBoundaryPanel>,
    radius: f64,
    exterior: PhysicsParams,
}

impl TwoDTransmissionAdjointSolution {
    /// Return the solved adjoint coefficients in operator ordering.
    pub fn coefficients(&self) -> &[Complex64] {
        &self.coefficients
    }

    /// Return the independently recomputed relative adjoint residual.
    pub fn relative_residual(&self) -> f64 {
        self.relative_residual
    }

    /// Evaluate the adjoint exterior single-layer pressure at exterior
    /// points. The result omits any incident-field contribution.
    pub fn evaluate_exterior_adjoint_pressure(
        &self,
        points: &[[f64; 2]],
    ) -> Result<Vec<Complex64>, TwoDBemError> {
        let n = self.panels.len();
        points
            .iter()
            .enumerate()
            .map(|(index, &point)| {
                let radius = point[0].hypot(point[1]);
                if point.iter().any(|value| !value.is_finite())
                    || !radius.is_finite()
                    || radius <= self.radius * (1.0 + 1.0e-10)
                {
                    return Err(TwoDBemError::InvalidEvaluationPoint { index });
                }
                let pressure = self
                    .panels
                    .iter()
                    .zip(&self.coefficients[..n])
                    .map(|(panel, &coefficient)| {
                        coefficient
                            * integrate_single_layer(
                                point,
                                panel,
                                self.exterior.wave_number,
                                self.exterior.harmonic_factor,
                            )
                    })
                    .sum::<Complex64>();
                if !complex_is_finite(pressure) {
                    return Err(TwoDBemError::NonFiniteResult {
                        quantity: "adjoint exterior pressure",
                    });
                }
                Ok(pressure)
            })
            .collect()
    }
}

/// Solved traces and field evaluator for [`TwoDFluidCylinder`].
#[derive(Debug, Clone)]
pub struct TwoDFluidCylinderSolution {
    panels: Vec<TwoDBoundaryPanel>,
    radius: f64,
    exterior: PhysicsParams,
    interior: PhysicsParams,
    incident: TwoDPlaneWave,
    exterior_density: Vec<Complex64>,
    interior_density: Vec<Complex64>,
    rms_amplitudes: bool,
    /// Relative residual certified against the assembled two-medium system.
    pub relative_residual: f64,
}

impl TwoDFluidCylinderSolution {
    /// Return the exterior scattered single-layer density.
    pub fn exterior_density(&self) -> &[Complex64] {
        &self.exterior_density
    }

    /// Return the interior transmitted single-layer density.
    pub fn interior_density(&self) -> &[Complex64] {
        &self.interior_density
    }

    /// Evaluate scattered exterior pressure at points strictly outside.
    pub fn evaluate_exterior_scattered_pressure(
        &self,
        points: &[[f64; 2]],
    ) -> Result<Vec<Complex64>, TwoDBemError> {
        let mut values = Vec::with_capacity(points.len());
        for (index, &point) in points.iter().enumerate() {
            self.validate_exterior_point(index, point)?;
            let pressure = self
                .panels
                .iter()
                .zip(&self.exterior_density)
                .map(|(panel, &density)| {
                    density
                        * integrate_single_layer(
                            point,
                            panel,
                            self.exterior.wave_number,
                            self.exterior.harmonic_factor,
                        )
                })
                .sum();
            if !complex_is_finite(pressure) {
                return Err(TwoDBemError::NonFiniteResult {
                    quantity: "exterior scattered pressure",
                });
            }
            values.push(pressure);
        }
        Ok(values)
    }

    /// Evaluate total exterior pressure at points strictly outside.
    pub fn evaluate_exterior_total_pressure(
        &self,
        points: &[[f64; 2]],
    ) -> Result<Vec<Complex64>, TwoDBemError> {
        let scattered = self.evaluate_exterior_scattered_pressure(points)?;
        let total = points
            .iter()
            .zip(scattered)
            .map(|(&point, scattered)| {
                scattered
                    + incident_pressure(
                        self.incident,
                        point,
                        self.exterior.wave_number,
                        self.exterior.harmonic_factor,
                    )
            })
            .collect::<Vec<_>>();
        if total.iter().any(|value| !complex_is_finite(*value)) {
            return Err(TwoDBemError::NonFiniteResult {
                quantity: "exterior total pressure",
            });
        }
        Ok(total)
    }

    /// Evaluate transmitted interior pressure at points strictly inside.
    pub fn evaluate_interior_pressure(
        &self,
        points: &[[f64; 2]],
    ) -> Result<Vec<Complex64>, TwoDBemError> {
        let mut values = Vec::with_capacity(points.len());
        for (index, &point) in points.iter().enumerate() {
            self.validate_interior_point(index, point)?;
            let pressure = self
                .panels
                .iter()
                .zip(&self.interior_density)
                .map(|(panel, &density)| {
                    density
                        * integrate_single_layer(
                            point,
                            panel,
                            self.interior.wave_number,
                            self.interior.harmonic_factor,
                        )
                })
                .sum();
            if !complex_is_finite(pressure) {
                return Err(TwoDBemError::NonFiniteResult {
                    quantity: "interior pressure",
                });
            }
            values.push(pressure);
        }
        Ok(values)
    }

    /// Evaluate the outgoing exterior far-field amplitude.
    pub fn far_field_amplitude(&self, angle: f64) -> Result<Complex64, TwoDBemError> {
        if !angle.is_finite() {
            return Err(TwoDBemError::NonFiniteInput {
                quantity: "far-field angle",
            });
        }
        let direction = [angle.cos(), angle.sin()];
        let amplitude = self
            .panels
            .iter()
            .zip(&self.exterior_density)
            .map(|(panel, &density)| {
                density
                    * integrate_far_field_phase(
                        direction,
                        panel,
                        self.exterior.wave_number,
                        self.exterior.harmonic_factor,
                    )
            })
            .sum();
        if !complex_is_finite(amplitude) {
            return Err(TwoDBemError::NonFiniteResult {
                quantity: "transmission far-field amplitude",
            });
        }
        Ok(amplitude)
    }

    /// Integrate scattered exterior power per unit cylinder length.
    pub fn scattered_power_per_length(&self, angular_samples: usize) -> Result<f64, TwoDBemError> {
        if angular_samples < 8 {
            return Err(TwoDBemError::InvalidAngularSamples {
                value: angular_samples,
            });
        }
        let sum = (0..angular_samples)
            .map(|index| {
                let angle = 2.0 * PI * index as f64 / angular_samples as f64;
                self.far_field_amplitude(angle)
                    .map(|amplitude| amplitude.norm_sqr())
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|values| values.into_iter().sum::<f64>())?;
        let angular_integral = 2.0 * PI * sum / angular_samples as f64;
        let amplitude_factor = if self.rms_amplitudes { 1.0 } else { 0.5 };
        let power = amplitude_factor * angular_integral
            / (8.0
                * PI
                * self.exterior.wave_number
                * self.exterior.density
                * self.exterior.speed_of_sound);
        if !power.is_finite() || power < 0.0 {
            return Err(TwoDBemError::NonFiniteResult {
                quantity: "transmission scattered power per length",
            });
        }
        Ok(power)
    }

    /// Maximum pressure continuity residual at panel collocation points.
    pub fn pressure_continuity_error(&self) -> f64 {
        self.panels
            .iter()
            .map(|panel| {
                let exterior = incident_pressure(
                    self.incident,
                    panel.center,
                    self.exterior.wave_number,
                    self.exterior.harmonic_factor,
                ) + self.layer_pressure_at(
                    panel.center,
                    &self.exterior_density,
                    &self.exterior,
                );
                let interior =
                    self.layer_pressure_at(panel.center, &self.interior_density, &self.interior);
                (exterior - interior).norm()
            })
            .fold(0.0, f64::max)
    }

    /// Maximum normal-velocity continuity residual at collocation points.
    pub fn velocity_continuity_error(&self) -> f64 {
        let i_omega = Complex64::new(0.0, self.exterior.harmonic_factor * self.exterior.omega);
        self.panels
            .iter()
            .enumerate()
            .map(|(row, panel)| {
                let exterior_action =
                    self.normal_layer_at(row, &self.exterior_density, &self.exterior, -0.5)
                        + incident_normal_derivative(
                            self.incident,
                            panel.center,
                            panel.normal,
                            self.exterior.wave_number,
                            self.exterior.harmonic_factor,
                        );
                let interior_action =
                    self.normal_layer_at(row, &self.interior_density, &self.interior, 0.5);
                let exterior_velocity = exterior_action / (i_omega * self.exterior.density);
                let interior_velocity = interior_action / (i_omega * self.interior.density);
                (exterior_velocity - interior_velocity).norm()
            })
            .fold(0.0, f64::max)
    }

    fn layer_pressure_at(
        &self,
        point: [f64; 2],
        density: &[Complex64],
        physics: &PhysicsParams,
    ) -> Complex64 {
        self.panels
            .iter()
            .zip(density)
            .map(|(panel, &value)| {
                value
                    * integrate_single_layer(
                        point,
                        panel,
                        physics.wave_number,
                        physics.harmonic_factor,
                    )
            })
            .sum()
    }

    fn normal_layer_at(
        &self,
        row: usize,
        density: &[Complex64],
        physics: &PhysicsParams,
        jump: f64,
    ) -> Complex64 {
        let field_panel = &self.panels[row];
        let off_diagonal = self
            .panels
            .iter()
            .enumerate()
            .filter(|(column, _)| *column != row)
            .map(|(column, source_panel)| {
                density[column]
                    * integrate_adjoint_kernel(
                        field_panel.center,
                        field_panel.normal,
                        source_panel,
                        physics.wave_number,
                        physics.harmonic_factor,
                    )
            })
            .sum::<Complex64>();
        off_diagonal + density[row] * jump
    }

    fn validate_exterior_point(&self, index: usize, point: [f64; 2]) -> Result<(), TwoDBemError> {
        let radius = point[0].hypot(point[1]);
        if point.iter().any(|value| !value.is_finite())
            || !radius.is_finite()
            || radius <= self.radius * (1.0 + 1.0e-10)
        {
            return Err(TwoDBemError::InvalidEvaluationPoint { index });
        }
        Ok(())
    }

    fn validate_interior_point(&self, index: usize, point: [f64; 2]) -> Result<(), TwoDBemError> {
        let radius = point[0].hypot(point[1]);
        if point.iter().any(|value| !value.is_finite())
            || !radius.is_finite()
            || radius >= self.radius * (1.0 - 1.0e-10)
        {
            return Err(TwoDBemError::InvalidEvaluationPoint { index });
        }
        Ok(())
    }
}

/// Build counter-clockwise constant panels for a star-shaped Fourier boundary.
fn build_star_panels(
    radius: f64,
    panel_count: usize,
    modes: &[TwoDFourierMode],
) -> Vec<TwoDBoundaryPanel> {
    let shape = |theta: f64| -> (f64, f64) {
        let mut radial = 1.0_f64;
        let mut slope = 0.0_f64;
        for mode in modes {
            let phase = mode.order as f64 * theta;
            radial += mode.cosine * phase.cos() + mode.sine * phase.sin();
            slope += mode.order as f64 * (mode.sine * phase.cos() - mode.cosine * phase.sin());
        }
        (radius * radial, radius * slope)
    };
    let point = |theta: f64| -> [f64; 2] {
        let (r, _) = shape(theta);
        [r * theta.cos(), r * theta.sin()]
    };
    let mut panels = Vec::with_capacity(panel_count);
    for index in 0..panel_count {
        let start_angle = 2.0 * PI * index as f64 / panel_count as f64;
        let end_angle = 2.0 * PI * (index + 1) as f64 / panel_count as f64;
        let center_angle = 0.5 * (start_angle + end_angle);
        let start = point(start_angle);
        let end = point(end_angle);
        let center = [0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1])];
        let (r, slope) = shape(center_angle);
        // Tangent of p(theta) = r * (cos, sin); rotating it by -90 degrees
        // gives the outward normal for counter-clockwise traversal.
        let (sin, cos) = center_angle.sin_cos();
        let tangent = [slope * cos - r * sin, slope * sin + r * cos];
        let tangent_length = tangent[0].hypot(tangent[1]).max(MIN_SCALE);
        let normal = [tangent[1] / tangent_length, -tangent[0] / tangent_length];
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        panels.push(TwoDBoundaryPanel {
            start,
            end,
            center,
            normal,
            length: dx.hypot(dy),
        });
    }
    panels
}

fn integrate_adjoint_kernel(
    field_point: [f64; 2],
    field_normal: [f64; 2],
    source_panel: &TwoDBoundaryPanel,
    wave_number: f64,
    harmonic_factor: f64,
) -> Complex64 {
    let mut integral = Complex64::default();
    let half_tangent = [
        0.5 * (source_panel.end[0] - source_panel.start[0]),
        0.5 * (source_panel.end[1] - source_panel.start[1]),
    ];
    for (&coordinate, &weight) in GAUSS_POINTS.iter().zip(GAUSS_WEIGHTS.iter()) {
        let source = [
            source_panel.center[0] + coordinate * half_tangent[0],
            source_panel.center[1] + coordinate * half_tangent[1],
        ];
        let dx = field_point[0] - source[0];
        let dy = field_point[1] - source[1];
        let distance = dx.hypot(dy);
        if distance <= 1.0e-14 {
            continue;
        }
        let radial_derivative = green_radial_derivative(distance, wave_number, harmonic_factor);
        let normal_projection = (dx * field_normal[0] + dy * field_normal[1]) / distance;
        integral += radial_derivative * normal_projection * (0.5 * source_panel.length * weight);
    }
    integral
}

fn integrate_single_layer(
    field_point: [f64; 2],
    source_panel: &TwoDBoundaryPanel,
    wave_number: f64,
    harmonic_factor: f64,
) -> Complex64 {
    let mut integral = Complex64::default();
    let half_tangent = [
        0.5 * (source_panel.end[0] - source_panel.start[0]),
        0.5 * (source_panel.end[1] - source_panel.start[1]),
    ];
    for (&coordinate, &weight) in GAUSS_POINTS.iter().zip(GAUSS_WEIGHTS.iter()) {
        let source = [
            source_panel.center[0] + coordinate * half_tangent[0],
            source_panel.center[1] + coordinate * half_tangent[1],
        ];
        let distance = (field_point[0] - source[0]).hypot(field_point[1] - source[1]);
        if distance <= 1.0e-14 {
            continue;
        }
        integral += green_function(distance, wave_number, harmonic_factor)
            * (0.5 * source_panel.length * weight);
    }
    integral
}

fn integrate_far_field_phase(
    direction: [f64; 2],
    source_panel: &TwoDBoundaryPanel,
    wave_number: f64,
    harmonic_factor: f64,
) -> Complex64 {
    let mut integral = Complex64::default();
    let half_tangent = [
        0.5 * (source_panel.end[0] - source_panel.start[0]),
        0.5 * (source_panel.end[1] - source_panel.start[1]),
    ];
    for (&coordinate, &weight) in GAUSS_POINTS.iter().zip(GAUSS_WEIGHTS.iter()) {
        let source = [
            source_panel.center[0] + coordinate * half_tangent[0],
            source_panel.center[1] + coordinate * half_tangent[1],
        ];
        let phase =
            -harmonic_factor * wave_number * (direction[0] * source[0] + direction[1] * source[1]);
        integral += Complex64::new(phase.cos(), phase.sin()) * (0.5 * source_panel.length * weight);
    }
    integral
}

fn incident_normal_derivative(
    incident: TwoDPlaneWave,
    point: [f64; 2],
    normal: [f64; 2],
    wave_number: f64,
    harmonic_factor: f64,
) -> Complex64 {
    let pressure = incident_pressure(incident, point, wave_number, harmonic_factor);
    let projection = incident.direction[0] * normal[0] + incident.direction[1] * normal[1];
    Complex64::new(0.0, harmonic_factor * wave_number * projection) * pressure
}

fn incident_pressure(
    incident: TwoDPlaneWave,
    point: [f64; 2],
    wave_number: f64,
    harmonic_factor: f64,
) -> Complex64 {
    let phase = harmonic_factor
        * wave_number
        * (incident.direction[0] * point[0] + incident.direction[1] * point[1]);
    incident.amplitude * Complex64::new(phase.cos(), phase.sin())
}

fn green_function(distance: f64, wave_number: f64, harmonic_factor: f64) -> Complex64 {
    let outgoing = greens_function_2d(distance, wave_number);
    if harmonic_factor > 0.0 {
        outgoing
    } else {
        outgoing.conj()
    }
}

fn green_radial_derivative(distance: f64, wave_number: f64, harmonic_factor: f64) -> Complex64 {
    // `sonium-wave` owns the 2-D Hankel implementation. Keep this BEM crate
    // independent of its private special-function dependency and use a
    // centered radial derivative away from the skipped self panel.
    let step = (1.0e-5 * (1.0 + distance)).min(0.25 * distance);
    let derivative = (green_function(distance + step, wave_number, 1.0)
        - green_function(distance - step, wave_number, 1.0))
        / (2.0 * step);
    if harmonic_factor > 0.0 {
        derivative
    } else {
        derivative.conj()
    }
}

fn relative_residual(
    matrix: &DenseMatrix<Complex64>,
    solution: &[Complex64],
    rhs: &[Complex64],
) -> f64 {
    let mut residual_squared = 0.0;
    let mut rhs_squared = 0.0;
    for row in 0..rhs.len() {
        let mut action = Complex64::default();
        for column in 0..solution.len() {
            action += matrix[[row, column]] * solution[column];
        }
        residual_squared += (action - rhs[row]).norm_sqr();
        rhs_squared += rhs[row].norm_sqr();
    }
    residual_squared.sqrt() / rhs_squared.sqrt().max(MIN_SCALE)
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}
